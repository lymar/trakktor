//! Reading one picture: prompt, greedy decode, and knowing when to stop.
//!
//! The prompt is the published chat template flattened by hand — there is one
//! turn, one picture and one instruction, so a template engine would be a
//! dependency to spell four constants.
//!
//! Stopping is the interesting part. A generative reader does not fail by
//! emitting a wrong glyph; it fails by **never stopping** — the same syllable,
//! or the same short phrase, repeated until the caller's budget runs out. That
//! is not a rare pathology but the model's documented behaviour when it is
//! handed more than it can parse, so the loop watches for it explicitly and
//! reports it, rather than returning a page of noise as if it were text.

use candle_core::{D, DType, Device, IndexOp, Result, Tensor};
use candle_nn::ops::softmax_last_dim;
use tokenizers::Tokenizer;

use super::{
    config::{ImageConfig, ModelConfig},
    ernie::{Cache, Decoder},
    image::Prepared,
    vision::{Projector, Tower},
};

/// What the model is asked to do with the picture.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Task {
    /// Read the text.
    Ocr,
    /// Read a table, as markup rather than as lines.
    Table,
    /// Read a formula as LaTeX.
    Formula,
    /// Describe a chart.
    Chart,
}

impl Task {
    /// The instruction that follows the picture in the prompt.
    ///
    /// Two more exist upstream — `Spotting:` and `Seal Recognition:` — and are
    /// deliberately not offered: `Spotting:` transliterates Cyrillic into
    /// look-alike Latin and turns Tibetan into noise.
    pub fn prompt(self) -> &'static str {
        match self {
            Self::Ocr => "OCR:",
            Self::Table => "Table Recognition:",
            Self::Formula => "Formula Recognition:",
            Self::Chart => "Chart Recognition:",
        }
    }

    /// Whether this task wants the page whole rather than cut into blocks. A
    /// table *is* a block; cutting it up destroys the structure that makes the
    /// task worth asking for.
    pub fn wants_whole_page(self) -> bool { self != Self::Ocr }
}

/// When to give up on one picture.
#[derive(Debug, Clone, Copy)]
pub struct Limits {
    /// Hard ceiling on the answer's length.
    ///
    /// It has to be generous, and how generous depends on the script: the
    /// vocabulary holds only about thirty tokens with Tibetan code points, so
    /// Tibetan costs roughly 1.2–1.35 tokens per character where English costs
    /// 0.23. A ceiling sized by an English reading cuts a Tibetan block in
    /// half.
    pub max_tokens: usize,
    /// How many times one token may repeat in a row before the run is called a
    /// loop.
    pub repeat_run: usize,
    /// The longest cycle looked for, in tokens. A phrase repeating with a
    /// period longer than this is not caught — and the period is measured in
    /// tokens, not characters, so a script that costs five tokens a character
    /// needs a great deal more room here than Latin does.
    pub repeat_period: usize,
    /// How many tokens of unbroken repetition it takes to call it a loop.
    /// Real text does repeat — a table column, a list of dates — so a short
    /// echo has to be allowed to pass.
    pub repeat_tokens: usize,
}

impl Default for Limits {
    fn default() -> Self {
        Self {
            max_tokens: DEFAULT_MAX_TOKENS,
            repeat_run: 16,
            repeat_period: 96,
            repeat_tokens: 64,
        }
    }
}

/// The published ceiling on one block's answer.
pub const DEFAULT_MAX_TOKENS: usize = 1024;

/// What the model made of one picture.
#[derive(Debug, Clone)]
pub struct Answer {
    pub text: String,
    /// Mean probability of the tokens that were kept.
    ///
    /// A generative reader has no per-character CTC probability to average, so
    /// this stands in its place: the softmax value of the token the greedy
    /// choice took, averaged over the answer. It behaves the same way — near
    /// one where the reading is certain, sagging where the model is inventing.
    pub score: f32,
    /// How many tokens were kept.
    pub tokens: usize,
    /// Whether the answer was cut short — by the ceiling or by the loop guard.
    pub truncated: bool,
}

/// The loaded networks and the tokenizer: everything needed to turn a picture
/// into text.
pub struct Reader {
    tower: Tower,
    projector: Projector,
    decoder: Decoder,
    tokenizer: Tokenizer,
    cfg: ModelConfig,
    image_cfg: ImageConfig,
    device: Device,
    dtype: DType,
    /// Ids that end an answer.
    stops: Vec<u32>,
    /// The id the prompt repeats once per place the picture takes.
    placeholder: u32,
}

impl Reader {
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        tower: Tower,
        projector: Projector,
        decoder: Decoder,
        tokenizer: Tokenizer,
        cfg: ModelConfig,
        image_cfg: ImageConfig,
        device: Device,
        dtype: DType,
    ) -> Result<Self> {
        let id = |token: &str| tokenizer.token_to_id(token);
        let placeholder = id("<|IMAGE_PLACEHOLDER|>").ok_or_else(|| {
            candle_core::Error::Msg(
                "the tokenizer has no <|IMAGE_PLACEHOLDER|> token".into(),
            )
        })?;
        let stops = ["</s>", "<|end_of_sentence|>"]
            .iter()
            .filter_map(|token| id(token))
            .collect::<Vec<_>>();
        Ok(Self {
            tower,
            projector,
            decoder,
            tokenizer,
            cfg,
            image_cfg,
            device,
            dtype,
            stops,
            placeholder,
        })
    }

    pub fn image_config(&self) -> &ImageConfig { &self.image_cfg }

    pub fn device(&self) -> &Device { &self.device }

    /// Reads one prepared picture.
    pub fn read(
        &self,
        picture: &Prepared,
        task: Task,
        limits: &Limits,
    ) -> Result<Answer> {
        let merge = self.cfg.vision.spatial_merge_size;
        let places = picture.tokens(merge);

        // 1. The picture, once: patches → tower → projector.
        let pixels = Tensor::from_vec(
            picture.pixels.clone(),
            (picture.patches(), picture.pixels.len() / picture.patches()),
            &self.device,
        )?
        .to_dtype(self.dtype)?;
        let features = self.tower.forward(&pixels, picture.grid)?;
        let vision = self.projector.forward(&features, picture.grid)?;

        // 2. The prompt around it.
        let (tokens, image_at) = self.prompt(task, places)?;
        let positions = self.positions(tokens.len(), image_at, picture.grid);

        // 3. The picture's tokens take the placeholders' places.
        let embedded = self.decoder.embed(&tokens, &self.device)?;
        let mut parts = Vec::with_capacity(3);
        if image_at > 0 {
            parts.push(embedded.narrow(1, 0, image_at)?);
        }
        parts.push(vision.unsqueeze(0)?);
        let after = image_at + places;
        if after < tokens.len() {
            parts.push(embedded.narrow(1, after, tokens.len() - after)?);
        }
        let inputs = Tensor::cat(&parts, 1)?.contiguous()?;

        // 4. Prefill, then one token at a time.
        let mut cache = Cache::new(self.decoder.layer_count());
        let mut logits =
            self.decoder.forward(&inputs, &positions, &mut cache)?;
        let mut next =
            positions.last().map(|axes| axes[0] + 1).unwrap_or_default();

        let mut generated: Vec<u32> = Vec::new();
        let mut scores: Vec<f32> = Vec::new();
        let mut truncated = false;

        loop {
            let (token, probability) = self.pick(&logits)?;
            if self.stops.contains(&token) {
                break;
            }
            generated.push(token);
            scores.push(probability);

            if let Some(keep) = self.looping(&generated, limits) {
                generated.truncate(keep);
                scores.truncate(keep);
                truncated = true;
                break;
            }
            if generated.len() >= limits.max_tokens {
                truncated = true;
                break;
            }

            let embedded = self.decoder.embed(&[token], &self.device)?;
            logits =
                self.decoder.forward(&embedded, &[[next; 3]], &mut cache)?;
            next += 1;
        }

        let text = self
            .tokenizer
            .decode(&generated, true)
            .map_err(|e| candle_core::Error::Msg(format!("decoding: {e}")))?;
        let score = if scores.is_empty() {
            0.0
        } else {
            scores.iter().sum::<f32>() / scores.len() as f32
        };
        Ok(Answer {
            text: text.trim().to_string(),
            score,
            tokens: generated.len(),
            truncated,
        })
    }

    /// The prompt's token ids, and where the picture's places start.
    pub(crate) fn prompt(
        &self,
        task: Task,
        places: usize,
    ) -> Result<(Vec<u32>, usize)> {
        let encode = |text: &str| -> Result<Vec<u32>> {
            self.tokenizer
                .encode(text, false)
                .map(|encoded| encoded.get_ids().to_vec())
                .map_err(|e| {
                    candle_core::Error::Msg(format!("encoding the prompt: {e}"))
                })
        };
        let prefix = encode("<|begin_of_sentence|>User: <|IMAGE_START|>")?;
        let suffix =
            encode(&format!("<|IMAGE_END|>{}\nAssistant:\n", task.prompt()))?;

        let image_at = prefix.len();
        let mut tokens =
            Vec::with_capacity(prefix.len() + places + suffix.len());
        tokens.extend(prefix);
        tokens.extend(std::iter::repeat_n(self.placeholder, places));
        tokens.extend(suffix);
        Ok((tokens, image_at))
    }

    /// The three-axis position of every prompt token.
    ///
    /// Text runs along all three axes at once; the picture spends one axis on
    /// nothing, one on the patch's row and one on its column, all offset by
    /// where the picture starts. The text after it resumes past the *larger* of
    /// the two spans, so a wide picture and a tall one both leave the sequence
    /// where the picture's furthest extent left it.
    pub(crate) fn positions(
        &self,
        length: usize,
        image_at: usize,
        grid: (usize, usize, usize),
    ) -> Vec<[i64; 3]> {
        let merge = self.cfg.vision.spatial_merge_size;
        let (_, height, width) = grid;
        let (rows, columns) = (height / merge, width / merge);

        let mut positions = Vec::with_capacity(length);
        for index in 0..image_at {
            let at = index as i64;
            positions.push([at, at, at]);
        }
        let base = image_at as i64;
        for row in 0..rows as i64 {
            for column in 0..columns as i64 {
                positions.push([base, base + row, base + column]);
            }
        }
        let mut at = base + rows.max(columns) as i64;
        while positions.len() < length {
            positions.push([at, at, at]);
            at += 1;
        }
        positions
    }

    /// The greedy choice and how sure the model was of it.
    fn pick(&self, logits: &Tensor) -> Result<(u32, f32)> {
        let probabilities = softmax_last_dim(&logits.to_dtype(DType::F32)?)?;
        let token = probabilities.argmax(D::Minus1)?.to_scalar::<u32>()?;
        let probability =
            probabilities.i(token as usize)?.to_scalar::<f32>()?;
        Ok((token, probability))
    }

    /// Where to cut, if the answer has started going in circles.
    ///
    /// Two shapes of loop, because they are two different failures: one token
    /// stuck on repeat is the model losing the thread mid-glyph; a whole phrase
    /// on repeat is it re-reading the same line forever.
    ///
    /// The phrase case is looked for by **period**, not by a fixed window. A
    /// window only catches cycles whose length divides it — a five-token
    /// phrase repeating a hundred times slips straight through a thirty-two
    /// token comparison — and the cycle length is exactly what is not known in
    /// advance.
    fn looping(&self, generated: &[u32], limits: &Limits) -> Option<usize> {
        let last = *generated.last()?;
        let run = generated
            .iter()
            .rev()
            .take_while(|token| **token == last)
            .count();
        if run >= limits.repeat_run {
            return Some(generated.len() - run);
        }

        for period in 1..=limits.repeat_period {
            // Enough tokens to see the phrase several times over, and enough
            // tokens in absolute terms that a genuine echo does not qualify.
            let span = (4 * period).max(limits.repeat_tokens);
            if generated.len() < span {
                continue;
            }
            let tail = &generated[generated.len() - span..];
            if tail.iter().skip(period).zip(tail).all(|(a, b)| a == b) {
                // Keep the first turn of the cycle: it is very likely the text
                // the model meant to emit before it got stuck on it.
                return Some(generated.len() - span + period);
            }
        }
        None
    }
}
