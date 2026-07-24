//! Driving one synthesis end to end.
//!
//! Ties the pieces together: tokenize the text, lay out the two-track prompt,
//! prime the talker with it, then generate frame by frame — the talker picks
//! codebook 0, the code predictor fills the rest, and the finished frame folds
//! back into the talker's next input. When the talker calls an end to the
//! speech, the collected frames go to the codec decoder and come back as a
//! waveform.

use std::path::Path;

use candle_core::{Device, Tensor};

use super::{
    Precision, Sampling, SynthesisOptions,
    config::{GenerationDefaults, ModelConfig, ModelType},
    error::Qwen3TtsError,
    prompt::{self, PromptSpec},
    runtime::{self, CodecDecoder, Talker},
    sampler::{Rule, Sampler},
    tokenizer::TextTokenizer,
};
use crate::tts::{Speech, Voice};

/// How many frames must be produced before the model is allowed to stop.
///
/// The reference holds generation open for two frames; without it a model that
/// opens with the end code would return silence.
const MIN_FRAMES: usize = 2;

/// The result of one synthesis: the audio and what it took to make it.
#[derive(Debug, Clone)]
pub struct Synthesis {
    /// The synthesized audio.
    pub speech: Speech,
    /// How the voice was chosen.
    pub voice: Voice,
    /// The target language, when one was requested.
    pub language: Option<String>,
    /// Frames generated — one per 1/12.5 s of audio.
    pub frames: usize,
}

/// A loaded engine, ready to synthesize.
pub struct Synthesizer {
    config: ModelConfig,
    defaults: GenerationDefaults,
    tokenizer: TextTokenizer,
    talker: Talker,
    codec: CodecDecoder,
    /// The newline that closes the prompt's opening role.
    newline_id: u32,
    /// Codes the talker must never emit for codebook 0.
    forbidden: Vec<u32>,
}

impl Synthesizer {
    /// Loads a checkpoint directory onto `device`.
    ///
    /// The talker and the code predictor run at `precision`; the codec decoder
    /// always runs in full precision.
    ///
    /// # Errors
    ///
    /// Returns [`Qwen3TtsError::InvalidModel`] when the checkpoint is missing
    /// files, malformed, or of an unsupported kind, and
    /// [`Qwen3TtsError::Tokenizer`] when the tokenizer cannot be assembled.
    pub fn load(
        model_dir: &Path,
        device: Device,
        precision: Precision,
    ) -> Result<Self, Qwen3TtsError> {
        let config = ModelConfig::parse(&read(model_dir, "config.json")?)?;
        if config.model_type != ModelType::CustomVoice {
            return Err(Qwen3TtsError::InvalidModel(format!(
                "this checkpoint conditions the voice in a way the engine \
                 does not serve yet ({:?}); use a CustomVoice checkpoint",
                config.model_type
            )));
        }
        let defaults = GenerationDefaults::parse(&read(
            model_dir,
            "generation_config.json",
        )?)?;

        let tokenizer = TextTokenizer::load(
            &model_dir.join("vocab.json"),
            &model_dir.join("merges.txt"),
        )?;
        // The prompt's opening role ends in a newline; ask the tokenizer for
        // it rather than assuming an id.
        let newline_id = *tokenizer.encode("\n")?.first().ok_or_else(|| {
            Qwen3TtsError::Tokenizer(
                "the tokenizer does not encode a newline".into(),
            )
        })?;

        let talker = load_talker(model_dir, &config, &device, precision)?;
        let codec = runtime::load_codec(model_dir, device)?;

        let forbidden = forbidden_codes(&config);
        Ok(Self {
            config,
            defaults,
            tokenizer,
            talker,
            codec,
            newline_id,
            forbidden,
        })
    }

    /// The checkpoint's configuration — its voices, languages, and geometry.
    #[must_use]
    pub fn config(&self) -> &ModelConfig { &self.config }

    /// The sampling defaults the checkpoint ships.
    #[must_use]
    pub fn defaults(&self) -> GenerationDefaults { self.defaults }

    /// Synthesizes `text`.
    ///
    /// # Errors
    ///
    /// Returns [`Qwen3TtsError::TextEmpty`] for empty text,
    /// [`Qwen3TtsError::UnsupportedVoice`] or
    /// [`Qwen3TtsError::UnsupportedLanguage`] when the checkpoint cannot serve
    /// the request, and [`Qwen3TtsError::InvalidModel`] on a backend failure.
    pub fn speak(
        &mut self,
        text: &str,
        options: &SynthesisOptions,
    ) -> Result<Synthesis, Qwen3TtsError> {
        if text.trim().is_empty() {
            return Err(Qwen3TtsError::TextEmpty);
        }
        let speaker_id = self.config.talker.speaker_id(&options.voice)?;
        let language_id =
            self.config.language_id(options.language.as_deref())?;

        let text_ids = self.tokenizer.encode(text)?;
        let positions = prompt::build(
            &self.config,
            &PromptSpec {
                text_ids: &text_ids,
                newline_id: self.newline_id,
                speaker_id: Some(speaker_id),
                language_id,
            },
        );

        let frames = self.generate(&positions, options)?;
        let samples = self
            .codec
            .decode(&frames)
            .map_err(|e| runtime::model_err("decoding the codec frames", e))?;

        Ok(Synthesis {
            speech: Speech {
                samples,
                sample_rate: self.codec.sample_rate(),
            },
            voice: Voice::Preset {
                name: options.voice.clone(),
            },
            language: options.language.clone(),
            frames: frames.len(),
        })
    }

    /// Runs the generation loop, returning the finished frames.
    fn generate(
        &mut self,
        positions: &[prompt::Position],
        options: &SynthesisOptions,
    ) -> Result<Vec<Vec<u32>>, Qwen3TtsError> {
        let fail = |what: &'static str| {
            move |e: candle_core::Error| runtime::model_err(what, e)
        };

        self.talker.reset();
        let prompt_embeds = self
            .embed_prompt(positions)
            .map_err(fail("embedding the prompt"))?;
        // During generation the text track has nothing left to say and simply
        // pads; the codec track carries the frames.
        let pad_embed = self
            .talker
            .embed_text(&[self.config.tts_pad_token_id])
            .map_err(fail("embedding the text padding"))?;

        let (talker_rule, predictor_rule, repetition_penalty, seed) =
            self.rules(options);
        let mut sampler = Sampler::new(seed);

        let (mut logits, mut state) = self
            .talker
            .forward(&prompt_embeds)
            .map_err(fail("priming the talker"))?;

        let eos = self.config.talker.codec_eos_token_id;
        let mut frames: Vec<Vec<u32>> = Vec::new();
        let mut history: Vec<u32> = Vec::new();

        loop {
            let mut scores = logits
                .to_vec1::<f32>()
                .map_err(fail("reading the logits"))?;
            // Holding the speech open for the first frames keeps a model that
            // opens with the end code from returning silence.
            let mut forbidden = self.forbidden.clone();
            if frames.len() < MIN_FRAMES {
                forbidden.push(eos);
            }
            let first = sampler.pick(
                &mut scores,
                talker_rule,
                &history,
                repetition_penalty,
                &forbidden,
            );
            if first == eos || frames.len() >= options.max_frames {
                break;
            }
            history.push(first);

            // The code predictor fills the rest of this frame, conditioned on
            // the talker's state and the code it just picked.
            let mut residual_error = None;
            let residuals = self
                .talker
                .predict_residuals(&state, first, |logits, _step| {
                    let mut scores = logits.to_vec1::<f32>()?;
                    Ok(sampler.pick(&mut scores, predictor_rule, &[], 1.0, &[]))
                })
                .map_err(|e| {
                    residual_error = Some(e);
                    runtime::model_err(
                        "predicting the residual codebooks",
                        residual_error.take().expect("just set"),
                    )
                })?;

            let mut frame = Vec::with_capacity(residuals.len() + 1);
            frame.push(first);
            frame.extend_from_slice(&residuals);

            // The finished frame folds into one embedding and, with the text
            // track's padding, becomes the talker's next input.
            let folded = self
                .talker
                .fold_frame(&frame)
                .map_err(fail("folding the frame"))?;
            let next =
                (folded + &pad_embed).map_err(fail("adding the padding"))?;
            frames.push(frame);

            let stepped = self
                .talker
                .forward(&next)
                .map_err(fail("advancing the talker"))?;
            logits = stepped.0;
            state = stepped.1;
        }

        Ok(frames)
    }

    /// Resolves the sampling rules for both levels.
    ///
    /// The repetition penalty is not part of sampling: the reference applies
    /// it to the logits either way, and without it greedy decoding never
    /// reaches the end-of-speech code on anything but the shortest text.
    fn rules(&self, options: &SynthesisOptions) -> (Rule, Rule, f32, u64) {
        match options.sampling {
            Sampling::Greedy => (
                Rule::Greedy,
                Rule::Greedy,
                self.defaults.repetition_penalty,
                0,
            ),
            Sampling::TopK {
                top_k,
                temperature,
                repetition_penalty,
                seed,
            } => (
                Rule::TopK { top_k, temperature },
                // The code predictor keeps the checkpoint's own settings: it
                // is a separate level with its own defaults.
                Rule::TopK {
                    top_k: self.defaults.predictor_top_k,
                    temperature: self.defaults.predictor_temperature,
                },
                repetition_penalty,
                seed,
            ),
        }
    }

    /// Embeds a laid-out prompt into `[1, positions, hidden]`.
    fn embed_prompt(
        &self,
        positions: &[prompt::Position],
    ) -> candle_core::Result<Tensor> {
        // Every position drives the text track; only the opening role, a
        // prefix, leaves the codec track silent.
        let text_ids: Vec<u32> = positions
            .iter()
            .filter_map(|position| position.text)
            .collect();
        let lead = positions
            .iter()
            .take_while(|position| position.codec.is_none())
            .count();
        let codec_ids: Vec<u32> = positions
            .iter()
            .filter_map(|position| position.codec)
            .collect();
        debug_assert_eq!(text_ids.len(), positions.len());
        debug_assert_eq!(codec_ids.len() + lead, positions.len());

        let text = self.talker.embed_text(&text_ids)?;
        let codec = self.talker.embed_codec(&codec_ids)?;
        let hidden = text.dim(2)?;
        let silent =
            Tensor::zeros((1, lead, hidden), codec.dtype(), codec.device())?;
        text + Tensor::cat(&[&silent, &codec], 1)?
    }
}

/// Reads a file from the checkpoint directory.
fn read(model_dir: &Path, name: &str) -> Result<String, Qwen3TtsError> {
    let path = model_dir.join(name);
    std::fs::read_to_string(&path).map_err(|e| {
        Qwen3TtsError::InvalidModel(format!("reading {}: {e}", path.display()))
    })
}

/// The codes the talker must never emit for codebook 0.
///
/// The tail of the codec vocabulary holds control ids rather than audio codes;
/// only the end-of-speech code is left reachable, so the model can stop.
fn forbidden_codes(config: &ModelConfig) -> Vec<u32> {
    let vocab = config.talker.vocab_size as u32;
    let predictor_vocab = config.talker.code_predictor.vocab_size as u32;
    (predictor_vocab..vocab)
        .filter(|code| *code != config.talker.codec_eos_token_id)
        .collect()
}

/// Loads the talker from a checkpoint directory.
fn load_talker(
    model_dir: &Path,
    config: &ModelConfig,
    device: &Device,
    precision: Precision,
) -> Result<Talker, Qwen3TtsError> {
    let weights = model_dir.join("model.safetensors");
    if !weights.is_file() {
        return Err(Qwen3TtsError::InvalidModel(format!(
            "no {} in the checkpoint",
            weights.display()
        )));
    }
    // SAFETY: the checkpoint is memory-mapped read-only; candle requires the
    // file not to be mutated while mapped, which nothing here does.
    let vb = unsafe {
        candle_nn::VarBuilder::from_mmaped_safetensors(
            &[&weights],
            precision.dtype(),
            device,
        )
        .map_err(|e| runtime::model_err(&weights.display().to_string(), e))?
    };
    Talker::load(&config.talker, vb, device.clone(), precision.dtype())
        .map_err(|e| runtime::model_err("loading the talker", e))
}

#[cfg(test)]
mod tests;
