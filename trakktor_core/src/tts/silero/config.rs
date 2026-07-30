//! Geometry of the networks, and the constants the weights do not state.
//!
//! Everything that *can* be read off the checkpoint is: the widths, the block
//! counts, the alphabet and speaker table sizes, even the analysis window all
//! come from the tensors themselves. Deriving them is both shorter and harder
//! to get wrong than restating a table of numbers that must then be kept in
//! step with four different models.
//!
//! What is pinned here is what the tensors genuinely do not say: how the
//! attention splits into heads, the frame rate, and the rules the reference
//! applies to the durations it predicts.

use std::collections::BTreeMap;

use super::error::SileroError;

/// Native sample rate of the vocoder. The other two rates the model offers are
/// derived from this one by its own filterbank, not by resampling.
pub const SAMPLE_RATE: u32 = 48_000;

/// Length of one mel frame, in seconds — `HOP / SAMPLE_RATE`, and the unit the
/// reference states pause lengths in.
pub const FRAME_SECONDS: f64 = 0.0125;

/// Attention heads. Both widths use two, and the weights are one fused matrix,
/// so the split is not visible in them.
pub const HEADS: usize = 2;

/// Epsilon of every normalization in the model.
pub const NORM_EPS: f64 = 1e-5;

/// How far the hourglass decoder shortens the sequence before its middle
/// blocks.
pub const SHORTEN_FACTOR: usize = 3;

/// Ceiling on the vocoder's predicted magnitudes — the reference's own guard
/// against a spectrum that would explode into a click.
pub const MAX_MAGNITUDE: f64 = 1e2;

/// Below this the pitch head's output is taken to mean "no pitch here" and
/// zeroed, which is what stops a rounding-level value from being scaled up by
/// `--pitch`.
pub const PITCH_FLOOR: f32 = 0.001;

/// Kernel of the convolutional feed-forward inside an FFT block.
pub const FFN_KERNEL: usize = 9;

/// Kernel of the vocoder's convolutions (both the input embedding and the
/// depthwise convolution of a ConvNeXt block).
pub const VOCODER_KERNEL: usize = 7;

/// Kernel of the pitch projection.
pub const PITCH_KERNEL: usize = 3;

/// Queries one attention tile covers.
///
/// The decoder attends over mel frames, so its cost and its memory are
/// quadratic in the length of the utterance: at the model's ceiling a single
/// score matrix would be 200 MB. Tiling the queries bounds that at
/// `TILE × frames` without changing a single result — the softmax still runs
/// over a whole row.
pub const ATTENTION_TILE: usize = 1024;

/// Text this many **bytes** of UTF-8 long is spoken in one piece; a paragraph
/// past it is split.
///
/// Cyrillic spends two bytes a letter, so this is around 350 characters, or
/// twenty seconds of speech — a third of what [`Config::max_frames`] allows,
/// and squarely inside the five-to-twenty-second window the reference measures
/// its own speed on. Bytes rather than characters because that is the unit the
/// shared splitting layer counts in.
pub const CHUNK_BUDGET: usize = 700;

/// Durations the reference forces on the last symbols of an utterance,
/// whatever the head predicted: the end-of-speech symbol, the punctuation
/// before it (set outright, not clamped), and the one before that.
pub const TAIL_DURATIONS: [(usize, u32, bool); 3] =
    [(1, 7, false), (2, 13, true), (3, 13, false)];

/// Ceiling on the first symbol's duration.
pub const HEAD_DURATION: u32 = 5;

/// Shapes of a checkpoint's tensors, by name: what the geometry is derived
/// from.
pub type ShapeIndex = BTreeMap<String, Vec<usize>>;

/// Geometry of one loaded model.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Config {
    /// Symbols in the model's alphabet.
    pub symbols: usize,
    /// Rows of the speaker table. One more than the speakers a model names
    /// when it keeps a slot for a generated voice.
    pub speaker_slots: usize,
    /// Width of the acoustic model.
    pub dim: usize,
    /// Inner width of its convolutional feed-forward.
    pub ff_inner: usize,
    /// FFT blocks in the encoder.
    pub encoder_layers: usize,
    /// Width of the duration and pitch predictors.
    pub predictor_dim: usize,
    /// Inner width of their feed-forward.
    pub predictor_ff_inner: usize,
    /// FFT blocks in each predictor.
    pub predictor_layers: usize,
    /// Utterance types the pitch predictor conditions on; zero when the model
    /// has no intonation head.
    pub utterance_types: usize,
    /// Mel channels the acoustic model produces.
    pub mel_channels: usize,
    /// Positions the encoder and decoder tables hold — the hard ceiling on one
    /// pass.
    pub positions: usize,
    /// Width of the vocoder backbone.
    pub vocoder_dim: usize,
    /// Inner width of its ConvNeXt blocks.
    pub vocoder_ff_inner: usize,
    /// ConvNeXt blocks.
    pub vocoder_layers: usize,
    /// FFT size of the inverse transform.
    pub n_fft: usize,
}

impl Config {
    /// Derives the geometry from the tensors a converted model holds.
    ///
    /// # Errors
    ///
    /// Returns [`SileroError::Checkpoint`] when a tensor the geometry is read
    /// from is missing or has an unexpected rank.
    pub fn derive(shapes: &ShapeIndex) -> Result<Self, SileroError> {
        let embedding = dims(shapes, "tacotron.embedding.weight", 2)?;
        let speakers = dims(shapes, "tacotron.speaker_embedding.weight", 2)?;
        let ffn = dims(shapes, "tacotron.encoder.layers.0.conv1.weight", 3)?;
        let pe = dims(shapes, "tacotron.encoder.pos_encoder.pe", 2)?;
        let lin = dims(shapes, "tacotron.lin.weight", 2)?;
        let predictor = dims(
            shapes,
            "dur_predictor.dur_pred.transformer.layers.0.conv1.weight",
            3,
        )?;
        let predictor_embedding =
            dims(shapes, "dur_predictor.dur_pred.embedding.weight", 2)?;
        let vocoder_embed = dims(shapes, "vocoder.backbone.embed.weight", 3)?;
        let vocoder_ffn =
            dims(shapes, "vocoder.backbone.convnext.0.pwconv1.weight", 2)?;
        let head = dims(shapes, "vocoder.head.out.weight", 2)?;

        let encoder_layers = count(shapes, "tacotron.encoder.layers.");
        let predictor_layers =
            count(shapes, "dur_predictor.dur_pred.transformer.layers.");
        let vocoder_layers = count(shapes, "vocoder.backbone.convnext.");
        if encoder_layers == 0 || predictor_layers == 0 || vocoder_layers == 0 {
            return Err(SileroError::Checkpoint(
                "the checkpoint holds no transformer or ConvNeXt blocks".into(),
            ));
        }

        let utterance_types = shapes
            .get("pitch_predictor.pitch_pred.type_embedding.weight")
            .map_or(0, |shape| shape[0]);
        // The head predicts a magnitude and a phase per frequency bin, so its
        // width is `n_fft + 2`.
        let n_fft = head[0].checked_sub(2).ok_or_else(|| {
            SileroError::Checkpoint("the vocoder head is too narrow".into())
        })?;

        let config = Self {
            symbols: embedding[0],
            speaker_slots: speakers[0],
            dim: embedding[1],
            ff_inner: ffn[0],
            encoder_layers,
            predictor_dim: predictor_embedding[1],
            predictor_ff_inner: predictor[0],
            predictor_layers,
            utterance_types,
            mel_channels: lin[0],
            positions: pe[0],
            vocoder_dim: vocoder_embed[0],
            vocoder_ff_inner: vocoder_ffn[0],
            vocoder_layers,
            n_fft,
        };
        for (width, what) in [
            (config.dim, "the acoustic model"),
            (config.predictor_dim, "the predictors"),
        ] {
            if width % HEADS != 0 {
                return Err(SileroError::Checkpoint(format!(
                    "a width of {width} in {what} does not split into {HEADS} \
                     heads"
                )));
            }
        }
        if vocoder_embed[1] != config.mel_channels {
            return Err(SileroError::Checkpoint(format!(
                "the vocoder takes {} mel channels but the acoustic model \
                 makes {}",
                vocoder_embed[1], config.mel_channels
            )));
        }
        Ok(config)
    }

    /// Hop of the inverse transform, in samples. It is what makes a frame
    /// [`FRAME_SECONDS`] long, so it follows from the frame rate rather than
    /// being stated twice.
    #[must_use]
    pub fn hop(&self) -> usize {
        (FRAME_SECONDS * f64::from(SAMPLE_RATE)) as usize
    }

    /// Frequency bins the vocoder head predicts.
    #[must_use]
    pub fn bins(&self) -> usize { self.n_fft / 2 + 1 }

    /// Longest utterance one pass can produce, in mel frames. Past this the
    /// decoder's position table runs out.
    #[must_use]
    pub fn max_frames(&self) -> usize { self.positions }

    /// The same ceiling in seconds of speech.
    #[must_use]
    pub fn max_seconds(&self) -> f64 { self.positions as f64 * FRAME_SECONDS }
}

/// The shape of `name`, checked to have `rank` axes.
fn dims(
    shapes: &ShapeIndex,
    name: &str,
    rank: usize,
) -> Result<Vec<usize>, SileroError> {
    let shape = shapes.get(name).ok_or_else(|| {
        SileroError::Checkpoint(format!("the checkpoint has no `{name}`"))
    })?;
    if shape.len() != rank {
        return Err(SileroError::Checkpoint(format!(
            "`{name}` should have {rank} axes, not {}",
            shape.len()
        )));
    }
    Ok(shape.clone())
}

/// How many indexed children `prefix` has — the depth of a repeated stack.
fn count(shapes: &ShapeIndex, prefix: &str) -> usize {
    let mut highest = None;
    for name in shapes.keys() {
        let Some(rest) = name.strip_prefix(prefix) else {
            continue;
        };
        let Some(index) = rest
            .split('.')
            .next()
            .and_then(|index| index.parse::<usize>().ok())
        else {
            continue;
        };
        highest = Some(highest.map_or(index, |top: usize| top.max(index)));
    }
    highest.map_or(0, |top| top + 1)
}
