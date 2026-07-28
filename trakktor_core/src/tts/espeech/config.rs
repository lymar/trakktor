//! Geometry of the two networks, and the constants that come from the
//! reference rather than from the weights.
//!
//! Everything that *can* be read off the checkpoint is read off it: the
//! published `MODEL_CFG` is six numbers, and the rest of the architecture comes
//! from the defaults of whatever F5-TTS version trained the checkpoint —
//! deriving the shapes from the tensors themselves is both shorter and harder
//! to get wrong than restating them. What is left pinned here is what the
//! weights genuinely do not say: the head split of the attention, the rotary
//! base, and the audio geometry the model was trained against.

use std::collections::BTreeMap;

use super::error::EspeechError;

/// Sample rate of everything in this engine: the reference is resampled to it
/// and the vocoder produces it.
pub const SAMPLE_RATE: u32 = 24_000;

/// Length of the analysis window and of the FFT.
pub const N_FFT: usize = 1024;

/// Hop between mel frames, in samples — one frame is 1/93.75 s of audio.
pub const HOP: usize = 256;

/// Floor the mel magnitudes are clamped to before the logarithm.
pub const MEL_FLOOR: f64 = 1e-5;

/// Heads the attention splits into is not visible in the weights (they are one
/// fused matrix), but the head size is fixed across the F5-TTS family.
pub const DIM_HEAD: usize = 64;

/// Base of the rotary embedding, and of the sinusoidal text positions.
pub const ROPE_THETA: f64 = 10_000.0;

/// Scale the timestep is multiplied by before the sinusoidal embedding.
pub const TIME_SCALE: f64 = 1000.0;

/// Loudness the reference is normalized to before the mel, when it is quieter
/// than this. The synthesized waveform is scaled back by the same factor, so a
/// quiet reference does not force a quiet result.
pub const TARGET_RMS: f32 = 0.1;

/// Level below which the edges of the reference count as silence, in dBFS.
pub const SILENCE_DBFS: f32 = -42.0;

/// Longest reference the model was trained to condition on, in seconds.
pub const REF_MAX_SECONDS: f64 = 12.0;

/// Silence appended to the trimmed reference, in seconds — the reference does
/// the same, and it gives the model a clean boundary to continue from.
pub const REF_TAIL_SILENCE: f64 = 0.05;

/// Seconds one utterance stays safely inside, reference included. The chunk
/// budget is what is left of it after the reference.
pub const UTTERANCE_WINDOW: f64 = 22.0;

/// Text this short (in bytes) is spoken with a fixed slow rate rather than the
/// requested one: a handful of characters against a whole reference makes the
/// duration estimate meaningless, and the reference guards it the same way.
pub const SHORT_TEXT_BYTES: usize = 10;

/// The rate short text is spoken at, whatever `--speed` says.
pub const SHORT_TEXT_SPEED: f32 = 0.3;

/// Shapes of a checkpoint's tensors, by name: what the geometry is derived
/// from.
pub type ShapeIndex = BTreeMap<String, Vec<usize>>;

/// Geometry of the DiT — the network that turns noise into a mel spectrogram.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct DitConfig {
    /// Width of the residual stream.
    pub dim: usize,
    /// Transformer blocks.
    pub depth: usize,
    /// Attention heads: `dim / DIM_HEAD`.
    pub heads: usize,
    /// Inner width of the feed-forward block.
    pub ff_inner: usize,
    /// Mel channels in and out.
    pub mel_channels: usize,
    /// Width of the text embedding.
    pub text_dim: usize,
    /// Inner width of the text encoder's ConvNeXt blocks.
    pub text_ff_inner: usize,
    /// ConvNeXt blocks in the text encoder.
    pub text_conv_layers: usize,
    /// Text vocabulary, without the filler token.
    pub vocab_size: usize,
    /// Width of the sinusoidal timestep embedding.
    pub time_dim: usize,
    /// Kernel of the convolutional position embedding.
    pub conv_pos_kernel: usize,
    /// Groups of the convolutional position embedding.
    pub conv_pos_groups: usize,
}

impl DitConfig {
    /// Derives the geometry from the tensors a checkpoint holds.
    ///
    /// # Errors
    ///
    /// Returns [`EspeechError::Checkpoint`] when a tensor the geometry is read
    /// from is missing or has an unexpected rank.
    pub fn derive(shapes: &ShapeIndex) -> Result<Self, EspeechError> {
        let proj_out = dims(shapes, "transformer.proj_out.weight", 2)?;
        let mel_channels = proj_out[0];
        let dim = proj_out[1];
        let text_embed =
            dims(shapes, "transformer.text_embed.text_embed.weight", 2)?;
        let text_dim = text_embed[1];
        // The table holds one extra row: index 0 is the filler token that pads
        // the text out to the audio length.
        let vocab_size = text_embed[0].checked_sub(1).ok_or_else(|| {
            EspeechError::Checkpoint("the text embedding is empty".into())
        })?;
        let ff = dims(
            shapes,
            "transformer.transformer_blocks.0.ff.ff.0.0.weight",
            2,
        )?;
        let text_pw = dims(
            shapes,
            "transformer.text_embed.text_blocks.0.pwconv1.weight",
            2,
        )?;
        let time_mlp =
            dims(shapes, "transformer.time_embed.time_mlp.0.weight", 2)?;
        let conv_pos = dims(
            shapes,
            "transformer.input_embed.conv_pos_embed.conv1d.0.weight",
            3,
        )?;

        let depth = count(shapes, "transformer.transformer_blocks.");
        let text_conv_layers =
            count(shapes, "transformer.text_embed.text_blocks.");
        if depth == 0 || text_conv_layers == 0 {
            return Err(EspeechError::Checkpoint(
                "the checkpoint holds no transformer blocks".into(),
            ));
        }
        if dim % DIM_HEAD != 0 {
            return Err(EspeechError::Checkpoint(format!(
                "a width of {dim} does not split into heads of {DIM_HEAD}"
            )));
        }
        Ok(Self {
            dim,
            depth,
            heads: dim / DIM_HEAD,
            ff_inner: ff[0],
            mel_channels,
            text_dim,
            text_ff_inner: text_pw[0],
            text_conv_layers,
            vocab_size,
            time_dim: time_mlp[1],
            conv_pos_kernel: conv_pos[2],
            conv_pos_groups: dim / conv_pos[1],
        })
    }
}

/// Geometry of the vocoder — the network that turns a mel spectrogram back
/// into a waveform.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct VocoderConfig {
    /// Mel channels it takes.
    pub mel_channels: usize,
    /// Width of the backbone.
    pub dim: usize,
    /// Inner width of its ConvNeXt blocks.
    pub ff_inner: usize,
    /// ConvNeXt blocks.
    pub layers: usize,
    /// FFT size of the inverse transform, and of the mel frontend.
    pub n_fft: usize,
}

impl VocoderConfig {
    /// Derives the geometry from the vocoder's tensors.
    ///
    /// # Errors
    ///
    /// Returns [`EspeechError::Checkpoint`] when a tensor the geometry is read
    /// from is missing or has an unexpected rank.
    pub fn derive(shapes: &ShapeIndex) -> Result<Self, EspeechError> {
        let embed = dims(shapes, "backbone.embed.weight", 3)?;
        let pw = dims(shapes, "backbone.convnext.0.pwconv1.weight", 2)?;
        let out = dims(shapes, "head.out.weight", 2)?;
        let layers = count(shapes, "backbone.convnext.");
        if layers == 0 {
            return Err(EspeechError::Checkpoint(
                "the vocoder holds no ConvNeXt blocks".into(),
            ));
        }
        // The head predicts a magnitude and a phase per frequency bin, so its
        // width is `n_fft + 2`.
        let n_fft = out[0].checked_sub(2).ok_or_else(|| {
            EspeechError::Checkpoint("the vocoder head is too narrow".into())
        })?;
        Ok(Self {
            mel_channels: embed[1],
            dim: embed[0],
            ff_inner: pw[0],
            layers,
            n_fft,
        })
    }
}

/// The shape of `name`, checked to have `rank` axes.
fn dims(
    shapes: &ShapeIndex,
    name: &str,
    rank: usize,
) -> Result<Vec<usize>, EspeechError> {
    let shape = shapes.get(name).ok_or_else(|| {
        EspeechError::Checkpoint(format!("the checkpoint has no `{name}`"))
    })?;
    if shape.len() != rank {
        return Err(EspeechError::Checkpoint(format!(
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

#[cfg(test)]
mod tests;
