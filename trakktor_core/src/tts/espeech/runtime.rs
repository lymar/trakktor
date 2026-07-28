//! The candle-backed runtime.
//!
//! Holds the two networks and the plumbing around them: device selection,
//! checkpoint loading, and the solver state that lives between steps. The
//! vocoder always runs in full precision — it is the stage whose output is
//! compared byte for byte — while the DiT follows the requested
//! [`Precision`](super::Precision).

mod dit;
pub(super) mod layers;
#[cfg(test)]
mod tests;
mod vocos;

use std::path::Path;

pub use candle_core::Device;
use candle_core::{DType, Tensor};
use candle_nn::VarBuilder;
pub use dit::Dit;
pub use vocos::Vocoder;

use super::{
    Precision,
    config::{DitConfig, SAMPLE_RATE, ShapeIndex, VocoderConfig},
    download::{MODEL_WEIGHTS, VOCODER_WEIGHTS},
    error::EspeechError,
    mel::MelBasis,
    model::SpeechModel,
};

/// Guidance below this counts as off — negative values included, exactly as
/// the reference treats them — and the unconditional branch is then skipped
/// rather than computed and ignored.
pub(super) const CFG_OFF: f32 = 1e-5;

/// Maps any backend failure onto the engine's checkpoint error.
pub(super) fn model_err(
    context: &str,
    e: impl std::fmt::Display,
) -> EspeechError {
    EspeechError::Checkpoint(format!("{context}: {e}"))
}

impl Precision {
    /// The tensor type this precision loads weights as.
    pub(super) fn dtype(self) -> DType {
        match self {
            Precision::F16 => DType::F16,
            Precision::F32 => DType::F32,
        }
    }
}

/// The engine's networks on candle, plus the state of the utterance being
/// solved.
pub struct CandleSpeech {
    dit: Dit,
    vocoder: Vocoder,
    basis: MelBasis,
    device: Device,
    solving: Option<Solving>,
}

/// What one utterance carries between solver steps.
struct Solving {
    /// The conditioning, `[1, frames, mel]`, kept for the final splice.
    cond: Tensor,
    /// Frames the conditioning covers.
    cond_frames: usize,
    /// The projected context of every guidance branch, `[branches, frames,
    /// dim]`.
    context: Tensor,
    /// The state being refined, `[1, frames, mel]`.
    x: Tensor,
}

/// Loads a checkpoint directory and the vocoder onto `device`.
///
/// The DiT runs at `precision`; the vocoder always runs in full precision.
///
/// # Errors
///
/// Returns [`EspeechError::Checkpoint`] when either checkpoint is missing,
/// malformed, or of an unexpected geometry.
pub fn load(
    model_dir: &Path,
    vocoder_dir: &Path,
    device: Device,
    precision: Precision,
) -> Result<CandleSpeech, EspeechError> {
    let weights = model_dir.join(MODEL_WEIGHTS);
    let vocoder_weights = vocoder_dir.join(VOCODER_WEIGHTS);
    for path in [&weights, &vocoder_weights] {
        if !path.is_file() {
            return Err(EspeechError::Checkpoint(format!(
                "no {}",
                path.display()
            )));
        }
    }

    let dit_shapes = shapes(&weights)?;
    let dit_cfg = DitConfig::derive(&dit_shapes)?;
    // SAFETY: the checkpoint is memory-mapped read-only; candle requires the
    // file not to be mutated while mapped, which nothing here does.
    let vb = unsafe {
        VarBuilder::from_mmaped_safetensors(
            &[&weights],
            precision.dtype(),
            &device,
        )
        .map_err(|e| model_err(&weights.display().to_string(), e))?
    };
    let dit = Dit::load(dit_cfg, precision.dtype(), vb, &device)
        .map_err(|e| model_err("loading the model", e))?;

    let (vocoder, basis) = load_vocoder(&vocoder_weights, &device)?;
    Ok(CandleSpeech {
        dit,
        vocoder,
        basis,
        device,
        solving: None,
    })
}

/// Loads the vocoder, along with the analysis basis it ships.
fn load_vocoder(
    weights: &Path,
    device: &Device,
) -> Result<(Vocoder, MelBasis), EspeechError> {
    let shapes = shapes(weights)?;
    let cfg = VocoderConfig::derive(&shapes)?;
    // SAFETY: as above — read-only mapping of a file nothing mutates.
    let vb = unsafe {
        VarBuilder::from_mmaped_safetensors(&[weights], DType::F32, device)
            .map_err(|e| model_err(&weights.display().to_string(), e))?
    };
    let vocoder = Vocoder::load(cfg, vb.clone())
        .map_err(|e| model_err("loading the vocoder", e))?;

    // The window and the filterbank ride along with the vocoder's weights; the
    // frontend reads them rather than rebuilding them.
    let read =
        |name: &str, shape: Vec<usize>| -> Result<Vec<f32>, EspeechError> {
            let missing = || {
                EspeechError::Checkpoint(format!(
                    "the vocoder checkpoint has no `{name}`: it is where the \
                     mel filterbank and the analysis window come from"
                ))
            };
            let dims = shapes.get(name).ok_or_else(missing)?;
            if *dims != shape {
                return Err(EspeechError::Checkpoint(format!(
                    "the vocoder's `{name}` is {dims:?}, expected {shape:?}"
                )));
            }
            vb.get(shape.clone(), name)
                .and_then(|tensor| tensor.flatten_all()?.to_vec1::<f32>())
                .map_err(|e| model_err(name, e))
        };
    let bins = cfg.n_fft / 2 + 1;
    let window = read(
        "feature_extractor.mel_spec.spectrogram.window",
        vec![cfg.n_fft],
    )?;
    let filters = read(
        "feature_extractor.mel_spec.mel_scale.fb",
        vec![bins, cfg.mel_channels],
    )?;
    let basis = MelBasis::new(window, filters, cfg.mel_channels)?;
    Ok((vocoder, basis))
}

/// The shapes of every tensor in a safetensors file, for deriving geometry.
pub(super) fn shapes(path: &Path) -> Result<ShapeIndex, EspeechError> {
    // SAFETY: read-only mapping of a file nothing mutates.
    let mapped = unsafe {
        candle_core::safetensors::MmapedSafetensors::new(path)
            .map_err(|e| model_err(&path.display().to_string(), e))?
    };
    Ok(mapped
        .tensors()
        .into_iter()
        .map(|(name, view)| (name, view.shape().to_vec()))
        .collect())
}

impl SpeechModel for CandleSpeech {
    fn config(&self) -> &DitConfig { self.dit.config() }

    fn mel_basis(&self) -> &MelBasis { &self.basis }

    fn sample_rate(&self) -> u32 { SAMPLE_RATE }

    fn prepare(
        &mut self,
        cond: &[f32],
        cond_frames: usize,
        text: &[u32],
        frames: usize,
        noise: &[f32],
    ) -> Result<(), EspeechError> {
        self.solving = None;
        let mel = self.dit.config().mel_channels;
        let dtype = self.dit.dtype();

        let cond_tensor =
            pad_rows(cond, cond_frames, frames, mel, &self.device)
                .map_err(|e| model_err("laying out the conditioning", e))?;
        let (ids, keep) = self
            .text_tensors(text, frames)
            .map_err(|e| model_err("laying out the text", e))?;

        let context = self
            .context(&cond_tensor, &ids, &keep, dtype)
            .map_err(|e| model_err("encoding the conditioning", e))?;
        let x = Tensor::from_slice(noise, (1, frames, mel), &self.device)
            .and_then(|noise| noise.to_dtype(dtype))
            .map_err(|e| model_err("laying out the noise", e))?;

        self.solving = Some(Solving {
            cond: cond_tensor,
            cond_frames,
            context,
            x,
        });
        Ok(())
    }

    fn step(
        &mut self,
        t: f32,
        dt: f32,
        cfg_strength: f32,
    ) -> Result<(), EspeechError> {
        let state = self.solving.as_mut().ok_or_else(|| {
            model_err("stepping the solver", "no utterance prepared")
        })?;
        // Without guidance the unconditional branch is dead weight, so it is
        // not computed at all — that is where "half the work" comes from.
        let context = if cfg_strength < CFG_OFF {
            state
                .context
                .narrow(0, 0, 1)
                .map_err(|e| model_err("dropping the guidance branch", e))?
        } else {
            state.context.clone()
        };
        let flow = self
            .dit
            .forward(&state.x, &context, t)
            .map_err(|e| model_err("predicting the flow", e))?;
        let flow = combine(&flow, cfg_strength)
            .map_err(|e| model_err("combining the guidance branches", e))?;
        state.x = flow
            .affine(f64::from(dt), 0.0)
            .and_then(|scaled| &state.x + scaled)
            .map_err(|e| model_err("advancing the solver", e))?;
        Ok(())
    }

    fn finish(&mut self, cut_frames: usize) -> Result<Vec<f32>, EspeechError> {
        let state = self.solving.take().ok_or_else(|| {
            model_err("finishing the utterance", "no utterance prepared")
        })?;
        let mel = self
            .generated_mel(&state, cut_frames)
            .map_err(|e| model_err("assembling the mel spectrogram", e))?;
        let frames = mel.dim(1).map_err(|e| model_err("the mel", e))?;
        let (magnitude, phase) = self
            .vocoder
            .spectrum(&mel)
            .map_err(|e| model_err("running the vocoder", e))?;
        Ok(self.basis.istft(&magnitude, &phase, frames))
    }
}

impl CandleSpeech {
    /// The mel to vocode: the conditioning over its own frames, the solved
    /// state after them, cut at `cut_frames`.
    fn generated_mel(
        &self,
        state: &Solving,
        cut_frames: usize,
    ) -> candle_core::Result<Tensor> {
        let frames = state.x.dim(1)?;
        // The vocoder runs in full precision, and the conditioning was kept
        // there; a half-precision solve joins it after the cast, not before.
        let solved = state.x.to_dtype(DType::F32)?;
        let spliced = if state.cond_frames >= frames {
            state.cond.clone()
        } else {
            Tensor::cat(
                &[
                    state.cond.narrow(1, 0, state.cond_frames)?,
                    solved.narrow(
                        1,
                        state.cond_frames,
                        frames - state.cond_frames,
                    )?,
                ],
                1,
            )?
        };
        let cut = cut_frames.min(frames);
        spliced.narrow(1, cut, frames - cut)?.contiguous()
    }

    /// The text laid out for the network: ids shifted so that zero is the
    /// filler token, and a mask that holds the padding at zero.
    fn text_tensors(
        &self,
        text: &[u32],
        frames: usize,
    ) -> candle_core::Result<(Tensor, Tensor)> {
        // The reference truncates text longer than the audio it has to fit in.
        let taken = text.len().min(frames);
        let mut ids = Vec::with_capacity(frames);
        ids.extend(text[..taken].iter().map(|id| id + 1));
        ids.resize(frames, 0);
        let keep: Vec<f32> = ids
            .iter()
            .map(|id| if *id == 0 { 0.0 } else { 1.0 })
            .collect();
        Ok((
            Tensor::from_vec(ids, (1, frames), &self.device)?,
            Tensor::from_vec(keep, (1, frames, 1), &self.device)?
                .to_dtype(self.dit.dtype())?,
        ))
    }

    /// Projects the context of both guidance branches: the conditioning and the
    /// encoded text for one, zeros and the filler-token text for the other.
    fn context(
        &self,
        cond: &Tensor,
        ids: &Tensor,
        keep: &Tensor,
        dtype: DType,
    ) -> candle_core::Result<Tensor> {
        let text_cond = self.dit.encode_text(ids, keep, false)?;
        let text_uncond = self.dit.encode_text(ids, keep, true)?;
        let cond = cond.to_dtype(dtype)?;
        let both_cond = Tensor::cat(&[&cond, &cond.zeros_like()?], 0)?;
        let both_text = Tensor::cat(&[&text_cond, &text_uncond], 0)?;
        self.dit.project_context(&both_cond, &both_text)
    }
}

/// Combines the guidance branches: the conditional prediction, pushed away from
/// the unconditional one by `strength`.
///
/// A strength of zero means the unconditional branch was never computed, and
/// the conditional prediction stands alone.
fn combine(flow: &Tensor, strength: f32) -> candle_core::Result<Tensor> {
    let conditional = flow.narrow(0, 0, 1)?;
    if flow.dim(0)? == 1 {
        return Ok(conditional);
    }
    let unconditional = flow.narrow(0, 1, 1)?;
    let difference = (&conditional - &unconditional)?;
    conditional + (difference * f64::from(strength))?
}

/// Lays `rows` out as `[1, frames, width]`, zero-padded past `rows_len`.
fn pad_rows(
    rows: &[f32],
    rows_len: usize,
    frames: usize,
    width: usize,
    device: &Device,
) -> candle_core::Result<Tensor> {
    let mut padded = Vec::with_capacity(frames * width);
    let taken = rows_len.min(frames) * width;
    padded.extend_from_slice(&rows[..taken.min(rows.len())]);
    padded.resize(frames * width, 0.0);
    Tensor::from_vec(padded, (1, frames, width), device)
}

/// Creates the compute device for a run.
///
/// # Errors
///
/// Returns [`EspeechError::InvalidOptions`] when the requested device is not
/// available in this build.
pub fn device(metal: bool) -> Result<Device, EspeechError> {
    if !metal {
        return Ok(Device::Cpu);
    }
    #[cfg(feature = "tts-metal")]
    {
        Device::new_metal(0)
            .map_err(|e| model_err("creating the metal device", e))
    }
    #[cfg(not(feature = "tts-metal"))]
    {
        Err(EspeechError::InvalidOptions(
            "this build has no Metal support; install or build trakktor with \
             the `metal` feature"
                .into(),
        ))
    }
}
