//! The burn-backed ESpeech runtime.
//!
//! An alternative [`SpeechModel`] implementation on [burn](burn), selectable at
//! run time next to the candle one. Backends: ndarray on the CPU, and wgpu with
//! MSL-compiled kernels on Metal, with operator fusion and autotuning.
//!
//! **This runtime computes in f32 only**, on either device: the ndarray backend
//! has no half-precision element at all, and the wgpu one is not asked for it
//! here — full precision is the mode the two runtimes are compared in, and this
//! network is small enough that its weights fit either way.
//!
//! The checkpoint is the same converted directory the candle runtime loads:
//! tensors are read through candle's safetensors reader and converted through
//! f32 into tensors of the target backend. The mel spectrogram and the inverse
//! transform are shared host-side code, so the two runtimes cannot disagree
//! about the model's input or about the last step of its output.
//!
//! The backend choice is erased behind a boxed [`SpeechModel`], so the
//! synthesis driver sees one type.

pub mod net;

use std::path::Path;

use burn::{
    backend::{
        ndarray::{NdArray, NdArrayDevice},
        wgpu::{Metal, WgpuDevice},
    },
    prelude::Int,
    tensor::{Tensor, TensorData, backend::Backend},
};

use super::{
    Precision,
    config::{DitConfig, SAMPLE_RATE, VocoderConfig},
    download::{MODEL_WEIGHTS, VOCODER_WEIGHTS},
    error::EspeechError,
    mel::MelBasis,
    model::SpeechModel,
    runtime::{CFG_OFF, model_err, shapes},
};

/// Lazy access to a checkpoint's tensors through candle's safetensors reader.
pub struct Weights(candle_core::safetensors::MmapedSafetensors);

impl Weights {
    /// Opens one safetensors file.
    fn open(path: &Path) -> Result<Self, EspeechError> {
        if !path.is_file() {
            return Err(EspeechError::Checkpoint(format!(
                "no {}",
                path.display()
            )));
        }
        // SAFETY: the checkpoint is memory-mapped read-only; candle requires
        // the file not to be mutated while mapped, which nothing here does.
        let inner =
            unsafe { candle_core::safetensors::MmapedSafetensors::new(path) }
                .map_err(|e| model_err(&path.display().to_string(), e))?;
        Ok(Self(inner))
    }

    /// The named tensor as f32 values and its shape.
    pub fn parts(
        &self,
        key: &str,
    ) -> Result<(Vec<f32>, Vec<usize>), EspeechError> {
        let tensor = self
            .0
            .load(key, &candle_core::Device::Cpu)
            .map_err(|e| model_err(key, e))?;
        let dims = tensor.dims().to_vec();
        let values = tensor
            .to_dtype(candle_core::DType::F32)
            .and_then(|t| t.flatten_all())
            .and_then(|t| t.to_vec1::<f32>())
            .map_err(|e| model_err(key, e))?;
        Ok((values, dims))
    }
}

/// The engine's networks on one burn backend, plus the state of the utterance
/// being solved.
pub struct BurnSpeech<B: Backend> {
    dit: net::Dit<B>,
    vocoder: net::Vocoder<B>,
    basis: MelBasis,
    device: B::Device,
    solving: Option<Solving<B>>,
}

/// What one utterance carries between solver steps.
struct Solving<B: Backend> {
    cond: Tensor<B, 3>,
    cond_frames: usize,
    context: Tensor<B, 3>,
    x: Tensor<B, 3>,
}

impl<B: Backend> BurnSpeech<B> {
    /// Loads a converted checkpoint and the vocoder onto `device`.
    ///
    /// # Errors
    ///
    /// Returns [`EspeechError::Checkpoint`] when either checkpoint is missing,
    /// malformed, or of an unexpected geometry.
    pub fn load(
        model_dir: &Path,
        vocoder_dir: &Path,
        device: B::Device,
    ) -> Result<Self, EspeechError> {
        let model_path = model_dir.join(MODEL_WEIGHTS);
        let vocoder_path = vocoder_dir.join(VOCODER_WEIGHTS);
        let dit_cfg = DitConfig::derive(&shapes(&model_path)?)?;
        let vocoder_shapes = shapes(&vocoder_path)?;
        let vocoder_cfg = VocoderConfig::derive(&vocoder_shapes)?;

        let model_weights = Weights::open(&model_path)?;
        let vocoder_weights = Weights::open(&vocoder_path)?;
        let dit = net::Dit::load(&model_weights, &device, dit_cfg)?;
        let vocoder =
            net::Vocoder::load(&vocoder_weights, &device, vocoder_cfg)?;

        // The window and the filterbank ride along with the vocoder's weights,
        // and are used on the host — read them as plain values.
        let read = |name: &str, expected: Vec<usize>| {
            let (values, dims) = vocoder_weights.parts(name)?;
            if dims != expected {
                return Err(EspeechError::Checkpoint(format!(
                    "the vocoder's `{name}` is {dims:?}, expected {expected:?}"
                )));
            }
            Ok(values)
        };
        let bins = vocoder_cfg.n_fft / 2 + 1;
        let basis = MelBasis::new(
            read(
                "feature_extractor.mel_spec.spectrogram.window",
                vec![vocoder_cfg.n_fft],
            )?,
            read(
                "feature_extractor.mel_spec.mel_scale.fb",
                vec![bins, vocoder_cfg.mel_channels],
            )?,
            vocoder_cfg.mel_channels,
        )?;

        Ok(Self {
            dit,
            vocoder,
            basis,
            device,
            solving: None,
        })
    }

    /// The mel to vocode: the conditioning over its own frames, the solved
    /// state after them, cut at `cut_frames`.
    fn generated_mel(
        &self,
        state: &Solving<B>,
        cut_frames: usize,
    ) -> Tensor<B, 3> {
        let [_, frames, mel] = state.x.dims();
        let spliced = if state.cond_frames >= frames {
            state.cond.clone()
        } else {
            Tensor::cat(
                vec![
                    state.cond.clone().slice([
                        0..1,
                        0..state.cond_frames,
                        0..mel,
                    ]),
                    state.x.clone().slice([
                        0..1,
                        state.cond_frames..frames,
                        0..mel,
                    ]),
                ],
                1,
            )
        };
        let cut = cut_frames.min(frames);
        spliced.slice([0..1, cut..frames, 0..mel])
    }
}

impl<B: Backend> SpeechModel for BurnSpeech<B> {
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

        let mut padded = Vec::with_capacity(frames * mel);
        let taken = (cond_frames.min(frames) * mel).min(cond.len());
        padded.extend_from_slice(&cond[..taken]);
        padded.resize(frames * mel, 0.0);
        let cond_tensor: Tensor<B, 3> = Tensor::from_data(
            TensorData::new(padded, [1, frames, mel]),
            &self.device,
        );

        // The reference truncates text longer than the audio it has to fit in.
        let taken = text.len().min(frames);
        let mut ids: Vec<i32> = Vec::with_capacity(frames);
        ids.extend(text[..taken].iter().map(|id| *id as i32 + 1));
        ids.resize(frames, 0);
        let keep: Vec<f32> = ids
            .iter()
            .map(|id| if *id == 0 { 0.0 } else { 1.0 })
            .collect();
        let ids: Tensor<B, 2, Int> =
            Tensor::from_data(TensorData::new(ids, [1, frames]), &self.device);
        let keep: Tensor<B, 3> = Tensor::from_data(
            TensorData::new(keep, [1, frames, 1]),
            &self.device,
        );

        let text_cond = self.dit.encode_text(ids.clone(), keep.clone(), false);
        let text_uncond = self.dit.encode_text(ids, keep, true);
        let both_cond =
            Tensor::cat(vec![cond_tensor.clone(), cond_tensor.zeros_like()], 0);
        let both_text = Tensor::cat(vec![text_cond, text_uncond], 0);
        let context = self.dit.project_context(both_cond, both_text);

        let x: Tensor<B, 3> = Tensor::from_data(
            TensorData::new(noise.to_vec(), [1, frames, mel]),
            &self.device,
        );
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
            let [_, frames, dim] = state.context.dims();
            state.context.clone().slice([0..1, 0..frames, 0..dim])
        } else {
            state.context.clone()
        };
        let flow = self.dit.forward(state.x.clone(), context, t);
        let [branches, frames, mel] = flow.dims();
        let conditional = flow.clone().slice([0..1, 0..frames, 0..mel]);
        let guided = if branches == 1 {
            conditional
        } else {
            let unconditional = flow.slice([1..2, 0..frames, 0..mel]);
            let difference = conditional.clone() - unconditional;
            conditional + difference * cfg_strength
        };
        state.x = state.x.clone() + guided * dt;
        Ok(())
    }

    fn finish(&mut self, cut_frames: usize) -> Result<Vec<f32>, EspeechError> {
        let state = self.solving.take().ok_or_else(|| {
            model_err("finishing the utterance", "no utterance prepared")
        })?;
        let mel = self.generated_mel(&state, cut_frames);
        let [_, frames, _] = mel.dims();
        let (magnitude, phase) = self.vocoder.spectrum(mel)?;
        Ok(self.basis.istft(&magnitude, &phase, frames))
    }
}

/// Loads a checkpoint on the burn CPU backend (ndarray).
///
/// # Errors
///
/// Returns [`EspeechError::InvalidOptions`] for half precision; otherwise see
/// [`BurnSpeech::load`].
pub fn load_cpu(
    model_dir: &Path,
    vocoder_dir: &Path,
    precision: Precision,
) -> Result<Box<dyn SpeechModel>, EspeechError> {
    match precision {
        Precision::F32 => Ok(Box::new(BurnSpeech::<NdArray<f32>>::load(
            model_dir,
            vocoder_dir,
            NdArrayDevice::Cpu,
        )?)),
        Precision::F16 => Err(half_precision_unavailable()),
    }
}

/// Loads a checkpoint on the burn Metal backend (wgpu with MSL-compiled
/// kernels).
///
/// # Errors
///
/// Returns [`EspeechError::InvalidOptions`] for half precision; otherwise see
/// [`BurnSpeech::load`].
pub fn load_metal(
    model_dir: &Path,
    vocoder_dir: &Path,
    precision: Precision,
) -> Result<Box<dyn SpeechModel>, EspeechError> {
    match precision {
        Precision::F32 => Ok(Box::new(BurnSpeech::<Metal<f32>>::load(
            model_dir,
            vocoder_dir,
            WgpuDevice::default(),
        )?)),
        Precision::F16 => Err(half_precision_unavailable()),
    }
}

/// Reports the one precision this runtime does not serve.
fn half_precision_unavailable() -> EspeechError {
    EspeechError::InvalidOptions(
        "the burn runtime computes in f32 only; use `--precision f32`, or \
         `--runtime candle` for f16"
            .into(),
    )
}

#[cfg(test)]
mod tests;
