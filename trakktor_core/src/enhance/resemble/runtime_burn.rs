//! The burn-backed resemble runtimes.
//!
//! Alternative [`EnhanceModel`] implementations on [burn](burn), selectable at
//! run time next to the candle ones. Backends: ndarray on the CPU, and wgpu
//! with MSL-compiled kernels on Metal, with operator fusion and autotuning.
//!
//! **These compute in f32 only**, on either device: the ndarray backend has no
//! half-precision element at all, and f32 is the mode the port is verified in.
//!
//! The checkpoint is the same converted pair the candle runtime loads. The
//! transforms, the mel, the chunking, the solver's time grid and the Gaussian
//! source are shared host-side code, so the two runtimes cannot disagree about
//! what the networks were given or about when the flow model was asked.
//!
//! The backend choice is erased behind a boxed [`EnhanceModel`], so the driver
//! sees one type.

pub mod cfm;
pub mod denoiser;
pub mod irmae;
#[cfg(test)]
mod tests;
pub mod univnet;

use std::path::Path;

use burn::{
    backend::{
        ndarray::{NdArray, NdArrayDevice},
        wgpu::{Metal, WgpuDevice},
    },
    tensor::{Tensor, backend::Backend},
};

use super::{
    EnhancerSettings,
    config::{MELS, Z_SCALE},
    download::{DENOISER_FILE, ENHANCER_FILE},
    mel,
    noise::Source,
    runtime::model_err,
    solver, stft,
};
use crate::enhance::{
    EnhanceError, EnhanceModel, Precision, Runtime,
    resemble::stft::{Prediction, Spectrum},
};

/// Lazy access to a converted checkpoint through candle's safetensors reader.
pub struct Weights(candle_core::safetensors::MmapedSafetensors);

impl Weights {
    /// Opens a converted file.
    fn open(path: &Path) -> Result<Self, EnhanceError> {
        if !path.is_file() {
            return Err(EnhanceError::Checkpoint(format!(
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
    ) -> Result<(Vec<f32>, Vec<usize>), EnhanceError> {
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

    /// The named tensor's values, whatever its shape.
    pub fn values(&self, key: &str) -> Result<Vec<f32>, EnhanceError> {
        Ok(self.parts(key)?.0)
    }
}

/// Reads a checkpoint tensor of the given shape as a burn tensor.
pub fn weight<B: Backend, const D: usize>(
    weights: &Weights,
    device: &B::Device,
    key: &str,
    shape: [usize; D],
) -> Result<Tensor<B, D>, EnhanceError> {
    let (values, dims) = weights.parts(key)?;
    if dims != shape {
        return Err(EnhanceError::Checkpoint(format!(
            "{key}: shape {dims:?}, expected {shape:?}"
        )));
    }
    Ok(Tensor::from_data(
        burn::tensor::TensorData::new(values, shape),
        device,
    ))
}

/// The masking network on one burn backend.
pub struct BurnDenoiser<B: Backend> {
    net: denoiser::Unet<B>,
    device: B::Device,
}

impl<B: Backend> BurnDenoiser<B> {
    /// Loads the converted denoiser from `dir` onto `device`.
    ///
    /// # Errors
    ///
    /// Returns [`EnhanceError::Checkpoint`] when the weights are missing or do
    /// not match the geometry the engine expects.
    pub fn load(dir: &Path, device: B::Device) -> Result<Self, EnhanceError> {
        let weights = Weights::open(&dir.join(DENOISER_FILE))?;
        Ok(Self {
            net: denoiser::Unet::load(&weights, &device, "net")?,
            device,
        })
    }

    /// What the network predicts for one analysed chunk.
    fn predict(&self, spectrum: &Spectrum) -> Prediction {
        self.net.predict(spectrum, &self.device)
    }
}

impl<B: Backend> EnhanceModel for BurnDenoiser<B> {
    fn enhance_window(
        &mut self,
        samples: &[f32],
        _lost: &[bool],
    ) -> Result<Vec<f32>, EnhanceError> {
        let normalized = super::runtime::normalize_peak(samples);
        let spectrum = stft::analyze(&normalized);
        let mut wave =
            stft::synthesize(&stft::apply(&spectrum, &self.predict(&spectrum)));
        wave.resize(samples.len(), 0.0);
        Ok(wave)
    }

    fn runtime(&self) -> Runtime { Runtime::Burn }
}

/// The whole generative pipeline on one burn backend.
pub struct BurnEnhancer<B: Backend> {
    denoiser: denoiser::Unet<B>,
    encoder: irmae::Encoder<B>,
    decoder: irmae::Decoder<B>,
    velocity: cfm::Velocity<B>,
    vocoder: univnet::UnivNet<B>,
    filterbank: Vec<f32>,
    mel_window: Vec<f32>,
    centre: (f32, f32),
    settings: EnhancerSettings,
    noise: Source,
    device: B::Device,
}

impl<B: Backend> BurnEnhancer<B> {
    /// Loads the converted pipeline from `dir` onto `device`.
    ///
    /// # Errors
    ///
    /// Returns [`EnhanceError::Checkpoint`] when the weights are missing or do
    /// not match the geometry the engine expects.
    pub fn load(
        dir: &Path,
        device: B::Device,
        settings: EnhancerSettings,
    ) -> Result<Self, EnhanceError> {
        let weights = Weights::open(&dir.join(ENHANCER_FILE))?;
        let centre = weights.values("mel_centre")?;
        Ok(Self {
            denoiser: denoiser::Unet::load(&weights, &device, "denoiser.net")?,
            encoder: irmae::Encoder::load(&weights, &device)?,
            decoder: irmae::Decoder::load(&weights, &device)?,
            velocity: cfm::Velocity::load(&weights, &device)?,
            vocoder: univnet::UnivNet::load(&weights, &device)?,
            filterbank: weights.values("mel_filterbank")?,
            mel_window: weights.values("mel_window")?,
            centre: (centre[0], centre[1]),
            noise: Source::seeded(settings.seed),
            settings,
            device,
        })
    }

    /// Replaces where the Gaussian draws come from — see
    /// [`EnhancerModel::with_noise`](super::runtime::EnhancerModel::with_noise).
    pub fn with_noise(&mut self, source: Source) { self.noise = source; }

    /// The mel of one waveform, centred, as `[1, MELS, frames]`.
    fn mel(&self, samples: &[f32]) -> Tensor<B, 3> {
        let values = mel::analyze(samples, &self.filterbank, &self.mel_window);
        let count = values.len() / MELS;
        let tensor: Tensor<B, 3> = Tensor::from_data(
            burn::tensor::TensorData::new(values, [1, MELS, count]),
            &self.device,
        );
        (tensor - self.centre.0) / self.centre.1
    }

    /// One chunk, end to end.
    fn run(&mut self, samples: &[f32]) -> Result<Vec<f32>, EnhanceError> {
        let wave = super::runtime::normalize_peak(samples);
        let original = self.mel(&wave);

        let lambda = self.settings.lambda;
        let conditioning = if lambda <= 0.0 {
            original.clone()
        } else {
            let denoised = self.denoise(&wave);
            self.mel(&denoised) * lambda + original.clone() * (1.0 - lambda)
        };

        let encoded = self.encoder.forward(original);
        let start = self.start(encoded);
        let latent = self.solve(conditioning, start);
        let decoded = self.decoder.forward(latent / Z_SCALE);
        let count = decoded.dims()[2];
        let noise = self.noise.draw(univnet::noise_len(count));
        Ok(self.vocoder.forward(decoded, &noise, &self.device))
    }

    /// The denoiser's own pass, which the enhancer uses as an input.
    fn denoise(&self, wave: &[f32]) -> Vec<f32> {
        let normalized = super::runtime::normalize_peak(wave);
        let spectrum = stft::analyze(&normalized);
        let prediction = self.denoiser.predict(&spectrum, &self.device);
        let mut out = stft::synthesize(&stft::apply(&spectrum, &prediction));
        out.resize(wave.len(), 0.0);
        out
    }

    /// Where the walk starts.
    fn start(&mut self, encoded: Tensor<B, 3>) -> Tensor<B, 3> {
        let scaled = encoded * Z_SCALE;
        let tau = self.settings.temperature;
        if tau <= 0.0 {
            return scaled;
        }
        let shape = scaled.dims();
        let drawn = self.noise.draw(shape[0] * shape[1] * shape[2]);
        let noise: Tensor<B, 3> = Tensor::from_data(
            burn::tensor::TensorData::new(drawn, shape),
            &self.device,
        );
        noise * tau + scaled * (1.0 - tau)
    }

    /// Integrates the flow from `start` to the latent the decoder takes.
    fn solve(
        &self,
        conditioning: Tensor<B, 3>,
        start: Tensor<B, 3>,
    ) -> Tensor<B, 3> {
        let prepared = self.velocity.condition(conditioning);
        let method = self.settings.method.resolve(self.settings.nfe);
        let steps = method.steps(self.settings.nfe);
        let times = solver::schedule(steps);
        let mut point = start;
        for index in 0..steps {
            let at = times[index];
            let step = times[index + 1] - at;
            point = match method {
                solver::Method::Euler => {
                    let velocity = self.velocity.forward(
                        point.clone(),
                        at,
                        &prepared,
                        &self.device,
                    );
                    point + velocity * step
                },
                solver::Method::Midpoint => {
                    let first = self.velocity.forward(
                        point.clone(),
                        at,
                        &prepared,
                        &self.device,
                    );
                    let middle = point.clone() + first * (step / 2.0);
                    let second = self.velocity.forward(
                        middle,
                        at + step / 2.0,
                        &prepared,
                        &self.device,
                    );
                    point + second * step
                },
                solver::Method::Rk4 => {
                    let k1 = self.velocity.forward(
                        point.clone(),
                        at,
                        &prepared,
                        &self.device,
                    );
                    let k2 = self.velocity.forward(
                        point.clone() + k1.clone() * (step / 2.0),
                        at + step / 2.0,
                        &prepared,
                        &self.device,
                    );
                    let k3 = self.velocity.forward(
                        point.clone() + k2.clone() * (step / 2.0),
                        at + step / 2.0,
                        &prepared,
                        &self.device,
                    );
                    let k4 = self.velocity.forward(
                        point.clone() + k3.clone() * step,
                        at + step,
                        &prepared,
                        &self.device,
                    );
                    point + (k1 + k2 * 2.0 + k3 * 2.0 + k4) * (step / 6.0)
                },
            };
        }
        point
    }
}

impl<B: Backend> EnhanceModel for BurnEnhancer<B> {
    fn enhance_window(
        &mut self,
        samples: &[f32],
        _lost: &[bool],
    ) -> Result<Vec<f32>, EnhanceError> {
        self.run(samples)
    }

    fn runtime(&self) -> Runtime { Runtime::Burn }
}

/// Loads the denoiser on the burn CPU backend (ndarray).
///
/// # Errors
///
/// Returns [`EnhanceError::InvalidOptions`] for half precision; otherwise see
/// [`BurnDenoiser::load`].
pub fn load_denoiser_cpu(
    dir: &Path,
    precision: Precision,
) -> Result<Box<dyn EnhanceModel>, EnhanceError> {
    match precision {
        Precision::F32 => Ok(Box::new(BurnDenoiser::<NdArray<f32>>::load(
            dir,
            NdArrayDevice::Cpu,
        )?)),
        Precision::F16 => Err(half_precision_unavailable()),
    }
}

/// Loads the denoiser on the burn Metal backend.
///
/// # Errors
///
/// Returns [`EnhanceError::InvalidOptions`] for half precision; otherwise see
/// [`BurnDenoiser::load`].
pub fn load_denoiser_metal(
    dir: &Path,
    precision: Precision,
) -> Result<Box<dyn EnhanceModel>, EnhanceError> {
    match precision {
        Precision::F32 => Ok(Box::new(BurnDenoiser::<Metal<f32>>::load(
            dir,
            WgpuDevice::default(),
        )?)),
        Precision::F16 => Err(half_precision_unavailable()),
    }
}

/// Loads the enhancer on the burn CPU backend (ndarray).
///
/// # Errors
///
/// Returns [`EnhanceError::InvalidOptions`] for half precision; otherwise see
/// [`BurnEnhancer::load`].
pub fn load_enhancer_cpu(
    dir: &Path,
    precision: Precision,
    settings: EnhancerSettings,
) -> Result<Box<dyn EnhanceModel>, EnhanceError> {
    match precision {
        Precision::F32 => Ok(Box::new(BurnEnhancer::<NdArray<f32>>::load(
            dir,
            NdArrayDevice::Cpu,
            settings,
        )?)),
        Precision::F16 => Err(half_precision_unavailable()),
    }
}

/// Loads the enhancer on the burn Metal backend.
///
/// # Errors
///
/// Returns [`EnhanceError::InvalidOptions`] for half precision; otherwise see
/// [`BurnEnhancer::load`].
pub fn load_enhancer_metal(
    dir: &Path,
    precision: Precision,
    settings: EnhancerSettings,
) -> Result<Box<dyn EnhanceModel>, EnhanceError> {
    match precision {
        Precision::F32 => Ok(Box::new(BurnEnhancer::<Metal<f32>>::load(
            dir,
            WgpuDevice::default(),
            settings,
        )?)),
        Precision::F16 => Err(half_precision_unavailable()),
    }
}

/// Reports the one precision this runtime does not serve.
fn half_precision_unavailable() -> EnhanceError {
    EnhanceError::InvalidOptions(
        "the burn runtime computes in f32 only; use `--precision f32`, or \
         `--runtime candle` for f16"
            .into(),
    )
}
