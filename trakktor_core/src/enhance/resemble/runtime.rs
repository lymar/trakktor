//! The candle runtime: loading either network and running one chunk.
//!
//! Two models live here because the project publishes two, in one checkpoint:
//! [`DenoiserModel`] is the masking half on its own — upstream's
//! `--denoise_only` — and [`EnhancerModel`] is the whole generative pipeline,
//! which uses the same denoiser as one of its inputs.
//!
//! Full precision is the default and the only mode parity is claimed in.

pub mod cfm;
pub mod denoiser;
pub mod irmae;
#[cfg(test)]
mod tests;
pub mod univnet;

use std::path::Path;

pub use candle_core::Device;
use candle_core::{DType, Tensor};
use candle_nn::VarBuilder;

use super::{
    EnhancerSettings,
    config::{MELS, PEAK_EPS, Z_SCALE},
    download::{DENOISER_FILE, ENHANCER_FILE},
    mel,
    mel::MEL_BINS,
    noise::Source,
    solver, stft,
};
use crate::enhance::{EnhanceError, EnhanceModel, Precision, Runtime};

/// Creates the compute device, reporting a build without Metal as a validation
/// error rather than falling back silently.
///
/// # Errors
///
/// Returns [`EnhanceError::InvalidOptions`] when Metal is asked for and this
/// build has no Metal support, and [`EnhanceError::Checkpoint`] when the device
/// cannot be created.
pub fn device(metal: bool) -> Result<Device, EnhanceError> {
    if !metal {
        return Ok(Device::Cpu);
    }
    #[cfg(feature = "enhance-metal")]
    {
        Device::new_metal(0).map_err(|e| {
            EnhanceError::Checkpoint(format!("opening the Metal device: {e}"))
        })
    }
    #[cfg(not(feature = "enhance-metal"))]
    Err(EnhanceError::InvalidOptions(
        "this build has no Metal support; rebuild with the `metal` feature, \
         or use `--device cpu`"
            .into(),
    ))
}

/// Wraps a candle failure as a checkpoint error, naming what was being read.
pub(super) fn model_err(what: &str, error: candle_core::Error) -> EnhanceError {
    EnhanceError::Checkpoint(format!("{what}: {error}"))
}

/// Opens a converted file as a variable builder.
fn open(
    path: &Path,
    device: &Device,
    precision: Precision,
) -> Result<VarBuilder<'static>, EnhanceError> {
    if !path.is_file() {
        return Err(EnhanceError::Checkpoint(format!("no {}", path.display())));
    }
    let dtype = match precision {
        Precision::F32 => DType::F32,
        Precision::F16 => DType::F16,
    };
    // SAFETY: the checkpoint is memory-mapped read-only; candle requires the
    // file not to be mutated while mapped, which nothing here does.
    unsafe { VarBuilder::from_mmaped_safetensors(&[path], dtype, device) }
        .map_err(|e| model_err(&path.display().to_string(), e))
}

/// A tensor as host `f32` values.
fn host(tensor: &Tensor) -> Result<Vec<f32>, EnhanceError> {
    tensor
        .to_dtype(DType::F32)
        .and_then(|t| t.flatten_all()?.to_vec1::<f32>())
        .map_err(|e| EnhanceError::Compute(e.to_string()))
}

/// The reference's `_normalize_wav`: divide by the peak, guarded by a floor
/// **added** to it rather than clamped — so a silent chunk stays silent instead
/// of being amplified.
pub(super) fn normalize_peak(samples: &[f32]) -> Vec<f32> {
    let peak = samples
        .iter()
        .fold(0f32, |peak, &sample| peak.max(sample.abs())) +
        PEAK_EPS;
    samples.iter().map(|&sample| sample / peak).collect()
}

/// The masking network on its own.
pub struct DenoiserModel {
    net: denoiser::Unet,
}

impl DenoiserModel {
    /// Loads the converted denoiser from `dir` onto `device`.
    ///
    /// # Errors
    ///
    /// Returns [`EnhanceError::Checkpoint`] when the weights are missing or do
    /// not match the geometry the engine expects.
    pub fn load(
        dir: &Path,
        device: Device,
        precision: Precision,
    ) -> Result<Self, EnhanceError> {
        let vb = open(&dir.join(DENOISER_FILE), &device, precision)?;
        Ok(Self {
            net: denoiser::Unet::load(vb.pp("net"))
                .map_err(|e| model_err("the denoiser", e))?,
        })
    }

    /// The network, for the parity tests.
    #[cfg(test)]
    pub(crate) fn net(&self) -> &denoiser::Unet { &self.net }

    /// One chunk of already-normalized samples, denoised.
    fn run(&self, samples: &[f32]) -> Result<Vec<f32>, EnhanceError> {
        let normalized = normalize_peak(samples);
        let spectrum = stft::analyze(&normalized);
        let prediction = self
            .net
            .predict(&spectrum)
            .map_err(|e| EnhanceError::Compute(e.to_string()))?;
        let mut wave = stft::synthesize(&stft::apply(&spectrum, &prediction));
        // The reference pads its output back to the length that went in; the
        // transform returns whole frames, which is never longer.
        wave.resize(samples.len(), 0.0);
        Ok(wave)
    }
}

impl EnhanceModel for DenoiserModel {
    fn enhance_window(
        &mut self,
        samples: &[f32],
        _lost: &[bool],
    ) -> Result<Vec<f32>, EnhanceError> {
        self.run(samples)
    }

    fn runtime(&self) -> Runtime { Runtime::Candle }
}

/// The whole generative pipeline.
pub struct EnhancerModel {
    denoiser: denoiser::Unet,
    encoder: irmae::Encoder,
    decoder: irmae::Decoder,
    velocity: cfm::Velocity,
    vocoder: univnet::UnivNet,
    /// The mel filterbank, as the checkpoint carries it.
    filterbank: Vec<f32>,
    /// And its analysis window, likewise — see [`mel`](super::mel) for why it
    /// is read rather than computed.
    mel_window: Vec<f32>,
    /// The mean and standard deviation the mel is centred by, learned during
    /// training and stored as two scalars.
    centre: (f64, f64),
    settings: EnhancerSettings,
    noise: Source,
    device: Device,
    dtype: DType,
}

impl EnhancerModel {
    /// Loads the converted pipeline from `dir` onto `device`.
    ///
    /// # Errors
    ///
    /// Returns [`EnhanceError::Checkpoint`] when the weights are missing or do
    /// not match the geometry the engine expects.
    pub fn load(
        dir: &Path,
        device: Device,
        precision: Precision,
        settings: EnhancerSettings,
    ) -> Result<Self, EnhanceError> {
        let vb = open(&dir.join(ENHANCER_FILE), &device, precision)?;
        let filterbank = vb
            .get((MEL_BINS, MELS), "mel_filterbank")
            .and_then(|t| {
                t.to_dtype(DType::F32)?.flatten_all()?.to_vec1::<f32>()
            })
            .map_err(|e| model_err("the mel filterbank", e))?;
        let mel_window = vb
            .get(super::config::MEL_N_FFT, "mel_window")
            .and_then(|t| t.to_dtype(DType::F32)?.to_vec1::<f32>())
            .map_err(|e| model_err("the mel window", e))?;
        let centre = vb
            .get(2, "mel_centre")
            .and_then(|t| t.to_dtype(DType::F32)?.to_vec1::<f32>())
            .map_err(|e| model_err("the mel centring", e))?;
        Ok(Self {
            denoiser: denoiser::Unet::load(vb.pp("denoiser.net"))
                .map_err(|e| model_err("the denoiser", e))?,
            encoder: irmae::Encoder::load(vb.pp("lcfm.ae.encoder"))
                .map_err(|e| model_err("the encoder", e))?,
            decoder: irmae::Decoder::load(vb.pp("lcfm.ae.decoder"))
                .map_err(|e| model_err("the decoder", e))?,
            velocity: cfm::Velocity::load(vb.pp("lcfm.cfm.net"))
                .map_err(|e| model_err("the flow model", e))?,
            vocoder: univnet::UnivNet::load(vb.pp("vocoder"))
                .map_err(|e| model_err("the vocoder", e))?,
            filterbank,
            mel_window,
            centre: (f64::from(centre[0]), f64::from(centre[1])),
            noise: Source::seeded(settings.seed),
            settings,
            device,
            dtype: match precision {
                Precision::F32 => DType::F32,
                Precision::F16 => DType::F16,
            },
        })
    }

    /// The vocoder, for the parity tests.
    #[cfg(test)]
    pub(crate) fn vocoder(&self) -> &univnet::UnivNet { &self.vocoder }

    /// The device it is on, likewise.
    #[cfg(test)]
    pub(crate) fn device(&self) -> &Device { &self.device }

    /// Replaces where the Gaussian draws come from.
    ///
    /// A run uses a stream fixed by `--seed`. Handing it
    /// [`Source::Scripted`](super::noise::Source::Scripted) instead is how the
    /// reference's own draws are fed in, which is the only way two generative
    /// pipelines can be compared sample by sample at all.
    pub fn with_noise(&mut self, source: Source) { self.noise = source; }

    /// The mel of one waveform, centred as the network expects it, as
    /// `[1, MELS, frames]`.
    fn mel(&self, samples: &[f32]) -> Result<Tensor, EnhanceError> {
        let values = mel::analyze(samples, &self.filterbank, &self.mel_window);
        let count = values.len() / MELS;
        Tensor::from_vec(values, (1, MELS, count), &self.device)
            .and_then(|t| t.to_dtype(self.dtype))
            .and_then(|t| (t - self.centre.0)? / self.centre.1)
            .map_err(|e| EnhanceError::Compute(e.to_string()))
    }

    /// One chunk, end to end.
    fn run(&mut self, samples: &[f32]) -> Result<Vec<f32>, EnhanceError> {
        self.run_recording(samples, None)
    }

    /// Every stage of one run, for the parity tests. The names are the
    /// reference stand's.
    #[cfg(test)]
    pub(crate) fn stages(
        &mut self,
        samples: &[f32],
    ) -> Result<Vec<(String, Vec<f32>)>, EnhanceError> {
        let mut stages = Vec::new();
        let out = self.run_recording(samples, Some(&mut stages))?;
        stages.push(("chunk_out".to_owned(), out));
        Ok(stages)
    }

    /// One chunk, recording what it passed through on the way if asked.
    fn run_recording(
        &mut self,
        samples: &[f32],
        mut stages: Option<&mut Vec<(String, Vec<f32>)>>,
    ) -> Result<Vec<f32>, EnhanceError> {
        let compute =
            |e: candle_core::Error| EnhanceError::Compute(e.to_string());
        let wave = normalize_peak(samples);
        let original = self.mel(&wave)?;
        if let Some(stages) = stages.as_deref_mut() {
            stages.push((
                "mel_0".to_owned(),
                mel::analyze(&wave, &self.filterbank, &self.mel_window),
            ));
            stages.push(("mel_norm_0".to_owned(), host(&original)?));
        }

        // The denoised mel is what the flow model is conditioned on; how much
        // of it is used rather than the recording's own is `--denoise`.
        let lambda = f64::from(self.settings.lambda);
        let conditioning = if lambda <= 0.0 {
            original.clone()
        } else {
            let denoised = self.denoise(&wave)?;
            let mel = self.mel(&denoised)?;
            if let Some(stages) = stages.as_deref_mut() {
                stages.push((
                    "mel_1".to_owned(),
                    mel::analyze(&denoised, &self.filterbank, &self.mel_window),
                ));
                stages.push(("mel_norm_1".to_owned(), host(&mel)?));
            }
            ((mel * lambda).map_err(compute)? +
                (original.clone() * (1.0 - lambda)).map_err(compute)?)
            .map_err(compute)?
        };

        let encoded = self.encoder.forward(&original).map_err(compute)?;
        if let Some(stages) = stages.as_deref_mut() {
            stages.push(("ae_encoder".to_owned(), host(&encoded)?));
        }
        let start = self.start(&encoded)?;
        let latent = self.solve(&conditioning, start)?;
        let decoded = self
            .decoder
            .forward(&(latent / f64::from(Z_SCALE)).map_err(compute)?)
            .map_err(compute)?;
        if let Some(stages) = stages.as_deref_mut() {
            stages.push(("ae_decoder".to_owned(), host(&decoded)?));
        }
        let count = decoded.dim(2).map_err(compute)?;
        let noise = self.noise.draw(univnet::UnivNet::noise_len(count));
        // The vocoder returns whole conditioning frames, which is always longer
        // than the chunk that went in; the driver trims it.
        match stages {
            Some(stages) => {
                let (collected, wave) =
                    self.vocoder.stages(&decoded, &noise).map_err(compute)?;
                for (name, tensor) in collected {
                    stages.push((name, host(&tensor)?));
                }
                Ok(wave)
            },
            None => self.vocoder.forward(&decoded, &noise).map_err(compute),
        }
    }

    /// The denoiser's own pass, which the enhancer uses as an input rather than
    /// as an output.
    fn denoise(&self, wave: &[f32]) -> Result<Vec<f32>, EnhanceError> {
        let normalized = normalize_peak(wave);
        let spectrum = stft::analyze(&normalized);
        let prediction = self
            .denoiser
            .predict(&spectrum)
            .map_err(|e| EnhanceError::Compute(e.to_string()))?;
        let mut out = stft::synthesize(&stft::apply(&spectrum, &prediction));
        out.resize(wave.len(), 0.0);
        Ok(out)
    }

    /// Where the walk starts: the encoding of the recording as it is, mixed
    /// with noise in the proportion `--temperature` names.
    fn start(&mut self, encoded: &Tensor) -> Result<Tensor, EnhanceError> {
        let compute =
            |e: candle_core::Error| EnhanceError::Compute(e.to_string());
        let scaled = (encoded.clone() * f64::from(Z_SCALE)).map_err(compute)?;
        let tau = f64::from(self.settings.temperature);
        if tau <= 0.0 {
            return Ok(scaled);
        }
        let shape = scaled.dims3().map_err(compute)?;
        let count = shape.0 * shape.1 * shape.2;
        let drawn = self.noise.draw(count);
        let noise = Tensor::from_vec(drawn, shape, &self.device)
            .and_then(|t| t.to_dtype(self.dtype))
            .map_err(compute)?;
        ((noise * tau).map_err(compute)? +
            (scaled * (1.0 - tau)).map_err(compute)?)
        .map_err(compute)
    }

    /// Integrates the flow from `start` to the latent the decoder takes.
    fn solve(
        &self,
        conditioning: &Tensor,
        start: Tensor,
    ) -> Result<Tensor, EnhanceError> {
        let compute =
            |e: candle_core::Error| EnhanceError::Compute(e.to_string());
        let prepared =
            self.velocity.condition(conditioning).map_err(compute)?;
        let method = self.settings.method.resolve(self.settings.nfe);
        let steps = method.steps(self.settings.nfe);
        let times = solver::schedule(steps);
        let mut point = start;
        for index in 0..steps {
            let at = times[index];
            let step = times[index + 1] - at;
            let velocity = |point: &Tensor, at: f64| {
                self.velocity.forward(point, at, &prepared)
            };
            point = match method {
                solver::Method::Euler => (&point +
                    (velocity(&point, at).map_err(compute)? * step)
                        .map_err(compute)?)
                .map_err(compute)?,
                solver::Method::Midpoint => {
                    let first = velocity(&point, at).map_err(compute)?;
                    let middle = (&point +
                        (first * (step / 2.0)).map_err(compute)?)
                    .map_err(compute)?;
                    let second =
                        velocity(&middle, at + step / 2.0).map_err(compute)?;
                    (&point + (second * step).map_err(compute)?)
                        .map_err(compute)?
                },
                solver::Method::Rk4 => {
                    let k1 = velocity(&point, at).map_err(compute)?;
                    let p2 = (&point +
                        (&k1 * (step / 2.0)).map_err(compute)?)
                    .map_err(compute)?;
                    let k2 = velocity(&p2, at + step / 2.0).map_err(compute)?;
                    let p3 = (&point +
                        (&k2 * (step / 2.0)).map_err(compute)?)
                    .map_err(compute)?;
                    let k3 = velocity(&p3, at + step / 2.0).map_err(compute)?;
                    let p4 = (&point + (&k3 * step).map_err(compute)?)
                        .map_err(compute)?;
                    let k4 = velocity(&p4, at + step).map_err(compute)?;
                    let sum = ((k1 + (k2 * 2.0).map_err(compute)?)
                        .map_err(compute)? +
                        ((k3 * 2.0).map_err(compute)? + k4)
                            .map_err(compute)?)
                    .map_err(compute)?;
                    (&point + (sum * (step / 6.0)).map_err(compute)?)
                        .map_err(compute)?
                },
            };
        }
        Ok(point)
    }
}

impl EnhanceModel for EnhancerModel {
    fn enhance_window(
        &mut self,
        samples: &[f32],
        _lost: &[bool],
    ) -> Result<Vec<f32>, EnhanceError> {
        self.run(samples)
    }

    fn runtime(&self) -> Runtime { Runtime::Candle }
}
