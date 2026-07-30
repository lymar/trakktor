//! The burn-backed Silero runtime.
//!
//! An alternative [`SpeechModel`] implementation on [burn](burn), selectable at
//! run time next to the candle one. Backends: ndarray on the CPU, and wgpu with
//! MSL-compiled kernels on Metal, with operator fusion and autotuning.
//!
//! **f32 only**, as on candle and for the same reason: this network is small,
//! half precision buys it nothing, and full precision is the mode the two
//! runtimes are compared in.
//!
//! The model is the same converted directory the candle runtime loads —
//! tensors read through candle's safetensors reader and converted through f32
//! — and the two stages that decide the shape of the result, the duration
//! rules and the inverse transform, are shared host-side code. So the runtimes
//! cannot disagree about how long a word is or about the last step of the
//! output; what is left to differ is the arithmetic in between.

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
    config::Config,
    download::WEIGHTS_FILE,
    error::SileroError,
    istft::{Pqmf, Window, split_spectrum},
    model::{self, Spectrum, SpeechModel, Utterance},
};

/// Lazy access to a converted model's tensors through candle's safetensors
/// reader.
pub struct Weights(candle_core::safetensors::MmapedSafetensors);

impl Weights {
    /// Opens one safetensors file.
    fn open(path: &Path) -> Result<Self, SileroError> {
        if !path.is_file() {
            return Err(SileroError::Checkpoint(format!(
                "no {}",
                path.display()
            )));
        }
        // SAFETY: the file is memory-mapped read-only and nothing mutates it.
        let inner =
            unsafe { candle_core::safetensors::MmapedSafetensors::new(path) }
                .map_err(|e| {
                SileroError::Checkpoint(format!(
                    "reading {}: {e}",
                    path.display()
                ))
            })?;
        Ok(Self(inner))
    }

    /// The named tensor as f32 values and its shape.
    pub fn parts(
        &self,
        key: &str,
    ) -> Result<(Vec<f32>, Vec<usize>), SileroError> {
        let tensor = self
            .0
            .load(key, &candle_core::Device::Cpu)
            .map_err(|e| SileroError::Checkpoint(format!("{key}: {e}")))?;
        let dims = tensor.dims().to_vec();
        let values = tensor
            .to_dtype(candle_core::DType::F32)
            .and_then(|t| t.flatten_all())
            .and_then(|t| t.to_vec1::<f32>())
            .map_err(|e| SileroError::Checkpoint(format!("{key}: {e}")))?;
        Ok((values, dims))
    }

    /// The named tensor as plain values, checked against a shape.
    fn values(
        &self,
        key: &str,
        expected: &[usize],
    ) -> Result<Vec<f32>, SileroError> {
        let (values, dims) = self.parts(key)?;
        if dims != expected {
            return Err(SileroError::Checkpoint(format!(
                "{key}: shape {dims:?}, expected {expected:?}"
            )));
        }
        Ok(values)
    }
}

/// The engine's networks on one burn backend.
pub struct BurnSpeech<B: Backend> {
    config: Config,
    device: B::Device,
    acoustic: net::Acoustic<B>,
    durations: net::SeriesPredictor<B>,
    pitch: net::SeriesPredictor<B>,
    vocoder: net::Vocoder<B>,
    mean_std: Vec<f32>,
    window: Window,
    filterbanks: Vec<Pqmf>,
}

impl<B: Backend> BurnSpeech<B> {
    /// Loads a converted model directory onto `device`.
    ///
    /// # Errors
    ///
    /// Returns [`SileroError::Checkpoint`] when the weights are missing or do
    /// not match the geometry derived from them.
    pub fn load(dir: &Path, device: B::Device) -> Result<Self, SileroError> {
        let path = dir.join(WEIGHTS_FILE);
        let config = Config::derive(&super::runtime::shape_index(&path)?)?;
        let weights = Weights::open(&path)?;
        let has_types = config.utterance_types > 0;

        let mut filterbanks = Vec::new();
        for (bands, name) in [(2usize, "pqmf_2"), (6, "pqmf_6")] {
            let key = format!("vocoder.{name}.filters");
            let Ok((filters, dims)) = weights.parts(&key) else {
                continue;
            };
            let [_, taps] = dims[..] else { continue };
            filterbanks.push(Pqmf {
                filters,
                bands,
                taps,
            });
        }

        Ok(Self {
            acoustic: net::Acoustic::load(&weights, &device, &config)?,
            durations: net::SeriesPredictor::load(
                &weights,
                &device,
                "dur_predictor.dur_pred",
                &config,
                false,
            )?,
            pitch: net::SeriesPredictor::load(
                &weights,
                &device,
                "pitch_predictor.pitch_pred",
                &config,
                has_types,
            )?,
            vocoder: net::Vocoder::load(&weights, &device, &config)?,
            mean_std: weights
                .values("pitch.mean_std_coef", &[config.speaker_slots])?,
            window: Window {
                samples: weights
                    .values("vocoder.head.istft.window", &[config.n_fft])?,
                hop: config.hop(),
            },
            filterbanks,
            config,
            device,
        })
    }

    /// A host-side index vector as a backend tensor.
    fn indices(&self, values: &[u32]) -> Tensor<B, 1, Int> {
        let values: Vec<i32> = values.iter().map(|v| *v as i32).collect();
        Tensor::from_data(
            TensorData::new(values.clone(), [values.len()]),
            &self.device,
        )
    }
}

impl<B: Backend> SpeechModel for BurnSpeech<B> {
    fn config(&self) -> &Config { &self.config }

    fn window(&self) -> &Window { &self.window }

    fn pqmf(&self, bands: usize) -> Option<&Pqmf> {
        self.filterbanks.iter().find(|bank| bank.bands == bands)
    }

    fn synthesize(
        &self,
        utterance: &Utterance,
    ) -> Result<Spectrum, SileroError> {
        model::check_input(utterance, &self.config)?;
        let ids = self.indices(&utterance.ids);
        let types = utterance.types.as_ref().map(|types| self.indices(types));

        let log_durations: Vec<f32> = self
            .durations
            .forward(ids.clone(), utterance.speaker, None)
            .into_data()
            .into_vec()
            .map_err(|e| {
                SileroError::Checkpoint(format!("reading the durations: {e:?}"))
            })?;
        let durations = model::durations(&log_durations, &utterance.rate);
        let expansion = model::expansion(&durations);
        model::check_frames(expansion.len(), &self.config)?;

        let mut pitch: Vec<f32> = self
            .pitch
            .forward(ids.clone(), utterance.speaker, types)
            .into_data()
            .into_vec()
            .map_err(|e| {
                SileroError::Checkpoint(format!("reading the pitch: {e:?}"))
            })?;
        model::shape_pitch(
            &mut pitch,
            &utterance.pitch,
            self.mean_std
                .get(utterance.speaker)
                .copied()
                .unwrap_or_default(),
        );

        let pitch = Tensor::<B, 1>::from_data(
            TensorData::new(pitch.clone(), [pitch.len()]),
            &self.device,
        );
        let mel = self.acoustic.forward(
            ids,
            utterance.speaker,
            pitch,
            self.indices(&expansion),
        );
        let head: Vec<f32> = self
            .vocoder
            .forward(mel)
            .into_data()
            .into_vec()
            .map_err(|e| {
            SileroError::Checkpoint(format!("reading the spectrum: {e:?}"))
        })?;
        let (magnitude, phase) = split_spectrum(&head, self.config.bins());
        Ok(Spectrum {
            magnitude,
            phase,
            frames: expansion.len(),
            durations,
        })
    }
}

/// Loads the model on burn's CPU backend.
///
/// # Errors
///
/// Returns [`SileroError::Checkpoint`] when the weights cannot be read.
pub fn load_cpu(dir: &Path) -> Result<Box<dyn SpeechModel>, SileroError> {
    Ok(Box::new(BurnSpeech::<NdArray>::load(
        dir,
        NdArrayDevice::Cpu,
    )?))
}

/// Loads the model on burn's Metal backend.
///
/// # Errors
///
/// Returns [`SileroError::Checkpoint`] when the weights cannot be read.
pub fn load_metal(dir: &Path) -> Result<Box<dyn SpeechModel>, SileroError> {
    Ok(Box::new(BurnSpeech::<Metal>::load(
        dir,
        WgpuDevice::DefaultDevice,
    )?))
}

#[cfg(test)]
mod tests;
