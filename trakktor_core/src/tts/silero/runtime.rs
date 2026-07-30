//! The candle runtime: loading a converted model and running one utterance.
//!
//! Everything computes in `f32`. On twenty-three million parameters half
//! precision saves nothing worth the risk to a voice, and the reference itself
//! offers no half-precision path to be compared against.

pub mod net;

use std::path::Path;

pub use candle_core::Device;
use candle_core::{DType, Tensor};
use candle_nn::VarBuilder;

use super::{
    config::Config,
    download::WEIGHTS_FILE,
    error::SileroError,
    istft::{Pqmf, Window, split_spectrum},
    model::{self, Spectrum, SpeechModel, Utterance},
};

/// A loaded model on candle.
pub struct SileroRuntime {
    config: Config,
    device: Device,
    acoustic: net::Acoustic,
    durations: net::SeriesPredictor,
    pitch: net::SeriesPredictor,
    vocoder: net::Vocoder,
    /// Per-speaker pitch range, used when the pitch is shifted.
    mean_std: Vec<f32>,
    window: Window,
    filterbanks: Vec<Pqmf>,
}

/// Creates the compute device, reporting a build without Metal as a validation
/// error rather than falling back silently.
///
/// # Errors
///
/// Returns [`SileroError::InvalidOptions`] when Metal is asked for and this
/// build has no Metal support, and [`SileroError::Checkpoint`] when the device
/// cannot be created.
pub fn device(metal: bool) -> Result<Device, SileroError> {
    if !metal {
        return Ok(Device::Cpu);
    }
    #[cfg(feature = "tts-metal")]
    {
        Device::new_metal(0).map_err(|e| {
            SileroError::Checkpoint(format!("opening the Metal device: {e}"))
        })
    }
    #[cfg(not(feature = "tts-metal"))]
    Err(SileroError::InvalidOptions(
        "this build has no Metal support; rebuild with the `metal` feature, \
         or use `--device cpu`"
            .into(),
    ))
}

/// Loads a converted model directory onto `device`.
///
/// # Errors
///
/// Returns [`SileroError::Checkpoint`] when the weights are missing or do not
/// match the geometry derived from them.
pub fn load(dir: &Path, device: Device) -> Result<SileroRuntime, SileroError> {
    let path = dir.join(WEIGHTS_FILE);
    let shapes = shape_index(&path)?;
    let config = Config::derive(&shapes)?;
    let vb = unsafe {
        VarBuilder::from_mmaped_safetensors(&[&path], DType::F32, &device)
    }
    .map_err(|e| {
        SileroError::Checkpoint(format!("reading {}: {e}", path.display()))
    })?;

    let has_types = config.utterance_types > 0;
    let build = || -> candle_core::Result<SileroRuntime> {
        let window = Window {
            samples: vb
                .get(config.n_fft, "vocoder.head.istft.window")?
                .to_vec1()?,
            hop: config.hop(),
        };
        let mut filterbanks = Vec::new();
        for (bands, name) in [(2usize, "pqmf_2"), (6, "pqmf_6")] {
            let key = format!("vocoder.{name}.filters");
            let Some(shape) = shapes.get(&key) else {
                continue;
            };
            let taps = shape[1];
            filterbanks.push(Pqmf {
                filters: vb
                    .get((bands, taps), &key)?
                    .flatten_all()?
                    .to_vec1()?,
                bands,
                taps,
            });
        }
        Ok(SileroRuntime {
            config,
            acoustic: net::Acoustic::load(&config, vb.pp("tacotron"))?,
            durations: net::SeriesPredictor::load(
                &config,
                false,
                vb.pp("dur_predictor").pp("dur_pred"),
            )?,
            pitch: net::SeriesPredictor::load(
                &config,
                has_types,
                vb.pp("pitch_predictor").pp("pitch_pred"),
            )?,
            vocoder: net::Vocoder::load(&config, vb.pp("vocoder"))?,
            mean_std: vb
                .get(config.speaker_slots, "pitch.mean_std_coef")?
                .to_vec1()?,
            window,
            filterbanks,
            device,
        })
    };
    build().map_err(|e| {
        SileroError::Checkpoint(format!("loading {}: {e}", path.display()))
    })
}

/// The shapes of every tensor a safetensors file holds, for deriving geometry.
pub(super) fn shape_index(
    path: &Path,
) -> Result<super::config::ShapeIndex, SileroError> {
    // SAFETY: read-only mapping of a file nothing mutates.
    let mapped = unsafe {
        candle_core::safetensors::MmapedSafetensors::new(path).map_err(|e| {
            SileroError::Checkpoint(format!("reading {}: {e}", path.display()))
        })?
    };
    Ok(mapped
        .tensors()
        .into_iter()
        .map(|(name, view)| (name, view.shape().to_vec()))
        .collect())
}

impl SpeechModel for SileroRuntime {
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
        let fail = |e: candle_core::Error| {
            SileroError::Checkpoint(format!("running the model: {e}"))
        };
        let ids = net::indices(&utterance.ids, &self.device).map_err(fail)?;
        let types = utterance
            .types
            .as_ref()
            .map(|types| net::indices(types, &self.device))
            .transpose()
            .map_err(fail)?;

        let log_durations: Vec<f32> = self
            .durations
            .forward(&ids, utterance.speaker, None)
            .and_then(|out| out.to_vec1())
            .map_err(fail)?;
        let durations = model::durations(&log_durations, &utterance.rate);
        let expansion = model::expansion(&durations);
        model::check_frames(expansion.len(), &self.config)?;

        let mut pitch: Vec<f32> = self
            .pitch
            .forward(&ids, utterance.speaker, types.as_ref())
            .and_then(|out| out.to_vec1())
            .map_err(fail)?;
        model::shape_pitch(
            &mut pitch,
            &utterance.pitch,
            self.mean_std
                .get(utterance.speaker)
                .copied()
                .unwrap_or_default(),
        );

        let spectrum = self
            .spectrum(&ids, utterance.speaker, &pitch, &expansion)
            .map_err(fail)?;
        let (magnitude, phase) = split_spectrum(&spectrum, self.config.bins());
        Ok(Spectrum {
            magnitude,
            phase,
            frames: expansion.len(),
            durations,
        })
    }
}

impl SileroRuntime {
    /// The acoustic model and the vocoder, from the pitch and the plan of
    /// frames onwards.
    fn spectrum(
        &self,
        ids: &Tensor,
        speaker: usize,
        pitch: &[f32],
        expansion: &[u32],
    ) -> candle_core::Result<Vec<f32>> {
        let pitch = net::values(pitch, &self.device)?;
        let expansion = net::indices(expansion, &self.device)?;
        let mel = self.acoustic.forward(ids, speaker, &pitch, &expansion)?;
        self.vocoder.forward(&mel)?.flatten_all()?.to_vec1()
    }
}

#[cfg(test)]
mod tests;
