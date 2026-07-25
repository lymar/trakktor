//! Splitting a paragraph the engine cannot speak in one utterance.
//!
//! The cutting itself belongs to `text structify` — it already scores every
//! position for a boundary, and a second splitter would only drift from it.
//! What lives here is the wiring: the same runtime and device the synthesis
//! runs on, structify's own defaults for everything else, and a model loaded
//! **lazily**, so a text whose paragraphs all fit never downloads it.

use std::path::{Path, PathBuf};

use trakktor_core::{
    structify::{
        self, BoundaryModel, SatRuntime, Structifier, StructifyOptions,
        XlmrTokenizer,
    },
    tts::qwen3_tts::{Qwen3TtsError, TextTokenizer},
};

use crate::cli::{DeviceArg, RuntimeArg};

/// Splits over-budget paragraphs, loading the model on first use.
pub(crate) struct Splitter {
    model_dir: PathBuf,
    runtime: RuntimeArg,
    device: DeviceArg,
    /// The engine's own tokenizer: the budget counts its tokens, so the cost
    /// of a candidate piece has to be measured with it.
    cost: TextTokenizer,
    loaded: Option<Structifier>,
}

impl Splitter {
    /// Prepares a splitter; nothing is loaded or downloaded yet.
    pub(crate) fn new(
        model_dir: &Path,
        runtime: RuntimeArg,
        device: DeviceArg,
        cost: TextTokenizer,
    ) -> Self {
        Self {
            model_dir: model_dir.to_path_buf(),
            runtime,
            device,
            cost,
            loaded: None,
        }
    }

    /// Splits `text` into pieces of at most `budget` engine tokens.
    ///
    /// # Errors
    ///
    /// Reports a failure to load or run the model as
    /// [`Qwen3TtsError::InvalidModel`] — from the caller's point of view this
    /// is one synthesis run, and its error contract is the engine's.
    pub(crate) fn split(
        &mut self,
        text: &str,
        budget: usize,
    ) -> Result<Vec<String>, Qwen3TtsError> {
        if self.loaded.is_none() {
            eprintln!(
                "a paragraph is longer than one utterance; loading the \
                 sentence splitter..."
            );
            self.loaded = Some(self.load().map_err(splitter_failed)?);
        }
        let structifier =
            self.loaded.as_ref().expect("the splitter was just loaded");

        let cost = |piece: &str| {
            // A piece whose cost cannot be measured is treated as too
            // expensive, so the search keeps cutting rather than accepting it.
            self.cost.encode(piece).map_or(usize::MAX, |ids| ids.len())
        };
        structifier
            .split_to_budget(text, budget, &cost, &StructifyOptions::default())
            .map_err(splitter_failed)
    }

    /// Loads the boundary model and its tokenizer, downloading them on first
    /// use — structify's defaults throughout, on this run's runtime and
    /// device.
    fn load(&self) -> Result<Structifier, structify::StructifyError> {
        let resolved = structify::resolve_model(
            &self.model_dir,
            structify::DEFAULT_MODEL,
            &mut crate::asr::progress::download_progress(),
        )?;
        let tokenizer_path = structify::resolve_tokenizer(
            &self.model_dir,
            &mut crate::asr::progress::download_progress(),
        )?;
        // structify's own default (f16), except where the backend cannot serve
        // it: burn on the CPU computes in f32 only, and this choice is the
        // engine's to make, not something to fail the run over.
        let precision = match (self.runtime, self.device) {
            (RuntimeArg::Burn, DeviceArg::Cpu) => structify::Precision::F32,
            _ => structify::Precision::default(),
        };

        let runtime: Box<dyn BoundaryModel> = match self.runtime {
            RuntimeArg::Candle => match self.device {
                DeviceArg::Cpu => {
                    Box::new(SatRuntime::load_cpu(&resolved.dir, precision)?)
                },
                DeviceArg::Metal => {
                    Box::new(load_metal(&resolved.dir, precision)?)
                },
            },
            RuntimeArg::Burn => {
                load_burn(&resolved.dir, self.device, precision)?
            },
        };
        Ok(Structifier::new(
            runtime,
            XlmrTokenizer::load(&tokenizer_path)?,
        ))
    }
}

/// Reports a splitter failure in the engine's error vocabulary: the run is a
/// synthesis, and the splitter is an implementation detail of it.
fn splitter_failed(err: structify::StructifyError) -> Qwen3TtsError {
    Qwen3TtsError::InvalidModel(format!("splitting a long paragraph: {err}"))
}

/// Loads the boundary model on Metal (builds with the `metal` feature).
#[cfg(feature = "metal")]
fn load_metal(
    model_dir: &Path,
    precision: structify::Precision,
) -> Result<SatRuntime, structify::StructifyError> {
    SatRuntime::load_metal(model_dir, precision)
}

/// Without the `metal` feature, `--device metal` is a validation error.
#[cfg(not(feature = "metal"))]
fn load_metal(
    _model_dir: &Path,
    _precision: structify::Precision,
) -> Result<SatRuntime, structify::StructifyError> {
    Err(structify::StructifyError::InvalidOptions(
        "this build has no Metal support; install or build trakktor with the \
         `metal` feature"
            .into(),
    ))
}

/// Loads the boundary model on the burn runtime (builds with the `burn`
/// feature).
#[cfg(feature = "burn")]
fn load_burn(
    model_dir: &Path,
    device: DeviceArg,
    precision: structify::Precision,
) -> Result<Box<dyn BoundaryModel>, structify::StructifyError> {
    use trakktor_core::structify::SatBurnRuntime;
    let runtime = match device {
        DeviceArg::Cpu => SatBurnRuntime::load_cpu(model_dir, precision)?,
        DeviceArg::Metal => {
            crate::burn_notice::announce_cold_gpu_start();
            SatBurnRuntime::load_metal(model_dir, precision)?
        },
    };
    Ok(Box::new(runtime))
}

/// Without the `burn` feature, `--runtime burn` is a validation error.
#[cfg(not(feature = "burn"))]
fn load_burn(
    _model_dir: &Path,
    _device: DeviceArg,
    _precision: structify::Precision,
) -> Result<Box<dyn BoundaryModel>, structify::StructifyError> {
    Err(structify::StructifyError::InvalidOptions(
        "this build has no burn runtime; install or build trakktor with the \
         `burn` feature"
            .into(),
    ))
}
