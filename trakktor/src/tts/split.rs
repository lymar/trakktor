//! Splitting a paragraph the engine cannot speak in one utterance.
//!
//! The cutting itself belongs to `text structify` — it already scores every
//! position for a boundary, and a second splitter would only drift from it.
//! What lives here is the wiring: the same runtime and device the synthesis
//! runs on, structify's own defaults for everything else, and a model loaded
//! **lazily**, so a text whose paragraphs all fit never downloads it.

use std::path::{Path, PathBuf};

use trakktor_core::structify::{
    self, BoundaryModel, SatRuntime, Structifier, StructifyOptions,
    XlmrTokenizer,
};

use crate::cli::{DeviceArg, RuntimeArg};

/// How a candidate piece is measured against the budget. Each engine states its
/// own unit — one counts its text tokens, another the bytes of UTF-8 — so the
/// splitter takes the measure rather than owning it.
pub(crate) type Cost<'a> = Box<dyn Fn(&str) -> usize + 'a>;

/// Splits over-budget paragraphs, loading the model on first use.
pub(crate) struct Splitter<'a> {
    model_dir: PathBuf,
    runtime: RuntimeArg,
    device: DeviceArg,
    cost: Cost<'a>,
    loaded: Option<Structifier>,
}

impl<'a> Splitter<'a> {
    /// Prepares a splitter; nothing is loaded or downloaded yet.
    pub(crate) fn new(
        model_dir: &Path,
        runtime: RuntimeArg,
        device: DeviceArg,
        cost: Cost<'a>,
    ) -> Self {
        Self {
            model_dir: model_dir.to_path_buf(),
            runtime,
            device,
            cost,
            loaded: None,
        }
    }

    /// Splits `text` into pieces of at most `budget`, in the caller's unit.
    ///
    /// # Errors
    ///
    /// Returns the failure as a message, for the calling engine to report in
    /// its own error vocabulary — from the caller's point of view this is one
    /// synthesis run, and the splitter an implementation detail of it.
    pub(crate) fn split(
        &mut self,
        text: &str,
        budget: usize,
    ) -> Result<Vec<String>, String> {
        if self.loaded.is_none() {
            eprintln!(
                "a paragraph is longer than one utterance; loading the \
                 sentence splitter..."
            );
            self.loaded = Some(self.load().map_err(splitter_failed)?);
        }
        let structifier =
            self.loaded.as_ref().expect("the splitter was just loaded");

        structifier
            .split_to_budget(
                text,
                budget,
                &self.cost,
                &StructifyOptions::default(),
            )
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

/// Phrases a splitter failure for the engine to wrap.
fn splitter_failed(err: structify::StructifyError) -> String {
    format!("splitting a long paragraph: {err}")
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
