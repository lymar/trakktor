//! `text structify`: flag mapping, model resolution, and the segmentation run.

use std::path::Path;

use trakktor_core::structify::{
    self, BoundaryModel, SatRuntime, Structifier, StructifyError,
    StructifyOptions, XlmrTokenizer,
};

use crate::{cli::StructifyArgs, error::CliError};

/// Runs one structuring end to end: read the text, resolve (and if needed
/// download) the model and tokenizer, load them, segment into paragraphs, and
/// print the result.
pub(crate) fn run_structify(
    args: &StructifyArgs,
    model_dir: &Path,
    json: bool,
    pretty: bool,
) -> Result<(), CliError> {
    // Read and normalize the input first, so a bad path fails before any
    // download or model load.
    let raw = std::fs::read_to_string(&args.input).map_err(|e| {
        StructifyError::Io(format!("{}: {e}", args.input.display()))
    })?;
    let text = structify::normalize(&raw);

    let resolved = structify::resolve_model(
        model_dir,
        &args.model,
        &mut crate::asr::progress::download_progress(),
    )?;
    let tokenizer_path = structify::resolve_tokenizer(
        model_dir,
        &mut crate::asr::progress::download_progress(),
    )?;

    let precision = args.precision.to_structify();
    let runtime: Box<dyn BoundaryModel> = match args.runtime {
        crate::cli::RuntimeArg::Candle => match args.device {
            crate::cli::DeviceArg::Cpu => {
                Box::new(SatRuntime::load_cpu(&resolved.dir, precision)?)
            },
            crate::cli::DeviceArg::Metal => {
                Box::new(load_metal(&resolved.dir, precision)?)
            },
        },
        crate::cli::RuntimeArg::Burn => {
            load_burn(&resolved.dir, args.device, precision)?
        },
    };
    let tokenizer = XlmrTokenizer::load(&tokenizer_path)?;
    let structifier = Structifier::new(runtime, tokenizer);

    let options = StructifyOptions {
        threshold: args.threshold,
        stride: args.stride,
        batch_size: args.batch_size,
        weighting: structify::Weighting::Uniform,
    };
    let paragraphs = structifier.paragraphs(&text, &options)?;

    crate::output::print_structify(&args.model, &paragraphs, json, pretty);
    Ok(())
}

/// Loads the model on Metal (builds with the `metal` feature).
#[cfg(feature = "metal")]
fn load_metal(
    model_dir: &Path,
    precision: structify::Precision,
) -> Result<SatRuntime, CliError> {
    Ok(SatRuntime::load_metal(model_dir, precision)?)
}

/// Without the `metal` feature, `--device metal` is a validation error.
#[cfg(not(feature = "metal"))]
fn load_metal(
    _model_dir: &Path,
    _precision: structify::Precision,
) -> Result<SatRuntime, CliError> {
    Err(StructifyError::InvalidOptions(
        "this build has no Metal support; install or build trakktor with the \
         `metal` feature"
            .into(),
    )
    .into())
}

/// Loads the model on the burn runtime (builds with the `burn` feature); the
/// burn Metal backend is independent of the candle `metal` feature.
#[cfg(feature = "burn")]
fn load_burn(
    model_dir: &Path,
    device: crate::cli::DeviceArg,
    precision: structify::Precision,
) -> Result<Box<dyn BoundaryModel>, CliError> {
    use trakktor_core::structify::SatBurnRuntime;
    let runtime = match device {
        crate::cli::DeviceArg::Cpu => {
            SatBurnRuntime::load_cpu(model_dir, precision)?
        },
        crate::cli::DeviceArg::Metal => {
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
    _device: crate::cli::DeviceArg,
    _precision: structify::Precision,
) -> Result<Box<dyn BoundaryModel>, CliError> {
    Err(CliError::from(StructifyError::InvalidOptions(
        "this build has no burn runtime; install or build trakktor with the \
         `burn` feature"
            .into(),
    )))
}
