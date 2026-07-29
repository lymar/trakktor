//! `text stress`: flag mapping, model resolution, and the marking run.

use std::path::Path;

use trakktor_core::stress::{
    self, Dictionary, StressError, StressModel, StressOptions, StressRuntime,
    Stressor,
};

use crate::{cli::StressArgs, error::CliError};

/// Runs one marking pass end to end: read the text and the dictionaries,
/// resolve (and if needed download and convert) the model, load it, mark the
/// text, and print the result.
pub(crate) fn run_stress(
    args: &StressArgs,
    model_dir: &Path,
    json: bool,
    pretty: bool,
) -> Result<(), CliError> {
    // Read the input first, so a bad path fails before any download or model
    // load; the dictionaries too, since a typo in one is the caller's to fix.
    let text = std::fs::read_to_string(&args.input).map_err(|e| {
        StressError::Io(format!("{}: {e}", args.input.display()))
    })?;
    let dictionary = Dictionary::load(&args.dictionaries)?;

    let resolved = stress::resolve_model(
        model_dir,
        &args.model,
        &mut crate::asr::progress::download_progress(),
    )?;

    let runtime = load(&resolved.dir, args)?;
    let stressor = Stressor::load(&resolved.dir, runtime)?;

    let options = StressOptions {
        marker: args.marker.to_core(),
        restore_yo: matches!(args.yo, crate::cli::YoArg::Auto),
        batch_size: args.batch_size,
    };
    let marked = stressor.mark(&text, &dictionary, &options)?;

    crate::output::print_stress(&resolved.label(), &marked, json, pretty);
    Ok(())
}

/// Loads the model on the runtime and device the flags ask for.
fn load(
    model_dir: &Path,
    args: &StressArgs,
) -> Result<Box<dyn StressModel>, CliError> {
    let precision = args.precision.to_stress();
    match args.runtime {
        crate::cli::RuntimeArg::Candle => match args.device {
            crate::cli::DeviceArg::Cpu => {
                Ok(Box::new(StressRuntime::load_cpu(model_dir, precision)?))
            },
            crate::cli::DeviceArg::Metal => {
                Ok(Box::new(load_metal(model_dir, precision)?))
            },
        },
        crate::cli::RuntimeArg::Burn => {
            load_burn(model_dir, args.device, precision)
        },
    }
}

/// Loads the model on Metal (builds with the `metal` feature).
#[cfg(feature = "metal")]
fn load_metal(
    model_dir: &Path,
    precision: stress::Precision,
) -> Result<StressRuntime, CliError> {
    Ok(StressRuntime::load_metal(model_dir, precision)?)
}

/// Without the `metal` feature, `--device metal` is a validation error.
#[cfg(not(feature = "metal"))]
fn load_metal(
    _model_dir: &Path,
    _precision: stress::Precision,
) -> Result<StressRuntime, CliError> {
    Err(StressError::InvalidOptions(
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
    precision: stress::Precision,
) -> Result<Box<dyn StressModel>, CliError> {
    use trakktor_core::stress::StressBurnRuntime;
    let runtime = match device {
        crate::cli::DeviceArg::Cpu => {
            StressBurnRuntime::load_cpu(model_dir, precision)?
        },
        crate::cli::DeviceArg::Metal => {
            crate::burn_notice::announce_cold_gpu_start();
            StressBurnRuntime::load_metal(model_dir, precision)?
        },
    };
    Ok(Box::new(runtime))
}

/// Without the `burn` feature, `--runtime burn` is a validation error.
#[cfg(not(feature = "burn"))]
fn load_burn(
    _model_dir: &Path,
    _device: crate::cli::DeviceArg,
    _precision: stress::Precision,
) -> Result<Box<dyn StressModel>, CliError> {
    Err(CliError::from(StressError::InvalidOptions(
        "this build has no burn runtime; install or build trakktor with the \
         `burn` feature"
            .into(),
    )))
}
