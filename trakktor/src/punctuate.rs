//! `text punctuate`: flag mapping, model resolution, and the punctuation run.

use std::path::Path;

use trakktor_core::punctuate::{
    self, PunctCapSegModel, PunctRuntime, PunctuateError, PunctuateOptions,
    Punctuator, SpeTokenizer,
};

use crate::{cli::PunctuateArgs, error::CliError};

/// Runs one punctuation pass end to end: read the text, resolve (and if needed
/// download) the model, load it and the tokenizer, restore punctuation and
/// casing, and print the result.
pub(crate) fn run_punctuate(
    args: &PunctuateArgs,
    model_dir: &Path,
    json: bool,
    pretty: bool,
) -> Result<(), CliError> {
    // Read and normalize the input first, so a bad path fails before any
    // download or model load.
    let raw = std::fs::read_to_string(&args.input).map_err(|e| {
        PunctuateError::Io(format!("{}: {e}", args.input.display()))
    })?;
    let text = punctuate::normalize(&raw);

    let resolved = punctuate::resolve_model(
        model_dir,
        &args.model,
        &mut crate::asr::progress::download_progress(),
    )?;

    let precision = args.precision.to_punctuate();
    let runtime: Box<dyn PunctCapSegModel> = match args.runtime {
        crate::cli::RuntimeArg::Candle => match args.device {
            crate::cli::DeviceArg::Cpu => {
                Box::new(PunctRuntime::load_cpu(&resolved.dir, precision)?)
            },
            crate::cli::DeviceArg::Metal => {
                Box::new(load_metal(&resolved.dir, precision)?)
            },
        },
        crate::cli::RuntimeArg::Burn => {
            load_burn(&resolved.dir, args.device, precision)?
        },
    };
    let tokenizer = SpeTokenizer::load(&resolved.spe_path())?;
    let punctuator = Punctuator::new(runtime, tokenizer);

    let options = PunctuateOptions {
        overlap: args.overlap,
        batch_size: args.batch_size,
        apply_sbd: true,
    };
    let sentences = punctuator.sentences(&text, &options)?;

    crate::output::print_punctuate(&args.model, &sentences, json, pretty);
    Ok(())
}

/// Loads the model on Metal (builds with the `metal` feature).
#[cfg(feature = "metal")]
fn load_metal(
    model_dir: &Path,
    precision: punctuate::Precision,
) -> Result<PunctRuntime, CliError> {
    Ok(PunctRuntime::load_metal(model_dir, precision)?)
}

/// Without the `metal` feature, `--device metal` is a validation error.
#[cfg(not(feature = "metal"))]
fn load_metal(
    _model_dir: &Path,
    _precision: punctuate::Precision,
) -> Result<PunctRuntime, CliError> {
    Err(PunctuateError::InvalidOptions(
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
    precision: punctuate::Precision,
) -> Result<Box<dyn PunctCapSegModel>, CliError> {
    use trakktor_core::punctuate::PunctBurnRuntime;
    let runtime = match device {
        crate::cli::DeviceArg::Cpu => {
            PunctBurnRuntime::load_cpu(model_dir, precision)?
        },
        crate::cli::DeviceArg::Metal => {
            crate::burn_notice::announce_cold_gpu_start();
            PunctBurnRuntime::load_metal(model_dir, precision)?
        },
    };
    Ok(Box::new(runtime))
}

/// Without the `burn` feature, `--runtime burn` is a validation error.
#[cfg(not(feature = "burn"))]
fn load_burn(
    _model_dir: &Path,
    _device: crate::cli::DeviceArg,
    _precision: punctuate::Precision,
) -> Result<Box<dyn PunctCapSegModel>, CliError> {
    Err(CliError::from(PunctuateError::InvalidOptions(
        "this build has no burn runtime; install or build trakktor with the \
         `burn` feature"
            .into(),
    )))
}
