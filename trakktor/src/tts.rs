//! `tts qwen3-tts`: flag mapping, model resolution, and the synthesis run.

use std::path::Path;

use trakktor_core::{
    audio::encode::Format,
    tts::qwen3_tts::{
        self, Precision, Qwen3TtsError, Sampling, SpeechModel,
        SynthesisOptions, Synthesizer, runtime::Device,
    },
};

use crate::{
    cli::{DeviceArg, Qwen3TtsArgs, RuntimeArg, TtsPrecisionArg},
    error::CliError,
};

/// Frames the engine may generate before giving up on an end-of-speech code.
/// At 12.5 frames per second this bounds one run to a few minutes of audio.
const MAX_FRAMES: usize = 2048;

/// Runs one synthesis end to end: resolve (and if needed download) the model,
/// load it, speak the text, write the audio, and print the result.
pub(crate) fn run_qwen3_tts(
    args: &Qwen3TtsArgs,
    model_dir: &Path,
    json: bool,
    pretty: bool,
) -> Result<(), CliError> {
    // Resolve the text, the container, and the precision before any download,
    // so a bad path, a bad extension, or a precision the runtime cannot serve
    // fails fast rather than after gigabytes.
    let text = resolve_text(args.text.as_deref(), args.text_file.as_deref())?;
    let format = output_format(&args.output)?;
    let precision = resolve_precision(args.precision, args.runtime)?;

    let resolved = qwen3_tts::resolve_model(
        model_dir,
        &args.model,
        &mut crate::asr::progress::download_progress(),
    )?;

    let model: Box<dyn SpeechModel> = match args.runtime {
        RuntimeArg::Candle => Box::new(qwen3_tts::runtime::load(
            &resolved.dir,
            device(args.device)?,
            precision,
        )?),
        RuntimeArg::Burn => load_burn(&resolved.dir, args.device, precision)?,
    };
    let mut synthesizer = Synthesizer::load(&resolved.dir, model)?;

    let sampling = if args.greedy {
        Sampling::Greedy
    } else {
        Sampling::TopK {
            top_k: args.top_k,
            temperature: args.temperature,
            repetition_penalty: args.repetition_penalty,
            seed: args.seed,
        }
    };
    // `auto` is how the CLI spells "let the model decide", which the engine
    // expresses as no language at all.
    let language = (!args.language.eq_ignore_ascii_case("auto"))
        .then(|| args.language.clone());

    let synthesis = synthesizer.speak(
        &text,
        &SynthesisOptions {
            voice: args.voice.clone(),
            language,
            sampling,
            max_frames: MAX_FRAMES,
        },
    )?;

    synthesis.speech.write(&args.output, format)?;

    crate::output::print_synthesis(
        &synthesis,
        &args.output,
        format_name(format),
        &resolved.label(),
        runtime_name(args.runtime),
        &sampling,
        json,
        pretty,
    );
    Ok(())
}

/// Loads the model on the burn runtime (builds with the `burn` feature); the
/// burn Metal backend is independent of the candle `metal` feature.
#[cfg(feature = "burn")]
fn load_burn(
    model_dir: &Path,
    device: DeviceArg,
    precision: Precision,
) -> Result<Box<dyn SpeechModel>, CliError> {
    use trakktor_core::tts::qwen3_tts::runtime_burn;
    let load = match device {
        DeviceArg::Cpu => runtime_burn::load_cpu,
        DeviceArg::Metal => {
            crate::burn_notice::announce_cold_gpu_start();
            runtime_burn::load_metal
        },
    };
    Ok(load(model_dir, precision)?)
}

/// Resolves the compute precision, whose default depends on the runtime: burn
/// serves `f32` only, so that is its default; candle keeps the reference's
/// `bf16`. Asking burn for `bf16` explicitly is rejected rather than quietly
/// downgraded.
fn resolve_precision(
    precision: Option<TtsPrecisionArg>,
    runtime: RuntimeArg,
) -> Result<Precision, CliError> {
    Ok(match (precision, runtime) {
        (Some(TtsPrecisionArg::F32), _) => Precision::F32,
        (Some(TtsPrecisionArg::Bf16), RuntimeArg::Candle) => Precision::Bf16,
        (Some(TtsPrecisionArg::Bf16), RuntimeArg::Burn) => {
            return Err(Qwen3TtsError::InvalidOptions(
                "the burn runtime computes in f32 only; use `--precision \
                 f32`, or `--runtime candle` for bf16"
                    .into(),
            )
            .into());
        },
        // Unspecified: burn serves only f32; candle keeps the reference's bf16.
        (None, RuntimeArg::Burn) => Precision::F32,
        (None, RuntimeArg::Candle) => Precision::Bf16,
    })
}

/// Without the `burn` feature, `--runtime burn` is a validation error.
#[cfg(not(feature = "burn"))]
fn load_burn(
    _model_dir: &Path,
    _device: DeviceArg,
    _precision: Precision,
) -> Result<Box<dyn SpeechModel>, CliError> {
    Err(CliError::from(Qwen3TtsError::InvalidOptions(
        "this build has no burn runtime; install or build trakktor with the \
         `burn` feature"
            .into(),
    )))
}

/// The name of a runtime, as the output contract spells it.
fn runtime_name(runtime: RuntimeArg) -> &'static str {
    match runtime {
        RuntimeArg::Candle => "candle",
        RuntimeArg::Burn => "burn",
    }
}

/// Resolves what to speak: the argument, or the contents of `--text-file`.
///
/// A file is read as UTF-8 and its whitespace collapsed — a hard-wrapped
/// paragraph should read as running prose, and the model was trained on single
/// lines, so raw newlines only confuse its prosody.
fn resolve_text(
    text: Option<&str>,
    file: Option<&Path>,
) -> Result<String, Qwen3TtsError> {
    if let Some(text) = text {
        return Ok(text.to_owned());
    }
    // clap enforces that exactly one of the two is given; this guards the
    // library-level contract rather than the command line.
    let path = file.ok_or_else(|| {
        Qwen3TtsError::InvalidOptions(
            "pass the text as an argument or with --text-file".into(),
        )
    })?;
    let raw = std::fs::read_to_string(path).map_err(|e| {
        Qwen3TtsError::Io(format!("reading {}: {e}", path.display()))
    })?;
    Ok(collapse_whitespace(&raw))
}

/// Collapses every run of whitespace to a single space and trims the ends.
fn collapse_whitespace(text: &str) -> String {
    text.split_whitespace().collect::<Vec<_>>().join(" ")
}

/// Picks the container from the output path's extension.
fn output_format(path: &Path) -> Result<Format, Qwen3TtsError> {
    match path
        .extension()
        .and_then(|extension| extension.to_str())
        .map(str::to_ascii_lowercase)
        .as_deref()
    {
        Some("wav") => Ok(Format::Wav),
        Some("flac") => Ok(Format::Flac),
        other => Err(Qwen3TtsError::InvalidOptions(format!(
            "cannot write `{}`: the output extension must be wav or flac{}",
            path.display(),
            other.map_or_else(
                || " (there is none)".to_owned(),
                |found| format!(" (found `{found}`)")
            )
        ))),
    }
}

/// The name of a container, as the output contract spells it.
fn format_name(format: Format) -> &'static str {
    match format {
        Format::Wav => "wav",
        Format::Flac => "flac",
    }
}

/// Creates the compute device, reporting a build without Metal as a validation
/// error rather than falling back silently.
fn device(device: DeviceArg) -> Result<Device, CliError> {
    Ok(qwen3_tts::runtime::device(matches!(
        device,
        DeviceArg::Metal
    ))?)
}

#[cfg(test)]
mod tests;
