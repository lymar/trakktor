//! `tts qwen3-tts`: flag mapping, model resolution, and the synthesis run.

use std::path::Path;

use trakktor_core::{
    audio::encode::Format,
    tts::qwen3_tts::{
        self, Precision, Qwen3TtsError, Sampling, SynthesisOptions,
        Synthesizer, runtime::Device,
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
    // Decide the container before any download, so a bad extension fails fast.
    let format = output_format(&args.output)?;

    if matches!(args.runtime, RuntimeArg::Burn) {
        return Err(Qwen3TtsError::InvalidOptions(
            "this engine has no burn runtime yet; use `--runtime candle`"
                .into(),
        )
        .into());
    }

    let resolved = qwen3_tts::resolve_model(
        model_dir,
        &args.model,
        &mut crate::asr::progress::download_progress(),
    )?;

    let device = device(args.device)?;
    let precision = match args.precision {
        TtsPrecisionArg::Bf16 => Precision::Bf16,
        TtsPrecisionArg::F32 => Precision::F32,
    };
    let mut synthesizer = Synthesizer::load(&resolved.dir, device, precision)?;

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
        &args.text,
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
        "candle",
        &sampling,
        json,
        pretty,
    );
    Ok(())
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
