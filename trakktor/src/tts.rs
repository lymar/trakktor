//! `tts qwen3-tts`: flag mapping, model resolution, and the synthesis run.

use std::{
    io::Read,
    path::{Path, PathBuf},
    time::Instant,
};

use trakktor_core::{
    audio::encode,
    tts::{
        Speech,
        qwen3_tts::{
            self, Precision, Qwen3TtsError, Sampling, SpeechModel,
            SynthesisOptions, Synthesizer, TextTokenizer, runtime::Device,
        },
        text,
    },
};

use crate::{
    cli::{
        AudioEncoderArg, DeviceArg, EspeechArgs, LevelsArg, Qwen3TtsArgs,
        RuntimeArg, TtsPrecisionArg,
    },
    error::CliError,
};

mod progress;
mod split;

/// Runs one synthesis end to end: read the text, resolve (and if needed
/// download) the model, load it, speak the paragraphs, write the audio, and
/// print the result.
pub(crate) fn run_qwen3_tts(
    args: &Qwen3TtsArgs,
    model_dir: &Path,
    json: bool,
    pretty: bool,
) -> Result<(), CliError> {
    // Resolve the text, the container, and the precision before any download,
    // so a bad path, a bad extension, or a precision the runtime cannot serve
    // fails fast rather than after gigabytes.
    let input = read_input(args.text.as_deref(), args.text_file.as_deref())?;
    let paragraphs = text::paragraphs(
        &input.text,
        args.text_format
            .to_core()
            .resolve(input.source.as_deref(), &input.text),
    );
    if paragraphs.is_empty() {
        return Err(Qwen3TtsError::TextEmpty.into());
    }
    let target = output_target(&args.output, args.audio_encoder)?;
    let precision =
        resolve_precision(args.precision, args.runtime, args.device)?;
    let sampling = if args.greedy {
        Sampling::Greedy
    } else {
        // The reference validates this too: zero would divide the logits away,
        // and "no randomness" is what --greedy is for.
        if args.temperature <= 0.0 {
            return Err(Qwen3TtsError::InvalidOptions(
                "temperature must be positive; for deterministic output use \
                 --greedy"
                    .into(),
            )
            .into());
        }
        Sampling::TopK {
            top_k: args.top_k,
            temperature: args.temperature,
            repetition_penalty: args.repetition_penalty,
            seed: args.seed,
        }
    };

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
    // `auto` is how the CLI spells "let the model decide", which the engine
    // expresses as no language at all.
    let language = (!args.language.eq_ignore_ascii_case("auto"))
        .then(|| args.language.clone());

    // Splitting an over-budget paragraph needs a second model; it is loaded
    // only if a paragraph actually turns out to need one.
    let tokenizer = TextTokenizer::load(
        &resolved.dir.join("vocab.json"),
        &resolved.dir.join("merges.txt"),
    )?;
    let mut splitter = split::Splitter::new(
        model_dir,
        args.runtime,
        args.device,
        // A piece whose cost cannot be measured is treated as too expensive,
        // so the search keeps cutting rather than accepting it.
        Box::new(move |piece: &str| {
            tokenizer.encode(piece).map_or(usize::MAX, |ids| ids.len())
        }),
    );

    let report = progress::Reporter::start(Instant::now());
    let synthesis = synthesizer.speak_paragraphs(
        &paragraphs,
        &SynthesisOptions {
            voice: args.voice.clone(),
            language,
            sampling,
            pause: f64::from(args.pause_ms) / 1000.0,
            match_levels: matches!(args.levels, LevelsArg::Match),
        },
        &mut |text, budget| {
            splitter
                .split(text, budget)
                .map_err(Qwen3TtsError::InvalidModel)
        },
        &mut |progress| report.update(progress),
    )?;
    report.finish(synthesis.chunks, synthesis.speech.duration());

    write_audio(
        &args.output,
        &synthesis.speech,
        target,
        args.bitrate.as_deref(),
    )?;

    crate::output::print_synthesis(
        &synthesis,
        &args.output,
        &format_name(&args.output),
        &resolved.label(),
        runtime_name(args.runtime),
        &sampling,
        json,
        pretty,
    );
    Ok(())
}

/// Runs one ESpeech synthesis end to end: read the text and the reference,
/// resolve (and if needed download and convert) the model, load it, speak the
/// pieces in the reference's voice, write the audio, and print the result.
pub(crate) fn run_espeech(
    args: &EspeechArgs,
    model_dir: &Path,
    json: bool,
    pretty: bool,
) -> Result<(), CliError> {
    use trakktor_core::tts::espeech;

    // Everything that can fail without touching the network fails first, so a
    // bad path, a bad extension, or an unsupported language does not cost a
    // download.
    let input = read_input(args.text.as_deref(), args.text_file.as_deref())
        .map_err(espeech_input)?;
    let paragraphs = text::paragraphs(
        &input.text,
        args.text_format
            .to_core()
            .resolve(input.source.as_deref(), &input.text),
    );
    if paragraphs.is_empty() {
        return Err(espeech::EspeechError::TextEmpty.into());
    }
    let target = output_target(&args.output, args.audio_encoder)
        .map_err(espeech_input)?;
    let language = (!args.language.eq_ignore_ascii_case("auto"))
        .then(|| args.language.clone());
    espeech::Synthesizer::check_language(language.as_deref())?;
    let ref_text = read_ref_text(args)?;
    let reference = espeech::reference::prepare(&args.ref_audio)?;

    let resolved = espeech::resolve_model(
        model_dir,
        &args.model,
        &mut crate::asr::progress::download_progress(),
    )?;

    let precision = match args.precision {
        crate::cli::EspeechPrecisionArg::F16 => espeech::Precision::F16,
        crate::cli::EspeechPrecisionArg::F32 => espeech::Precision::F32,
    };
    let model: Box<dyn espeech::SpeechModel> = match args.runtime {
        RuntimeArg::Candle => Box::new(espeech::runtime::load(
            &resolved.dir,
            &resolved.vocoder_dir,
            espeech::runtime::device(matches!(args.device, DeviceArg::Metal))?,
            precision,
        )?),
        RuntimeArg::Burn => {
            load_espeech_burn(&resolved, args.device, precision)?
        },
    };

    let mut synthesizer = espeech::Synthesizer::load(
        &resolved.dir,
        model,
        reference,
        &ref_text,
        &args.ref_audio.display().to_string(),
    )?;

    let options = espeech::SynthesisOptions {
        nfe_step: args.nfe_step,
        cfg_strength: args.cfg_strength,
        speed: args.speed,
        seed: args.seed,
        pause: f64::from(args.pause_ms) / 1000.0,
        match_levels: matches!(args.levels, LevelsArg::Match),
    };
    // The budget is in bytes of UTF-8 — the unit the duration estimate is
    // stated in — so that is what a candidate piece is measured in too.
    let mut splitter = split::Splitter::new(
        model_dir,
        args.runtime,
        args.device,
        Box::new(str::len),
    );

    let report = progress::Reporter::start(Instant::now());
    let synthesis = synthesizer.speak_paragraphs(
        &paragraphs,
        &options,
        &mut |text, budget| {
            splitter
                .split(text, budget)
                .map_err(espeech::EspeechError::Checkpoint)
        },
        &mut |progress| report.update(progress),
    )?;
    report.finish(synthesis.chunks, synthesis.speech.duration());

    write_audio(
        &args.output,
        &synthesis.speech,
        target,
        args.bitrate.as_deref(),
    )?;

    crate::output::print_espeech_synthesis(
        &synthesis,
        &args.output,
        &format_name(&args.output),
        &resolved.label(),
        runtime_name(args.runtime),
        &options,
        language.as_deref(),
        json,
        pretty,
    );
    Ok(())
}

/// Re-reports an input failure of the shared helpers in the ESpeech vocabulary.
fn espeech_input(
    err: Qwen3TtsError,
) -> trakktor_core::tts::espeech::EspeechError {
    use trakktor_core::tts::espeech::EspeechError;
    match err {
        Qwen3TtsError::TextEmpty => EspeechError::TextEmpty,
        Qwen3TtsError::Io(message) => EspeechError::Io(message),
        other => EspeechError::InvalidOptions(other.to_string()),
    }
}

/// Resolves the reference transcript: the argument or a UTF-8 file.
fn read_ref_text(args: &EspeechArgs) -> Result<String, CliError> {
    use trakktor_core::tts::espeech::EspeechError;
    if let Some(text) = &args.ref_text {
        return Ok(text.clone());
    }
    // clap enforces that exactly one of the two is given; this guards the
    // library-level contract rather than the command line.
    let path = args
        .ref_text_file
        .as_deref()
        .ok_or_else(|| CliError::from(EspeechError::RefTextRequired))?;
    std::fs::read_to_string(path).map_err(|e| {
        CliError::from(EspeechError::Io(format!(
            "reading {}: {e}",
            path.display()
        )))
    })
}

/// Loads the ESpeech model on the burn runtime (builds with the `burn`
/// feature).
#[cfg(feature = "burn")]
fn load_espeech_burn(
    resolved: &trakktor_core::tts::espeech::ResolvedModel,
    device: DeviceArg,
    precision: trakktor_core::tts::espeech::Precision,
) -> Result<Box<dyn trakktor_core::tts::espeech::SpeechModel>, CliError> {
    use trakktor_core::tts::espeech::runtime_burn;
    let load = match device {
        DeviceArg::Cpu => runtime_burn::load_cpu,
        DeviceArg::Metal => {
            crate::burn_notice::announce_cold_gpu_start();
            runtime_burn::load_metal
        },
    };
    Ok(load(&resolved.dir, &resolved.vocoder_dir, precision)?)
}

/// Without the `burn` feature, `--runtime burn` is a validation error.
#[cfg(not(feature = "burn"))]
fn load_espeech_burn(
    _resolved: &trakktor_core::tts::espeech::ResolvedModel,
    _device: DeviceArg,
    _precision: trakktor_core::tts::espeech::Precision,
) -> Result<Box<dyn trakktor_core::tts::espeech::SpeechModel>, CliError> {
    Err(CliError::from(
        trakktor_core::tts::espeech::EspeechError::InvalidOptions(
            "this build has no burn runtime; install or build trakktor with \
             the `burn` feature"
                .into(),
        ),
    ))
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

/// Resolves the compute precision, whose default depends on the runtime and
/// the device. burn serves `f32` only; candle keeps the reference's `bf16` on
/// Metal — where it is what holds the larger model in memory — but its CPU
/// backend has no `bf16` arithmetic, so the CPU default is `f32`. Asking for
/// a pairing the backend cannot serve is rejected rather than quietly
/// downgraded.
fn resolve_precision(
    precision: Option<TtsPrecisionArg>,
    runtime: RuntimeArg,
    device: DeviceArg,
) -> Result<Precision, CliError> {
    Ok(match (precision, runtime, device) {
        (Some(TtsPrecisionArg::F32), _, _) => Precision::F32,
        (Some(TtsPrecisionArg::Bf16), RuntimeArg::Candle, DeviceArg::Metal) => {
            Precision::Bf16
        },
        (Some(TtsPrecisionArg::Bf16), RuntimeArg::Candle, DeviceArg::Cpu) => {
            return Err(Qwen3TtsError::InvalidOptions(
                "candle on the CPU computes in f32 only (its CPU backend has \
                 no bf16 arithmetic); use `--precision f32`, or `--device \
                 metal` for bf16"
                    .into(),
            )
            .into());
        },
        (Some(TtsPrecisionArg::Bf16), RuntimeArg::Burn, _) => {
            return Err(Qwen3TtsError::InvalidOptions(
                "the burn runtime computes in f32 only; use `--precision \
                 f32`, or `--runtime candle --device metal` for bf16"
                    .into(),
            )
            .into());
        },
        // Unspecified: bf16 where it is served, f32 everywhere else.
        (None, RuntimeArg::Candle, DeviceArg::Metal) => Precision::Bf16,
        (None, RuntimeArg::Candle, DeviceArg::Cpu) |
        (None, RuntimeArg::Burn, _) => Precision::F32,
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

/// The text to speak, and where it came from — the source name is what lets
/// `--text-format auto` believe an extension.
#[derive(Debug)]
struct Input {
    text: String,
    source: Option<PathBuf>,
}

/// Resolves what to speak: the argument, a UTF-8 file, or standard input.
fn read_input(
    text: Option<&str>,
    file: Option<&Path>,
) -> Result<Input, Qwen3TtsError> {
    if let Some(text) = text {
        return Ok(Input {
            text: text.to_owned(),
            source: None,
        });
    }
    // clap enforces that exactly one of the two is given; this guards the
    // library-level contract rather than the command line.
    let path = file.ok_or_else(|| {
        Qwen3TtsError::InvalidOptions(
            "pass the text as an argument or with --text-file".into(),
        )
    })?;
    if path == Path::new("-") {
        let mut text = String::new();
        std::io::stdin().read_to_string(&mut text).map_err(|e| {
            Qwen3TtsError::Io(format!("reading standard input: {e}"))
        })?;
        return Ok(Input { text, source: None });
    }
    let text = std::fs::read_to_string(path).map_err(|e| {
        Qwen3TtsError::Io(format!("reading {}: {e}", path.display()))
    })?;
    Ok(Input {
        text,
        source: Some(path.to_path_buf()),
    })
}

/// Where the audio goes: a built-in encoder, or an external ffmpeg process.
#[derive(Clone, Copy)]
enum OutputTarget {
    Native(encode::Format),
    Ffmpeg,
}

/// Resolves the `--audio-encoder`/extension pair into a concrete target. With
/// `auto` an extension the built-in encoders do not write is handed to ffmpeg;
/// with `builtin` it is an error, before anything is generated.
fn output_target(
    path: &Path,
    encoder: AudioEncoderArg,
) -> Result<OutputTarget, Qwen3TtsError> {
    let extension = path
        .extension()
        .and_then(|extension| extension.to_str())
        .map(str::to_ascii_lowercase);
    let native = match extension.as_deref() {
        Some("wav") => Some(encode::Format::Wav),
        Some("flac") => Some(encode::Format::Flac),
        _ => None,
    };
    match (encoder, native) {
        (AudioEncoderArg::Ffmpeg, _) | (AudioEncoderArg::Auto, None) => {
            Ok(OutputTarget::Ffmpeg)
        },
        (_, Some(format)) => Ok(OutputTarget::Native(format)),
        (AudioEncoderArg::Builtin, None) => {
            Err(Qwen3TtsError::InvalidOptions(format!(
                "cannot write `{}` with the builtin encoder: it writes wav or \
                 flac{}; use `--audio-encoder ffmpeg` for other formats",
                path.display(),
                extension.map_or_else(
                    || " (the output has no extension)".to_owned(),
                    |found| format!(" (found `{found}`)")
                )
            )))
        },
    }
}

/// The format name reported in the output contract: the output's extension,
/// lowercased.
fn format_name(path: &Path) -> String {
    path.extension()
        .and_then(|extension| extension.to_str())
        .map(str::to_ascii_lowercase)
        .unwrap_or_default()
}

/// Writes the synthesized audio through the resolved target.
fn write_audio(
    path: &Path,
    speech: &Speech,
    target: OutputTarget,
    bitrate: Option<&str>,
) -> Result<(), CliError> {
    match target {
        OutputTarget::Native(format) => speech.write(path, format)?,
        OutputTarget::Ffmpeg => speech.write_ffmpeg(path, bitrate)?,
    }
    Ok(())
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
