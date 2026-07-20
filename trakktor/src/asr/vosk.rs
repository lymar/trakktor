//! `asr vosk`: model resolution, streamed decoding, and the transcription
//! run.
//!
//! The audio is decoded block by block and fed straight into the engine's
//! streaming session. For an offline model that session detects speech,
//! plans chunk cuts, transcribes committed chunks, and releases their PCM; a
//! streaming model runs the native chunked encoder instead. Either way memory
//! stays bounded regardless of the file length.

use std::{io::Read, path::Path, time::Instant};

use trakktor_core::asr::vosk::{
    self, Decoding, TranscribeOptions, TranscribeProgress, TransducerHead,
    VoskError,
};

use crate::{cli::VoskArgs, error::CliError, output};

/// Samples per block fed into the session over the ffmpeg pipe (~2 s).
const FFMPEG_BLOCK_SAMPLES: usize = 32 * 1024;

/// Runs one Vosk transcription end to end: resolve (and if needed download)
/// the model, load it, then stream-decode and transcribe.
pub(crate) fn run_vosk(
    args: &VoskArgs,
    model_dir: &Path,
    json: bool,
    pretty: bool,
) -> Result<(), CliError> {
    let resolved = vosk::resolve_model(
        model_dir,
        &args.model,
        &mut crate::asr::progress::download_progress(),
    )?;
    let weights = vosk::weights::load_dir(&resolved.dir)?;
    let tokens = std::fs::read_to_string(resolved.dir.join("tokens.txt"))
        .map_err(|e| {
            VoskError::InvalidModel(format!("reading tokens.txt: {e}"))
        })?;
    let tokenizer = vosk::Tokenizer::parse(&tokens)?;
    let head = TransducerHead::load(&weights, tokenizer.unk_id())?;
    let precision = args.precision.to_vosk();

    let model: Box<dyn vosk::EncoderSeam> = match args.runtime {
        crate::cli::RuntimeArg::Candle => match args.device {
            crate::cli::DeviceArg::Cpu => {
                Box::new(vosk::VoskModel::load_cpu(&weights, precision)?)
            },
            crate::cli::DeviceArg::Metal => {
                Box::new(load_metal(&weights, precision)?)
            },
        },
        crate::cli::RuntimeArg::Burn => {
            load_burn(&weights, args.device, precision)?
        },
    };

    let options = TranscribeOptions {
        word_timestamps: matches!(
            args.timestamps,
            crate::cli::TimestampsArg::Word
        ),
        decoding: match args.decoding {
            crate::cli::DecodingArg::Beam => Decoding::Beam {
                max_active: vosk::decode::DEFAULT_MAX_ACTIVE,
            },
            crate::cli::DecodingArg::Greedy => Decoding::Greedy,
        },
    };

    let started = Instant::now();
    let mut live = crate::asr::progress::live_reporter(started);
    let mut report =
        |p: TranscribeProgress| live(p.processed_seconds, p.total_seconds);

    let transcription = match args.audio_decoder {
        crate::cli::AudioDecoderArg::Builtin => {
            run_builtin(args, &*model, &head, &tokenizer, options, &mut report)?
        },
        crate::cli::AudioDecoderArg::Ffmpeg => {
            run_ffmpeg(args, &*model, &head, &tokenizer, options, &mut report)?
        },
    };

    crate::asr::progress::finish_line(started, transcription.duration);

    output::print_vosk_transcription(
        &transcription,
        &args.model,
        args.language.as_deref(),
        args.timestamps,
        json,
        pretty,
    );

    if let Some(format) = args.output_format {
        let with_words =
            matches!(args.timestamps, crate::cli::TimestampsArg::Word);
        let paths = crate::asr::writers::write_vosk_outputs(
            &transcription,
            &args.model,
            &args.audio,
            format,
            &args.output_dir,
            with_words,
        )?;
        for path in &paths {
            eprintln!("wrote {}", path.display());
        }
    }
    Ok(())
}

/// Streams the built-in decoder's blocks through a transcription session.
fn run_builtin(
    args: &VoskArgs,
    model: &dyn vosk::EncoderSeam,
    head: &TransducerHead,
    tokenizer: &vosk::Tokenizer,
    options: TranscribeOptions,
    report: &mut dyn FnMut(TranscribeProgress),
) -> Result<vosk::Transcription, CliError> {
    let mut stream =
        trakktor_core::audio::MonoS16Stream::open(&args.audio, 16_000)
            .map_err(|e| VoskError::AudioDecode(e.to_string()))?;
    let mut session = vosk::StreamTranscriber::new(
        model,
        head,
        tokenizer,
        options,
        stream.duration_hint(),
    )?;
    while let Some(block) = stream
        .next_block()
        .map_err(|e| VoskError::AudioDecode(e.to_string()))?
    {
        let samples: Vec<f32> =
            block.iter().map(|&s| f32::from(s) / 32768.0).collect();
        session.push(&samples, report)?;
    }
    Ok(session.finish(report)?)
}

/// Streams 16 kHz mono s16 PCM from an external `ffmpeg` process through a
/// transcription session (formats the built-in decoder does not cover).
fn run_ffmpeg(
    args: &VoskArgs,
    model: &dyn vosk::EncoderSeam,
    head: &TransducerHead,
    tokenizer: &vosk::Tokenizer,
    options: TranscribeOptions,
    report: &mut dyn FnMut(TranscribeProgress),
) -> Result<vosk::Transcription, CliError> {
    use std::process::{Command, Stdio};

    let mut child = Command::new("ffmpeg")
        .args(["-nostdin", "-threads", "0", "-i"])
        .arg(&args.audio)
        .args(["-f", "s16le", "-ac", "1", "-acodec", "pcm_s16le", "-ar"])
        .arg("16000")
        .arg("-")
        .stdin(Stdio::null())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .map_err(|e| {
            VoskError::AudioDecode(format!(
                "could not run ffmpeg ({e}); is it installed and on PATH?"
            ))
        })?;

    // The pipe has no length; progress runs without a total.
    let mut session =
        vosk::StreamTranscriber::new(model, head, tokenizer, options, None)?;
    let mut stdout = child.stdout.take().expect("stdout is piped");
    let mut bytes = vec![0u8; FFMPEG_BLOCK_SAMPLES * 2];
    // A block may end mid-sample; the odd byte carries into the next read.
    let mut carry: Option<u8> = None;
    loop {
        let offset = match carry.take() {
            Some(byte) => {
                bytes[0] = byte;
                1
            },
            None => 0,
        };
        let read = stdout
            .read(&mut bytes[offset..])
            .map_err(|e| VoskError::AudioDecode(format!("ffmpeg: {e}")))?;
        if read == 0 {
            break;
        }
        let total = offset + read;
        let whole = total & !1;
        if total > whole {
            carry = Some(bytes[whole]);
        }
        let samples: Vec<f32> = bytes[..whole]
            .chunks_exact(2)
            .map(|pair| {
                f32::from(i16::from_le_bytes([pair[0], pair[1]])) / 32768.0
            })
            .collect();
        session.push(&samples, report)?;
    }
    drop(stdout);

    let status = child
        .wait()
        .map_err(|e| VoskError::AudioDecode(format!("ffmpeg: {e}")))?;
    if !status.success() {
        let mut stderr = String::new();
        if let Some(mut pipe) = child.stderr.take() {
            let _ = pipe.read_to_string(&mut stderr);
        }
        let detail = stderr
            .lines()
            .rev()
            .find(|line| !line.trim().is_empty())
            .unwrap_or("")
            .trim()
            .to_string();
        return Err(CliError::from(VoskError::AudioDecode(format!(
            "ffmpeg could not decode {}: {detail}",
            args.audio.display()
        ))));
    }
    Ok(session.finish(report)?)
}

/// Loads a model on Metal (builds with the `metal` feature).
#[cfg(feature = "metal")]
fn load_metal(
    weights: &vosk::weights::ModelWeights,
    precision: vosk::Precision,
) -> Result<vosk::VoskModel, CliError> {
    Ok(vosk::VoskModel::load_metal(weights, precision)?)
}

/// Without the `metal` feature, `--device metal` is a validation error.
#[cfg(not(feature = "metal"))]
fn load_metal(
    _weights: &vosk::weights::ModelWeights,
    _precision: vosk::Precision,
) -> Result<vosk::VoskModel, CliError> {
    Err(CliError::from(VoskError::InvalidOptions(
        "this build has no Metal support; install or build trakktor with the \
         `metal` feature"
            .into(),
    )))
}

/// Loads a model on the burn runtime (builds with the `burn` feature); the
/// burn Metal backend is independent of the candle `metal` feature.
#[cfg(feature = "burn")]
fn load_burn(
    weights: &vosk::weights::ModelWeights,
    device: crate::cli::DeviceArg,
    precision: vosk::Precision,
) -> Result<Box<dyn vosk::EncoderSeam>, CliError> {
    use trakktor_core::asr::vosk::runtime_burn;
    let load = match device {
        crate::cli::DeviceArg::Cpu => runtime_burn::load_cpu,
        crate::cli::DeviceArg::Metal => {
            crate::burn_notice::announce_cold_gpu_start();
            runtime_burn::load_metal
        },
    };
    Ok(load(weights, precision)?)
}

/// Without the `burn` feature, `--runtime burn` is a validation error.
#[cfg(not(feature = "burn"))]
fn load_burn(
    _weights: &vosk::weights::ModelWeights,
    _device: crate::cli::DeviceArg,
    _precision: vosk::Precision,
) -> Result<Box<dyn vosk::EncoderSeam>, CliError> {
    Err(CliError::from(VoskError::InvalidOptions(
        "this build has no burn runtime; install or build trakktor with the \
         `burn` feature"
            .into(),
    )))
}
