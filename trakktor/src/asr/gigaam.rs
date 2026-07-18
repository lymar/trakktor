//! `asr gigaam`: model resolution, streamed decoding, and the transcription
//! run.
//!
//! The audio is decoded block by block and fed straight into the engine's
//! streaming session, which detects speech, plans chunk cuts, transcribes
//! committed chunks, and releases their PCM — so memory stays bounded by a few
//! minutes of audio regardless of the file length.

use std::{
    io::{IsTerminal, Read, Write},
    path::Path,
    time::Instant,
};

use trakktor_core::asr::gigaam::{
    self, GigaamError, StreamTranscriber, TranscribeOptions, TranscribeProgress,
};

use crate::{cli::GigaamArgs, error::CliError, output};

/// Samples per block fed into the session over the ffmpeg pipe (~2 s).
const FFMPEG_BLOCK_SAMPLES: usize = 32 * 1024;

/// Runs one GigaAM transcription end to end: resolve (and if needed download)
/// the model, load it, then stream-decode, segment, and transcribe.
pub(crate) fn run_gigaam(
    args: &GigaamArgs,
    model_dir: &Path,
    json: bool,
    pretty: bool,
) -> Result<(), CliError> {
    let resolved = gigaam::resolve_model(
        model_dir,
        &args.model,
        &mut download_progress(),
    )?;
    let precision = args.precision.to_gigaam();
    let model: Box<dyn gigaam::CtcModel> = match args.runtime {
        crate::cli::RuntimeArg::Candle => match args.device {
            crate::cli::DeviceArg::Cpu => {
                Box::new(gigaam::GigaamModel::load_ctc_cpu(
                    &resolved.ckpt,
                    resolved.config.encoder,
                    resolved.config.mel,
                    resolved.config.num_classes,
                    precision,
                )?)
            },
            crate::cli::DeviceArg::Metal => Box::new(load_metal(
                &resolved.ckpt,
                &resolved.config,
                precision,
            )?),
        },
        crate::cli::RuntimeArg::Burn => {
            load_burn(&resolved.ckpt, &resolved.config, args.device, precision)?
        },
    };
    let tokenizer = resolved.config.tokenizer.build();

    let options = TranscribeOptions {
        word_timestamps: matches!(
            args.timestamps,
            crate::cli::TimestampsArg::Word
        ),
    };

    let started = Instant::now();
    let mut report = transcribe_progress(started);

    let transcription = match args.audio_decoder {
        crate::cli::AudioDecoderArg::Builtin => {
            run_builtin(args, &*model, &tokenizer, options, &mut report)?
        },
        crate::cli::AudioDecoderArg::Ffmpeg => {
            run_ffmpeg(args, &*model, &tokenizer, options, &mut report)?
        },
    };

    if std::io::stderr().is_terminal() {
        let total = clock(transcription.duration);
        let elapsed = clock(started.elapsed().as_secs_f64());
        eprintln!(
            "\r✓ transcribing {total} / {total} (100%) · {elapsed} \
             elapsed\x1b[K"
        );
    }

    output::print_gigaam_transcription(
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
        let paths = crate::asr::writers::write_gigaam_outputs(
            &transcription,
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
    args: &GigaamArgs,
    model: &dyn gigaam::CtcModel,
    tokenizer: &gigaam::Tokenizer,
    options: TranscribeOptions,
    report: &mut dyn FnMut(TranscribeProgress),
) -> Result<gigaam::Transcription, CliError> {
    let mut stream =
        trakktor_core::audio::MonoS16Stream::open(&args.audio, 16_000)
            .map_err(|e| GigaamError::AudioDecode(e.to_string()))?;
    let mut session = StreamTranscriber::new(
        model,
        tokenizer,
        options,
        stream.duration_hint(),
    )?;
    while let Some(block) = stream
        .next_block()
        .map_err(|e| GigaamError::AudioDecode(e.to_string()))?
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
    args: &GigaamArgs,
    model: &dyn gigaam::CtcModel,
    tokenizer: &gigaam::Tokenizer,
    options: TranscribeOptions,
    report: &mut dyn FnMut(TranscribeProgress),
) -> Result<gigaam::Transcription, CliError> {
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
            GigaamError::AudioDecode(format!(
                "could not run ffmpeg ({e}); is it installed and on PATH?"
            ))
        })?;

    // The pipe has no length; progress runs without a total.
    let mut session = StreamTranscriber::new(model, tokenizer, options, None)?;
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
            .map_err(|e| GigaamError::AudioDecode(format!("ffmpeg: {e}")))?;
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
        .map_err(|e| GigaamError::AudioDecode(format!("ffmpeg: {e}")))?;
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
        return Err(CliError::from(GigaamError::AudioDecode(format!(
            "ffmpeg could not decode {}: {detail}",
            args.audio.display()
        ))));
    }
    Ok(session.finish(report)?)
}

/// Loads a CTC model on Metal (builds with the `metal` feature).
#[cfg(feature = "metal")]
fn load_metal(
    ckpt: &Path,
    config: &gigaam::ModelConfig,
    precision: gigaam::Precision,
) -> Result<gigaam::GigaamModel, CliError> {
    Ok(gigaam::GigaamModel::load_ctc_metal(
        ckpt,
        config.encoder,
        config.mel,
        config.num_classes,
        precision,
    )?)
}

/// Without the `metal` feature, `--device metal` is a validation error.
#[cfg(not(feature = "metal"))]
fn load_metal(
    _ckpt: &Path,
    _config: &gigaam::ModelConfig,
    _precision: gigaam::Precision,
) -> Result<gigaam::GigaamModel, CliError> {
    Err(CliError::from(GigaamError::InvalidOptions(
        "this build has no Metal support; install or build trakktor with the \
         `metal` feature"
            .into(),
    )))
}

/// Loads a model on the burn runtime (builds with the `burn` feature); the
/// burn Metal backend is independent of the candle `metal` feature.
#[cfg(feature = "burn")]
fn load_burn(
    ckpt: &Path,
    config: &gigaam::ModelConfig,
    device: crate::cli::DeviceArg,
    precision: gigaam::Precision,
) -> Result<Box<dyn gigaam::CtcModel>, CliError> {
    use trakktor_core::asr::gigaam::runtime_burn;
    let load = match device {
        crate::cli::DeviceArg::Cpu => runtime_burn::load_ctc_cpu,
        crate::cli::DeviceArg::Metal => runtime_burn::load_ctc_metal,
    };
    Ok(load(
        ckpt,
        config.encoder,
        config.mel,
        config.num_classes,
        precision,
    )?)
}

/// Without the `burn` feature, `--runtime burn` is a validation error.
#[cfg(not(feature = "burn"))]
fn load_burn(
    _ckpt: &Path,
    _config: &gigaam::ModelConfig,
    _device: crate::cli::DeviceArg,
    _precision: gigaam::Precision,
) -> Result<Box<dyn gigaam::CtcModel>, CliError> {
    Err(CliError::from(GigaamError::InvalidOptions(
        "this build has no burn runtime; install or build trakktor with the \
         `burn` feature"
            .into(),
    )))
}

/// A transcription progress reporter: the audio position (and percentage when
/// the total is known) on stderr, rewritten in place on a terminal and printed
/// once per advance otherwise. Diagnostics only — the result never goes to
/// stderr.
fn transcribe_progress(started: Instant) -> impl FnMut(TranscribeProgress) {
    let interactive = std::io::stderr().is_terminal();
    let mut last_line = String::new();
    let mut frame: usize = 0;
    move |p: TranscribeProgress| {
        let processed = clock(p.processed_seconds);
        let elapsed = clock(started.elapsed().as_secs_f64());
        let position = match p.total_seconds {
            Some(total) if total > 0.0 => {
                let percent = (p.processed_seconds / total * 100.0)
                    .round()
                    .min(100.0) as u64;
                format!("{processed} / {} ({percent}%)", clock(total))
            },
            _ => processed,
        };
        let line = format!("transcribing {position} · {elapsed} elapsed");
        if interactive {
            const SPINNER: [char; 10] =
                ['⠋', '⠙', '⠹', '⠸', '⠼', '⠴', '⠦', '⠧', '⠇', '⠏'];
            let spin = SPINNER[frame % SPINNER.len()];
            frame = frame.wrapping_add(1);
            eprint!("\r{spin} {line}\x1b[K");
            let _ = std::io::stderr().flush();
        } else if line != last_line {
            eprintln!("{line}");
        }
        last_line = line;
    }
}

/// Formats a number of seconds as `mm:ss`, or `h:mm:ss` past an hour.
fn clock(seconds: f64) -> String {
    let total = seconds.max(0.0) as u64;
    let (hours, minutes, secs) =
        (total / 3600, (total % 3600) / 60, total % 60);
    if hours > 0 {
        format!("{hours}:{minutes:02}:{secs:02}")
    } else {
        format!("{minutes:02}:{secs:02}")
    }
}

/// A download progress reporter: percentages on stderr when it is a terminal,
/// one line per file otherwise.
fn download_progress() -> impl FnMut(&str, u64, Option<u64>) {
    let interactive = std::io::stderr().is_terminal();
    let mut announced: Option<String> = None;
    let mut last_percent: u64 = u64::MAX;
    move |file: &str, done: u64, total: Option<u64>| {
        if announced.as_deref() != Some(file) {
            announced = Some(file.to_string());
            last_percent = u64::MAX;
            if !interactive {
                eprintln!("downloading {file}...");
            }
        }
        if !interactive {
            return;
        }
        match total {
            Some(total) if total > 0 => {
                let percent = done * 100 / total;
                if percent != last_percent {
                    last_percent = percent;
                    eprint!("\rdownloading {file}: {percent}%");
                    if percent == 100 {
                        eprintln!();
                    }
                    let _ = std::io::stderr().flush();
                }
            },
            _ => {
                eprint!("\rdownloading {file}: {} MiB", done >> 20);
                let _ = std::io::stderr().flush();
            },
        }
    }
}
