//! `asr whisper`: flag mapping, model resolution, and the transcription run.

use std::{
    io::{IsTerminal, Write},
    path::Path,
};

use trakktor_core::{
    asr::whisper::{
        self, AudioDecoder, ForwardProvider, TranscribeOptions, WhisperError,
        alignment_heads,
    },
    vad::{self, SpeechSegment, VadOptions},
};

use crate::{cli::WhisperArgs, error::CliError, output};

mod gigaam;
mod writers;

pub(crate) use gigaam::run_gigaam;

/// Runs one transcription end to end: resolve (and if needed download) the
/// model, load it, decode the audio, transcribe, and print the result.
pub(crate) fn run_whisper(
    args: &WhisperArgs,
    model_dir: &Path,
    json: bool,
    pretty: bool,
) -> Result<(), CliError> {
    let suppress_tokens = parse_suppress_tokens(&args.suppress_tokens)?;
    // Resolve clip/range options up front so bad ones fail before any download
    // or decode. With `--vad`, clips come from detected speech instead, but the
    // optional `--start`/`--end` range is still validated here.
    let manual_clips = if args.vad {
        Vec::new()
    } else {
        resolve_clip_timestamps(args)?
    };
    let vad_range = if args.vad {
        parse_vad_range(args)?
    } else {
        None
    };
    let temperature = temperature_schedule(
        args.temperature,
        args.temperature_increment_on_fallback.0,
    );

    let resolved = whisper::resolve_model(
        model_dir,
        &args.model,
        &mut download_progress(),
    )?;
    let precision = args.precision.to_core();

    let options = TranscribeOptions {
        temperature,
        compression_ratio_threshold: args
            .compression_ratio_threshold
            .0
            .map(|v| v as f32),
        logprob_threshold: args.logprob_threshold.0.map(|v| v as f32),
        no_speech_threshold: args.no_speech_threshold.0.map(|v| v as f32),
        condition_on_previous_text: args.condition_on_previous_text,
        initial_prompt: args.initial_prompt.clone(),
        carry_initial_prompt: args.carry_initial_prompt,
        clip_timestamps: manual_clips,
        language: args.language.clone(),
        task: args.task.to_core(),
        beam_size: args.beam_size.0,
        best_of: args.best_of.0,
        patience: args.patience.0,
        length_penalty: args.length_penalty.0,
        suppress_tokens,
        word_timestamps: matches!(
            args.timestamps,
            crate::cli::TimestampsArg::Word
        ),
        prepend_punctuations: args.prepend_punctuations.clone(),
        append_punctuations: args.append_punctuations.clone(),
        hallucination_silence_threshold: args.hallucination_silence_threshold.0,
        alignment_heads: resolved
            .name
            .and_then(alignment_heads)
            .map(<[(usize, usize)]>::to_vec),
    };

    match args.runtime {
        crate::cli::RuntimeArg::Candle => {
            let runtime = match args.device {
                crate::cli::DeviceArg::Cpu => {
                    whisper::CandleRuntime::load_cpu(&resolved.dir, precision)?
                },
                crate::cli::DeviceArg::Metal => {
                    load_metal(&resolved.dir, precision)?
                },
            };
            transcribe_and_output(
                runtime, args, options, vad_range, json, pretty,
            )
        },
        crate::cli::RuntimeArg::Burn => run_burn(
            args,
            &resolved.dir,
            precision,
            options,
            vad_range,
            json,
            pretty,
        ),
    }
}

/// Decodes the audio, optionally detects speech, runs the transcription on
/// the loaded runtime, and prints (and optionally writes) the result. The
/// runtime-agnostic tail of `run_whisper`.
fn transcribe_and_output<P: ForwardProvider>(
    mut runtime: P,
    args: &WhisperArgs,
    options: TranscribeOptions,
    vad_range: Option<(f64, f64)>,
    json: bool,
    pretty: bool,
) -> Result<(), CliError> {
    eprintln!("decoding audio...");
    let audio = decode_audio(args)?;

    // Optional VAD: detect speech, then restrict to the `--start`/`--end`
    // range if one was given.
    let vad_speech: Option<Vec<SpeechSegment>> = if args.vad {
        eprintln!("detecting speech...");
        let mut speech = vad::detect_speech(&audio, &vad_options(args))?;
        if let Some((lo, hi)) = vad_range {
            speech = intersect_speech(&speech, lo, hi);
        }
        Some(speech)
    } else {
        None
    };

    // With VAD, collapse the detected speech into a dense buffer, transcribed
    // whole and then mapped back to the original timeline.
    let collapsed = match &vad_speech {
        Some(speech) if !speech.is_empty() => {
            Some(vad::collapse::collapse(&audio, speech))
        },
        _ => None,
    };
    // VAD that found nothing means an empty result — never a whole-file run.
    let no_speech = matches!(&vad_speech, Some(speech) if speech.is_empty());
    let buffer: &[f32] = collapsed
        .as_ref()
        .map_or(&audio[..], |c| c.buffer.as_slice());

    let started = std::time::Instant::now();
    let mut transcription = if no_speech {
        empty_transcription(&audio, options.language.clone())
    } else {
        let mut report = transcribe_progress(started);
        // With VAD the loop runs on the dense speech buffer, so its progress is
        // in buffer time against the (shorter) speech duration. Map the
        // position back to the original timeline and report against the
        // original duration, so the length shown is the file's, not the
        // speech's. Without VAD the position passes through unchanged.
        let mut progress = |p: whisper::TranscribeProgress| match &collapsed {
            Some(collapsed) => report(whisper::TranscribeProgress {
                processed_seconds: collapsed.mapping.map(p.processed_seconds),
                total_seconds: collapsed.duration,
            }),
            None => report(p),
        };
        whisper::transcribe_with_progress(
            &mut runtime,
            buffer,
            &options,
            &mut progress,
        )?
    };
    // With VAD, map the buffer-timeline timestamps back to the original.
    if let Some(collapsed) = &collapsed {
        remap_transcription(&mut transcription, &collapsed.mapping);
        transcription.duration = collapsed.duration;
    }

    // Close the in-place progress line on a terminal with a final 100% and the
    // total wall time; `\x1b[K` clears the leftover of the longer live line.
    if !no_speech && std::io::stderr().is_terminal() {
        let total = clock(transcription.duration);
        let elapsed = clock(started.elapsed().as_secs_f64());
        eprintln!(
            "\r✓ transcribing {total} / {total} (100%) · {elapsed} \
             elapsed\x1b[K"
        );
    }

    let vad_speech = vad_speech.as_deref();
    output::print_transcription(
        &transcription,
        &args.model,
        args.timestamps,
        vad_speech,
        json,
        pretty,
    );

    // Optionally persist the transcript to files; stdout above already carried
    // the result, so the written paths are reported on stderr as diagnostics.
    if let Some(format) = args.output_format {
        let with_words =
            matches!(args.timestamps, crate::cli::TimestampsArg::Word);
        let paths = writers::write_outputs(
            &transcription,
            &args.model,
            &args.audio,
            format,
            &args.output_dir,
            with_words,
            vad_speech,
        )?;
        for path in &paths {
            eprintln!("wrote {}", path.display());
        }
    }
    Ok(())
}

/// Builds [`VadOptions`] from the `--vad-*` flags.
fn vad_options(args: &WhisperArgs) -> VadOptions {
    VadOptions {
        threshold: args.vad_threshold,
        min_speech_duration_ms: args.vad_min_speech_duration_ms,
        min_silence_duration_ms: args.vad_min_silence_duration_ms,
        speech_pad_ms: args.vad_speech_pad_ms,
        max_speech_duration_s: args
            .vad_max_speech_duration_s
            .0
            .map(|s| s as f32),
    }
}

/// The `--start`/`--end` range (seconds) for `--vad`, validated. `None` when
/// neither is given (the whole file).
fn parse_vad_range(args: &WhisperArgs) -> Result<Option<(f64, f64)>, CliError> {
    if args.start.is_none() && args.end.is_none() {
        return Ok(None);
    }
    let lo = args.start.map_or(0.0, |t| t.0);
    let hi = match args.end {
        Some(end) if end.0 <= lo => {
            return Err(WhisperError::InvalidOptions(format!(
                "--end ({}) must be greater than --start ({lo})",
                end.0
            ))
            .into());
        },
        Some(end) => end.0,
        None => f64::INFINITY,
    };
    Ok(Some((lo, hi)))
}

/// Clips each speech segment to `[lo, hi]`, dropping any that fall outside.
fn intersect_speech(
    speech: &[SpeechSegment],
    lo: f64,
    hi: f64,
) -> Vec<SpeechSegment> {
    speech
        .iter()
        .filter_map(|segment| {
            let start = segment.start.max(lo);
            let end = segment.end.min(hi);
            (end > start).then_some(SpeechSegment { start, end })
        })
        .collect()
}

/// Maps every segment and word timestamp of a collapse transcription back to
/// the original timeline.
fn remap_transcription(
    transcription: &mut whisper::Transcription,
    mapping: &vad::TimeMapping,
) {
    for segment in &mut transcription.segments {
        segment.start = mapping.map(segment.start);
        segment.end = mapping.map(segment.end);
        for word in &mut segment.words {
            word.start = mapping.map(word.start);
            word.end = mapping.map(word.end);
        }
    }
}

/// An empty transcription for when VAD found no speech: no text, no segments,
/// the original audio duration.
fn empty_transcription(
    audio: &[f32],
    language: Option<String>,
) -> whisper::Transcription {
    whisper::Transcription {
        text: String::new(),
        segments: Vec::new(),
        language: language.unwrap_or_default(),
        duration: audio.len() as f64 / whisper::constants::SAMPLE_RATE as f64,
    }
}

/// Decodes the input audio with the decoder selected by `--audio-decoder`.
fn decode_audio(args: &WhisperArgs) -> Result<Vec<f32>, CliError> {
    let audio = match args.audio_decoder {
        crate::cli::AudioDecoderArg::Builtin => {
            whisper::BuiltinDecoder.decode(&args.audio)?
        },
        crate::cli::AudioDecoderArg::Ffmpeg => {
            whisper::FfmpegDecoder.decode(&args.audio)?
        },
    };
    Ok(audio)
}

/// Loads the model on Metal (builds with the `metal` feature).
#[cfg(feature = "metal")]
fn load_metal(
    model_dir: &Path,
    precision: whisper::Precision,
) -> Result<whisper::CandleRuntime, CliError> {
    Ok(whisper::CandleRuntime::load_metal(model_dir, precision)?)
}

/// Without the `metal` feature, `--device metal` is a validation error.
#[cfg(not(feature = "metal"))]
fn load_metal(
    _model_dir: &Path,
    _precision: whisper::Precision,
) -> Result<whisper::CandleRuntime, CliError> {
    Err(WhisperError::InvalidOptions(
        "this build has no Metal support; install or build trakktor with the \
         `metal` feature"
            .into(),
    )
    .into())
}

/// Runs the transcription on the burn runtime (builds with the `burn`
/// feature); the burn Metal backend is independent of the candle `metal`
/// feature.
#[cfg(feature = "burn")]
fn run_burn(
    args: &WhisperArgs,
    model_dir: &Path,
    precision: whisper::Precision,
    options: TranscribeOptions,
    vad_range: Option<(f64, f64)>,
    json: bool,
    pretty: bool,
) -> Result<(), CliError> {
    let runtime = match args.device {
        crate::cli::DeviceArg::Cpu => {
            whisper::BurnRuntime::load_cpu(model_dir, precision)?
        },
        crate::cli::DeviceArg::Metal => {
            whisper::BurnRuntime::load_metal(model_dir, precision)?
        },
    };
    transcribe_and_output(runtime, args, options, vad_range, json, pretty)
}

/// Without the `burn` feature, `--runtime burn` is a validation error.
#[cfg(not(feature = "burn"))]
fn run_burn(
    _args: &WhisperArgs,
    _model_dir: &Path,
    _precision: whisper::Precision,
    _options: TranscribeOptions,
    _vad_range: Option<(f64, f64)>,
    _json: bool,
    _pretty: bool,
) -> Result<(), CliError> {
    Err(WhisperError::InvalidOptions(
        "this build has no burn runtime; install or build trakktor with the \
         `burn` feature"
            .into(),
    )
    .into())
}

/// The reference schedule: from the starting temperature up to 1.0 in
/// `increment` steps; no increment means a single fixed temperature.
fn temperature_schedule(start: f32, increment: Option<f64>) -> Vec<f32> {
    let Some(step) = increment else {
        return vec![start];
    };
    let mut schedule = Vec::new();
    let mut index = 0u32;
    loop {
        let t = f64::from(start) + step * f64::from(index);
        if t > 1.0 + 1e-6 {
            break;
        }
        schedule.push(t as f32);
        index += 1;
    }
    if schedule.is_empty() {
        // A start above 1.0: keep it as a single-temperature schedule.
        schedule.push(start);
    }
    schedule
}

/// `--suppress-tokens` csv → token ids; empty disables suppression.
fn parse_suppress_tokens(csv: &str) -> Result<Option<Vec<i64>>, CliError> {
    let trimmed = csv.trim();
    if trimmed.is_empty() {
        return Ok(Some(Vec::new()));
    }
    let mut tokens = Vec::new();
    for part in trimmed.split(',') {
        let token: i64 = part.trim().parse().map_err(|_| {
            WhisperError::InvalidOptions(format!(
                "suppress-tokens: `{part}` is not an integer"
            ))
        })?;
        tokens.push(token);
    }
    Ok(Some(tokens))
}

/// `--clip-timestamps` csv → second offsets.
fn parse_clip_timestamps(csv: &str) -> Result<Vec<f32>, CliError> {
    let trimmed = csv.trim();
    if trimmed.is_empty() {
        return Ok(Vec::new());
    }
    let mut clips = Vec::new();
    for part in trimmed.split(',') {
        let seconds: f32 = part.trim().parse().map_err(|_| {
            WhisperError::InvalidOptions(format!(
                "clip-timestamps: `{part}` is not a number"
            ))
        })?;
        clips.push(seconds);
    }
    Ok(clips)
}

/// Resolves the clips to transcribe. `--start`/`--end` (mutually exclusive with
/// `--clip-timestamps`) form a single clip; otherwise the raw
/// `--clip-timestamps` list is used. A lone `--start` runs to the end of the
/// audio; a lone `--end` runs from the beginning.
fn resolve_clip_timestamps(args: &WhisperArgs) -> Result<Vec<f32>, CliError> {
    if args.start.is_none() && args.end.is_none() {
        return parse_clip_timestamps(&args.clip_timestamps);
    }
    let start = args.start.map_or(0.0, |t| t.0);
    let mut clips = vec![start as f32];
    if let Some(end) = args.end {
        if end.0 <= start {
            return Err(WhisperError::InvalidOptions(format!(
                "--end ({}) must be greater than --start ({start})",
                end.0
            ))
            .into());
        }
        clips.push(end.0 as f32);
    }
    Ok(clips)
}

/// A transcription progress reporter: the audio position and percentage on
/// stderr, rewritten in place on a terminal and printed once per advance
/// otherwise. Diagnostics only — the result never goes to stderr.
fn transcribe_progress(
    started: std::time::Instant,
) -> impl FnMut(whisper::TranscribeProgress) {
    let interactive = std::io::stderr().is_terminal();
    let mut last_percent: u64 = u64::MAX;
    let mut frame: usize = 0;
    let mut last_render: Option<std::time::Instant> = None;
    let mut last_shown: f64 = 0.0;
    let mut eta_pos: f64 = 0.0;
    let mut eta: Option<String> = None;
    move |p: whisper::TranscribeProgress| {
        // The callback fires on every sampled token; render at most every
        // quarter second so a fast model does not flood stderr.
        let now = std::time::Instant::now();
        let too_soon = last_render.is_some_and(|last| {
            now.duration_since(last) < std::time::Duration::from_millis(250)
        });
        if too_soon {
            return;
        }
        // Clamp to the furthest position shown so a beam reshuffle can't tick
        // the mid-window position backward.
        let position = p.processed_seconds.max(last_shown);
        let percent = if p.total_seconds > 0.0 {
            (position / p.total_seconds * 100.0).round() as u64
        } else {
            100
        };
        // Off a terminal, only an advancing position is worth a line.
        if !interactive && percent == last_percent {
            return;
        }
        last_render = Some(now);
        last_shown = position;
        last_percent = percent;
        let processed = clock(position);
        let total = clock(p.total_seconds);
        let elapsed_s = started.elapsed().as_secs_f64();
        // Recompute the time-left estimate only when the position actually
        // advances. Otherwise the wall clock keeps growing while progress
        // stalls and the estimate would creep upward, then snap down on the
        // next advance — jumpy. Hold the last value between advances, and show
        // a placeholder until the first one exists. (Approximate anyway: the
        // first window carries one-time setup, so early estimates run high.)
        if position > eta_pos && position < p.total_seconds {
            eta_pos = position;
            eta = Some(clock(
                elapsed_s * (p.total_seconds - position) / position,
            ));
        }
        let left = eta.as_deref().unwrap_or("--:--");
        let elapsed = clock(elapsed_s);
        if interactive {
            // A spinner shows life even inside a single slow window; the
            // position, percent, and estimate update as the window decodes.
            // `\x1b[K` clears the previous, possibly longer, line.
            const SPINNER: [char; 10] =
                ['⠋', '⠙', '⠹', '⠸', '⠼', '⠴', '⠦', '⠧', '⠇', '⠏'];
            let spin = SPINNER[frame % SPINNER.len()];
            frame = frame.wrapping_add(1);
            eprint!(
                "\r{spin} transcribing {processed} / {total} ({percent}%) · \
                 {elapsed} elapsed · ~{left} left\x1b[K"
            );
            let _ = std::io::stderr().flush();
        } else {
            eprintln!(
                "transcribing {processed} / {total} ({percent}%) · {elapsed} \
                 elapsed · ~{left} left"
            );
        }
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

/// A download progress reporter: percentages on stderr when it is a
/// terminal, one line per file otherwise. Shared with other commands that
/// download model weights.
pub(crate) fn download_progress() -> impl FnMut(&str, u64, Option<u64>) {
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
