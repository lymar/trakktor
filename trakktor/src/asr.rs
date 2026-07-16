//! `asr whisper`: flag mapping, model resolution, and the transcription run.

use std::{
    io::{IsTerminal, Write},
    path::Path,
};

use trakktor_core::asr::whisper::{
    self, AudioDecoder, TranscribeOptions, WhisperError, alignment_heads,
};

use crate::{cli::WhisperArgs, error::CliError, output};

/// Runs one transcription end to end: resolve (and if needed download) the
/// model, load it, decode the audio, transcribe, and print the result.
pub(crate) fn run_whisper(
    args: &WhisperArgs,
    work_dir: &Path,
    json: bool,
    pretty: bool,
) -> Result<(), CliError> {
    let suppress_tokens = parse_suppress_tokens(&args.suppress_tokens)?;
    let clip_timestamps = parse_clip_timestamps(&args.clip_timestamps)?;
    let temperature = temperature_schedule(
        args.temperature,
        args.temperature_increment_on_fallback.0,
    );

    let resolved = whisper::resolve_model(
        work_dir,
        &args.model,
        &mut download_progress(),
    )?;
    let mut runtime = match args.device {
        crate::cli::DeviceArg::Cpu => {
            whisper::CandleRuntime::load_cpu(&resolved.dir)?
        },
        crate::cli::DeviceArg::Metal => load_metal(&resolved.dir)?,
    };
    let audio = whisper::FfmpegDecoder::default().decode(&args.audio)?;

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
        clip_timestamps,
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

    let transcription = whisper::transcribe(&mut runtime, &audio, &options)?;
    output::print_transcription(
        &transcription,
        &args.model,
        args.timestamps,
        json,
        pretty,
    );
    Ok(())
}

/// Loads the model on Metal (builds with the `metal` feature).
#[cfg(feature = "metal")]
fn load_metal(model_dir: &Path) -> Result<whisper::CandleRuntime, CliError> {
    Ok(whisper::CandleRuntime::load_metal(model_dir)?)
}

/// Without the `metal` feature, `--device metal` is a validation error.
#[cfg(not(feature = "metal"))]
fn load_metal(_model_dir: &Path) -> Result<whisper::CandleRuntime, CliError> {
    Err(WhisperError::InvalidOptions(
        "this build has no Metal support; install or build trakktor with the \
         `metal` feature"
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

/// A download progress reporter: percentages on stderr when it is a
/// terminal, one line per file otherwise.
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
