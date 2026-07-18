//! `vad`: standalone voice-activity editing — timeline, cut, and split.
//!
//! Detection runs on 16 kHz mono (the shape Silero needs). Cutting and
//! splitting run on the decoded original at full quality — its own rate,
//! channels, and bit depth — mapping the speech times (seconds) to native
//! samples. Detection and the native decode are two passes, so only one large
//! buffer is held at a time.

use std::{fs, path::Path};

use trakktor_core::{
    audio::{self, encode},
    vad::{self, EditOptions, Keep, SpeechSegment, VadOptions},
};

use crate::{
    cli::{
        AudioEncoderArg, KeepArg, PresetArg, VadCutArgs, VadDetectArgs,
        VadShapeArgs, VadSplitArgs, VadTimelineArgs,
    },
    error::CliError,
    output::{self, VadClip},
};

/// The sample rate the detector consumes.
const SAMPLE_RATE: u32 = 16_000;

/// Runs `vad timeline`: detect speech and print the spans plus statistics.
pub(crate) fn run_timeline(
    args: &VadTimelineArgs,
    json: bool,
    pretty: bool,
) -> Result<(), CliError> {
    eprintln!("detecting speech...");
    let mono = audio::decode_to_mono_f32(&args.detect.audio, SAMPLE_RATE)?;
    let win = window(&args.detect, samples_seconds(mono.len()));
    let show_window = detection_range(&args.detect).is_some();
    let speech = find_speech(&args.detect, &mono)?;
    output::print_vad_timeline(&speech, win, show_window, json, pretty);
    Ok(())
}

/// Runs `vad cut`: detect speech, then write one file of the kept audio.
pub(crate) fn run_cut(
    args: &VadCutArgs,
    json: bool,
    pretty: bool,
) -> Result<(), CliError> {
    let target = resolve_target(&args.shape);
    let ext = format_ext(&args.shape);
    let speech = find_speech_in_file(&args.detect)?;
    let opts =
        edit_options(&args.shape, args.detect.preset, args.max_silence_ms, 0);
    let fade = resolve_fade(&args.shape, args.detect.preset);

    eprintln!("decoding audio...");
    let decoded = audio::decode_file(&args.detect.audio)?;
    let win = window(&args.detect, decoded.duration_seconds());
    let ranges = vad::edit::plan(&speech, win, &opts);

    if ranges.is_empty() {
        eprintln!("no {} detected; nothing written", kept_word(opts.keep));
        output::print_vad_cut(
            None,
            0.0,
            window_len(win),
            opts.keep,
            0,
            &ext,
            json,
            pretty,
        );
        return Ok(());
    }

    let cut = audio::edit::cut(&decoded, &ranges, fade);
    create_output_dir(&args.shape.output_dir)?;
    let path =
        cut_path(&args.shape.output_dir, &args.detect.audio, opts.keep, &ext);
    write_output(&path, &cut, target, args.shape.bitrate.as_deref())?;
    eprintln!("wrote {}", path.display());
    output::print_vad_cut(
        Some(&path),
        cut.duration_seconds(),
        window_len(win),
        opts.keep,
        ranges.len(),
        &ext,
        json,
        pretty,
    );
    Ok(())
}

/// Runs `vad split`: detect speech, then write one file per kept span.
pub(crate) fn run_split(
    args: &VadSplitArgs,
    json: bool,
    pretty: bool,
) -> Result<(), CliError> {
    let target = resolve_target(&args.shape);
    let ext = format_ext(&args.shape);
    let speech = find_speech_in_file(&args.detect)?;
    let opts = edit_options(
        &args.shape,
        args.detect.preset,
        None,
        args.min_duration_ms,
    );
    let fade = resolve_fade(&args.shape, args.detect.preset);

    eprintln!("decoding audio...");
    let decoded = audio::decode_file(&args.detect.audio)?;
    let win = window(&args.detect, decoded.duration_seconds());
    let ranges = vad::edit::plan(&speech, win, &opts);
    create_output_dir(&args.shape.output_dir)?;

    let clips = audio::edit::split(&decoded, &ranges, fade);
    let mut written: Vec<VadClip> = Vec::new();
    for (range, clip) in ranges.iter().zip(clips.iter()) {
        if clip.frames() == 0 {
            continue;
        }
        let index = written.len() + 1;
        let path = split_path(
            &args.shape.output_dir,
            &args.detect.audio,
            opts.keep,
            index,
            &ext,
        );
        write_output(&path, clip, target, args.shape.bitrate.as_deref())?;
        eprintln!("wrote {}", path.display());
        written.push(VadClip {
            path,
            start: range.0,
            end: range.1,
            duration: clip.duration_seconds(),
        });
    }
    output::print_vad_split(
        &args.shape.output_dir,
        &written,
        opts.keep,
        &ext,
        json,
        pretty,
    );
    Ok(())
}

// ---------------------------------------------------------------------------
// detection
// ---------------------------------------------------------------------------

/// Decodes the file to mono and detects speech (used by cut/split before the
/// separate full-quality decode).
fn find_speech_in_file(
    detect: &VadDetectArgs,
) -> Result<Vec<SpeechSegment>, CliError> {
    eprintln!("detecting speech...");
    let mono = audio::decode_to_mono_f32(&detect.audio, SAMPLE_RATE)?;
    find_speech(detect, &mono)
}

/// Runs detection on an already-decoded mono buffer, then restricts the result
/// to the `--start`/`--end` range if one was given.
fn find_speech(
    detect: &VadDetectArgs,
    mono: &[f32],
) -> Result<Vec<SpeechSegment>, CliError> {
    let mut speech = vad::detect_speech(mono, &vad_options(detect))?;
    if let Some((lo, hi)) = detection_range(detect) {
        speech = intersect(&speech, lo, hi);
    }
    Ok(speech)
}

/// The `--start`/`--end` detection range (seconds), or `None` for the whole
/// file. A non-positive-length range is a usage error.
fn detection_range(detect: &VadDetectArgs) -> Option<(f64, f64)> {
    if detect.start.is_none() && detect.end.is_none() {
        return None;
    }
    let lo = detect.start.map_or(0.0, |t| t.0);
    let hi = match detect.end {
        Some(end) if end.0 <= lo => crate::cli::usage_error(&format!(
            "--end ({}) must be greater than --start ({lo})",
            end.0
        )),
        Some(end) => end.0,
        None => f64::INFINITY,
    };
    Some((lo, hi))
}

/// Clips each speech span to `[lo, hi]`, dropping any that fall outside.
fn intersect(speech: &[SpeechSegment], lo: f64, hi: f64) -> Vec<SpeechSegment> {
    speech
        .iter()
        .filter_map(|segment| {
            let start = segment.start.max(lo);
            let end = segment.end.min(hi);
            (end > start).then_some(SpeechSegment { start, end })
        })
        .collect()
}

/// The working window `[start, end]` in seconds: the `--start`/`--end` range
/// clamped to the audio, or the whole file. All editing happens within it, so a
/// windowed inversion stays inside the window.
fn window(detect: &VadDetectArgs, duration: f64) -> (f64, f64) {
    match detection_range(detect) {
        Some((lo, hi)) => (lo.min(duration), hi.min(duration)),
        None => (0.0, duration),
    }
}

/// The window's length in seconds.
fn window_len((start, end): (f64, f64)) -> f64 { (end - start).max(0.0) }

// ---------------------------------------------------------------------------
// options and presets
// ---------------------------------------------------------------------------

/// The base values a `--preset` supplies for the flags left unset.
struct PresetDefaults {
    threshold: f32,
    min_speech_ms: u32,
    min_silence_ms: u32,
    speech_pad_ms: u32,
    margin_ms: u32,
    merge_gap_ms: u32,
    fade_ms: u32,
}

/// Resolves a preset to its defaults. `tight` is canonical Silero with a small
/// cut margin; `asr` bridges long pauses to keep speech in large chunks;
/// `natural` keeps more room around speech.
fn preset_defaults(preset: PresetArg) -> PresetDefaults {
    match preset {
        PresetArg::Tight => PresetDefaults {
            threshold: 0.5,
            min_speech_ms: 250,
            min_silence_ms: 100,
            speech_pad_ms: 30,
            margin_ms: 50,
            merge_gap_ms: 150,
            fade_ms: 10,
        },
        PresetArg::Asr => PresetDefaults {
            threshold: 0.5,
            min_speech_ms: 0,
            min_silence_ms: 2000,
            speech_pad_ms: 400,
            margin_ms: 0,
            merge_gap_ms: 200,
            fade_ms: 10,
        },
        PresetArg::Natural => PresetDefaults {
            threshold: 0.5,
            min_speech_ms: 250,
            min_silence_ms: 100,
            speech_pad_ms: 30,
            margin_ms: 200,
            merge_gap_ms: 400,
            fade_ms: 15,
        },
    }
}

/// Builds [`VadOptions`] from the detection flags, filling any left unset from
/// the preset.
fn vad_options(detect: &VadDetectArgs) -> VadOptions {
    let base = preset_defaults(detect.preset);
    VadOptions {
        threshold: detect.threshold.unwrap_or(base.threshold),
        min_speech_duration_ms: detect
            .min_speech_duration_ms
            .unwrap_or(base.min_speech_ms),
        min_silence_duration_ms: detect
            .min_silence_duration_ms
            .unwrap_or(base.min_silence_ms),
        speech_pad_ms: detect.speech_pad_ms.unwrap_or(base.speech_pad_ms),
        max_speech_duration_s: detect
            .max_speech_duration_s
            .map(|seconds| seconds as f32),
    }
}

/// Builds [`EditOptions`] from the shaping flags, filling any left unset from
/// the preset.
fn edit_options(
    shape: &VadShapeArgs,
    preset: PresetArg,
    max_silence_ms: Option<u32>,
    min_duration_ms: u32,
) -> EditOptions {
    let base = preset_defaults(preset);
    EditOptions {
        keep: to_keep(shape.keep),
        margin: ms(shape.margin_ms.unwrap_or(base.margin_ms)),
        merge_gap: ms(shape.merge_gap_ms.unwrap_or(base.merge_gap_ms)),
        max_silence: max_silence_ms.map(ms),
        min_duration: ms(min_duration_ms),
    }
}

fn resolve_fade(shape: &VadShapeArgs, preset: PresetArg) -> u32 {
    shape.fade_ms.unwrap_or(preset_defaults(preset).fade_ms)
}

fn to_keep(keep: KeepArg) -> Keep {
    match keep {
        KeepArg::Speech => Keep::Speech,
        KeepArg::NonSpeech => Keep::NonSpeech,
    }
}

fn ms(value: u32) -> f64 { f64::from(value) / 1000.0 }

fn samples_seconds(count: usize) -> f64 {
    count as f64 / f64::from(SAMPLE_RATE)
}

// ---------------------------------------------------------------------------
// output files
// ---------------------------------------------------------------------------

/// Where the output goes: a built-in encoder, or an external ffmpeg process.
#[derive(Clone, Copy)]
enum OutputTarget {
    Native(encode::Format),
    Ffmpeg,
}

/// Resolves the `--audio-encoder`/`--format` pair into a concrete target. An
/// unknown format with the built-in encoder is a usage error.
fn resolve_target(shape: &VadShapeArgs) -> OutputTarget {
    match shape.audio_encoder {
        AudioEncoderArg::Ffmpeg => OutputTarget::Ffmpeg,
        AudioEncoderArg::Builtin => {
            match shape.format.to_ascii_lowercase().as_str() {
                "wav" => OutputTarget::Native(encode::Format::Wav),
                "flac" => OutputTarget::Native(encode::Format::Flac),
                other => crate::cli::usage_error(&format!(
                    "the builtin encoder writes only `wav` or `flac`, not \
                     `{other}`; use `--audio-encoder ffmpeg` for other formats"
                )),
            }
        },
    }
}

/// The lowercase output file extension, also reported as the format name.
fn format_ext(shape: &VadShapeArgs) -> String {
    shape.format.to_ascii_lowercase()
}

/// Writes one buffer to `path` via the resolved target.
fn write_output(
    path: &Path,
    audio: &audio::DecodedAudio,
    target: OutputTarget,
    bitrate: Option<&str>,
) -> Result<(), CliError> {
    match target {
        OutputTarget::Native(format) => encode::write(path, audio, format)?,
        OutputTarget::Ffmpeg => encode::write_ffmpeg(path, audio, bitrate)?,
    }
    Ok(())
}

fn kept_word(keep: Keep) -> &'static str {
    match keep {
        Keep::Speech => "speech",
        Keep::NonSpeech => "non-speech",
    }
}

/// Creates the output directory (and parents) if it does not exist.
fn create_output_dir(dir: &Path) -> Result<(), CliError> {
    fs::create_dir_all(dir).map_err(|source| audio::AudioError::Encode {
        path: dir.to_path_buf(),
        message: source.to_string(),
    })?;
    Ok(())
}

/// The audio file's stem, defaulting to `audio` when it has none.
fn stem(audio: &Path) -> &str {
    audio
        .file_stem()
        .and_then(|name| name.to_str())
        .unwrap_or("audio")
}

/// `<dir>/<stem>.<speech|non-speech>.<ext>` for `cut`.
fn cut_path(
    dir: &Path,
    audio: &Path,
    keep: Keep,
    ext: &str,
) -> std::path::PathBuf {
    dir.join(format!("{}.{}.{ext}", stem(audio), kept_word(keep)))
}

/// `<dir>/<stem>.<speech|non-speech>.<NNN>.<ext>` for `split`.
fn split_path(
    dir: &Path,
    audio: &Path,
    keep: Keep,
    index: usize,
    ext: &str,
) -> std::path::PathBuf {
    dir.join(format!(
        "{}.{}.{index:03}.{ext}",
        stem(audio),
        kept_word(keep)
    ))
}
