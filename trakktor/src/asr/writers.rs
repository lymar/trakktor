//! Transcript file writers for `asr whisper --output-format`.
//!
//! Ports the reference Whisper output writers: plain text, WebVTT and SubRip
//! subtitles, TSV, and the full JSON result. stdout keeps printing the result
//! as usual; these persist copies to files named after the audio, one per
//! requested format (`all` writes every format).

use std::{
    fs,
    path::{Path, PathBuf},
};

use trakktor_core::{
    asr::{
        gigaam, vosk,
        whisper::{Transcription, WhisperError},
    },
    vad::SpeechSegment,
};

use crate::cli::OutputFormatArg;

/// A segment as the text subtitle writers see it: a time span and its text.
/// Implemented by every engine's segment type so the txt/vtt/srt/tsv renderers
/// are shared.
trait CueSegment {
    fn start(&self) -> f64;
    fn end(&self) -> f64;
    fn text(&self) -> &str;
}

impl CueSegment for trakktor_core::asr::whisper::Segment {
    fn start(&self) -> f64 { self.start }
    fn end(&self) -> f64 { self.end }
    fn text(&self) -> &str { &self.text }
}

impl CueSegment for gigaam::Segment {
    fn start(&self) -> f64 { self.start }
    fn end(&self) -> f64 { self.end }
    fn text(&self) -> &str { &self.text }
}

impl CueSegment for vosk::Segment {
    fn start(&self) -> f64 { self.start }
    fn end(&self) -> f64 { self.end }
    fn text(&self) -> &str { &self.text }
}

/// The formats [`OutputFormatArg::All`] expands to, in a stable order.
const ALL_FORMATS: [OutputFormatArg; 5] = [
    OutputFormatArg::Txt,
    OutputFormatArg::Vtt,
    OutputFormatArg::Srt,
    OutputFormatArg::Tsv,
    OutputFormatArg::Json,
];

/// Writes the transcript in the requested format(s) into `output_dir`, naming
/// each file after the audio's stem (for example `talk.srt`). Returns the paths
/// written, in the order produced. Creates `output_dir` if it does not exist.
pub fn write_outputs(
    transcription: &Transcription,
    model: &str,
    audio_path: &Path,
    format: OutputFormatArg,
    output_dir: &Path,
    with_words: bool,
    vad_speech: Option<&[SpeechSegment]>,
) -> Result<Vec<PathBuf>, WhisperError> {
    let stem = audio_path
        .file_stem()
        .map(|s| s.to_string_lossy().into_owned())
        .unwrap_or_else(|| "transcript".to_string());

    fs::create_dir_all(output_dir).map_err(|e| {
        WhisperError::Io(format!("{}: {e}", output_dir.display()))
    })?;

    let formats: &[OutputFormatArg] = match format {
        OutputFormatArg::All => &ALL_FORMATS,
        ref single => std::slice::from_ref(single),
    };

    let mut written = Vec::with_capacity(formats.len());
    for &format in formats {
        let (extension, content) =
            render(transcription, model, format, with_words, vad_speech);
        let path = output_dir.join(format!("{stem}.{extension}"));
        fs::write(&path, content).map_err(|e| {
            WhisperError::Io(format!("{}: {e}", path.display()))
        })?;
        written.push(path);
    }
    Ok(written)
}

/// Renders one format to `(extension, file contents)`.
fn render(
    transcription: &Transcription,
    model: &str,
    format: OutputFormatArg,
    with_words: bool,
    vad_speech: Option<&[SpeechSegment]>,
) -> (&'static str, String) {
    match format {
        OutputFormatArg::Txt => ("txt", render_txt(&transcription.segments)),
        OutputFormatArg::Vtt => ("vtt", render_vtt(&transcription.segments)),
        OutputFormatArg::Srt => ("srt", render_srt(&transcription.segments)),
        OutputFormatArg::Tsv => ("tsv", render_tsv(&transcription.segments)),
        OutputFormatArg::Json => (
            "json",
            render_json(transcription, model, with_words, vad_speech),
        ),
        // `All` is expanded by the caller into concrete formats.
        OutputFormatArg::All => unreachable!("all is expanded before render"),
    }
}

/// Writes a GigaAM transcript in the requested format(s) into `output_dir`.
/// Same file layout as [`write_outputs`]; the JSON form is GigaAM's envelope.
pub fn write_gigaam_outputs(
    transcription: &gigaam::Transcription,
    audio_path: &Path,
    format: OutputFormatArg,
    output_dir: &Path,
    with_words: bool,
) -> Result<Vec<PathBuf>, WhisperError> {
    let stem = audio_path
        .file_stem()
        .map(|s| s.to_string_lossy().into_owned())
        .unwrap_or_else(|| "transcript".to_string());

    fs::create_dir_all(output_dir).map_err(|e| {
        WhisperError::Io(format!("{}: {e}", output_dir.display()))
    })?;

    let formats: &[OutputFormatArg] = match format {
        OutputFormatArg::All => &ALL_FORMATS,
        ref single => std::slice::from_ref(single),
    };

    let mut written = Vec::with_capacity(formats.len());
    for &format in formats {
        let (extension, content) = match format {
            OutputFormatArg::Txt => {
                ("txt", render_txt(&transcription.segments))
            },
            OutputFormatArg::Vtt => {
                ("vtt", render_vtt(&transcription.segments))
            },
            OutputFormatArg::Srt => {
                ("srt", render_srt(&transcription.segments))
            },
            OutputFormatArg::Tsv => {
                ("tsv", render_tsv(&transcription.segments))
            },
            OutputFormatArg::Json => {
                let value = crate::output::gigaam_transcription_to_json(
                    transcription,
                    audio_path
                        .file_stem()
                        .and_then(|s| s.to_str())
                        .unwrap_or("gigaam"),
                    None,
                    true,
                    with_words,
                );
                (
                    "json",
                    serde_json::to_string(&value)
                        .expect("serializing a serde_json::Value never fails"),
                )
            },
            OutputFormatArg::All => unreachable!("all is expanded"),
        };
        let path = output_dir.join(format!("{stem}.{extension}"));
        fs::write(&path, content).map_err(|e| {
            WhisperError::Io(format!("{}: {e}", path.display()))
        })?;
        written.push(path);
    }
    Ok(written)
}

/// Writes a Vosk transcript in the requested format(s) into `output_dir`.
/// Same file layout as [`write_outputs`]; the JSON form is Vosk's envelope.
pub fn write_vosk_outputs(
    transcription: &vosk::Transcription,
    model: &str,
    audio_path: &Path,
    format: OutputFormatArg,
    output_dir: &Path,
    with_words: bool,
) -> Result<Vec<PathBuf>, WhisperError> {
    let stem = audio_path
        .file_stem()
        .map(|s| s.to_string_lossy().into_owned())
        .unwrap_or_else(|| "transcript".to_string());

    fs::create_dir_all(output_dir).map_err(|e| {
        WhisperError::Io(format!("{}: {e}", output_dir.display()))
    })?;

    let formats: &[OutputFormatArg] = match format {
        OutputFormatArg::All => &ALL_FORMATS,
        ref single => std::slice::from_ref(single),
    };

    let mut written = Vec::with_capacity(formats.len());
    for &format in formats {
        let (extension, content) = match format {
            OutputFormatArg::Txt => {
                ("txt", render_txt(&transcription.segments))
            },
            OutputFormatArg::Vtt => {
                ("vtt", render_vtt(&transcription.segments))
            },
            OutputFormatArg::Srt => {
                ("srt", render_srt(&transcription.segments))
            },
            OutputFormatArg::Tsv => {
                ("tsv", render_tsv(&transcription.segments))
            },
            OutputFormatArg::Json => {
                let value = crate::output::vosk_transcription_to_json(
                    transcription,
                    model,
                    None,
                    true,
                    with_words,
                );
                (
                    "json",
                    serde_json::to_string(&value)
                        .expect("serializing a serde_json::Value never fails"),
                )
            },
            OutputFormatArg::All => unreachable!("all is expanded"),
        };
        let path = output_dir.join(format!("{stem}.{extension}"));
        fs::write(&path, content).map_err(|e| {
            WhisperError::Io(format!("{}: {e}", path.display()))
        })?;
        written.push(path);
    }
    Ok(written)
}

/// Plain text: one trimmed segment per line.
fn render_txt<S: CueSegment>(segments: &[S]) -> String {
    let mut out = String::new();
    for segment in segments {
        out.push_str(segment.text().trim());
        out.push('\n');
    }
    out
}

/// WebVTT: a `WEBVTT` header then one cue per segment.
fn render_vtt<S: CueSegment>(segments: &[S]) -> String {
    let mut out = String::from("WEBVTT\n\n");
    for segment in segments {
        let start = format_timestamp(segment.start(), false, '.');
        let end = format_timestamp(segment.end(), false, '.');
        out.push_str(&format!("{start} --> {end}\n{}\n\n", cue_text(segment)));
    }
    out
}

/// SubRip (SRT): numbered cues with comma decimal marks and hours always
/// present.
fn render_srt<S: CueSegment>(segments: &[S]) -> String {
    let mut out = String::new();
    for (index, segment) in segments.iter().enumerate() {
        let start = format_timestamp(segment.start(), true, ',');
        let end = format_timestamp(segment.end(), true, ',');
        out.push_str(&format!(
            "{}\n{start} --> {end}\n{}\n\n",
            index + 1,
            cue_text(segment)
        ));
    }
    out
}

/// TSV: a header then `start<TAB>end<TAB>text`, times as integer milliseconds.
fn render_tsv<S: CueSegment>(segments: &[S]) -> String {
    let mut out = String::from("start\tend\ttext\n");
    for segment in segments {
        let start = (segment.start() * 1000.0).round() as i64;
        let end = (segment.end() * 1000.0).round() as i64;
        let text = segment.text().trim().replace('\t', " ");
        out.push_str(&format!("{start}\t{end}\t{text}\n"));
    }
    out
}

/// The full JSON result — the same envelope printed to stdout, compact.
fn render_json(
    transcription: &Transcription,
    model: &str,
    with_words: bool,
    vad_speech: Option<&[SpeechSegment]>,
) -> String {
    let value = crate::output::transcription_to_json(
        transcription,
        model,
        true,
        with_words,
        vad_speech,
    );
    serde_json::to_string(&value)
        .expect("serializing a serde_json::Value never fails")
}

/// Segment text for a subtitle cue: trimmed, with any `-->` neutralized so it
/// cannot be mistaken for a cue-timing arrow.
fn cue_text<S: CueSegment>(segment: &S) -> String {
    segment.text().trim().replace("-->", "->")
}

/// Formats `seconds` as `[HH:]MM:SS<marker>mmm`, rounding to whole
/// milliseconds. `always_hours` forces the `HH:` field (SRT); otherwise it
/// appears only past an hour (VTT).
fn format_timestamp(seconds: f64, always_hours: bool, marker: char) -> String {
    let total_ms = (seconds.max(0.0) * 1000.0).round() as u64;
    let hours = total_ms / 3_600_000;
    let minutes = (total_ms % 3_600_000) / 60_000;
    let secs = (total_ms % 60_000) / 1000;
    let millis = total_ms % 1000;
    let hours_marker = if always_hours || hours > 0 {
        format!("{hours:02}:")
    } else {
        String::new()
    };
    format!("{hours_marker}{minutes:02}:{secs:02}{marker}{millis:03}")
}

#[cfg(test)]
mod tests;
