//! Output formatting: JSON (default) and human-readable text (`--text`).
//!
//! Data goes to stdout; errors go to stderr. In JSON, lists are arrays and a
//! single result is an object; absent values are omitted (never `null`). In
//! text, lists are one record per line with tab-separated fields and composite
//! values collapsed onto one line.

use std::path::{Path, PathBuf};

use serde_json::{Map, Value, json};
use trakktor_core::{
    asr::{
        gigaam,
        whisper::{Segment, Transcription, Word},
    },
    feed::{
        Author, ContentBlock, DiscoveredFeed, Field, MarkReadSummary,
        Publication,
    },
    skill::WriteOutcome,
    structify::Paragraph,
    vad::{Keep, SpeechSegment},
};

use crate::{cli::TimestampsArg, error::CliError};

// ---------------------------------------------------------------------------
// asr whisper
// ---------------------------------------------------------------------------

/// Prints a transcription. JSON: the common ASR envelope — `text`,
/// `language`, `duration`, `engine`, and (unless `--timestamps none`)
/// `segments` with engine diagnostics under a `whisper` key and, with
/// `--timestamps word`, per-word timings. Text: one `[start --> end] text`
/// line per segment, or just the text with `--timestamps none`.
pub fn print_transcription(
    transcription: &Transcription,
    model: &str,
    timestamps: TimestampsArg,
    vad_speech: Option<&[SpeechSegment]>,
    json: bool,
    pretty: bool,
) {
    if json {
        let value = transcription_to_json(
            transcription,
            model,
            timestamps != TimestampsArg::None,
            timestamps == TimestampsArg::Word,
            vad_speech,
        );
        print_json(&value, pretty);
        return;
    }

    if timestamps == TimestampsArg::None {
        println!("{}", transcription.text.trim());
        return;
    }
    for segment in &transcription.segments {
        println!(
            "[{:8.2} --> {:8.2}] {}",
            segment.start,
            segment.end,
            collapse(segment.text.trim())
        );
    }
}

/// Builds the common ASR envelope as a JSON value: `text`, `language`,
/// `duration`, `engine`, and — when `include_segments` — the `segments` array
/// (each with the `whisper` diagnostics, and per-word timings when
/// `with_words`). Shared by stdout printing and the `json` output file.
pub fn transcription_to_json(
    transcription: &Transcription,
    model: &str,
    include_segments: bool,
    with_words: bool,
    vad_speech: Option<&[SpeechSegment]>,
) -> Value {
    let mut object = Map::new();
    object.insert("text".into(), Value::String(transcription.text.clone()));
    object.insert(
        "language".into(),
        Value::String(transcription.language.clone()),
    );
    insert_f64(&mut object, "duration", transcription.duration);
    object.insert(
        "engine".into(),
        json!({ "name": "whisper", "model": model }),
    );
    // When VAD ran, list the detected speech spans (original timeline) as an
    // engine-independent diagnostic — present even if empty.
    if let Some(speech) = vad_speech {
        let spans: Vec<Value> = speech
            .iter()
            .map(|segment| {
                let mut object = Map::new();
                insert_f64(&mut object, "start", segment.start);
                insert_f64(&mut object, "end", segment.end);
                Value::Object(object)
            })
            .collect();
        object.insert("vad".into(), json!({ "speech": spans }));
    }
    if include_segments {
        let segments = transcription
            .segments
            .iter()
            .map(|segment| segment_to_json(segment, with_words))
            .collect();
        object.insert("segments".into(), Value::Array(segments));
    }
    Value::Object(object)
}

/// One segment of the envelope: common fields, then the engine-specific
/// diagnostics under `whisper`.
fn segment_to_json(segment: &Segment, with_words: bool) -> Value {
    let mut object = Map::new();
    object.insert("id".into(), Value::from(segment.id));
    insert_f64(&mut object, "start", segment.start);
    insert_f64(&mut object, "end", segment.end);
    object.insert("text".into(), Value::String(segment.text.clone()));
    if with_words {
        let words = segment.words.iter().map(word_to_json).collect();
        object.insert("words".into(), Value::Array(words));
    }
    object.insert(
        "whisper".into(),
        json!({
            "avg_logprob": f64::from(segment.avg_logprob),
            "compression_ratio": f64::from(segment.compression_ratio),
            "no_speech_prob": f64::from(segment.no_speech_prob),
            "temperature": f64::from(segment.temperature),
        }),
    );
    Value::Object(object)
}

fn word_to_json(word: &Word) -> Value {
    let mut object = Map::new();
    insert_f64(&mut object, "start", word.start);
    insert_f64(&mut object, "end", word.end);
    object.insert("word".into(), Value::String(word.word.clone()));
    object.insert(
        "probability".into(),
        Value::from(f64::from(word.probability)),
    );
    Value::Object(object)
}

// ---------------------------------------------------------------------------
// asr gigaam
// ---------------------------------------------------------------------------

/// Prints a GigaAM transcription. Same common ASR envelope as Whisper, minus
/// the engine-specific diagnostics (GigaAM's CTC decoding has none): `text`,
/// optional `language`, `duration`, `engine`, and (unless `--timestamps none`)
/// `segments`, with per-word timings under `--timestamps word`.
pub fn print_gigaam_transcription(
    transcription: &gigaam::Transcription,
    model: &str,
    language: Option<&str>,
    timestamps: TimestampsArg,
    json: bool,
    pretty: bool,
) {
    if json {
        let value = gigaam_transcription_to_json(
            transcription,
            model,
            language,
            timestamps != TimestampsArg::None,
            timestamps == TimestampsArg::Word,
        );
        print_json(&value, pretty);
        return;
    }

    if timestamps == TimestampsArg::None {
        println!("{}", transcription.text.trim());
        return;
    }
    for segment in &transcription.segments {
        println!(
            "[{:8.2} --> {:8.2}] {}",
            segment.start,
            segment.end,
            collapse(segment.text.trim())
        );
    }
}

/// Builds GigaAM's ASR envelope as a JSON value. Shared by stdout printing and
/// the `json` output file.
pub fn gigaam_transcription_to_json(
    transcription: &gigaam::Transcription,
    model: &str,
    language: Option<&str>,
    include_segments: bool,
    with_words: bool,
) -> Value {
    let mut object = Map::new();
    object.insert("text".into(), Value::String(transcription.text.clone()));
    // GigaAM does not detect the language; report it only when the caller
    // supplied a label.
    if let Some(language) = language {
        object.insert("language".into(), Value::String(language.to_string()));
    }
    insert_f64(&mut object, "duration", transcription.duration);
    object.insert("engine".into(), json!({ "name": "gigaam", "model": model }));
    if include_segments {
        let segments = transcription
            .segments
            .iter()
            .map(|segment| gigaam_segment_to_json(segment, with_words))
            .collect();
        object.insert("segments".into(), Value::Array(segments));
    }
    Value::Object(object)
}

/// One GigaAM segment of the envelope: the common fields, and per-word timings
/// under `--timestamps word`. GigaAM has no engine-specific diagnostics block.
fn gigaam_segment_to_json(
    segment: &gigaam::Segment,
    with_words: bool,
) -> Value {
    let mut object = Map::new();
    object.insert("id".into(), Value::from(segment.id));
    insert_f64(&mut object, "start", segment.start);
    insert_f64(&mut object, "end", segment.end);
    object.insert("text".into(), Value::String(segment.text.clone()));
    if with_words {
        let words = segment
            .words
            .iter()
            .map(|word| {
                let mut object = Map::new();
                insert_f64(&mut object, "start", word.start);
                insert_f64(&mut object, "end", word.end);
                object.insert("word".into(), Value::String(word.text.clone()));
                Value::Object(object)
            })
            .collect();
        object.insert("words".into(), Value::Array(words));
    }
    Value::Object(object)
}

/// Inserts a float, tolerating non-finite values by omission (JSON has no
/// NaN).
fn insert_f64(object: &mut Map<String, Value>, key: &str, value: f64) {
    if let Some(number) = serde_json::Number::from_f64(value) {
        object.insert(key.to_string(), Value::Number(number));
    }
}

// ---------------------------------------------------------------------------
// vad timeline / cut / split
// ---------------------------------------------------------------------------

/// One written clip from `vad split`, for output rendering.
pub struct VadClip {
    /// The written file.
    pub path: PathBuf,
    /// Clip start on the original timeline, seconds.
    pub start: f64,
    /// Clip end on the original timeline, seconds.
    pub end: f64,
    /// Clip duration, seconds.
    pub duration: f64,
}

/// Prints the detected speech timeline (`vad timeline`). JSON: `duration`, a
/// `speech` array of `{start,end}` spans, and a `stats` object (segment count,
/// speech/silence seconds, speech ratio, longest pause). Text: one
/// tab-separated `start<TAB>end` per span.
pub fn print_vad_timeline(
    speech: &[SpeechSegment],
    window: (f64, f64),
    show_window: bool,
    json: bool,
    pretty: bool,
) {
    if json {
        let (start, end) = window;
        let duration = (end - start).max(0.0);
        let (speech_seconds, longest_pause) = speech_stats(speech, window);
        let silence = (duration - speech_seconds).max(0.0);
        let ratio = if duration > 0.0 {
            speech_seconds / duration
        } else {
            0.0
        };
        let mut stats = Map::new();
        stats.insert("segments".into(), Value::from(speech.len() as u64));
        insert_f64(&mut stats, "speech_seconds", speech_seconds);
        insert_f64(&mut stats, "silence_seconds", silence);
        insert_f64(&mut stats, "speech_ratio", ratio);
        insert_f64(&mut stats, "longest_pause_seconds", longest_pause);

        let mut object = Map::new();
        insert_f64(&mut object, "duration", duration);
        // With a `--start`/`--end` window, report its bounds so absolute span
        // times stay unambiguous.
        if show_window {
            let mut bounds = Map::new();
            insert_f64(&mut bounds, "start", start);
            insert_f64(&mut bounds, "end", end);
            object.insert("window".into(), Value::Object(bounds));
        }
        object.insert("speech".into(), Value::Array(spans_to_json(speech)));
        object.insert("stats".into(), Value::Object(stats));
        print_json(&Value::Object(object), pretty);
        return;
    }
    for segment in speech {
        println!("{:.3}\t{:.3}", segment.start, segment.end);
    }
}

/// Prints the result of `vad cut`. JSON: the written `output` path (omitted
/// when nothing was written), `format`, `kept`, the number of kept `segments`,
/// and the source and output durations. Text: the written path, if any.
#[allow(clippy::too_many_arguments)]
pub fn print_vad_cut(
    output: Option<&Path>,
    output_duration: f64,
    source_duration: f64,
    keep: Keep,
    segments: usize,
    format: &str,
    json: bool,
    pretty: bool,
) {
    if json {
        let mut object = Map::new();
        if let Some(path) = output {
            object.insert(
                "output".into(),
                Value::String(path.display().to_string()),
            );
        }
        object.insert("format".into(), Value::String(format.to_string()));
        object
            .insert("kept".into(), Value::String(keep_word(keep).to_string()));
        object.insert("segments".into(), Value::from(segments as u64));
        insert_f64(&mut object, "source_duration", source_duration);
        insert_f64(&mut object, "output_duration", output_duration);
        print_json(&Value::Object(object), pretty);
        return;
    }
    if let Some(path) = output {
        println!("{}", path.display());
    }
}

/// Prints the result of `vad split`. JSON: `output_dir`, `format`, `kept`, the
/// clip `count`, and a `clips` array of `{path,index,start,end,duration}`.
/// Text: one written path per line.
pub fn print_vad_split(
    dir: &Path,
    clips: &[VadClip],
    keep: Keep,
    format: &str,
    json: bool,
    pretty: bool,
) {
    if json {
        let array: Vec<Value> = clips
            .iter()
            .enumerate()
            .map(|(index, clip)| {
                let mut object = Map::new();
                object.insert(
                    "path".into(),
                    Value::String(clip.path.display().to_string()),
                );
                object.insert("index".into(), Value::from(index as u64 + 1));
                insert_f64(&mut object, "start", clip.start);
                insert_f64(&mut object, "end", clip.end);
                insert_f64(&mut object, "duration", clip.duration);
                Value::Object(object)
            })
            .collect();
        let mut object = Map::new();
        object.insert(
            "output_dir".into(),
            Value::String(dir.display().to_string()),
        );
        object.insert("format".into(), Value::String(format.to_string()));
        object
            .insert("kept".into(), Value::String(keep_word(keep).to_string()));
        object.insert("count".into(), Value::from(clips.len() as u64));
        object.insert("clips".into(), Value::Array(array));
        print_json(&Value::Object(object), pretty);
        return;
    }
    for clip in clips {
        println!("{}", clip.path.display());
    }
}

/// `speech` spans as an array of `{start,end}` JSON objects.
fn spans_to_json(speech: &[SpeechSegment]) -> Vec<Value> {
    speech
        .iter()
        .map(|segment| {
            let mut object = Map::new();
            insert_f64(&mut object, "start", segment.start);
            insert_f64(&mut object, "end", segment.end);
            Value::Object(object)
        })
        .collect()
}

/// Total speech seconds and the longest pause (leading, internal, or trailing)
/// within the window. Spans are clipped to the window so windowed stats are
/// correct even if a span straddles an edge.
fn speech_stats(speech: &[SpeechSegment], window: (f64, f64)) -> (f64, f64) {
    let (start, end) = window;
    let speech_seconds: f64 = speech
        .iter()
        .map(|s| (s.end.min(end) - s.start.max(start)).max(0.0))
        .sum();
    let mut longest_pause = 0.0f64;
    let mut cursor = start;
    for segment in speech {
        longest_pause = longest_pause.max(segment.start.min(end) - cursor);
        cursor = cursor.max(segment.end.min(end));
    }
    longest_pause = longest_pause.max(end - cursor);
    (speech_seconds, longest_pause.max(0.0))
}

fn keep_word(keep: Keep) -> &'static str {
    match keep {
        Keep::Speech => "speech",
        Keep::NonSpeech => "non-speech",
    }
}

// ---------------------------------------------------------------------------
// text structify
// ---------------------------------------------------------------------------

/// Prints the paragraphs of a structify run. JSON: an object with the `model`
/// and a `paragraphs` array, each `{ start, end, text }` where `start`/`end`
/// are character offsets into the normalized text. Text: the paragraphs
/// separated by a blank line.
pub fn print_structify(
    model: &str,
    paragraphs: &[Paragraph],
    json: bool,
    pretty: bool,
) {
    if json {
        let array: Vec<Value> = paragraphs
            .iter()
            .map(|paragraph| {
                json!({
                    "start": paragraph.start,
                    "end": paragraph.end,
                    "text": paragraph.text,
                })
            })
            .collect();
        print_json(&json!({ "model": model, "paragraphs": array }), pretty);
        return;
    }

    let joined = paragraphs
        .iter()
        .map(|paragraph| paragraph.text.as_str())
        .collect::<Vec<_>>()
        .join("\n\n");
    println!("{joined}");
}

// ---------------------------------------------------------------------------
// feed discover
// ---------------------------------------------------------------------------

/// Prints discovered feeds.
pub fn print_discover(feeds: &[DiscoveredFeed], json: bool, pretty: bool) {
    if json {
        let array = feeds
            .iter()
            .map(|feed| {
                let mut object = Map::new();
                object.insert("url".into(), Value::String(feed.url.clone()));
                insert_opt(&mut object, "type", feed.mime.as_deref());
                insert_opt(&mut object, "title", feed.title.as_deref());
                Value::Object(object)
            })
            .collect();
        print_json(&Value::Array(array), pretty);
        return;
    }

    for feed in feeds {
        let mut line = feed.url.clone();
        if let Some(mime) = &feed.mime {
            line.push('\t');
            line.push_str(mime);
        }
        if let Some(title) = &feed.title {
            line.push('\t');
            line.push_str(title);
        }
        println!("{line}");
    }
}

// ---------------------------------------------------------------------------
// feed read
// ---------------------------------------------------------------------------

/// Prints publications, projected to the selected fields.
pub fn print_read(
    publications: &[Publication],
    selection: &[Field],
    json: bool,
    pretty: bool,
) {
    if json {
        let array = publications
            .iter()
            .map(|publication| publication_to_json(publication, selection))
            .collect();
        print_json(&Value::Array(array), pretty);
        return;
    }

    for publication in publications {
        println!("{}", text_line(publication, selection));
    }
    if !publications.is_empty() {
        // The `uid` (first column) is each record's primary key; show how to
        // act on it. Printed to stderr so stdout stays a clean data
        // stream.
        eprintln!(
            "hint: each publication's \"uid\" (the first column) is its \
             primary key; mark an item read with: trakktor feed mark-read \
             <uid>"
        );
    }
}

/// Renders one publication as a single text line: the primary-key `uid` first,
/// then the selected fields in order, tab-separated. Every cell is collapsed so
/// newlines/tabs in any field — author names included — cannot break the
/// one-record-per-line layout.
fn text_line(publication: &Publication, selection: &[Field]) -> String {
    let mut cells = vec![publication.uid.clone()];
    cells.extend(
        selection
            .iter()
            .map(|field| collapse(&cell(publication, *field))),
    );
    cells.join("\t")
}

/// Builds the JSON object for one publication: the primary-key `uid` is always
/// present, followed by the selected fields that are present (absent values are
/// omitted).
fn publication_to_json(
    publication: &Publication,
    selection: &[Field],
) -> Value {
    let mut object = Map::new();
    object.insert("uid".into(), Value::String(publication.uid.clone()));
    for field in selection {
        let key = field.key();
        match field {
            Field::IsRead => {
                object.insert(key.into(), Value::Bool(publication.is_read));
            },
            Field::Title => {
                insert_opt(&mut object, key, publication.title.as_deref())
            },
            Field::Link => {
                insert_opt(&mut object, key, publication.link.as_deref())
            },
            Field::Published => {
                insert_opt(&mut object, key, publication.published.as_deref());
            },
            Field::Updated => {
                insert_opt(&mut object, key, publication.updated.as_deref());
            },
            Field::Summary => {
                insert_opt(&mut object, key, publication.summary.as_deref());
            },
            Field::Content => {
                if !publication.content.is_empty() {
                    object.insert(
                        key.into(),
                        content_to_json(&publication.content),
                    );
                }
            },
            Field::Authors => {
                if !publication.authors.is_empty() {
                    object.insert(
                        key.into(),
                        authors_to_json(&publication.authors),
                    );
                }
            },
        }
    }
    Value::Object(object)
}

/// `content` as an array of `{ type?, value }`.
fn content_to_json(blocks: &[ContentBlock]) -> Value {
    let array = blocks
        .iter()
        .map(|block| {
            let mut object = Map::new();
            insert_opt(&mut object, "type", block.mime.as_deref());
            object.insert("value".into(), Value::String(block.value.clone()));
            Value::Object(object)
        })
        .collect();
    Value::Array(array)
}

/// `authors` as an array of `{ name?, email?, uri? }`.
fn authors_to_json(authors: &[Author]) -> Value {
    let array = authors
        .iter()
        .map(|author| {
            let mut object = Map::new();
            insert_opt(&mut object, "name", author.name.as_deref());
            insert_opt(&mut object, "email", author.email.as_deref());
            insert_opt(&mut object, "uri", author.uri.as_deref());
            Value::Object(object)
        })
        .collect();
    Value::Array(array)
}

/// Renders one field of a publication as a text cell. The caller collapses
/// newlines/tabs to spaces uniformly, so this only shapes composite values:
/// content blocks join with spaces, authors join with commas.
fn cell(publication: &Publication, field: Field) -> String {
    match field {
        Field::IsRead => bool_text(publication.is_read).to_string(),
        Field::Title => publication.title.clone().unwrap_or_default(),
        Field::Link => publication.link.clone().unwrap_or_default(),
        Field::Published => publication.published.clone().unwrap_or_default(),
        Field::Updated => publication.updated.clone().unwrap_or_default(),
        Field::Summary => publication.summary.clone().unwrap_or_default(),
        Field::Content => publication
            .content
            .iter()
            .map(|block| block.value.as_str())
            .collect::<Vec<_>>()
            .join(" "),
        Field::Authors => publication
            .authors
            .iter()
            .map(author_display)
            .collect::<Vec<_>>()
            .join(", "),
    }
}

/// Text display of an author: name, else email, else uri.
fn author_display(author: &Author) -> String {
    author
        .name
        .clone()
        .or_else(|| author.email.clone())
        .or_else(|| author.uri.clone())
        .unwrap_or_default()
}

// ---------------------------------------------------------------------------
// feed mark-read
// ---------------------------------------------------------------------------

/// Prints the mark-read summary (single-object form).
pub fn print_mark_read(summary: &MarkReadSummary, json: bool, pretty: bool) {
    if json {
        let value = json!({
            "marked": summary.marked,
            "already_read": summary.already_read,
        });
        print_json(&value, pretty);
    } else {
        println!("marked: {}", summary.marked);
        println!("already_read: {}", summary.already_read);
    }
}

// ---------------------------------------------------------------------------
// skill show / install
// ---------------------------------------------------------------------------

/// Prints the generated skill guide (`skill show`). The guide is a prose
/// document; by default it is wrapped as `{ "content": "…" }`, and `--text`
/// prints the Markdown verbatim.
pub fn print_skill_show(content: &str, json: bool, pretty: bool) {
    if json {
        print_json(&json!({ "content": content }), pretty);
    } else {
        println!("{content}");
    }
}

/// Prints the install result (`skill install`). One destination per run, so the
/// JSON form is a single object
/// `{ "path": "<path>", "status": "written" | "skipped" }`: `written` when the
/// stub was created or overwritten, `skipped` when an existing file was left in
/// place without `--force`.
pub fn print_skill_install(
    path: &Path,
    outcome: WriteOutcome,
    json: bool,
    pretty: bool,
) {
    let status = match outcome {
        WriteOutcome::Written => "written",
        WriteOutcome::Skipped => "skipped",
    };
    if json {
        print_json(
            &json!({ "path": path.display().to_string(), "status": status }),
            pretty,
        );
        return;
    }
    match outcome {
        WriteOutcome::Written => println!("installed: {}", path.display()),
        WriteOutcome::Skipped => {
            println!("skipped (exists, use --force): {}", path.display());
        },
    }
}

// ---------------------------------------------------------------------------
// errors
// ---------------------------------------------------------------------------

/// Emits an error to stderr. JSON form:
/// `{ "error": { "code", "message" } }`; text form: a plain message. The stable
/// `code` is resolved by [`CliError::code`] at the bin boundary.
pub fn emit_error(err: &CliError, json: bool, pretty: bool) {
    if json {
        let value = json!({
            "error": { "code": err.code(), "message": err.to_string() },
        });
        let rendered = render_json(&value, pretty);
        eprintln!("{rendered}");
    } else {
        eprintln!("error: {err}");
    }
}

// ---------------------------------------------------------------------------
// helpers
// ---------------------------------------------------------------------------

/// Inserts `key => value` only when `value` is `Some` (omit absent).
fn insert_opt(object: &mut Map<String, Value>, key: &str, value: Option<&str>) {
    if let Some(value) = value {
        object.insert(key.to_string(), Value::String(value.to_string()));
    }
}

fn bool_text(value: bool) -> &'static str {
    if value { "true" } else { "false" }
}

/// Collapses newlines and tabs to spaces so a value fits one text cell.
fn collapse(value: &str) -> String {
    value
        .chars()
        .map(|c| {
            if matches!(c, '\n' | '\r' | '\t') {
                ' '
            } else {
                c
            }
        })
        .collect()
}

fn print_json(value: &Value, pretty: bool) {
    println!("{}", render_json(value, pretty));
}

fn render_json(value: &Value, pretty: bool) -> String {
    if pretty {
        serde_json::to_string_pretty(value)
    } else {
        serde_json::to_string(value)
    }
    .expect("serializing a serde_json::Value never fails")
}

#[cfg(test)]
mod tests;
