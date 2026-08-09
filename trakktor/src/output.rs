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
        gigaam, vosk,
        whisper::{Segment, Transcription, Word},
    },
    enhance::Enhanced,
    feed::{
        Author, ContentBlock, DiscoveredFeed, Field, MarkReadSummary,
        Publication,
    },
    skill::WriteOutcome,
    structify::Paragraph,
    tts::{
        Voice,
        espeech::{
            Synthesis as EspeechSynthesis, SynthesisOptions as EspeechOptions,
        },
        qwen3_tts::{Sampling, Synthesis},
        silero::{
            ResolvedModel as SileroModel, Synthesis as SileroSynthesis,
            SynthesisOptions as SileroOptions,
        },
    },
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

// ---------------------------------------------------------------------------
// asr vosk
// ---------------------------------------------------------------------------

/// Prints a Vosk transcription. Same common ASR envelope as the other
/// engines, minus any engine-specific diagnostics (transducer decoding has
/// none): `text`, optional `language`, `duration`, `engine`, and (unless
/// `--timestamps none`) `segments`, with per-word timings under
/// `--timestamps word`.
pub fn print_vosk_transcription(
    transcription: &vosk::Transcription,
    model: &str,
    language: Option<&str>,
    timestamps: TimestampsArg,
    json: bool,
    pretty: bool,
) {
    if json {
        let value = vosk_transcription_to_json(
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

/// Builds Vosk's ASR envelope as a JSON value. Shared by stdout printing and
/// the `json` output file.
pub fn vosk_transcription_to_json(
    transcription: &vosk::Transcription,
    model: &str,
    language: Option<&str>,
    include_segments: bool,
    with_words: bool,
) -> Value {
    let mut object = Map::new();
    object.insert("text".into(), Value::String(transcription.text.clone()));
    // Vosk does not detect the language; report it only when the caller
    // supplied a label.
    if let Some(language) = language {
        object.insert("language".into(), Value::String(language.to_string()));
    }
    insert_f64(&mut object, "duration", transcription.duration);
    object.insert("engine".into(), json!({ "name": "vosk", "model": model }));
    if include_segments {
        let segments = transcription
            .segments
            .iter()
            .map(|segment| vosk_segment_to_json(segment, with_words))
            .collect();
        object.insert("segments".into(), Value::Array(segments));
    }
    Value::Object(object)
}

/// One Vosk segment of the envelope: the common fields, and per-word timings
/// under `--timestamps word`. Vosk has no engine-specific diagnostics block.
fn vosk_segment_to_json(segment: &vosk::Segment, with_words: bool) -> Value {
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
// text punctuate
// ---------------------------------------------------------------------------

/// Prints the result of a punctuate run. JSON: an object with the `model`, the
/// restored `text` (sentences joined by a space), and the `sentences` array.
/// Text: the restored text on one line.
pub fn print_punctuate(
    model: &str,
    sentences: &[String],
    json: bool,
    pretty: bool,
) {
    let text = sentences.join(" ");
    if json {
        print_json(
            &json!({ "model": model, "text": text, "sentences": sentences }),
            pretty,
        );
        return;
    }
    println!("{text}");
}

/// Prints the result of a stress-marking run. JSON: the model, the mark form,
/// the marked text, what the run did, and the words it left unmarked. Text: the
/// marked text alone — that is the thing the caller came for.
///
/// `unstressed` is the actionable half of the output: those are exactly the
/// words a `--dict` entry would fix, so the list is always present, even empty.
pub fn print_stress(
    model: &str,
    marked: &trakktor_core::stress::Stressed,
    json: bool,
    pretty: bool,
) {
    if json {
        let stats = &marked.stats;
        print_json(
            &json!({
                "model": model,
                "marker": marked.marker.as_str(),
                "text": marked.text,
                "stats": {
                    "words": stats.words,
                    "stressed": stats.stressed,
                    "yo_restored": stats.yo_restored,
                    "homographs": stats.homographs,
                    "from_dictionary": stats.from_dictionary,
                },
                "unstressed": marked.unstressed,
            }),
            pretty,
        );
        return;
    }
    println!("{}", marked.text);
}

// ---------------------------------------------------------------------------
// tts
// ---------------------------------------------------------------------------

/// Prints the result of a synthesis run. JSON: the common TTS envelope — where
/// the audio was written, its `format`, `sample_rate` and `duration`, the
/// `language` and `voice` used, and the `engine` — with engine diagnostics
/// under the engine's own key. Text: the path, then a one-line summary.
///
/// The audio itself is always a file: raw samples on stdout would not survive
/// the machine-readable contract.
pub fn print_synthesis(
    synthesis: &Synthesis,
    output: &Path,
    format: &str,
    model: &str,
    runtime: &str,
    sampling: &Sampling,
    json: bool,
    pretty: bool,
) {
    let duration = synthesis.speech.duration();
    if json {
        let mut voice = Map::new();
        voice.insert(
            "kind".into(),
            Value::String(synthesis.voice.kind().into()),
        );
        if let Voice::Preset { name } = &synthesis.voice {
            voice.insert("name".into(), Value::String(name.clone()));
        }

        let mut engine_block = json!({
            "frames": synthesis.frames,
            "sampling": sampling.as_str(),
        });
        if synthesis.truncated {
            engine_block
                .as_object_mut()
                .expect("object")
                .insert("truncated".into(), json!(true));
        }
        if let Sampling::TopK {
            top_k,
            temperature,
            seed,
            ..
        } = sampling
        {
            let block = engine_block.as_object_mut().expect("object");
            block.insert("seed".into(), json!(seed));
            block.insert("top_k".into(), json!(top_k));
            block.insert("temperature".into(), json!(temperature));
        }

        let mut object = Map::new();
        object.insert(
            "output".into(),
            Value::String(output.display().to_string()),
        );
        object.insert("format".into(), Value::String(format.into()));
        object
            .insert("sample_rate".into(), json!(synthesis.speech.sample_rate));
        insert_f64(&mut object, "duration", duration);
        object.insert("chunks".into(), json!(synthesis.chunks));
        insert_opt(&mut object, "language", synthesis.language.as_deref());
        object.insert("voice".into(), Value::Object(voice));
        object.insert(
            "engine".into(),
            json!({ "name": "qwen3-tts", "model": model, "runtime": runtime }),
        );
        object.insert("qwen3-tts".into(), engine_block);
        print_json(&Value::Object(object), pretty);
        return;
    }

    println!("{}", output.display());
    println!(
        "{:.2}s\t{} Hz\t{} frames\t{} chunk{}\tvoice {}{}",
        duration,
        synthesis.speech.sample_rate,
        synthesis.frames,
        synthesis.chunks,
        if synthesis.chunks == 1 { "" } else { "s" },
        match &synthesis.voice {
            Voice::Preset { name } => name.as_str(),
            other => other.kind(),
        },
        if synthesis.truncated {
            "\ttruncated"
        } else {
            ""
        }
    );
}

/// Prints the result of one `tts espeech` run.
///
/// The envelope is the shared one; what differs from the other engine is the
/// namespaced block — a flow-matching run has no sampling and cannot be
/// truncated, and instead reports the solver's settings and the reference the
/// voice came from.
#[allow(clippy::too_many_arguments)]
pub fn print_espeech_synthesis(
    synthesis: &EspeechSynthesis,
    output: &Path,
    format: &str,
    model: &str,
    runtime: &str,
    options: &EspeechOptions,
    language: Option<&str>,
    stressed: Option<&crate::tts::stress::Marked>,
    json: bool,
    pretty: bool,
) {
    let duration = synthesis.speech.duration();
    if json {
        let mut voice = Map::new();
        voice.insert(
            "kind".into(),
            Value::String(synthesis.voice.kind().into()),
        );
        if let Voice::Clone { mode, ref_audio } = &synthesis.voice {
            voice.insert("mode".into(), Value::String(mode.as_str().into()));
            voice.insert("ref_audio".into(), Value::String(ref_audio.clone()));
        }

        let mut engine_block = Map::new();
        engine_block.insert("frames".into(), json!(synthesis.frames));
        engine_block.insert("nfe_step".into(), json!(options.nfe_step));
        insert_f64(
            &mut engine_block,
            "cfg_strength",
            f64::from(options.cfg_strength),
        );
        insert_f64(&mut engine_block, "speed", f64::from(options.speed));
        engine_block.insert("seed".into(), json!(options.seed));
        insert_f64(&mut engine_block, "ref_seconds", synthesis.ref_seconds);
        if synthesis.ref_clipped {
            engine_block.insert("ref_clipped".into(), json!(true));
        }
        // What the model actually read, when the marks are not the caller's:
        // without this it is impossible to tell a bad reading from a bad mark.
        if let Some(marked) = stressed {
            engine_block.insert(
                "stressed_text".into(),
                json!(marked.paragraphs.join("\n")),
            );
            engine_block
                .insert("stressed_ref_text".into(), json!(marked.ref_text));
        }

        let mut object = Map::new();
        object.insert(
            "output".into(),
            Value::String(output.display().to_string()),
        );
        object.insert("format".into(), Value::String(format.into()));
        object
            .insert("sample_rate".into(), json!(synthesis.speech.sample_rate));
        insert_f64(&mut object, "duration", duration);
        object.insert("chunks".into(), json!(synthesis.chunks));
        insert_opt(&mut object, "language", language);
        object.insert("voice".into(), Value::Object(voice));
        object.insert(
            "engine".into(),
            json!({ "name": "espeech", "model": model, "runtime": runtime }),
        );
        object.insert("espeech".into(), Value::Object(engine_block));
        print_json(&Value::Object(object), pretty);
        return;
    }

    println!("{}", output.display());
    println!(
        "{:.2}s\t{} Hz\t{} frames\t{} chunk{}\tcloned from {} ({:.2}s{})",
        duration,
        synthesis.speech.sample_rate,
        synthesis.frames,
        synthesis.chunks,
        if synthesis.chunks == 1 { "" } else { "s" },
        match &synthesis.voice {
            Voice::Clone { ref_audio, .. } => ref_audio.as_str(),
            other => other.kind(),
        },
        synthesis.ref_seconds,
        if synthesis.ref_clipped {
            ", clipped"
        } else {
            ""
        }
    );
}

/// Prints the result of one `tts silero` run.
///
/// The envelope is the shared one; the namespaced block carries what only this
/// engine has — the **licence** of the model that was used, which is the point
/// at which a caller can still notice it, and the counts that say what the
/// frontend made of the text.
#[allow(clippy::too_many_arguments)]
pub fn print_silero_synthesis(
    synthesis: &SileroSynthesis,
    output: &Path,
    format: &str,
    model: &SileroModel,
    runtime: &str,
    options: &SileroOptions,
    stressed: Option<&[String]>,
    json: bool,
    pretty: bool,
) {
    let duration = synthesis.speech.duration();
    if json {
        let mut engine_block = Map::new();
        insert_opt(&mut engine_block, "license", model.license());
        engine_block.insert("symbols".into(), json!(synthesis.symbols));
        engine_block.insert("frames".into(), json!(synthesis.frames));
        insert_f64(&mut engine_block, "rate", f64::from(options.rate));
        insert_f64(&mut engine_block, "pitch", f64::from(options.pitch));
        // Silence about what was thrown away would be the wrong kind of quiet:
        // this frontend removes what it cannot spell, Latin included.
        if synthesis.dropped > 0 {
            engine_block
                .insert("dropped_characters".into(), json!(synthesis.dropped));
        }
        if synthesis.skipped > 0 {
            engine_block
                .insert("skipped_paragraphs".into(), json!(synthesis.skipped));
        }
        if !synthesis.utterances.is_empty() {
            engine_block
                .insert("utterances".into(), json!(synthesis.utterances));
        }
        // What the model actually read, when the marks are not the caller's:
        // without this it is impossible to tell a bad reading from a bad mark.
        if let Some(marked) = stressed {
            engine_block
                .insert("stressed_text".into(), json!(marked.join("\n")));
        }

        let mut object = Map::new();
        object.insert(
            "output".into(),
            Value::String(output.display().to_string()),
        );
        object.insert("format".into(), Value::String(format.into()));
        object
            .insert("sample_rate".into(), json!(synthesis.speech.sample_rate));
        insert_f64(&mut object, "duration", duration);
        object.insert("chunks".into(), json!(synthesis.chunks));
        object.insert(
            "voice".into(),
            json!({
                "kind": synthesis.voice.kind(),
                "name": options.voice,
            }),
        );
        object.insert(
            "engine".into(),
            json!({
                "name": "silero",
                "model": model.label(),
                "runtime": runtime,
            }),
        );
        object.insert("silero".into(), Value::Object(engine_block));
        print_json(&Value::Object(object), pretty);
        return;
    }

    println!("{}", output.display());
    println!(
        "{:.2}s\t{} Hz\t{} frames\t{} chunk{}\tvoice {}\t{} ({})",
        duration,
        synthesis.speech.sample_rate,
        synthesis.frames,
        synthesis.chunks,
        if synthesis.chunks == 1 { "" } else { "s" },
        options.voice,
        model.label(),
        model.license().unwrap_or("license unknown"),
    );
}

/// Prints the result of a speech-enhancement run.
pub fn print_enhance(
    enhanced: &Enhanced,
    output: &Path,
    format: &str,
    engine: &str,
    model: &str,
    runtime: &str,
    device: &str,
    plc: bool,
    json: bool,
    pretty: bool,
) {
    let duration = enhanced.duration();
    if json {
        let mut object = Map::new();
        object.insert(
            "output".into(),
            Value::String(output.display().to_string()),
        );
        object.insert("format".into(), Value::String(format.into()));
        object.insert("sample_rate".into(), json!(enhanced.sample_rate));
        object.insert(
            "source_sample_rate".into(),
            json!(enhanced.source_sample_rate),
        );
        insert_f64(&mut object, "duration", duration);
        object.insert("windows".into(), json!(enhanced.windows));
        // What concealment actually did, so a run that filled in half the
        // recording cannot look like one that changed nothing.
        let mut plc_block = Map::new();
        plc_block.insert("enabled".into(), json!(plc));
        plc_block.insert("frames".into(), json!(enhanced.concealed_frames));
        insert_f64(&mut plc_block, "seconds", enhanced.concealed_seconds);
        object.insert("packet_loss".into(), Value::Object(plc_block));
        object.insert(
            "engine".into(),
            json!({
                "name": engine,
                "model": model,
                "runtime": runtime,
                "device": device,
            }),
        );
        print_json(&Value::Object(object), pretty);
        return;
    }

    println!("{}", output.display());
    println!(
        "{:.2}s\t{} Hz\t{} window{}\t{:.2}s concealed\t{engine} {model} \
         ({runtime}, {device})",
        duration,
        enhanced.sample_rate,
        enhanced.windows,
        if enhanced.windows == 1 { "" } else { "s" },
        enhanced.concealed_seconds,
    );
}

/// Prints the voices a model speaks with (`--voice list`).
pub fn print_voices(
    voices: &[&str],
    model: &str,
    license: Option<&str>,
    json: bool,
    pretty: bool,
) {
    if json {
        let mut object = Map::new();
        object.insert("model".into(), Value::String(model.into()));
        insert_opt(&mut object, "license", license);
        object.insert("voices".into(), json!(voices));
        print_json(&Value::Object(object), pretty);
        return;
    }
    for voice in voices {
        println!("{voice}");
    }
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

/// Prints the result of an OCR run. JSON: one object for the run, holding the
/// pages, each page's lines with their quadrangles and confidences, and the
/// models that produced them. Text: the page text, or the assembled Markdown.
///
/// The page separator goes to `stdout` as part of the data rather than to
/// `stderr` as diagnostics: without it the text of a multi-page document
/// cannot be taken apart again.
#[allow(clippy::too_many_arguments)]
pub fn print_ocr(
    pages: &[trakktor_core::ocr::Page],
    figures: &[Vec<trakktor_core::ocr::Quad>],
    // What the layout model made of each page, when it ran at all. Empty means
    // it did not, and the analysis falls back to geometry — which is also what
    // happens page by page where the model found nothing.
    regions: &[Vec<trakktor_core::ocr::layout::Region>],
    // The language the run was told to read, when the engine takes one at all:
    // a generative engine works the writing system out for itself, and
    // reporting a language it never used would be an invention.
    language: Option<&str>,
    models: OcrModels<'_>,
    // What the preprocessing stage did to each page, when it ran. Every
    // quadrangle reported below is put back through this, so that a caller who
    // straightened a photograph still gets boxes on the file they handed in —
    // while the Markdown above is assembled on the straightened page, where
    // the columns are columns and the lines are level.
    prepared: &[trakktor_core::ocr::preprocess::Prepared],
    markdown: bool,
    out: Option<&std::path::Path>,
    json: bool,
    pretty: bool,
) -> Result<(), trakktor_core::ocr::OcrError> {
    use trakktor_core::ocr::{layout, markdown as md};

    // Running heads and page numbers are only recognizable across a
    // document, so the repeated text is found once and handed to every page.
    let repeated = layout::document_furniture(pages);
    let settings = layout::Settings::default();
    let analysed: Vec<(trakktor_core::ocr::Page, layout::Layout)> = pages
        .iter()
        .enumerate()
        .map(|(at, page)| {
            let marked = regions.get(at).map(Vec::as_slice).unwrap_or_default();
            if marked.is_empty() {
                let layout = layout::analyse(page, &repeated, &settings);
                let found = figures.get(at).cloned().unwrap_or_default();
                (page.clone(), layout.with_figures(found))
            } else {
                // The model's own pictures replace the ink-based ones: it says
                // what a picture *is*, and the raster path only says where ink
                // stands that no text box covers.
                (
                    page.clone(),
                    layout::analyse_with(page, marked, &repeated, &settings),
                )
            }
        })
        .collect();

    let text = if markdown {
        let options = md::Options {
            page_separators: pages.len() > 1,
            // The links are only worth writing once the crops are on disk,
            // which is what a run writing the Markdown to a file does.
            image_dir: out.map(|_| crate::ocr::IMAGE_DIR.to_string()),
        };
        md::render(&analysed, &options)
    } else {
        md::plain(pages)
    };

    if let Some(path) = out {
        std::fs::write(path, &text).map_err(|source| {
            trakktor_core::ocr::OcrError::Write {
                path: path.display().to_string(),
                source,
            }
        })?;
    }

    if json {
        // From here on the pages are the caller's own again: same lines, same
        // order, quadrangles moved back onto the file they came from.
        let located: Vec<trakktor_core::ocr::Page> = pages
            .iter()
            .enumerate()
            .map(|(at, page)| {
                let mut page = page.clone();
                if let Some(prepared) = prepared.get(at) {
                    prepared.relocate(&mut page);
                }
                page
            })
            .collect();
        let pages_json: Vec<Value> = located
            .iter()
            .zip(&analysed)
            .enumerate()
            .map(|(at, (page, (_, layout)))| {
                let mut item = json!({
                    "number": page.number,
                    "source": page.source,
                    "width": page.width,
                    "height": page.height,
                    "text": page.text(),
                    "lines": page.lines.iter().map(|line| {
                        let mut item = json!({
                            "text": line.text,
                            "score": round4(line.score),
                            "quad": line.quad.points.iter()
                                .map(|(x, y)| json!([round1(*x), round1(*y)]))
                                .collect::<Vec<_>>(),
                            "rotated": line.rotated,
                        });
                        // Only when it happened: an engine that cannot run
                        // away should not carry a field about running away.
                        if line.truncated {
                            item["truncated"] = json!(true);
                        }
                        item
                    }).collect::<Vec<_>>(),
                });
                // Blocks are the layout model's word, so they appear only when
                // it ran. The geometry's own guesses stay where they are used —
                // in the Markdown — rather than being reported as facts.
                let marked =
                    regions.get(at).map(Vec::as_slice).unwrap_or_default();
                if !marked.is_empty() {
                    item["blocks"] =
                        json!(blocks_json(layout, prepared.get(at)));
                }
                item
            })
            .collect();
        let mut envelope = json!({
            "pages": pages_json,
            "models": models.json(),
        });
        if let Some(language) = language {
            envelope["language"] = json!(language);
        }
        if markdown {
            envelope["markdown"] = json!(text);
        }
        if let Some(path) = out {
            envelope["out"] = json!(path.display().to_string());
        }
        print_json(&envelope, pretty);
        return Ok(());
    }

    print!("{text}");
    Ok(())
}

/// The models a run went through, for the result envelope.
pub struct OcrModels<'a> {
    pub detection: &'a str,
    pub recognition: &'a str,
    /// The layout model, when one ran.
    pub layout: Option<&'a str>,
}

impl OcrModels<'_> {
    fn json(&self) -> Value {
        let mut models = json!({
            "detection": self.detection,
            "recognition": self.recognition,
        });
        if let Some(layout) = self.layout {
            models["layout"] = json!(layout);
        }
        models
    }
}

/// One page's blocks, in reading order, each with the lines it caught.
///
/// Two names travel with a block and they are not the same thing: `label` is
/// the layout model's word, present only where the model claimed the block,
/// and `kind` is what the analysis made of it, which is what the Markdown was
/// built from. A block the model never saw carries the kind alone.
fn blocks_json(
    layout: &trakktor_core::ocr::layout::Layout,
    prepared: Option<&trakktor_core::ocr::preprocess::Prepared>,
) -> Vec<Value> {
    use trakktor_core::ocr::layout::BlockKind;

    layout
        .blocks
        .iter()
        .map(|block| {
            let kind = match block.kind {
                BlockKind::Figure { .. } => "figure",
                BlockKind::Heading { .. } => "heading",
                BlockKind::Paragraph => "paragraph",
                BlockKind::Footnote => "footnote",
                BlockKind::PageFurniture => "furniture",
                BlockKind::Caption => "caption",
            };
            let quad = match prepared {
                None => block.quad,
                Some(prepared) => prepared.locate(&block.quad),
            };
            let mut item = json!({
                "kind": kind,
                "quad": quad.points.iter()
                    .map(|(x, y)| json!([round1(*x), round1(*y)]))
                    .collect::<Vec<_>>(),
                "lines": block.lines,
            });
            if let Some(label) = block.label {
                item["label"] = json!(label.name());
            }
            if let Some(score) = block.score {
                item["score"] = json!(round4(score));
            }
            if let BlockKind::Heading { level } = block.kind {
                item["level"] = json!(level);
            }
            item
        })
        .collect()
}

/// Prints what the layout stage made of the pages, on its own.
///
/// No text: this command answers "what is on this page", and the blocks come
/// back in the order the model ranked them — by confidence, which is what the
/// caller wants when the question is what the model is sure of. Reading order
/// is a property of a *read* page and belongs to the engines.
pub fn print_ocr_layout(
    pages: &[(
        usize,
        String,
        u32,
        u32,
        Vec<trakktor_core::ocr::layout::Region>,
    )],
    model: &str,
    json: bool,
    pretty: bool,
) {
    if json {
        let pages_json: Vec<Value> = pages
            .iter()
            .map(|(number, source, width, height, regions)| {
                json!({
                    "number": number,
                    "source": source,
                    "width": width,
                    "height": height,
                    "blocks": regions.iter().map(|region| json!({
                        "label": region.label.name(),
                        "score": round4(region.score),
                        "quad": region.quad.points.iter()
                            .map(|(x, y)| json!([round1(*x), round1(*y)]))
                            .collect::<Vec<_>>(),
                    })).collect::<Vec<_>>(),
                })
            })
            .collect();
        print_json(
            &json!({ "pages": pages_json, "models": { "layout": model } }),
            pretty,
        );
        return;
    }

    for (at, (number, source, _, _, regions)) in pages.iter().enumerate() {
        if at > 0 {
            println!();
        }
        if pages.len() > 1 {
            println!("=== page {number} · {source} ===");
        }
        for region in regions {
            let (x0, y0, x1, y1) = region.bounds();
            println!(
                "{:<16} {:.2}  {:.0},{:.0} {:.0}×{:.0}",
                region.label.name(),
                region.score,
                x0,
                y0,
                x1 - x0,
                y1 - y0
            );
        }
    }
}

/// Prints the language codes an OCR engine covers, with the recognizer each
/// one selects — the mapping is many-to-one, and knowing which model a code
/// lands on is what tells you which alphabet you will get.
pub fn print_ocr_languages(
    languages: &[(&str, &str)],
    json: bool,
    pretty: bool,
) {
    if json {
        let items: Vec<Value> = languages
            .iter()
            .map(|(code, model)| json!({ "code": code, "model": model }))
            .collect();
        print_json(&json!(items), pretty);
        return;
    }
    for (code, model) in languages {
        println!("{code}\t{model}");
    }
}

fn round4(value: f32) -> f64 { (f64::from(value) * 1e4).round() / 1e4 }

fn round1(value: f32) -> f64 { (f64::from(value) * 10.0).round() / 10.0 }
