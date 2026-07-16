use std::path::PathBuf;

use serde_json::Value;

use super::{
    super::{
        constants::SAMPLE_RATE,
        testing::{FakeProvider, peak},
    },
    *,
};

fn tokenizer() -> Tokenizer {
    Tokenizer::new(true, 99, Some("en"), Some(Task::Transcribe)).unwrap()
}

fn repo_path(relative: &str) -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("..")
        .join(relative)
}

fn ids(value: &Value) -> Vec<TokenId> {
    value
        .as_array()
        .unwrap()
        .iter()
        .map(|v| v.as_u64().unwrap() as TokenId)
        .collect()
}

fn transcribe_golden() -> Value {
    let path = repo_path("tmp/whisper_golden/tiny_transcribe.json");
    let raw = std::fs::read_to_string(&path)
        .unwrap_or_else(|e| panic!("reading {}: {e}", path.display()));
    serde_json::from_str(&raw).unwrap()
}

/// Replays the reference trace's decode outputs through a scripted provider:
/// with identical per-window tokens, the segmentation, seek trajectory, and
/// final text must reproduce the reference exactly. Fast (no model), but it
/// needs the developer-local trace golden.
#[test]
#[ignore = "requires the reference trace golden"]
fn segmentation_replays_reference_trace() {
    let golden = transcribe_golden();
    assert!(golden["deterministic"].as_bool().unwrap());

    let tok = Tokenizer::new(
        true,
        99,
        Some(golden["language"].as_str().unwrap()),
        Some(Task::Transcribe),
    )
    .unwrap();
    let n_vocab = tok.n_vocab();

    // One row per accepted token, then eot, per window. Windows the
    // reference judged silent (and skipped) are rebuilt in a "silent" shape
    // — a towering <|nospeech|> over a flat text mass — so the replay takes
    // the same skip decision.
    let ts_begin = tok.timestamp_begin() as usize;
    let no_speech = tok.no_speech() as usize;
    let row = |intended: TokenId, silent: bool| {
        if silent {
            let mut row = vec![0.0f32; n_vocab];
            for x in &mut row[..ts_begin] {
                *x = 9.9;
            }
            row[no_speech] = 50.0;
            row[intended as usize] = 10.0;
            row
        } else {
            peak(n_vocab, intended, 50.0)
        }
    };

    let mut script: Vec<Vec<Vec<f32>>> = Vec::new();
    for call in golden["calls"].as_array().unwrap() {
        let no_speech_prob = call["no_speech_prob"].as_f64().unwrap() as f32;
        let avg_logprob = call["avg_logprob"].as_f64().unwrap() as f32;
        let silent = no_speech_prob > 0.6 && avg_logprob <= -1.0;
        for token in ids(&call["tokens"]) {
            script.push(vec![row(token, silent)]);
        }
        script.push(vec![row(tok.eot(), silent)]);
    }
    let mut provider = FakeProvider::new(n_vocab, script);

    let duration = golden["duration"].as_u64().unwrap() as usize;
    let options = TranscribeOptions {
        language: Some(golden["language"].as_str().unwrap().to_string()),
        ..Default::default()
    };
    let (transcription, traces) =
        transcribe_with_trace(&mut provider, &audio(duration), &options)
            .unwrap();

    let want_segments = golden["segments"].as_array().unwrap();
    if transcription.segments.len() != want_segments.len() {
        for trace in &traces {
            eprintln!(
                "window seek {} -> {} (single_ending={}, consecutive={}, \
                 segments={:?})",
                trace.seek,
                trace.seek_after,
                trace.single_timestamp_ending,
                trace.consecutive_count,
                trace.segments,
            );
        }
        for segment in &transcription.segments {
            eprintln!(
                "ours: id={} seek={} [{:.2}..{:.2}] {:?}",
                segment.id,
                segment.seek,
                segment.start,
                segment.end,
                segment.text
            );
        }
        for want in want_segments {
            eprintln!(
                "want: id={} seek={} [{:.2}..{:.2}] {:?}",
                want["id"],
                want["seek"],
                want["start"],
                want["end"],
                want["text"].as_str().unwrap()
            );
        }
    }
    assert_eq!(
        transcription.segments.len(),
        want_segments.len(),
        "segment count"
    );
    for (segment, want) in transcription.segments.iter().zip(want_segments) {
        let index = segment.id;
        assert_eq!(
            segment.id,
            want["id"].as_u64().unwrap() as usize,
            "segment {index}: id"
        );
        assert_eq!(
            segment.seek,
            want["seek"].as_u64().unwrap() as usize,
            "segment {index}: seek"
        );
        assert_eq!(
            segment.start,
            want["start"].as_f64().unwrap(),
            "segment {index}: start ({})",
            want["text"].as_str().unwrap()
        );
        assert_eq!(
            segment.end,
            want["end"].as_f64().unwrap(),
            "segment {index}: end ({})",
            want["text"].as_str().unwrap()
        );
        assert_eq!(
            segment.text,
            want["text"].as_str().unwrap(),
            "segment {index}: text"
        );
        assert_eq!(
            segment.tokens,
            ids(&want["tokens"]),
            "segment {index}: tokens"
        );
    }
    assert_eq!(transcription.text, golden["text"].as_str().unwrap());
}

fn options() -> TranscribeOptions {
    TranscribeOptions {
        language: Some("en".to_string()),
        ..Default::default()
    }
}

/// Silence of `seconds`; the scripted provider ignores the mel anyway.
fn audio(seconds: usize) -> Vec<f32> { vec![0.0f32; seconds * SAMPLE_RATE] }

#[test]
fn seek_advances_by_timestamps_across_windows() {
    let tok = tokenizer();
    let n_vocab = tok.n_vocab();
    let ts = tok.timestamp_begin();
    let eot = tok.eot();
    let hello = tok.encode(" hello")[0];
    let world = tok.encode(" world")[0];
    let again = tok.encode(" again")[0];
    let done = tok.encode(" done")[0];

    let one = |token: TokenId| vec![peak(n_vocab, token, 50.0)];
    let script = vec![
        // Window 1: two closed segments; the unfinished tail is impossible
        // here, the window ends right at the second pair.
        one(ts),
        one(hello),
        one(ts + 100),
        one(ts + 100),
        one(world),
        one(ts + 200),
        one(ts + 200),
        one(eot),
        // Window 2: single trailing timestamp (no speech after it).
        one(ts),
        one(again),
        one(ts + 50),
        one(eot),
        // Window 3: last (short) window.
        one(ts),
        one(done),
        one(ts + 10),
        one(eot),
    ];
    let mut provider = FakeProvider::new(n_vocab, script);

    let (transcription, traces) =
        transcribe_with_trace(&mut provider, &audio(60), &options()).unwrap();

    // 60 s of content = 6000 frames.
    assert_eq!(transcription.duration, 60.0);
    assert_eq!(transcription.language, "en");
    assert_eq!(transcription.text, " hello world again done");

    let segments = &transcription.segments;
    assert_eq!(segments.len(), 4);
    assert_eq!(
        segments.iter().map(|s| s.id).collect::<Vec<_>>(),
        vec![0, 1, 2, 3]
    );
    assert_eq!(
        segments.iter().map(|s| s.seek).collect::<Vec<_>>(),
        vec![0, 0, 400, 3400]
    );
    assert_eq!(
        segments
            .iter()
            .map(|s| (s.start, s.end))
            .collect::<Vec<_>>(),
        vec![(0.0, 2.0), (2.0, 4.0), (4.0, 5.0), (34.0, 34.2)]
    );
    assert_eq!(segments[0].tokens, vec![ts, hello, ts + 100]);

    assert_eq!(traces.len(), 3);
    // Window 1 ends in a closed pair: seek moves to the last timestamp
    // (200 tokens * 2 frames); the others end in a lone timestamp and jump
    // a whole window.
    assert_eq!(
        traces.iter().map(|t| t.seek_after).collect::<Vec<_>>(),
        vec![400, 3400, 6000]
    );
    assert_eq!(
        traces
            .iter()
            .map(|t| t.single_timestamp_ending)
            .collect::<Vec<_>>(),
        vec![false, true, true]
    );
    assert_eq!(
        traces
            .iter()
            .map(|t| t.consecutive_count)
            .collect::<Vec<_>>(),
        vec![2, 0, 0]
    );
    // The context accumulates across windows.
    assert_eq!(
        traces.iter().map(|t| t.prompt_len).collect::<Vec<_>>(),
        vec![0, 6, 9]
    );
    assert!(traces.iter().all(|t| !t.context_reset));
}

#[test]
fn repetitive_output_falls_back_to_a_higher_temperature() {
    let tok = tokenizer();
    let n_vocab = tok.n_vocab();
    let ts = tok.timestamp_begin();
    let eot = tok.eot();
    let and = tok.encode(" and")[0];
    let hello = tok.encode(" hello")[0];

    let one = |token: TokenId| vec![peak(n_vocab, token, 50.0)];
    let mut script: Vec<Vec<Vec<f32>>> = Vec::new();
    // Attempt at t = 0: " and" times 30 — compresses far too well.
    script.push(one(ts));
    for _ in 0..30 {
        script.push(one(and));
    }
    script.push(one(ts + 100));
    script.push(one(eot));
    // Attempt at t = 0.2: a normal short segment, accepted.
    script.extend([one(ts), one(hello), one(ts + 50), one(eot)]);

    let mut provider = FakeProvider::new(n_vocab, script);
    let (transcription, traces) =
        transcribe_with_trace(&mut provider, &audio(30), &options()).unwrap();

    assert_eq!(traces.len(), 1);
    let attempts = &traces[0].attempts;
    assert_eq!(attempts.len(), 2);
    assert!(attempts[0].needs_fallback);
    assert!(attempts[0].compression_ratio > 2.4);
    assert_eq!(attempts[0].temperature, 0.0);
    assert!(!attempts[1].needs_fallback);
    assert_eq!(attempts[1].temperature, 0.2);

    // The accepted (second) attempt is what lands in the transcription;
    // 0.2 is below the context-reset threshold.
    assert_eq!(transcription.text, " hello");
    assert_eq!(transcription.segments[0].temperature, 0.2);
    assert!(!traces[0].context_reset);
}

#[test]
fn high_temperature_resets_the_context() {
    let tok = tokenizer();
    let n_vocab = tok.n_vocab();
    let ts = tok.timestamp_begin();
    let eot = tok.eot();
    let hello = tok.encode(" hello")[0];
    let world = tok.encode(" world")[0];

    let one = |token: TokenId| vec![peak(n_vocab, token, 50.0)];
    let script = vec![
        one(ts),
        one(hello),
        one(ts + 50),
        one(eot),
        one(ts),
        one(world),
        one(ts + 50),
        one(eot),
    ];
    let mut provider = FakeProvider::new(n_vocab, script);

    let mut options = options();
    options.temperature = vec![0.6]; // accepted immediately, but > 0.5
    let (_, traces) =
        transcribe_with_trace(&mut provider, &audio(60), &options).unwrap();

    assert_eq!(traces.len(), 2);
    assert!(traces[0].context_reset);
    // The reset empties the prompt of the following window.
    assert_eq!(traces[1].prompt_len, 0);
}

#[test]
fn silent_windows_are_skipped() {
    let tok = tokenizer();
    let n_vocab = tok.n_vocab();
    let ts = tok.timestamp_begin();
    let eot = tok.eot();
    let no_speech = tok.no_speech();
    let hello = tok.encode(" hello")[0];

    // Every step: <|nospeech|> towers (the probe reads it before any
    // filtering), while the intended token barely rises above a flat text
    // mass — so the selected text carries dismal log-probabilities, which
    // is exactly the silent-window pattern. Timestamps stay at a low floor
    // so their joint mass does not trip the timestamp-forcing rule.
    let step = |intended: TokenId| {
        let mut row = vec![0.0f32; n_vocab];
        for x in &mut row[..ts as usize] {
            *x = 9.9;
        }
        row[no_speech as usize] = 50.0;
        row[intended as usize] = 10.0;
        vec![row]
    };
    let script = vec![step(ts), step(hello), step(ts + 50), step(eot)];
    let mut provider = FakeProvider::new(n_vocab, script);

    let (transcription, traces) =
        transcribe_with_trace(&mut provider, &audio(30), &options()).unwrap();

    assert_eq!(traces.len(), 1);
    let trace = &traces[0];
    // The silence exception accepts the window at t = 0 (no fallback), and
    // the skip check then drops it whole.
    assert_eq!(trace.attempts.len(), 1);
    assert!(!trace.attempts[0].needs_fallback);
    assert!(trace.attempts[0].no_speech_prob > 0.6);
    assert!(trace.attempts[0].avg_logprob < -1.0);
    assert!(trace.should_skip);
    assert_eq!(trace.seek_after, 3000);
    assert!(trace.segments.is_empty());
    assert!(transcription.segments.is_empty());
    assert_eq!(transcription.text, "");
}

// ---------------------------------------------------------------------------
// Opt-in differential test against the reference implementation. Needs the
// developer-local checkpoint (`tmp/models/whisper-tiny/`), the local sample
// audio, and the trace produced by
// `scripts/asr/whisper/gen_transcribe_golden.py`; run with --ignored.
// ---------------------------------------------------------------------------

#[cfg(feature = "whisper-runtime")]
mod reference_parity {
    use candle_core::Device;

    use super::{
        super::super::{
            audio::{AudioDecoder, FfmpegDecoder},
            runtime::CandleRuntime,
        },
        *,
    };

    #[test]
    #[ignore = "requires the local checkpoint, sample audio, and trace golden"]
    fn decision_trace_matches_reference() {
        let golden = transcribe_golden();
        assert!(
            golden["deterministic"].as_bool().unwrap(),
            "the reference trace used non-zero temperatures; pick another \
             slice for exact comparison"
        );

        // The identical audio slice the reference transcribed.
        let offset = golden["offset"].as_u64().unwrap() as usize;
        let duration = golden["duration"].as_u64().unwrap() as usize;
        let pcm = FfmpegDecoder::default()
            .decode(&repo_path("tmp/sample.mp3"))
            .expect("decoding the local sample");
        let start = (offset * SAMPLE_RATE).min(pcm.len());
        let end = ((offset + duration) * SAMPLE_RATE).min(pcm.len());
        let audio = &pcm[start..end];

        let mut runtime = CandleRuntime::load(
            &repo_path("tmp/models/whisper-tiny"),
            Device::Cpu,
        )
        .expect("loading the local whisper-tiny checkpoint");

        let options = TranscribeOptions {
            language: None, // detected, as in the reference run
            beam_size: Some(5),
            best_of: Some(5),
            ..Default::default()
        };
        let (transcription, traces) =
            transcribe_with_trace(&mut runtime, audio, &options).unwrap();

        assert_eq!(
            transcription.language,
            golden["language"].as_str().unwrap(),
            "detected language"
        );

        // Every decode attempt, in order: same temperature, same tokens.
        let calls = golden["calls"].as_array().unwrap();
        let ours: Vec<&FallbackAttempt> =
            traces.iter().flat_map(|t| &t.attempts).collect();
        assert_eq!(ours.len(), calls.len(), "number of decode attempts");
        for (index, (attempt, call)) in ours.iter().zip(calls).enumerate() {
            assert_eq!(
                f64::from(attempt.temperature),
                call["temperature"].as_f64().unwrap(),
                "attempt {index}: temperature"
            );
            assert_eq!(
                attempt.tokens,
                ids(&call["tokens"]),
                "attempt {index}: tokens"
            );
            let avg = call["avg_logprob"].as_f64().unwrap() as f32;
            assert!(
                (attempt.avg_logprob - avg).abs() < 5e-3,
                "attempt {index}: avg_logprob {} vs {avg}",
                attempt.avg_logprob
            );
            let no_speech = call["no_speech_prob"].as_f64().unwrap() as f32;
            assert!(
                (attempt.no_speech_prob - no_speech).abs() < 5e-3,
                "attempt {index}: no_speech_prob {} vs {no_speech}",
                attempt.no_speech_prob
            );
        }

        // Segments: identical boundaries, text, and tokens.
        let want_segments = golden["segments"].as_array().unwrap();
        assert_eq!(transcription.segments.len(), want_segments.len());
        for (segment, want) in transcription.segments.iter().zip(want_segments)
        {
            assert_eq!(segment.id, want["id"].as_u64().unwrap() as usize);
            assert_eq!(segment.seek, want["seek"].as_u64().unwrap() as usize);
            assert_eq!(segment.start, want["start"].as_f64().unwrap());
            assert_eq!(segment.end, want["end"].as_f64().unwrap());
            assert_eq!(segment.text, want["text"].as_str().unwrap());
            assert_eq!(segment.tokens, ids(&want["tokens"]));
        }

        assert_eq!(transcription.text, golden["text"].as_str().unwrap());
    }
}

#[test]
fn instantaneous_or_empty_segments_are_cleared() {
    let tok = tokenizer();
    let n_vocab = tok.n_vocab();
    let ts = tok.timestamp_begin();
    let eot = tok.eot();
    let space = tok.encode(" ")[0];

    // A segment whose text is a lone space: kept as a segment, but emptied,
    // and it contributes nothing to the context.
    let one = |token: TokenId| vec![peak(n_vocab, token, 50.0)];
    let script = vec![one(ts), one(space), one(ts + 50), one(eot)];
    let mut provider = FakeProvider::new(n_vocab, script);

    let (transcription, traces) =
        transcribe_with_trace(&mut provider, &audio(30), &options()).unwrap();

    assert_eq!(transcription.segments.len(), 1);
    let segment = &transcription.segments[0];
    assert_eq!(segment.text, "");
    assert!(segment.tokens.is_empty());
    assert_eq!(transcription.text, "");
    assert_eq!(traces[0].segments.len(), 1);
}
