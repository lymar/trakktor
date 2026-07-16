use super::{
    super::model::{CrossQk, ModelDims},
    decoder::{BeamSearchDecoder, GreedyDecoder, TokenDecoder},
    filters::{ApplyTimestampRules, LogitFilter, SuppressBlank},
    *,
};

fn tokenizer() -> Tokenizer {
    Tokenizer::new(true, 99, Some("en"), Some(Task::Transcribe)).unwrap()
}

// ---------------------------------------------------------------------------
// compression ratio, slicing helpers, option validation
// ---------------------------------------------------------------------------

#[test]
fn compression_ratio_flags_repetition() {
    let repetitive = "again and again and again and again and again and again";
    let varied = "the quick brown fox jumps over one very lazy sleeping dog";
    assert!(compression_ratio(repetitive) > compression_ratio(varied));
    assert_eq!(compression_ratio(""), 0.0);
}

#[test]
fn python_tail_matches_slice_semantics() {
    let values: Vec<TokenId> = vec![1, 2, 3, 4, 5];
    assert_eq!(python_tail(values.clone(), 2), vec![4, 5]);
    assert_eq!(python_tail(values.clone(), 0), values); // [-0:] is everything
    assert_eq!(python_tail(values.clone(), -2), vec![3, 4, 5]);
    assert_eq!(python_tail(values.clone(), 99), values);
}

#[test]
fn suppress_set_expands_the_default_sentinel() {
    let tok = tokenizer();
    let options = DecodingOptions::default();
    let set = build_suppress_set(&tok, &options).unwrap();

    let mut expected: Vec<TokenId> = tok.non_speech_tokens();
    expected.extend([
        tok.transcribe(),
        tok.translate(),
        tok.sot(),
        tok.sot_prev(),
        tok.sot_lm(),
        tok.no_speech(),
    ]);
    expected.sort_unstable();
    expected.dedup();
    assert_eq!(set, expected);
}

#[test]
fn inconsistent_options_fail_validation() {
    let cases = [
        DecodingOptions {
            beam_size: Some(5),
            best_of: Some(5),
            temperature: 1.0,
            ..Default::default()
        },
        DecodingOptions {
            best_of: Some(5),
            ..Default::default()
        },
        DecodingOptions {
            patience: Some(2.0),
            ..Default::default()
        },
        DecodingOptions {
            length_penalty: Some(1.5),
            ..Default::default()
        },
        DecodingOptions {
            suppress_tokens: Some(vec![-2]),
            ..Default::default()
        },
    ];
    for options in cases {
        assert!(matches!(
            verify_options(&options),
            Err(WhisperError::InvalidOptions(_))
        ));
    }
    assert!(verify_options(&DecodingOptions::default()).is_ok());
}

// ---------------------------------------------------------------------------
// token decoders
// ---------------------------------------------------------------------------

#[test]
fn greedy_freezes_finished_sequences() {
    let eot: TokenId = 3;
    let mut decoder = GreedyDecoder::new(0.0, eot, Some(7));
    let mut tokens: Vec<Vec<TokenId>> = vec![vec![9], vec![9]];
    let mut sums = vec![0.0f32; 2];

    // Row 0 picks token 1; row 1 picks eot.
    let step1 = vec![vec![1.0, 9.0, 0.0, 0.0], vec![0.0, 0.0, 0.0, 9.0]];
    let (completed, rearrange) =
        decoder.update(&mut tokens, &step1, &mut sums).unwrap();
    assert!(!completed);
    assert!(rearrange.is_none());
    assert_eq!(tokens, vec![vec![9, 1], vec![9, 3]]);
    let sum_after_step1 = sums[1];
    assert!(sum_after_step1 < 0.0);

    // Row 1 is finished: the sampled token is replaced by eot and its
    // cumulative log-probability stops growing.
    let step2 = vec![vec![9.0, 0.0, 0.0, 0.0], vec![9.0, 0.0, 0.0, 0.0]];
    let (completed, _) =
        decoder.update(&mut tokens, &step2, &mut sums).unwrap();
    assert!(!completed);
    assert_eq!(tokens, vec![vec![9, 1, 0], vec![9, 3, 3]]);
    assert_eq!(sums[1], sum_after_step1);

    // Row 0 reaches eot: all sequences complete.
    let step3 = vec![vec![0.0, 0.0, 0.0, 9.0], vec![9.0, 0.0, 0.0, 0.0]];
    let (completed, _) =
        decoder.update(&mut tokens, &step3, &mut sums).unwrap();
    assert!(completed);

    // Finalize appends one eot to every row.
    let (finalized, _) = decoder.finalize(vec![tokens], vec![sums.clone()]);
    assert_eq!(finalized[0][0], vec![9, 1, 0, 3, 3]);
    assert_eq!(finalized[0][1], vec![9, 3, 3, 3, 3]);
}

#[test]
fn beam_search_dedups_ranks_and_finishes() {
    let eot: TokenId = 5;
    let mut decoder = BeamSearchDecoder::new(2, eot, None).unwrap();
    decoder.reset();
    let mut tokens: Vec<Vec<TokenId>> = vec![vec![7], vec![7]];
    let mut sums = vec![0.0f32; 2];

    // Both rows carry identical prefixes, so the candidate sets collapse
    // (dictionary semantics) and the sources point at the last producer.
    let step1 = vec![
        vec![0.0, 5.0, 4.0, 0.0, 0.0, 0.0],
        vec![0.0, 5.0, 4.0, 0.0, 0.0, 0.0],
    ];
    let (completed, rearrange) =
        decoder.update(&mut tokens, &step1, &mut sums).unwrap();
    assert!(!completed);
    assert_eq!(tokens, vec![vec![7, 1], vec![7, 2]]);
    assert_eq!(rearrange.unwrap(), vec![1, 1]);
    assert!(sums[0] > sums[1]);

    // The best continuation of beam 0 is eot: it moves to the finished
    // pool, and both surviving beams descend from row 1.
    let step2 = vec![
        vec![0.0, 0.0, 0.0, 0.0, 0.0, 12.0],
        vec![0.0, 0.0, 6.0, 0.0, 0.0, 0.0],
    ];
    let (completed, rearrange) =
        decoder.update(&mut tokens, &step2, &mut sums).unwrap();
    assert!(!completed); // one finished candidate of the required two
    assert_eq!(tokens, vec![vec![7, 2, 2], vec![7, 2, 0]]);
    assert_eq!(rearrange.unwrap(), vec![1, 1]);

    // Finalize pads with the best unfinished beam, eot-terminated.
    let (candidates, scores) =
        decoder.finalize(vec![tokens], vec![sums.clone()]);
    assert_eq!(candidates[0].len(), 2);
    assert_eq!(candidates[0][0], vec![7, 1, 5]);
    assert_eq!(candidates[0][1], vec![7, 2, 2, 5]);
    assert_eq!(scores[0].len(), 2);
}

// ---------------------------------------------------------------------------
// logit filters
// ---------------------------------------------------------------------------

#[test]
fn suppress_blank_applies_only_at_the_first_sample() {
    let tok = tokenizer();
    let filter = SuppressBlank::new(&tok, 3);
    let n_vocab = tok.n_vocab();
    let space = tok.encode(" ")[0] as usize;

    let mut logits = vec![vec![0.0f32; n_vocab]];
    filter.apply(&mut logits, &[vec![1, 2, 3]]);
    assert_eq!(logits[0][space], f32::NEG_INFINITY);
    assert_eq!(logits[0][tok.eot() as usize], f32::NEG_INFINITY);

    let mut logits = vec![vec![0.0f32; n_vocab]];
    filter.apply(&mut logits, &[vec![1, 2, 3, 4]]);
    assert_eq!(logits[0][space], 0.0);
}

#[test]
fn timestamp_rules_enforce_the_grammar() {
    let tok = tokenizer();
    let n_vocab = tok.n_vocab();
    let ts = tok.timestamp_begin();
    let sample_begin = 3usize;
    let filter = ApplyTimestampRules::new(&tok, sample_begin, Some(50));
    let context_prefix = vec![1, 2, 3];

    // At the very first sample: text is masked, and initial timestamps are
    // capped at index 50.
    let mut logits = vec![vec![0.0f32; n_vocab]];
    filter.apply(&mut logits, &[context_prefix.clone()]);
    assert_eq!(logits[0][100], f32::NEG_INFINITY, "text is masked");
    assert!(logits[0][(ts + 50) as usize].is_finite());
    assert_eq!(logits[0][(ts + 51) as usize], f32::NEG_INFINITY);
    assert_eq!(logits[0][tok.no_timestamps() as usize], f32::NEG_INFINITY);

    // After a lone opening timestamp the segment must get text: every
    // timestamp is masked (closing immediately would make a zero-length
    // segment), while text flows.
    let mut context = context_prefix.clone();
    context.push(ts + 10);
    let mut logits = vec![vec![0.0f32; n_vocab]];
    filter.apply(&mut logits, &[context.clone()]);
    assert!(logits[0][100].is_finite());
    assert_eq!(logits[0][(ts + 9) as usize], f32::NEG_INFINITY);
    assert_eq!(logits[0][(ts + 10) as usize], f32::NEG_INFINITY);

    // After a closed pair (a segment boundary): timestamps are masked,
    // text flows.
    context.push(ts + 10);
    let mut logits = vec![vec![0.0f32; n_vocab]];
    // Give text real mass so the probability rule does not kick in.
    logits[0][100] = 10.0;
    filter.apply(&mut logits, &[context.clone()]);
    assert_eq!(logits[0][(ts + 10) as usize], f32::NEG_INFINITY);
    assert_eq!(logits[0][(ts + 1200) as usize], f32::NEG_INFINITY);
    assert!(logits[0][100].is_finite());

    // Monotonicity after text inside a segment: the next timestamp may not
    // precede the last one (nonzero-length segments).
    let mut context = context_prefix.clone();
    context.extend([ts + 10, ts + 10, 100]);
    let mut logits = vec![vec![0.0f32; n_vocab]];
    logits[0][100] = 10.0;
    filter.apply(&mut logits, &[context]);
    assert_eq!(logits[0][(ts + 10) as usize], f32::NEG_INFINITY);
    assert!(logits[0][(ts + 11) as usize].is_finite());
}

#[test]
fn timestamp_probability_rule_forces_timestamps() {
    let tok = tokenizer();
    let n_vocab = tok.n_vocab();
    let ts = tok.timestamp_begin() as usize;
    let filter = ApplyTimestampRules::new(&tok, 3, Some(50));

    // Context past the first sample, inside no open pair: text would be
    // allowed, but the timestamp mass dominates every single text token.
    let context =
        vec![1, 2, 3, tok.timestamp_begin(), tok.timestamp_begin(), 100];
    let mut logits = vec![vec![0.0f32; n_vocab]];
    // Every timestamp slightly positive: their joint mass wins.
    for x in &mut logits[0][ts..] {
        *x = 2.0;
    }
    filter.apply(&mut logits, &[context]);
    assert_eq!(logits[0][100], f32::NEG_INFINITY, "text is forced out");
}

// ---------------------------------------------------------------------------
// end-to-end decode over a scripted provider
// ---------------------------------------------------------------------------

/// A provider that replays scripted logits: call `n` fills every fed
/// position of row `r` with `script[n][r]`.
struct FakeProvider {
    dims: ModelDims,
    script: Vec<Vec<Vec<f32>>>,
    calls: usize,
    rearranges: Vec<Vec<usize>>,
    sessions: usize,
}

impl FakeProvider {
    fn new(n_vocab: usize, script: Vec<Vec<Vec<f32>>>) -> Self {
        Self {
            dims: ModelDims {
                n_mels: 80,
                n_audio_ctx: 1500,
                n_audio_state: 8,
                n_audio_head: 2,
                n_audio_layer: 1,
                n_vocab,
                n_text_ctx: 448,
                n_text_state: 8,
                n_text_head: 2,
                n_text_layer: 1,
            },
            script,
            calls: 0,
            rearranges: Vec::new(),
            sessions: 0,
        }
    }
}

impl ForwardProvider for FakeProvider {
    type AudioFeatures = ();

    fn dims(&self) -> &ModelDims { &self.dims }

    fn encode(
        &mut self,
        _mel_window: &super::super::feature::MelWindow,
    ) -> Result<(), WhisperError> {
        Ok(())
    }

    fn begin_decode(
        &mut self,
        _n_batch: usize,
        _features: &(),
    ) -> Result<(), WhisperError> {
        self.sessions += 1;
        Ok(())
    }

    fn decode_step(
        &mut self,
        step_tokens: &[TokenId],
        n_batch: usize,
    ) -> Result<Logits, WhisperError> {
        let rows = &self.script[self.calls];
        self.calls += 1;
        let n_positions = step_tokens.len() / n_batch;
        let mut data =
            Vec::with_capacity(n_batch * n_positions * self.dims.n_vocab);
        for row in rows.iter().take(n_batch) {
            for _ in 0..n_positions {
                data.extend_from_slice(row);
            }
        }
        Ok(Logits::new(n_batch, n_positions, self.dims.n_vocab, data))
    }

    fn rearrange_kv_cache(
        &mut self,
        source_indices: &[usize],
    ) -> Result<(), WhisperError> {
        self.rearranges.push(source_indices.to_vec());
        Ok(())
    }

    fn end_decode(&mut self) {}

    fn forward_with_cross_qk(
        &mut self,
        _tokens: &[TokenId],
        _features: &(),
    ) -> Result<(Logits, CrossQk), WhisperError> {
        Err(WhisperError::InvalidModel("not scripted".into()))
    }
}

/// A logits row with one dominant token.
fn peak(n_vocab: usize, index: TokenId, height: f32) -> Vec<f32> {
    let mut row = vec![0.0f32; n_vocab];
    row[index as usize] = height;
    row
}

#[test]
fn decode_greedy_produces_a_timestamped_segment() {
    let tok = tokenizer();
    let n_vocab = tok.n_vocab();
    let ts = tok.timestamp_begin();
    let text_token: TokenId = tok.encode(" hello")[0];

    // <|0.00|>, text, <|0.20|>, eot — a single closed segment.
    let script = vec![
        vec![peak(n_vocab, ts, 12.0)],
        vec![peak(n_vocab, text_token, 12.0)],
        vec![peak(n_vocab, ts + 10, 12.0)],
        vec![peak(n_vocab, tok.eot(), 12.0)],
    ];
    let mut provider = FakeProvider::new(n_vocab, script);

    let options = DecodingOptions {
        language: Some("en".into()),
        ..Default::default()
    };
    let result = decode(&mut provider, &tok, &(), &options).unwrap();

    assert_eq!(result.tokens, vec![ts, text_token, ts + 10]);
    // Decoded, timestamps dropped, trimmed.
    assert_eq!(result.text, tok.decode(&[text_token]).trim());
    assert_eq!(result.language, "en");
    assert_eq!(result.temperature, 0.0);
    assert!(result.avg_logprob < 0.0);
    assert!((0.0..=1.0).contains(&result.no_speech_prob));
    assert!(result.compression_ratio > 0.0);
    assert_eq!(provider.sessions, 1);
    assert_eq!(provider.calls, 4);
}

#[test]
fn decode_beam_reorders_the_cache() {
    let tok = tokenizer();
    let n_vocab = tok.n_vocab();
    let ts = tok.timestamp_begin();
    let text_token: TokenId = tok.encode(" hello")[0];

    // Two beams; every step peaks the same way for both rows, ending after
    // one segment. What matters here is that the cache is reordered on
    // every update with in-range indices.
    let script = vec![
        vec![peak(n_vocab, ts, 12.0), peak(n_vocab, ts, 12.0)],
        vec![
            peak(n_vocab, text_token, 12.0),
            peak(n_vocab, text_token, 12.0),
        ],
        vec![peak(n_vocab, ts + 10, 12.0), peak(n_vocab, ts + 10, 12.0)],
        vec![
            peak(n_vocab, tok.eot(), 12.0),
            peak(n_vocab, tok.eot(), 12.0),
        ],
        vec![
            peak(n_vocab, tok.eot(), 12.0),
            peak(n_vocab, tok.eot(), 12.0),
        ],
        vec![
            peak(n_vocab, tok.eot(), 12.0),
            peak(n_vocab, tok.eot(), 12.0),
        ],
    ];
    let mut provider = FakeProvider::new(n_vocab, script);

    let options = DecodingOptions {
        language: Some("en".into()),
        beam_size: Some(2),
        ..Default::default()
    };
    let result = decode(&mut provider, &tok, &(), &options).unwrap();

    assert_eq!(result.tokens, vec![ts, text_token, ts + 10]);
    assert!(!provider.rearranges.is_empty());
    for indices in &provider.rearranges {
        assert_eq!(indices.len(), 2);
        assert!(indices.iter().all(|&i| i < 2));
    }
}

#[test]
fn decode_detects_language_when_unset() {
    let tok = tokenizer();
    let n_vocab = tok.n_vocab();
    let ts = tok.timestamp_begin();
    let ru = tok.to_language_token("ru").unwrap();

    // First call serves language detection (peak on <|ru|>), the rest the
    // main loop.
    let script = vec![
        vec![peak(n_vocab, ru, 12.0)],
        vec![peak(n_vocab, ts, 12.0)],
        vec![peak(n_vocab, 6550, 12.0)],
        vec![peak(n_vocab, ts + 10, 12.0)],
        vec![peak(n_vocab, tok.eot(), 12.0)],
    ];
    let mut provider = FakeProvider::new(n_vocab, script);

    let options = DecodingOptions::default(); // language: None
    let result = decode(&mut provider, &tok, &(), &options).unwrap();

    assert_eq!(result.language, "ru");
    assert_eq!(provider.sessions, 2); // detection + main loop
}

#[test]
fn detect_language_masks_everything_else() {
    let tok = tokenizer();
    let n_vocab = tok.n_vocab();
    let ru = tok.to_language_token("ru").unwrap();

    // Even with a huge peak on a text token, only language tokens compete.
    let mut row = peak(n_vocab, 42, 50.0);
    row[ru as usize] = 5.0;
    let mut provider = FakeProvider::new(n_vocab, vec![vec![row]]);

    let (code, distribution) =
        detect_language(&mut provider, &tok, &()).unwrap();
    assert_eq!(code, "ru");
    assert_eq!(distribution.len(), 99);
    let total: f32 = distribution.iter().map(|&(_, p)| p).sum();
    assert!((total - 1.0).abs() < 1e-3);
}

// ---------------------------------------------------------------------------
// Opt-in parity tests against the reference implementation. They need the
// developer-local checkpoint (`tmp/models/whisper-tiny/`) and the goldens
// produced by `scripts/asr/whisper/gen_decode_golden.py`; run with --ignored.
// ---------------------------------------------------------------------------

#[cfg(feature = "whisper-runtime")]
mod reference_parity {
    use std::path::PathBuf;

    use candle_core::Device;
    use serde_json::Value;

    use super::{
        super::super::{
            audio::pad_or_trim,
            constants::{N_FRAMES, N_SAMPLES},
            feature::{MelBands, log_mel_spectrogram},
            runtime::CandleRuntime,
        },
        *,
    };

    fn repo_path(relative: &str) -> PathBuf {
        PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .join("..")
            .join(relative)
    }

    fn golden() -> Value {
        let path = repo_path("tmp/whisper_golden/tiny_decode.json");
        let raw = std::fs::read_to_string(&path)
            .unwrap_or_else(|e| panic!("reading {}: {e}", path.display()));
        serde_json::from_str(&raw).unwrap()
    }

    fn fixture_features(
        runtime: &mut CandleRuntime,
    ) -> <CandleRuntime as ForwardProvider>::AudioFeatures {
        let pcm: Vec<f32> = include_bytes!("../testdata/sample_2s.pcm.bin")
            .chunks_exact(4)
            .map(|b| f32::from_le_bytes([b[0], b[1], b[2], b[3]]))
            .collect();
        let padded = pad_or_trim(&pcm, N_SAMPLES);
        let mel = log_mel_spectrogram(&padded, MelBands::Mel80, 0);
        runtime.encode(&mel.window(0, N_FRAMES)).unwrap()
    }

    fn check_case(case: &Value, options: &DecodingOptions) {
        let mut runtime = CandleRuntime::load(
            &repo_path("tmp/models/whisper-tiny"),
            Device::Cpu,
        )
        .expect("loading the local whisper-tiny checkpoint");
        let features = fixture_features(&mut runtime);
        let tok = tokenizer();

        let result = decode(&mut runtime, &tok, &features, options).unwrap();

        let want_tokens: Vec<TokenId> = case["tokens"]
            .as_array()
            .unwrap()
            .iter()
            .map(|v| v.as_u64().unwrap() as TokenId)
            .collect();
        assert_eq!(result.tokens, want_tokens, "token sequences must match");
        assert_eq!(result.text, case["text"].as_str().unwrap());
        let avg = case["avg_logprob"].as_f64().unwrap() as f32;
        assert!(
            (result.avg_logprob - avg).abs() < 5e-3,
            "avg_logprob {} vs reference {avg}",
            result.avg_logprob
        );
        let no_speech = case["no_speech_prob"].as_f64().unwrap() as f32;
        assert!(
            (result.no_speech_prob - no_speech).abs() < 5e-3,
            "no_speech_prob {} vs reference {no_speech}",
            result.no_speech_prob
        );
        let ratio = case["compression_ratio"].as_f64().unwrap() as f32;
        assert!(
            (result.compression_ratio - ratio).abs() < 1e-4,
            "compression_ratio {} vs reference {ratio}",
            result.compression_ratio
        );
    }

    #[test]
    #[ignore = "requires the local checkpoint and reference goldens"]
    fn greedy_window_matches_reference() {
        let golden = golden();
        let options = DecodingOptions {
            language: Some("en".into()),
            ..Default::default()
        };
        check_case(&golden["greedy"], &options);
    }

    #[test]
    #[ignore = "requires the local checkpoint and reference goldens"]
    fn beam_window_matches_reference() {
        let golden = golden();
        let options = DecodingOptions {
            language: Some("en".into()),
            beam_size: Some(5),
            ..Default::default()
        };
        check_case(&golden["beam5"], &options);
    }

    #[test]
    #[ignore = "requires the local checkpoint and reference goldens"]
    fn language_detection_matches_reference() {
        let golden = golden();
        let mut runtime = CandleRuntime::load(
            &repo_path("tmp/models/whisper-tiny"),
            Device::Cpu,
        )
        .expect("loading the local whisper-tiny checkpoint");
        let features = fixture_features(&mut runtime);
        let tok = tokenizer();

        let (code, distribution) =
            detect_language(&mut runtime, &tok, &features).unwrap();
        assert_eq!(code, golden["detected_language"].as_str().unwrap());
        let want_prob =
            golden["detected_language_prob"].as_f64().unwrap() as f32;
        let got_prob = distribution
            .iter()
            .find(|&&(c, _)| c == code)
            .map(|&(_, p)| p)
            .unwrap();
        assert!(
            (got_prob - want_prob).abs() < 5e-3,
            "prob {got_prob} vs reference {want_prob}"
        );
    }
}
