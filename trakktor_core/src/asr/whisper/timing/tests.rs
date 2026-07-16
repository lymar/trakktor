use super::{
    super::{testing::FakeProvider, tokenizer::Task},
    *,
};

fn tokenizer() -> Tokenizer {
    Tokenizer::new(true, 99, Some("en"), Some(Task::Transcribe)).unwrap()
}

#[test]
fn median_filter_reflects_at_the_edges() {
    let mut rows = vec![vec![1.0f32, 9.0, 1.0, 9.0, 1.0]];
    median_filter(&mut rows, 3);
    assert_eq!(rows[0], vec![9.0, 1.0, 9.0, 1.0, 9.0]);

    // Rows shorter than the padding stay unchanged.
    let mut short = vec![vec![5.0f32]];
    median_filter(&mut short, 3);
    assert_eq!(short[0], vec![5.0]);
}

#[test]
fn dtw_follows_the_cheap_diagonal() {
    let cost = vec![
        vec![0.0f32, 5.0, 5.0, 5.0],
        vec![5.0, 0.0, 5.0, 5.0],
        vec![5.0, 5.0, 0.0, 0.0],
    ];
    let (text_indices, time_indices) = dtw(&cost);
    assert_eq!(text_indices, vec![0, 1, 2, 2]);
    assert_eq!(time_indices, vec![0, 1, 2, 3]);
}

#[test]
fn merge_punctuations_attaches_marks_to_words() {
    let word = |text: &str, tokens: Vec<TokenId>| WordTiming {
        word: text.to_string(),
        tokens,
        start: 0.0,
        end: 0.1,
        probability: 1.0,
    };
    let mut alignment = vec![
        word(" (", vec![1]),
        word(" hi", vec![2]),
        word(",", vec![3]),
        word(" there", vec![4]),
    ];
    merge_punctuations(&mut alignment, "\"'“¿([{-", "\"'.。,，!！?？:：”)]}、");

    // The opening bracket prepends to " hi"; the comma appends to it.
    assert_eq!(alignment[0].word, "");
    assert!(alignment[0].tokens.is_empty());
    assert_eq!(alignment[1].word, " ( hi,");
    assert_eq!(alignment[1].tokens, vec![1, 2, 3]);
    assert_eq!(alignment[2].word, "");
    assert_eq!(alignment[3].word, " there");
}

#[test]
fn find_alignment_yields_monotonic_words() {
    let tok = tokenizer();
    let mut provider = FakeProvider::new(tok.n_vocab(), Vec::new());
    provider.cross_qk_diagonal = Some((1500, 0));

    let text_tokens = tok.encode(" hello world again");
    let timings =
        find_alignment(&mut provider, &tok, &text_tokens, &(), 3000, &[(0, 0)])
            .unwrap();

    let words: Vec<&str> =
        timings.iter().map(|timing| timing.word.as_str()).collect();
    assert_eq!(words, vec![" hello", " world", " again"]);
    for timing in &timings {
        assert!(timing.start <= timing.end);
        assert!(timing.probability > 0.0);
    }
    for pair in timings.windows(2) {
        assert!(pair[0].end <= pair[1].start);
    }

    // Empty input aligns to nothing.
    assert!(
        find_alignment(&mut provider, &tok, &[], &(), 3000, &[(0, 0)])
            .unwrap()
            .is_empty()
    );
}

#[test]
fn get_end_prefers_words_and_falls_back_to_segments() {
    let segment = |end: f64, words: Vec<Word>| Segment {
        id: 0,
        seek: 0,
        start: 0.0,
        end,
        text: String::new(),
        tokens: Vec::new(),
        words,
        temperature: 0.0,
        avg_logprob: 0.0,
        compression_ratio: 0.0,
        no_speech_prob: 0.0,
    };
    let word = Word {
        word: " hi".to_string(),
        start: 1.0,
        end: 1.5,
        probability: 0.9,
    };

    assert_eq!(get_end(&[]), None);
    assert_eq!(get_end(&[segment(3.0, vec![])]), Some(3.0));
    assert_eq!(
        get_end(&[segment(3.0, vec![word.clone()]), segment(4.0, vec![])]),
        Some(1.5)
    );
}
