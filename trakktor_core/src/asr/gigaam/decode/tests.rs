use super::{ctc_greedy, decode_chunk, frames_to_words};
use crate::asr::gigaam::tokenizer::Tokenizer;

/// A tiny vocab: 0=' ', 1='a', 2='b', 3='c'; blank = 4.
fn tok() -> Tokenizer {
    Tokenizer::charwise(
        [" ", "a", "b", "c"].iter().map(|s| s.to_string()).collect(),
    )
}

#[test]
fn collapse_removes_blanks_and_repeats() {
    // frames: a a _ a b b _ _ c   (blank=4)
    let labels = [1u32, 1, 4, 1, 2, 2, 4, 4, 3];
    let (ids, frames) = ctc_greedy(&labels, labels.len(), 4);
    // First 'a' at 0; 'a' at 3 (separated by blank); 'b' at 4; 'c' at 8.
    assert_eq!(ids, vec![1, 1, 2, 3]);
    assert_eq!(frames, vec![0, 3, 4, 8]);
}

#[test]
fn collapse_respects_length() {
    let labels = [1u32, 2, 3, 3];
    let (ids, frames) = ctc_greedy(&labels, 2, 4);
    assert_eq!(ids, vec![1, 2]);
    assert_eq!(frames, vec![0, 1]);
}

#[test]
fn decode_joins_characters() {
    // "ab c": a b space c
    let labels = [1u32, 2, 0, 3];
    let decoded = decode_chunk(&tok(), &labels, labels.len());
    assert_eq!(decoded.text, "ab c");
    assert_eq!(decoded.token_ids, vec![1, 2, 0, 3]);
}

#[test]
fn words_split_on_space_with_times() {
    // "ab c": tokens a(0) b(1) space(2) c(3); frame_shift 0.04.
    let ids = [1u32, 2, 0, 3];
    let frames = [0usize, 1, 2, 3];
    let words = frames_to_words(&tok(), &ids, &frames, 0.04);
    assert_eq!(words.len(), 2);
    assert_eq!(words[0].text, "ab");
    assert!((words[0].start - 0.0).abs() < 1e-9);
    assert!((words[0].end - 0.08).abs() < 1e-9); // (1+1)*0.04
    assert_eq!(words[1].text, "c");
    assert!((words[1].start - 0.12).abs() < 1e-9); // 3*0.04
    assert!((words[1].end - 0.16).abs() < 1e-9); // (3+1)*0.04
}
