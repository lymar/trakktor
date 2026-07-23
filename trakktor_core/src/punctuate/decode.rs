//! Rebuilding punctuated, true-cased, sentence-split text from the merged
//! per-token predictions.
//!
//! Port of the reconstruction loop in `punctuators`
//! (`collectors/pcs_collector.py::produce`) and the README "Manual Usage"
//! snippet. The text is assembled from the SentencePiece **piece strings** (a
//! leading `▁` marks a word start), applying, per token: the pre-punctuation
//! before the first character, per-character upper-casing, the post-punctuation
//! after the last character (or a period after **every** character for
//! `<ACRONYM>`), and a sentence split when the boundary flag is set.

#[cfg(test)]
mod tests;

use super::{
    model::{
        ACRONYM_LABEL, MAX_SUBWORD_LEN, NULL_LABEL, POST_LABELS, PRE_LABELS,
    },
    segment::TokenPred,
    tokenizer::{SpeTokenizer, WORD_PREFIX},
};

/// Rebuilds the sentences from the stream of `(token id, prediction)` pairs.
///
/// With `apply_sbd`, the stream is split into sentences at every predicted full
/// stop; otherwise a single-element vector holds the whole text.
#[must_use]
pub fn reconstruct(
    merged: &[(u32, TokenPred)],
    tokenizer: &SpeTokenizer,
    apply_sbd: bool,
) -> Vec<String> {
    reconstruct_with(merged, |id| tokenizer.piece(id), apply_sbd)
}

/// [`reconstruct`] over an arbitrary piece lookup — the testable core.
fn reconstruct_with<'a>(
    merged: &[(u32, TokenPred)],
    piece: impl Fn(u32) -> &'a str,
    apply_sbd: bool,
) -> Vec<String> {
    let mut sentences: Vec<String> = Vec::new();
    let mut current = String::new();

    for &(id, pred) in merged {
        let chars: Vec<char> = piece(id).chars().collect();
        if chars.is_empty() {
            continue;
        }
        let starts_word = chars[0] == WORD_PREFIX;
        // A word-initial piece emits a separating space (the `▁` itself is
        // skipped).
        if starts_word && !current.is_empty() {
            current.push(' ');
        }
        let char_start = usize::from(starts_word);
        let last = chars.len() - 1;
        let pre_label = PRE_LABELS[usize::from(pred.pre)];
        let post_label = POST_LABELS[usize::from(pred.post)];

        for (ci, &ch) in chars.iter().enumerate().skip(char_start) {
            // Pre-punctuation goes before the piece's first character.
            if ci == char_start && pre_label != NULL_LABEL {
                current.push_str(pre_label);
            }
            // Per-character casing, indexed by the character's position within
            // the piece (position 0 is the `▁`). Pieces longer than the cap
            // head's width are left lowercase past it.
            let upper = ci < MAX_SUBWORD_LEN && pred.cap[ci];
            if upper {
                current.extend(ch.to_uppercase());
            } else {
                current.push(ch);
            }
            // Post-punctuation: `<ACRONYM>` puts a period after every
            // character; any other mark goes after the last character only.
            if post_label == ACRONYM_LABEL {
                current.push('.');
            } else if ci == last && post_label != NULL_LABEL {
                current.push_str(post_label);
            }
            // A sentence boundary at the last character flushes the sentence.
            if apply_sbd && ci == last && pred.sbd {
                sentences.push(std::mem::take(&mut current));
            }
        }
    }

    if !current.is_empty() {
        sentences.push(current);
    }
    sentences
}
