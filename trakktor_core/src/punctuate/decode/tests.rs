//! Unit tests for text reconstruction, on fixed pieces and predictions.

use super::*;

/// A prediction with `pre`/`post` label ids, a boundary flag, and the character
/// positions (within the piece) to upper-case.
fn pred(pre: u8, post: u8, sbd: bool, upper: &[usize]) -> TokenPred {
    let mut cap = [false; MAX_SUBWORD_LEN];
    for &i in upper {
        cap[i] = true;
    }
    TokenPred {
        pre,
        post,
        sbd,
        cap,
    }
}

/// Reconstructs from `(piece, prediction)` pairs, numbering the pieces as ids.
fn run(items: &[(&'static str, TokenPred)], apply_sbd: bool) -> Vec<String> {
    let pieces: Vec<&str> = items.iter().map(|&(p, _)| p).collect();
    let merged: Vec<(u32, TokenPred)> = items
        .iter()
        .enumerate()
        .map(|(i, &(_, p))| (i as u32, p))
        .collect();
    reconstruct_with(&merged, |id| pieces[id as usize], apply_sbd)
}

// Post label ids: 0 = <NULL>, 1 = <ACRONYM>, 2 = ".", 3 = ",". Pre: 1 = ¿.

#[test]
fn capitalizes_and_spaces_and_splits() {
    // "▁hello" with the first letter upper, "▁world" plain + a full stop.
    let out = run(
        &[
            ("▁hello", pred(0, 0, false, &[1])),
            ("▁world", pred(0, 2, true, &[])),
        ],
        true,
    );
    assert_eq!(out, vec!["Hello world.".to_string()]);
}

#[test]
fn acronym_periods_after_every_char() {
    // "▁us" as an acronym, both letters upper -> "U.S."
    let out = run(&[("▁us", pred(0, 1, false, &[1, 2]))], true);
    assert_eq!(out, vec!["U.S.".to_string()]);
}

#[test]
fn per_character_casing() {
    // "▁mcdavid": upper 'M' (pos 1) and 'D' (pos 3) -> "McDavid".
    let out = run(&[("▁mcdavid", pred(0, 0, false, &[1, 3]))], true);
    assert_eq!(out, vec!["McDavid".to_string()]);
}

#[test]
fn pre_punctuation_before_first_char() {
    // Spanish inverted question mark before the word.
    let out = run(
        &[
            ("▁como", pred(1, 0, false, &[])),
            ("▁estas", pred(0, 4, true, &[])),
        ],
        true,
    );
    assert_eq!(out, vec!["¿como estas?".to_string()]);
}

#[test]
fn continuation_pieces_join_without_space() {
    // A word split into two pieces: only the word-initial piece emits a space.
    let out = run(
        &[
            ("▁sun", pred(0, 0, false, &[1])),
            ("shine", pred(0, 2, true, &[])),
        ],
        true,
    );
    assert_eq!(out, vec!["Sunshine.".to_string()]);
}

#[test]
fn without_sbd_returns_one_string() {
    let out = run(
        &[
            ("▁a", pred(0, 2, true, &[1])),
            ("▁b", pred(0, 2, true, &[1])),
        ],
        false,
    );
    assert_eq!(out, vec!["A. B.".to_string()]);
}
