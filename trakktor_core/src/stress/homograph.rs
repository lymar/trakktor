//! The homograph stage: find the words whose spelling alone does not say how
//! they are read, ask the encoder which of the two readings this sentence
//! wants, and write that reading back.
//!
//! It runs **before** the accentor and rewrites whole words, so everything
//! downstream sees an already-marked word and leaves it alone. Words the caller
//! marked themselves are not candidates at all: a `+` inside a run makes it
//! stop matching the table.

use super::{
    error::StressError,
    model::HomographContext,
    tables::Tables,
    text::{STRESS, is_russian, lower, upper},
    tokenizer::WordPieces,
};

/// How much of the sentence the encoder is shown on each side of the word.
///
/// A sentence is normally far shorter than this, and then the whole of it is
/// used, exactly as the reference does. The cap only bites on pathological
/// input — a text with no sentence punctuation at all — where an unbounded
/// context would run past the encoder's 2 048 positions.
const CONTEXT_CHARS: usize = 1000;

/// One homograph found in a segment.
#[derive(Debug, Clone)]
pub(super) struct Occurrence {
    /// Byte range of the word inside its segment.
    pub start: usize,
    pub end: usize,
    /// The word as written.
    pub word: String,
}

/// Whether `c` can be part of a word for this stage — Russian letters plus the
/// stress mark, so an already-marked word is seen whole and rejected as a
/// candidate.
fn is_word(c: char) -> bool { is_russian(c) || c == STRESS }

/// Finds the homographs of one segment, left to right.
#[must_use]
pub(super) fn occurrences(segment: &str, tables: &Tables) -> Vec<Occurrence> {
    let mut out = Vec::new();
    let mut run: Option<usize> = None;
    let close = |run: Option<usize>, end: usize, out: &mut Vec<Occurrence>| {
        let Some(start) = run else { return };
        let word = &segment[start..end];
        if !word.chars().any(is_russian) {
            return;
        }
        let key = word.to_lowercase();
        if tables.homographs.contains_key(key.as_str()) {
            out.push(Occurrence {
                start,
                end,
                word: word.to_owned(),
            });
        }
    };
    for (at, c) in segment.char_indices() {
        if is_word(c) {
            run.get_or_insert(at);
        } else {
            close(run.take(), at, &mut out);
        }
    }
    close(run.take(), segment.len(), &mut out);
    out
}

/// Builds the encoder's input for one occurrence: the segment with the word
/// wrapped in markers.
///
/// The reference marks up the sentence as text and relies on `never_split` to
/// keep the markers whole; the markers are always surrounded by spaces, so
/// encoding the three pieces separately and splicing the marker ids between
/// them gives the same tokens.
pub(super) fn context(
    segment: &str,
    occurrence: &Occurrence,
    tokenizer: &WordPieces,
) -> Result<HomographContext, StressError> {
    let prefix = trim_left(&segment[..occurrence.start]);
    let suffix = trim_right(&segment[occurrence.end..]);

    let mut ids = vec![tokenizer.cls];
    ids.extend(tokenizer.encode(prefix)?);
    let start = ids.len();
    ids.push(tokenizer.homo_start);
    ids.extend(tokenizer.encode(&occurrence.word)?);
    let end = ids.len();
    ids.push(tokenizer.homo_end);
    ids.extend(tokenizer.encode(suffix)?);
    ids.push(tokenizer.sep);
    Ok(HomographContext { ids, start, end })
}

/// Keeps at most [`CONTEXT_CHARS`] characters of the left context, cut at a
/// word boundary.
fn trim_left(text: &str) -> &str {
    let mut chars = text.char_indices().rev();
    let Some((cut, _)) = chars.nth(CONTEXT_CHARS) else {
        return text;
    };
    let tail = &text[cut..];
    match tail.find(char::is_whitespace) {
        Some(space) => &tail[space..],
        None => tail,
    }
}

/// The same on the right.
fn trim_right(text: &str) -> &str {
    let Some((cut, _)) = text.char_indices().nth(CONTEXT_CHARS) else {
        return text;
    };
    let head = &text[..cut];
    match head.rfind(char::is_whitespace) {
        Some(space) => &head[..space],
        None => head,
    }
}

/// Writes the chosen readings back into a segment.
///
/// Returns the new text and how many of the readings restored a `ё` — this
/// stage is the only one that can, for the pairs a stress mark cannot tell
/// apart. The variants differ from the word only by the stress mark and by
/// `е` ↔ `ё`, and both letters are the same width in UTF-8, so nothing else
/// about the segment moves.
#[must_use]
pub(super) fn apply(
    segment: &str,
    occurrences: &[Occurrence],
    picks: &[usize],
    tables: &Tables,
    put_yo: bool,
) -> (String, usize) {
    let mut out = String::with_capacity(segment.len() + occurrences.len());
    let mut restored_yo = 0;
    let mut at = 0;
    for (occurrence, pick) in occurrences.iter().zip(picks) {
        let key = occurrence.word.to_lowercase();
        let Some(variants) = tables.homographs.get(key.as_str()) else {
            continue;
        };
        let variant = &variants[(*pick).min(1)];
        let reading = reading(&occurrence.word, variant, put_yo);
        if reading.contains('ё') != occurrence.word.contains('ё') ||
            reading.contains('Ё') != occurrence.word.contains('Ё')
        {
            restored_yo += 1;
        }
        out.push_str(&segment[at..occurrence.start]);
        out.push_str(&reading);
        at = occurrence.end;
    }
    out.push_str(&segment[at..]);
    (out, restored_yo)
}

/// Renders one variant in the case of the word it replaces.
fn reading(word: &str, variant: &str, put_yo: bool) -> String {
    let variant: Vec<char> = variant
        .chars()
        .map(|c| if put_yo || c != 'ё' { c } else { 'е' })
        .collect();
    let Some(mark) = variant.iter().position(|c| *c == STRESS) else {
        return word.to_owned();
    };
    let letters: Vec<char> =
        variant.into_iter().filter(|c| *c != STRESS).collect();

    let mut cased: Vec<char> = word
        .chars()
        .zip(&letters)
        .map(|(original, letter)| {
            if original.is_lowercase() {
                lower(*letter)
            } else {
                upper(*letter)
            }
        })
        .collect();
    if mark <= cased.len() {
        cased.insert(mark, STRESS);
    }
    cased.into_iter().collect()
}

#[cfg(test)]
mod tests;
