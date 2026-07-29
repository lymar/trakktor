//! The four lookup tables the model ships with, and the plain-text files they
//! are kept in.
//!
//! The reference carries them inside its pickle; the conversion writes them out
//! as line-based text (one entry per line, index = line number where an index
//! is meant), which is what every later run reads. All four are verified to
//! hold no whitespace, so the format is unambiguous.

use std::path::Path;

use rustc_hash::FxHashMap;

use super::{error::StressError, text::STRESS};

/// `n-gram → row of the embedding table`; index is the line number.
pub(super) const NGRAMS_FILE: &str = "ngrams.txt";
/// `word stress-position yo-position` per line.
pub(super) const EXCEPTIONS_FILE: &str = "exceptions.txt";
/// `word first-variant second-variant` per line, variants already ordered.
pub(super) const HOMOGRAPHS_FILE: &str = "homographs.txt";
/// The word-piece vocabulary; index is the line number.
pub(super) const VOCAB_FILE: &str = "vocab.txt";

/// The n-gram the reference falls back to when a word matches nothing.
const UNK_NGRAM: &str = "UNK";

/// One exception entry: where the stress goes and which letter is really `ё`.
///
/// Both are **character** positions in the word, and `yo` is `None` when the
/// word has no `ё`.
#[derive(Debug, Clone, Copy)]
pub(super) struct Exception {
    pub stress: usize,
    pub yo: Option<usize>,
}

/// Everything the pipeline looks words up in.
pub(super) struct Tables {
    /// Character n-grams to embedding rows.
    pub ngrams: FxHashMap<Box<str>, u32>,
    /// The row used when a word matches no n-gram at all.
    pub unk: u32,
    /// Words the reference marks from a table rather than from the network.
    pub exceptions: FxHashMap<Box<str>, Exception>,
    /// Homographs and their two spellings, in the reference's order.
    pub homographs: FxHashMap<Box<str>, [Box<str>; 2]>,
}

/// Wraps a table-level failure.
fn bad(file: &str, detail: impl std::fmt::Display) -> StressError {
    StressError::InvalidModel(format!("{file}: {detail}"))
}

fn read(dir: &Path, file: &str) -> Result<String, StressError> {
    std::fs::read_to_string(dir.join(file))
        .map_err(|e| bad(file, format!("cannot read ({e})")))
}

impl Tables {
    /// Loads the four tables from a converted model directory.
    pub fn load(dir: &Path) -> Result<Self, StressError> {
        let ngrams_text = read(dir, NGRAMS_FILE)?;
        let mut ngrams = FxHashMap::default();
        for (index, line) in ngrams_text.lines().enumerate() {
            ngrams
                .insert(line.into(), u32::try_from(index).unwrap_or(u32::MAX));
        }
        let unk = *ngrams
            .get(UNK_NGRAM)
            .ok_or_else(|| bad(NGRAMS_FILE, "no `UNK` row"))?;

        let exceptions_text = read(dir, EXCEPTIONS_FILE)?;
        let mut exceptions = FxHashMap::default();
        for line in exceptions_text.lines() {
            let mut parts = line.split(' ');
            let (Some(word), Some(stress), Some(yo), None) =
                (parts.next(), parts.next(), parts.next(), parts.next())
            else {
                return Err(bad(EXCEPTIONS_FILE, format!("bad line `{line}`")));
            };
            let stress: usize = stress.parse().map_err(|_| {
                bad(EXCEPTIONS_FILE, format!("bad line `{line}`"))
            })?;
            let yo: i64 = yo.parse().map_err(|_| {
                bad(EXCEPTIONS_FILE, format!("bad line `{line}`"))
            })?;
            let yo = usize::try_from(yo).ok();
            // Both are character positions into the word; the marking rules
            // index by them, so an out-of-range one is a corrupt table, not a
            // word to mark slightly wrong.
            let length = word.chars().count();
            if stress >= length || yo.is_some_and(|yo| yo >= length) {
                return Err(bad(
                    EXCEPTIONS_FILE,
                    format!("`{line}`: position outside the word"),
                ));
            }
            exceptions.insert(word.into(), Exception { stress, yo });
        }

        let homographs_text = read(dir, HOMOGRAPHS_FILE)?;
        let mut homographs = FxHashMap::default();
        for line in homographs_text.lines() {
            let mut parts = line.split(' ');
            let (Some(word), Some(first), Some(second), None) =
                (parts.next(), parts.next(), parts.next(), parts.next())
            else {
                return Err(bad(HOMOGRAPHS_FILE, format!("bad line `{line}`")));
            };
            // A variant is the word plus exactly one mark. Writing a variant
            // back walks it against the word letter by letter, so a variant of
            // any other shape would silently misplace letters — checked here,
            // once, instead.
            let length = word.chars().count();
            for variant in [first, second] {
                let marks = variant.chars().filter(|c| *c == STRESS).count();
                if marks != 1 || variant.chars().count() != length + 1 {
                    return Err(bad(
                        HOMOGRAPHS_FILE,
                        format!(
                            "`{line}`: variant `{variant}` does not fit \
                             `{word}`"
                        ),
                    ));
                }
            }
            homographs.insert(word.into(), [first.into(), second.into()]);
        }

        Ok(Self {
            ngrams,
            unk,
            exceptions,
            homographs,
        })
    }
}

#[cfg(test)]
mod tests;

#[cfg(test)]
impl Tables {
    /// A hand-built table for the tests that exercise the rules rather than
    /// the model.
    pub(super) fn fixture(
        ngrams: &[&str],
        exceptions: &[(&str, usize, Option<usize>)],
        homographs: &[(&str, &str, &str)],
    ) -> Self {
        let mut rows = FxHashMap::default();
        for (index, gram) in ngrams.iter().enumerate() {
            rows.insert((*gram).into(), index as u32);
        }
        let unk = rows.len() as u32;
        rows.insert(UNK_NGRAM.into(), unk);
        Self {
            ngrams: rows,
            unk,
            exceptions: exceptions
                .iter()
                .map(|(word, stress, yo)| {
                    (
                        (*word).into(),
                        Exception {
                            stress: *stress,
                            yo: *yo,
                        },
                    )
                })
                .collect(),
            homographs: homographs
                .iter()
                .map(|(word, first, second)| {
                    ((*word).into(), [(*first).into(), (*second).into()])
                })
                .collect(),
        }
    }
}

/// Renders the tables as the text files [`Tables::load`] reads. Used once, by
/// the conversion.
pub(super) fn render_ngrams(
    entries: &[(String, i64)],
) -> Result<String, StressError> {
    render_indexed(entries, "n-gram")
}

/// Renders the word-piece vocabulary, which is stored the same way.
pub(super) fn render_vocab(
    entries: &[(String, i64)],
) -> Result<String, StressError> {
    render_indexed(entries, "vocabulary entry")
}

/// Lays an `entry → index` map out as lines, line number = index — after
/// checking the indices really are the dense `0..N`.
fn render_indexed(
    entries: &[(String, i64)],
    what: &str,
) -> Result<String, StressError> {
    let mut rows = vec![None; entries.len()];
    for (entry, index) in entries {
        let index = usize::try_from(*index)
            .ok()
            .filter(|index| *index < rows.len())
            .ok_or_else(|| {
                StressError::ModelDownload(format!(
                    "{what} `{entry}` has row {index}, outside 0..{}",
                    rows.len()
                ))
            })?;
        rows[index] = Some(entry.as_str());
    }
    let mut out = String::new();
    for (index, row) in rows.iter().enumerate() {
        let entry = row.ok_or_else(|| {
            StressError::ModelDownload(format!("{what} row {index} is missing"))
        })?;
        out.push_str(entry);
        out.push('\n');
    }
    Ok(out)
}
