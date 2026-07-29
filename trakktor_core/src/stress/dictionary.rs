//! The user dictionary — the part the reference does not have at all.
//!
//! Upstream offers no way to say "this word is stressed like *this*", and there
//! will always be words that need it: names, terms, rare vocabulary, and the
//! three-way homographs the solver cannot express. The mechanism is as small as
//! it can be, because nothing downstream overwrites a mark that is already
//! there: the dictionary simply writes its spelling into the text, before the
//! model ever sees it.
//!
//! One marked word per line; `#` starts a comment. The key is the line itself
//! with the marks removed, lowercased and with `ё` folded into `е`, so a single
//! entry covers the word whether or not its `ё` is written:
//!
//! ```text
//! ф+орзац
//! Корол+ёв
//! ```

use std::path::{Path, PathBuf};

use rustc_hash::FxHashMap;

use super::{
    error::StressError,
    text::{STRESS, Token, fold_yo, is_russian, lower, tokenize, upper},
};

/// The loaded dictionary: lookup key → the marked spelling, as characters.
#[derive(Debug, Default)]
pub struct Dictionary {
    entries: FxHashMap<String, Vec<char>>,
}

impl Dictionary {
    /// Reads and merges dictionaries, in the order given.
    ///
    /// # Errors
    ///
    /// Returns [`StressError::Io`] when a file cannot be read and
    /// [`StressError::Dictionary`] for a malformed or duplicated entry — a
    /// silent overwrite would be the worst way to resolve a conflict between
    /// two of the caller's own files.
    pub fn load(paths: &[PathBuf]) -> Result<Self, StressError> {
        let mut dictionary = Self::default();
        for path in paths {
            let text = std::fs::read_to_string(path).map_err(|e| {
                StressError::Io(format!("{}: {e}", path.display()))
            })?;
            dictionary.extend(&text, path)?;
        }
        Ok(dictionary)
    }

    /// Parses one dictionary's text into the map.
    fn extend(&mut self, text: &str, path: &Path) -> Result<(), StressError> {
        for (number, line) in text.lines().enumerate() {
            let line = line.trim();
            if line.is_empty() || line.starts_with('#') {
                continue;
            }
            let at = || format!("{}:{}", path.display(), number + 1);
            let form: Vec<char> = line.chars().collect();
            if form.iter().any(|c| c.is_whitespace()) {
                return Err(StressError::Dictionary(format!(
                    "{}: `{line}` has more than one word on the line",
                    at()
                )));
            }
            let letters: Vec<char> =
                form.iter().copied().filter(|c| *c != STRESS).collect();
            if !letters.iter().all(|c| is_russian(*c)) {
                return Err(StressError::Dictionary(format!(
                    "{}: `{line}` is not a Russian word with `+` marks",
                    at()
                )));
            }
            if !form.contains(&STRESS) &&
                !letters.iter().any(|c| lower(*c) == 'ё')
            {
                return Err(StressError::Dictionary(format!(
                    "{}: `{line}` marks nothing — write the stress as `+` \
                     before the stressed vowel, or spell the `ё`",
                    at()
                )));
            }
            let key = fold_yo(
                &letters.iter().copied().map(lower).collect::<String>(),
            );
            if self.entries.insert(key.clone(), form).is_some() {
                return Err(StressError::Dictionary(format!(
                    "{}: `{key}` is already in the dictionary",
                    at()
                )));
            }
        }
        Ok(())
    }

    /// Whether the dictionary would change anything.
    #[must_use]
    pub fn is_empty(&self) -> bool { self.entries.is_empty() }

    /// The marked spelling for a word, if the dictionary has one.
    fn spelling(&self, token: &Token) -> Option<&[char]> {
        self.entries.get(&fold_yo(&token.clean)).map(Vec::as_slice)
    }

    /// Writes the dictionary's spellings into `text`, returning the new text
    /// and how many words it changed.
    ///
    /// A word the caller has already marked — with `+` or a written `ё` — is
    /// left alone: the caller's own text outranks a stored preference.
    #[must_use]
    pub(super) fn apply(&self, text: &str) -> (String, usize) {
        if self.entries.is_empty() {
            return (text.to_owned(), 0);
        }
        let mut out = String::with_capacity(text.len());
        let mut applied = 0;
        for token in tokenize(text) {
            let spelling =
                if token.process && !token.has_stress() && !token.has_yo() {
                    self.spelling(&token)
                } else {
                    None
                };
            match spelling {
                Some(spelling) => {
                    out.push_str(&splice(&token, spelling));
                    applied += 1;
                },
                None => out.extend(&token.raw),
            }
        }
        (out, applied)
    }
}

/// Rewrites one word with the dictionary's spelling, keeping everything the
/// dictionary does not speak for: the word's own case and whatever non-letters
/// surround it inside the token (quotes, digits, a trailing hyphen).
///
/// The spelling's letters correspond one-to-one to the word's Russian letters —
/// that is what the key guarantees — so the two are walked together.
fn splice(token: &Token, spelling: &[char]) -> String {
    let mut out = String::with_capacity(token.raw.len() + 4);
    let mut at = 0;
    for c in &token.raw {
        if !is_russian(*c) {
            out.push(*c);
            continue;
        }
        while spelling.get(at) == Some(&STRESS) {
            out.push(STRESS);
            at += 1;
        }
        match spelling.get(at) {
            Some(letter) => {
                out.push(if c.is_lowercase() {
                    lower(*letter)
                } else {
                    upper(*letter)
                });
                at += 1;
            },
            None => out.push(*c),
        }
    }
    // A mark written after the last letter would otherwise be dropped.
    while spelling.get(at) == Some(&STRESS) {
        out.push(STRESS);
        at += 1;
    }
    out
}

#[cfg(test)]
mod tests;
