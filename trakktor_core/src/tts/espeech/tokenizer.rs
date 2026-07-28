//! Turning text into the model's tokens, which are plain characters.
//!
//! There is no BPE and no grapheme-to-phoneme step: the checkpoint ships a
//! `vocab.txt` of single characters, one per line, and a character's line
//! number *is* its token. Anything not in the table maps to line 0 (the space).
//!
//! Stress is part of the text, not of the tokenizer: the model was trained on
//! Russian marked with `+` before the stressed vowel (`прив+ет`), and `+` is an
//! ordinary entry of the table. Text is therefore passed through as written —
//! whoever wrote it decides where the stresses go.

use std::{collections::HashMap, path::Path};

use super::error::EspeechError;

/// The character table of one checkpoint.
#[derive(Debug, Clone)]
pub struct CharTokenizer {
    map: HashMap<char, u32>,
    size: usize,
}

impl CharTokenizer {
    /// Reads `vocab.txt`: one character per line, the line number is the id.
    ///
    /// # Errors
    ///
    /// Returns [`EspeechError::Checkpoint`] when the file cannot be read or
    /// holds no entries.
    pub fn load(path: &Path) -> Result<Self, EspeechError> {
        let text = std::fs::read_to_string(path).map_err(|e| {
            EspeechError::Checkpoint(format!("reading {}: {e}", path.display()))
        })?;
        // The reference strips exactly the trailing newline of each line, so a
        // line holding several characters (there are none in the published
        // table) would enter the map whole and simply never match.
        let mut map = HashMap::new();
        let mut size = 0usize;
        for (index, line) in text.lines().enumerate() {
            size = index + 1;
            let mut chars = line.chars();
            if let (Some(character), None) = (chars.next(), chars.next()) {
                map.insert(character, index as u32);
            }
        }
        if size == 0 {
            return Err(EspeechError::Checkpoint(format!(
                "{} holds no characters",
                path.display()
            )));
        }
        Ok(Self { map, size })
    }

    /// Entries in the table — the vocabulary the embedding was sized for.
    #[must_use]
    pub fn size(&self) -> usize { self.size }

    /// Encodes `text` into token ids, applying the reference's own
    /// normalization first.
    #[must_use]
    pub fn encode(&self, text: &str) -> Vec<u32> {
        normalize(text)
            .chars()
            .map(|character| {
                self.map.get(&character).copied().unwrap_or_default()
            })
            .collect()
    }
}

/// Rewrites `text` the way the reference does before tokenizing.
///
/// Two things happen there. A handful of characters are substituted for ones
/// the table actually holds (the reference's `custom_trans`). And the Chinese
/// segmenter it runs — which for text without CJK does nothing but hand back
/// the characters — inserts a space before a run of Latin letters or digits
/// when the character before it is neither a space nor a quote. That second
/// rule is the only visible effect of the segmenter on Russian text, so it is
/// reproduced rather than the segmenter.
#[must_use]
pub fn normalize(text: &str) -> String {
    let substituted: String = text
        .chars()
        .map(|character| match character {
            ';' => ',',
            '\u{201c}' | '\u{201d}' => '"',
            '\u{2018}' | '\u{2019}' => '\'',
            other => other,
        })
        .collect();

    let chars: Vec<char> = substituted.chars().collect();
    let mut out = String::with_capacity(substituted.len());
    let mut index = 0;
    while index < chars.len() {
        let run = ascii_word(&chars[index..]);
        if run > 1 &&
            !out.is_empty() &&
            !matches!(out.chars().next_back(), Some(' ' | ':' | '\'' | '"'))
        {
            out.push(' ');
        }
        let take = run.max(1);
        out.extend(&chars[index..index + take]);
        index += take;
    }
    out
}

/// Length of the run of Latin letters and digits `chars` opens with, counted
/// the way the segmenter groups them: alphanumerics, then optionally a decimal
/// tail and a percent sign, so `1.5%` is one run rather than three.
fn ascii_word(chars: &[char]) -> usize {
    let mut length = 0;
    while length < chars.len() && chars[length].is_ascii_alphanumeric() {
        length += 1;
    }
    if length == 0 {
        return 0;
    }
    if chars.get(length) == Some(&'.') &&
        chars.get(length + 1).is_some_and(char::is_ascii_digit)
    {
        length += 1;
        while length < chars.len() && chars[length].is_ascii_digit() {
            length += 1;
        }
    }
    if chars.get(length) == Some(&'%') {
        length += 1;
    }
    length
}

/// Closes the reference transcript the way the model expects to see it: ending
/// in a sentence-final period and a space.
///
/// The reference does this in two places, and both apply: first it appends
/// `". "` (or just the space, if the text already ends in a period), then it
/// appends another space because the last character is single-byte. The result
/// ends in two spaces, and that is what the model was fed.
#[must_use]
pub fn close_reference_text(text: &str) -> String {
    let mut closed = text.to_owned();
    if !closed.ends_with(". ") && !closed.ends_with('\u{3002}') {
        if closed.ends_with('.') {
            closed.push(' ');
        } else {
            closed.push_str(". ");
        }
    }
    if closed
        .chars()
        .next_back()
        .is_some_and(|last| last.len_utf8() == 1)
    {
        closed.push(' ');
    }
    closed
}

#[cfg(test)]
mod tests;
