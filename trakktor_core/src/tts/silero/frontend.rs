//! Text in, symbol ids out.
//!
//! The reference's frontend is small — one file, eighty-five symbols, no
//! phonemizer — and it is ported **word for word**, including the parts that
//! look like oversights. Two reasons: the model was trained on exactly this
//! input, and anything else here breaks the golden trace at its very first
//! stage.
//!
//! Three of those parts are worth knowing about, because they are audible:
//!
//! - **there is no Latin in the alphabet** (bar a lone `h` in the base models),
//!   so an English word is not transliterated, it disappears;
//! - **an em dash disappears too**, in the base models: the frontend rewrites
//!   it as an en dash, which their alphabet does not contain. The Russian model
//!   keeps the en dash and so keeps the pause;
//! - **`!` disappears** in the base models, for a different reason — the
//!   frontend keeps `symbols[3:]`, and in their table `!` falls inside those
//!   first three.
//!
//! Stress is not a special case here: `+` is symbol id 5, an ordinary input.

use super::{
    error::SileroError,
    intonation::{self, Utterance},
    tables::Tables,
};

/// One utterance, ready for the network.
#[derive(Debug, Clone)]
pub struct Utterances {
    /// Symbol ids, opened with the start symbol and closed with the end one.
    pub ids: Vec<u32>,
    /// Speech-rate multiplier per symbol.
    pub rate: Vec<f32>,
    /// Pitch multiplier per symbol.
    pub pitch: Vec<f32>,
    /// Utterance-type id per symbol, for a model with an intonation head.
    pub types: Option<Vec<u32>>,
    /// Characters the cleaning threw away — a count worth reporting, because
    /// what this frontend cannot spell it removes without a word.
    pub dropped: usize,
}

/// Turns `text` into the network's input.
///
/// `language` selects a transliteration table and is normally the prefix of the
/// speaker's name (`kat_vika` speaks Georgian); a language with no table, or
/// none at all, leaves the text alone.
///
/// # Errors
///
/// Returns [`SileroError::TextEmpty`] when nothing readable is left after the
/// cleaning — a line of Latin, say, of which this frontend keeps nothing.
pub fn prepare(
    text: &str,
    tables: &Tables,
    language: Option<&str>,
    rate: f32,
    pitch: f32,
    with_types: bool,
) -> Result<Utterances, SileroError> {
    let raw = text.trim();
    let joined = join_lines(raw);
    let Cleaned { spoken, dropped } = clean(&joined, tables, language);
    if !has_speech(&spoken) {
        return Err(SileroError::TextEmpty);
    }

    let index = tables.index();
    let mut ids = Vec::with_capacity(spoken.chars().count() + 2);
    let sos = tables.sos.chars().next().unwrap_or('|');
    let eos = tables.eos.chars().next().unwrap_or('~');
    for symbol in std::iter::once(sos)
        .chain(spoken.chars())
        .chain(std::iter::once(eos))
    {
        let id = index.get(&symbol).copied().ok_or_else(|| {
            SileroError::Checkpoint(format!(
                "the model's alphabet has no `{symbol}`"
            ))
        })?;
        ids.push(id);
    }

    let types = with_types.then(|| {
        intonation::type_ids(raw, &intonation::classify_text(raw), ids.len())
    });
    Ok(Utterances {
        rate: vec![rate; ids.len()],
        pitch: vec![pitch; ids.len()],
        types,
        ids,
        dropped,
    })
}

/// Replaces the line breaks: a break right after a clause mark only joins the
/// lines, any other break ends a sentence.
fn join_lines(text: &str) -> String {
    let mut out = String::with_capacity(text.len() + 8);
    let mut previous: Option<char> = None;
    for character in text.chars() {
        if character == '\n' {
            match previous {
                Some(',' | ';' | ':' | '.' | '!' | '?') => out.push(' '),
                _ => out.push_str(". "),
            }
        } else {
            out.push(character);
        }
        previous = Some(character);
    }
    out
}

/// What the cleaning produced, and how much of the input it could not spell.
struct Cleaned {
    spoken: String,
    /// Characters the alphabet rejected. Whitespace the frontend collapses is
    /// **not** counted: that is normalization, not loss.
    dropped: usize,
}

/// The frontend's cleaning: case, dashes, transliteration, then everything the
/// alphabet does not contain removed and the spacing normalized.
fn clean(text: &str, tables: &Tables, language: Option<&str>) -> Cleaned {
    let lowered: String = text
        .to_lowercase()
        .chars()
        .map(|character| match character {
            // An em dash becomes an en dash — which only one published
            // alphabet contains, so in the others this is where it is lost.
            '\u{2014}' => '\u{2013}',
            // A non-breaking hyphen becomes an ordinary one.
            '\u{2011}' => '-',
            other => other,
        })
        .collect();
    let converted = transliterate(&lowered, tables, language);

    let mut out = String::with_capacity(converted.len());
    let mut dropped = 0;
    let mut space = false;
    for character in converted.chars() {
        if !tables.keeps(character) {
            dropped += usize::from(!character.is_whitespace());
            continue;
        }
        if character == ' ' {
            space = true;
            continue;
        }
        if space && !out.is_empty() {
            out.push(' ');
        }
        space = false;
        out.push(character);
    }
    Cleaned {
        spoken: out,
        dropped,
    }
}

/// Rewrites a text in the model's own script, when it is written in one the
/// model does not have and a table for it exists.
fn transliterate(
    text: &str,
    tables: &Tables,
    language: Option<&str>,
) -> String {
    let Some(table) = language.and_then(|code| tables.translit.get(code))
    else {
        return text.to_owned();
    };
    let mut foreign = false;
    let mut any = false;
    for character in text.chars().filter(|c| c.is_alphabetic()) {
        any = true;
        if !tables.is_letter(character) {
            foreign = true;
        }
    }
    if !any || !foreign {
        return text.to_owned();
    }
    let mut out = String::with_capacity(text.len());
    let mut buffer = [0u8; 4];
    for character in text.chars() {
        match table.get(character.encode_utf8(&mut buffer) as &str) {
            Some(replacement) => out.push_str(replacement),
            None => out.push(character),
        }
    }
    out
}

/// Whether anything readable is left: the reference asks the same question of
/// the plain Russian letters, and refuses a text that has none.
fn has_speech(text: &str) -> bool {
    text.chars()
        .any(|character| matches!(character, 'а'..='я') || character == '-')
}

/// The utterance types of a text, for the output contract.
#[must_use]
pub fn utterance_types(text: &str) -> Vec<Utterance> {
    intonation::classify_text(text.trim())
}

#[cfg(test)]
mod tests;
