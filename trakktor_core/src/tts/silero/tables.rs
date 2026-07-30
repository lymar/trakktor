//! The lookup tables a model carries beside its weights.
//!
//! All four published models share one architecture and differ exactly here:
//! in their alphabet, their speakers and — for the models built for the
//! languages of the region — their transliteration tables. So the tables are
//! read from the artifact rather than restated in the code, and the conversion
//! step writes them out as one small JSON file.

use std::{collections::HashMap, path::Path};

use serde_json::{Value, json};

use super::error::SileroError;

/// The tables file inside a converted model directory.
pub const TABLES_FILE: &str = "tables.json";

/// Everything the text frontend needs to turn a string into symbol ids.
#[derive(Debug, Clone, Default)]
pub struct Tables {
    /// The model's symbols in id order; index is the id.
    pub symbols: Vec<String>,
    /// The reference's own `symbols` string — **not** the same thing as
    /// [`Self::symbols`]. It is the artifact's own datum, and the frontend
    /// keeps only characters from its fourth onward (see
    /// [`Tables::keeps`]).
    pub alphabet: String,
    /// The letters of the model's language(s), against which a text is judged
    /// to need transliteration.
    pub letters: String,
    /// Start-of-sequence symbol.
    pub sos: String,
    /// End-of-sequence symbol.
    pub eos: String,
    /// Speaker names in id order.
    pub speakers: Vec<String>,
    /// Per-language transliteration into the model's alphabet. A replacement
    /// may be several characters long.
    pub translit: HashMap<String, HashMap<String, String>>,
}

impl Tables {
    /// Reads the tables of a converted model.
    ///
    /// # Errors
    ///
    /// Returns [`SileroError::Checkpoint`] when the file is missing or
    /// malformed.
    pub fn load(dir: &Path) -> Result<Self, SileroError> {
        let path = dir.join(TABLES_FILE);
        let text = std::fs::read_to_string(&path).map_err(|e| {
            SileroError::Checkpoint(format!("reading {}: {e}", path.display()))
        })?;
        let root: Value = serde_json::from_str(&text).map_err(|e| {
            SileroError::Checkpoint(format!("parsing {}: {e}", path.display()))
        })?;
        let broken = |what: &str| {
            SileroError::Checkpoint(format!("{}: no {what}", path.display()))
        };
        let strings = |key: &str| -> Option<Vec<String>> {
            root.get(key)?
                .as_array()?
                .iter()
                .map(|value| value.as_str().map(str::to_owned))
                .collect()
        };
        let text_of = |key: &str| -> Option<String> {
            root.get(key)?.as_str().map(str::to_owned)
        };
        let tables = Self {
            symbols: strings("symbols").ok_or_else(|| broken("symbols"))?,
            alphabet: text_of("alphabet").ok_or_else(|| broken("alphabet"))?,
            letters: text_of("letters").unwrap_or_default(),
            sos: text_of("sos").ok_or_else(|| broken("start symbol"))?,
            eos: text_of("eos").ok_or_else(|| broken("end symbol"))?,
            speakers: strings("speakers").ok_or_else(|| broken("speakers"))?,
            translit: read_translit(root.get("translit")),
        };
        if tables.symbols.is_empty() || tables.speakers.is_empty() {
            return Err(broken("symbols or speakers"));
        }
        Ok(tables)
    }

    /// Writes the tables into a converted model directory.
    ///
    /// # Errors
    ///
    /// Returns [`SileroError::ModelDownload`] when the file cannot be written.
    pub fn store(&self, dir: &Path) -> Result<(), SileroError> {
        let translit: serde_json::Map<String, Value> = self
            .translit
            .iter()
            .map(|(language, table)| {
                let entries: serde_json::Map<String, Value> = table
                    .iter()
                    .map(|(from, to)| (from.clone(), json!(to)))
                    .collect();
                (language.clone(), Value::Object(entries))
            })
            .collect();
        let root = json!({
            "symbols": self.symbols,
            "alphabet": self.alphabet,
            "letters": self.letters,
            "sos": self.sos,
            "eos": self.eos,
            "speakers": self.speakers,
            "translit": Value::Object(translit),
        });
        let text = serde_json::to_string_pretty(&root).map_err(|e| {
            SileroError::ModelDownload(format!("rendering the tables: {e}"))
        })?;
        let temp = dir.join(format!("{TABLES_FILE}.partial"));
        std::fs::write(&temp, text).map_err(|e| {
            SileroError::ModelDownload(format!("writing the tables: {e}"))
        })?;
        std::fs::rename(&temp, dir.join(TABLES_FILE)).map_err(|e| {
            SileroError::ModelDownload(format!("finalizing the tables: {e}"))
        })
    }

    /// The alphabet as a lookup: symbol to id.
    #[must_use]
    pub fn index(&self) -> HashMap<char, u32> {
        self.symbols
            .iter()
            .enumerate()
            .filter_map(|(id, symbol)| {
                let mut characters = symbol.chars();
                let first = characters.next()?;
                characters.next().is_none().then_some((first, id as u32))
            })
            .collect()
    }

    /// Whether the frontend keeps `symbol` in the text at all.
    ///
    /// The reference builds this set as `symbols[3:]` — its own string minus
    /// its first three entries — which is why the punctuation a model reads
    /// differs between models even though their tables overlap: the base
    /// models lose `!` to that slice, and the Russian one keeps it.
    #[must_use]
    pub fn keeps(&self, symbol: char) -> bool {
        self.alphabet.chars().skip(3).any(|kept| kept == symbol)
    }

    /// Whether `symbol` is one of the model's letters.
    #[must_use]
    pub fn is_letter(&self, symbol: char) -> bool {
        self.letters.chars().any(|letter| letter == symbol)
    }

    /// The id of a named speaker.
    #[must_use]
    pub fn speaker(&self, name: &str) -> Option<usize> {
        if name.is_empty() {
            return None;
        }
        self.speakers.iter().position(|known| known == name)
    }

    /// The speakers a model actually names, in id order. A model may hold a
    /// row it names nothing for; that is not a voice anyone can ask for.
    #[must_use]
    pub fn voices(&self) -> Vec<&str> {
        self.speakers
            .iter()
            .filter(|name| !name.is_empty())
            .map(String::as_str)
            .collect()
    }

    /// The speakers, comma-separated, for an error message.
    #[must_use]
    pub fn speaker_list(&self) -> String { self.voices().join(", ") }
}

/// Reads the per-language transliteration tables, if there are any.
fn read_translit(
    value: Option<&Value>,
) -> HashMap<String, HashMap<String, String>> {
    let Some(Value::Object(languages)) = value else {
        return HashMap::new();
    };
    languages
        .iter()
        .filter_map(|(language, table)| {
            let Value::Object(entries) = table else {
                return None;
            };
            let table = entries
                .iter()
                .filter_map(|(from, to)| {
                    Some((from.clone(), to.as_str()?.to_owned()))
                })
                .collect();
            Some((language.clone(), table))
        })
        .collect()
}

#[cfg(test)]
mod tests;
