//! Russian stress marking and `ё` restoration (`text stress`).
//!
//! Takes Russian text and gives back the same text with the stressed vowel
//! marked — `+` before it, the form the speech engines read, or a combining
//! acute after it — and with the letter `ё` written where it belongs. Russian
//! does not write either, and a speech synthesizer needs both: an unmarked word
//! is read by guesswork, and a pair like «все»/«всё» cannot be told apart by a
//! stress mark at all.
//!
//! Three layers, in this order:
//!
//! 1. **The user's dictionary** ([`Dictionary`]) — the caller's own spellings,
//!    written in before anything else looks at the text. Nothing downstream
//!    overwrites a mark that is already there, so this is all it takes.
//! 2. **The homograph solver** — for words whose spelling does not say how they
//!    are read, an encoder reads the sentence around the word and picks one of
//!    its two readings.
//! 3. **The accentor** — for everything else, a network over the word's
//!    character n-grams names the stressed vowel and any hidden `ё`. When it is
//!    not confident, the word is left unmarked and reported, which is the right
//!    failure here: a miss beats a confident mistake, and the caller can fix a
//!    miss with a dictionary entry.
//!
//! # Credits
//!
//! Ported from [`silero-stress`](https://github.com/snakers4/silero-stress)
//! (MIT) by the Silero Team: the n-gram accentor with its stress and `ё` heads,
//! the `rubert-tiny`-class homograph solver, and the marking rules. The user
//! dictionary is this port's own — upstream has no equivalent.

mod accentor;
mod dictionary;
pub mod download;
mod error;
mod homograph;
pub mod model;
pub mod runtime;
#[cfg(feature = "stress-burn")]
pub mod runtime_burn;
mod tables;
mod text;
mod tokenizer;

use std::path::Path;

pub use dictionary::Dictionary;
pub use download::{
    DEFAULT_MODEL, ResolvedModel, known_model_names, resolve_model,
};
pub use error::StressError;
pub use model::StressModel;
pub use runtime::{Precision, StressRuntime};
#[cfg(feature = "stress-burn")]
pub use runtime_burn::StressBurnRuntime;
pub use text::Marker;

use crate::stress::{tables::Tables, tokenizer::WordPieces};

/// Tunables for one marking run.
#[derive(Debug, Clone, Copy)]
pub struct StressOptions {
    /// Which form the stress mark takes in the output.
    pub marker: Marker,
    /// Whether `ё` is restored. With this off the letters of the input are not
    /// touched at all — only marks are added.
    pub restore_yo: bool,
    /// Words per forward batch of the accentor.
    pub batch_size: usize,
}

impl Default for StressOptions {
    fn default() -> Self {
        Self {
            marker: Marker::Plus,
            restore_yo: true,
            batch_size: 256,
        }
    }
}

/// What one run did, for the machine-readable output.
#[derive(Debug, Default, Clone, Copy)]
pub struct Stats {
    /// Words that can carry a stress at all — those with at least one vowel.
    pub words: usize,
    /// Of those, the ones carrying a stress mark afterwards.
    pub stressed: usize,
    /// Letters `е` rewritten as `ё`.
    pub yo_restored: usize,
    /// Homographs resolved by the encoder.
    pub homographs: usize,
    /// Words marked from the user's dictionary.
    pub from_dictionary: usize,
}

/// The marked text and what it took.
#[derive(Debug)]
pub struct Stressed {
    /// The input text with the marks in place.
    pub text: String,
    /// The form those marks take.
    pub marker: Marker,
    /// Counts for the output.
    pub stats: Stats,
    /// The words left unmarked, deduplicated and in the order they first
    /// appear. These are exactly the words a dictionary entry would fix.
    pub unstressed: Vec<String>,
}

/// A loaded marker: the networks on either runtime, plus the model's tables and
/// its word-piece vocabulary.
pub struct Stressor {
    runtime: Box<dyn StressModel>,
    tables: Tables,
    tokenizer: WordPieces,
}

impl Stressor {
    /// Loads the model's tables and tokenizer from a converted model directory
    /// and pairs them with an already-loaded runtime.
    ///
    /// # Errors
    ///
    /// Returns [`StressError`] when a table is missing or malformed.
    pub fn load(
        model_dir: &Path,
        runtime: Box<dyn StressModel>,
    ) -> Result<Self, StressError> {
        Ok(Self {
            tables: Tables::load(model_dir)?,
            tokenizer: WordPieces::load(model_dir)?,
            runtime,
        })
    }

    /// Marks `text`, applying `dictionary` first.
    ///
    /// # Errors
    ///
    /// Returns [`StressError`] on a tokenizer or backend failure.
    pub fn mark(
        &self,
        text: &str,
        dictionary: &Dictionary,
        options: &StressOptions,
    ) -> Result<Stressed, StressError> {
        // An acute in the input is the same statement as a plus; normalizing
        // it first means the operation accepts its own `--marker acute` output.
        let text = text::acute_to_plus(text);
        let (text, from_dictionary) = dictionary.apply(&text);
        let (text, homographs, homograph_yo) =
            self.resolve_homographs(&text, options)?;
        let (text, mut stats, unstressed) = self.accentuate(&text, options)?;

        stats.homographs = homographs;
        // A `ё` the solver put there is one the accentor never sees, so the two
        // stages are counted together.
        stats.yo_restored += homograph_yo;
        stats.from_dictionary = from_dictionary;
        Ok(Stressed {
            text: text::render(&text, options.marker),
            marker: options.marker,
            stats,
            unstressed,
        })
    }

    /// Runs the homograph solver over every sentence of the text.
    ///
    /// The batch is one sentence's worth of homographs, exactly as the
    /// reference batches them: it pads without an attention mask, so the batch
    /// is part of the arithmetic and regrouping it would change the answers.
    fn resolve_homographs(
        &self,
        text: &str,
        options: &StressOptions,
    ) -> Result<(String, usize, usize), StressError> {
        let mut out = String::with_capacity(text.len());
        let mut resolved = 0;
        let mut restored_yo = 0;
        for segment in text::segments(text) {
            let occurrences = homograph::occurrences(segment, &self.tables);
            if occurrences.is_empty() {
                out.push_str(segment);
                continue;
            }
            let contexts = occurrences
                .iter()
                .map(|occurrence| {
                    homograph::context(segment, occurrence, &self.tokenizer)
                })
                .collect::<Result<Vec<_>, _>>()?;
            let picks =
                self.runtime.homographs(&contexts, self.tokenizer.pad)?;
            let (marked, yo) = homograph::apply(
                segment,
                &occurrences,
                &picks,
                &self.tables,
                options.restore_yo,
            );
            out.push_str(&marked);
            resolved += occurrences.len();
            restored_yo += yo;
        }
        Ok((out, resolved, restored_yo))
    }

    /// Runs the accentor over the whole text: it has no context to lose, so it
    /// sees every word at once.
    fn accentuate(
        &self,
        text: &str,
        options: &StressOptions,
    ) -> Result<(String, Stats, Vec<String>), StressError> {
        let tokens = text::tokenize(text);
        let wanted: Vec<usize> = tokens
            .iter()
            .enumerate()
            .filter(|(_, token)| token.process)
            .map(|(index, _)| index)
            .collect();

        let bags: Vec<Vec<u32>> = wanted
            .iter()
            .map(|index| accentor::bag(&tokens[*index].clean, &self.tables))
            .collect();
        let batch = options.batch_size.max(1);
        let mut scored = Vec::with_capacity(bags.len());
        for chunk in bags.chunks(batch) {
            scored.extend(self.runtime.accentor(chunk)?);
        }

        let mut scores = vec![None; tokens.len()];
        for (slot, index) in wanted.iter().enumerate() {
            scores[*index] = scored.get(slot).copied();
        }

        let mut out = String::with_capacity(text.len());
        let mut stats = Stats::default();
        let mut unstressed = Vec::new();
        let mut seen = rustc_hash::FxHashSet::default();
        for (index, token) in tokens.iter().enumerate() {
            let marked = accentor::mark(
                token,
                scores[index].as_ref(),
                &self.tables,
                options.restore_yo,
            );
            out.push_str(&marked.text);
            // A word without a vowel has no stress to carry, so counting it
            // would only dilute both numbers — and it can never appear in
            // `unstressed`, where every entry is meant to be actionable.
            if !token.process || token.vowels().is_empty() {
                continue;
            }
            stats.words += 1;
            if marked.outcome.restored_yo {
                stats.yo_restored += 1;
            }
            if marked.outcome.stressed {
                stats.stressed += 1;
            } else if seen.insert(token.clean.clone()) {
                unstressed.push(token.clean.clone());
            }
        }
        Ok((out, stats, unstressed))
    }
}
