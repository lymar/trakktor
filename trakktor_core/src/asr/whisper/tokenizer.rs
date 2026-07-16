//! The BPE tokenizer of the Whisper models.
//!
//! Wraps a tiktoken-compatible byte-pair encoding with Whisper's special
//! tokens: transcript markers, per-language tokens, task selectors, and 1501
//! timestamp tokens in 0.02 s steps. Special ids follow the base vocabulary in
//! a fixed order, so the decoding logic can rely on stable arithmetic
//! relations between them (e.g. everything at or above `timestamp_begin` is a
//! timestamp).

mod languages;
#[cfg(test)]
mod tests;

use std::collections::BTreeSet;

use base64::{Engine as _, engine::general_purpose::STANDARD as BASE64};
use rustc_hash::FxHashMap;
use tiktoken_rs::CoreBPE;

use super::{assets, error::WhisperError};

/// A token identifier in the Whisper vocabulary.
pub type TokenId = u32;

/// The task the decoder is asked to perform.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Task {
    /// Transcribe in the source language.
    Transcribe,
    /// Translate into English.
    Translate,
}

/// Number of timestamp tokens: `<|0.00|>` through `<|30.00|>`, 0.02 s steps.
const N_TIMESTAMP_TOKENS: TokenId = 1501;

/// Pre-tokenization pattern of the BPE. Needs Unicode-aware classes and a
/// look-ahead, which the BPE's fancy-regex engine provides.
const PAT_STR: &str = r"'s|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+";

/// Languages typically written without spaces between words; word splitting
/// falls back to unicode-point boundaries for these.
const NO_SPACE_LANGUAGES: [&str; 6] = ["zh", "ja", "th", "lo", "my", "yue"];

/// ASCII punctuation (the classic C-locale set), used to classify decoded
/// fragments when grouping tokens into words.
const ASCII_PUNCTUATION: &str = "!\"#$%&'()*+,-./:;<=>?@[\\]^_`{|}~";

/// The Whisper tokenizer: a byte-pair encoding plus special-token layout.
///
/// Construction mirrors the model family: English-only models use the gpt2
/// vocabulary and carry no language or task tokens in the start sequence;
/// multilingual models use the multilingual vocabulary, defaulting to English
/// and transcription when no language or task is given.
pub struct Tokenizer {
    bpe: CoreBPE,
    num_languages: usize,
    language: Option<&'static str>,
    task: Option<Task>,
    eot: TokenId,
    sot: TokenId,
    translate: TokenId,
    transcribe: TokenId,
    sot_lm: TokenId,
    sot_prev: TokenId,
    no_speech: TokenId,
    no_timestamps: TokenId,
    timestamp_begin: TokenId,
    sot_sequence: Vec<TokenId>,
}

impl std::fmt::Debug for Tokenizer {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Tokenizer")
            .field("num_languages", &self.num_languages)
            .field("language", &self.language)
            .field("task", &self.task)
            .field("n_vocab", &self.n_vocab())
            .finish_non_exhaustive()
    }
}

impl Tokenizer {
    /// Builds a tokenizer for a model.
    ///
    /// `multilingual` and `num_languages` come from the model (99 languages
    /// for most multilingual models, 100 for large-v3 and its turbo variant).
    /// `language` accepts a code, an English name, or a known alias,
    /// case-insensitively. For English-only models the language and task are
    /// ignored (the start sequence carries neither), but an unknown language
    /// still fails.
    ///
    /// # Errors
    ///
    /// Returns [`WhisperError::UnsupportedLanguage`] when the language is
    /// unknown, or known but outside the model's first `num_languages`
    /// languages.
    pub fn new(
        multilingual: bool,
        num_languages: usize,
        language: Option<&str>,
        task: Option<Task>,
    ) -> Result<Self, WhisperError> {
        let normalized = match language {
            Some(l) => Some(languages::normalize(l).ok_or_else(|| {
                WhisperError::UnsupportedLanguage(l.to_string())
            })?),
            None => None,
        };

        let (vocab, language, task) = if multilingual {
            let language = normalized.or(Some("en"));
            let task = task.or(Some(Task::Transcribe));
            (assets::MULTILINGUAL_TIKTOKEN, language, task)
        } else {
            (assets::GPT2_TIKTOKEN, None, None)
        };

        let (bpe, base) = build_bpe(vocab, num_languages);

        let eot = base;
        let sot = eot + 1;
        let translate = sot + 1 + num_languages as TokenId;
        let transcribe = translate + 1;
        let sot_lm = translate + 2;
        let sot_prev = translate + 3;
        let no_speech = translate + 4;
        let no_timestamps = translate + 5;
        let timestamp_begin = translate + 6;

        let mut sot_sequence = vec![sot];
        if let Some(code) = language {
            let index =
                languages::index_of(code, num_languages).ok_or_else(|| {
                    WhisperError::UnsupportedLanguage(code.to_string())
                })?;
            sot_sequence.push(sot + 1 + index as TokenId);
        }
        if let Some(task) = task {
            sot_sequence.push(match task {
                Task::Transcribe => transcribe,
                Task::Translate => translate,
            });
        }

        Ok(Self {
            bpe,
            num_languages,
            language,
            task,
            eot,
            sot,
            translate,
            transcribe,
            sot_lm,
            sot_prev,
            no_speech,
            no_timestamps,
            timestamp_begin,
            sot_sequence,
        })
    }

    /// The normalized language code, when configured.
    pub fn language(&self) -> Option<&'static str> { self.language }

    /// The task, when configured.
    pub fn task(&self) -> Option<Task> { self.task }

    /// `<|endoftext|>`.
    pub fn eot(&self) -> TokenId { self.eot }

    /// `<|startoftranscript|>`.
    pub fn sot(&self) -> TokenId { self.sot }

    /// `<|translate|>`.
    pub fn translate(&self) -> TokenId { self.translate }

    /// `<|transcribe|>`.
    pub fn transcribe(&self) -> TokenId { self.transcribe }

    /// `<|startoflm|>`.
    pub fn sot_lm(&self) -> TokenId { self.sot_lm }

    /// `<|startofprev|>`.
    pub fn sot_prev(&self) -> TokenId { self.sot_prev }

    /// `<|nospeech|>`.
    pub fn no_speech(&self) -> TokenId { self.no_speech }

    /// `<|notimestamps|>`.
    pub fn no_timestamps(&self) -> TokenId { self.no_timestamps }

    /// `<|0.00|>` — every id at or above this one is a timestamp token.
    pub fn timestamp_begin(&self) -> TokenId { self.timestamp_begin }

    /// Total vocabulary size (base tokens plus all specials).
    pub fn n_vocab(&self) -> usize {
        (self.timestamp_begin + N_TIMESTAMP_TOKENS) as usize
    }

    /// The start sequence: `sot`, then the language token and the task token
    /// when configured.
    pub fn sot_sequence(&self) -> &[TokenId] { &self.sot_sequence }

    /// The start sequence with `<|notimestamps|>` appended.
    pub fn sot_sequence_including_notimestamps(&self) -> Vec<TokenId> {
        let mut seq = self.sot_sequence.clone();
        seq.push(self.no_timestamps);
        seq
    }

    /// The token of the configured language, when one is set.
    pub fn language_token(&self) -> Option<TokenId> {
        self.language.and_then(|code| self.to_language_token(code))
    }

    /// The token of `code`, when it is inside the model's language set.
    pub fn to_language_token(&self, code: &str) -> Option<TokenId> {
        languages::index_of(code, self.num_languages)
            .map(|index| self.sot + 1 + index as TokenId)
    }

    /// All language tokens, in vocabulary order.
    pub fn all_language_tokens(&self) -> Vec<TokenId> {
        (0..self.num_languages as TokenId)
            .map(|i| self.sot + 1 + i)
            .collect()
    }

    /// All language codes, in vocabulary order.
    pub fn all_language_codes(&self) -> Vec<&'static str> {
        languages::LANGUAGES
            .iter()
            .take(self.num_languages)
            .map(|(code, _)| *code)
            .collect()
    }

    /// Encodes plain text (no special-token matching).
    pub fn encode(&self, text: &str) -> Vec<TokenId> {
        self.bpe.encode_ordinary(text)
    }

    /// Decodes tokens to text, dropping timestamp tokens.
    ///
    /// Non-timestamp special tokens are rendered in their `<|...|>` form;
    /// invalid UTF-8 becomes U+FFFD replacement characters.
    pub fn decode(&self, tokens: &[TokenId]) -> String {
        let filtered: Vec<TokenId> = tokens
            .iter()
            .copied()
            .filter(|&t| t < self.timestamp_begin)
            .collect();
        self.decode_lossy(&filtered)
    }

    /// Decodes tokens to text with timestamp tokens rendered as `<|1.08|>`.
    pub fn decode_with_timestamps(&self, tokens: &[TokenId]) -> String {
        self.decode_lossy(tokens)
    }

    fn decode_lossy(&self, tokens: &[TokenId]) -> String {
        let bytes = self
            .bpe
            .decode_bytes(tokens)
            .expect("token id outside the vocabulary");
        String::from_utf8_lossy(&bytes).into_owned()
    }

    /// Tokens to suppress during sampling to avoid non-speech annotations
    /// (`♪♪♪`, `(SPEAKING FOREIGN LANGUAGE)`, `[DAVID]`, ...), while keeping
    /// basic punctuation.
    ///
    /// Hyphens and single quotes are still allowed between words: the set
    /// contains their word-initial (space-prefixed) forms only.
    pub fn non_speech_tokens(&self) -> Vec<TokenId> {
        let singles = "\"#()*+/:;<=>@[\\]^_`{|}~「」『』";
        let multis = [
            "<<",
            ">>",
            "<<<",
            ">>>",
            "--",
            "---",
            "-(",
            "-[",
            "('",
            "(\"",
            "((",
            "))",
            "(((",
            ")))",
            "[[",
            "]]",
            "{{",
            "}}",
            "♪♪",
            "♪♪♪",
        ];
        let miscellaneous = "♩♪♫♬♭♮♯";

        let mut result: BTreeSet<TokenId> = BTreeSet::new();
        result.insert(self.encode(" -")[0]);
        result.insert(self.encode(" '")[0]);

        let symbols = singles
            .chars()
            .map(String::from)
            .chain(multis.iter().map(|s| (*s).to_string()))
            .chain(miscellaneous.chars().map(String::from));
        for symbol in symbols {
            let is_miscellaneous =
                symbol.chars().count() == 1 && miscellaneous.contains(&symbol);
            for tokens in
                [self.encode(&symbol), self.encode(&format!(" {symbol}"))]
            {
                if tokens.len() == 1 || is_miscellaneous {
                    result.insert(tokens[0]);
                }
            }
        }

        result.into_iter().collect()
    }

    /// Splits decoded tokens into words with their token spans.
    ///
    /// Languages written without spaces split on unicode-point boundaries;
    /// the rest split on spaces and punctuation.
    pub fn split_to_word_tokens(
        &self,
        tokens: &[TokenId],
    ) -> (Vec<String>, Vec<Vec<TokenId>>) {
        if self
            .language
            .is_some_and(|l| NO_SPACE_LANGUAGES.contains(&l))
        {
            self.split_tokens_on_unicode(tokens)
        } else {
            self.split_tokens_on_spaces(tokens)
        }
    }

    /// Groups tokens at every boundary where the accumulated bytes decode to
    /// valid unicode (a U+FFFD in the fragment is accepted only when the full
    /// decode genuinely contains one at that position).
    fn split_tokens_on_unicode(
        &self,
        tokens: &[TokenId],
    ) -> (Vec<String>, Vec<Vec<TokenId>>) {
        const REPLACEMENT: char = '\u{fffd}';
        let decoded_full = self.decode_with_timestamps(tokens);
        let full_chars: Vec<char> = decoded_full.chars().collect();

        let mut words = Vec::new();
        let mut word_tokens: Vec<Vec<TokenId>> = Vec::new();
        let mut current: Vec<TokenId> = Vec::new();
        let mut unicode_offset = 0usize; // in chars

        for &token in tokens {
            current.push(token);
            let decoded = self.decode_with_timestamps(&current);

            let is_boundary =
                match decoded.chars().position(|c| c == REPLACEMENT) {
                    None => true,
                    Some(pos) => {
                        full_chars.get(unicode_offset + pos) ==
                            Some(&REPLACEMENT)
                    },
                };
            if is_boundary {
                unicode_offset += decoded.chars().count();
                words.push(decoded);
                word_tokens.push(std::mem::take(&mut current));
            }
        }

        (words, word_tokens)
    }

    /// Groups unicode fragments into words, starting a new word at specials,
    /// space-prefixed fragments, and punctuation.
    fn split_tokens_on_spaces(
        &self,
        tokens: &[TokenId],
    ) -> (Vec<String>, Vec<Vec<TokenId>>) {
        let (subwords, subword_tokens_list) =
            self.split_tokens_on_unicode(tokens);

        let mut words: Vec<String> = Vec::new();
        let mut word_tokens: Vec<Vec<TokenId>> = Vec::new();

        for (subword, subword_tokens) in
            subwords.into_iter().zip(subword_tokens_list)
        {
            let special = subword_tokens[0] >= self.eot;
            let with_space = subword.starts_with(' ');
            // Substring check on purpose: any contiguous run of ASCII
            // punctuation (and the empty string) counts as punctuation.
            let punctuation = ASCII_PUNCTUATION.contains(subword.trim());
            if special || with_space || punctuation || words.is_empty() {
                words.push(subword);
                word_tokens.push(subword_tokens);
            } else {
                let last = words.len() - 1;
                words[last].push_str(&subword);
                word_tokens[last].extend(subword_tokens);
            }
        }

        (words, word_tokens)
    }
}

/// Parses a `.tiktoken` vocabulary and appends Whisper's special tokens in
/// their canonical order. Returns the BPE and the base vocabulary size (the
/// id of the first special token).
fn build_bpe(vocab: &str, num_languages: usize) -> (CoreBPE, TokenId) {
    let mut encoder: FxHashMap<Vec<u8>, TokenId> = FxHashMap::default();
    for line in vocab.lines().filter(|l| !l.is_empty()) {
        let (token, rank) = line
            .split_once(' ')
            .expect("embedded vocabulary line must be `base64 rank`");
        let bytes = match BASE64.decode(token) {
            Ok(bytes) => bytes,
            // The multilingual vocabulary ends with a bare padding character
            // (`= 50256`); the reference reads it with a lenient base64
            // parser, yielding an empty token. Replicate exactly that.
            Err(_) if token.chars().all(|c| c == '=') => Vec::new(),
            Err(e) => {
                panic!("embedded vocabulary token must be valid base64: {e}")
            },
        };
        let rank: TokenId = rank
            .parse()
            .expect("embedded vocabulary rank must be an integer");
        encoder.insert(bytes, rank);
    }
    let base = encoder.len() as TokenId;

    let mut names: Vec<String> =
        Vec::with_capacity(8 + num_languages + N_TIMESTAMP_TOKENS as usize);
    names.push("<|endoftext|>".to_string());
    names.push("<|startoftranscript|>".to_string());
    for (code, _) in languages::LANGUAGES.iter().take(num_languages) {
        names.push(format!("<|{code}|>"));
    }
    for name in [
        "<|translate|>",
        "<|transcribe|>",
        "<|startoflm|>",
        "<|startofprev|>",
        "<|nospeech|>",
        "<|notimestamps|>",
    ] {
        names.push(name.to_string());
    }
    for i in 0..N_TIMESTAMP_TOKENS {
        // Hundredths of a second, in exact integer arithmetic: 0.02 s steps.
        let hundredths = i * 2;
        names.push(format!("<|{}.{:02}|>", hundredths / 100, hundredths % 100));
    }

    let special_tokens: FxHashMap<String, TokenId> = names
        .into_iter()
        .enumerate()
        .map(|(i, name)| (name, base + i as TokenId))
        .collect();

    let bpe = CoreBPE::new(encoder, special_tokens, PAT_STR)
        .expect("embedded vocabulary must produce a valid encoding");
    (bpe, base)
}
