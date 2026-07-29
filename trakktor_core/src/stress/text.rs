//! The text-level layer: the alphabet, the reference's word split, sentence
//! segmentation, and the two forms a stress mark can take.
//!
//! No tensors and no model here — everything in this module is exact string
//! work, which is also where most of the reference's edge cases live.

/// The mark the reference uses: a plus sign **before** the stressed vowel.
pub(super) const STRESS: char = '+';

/// The combining acute accent, which goes **after** the stressed vowel — the
/// form Russian corpora and dictionaries use.
pub(super) const ACUTE: char = '\u{0301}';

/// The vowels, in the reference's own order (only membership matters).
const VOWELS: [char; 10] = ['а', 'о', 'у', 'ы', 'э', 'и', 'е', 'я', 'ё', 'ю'];

/// Characters that end a sentence, for segmentation.
const ENDERS: [char; 4] = ['.', '!', '?', '…'];

/// The characters the reference splits words on, besides whitespace.
const SEPARATORS: [char; 13] = [
    '.', ',', '!', '?', ';', ':', '<', '>', '=', '(', ')', '/', '\\',
];

/// Which form the output marks stress in.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum Marker {
    /// `+` before the stressed vowel — what the speech engines read.
    #[default]
    Plus,
    /// Combining acute U+0301 after the stressed vowel.
    Acute,
}

impl Marker {
    /// The value reported in the output contract.
    #[must_use]
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Plus => "plus",
            Self::Acute => "acute",
        }
    }
}

/// Whether `c` is a Russian letter — the alphabet the model knows.
#[must_use]
pub(super) fn is_russian(c: char) -> bool {
    matches!(c, 'А'..='Я' | 'а'..='я' | 'ё' | 'Ё')
}

/// Whether an already-lowercased character is a vowel.
#[must_use]
pub(super) fn is_vowel(c: char) -> bool { VOWELS.contains(&c) }

/// Lowercases one character, keeping the one-to-one correspondence the whole
/// word-level algorithm indexes by. (For the Russian alphabet this is exact;
/// for the rare character whose lowercase is longer, the first character is
/// taken — the reference has the same blind spot, and its own positions would
/// slide the same way.)
#[must_use]
pub(super) fn lower(c: char) -> char { c.to_lowercase().next().unwrap_or(c) }

/// Uppercases one character, the same way round.
#[must_use]
pub(super) fn upper(c: char) -> char { c.to_uppercase().next().unwrap_or(c) }

/// The model's view of a word: lowercase, Russian letters only.
#[must_use]
pub(super) fn clean(word: &[char]) -> String {
    word.iter()
        .copied()
        .map(lower)
        .filter(|c| is_russian(*c))
        .collect()
}

/// The key both the reference's own skip lists and our user dictionary use:
/// the clean form with `ё` folded into `е`, so one entry covers a word whether
/// or not its `ё` is written.
#[must_use]
pub(super) fn fold_yo(word: &str) -> String { word.replace('ё', "е") }

/// One word of the reference's split, with everything the rules need.
#[derive(Debug, Clone)]
pub(super) struct Token {
    /// The characters exactly as they appear in the input.
    pub raw: Vec<char>,
    /// The same characters lowercased one-to-one, so positions line up.
    pub lower: Vec<char>,
    /// The model's view of the word (may be empty for punctuation).
    pub clean: String,
    /// Whether the rules apply to this token at all.
    pub process: bool,
}

impl Token {
    fn new(raw: Vec<char>, process: bool) -> Self {
        let lower: Vec<char> = raw.iter().copied().map(lower).collect();
        let clean = clean(&raw);
        let process = process && !clean.is_empty();
        Self {
            raw,
            lower,
            clean,
            process,
        }
    }

    /// Whether the caller already marked the stress.
    pub fn has_stress(&self) -> bool { self.lower.contains(&STRESS) }

    /// Whether the caller already wrote `ё`.
    pub fn has_yo(&self) -> bool { self.lower.contains(&'ё') }

    /// Character positions of the vowels.
    pub fn vowels(&self) -> Vec<usize> {
        self.lower
            .iter()
            .enumerate()
            .filter(|(_, c)| is_vowel(**c))
            .map(|(index, _)| index)
            .collect()
    }

    /// Character positions of the letter `е`, which is what the `ё` head
    /// numbers.
    pub fn ye(&self) -> Vec<usize> {
        self.lower
            .iter()
            .enumerate()
            .filter(|(_, c)| **c == 'е')
            .map(|(index, _)| index)
            .collect()
    }
}

/// Splits text the way the reference does: on runs of whitespace and its
/// punctuation set (the separators are kept as their own tokens), then each
/// piece again on hyphens.
///
/// Concatenating every token's `raw` reproduces the input exactly.
#[must_use]
pub(super) fn tokenize(text: &str) -> Vec<Token> {
    let mut tokens = Vec::new();
    let chars: Vec<char> = text.chars().collect();
    let separator = |c: char| c.is_whitespace() || SEPARATORS.contains(&c);

    let mut at = 0;
    while at < chars.len() {
        let start = at;
        let is_separator = separator(chars[at]);
        while at < chars.len() && separator(chars[at]) == is_separator {
            at += 1;
        }
        let chunk = &chars[start..at];
        if is_separator {
            tokens.push(Token::new(chunk.to_vec(), false));
        } else {
            push_hyphenated(chunk, &mut tokens);
        }
    }
    tokens
}

/// Splits one non-separator chunk on hyphens, keeping the hyphen attached to
/// the part before it. The reference leaves the trailing `-то` of `что-то`,
/// `кому-то` and friends unmarked, on purpose.
fn push_hyphenated(chunk: &[char], tokens: &mut Vec<Token>) {
    let parts: Vec<&[char]> = chunk.split(|c| *c == '-').collect();
    let last = parts.len() - 1;
    // A word without a hyphen is always marked — the `-то` exemption belongs to
    // the hyphenated form, not to the standalone word «то».
    let hyphenated = last > 0;
    for (index, part) in parts.iter().enumerate() {
        let mut raw = part.to_vec();
        let process = if index == last {
            !hyphenated || part.iter().collect::<String>() != "то"
        } else {
            raw.push('-');
            true
        };
        tokens.push(Token::new(raw, process));
    }
}

/// Cuts text into the units the homograph solver takes as context: sentences.
///
/// A unit ends after a run of sentence-ending punctuation followed by
/// whitespace (the whitespace stays with it), and after every line break.
/// Concatenating the pieces reproduces the input exactly — the operation must
/// give back the same text it was handed.
#[must_use]
pub(super) fn segments(text: &str) -> Vec<&str> {
    let mut out = Vec::new();
    let mut start = 0;
    let mut at = 0;
    let bytes = text.as_bytes();
    while at < text.len() {
        let c = text[at..].chars().next().expect("boundary");
        if ENDERS.contains(&c) {
            let mut end = at;
            while let Some(next) = text[end..].chars().next() {
                if !ENDERS.contains(&next) {
                    break;
                }
                end += next.len_utf8();
            }
            let mut after = end;
            while let Some(next) = text[after..].chars().next() {
                if !next.is_whitespace() {
                    break;
                }
                after += next.len_utf8();
            }
            if after > end {
                out.push(&text[start..after]);
                start = after;
                at = after;
                continue;
            }
            at = end;
            continue;
        }
        if bytes[at] == b'\n' {
            out.push(&text[start..=at]);
            start = at + 1;
            at = start;
            continue;
        }
        at += c.len_utf8();
    }
    if start < text.len() {
        out.push(&text[start..]);
    }
    out
}

/// Turns any combining acute in the input into the `+` form the pipeline works
/// in, so the operation accepts its own `--marker acute` output.
#[must_use]
pub(super) fn acute_to_plus(text: &str) -> String {
    if !text.contains(ACUTE) {
        return text.to_owned();
    }
    let chars: Vec<char> = text.chars().collect();
    let mut out = String::with_capacity(text.len());
    let mut at = 0;
    while at < chars.len() {
        let c = chars[at];
        // The acute follows its vowel; the plus goes before it. An acute after
        // anything else is left where it is rather than silently dropped.
        if chars.get(at + 1) == Some(&ACUTE) && is_vowel(lower(c)) {
            out.push(STRESS);
            out.push(c);
            at += 2;
            continue;
        }
        out.push(c);
        at += 1;
    }
    out
}

/// Renders `+`-marked text in the requested form.
///
/// Two details of the acute form are conventions, not arithmetic: `ё` carries
/// no accent (the letter is stressed by definition, and no Russian dictionary
/// marks it), and a `+` that does not sit in front of a vowel is left alone
/// rather than silently dropped — the pipeline never produces one, so it can
/// only have come from the caller's own text.
#[must_use]
pub(super) fn render(text: &str, marker: Marker) -> String {
    if marker == Marker::Plus {
        return text.to_owned();
    }
    let mut out = String::with_capacity(text.len());
    let mut chars = text.chars().peekable();
    while let Some(c) = chars.next() {
        if c == STRESS &&
            chars.peek().is_some_and(|next| is_vowel(lower(*next)))
        {
            let vowel = chars.next().expect("peeked");
            out.push(vowel);
            if lower(vowel) != 'ё' {
                out.push(ACUTE);
            }
            continue;
        }
        out.push(c);
    }
    out
}

#[cfg(test)]
mod tests;
