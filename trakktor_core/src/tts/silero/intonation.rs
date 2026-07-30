//! Which way a sentence is read: statement, question, exclamation.
//!
//! One published model conditions its pitch head on an **utterance type** —
//! six of them — and works out that type in its frontend, by rule, from the
//! punctuation and the opening words. So the rule set is data of the reference
//! that lives in code rather than in the weights, and it is ported here word
//! for word, lists included.
//!
//! The other models have no such head; for them this module is never called.

/// The utterance types, in the order the model's embedding numbers them.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Utterance {
    /// A statement — and the fallback for anything unclassified.
    Statement,
    /// A question opening with a question word ("who", "where", …).
    WhQuestion,
    /// A yes/no question.
    GeneralQuestion,
    /// A question offering alternatives ("… или …?").
    Alternative,
    /// A question tagged onto a statement ("…, правда?").
    Tag,
    /// An exclamation.
    Exclamation,
}

impl Utterance {
    /// The id the pitch head's embedding is indexed by.
    #[must_use]
    pub fn id(self) -> u32 {
        match self {
            Self::Statement => 0,
            Self::WhQuestion => 1,
            Self::GeneralQuestion => 2,
            Self::Alternative => 3,
            Self::Tag => 4,
            Self::Exclamation => 5,
        }
    }
}

/// Question words, any of which in the opening of a question makes it a
/// wh-question.
const QUESTION_WORDS: &[&str] = &[
    "кто",
    "кого",
    "кому",
    "кем",
    "ком",
    "что",
    "чего",
    "чему",
    "чем",
    "чём",
    "чё",
    "чо",
    "где",
    "куда",
    "откуда",
    "когда",
    "почему",
    "зачем",
    "как",
    "почём",
    "отчего",
    "насколько",
    "сколько",
    "скольких",
    "скольким",
    "сколькими",
    "какой",
    "какая",
    "какое",
    "какие",
    "какого",
    "каких",
    "какому",
    "каким",
    "какими",
    "каком",
    "какую",
    "чей",
    "чья",
    "чьё",
    "чье",
    "чьи",
    "чьего",
    "чьей",
    "чьих",
    "чьему",
    "чьим",
    "чьими",
    "чьём",
    "чьем",
    "чью",
    "который",
    "которая",
    "которое",
    "которые",
    "которого",
    "которой",
    "которых",
    "которому",
    "которым",
    "которыми",
    "котором",
    "которую",
    "каков",
    "какова",
    "каково",
    "каковы",
];

/// Words that may open a sentence without counting as its first content word.
const FILLERS: &[&str] = &[
    "а",
    "ну",
    "и",
    "так",
    "вот",
    "слышь",
    "слушай",
    "скажите",
    "скажи",
    "пожалуйста",
    "вобще",
    "вообще",
    "типа",
    "короче",
];

/// The phrases that turn a question into a tag question when they close it,
/// after at least one comma or space.
const TAG_PHRASES: &[&str] = &[
    "правда",
    "верно",
    "да",
    "не так ли",
    "не правда ли",
    "разве не так",
    "ведь так",
    "ведь",
    "а",
];

/// Content words of the opening that are looked at for a question word.
const OPENING_CONTENT_WORDS: usize = 4;

/// Words of the opening that are looked at at all.
const OPENING_WORDS: usize = 8;

/// Quotation marks a sentence may be wrapped in.
const QUOTES_OPEN: [char; 4] = ['"', '«', '\u{201c}', '\u{201e}'];
const QUOTES_CLOSE: [char; 4] = ['"', '»', '\u{201d}', '\u{2019}'];

/// Splits a text into sentences the way the reference does: at a run of
/// whitespace that follows `.`, `!` or `?`.
///
/// Empty pieces are **kept** — the reference's own splitter keeps them, and the
/// per-symbol type ids are laid out against exactly this list.
#[must_use]
pub fn sentences(text: &str) -> Vec<&str> {
    let text = text.trim();
    let mut out = Vec::new();
    let mut start = 0;
    let mut previous: Option<char> = None;
    let mut at = 0;
    while at < text.len() {
        let rest = &text[at..];
        let character = rest.chars().next().expect("a char at a boundary");
        if character.is_whitespace() &&
            matches!(previous, Some('.' | '!' | '?'))
        {
            let run = rest
                .find(|c: char| !c.is_whitespace())
                .unwrap_or(rest.len());
            out.push(&text[start..at]);
            at += run;
            start = at;
            previous = None;
            continue;
        }
        previous = Some(character);
        at += character.len_utf8();
    }
    out.push(&text[start..]);
    out
}

/// Classifies one sentence.
#[must_use]
pub fn classify(sentence: &str) -> Utterance {
    let sentence = sentence.trim();
    let stripped = strip_quotes(sentence);
    if stripped.is_empty() {
        return Utterance::Statement;
    }
    // A question mark followed by an exclamation or an ellipsis is still a
    // question; the reference peels those off before looking.
    let tail = if stripped.ends_with("?!") ||
        stripped.ends_with("?..") ||
        stripped.ends_with("?...")
    {
        stripped.trim_end_matches(['.', '!'])
    } else {
        stripped
    };
    if tail.ends_with('?') {
        let question = normalize(tail);
        if is_tag(&question) {
            return Utterance::Tag;
        }
        if opens_with_question_word(&question) {
            return Utterance::WhQuestion;
        }
        if contains_word(&question, "или") {
            return Utterance::Alternative;
        }
        return Utterance::GeneralQuestion;
    }
    if stripped.ends_with('!') {
        return Utterance::Exclamation;
    }
    Utterance::Statement
}

/// Classifies every sentence of a text.
#[must_use]
pub fn classify_text(text: &str) -> Vec<Utterance> {
    let types: Vec<Utterance> = sentences(text)
        .into_iter()
        .filter(|sentence| !sentence.trim().is_empty())
        .map(classify)
        .collect();
    if types.is_empty() {
        vec![Utterance::Statement]
    } else {
        types
    }
}

/// The per-symbol type ids for one utterance, as the reference lays them out.
///
/// The mapping is by character position in the **raw** text, not in the cleaned
/// one the ids were built from, so a text whose cleaning removed characters
/// drifts against its own sequence — and the tail is filled with the first
/// sentence's type. That is the reference's behaviour, and it only shows on a
/// text that mixes sentence types.
#[must_use]
pub fn type_ids(text: &str, types: &[Utterance], length: usize) -> Vec<u32> {
    let default = types.first().copied().unwrap_or(Utterance::Statement).id();
    let mut per_character: Vec<u32> = Vec::new();
    let pieces = sentences(text);
    for (index, sentence) in pieces.iter().enumerate() {
        let id = types
            .get(index)
            .or_else(|| types.last())
            .copied()
            .unwrap_or(Utterance::Statement)
            .id();
        per_character.extend(std::iter::repeat_n(id, sentence.chars().count()));
        if index + 1 < pieces.len() {
            // The whitespace the split consumed counts as one character.
            per_character.push(id);
        }
    }
    let mut ids = vec![0u32; length];
    if per_character.is_empty() {
        return ids;
    }
    // Position zero is the start-of-sequence symbol, which every published
    // model has.
    if let Some(slot) = ids.first_mut() {
        *slot = default;
    }
    for (slot, id) in ids.iter_mut().skip(1).zip(&per_character) {
        *slot = *id;
    }
    for slot in ids.iter_mut().skip(1 + per_character.len()) {
        *slot = default;
    }
    ids
}

/// Drops the quotation marks a sentence may be wrapped in.
fn strip_quotes(text: &str) -> &str {
    let mut text = text.trim();
    while let Some(rest) = text.strip_prefix(QUOTES_OPEN) {
        text = rest.trim();
    }
    while let Some(rest) = text.strip_suffix(QUOTES_CLOSE) {
        text = rest.trim();
    }
    text
}

/// The form the rules below are matched against: no quotes, no stress marks,
/// `ё` folded into `е`, whitespace collapsed.
fn normalize(text: &str) -> String {
    let stripped = strip_quotes(text);
    let mut out = String::with_capacity(stripped.len());
    let mut space = false;
    for character in stripped.chars() {
        match character {
            '+' => continue,
            'ё' => space = push(&mut out, 'е', space),
            'Ё' => space = push(&mut out, 'Е', space),
            c if c.is_whitespace() => space = true,
            c => space = push(&mut out, c, space),
        }
    }
    out.trim().to_owned()
}

/// Appends one character, inserting the whitespace that was collapsed before
/// it. Returns the new "a space is pending" state.
fn push(out: &mut String, character: char, space: bool) -> bool {
    if space && !out.is_empty() {
        out.push(' ');
    }
    out.push(character);
    false
}

/// Whether a question closes with one of the tag phrases.
fn is_tag(question: &str) -> bool {
    let lowered = question.to_lowercase();
    let Some(body) = lowered.strip_suffix('?') else {
        return false;
    };
    let body = body.trim_end();
    TAG_PHRASES.iter().any(|phrase| {
        body.strip_suffix(phrase).is_some_and(|before| {
            // The phrase has to be set off by at least one comma or space, so
            // that «да?» on its own is not a tag and «…, да?» is.
            before.ends_with([',', ' '])
        })
    })
}

/// Whether one of the first content words is a question word.
fn opens_with_question_word(question: &str) -> bool {
    let lowered = question.to_lowercase();
    words(&lowered)
        .take(OPENING_WORDS)
        .filter(|word| !FILLERS.contains(&word.as_str()))
        .take(OPENING_CONTENT_WORDS)
        .any(|word| QUESTION_WORDS.contains(&word.as_str()))
}

/// The words of a text: runs of Cyrillic or Latin letters.
fn words(text: &str) -> impl Iterator<Item = String> + '_ {
    let letter = |c: char| {
        matches!(c, 'а'..='я' | 'ё' | 'А'..='Я' | 'Ё') ||
            c.is_ascii_alphabetic()
    };
    text.split(move |c: char| !letter(c))
        .filter(|word| !word.is_empty())
        .map(str::to_owned)
}

/// Whether `needle` appears in `text` as a whole word.
fn contains_word(text: &str, needle: &str) -> bool {
    let lowered = text.to_lowercase();
    let word = |c: char| c.is_alphanumeric() || c == '_';
    let mut at = 0;
    while let Some(found) = lowered[at..].find(needle) {
        let start = at + found;
        let end = start + needle.len();
        let before = lowered[..start].chars().next_back();
        let after = lowered[end..].chars().next();
        if !before.is_some_and(word) && !after.is_some_and(word) {
            return true;
        }
        at = end;
    }
    false
}

#[cfg(test)]
mod tests;
