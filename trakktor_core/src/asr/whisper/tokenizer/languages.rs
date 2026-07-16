//! The language inventory of the multilingual Whisper models.
//!
//! The order is significant: language token ids follow the start-of-transcript
//! token in exactly this order, and a model's language set is the first
//! `num_languages` entries (99 for most multilingual models, 100 for large-v3
//! and its turbo variant, whose vocabulary adds Cantonese).

/// `(code, english name)` pairs, in vocabulary order.
pub(crate) static LANGUAGES: [(&str, &str); 100] = [
    ("en", "english"),
    ("zh", "chinese"),
    ("de", "german"),
    ("es", "spanish"),
    ("ru", "russian"),
    ("ko", "korean"),
    ("fr", "french"),
    ("ja", "japanese"),
    ("pt", "portuguese"),
    ("tr", "turkish"),
    ("pl", "polish"),
    ("ca", "catalan"),
    ("nl", "dutch"),
    ("ar", "arabic"),
    ("sv", "swedish"),
    ("it", "italian"),
    ("id", "indonesian"),
    ("hi", "hindi"),
    ("fi", "finnish"),
    ("vi", "vietnamese"),
    ("he", "hebrew"),
    ("uk", "ukrainian"),
    ("el", "greek"),
    ("ms", "malay"),
    ("cs", "czech"),
    ("ro", "romanian"),
    ("da", "danish"),
    ("hu", "hungarian"),
    ("ta", "tamil"),
    ("no", "norwegian"),
    ("th", "thai"),
    ("ur", "urdu"),
    ("hr", "croatian"),
    ("bg", "bulgarian"),
    ("lt", "lithuanian"),
    ("la", "latin"),
    ("mi", "maori"),
    ("ml", "malayalam"),
    ("cy", "welsh"),
    ("sk", "slovak"),
    ("te", "telugu"),
    ("fa", "persian"),
    ("lv", "latvian"),
    ("bn", "bengali"),
    ("sr", "serbian"),
    ("az", "azerbaijani"),
    ("sl", "slovenian"),
    ("kn", "kannada"),
    ("et", "estonian"),
    ("mk", "macedonian"),
    ("br", "breton"),
    ("eu", "basque"),
    ("is", "icelandic"),
    ("hy", "armenian"),
    ("ne", "nepali"),
    ("mn", "mongolian"),
    ("bs", "bosnian"),
    ("kk", "kazakh"),
    ("sq", "albanian"),
    ("sw", "swahili"),
    ("gl", "galician"),
    ("mr", "marathi"),
    ("pa", "punjabi"),
    ("si", "sinhala"),
    ("km", "khmer"),
    ("sn", "shona"),
    ("yo", "yoruba"),
    ("so", "somali"),
    ("af", "afrikaans"),
    ("oc", "occitan"),
    ("ka", "georgian"),
    ("be", "belarusian"),
    ("tg", "tajik"),
    ("sd", "sindhi"),
    ("gu", "gujarati"),
    ("am", "amharic"),
    ("yi", "yiddish"),
    ("lo", "lao"),
    ("uz", "uzbek"),
    ("fo", "faroese"),
    ("ht", "haitian creole"),
    ("ps", "pashto"),
    ("tk", "turkmen"),
    ("nn", "nynorsk"),
    ("mt", "maltese"),
    ("sa", "sanskrit"),
    ("lb", "luxembourgish"),
    ("my", "myanmar"),
    ("bo", "tibetan"),
    ("tl", "tagalog"),
    ("mg", "malagasy"),
    ("as", "assamese"),
    ("tt", "tatar"),
    ("haw", "hawaiian"),
    ("ln", "lingala"),
    ("ha", "hausa"),
    ("ba", "bashkir"),
    ("jw", "javanese"),
    ("su", "sundanese"),
    ("yue", "cantonese"),
];

/// Additional `(alias, code)` names accepted on input.
static LANGUAGE_ALIASES: [(&str, &str); 12] = [
    ("burmese", "my"),
    ("valencian", "ca"),
    ("flemish", "nl"),
    ("haitian", "ht"),
    ("letzeburgesch", "lb"),
    ("pushto", "ps"),
    ("panjabi", "pa"),
    ("moldavian", "ro"),
    ("moldovan", "ro"),
    ("sinhalese", "si"),
    ("castilian", "es"),
    ("mandarin", "zh"),
];

/// Normalizes a user-supplied language (code, English name, or alias,
/// case-insensitive) to its canonical code. Returns `None` when unknown.
pub(crate) fn normalize(language: &str) -> Option<&'static str> {
    let lower = language.to_lowercase();
    for (code, name) in &LANGUAGES {
        if lower == *code || lower == *name {
            return Some(code);
        }
    }
    for (alias, code) in &LANGUAGE_ALIASES {
        if lower == *alias {
            return Some(code);
        }
    }
    None
}

/// Position of `code` within the first `num_languages` languages, or `None`
/// when the code lies outside that set.
pub(crate) fn index_of(code: &str, num_languages: usize) -> Option<usize> {
    LANGUAGES
        .iter()
        .take(num_languages)
        .position(|(c, _)| *c == code)
}
