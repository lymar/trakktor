//! Recovering the built-in encoding of embedded Type 1 fonts.
//!
//! A PDF produced by TeX embeds Type 1 fonts whose encoding lives **inside the
//! font program** — a run of `dup <code> /<glyph> put` in the cleartext part,
//! before `eexec` — and writes neither `/Encoding` nor `/ToUnicode` into the
//! font dictionary. An extractor that does not read font programs has nothing
//! to go on and falls back to StandardEncoding, where the ligature codes are
//! unassigned and the dash and quote codes mean something else. The damage is
//! silent: unassigned codes are dropped rather than replaced, so a "did the
//! text come out with replacement characters" guard never fires. On an ordinary
//! typeset paper that costs the ligatures (`fi`, `ff`, `ffi`), the dashes, the
//! quotes and — worst, because it still reads as mathematics — the operators:
//! `≤` arrives as `6`, `−` and `·` vanish, `⌊⌋` become letters.
//!
//! So the input is repaired rather than the output. The encoding vector found
//! in the font program is written into the font dictionary as
//! `/Encoding << /Differences […] >>`, which is the field PDF provides for
//! exactly this and which every extractor reads. Nothing is invented: what is
//! written is what the file already said, one indirection away.
//!
//! One thing a code-to-character map cannot do is reorder, and TeX draws an
//! accent as a separate glyph *before* its letter (`\=a` is a macron, then an
//! `a`). [`compose_diacritics`] closes that gap afterwards, on the text.

use lopdf::{Dictionary, Document, Object, ObjectId};

/// What a repair pass found and did.
pub struct Repair {
    /// The rewritten document, when there was something to rewrite. `None`
    /// means the caller should go on using the bytes it already has — no PDF
    /// was re-serialized, so a document that needs no repair pays nothing.
    pub bytes: Option<Vec<u8>>,
    /// How many fonts got their encoding back.
    pub fonts: usize,
    /// Whether the document also holds a simple font whose encoding is beyond
    /// recovery — no `/Encoding`, no `/ToUnicode`, and no embedded program to
    /// read the vector out of. Its text may come out wrong and there is
    /// nothing in the file to check it against.
    pub unmapped: bool,
    /// Pages in the document, as the parser counts them. `None` when the
    /// document did not parse here at all, which is not an error in itself:
    /// the engine gets its own turn and reports the failure properly.
    pub page_count: Option<u32>,
}

/// Reads `source`, gives every embedded Type 1 font its own encoding back, and
/// returns the rewritten document.
///
/// Best effort throughout: a document that does not parse, a font program that
/// does not decompress, an encoding vector that is not there — each of those
/// simply means less is repaired, never an error. The caller keeps the original
/// bytes and loses nothing by trying.
pub fn repair(source: &[u8]) -> Repair {
    let Ok(mut document) = Document::load_mem(source) else {
        return Repair {
            bytes: None,
            fonts: 0,
            unmapped: false,
            page_count: None,
        };
    };
    let page_count = u32::try_from(document.get_pages().len()).ok();

    let mut unmapped = false;
    let mut found: Vec<(ObjectId, Vec<(u8, String)>)> = Vec::new();
    for (id, object) in &document.objects {
        let Ok(font) = object.as_dict() else { continue };
        if !is_unencoded_simple_font(font) {
            continue;
        }
        match font_program(&document, font)
            .as_deref()
            .map(builtin_encoding)
        {
            Some(vector) if !vector.is_empty() => found.push((*id, vector)),
            // A font with a descriptor is this document's own — subset,
            // renamed, and only it knows what its codes mean, so failing to
            // read it is worth saying out loud. A font without one is a
            // standard font named for the reader to supply, and its encoding is
            // standard too; there is nothing to recover and nothing to warn
            // about.
            _ => unmapped |= font.get(b"FontDescriptor").is_ok(),
        }
    }

    if found.is_empty() {
        return Repair {
            bytes: None,
            fonts: 0,
            unmapped,
            page_count,
        };
    }

    let fonts = found.len();
    for (id, vector) in found {
        let encoding = differences(&vector);
        if let Ok(font) =
            document.get_object_mut(id).and_then(Object::as_dict_mut)
        {
            font.set("Encoding", Object::Dictionary(encoding));
        }
    }

    let mut bytes = Vec::with_capacity(source.len());
    let saved = document.save_to(&mut bytes).is_ok();
    Repair {
        bytes: saved.then_some(bytes),
        fonts: if saved { fonts } else { 0 },
        unmapped,
        page_count,
    }
}

/// A Type 1 font that says nothing about its own encoding: no `/Encoding`, no
/// `/ToUnicode`. Those are the only fonts this module touches — a font that
/// declares an encoding is telling the truth about itself and must be left
/// alone.
fn is_unencoded_simple_font(font: &Dictionary) -> bool {
    if font.get(b"Type").and_then(Object::as_name).ok() != Some(b"Font") {
        return false;
    }
    if !matches!(
        font.get(b"Subtype").and_then(Object::as_name).ok(),
        Some(b"Type1") | Some(b"MMType1")
    ) {
        return false;
    }
    font.get(b"Encoding").is_err() && font.get(b"ToUnicode").is_err()
}

/// The decompressed Type 1 font program of a font, when it embeds one.
fn font_program(document: &Document, font: &Dictionary) -> Option<Vec<u8>> {
    let descriptor = match font.get(b"FontDescriptor").ok()? {
        Object::Reference(id) => document.get_dictionary(*id).ok()?,
        Object::Dictionary(dictionary) => dictionary,
        _ => return None,
    };
    let stream = match descriptor.get(b"FontFile").ok()? {
        Object::Reference(id) => {
            document.get_object(*id).ok()?.as_stream().ok()?
        },
        Object::Stream(stream) => stream,
        _ => return None,
    };
    stream.decompressed_content().ok()
}

/// The `code → glyph name` pairs of a Type 1 font program's encoding vector.
///
/// Only the cleartext part is read — everything up to `eexec`, which is where
/// the vector lives and where the program stops being plain text. Entries look
/// like `dup 11 /ff put`, one per line, and anything that does not is skipped:
/// this is a scan for a known shape, not a PostScript interpreter.
fn builtin_encoding(program: &[u8]) -> Vec<(u8, String)> {
    let cleartext = program
        .windows(5)
        .position(|window| window == b"eexec")
        .map_or(program, |end| &program[..end]);
    let cleartext = String::from_utf8_lossy(cleartext);

    let mut vector = Vec::new();
    for entry in cleartext.split("dup ").skip(1) {
        let mut fields = entry.split_whitespace();
        let (Some(code), Some(glyph), Some(b"put")) = (
            fields.next(),
            fields.next(),
            fields.next().map(str::as_bytes),
        ) else {
            continue;
        };
        let Some(glyph) = glyph.strip_prefix('/') else {
            continue;
        };
        // `/.notdef` is the absence of a glyph, and a name with a delimiter in
        // it is not a name at all — either would poison the array we build.
        if glyph.is_empty() ||
            glyph == ".notdef" ||
            glyph.contains(|ch: char| {
                ch.is_ascii_whitespace() || "/[]{}<>()%".contains(ch)
            })
        {
            continue;
        }
        if let Ok(code) = code.parse::<u8>() {
            vector.push((code, glyph.to_string()));
        }
    }
    vector.sort_unstable_by_key(|&(code, _)| code);
    vector.dedup_by_key(|&mut (code, _)| code);
    vector
}

/// The encoding dictionary for a vector: `/Differences` in its run-length form,
/// where a code is written only when it does not follow the previous one.
fn differences(vector: &[(u8, String)]) -> Dictionary {
    let mut array: Vec<Object> = Vec::with_capacity(vector.len() + 8);
    let mut previous: Option<u8> = None;
    for (code, glyph) in vector {
        if previous.and_then(|p| p.checked_add(1)) != Some(*code) {
            array.push(Object::Integer(i64::from(*code)));
        }
        array.push(Object::Name(glyph.as_bytes().to_vec()));
        previous = Some(*code);
    }
    let mut encoding = Dictionary::new();
    encoding.set("Type", Object::Name(b"Encoding".to_vec()));
    encoding.set("Differences", Object::Array(array));
    encoding
}

/// Turns Adobe's corporate-use characters into the text they stand for.
///
/// The Adobe Glyph List resolves a family of presentation glyph names into
/// U+F600…U+F8FF, a stretch of the private use area Adobe reserved for itself:
/// small capitals, old-style figures, and the slices a typesetter stacks to
/// draw a brace or a radical taller than one glyph. They arrive here because
/// they are named correctly — this is the opposite of the unrecoverable
/// private-use text the conversion warns about — but nothing downstream can do
/// anything with them. So a small capital becomes its letter, an old-style
/// figure becomes its digit, and a slice of a big brace, which is a piece of a
/// drawing rather than a character, goes away.
///
/// Only the unambiguous blocks are handled. Anything else in the private use
/// area is left where it is and reported, because guessing at it would be
/// exactly the invention this whole path avoids.
pub fn normalize_presentation_forms(text: &str) -> String {
    if !text.chars().any(is_corporate_use) {
        return text.to_string();
    }
    let mut out = String::with_capacity(text.len());
    for ch in text.chars() {
        match ch as u32 {
            // zerooldstyle…nineoldstyle
            code @ 0xF730..=0xF739 => {
                out.push(char::from(b'0' + (code - 0xF730) as u8));
            },
            // Asmall…Zsmall. The glyph is a small capital; the character it
            // spells is the capital, which is what every other reader does
            // with it.
            code @ 0xF761..=0xF77A => {
                out.push(char::from(b'A' + (code - 0xF761) as u8));
            },
            0xF8E8 => out.push('®'),
            0xF8E9 => out.push('©'),
            0xF8EA => out.push('™'),
            // radicalex, the arrow and delimiter extensions, and the pieces of
            // a multi-glyph brace, bracket and parenthesis — drawing, not text.
            0xF8E5..=0xF8E7 | 0xF8EB..=0xF8FF => {},
            _ => out.push(ch),
        }
    }
    out
}

/// Whether a character falls in the stretch of the private use area the Adobe
/// Glyph List assigns names in.
fn is_corporate_use(ch: char) -> bool { matches!(ch as u32, 0xF600..=0xF8FF) }

/// The spacing diacritics TeX draws ahead of the letter they belong to, paired
/// with the combining mark that says the same thing.
///
/// Deliberately without the ASCII lookalikes — `` ` ``, `^`, `~`, `"` — which
/// carry accents in one TeX font encoding and are ordinary punctuation in
/// every other context. Composing those would corrupt code, paths and shell
/// snippets to fix an accent.
const SPACING_DIACRITICS: &[(char, char)] = &[
    ('\u{00A8}', '\u{0308}'), // dieresis
    ('\u{00AF}', '\u{0304}'), // macron
    ('\u{00B4}', '\u{0301}'), // acute
    ('\u{00B8}', '\u{0327}'), // cedilla
    ('\u{02C6}', '\u{0302}'), // modifier circumflex
    ('\u{02C7}', '\u{030C}'), // caron
    ('\u{02D8}', '\u{0306}'), // breve
    ('\u{02D9}', '\u{0307}'), // dot above
    ('\u{02DA}', '\u{030A}'), // ring above
    ('\u{02DB}', '\u{0328}'), // ogonek
    ('\u{02DC}', '\u{0303}'), // small tilde
    ('\u{02DD}', '\u{030B}'), // double acute
];

/// Joins a free-standing accent to the letter it was drawn in front of.
///
/// `¯` + `a` becomes `ā`. Only the accents in [`SPACING_DIACRITICS`] are
/// joined, and only to a letter directly after them: a diacritic before a
/// space, a digit or another accent is left exactly where it is, because then
/// it is a character in its own right rather than a misplaced mark.
pub fn compose_diacritics(text: &str) -> String {
    if !text.chars().any(|ch| mark_for(ch).is_some()) {
        return text.to_string();
    }
    let mut out = String::with_capacity(text.len());
    let mut chars = text.chars().peekable();
    while let Some(ch) = chars.next() {
        match (mark_for(ch), chars.peek()) {
            (Some(mark), Some(&next)) if next.is_alphabetic() => {
                chars.next();
                out.extend(compose(next, mark));
            },
            _ => out.push(ch),
        }
    }
    out
}

/// The combining mark equivalent to a spacing diacritic, if it is one.
fn mark_for(ch: char) -> Option<char> {
    SPACING_DIACRITICS
        .iter()
        .find(|&&(spacing, _)| spacing == ch)
        .map(|&(_, mark)| mark)
}

/// `base` with `mark` on it, as one character where Unicode has one and as the
/// two-character sequence where it does not.
///
/// A dotless `ı` or `ȷ` gets its dot back first: those letters exist so that an
/// accent can be set over them, so an accent is exactly the context in which
/// they mean the ordinary letter.
fn compose(base: char, mark: char) -> impl Iterator<Item = char> {
    let base = match base {
        'ı' => 'i',
        'ȷ' => 'j',
        other => other,
    };
    precomposed(base, mark)
        .map(|composed| vec![composed])
        .unwrap_or_else(|| vec![base, mark])
        .into_iter()
}

/// The single code point for `base` + `mark`, when Unicode assigns one.
///
/// The table covers the Latin letters a canonical composition would reach; the
/// point of listing them is that no full normalizer is pulled in for a dozen
/// pairs, and that a pair with no precomposed form falls back to the sequence
/// rather than to nothing.
fn precomposed(base: char, mark: char) -> Option<char> {
    const COMPOSED: &[(char, char, char)] = &[
        ('a', '\u{0304}', 'ā'),
        ('e', '\u{0304}', 'ē'),
        ('i', '\u{0304}', 'ī'),
        ('o', '\u{0304}', 'ō'),
        ('u', '\u{0304}', 'ū'),
        ('A', '\u{0304}', 'Ā'),
        ('E', '\u{0304}', 'Ē'),
        ('I', '\u{0304}', 'Ī'),
        ('O', '\u{0304}', 'Ō'),
        ('U', '\u{0304}', 'Ū'),
        ('a', '\u{0301}', 'á'),
        ('e', '\u{0301}', 'é'),
        ('i', '\u{0301}', 'í'),
        ('o', '\u{0301}', 'ó'),
        ('u', '\u{0301}', 'ú'),
        ('y', '\u{0301}', 'ý'),
        ('n', '\u{0301}', 'ń'),
        ('s', '\u{0301}', 'ś'),
        ('c', '\u{0301}', 'ć'),
        ('z', '\u{0301}', 'ź'),
        ('A', '\u{0301}', 'Á'),
        ('E', '\u{0301}', 'É'),
        ('I', '\u{0301}', 'Í'),
        ('O', '\u{0301}', 'Ó'),
        ('U', '\u{0301}', 'Ú'),
        ('a', '\u{0308}', 'ä'),
        ('e', '\u{0308}', 'ë'),
        ('i', '\u{0308}', 'ï'),
        ('o', '\u{0308}', 'ö'),
        ('u', '\u{0308}', 'ü'),
        ('y', '\u{0308}', 'ÿ'),
        ('A', '\u{0308}', 'Ä'),
        ('E', '\u{0308}', 'Ë'),
        ('I', '\u{0308}', 'Ï'),
        ('O', '\u{0308}', 'Ö'),
        ('U', '\u{0308}', 'Ü'),
        ('a', '\u{0302}', 'â'),
        ('e', '\u{0302}', 'ê'),
        ('i', '\u{0302}', 'î'),
        ('o', '\u{0302}', 'ô'),
        ('u', '\u{0302}', 'û'),
        ('A', '\u{0302}', 'Â'),
        ('E', '\u{0302}', 'Ê'),
        ('I', '\u{0302}', 'Î'),
        ('O', '\u{0302}', 'Ô'),
        ('U', '\u{0302}', 'Û'),
        ('a', '\u{0303}', 'ã'),
        ('n', '\u{0303}', 'ñ'),
        ('o', '\u{0303}', 'õ'),
        ('A', '\u{0303}', 'Ã'),
        ('N', '\u{0303}', 'Ñ'),
        ('O', '\u{0303}', 'Õ'),
        ('a', '\u{030C}', 'ǎ'),
        ('c', '\u{030C}', 'č'),
        ('i', '\u{030C}', 'ǐ'),
        ('o', '\u{030C}', 'ǒ'),
        ('s', '\u{030C}', 'š'),
        ('u', '\u{030C}', 'ǔ'),
        ('z', '\u{030C}', 'ž'),
        ('r', '\u{030C}', 'ř'),
        ('e', '\u{030C}', 'ě'),
        ('C', '\u{030C}', 'Č'),
        ('S', '\u{030C}', 'Š'),
        ('Z', '\u{030C}', 'Ž'),
        ('a', '\u{0306}', 'ă'),
        ('g', '\u{0306}', 'ğ'),
        ('u', '\u{0306}', 'ŭ'),
        ('c', '\u{0327}', 'ç'),
        ('s', '\u{0327}', 'ş'),
        ('t', '\u{0327}', 'ţ'),
        ('C', '\u{0327}', 'Ç'),
        ('a', '\u{030A}', 'å'),
        ('u', '\u{030A}', 'ů'),
        ('A', '\u{030A}', 'Å'),
        ('z', '\u{0307}', 'ż'),
        ('e', '\u{0307}', 'ė'),
        ('a', '\u{0328}', 'ą'),
        ('e', '\u{0328}', 'ę'),
        ('o', '\u{030B}', 'ő'),
        ('u', '\u{030B}', 'ű'),
    ];
    COMPOSED
        .iter()
        .find(|&&(letter, combining, _)| letter == base && combining == mark)
        .map(|&(_, _, composed)| composed)
}

#[cfg(test)]
mod tests;
