//! `convert pdf`: the text layer of a PDF, as Markdown.
//!
//! The heavy lifting — parsing, classifying and laying the text out — belongs
//! to the `pdf-inspector` crate. What lives here is the part that makes its
//! answer trustworthy:
//!
//! * the [encoding repair](encoding) applied to the document before it is read;
//! * a per-page decision about what actually converted, taking the engine's
//!   "there is nothing to read here" verdicts and leaving its "this text looks
//!   wrong" one alone, because that one was measured and misses more than half
//!   of what it should catch while flagging clean pages;
//! * two checks of trakktor's own — text that maps into a private use area, and
//!   a page carrying nothing but what is stamped across the whole document.

pub mod encoding;
pub mod pages;

use std::{collections::BTreeMap, path::Path};

pub use pages::Selection;
use pdf_inspector::{
    DetectionConfig, MarkdownOptions, PdfError, PdfOptions, PdfType,
    ProcessMode, ScanStrategy,
};

use crate::convert::ConvertError;

/// What the engine made of the document as a whole.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Kind {
    /// Text throughout: a document made from a layout program.
    Text,
    /// Some pages carry text, some do not.
    Mixed,
    /// Scanned pages: pictures of a document.
    Scanned,
    /// Images that are not scans of text.
    Image,
}

impl Kind {
    /// The word that goes into the output.
    #[must_use]
    pub fn as_str(self) -> &'static str {
        match self {
            Kind::Text => "text",
            Kind::Mixed => "mixed",
            Kind::Scanned => "scanned",
            Kind::Image => "image",
        }
    }
}

/// Why a page came back without text.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Reason {
    /// The page is a picture of a document.
    Scanned,
    /// The page holds no text at all.
    NoText,
    /// The text is drawn as vector outlines: letters on the page, no
    /// characters in the file.
    VectorText,
    /// Everything on the page is stamped across the document as well — a
    /// watermark, a running head — and there is nothing else.
    NoUniqueText,
}

impl Reason {
    /// The stable string that goes into the output.
    #[must_use]
    pub fn as_str(self) -> &'static str {
        match self {
            Reason::Scanned => "scanned",
            Reason::NoText => "no_text",
            Reason::VectorText => "vector_text",
            Reason::NoUniqueText => "no_unique_text",
        }
    }

    /// How to say it in a sentence, in the plural.
    fn plural(self) -> &'static str {
        match self {
            Reason::Scanned => "scanned images",
            Reason::NoText => "empty of text",
            Reason::VectorText => "text drawn as vector outlines",
            Reason::NoUniqueText => "blank but for a watermark",
        }
    }
}

/// A page of the result: converted, or not and why.
#[derive(Debug, Clone)]
pub struct Page {
    /// The page's own number in the document, counting from one.
    pub number: u32,
    /// The Markdown of the page, when it converted.
    pub markdown: Option<String>,
    /// Why it did not, when it did not.
    pub reason: Option<Reason>,
}

/// Something wrong with text that *did* convert — a warning, not a refusal.
#[derive(Debug, Clone)]
pub struct Issue {
    /// A stable identifier for the kind of damage.
    pub code: &'static str,
    /// The pages it was found on. Empty when it is a property of the document.
    pub pages: Vec<u32>,
    /// What it means, and what can be done about it.
    pub message: String,
}

/// The converted document.
#[derive(Debug, Clone)]
pub struct Converted {
    /// How the engine classified the document.
    pub kind: Kind,
    /// Pages in the document — not pages in this result, which `--pages` may
    /// have narrowed.
    pub page_count: u32,
    /// The pages asked for, in order.
    pub pages: Vec<Page>,
    /// Warnings about the text that did convert.
    pub issues: Vec<Issue>,
    /// How many fonts had their encoding repaired before reading.
    pub fonts_repaired: usize,
}

impl Converted {
    /// The pages that need recognizing instead of reading.
    #[must_use]
    pub fn pages_needing_ocr(&self) -> Vec<u32> {
        self.pages
            .iter()
            .filter(|page| page.markdown.is_none())
            .map(|page| page.number)
            .collect()
    }

    /// The whole document as one Markdown string.
    ///
    /// With more than one page, each page opens with a `<!-- page N -->`
    /// comment, as `ocr --format md` does. A page that did not convert leaves a
    /// comment of its own rather than a silent gap: a hole in the document has
    /// to be visible to whoever reads the source, and invisible in the rendered
    /// text.
    #[must_use]
    pub fn markdown(&self) -> String {
        let separators = self.pages.len() > 1;
        let mut out = String::new();
        for page in &self.pages {
            if !out.is_empty() {
                out.push_str("\n\n");
            }
            match (&page.markdown, page.reason) {
                (Some(markdown), _) => {
                    if separators {
                        out.push_str(&format!(
                            "<!-- page {} -->\n\n",
                            page.number
                        ));
                    }
                    out.push_str(markdown.trim_end());
                },
                // No separator ahead of this one: the comment names the page
                // itself, and two comments in a row would say it twice.
                (None, reason) => {
                    let reason = reason.unwrap_or(Reason::NoText);
                    out.push_str(&format!(
                        "<!-- page {}: no text layer ({}) — read this page \
                         with `trakktor ocr` -->",
                        page.number,
                        reason.as_str()
                    ));
                },
            }
        }
        if !out.is_empty() {
            out.push('\n');
        }
        out
    }
}

/// How to convert.
#[derive(Debug, Clone, Default)]
pub struct Options {
    /// Which pages to take. `None` takes them all.
    pub pages: Option<Selection>,
    /// The password of an encrypted document.
    pub password: Option<String>,
}

/// Converts the text layer of a PDF to Markdown.
///
/// # Errors
///
/// Returns a [`ConvertError`] when the file cannot be read or is not a PDF
/// (`invalid_input`), when its structure does not parse (`parse_failed`), when
/// it is encrypted and the password is missing or wrong (`encrypted`), when
/// `--pages` does not fit the document (`invalid_options`), or when not one
/// page carries text to convert (`no_text_layer`).
pub fn convert(
    path: &Path,
    options: &Options,
) -> Result<Converted, ConvertError> {
    let source = std::fs::read(path).map_err(|source| {
        ConvertError::InvalidInput(format!(
            "cannot read {}: {source}",
            path.display()
        ))
    })?;
    if !source.starts_with(b"%PDF-") {
        return Err(ConvertError::InvalidInput(format!(
            "{} is not a PDF",
            path.display()
        )));
    }

    let repair = encoding::repair(&source);
    let page_count = repair.page_count.filter(|&count| count > 0);
    let selected = match (&options.pages, page_count) {
        (None, _) => None,
        (Some(selection), Some(count)) => Some(selection.resolve(count)?),
        // The document did not parse here, so there is no length to resolve a
        // range against. The engine gets its turn and reports the real reason
        // — encrypted, or broken — which beats guessing at the page count.
        (Some(_), None) => None,
    };

    let result = read(
        &source,
        repair.bytes.as_deref(),
        selected.as_deref(),
        options,
    )?;

    let page_count = result.page_count.max(page_count.unwrap_or(0));
    let requested: Vec<u32> = selected
        .clone()
        .unwrap_or_else(|| (1..=page_count).collect());

    // Two passes over the text before it is anyone's answer. The first is only
    // owed by a document whose encoding we put back — that is where a
    // free-standing accent comes from. The second is owed by any document: the
    // presentation glyph names it normalizes are resolved the same way whether
    // the map came from us or from the file.
    let markdown = result.markdown.unwrap_or_default();
    let markdown = if repair.fonts > 0 {
        encoding::compose_diacritics(&markdown)
    } else {
        markdown
    };
    let markdown = encoding::normalize_presentation_forms(&markdown);
    let mut converted = pages::split_pages(&markdown, &requested);
    let furniture = pages::furniture_pages(&converted);

    let refused = refusals(&result.ocr_reasons_by_page);
    let mut pages = Vec::with_capacity(requested.len());
    for number in requested {
        let reason = refused.get(&number).copied().or_else(|| {
            furniture.contains(&number).then_some(Reason::NoUniqueText)
        });
        let markdown = match reason {
            Some(_) => {
                converted.remove(&number);
                None
            },
            None => converted.remove(&number),
        };
        let reason = match (&markdown, reason) {
            (None, None) => Some(Reason::NoText),
            (_, reason) => reason,
        };
        pages.push(Page {
            number,
            markdown,
            reason,
        });
    }

    if pages.iter().all(|page| page.markdown.is_none()) {
        // A PDF locked with an owner password opens without one and reads as an
        // empty scan, so "nothing to convert" from an encrypted document is a
        // password problem, not a scanner.
        if is_encrypted(&source) {
            return Err(ConvertError::Encrypted);
        }
        return Err(ConvertError::NoTextLayer {
            what: dominant_reason(&pages).plural().to_string(),
        });
    }

    Ok(Converted {
        kind: kind(result.pdf_type),
        page_count,
        issues: issues(&pages, repair.unmapped),
        pages,
        fonts_repaired: repair.fonts,
    })
}

/// Runs the engine, once on the repaired document and, if that fails, once more
/// on the bytes as they came.
///
/// The repair rewrites the whole file, and a rewrite can go wrong in ways the
/// original would not; falling back means the worst it can do is cost a second
/// parse. A document that needed no repair takes this path once.
fn read(
    source: &[u8],
    repaired: Option<&[u8]>,
    selected: Option<&[u32]>,
    options: &Options,
) -> Result<pdf_inspector::PdfProcessResult, ConvertError> {
    if let Some(repaired) = repaired &&
        let Ok(result) = pdf_inspector::process_pdf_mem_with_options(
            repaired,
            engine_options(selected, options),
        )
    {
        return Ok(result);
    }
    pdf_inspector::process_pdf_mem_with_options(
        source,
        engine_options(selected, options),
    )
    .map_err(translate)
}

/// The engine's options for one run.
///
/// Two of them are not defaults and both matter. Detection walks **every**
/// page: sampling eight of them is how a document with a scanned insert comes
/// back "text" and the insert comes back converted-and-empty. Page markers are
/// switched on because they are the only way the one Markdown string can be cut
/// back into pages — the crate's own per-page entry point skips detection
/// entirely and calls a page of vector outlines fine.
fn engine_options(selected: Option<&[u32]>, options: &Options) -> PdfOptions {
    let markdown = MarkdownOptions {
        include_page_numbers: true,
        ..MarkdownOptions::default()
    };
    let strategy = match selected {
        Some(pages) => ScanStrategy::Pages(pages.to_vec()),
        None => ScanStrategy::Full,
    };
    let mut engine = PdfOptions::new()
        .mode(ProcessMode::Full)
        .markdown(markdown)
        .detection(DetectionConfig {
            strategy,
            ..DetectionConfig::default()
        });
    if let Some(pages) = selected {
        engine = engine.pages(pages.iter().copied());
    }
    if let Some(password) = &options.password {
        engine = engine.password(password.clone());
    }
    engine
}

/// The engine's per-page verdicts, reduced to the ones that mean "there is
/// nothing here to read".
///
/// `suspected_garbled_text` is deliberately absent. Measured on a bilingual
/// book whose second script is unrecoverable, it named 124 of the 267 damaged
/// pages and two clean ones — and the price of believing it is a page of
/// perfectly good text thrown away. Damaged text is reported as an issue
/// instead, from a signal of our own.
fn refusals(
    reasons: &[pdf_inspector::PageOcrReasons],
) -> BTreeMap<u32, Reason> {
    let mut refused = BTreeMap::new();
    for page in reasons {
        let reason =
            page.reasons
                .iter()
                .find_map(|reason| match reason.as_str() {
                    pdf_inspector::OCR_REASON_SCANNED => Some(Reason::Scanned),
                    pdf_inspector::OCR_REASON_VECTOR_TEXT => {
                        Some(Reason::VectorText)
                    },
                    pdf_inspector::OCR_REASON_NO_TEXT => Some(Reason::NoText),
                    _ => None,
                });
        if let Some(reason) = reason {
            refused.insert(page.page, reason);
        }
    }
    refused
}

/// The reason to name when the whole document has to go to OCR: the one that
/// accounts for most of its pages.
fn dominant_reason(pages: &[Page]) -> Reason {
    let mut counts: BTreeMap<&'static str, (usize, Reason)> = BTreeMap::new();
    for page in pages {
        let reason = page.reason.unwrap_or(Reason::NoText);
        let entry = counts.entry(reason.as_str()).or_insert((0, reason));
        entry.0 += 1;
    }
    counts
        .into_values()
        .max_by_key(|&(count, _)| count)
        .map_or(Reason::NoText, |(_, reason)| reason)
}

/// Warnings about the text that did convert.
fn issues(pages: &[Page], unmapped_fonts: bool) -> Vec<Issue> {
    let mut issues = Vec::new();

    let private_use: Vec<u32> = pages
        .iter()
        .filter(|page| {
            page.markdown
                .as_deref()
                .is_some_and(|text| pages::private_use_chars(text) > 0)
        })
        .map(|page| page.number)
        .collect();
    if !private_use.is_empty() {
        issues.push(Issue {
            code: "private_use_text",
            pages: private_use,
            message: "some characters on these pages map into a private use \
                      area of Unicode: the font says which glyph to draw but \
                      not which character it is, so that text is lost and \
                      cannot be recovered from the file. Recognize those \
                      pages with `trakktor ocr` if you need them"
                .to_string(),
        });
    }

    if unmapped_fonts {
        issues.push(Issue {
            code: "unmapped_font_encoding",
            pages: Vec::new(),
            message: "this document has a font that declares no encoding and \
                      embeds no program to recover one from, so some \
                      characters may come out wrong or missing"
                .to_string(),
        });
    }

    issues
}

/// Whether the file declares itself encrypted.
///
/// A scan of the raw bytes rather than a parse: by the time this is asked the
/// document has already been through the engine, and the question is only
/// which of two messages to print.
fn is_encrypted(source: &[u8]) -> bool {
    source.windows(8).any(|window| window == b"/Encrypt")
}

/// The engine's classification, in our words.
fn kind(pdf_type: PdfType) -> Kind {
    match pdf_type {
        PdfType::TextBased => Kind::Text,
        PdfType::Mixed => Kind::Mixed,
        PdfType::Scanned => Kind::Scanned,
        PdfType::ImageBased => Kind::Image,
    }
}

/// The engine's error, in our contract.
fn translate(error: PdfError) -> ConvertError {
    match error {
        PdfError::Encrypted => ConvertError::Encrypted,
        PdfError::NotAPdf(what) => ConvertError::InvalidInput(what),
        PdfError::Io(source) => ConvertError::InvalidInput(format!(
            "cannot read the file: {source}"
        )),
        PdfError::Parse(what) => ConvertError::ParseFailed(what),
        PdfError::InvalidStructure => {
            ConvertError::ParseFailed("invalid PDF structure".to_string())
        },
    }
}

#[cfg(test)]
mod tests;
