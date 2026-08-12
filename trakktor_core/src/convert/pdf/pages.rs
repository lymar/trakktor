//! Page bookkeeping for `convert pdf`: which pages were asked for, how the
//! engine's one Markdown string is cut back into pages, and the two checks
//! trakktor makes on a page that the engine does not make itself.

use std::collections::{BTreeMap, BTreeSet, HashMap};

pub use crate::pages::Selection;

/// The marker the engine writes ahead of each page when page numbering is on.
/// Splitting on it is how per-page Markdown is recovered from the one string
/// the engine returns — an undocumented contract, so it is pinned by a test.
const PAGE_MARKER_PREFIX: &str = "<!-- Page ";

/// Longest a page's text may be and still count as document furniture rather
/// than content. A watermark or a running head is a handful of words; a page of
/// prose is not.
const FURNITURE_MAX_CHARS: usize = 200;

/// How many pages must carry exactly the same text before it counts as
/// furniture. Two identical pages are a coincidence worth keeping; five are a
/// stamp applied to the whole document.
const FURNITURE_MIN_PAGES: usize = 5;

/// Cuts the engine's single Markdown string back into pages along its page
/// markers.
///
/// `expected` is the pages that were asked for, in order. It settles two cases
/// the markers cannot: a run of text before the first marker (it belongs to the
/// first page asked for) and a document that emitted no markers at all (all of
/// it is the one page asked for). A page the engine wrote nothing for is
/// absent from the result rather than present and empty.
pub fn split_pages(markdown: &str, expected: &[u32]) -> BTreeMap<u32, String> {
    let mut pages: BTreeMap<u32, String> = BTreeMap::new();
    let mut current = expected.first().copied();
    let mut buffer = String::new();
    let mut seen_marker = false;

    for line in markdown.lines() {
        if let Some(number) = marker_page(line) {
            flush(&mut pages, current.take(), &mut buffer);
            current = Some(number);
            seen_marker = true;
            continue;
        }
        buffer.push_str(line);
        buffer.push('\n');
    }
    flush(&mut pages, current, &mut buffer);

    // Without a single marker the engine cannot be telling us about more than
    // one page; attributing its output to several would be an invention.
    if !seen_marker && expected.len() > 1 {
        pages.clear();
    }
    pages
}

/// The page number of a marker line, when the line is one.
fn marker_page(line: &str) -> Option<u32> {
    let rest = line.trim().strip_prefix(PAGE_MARKER_PREFIX)?;
    rest.strip_suffix("-->")?.trim().parse().ok()
}

/// Files the buffered lines under `number`, dropping them when they are blank
/// or when there is no page to file them under.
fn flush(
    pages: &mut BTreeMap<u32, String>,
    number: Option<u32>,
    buffer: &mut String,
) {
    let text = std::mem::take(buffer);
    let text = text.trim();
    if text.is_empty() {
        return;
    }
    let Some(number) = number else { return };
    let entry = pages.entry(number).or_default();
    if !entry.is_empty() {
        entry.push_str("\n\n");
    }
    entry.push_str(text);
}

/// How many characters of `text` land in a Unicode private use area.
///
/// A font may carry a `/ToUnicode` map that points there instead of at real
/// characters, and then the text comes out confident, non-empty and
/// unrecoverable — the glyph is on the page, but nothing in the file says which
/// character it is. Counting is the whole check: one such character is already
/// a character the reader will not get.
pub fn private_use_chars(text: &str) -> usize {
    text.chars()
        .filter(|&ch| {
            matches!(ch as u32,
                0xE000..=0xF8FF | 0xF_0000..=0xF_FFFD | 0x10_0000..=0x10_FFFD)
        })
        .count()
}

/// The pages whose entire text is something stamped across the document.
///
/// A page drawn as vector outlines still carries whatever *is* real text on it
/// — a watermark, a running head — so it converts to a confident, non-empty,
/// completely useless page. The tell is not the page but the document: the same
/// few words, alone, on page after page. No threshold is tuned to a file; the
/// rule is "identical to several other pages, and short".
pub fn furniture_pages(pages: &BTreeMap<u32, String>) -> BTreeSet<u32> {
    let mut by_text: HashMap<String, Vec<u32>> = HashMap::new();
    for (&number, text) in pages {
        let normalized = collapse(text);
        if normalized.is_empty() ||
            normalized.chars().count() > FURNITURE_MAX_CHARS
        {
            continue;
        }
        by_text.entry(normalized).or_default().push(number);
    }
    by_text
        .into_values()
        .filter(|numbers| numbers.len() >= FURNITURE_MIN_PAGES)
        .flatten()
        .collect()
}

/// The text with every run of whitespace reduced to one space, so that two
/// pages differing only in line breaks compare equal.
fn collapse(text: &str) -> String {
    text.split_whitespace().collect::<Vec<_>>().join(" ")
}

#[cfg(test)]
mod tests;
