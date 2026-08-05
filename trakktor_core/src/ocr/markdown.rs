//! Turning an analysed document into Markdown.
//!
//! The reading order and the kind of every block arrive already decided (see
//! [`crate::ocr::layout`]); what is left is the text. That is less mechanical
//! than it sounds, because a page arrives as one string per detected row: the
//! paragraph a person reads was cut into lines by whoever set the page, words
//! were split across the cut with a hyphen, and a sentence that runs over a
//! page break leaves no trace in the line list at all.
//!
//! Three rules do that work, and two of them are deliberately not the shape a
//! quick implementation takes:
//!
//! - **An end-of-line hyphen is dropped only between letters of a script that
//!   hyphenates.** Dropping it unconditionally is the tempting version and it
//!   is wrong: a line that ends on a dash between two numbers is a range, not a
//!   broken word, and closing it up turns `358-359` into `358359`. So the
//!   hyphen survives whenever either side of it is not a letter, and a dash
//!   between digits keeps its place without a space around it.
//! - **A word space is inserted for any Unicode letter, not only `[A-Za-z]`.**
//!   Testing the ASCII range is the usual shortcut, and it silently welds
//!   together every pair of lines in every other alphabet — Cyrillic, Greek,
//!   Armenian — because none of their letters match it. The test here is
//!   [`char::is_alphabetic`], so Russian prose keeps its spaces.
//! - **No space is inserted where a script does not use one.** Chinese,
//!   Japanese, Tibetan, Thai, Khmer, Lao and Burmese write without word spaces,
//!   so lines in those scripts are concatenated directly; a space there is as
//!   wrong as its absence is in Latin.
//!
//! Everything else follows the page. Headings become `#` lines at the level
//! the layout gave them; footnotes are emitted after the body of the page
//! they belong to, as Markdown footnote definitions when the recognizer read
//! a leading marker; running heads, running feet and page numbers are
//! dropped; and a paragraph that runs over a page break is joined to its
//! beginning instead of starting a second one — detected geometrically, from
//! a last line that reached the right margin followed by a first line that is
//! not indented.

pub mod table;
#[cfg(test)]
mod tests;

use std::collections::HashSet;

use crate::ocr::{
    layout::{Block, BlockKind, Layout},
    page::Page,
};

/// How far short of its own right margin the last line of a paragraph may
/// stop and still count as having reached it, as a fraction of the
/// paragraph's width. A paragraph that continues fills its last line to
/// within a hair of the margin; a paragraph that is finished leaves a ragged
/// tail far longer than this.
const MARGIN_TOLERANCE: f32 = 0.12;

/// How far right of its left margin the first line of a paragraph may start
/// before it counts as indented, in multiples of the page's median line
/// height. An indent worth the name is about one em quad wide, which is close
/// to the line height; anything smaller is the wobble of the detector.
const INDENT_TOLERANCE: f32 = 0.9;

/// What the rendered document carries beyond the text itself.
#[derive(Debug, Clone, Default)]
pub struct Options {
    /// Mark where every page after the first begins. The marker is an HTML
    /// comment, so it is a note to whoever reads the Markdown source and
    /// disappears when the document is rendered — which also lets it sit
    /// inside a paragraph that runs across the break without cutting the
    /// paragraph in two.
    pub page_separators: bool,
    /// The directory this run's illustration crops were written to, spelled
    /// as it should appear in a link (`Some("figures")` yields `figures/…`).
    /// It prefixes the reference of every illustration that reaches the
    /// renderer as a block of its own; a document of nothing but text never
    /// mentions it.
    pub image_dir: Option<String>,
}

/// Renders the analysed pages as one Markdown document.
///
/// Blocks come out in the order the layout put them in, separated by a blank
/// line, with the footnotes of a page after that page's body. The result ends
/// with a single newline, or is empty when the pages hold no text.
pub fn render(pages: &[(Page, Layout)], options: &Options) -> String {
    let mut document = Document::new(options);
    for (index, (page, layout)) in pages.iter().enumerate() {
        document.page(page, layout, index == 0);
    }
    document.finish()
}

/// The file name of one illustration crop: the page it was found on and its
/// position among that page's figures.
///
/// Both the writer of the file and the link in the Markdown come from here, so
/// there is one place to change and no way for them to disagree.
pub fn figure_name(page: usize, index: usize) -> String {
    format!("p{page:03}-f{index:02}.png")
}

/// Renders the pages as plain text: every recognized line on a line of its
/// own, in reading order, with a marker between pages naming the page and the
/// file it was read from. A single page is emitted on its own, without a
/// marker.
pub fn plain(pages: &[Page]) -> String {
    let mut out = String::new();
    for (index, page) in pages.iter().enumerate() {
        if index > 0 {
            out.push('\n');
            out.push_str(&plain_marker(page));
            out.push_str("\n\n");
        }
        for line in &page.lines {
            out.push_str(&line.text);
            out.push('\n');
        }
    }
    out
}

/// The document under construction: the chunks that will be joined by blank
/// lines, plus what carrying a paragraph over a page break needs.
struct Document<'a> {
    options: &'a Options,
    chunks: Vec<String>,
    /// The paragraph a following page may still be joined to.
    open: Option<Open>,
    /// The footnote labels already defined, so that two pages numbering their
    /// notes from `1` do not define one label twice.
    labels: HashSet<String>,
}

/// A paragraph already written out that a later page can still continue.
#[derive(Debug, Clone, Copy)]
struct Open {
    /// Which chunk holds it.
    chunk: usize,
    /// Whether its last line reached the right margin, i.e. whether the
    /// paragraph looks unfinished.
    reaches_right: bool,
}

impl<'a> Document<'a> {
    fn new(options: &'a Options) -> Self {
        Self {
            options,
            chunks: Vec::new(),
            open: None,
            labels: HashSet::new(),
        }
    }

    /// Appends one analysed page.
    fn page(&mut self, page: &Page, layout: &Layout, first: bool) {
        let height = median_height(page);
        // Held back until the page writes something, so that a page of pure
        // furniture does not strand its marker above the next page's; the
        // first chunk written takes it.
        let mut separator =
            (self.options.page_separators && !first).then(|| page_marker(page));
        let mut notes = Vec::new();
        let mut ruled = false;
        // Only the first block of a page's body may continue the paragraph
        // the previous page left open; inside a page the layout has already
        // decided where the paragraphs end.
        let mut opening = true;

        for block in &layout.blocks {
            // An illustration is the one block with nothing to read in it, so
            // it is emitted before the empty-text guard rather than after.
            if let BlockKind::Figure { index } = block.kind {
                if let Some(link) = self.figure(page.number, index) {
                    self.push(&mut separator, link);
                    self.open = None;
                    opening = false;
                }
                continue;
            }
            let text = block_text(page, block);
            if text.is_empty() {
                continue;
            }
            // A table the model answered with cell markup is a table, whatever
            // the analysis made of the block it sits in: the markup *is* the
            // structure, and it is the one thing on the page that Markdown can
            // represent better than the text it came from.
            if table::is_markup(&text) {
                if let Some(rendered) = table::to_markdown(&text) {
                    self.push(&mut separator, rendered);
                    self.open = None;
                    opening = false;
                    continue;
                }
            }
            match block.kind {
                BlockKind::Figure { .. } => {},
                BlockKind::PageFurniture => {},
                BlockKind::Footnote => {
                    let split = split_marker(&text)
                        .map(|(label, note)| (label, note.to_string()));
                    match split {
                        Some((marker, note)) => {
                            let label = self.label(&marker, page.number);
                            notes.push(format!("[^{label}]: {note}"));
                        },
                        None => {
                            if !ruled {
                                notes.push("---".to_string());
                                ruled = true;
                            }
                            notes.push(text);
                        },
                    }
                },
                BlockKind::Heading { level } => {
                    self.push(&mut separator, heading(level, &one_line(&text)));
                    self.open = None;
                    opening = false;
                },
                BlockKind::Caption => {
                    let text = one_line(&text);
                    self.push(&mut separator, format!("*{text}*"));
                    self.open = None;
                    opening = false;
                },
                BlockKind::Paragraph => {
                    let span = span(page, block);
                    let reaches = span.is_some_and(|it| it.reaches_right());
                    let flush = span.is_some_and(|it| it.flush_left(height));
                    let carry = self
                        .open
                        .filter(|open| opening && flush && open.reaches_right);
                    match carry {
                        Some(open) => {
                            let marker = separator.take();
                            let chunk = &mut self.chunks[open.chunk];
                            append(chunk, marker.as_deref(), &text);
                            self.open = Some(Open {
                                reaches_right: reaches,
                                ..open
                            });
                        },
                        None => {
                            self.push(&mut separator, text);
                            self.open = Some(Open {
                                chunk: self.chunks.len() - 1,
                                reaches_right: reaches,
                            });
                        },
                    }
                    opening = false;
                },
            }
        }

        // The notes go after the page's body but leave the open paragraph
        // alone: they are apparatus, and the paragraph above them may still
        // be continued by the next page.
        for note in notes {
            self.push(&mut separator, note);
        }
        if let Some(separator) = separator.take() {
            self.chunks.push(separator);
        }
    }

    /// Writes one chunk, preceded by the page separator when one is due.
    fn push(&mut self, separator: &mut Option<String>, chunk: String) {
        if let Some(separator) = separator.take() {
            self.chunks.push(separator);
        }
        self.chunks.push(chunk);
    }

    /// The Markdown for one illustration, or nothing when this run was not
    /// told where the crops were written.
    ///
    /// The name is the page and the figure's position on it, which is what the
    /// caller writes the file under; keeping the two in one place is the whole
    /// of the contract between them.
    fn figure(&self, page: usize, index: usize) -> Option<String> {
        let dir = self.options.image_dir.as_ref()?;
        Some(format!("![]({dir}/{})", figure_name(page, index)))
    }

    /// Reserves a footnote label, qualifying it with the page number when the
    /// document has defined it already.
    fn label(&mut self, marker: &str, page: usize) -> String {
        if self.labels.insert(marker.to_string()) {
            return marker.to_string();
        }
        let mut label = format!("{marker}-{page}");
        let mut copy = 2;
        while !self.labels.insert(label.clone()) {
            label = format!("{marker}-{page}-{copy}");
            copy += 1;
        }
        label
    }

    fn finish(self) -> String {
        let mut out = self.chunks.join("\n\n");
        if !out.is_empty() {
            out.push('\n');
        }
        out
    }
}

/// Flattens a block that must occupy one line.
///
/// A heading and a caption are single-line constructs in Markdown: a hard break
/// inside either is not a break, it is a stray backslash. They are the only two
/// places where the reader's own line structure has to give way to the
/// format's.
fn one_line(text: &str) -> String { text.replace(HARD_BREAK, " ") }

/// A heading line at the level the layout assigned. Markdown has six levels;
/// a deeper one is written at the deepest that exists rather than dropped.
fn heading(level: u8, text: &str) -> String {
    let depth = usize::from(level.clamp(1, 6));
    format!("{} {text}", "#".repeat(depth))
}

/// The page separator: a comment, so it does not show in a rendered document
/// and may sit inside a paragraph that spans the break.
fn page_marker(page: &Page) -> String {
    if page.source.is_empty() {
        format!("<!-- page {} -->", page.number)
    } else {
        format!("<!-- page {} · {} -->", page.number, page.source)
    }
}

/// The between-pages marker of the plain-text rendering.
fn plain_marker(page: &Page) -> String {
    if page.source.is_empty() {
        format!("=== page {} ===", page.number)
    } else {
        format!("=== page {} · {} ===", page.number, page.source)
    }
}

/// The hard line break: a backslash at the end of a line.
///
/// The other spelling — two trailing spaces — is invisible in the source and
/// is stripped by half the tools that touch a file, so a break written that way
/// is a break that quietly stops being one.
const HARD_BREAK: &str = "\\\n";

/// One block's lines, glued back into running text.
///
/// Whether they *should* be glued depends on where they came from
/// ([`Page::reflowed`]): a recognizer's lines are the printed rows and a
/// paragraph has to be reassembled out of them, while a generative reader has
/// already done that and the lines it returns are the ones it meant to keep
/// apart. Running the second kind together turns a verse and its gloss into one
/// paragraph.
fn block_text(page: &Page, block: &Block) -> String {
    let mut text = String::new();
    for &index in &block.lines {
        let Some(line) = page.lines.get(index) else {
            continue;
        };
        let piece = line.text.trim();
        if piece.is_empty() {
            continue;
        }
        if page.reflowed {
            if !text.is_empty() {
                text.push_str(HARD_BREAK);
            }
            text.push_str(piece);
        } else {
            append(&mut text, None, piece);
        }
    }
    text
}

/// How the next line attaches to the text written so far.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Glue {
    /// The trailing hyphen is a word break: drop it and close the word up.
    Hyphen,
    /// Concatenate with nothing in between.
    Direct,
    /// Concatenate with one space.
    Space,
}

/// Appends `next` to `text`, gluing the two according to the characters that
/// meet at the seam. `marker` is written into the seam itself, which is how a
/// page break that has to stay visible survives inside a paragraph.
fn append(text: &mut String, marker: Option<&str>, next: &str) {
    if text.is_empty() {
        if let Some(marker) = marker {
            text.push_str(marker);
        }
        text.push_str(next);
        return;
    }
    let glue = glue(text, next);
    if glue == Glue::Hyphen {
        text.pop();
    }
    if let Some(marker) = marker {
        text.push_str(marker);
    }
    if glue == Glue::Space {
        text.push(' ');
    }
    text.push_str(next);
}

/// Decides the seam between the end of `text` and the start of `next`.
fn glue(text: &str, next: &str) -> Glue {
    let mut tail = text.chars().rev();
    let (Some(last), Some(first)) = (tail.next(), next.chars().next()) else {
        return Glue::Direct;
    };
    let before = tail.next();
    if is_word_hyphen(last) {
        // A hyphen closes a word up only when it stands between two letters
        // of a script that breaks words that way. Everywhere else — beside a
        // digit, a bracket, a syllable of a script that does not hyphenate —
        // it belongs to the text and stays.
        let word_break = before.is_some_and(hyphenates) && hyphenates(first);
        return if word_break {
            Glue::Hyphen
        } else {
            Glue::Direct
        };
    }
    if is_dash(last) &&
        before.is_some_and(char::is_numeric) &&
        first.is_numeric()
    {
        // A range broken over the line end: `358–` then `359`.
        return Glue::Direct;
    }
    // A script that writes without word spaces joins to itself without one —
    // but only to itself. Across a **change** of script the seam is not a word
    // boundary that happens to need no space, it is two different things
    // meeting, and welding them produces `МОЛИТВАཧཱུྃ`. The reference makes no
    // such distinction (it tests only for Latin); this is the second place
    // where this port parts company with it, for the same reason as the first.
    if !uses_spaces(last) && !uses_spaces(first) {
        return Glue::Direct;
    }
    Glue::Space
}

/// Whether the character can be the hyphen a word was broken with.
fn is_word_hyphen(c: char) -> bool {
    matches!(c, '-' | '\u{00ad}' | '\u{2010}' | '\u{2011}')
}

/// Whether the character is a dash long enough to be punctuation rather than
/// a word break.
fn is_dash(c: char) -> bool { matches!(c, '\u{2012}'..='\u{2015}' | '−') }

/// Whether the character belongs to a script that breaks words across lines
/// with a hyphen: an alphabet that also separates its words with spaces.
fn hyphenates(c: char) -> bool { c.is_alphabetic() && uses_spaces(c) }

/// Whether the character belongs to a script that separates words with
/// spaces.
///
/// The listed ranges are the scripts that do not: Han and the Japanese kana,
/// together with the full-width forms and the CJK punctuation that travels
/// with them, then Tibetan, Thai, Lao, Khmer and Burmese. Everything else
/// falls through to `true` — including Hangul and Devanagari, which do use
/// spaces, and digits and punctuation, which never decide a seam on their
/// own.
fn uses_spaces(c: char) -> bool {
    !matches!(
        u32::from(c),
        0x0E00..=0x0EFF        // Thai, Lao
            | 0x0F00..=0x0FFF  // Tibetan
            | 0x1000..=0x109F  // Burmese
            | 0x1780..=0x17FF  // Khmer
            | 0x3000..=0x30FF  // CJK punctuation, kana
            | 0x3400..=0x4DBF  // Han, extension A
            | 0x4E00..=0x9FFF  // Han
            | 0xF900..=0xFAFF  // Han, compatibility
            | 0xFF00..=0xFF65  // full-width forms, half-width kana
            | 0x20000..=0x2FFFF // Han, supplementary extensions
    )
}

/// Splits a recognized footnote into its marker and the note itself.
///
/// Four shapes are accepted, because those are what a recognizer returns for
/// the number a typesetter set in superscript: a run of the classic reference
/// symbols, superscript digits that survived as such, a bracketed number, and
/// a small number followed by a full stop, a bracket or a space. A bare
/// number that is *not* delimited is not a marker — a note opening on a year
/// or a page range would otherwise lose its first digits.
fn split_marker(text: &str) -> Option<(String, &str)> {
    let text = text.trim_start();

    let symbols: String =
        text.chars().take_while(|c| "*†‡§¶".contains(*c)).collect();
    if !symbols.is_empty() {
        let note = text[symbols.len()..].trim_start();
        return (!note.is_empty()).then_some((symbols, note));
    }

    let superscripts: String =
        text.chars().map_while(superscript_digit).take(3).collect();
    if !superscripts.is_empty() {
        let width: usize = text
            .chars()
            .take(superscripts.len())
            .map(char::len_utf8)
            .sum();
        let note = text[width..].trim_start();
        return (!note.is_empty()).then_some((superscripts, note));
    }

    if let Some(rest) = text.strip_prefix('[') {
        let (label, note) = rest.split_once(']')?;
        let note = note.trim_start();
        let numbered = !label.is_empty() &&
            label.len() <= 3 &&
            label.chars().all(|c| c.is_ascii_digit());
        return (numbered && !note.is_empty())
            .then(|| (label.to_string(), note));
    }

    let digits: String =
        text.chars().take_while(char::is_ascii_digit).collect();
    if digits.is_empty() || digits.len() > 3 {
        return None;
    }
    let rest = &text[digits.len()..];
    let delimited = rest.strip_prefix(['.', ')']);
    let note = delimited.unwrap_or(rest);
    if delimited.is_none() && !note.starts_with(char::is_whitespace) {
        return None;
    }
    let note = note.trim_start();
    (!note.is_empty()).then_some((digits, note))
}

/// The plain digit a superscript one stands for.
fn superscript_digit(c: char) -> Option<char> {
    match c {
        '\u{00b9}' => Some('1'),
        '\u{00b2}' => Some('2'),
        '\u{00b3}' => Some('3'),
        '\u{2070}' => Some('0'),
        '\u{2074}'..='\u{2079}' => char::from_digit(u32::from(c) - 0x2070, 10),
        _ => None,
    }
}

/// Where a block's lines start and stop horizontally — the only geometry the
/// cross-page rule needs.
#[derive(Debug, Clone, Copy)]
struct Span {
    left: f32,
    right: f32,
    first_start: f32,
    last_end: f32,
    lines: usize,
}

impl Span {
    /// Whether the block's last line runs out to the right margin, which is
    /// what an unfinished paragraph looks like.
    ///
    /// A one-line block cannot say: its right margin *is* its only line, so
    /// the test would pass for every one of them, and a page ending on a
    /// short single line would swallow the paragraph that opens the next
    /// page. Such a block therefore never carries over.
    fn reaches_right(&self) -> bool {
        let width = self.right - self.left;
        self.lines > 1 &&
            width > 0.0 &&
            (self.right - self.last_end) <= MARGIN_TOLERANCE * width
    }

    /// Whether the block's first line starts at the left margin rather than
    /// being indented, which is what a paragraph that is *not* a new one
    /// looks like.
    fn flush_left(&self, height: f32) -> bool {
        let tolerance = if height > 0.0 {
            INDENT_TOLERANCE * height
        } else {
            0.0
        };
        (self.first_start - self.left) <= tolerance
    }
}

/// The horizontal extent of a block, measured over the lines that carry text.
fn span(page: &Page, block: &Block) -> Option<Span> {
    let mut span: Option<Span> = None;
    for &index in &block.lines {
        let Some(line) = page.lines.get(index) else {
            continue;
        };
        if line.text.trim().is_empty() {
            continue;
        }
        let (x0, _, x1, _) = line.quad.bounds();
        span = Some(match span {
            None => Span {
                left: x0,
                right: x1,
                first_start: x0,
                last_end: x1,
                lines: 1,
            },
            Some(span) => Span {
                left: span.left.min(x0),
                right: span.right.max(x1),
                first_start: span.first_start,
                last_end: x1,
                lines: span.lines + 1,
            },
        });
    }
    span
}

/// The page's median line height, the unit an indent is measured in. The
/// median rather than the mean, so that a tall script or a stray display line
/// does not set the scale for the whole page.
fn median_height(page: &Page) -> f32 {
    let mut heights: Vec<f32> = page
        .lines
        .iter()
        .map(|line| line.quad.height())
        .filter(|height| *height > 0.0)
        .collect();
    if heights.is_empty() {
        return 0.0;
    }
    heights.sort_by(f32::total_cmp);
    heights[heights.len() / 2]
}
