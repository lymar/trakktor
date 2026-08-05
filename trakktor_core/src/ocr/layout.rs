//! Layout analysis: a page of recognized lines becomes an ordered sequence of
//! typed blocks.
//!
//! The input is text lines and nothing else — one quadrangle, one string and
//! one confidence per line. There is no layout network behind this module, so
//! every decision is geometry plus a little textual evidence, and the module is
//! deliberate about where that reaches: columns, reading order, paragraphs,
//! headings, footnotes and page furniture are within reach; illustrations,
//! display formulas and the internal structure of a table are not, and nothing
//! here pretends otherwise.
//!
//! Two mistakes decide whether the result is readable, and both are easy to
//! walk into:
//!
//! - **A box height is not a type size.** A line of capitals carries no
//!   descenders and comes out *shorter* than the body text around it, so a
//!   centred title can be the smallest box on the page; a stacking script sets
//!   two or three registers above and below the baseline and comes out twice as
//!   tall as the alphabet beside it at the same type size. Every size judgement
//!   here is therefore made on the **pitch** — the distance between consecutive
//!   line tops — and the few places that must compare a height at all do it
//!   within one script class ([`Script`]).
//! - **A pixel is not a unit.** A constant like "boxes within ten pixels are on
//!   the same line" is a statement about one rasterization: the same page
//!   scanned at twice the resolution comes out progressively more scrambled the
//!   better it was scanned. Every threshold below is a multiple of the page's
//!   own median line height, its median line pitch, or the page size.
//!
//! The passes run in the order they depend on each other: deskew, page
//! furniture, a recursive cut into columns and bands, rows inside a band,
//! paragraphs inside a band, and finally the classification of each paragraph
//! and the reading order that puts headers first and footnotes, feet and page
//! numbers last.
//!
//! Running heads, running feet and watermarks cannot be told from a first
//! heading by looking at one page — the only reliable signal is that the same
//! short string sits at the same height on many pages. That pass is therefore
//! separate: [`document_furniture`] builds the histogram over a whole document
//! and [`analyse`] consumes its verdict.
//!
//! # When a layout model has run
//!
//! Everything above is what geometry can reach on its own. A caller that ran
//! the [layout model](detect) hands its [regions](region::Region) to
//! [`analyse_with`], and the passes stop guessing where the model spoke: a
//! block's kind comes from its label, and the cut treats a region as one
//! object rather than as the lines inside it.
//!
//! The model is an *improvement*, not a precondition, and the code is written
//! that way on purpose. It is trained on Chinese and English documents, and its
//! confidence outside that circle drops to where a threshold decides whether a
//! page marks up at all (measured: 0.97 for an English paragraph against 0.40
//! for a Tibetan one). So a line that fell in no region is *not* dropped —
//! which is what the reference pipeline does with it — and a page with no
//! regions is read exactly as it would have been without the model.

#[cfg(feature = "ocr-runtime")]
pub mod config;
#[cfg(feature = "ocr-runtime")]
pub mod detect;
#[cfg(feature = "ocr-runtime")]
pub mod download;
#[cfg(feature = "ocr-runtime")]
pub mod image;
#[cfg(feature = "ocr-runtime")]
pub mod model;
#[cfg(feature = "ocr-runtime")]
pub mod net;
#[cfg(all(feature = "ocr-runtime", feature = "ocr-burn"))]
pub mod net_burn;
pub mod post;
pub mod region;
#[cfg(feature = "ocr-runtime")]
pub mod sampling;
#[cfg(test)]
mod tests;

use std::collections::{HashMap, HashSet};

pub use region::{Label, Region};

use crate::ocr::page::{Line, Page, Quad};

/// What a block is, as far as geometry and a leading token can tell.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BlockKind {
    /// A heading, `level` counting from 1 for the largest on the page.
    Heading { level: u8 },
    /// Running text.
    Paragraph,
    /// A note at the foot of the page or of a column.
    Footnote,
    /// A running head or foot, a page number, or a watermark — text that
    /// belongs to the sheet rather than to the document.
    PageFurniture,
    /// A short block that sat beside the text flow rather than in it: a figure
    /// caption, a marginal note, a formula number.
    Caption,
    /// An illustration: a region of ink that no text box covers. `index`
    /// points into [`Layout::figures`]. It holds no lines — there is nothing
    /// recognized in it — and it is placed by where it sits on the page.
    Figure { index: usize },
}

/// One block: what it is, which lines it holds, and where it sits.
///
/// `lines` indexes [`Page::lines`], so nothing is copied and a caller can still
/// reach each line's text, score and quadrangle. Lines the confidence filter
/// dropped belong to no block.
#[derive(Debug, Clone)]
pub struct Block {
    pub kind: BlockKind,
    pub lines: Vec<usize>,
    /// The block's extent on the page. It is a quadrangle rather than a
    /// rectangle because the analysis works on a deskewed page and maps its
    /// result back onto the original one.
    pub quad: Quad,
    /// What the layout model called this block, when one ran and claimed it.
    ///
    /// `kind` is what this module made of the block and is always there;
    /// `label` is what the model said and is finer — it tells a table from a
    /// paragraph, an abstract from running text, a stamp from a picture.
    /// Reporting both is the point: the kind drives the Markdown, the label is
    /// what a caller can act on.
    pub label: Option<Label>,
    /// The model's confidence in the label.
    pub score: Option<f32>,
}

/// A page's blocks, in reading order.
#[derive(Debug, Clone)]
pub struct Layout {
    pub blocks: Vec<Block>,
    /// The illustrations found on the page, in the order
    /// [`BlockKind::Figure`] indexes them. Empty unless a caller supplied
    /// them: finding them takes the page raster, which the analysis of a line
    /// list does not have.
    pub figures: Vec<Quad>,
}

impl Layout {
    /// Places illustrations among the blocks.
    ///
    /// A figure has no lines to sort it by, so it is placed by its own
    /// position: after the last block that starts above it. That is the right
    /// answer for a figure between two paragraphs, which is the common case,
    /// and an approximation for one the text wraps around — there the figure
    /// lands before the text beside it rather than in the middle of it.
    pub fn with_figures(mut self, figures: Vec<Quad>) -> Self {
        for (index, quad) in figures.iter().enumerate() {
            let top = quad.bounds().1;
            let at = self
                .blocks
                .iter()
                .position(|block| block.quad.bounds().1 > top)
                .unwrap_or(self.blocks.len());
            self.blocks.insert(
                at,
                Block {
                    kind: BlockKind::Figure { index },
                    lines: Vec::new(),
                    quad: *quad,
                    label: None,
                    score: None,
                },
            );
        }
        self.figures = figures;
        self
    }
}

/// The script class a line is written in, as far as the geometry cares.
///
/// This is not a language: it is the answer to "how tall is a line of this at a
/// given type size", which is the only thing the layout pass needs and the one
/// thing a box height cannot be read for without it.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Script {
    /// Latin, Cyrillic, Greek, Armenian, Georgian — alphabets that stay within
    /// one ascender and one descender.
    Alphabetic,
    /// Han, kana, Hangul — square glyphs on a fixed body.
    Ideographic,
    /// Tibetan, Devanagari, Thai, Myanmar, Khmer and the rest of the stacking
    /// scripts: subjoined letters and vowel signs pile up above and below the
    /// baseline, so a line runs about twice the height of an alphabetic line of
    /// the same type size.
    Stacking,
    /// Digits, punctuation, symbols — nothing that says anything about height.
    Neutral,
}

/// The text that repeats in the same place across a document: running heads
/// and feet, and watermarks.
///
/// Built by [`document_furniture`]; the default value is "nothing repeats",
/// which is the right answer when there is only one page to look at.
#[derive(Debug, Clone, Default)]
pub struct Repeated {
    keys: HashSet<(String, u32)>,
}

impl Repeated {
    /// Whether this line is one of the repeated ones. `page_height` is the
    /// height of the page the line came from: the band a line falls in is a
    /// fraction of the page, so pages of different sizes still compare.
    pub fn contains(&self, line: &Line, page_height: u32) -> bool {
        match repeat_key(line, page_height) {
            Some(key) => self.keys.contains(&key),
            None => false,
        }
    }

    /// How many distinct strings were found to repeat.
    pub fn len(&self) -> usize { self.keys.len() }

    pub fn is_empty(&self) -> bool { self.keys.is_empty() }
}

/// The named thresholds of the analysis.
///
/// Every field is dimensionless: a multiple of the page's median line height
/// `H`, of the body line pitch `P`, or of the page size. The defaults are the
/// measured ones — the constants below say what each was measured against.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Settings {
    /// Mean character probability a line must reach to be considered.
    pub min_score: f32,
    /// Skew in degrees above which the page is turned straight.
    pub deskew_degrees: f32,
    /// Vertical overlap, as a fraction of the shorter box, for two lines to
    /// share a row.
    pub row_overlap: f32,
    /// × `H`: the largest horizontal gap allowed inside one row.
    pub row_merge_gap: f32,
    /// × `H`: the narrowest column gutter.
    pub gutter_height: f32,
    /// × page width: the narrowest column gutter. The wider bound wins.
    pub gutter_width: f32,
    /// How many lines a part must hold to be a column rather than a floating
    /// block.
    pub min_column_lines: usize,
    /// Fraction of the shorter vertical span two columns must share.
    pub column_y_overlap: f32,
    /// Fraction of one part's lines that must line up with every other part
    /// for the parts to be a table rather than text columns.
    pub table_alignment: f32,
    /// × `H`: the vertical gap that ends a band.
    pub cut_gap: f32,
    /// × `P`: the pitch ratio that starts a paragraph on its own.
    pub paragraph_pitch: f32,
    /// × `P`: the smaller pitch bump, believed only when the previous line
    /// ended short.
    pub soft_pitch: f32,
    /// × `H`: the first-line indent that starts a paragraph.
    pub indent: f32,
    /// × column width: how far short of the right edge a line must stop to
    /// count as having ended short.
    pub short_line: f32,
    /// × `P`: the pitch ratio that makes a block a heading.
    pub heading_pitch: f32,
    /// × column width: how far off the column's centre line a heading may sit.
    pub center_tolerance: f32,
    /// × column width: how wide a heading may be.
    pub heading_max_width: f32,
    /// How many rows a heading may hold.
    pub heading_max_rows: usize,
    /// × `P`: the pitch ratio below which a block is footnote-sized.
    pub footnote_pitch: f32,
    /// × page height: footnotes start below this.
    pub footnote_top: f32,
    /// × `P`: the gap that must separate a footnote from the body.
    pub footnote_gap: f32,
    /// × page width: how narrow a page-number box is.
    pub page_number_width: f32,
    /// × page width: how close to a margin, or to the page's centre line, a
    /// page number sits.
    pub page_number_edge: f32,
    /// × page height: the band at each end of the page in which page furniture
    /// is looked for.
    pub furniture_band: f32,
    /// × `H`: above this a block is a taller script, not a larger size.
    pub tall_script: f32,
}

/// Lines below this mean character probability are ignored. A spurious box in
/// the middle of a column moves every statistic on the page, and the
/// recognition stage that produced it already treats this as its own floor.
const MIN_SCORE: f32 = 0.50;

/// Half a degree of skew moves one end of a full-width line by about a whole
/// letter on a page-sized raster, which is enough to reorder two boxes of one
/// row; a third of a degree is where correcting starts to pay.
const DESKEW_DEGREES: f32 = 0.30;

/// Two boxes share a row when the shorter one lies this far inside the
/// taller's vertical span. Measured against prose whose ascenders and
/// descenders make neighbouring boxes of one row differ in height by half.
const ROW_OVERLAP: f32 = 0.55;

/// × `H`. Without this guard a caption at one margin is swept into the row of
/// a body line at the other and the two texts run together; measured on a page
/// whose text wraps around an illustration.
const ROW_MERGE_GAP: f32 = 1.2;

/// × `H` and × page width. A column gutter is at least one line height wide
/// and at least a fortieth of the page, whichever is more. Below that the word
/// spaces of a justified line start to look like gutters.
const GUTTER_HEIGHT: f32 = 1.2;
const GUTTER_WIDTH: f32 = 0.025;

/// A part holding fewer lines than this is not a column: it is a caption, a
/// marginal note or a formula number, and it is put back by distance instead
/// of being read as a column of its own.
const MIN_COLUMN_LINES: usize = 3;

/// Two parts sit side by side only if they share this much of the shorter
/// one's vertical span; otherwise they are two bands that happen to start at
/// different indents.
const COLUMN_Y_OVERLAP: f32 = 0.50;

/// Row-aligned parts are a table and read across; parts that are not are text
/// columns and read down. Measured on a multi-column numeric table against
/// multi-column prose.
const TABLE_ALIGNMENT: f32 = 0.60;

/// × `H`. The vertical gap that ends a band: above the gap ordinary leading
/// leaves between two consecutive lines, below the one a paragraph break or a
/// section opening leaves.
const CUT_GAP: f32 = 1.55;

/// × `P`. Extra leading of a third of a line starts a paragraph on its own.
const PARAGRAPH_PITCH: f32 = 1.35;

/// × `P`. A smaller bump, believed only when the previous line ended short —
/// the signature of block-style paragraphs, which carry no indent.
const SOFT_PITCH: f32 = 1.12;

/// × `H`. A first-line indent runs about one em, a little under a line height;
/// measured on justified prose whose indent came to 1.3 `H`.
const INDENT: f32 = 0.9;

/// × column width. How far short of the column's right edge a line must stop
/// before it counts as the last line of a paragraph.
const SHORT_LINE: f32 = 0.12;

/// × `P`. A heading's own pitch against the body's. Measured on pages where a
/// section head was set one size up: the box heights of head and body
/// overlapped completely there, which is why this is a pitch ratio.
const HEADING_PITCH: f32 = 1.22;

/// × column width. Centring tolerance, and the width a heading must stay
/// under. Measured on an article page whose two-line abstract sits at 0.79 of
/// the column and must *not* read as a heading, while the centred title, the
/// author line and the section head all sit well under 0.75.
const CENTER_TOLERANCE: f32 = 0.06;
const HEADING_MAX_WIDTH: f32 = 0.75;
const HEADING_MAX_ROWS: usize = 3;

/// × `P`. Footnote pitch against body pitch. Measured on a page setting eight
/// point notes under ten point body: the two pitches came out at 0.84 of each
/// other while their *box heights* overlapped entirely. The margin is thin,
/// which is why a leading marker is asked for as well.
const FOOTNOTE_PITCH: f32 = 0.90;

/// × page height, and × `P`. Notes sit low on the page and are separated from
/// the body by more than ordinary leading.
const FOOTNOTE_TOP: f32 = 0.62;
const FOOTNOTE_GAP: f32 = 1.4;

/// × page width. A page number is a narrow box anchored to a margin or to the
/// page's centre line. Measured on pages whose number sat a few pixels under
/// the last note on one and at the very top of the sheet on another — so
/// isolation alone does not find it, and narrowness plus anchoring does.
const PAGE_NUMBER_WIDTH: f32 = 0.10;
const PAGE_NUMBER_EDGE: f32 = 0.15;

/// × page height. The band at each end of the page in which page furniture is
/// looked for at all.
const FURNITURE_BAND: f32 = 0.15;

/// × `H`. Above this a block is written in a taller script rather than set in
/// a larger size. Measured on pages mixing a stacking script with an
/// alphabetic one, where the stacking lines ran 2.0 to 2.3 times the height of
/// the alphabetic lines beside them at the same type size.
const TALL_SCRIPT: f32 = 1.6;

/// How many rows at each end of a page may be peeled off as furniture. Two,
/// because a foot can carry a folio number at one margin and a page number at
/// the other.
const MAX_FURNITURE_ROWS: usize = 2;

/// How many bands a page is divided into when looking for text that repeats
/// across pages. Twenty bands is a twentieth of the page — coarser than a
/// running head's drift from page to page, finer than the distance from a head
/// to the body.
const REPEAT_BANDS: u32 = 20;

/// A string must appear on this fraction of the document's pages, and on at
/// least this many, before it is called furniture rather than content.
const REPEAT_FRACTION: f32 = 0.60;
const REPEAT_MIN_PAGES: usize = 3;

/// Two headings are the same level when their sizes agree to within this.
const LEVEL_TOLERANCE: f32 = 0.08;

/// The deepest heading level ever assigned.
const MAX_LEVEL: u8 = 6;

impl Default for Settings {
    fn default() -> Self {
        Self {
            min_score: MIN_SCORE,
            deskew_degrees: DESKEW_DEGREES,
            row_overlap: ROW_OVERLAP,
            row_merge_gap: ROW_MERGE_GAP,
            gutter_height: GUTTER_HEIGHT,
            gutter_width: GUTTER_WIDTH,
            min_column_lines: MIN_COLUMN_LINES,
            column_y_overlap: COLUMN_Y_OVERLAP,
            table_alignment: TABLE_ALIGNMENT,
            cut_gap: CUT_GAP,
            paragraph_pitch: PARAGRAPH_PITCH,
            soft_pitch: SOFT_PITCH,
            indent: INDENT,
            short_line: SHORT_LINE,
            heading_pitch: HEADING_PITCH,
            center_tolerance: CENTER_TOLERANCE,
            heading_max_width: HEADING_MAX_WIDTH,
            heading_max_rows: HEADING_MAX_ROWS,
            footnote_pitch: FOOTNOTE_PITCH,
            footnote_top: FOOTNOTE_TOP,
            footnote_gap: FOOTNOTE_GAP,
            page_number_width: PAGE_NUMBER_WIDTH,
            page_number_edge: PAGE_NUMBER_EDGE,
            furniture_band: FURNITURE_BAND,
            tall_script: TALL_SCRIPT,
        }
    }
}

/// Finds the text that repeats in the same place across a document.
///
/// A running head cannot be told from a first heading on one page: both are a
/// short line alone at the top. What separates them is that the head comes
/// back, page after page, at the same height. This builds that histogram —
/// keyed on a line's letters, so a head carrying the page number still matches
/// itself, and on the band of the page it sits in — and keeps the entries that
/// turn up on most of the pages. The same rule catches a watermark, which
/// repeats too and sits nowhere near a margin.
///
/// Under [`REPEAT_MIN_PAGES`] pages the question cannot be answered and the
/// result is empty.
pub fn document_furniture(pages: &[Page]) -> Repeated {
    if pages.len() < REPEAT_MIN_PAGES {
        return Repeated::default();
    }
    let mut seen: HashMap<(String, u32), HashSet<usize>> = HashMap::new();
    for (at, page) in pages.iter().enumerate() {
        for line in &page.lines {
            if let Some(key) = repeat_key(line, page.height) {
                seen.entry(key).or_default().insert(at);
            }
        }
    }
    let needed = ((pages.len() as f32) * REPEAT_FRACTION).ceil() as usize;
    let needed = needed.max(REPEAT_MIN_PAGES);
    Repeated {
        keys: seen
            .into_iter()
            .filter(|(_, on)| on.len() >= needed)
            .map(|(key, _)| key)
            .collect(),
    }
}

/// The histogram key of a line: its letters, folded to lower case, and the
/// band of the page it sits in. A line carrying no letters at all — a page
/// number, a formula number — is not keyed: those are what the page-number
/// rule is for, and that one needs no document behind it.
fn repeat_key(line: &Line, page_height: u32) -> Option<(String, u32)> {
    if page_height == 0 {
        return None;
    }
    let text: String = line
        .text
        .chars()
        .filter(|c| c.is_alphabetic())
        .flat_map(|c| c.to_lowercase())
        .collect();
    if text.is_empty() {
        return None;
    }
    let band = (line.quad.center_y() / page_height as f32 * REPEAT_BANDS as f32)
        as i32;
    Some((text, band.clamp(0, REPEAT_BANDS as i32 - 1) as u32))
}

/// Classifies a string by the Unicode blocks its characters fall in.
///
/// The dominant class wins; a stacking script takes a tie, because it is the
/// one that would otherwise be mistaken for a larger type size. A string with
/// no letters at all is [`Script::Neutral`] and says nothing about height.
pub fn script(text: &str) -> Script {
    let (mut alphabetic, mut ideographic, mut stacking) =
        (0usize, 0usize, 0usize);
    for ch in text.chars() {
        match ch as u32 {
            // Devanagari through Tibetan, plus Myanmar and Khmer: the scripts
            // that stack marks into several registers around the baseline.
            0x0900..=0x0FFF | 0x1000..=0x109F | 0x1780..=0x17FF => {
                stacking += 1
            },
            // Hangul jamo, the CJK blocks with their kana, and the Hangul
            // syllables.
            0x1100..=0x11FF |
            0x2E80..=0x9FFF |
            0xA960..=0xA97F |
            0xAC00..=0xD7FF => ideographic += 1,
            _ if ch.is_alphabetic() => alphabetic += 1,
            _ => {},
        }
    }
    if stacking == 0 && ideographic == 0 && alphabetic == 0 {
        Script::Neutral
    } else if stacking >= ideographic && stacking >= alphabetic {
        Script::Stacking
    } else if ideographic >= alphabetic {
        Script::Ideographic
    } else {
        Script::Alphabetic
    }
}

/// Turns one page's lines into ordered, typed blocks.
///
/// `repeated` is the document-level verdict from [`document_furniture`]; pass
/// `&Repeated::default()` when there is only one page to look at.
pub fn analyse(
    page: &Page,
    repeated: &Repeated,
    settings: &Settings,
) -> Layout {
    let sheet = Sheet::new(page, settings);
    if sheet.lines.is_empty() {
        return Layout {
            blocks: Vec::new(),
            figures: Vec::new(),
        };
    }
    let unit = sheet.median_height();

    let furniture = strip_furniture(&sheet, repeated, unit, settings);
    let body: Vec<usize> = (0..sheet.lines.len())
        .filter(|at| !furniture.taken.contains(at))
        .collect();

    let leaves = if body.is_empty() {
        Vec::new()
    } else {
        let column = sheet.bounds_of(&body);
        xy_cut(&sheet, &body, column, unit, settings)
    };
    let segments = segment(&sheet, &leaves, unit, settings);
    let stats = Stats::new(&sheet, &segments, unit);

    let mut kinds: Vec<BlockKind> = segments
        .iter()
        .enumerate()
        .map(|(at, seg)| {
            classify(seg, &stats, at + 1 == segments.len(), settings)
        })
        .collect();
    assign_levels(&segments, &mut kinds, &stats);

    // Heads first, then the body in cut order, then the notes, then the feet
    // and the page number. A note in the middle of a column is therefore
    // moved to the end of the page rather than read where it sits, which is
    // what makes the body read continuously.
    let mut blocks = Vec::with_capacity(segments.len() + 2);
    for row in &furniture.top {
        blocks.push(sheet.block(
            BlockKind::PageFurniture,
            &row.lines,
            row.rect,
        ));
    }
    for (seg, kind) in segments.iter().zip(&kinds) {
        if *kind != BlockKind::Footnote {
            blocks.push(sheet.block(*kind, &seg.lines, seg.rect));
        }
    }
    for (seg, kind) in segments.iter().zip(&kinds) {
        if *kind == BlockKind::Footnote {
            blocks.push(sheet.block(*kind, &seg.lines, seg.rect));
        }
    }
    for row in &furniture.bottom {
        blocks.push(sheet.block(
            BlockKind::PageFurniture,
            &row.lines,
            row.rect,
        ));
    }
    Layout {
        blocks,
        figures: Vec::new(),
    }
}

// ------------------------------------------------------------------------
// With a layout model
// ------------------------------------------------------------------------

/// Turns one page's lines into ordered, typed blocks, using a layout model's
/// regions where they reach and the geometry everywhere else.
///
/// What the labels replace is **classification**, not the reading of the page:
/// a region says what a block *is*, and it says where a paragraph ends — the
/// model returns one box per paragraph, which is the split the geometry has to
/// guess from leading and indents. What the labels do *not* replace is the
/// ordering, which stays this module's own recursive cut, run over regions as
/// whole objects rather than over the lines inside them. That is what puts a
/// page whose text wraps around a picture back in order.
///
/// Lines that fell in no region are not dropped. They are cut, grouped and
/// classified exactly as [`analyse`] would have done, and take their place
/// among the regions by position. On a page the model said nothing about, this
/// function and [`analyse`] agree.
pub fn analyse_with(
    page: &Page,
    regions: &[Region],
    repeated: &Repeated,
    settings: &Settings,
) -> Layout {
    let sheet = Sheet::new(page, settings);
    if sheet.lines.is_empty() && regions.is_empty() {
        return Layout {
            blocks: Vec::new(),
            figures: Vec::new(),
        };
    }
    let unit = sheet.median_height();

    // Regions come in page coordinates and the analysis works on a deskewed
    // page, so they are turned the same way the lines were.
    let placed: Vec<(Label, Rect)> = regions
        .iter()
        .map(|region| {
            let points = region
                .quad
                .points
                .map(|p| unrotate(p, sheet.center, sheet.sin, sheet.cos));
            (region.label, bounds(&points))
        })
        .collect();

    // Which region each line belongs to. The rule is the innermost container,
    // not the largest overlap — see [`region::owner`], which both this and the
    // generative engine's block assembly go through so they cannot disagree
    // about what is inside what.
    let deskewed: Vec<Region> = placed
        .iter()
        .zip(regions)
        .map(|((label, rect), region)| Region {
            label: *label,
            score: region.score,
            quad: Quad::new([
                (rect.x0, rect.y0),
                (rect.x1, rect.y0),
                (rect.x1, rect.y1),
                (rect.x0, rect.y1),
            ]),
        })
        .collect();
    let owner: Vec<Option<usize>> = sheet
        .lines
        .iter()
        .map(|line| {
            region::owner(
                (line.rect.x0, line.rect.y0, line.rect.x1, line.rect.y1),
                &deskewed,
            )
        })
        .collect();

    // Pictures have no lines to carry them; they are their own blocks and the
    // caller crops them from the region, not from the ink.
    let mut figures = Vec::new();
    let mut figure_labels = Vec::new();
    let mut figure_scores = Vec::new();
    let mut figure_of: HashMap<usize, usize> = HashMap::new();
    for (index, (label, _)) in placed.iter().enumerate() {
        if label.is_pictorial() {
            figure_of.insert(index, figures.len());
            figures.push(regions[index].quad);
            figure_labels.push(*label);
            figure_scores.push(regions[index].score);
        }
    }

    // One segment per region that caught something, plus the geometric
    // segments of everything that fell outside every region.
    let mut segments: Vec<Segment> = Vec::new();
    let mut labels: Vec<Option<Label>> = Vec::new();
    let mut scores: Vec<Option<f32>> = Vec::new();
    for (index, (label, rect)) in placed.iter().enumerate() {
        if label.is_pictorial() {
            continue;
        }
        let lines: Vec<usize> = (0..sheet.lines.len())
            .filter(|at| owner[*at] == Some(index))
            .collect();
        if lines.is_empty() {
            continue;
        }
        segments.push(sheet.segment_of(&lines, *rect, unit, settings));
        labels.push(Some(*label));
        scores.push(Some(regions[index].score));
    }

    let orphans: Vec<usize> = (0..sheet.lines.len())
        .filter(|at| owner[*at].is_none())
        .collect();
    let mut furniture = Furniture {
        taken: HashSet::new(),
        top: Vec::new(),
        bottom: Vec::new(),
    };
    if !orphans.is_empty() {
        furniture =
            strip_furniture_of(&sheet, &orphans, repeated, unit, settings);
        let body: Vec<usize> = orphans
            .iter()
            .copied()
            .filter(|at| !furniture.taken.contains(at))
            .collect();
        if !body.is_empty() {
            let column = sheet.bounds_of(&body);
            let leaves = xy_cut(&sheet, &body, column, unit, settings);
            for seg in segment(&sheet, &leaves, unit, settings) {
                segments.push(seg);
                labels.push(None);
                scores.push(None);
            }
        }
    }

    // Reading order over whole blocks. Figures take part in it: a picture the
    // text wraps around is an obstacle, and leaving it out is exactly what
    // makes the geometric layer rebuild the page wrongly.
    let mut atoms: Vec<(Rect, Piece)> = segments
        .iter()
        .enumerate()
        .map(|(at, seg)| (seg.rect, Piece::Segment(at)))
        .collect();
    for (index, at) in &figure_of {
        atoms.push((placed[*index].1, Piece::Figure(*at)));
    }
    let order = order_blocks(&atoms, unit, settings);

    let stats = Stats::new(&sheet, &segments, unit);
    let mut kinds: Vec<BlockKind> = segments
        .iter()
        .zip(&labels)
        .enumerate()
        .map(|(at, (seg, label))| match label {
            Some(label) => kind_of(*label),
            None => classify(seg, &stats, at + 1 == segments.len(), settings),
        })
        .collect();
    assign_levels(&segments, &mut kinds, &stats);

    // The ranking above judges by size, which is the right answer when nothing
    // else is known and the wrong one here: the model has already said which
    // heading is *the* title, and a numbered section heading set in the same
    // size as the title would otherwise rank alongside it. So a labelled
    // heading takes its level from the label, and its numbering below that.
    for (at, label) in labels.iter().enumerate() {
        let level = match label {
            Some(Label::DocTitle) => 1,
            Some(Label::ParagraphTitle) => {
                1 + leading_number(&segments[at].lead).unwrap_or(1)
            },
            _ => continue,
        };
        kinds[at] = BlockKind::Heading {
            level: level.min(MAX_LEVEL),
        };
    }

    let mut blocks = Vec::with_capacity(atoms.len() + 2);
    for row in &furniture.top {
        blocks.push(sheet.block(
            BlockKind::PageFurniture,
            &row.lines,
            row.rect,
        ));
    }
    let mut notes = Vec::new();
    for at in order {
        match atoms[at].1 {
            Piece::Figure(index) => blocks.push(Block {
                kind: BlockKind::Figure { index },
                lines: Vec::new(),
                quad: figures[index],
                label: Some(figure_labels[index]),
                score: Some(figure_scores[index]),
            }),
            Piece::Segment(at) => {
                let mut block = sheet.block(
                    kinds[at],
                    &segments[at].lines,
                    segments[at].rect,
                );
                block.label = labels[at];
                block.score = scores[at];
                // Notes go after the body of the page, as they do without a
                // model: a note in the middle of a column would otherwise cut
                // the text in two.
                if kinds[at] == BlockKind::Footnote {
                    notes.push(block);
                } else {
                    blocks.push(block);
                }
            },
        }
    }
    blocks.extend(notes);
    for row in &furniture.bottom {
        blocks.push(sheet.block(
            BlockKind::PageFurniture,
            &row.lines,
            row.rect,
        ));
    }
    Layout { blocks, figures }
}

/// What sits at one place in the reading order.
#[derive(Debug, Clone, Copy)]
enum Piece {
    Segment(usize),
    Figure(usize),
}

/// The block a label stands for.
///
/// Two of the twenty are deliberately *not* mapped to something of their own.
/// A `formula` and a `table` are read as running text here, because that is
/// what the recognizer gave us for them — the label says what the block is, and
/// a caller that can do better with it (the generative engine can) reads the
/// label itself.
fn kind_of(label: Label) -> BlockKind {
    match label {
        // The level is provisional; the ranking pass replaces it.
        Label::DocTitle => BlockKind::Heading { level: 1 },
        Label::ParagraphTitle => BlockKind::Heading { level: 2 },
        Label::Footnote => BlockKind::Footnote,
        Label::Header | Label::Footer | Label::Number | Label::AsideText => {
            BlockKind::PageFurniture
        },
        Label::FigureTitle | Label::FormulaNumber => BlockKind::Caption,
        Label::Image | Label::Chart | Label::Seal => {
            // Pictures never reach here — they carry no lines — but the match
            // has to be total.
            BlockKind::Paragraph
        },
        _ => BlockKind::Paragraph,
    }
}

/// Orders whole blocks: columns first, then bands, recursively.
///
/// The same cut as [`xy_cut`], on boxes that have already swallowed the space
/// between their own lines — which is what lets the gap that separates two
/// columns be measured against the page's line height rather than guessed.
fn order_blocks(
    atoms: &[(Rect, Piece)],
    unit: f32,
    settings: &Settings,
) -> Vec<usize> {
    fn cut(
        atoms: &[(Rect, Piece)],
        ids: &[usize],
        unit: f32,
        settings: &Settings,
        out: &mut Vec<usize>,
    ) {
        if ids.len() <= 1 {
            out.extend_from_slice(ids);
            return;
        }
        let gutter = settings.gutter_height * unit;
        let columns = split_rects(atoms, ids, Axis::X, gutter);
        if columns.len() >= 2 {
            // Side by side only if they share a vertical span; otherwise these
            // are bands that happen to start at different indents.
            let first = union_of(atoms, &columns[0]);
            let last = union_of(atoms, columns.last().expect("a column"));
            if first.y_overlap(&last) >= settings.column_y_overlap {
                for column in &columns {
                    cut(atoms, column, unit, settings, out);
                }
                return;
            }
        }
        let bands = split_rects(atoms, ids, Axis::Y, 0.0);
        if bands.len() >= 2 {
            for band in &bands {
                cut(atoms, band, unit, settings, out);
            }
            return;
        }
        let mut sorted = ids.to_vec();
        sorted.sort_by(|a, b| {
            let (l, r) = (atoms[*a].0, atoms[*b].0);
            compare(l.y0, r.y0).then(compare(l.x0, r.x0))
        });
        out.extend(sorted);
    }

    let ids: Vec<usize> = (0..atoms.len()).collect();
    let mut out = Vec::with_capacity(atoms.len());
    cut(atoms, &ids, unit, settings, &mut out);
    out
}

/// Sorts blocks along one axis and cuts wherever they leave a gap.
fn split_rects(
    atoms: &[(Rect, Piece)],
    ids: &[usize],
    axis: Axis,
    min_gap: f32,
) -> Vec<Vec<usize>> {
    let span = |at: &usize| {
        let rect = &atoms[*at].0;
        match axis {
            Axis::X => (rect.x0, rect.x1),
            Axis::Y => (rect.y0, rect.y1),
        }
    };
    let mut sorted = ids.to_vec();
    sorted.sort_by(|a, b| compare(span(a).0, span(b).0));

    let mut parts: Vec<Vec<usize>> = Vec::new();
    let mut end = f32::NEG_INFINITY;
    for at in sorted {
        let (start, stop) = span(&at);
        if parts.is_empty() || start - end >= min_gap.max(f32::EPSILON) {
            parts.push(Vec::new());
            end = stop;
        } else {
            end = end.max(stop);
        }
        parts.last_mut().expect("a part").push(at);
    }
    parts
}

fn union_of(atoms: &[(Rect, Piece)], ids: &[usize]) -> Rect {
    let mut rect = atoms[ids[0]].0;
    for at in &ids[1..] {
        rect.merge(&atoms[*at].0);
    }
    rect
}

// ------------------------------------------------------------------------
// Geometry
// ------------------------------------------------------------------------

/// An axis-aligned rectangle in deskewed page pixels.
#[derive(Debug, Clone, Copy)]
struct Rect {
    x0: f32,
    y0: f32,
    x1: f32,
    y1: f32,
}

impl Rect {
    fn width(&self) -> f32 { self.x1 - self.x0 }

    fn height(&self) -> f32 { self.y1 - self.y0 }

    fn center_x(&self) -> f32 { (self.x0 + self.x1) / 2.0 }

    fn center_y(&self) -> f32 { (self.y0 + self.y1) / 2.0 }

    fn merge(&mut self, other: &Rect) {
        self.x0 = self.x0.min(other.x0);
        self.y0 = self.y0.min(other.y0);
        self.x1 = self.x1.max(other.x1);
        self.y1 = self.y1.max(other.y1);
    }

    /// How much of the shorter of the two vertical spans the two share.
    fn y_overlap(&self, other: &Rect) -> f32 {
        let shared = self.y1.min(other.y1) - self.y0.max(other.y0);
        let shortest = self.height().min(other.height());
        if shortest <= 0.0 {
            return 0.0;
        }
        (shared / shortest).clamp(0.0, 1.0)
    }
}

/// One line as the analysis sees it.
#[derive(Debug, Clone, Copy)]
struct Geom {
    /// Index into [`Page::lines`].
    index: usize,
    rect: Rect,
    /// The mean of the quadrangle's two vertical edges — the line's own
    /// height, which survives the deskew rotation, unlike its bounding box.
    height: f32,
    script: Script,
}

/// A page prepared for analysis: its lines deskewed, plus the rotation that
/// got them there, so a block's extent can be mapped back onto the page.
struct Sheet<'a> {
    page: &'a Page,
    lines: Vec<Geom>,
    width: f32,
    height: f32,
    sin: f32,
    cos: f32,
    center: (f32, f32),
}

impl<'a> Sheet<'a> {
    fn new(page: &'a Page, settings: &Settings) -> Self {
        let center = (page.width as f32 / 2.0, page.height as f32 / 2.0);
        let kept: Vec<usize> = page
            .lines
            .iter()
            .enumerate()
            .filter(|(_, line)| line.score >= settings.min_score)
            .map(|(at, _)| at)
            .collect();

        // The skew is the median direction of the lines long enough for their
        // direction to mean anything: a short box is about as wide as it is
        // tall, and its top edge points wherever the fit happened to leave it.
        let mut angles: Vec<f32> = kept
            .iter()
            .filter_map(|at| {
                let quad = &page.lines[*at].quad;
                if edge_width(quad) <= 3.0 * edge_height(quad) {
                    return None;
                }
                let [tl, tr, _, _] = quad.points;
                Some((tr.1 - tl.1).atan2(tr.0 - tl.0))
            })
            .collect();
        let theta = median(&mut angles).unwrap_or(0.0);
        let theta = if theta.abs() > settings.deskew_degrees.to_radians() {
            theta
        } else {
            0.0
        };
        let (sin, cos) = (theta.sin(), theta.cos());

        let lines = kept
            .into_iter()
            .map(|index| {
                let quad = &page.lines[index].quad;
                let points = quad.points.map(|p| unrotate(p, center, sin, cos));
                Geom {
                    index,
                    rect: bounds(&points),
                    height: edge_height(quad),
                    script: script(&page.lines[index].text),
                }
            })
            .collect();

        Self {
            page,
            lines,
            width: page.width as f32,
            height: page.height as f32,
            sin,
            cos,
            center,
        }
    }

    fn text(&self, at: usize) -> &str {
        &self.page.lines[self.lines[at].index].text
    }

    fn median_height(&self) -> f32 {
        let mut heights: Vec<f32> =
            self.lines.iter().map(|line| line.height).collect();
        median(&mut heights).filter(|h| *h > 0.0).unwrap_or(1.0)
    }

    fn bounds_of(&self, ids: &[usize]) -> Rect {
        let mut rect = self.lines[ids[0]].rect;
        for at in &ids[1..] {
            rect.merge(&self.lines[*at].rect);
        }
        rect
    }

    /// Builds a segment out of the lines a layout region caught.
    ///
    /// The region *is* the paragraph — the model returns one box per paragraph
    /// — so there is nothing to split here; what still has to be measured is
    /// everything the classification and the heading ranking read: how many
    /// rows, at what pitch, in what script, opening with what.
    fn segment_of(
        &self,
        lines: &[usize],
        rect: Rect,
        unit: f32,
        settings: &Settings,
    ) -> Segment {
        let rows = make_rows(self, lines, unit, settings);
        let mut tops: Vec<f32> = rows.iter().map(|row| row.rect.y0).collect();
        tops.sort_by(|a, b| compare(*a, *b));
        let mut steps: Vec<f32> = tops
            .windows(2)
            .map(|pair| pair[1] - pair[0])
            .filter(|step| *step > 0.0)
            .collect();
        let mut heights: Vec<f32> =
            lines.iter().map(|at| self.lines[*at].height).collect();
        let lead: Vec<&str> = rows
            .first()
            .map(|row| row.lines.iter().map(|at| self.text(*at)).collect())
            .unwrap_or_default();
        Segment {
            lines: lines.to_vec(),
            rect,
            column: rect,
            rows: rows.len(),
            pitch: median(&mut steps).filter(|pitch| *pitch > 0.0),
            height: median(&mut heights).unwrap_or(unit),
            script: dominant_script(self, lines),
            floating: false,
            gap_above: f32::INFINITY,
            gap_below: f32::INFINITY,
            from_end: 0,
            lead: lead.join(" "),
        }
    }

    /// Builds a block, mapping its extent back onto the original page and its
    /// line positions back onto [`Page::lines`].
    fn block(&self, kind: BlockKind, lines: &[usize], rect: Rect) -> Block {
        let corners = [
            (rect.x0, rect.y0),
            (rect.x1, rect.y0),
            (rect.x1, rect.y1),
            (rect.x0, rect.y1),
        ];
        Block {
            kind,
            lines: lines.iter().map(|at| self.lines[*at].index).collect(),
            quad: Quad::new(
                corners.map(|p| rotate(p, self.center, self.sin, self.cos)),
            ),
            label: None,
            score: None,
        }
    }
}

/// The mean of a quadrangle's two vertical edges.
fn edge_height(quad: &Quad) -> f32 {
    let [tl, tr, br, bl] = quad.points;
    (distance(tl, bl) + distance(tr, br)) / 2.0
}

/// The mean of a quadrangle's two horizontal edges.
fn edge_width(quad: &Quad) -> f32 {
    let [tl, tr, br, bl] = quad.points;
    (distance(tl, tr) + distance(bl, br)) / 2.0
}

fn distance(from: (f32, f32), to: (f32, f32)) -> f32 {
    ((to.0 - from.0).powi(2) + (to.1 - from.1).powi(2)).sqrt()
}

/// Rotates a point by `-theta` about `center`: the deskew itself.
fn unrotate(
    p: (f32, f32),
    center: (f32, f32),
    sin: f32,
    cos: f32,
) -> (f32, f32) {
    let (dx, dy) = (p.0 - center.0, p.1 - center.1);
    (
        center.0 + dx * cos + dy * sin,
        center.1 - dx * sin + dy * cos,
    )
}

/// Rotates a point by `+theta` about `center`: back onto the original page.
fn rotate(p: (f32, f32), center: (f32, f32), sin: f32, cos: f32) -> (f32, f32) {
    let (dx, dy) = (p.0 - center.0, p.1 - center.1);
    (
        center.0 + dx * cos - dy * sin,
        center.1 + dx * sin + dy * cos,
    )
}

fn bounds(points: &[(f32, f32); 4]) -> Rect {
    let xs = points.map(|p| p.0);
    let ys = points.map(|p| p.1);
    Rect {
        x0: xs.iter().copied().fold(f32::INFINITY, f32::min),
        y0: ys.iter().copied().fold(f32::INFINITY, f32::min),
        x1: xs.iter().copied().fold(f32::NEG_INFINITY, f32::max),
        y1: ys.iter().copied().fold(f32::NEG_INFINITY, f32::max),
    }
}

fn median(values: &mut [f32]) -> Option<f32> {
    if values.is_empty() {
        return None;
    }
    values.sort_by(|a, b| compare(*a, *b));
    let n = values.len();
    Some(if n % 2 == 1 {
        values[n / 2]
    } else {
        (values[n / 2 - 1] + values[n / 2]) / 2.0
    })
}

fn compare(a: f32, b: f32) -> std::cmp::Ordering {
    a.partial_cmp(&b).unwrap_or(std::cmp::Ordering::Equal)
}

// ------------------------------------------------------------------------
// Page furniture
// ------------------------------------------------------------------------

/// What was taken off the page before the body was read.
struct Furniture {
    taken: HashSet<usize>,
    top: Vec<Row>,
    bottom: Vec<Row>,
}

fn strip_furniture(
    sheet: &Sheet,
    repeated: &Repeated,
    unit: f32,
    settings: &Settings,
) -> Furniture {
    let all: Vec<usize> = (0..sheet.lines.len()).collect();
    strip_furniture_of(sheet, &all, repeated, unit, settings)
}

/// The same pass over a subset of the page's lines — what is left after a
/// layout model has claimed the rest.
fn strip_furniture_of(
    sheet: &Sheet,
    ids: &[usize],
    repeated: &Repeated,
    unit: f32,
    settings: &Settings,
) -> Furniture {
    let mut taken: HashSet<usize> = HashSet::new();

    // Whatever the document-level pass called repeated: running heads and
    // feet, and watermarks, which sit nowhere near a margin and would
    // otherwise be interleaved into the body.
    for at in ids {
        let geom = &sheet.lines[*at];
        if repeated.contains(&sheet.page.lines[geom.index], sheet.page.height) {
            taken.insert(*at);
        }
    }

    // Page numbers. Peeling rows rather than looking only at the first and the
    // last lets a foot carrying two numbers — one at each margin, which the
    // row merge deliberately keeps apart — come off in one piece.
    let rest: Vec<usize> = ids
        .iter()
        .copied()
        .filter(|at| !taken.contains(at))
        .collect();
    if !rest.is_empty() {
        let rows = make_rows(sheet, &rest, unit, settings);
        let band = settings.furniture_band * sheet.height;
        for row in rows.iter().take(MAX_FURNITURE_ROWS) {
            if row.rect.y1 > band || !is_page_number(sheet, row, settings) {
                break;
            }
            taken.extend(row.lines.iter().copied());
        }
        for row in rows.iter().rev().take(MAX_FURNITURE_ROWS) {
            if row.rect.y0 < sheet.height - band ||
                row.lines.iter().any(|at| taken.contains(at)) ||
                !is_page_number(sheet, row, settings)
            {
                break;
            }
            taken.extend(row.lines.iter().copied());
        }
    }

    let mut top = Vec::new();
    let mut bottom = Vec::new();
    if !taken.is_empty() {
        let mut ids: Vec<usize> = taken.iter().copied().collect();
        ids.sort_unstable();
        for row in make_rows(sheet, &ids, unit, settings) {
            if row.rect.center_y() < sheet.height / 2.0 {
                top.push(row);
            } else {
                bottom.push(row);
            }
        }
    }
    Furniture { taken, top, bottom }
}

/// Whether a row is a page number: one or two narrow boxes, anchored to a
/// margin or to the page's centre line, reading as a numeral.
fn is_page_number(sheet: &Sheet, row: &Row, settings: &Settings) -> bool {
    if row.lines.is_empty() || row.lines.len() > 2 || sheet.width <= 0.0 {
        return false;
    }
    let edge = settings.page_number_edge * sheet.width;
    row.lines.iter().all(|at| {
        let rect = &sheet.lines[*at].rect;
        let narrow = rect.width() < settings.page_number_width * sheet.width;
        let anchored = rect.x0 < edge ||
            sheet.width - rect.x1 < edge ||
            (rect.center_x() - sheet.width / 2.0).abs() < edge;
        let text = sheet.text(*at);
        narrow && anchored && (text.trim().is_empty() || is_numeral(text))
    })
}

/// Whether a string is a bare numeral once the ornaments a folio carries —
/// brackets, dashes, a full stop — are taken off either end. Arabic, Roman,
/// Tibetan and Devanagari digits all count: a folio is numbered in whatever
/// script the page is set in.
fn is_numeral(text: &str) -> bool {
    let core: Vec<char> = text
        .trim_matches(|c: char| c.is_whitespace() || "-–—.[]()".contains(c))
        .chars()
        .collect();
    if core.is_empty() || core.len() > 6 {
        return false;
    }
    core.iter().all(|c| {
        c.is_ascii_digit() ||
            "IVXLCivxlc".contains(*c) ||
            matches!(*c as u32, 0x0F20..=0x0F29 | 0x0966..=0x096F)
    })
}

// ------------------------------------------------------------------------
// The cut
// ------------------------------------------------------------------------

/// A leaf of the cut: a group of lines with no gutter and no band break left
/// inside it.
struct Leaf {
    ids: Vec<usize>,
    /// The extent of the column the leaf was cut out of, which is what a
    /// heading's centring and width are judged against. It has to travel with
    /// the leaf: a band split leaves a centred title alone in a leaf of its
    /// own, and a title measured against its own bounds is neither centred nor
    /// narrow.
    column: Rect,
    /// Whether the group was pulled out of a column split as too small to be a
    /// column and put back by distance.
    floating: bool,
}

/// Cuts a set of lines into leaves, in reading order.
///
/// A column split is tried first and a band split second, which is what makes
/// a two-column page read down each column instead of across the page. The
/// published form of this cut runs on layout blocks, whose boxes have already
/// swallowed the space between their lines, and can therefore split on a
/// one-pixel gap; on raw text lines a one-pixel gap cuts every line into its
/// own column, so both gaps here are multiples of the page's own line height.
fn xy_cut(
    sheet: &Sheet,
    ids: &[usize],
    column: Rect,
    unit: f32,
    settings: &Settings,
) -> Vec<Leaf> {
    if ids.len() <= 1 {
        return vec![Leaf {
            ids: ids.to_vec(),
            column,
            floating: false,
        }];
    }

    let gutter = (settings.gutter_height * unit)
        .max(settings.gutter_width * sheet.width);
    let parts = split_axis(sheet, ids, Axis::X, gutter);
    if parts.len() >= 2 {
        let big: Vec<usize> = (0..parts.len())
            .filter(|at| parts[*at].len() >= settings.min_column_lines)
            .collect();
        if big.len() >= 2 {
            let first = sheet.bounds_of(&parts[big[0]]);
            let last = sheet.bounds_of(&parts[*big.last().expect("a part")]);
            if first.y_overlap(&last) >= settings.column_y_overlap {
                if looks_like_table(sheet, &parts, &big, settings) {
                    // Row-aligned parts are a table: it reads across, not
                    // down, and geometry alone can do no better than that.
                    let mut all = ids.to_vec();
                    all.sort_by(|a, b| {
                        let l = &sheet.lines[*a].rect;
                        let r = &sheet.lines[*b].rect;
                        compare(l.y0, r.y0).then(compare(l.x0, r.x0))
                    });
                    return vec![Leaf {
                        ids: all,
                        column,
                        floating: false,
                    }];
                }
                return parts
                    .iter()
                    .flat_map(|part| {
                        let column = sheet.bounds_of(part);
                        xy_cut(sheet, part, column, unit, settings)
                    })
                    .collect();
            }
        }
        if !big.is_empty() && big.len() < parts.len() {
            // Some parts are too small to be columns: a caption beside the
            // text, a marginal note, a formula number. They come out, the rest
            // is ordered, and each one goes back next to what it sits closest
            // to.
            let kept: Vec<usize> = big
                .iter()
                .flat_map(|at| parts[*at].iter().copied())
                .collect();
            let text_column = sheet.bounds_of(&kept);
            let mut ordered = xy_cut(sheet, &kept, text_column, unit, settings);
            for (at, part) in parts.iter().enumerate() {
                if !big.contains(&at) {
                    let float = Leaf {
                        ids: part.clone(),
                        column: text_column,
                        floating: true,
                    };
                    insert_floating(sheet, float, &mut ordered);
                }
            }
            return ordered;
        }
    }

    let pitch = line_pitch(sheet, ids).unwrap_or(unit);
    let gap = (settings.cut_gap * unit).max(0.55 * pitch);
    let bands = split_axis(sheet, ids, Axis::Y, gap);
    if bands.len() >= 2 {
        return bands
            .iter()
            .flat_map(|band| xy_cut(sheet, band, column, unit, settings))
            .collect();
    }
    vec![Leaf {
        ids: ids.to_vec(),
        column,
        floating: false,
    }]
}

#[derive(Clone, Copy, PartialEq, Eq)]
enum Axis {
    X,
    Y,
}

/// Sorts the lines along one axis and cuts wherever they leave a gap of at
/// least `min_gap`.
///
/// This is the answer a projection profile gives, without allocating an array
/// as long as the page for every level of the recursion.
fn split_axis(
    sheet: &Sheet,
    ids: &[usize],
    axis: Axis,
    min_gap: f32,
) -> Vec<Vec<usize>> {
    let span = |at: &usize| {
        let rect = &sheet.lines[*at].rect;
        match axis {
            Axis::X => (rect.x0, rect.x1),
            Axis::Y => (rect.y0, rect.y1),
        }
    };
    let mut sorted = ids.to_vec();
    sorted.sort_by(|a, b| compare(span(a).0, span(b).0));

    let mut parts: Vec<Vec<usize>> = Vec::new();
    let mut end = f32::NEG_INFINITY;
    for at in sorted {
        let (start, stop) = span(&at);
        if parts.is_empty() || start - end >= min_gap {
            parts.push(Vec::new());
            end = stop;
        } else {
            end = end.max(stop);
        }
        parts.last_mut().expect("a part").push(at);
    }
    parts
}

/// Whether the parts of a column split are the columns of a table.
///
/// A table's entries line up across its columns; the lines of two text columns
/// do not, because each column takes its leading from its own top. This is the
/// only distinction between the two that geometry offers.
fn looks_like_table(
    sheet: &Sheet,
    parts: &[Vec<usize>],
    big: &[usize],
    settings: &Settings,
) -> bool {
    if big.len() < 2 {
        return false;
    }
    let reference = &parts[big[0]];
    let hits = reference
        .iter()
        .filter(|at| {
            let rect = sheet.lines[**at].rect;
            big[1..].iter().all(|other| {
                parts[*other]
                    .iter()
                    .any(|peer| rect.y_overlap(&sheet.lines[*peer].rect) > 0.4)
            })
        })
        .count();
    hits as f32 >= settings.table_alignment * reference.len() as f32
}

/// Puts a floating group back among the ordered ones, next to whichever it
/// sits closest to.
///
/// The cost is dominated by the nearest-edge distance; the vertical and
/// horizontal gaps only break ties, and the gap upwards is discounted so that
/// a caption prefers the block above it to the one below.
fn insert_floating(sheet: &Sheet, float: Leaf, ordered: &mut Vec<Leaf>) {
    if float.ids.is_empty() {
        return;
    }
    if ordered.is_empty() {
        ordered.push(float);
        return;
    }
    let rect = sheet.bounds_of(&float.ids);
    let mut best = 0usize;
    let mut best_cost = f32::INFINITY;
    for (at, leaf) in ordered.iter().enumerate() {
        if leaf.ids.is_empty() {
            continue;
        }
        let cost = float_cost(&rect, &sheet.bounds_of(&leaf.ids));
        if cost < best_cost {
            best_cost = cost;
            best = at;
        }
    }
    let neighbour = sheet.bounds_of(&ordered[best].ids);
    let after = rect.center_y() > neighbour.center_y() ||
        (rect.center_y() == neighbour.center_y() &&
            rect.center_x() >= neighbour.center_x());
    ordered.insert(if after { best + 1 } else { best }, float);
}

fn float_cost(block: &Rect, other: &Rect) -> f32 {
    let up = (block.y0 - other.y1).max(0.0);
    let down = (other.y0 - block.y1).max(0.0);
    let left = (block.x0 - other.x1).max(0.0);
    let right = (other.x0 - block.x1).max(0.0);
    let touching = up == 0.0 && down == 0.0 && left == 0.0 && right == 0.0;
    let edge = if touching {
        0.0
    } else {
        left + right + 0.1 * up + down
    };
    edge * 1e4 + up + left * 1e-4
}

/// The line pitch over a set of lines: the median step from one line's top to
/// the next's.
fn line_pitch(sheet: &Sheet, ids: &[usize]) -> Option<f32> {
    let mut tops: Vec<f32> =
        ids.iter().map(|at| sheet.lines[*at].rect.y0).collect();
    tops.sort_by(|a, b| compare(*a, *b));
    let mut steps: Vec<f32> = tops
        .windows(2)
        .map(|pair| pair[1] - pair[0])
        .filter(|step| *step > 0.0)
        .collect();
    median(&mut steps).filter(|pitch| *pitch > 0.0)
}

// ------------------------------------------------------------------------
// Rows and paragraphs
// ------------------------------------------------------------------------

/// One visual row: the lines that sit side by side on one baseline.
struct Row {
    lines: Vec<usize>,
    rect: Rect,
}

/// Groups lines into rows.
///
/// Two boxes share a row when they overlap vertically *and* are horizontally
/// adjacent. The second half matters: without it a caption at one margin joins
/// the body line at the other and their texts run together.
fn make_rows(
    sheet: &Sheet,
    ids: &[usize],
    unit: f32,
    settings: &Settings,
) -> Vec<Row> {
    let mut sorted = ids.to_vec();
    sorted.sort_by(|a, b| {
        let (l, r) = (&sheet.lines[*a].rect, &sheet.lines[*b].rect);
        compare(l.center_y(), r.center_y()).then(compare(l.x0, r.x0))
    });

    let mut rows: Vec<Row> = Vec::new();
    for at in sorted {
        let rect = sheet.lines[at].rect;
        let joins = rows.last().is_some_and(|row| {
            row.rect.y_overlap(&rect) >= settings.row_overlap &&
                rect.x0 - row.rect.x1 < settings.row_merge_gap * unit
        });
        if joins {
            let row = rows.last_mut().expect("a row");
            row.lines.push(at);
            row.rect.merge(&rect);
        } else {
            rows.push(Row {
                lines: vec![at],
                rect,
            });
        }
    }
    for row in &mut rows {
        row.lines.sort_by(|a, b| {
            compare(sheet.lines[*a].rect.x0, sheet.lines[*b].rect.x0)
        });
    }
    rows
}

/// One paragraph-sized group of rows, with everything the classification needs
/// in order to judge it.
struct Segment {
    lines: Vec<usize>,
    rect: Rect,
    /// The extent of the column the segment was cut out of.
    column: Rect,
    rows: usize,
    /// The segment's own line pitch, when it holds more than one row.
    pitch: Option<f32>,
    /// The median height of its lines.
    height: f32,
    script: Script,
    floating: bool,
    /// The gap to the previous and to the next segment of the same column.
    /// Infinite at a column's ends, where the cut has already found a gap
    /// larger than any left inside a band.
    gap_above: f32,
    gap_below: f32,
    /// How far the segment sits from the end of its column, counting from 0.
    from_end: usize,
    /// The text of the first row, for the leading-token rules.
    lead: String,
}

/// Turns the leaves of the cut into segments: rows first, then paragraphs.
fn segment(
    sheet: &Sheet,
    leaves: &[Leaf],
    unit: f32,
    settings: &Settings,
) -> Vec<Segment> {
    let mut out: Vec<Segment> = Vec::new();
    for leaf in leaves {
        if leaf.ids.is_empty() {
            continue;
        }
        let rows = make_rows(sheet, &leaf.ids, unit, settings);
        let column = leaf.column;
        let groups = split_paragraphs(&rows, &column, unit, settings);

        let first = out.len();
        for group in &groups {
            let mut lines = Vec::new();
            let mut rect = rows[group[0]].rect;
            let mut heights = Vec::new();
            let mut tops = Vec::new();
            for at in group {
                let row = &rows[*at];
                rect.merge(&row.rect);
                tops.push(row.rect.y0);
                for line in &row.lines {
                    lines.push(*line);
                    heights.push(sheet.lines[*line].height);
                }
            }
            let mut steps: Vec<f32> = tops
                .windows(2)
                .map(|pair| pair[1] - pair[0])
                .filter(|step| *step > 0.0)
                .collect();
            let lead: Vec<&str> = rows[group[0]]
                .lines
                .iter()
                .map(|at| sheet.text(*at))
                .collect();
            out.push(Segment {
                rect,
                column,
                rows: group.len(),
                pitch: median(&mut steps).filter(|pitch| *pitch > 0.0),
                height: median(&mut heights).unwrap_or(unit),
                script: dominant_script(sheet, &lines),
                floating: leaf.floating,
                gap_above: f32::INFINITY,
                gap_below: f32::INFINITY,
                from_end: 0,
                lead: lead.join(" "),
                lines,
            });
        }
        let last = out.len();
        for at in first..last {
            out[at].from_end = last - 1 - at;
            if at > first {
                let gap = out[at].rect.y0 - out[at - 1].rect.y1;
                out[at].gap_above = gap;
                out[at - 1].gap_below = gap;
            }
        }
    }
    out
}

/// Cuts a band's rows into paragraphs.
///
/// Extra leading, a first-line indent, or a smaller bump after a line that
/// ended short. Never a jump in box height: adjacent lines of one paragraph
/// routinely differ by half depending on whether they happen to carry a
/// descender, and breaking on that gives a paragraph per line.
fn split_paragraphs(
    rows: &[Row],
    column: &Rect,
    unit: f32,
    settings: &Settings,
) -> Vec<Vec<usize>> {
    if rows.is_empty() {
        return Vec::new();
    }
    let mut steps: Vec<f32> = rows
        .windows(2)
        .map(|pair| pair[1].rect.y0 - pair[0].rect.y0)
        .filter(|step| *step > 0.0)
        .collect();
    let pitch = median(&mut steps).filter(|p| *p > 0.0).unwrap_or(unit);
    let width = column.width().max(1.0);

    let mut groups = vec![vec![0usize]];
    for at in 1..rows.len() {
        let (previous, current) = (&rows[at - 1], &rows[at]);
        let step = current.rect.y0 - previous.rect.y0;
        let ended_short =
            column.x1 - previous.rect.x1 > settings.short_line * width;
        let indented = current.rect.x0 - column.x0 > settings.indent * unit;
        let breaks = step > settings.paragraph_pitch * pitch ||
            indented ||
            (ended_short && step > settings.soft_pitch * pitch);
        if breaks {
            groups.push(Vec::new());
        }
        groups.last_mut().expect("a group").push(at);
    }
    groups
}

/// The script most of a set of lines is written in. A tie goes to the
/// stacking script, and a set of nothing but digits and punctuation comes back
/// [`Script::Neutral`].
fn dominant_script(sheet: &Sheet, ids: &[usize]) -> Script {
    let mut counts: HashMap<Script, usize> = HashMap::new();
    for at in ids {
        *counts.entry(sheet.lines[*at].script).or_default() += 1;
    }
    let mut best = Script::Neutral;
    let mut best_count = 0usize;
    for class in [Script::Stacking, Script::Ideographic, Script::Alphabetic] {
        let count = counts.get(&class).copied().unwrap_or(0);
        if count > best_count {
            best_count = count;
            best = class;
        }
    }
    best
}

// ------------------------------------------------------------------------
// Classification
// ------------------------------------------------------------------------

/// The page-level statistics a segment is judged against.
struct Stats {
    /// The page's median line height.
    unit: f32,
    /// The body's line pitch: the pitch of the largest run of body rows on the
    /// page. This, and no box height, is the page's type-size scale.
    body_pitch: f32,
    page_height: f32,
    /// The script most of the page's lines are written in.
    dominant: Script,
    /// The median line height per script class, so that a size can be compared
    /// within a script even on a page that mixes two.
    heights: HashMap<Script, f32>,
}

impl Stats {
    fn new(sheet: &Sheet, segments: &[Segment], unit: f32) -> Self {
        let all: Vec<usize> = (0..sheet.lines.len()).collect();
        let dominant = dominant_script(sheet, &all);

        let mut per_script: HashMap<Script, Vec<f32>> = HashMap::new();
        for line in &sheet.lines {
            per_script.entry(line.script).or_default().push(line.height);
        }
        let heights = per_script
            .into_iter()
            .filter_map(|(class, mut values)| {
                median(&mut values).map(|value| (class, value))
            })
            .collect();

        // The body pitch is the pitch of the biggest multi-row segment,
        // preferring one that is not written in a stacking script: a page that
        // interleaves two scripts should measure itself against the one its
        // body is set in.
        let pick = |stacking_ok: bool| -> Option<f32> {
            segments
                .iter()
                .filter(|seg| {
                    seg.rows >= 4 &&
                        seg.pitch.is_some() &&
                        (stacking_ok || seg.script != Script::Stacking)
                })
                .max_by_key(|seg| seg.rows)
                .and_then(|seg| seg.pitch)
        };
        let any_pitch = || {
            let mut pitches: Vec<f32> =
                segments.iter().filter_map(|seg| seg.pitch).collect();
            median(&mut pitches)
        };
        let body_pitch = pick(false)
            .or_else(|| pick(true))
            .or_else(any_pitch)
            .filter(|pitch| *pitch > 0.0)
            .unwrap_or(1.3 * unit);

        Self {
            unit,
            body_pitch,
            page_height: sheet.height,
            dominant,
            heights,
        }
    }

    /// The reference line height for a script class, falling back to the
    /// page's own median where the class is barely represented.
    fn height_of(&self, class: Script) -> f32 {
        self.heights
            .get(&class)
            .copied()
            .filter(|height| *height > 0.0)
            .unwrap_or(self.unit)
    }

    /// Whether a segment is tall because of its script rather than its type
    /// size, and must therefore stay out of every size judgement.
    ///
    /// Both halves are needed. The measured one catches an outsized block on
    /// any page; the script one catches a stacking-script line on a page whose
    /// body is alphabetic, where the page's median height already sits near
    /// the alphabetic lines. On a page set wholly in a stacking script the
    /// median is that script's own, so neither half fires and its headings are
    /// still found.
    fn is_tall(&self, seg: &Segment, settings: &Settings) -> bool {
        seg.height > settings.tall_script * self.unit ||
            (seg.script == Script::Stacking &&
                self.dominant != Script::Stacking)
    }

    /// A segment's size as a multiple of the body's: its pitch against the
    /// body pitch, or — for a single-row segment, which has no pitch — its
    /// height against the reference height of its own script. Used only to
    /// rank headings that have already been found, never to find one.
    fn size_of(&self, seg: &Segment) -> f32 {
        match seg.pitch {
            Some(pitch) => pitch / self.body_pitch,
            None => seg.height / self.height_of(seg.script),
        }
    }
}

fn classify(
    seg: &Segment,
    stats: &Stats,
    last_on_page: bool,
    settings: &Settings,
) -> BlockKind {
    if is_footnote(seg, stats, last_on_page, settings) {
        return BlockKind::Footnote;
    }
    if let Some(level) = heading_level(seg, stats, settings) {
        return BlockKind::Heading { level };
    }
    let narrow = seg.rect.width() <
        settings.heading_max_width * seg.column.width().max(1.0);
    if seg.floating && seg.rows <= settings.heading_max_rows && narrow {
        return BlockKind::Caption;
    }
    BlockKind::Paragraph
}

/// A footnote is set smaller than the body, sits low on the page, is separated
/// from it by more than ordinary leading, and comes at the end of its column.
///
/// The pitch margin between an eight-point note and a ten-point body is only
/// about a sixth, so the geometry alone is too thin to act on: a leading
/// marker — a number, or one of the reference symbols that stand in for one —
/// or being the last thing on the page has to confirm it.
fn is_footnote(
    seg: &Segment,
    stats: &Stats,
    last_on_page: bool,
    settings: &Settings,
) -> bool {
    let positioned = seg.rect.y0 > settings.footnote_top * stats.page_height &&
        seg.gap_above >= settings.footnote_gap * stats.body_pitch &&
        seg.from_end < 2;
    if !positioned {
        return false;
    }
    let small = seg.pitch.is_some_and(|pitch| {
        pitch <= settings.footnote_pitch * stats.body_pitch
    });
    footnote_marker(&seg.lead) || (small && last_on_page)
}

/// A heading is centred in its column, or set larger than the body, or
/// numbered — and in every case isolated from what surrounds it.
///
/// It is never found by box height. A line of capitals carries no descenders
/// and comes out shorter than the body around it, so the title of a page can
/// be its smallest box; the fragments of a display formula, meanwhile, come
/// out taller than anything else on the page and are not headings at all.
fn heading_level(
    seg: &Segment,
    stats: &Stats,
    settings: &Settings,
) -> Option<u8> {
    if seg.rows > settings.heading_max_rows || stats.is_tall(seg, settings) {
        return None;
    }
    let isolated = seg.gap_above >= 1.5 * stats.body_pitch &&
        seg.gap_below >= 1.2 * stats.body_pitch;
    if !isolated {
        return None;
    }
    let column = seg.column.width().max(1.0);
    let narrow = seg.rect.width() < settings.heading_max_width * column;
    let centered = narrow &&
        (seg.rect.center_x() - seg.column.center_x()).abs() <
            settings.center_tolerance * column;
    let bigger = seg.pitch.is_some_and(|pitch| {
        pitch >= settings.heading_pitch * stats.body_pitch
    });
    let numbered = leading_number(&seg.lead);

    if centered || bigger {
        // The level here is provisional; the ranking pass replaces it.
        return Some(numbered.unwrap_or(1));
    }
    match numbered {
        Some(depth) if narrow => Some(depth),
        _ => None,
    }
}

/// Ranks the headings found on the page by size, largest first.
///
/// A numbered heading keeps the depth its numbering states — `2.3` sits one
/// level under `2` however the two are set — and every other takes the rank of
/// its size among the distinct heading sizes of the page.
fn assign_levels(segments: &[Segment], kinds: &mut [BlockKind], stats: &Stats) {
    let mut sizes: Vec<f32> = segments
        .iter()
        .zip(kinds.iter())
        .filter(|(_, kind)| matches!(kind, BlockKind::Heading { .. }))
        .map(|(seg, _)| stats.size_of(seg))
        .collect();
    if sizes.is_empty() {
        return;
    }
    sizes.sort_by(|a, b| compare(*b, *a));
    let mut tiers: Vec<f32> = Vec::new();
    for size in sizes {
        let opens_a_tier = tiers
            .last()
            .is_none_or(|top| size < top * (1.0 - LEVEL_TOLERANCE));
        if opens_a_tier {
            tiers.push(size);
        }
    }

    for (seg, kind) in segments.iter().zip(kinds.iter_mut()) {
        let BlockKind::Heading { level } = kind else {
            continue;
        };
        *level = match leading_number(&seg.lead) {
            Some(depth) => depth,
            None => {
                let size = stats.size_of(seg);
                let rank = tiers
                    .iter()
                    .position(|top| size >= top * (1.0 - LEVEL_TOLERANCE))
                    .unwrap_or(tiers.len() - 1);
                (rank + 1).min(MAX_LEVEL as usize) as u8
            },
        };
    }
}

// ------------------------------------------------------------------------
// Leading tokens
// ------------------------------------------------------------------------

/// The longest run of digits one component of a numbering may have. Three,
/// because a section is never the thousandth of anything and a sentence
/// opening with a four-digit year is not a heading.
const MAX_NUMBER_DIGITS: usize = 3;

/// The depth of a heading's numbering, when it carries one: `2` is one deep,
/// `2.3` two, `IV.` and `B)` one. The token has to be closed by a full stop, a
/// bracket or a space.
fn leading_number(text: &str) -> Option<u8> {
    let rest = text.trim_start();
    let first = rest.chars().next()?;

    let (token, at) = if first.is_ascii_digit() {
        let end = rest
            .find(|c: char| !c.is_ascii_digit() && c != '.')
            .unwrap_or(rest.len());
        let token = &rest[..end];
        let long = token.split('.').any(|part| part.len() > MAX_NUMBER_DIGITS);
        if long {
            return None;
        }
        (token, end)
    } else if "IVXLC".contains(first) {
        let end = rest
            .find(|c: char| !"IVXLC".contains(c))
            .unwrap_or(rest.len());
        (&rest[..end], end)
    } else if first.is_ascii_uppercase() {
        (&rest[..1], 1)
    } else {
        return None;
    };

    let closed = match rest[at..].chars().next() {
        Some(c) => c == '.' || c == ')' || c.is_whitespace(),
        None => token.ends_with('.'),
    };
    if !closed {
        return None;
    }
    let depth = token.trim_end_matches('.').matches('.').count() + 1;
    Some(depth.min(MAX_LEVEL as usize) as u8)
}

/// Whether a line opens with a footnote marker: a small number, or one of the
/// reference symbols that take the place of one.
fn footnote_marker(text: &str) -> bool {
    let rest = text.trim_start();
    let Some(first) = rest.chars().next() else {
        return false;
    };
    if "*†‡§¶".contains(first) {
        return true;
    }
    if !first.is_ascii_digit() {
        return false;
    }
    let digits = rest.chars().take_while(|c| c.is_ascii_digit()).count();
    if digits > 3 {
        return false;
    }
    match rest.chars().nth(digits) {
        Some(c) => c == '.' || c == ')' || c.is_whitespace(),
        None => true,
    }
}
