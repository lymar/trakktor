//! What the layout model returns: a rectangle, a label and a confidence.
//!
//! The labels are the published ones, spelled as the model spells them. They
//! are *not* folded into this port's own vocabulary at the network boundary:
//! a label is what the model said, and a caller that wants to show the reader
//! "this is a footnote" must be able to. The mapping from label to role in the
//! rendered document belongs to the consumer, not here.

#[cfg(test)]
mod tests;

use crate::ocr::page::Quad;

/// One labelled region of a page.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Region {
    pub label: Label,
    /// Confidence in `0.0..=1.0`.
    pub score: f32,
    /// Where it sits, in page pixels. Axis-aligned: this model returns
    /// rectangles, not quadrangles. It is a [`Quad`] anyway so that a block
    /// and a text line can be sorted by the same code.
    pub quad: Quad,
}

impl Region {
    /// The axis-aligned bounds `(x0, y0, x1, y1)`.
    pub fn bounds(&self) -> (f32, f32, f32, f32) { self.quad.bounds() }

    /// Area in square pixels.
    pub fn area(&self) -> f32 {
        let (x0, y0, x1, y1) = self.bounds();
        (x1 - x0).max(0.0) * (y1 - y0).max(0.0)
    }
}

/// How far two boxes must overlap, in pixels of each axis, before a line
/// counts as being in a region at all. The reference's three: a line grazing
/// the edge of the block above it is not part of it.
const MIN_OVERLAP: f32 = 3.0;

/// How much of a line a region must cover to be considered its container
/// rather than merely a box it overlaps.
const CONTAINS: f32 = 0.5;

/// The smallest share of a line's width a region beside its owner has to hold
/// before the line is cut apart at the boundary between them.
///
/// It is a bar against slivers, and the measurements sit far away from it on
/// both sides: a line genuinely glued across the gutter of a two-column page
/// gives the neighbouring column between two fifths and half of its width,
/// while a box merely poking out of the paragraph it belongs to reaches into
/// the next region by a percent or two.
const PIECE: f32 = 0.1;

/// Which region a line belongs to: the **innermost** one that holds it.
///
/// The rule that suggests itself — "the region it overlaps most" — is wrong,
/// and wrong in a way that quietly rearranges a page. The model does not only
/// return the paragraphs of a page; on a page it is unsure of it also returns a
/// box around the *whole* column, and that box overlaps every line completely,
/// while the paragraph box around one line covers only the nine tenths of it
/// that fit inside (the ascenders of a stacking script poke out of every box
/// drawn around it). Most-overlap therefore hands every such line to the
/// column, the column becomes one enormous block, and the reading order
/// collapses into it.
///
/// So: among the regions that cover at least half of the line, the **smallest**
/// wins — a paragraph beats the column that contains it. A line no region
/// covers that far falls back to the region it overlaps most, and a line no
/// region touches belongs to none, which is not an error: it is read by the
/// geometry, as it would have been without a model at all.
///
/// Pictures are never candidates. A line lying over one is a caption or a label
/// printed on it, not part of it.
pub fn owner(line: (f32, f32, f32, f32), regions: &[Region]) -> Option<usize> {
    let area =
        |b: (f32, f32, f32, f32)| (b.2 - b.0).max(0.0) * (b.3 - b.1).max(0.0);
    let line_area = area(line);
    let mut inner: Option<(usize, f32)> = None;
    let mut overlapping: Option<(usize, f32)> = None;

    for (at, region) in regions.iter().enumerate() {
        if region.label.is_pictorial() {
            continue;
        }
        let bounds = region.bounds();
        let width = (line.2.min(bounds.2) - line.0.max(bounds.0)).max(0.0);
        let height = (line.3.min(bounds.3) - line.1.max(bounds.1)).max(0.0);
        if width <= MIN_OVERLAP || height <= MIN_OVERLAP {
            continue;
        }
        let shared = width * height;
        if overlapping.is_none_or(|(_, best)| shared > best) {
            overlapping = Some((at, shared));
        }
        if line_area > 0.0 && shared / line_area >= CONTAINS {
            let own = area(bounds);
            if inner.is_none_or(|(_, smallest)| own < smallest) {
                inner = Some((at, own));
            }
        }
    }
    inner.or(overlapping).map(|(at, _)| at)
}

/// Where a line has to be cut apart, because the box the detector drew for it
/// reaches past its own region into the one beside it.
///
/// A line detector works from ink, not from structure, and where two columns
/// are set close together it will happily join the end of a line in the left
/// column to the line facing it in the right one. The result is a box that
/// covers both, a reading that runs the two sentences together, and — since the
/// box has to belong somewhere — half of one column parked in the middle of the
/// other. The layout model, which does know the structure, has already drawn
/// the boundary the box crossed.
///
/// The pieces come back as the x intervals to cut the line into, in order, or
/// empty when it does not straddle anything. They tile the line: the ends reach
/// its own edges and neighbours meet halfway across the gap between the two
/// regions, so nothing between them is dropped.
///
/// **Only a region standing *beside* the owner can cut a line**, and that is
/// the whole difference from the reference, which cuts a line against every
/// region it falls into. Regions nest — a formula inside a paragraph, a box
/// around a whole column outside it, a footnote overlapping the text above it —
/// and on our own pages a line touches a second region that way about fifteen
/// times in a hundred without a single one of them being a straddle. Cutting
/// there would take an ordinary line of running text and slice an inline
/// formula out of the middle of it.
pub fn split(
    line: (f32, f32, f32, f32),
    regions: &[Region],
) -> Vec<(f32, f32)> {
    let (width, height) = (line.2 - line.0, line.3 - line.1);
    if width <= 0.0 {
        return Vec::new();
    }
    // A line no region holds is not cut, for the same reason it is not dropped:
    // there is nothing to cut it against, and the geometry reads it whole.
    let Some(own) = owner(line, regions) else {
        return Vec::new();
    };
    // A piece has to be worth reading on its own: a tenth of the line, and
    // wider than the line is tall — one character is about that square, and
    // less than a character is a detection artefact, not a column.
    let least = (width * PIECE).max(height);

    // How much of the line a region covers, along x, if it touches it at all.
    let held = |bounds: (f32, f32, f32, f32)| {
        let (from, to) = (line.0.max(bounds.0), line.2.min(bounds.2));
        let tall = (line.3.min(bounds.3) - line.1.max(bounds.1)).max(0.0);
        (to - from > MIN_OVERLAP && tall > MIN_OVERLAP).then_some((from, to))
    };

    let mine = regions[own].bounds();
    let Some(kept) = held(mine).filter(|(from, to)| to - from >= least) else {
        return Vec::new();
    };

    let mut beside: Vec<(f32, f32)> = regions
        .iter()
        .enumerate()
        .filter(|(at, region)| *at != own && !region.label.is_pictorial())
        .filter_map(|(_, region)| {
            let bounds = region.bounds();
            // Beside, not around: a region whose own x range overlaps the
            // owner's is nested in it, holds it, or stands above or below it in
            // the same column — none of which is a boundary a line can cross
            // sideways.
            let shared = bounds.2.min(mine.2) - bounds.0.max(mine.0);
            (shared <= MIN_OVERLAP).then(|| held(bounds)).flatten()
        })
        .filter(|(from, to)| to - from >= least)
        .collect();
    // Widest first, so that when two candidates cover the same stretch of the
    // line the one with more of it in view wins.
    beside.sort_by(|a, b| (b.1 - b.0).total_cmp(&(a.1 - a.0)));

    let mut spans = vec![kept];
    for span in beside {
        let clear = spans
            .iter()
            .all(|kept| span.1.min(kept.1) - span.0.max(kept.0) <= MIN_OVERLAP);
        if clear {
            spans.push(span);
        }
    }
    if spans.len() < 2 {
        return Vec::new();
    }

    spans.sort_by(|a, b| a.0.total_cmp(&b.0));
    let mut pieces = Vec::with_capacity(spans.len());
    let mut from = line.0;
    for pair in spans.windows(2) {
        let boundary = (pair[0].1 + pair[1].0) / 2.0;
        pieces.push((from, boundary));
        from = boundary;
    }
    pieces.push((from, line.2));
    pieces
}

/// The twenty classes `PP-DocLayout_plus-L` distinguishes.
///
/// The order of the variants is the order of the published label list, which is
/// also the order of the class indices the network emits — getting it wrong
/// yields plausible nonsense (`text` where the page has a picture), so the two
/// are tied together in one place and checked against the model's own
/// `config.json` at load time.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Label {
    ParagraphTitle,
    Image,
    Text,
    Number,
    Abstract,
    Content,
    FigureTitle,
    Formula,
    Table,
    Reference,
    DocTitle,
    Footnote,
    Header,
    Algorithm,
    Footer,
    Seal,
    Chart,
    FormulaNumber,
    AsideText,
    ReferenceContent,
}

/// The published names, in class-index order.
pub const LABEL_NAMES: [&str; Label::COUNT] = [
    "paragraph_title",
    "image",
    "text",
    "number",
    "abstract",
    "content",
    "figure_title",
    "formula",
    "table",
    "reference",
    "doc_title",
    "footnote",
    "header",
    "algorithm",
    "footer",
    "seal",
    "chart",
    "formula_number",
    "aside_text",
    "reference_content",
];

/// Every label, in class-index order.
pub const LABELS: [Label; Label::COUNT] = [
    Label::ParagraphTitle,
    Label::Image,
    Label::Text,
    Label::Number,
    Label::Abstract,
    Label::Content,
    Label::FigureTitle,
    Label::Formula,
    Label::Table,
    Label::Reference,
    Label::DocTitle,
    Label::Footnote,
    Label::Header,
    Label::Algorithm,
    Label::Footer,
    Label::Seal,
    Label::Chart,
    Label::FormulaNumber,
    Label::AsideText,
    Label::ReferenceContent,
];

impl Label {
    pub const COUNT: usize = 20;

    /// The class index the network emits for this label.
    pub fn index(self) -> usize {
        LABELS
            .iter()
            .position(|l| *l == self)
            .expect("a known label")
    }

    /// The label a class index stands for.
    pub fn from_index(index: usize) -> Option<Self> {
        LABELS.get(index).copied()
    }

    /// The published name.
    pub fn name(self) -> &'static str { LABEL_NAMES[self.index()] }

    /// Whether the region holds no readable text of its own — a picture, a
    /// chart, a stamp. A recognizer pointed at one of these returns whatever
    /// characters it can find in the ink, which is worse than nothing.
    pub fn is_pictorial(self) -> bool {
        matches!(self, Self::Image | Self::Chart | Self::Seal)
    }

    /// Whether the region belongs to the sheet rather than to the document:
    /// running heads and feet, page numbers, marginal notes.
    pub fn is_furniture(self) -> bool {
        matches!(
            self,
            Self::Header | Self::Footer | Self::Number | Self::AsideText
        )
    }

    /// Whether the region is a heading of some kind.
    pub fn is_heading(self) -> bool {
        matches!(self, Self::DocTitle | Self::ParagraphTitle)
    }
}

impl std::fmt::Display for Label {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(self.name())
    }
}
