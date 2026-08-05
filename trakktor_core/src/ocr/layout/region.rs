//! What the layout model returns: a rectangle, a label and a confidence.
//!
//! The labels are the published ones, spelled as the model spells them. They
//! are *not* folded into this port's own vocabulary at the network boundary:
//! a label is what the model said, and a caller that wants to show the reader
//! "this is a footnote" must be able to. The mapping from label to role in the
//! rendered document belongs to the consumer, not here.

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
