//! The engine-independent result shape: a page, its text lines, and the
//! quadrangle each line sits in.
//!
//! A run always produces a *sequence* of pages, one per input image, even for
//! a single file — a scanned document is normally several pages, and the
//! caller should not have to special-case the count. Page indices are
//! one-based, matching how people number the pages of a document.

/// A quadrangle in page pixels, corners clockwise from the top-left.
///
/// Text lines are not axis-aligned in general (a scan is never quite
/// straight), so a box is four points rather than a rectangle.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Quad {
    pub points: [(f32, f32); 4],
}

impl Quad {
    pub fn new(points: [(f32, f32); 4]) -> Self { Self { points } }

    /// The axis-aligned bounds `(x0, y0, x1, y1)`.
    pub fn bounds(&self) -> (f32, f32, f32, f32) {
        let xs = self.points.map(|p| p.0);
        let ys = self.points.map(|p| p.1);
        (
            xs.iter().copied().fold(f32::INFINITY, f32::min),
            ys.iter().copied().fold(f32::INFINITY, f32::min),
            xs.iter().copied().fold(f32::NEG_INFINITY, f32::max),
            ys.iter().copied().fold(f32::NEG_INFINITY, f32::max),
        )
    }

    /// Mid-height of the quadrangle, the vertical position a reading order
    /// sorts on.
    pub fn center_y(&self) -> f32 {
        let (_, y0, _, y1) = self.bounds();
        (y0 + y1) / 2.0
    }

    /// Height of the left edge — the type size of the line, as far as the
    /// geometry can tell.
    pub fn height(&self) -> f32 {
        let [tl, _, _, bl] = self.points;
        ((tl.0 - bl.0).powi(2) + (tl.1 - bl.1).powi(2)).sqrt()
    }

    /// Length of the top edge.
    pub fn width(&self) -> f32 {
        let [tl, tr, _, _] = self.points;
        ((tl.0 - tr.0).powi(2) + (tl.1 - tr.1).powi(2)).sqrt()
    }
}

/// One recognized text line.
#[derive(Debug, Clone)]
pub struct Line {
    /// The recognized text.
    pub text: String,
    /// Mean probability of the characters that were kept, in `0.0..=1.0`.
    pub score: f32,
    /// Where the line sits on the page.
    pub quad: Quad,
    /// Whether the text-line orientation classifier turned this crop around
    /// before reading it.
    pub rotated: bool,
    /// Whether the reading this line came from was **cut short** — the
    /// generative engine started repeating itself and the answer was trimmed
    /// back to where the repetition began.
    ///
    /// It is a property of a reading, not of a line, but it is reported per
    /// line because a line is what the caller has. Always false for an engine
    /// that classifies characters: that kind of recognizer cannot run away.
    pub truncated: bool,
}

/// One page of a run.
#[derive(Debug, Clone)]
pub struct Page {
    /// One-based page number within the run.
    pub number: usize,
    /// The file this page came from.
    pub source: String,
    /// Page size in pixels.
    pub width: u32,
    pub height: u32,
    /// The lines, in reading order.
    pub lines: Vec<Line>,
    /// Whether these lines are the **reader's** own, rather than the rows the
    /// page was printed in.
    ///
    /// An engine that classifies characters returns one line per detected row,
    /// and a paragraph has to be put back together out of them — that is what
    /// the Markdown assembly is for. A generative engine reflows the text
    /// itself: it joins words broken across a printed line and runs the rows
    /// of a paragraph together, so the lines it *does* return are lines it
    /// kept apart on purpose — a verse, a gloss under it, a list.
    ///
    /// Reflowing those a second time runs the verse into a paragraph, which is
    /// why the two cases are told apart here rather than guessed at
    /// downstream.
    pub reflowed: bool,
}

impl Page {
    /// The page's text, one line per line, in reading order.
    pub fn text(&self) -> String {
        let mut text = String::new();
        for line in &self.lines {
            text.push_str(&line.text);
            text.push('\n');
        }
        text
    }
}
