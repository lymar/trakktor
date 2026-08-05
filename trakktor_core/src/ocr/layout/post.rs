//! What happens to the network's three hundred boxes before anyone sees them.
//!
//! The network already decodes, scores and sorts its own output — that part is
//! inside the exported graph. What is left is box bookkeeping, and it is
//! reproduced in the reference's order, because the order is load-bearing:
//! suppression runs before the containment filters, and the widening runs after
//! them.
//!
//! **The thresholds and the merge modes are not the model's.** Its own
//! description carries a single `draw_threshold: 0.5` and nothing else; the
//! per-class thresholds, the two suppression thresholds, the modes below and
//! the widening ratio all come from the *pipeline* configuration upstream ships
//! alongside, and they can change without the weights changing. They are
//! written down here, and in the maintenance notes, as this port's constants.

#[cfg(test)]
mod tests;

use super::region::{Label, Region};
use crate::ocr::page::Quad;

/// One box as it comes out of the network.
#[derive(Debug, Clone, Copy)]
pub struct Raw {
    pub class: usize,
    pub score: f32,
    pub bounds: (f32, f32, f32, f32),
}

/// The knobs of this stage.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Settings {
    /// A single score floor replacing every per-class one. `None` keeps the
    /// per-class defaults.
    pub threshold: Option<f32>,
    /// Suppress a box overlapping a kept one of the **same** class by this.
    pub nms_same: f32,
    /// …and of a **different** class by this. Nearly one: two different kinds
    /// of block genuinely do sit on top of each other (a formula inside a
    /// paragraph, a caption inside a figure), and only a near-duplicate is a
    /// mistake.
    pub nms_other: f32,
    /// Widening of every box about its own centre, per axis.
    pub unclip: (f32, f32),
}

/// The per-class score floors, in class-index order.
///
/// Three classes sit below the general half: a section heading, a formula and
/// running text. The last is the one that matters most, and it is easy to think
/// it should not: the model is *sure* of a paragraph of English (0.97 and up)
/// and much less sure of the same paragraph in a script it has not seen much of
/// (0.40 to 0.69 on the Tibetan pages of the measurement set). At a flat half —
/// which is what the model's own description carries — those pages come back
/// nearly empty; at the pipeline's floors they mark up properly. The floors are
/// therefore not a formality but the difference between the stage working on a
/// page and returning nothing.
const THRESHOLDS: [f32; Label::COUNT] = [
    0.3,  // paragraph_title
    0.5,  // image
    0.4,  // text
    0.5,  // number
    0.5,  // abstract
    0.5,  // content
    0.5,  // figure_title
    0.3,  // formula
    0.5,  // table
    0.5,  // reference
    0.5,  // doc_title
    0.5,  // footnote
    0.5,  // header
    0.5,  // algorithm
    0.5,  // footer
    0.45, // seal
    0.5,  // chart
    0.5,  // formula_number
    0.5,  // aside_text
    0.5,  // reference_content
];

/// Classes that swallow what sits inside them. Everything else keeps its
/// nested boxes: a figure caption inside a figure is two blocks, not one.
const SWALLOWS: [Label; 4] = [
    Label::ParagraphTitle,
    Label::Image,
    Label::Formula,
    Label::Chart,
];

/// How much of a box must lie inside another before the outer one swallows it.
const CONTAINMENT: f32 = 0.9;

/// A picture covering this much of the page is the scan itself — a border, a
/// shadow, a full-bleed background — and not an illustration. The bar is lower
/// for a landscape page, where a genuine full-width picture is rarer.
const GIANT_LANDSCAPE: f32 = 0.82;
const GIANT_PORTRAIT: f32 = 0.93;

const NMS_SAME: f32 = 0.6;
const NMS_OTHER: f32 = 0.98;

impl Default for Settings {
    fn default() -> Self {
        Self {
            threshold: None,
            nms_same: NMS_SAME,
            nms_other: NMS_OTHER,
            unclip: (1.0, 1.0),
        }
    }
}

impl Settings {
    /// The floor a class must clear.
    fn floor(&self, class: usize) -> f32 {
        self.threshold
            .unwrap_or_else(|| THRESHOLDS.get(class).copied().unwrap_or(0.5))
    }

    /// The floor a label must clear.
    pub fn floor_for(&self, label: Label) -> f32 { self.floor(label.index()) }
}

/// Turns the network's boxes into the page's regions.
pub fn regions(
    raw: &[Raw],
    page: (u32, u32),
    settings: &Settings,
) -> Vec<Region> {
    let kept: Vec<Raw> = raw
        .iter()
        .copied()
        .filter(|box_| box_.score > settings.floor(box_.class))
        .collect();
    let kept = suppress(&kept, settings);
    let kept = drop_giant_pictures(kept, page);
    let kept = drop_swallowed(kept);

    kept.into_iter()
        .filter_map(|box_| {
            let label = Label::from_index(box_.class)?;
            let (x0, y0, x1, y1) = widen(box_.bounds, settings.unclip);
            let x0 = x0.max(0.0);
            let y0 = y0.max(0.0);
            let x1 = x1.min(page.0 as f32);
            let y1 = y1.min(page.1 as f32);
            if x1 <= x0 || y1 <= y0 {
                return None;
            }
            Some(Region {
                label,
                score: box_.score,
                quad: Quad::new([(x0, y0), (x1, y0), (x1, y1), (x0, y1)]),
            })
        })
        .collect()
}

/// Greedy non-maximum suppression, by score, with **two** thresholds.
///
/// The boxes arrive sorted by score — the graph's own top-k did that — but the
/// sort is redone here rather than assumed: the per-class score floor above is
/// free to reorder nothing, and a caller feeding boxes from somewhere else
/// should still get suppression.
fn suppress(boxes: &[Raw], settings: &Settings) -> Vec<Raw> {
    let mut order: Vec<usize> = (0..boxes.len()).collect();
    order.sort_by(|a, b| boxes[*b].score.total_cmp(&boxes[*a].score));

    let mut kept: Vec<Raw> = Vec::new();
    let mut alive = vec![true; boxes.len()];
    for at in order {
        if !alive[at] {
            continue;
        }
        let current = boxes[at];
        kept.push(current);
        for (other, box_) in boxes.iter().enumerate() {
            if !alive[other] || other == at {
                continue;
            }
            let limit = if box_.class == current.class {
                settings.nms_same
            } else {
                settings.nms_other
            };
            if iou(current.bounds, box_.bounds) >= limit {
                alive[other] = false;
            }
        }
    }
    kept
}

/// Intersection over union, with the reference's `+1` on every side: a
/// one-pixel box has an area of one, not zero.
fn iou(a: (f32, f32, f32, f32), b: (f32, f32, f32, f32)) -> f32 {
    let width = (a.2.min(b.2) - a.0.max(b.0) + 1.0).max(0.0);
    let height = (a.3.min(b.3) - a.1.max(b.1) + 1.0).max(0.0);
    let intersection = width * height;
    let area = |r: (f32, f32, f32, f32)| (r.2 - r.0 + 1.0) * (r.3 - r.1 + 1.0);
    let union = area(a) + area(b) - intersection;
    if union <= 0.0 {
        0.0
    } else {
        intersection / union
    }
}

/// How much of `inner` lies inside `outer`, as a fraction of `inner`'s area.
fn containment(
    inner: (f32, f32, f32, f32),
    outer: (f32, f32, f32, f32),
) -> f32 {
    let area = (inner.2 - inner.0) * (inner.3 - inner.1);
    if area <= 0.0 {
        return 0.0;
    }
    let width = (inner.2.min(outer.2) - inner.0.max(outer.0)).max(0.0);
    let height = (inner.3.min(outer.3) - inner.1.max(outer.1)).max(0.0);
    width * height / area
}

/// Drops an `image` box that covers most of the page.
///
/// If that would empty the page the filter gives up: a page that really is one
/// picture should come back as one picture.
fn drop_giant_pictures(boxes: Vec<Raw>, page: (u32, u32)) -> Vec<Raw> {
    if boxes.len() < 2 {
        return boxes;
    }
    let (width, height) = (page.0 as f32, page.1 as f32);
    let limit = if width > height {
        GIANT_LANDSCAPE
    } else {
        GIANT_PORTRAIT
    } * width *
        height;
    let picture = Label::Image.index();
    let kept: Vec<Raw> = boxes
        .iter()
        .copied()
        .filter(|box_| {
            if box_.class != picture {
                return true;
            }
            let x0 = box_.bounds.0.max(0.0);
            let y0 = box_.bounds.1.max(0.0);
            let x1 = box_.bounds.2.min(width);
            let y1 = box_.bounds.3.min(height);
            (x1 - x0).max(0.0) * (y1 - y0).max(0.0) <= limit
        })
        .collect();
    if kept.is_empty() { boxes } else { kept }
}

/// Drops a box that sits inside a box of a class that swallows.
///
/// One exception, and it is the reference's: a formula is never swallowed by
/// anything that is not itself a formula. A display formula lies inside the
/// paragraph that introduces it far more often than it is a duplicate of it.
fn drop_swallowed(boxes: Vec<Raw>) -> Vec<Raw> {
    let swallows: Vec<usize> = SWALLOWS.iter().map(|l| l.index()).collect();
    let formula = Label::Formula.index();
    let eaten: Vec<bool> = boxes
        .iter()
        .enumerate()
        .map(|(at, inner)| {
            boxes.iter().enumerate().any(|(other, outer)| {
                other != at &&
                    swallows.contains(&outer.class) &&
                    !(inner.class == formula && outer.class != formula) &&
                    containment(inner.bounds, outer.bounds) >= CONTAINMENT
            })
        })
        .collect();
    boxes
        .into_iter()
        .zip(eaten)
        .filter(|(_, eaten)| !*eaten)
        .map(|(box_, _)| box_)
        .collect()
}

/// Scales a box about its own centre.
fn widen(
    bounds: (f32, f32, f32, f32),
    ratio: (f32, f32),
) -> (f32, f32, f32, f32) {
    let (x0, y0, x1, y1) = bounds;
    let width = (x1 - x0) * ratio.0;
    let height = (y1 - y0) * ratio.1;
    let cx = (x0 + x1) / 2.0;
    let cy = (y0 + y1) / 2.0;
    (
        cx - width / 2.0,
        cy - height / 2.0,
        cx + width / 2.0,
        cy + height / 2.0,
    )
}
