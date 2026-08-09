//! Differentiable-binarization post-processing: the detector's probability
//! map becomes one quadrangle per text line.
//!
//! The map is thresholded into a bitmap, the bitmap is traced into contours
//! (see [`contour`]), each contour is reduced to its minimum-area rectangle
//! and scored — against the *probability* map, never the bitmap — then
//! expanded outwards, because a DB map marks a shrunken core of each text
//! line rather than its full extent. What survives is rescaled from the map's
//! grid onto the source page.
//!
//! Two conventions run through the whole stage and are easy to get subtly
//! wrong:
//!
//! - **Integer pixel-index space, with no half-pixel offset anywhere.** Contour
//!   points are pixel indices, a rectangle's size is `max - min` (so a
//!   six-pixel-wide blob is five wide), and the scoring mask fills its boundary
//!   inclusively. Adopting a "pixel centre at `x + 0.5`" rasterizer instead
//!   shrinks every scoring mask by about a row and a column, which biases every
//!   score and flips boxes that sit near `box_thresh`.
//! - **The scoring mask is OpenCV's `fillPoly`**, which is the union of the
//!   polygon's Bresenham outline and its strictly-interior even-odd fill — not
//!   a point-in-polygon test. The outline ring is worth several percent of the
//!   mask.
//!
//! The expansion step runs on an integer lattice: the rectangle's corners are
//! truncated to integers before offsetting and the offset ring comes back as
//! integers, so the sub-pixel part of a corner is discarded on the way in.
//! That is the upstream behaviour, and the rescale that follows quantizes to
//! whole source pixels anyway.
//!
//! [`contour`] supplies the two primitives this module builds on:
//! `find_contours`, which traces a bitmap into boundary chains in OpenCV's
//! `RETR_LIST` order (outer borders *and* hole borders — a hole is a box
//! candidate like any other), and `mini_box`, which fits a minimum-area
//! rectangle to a point set and returns its four corners in the order this
//! stage wants, together with the length of its shorter side.

pub mod contour;

#[cfg(test)]
mod tests;

use clipper2_rust::{
    clipper::inflate_paths_64,
    core::{Path64, Point64},
    offset::{EndType, JoinType},
};

use self::contour::{find_contours, mini_box};
use crate::ocr::{paddle::config::DbParams, page::Quad};

/// How a candidate is scored against the probability map.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ScoreMode {
    /// Average the probability over the candidate's minimum-area rectangle.
    /// Cheap, and the published default.
    Fast,
    /// Average it over the traced contour itself. For a slanted or curved
    /// line the enclosing rectangle holds a lot of background, so the two
    /// disagree sharply there — a diagonal line can score `0.34` fast and
    /// `0.82` slow, i.e. be dropped by the default threshold under one mode
    /// and kept under the other.
    Slow,
}

/// Post-processing parameters.
///
/// The defaults are the ones the PP-OCRv5 pipeline ships with. Note
/// `unclip_ratio`: the pipeline's own code carries `2.0` as a fallback, but the
/// configuration it ships — and every published detector's own description —
/// says `1.5`, so `1.5` is what a run actually uses. Getting this wrong is not
/// a crash but a systematic one: every box comes out a few pixels wider on
/// every side, which still reads but crops neighbouring lines into each other.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Params {
    /// Probability above which a pixel is text. Compared strictly.
    pub thresh: f32,
    /// Mean probability a candidate must reach to be kept.
    pub box_thresh: f32,
    /// Upper bound on the number of contours considered. The cut keeps the
    /// contours the tracer returns first.
    pub max_candidates: usize,
    /// How far the fitted rectangle is expanded, as a multiple of its
    /// area-to-perimeter ratio.
    pub unclip_ratio: f32,
    /// Whether to grow the bitmap by one pixel before tracing it.
    pub use_dilation: bool,
    pub score_mode: ScoreMode,
}

impl Default for Params {
    fn default() -> Self {
        Self {
            thresh: 0.3,
            box_thresh: 0.6,
            max_candidates: 1000,
            unclip_ratio: 1.5,
            use_dilation: false,
            score_mode: ScoreMode::Fast,
        }
    }
}

/// The thresholds a run names, if it names any.
///
/// What is left unset comes from **the detector's own description**, not from
/// the values above. That is not a nicety: a generation calibrates its
/// probability map with its post-processing, and the newest one asks for 0.2
/// where the older ones ask for 0.3 and for a smaller expansion of the box it
/// fits. Running one generation's map through another's thresholds finds a
/// different set of lines and says nothing about it. Upstream's own runtime
/// does the same thing — its thresholds default to nothing and are filled in
/// from the model.
#[derive(Debug, Clone, Copy, Default, PartialEq)]
pub struct Thresholds {
    pub thresh: Option<f32>,
    pub box_thresh: Option<f32>,
    pub unclip_ratio: Option<f32>,
}

impl Params {
    /// The parameters a run works with: the detector's declared ones, with
    /// whatever the caller named laid over them.
    pub fn resolved(published: &DbParams, named: Thresholds) -> Self {
        Self {
            thresh: named.thresh.unwrap_or(published.thresh),
            box_thresh: named.box_thresh.unwrap_or(published.box_thresh),
            max_candidates: published.max_candidates,
            unclip_ratio: named.unclip_ratio.unwrap_or(published.unclip_ratio),
            ..Self::default()
        }
    }
}

/// Shortest side a fitted rectangle must have before it is worth scoring.
const MIN_SIZE: f32 = 3.0;

/// Shortest side the *expanded* rectangle must have. The expansion adds at
/// least a pixel on each side, so the bar is two higher than [`MIN_SIZE`].
const MIN_SIZE_UNCLIPPED: f32 = MIN_SIZE + 2.0;

/// Turns one probability map into the text-line quadrangles of the source
/// page, each with the mean probability that got it accepted.
///
/// `prob` is the map in row-major order, `width` × `height`; it is read as
/// probabilities, so it must be the network's output and not the thresholded
/// bitmap. `source` is the size `(width, height)` of the page the map was
/// computed from — the *original* page, before any padding the resize stage
/// applied — and the boxes come back in its pixel coordinates. The map is
/// not assumed to be the same size as the page: the rescale goes straight
/// from one grid to the other.
///
/// `ratio` is the resize stage's `(height, width)` scale factors. The
/// differentiable-binarization path does not use them — they exist for the
/// other detector heads, which rescale through the resized geometry instead
/// of straight to the page — but they travel with every detector output, so
/// the parameter keeps the call sites uniform.
///
/// The order of the result follows the order the contour tracer returns, and
/// is not a reading order.
pub fn boxes_from_bitmap(
    prob: &[f32],
    width: usize,
    height: usize,
    source: (u32, u32),
    _ratio: (f32, f32),
    params: &Params,
) -> Vec<(Quad, f32)> {
    let (page_w, page_h) = (source.0 as f32, source.1 as f32);
    candidates(prob, width, height, source, params)
        .into_iter()
        .filter_map(|(corners, score)| {
            let corners =
                clip_to_page(order_points_clockwise(corners), page_w, page_h);
            // A box thinner than four pixels on either side carries no
            // readable text; the sides are truncated to whole pixels before
            // the comparison, so "three or less" means "under four".
            let [tl, tr, _, bl] = corners;
            if side(tl, tr) <= 3.0 || side(tl, bl) <= 3.0 {
                return None;
            }
            Some((Quad::new(corners), score as f32))
        })
        .collect()
}

/// The boxes before the page-level filtering: fitted, scored, expanded,
/// re-fitted and rescaled onto the source grid, in contour order.
fn candidates(
    prob: &[f32],
    width: usize,
    height: usize,
    source: (u32, u32),
    params: &Params,
) -> Vec<([(f32, f32); 4], f64)> {
    if width == 0 ||
        height == 0 ||
        source.0 == 0 ||
        source.1 == 0 ||
        prob.len() < width * height
    {
        return Vec::new();
    }

    let mut mask: Vec<bool> = prob.iter().map(|&v| v > params.thresh).collect();
    if params.use_dilation {
        mask = dilate(&mask, width, height);
    }

    let box_thresh = f64::from(params.box_thresh);
    let mut boxes = Vec::new();
    for traced in find_contours(&mask, width, height)
        .iter()
        .take(params.max_candidates)
    {
        let Some((corners, sside)) = mini_box(&traced.points) else {
            continue;
        };
        if sside < MIN_SIZE {
            continue;
        }
        let score = match params.score_mode {
            ScoreMode::Fast => score_quad(prob, width, height, &corners),
            ScoreMode::Slow => {
                score_contour(prob, width, height, &traced.points)
            },
        };
        if box_thresh > score {
            continue;
        }
        let Some(ring) = unclip(&corners, params.unclip_ratio) else {
            continue;
        };
        let Some((expanded, sside)) = mini_box(&ring) else {
            continue;
        };
        if sside < MIN_SIZE_UNCLIPPED {
            continue;
        }
        boxes.push((rescale(&expanded, width, height, source), score));
    }
    boxes
}

/// Grows the bitmap by one pixel toward `+x` and `+y`.
///
/// This is a 2×2 kernel anchored at its bottom-right element, which is what
/// the upstream dilation option uses — deliberately asymmetric. A symmetric
/// 3×3 dilation is not a substitute: it grows in all eight directions and
/// merges blobs this one leaves apart.
fn dilate(mask: &[bool], width: usize, height: usize) -> Vec<bool> {
    let mut out = vec![false; mask.len()];
    for y in 0..height {
        for x in 0..width {
            let at = y * width + x;
            out[at] = mask[at] ||
                (y > 0 && mask[at - width]) ||
                (x > 0 && mask[at - 1]) ||
                (y > 0 && x > 0 && mask[at - width - 1]);
        }
    }
    out
}

/// Mean probability over the candidate's own rectangle.
fn score_quad(
    prob: &[f32],
    width: usize,
    height: usize,
    corners: &[(f32, f32); 4],
) -> f64 {
    let xs = corners.map(|p| p.0);
    let ys = corners.map(|p| p.1);
    // The bounds are floored/ceiled first and clamped afterwards, and both
    // ends clamp to the last valid index — the high end to `dim - 1`, not to
    // `dim`.
    let xmin = clamp_index(fold_min(&xs).floor(), width);
    let xmax = clamp_index(fold_max(&xs).ceil(), width);
    let ymin = clamp_index(fold_min(&ys).floor(), height);
    let ymax = clamp_index(fold_max(&ys).ceil(), height);
    // Shifted into mask-local coordinates while still fractional, then
    // truncated toward zero. Truncation only differs from flooring when a
    // coordinate stayed negative, which happens when the bounds clamped at
    // zero while the box reached off the map.
    let polygon: Vec<(i32, i32)> = corners
        .iter()
        .map(|&(x, y)| ((x - xmin as f32) as i32, (y - ymin as f32) as i32))
        .collect();
    masked_mean(prob, width, (xmin, ymin), (xmax, ymax), &polygon)
}

/// Mean probability over the traced contour itself.
fn score_contour(
    prob: &[f32],
    width: usize,
    height: usize,
    points: &[(i32, i32)],
) -> f64 {
    if points.is_empty() {
        return 0.0;
    }
    // The contour is already integral, so the bounds are plain extremes.
    let mut xmin = i32::MAX;
    let mut xmax = i32::MIN;
    let mut ymin = i32::MAX;
    let mut ymax = i32::MIN;
    for &(x, y) in points {
        xmin = xmin.min(x);
        xmax = xmax.max(x);
        ymin = ymin.min(y);
        ymax = ymax.max(y);
    }
    let xmin = xmin.clamp(0, width as i32 - 1) as usize;
    let xmax = xmax.clamp(0, width as i32 - 1) as usize;
    let ymin = ymin.clamp(0, height as i32 - 1) as usize;
    let ymax = ymax.clamp(0, height as i32 - 1) as usize;
    let polygon: Vec<(i32, i32)> = points
        .iter()
        .map(|&(x, y)| (x - xmin as i32, y - ymin as i32))
        .collect();
    masked_mean(prob, width, (xmin, ymin), (xmax, ymax), &polygon)
}

/// Mean of `prob` over the filled `polygon`, inside the inclusive window
/// `min..=max`. An empty mask means zero, not a division by zero.
fn masked_mean(
    prob: &[f32],
    width: usize,
    min: (usize, usize),
    max: (usize, usize),
    polygon: &[(i32, i32)],
) -> f64 {
    let (xmin, ymin) = min;
    let (xmax, ymax) = max;
    let (mask_w, mask_h) = (xmax + 1 - xmin, ymax + 1 - ymin);
    let mask = fill_poly(polygon, mask_w, mask_h);
    // Accumulated in f64 in row-major order: the reference sums a float32
    // window in double precision, and matching the order keeps the last bits
    // from drifting on large masks.
    let mut sum = 0.0f64;
    let mut count = 0usize;
    for y in 0..mask_h {
        for x in 0..mask_w {
            if mask[y * mask_w + x] {
                sum += f64::from(prob[(ymin + y) * width + xmin + x]);
                count += 1;
            }
        }
    }
    if count == 0 { 0.0 } else { sum / count as f64 }
}

/// Rasterizes a closed polygon the way OpenCV's `fillPoly` does: the
/// eight-connected outline of every edge, plus the strict interior found by
/// an even-odd scanline with half-open edge spans.
///
/// The outline pass is what makes a degenerate polygon behave: a polygon
/// collapsed to a horizontal segment still fills that segment, and a repeated
/// single point still fills one pixel, so a degenerate candidate gets a real
/// score instead of an empty mask.
///
/// A polygon may reach outside the mask — the scoring window is clamped to
/// the map, so a box that runs off the page keeps corners the window cannot
/// hold — and the two passes handle that differently, exactly as OpenCV does:
/// the interior spans are simply cut at the border, while the outline is
/// re-anchored by [`clip_line`] first and then walked.
fn fill_poly(polygon: &[(i32, i32)], width: usize, height: usize) -> Vec<bool> {
    let mut mask = vec![false; width * height];
    if polygon.is_empty() || width == 0 || height == 0 {
        return mask;
    }
    let n = polygon.len();

    let outside = |p: (i32, i32)| {
        p.0 < 0 || p.1 < 0 || p.0 >= width as i32 || p.1 >= height as i32
    };
    for i in 0..n {
        let mut from = polygon[i];
        let mut to = polygon[(i + 1) % n];
        if (outside(from) || outside(to)) &&
            !clip_line(width as i32, height as i32, &mut from, &mut to)
        {
            continue;
        }
        line8(from, to, |x, y| {
            if x >= 0 && y >= 0 && (x as usize) < width && (y as usize) < height
            {
                mask[y as usize * width + x as usize] = true;
            }
        });
    }

    let ymin = polygon.iter().map(|p| p.1).min().expect("a vertex");
    let ymax = polygon.iter().map(|p| p.1).max().expect("a vertex");
    let mut crossings: Vec<(i64, i64)> = Vec::with_capacity(n);
    for y in ymin.max(0)..=ymax.min(height as i32 - 1) {
        crossings.clear();
        for i in 0..n {
            let (ax, ay) = polygon[i];
            let (bx, by) = polygon[(i + 1) % n];
            // Horizontal edges never contribute a crossing — the outline pass
            // already drew them — and the span is half-open in y, so a vertex
            // shared by two edges is counted once.
            if ay == by || y < ay.min(by) || y >= ay.max(by) {
                continue;
            }
            // The crossing is kept as an exact rational: rounding it to a
            // float first would move a span end by a pixel whenever the true
            // crossing lands on an integer.
            let (mut num, mut den) = (
                i64::from(ax) * i64::from(by - ay) +
                    i64::from(y - ay) * i64::from(bx - ax),
                i64::from(by - ay),
            );
            if den < 0 {
                num = -num;
                den = -den;
            }
            crossings.push((num, den));
        }
        crossings.sort_by(|a, b| {
            (i128::from(a.0) * i128::from(b.1))
                .cmp(&(i128::from(b.0) * i128::from(a.1)))
        });
        for span in crossings.chunks(2) {
            let [left, right] = span else { break };
            let from = ceil_ratio(*left).max(0);
            let to = floor_ratio(*right).min(width as i64 - 1);
            for x in from..=to {
                mask[y as usize * width + x as usize] = true;
            }
        }
    }
    mask
}

/// `ceil(num / den)` for a positive denominator.
fn ceil_ratio((num, den): (i64, i64)) -> i64 { -((-num).div_euclid(den)) }

/// `floor(num / den)` for a positive denominator.
fn floor_ratio((num, den): (i64, i64)) -> i64 { num.div_euclid(den) }

/// Moves the ends of a segment onto the raster, returning whether anything
/// of it is left.
///
/// This is not the same as walking the whole segment and dropping the pixels
/// that fall outside: the walk restarts from the moved end, and the truncated
/// slope it then follows steps on a slightly different set of pixels. The
/// difference is worth a percent of the scoring mask for a slanted box that
/// runs off the edge of the map, which is enough to move a score by more than
/// the width of a threshold decision.
///
/// The endpoint is pushed to the horizontal border first and to the vertical
/// one second, and each intersection is truncated toward zero — order and
/// rounding both matter, because they decide which pixel the walk starts on.
fn clip_line(
    width: i32,
    height: i32,
    from: &mut (i32, i32),
    to: &mut (i32, i32),
) -> bool {
    if width <= 0 || height <= 0 {
        return false;
    }
    let (right, bottom) = (i64::from(width) - 1, i64::from(height) - 1);
    let (mut x1, mut y1) = (i64::from(from.0), i64::from(from.1));
    let (mut x2, mut y2) = (i64::from(to.0), i64::from(to.1));
    // Bit 0 is "left of the raster", 1 is "right of it", 2 is "above", 3 is
    // "below". Both ends sharing a bit means the segment misses entirely.
    let region = |x: i64, y: i64| {
        i32::from(x < 0) +
            i32::from(x > right) * 2 +
            i32::from(y < 0) * 4 +
            i32::from(y > bottom) * 8
    };
    let mut c1 = region(x1, y1);
    let mut c2 = region(x2, y2);
    if (c1 & c2) == 0 && (c1 | c2) != 0 {
        if c1 & 12 != 0 {
            let edge = if c1 < 8 { 0 } else { bottom };
            x1 += ((edge - y1) as f64 * (x2 - x1) as f64 / (y2 - y1) as f64)
                as i64;
            y1 = edge;
            c1 = i32::from(x1 < 0) + i32::from(x1 > right) * 2;
        }
        if c2 & 12 != 0 {
            let edge = if c2 < 8 { 0 } else { bottom };
            x2 += ((edge - y2) as f64 * (x1 - x2) as f64 / (y1 - y2) as f64)
                as i64;
            y2 = edge;
            c2 = i32::from(x2 < 0) + i32::from(x2 > right) * 2;
        }
        if (c1 & c2) == 0 && (c1 | c2) != 0 {
            if c1 != 0 {
                let edge = if c1 == 1 { 0 } else { right };
                y1 += ((edge - x1) as f64 * (y2 - y1) as f64 / (x2 - x1) as f64)
                    as i64;
                x1 = edge;
                c1 = 0;
            }
            if c2 != 0 {
                let edge = if c2 == 1 { 0 } else { right };
                y2 += ((edge - x2) as f64 * (y1 - y2) as f64 / (x1 - x2) as f64)
                    as i64;
                x2 = edge;
                c2 = 0;
            }
        }
        *from = (x1 as i32, y1 as i32);
        *to = (x2 as i32, y2 as i32);
    }
    (c1 | c2) == 0
}

/// Walks the eight-connected Bresenham line between two pixels, left to
/// right, and hands every pixel to `emit`.
fn line8(from: (i32, i32), to: (i32, i32), mut emit: impl FnMut(i32, i32)) {
    // The traversal is normalized left-to-right, so the pixels of a segment
    // do not depend on which end it was given from.
    let (start, end) = if from.0 > to.0 {
        (to, from)
    } else {
        (from, to)
    };
    let (mut major, mut minor) = (end.0 - start.0, end.1 - start.1);
    let (step_x, step_y) = (major.signum(), minor.signum());
    major = major.abs();
    minor = minor.abs();
    let (major_x, major_y, minor_x, minor_y) = if minor > major {
        core::mem::swap(&mut major, &mut minor);
        (0, step_y, step_x, 0)
    } else {
        (step_x, 0, 0, step_y)
    };

    let (mut x, mut y) = start;
    let mut error = major - 2 * minor;
    for _ in 0..=major {
        emit(x, y);
        let sidestep = error < 0;
        error += -2 * minor + if sidestep { 2 * major } else { 0 };
        x += major_x + if sidestep { minor_x } else { 0 };
        y += major_y + if sidestep { minor_y } else { 0 };
    }
}

/// Expands a fitted rectangle outwards by `area * ratio / perimeter`, and
/// returns the resulting ring.
///
/// The area and the perimeter are measured on the rectangle's own fractional
/// corners, but the corners handed to the offsetter are truncated toward
/// zero: the offset itself runs on an integer lattice and gives integers
/// back. An expansion that splits into several rings means the candidate was
/// not a single region, and the candidate is dropped rather than repaired.
fn unclip(corners: &[(f32, f32); 4], ratio: f32) -> Option<Vec<(i32, i32)>> {
    let points = corners.map(|(x, y)| (f64::from(x), f64::from(y)));
    let mut twice_area = 0.0f64;
    let mut perimeter = 0.0f64;
    for i in 0..4 {
        let (from, to) = (points[i], points[(i + 1) % 4]);
        twice_area += from.0 * to.1 - to.0 * from.1;
        let (dx, dy) = (to.0 - from.0, to.1 - from.1);
        perimeter += (dx * dx + dy * dy).sqrt();
    }
    if perimeter.is_nan() || perimeter <= f64::EPSILON {
        return None;
    }
    let delta = (twice_area * 0.5).abs() * f64::from(ratio) / perimeter;

    let path: Path64 = points
        .iter()
        .map(|&(x, y)| Point64 {
            x: x as i64,
            y: y as i64,
        })
        .collect();
    // The round joins are resolved to a quarter-pixel chord tolerance, capped
    // at the offset distance itself; leaving the tolerance at zero would ask
    // for arcs about two orders of magnitude finer than intended.
    let arc_tolerance = 0.25f64.min(0.25 * delta.abs());
    let rings = inflate_paths_64(
        &vec![path],
        delta,
        JoinType::Round,
        EndType::Polygon,
        2.0,
        arc_tolerance,
    );
    let [ring] = rings.as_slice() else {
        return None;
    };
    ring.iter()
        .map(|p| Some((i32::try_from(p.x).ok()?, i32::try_from(p.y).ok()?)))
        .collect()
}

/// Maps the map-grid corners onto the source page.
///
/// The division stays in the map's own precision and the multiplication is
/// done in double precision, matching the reference's dtype chain; the
/// rounding is ties-to-even, not ties-away-from-zero. The clamp is inclusive
/// of the page size, so a coordinate may land one past the last column here —
/// [`clip_to_page`] is what pulls it back onto the page.
fn rescale(
    corners: &[(f32, f32); 4],
    width: usize,
    height: usize,
    source: (u32, u32),
) -> [(f32, f32); 4] {
    let (page_w, page_h) = (f64::from(source.0), f64::from(source.1));
    corners.map(|(x, y)| {
        let x = f64::from(x / width as f32) * page_w;
        let y = f64::from(y / height as f32) * page_h;
        (
            x.round_ties_even().clamp(0.0, page_w) as f32,
            y.round_ties_even().clamp(0.0, page_h) as f32,
        )
    })
}

/// Reorders four corners into "clockwise from the top-left".
///
/// The corner with the smallest `x + y` is the top-left and the one with the
/// largest is the bottom-right; of the remaining two, the smaller `y - x` is
/// the top-right. This is *not* a no-op after the rectangle fit: for a box
/// rotated past 45° the fit's own order is a different rotation of the same
/// cycle, and the crop that follows depends on which corner comes first.
///
/// Ties take the first candidate, and the degenerate case where one corner is
/// both the smallest and the largest leaves three candidates for the other
/// two slots — which is harmless, since all four corners then coincide.
fn order_points_clockwise(corners: [(f32, f32); 4]) -> [(f32, f32); 4] {
    let sums = corners.map(|(x, y)| x + y);
    let first = argmin(&sums);
    let third = argmax(&sums);
    let rest: Vec<(f32, f32)> = (0..4)
        .filter(|i| *i != first && *i != third)
        .map(|i| corners[i])
        .collect();
    let diffs: Vec<f32> = rest.iter().map(|(x, y)| y - x).collect();
    let second = argmin(&diffs);
    let fourth = argmax(&diffs);
    [corners[first], rest[second], corners[third], rest[fourth]]
}

/// Pulls every corner onto the page: `0 ..= size - 1`, truncated to whole
/// pixels.
fn clip_to_page(
    corners: [(f32, f32); 4],
    width: f32,
    height: f32,
) -> [(f32, f32); 4] {
    corners.map(|(x, y)| {
        (
            x.max(0.0).min(width - 1.0).trunc(),
            y.max(0.0).min(height - 1.0).trunc(),
        )
    })
}

/// Length of a box side, truncated to whole pixels.
fn side(from: (f32, f32), to: (f32, f32)) -> f64 {
    let (dx, dy) = (f64::from(to.0 - from.0), f64::from(to.1 - from.1));
    (dx * dx + dy * dy).sqrt().trunc()
}

/// Index of the smallest value; ties take the first.
fn argmin(values: &[f32]) -> usize {
    let mut best = 0;
    for (i, value) in values.iter().enumerate() {
        if *value < values[best] {
            best = i;
        }
    }
    best
}

/// Index of the largest value; ties take the first.
fn argmax(values: &[f32]) -> usize {
    let mut best = 0;
    for (i, value) in values.iter().enumerate() {
        if *value > values[best] {
            best = i;
        }
    }
    best
}

fn fold_min(values: &[f32; 4]) -> f32 {
    values.iter().copied().fold(f32::INFINITY, f32::min)
}

fn fold_max(values: &[f32; 4]) -> f32 {
    values.iter().copied().fold(f32::NEG_INFINITY, f32::max)
}

/// Clamps a map coordinate to a valid index of a `dim`-long axis.
fn clamp_index(value: f32, dim: usize) -> usize {
    (value as i64).clamp(0, dim as i64 - 1) as usize
}
