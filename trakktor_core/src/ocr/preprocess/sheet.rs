//! Finding the sheet in the frame, and cutting it out.
//!
//! This one is trakktor's own: upstream's document pre-processor has no such
//! step, and the measurements say it needs one. [`UVDoc`](super::unwarp)
//! straightens the *frame*, so it works on a photograph where the page fills
//! the picture and gives up on one where it does not — a sheet lying on a desk
//! at the far end of a wide shot comes out of it still small, still oblique,
//! and read worse than it would have been unstraightened.
//!
//! What the step needs is already in the tree. A page photographed from a hand
//! is a bright convex quadrangle on a darker ground, which is a threshold and
//! a connected component away; and turning four corners into an upright
//! rectangle is the perspective warp that straightens every text line
//! ([`crate::ocr::paddle::crop::straighten`]). So this module is a finder and
//! nothing else — the warp is borrowed.
//!
//! Two properties are worth having in mind:
//!
//! * **The map back is exact.** A perspective warp is a homography, and a
//!   homography inverts in closed form; unlike the unwarper's field, nothing
//!   here is only invertible numerically.
//! * **It declines more often than it fires.** A frame whose bright region is
//!   nearly the whole picture has no sheet to cut out — that is a scan, or a
//!   photograph already filled by the page — and a frame whose brightest region
//!   is a sliver is not a sheet at all. In both cases the step returns nothing
//!   and the page goes on unchanged, which is the behaviour that makes it safe
//!   to leave on.

#[cfg(test)]
mod tests;

use crate::ocr::{
    paddle::{
        crop::{self, source_of},
        image::{CHANNELS, Page},
    },
    page::Quad,
};

/// The width the search runs at. The sheet's outline does not need pixels,
/// and a photograph is millions of them.
const WORKING_WIDTH: usize = 512;

/// How much of the frame the bright region has to cover to be a sheet worth
/// cutting out, and how much is too much.
///
/// The upper bound is what keeps this step from firing on a scan: a page that
/// already fills its picture has nothing around it to remove, and cutting to
/// "the page" would only shave its margins.
const MIN_COVERAGE: f32 = 0.04;
const MAX_COVERAGE: f32 = 0.92;

/// How far outside the found corners the cut is taken, as a fraction of the
/// sheet's size. A page's own margin carries no text, but the finder tends to
/// stop a pixel or two inside the paper, and a cut that clips the first line
/// of a page costs more than a strip of desk.
const MARGIN: f32 = 0.012;

/// A sheet found in a frame, and the cut that straightens it.
#[derive(Debug, Clone)]
pub struct Sheet {
    /// Where the sheet sits in the photograph, clockwise from the top left.
    pub quad: Quad,
    /// The straightened sheet's size.
    pub width: usize,
    pub height: usize,
    /// Destination to source: a point of the straightened sheet, back onto the
    /// photograph.
    map: [f64; 9],
}

impl Sheet {
    /// Where a point of the straightened sheet sits in the photograph.
    pub fn source(&self, x: f32, y: f32) -> (f32, f32) {
        let (sx, sy) = source_of(&self.map, f64::from(x), f64::from(y));
        (sx as f32, sy as f32)
    }

    /// The straightened sheet.
    pub fn apply(&self, page: &Page) -> Page {
        let (cut, _) = crop::straighten(
            &page.bgr,
            page.width as usize,
            page.height as usize,
            &self.quad,
        );
        Page {
            width: cut.width as u32,
            height: cut.height as u32,
            bgr: cut.bgr,
        }
    }
}

/// Finds the sheet in a frame, or decides there is nothing to cut out.
pub fn find(page: &Page) -> Option<Sheet> {
    let (width, height) = (page.width as usize, page.height as usize);
    if width < 16 || height < 16 {
        return None;
    }
    let scale = (width as f32 / WORKING_WIDTH as f32).max(1.0);
    let (small_w, small_h) = (
        (width as f32 / scale).round().max(1.0) as usize,
        (height as f32 / scale).round().max(1.0) as usize,
    );

    let grey = luminance(page, small_w, small_h);
    let level = otsu(&grey);
    let mask: Vec<bool> = grey.iter().map(|value| *value > level).collect();
    let region = largest_region(&mask, small_w, small_h)?;

    let coverage = region.len() as f32 / (small_w * small_h) as f32;
    if !(MIN_COVERAGE..=MAX_COVERAGE).contains(&coverage) {
        return None;
    }

    // Sides first, extremes only if the sides cannot be made out: a corner
    // that fell outside the frame has to be reconstructed, and only the sides
    // know where it was.
    let quad = sides(&region, small_w, small_h)
        .unwrap_or_else(|| corners(&region, small_w));
    build(page, scaled(quad, scale))
}

/// Scales a quadrangle from the working image back to the page.
fn scaled(quad: Quad, scale: f32) -> Quad {
    let mut points = quad.points;
    for point in &mut points {
        *point = (point.0 * scale, point.1 * scale);
    }
    Quad::new(points)
}

/// Turns four corners into a sheet, widening the cut by [`MARGIN`] and
/// refusing anything degenerate.
fn build(page: &Page, quad: Quad) -> Option<Sheet> {
    let centre = quad.points.iter().fold((0.0f32, 0.0f32), |acc, point| {
        (acc.0 + point.0 / 4.0, acc.1 + point.1 / 4.0)
    });
    let mut points = quad.points;
    for point in &mut points {
        point.0 = centre.0 + (point.0 - centre.0) * (1.0 + MARGIN);
        point.1 = centre.1 + (point.1 - centre.1) * (1.0 + MARGIN);
    }
    let quad = Quad::new(points);

    let (cut, map) = crop::straighten(
        &page.bgr,
        page.width as usize,
        page.height as usize,
        &quad,
    );
    let map = map?;
    if cut.width < 16 || cut.height < 16 {
        return None;
    }
    Some(Sheet {
        quad,
        width: cut.width,
        height: cut.height,
        map,
    })
}

/// The frame in grey, scaled down, as the search sees it.
///
/// Nearest-neighbour on purpose: the outline of a sheet is a step several
/// hundred pixels long, and smoothing it costs time without moving it.
fn luminance(page: &Page, width: usize, height: usize) -> Vec<u8> {
    let (source_w, source_h) = (page.width as usize, page.height as usize);
    let row = source_w * CHANNELS;
    let mut grey = vec![0u8; width * height];
    for y in 0..height {
        let sy = (y * source_h / height).min(source_h - 1);
        for x in 0..width {
            let sx = (x * source_w / width).min(source_w - 1);
            let at = sy * row + sx * CHANNELS;
            // Blue, green, red — the page's own order — with the usual
            // luma weights.
            let value = 0.114 * f32::from(page.bgr[at]) +
                0.587 * f32::from(page.bgr[at + 1]) +
                0.299 * f32::from(page.bgr[at + 2]);
            grey[y * width + x] = value.round().clamp(0.0, 255.0) as u8;
        }
    }
    grey
}

/// Otsu's threshold: the level that splits the histogram into two classes of
/// least combined variance.
fn otsu(grey: &[u8]) -> u8 {
    let mut histogram = [0u32; 256];
    for value in grey {
        histogram[*value as usize] += 1;
    }
    let total: f64 = grey.len() as f64;
    let sum: f64 = histogram
        .iter()
        .enumerate()
        .map(|(value, count)| value as f64 * f64::from(*count))
        .sum();

    let (mut below, mut weight_below) = (0.0f64, 0.0f64);
    let (mut best, mut best_level) = (-1.0f64, 0u8);
    for level in 0..256 {
        weight_below += f64::from(histogram[level]);
        if weight_below == 0.0 {
            continue;
        }
        let weight_above = total - weight_below;
        if weight_above == 0.0 {
            break;
        }
        below += level as f64 * f64::from(histogram[level]);
        let mean_below = below / weight_below;
        let mean_above = (sum - below) / weight_above;
        let spread =
            weight_below * weight_above * (mean_below - mean_above).powi(2);
        if spread > best {
            best = spread;
            best_level = level as u8;
        }
    }
    best_level
}

/// The largest four-connected run of set pixels, as a list of positions.
fn largest_region(
    mask: &[bool],
    width: usize,
    height: usize,
) -> Option<Vec<usize>> {
    let mut seen = vec![false; mask.len()];
    let mut best: Option<Vec<usize>> = None;
    let mut stack = Vec::new();
    for start in 0..mask.len() {
        if seen[start] || !mask[start] {
            continue;
        }
        let mut region = Vec::new();
        stack.push(start);
        seen[start] = true;
        while let Some(at) = stack.pop() {
            region.push(at);
            let (x, y) = (at % width, at / width);
            let mut visit = |nx: usize, ny: usize, stack: &mut Vec<usize>| {
                let next = ny * width + nx;
                if !seen[next] && mask[next] {
                    seen[next] = true;
                    stack.push(next);
                }
            };
            if x > 0 {
                visit(x - 1, y, &mut stack);
            }
            if x + 1 < width {
                visit(x + 1, y, &mut stack);
            }
            if y > 0 {
                visit(x, y - 1, &mut stack);
            }
            if y + 1 < height {
                visit(x, y + 1, &mut stack);
            }
        }
        if best.as_ref().is_none_or(|found| region.len() > found.len()) {
            best = Some(region);
        }
    }
    best
}

/// The sheet's four corners, found by fitting its sides and intersecting them.
///
/// This is what makes a **tightly framed** photograph work. A page shot from
/// close up often runs out of the picture at one corner, and then that corner
/// simply is not in the image: the brightest region's own extreme point sits
/// on the frame's edge, several hundred pixels from where the paper's corner
/// really was. Cutting to that point shears the page — and shearing a page
/// pushes the far side of it out of the cut, which is text lost from a
/// photograph that had it.
///
/// The sides, on the other hand, are visible almost to the end, and two lines
/// meet whether or not their meeting point was photographed. So: convex hull,
/// simplified to a handful of vertices, edges lying along the frame's border
/// thrown away as artefacts of the cut rather than sides of the sheet, and the
/// four longest of what remains intersected in order.
///
/// `None` when the outline does not look like four sides — a torn sheet, two
/// pages read as one, a region that is not a sheet at all — and then the
/// caller falls back to [`corners`].
fn sides(region: &[usize], width: usize, height: usize) -> Option<Quad> {
    let hull = hull(outline(region, width, height));
    let polygon = simplify(&hull, (width.min(height) as f32) * 0.012);
    if polygon.len() < 4 {
        return None;
    }

    // An edge whose ends both sit on one border of the frame is where the
    // picture ended, not where the paper did.
    let border = |point: (f32, f32)| {
        [
            point.0 <= 1.5,
            point.0 >= width as f32 - 2.5,
            point.1 <= 1.5,
            point.1 >= height as f32 - 2.5,
        ]
    };
    let mut kept: Vec<usize> = Vec::with_capacity(polygon.len());
    for at in 0..polygon.len() {
        let (from, to) = (polygon[at], polygon[(at + 1) % polygon.len()]);
        let (a, b) = (border(from), border(to));
        if (0..4).any(|side| a[side] && b[side]) {
            continue;
        }
        kept.push(at);
    }
    if kept.len() < 4 {
        return None;
    }

    // The four longest of what is left, back in the order they run round the
    // outline — consecutive sides have to stay consecutive to be intersected.
    let length = |at: usize| {
        let (from, to) = (polygon[at], polygon[(at + 1) % polygon.len()]);
        (to.0 - from.0).hypot(to.1 - from.1)
    };
    kept.sort_by(|a, b| length(*b).total_cmp(&length(*a)));
    kept.truncate(4);
    kept.sort_unstable();

    let mut found = [(0.0f32, 0.0f32); 4];
    for at in 0..4 {
        let first = kept[at];
        let second = kept[(at + 1) % 4];
        found[at] = meet(
            (polygon[first], polygon[(first + 1) % polygon.len()]),
            (polygon[second], polygon[(second + 1) % polygon.len()]),
        )?;
    }

    // A reconstructed corner may legitimately sit outside the frame, but not
    // far outside: a pair of nearly parallel sides meets somewhere useless,
    // and the arithmetic will not say so.
    let reach = (width as f32).hypot(height as f32);
    if found.iter().any(|point| {
        point.0 < -reach ||
            point.1 < -reach ||
            point.0 > width as f32 + reach ||
            point.1 > height as f32 + reach
    }) {
        return None;
    }
    Some(clockwise(found))
}

/// Where two lines, each given by two points on it, cross.
fn meet(
    first: ((f32, f32), (f32, f32)),
    second: ((f32, f32), (f32, f32)),
) -> Option<(f32, f32)> {
    let (p, r) = (first.0, (first.1.0 - first.0.0, first.1.1 - first.0.1));
    let (q, s) = (second.0, (second.1.0 - second.0.0, second.1.1 - second.0.1));
    let cross = r.0 * s.1 - r.1 * s.0;
    let scale = r.0.hypot(r.1) * s.0.hypot(s.1);
    // Sides of a quadrangle meet at a decent angle; anything shallower is two
    // stretches of the same side and its crossing point is noise.
    if scale <= f32::EPSILON || (cross / scale).abs() < 0.08 {
        return None;
    }
    let t = ((q.0 - p.0) * s.1 - (q.1 - p.1) * s.0) / cross;
    Some((p.0 + r.0 * t, p.1 + r.1 * t))
}

/// Four corners in the order the domain keeps them: clockwise from the top
/// left, on a screen where `y` runs downwards.
fn clockwise(mut points: [(f32, f32); 4]) -> Quad {
    let centre = points.iter().fold((0.0f32, 0.0f32), |acc, point| {
        (acc.0 + point.0 / 4.0, acc.1 + point.1 / 4.0)
    });
    points.sort_by(|a, b| {
        let angle = |p: &(f32, f32)| (p.1 - centre.1).atan2(p.0 - centre.0);
        angle(a).total_cmp(&angle(b))
    });
    let first = (0..4)
        .min_by(|a, b| {
            let sum = |i: usize| points[i].0 + points[i].1;
            sum(*a).total_cmp(&sum(*b))
        })
        .unwrap_or(0);
    Quad::new([
        points[first],
        points[(first + 1) % 4],
        points[(first + 2) % 4],
        points[(first + 3) % 4],
    ])
}

/// The points the hull can possibly pass through: the extreme pixel of every
/// row and of every column. A few thousand instead of a few hundred thousand,
/// and no vertex of the hull is among the ones left out.
fn outline(region: &[usize], width: usize, height: usize) -> Vec<(f32, f32)> {
    let mut left = vec![usize::MAX; height];
    let mut right = vec![0usize; height];
    let mut top = vec![usize::MAX; width];
    let mut bottom = vec![0usize; width];
    for at in region {
        let (x, y) = (at % width, at / width);
        left[y] = left[y].min(x);
        right[y] = right[y].max(x);
        top[x] = top[x].min(y);
        bottom[x] = bottom[x].max(y);
    }
    let mut points = Vec::with_capacity(2 * (width + height));
    for y in 0..height {
        if left[y] != usize::MAX {
            points.push((left[y] as f32, y as f32));
            points.push((right[y] as f32, y as f32));
        }
    }
    for x in 0..width {
        if top[x] != usize::MAX {
            points.push((x as f32, top[x] as f32));
            points.push((x as f32, bottom[x] as f32));
        }
    }
    points
}

/// The convex hull, by the monotone chain, in the order it runs round.
fn hull(mut points: Vec<(f32, f32)>) -> Vec<(f32, f32)> {
    points.sort_by(|a, b| a.0.total_cmp(&b.0).then(a.1.total_cmp(&b.1)));
    points.dedup();
    if points.len() < 3 {
        return points;
    }
    let turn = |o: (f32, f32), a: (f32, f32), b: (f32, f32)| {
        (a.0 - o.0) * (b.1 - o.1) - (a.1 - o.1) * (b.0 - o.0)
    };
    let mut chain: Vec<(f32, f32)> = Vec::with_capacity(points.len() * 2);
    for pass in 0..2 {
        let start = chain.len();
        let ordered: Vec<(f32, f32)> = if pass == 0 {
            points.clone()
        } else {
            points.iter().rev().copied().collect()
        };
        for point in ordered {
            while chain.len() >= start + 2 &&
                turn(chain[chain.len() - 2], chain[chain.len() - 1], point) <=
                    0.0
            {
                chain.pop();
            }
            chain.push(point);
        }
        chain.pop();
    }
    chain
}

/// Drops the vertices a polygon does not need: the one that bulges least is
/// removed until every remaining one carries more than `tolerance`, or only
/// four are left.
fn simplify(polygon: &[(f32, f32)], tolerance: f32) -> Vec<(f32, f32)> {
    let mut points = polygon.to_vec();
    while points.len() > 4 {
        let mut worst = (f32::MAX, 0usize);
        for at in 0..points.len() {
            let before = points[(at + points.len() - 1) % points.len()];
            let after = points[(at + 1) % points.len()];
            let point = points[at];
            let span = (after.0 - before.0).hypot(after.1 - before.1);
            let bulge = if span <= f32::EPSILON {
                0.0
            } else {
                ((after.0 - before.0) * (before.1 - point.1) -
                    (before.0 - point.0) * (after.1 - before.1))
                    .abs() /
                    span
            };
            if bulge < worst.0 {
                worst = (bulge, at);
            }
        }
        if worst.0 > tolerance {
            break;
        }
        points.remove(worst.1);
    }
    points
}

/// The four corners of a convex region, taken along its diagonals.
///
/// The fallback, for an outline whose sides could not be made out. The
/// extremes of `x + y` and `x - y` are the corners of any quadrangle whose
/// sides are not far from the axes, which a photographed page usually is — but
/// only while all four of its corners are in the picture, which is the
/// limitation [`sides`] exists to lift.
fn corners(region: &[usize], width: usize) -> Quad {
    let mut sum = (f32::MAX, f32::MIN, 0usize, 0usize);
    let mut difference = (f32::MAX, f32::MIN, 0usize, 0usize);
    for at in region {
        let (x, y) = ((at % width) as f32, (at / width) as f32);
        if x + y < sum.0 {
            sum.0 = x + y;
            sum.2 = *at;
        }
        if x + y > sum.1 {
            sum.1 = x + y;
            sum.3 = *at;
        }
        if x - y < difference.0 {
            difference.0 = x - y;
            difference.2 = *at;
        }
        if x - y > difference.1 {
            difference.1 = x - y;
            difference.3 = *at;
        }
    }
    let point = |at: usize| ((at % width) as f32, (at / width) as f32);
    // Clockwise from the top left, which is the order every quadrangle in the
    // domain is in.
    Quad::new([
        point(sum.2),
        point(difference.3),
        point(sum.3),
        point(difference.2),
    ])
}
