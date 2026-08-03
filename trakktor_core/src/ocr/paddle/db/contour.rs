//! Contours of the binarized map, and the rectangles that bound them.
//!
//! Two primitives sit between the detector's probability map and the boxes it
//! yields, and both are written out here rather than taken from a
//! general-purpose crate: the choices such a crate is free to make — what
//! order the contours come back in, where a closed border starts, whether a
//! rectangle's corners are rounded to whole pixels — are precisely the ones
//! the port has to match.
//!
//! [`find_contours`] is Suzuki-Abe border following over the mask, in the two
//! modes the pipeline asks of OpenCV: every border, outer and hole alike, flat
//! and unnested (a hole is a box candidate like any other), with only the
//! vertices where the walk changes direction kept — straight runs collapse to
//! their two endpoints. Three properties of it are load-bearing:
//!
//! * **The order is the reverse of the raster scan's discovery order.** The
//!   caller caps the candidate list by truncating it, so the order decides
//!   which borders survive on a busy page, and it decides the order the boxes
//!   come out in.
//! * **The scan runs over a mask padded with one background pixel on every
//!   side.** Without the pad, a component that touches column 0 is entered from
//!   a different neighbour and its point list starts at a different vertex.
//! * **Points are inclusive boundary-pixel indices**, so the extent of a run is
//!   `max - min` — one less than the number of pixels it covers. The size
//!   filters downstream are calibrated to that, not to a pixel count.
//!
//! [`mini_box`] is the minimum-area rectangle of a point set: rotating
//! calipers over the convex hull, in `f64`, with the corners handed back in
//! the order the box builder wants them and with the length of the
//! rectangle's shorter side. Keeping the corners in floating point is the
//! whole point — the short side is what both size filters test, and a
//! rectangle rounded outward to whole pixels passes filters the real one
//! fails.

#[cfg(test)]
mod tests;

use std::cmp::Ordering;

/// A connected-component contour in mask pixel coordinates.
///
/// The points are boundary pixels — inclusive `(x, y)` indices into the mask —
/// walked once around a closed loop, with the interior of every straight run
/// dropped. Outer borders wind one way and hole borders the other; the
/// detector scores and expands both the same way, so which one this is is not
/// recorded.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Contour {
    pub points: Vec<(i32, i32)>,
}

/// The eight neighbours in the order border following steps through them.
/// Index 0 is `+x`; the index rises counter-clockwise on screen, which — the
/// mask's `y` growing downwards — is clockwise in the usual orientation.
const NEIGHBOURS: [(i32, i32); 8] = [
    (1, 0),
    (1, -1),
    (0, -1),
    (-1, -1),
    (-1, 0),
    (-1, 1),
    (0, 1),
    (1, 1),
];

/// Every border of every connected component of `mask`, outer and hole alike.
///
/// `mask` is row-major and `true` means foreground. Panics if its length does
/// not match `width * height`, which would be a caller bug rather than bad
/// input.
pub fn find_contours(
    mask: &[bool],
    width: usize,
    height: usize,
) -> Vec<Contour> {
    assert_eq!(
        mask.len(),
        width * height,
        "a {width}x{height} mask must hold {} pixels",
        width * height
    );
    if width == 0 || height == 0 {
        return Vec::new();
    }

    // The scan works on a copy fringed with one background pixel, so that
    // every foreground pixel has all eight neighbours and a component touching
    // the mask's own edge is entered exactly as an interior one would be.
    //
    // The copy doubles as the algorithm's scratch state: 0 is background, 1 is
    // foreground not yet walked, a value above 1 marks a walked pixel, and a
    // negative one marks a walked pixel that has background to its right —
    // which is what keeps the scan from re-entering an outer border as if it
    // were the start of a hole.
    let stride = width + 2;
    let mut labels = vec![0i32; stride * (height + 2)];
    for y in 0..height {
        for x in 0..width {
            if mask[y * width + x] {
                labels[(y + 1) * stride + x + 1] = 1;
            }
        }
    }

    let mut contours = Vec::new();
    let mut border = 1i32;
    for y in 1..=height {
        for x in 1..=width {
            let at = y * stride + x;
            let here = labels[at];
            // A border starts where a row crosses into a component that has
            // not been walked (an outer border) or crosses out of one that
            // still has background on its right (a hole border).
            let hole = if here == 1 && labels[at - 1] == 0 {
                false
            } else if here >= 1 && labels[at + 1] == 0 {
                true
            } else {
                continue;
            };
            border += 1;
            let mut points =
                follow(&mut labels, stride, (x as i32, y as i32), hole, border);
            for point in &mut points {
                point.0 -= 1;
                point.1 -= 1;
            }
            contours.push(Contour { points });
        }
    }

    // The flat contour list comes back in the reverse of the order the raster
    // scan discovers the borders in: last found, first returned.
    contours.reverse();
    contours
}

/// Walks one border from `start`, marking the pixels it visits, and returns
/// its vertices in padded coordinates.
fn follow(
    labels: &mut [i32],
    stride: usize,
    start: (i32, i32),
    hole: bool,
    border: i32,
) -> Vec<(i32, i32)> {
    let index = |p: (i32, i32)| p.1 as usize * stride + p.0 as usize;
    let step = |p: (i32, i32), dir: usize| {
        let (dx, dy) = NEIGHBOURS[dir];
        (p.0 + dx, p.1 + dy)
    };

    // The walk is entered from the neighbour the scan crossed: the left one
    // for an outer border, the right one for a hole border. That neighbour is
    // background by construction, so the first foreground pixel clockwise from
    // it is where the loop leaves; coming all the way back around to the entry
    // means the component is a single pixel.
    let entry = if hole { 0 } else { 4 };
    let mut dir = entry;
    loop {
        dir = (dir + 7) & 7;
        if dir == entry || labels[index(step(start, dir))] != 0 {
            break;
        }
    }
    if dir == entry {
        labels[index(start)] = -border;
        return vec![start];
    }
    let entered_at = step(start, dir);

    let mut points = Vec::new();
    let mut here = start;
    // The walk behaves as though it had arrived at the start pixel heading
    // away from that first neighbour, so a start pixel that merely continues
    // the run it closes on is dropped like any other interior point.
    let mut left_in = dir ^ 4;
    loop {
        // Turn counter-clockwise from the neighbour the walk arrived from —
        // one past it — until the next foreground pixel. A border pixel always
        // has one, so the sweep never runs past a full turn.
        let from = dir;
        let mut probe = from + 1;
        let next = loop {
            let candidate = step(here, probe & 7);
            if labels[index(candidate)] != 0 {
                break candidate;
            }
            probe += 1;
            debug_assert!(probe <= from + 8, "a border pixel has a neighbour");
        };
        let to = probe & 7;

        // If the sweep passed the right-hand neighbour and found it
        // background, this pixel is a right edge of the component; the
        // negative mark is what later stops the raster scan from reading it as
        // a hole start. Otherwise the pixel is claimed for this border, unless
        // an earlier border claimed it first.
        if to != 0 && to <= from {
            labels[index(here)] = -border;
        } else if labels[index(here)] == 1 {
            labels[index(here)] = border;
        }

        // A vertex is kept only where the direction the walk leaves in
        // changes, which reduces every straight or diagonal run to its two
        // endpoints.
        if left_in != to {
            points.push(here);
            left_in = to;
        }

        // The loop closes when the walk is about to re-enter the start pixel
        // from the very pixel it first left it for.
        if next == start && here == entered_at {
            break;
        }
        here = next;
        dir = (to + 4) & 7;
    }
    points
}

/// The minimum-area rectangle of a point set, with its corners in the order
/// the box builder expects and the length of its shorter side.
///
/// The corners come out left-most pair first and, within each pair, upper
/// corner before lower one: for a near-axis-aligned rectangle that is
/// top-left, top-right, bottom-right, bottom-left, and in general a clockwise
/// quadrangle that starts on the left. `None` only for an empty point set.
///
/// The rule reads nothing but the corners' coordinates, so for any rectangle
/// whose four corners have four distinct `x` it does not matter which corner
/// the rectangle was built from. Where two corners share an `x` — a square
/// standing on its corner, and nothing else — the sort falls back on the order
/// [`min_area_rect`] emitted them in, which is why that order is the
/// reference's and not a convenient one. A shape with several equally minimal
/// orientations is likewise settled arbitrarily, by the first hull edge to
/// reach the minimum. Both cases need an exact tie in real arithmetic, so they
/// are confined to tiny near-square blobs; a text line's rectangle has one
/// best orientation and four distinct corner columns.
///
/// The coordinates are expected to be page-scale. The exact whole-number
/// arithmetic underneath multiplies pairs of them, so it holds comfortably for
/// anything the detector produces — the map it works on is capped at a few
/// thousand pixels a side — but not for the full range of an `i32`.
pub fn mini_box(points: &[(i32, i32)]) -> Option<([(f32, f32); 4], f32)> {
    let (corners, short_side) = min_area_rect(points)?;

    let mut sorted = corners;
    // Ties keep the order the corners came in, and `-0.0` has to compare equal
    // to `0.0` here — an expanded box may legitimately carry both.
    sorted.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(Ordering::Equal));
    let (first, fourth) = if sorted[1].1 > sorted[0].1 {
        (0, 1)
    } else {
        (1, 0)
    };
    let (second, third) = if sorted[3].1 > sorted[2].1 {
        (2, 3)
    } else {
        (3, 2)
    };

    let corner = |i: usize| (sorted[i].0 as f32, sorted[i].1 as f32);
    Some((
        [corner(first), corner(second), corner(third), corner(fourth)],
        short_side as f32,
    ))
}

/// Rotating calipers over the convex hull: the rectangle of least area that
/// contains `points`, as its four corners and its shorter side.
///
/// The minimum is always attained flush with a hull edge, so every edge — the
/// one that closes the ring included, which is the edge a windowed pass over
/// the hull quietly drops — is tried in turn and the hull is projected onto it
/// and onto its normal.
///
/// The corners come out in the order the reference emits them, which is fixed
/// by the rectangle's own axes rather than by the edge that happened to win:
/// its first axis is whichever of the four perpendicular directions points
/// right and up on screen, and the corners run from the far end of the second
/// axis, against it, and then back along it on the other side.
///
/// Everything up to the last step stays on whole numbers. The points are
/// integers and so is every edge, so projecting on an *un-normalized* edge is
/// exact; a rectangle that really is square then comes out with two sides that
/// agree bit for bit, which is what makes the corner sort in [`mini_box`]
/// resolve a tie the same way the reference does instead of on rounding noise.
fn min_area_rect(points: &[(i32, i32)]) -> Option<([(f64, f64); 4], f64)> {
    let hull = convex_hull(points);
    let (&only, rest) = hull.split_first()?;
    if rest.is_empty() {
        // One point, repeated as often as it likes: a rectangle with no
        // extent, which every size filter downstream rejects.
        return Some(([(only.0 as f64, only.1 as f64); 4], 0.0));
    }

    let mut best: Option<(f64, (i64, i64))> = None;
    for i in 0..hull.len() {
        let a = hull[i];
        let b = hull[(i + 1) % hull.len()];
        let edge = (b.0 - a.0, b.1 - a.1);
        let square_length = edge.0 * edge.0 + edge.1 * edge.1;
        if square_length == 0 {
            continue;
        }
        // Both projections carry a factor of the edge's length, so the area
        // they bound is out by its square.
        let (span, rise) = extents(&hull, edge);
        let area = (span.1 - span.0) as f64 * (rise.1 - rise.0) as f64 /
            square_length as f64;
        // Ties keep the earlier edge. They only arise for shapes with several
        // equally good orientations — a square, say — where the alternatives
        // are equally correct answers.
        if best.is_none_or(|(least, _)| area < least) {
            best = Some((area, edge));
        }
    }
    let (_, edge) = best?;

    // Of four directions each a right angle from the last, exactly one has a
    // non-negative x and a negative y — the rectangle's first axis. An
    // axis-aligned rectangle therefore takes the one pointing straight up,
    // not the one pointing right.
    let edge = [
        edge,
        (-edge.1, edge.0),
        (-edge.0, -edge.1),
        (edge.1, -edge.0),
    ]
    .into_iter()
    .find(|&(x, y)| x >= 0 && y < 0)
    .expect("one of four perpendicular axes points right and up");
    let square_length = (edge.0 * edge.0 + edge.1 * edge.1) as f64;
    let length = square_length.sqrt();
    let axis = (edge.0 as f64 / length, edge.1 as f64 / length);
    let normal = (-axis.1, axis.0);

    let (span, rise) = extents(&hull, edge);
    let width = (span.1 - span.0) as f64 / length;
    let height = (rise.1 - rise.0) as f64 / length;
    let centre = (
        ((span.0 + span.1) as f64 * edge.0 as f64 -
            (rise.0 + rise.1) as f64 * edge.1 as f64) /
            (2.0 * square_length),
        ((span.0 + span.1) as f64 * edge.1 as f64 +
            (rise.0 + rise.1) as f64 * edge.0 as f64) /
            (2.0 * square_length),
    );

    // The two far corners are the centre's reflection of the two near ones
    // rather than a second reading of the extents. The reference builds them
    // that way, and it is not cosmetic: corners that face each other across an
    // axis then hold *exactly* equal coordinates, which is again what a tied
    // corner sort turns on.
    let (half_width, half_height) = (width / 2.0, height / 2.0);
    let near = (
        centre.0 - half_width * axis.0 + half_height * normal.0,
        centre.1 - half_width * axis.1 + half_height * normal.1,
    );
    let side = (
        centre.0 - half_width * axis.0 - half_height * normal.0,
        centre.1 - half_width * axis.1 - half_height * normal.1,
    );
    Some((
        [
            near,
            side,
            (2.0 * centre.0 - near.0, 2.0 * centre.1 - near.1),
            (2.0 * centre.0 - side.0, 2.0 * centre.1 - side.1),
        ],
        width.min(height),
    ))
}

/// The hull's `(low, high)` bounds projected on `edge` and on its normal.
///
/// The two directions are perpendicular but not unit-length: every value comes
/// out scaled by the edge's length, which the caller divides out once. Keeping
/// them unscaled is what keeps this exact.
fn extents(hull: &[(i64, i64)], edge: (i64, i64)) -> ((i64, i64), (i64, i64)) {
    let (mut low_axis, mut high_axis) = (i64::MAX, i64::MIN);
    let (mut low_normal, mut high_normal) = (i64::MAX, i64::MIN);
    for &p in hull {
        let u = p.0 * edge.0 + p.1 * edge.1;
        let v = p.1 * edge.0 - p.0 * edge.1;
        low_axis = low_axis.min(u);
        high_axis = high_axis.max(u);
        low_normal = low_normal.min(v);
        high_normal = high_normal.max(v);
    }
    ((low_axis, high_axis), (low_normal, high_normal))
}

/// The convex hull of a point set, as a ring without collinear vertices.
///
/// Andrew's monotone chain, kept on whole numbers throughout so that the turn
/// test is exact: a hull decided by rounded cross products can come out
/// slightly non-convex, and a non-convex ring loses the supporting edge the
/// calipers need. Degenerate input yields a shorter ring — two points for a
/// collinear set, one for a set of identical points.
fn convex_hull(points: &[(i32, i32)]) -> Vec<(i64, i64)> {
    let mut sorted: Vec<(i64, i64)> =
        points.iter().map(|&(x, y)| (x as i64, y as i64)).collect();
    sorted.sort_unstable();
    sorted.dedup();
    if sorted.len() < 3 {
        return sorted;
    }

    let turn = |o: (i64, i64), a: (i64, i64), b: (i64, i64)| {
        (a.0 - o.0) * (b.1 - o.1) - (a.1 - o.1) * (b.0 - o.0)
    };
    let chain = |ring: &mut Vec<(i64, i64)>, p: (i64, i64)| {
        while ring.len() >= 2 {
            let n = ring.len();
            if turn(ring[n - 2], ring[n - 1], p) > 0 {
                break;
            }
            ring.pop();
        }
        ring.push(p);
    };

    let mut lower = Vec::with_capacity(sorted.len());
    for &p in &sorted {
        chain(&mut lower, p);
    }
    let mut upper = Vec::with_capacity(sorted.len());
    for &p in sorted.iter().rev() {
        chain(&mut upper, p);
    }
    lower.pop();
    upper.pop();
    lower.extend(upper);
    lower
}
