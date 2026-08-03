use super::*;

/// A mask under construction, addressed the way the caller addresses it.
struct Mask {
    pixels: Vec<bool>,
    width: usize,
    height: usize,
}

impl Mask {
    fn new(width: usize, height: usize) -> Self {
        Self {
            pixels: vec![false; width * height],
            width,
            height,
        }
    }

    /// Reads a mask off a drawing, `#` for foreground.
    fn from_rows(rows: &[&str]) -> Self {
        let width = rows.first().map_or(0, |r| r.len());
        let mut mask = Self::new(width, rows.len());
        for (y, row) in rows.iter().enumerate() {
            assert_eq!(row.len(), width, "row {y} is a different length");
            for (x, cell) in row.bytes().enumerate() {
                mask.pixels[y * width + x] = cell == b'#';
            }
        }
        mask
    }

    /// Fills the half-open pixel range `[x0, x1) x [y0, y1)`.
    fn fill(&mut self, x0: usize, x1: usize, y0: usize, y1: usize) {
        for y in y0..y1 {
            for x in x0..x1 {
                self.pixels[y * self.width + x] = true;
            }
        }
    }

    fn contours(&self) -> Vec<Contour> {
        find_contours(&self.pixels, self.width, self.height)
    }
}

/// The synthetic map the vectors in this file were taken from: a solid bar, a
/// staircase that runs into an L and merges with it, and a small square. The
/// bar and the square are plain rectangles; the staircase is what exercises a
/// rotated minimum-area rectangle.
fn golden_mask() -> Mask {
    let mut mask = Mask::new(96, 64);
    mask.fill(6, 40, 7, 19);
    for i in 0..40 {
        mask.fill(44 + i, 46 + i, 26 + i / 4, 34 + i / 4);
    }
    mask.fill(10, 16, 46, 52);
    mask.fill(70, 78, 40, 60);
    mask.fill(70, 92, 52, 60);
    mask
}

#[track_caller]
fn assert_corners(got: [(f32, f32); 4], want: [(f32, f32); 4]) {
    for (i, (g, w)) in got.iter().zip(&want).enumerate() {
        assert!(
            (g.0 - w.0).abs() < 1e-4 && (g.1 - w.1).abs() < 1e-4,
            "corner {i}: got {g:?}, want {w:?}; whole box {got:?}"
        );
    }
}

#[test]
fn a_solid_rectangle_keeps_only_its_corners() {
    let mut mask = Mask::new(12, 10);
    mask.fill(3, 8, 2, 7);
    let contours = mask.contours();
    assert_eq!(contours.len(), 1);
    // Down the left edge first, and the interior of every run is dropped.
    assert_eq!(contours[0].points, vec![(3, 2), (3, 6), (7, 6), (7, 2)]);
}

#[test]
fn a_component_touching_the_edge_starts_at_its_top_left_pixel() {
    // Without the background fringe the walk would enter this one from a
    // different neighbour and the point list would start elsewhere.
    let mut mask = Mask::new(6, 6);
    mask.fill(0, 3, 0, 3);
    let contours = mask.contours();
    assert_eq!(contours.len(), 1);
    assert_eq!(contours[0].points, vec![(0, 0), (0, 2), (2, 2), (2, 0)]);
}

#[test]
fn a_lone_pixel_is_a_contour_of_one_point() {
    let mut mask = Mask::new(5, 5);
    mask.fill(2, 3, 2, 3);
    let contours = mask.contours();
    assert_eq!(contours.len(), 1);
    assert_eq!(contours[0].points, vec![(2, 2)]);
    // A one-pixel run has no extent at all, which is how the size filters see
    // it too.
    let (_, short_side) = mini_box(&contours[0].points).expect("a rectangle");
    assert_eq!(short_side, 0.0);
}

#[test]
fn a_hole_is_returned_as_a_contour_of_its_own() {
    let mut mask = Mask::new(14, 14);
    mask.fill(2, 11, 2, 11);
    mask.fill(4, 8, 4, 8);
    for y in 4..8 {
        for x in 4..8 {
            mask.pixels[y * mask.width + x] = false;
        }
    }
    let contours = mask.contours();
    // The hole is a box candidate like any other, and it is discovered after
    // the outer border, so it comes back first.
    assert_eq!(contours.len(), 2);
    assert_eq!(contours[1].points, vec![(2, 2), (2, 10), (10, 10), (10, 2)]);
    assert_eq!(
        contours[0].points,
        vec![
            (3, 4),
            (4, 3),
            (7, 3),
            (8, 4),
            (8, 7),
            (7, 8),
            (4, 8),
            (3, 7)
        ]
    );
}

#[test]
fn a_walk_that_runs_through_its_own_start_pixel_drops_it() {
    // The hole inside the lower cluster is entered at (2, 5), but the walk
    // leaves that pixel in the very direction it arrived in, so the pixel is
    // interior to a diagonal run and never becomes a vertex — the loop just
    // closes over it. Getting this wrong adds one spurious leading point to
    // roughly one contour in a hundred, and to no others.
    let mask = Mask::from_rows(&[
        ".........",
        ".#.#.....",
        ".##......",
        ".#.......",
        "...#...#.",
        ".##.#....",
        ".#..##...",
        "..####.#.",
        ".........",
    ]);
    let contours = mask.contours();
    let points: Vec<&[(i32, i32)]> =
        contours.iter().map(|c| c.points.as_slice()).collect();
    assert_eq!(
        points,
        vec![
            &[(7, 7)][..],
            &[(3, 4), (4, 5), (4, 6), (3, 7), (2, 7), (1, 6)][..],
            &[(7, 4)][..],
            &[(3, 4), (2, 5), (1, 5), (1, 6), (2, 7), (5, 7), (5, 6)][..],
            &[(1, 1), (1, 3), (3, 1), (2, 2)][..],
        ]
    );
}

#[test]
fn contours_come_back_in_reverse_discovery_order() {
    let mut mask = Mask::new(32, 32);
    // Six squares whose top-left pixels the raster scan meets in this order.
    let corners = [(2, 2), (12, 2), (22, 2), (2, 12), (12, 12), (22, 22)];
    for (x, y) in corners {
        mask.fill(x, x + 5, y, y + 5);
    }
    let first: Vec<(i32, i32)> =
        mask.contours().iter().map(|c| c.points[0]).collect();
    assert_eq!(
        first,
        vec![(22, 22), (12, 12), (2, 12), (22, 2), (12, 2), (2, 2)]
    );
}

#[test]
fn the_golden_map_yields_three_contours() {
    let contours = golden_mask().contours();
    assert_eq!(contours.len(), 3);
    let shape: Vec<(usize, (i32, i32))> = contours
        .iter()
        .map(|c| (c.points.len(), c.points[0]))
        .collect();
    // The staircase and the L touch, so they arrive as one contour.
    assert_eq!(shape, vec![(4, (10, 46)), (45, (44, 26)), (4, (6, 7))]);
}

#[test]
fn the_golden_contours_have_the_expected_mini_boxes() {
    let contours = golden_mask().contours();
    let boxes: Vec<([(f32, f32); 4], f32)> = contours
        .iter()
        .map(|c| mini_box(&c.points).expect("a rectangle"))
        .collect();

    assert_corners(
        boxes[0].0,
        [(10.0, 46.0), (15.0, 46.0), (15.0, 51.0), (10.0, 51.0)],
    );
    assert_eq!(boxes[0].1, 5.0);

    assert_corners(
        boxes[1].0,
        [
            (44.2353, 25.0588),
            (96.2353, 38.0588),
            (89.7647, 63.9412),
            (37.7647, 50.9412),
        ],
    );
    assert!(
        (f64::from(boxes[1].1) - 26.678921).abs() < 1e-5,
        "short side {}",
        boxes[1].1
    );

    assert_corners(
        boxes[2].0,
        [(6.0, 7.0), (39.0, 7.0), (39.0, 18.0), (6.0, 18.0)],
    );
    assert_eq!(boxes[2].1, 11.0);
}

#[test]
fn an_expanded_ring_is_re_fitted_to_a_rectangle() {
    // What the box expansion hands back for the golden map's small square and
    // its solid bar: integer rings that no longer are rectangles.
    let square = [
        (17, 46),
        (17, 51),
        (15, 53),
        (10, 53),
        (8, 51),
        (8, 46),
        (10, 44),
        (15, 44),
    ];
    let (corners, short_side) = mini_box(&square).expect("a rectangle");
    assert_corners(
        corners,
        [(8.0, 44.0), (17.0, 44.0), (17.0, 53.0), (8.0, 53.0)],
    );
    assert_eq!(short_side, 9.0);

    let bar = [
        (42, 2),
        (45, 4),
        (45, 18),
        (44, 21),
        (42, 24),
        (6, 24),
        (3, 23),
        (0, 21),
        (0, 7),
        (1, 4),
        (3, 1),
        (39, 1),
    ];
    let (corners, short_side) = mini_box(&bar).expect("a rectangle");
    assert_corners(
        corners,
        [(0.0, 1.0), (45.0, 1.0), (45.0, 24.0), (0.0, 24.0)],
    );
    assert_eq!(short_side, 23.0);
}

#[test]
fn mini_box_orders_a_tilted_rectangle_clockwise_from_the_left() {
    // A 10x5 rectangle tilted by a 3-4-5 triangle: the minimum-area rectangle
    // is the tilted one (50) and not its axis-aligned bounding box (110).
    let tilted = [(3, 0), (11, 6), (8, 10), (0, 4)];
    let (corners, short_side) = mini_box(&tilted).expect("a rectangle");
    assert_corners(corners, [(3.0, 0.0), (11.0, 6.0), (8.0, 10.0), (0.0, 4.0)]);
    assert!((short_side - 5.0).abs() < 1e-4, "short side {short_side}");
}

#[test]
fn a_square_on_its_corner_still_lands_on_one_ordering() {
    // Four orientations of this one are equally minimal, and two of its
    // corners share an x, so both the choice of rectangle and the corner sort
    // come down to ties. They are only decided the same way as the reference
    // because the sides come out of whole-number projections and so agree
    // exactly, and because the far corners are reflections of the near ones.
    let diamond = [(17, 21), (18, 20), (19, 21), (18, 22)];
    let (corners, short_side) = mini_box(&diamond).expect("a rectangle");
    assert_corners(
        corners,
        [(17.0, 21.0), (18.0, 20.0), (19.0, 21.0), (18.0, 22.0)],
    );
    assert!(
        (short_side - 2f32.sqrt()).abs() < 1e-6,
        "short side {short_side}"
    );
}

#[test]
fn mini_box_reads_the_extent_not_the_pixel_count() {
    // Three by four pixels span two by three, because both ends are inclusive.
    let block = [(4, 5), (6, 5), (6, 8), (4, 8), (5, 6)];
    let (corners, short_side) = mini_box(&block).expect("a rectangle");
    assert_corners(corners, [(4.0, 5.0), (6.0, 5.0), (6.0, 8.0), (4.0, 8.0)]);
    assert_eq!(short_side, 2.0);
}

#[test]
fn mini_box_of_nothing_is_nothing() {
    assert!(mini_box(&[]).is_none());
}

#[test]
fn an_empty_mask_has_no_contours() {
    assert!(find_contours(&[], 0, 0).is_empty());
    assert!(find_contours(&[false; 9], 3, 3).is_empty());
}
