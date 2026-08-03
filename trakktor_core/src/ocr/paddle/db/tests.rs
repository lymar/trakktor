use super::*;

/// The synthetic probability map every geometry test below runs on. It is
/// `MAP_W` × `MAP_H` and holds four invented blobs whose shapes exercise the
/// interesting paths: an upright bar with a fainter row above and below it,
/// so the threshold decides the bar's height; a staircase that rises four
/// pixels at a time, so it is a slanted line rather than an upright one; a
/// small square that survives on its own but starves once it is dilated; and
/// an L of two overlapping rectangles that the staircase runs into, so two
/// blobs merge into one contour.
const MAP_W: usize = 96;
const MAP_H: usize = 64;

fn fill(
    map: &mut [f32],
    rows: std::ops::Range<usize>,
    cols: std::ops::Range<usize>,
    value: f32,
) {
    for y in rows {
        for x in cols.clone() {
            map[y * MAP_W + x] = value;
        }
    }
}

fn golden_map() -> Vec<f32> {
    let mut map = vec![0.0f32; MAP_W * MAP_H];
    fill(&mut map, 8..18, 6..40, 0.90);
    fill(&mut map, 7..8, 6..40, 0.45);
    fill(&mut map, 18..19, 6..40, 0.45);
    for step in 0..40 {
        let x = 44 + step;
        let y = 26 + step / 4;
        fill(&mut map, y..y + 8, x..x + 2, 0.80);
    }
    fill(&mut map, 46..52, 10..16, 0.75);
    fill(&mut map, 40..60, 70..78, 0.85);
    fill(&mut map, 52..60, 70..92, 0.85);
    map
}

/// The source page the golden map is rescaled onto: twice the map in both
/// directions, so every rescale is an exact doubling and the expected numbers
/// stay readable.
const PAGE: (u32, u32) = (192, 128);

fn params(score_mode: ScoreMode) -> Params {
    Params {
        thresh: 0.3,
        box_thresh: 0.6,
        max_candidates: 1000,
        // The training-export value, which is what the golden numbers below
        // were produced with; it is not the default.
        unclip_ratio: 1.5,
        use_dilation: false,
        score_mode,
    }
}

fn raw(map: &[f32], params: &Params) -> Vec<[(i32, i32); 4]> {
    candidates(map, MAP_W, MAP_H, PAGE, params)
        .into_iter()
        .map(|(corners, _)| corners.map(|(x, y)| (x as i32, y as i32)))
        .collect()
}

fn filtered(map: &[f32], params: &Params) -> Vec<[(i32, i32); 4]> {
    boxes_from_bitmap(map, MAP_W, MAP_H, PAGE, (2.0, 2.0), params)
        .into_iter()
        .map(|(quad, _)| quad.points.map(|(x, y)| (x as i32, y as i32)))
        .collect()
}

const SMALL_SQUARE: [(i32, i32); 4] =
    [(16, 88), (34, 88), (34, 106), (16, 106)];
const BAR: [(i32, i32); 4] = [(0, 2), (90, 2), (90, 48), (0, 48)];
const STAIRCASE: [(i32, i32); 4] = [(67, 17), (192, 57), (192, 128), (41, 119)];

#[test]
fn the_map_is_the_one_the_expectations_were_taken_from() {
    let map = golden_map();
    let total: f64 = map.iter().map(|&v| f64::from(v)).sum();
    assert!((total - 858.0).abs() < 1e-3, "map total is {total}");
}

#[test]
fn fast_scoring_drops_the_slanted_line() {
    // The staircase's own rectangle is mostly background, so averaging over
    // the rectangle scores it well under the threshold.
    assert_eq!(
        raw(&golden_map(), &params(ScoreMode::Fast)),
        vec![SMALL_SQUARE, BAR]
    );
}

#[test]
fn slow_scoring_keeps_the_slanted_line() {
    assert_eq!(
        raw(&golden_map(), &params(ScoreMode::Slow)),
        vec![SMALL_SQUARE, STAIRCASE, BAR]
    );
}

#[test]
fn the_two_score_modes_disagree_by_half_on_a_slanted_line() {
    let map = golden_map();
    let scores = |mode| {
        let params = Params {
            box_thresh: 0.0,
            ..params(mode)
        };
        candidates(&map, MAP_W, MAP_H, PAGE, &params)
            .into_iter()
            .map(|(_, score)| score)
            .collect::<Vec<_>>()
    };
    // The staircase's rectangle reaches past the right edge of the map, so
    // its fast score also pins the border handling of the scoring mask.
    let fast = scores(ScoreMode::Fast);
    let slow = scores(ScoreMode::Slow);
    assert_eq!(fast.len(), 3);
    assert_eq!(slow.len(), 3);
    for (mode, (got, want)) in [
        ("fast", (&fast, [0.75, 0.338_154_28, 0.824_999_98])),
        ("slow", (&slow, [0.75, 0.822_628_97, 0.824_999_98])),
    ] {
        for (got, want) in got.iter().zip(want) {
            assert!(
                (got - want).abs() < 1e-6,
                "{mode}: scored {got}, expected {want}"
            );
        }
    }
}

#[test]
fn the_page_filters_pull_the_slanted_line_back_onto_the_page() {
    // Two of the staircase's corners land exactly on the page size, which is
    // one past the last pixel; the clip is what makes them addressable.
    assert_eq!(
        filtered(&golden_map(), &params(ScoreMode::Slow)),
        vec![
            SMALL_SQUARE,
            [(67, 17), (191, 57), (191, 127), (41, 119)],
            BAR,
        ]
    );
    assert_eq!(
        filtered(&golden_map(), &params(ScoreMode::Fast)),
        vec![SMALL_SQUARE, BAR]
    );
}

#[test]
fn dilation_merges_the_blobs_and_starves_the_small_square() {
    // Grown by a pixel, the small square's rectangle covers thirteen more
    // cells than the square itself, which pulls its mean under the threshold.
    let dilated = |mode| Params {
        use_dilation: true,
        ..params(mode)
    };
    assert_eq!(
        filtered(&golden_map(), &dilated(ScoreMode::Fast)),
        vec![[(0, 0), (94, 0), (94, 52), (0, 52)]]
    );

    let slow = filtered(&golden_map(), &dilated(ScoreMode::Slow));
    assert_eq!(slow.len(), 2);
    assert_eq!(slow[1], [(0, 0), (94, 0), (94, 52), (0, 52)]);
    // The merged slanted blob is the one place where the offsetter's own
    // arithmetic shows through. PaddleOCR expands with an earlier generation
    // of the same library, whose round joins are stepped by a different law;
    // the two rings differ by a vertex or so, and the rectangle re-fitted to
    // them can land a pixel apart on the map — two here, since this page is
    // twice the map. Upright boxes are unaffected: their joins are exact
    // quarter-turns either way, which is why every other expectation in this
    // file is an equality.
    for (got, want) in
        slow[0]
            .iter()
            .zip([(69, 15), (191, 57), (191, 127), (41, 121)])
    {
        assert!(
            (got.0 - want.0).abs() <= 2 && (got.1 - want.1).abs() <= 2,
            "corner {got:?} is more than two pixels from {want:?}"
        );
    }
}

#[test]
fn the_candidate_cut_keeps_the_contours_that_come_first() {
    let params = Params {
        max_candidates: 1,
        ..params(ScoreMode::Fast)
    };
    assert_eq!(filtered(&golden_map(), &params), vec![SMALL_SQUARE]);
}

#[test]
fn the_threshold_is_strict() {
    let mut map = vec![0.0f32; MAP_W * MAP_H];
    fill(&mut map, 10..30, 10..30, 0.3);
    let params = Params {
        box_thresh: 0.0,
        ..params(ScoreMode::Fast)
    };
    assert!(filtered(&map, &params).is_empty());

    let mut map = vec![0.0f32; MAP_W * MAP_H];
    fill(
        &mut map,
        10..30,
        10..30,
        f32::from_bits(0.3f32.to_bits() + 1),
    );
    assert_eq!(filtered(&map, &params).len(), 1);
}

/// A box that rescales down to a couple of pixels is dropped by the page
/// filter even though it passed every threshold on the map.
#[test]
fn boxes_that_shrink_to_nothing_on_the_page_are_dropped() {
    let mut map = vec![0.0f32; MAP_W * MAP_H];
    fill(&mut map, 10..14, 10..14, 1.0);
    let params = params(ScoreMode::Fast);
    let tiny_page = (MAP_W as u32 / 8, MAP_H as u32 / 8);

    let survivors = candidates(&map, MAP_W, MAP_H, tiny_page, &params);
    assert_eq!(
        survivors
            .iter()
            .map(|(corners, _)| corners.map(|(x, y)| (x as i32, y as i32)))
            .collect::<Vec<_>>(),
        vec![[(1, 1), (2, 1), (2, 2), (1, 2)]]
    );
    assert!(
        boxes_from_bitmap(&map, MAP_W, MAP_H, tiny_page, (1.0, 1.0), &params)
            .is_empty()
    );
}

#[test]
fn a_blob_too_thin_to_fit_a_rectangle_never_reaches_scoring() {
    let mut map = vec![0.0f32; MAP_W * MAP_H];
    // Three pixels across is a two-pixel extent, under the minimum.
    fill(&mut map, 10..13, 10..13, 1.0);
    assert!(filtered(&map, &params(ScoreMode::Fast)).is_empty());
}

#[test]
fn dilation_grows_only_towards_the_far_corner() {
    let (w, h) = (8usize, 8usize);
    let mut mask = vec![false; w * h];
    mask[3 * w + 3] = true;
    let grown = dilate(&mask, w, h);
    let lit: Vec<(usize, usize)> = (0..h)
        .flat_map(|y| (0..w).map(move |x| (x, y)))
        .filter(|(x, y)| grown[y * w + x])
        .collect();
    assert_eq!(lit, vec![(3, 3), (4, 3), (3, 4), (4, 4)]);
}

#[test]
fn filling_a_rectangle_covers_its_boundary_too() {
    // Interior scanlines alone would miss the bottom row and the right
    // column; the outline pass is what makes the fill inclusive.
    let mask = fill_poly(&[(1, 1), (5, 1), (5, 3), (1, 3)], 8, 6);
    assert_eq!(mask.iter().filter(|lit| **lit).count(), 15);
    for y in 1..=3 {
        for x in 1..=5 {
            assert!(mask[y * 8 + x], "({x},{y}) should be filled");
        }
    }
}

#[test]
fn a_degenerate_polygon_still_fills_something() {
    // A candidate collapsed to a segment or a point must score against a
    // non-empty mask rather than divide by zero.
    let segment = fill_poly(&[(1, 1), (8, 1)], 10, 3);
    assert_eq!(segment.iter().filter(|lit| **lit).count(), 8);
    let point = fill_poly(&[(4, 4), (4, 4), (4, 4)], 6, 6);
    assert_eq!(point.iter().filter(|lit| **lit).count(), 1);
    assert!(point[4 * 6 + 4]);
}

/// The scoring window is clamped to the map, so a box that runs off the page
/// is handed to the fill with a corner outside the mask. The outline is then
/// re-anchored on the border rather than merely cut, which changes the pixels
/// the walk visits — here the first row gets two of them and not three.
#[test]
fn filling_re_anchors_an_outline_that_leaves_the_mask() {
    let (w, h) = (59usize, 39usize);
    let mask = fill_poly(&[(7, 0), (59, 13), (52, 38), (0, 25)], w, h);
    assert_eq!(mask.iter().filter(|lit| **lit).count(), 1452);
    let first_row: Vec<usize> = (0..w).filter(|x| mask[*x]).collect();
    assert_eq!(first_row, vec![7, 8]);
}

#[test]
fn filling_ignores_the_winding_direction() {
    let clockwise = fill_poly(&[(0, 0), (4, 0), (4, 4), (0, 4)], 6, 6);
    let widdershins = fill_poly(&[(0, 4), (4, 4), (4, 0), (0, 0)], 6, 6);
    assert_eq!(clockwise, widdershins);
    assert_eq!(clockwise.iter().filter(|lit| **lit).count(), 25);
}

#[test]
fn expanding_a_rectangle_moves_every_side_out_by_the_offset() {
    // A 20x10 rectangle has area 200 and perimeter 60, so at ratio 1.5 the
    // offset is exactly 5 pixels on every side.
    let ring = unclip(
        &[(10.0, 10.0), (30.0, 10.0), (30.0, 20.0), (10.0, 20.0)],
        1.5,
    )
    .expect("one ring");
    let xs: Vec<i32> = ring.iter().map(|p| p.0).collect();
    let ys: Vec<i32> = ring.iter().map(|p| p.1).collect();
    assert_eq!(xs.iter().min(), Some(&5));
    assert_eq!(xs.iter().max(), Some(&35));
    assert_eq!(ys.iter().min(), Some(&5));
    assert_eq!(ys.iter().max(), Some(&25));
}

#[test]
fn expanding_truncates_the_corners_on_the_way_in() {
    // The area and the perimeter are measured on the fractional corners, but
    // the offset itself starts from the truncated ones — which is why the
    // ring below is symmetric about a lattice rectangle rather than about the
    // input.
    let ring = unclip(
        &[(10.4, 10.6), (30.4, 10.6), (30.4, 20.2), (10.4, 20.2)],
        2.0,
    )
    .expect("one ring");
    assert_eq!(
        ring,
        vec![
            (36, 7),
            (36, 23),
            (33, 26),
            (7, 26),
            (4, 23),
            (4, 7),
            (7, 4),
            (33, 4),
        ]
    );
}

#[test]
fn ordering_rotates_the_corners_of_a_steeply_slanted_box() {
    // The rectangle fit hands back its left-most corner first; past 45
    // degrees that corner is not the top-left one, and the crop that follows
    // cares which is which.
    let fitted = [(14.0, 0.0), (34.0, 35.0), (26.0, 40.0), (6.0, 5.0)];
    assert_eq!(
        order_points_clockwise(fitted),
        [(6.0, 5.0), (14.0, 0.0), (34.0, 35.0), (26.0, 40.0)]
    );
}

#[test]
fn ordering_leaves_an_upright_box_alone() {
    let upright = [(1.0, 2.0), (9.0, 2.0), (9.0, 6.0), (1.0, 6.0)];
    assert_eq!(order_points_clockwise(upright), upright);
}

#[test]
fn ordering_survives_four_identical_corners() {
    let degenerate = [(3.0, 4.0); 4];
    assert_eq!(order_points_clockwise(degenerate), degenerate);
}

#[test]
fn an_empty_or_mismatched_map_yields_nothing() {
    let params = params(ScoreMode::Fast);
    assert!(boxes_from_bitmap(&[], 0, 0, PAGE, (1.0, 1.0), &params).is_empty());
    assert!(
        boxes_from_bitmap(&[1.0; 4], 8, 8, PAGE, (1.0, 1.0), &params)
            .is_empty()
    );
    assert!(
        boxes_from_bitmap(&[1.0; 64], 8, 8, (0, 0), (1.0, 1.0), &params)
            .is_empty()
    );
}
