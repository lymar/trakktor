use super::Quad;

/// A cut follows the box's own slant rather than the page's vertical.
///
/// A detected box is the smallest quadrangle around the ink, which on a scan
/// sits a degree or two off level. Cutting straight down the page would take a
/// bite out of one piece and leave a wedge of the neighbour in the other.
#[test]
fn a_slice_follows_the_slant_of_the_box() {
    // A hundred pixels long, dropping ten as it goes.
    let tilted = Quad::new([
        (100.0, 200.0),
        (200.0, 210.0),
        (200.0, 240.0),
        (100.0, 230.0),
    ]);
    let left = tilted.slice(100.0, 150.0);
    assert_eq!(left.points[0], (100.0, 200.0));
    assert_eq!(left.points[1], (150.0, 205.0));
    assert_eq!(left.points[2], (150.0, 235.0));
    assert_eq!(left.points[3], (100.0, 230.0));

    // The two pieces meet exactly, corner for corner, and the far one keeps
    // the box's own end.
    let right = tilted.slice(150.0, 200.0);
    assert_eq!(right.points[0], left.points[1]);
    assert_eq!(right.points[3], left.points[2]);
    assert_eq!(right.points[1], tilted.points[1]);
}

/// A box with no width at all cannot be cut, and says so by staying whole
/// rather than by collapsing to a point.
#[test]
fn a_box_with_no_width_survives_a_slice() {
    let upright =
        Quad::new([(50.0, 10.0), (50.0, 10.0), (50.0, 90.0), (50.0, 90.0)]);
    assert_eq!(upright.slice(20.0, 40.0), upright);
}
