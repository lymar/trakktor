//! Which region a line belongs to, and where a line has to be taken apart
//! because it belongs to two.
//!
//! Both rules are the port's own rather than the reference's, and both were
//! written against pages that had gone wrong, so the cases below are the pages
//! rather than the algebra.

use super::*;
use crate::ocr::page::Quad;

fn rect(x0: f32, y0: f32, x1: f32, y1: f32) -> Quad {
    Quad::new([(x0, y0), (x1, y0), (x1, y1), (x0, y1)])
}

fn region(label: Label, x0: f32, y0: f32, x1: f32, y1: f32) -> Region {
    Region {
        label,
        score: 0.9,
        quad: rect(x0, y0, x1, y1),
    }
}

/// The two columns of a page and one line of each, at the same height.
fn columns() -> Vec<Region> {
    vec![
        region(Label::Text, 200.0, 2430.0, 1223.0, 2772.0),
        region(Label::Text, 1244.0, 2224.0, 2265.0, 2567.0),
    ]
}

/// A line belongs to the innermost region that holds it.
///
/// The page this is drawn from is real: the model returned nine paragraph boxes
/// **and** a box around the whole column, and the lines of a stacking script
/// poke a few pixels out of every box drawn around them. Picking the region a
/// line overlaps most therefore hands every one of those lines to the column,
/// which then becomes one page-sized block and swallows the reading order.
#[test]
fn a_line_goes_to_the_smallest_region_that_holds_it() {
    let regions = vec![
        region(Label::Text, 245.0, 781.0, 1465.0, 936.0),
        // The column: it covers every line completely.
        region(Label::Text, 236.0, 233.0, 2316.0, 3057.0),
    ];
    // A line that sticks a few pixels out of the paragraph around it.
    let line = (240.0, 772.0, 1476.0, 943.0);
    assert_eq!(owner(line, &regions), Some(0));

    // A line only the column holds still goes to the column.
    let stray = (250.0, 2000.0, 1400.0, 2100.0);
    assert_eq!(owner(stray, &regions), Some(1));

    // A line nowhere near either belongs to neither, and is read by geometry.
    let outside = (250.0, 3200.0, 400.0, 3260.0);
    assert_eq!(owner(outside, &regions), None);
}

/// A picture is never a line's container: text over one is a caption printed on
/// it, not part of it.
#[test]
fn a_picture_never_owns_a_line() {
    let regions = vec![region(Label::Image, 100.0, 100.0, 900.0, 900.0)];
    assert_eq!(owner((200.0, 200.0, 800.0, 260.0), &regions), None);
}

/// A line glued across the gutter is cut, and cut between the columns.
///
/// The measured case: a two-column page set with a narrow gutter, where the
/// detector joined the first line of a paragraph in the left column to the line
/// facing it in the right one. Read whole, the two sentences run together and
/// half of one column is filed inside the other.
#[test]
fn a_line_glued_across_the_gutter_is_cut_between_the_columns() {
    let regions = columns();
    let glued = (203.0, 2423.0, 2271.0, 2478.0);
    let pieces = split(glued, &regions);
    assert_eq!(pieces.len(), 2, "{pieces:?}");

    // The pieces tile the line — its own edges at the ends, and the cut halfway
    // across the gutter, so no ink between the columns is dropped.
    assert_eq!(pieces[0].0, 203.0);
    assert_eq!(pieces[1].1, 2271.0);
    assert_eq!(pieces[0].1, pieces[1].0);
    assert!(
        (pieces[0].1 - (1223.0 + 1244.0) / 2.0).abs() < 0.01,
        "cut at {}",
        pieces[0].1
    );
}

/// A line that stays inside its own column is left alone, however close the
/// next column stands.
#[test]
fn a_line_inside_its_column_is_not_cut() {
    let regions = columns();
    assert!(split((203.0, 2477.0, 1259.0, 2532.0), &regions).is_empty());
    assert!(split((1248.0, 2375.0, 2274.0, 2423.0), &regions).is_empty());
}

/// A region **inside** the one a line belongs to never cuts it.
///
/// This is the difference from the reference, which cuts a line against every
/// region it falls into, and it is not a detail: a paragraph of running text
/// with a formula set inside it puts a `formula` box around part of nearly
/// every line, and cutting there would slice the middle out of a line that no
/// region disagrees about. On our own pages this is what a second region almost
/// always is.
#[test]
fn a_region_nested_in_the_owner_never_cuts_a_line() {
    let regions = vec![
        region(Label::Text, 200.0, 1000.0, 2000.0, 1400.0),
        // A formula set into the middle of a line of running text.
        region(Label::Formula, 800.0, 1180.0, 1400.0, 1240.0),
        // And one on a line of its own, further down.
        region(Label::Formula, 780.0, 1300.0, 1420.0, 1360.0),
    ];
    let inline = (210.0, 1185.0, 1990.0, 1235.0);
    assert_eq!(owner(inline, &regions), Some(0));
    assert!(split(inline, &regions).is_empty());

    // The other way round: the formula owns the line and the paragraph is the
    // one around it, which is no more a boundary to cross than the reverse.
    let display = (800.0, 1305.0, 1400.0, 1355.0);
    assert_eq!(owner(display, &regions), Some(2));
    assert!(split(display, &regions).is_empty());
}

/// Neither does a region the line merely grazes: a box that pokes a few pixels
/// into the next column is a detection artefact, not a straddle.
#[test]
fn a_line_that_only_grazes_the_next_column_is_not_cut() {
    let regions = columns();
    assert!(split((203.0, 2477.0, 1290.0, 2532.0), &regions).is_empty());
}

/// A line in no region at all is left whole, as it is left whole everywhere
/// else: the model is silent on scripts outside its own, and losing text to
/// that silence is what the ownership rule exists to prevent.
#[test]
fn a_line_no_region_holds_is_not_cut() {
    let regions = columns();
    assert!(split((203.0, 100.0, 2271.0, 160.0), &regions).is_empty());
}

/// Three columns, one line across all of them: two cuts, each in its gutter.
#[test]
fn a_line_across_three_columns_is_cut_twice() {
    let regions = vec![
        region(Label::Text, 100.0, 500.0, 700.0, 900.0),
        region(Label::Text, 750.0, 500.0, 1350.0, 900.0),
        region(Label::Text, 1400.0, 500.0, 2000.0, 900.0),
    ];
    let pieces = split((100.0, 600.0, 2000.0, 650.0), &regions);
    assert_eq!(pieces.len(), 3, "{pieces:?}");
    assert!((pieces[0].1 - 725.0).abs() < 0.01);
    assert!((pieces[1].1 - 1375.0).abs() < 0.01);
}
