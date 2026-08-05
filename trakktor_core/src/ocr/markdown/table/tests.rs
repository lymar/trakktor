//! Cell markup into a Markdown table.

use super::*;

#[test]
fn prose_is_not_markup() {
    assert!(!is_markup("A table of month names, see Section 5.1."));
    assert!(!is_markup(""));
    assert!(is_markup("<fcel>1<fcel>mchu<nl>"));
    assert!(is_markup("<ecel><fcel>Tibetan<nl>"));
}

#[test]
fn a_table_becomes_a_pipe_table() {
    let markup = "<ecel><fcel>Tibetan<fcel>Sanskrit<nl>\
                  <fcel>1<fcel>mchu<fcel>Magha<nl>\
                  <fcel>2<fcel>dbo<fcel>Phalguna<nl>";
    let out = to_markdown(markup).expect("it converts");
    assert_eq!(
        out,
        "|  | Tibetan | Sanskrit |\n| --- | --- | --- |\n| 1 | mchu | Magha \
         |\n| 2 | dbo | Phalguna |\n"
    );
}

#[test]
fn a_merged_cell_leaves_its_place_empty() {
    // A pipe table has no colspan; the text stays where it was read and the
    // continuation is blank, which lays out right even though it says less.
    let markup = "<fcel>Sat<lcel><nl><fcel>22<fcel>9<nl>";
    let out = to_markdown(markup).expect("it converts");
    assert_eq!(out, "| Sat |  |\n| --- | --- |\n| 22 | 9 |\n");
}

#[test]
fn a_short_row_is_padded() {
    let markup = "<fcel>a<fcel>b<fcel>c<nl><fcel>d<nl>";
    let out = to_markdown(markup).expect("it converts");
    assert_eq!(out, "| a | b | c |\n| --- | --- | --- |\n| d |  |  |\n");
}

#[test]
fn a_row_left_open_still_converts() {
    // The answer was cut short by the token ceiling mid-row.
    let markup = "<fcel>a<fcel>b<nl><fcel>c";
    let out = to_markdown(markup).expect("it converts");
    assert_eq!(out, "| a | b |\n| --- | --- |\n| c |  |\n");
}

#[test]
fn a_pipe_inside_a_cell_is_escaped() {
    let out = to_markdown("<fcel>a|b<fcel>c<nl>").expect("it converts");
    assert!(out.contains("a\\|b"), "{out}");
}

#[test]
fn a_break_inside_a_cell_becomes_a_tag() {
    // Both spellings the model uses: a real newline and the two characters.
    let out = to_markdown("<fcel>one\ntwo<fcel>three\\nfour<nl>")
        .expect("it converts");
    assert!(out.contains("one<br>two"), "{out}");
    assert!(out.contains("three<br>four"), "{out}");
}

#[test]
fn a_single_row_gets_a_body() {
    // A header with no rows under it renders as nothing at all, so an empty
    // row is added rather than losing the text.
    let out = to_markdown("<fcel>only<fcel>row<nl>").expect("it converts");
    assert_eq!(out, "| only | row |\n| --- | --- |\n|  |  |\n");
}

#[test]
fn header_markers_carry_no_cell() {
    let out = to_markdown("<ched><fcel>a<fcel>b<nl>").expect("it converts");
    assert_eq!(out, "| a | b |\n| --- | --- |\n|  |  |\n");
}

#[test]
fn markup_with_no_cells_converts_to_nothing() {
    assert!(to_markdown("").is_none());
    assert!(to_markdown("<nl><nl>").is_none());
}
