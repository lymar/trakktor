//! Cell markup into a Markdown table.
//!
//! The tables below are what the model actually answered, copied out of the
//! runs unchanged, not markup written here to suit the converter: the point of
//! a merged cell is that the model's spelling of it is the thing under test.
//! They come from pages typeset for these measurements — a ruled table with a
//! head merged two columns wide and two rows deep, a timetable with a day
//! standing against three departures, a table with no merge in it at all.

use super::*;

/// A ruled table whose head is merged both ways: `Quarry and bench` covers two
/// columns and two rows, `Grain size (mm)` covers three columns. In the body
/// the model reported the row heads **not** as merges but as text against the
/// row it is printed against, with a blank above and below — the loss this
/// conversion cannot undo.
const BOTH_WAYS: &str = "<fcel>Quarry and bench<lcel><fcel>Grain size \
                         (mm)<lcel><lcel><nl><ucel><xcel><fcel>Coarse<fcel>\
                         Medium<fcel>Fine<nl><ecel><fcel>Upper<fcel>12.4<fcel>\
                         31.0<fcel>56.6<nl><fcel>Ravensport<fcel>Middle<fcel>\
                         10.8<fcel>29.4<fcel>59.8<nl><ecel><fcel>Lower<fcel>\
                         9.1<fcel>26.7<fcel>64.2<nl><fcel>Halden<fcel>Upper\
                         <fcel>14.0<fcel>33.2<fcel>52.8<nl><ecel><fcel>Lower\
                         <fcel>11.6<fcel>30.5<fcel>57.9<nl>";

/// The same table on an ordinary page — running text above it, a caption, a
/// footnote below — where the model reported the row heads as merges instead.
/// It also wrote a newline mid-answer and repeated the row it had just
/// written; both are left in, because that is what arrived.
const ROW_HEADS: &str = "<fcel>Quarry and bench<lcel><fcel>Grain size \
                         (mm)<lcel><lcel><nl><ucel><xcel><fcel>Coarse<fcel>\
                         Medium<fcel>Fine<nl><fcel>Ravensport<fcel>Upper<fcel>\
                         12.4<fcel>31.0<fcel>56.6<nl><ucel><fcel>Middle<fcel>\
                         10.8<fcel>29.4<fcel>59.8<nl><ucel><fcel>Lower<fcel>\
                         9.1<fcel>26.7<fcel>64.2<nl>\n<ecel><fcel>Lower<fcel>\
                         9.1<fcel>26.7<fcel>64.2<nl><fcel>Halden<fcel>Upper\
                         <fcel>14.0<fcel>33.2<fcel>52.8<nl><ucel><fcel>Lower\
                         <fcel>11.6<fcel>30.5<fcel>57.9<nl>";

/// A timetable: a day standing against three departures, and a Notes column
/// that is genuinely empty on most rows. Both come back, and they come back
/// differently — which is why a blank must not be read as a merge.
const TIMETABLE: &str =
    "<fcel>Days<fcel>Departure<fcel>Vessel<fcel>Notes<nl><fcel>Monday to \
     Thursday<fcel>06:40<fcel>Kittiwake<ecel><nl><ucel><fcel>11:\
     15<fcel>Kittiwake<ecel><nl><ucel><fcel>17:50<fcel>Fulmar<fcel>Cargo \
     only<nl><fcel>Friday<fcel>06:40<fcel>Fulmar<ecel><nl><ucel><fcel>15:\
     05<fcel>Fulmar<fcel>Two \
     crossings<nl><fcel>Saturday<fcel>09:30<fcel>Kittiwake<ecel><nl>";

/// A title row across the whole width, and no other merge.
const TITLE_ROW: &str = "<fcel>Harbour dues, season of 2027<lcel><lcel><lcel>\
                         <nl><fcel>Class<fcel>Season<fcel>Weekly<fcel>Daily\
                         <nl><fcel>Under 6 m<fcel>240<fcel>38<fcel>9<nl><fcel>\
                         6 to 12 m<fcel>410<fcel>62<fcel>14<nl><fcel>Over 12 m\
                         <fcel>735<fcel>105<fcel>23<nl>";

/// An ordinary table with nothing merged in it.
const PLAIN: &str = "<fcel>Screen<fcel>Opening (mm)<fcel>Loads \
                     sorted<nl><fcel>Coarse<fcel>20.\
                     0<fcel>41<nl><fcel>Medium<fcel>6.\
                     3<fcel>118<nl><fcel>Fine<fcel>2.\
                     0<fcel>96<nl><fcel>Rejects<fcel>0.5<fcel>12<nl>";

/// The first table again, read with the length ceiling low enough to cut the
/// answer off. It stops inside a number: the page reads `29.4`.
const CUT: &str = "<fcel>Quarry and bench<lcel><fcel>Grain size (mm)<lcel>\
                   <lcel><nl><ucel><xcel><fcel>Coarse<fcel>Medium<fcel>Fine\
                   <nl><ecel><fcel>Upper<fcel>12.4<fcel>31.0<fcel>56.6<nl>\
                   <fcel>Ravensport<fcel>Middle<fcel>10.8<fcel>29";

#[test]
fn prose_is_not_markup() {
    assert!(!is_markup("A table of month names, see Section 5.1."));
    assert!(!is_markup(""));
    assert!(is_markup("<fcel>1<fcel>mchu<nl>"));
    assert!(is_markup("<ecel><fcel>Tibetan<nl>"));
}

#[test]
fn a_table_with_nothing_merged_is_a_pipe_table() {
    let out = to_markdown(PLAIN, false).expect("it converts");
    assert_eq!(
        out,
        concat!(
            "| Screen | Opening (mm) | Loads sorted |\n",
            "| --- | --- | --- |\n",
            "| Coarse | 20.0 | 41 |\n",
            "| Medium | 6.3 | 118 |\n",
            "| Fine | 2.0 | 96 |\n",
            "| Rejects | 0.5 | 12 |\n",
        )
    );
}

#[test]
fn a_cell_merged_both_ways_becomes_a_span_of_each() {
    // The head covers two columns and two rows; the second head covers three
    // columns. Every position they reach into is left out of its row, which is
    // what makes the rest of the row line up.
    let out = to_markdown(BOTH_WAYS, false).expect("it converts");
    assert_eq!(
        out,
        concat!(
            "<table>\n",
            "<tr><td rowspan=\"2\" colspan=\"2\">Quarry and bench</td>",
            "<td colspan=\"3\">Grain size (mm)</td></tr>\n",
            "<tr><td>Coarse</td><td>Medium</td><td>Fine</td></tr>\n",
            "<tr><td></td><td>Upper</td><td>12.4</td><td>31.0</td>",
            "<td>56.6</td></tr>\n",
            "<tr><td>Ravensport</td><td>Middle</td><td>10.8</td><td>29.4</td>",
            "<td>59.8</td></tr>\n",
            "<tr><td></td><td>Lower</td><td>9.1</td><td>26.7</td>",
            "<td>64.2</td></tr>\n",
            "<tr><td>Halden</td><td>Upper</td><td>14.0</td><td>33.2</td>",
            "<td>52.8</td></tr>\n",
            "<tr><td></td><td>Lower</td><td>11.6</td><td>30.5</td>",
            "<td>57.9</td></tr>\n",
            "</table>\n",
        )
    );
}

#[test]
fn a_row_head_the_model_did_report_as_merged_becomes_a_rowspan() {
    // The same table as above, read off an ordinary page: here the model did
    // report the row heads, and each becomes one cell three rows deep. The
    // stray newline it wrote mid-answer belongs to no cell and disappears; the
    // row it repeated after that newline stays, because dropping a row on the
    // suspicion that it is a repeat would drop the real ones too.
    let out = to_markdown(ROW_HEADS, false).expect("it converts");
    assert!(
        out.contains("<td rowspan=\"3\">Ravensport</td>"),
        "the row head should cover its three rows: {out}"
    );
    assert_eq!(out.matches("<tr>").count(), 8, "{out}");
    assert_eq!(
        out.matches("<td>9.1</td>").count(),
        2,
        "the repeated row is kept: {out}"
    );
}

#[test]
fn an_empty_cell_is_not_a_merge() {
    // Both spellings are in this table: `Monday to Thursday` stands against
    // three departures and comes back merged, while the Notes column is
    // simply blank on four rows. They must not come out the same.
    let out = to_markdown(TIMETABLE, false).expect("it converts");
    assert_eq!(
        out,
        concat!(
            "<table>\n",
            "<tr><td>Days</td><td>Departure</td><td>Vessel</td>",
            "<td>Notes</td></tr>\n",
            "<tr><td rowspan=\"3\">Monday to Thursday</td><td>06:40</td>",
            "<td>Kittiwake</td><td></td></tr>\n",
            "<tr><td>11:15</td><td>Kittiwake</td><td></td></tr>\n",
            "<tr><td>17:50</td><td>Fulmar</td><td>Cargo only</td></tr>\n",
            "<tr><td rowspan=\"2\">Friday</td><td>06:40</td><td>Fulmar</td>",
            "<td></td></tr>\n",
            "<tr><td>15:05</td><td>Fulmar</td><td>Two crossings</td></tr>\n",
            "<tr><td>Saturday</td><td>09:30</td><td>Kittiwake</td>",
            "<td></td></tr>\n",
            "</table>\n",
        )
    );
}

#[test]
fn a_title_row_becomes_a_colspan() {
    let out = to_markdown(TITLE_ROW, false).expect("it converts");
    assert!(
        out.starts_with(
            "<table>\n<tr><td colspan=\"4\">Harbour dues, season of \
             2027</td></tr>\n"
        ),
        "{out}"
    );
    assert!(out.contains("<tr><td>Class</td>"), "{out}");
}

#[test]
fn a_table_cut_short_is_marked() {
    // The answer stopped inside a cell, and nothing about `29` says so — hence
    // the marker. The short row is still filled out to the width of the table.
    let out = to_markdown(CUT, true).expect("it converts");
    assert!(out.ends_with("<!-- table cut short -->\n"), "{out}");
    assert!(out.contains("<td>29</td></tr>"), "{out}");
    // Nothing is added when the answer ran to its end.
    assert!(!to_markdown(CUT, false).unwrap().contains("cut short"));
}

#[test]
fn a_merge_with_nothing_to_merge_into_is_a_cell_of_its_own() {
    // A row that opens with a continuation — which is how the second half of
    // an answer split in two arrives — must not lose the position: `a` stands
    // in the second column of this table, not the first.
    let out = to_markdown("<lcel><fcel>a<nl><fcel>b<fcel>c<nl>", false)
        .expect("it converts");
    assert_eq!(out, "|  | a |\n| --- | --- |\n| b | c |\n");
    // The same upwards: there is nothing above the first row.
    let out = to_markdown("<ucel><fcel>a<nl>", false).expect("it converts");
    assert_eq!(out, "|  | a |\n| --- | --- |\n|  |  |\n");
}

#[test]
fn an_angle_bracket_that_is_not_a_tag_is_text() {
    // The markup has six tags and no others, so this is a page that mentions a
    // threshold — not a tag this reader has to know about. The old parser read
    // everything between `<` and `>` as a tag and swallowed it.
    let out = to_markdown("<fcel>p < 0.05<fcel>n > 30<nl>", false)
        .expect("it converts");
    assert!(out.contains("| p < 0.05 | n > 30 |"), "{out}");
}

#[test]
fn html_escapes_what_would_otherwise_close_a_cell() {
    let out = to_markdown("<fcel>p < 0.05 & n > 30<lcel><nl>", false)
        .expect("it converts");
    assert!(
        out.contains("<td colspan=\"2\">p &lt; 0.05 &amp; n &gt; 30</td>"),
        "{out}"
    );
}

#[test]
fn a_pipe_inside_a_cell_is_escaped() {
    let out = to_markdown("<fcel>a|b<fcel>c<nl>", false).expect("it converts");
    assert!(out.contains("a\\|b"), "{out}");
}

#[test]
fn a_break_inside_a_cell_becomes_a_tag() {
    // Both spellings the model uses: a real newline and the two characters.
    let out = to_markdown("<fcel>one\ntwo<fcel>three\\nfour<nl>", false)
        .expect("it converts");
    assert!(out.contains("one<br>two"), "{out}");
    assert!(out.contains("three<br>four"), "{out}");
}

#[test]
fn a_short_row_is_padded() {
    let out = to_markdown("<fcel>a<fcel>b<fcel>c<nl><fcel>d<nl>", false)
        .expect("it converts");
    assert_eq!(out, "| a | b | c |\n| --- | --- | --- |\n| d |  |  |\n");
}

#[test]
fn a_single_row_gets_a_body() {
    // A header with no rows under it renders as nothing at all, so an empty
    // row is added rather than losing the text.
    let out =
        to_markdown("<fcel>only<fcel>row<nl>", false).expect("it converts");
    assert_eq!(out, "| only | row |\n| --- | --- |\n|  |  |\n");
}

#[test]
fn markup_with_no_cells_converts_to_nothing() {
    assert!(to_markdown("", false).is_none());
    assert!(to_markdown("<nl><nl>", false).is_none());
}
