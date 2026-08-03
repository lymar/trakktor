use super::*;
use crate::ocr::page::{Line, Quad};

/// The invented page every fixture below is set on: a column of text between
/// `LEFT` and `RIGHT` on a 1000 px page, set at a line height of `HEIGHT`.
/// The numbers are only ever compared with each other, so they are round
/// rather than realistic.
const LEFT: f32 = 100.0;
const RIGHT: f32 = 900.0;
const HEIGHT: f32 = 20.0;

/// One recognized line. An upright quadrangle is all the geometry the
/// renderer reads, so the fixtures give one directly.
fn line(text: &str, x0: f32, y: f32, x1: f32) -> Line {
    Line {
        text: text.to_string(),
        score: 1.0,
        quad: Quad::new([(x0, y), (x1, y), (x1, y + HEIGHT), (x0, y + HEIGHT)]),
        rotated: false,
        truncated: false,
    }
}

/// A line filling the column from the left margin to the right one.
fn full(text: &str, y: f32) -> Line { line(text, LEFT, y, RIGHT) }

/// A line that starts at the left margin and stops well short of the right.
fn short(text: &str, y: f32) -> Line { line(text, LEFT, y, 400.0) }

/// The bounding quadrangle of a block's lines.
fn cover(page: &Page, lines: &[usize]) -> Quad {
    let (mut x0, mut y0) = (f32::INFINITY, f32::INFINITY);
    let (mut x1, mut y1) = (f32::NEG_INFINITY, f32::NEG_INFINITY);
    for &index in lines {
        let (a, b, c, d) = page.lines[index].quad.bounds();
        x0 = x0.min(a);
        y0 = y0.min(b);
        x1 = x1.max(c);
        y1 = y1.max(d);
    }
    Quad::new([(x0, y0), (x1, y0), (x1, y1), (x0, y1)])
}

/// A page of invented lines together with the layout over them: every entry
/// is a block kind and the lines that block owns.
fn analysed(
    number: usize,
    source: &str,
    lines: Vec<Line>,
    blocks: Vec<(BlockKind, Vec<usize>)>,
) -> (Page, Layout) {
    let page = Page {
        number,
        source: source.to_string(),
        width: 1000,
        height: 1400,
        lines,
    };
    let blocks = blocks
        .into_iter()
        .map(|(kind, lines)| Block {
            quad: cover(&page, &lines),
            kind,
            lines,
        })
        .collect();
    (
        page,
        Layout {
            blocks,
            figures: Vec::new(),
        },
    )
}

/// A page holding one paragraph over the given lines.
fn paragraph(number: usize, source: &str, lines: Vec<Line>) -> (Page, Layout) {
    let indices = (0..lines.len()).collect();
    analysed(number, source, lines, vec![(BlockKind::Paragraph, indices)])
}

fn markdown(pages: &[(Page, Layout)]) -> String {
    render(pages, &Options::default())
}

/// Every case of the line seam: what the two characters that meet decide.
/// The first three columns are the text so far, the line being added and the
/// result; the comment on each row is the rule it stands for.
const SEAMS: &[(&str, &str, &str)] = &[
    // A hyphen between letters is a broken word, in any alphabet.
    ("under-", "standing", "understanding"),
    ("раз-", "бивается", "разбивается"),
    // A hyphen with a digit on either side belongs to the text.
    ("COVID-", "19", "COVID-19"),
    ("358-", "359", "358-359"),
    ("3-", "мерный", "3-мерный"),
    // So does a longer dash between numbers, and without a space.
    ("358–", "359", "358–359"),
    // A space goes in for any letter, not only the Latin ones.
    ("word", "next", "word next"),
    ("слово", "следующее", "слово следующее"),
    ("λόγος", "νέος", "λόγος νέος"),
    ("word,", "next", "word, next"),
    ("2024", "was", "2024 was"),
    // Scripts written without word spaces get none.
    ("第一页结束", "于句中并继续", "第一页结束于句中并继续"),
    ("བཀྲ་ཤིས་", "བདེ་ལེགས", "བཀྲ་ཤིས་བདེ་ལེགས"),
    ("word", "བདེ", "wordབདེ"),
];

#[test]
fn a_line_seam_follows_the_scripts_that_meet_at_it() {
    for (text, next, want) in SEAMS {
        let mut got = (*text).to_string();
        append(&mut got, None, next);
        assert_eq!(&got, want, "joining `{text}` to `{next}`");
    }
}

#[test]
fn a_broken_word_is_closed_up_and_a_number_range_is_not() {
    let page = paragraph(
        1,
        "a.png",
        vec![
            full("Слово раз-", 100.0),
            short("бивается на строки.", 130.0),
        ],
    );
    assert_eq!(markdown(&[page]), "Слово разбивается на строки.\n");

    let page = paragraph(
        1,
        "a.png",
        vec![
            full("См. страницы 358-", 100.0),
            short("359 в томе.", 130.0),
        ],
    );
    assert_eq!(markdown(&[page]), "См. страницы 358-359 в томе.\n");
}

#[test]
fn a_space_is_kept_between_lines_of_a_non_latin_script() {
    let page = paragraph(
        1,
        "a.png",
        vec![full("Кошка сидела на", 100.0), short("тёплом окне.", 130.0)],
    );
    assert_eq!(markdown(&[page]), "Кошка сидела на тёплом окне.\n");
}

/// The level the layout assigned, and the line it becomes. Markdown stops at
/// six levels, and a level of zero is still a heading.
const HEADINGS: &[(u8, &str)] = &[
    (0, "# Title"),
    (1, "# Title"),
    (2, "## Title"),
    (3, "### Title"),
    (6, "###### Title"),
    (9, "###### Title"),
];

#[test]
fn a_heading_is_written_at_the_level_it_was_given() {
    for (level, want) in HEADINGS {
        assert_eq!(&heading(*level, "Title"), want, "level {level}");
    }
}

#[test]
fn a_heading_and_its_paragraph_are_separate_blocks() {
    let page = analysed(
        1,
        "a.png",
        vec![
            line("Глава вторая", 400.0, 100.0, 600.0),
            short("Начало.", 140.0),
        ],
        vec![
            (BlockKind::Heading { level: 2 }, vec![0]),
            (BlockKind::Paragraph, vec![1]),
        ],
    );
    assert_eq!(markdown(&[page]), "## Глава вторая\n\nНачало.\n");
}

/// What a recognized footnote splits into: the marker that becomes the label
/// and the note itself, or nothing when the opening characters are simply
/// text.
const MARKERS: &[(&str, Option<(&str, &str)>)] = &[
    ("1. Первая сноска.", Some(("1", "Первая сноска."))),
    ("2 Вторая сноска.", Some(("2", "Вторая сноска."))),
    ("3) Третья сноска.", Some(("3", "Третья сноска."))),
    ("[4] Четвёртая сноска.", Some(("4", "Четвёртая сноска."))),
    ("* Со звёздочкой.", Some(("*", "Со звёздочкой."))),
    ("† С крестиком.", Some(("†", "С крестиком."))),
    (
        "¹ Надстрочная единица.",
        Some(("1", "Надстрочная единица.")),
    ),
    ("1998 год был иным.", None),
    ("358–359 в указателе.", None),
    ("Без всякого маркера.", None),
    ("12", None),
];

#[test]
fn a_footnote_marker_is_only_read_where_one_was_set() {
    for (text, want) in MARKERS {
        let got = split_marker(text);
        let got = got.as_ref().map(|(label, note)| (&label[..], *note));
        assert_eq!(got, *want, "reading `{text}`");
    }
}

#[test]
fn footnotes_come_after_the_body_of_their_page() {
    let page = analysed(
        1,
        "a.png",
        vec![
            short("Тело страницы.", 100.0),
            short("1. Первая сноска.", 1200.0),
            short("Без маркера.", 1240.0),
        ],
        vec![
            (BlockKind::Paragraph, vec![0]),
            (BlockKind::Footnote, vec![1]),
            (BlockKind::Footnote, vec![2]),
        ],
    );
    assert_eq!(
        markdown(&[page]),
        "Тело страницы.\n\n[^1]: Первая сноска.\n\n---\n\nБез маркера.\n"
    );
}

#[test]
fn a_label_two_pages_reuse_is_qualified_by_the_page() {
    let first = analysed(
        1,
        "a.png",
        vec![short("1. Сноска первой страницы.", 1200.0)],
        vec![(BlockKind::Footnote, vec![0])],
    );
    let second = analysed(
        2,
        "b.png",
        vec![short("1. Сноска второй страницы.", 1200.0)],
        vec![(BlockKind::Footnote, vec![0])],
    );
    assert_eq!(
        markdown(&[first, second]),
        "[^1]: Сноска первой страницы.\n\n[^1-2]: Сноска второй страницы.\n"
    );
}

#[test]
fn page_furniture_never_reaches_the_document() {
    let page = analysed(
        1,
        "a.png",
        vec![
            line("Бегущий заголовок", 400.0, 40.0, 600.0),
            short("Тело страницы.", 100.0),
            line("53", 480.0, 1300.0, 520.0),
        ],
        vec![
            (BlockKind::PageFurniture, vec![0]),
            (BlockKind::Paragraph, vec![1]),
            (BlockKind::PageFurniture, vec![2]),
        ],
    );
    assert_eq!(markdown(&[page]), "Тело страницы.\n");
}

#[test]
fn a_caption_is_set_apart_from_the_body() {
    let page = analysed(
        1,
        "a.png",
        vec![
            short("Тело страницы.", 100.0),
            line("Рис. 28. Дерево.", 200.0, 400.0, 500.0),
        ],
        vec![
            (BlockKind::Paragraph, vec![0]),
            (BlockKind::Caption, vec![1]),
        ],
    );
    assert_eq!(markdown(&[page]), "Тело страницы.\n\n*Рис. 28. Дерево.*\n");
}

/// A page whose paragraph runs out to the right margin, i.e. one that is not
/// finished where the page ends.
fn open_page() -> (Page, Layout) {
    paragraph(
        1,
        "a.png",
        vec![
            full("Первый абзац идёт до", 100.0),
            full("самого правого поля и", 130.0),
            line("продолжается", LEFT, 160.0, 890.0),
        ],
    )
}

#[test]
fn a_paragraph_that_runs_over_the_break_is_joined_to_its_beginning() {
    let next =
        paragraph(2, "b.png", vec![short("на следующей странице.", 100.0)]);
    assert_eq!(
        markdown(&[open_page(), next]),
        "Первый абзац идёт до самого правого поля и продолжается на следующей \
         странице.\n"
    );
}

#[test]
fn a_footnote_between_the_two_halves_does_not_break_the_join() {
    let mut first = open_page();
    first
        .0
        .lines
        .push(short("1. Сноска первой страницы.", 1200.0));
    let note = Block {
        kind: BlockKind::Footnote,
        lines: vec![first.0.lines.len() - 1],
        quad: cover(&first.0, &[first.0.lines.len() - 1]),
    };
    first.1.blocks.push(note);
    let next =
        paragraph(2, "b.png", vec![short("на следующей странице.", 100.0)]);
    assert_eq!(
        markdown(&[first, next]),
        "Первый абзац идёт до самого правого поля и продолжается на следующей \
         странице.\n\n[^1]: Сноска первой страницы.\n"
    );
}

#[test]
fn an_indented_first_line_starts_a_paragraph_of_its_own() {
    let next = analysed(
        2,
        "b.png",
        vec![
            line("Новый абзац с отступом", LEFT + 40.0, 100.0, RIGHT),
            short("во второй строке.", 130.0),
        ],
        vec![(BlockKind::Paragraph, vec![0, 1])],
    );
    assert_eq!(
        markdown(&[open_page(), next]),
        "Первый абзац идёт до самого правого поля и продолжается\n\nНовый \
         абзац с отступом во второй строке.\n"
    );
}

#[test]
fn a_paragraph_that_ended_short_takes_no_continuation() {
    let first = paragraph(
        1,
        "a.png",
        vec![
            full("Абзац кончается", 100.0),
            short("на этой странице.", 130.0),
        ],
    );
    let next = paragraph(2, "b.png", vec![short("Другой абзац.", 100.0)]);
    assert_eq!(
        markdown(&[first, next]),
        "Абзац кончается на этой странице.\n\nДругой абзац.\n"
    );
}

#[test]
fn a_page_separator_is_written_between_the_pages_when_asked() {
    let first = paragraph(1, "a.png", vec![short("Первая страница.", 100.0)]);
    let second = paragraph(2, "b.png", vec![short("Вторая страница.", 100.0)]);
    let options = Options {
        page_separators: true,
        image_dir: None,
    };
    assert_eq!(
        render(&[first, second], &options),
        "Первая страница.\n\n<!-- page 2 · b.png -->\n\nВторая страница.\n"
    );
}

#[test]
fn a_separator_sits_inside_the_paragraph_it_would_have_cut_in_two() {
    let next =
        paragraph(2, "b.png", vec![short("на следующей странице.", 100.0)]);
    let options = Options {
        page_separators: true,
        image_dir: None,
    };
    assert_eq!(
        render(&[open_page(), next], &options),
        "Первый абзац идёт до самого правого поля и продолжается<!-- page 2 · \
         b.png --> на следующей странице.\n"
    );
}

#[test]
fn pages_without_text_render_as_nothing() {
    let page = analysed(
        1,
        "a.png",
        vec![line("53", 480.0, 1300.0, 520.0)],
        vec![(BlockKind::PageFurniture, vec![0])],
    );
    assert!(markdown(&[page]).is_empty());
    assert!(markdown(&[]).is_empty());
}

#[test]
fn plain_text_names_every_page_after_the_first() {
    let first = Page {
        number: 1,
        source: "a.png".to_string(),
        width: 1000,
        height: 1400,
        lines: vec![short("Первая строка.", 100.0), short("Вторая.", 130.0)],
    };
    let second = Page {
        number: 2,
        source: "b.png".to_string(),
        width: 1000,
        height: 1400,
        lines: vec![short("Строка второй страницы.", 100.0)],
    };
    assert_eq!(
        plain(std::slice::from_ref(&first)),
        "Первая строка.\nВторая.\n"
    );
    assert_eq!(
        plain(&[first, second]),
        "Первая строка.\nВторая.\n\n=== page 2 · b.png ===\n\nСтрока второй \
         страницы.\n"
    );
}
