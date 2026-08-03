//! Synthetic pages, built here from quadrangles and invented strings, that
//! exercise one layout decision each.

use super::*;

/// One horizontal line, `text` sitting inside `(x0, y0)..(x1, y1)`.
fn line(text: &str, x0: f32, y0: f32, x1: f32, y1: f32) -> Line {
    Line {
        text: text.to_string(),
        score: 0.95,
        quad: Quad::new([(x0, y0), (x1, y0), (x1, y1), (x0, y1)]),
        rotated: false,
        truncated: false,
    }
}

fn sheet(width: u32, height: u32, lines: Vec<Line>) -> Page {
    Page {
        number: 1,
        source: "synthetic".into(),
        width,
        height,
        lines,
    }
}

fn read(page: &Page) -> Layout {
    analyse(page, &Repeated::default(), &Settings::default())
}

fn kinds(layout: &Layout) -> Vec<BlockKind> {
    layout.blocks.iter().map(|block| block.kind).collect()
}

/// The page-line indices in the order the layout reads them.
fn order(layout: &Layout) -> Vec<usize> {
    layout
        .blocks
        .iter()
        .flat_map(|block| block.lines.iter().copied())
        .collect()
}

/// Turns a page's quadrangles by `degrees` about its centre, so a test can
/// feed the analysis a page that was scanned crooked.
fn tilt(page: &Page, degrees: f32) -> Page {
    let theta = degrees.to_radians();
    let (sin, cos) = (theta.sin(), theta.cos());
    let center = (page.width as f32 / 2.0, page.height as f32 / 2.0);
    let lines = page
        .lines
        .iter()
        .map(|line| Line {
            quad: Quad::new(line.quad.points.map(|p| {
                let (dx, dy) = (p.0 - center.0, p.1 - center.1);
                (
                    center.0 + dx * cos - dy * sin,
                    center.1 + dx * sin + dy * cos,
                )
            })),
            ..line.clone()
        })
        .collect();
    Page {
        lines,
        ..page.clone()
    }
}

/// Two columns of prose read down the left one and then down the right one,
/// however interleaved the detector handed them over.
#[test]
fn two_columns_read_one_after_the_other() {
    // The lines go in interleaved — left 0, right 0, left 1, … — which is the
    // order a naive sort by `y` then `x` produces and the one that makes a
    // two-column page unreadable.
    let mut lines = Vec::new();
    for i in 0..6 {
        let y = 100.0 + 30.0 * i as f32;
        lines.push(line("left column line", 80.0, y, 460.0, y + 20.0));
        lines.push(line("right column line", 540.0, y + 13.0, 920.0, y + 33.0));
    }
    for i in 6..8 {
        let y = 100.0 + 30.0 * i as f32;
        lines.push(line("left column line", 80.0, y, 460.0, y + 20.0));
    }
    let page = sheet(1000, 1400, lines);
    let layout = read(&page);

    assert_eq!(kinds(&layout), vec![BlockKind::Paragraph; 2]);
    assert_eq!(
        order(&layout),
        vec![0, 2, 4, 6, 8, 10, 12, 13, 1, 3, 5, 7, 9, 11]
    );
}

/// A page with a centred heading and two paragraphs comes out as exactly
/// that. The heading is found by its centring and its isolation: its box is
/// barely taller than a body line.
#[test]
fn a_centred_heading_over_two_paragraphs() {
    let mut lines =
        vec![line("Widgets and gadgets", 400.0, 100.0, 600.0, 122.0)];
    for i in 0..5 {
        let y = 200.0 + 30.0 * i as f32;
        lines.push(line("first paragraph line", 200.0, y, 800.0, y + 20.0));
    }
    for i in 0..4 {
        let y = 362.0 + 30.0 * i as f32;
        lines.push(line("second paragraph line", 200.0, y, 800.0, y + 20.0));
    }
    let page = sheet(1000, 1400, lines);
    let layout = read(&page);

    assert_eq!(
        kinds(&layout),
        vec![
            BlockKind::Heading { level: 1 },
            BlockKind::Paragraph,
            BlockKind::Paragraph,
        ]
    );
    assert_eq!(layout.blocks[0].lines, vec![0]);
    assert_eq!(layout.blocks[1].lines.len(), 5);
    assert_eq!(layout.blocks[2].lines.len(), 4);
}

/// A note set smaller than the body, low on the page and opening with a
/// number, is a footnote — and box height says nothing here, because the
/// note's boxes and the body's overlap.
#[test]
fn a_small_numbered_block_low_on_the_page_is_a_footnote() {
    let mut lines = Vec::new();
    for i in 0..8 {
        let y = 200.0 + 32.0 * i as f32;
        let stop = if i == 7 { 700.0 } else { 850.0 };
        lines.push(line("a line of body text", 150.0, y, stop, y + 22.0));
    }
    lines.push(line(
        "1 The note explains a word above.",
        150.0,
        950.0,
        600.0,
        966.0,
    ));
    lines.push(line(
        "It runs onto a second line.",
        150.0,
        976.0,
        500.0,
        992.0,
    ));
    let page = sheet(1000, 1400, lines);
    let layout = read(&page);

    assert_eq!(
        kinds(&layout),
        vec![BlockKind::Paragraph, BlockKind::Footnote]
    );
    assert_eq!(layout.blocks[1].lines, vec![8, 9]);
}

/// A narrow numeral anchored to the page's centre line at the foot is page
/// furniture, and page furniture is read last — after the footnote, which is
/// itself read after the body.
#[test]
fn a_page_number_is_furniture_and_comes_last() {
    let mut lines = Vec::new();
    for i in 0..8 {
        let y = 200.0 + 32.0 * i as f32;
        lines.push(line("a line of body text", 150.0, y, 850.0, y + 22.0));
    }
    lines.push(line("1 A note at the foot.", 150.0, 950.0, 600.0, 966.0));
    lines.push(line("Its second line.", 150.0, 976.0, 500.0, 992.0));
    lines.push(line("7", 490.0, 1300.0, 510.0, 1320.0));
    let page = sheet(1000, 1400, lines);
    let layout = read(&page);

    assert_eq!(
        kinds(&layout),
        vec![
            BlockKind::Paragraph,
            BlockKind::Footnote,
            BlockKind::PageFurniture,
        ]
    );
    assert_eq!(layout.blocks[2].lines, vec![10]);
}

/// A running head cannot be told from a first heading on one page; it is told
/// by coming back on every page. The head here carries the page number, which
/// changes from page to page, so the histogram has to key on the letters
/// alone.
#[test]
fn a_running_head_is_found_only_across_pages() {
    const WORDS: [&str; 20] = [
        "alder", "birch", "cedar", "elm", "fir", "hazel", "ilex", "juniper",
        "larch", "maple", "oak", "pine", "quince", "rowan", "spruce", "teak",
        "willow", "yew", "aspen", "beech",
    ];
    let pages: Vec<Page> = (0..4)
        .map(|p| {
            let mut lines = vec![line(
                &format!("Notes on trees {}", 12 + p),
                380.0,
                40.0,
                620.0,
                62.0,
            )];
            for i in 0..5 {
                let y = 300.0 + 30.0 * i as f32;
                let word = WORDS[(p as usize * 5 + i) % WORDS.len()];
                lines.push(line(word, 200.0, y, 800.0, y + 20.0));
            }
            sheet(1000, 1000, lines)
        })
        .collect();

    let repeated = document_furniture(&pages);
    assert_eq!(repeated.len(), 1);

    let with = analyse(&pages[0], &repeated, &Settings::default());
    assert_eq!(with.blocks[0].kind, BlockKind::PageFurniture);
    assert_eq!(with.blocks[0].lines, vec![0]);
    assert!(
        with.blocks[1..]
            .iter()
            .all(|block| block.kind != BlockKind::PageFurniture)
    );

    // Without the document-level pass the same line reads as a heading, which
    // is exactly why the pass exists.
    let without = read(&pages[0]);
    assert_eq!(without.blocks[0].kind, BlockKind::Heading { level: 1 });
}

/// On a page that interleaves a stacking script with an alphabetic one, the
/// stacking lines are twice as tall at the same type size. They must not
/// become headings, while a genuinely centred alphabetic title still must.
#[test]
fn tall_script_lines_are_not_headings() {
    // Tibetan letters ka, kha, ga: invented syllables, not a text.
    let tall = "\u{0F40}\u{0F41}\u{0F42} \u{0F44}\u{0F45}\u{0F46}";
    let mut lines = vec![
        line("Заголовок", 380.0, 60.0, 495.0, 82.0),
        line(tall, 370.0, 130.0, 505.0, 176.0),
    ];
    for s in 0..3 {
        let base = 260.0 + 160.0 * s as f32;
        lines.push(line(tall, 120.0, base, 750.0, base + 46.0));
        lines.push(line(
            "транслитерация",
            100.0,
            base + 60.0,
            775.0,
            base + 80.0,
        ));
        lines.push(line(
            "перевод строки",
            100.0,
            base + 90.0,
            775.0,
            base + 110.0,
        ));
    }
    let page = sheet(875, 1250, lines);
    let layout = read(&page);

    assert_eq!(order(&layout), (0..11).collect::<Vec<_>>());
    assert_eq!(layout.blocks[0].kind, BlockKind::Heading { level: 1 });
    // The tall line sitting alone under the title is centred and isolated —
    // everything a heading needs except that it is only tall because of its
    // script.
    assert_eq!(layout.blocks[1].lines, vec![1]);
    assert_eq!(layout.blocks[1].kind, BlockKind::Paragraph);
    assert_eq!(
        layout
            .blocks
            .iter()
            .filter(|block| matches!(block.kind, BlockKind::Heading { .. }))
            .count(),
        1
    );
}

/// Row-aligned parts are a table and read across, not down each part in turn.
#[test]
fn a_table_reads_row_major() {
    // Pushed column by column, which is the wrong reading order.
    let mut lines = Vec::new();
    for (column, x0) in [(0, 100.0), (1, 350.0), (2, 600.0)] {
        for row in 0..5 {
            let y = 200.0 + 40.0 * row as f32;
            lines.push(line(
                &format!("cell {column} {row}"),
                x0,
                y,
                x0 + 150.0,
                y + 20.0,
            ));
        }
    }
    let page = sheet(1000, 1400, lines);
    let layout = read(&page);

    assert_eq!(
        order(&layout),
        vec![0, 5, 10, 1, 6, 11, 2, 7, 12, 3, 8, 13, 4, 9, 14]
    );
}

/// Adjacent lines of one paragraph differ in box height by half, depending on
/// whether they happen to carry a descender. Splitting on that gives a
/// paragraph per line; splitting on the pitch does not.
#[test]
fn line_height_jitter_does_not_split_a_paragraph() {
    let lines: Vec<Line> = (0..8)
        .map(|i| {
            let y = 200.0 + 30.0 * i as f32;
            let height = if i % 2 == 0 { 12.0 } else { 21.0 };
            line("a line of body text", 200.0, y, 800.0, y + height)
        })
        .collect();
    let page = sheet(1000, 1400, lines);
    let layout = read(&page);

    assert_eq!(kinds(&layout), vec![BlockKind::Paragraph]);
    assert_eq!(layout.blocks[0].lines.len(), 8);
}

/// A page scanned crooked reads the same as a straight one, and the blocks
/// come back in the crooked page's own coordinates rather than in the
/// straightened ones.
#[test]
fn a_crooked_page_reads_the_same_and_reports_crooked_blocks() {
    let mut lines =
        vec![line("Widgets and gadgets", 400.0, 100.0, 600.0, 122.0)];
    for i in 0..5 {
        let y = 200.0 + 30.0 * i as f32;
        lines.push(line("first paragraph line", 200.0, y, 800.0, y + 20.0));
    }
    for i in 0..4 {
        let y = 362.0 + 30.0 * i as f32;
        lines.push(line("second paragraph line", 200.0, y, 800.0, y + 20.0));
    }
    let straight = sheet(1000, 1400, lines);
    let crooked = tilt(&straight, 1.2);

    let expected = read(&straight);
    let layout = read(&crooked);
    assert_eq!(kinds(&layout), kinds(&expected));
    assert_eq!(order(&layout), order(&expected));

    let [tl, tr, ..] = layout.blocks[0].quad.points;
    assert!(
        (tr.1 - tl.1) > 1.0,
        "the block should be reported on the crooked page, not the \
         straightened one"
    );
}

/// A short block beside the text, too small to be a column of its own, is
/// taken out of the cut, put back next to what it sits closest to, and called
/// a caption.
#[test]
fn a_block_beside_the_text_becomes_a_caption() {
    let mut lines: Vec<Line> = (0..10)
        .map(|i| {
            let y = 200.0 + 30.0 * i as f32;
            line("a line of body text", 400.0, y, 900.0, y + 20.0)
        })
        .collect();
    lines.push(line(
        "Figure 4 shows the widget",
        100.0,
        500.0,
        330.0,
        518.0,
    ));
    lines.push(line("in its second housing.", 100.0, 526.0, 300.0, 544.0));
    let page = sheet(1000, 1400, lines);
    let layout = read(&page);

    assert_eq!(
        kinds(&layout),
        vec![BlockKind::Paragraph, BlockKind::Caption]
    );
    assert_eq!(layout.blocks[1].lines, vec![10, 11]);
}

/// An empty page, and a page whose every line was rejected by the confidence
/// filter, come back empty rather than panicking.
#[test]
fn degenerate_pages_come_back_empty() {
    assert!(read(&sheet(1000, 1400, Vec::new())).blocks.is_empty());

    let mut faint = line("barely read", 200.0, 200.0, 800.0, 220.0);
    faint.score = 0.1;
    assert!(read(&sheet(1000, 1400, vec![faint])).blocks.is_empty());
}

#[test]
fn scripts_are_classified_by_their_unicode_block() {
    assert_eq!(script("Widgets"), Script::Alphabetic);
    assert_eq!(script("Заголовок"), Script::Alphabetic);
    assert_eq!(script("\u{0F40}\u{0F41}\u{0F42}"), Script::Stacking);
    assert_eq!(script("\u{0915}\u{0916}"), Script::Stacking);
    assert_eq!(script("\u{4E2D}\u{6587}"), Script::Ideographic);
    assert_eq!(script("12 (3)"), Script::Neutral);
    // A stray Latin word inside a stacking-script line does not change what
    // the line's height means.
    assert_eq!(script("\u{0F40}\u{0F41}\u{0F42} ok"), Script::Stacking);
}

#[test]
fn heading_numbering_gives_its_own_depth() {
    assert_eq!(leading_number("2 Methods"), Some(1));
    assert_eq!(leading_number("2. Methods"), Some(1));
    assert_eq!(leading_number("2.3 Details"), Some(2));
    assert_eq!(leading_number("2.3.1 More"), Some(3));
    assert_eq!(leading_number("IV. Results"), Some(1));
    assert_eq!(leading_number("B) Notes"), Some(1));
    // A sentence opening with a year is not a numbering.
    assert_eq!(leading_number("1998 was a quiet year"), None);
    assert_eq!(leading_number("Introduction"), None);
}

#[test]
fn footnote_markers_are_small_numbers_and_reference_symbols() {
    assert!(footnote_marker("1 The note."));
    assert!(footnote_marker("12. The note."));
    assert!(footnote_marker("* The note."));
    assert!(footnote_marker("† The note."));
    assert!(!footnote_marker("1998 was a quiet year"));
    assert!(!footnote_marker("The note."));
    assert!(!footnote_marker(""));
}

#[test]
fn page_numbers_are_read_in_whatever_script_the_page_uses() {
    assert!(is_numeral("7"));
    assert!(is_numeral("[12]"));
    assert!(is_numeral("- 341 -"));
    assert!(is_numeral("xiv"));
    assert!(is_numeral("\u{0F22}\u{0F27}"));
    assert!(is_numeral("\u{0967}\u{0968}"));
    assert!(!is_numeral("Notes"));
    assert!(!is_numeral(""));
    assert!(!is_numeral("1234567"));
}
