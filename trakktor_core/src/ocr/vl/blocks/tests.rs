use super::*;

/// A line box `height` tall whose top-left corner is at `(x, y)`.
fn line(x: f32, y: f32, width: f32, height: f32) -> Quad {
    Quad::new([
        (x, y),
        (x + width, y),
        (x + width, y + height),
        (x, y + height),
    ])
}

const PAGE: (u32, u32) = (1000, 1400);

#[test]
fn consecutive_lines_of_one_paragraph_make_one_block() {
    let quads: Vec<Quad> = (0..5)
        .map(|i| line(100.0, 100.0 + i as f32 * 30.0, 600.0, 20.0))
        .collect();
    let blocks = assemble(&quads, PAGE, None, &Settings::default());
    assert_eq!(blocks.len(), 1);
    assert_eq!(blocks[0].lines, vec![0, 1, 2, 3, 4]);
}

#[test]
fn a_wide_vertical_gap_starts_a_new_block() {
    let quads = vec![
        line(100.0, 100.0, 600.0, 20.0),
        line(100.0, 130.0, 600.0, 20.0),
        // Four line heights below the last one.
        line(100.0, 260.0, 600.0, 20.0),
    ];
    let blocks = assemble(&quads, PAGE, None, &Settings::default());
    assert_eq!(blocks.len(), 2);
    assert_eq!(blocks[0].lines, vec![0, 1]);
    assert_eq!(blocks[1].lines, vec![2]);
}

#[test]
fn two_columns_stay_apart_though_their_lines_interleave() {
    // The reading order a detector hands over walks both columns row by row.
    let mut quads = Vec::new();
    for row in 0..4 {
        let y = 100.0 + row as f32 * 30.0;
        quads.push(line(60.0, y, 380.0, 20.0));
        quads.push(line(520.0, y, 380.0, 20.0));
    }
    let blocks = assemble(&quads, PAGE, None, &Settings::default());
    assert_eq!(blocks.len(), 2);
    assert_eq!(blocks[0].lines, vec![0, 2, 4, 6]);
    assert_eq!(blocks[1].lines, vec![1, 3, 5, 7]);
}

#[test]
fn a_heading_does_not_join_the_paragraph_under_it() {
    // Twice the type size of the text under it.
    let quads = vec![
        line(100.0, 100.0, 500.0, 44.0),
        line(100.0, 160.0, 600.0, 20.0),
        line(100.0, 190.0, 600.0, 20.0),
    ];
    let blocks = assemble(&quads, PAGE, None, &Settings::default());
    assert_eq!(blocks.len(), 2);
    assert_eq!(blocks[0].lines, vec![0]);
    assert_eq!(blocks[1].lines, vec![1, 2]);
}

#[test]
fn a_stacking_line_does_not_swallow_the_alphabetic_lines_around_it() {
    // A Tibetan verse between its transliteration and its translation: the
    // middle line runs twice the height of the other two. Joining all three
    // makes a block that mixes writing systems line by line, which is measured
    // to be worse for this model than the lines on their own.
    let quads = vec![
        line(100.0, 100.0, 600.0, 24.0),
        line(100.0, 132.0, 500.0, 50.0),
        line(100.0, 190.0, 600.0, 24.0),
    ];
    let blocks = assemble(&quads, PAGE, None, &Settings::default());
    assert_eq!(blocks.len(), 3);
}

#[test]
fn a_block_never_grows_past_the_line_ceiling() {
    let settings = Settings {
        max_lines: 3,
        ..Settings::default()
    };
    let quads: Vec<Quad> = (0..7)
        .map(|i| line(100.0, 100.0 + i as f32 * 30.0, 600.0, 20.0))
        .collect();
    let blocks = assemble(&quads, PAGE, None, &settings);
    assert!(blocks.iter().all(|block| block.lines.len() <= 3));
    assert_eq!(blocks.iter().map(|b| b.lines.len()).sum::<usize>(), 7);
}

#[test]
fn the_cut_is_padded_and_clipped_to_the_page() {
    let quads = vec![line(2.0, 2.0, 990.0, 40.0)];
    let blocks = assemble(&quads, PAGE, None, &Settings::default());
    let rect = blocks[0].rect;
    assert_eq!(rect.x, 0);
    assert_eq!(rect.y, 0);
    assert!(rect.x + rect.width <= PAGE.0);
    assert!(rect.y + rect.height <= PAGE.1);
    // The padding did reach past the box on the sides that had room.
    assert!(rect.height > 40);
}

#[test]
fn boxes_of_one_row_are_joined() {
    // Two halves of one line, as the detector often splits one: side by side,
    // one word space apart. Cutting them into separate blocks would hand the
    // model half a line, which is the input it reads worst.
    let quads = vec![
        line(100.0, 100.0, 200.0, 20.0),
        line(310.0, 100.0, 200.0, 20.0),
    ];
    let blocks = assemble(&quads, PAGE, None, &Settings::default());
    assert_eq!(blocks.len(), 1);
    assert_eq!(blocks[0].lines, vec![0, 1]);
    // And the cut spans both halves.
    assert!(blocks[0].rect.width >= 410);
}

#[test]
fn a_column_gutter_is_not_a_word_space() {
    // The same two boxes, now a gutter apart rather than a word apart.
    let quads = vec![
        line(100.0, 100.0, 200.0, 20.0),
        line(420.0, 100.0, 200.0, 20.0),
    ];
    let blocks = assemble(&quads, PAGE, None, &Settings::default());
    assert_eq!(blocks.len(), 2);
}

#[test]
fn a_two_column_page_of_split_lines_still_makes_two_blocks() {
    // Each column's lines arrive split in two, interleaved between columns —
    // the worst case for both passes at once.
    let mut quads = Vec::new();
    for row in 0..3 {
        let y = 100.0 + row as f32 * 30.0;
        quads.push(line(60.0, y, 180.0, 20.0));
        quads.push(line(250.0, y, 180.0, 20.0));
        quads.push(line(520.0, y, 180.0, 20.0));
        quads.push(line(710.0, y, 180.0, 20.0));
    }
    let blocks = assemble(&quads, PAGE, None, &Settings::default());
    assert_eq!(blocks.len(), 2);
    assert_eq!(blocks[0].lines, vec![0, 1, 4, 5, 8, 9]);
    assert_eq!(blocks[1].lines, vec![2, 3, 6, 7, 10, 11]);
}

#[test]
fn cutting_swaps_the_channel_order() {
    // One pixel, blue-first on the page, and it is the block's own row.
    let page = (1u32, 1u32);
    let bgr = vec![10u8, 20, 30];
    let quads = vec![line(0.0, 0.0, 1.0, 1.0)];
    let block = &assemble(&quads, page, None, &Settings::default())[0];
    let cut = cut(&bgr, page, block);
    assert_eq!(cut.get_pixel(0, 0).0, [30, 20, 10]);
}

#[test]
fn a_neighbour_caught_in_the_frame_is_painted_out() {
    // A narrow block with a wide line just above it: the wide line crosses the
    // frame, and reading its middle would duplicate half a line.
    let page = (400u32, 200u32);
    let quads = vec![
        line(10.0, 10.0, 380.0, 20.0),
        line(150.0, 120.0, 100.0, 20.0),
    ];
    let blocks = assemble(&quads, page, None, &Settings::default());
    assert_eq!(blocks.len(), 2);
    let narrow = &blocks[1];
    // The frame stops halfway to the line above rather than reaching into it.
    assert!(narrow.rect.y >= 75, "frame starts at {}", narrow.rect.y);
    // Ink everywhere on the page: whatever survives the cut is the block's.
    let bgr = vec![0u8; (page.0 * page.1 * 3) as usize];
    let cut = cut(&bgr, page, narrow);
    let middle = 120 - narrow.rect.y;
    assert_eq!(cut.get_pixel(0, middle).0, [0, 0, 0]);
}

#[test]
fn a_gap_inside_a_block_is_painted_out() {
    // Two rows of one block with a third line of another block between them
    // horizontally — the frame spans the gap, the mask does not.
    let page = (400u32, 300u32);
    let quads =
        vec![line(20.0, 20.0, 200.0, 20.0), line(20.0, 50.0, 200.0, 20.0)];
    let blocks = assemble(&quads, page, None, &Settings::default());
    assert_eq!(blocks.len(), 1);
    let bgr = vec![0u8; (page.0 * page.1 * 3) as usize];
    let cut = cut(&bgr, page, &blocks[0]);
    // Both rows kept their ink.
    assert_eq!(cut.get_pixel(0, 25 - blocks[0].rect.y).0, [0, 0, 0]);
    assert_eq!(cut.get_pixel(0, 55 - blocks[0].rect.y).0, [0, 0, 0]);
}

#[test]
fn the_boundary_follows_the_ink_when_a_box_is_short() {
    // Two lines whose boxes stop short of their ink: the box above ends at 40,
    // but the line's descenders run to 58, and the box below starts at 62.
    // Halfway (51) would cut through the descenders; the whitest row (60) is
    // where the two lines actually part.
    let page = (200u32, 200u32);
    let quads =
        vec![line(10.0, 20.0, 180.0, 20.0), line(10.0, 62.0, 60.0, 20.0)];
    let mut ink = vec![0u32; page.1 as usize];
    ink[20..58].fill(100);
    ink[62..82].fill(40);
    let blocks = assemble(&quads, page, Some(&ink), &Settings::default());
    assert_eq!(blocks.len(), 2);
    let bgr = vec![0u8; (page.0 * page.1 * 3) as usize];
    let cut = cut(&bgr, page, &blocks[1]);
    // The frame of the lower block starts below the upper line's ink.
    assert!(
        blocks[1].rect.y >= 58,
        "frame starts at {}",
        blocks[1].rect.y
    );
    assert!(cut.height() >= 20);
}
