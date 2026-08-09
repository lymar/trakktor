use super::*;

/// A page whose every pixel says where it is, so a turn can be checked pixel
/// by pixel rather than by eye.
fn marked(width: usize, height: usize) -> Page {
    let mut bgr = vec![0u8; width * height * CHANNELS];
    for y in 0..height {
        for x in 0..width {
            let at = (y * width + x) * CHANNELS;
            bgr[at] = x as u8;
            bgr[at + 1] = y as u8;
            bgr[at + 2] = 0;
        }
    }
    Page {
        width: width as u32,
        height: height as u32,
        bgr,
    }
}

fn pixel(page: &Page, x: usize, y: usize) -> (u8, u8) {
    let at = (y * page.width as usize + x) * CHANNELS;
    (page.bgr[at], page.bgr[at + 1])
}

#[test]
fn a_quarter_turn_transposes_the_page() {
    let page = marked(7, 5);
    let turned = turn(&page, 90);
    assert_eq!((turned.width, turned.height), (5, 7));
    // Counter-clockwise: the top right corner of the page becomes the top
    // left of the turn.
    assert_eq!(pixel(&page, 6, 0), (6, 0));
    assert_eq!(pixel(&turned, 0, 0), (6, 0));
}

#[test]
fn three_quarter_turns_are_the_other_way_round() {
    let page = marked(7, 5);
    let turned = turn(&page, 270);
    assert_eq!((turned.width, turned.height), (5, 7));
    // Clockwise: the top left corner goes to the top right.
    assert_eq!(pixel(&turned, 4, 0), (0, 0));
}

#[test]
fn four_quarter_turns_come_back_to_the_page() {
    let page = marked(7, 5);
    let mut turned = page.clone();
    for _ in 0..4 {
        turned = turn(&turned, 90);
    }
    assert_eq!(turned.width, page.width);
    assert_eq!(turned.height, page.height);
    assert_eq!(turned.bgr, page.bgr);
}

#[test]
fn undoing_a_turn_lands_on_the_pixel_it_came_from() {
    let page = marked(7, 5);
    for degrees in [0u16, 90, 180, 270] {
        let turned = turn(&page, degrees);
        for y in 0..turned.height as usize {
            for x in 0..turned.width as usize {
                let (ox, oy) = unturn((x as f32, y as f32), degrees, 7.0, 5.0);
                assert_eq!(
                    pixel(&turned, x, y),
                    (ox as u8, oy as u8),
                    "{degrees} at ({x}, {y})"
                );
            }
        }
    }
}

#[test]
fn the_square_is_cut_from_the_middle_of_the_short_side() {
    let page = marked(120, 300);
    let tensor = page_tensor(&page, &candle_core::Device::Cpu).unwrap();
    assert_eq!(tensor.dims(), &[1, CHANNELS, SIDE, SIDE]);
}

#[test]
fn a_page_with_no_pixels_is_refused() {
    let page = Page {
        width: 0,
        height: 0,
        bgr: Vec::new(),
    };
    assert!(page_tensor(&page, &candle_core::Device::Cpu).is_err());
}
