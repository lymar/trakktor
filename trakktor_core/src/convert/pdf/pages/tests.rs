use std::collections::BTreeMap;

use super::*;

#[test]
fn pages_are_cut_apart_at_the_markers() {
    let markdown = "<!-- Page 1 -->\n\nfirst\n\n<!-- Page 2 -->\n\nsecond\n";
    let pages = split_pages(markdown, &[1, 2]);
    assert_eq!(pages.get(&1).map(String::as_str), Some("first"));
    assert_eq!(pages.get(&2).map(String::as_str), Some("second"));
}

#[test]
fn a_page_the_engine_wrote_nothing_for_is_absent_rather_than_empty() {
    let markdown = "<!-- Page 1 -->\n\nfirst\n\n<!-- Page 2 -->\n\n\n<!-- \
                    Page 3 -->\n\nthird\n";
    let pages = split_pages(markdown, &[1, 2, 3]);
    assert_eq!(pages.keys().copied().collect::<Vec<_>>(), vec![1, 3]);
}

#[test]
fn markers_carry_the_documents_own_numbers_not_a_counter() {
    let markdown = "<!-- Page 7 -->\n\nseventh\n\n<!-- Page 9 -->\n\nninth\n";
    let pages = split_pages(markdown, &[7, 9]);
    assert_eq!(pages.get(&7).map(String::as_str), Some("seventh"));
    assert_eq!(pages.get(&9).map(String::as_str), Some("ninth"));
}

#[test]
fn text_ahead_of_the_first_marker_belongs_to_the_first_page() {
    let pages = split_pages("stray\n\n<!-- Page 4 -->\n\nfourth\n", &[4, 5]);
    assert_eq!(pages.get(&4).map(String::as_str), Some("stray\n\nfourth"));
}

#[test]
fn one_page_without_markers_is_still_that_page() {
    assert_eq!(
        split_pages("only\n", &[6]).get(&6).map(String::as_str),
        Some("only")
    );
}

#[test]
fn several_pages_without_markers_are_attributed_to_none() {
    // Handing one blob of text to two pages would be an invention; better to
    // report both as having produced nothing.
    assert!(split_pages("only\n", &[1, 2]).is_empty());
}

#[test]
fn private_use_characters_are_counted_across_all_three_areas() {
    assert_eq!(private_use_chars("plain text"), 0);
    assert_eq!(private_use_chars("a\u{E000}b\u{F8FF}"), 2);
    assert_eq!(private_use_chars("\u{F0000}\u{100000}"), 2);
    // Tibetan is not private use, however unfamiliar it looks.
    assert_eq!(private_use_chars("\u{0F40}\u{0F0B}"), 0);
}

#[test]
fn a_stamp_repeated_across_the_document_is_furniture() {
    let mut pages = BTreeMap::new();
    for number in 1..=6 {
        pages.insert(number, "DRAFT COPY".to_string());
    }
    pages.insert(7, "A real page with something on it.".to_string());
    let furniture = furniture_pages(&pages);
    assert_eq!(furniture.len(), 6);
    assert!(!furniture.contains(&7));
}

#[test]
fn line_breaks_do_not_make_two_stamps_different() {
    let mut pages = BTreeMap::new();
    for number in 1..=5 {
        let text = if number % 2 == 0 {
            "DRAFT\nCOPY"
        } else {
            "DRAFT COPY"
        };
        pages.insert(number, text.to_string());
    }
    assert_eq!(furniture_pages(&pages).len(), 5);
}

#[test]
fn a_few_identical_pages_are_a_coincidence_and_are_kept() {
    let mut pages = BTreeMap::new();
    for number in 1..=4 {
        pages.insert(number, "Blank page".to_string());
    }
    assert!(furniture_pages(&pages).is_empty());
}

#[test]
fn a_long_page_repeated_is_content_not_furniture() {
    let long = "word ".repeat(80);
    let mut pages = BTreeMap::new();
    for number in 1..=8 {
        pages.insert(number, long.clone());
    }
    assert!(furniture_pages(&pages).is_empty());
}
