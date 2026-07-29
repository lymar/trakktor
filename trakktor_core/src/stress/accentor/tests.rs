//! The word-level rules, tested against hand-built tables and hand-made
//! predictions — the network's numbers are checked separately, against the
//! reference.

use super::*;
use crate::stress::text::tokenize;

/// The one token a text of a single word produces.
fn token(text: &str) -> Token {
    tokenize(text).into_iter().next().expect("one token")
}

/// A prediction with both heads confident.
fn confident(stress_id: usize, yo_id: usize) -> WordScores {
    WordScores {
        stress_id,
        stress_prob: 0.9,
        yo_id,
        yo_prob: 0.9,
    }
}

fn empty_tables() -> Tables { Tables::fixture(&[], &[], &[]) }

#[test]
fn a_bag_is_every_substring_of_the_wrapped_word() {
    // `<кот>` has four one-character n-grams in the table and one three-
    // character one; everything else it slices is unknown and dropped.
    let tables = Tables::fixture(&["<", "к", "о", "т", ">", "кот"], &[], &[]);
    let rows = bag("кот", &tables);
    let names: Vec<&str> = rows
        .iter()
        .map(|row| {
            tables
                .ngrams
                .iter()
                .find(|(_, index)| *index == row)
                .map(|(gram, _)| &**gram)
                .expect("row")
        })
        .collect();
    assert_eq!(names, ["<", "к", "о", "т", ">", "кот"]);
}

#[test]
fn a_word_matching_nothing_falls_back_to_one_row() {
    let tables = Tables::fixture(&["а"], &[], &[]);
    assert_eq!(bag("кот", &tables), vec![tables.unk]);
}

#[test]
fn the_head_marks_the_vowel_it_names() {
    let tables = empty_tables();
    // `молоко`: vowels are о(1), о(3), о(5); the head names the third.
    let marked = mark(&token("молоко"), Some(&confident(2, 0)), &tables, true);
    assert_eq!(marked.text, "молок+о");
    assert!(marked.outcome.stressed);
}

#[test]
fn an_unsure_head_leaves_the_word_alone() {
    let tables = empty_tables();
    let unsure = WordScores {
        stress_id: 2,
        stress_prob: 0.4,
        yo_id: 0,
        yo_prob: 0.9,
    };
    let marked = mark(&token("молоко"), Some(&unsure), &tables, true);
    assert_eq!(marked.text, "молоко");
    assert!(!marked.outcome.stressed);
}

#[test]
fn a_word_with_one_vowel_is_marked_whatever_the_head_thinks() {
    let tables = empty_tables();
    let unsure = WordScores {
        stress_id: 5,
        stress_prob: 0.1,
        yo_id: 0,
        yo_prob: 0.1,
    };
    let marked = mark(&token("кот"), Some(&unsure), &tables, true);
    assert_eq!(marked.text, "к+от");
    assert!(marked.outcome.stressed);
}

#[test]
fn a_word_without_vowels_is_left_alone() {
    let tables = empty_tables();
    let marked = mark(&token("кс"), Some(&confident(0, 0)), &tables, true);
    assert_eq!(marked.text, "кс");
    assert!(!marked.outcome.stressed);
}

#[test]
fn yo_is_restored_only_where_the_stress_agrees_with_it() {
    let tables = empty_tables();
    // `королев`: the vowels are о(1), о(3), е(5) and the only `е` is at 5.
    // Stress on the third vowel and `ё` on the first `е` — the same letter.
    let marked = mark(&token("королев"), Some(&confident(2, 1)), &tables, true);
    assert_eq!(marked.text, "корол+ёв");
    assert!(marked.outcome.restored_yo);

    // The same `ё` prediction with the stress elsewhere is a contradiction,
    // and the letter stays as written.
    let marked = mark(&token("королев"), Some(&confident(1, 1)), &tables, true);
    assert_eq!(marked.text, "кор+олев");
    assert!(!marked.outcome.restored_yo);
}

#[test]
fn turning_yo_off_leaves_every_letter_where_it_was() {
    let tables = Tables::fixture(&[], &[("учетом", 2, Some(2))], &[]);
    // From the model…
    let marked =
        mark(&token("королев"), Some(&confident(2, 1)), &tables, false);
    assert_eq!(marked.text, "корол+ев");
    // …and from the exception table, where the reference writes it anyway.
    let marked = mark(&token("учетом"), None, &tables, false);
    assert_eq!(marked.text, "уч+етом");
    let marked = mark(&token("учетом"), None, &tables, true);
    assert_eq!(marked.text, "уч+ётом");
}

#[test]
fn a_written_yo_is_stressed_without_asking_the_model() {
    let tables = empty_tables();
    let marked = mark(&token("мёд"), None, &tables, true);
    assert_eq!(marked.text, "м+ёд");
    // Both marks already there: nothing to do.
    let marked = mark(&token("м+ёд"), None, &tables, true);
    assert_eq!(marked.text, "м+ёд");
}

#[test]
fn the_callers_own_mark_is_never_moved() {
    let tables = empty_tables();
    let marked = mark(&token("мол+око"), Some(&confident(2, 0)), &tables, true);
    assert_eq!(marked.text, "мол+око");
    assert!(marked.outcome.stressed);
}

#[test]
fn an_exception_is_marked_by_position() {
    let tables =
        Tables::fixture(&[], &[("его", 2, None), ("нее", 2, Some(2))], &[]);
    assert_eq!(mark(&token("его"), None, &tables, true).text, "ег+о");
    assert_eq!(mark(&token("Нее"), None, &tables, true).text, "Не+ё");
}

#[test]
fn a_word_the_rules_skip_comes_back_untouched() {
    let tables = empty_tables();
    let marked = mark(&token(", "), None, &tables, true);
    assert_eq!(marked.text, ", ");
    assert!(!marked.outcome.stressed);
}
