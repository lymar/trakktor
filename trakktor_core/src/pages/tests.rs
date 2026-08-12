use super::*;

fn selection(spec: &str) -> Selection { Selection::parse(spec).expect(spec) }

#[test]
fn a_single_page_a_range_and_a_list_all_parse() {
    assert_eq!(selection("3").resolve(10).unwrap(), vec![3]);
    assert_eq!(selection("1-3").resolve(10).unwrap(), vec![1, 2, 3]);
    assert_eq!(
        selection("1-2,5,8-9").resolve(10).unwrap(),
        vec![1, 2, 5, 8, 9]
    );
    assert_eq!(selection(" 4 , 2 ").resolve(10).unwrap(), vec![2, 4]);
}

#[test]
fn overlapping_parts_are_merged_and_sorted() {
    assert_eq!(
        selection("5,1-3,2-4").resolve(10).unwrap(),
        vec![1, 2, 3, 4, 5]
    );
}

#[test]
fn an_open_range_runs_to_the_end_and_clamps() {
    assert_eq!(selection("8-").resolve(10).unwrap(), vec![8, 9, 10]);
    // Asking for "the rest" of a document that ends sooner is not an error:
    // the answer is the rest.
    assert_eq!(selection("8-99").resolve(10).unwrap(), vec![8, 9, 10]);
}

#[test]
fn a_selection_starting_past_the_end_is_an_error() {
    let err = selection("11-").resolve(10).unwrap_err();
    assert!(err.to_string().contains("has 10"), "{err}");
}

#[test]
fn nonsense_page_specs_are_rejected() {
    for spec in ["", "abc", "1-x", "5-2", "0", "1,0", ",", "1..3"] {
        assert!(Selection::parse(spec).is_err(), "`{spec}` should not parse");
    }
}
