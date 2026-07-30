//! Tests for the lookup tables a converted model carries.

use std::collections::HashMap;

use super::Tables;

/// A stand-in for the published base alphabet: the same first entries, the
/// same `symbols` string that starts three symbols in.
fn tables() -> Tables {
    Tables {
        symbols: "_~|!'+,-. абя".chars().map(|c| c.to_string()).collect(),
        alphabet: "|!'+,-. абя".to_owned(),
        letters: "абя".to_owned(),
        sos: "|".to_owned(),
        eos: "~".to_owned(),
        speakers: vec!["ru_one".to_owned(), "kat_two".to_owned()],
        translit: HashMap::from([(
            "kat".to_owned(),
            HashMap::from([("ა".to_owned(), "а".to_owned())]),
        )]),
    }
}

#[test]
fn the_kept_set_starts_three_symbols_into_the_alphabet_string() {
    let tables = tables();
    // `|`, `!` and `'` are the three the reference's own slice drops.
    for dropped in ['|', '!', '\''] {
        assert!(!tables.keeps(dropped), "{dropped} should be dropped");
    }
    for kept in ['+', ',', '-', '.', ' ', 'а'] {
        assert!(tables.keeps(kept), "{kept} should be kept");
    }
}

#[test]
fn a_symbol_maps_to_its_position() {
    let index = tables().index();
    assert_eq!(index.get(&'_'), Some(&0));
    assert_eq!(index.get(&'+'), Some(&5));
    assert_eq!(index.get(&'э'), None);
}

#[test]
fn a_speaker_maps_to_its_row() {
    let tables = tables();
    assert_eq!(tables.speaker("kat_two"), Some(1));
    assert_eq!(tables.speaker("nobody"), None);
}

#[test]
fn tables_survive_a_round_trip_through_the_file() {
    let dir = tempfile::tempdir().expect("a temporary directory");
    let original = tables();
    original.store(dir.path()).expect("should write");
    let read = Tables::load(dir.path()).expect("should read");
    assert_eq!(read.symbols, original.symbols);
    assert_eq!(read.alphabet, original.alphabet);
    assert_eq!(read.letters, original.letters);
    assert_eq!(read.speakers, original.speakers);
    assert_eq!(read.translit, original.translit);
}
