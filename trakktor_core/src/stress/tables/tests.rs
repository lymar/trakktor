//! Loading the model's tables: the line format and the invariants the marking
//! rules index by. The real tables are covered by the ignored parity tests;
//! here the malformed cases are rejected loudly instead of marking words
//! slightly wrong.

use std::path::Path;

use super::*;

/// Writes the three table files into `dir` and loads them.
fn load(
    dir: &Path,
    ngrams: &str,
    exceptions: &str,
    homographs: &str,
) -> Result<Tables, StressError> {
    std::fs::write(dir.join(NGRAMS_FILE), ngrams).expect("ngrams");
    std::fs::write(dir.join(EXCEPTIONS_FILE), exceptions).expect("exceptions");
    std::fs::write(dir.join(HOMOGRAPHS_FILE), homographs).expect("homographs");
    Tables::load(dir)
}

#[test]
fn well_formed_tables_load() {
    let dir = tempfile::tempdir().expect("tempdir");
    let tables = load(
        dir.path(),
        "а\nUNK\n",
        "кот 1 -1\nнее 2 2\n",
        "замки з+амки замк+и\n",
    )
    .expect("load");
    assert_eq!(tables.unk, 1);
    assert_eq!(tables.exceptions.len(), 2);
    assert_eq!(tables.homographs.len(), 1);
}

#[test]
fn a_missing_unk_row_is_a_corrupt_table() {
    let dir = tempfile::tempdir().expect("tempdir");
    let error = load(dir.path(), "а\n", "", "");
    assert!(matches!(error, Err(StressError::InvalidModel(_))));
}

#[test]
fn a_position_outside_the_word_is_a_corrupt_table() {
    let dir = tempfile::tempdir().expect("tempdir");
    for exceptions in ["кот 3 -1\n", "кот 1 3\n"] {
        let error = load(dir.path(), "UNK\n", exceptions, "");
        assert!(
            matches!(error, Err(StressError::InvalidModel(_))),
            "expected a table error for {exceptions:?}"
        );
    }
}

#[test]
fn a_variant_that_does_not_fit_its_word_is_a_corrupt_table() {
    let dir = tempfile::tempdir().expect("tempdir");
    for homographs in [
        "замки замки замк+и\n", // the first variant carries no mark
        "замки з+амки за+мк+и\n", // two marks
        "замки з+амк замк+и\n", // a variant of the wrong length
    ] {
        let error = load(dir.path(), "UNK\n", "", homographs);
        assert!(
            matches!(error, Err(StressError::InvalidModel(_))),
            "expected a table error for {homographs:?}"
        );
    }
}
