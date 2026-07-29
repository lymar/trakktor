//! The one-time conversion of the published archive. `#[ignore]` — it needs
//! the archive itself, which a first run downloads into the model directory.

use std::path::PathBuf;

use super::*;

/// Converts whatever archive is staged in the model directory, exactly as a
/// first run does. Re-running is a no-op once the files are there.
#[test]
#[ignore = "needs ~/.trakktor/text/stress/silero-ru/accentor.pt"]
fn the_published_archive_converts_into_the_files_a_run_reads() {
    let models =
        PathBuf::from(std::env::var("HOME").expect("HOME")).join(".trakktor");
    let dir = feature_dir(&models).join(DEFAULT_MODEL);
    assert!(
        dir.join("accentor.pt").is_file() || is_converted(&dir),
        "stage the archive in {} first",
        dir.display()
    );
    let resolved = resolve_model(&models, DEFAULT_MODEL, &mut |_, _, _| {})
        .expect("resolve");
    assert!(is_converted(&resolved.dir));
    assert!(
        !resolved.dir.join("accentor.pt").exists(),
        "the archive should be gone once it has been converted"
    );

    // The tables have the sizes the reference reports.
    let count = |file: &str| {
        std::fs::read_to_string(resolved.dir.join(file))
            .expect(file)
            .lines()
            .count()
    };
    assert_eq!(count(NGRAMS_FILE), 126_523);
    assert_eq!(count(EXCEPTIONS_FILE), 16_976);
    assert_eq!(count(HOMOGRAPHS_FILE), 1_924);
    assert_eq!(count(VOCAB_FILE), 83_830);
}
