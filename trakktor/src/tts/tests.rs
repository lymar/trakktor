//! Tests for resolving what to speak and where to write it.

use super::*;

#[test]
fn the_argument_is_taken_verbatim() {
    let text = resolve_text(Some("Привет, мир."), None).expect("resolves");
    assert_eq!(text, "Привет, мир.");
}

#[test]
fn a_file_is_read_and_its_line_breaks_collapsed() {
    let dir = tempfile::tempdir().expect("temp dir");
    let path = dir.path().join("speech.txt");
    // Hard-wrapped prose with a blank line, as a real text file has.
    std::fs::write(
        &path,
        "  Речь в этой книге\nпойдет главным образом\n\nо хоббитах.  \n",
    )
    .expect("write");

    let text = resolve_text(None, Some(&path)).expect("resolves");

    // One utterance of running prose: no newlines, no doubled spaces, trimmed.
    assert_eq!(text, "Речь в этой книге пойдет главным образом о хоббитах.");
}

#[test]
fn a_missing_file_is_an_io_error() {
    let dir = tempfile::tempdir().expect("temp dir");
    let missing = dir.path().join("nope.txt");

    let err = resolve_text(None, Some(&missing)).unwrap_err();

    let Qwen3TtsError::Io(message) = err else {
        panic!("expected Io, got {err:?}");
    };
    assert!(message.contains("nope.txt"), "{message}");
}

#[test]
fn neither_source_is_an_options_error() {
    let err = resolve_text(None, None).unwrap_err();
    assert!(matches!(err, Qwen3TtsError::InvalidOptions(_)));
}

#[test]
fn the_precision_default_follows_the_runtime() {
    use crate::cli::{RuntimeArg, TtsPrecisionArg};

    // Unspecified: candle keeps bf16, burn falls to the only precision it
    // serves.
    assert_eq!(
        resolve_precision(None, RuntimeArg::Candle).expect("candle default"),
        Precision::Bf16
    );
    assert_eq!(
        resolve_precision(None, RuntimeArg::Burn).expect("burn default"),
        Precision::F32
    );

    // An explicit choice is honored on candle, either way.
    assert_eq!(
        resolve_precision(Some(TtsPrecisionArg::F32), RuntimeArg::Candle)
            .expect("candle f32"),
        Precision::F32
    );
    assert_eq!(
        resolve_precision(Some(TtsPrecisionArg::Bf16), RuntimeArg::Candle)
            .expect("candle bf16"),
        Precision::Bf16
    );

    // f32 is fine on burn; bf16 is rejected rather than downgraded.
    assert_eq!(
        resolve_precision(Some(TtsPrecisionArg::F32), RuntimeArg::Burn)
            .expect("burn f32"),
        Precision::F32
    );
    assert!(
        resolve_precision(Some(TtsPrecisionArg::Bf16), RuntimeArg::Burn)
            .is_err()
    );
}

#[test]
fn the_output_extension_picks_the_container() {
    assert!(matches!(
        output_format(Path::new("a.wav")).expect("wav"),
        Format::Wav
    ));
    // The extension is matched case-insensitively.
    assert!(matches!(
        output_format(Path::new("a.FLAC")).expect("flac"),
        Format::Flac
    ));
    // An unusable extension is reported before any model is downloaded.
    assert!(output_format(Path::new("a.mp3")).is_err());
    assert!(output_format(Path::new("a")).is_err());
}
