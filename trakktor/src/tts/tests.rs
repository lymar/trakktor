//! Tests for resolving what to speak and where to write it.

use super::*;

#[test]
fn the_argument_is_taken_verbatim() {
    let input = read_input(Some("Привет, мир."), None).expect("resolves");

    assert_eq!(input.text, "Привет, мир.");
    // Nothing to take a format hint from.
    assert!(input.source.is_none());
}

#[test]
fn a_file_is_read_whole_and_remembers_its_name() {
    let dir = tempfile::tempdir().expect("temp dir");
    let path = dir.path().join("speech.txt");
    std::fs::write(&path, "Первая строка.\nВторая строка.\n").expect("write");

    let input = read_input(None, Some(&path)).expect("resolves");

    // The text arrives untouched: paragraph splitting happens later, and it
    // needs the line breaks.
    assert_eq!(input.text, "Первая строка.\nВторая строка.\n");
    // The name is kept so `--text-format auto` can believe the extension.
    assert_eq!(input.source.as_deref(), Some(path.as_path()));
}

#[test]
fn a_missing_file_is_an_io_error() {
    let dir = tempfile::tempdir().expect("temp dir");
    let missing = dir.path().join("nope.txt");

    let err = read_input(None, Some(&missing)).unwrap_err();

    let Qwen3TtsError::Io(message) = err else {
        panic!("expected Io, got {err:?}");
    };
    assert!(message.contains("nope.txt"), "{message}");
}

#[test]
fn neither_source_is_an_options_error() {
    let err = read_input(None, None).unwrap_err();
    assert!(matches!(err, Qwen3TtsError::InvalidOptions(_)));
}

#[test]
fn the_precision_default_follows_the_runtime_and_device() {
    use crate::cli::{DeviceArg, RuntimeArg, TtsPrecisionArg};

    let resolve = |precision, runtime, device| {
        resolve_precision(precision, runtime, device)
    };

    // Unspecified: bf16 only where it is served — candle on Metal; the burn
    // runtime and candle's CPU backend fall to the one precision they have.
    assert_eq!(
        resolve(None, RuntimeArg::Candle, DeviceArg::Metal)
            .expect("candle metal default"),
        Precision::Bf16
    );
    assert_eq!(
        resolve(None, RuntimeArg::Candle, DeviceArg::Cpu)
            .expect("candle cpu default"),
        Precision::F32
    );
    assert_eq!(
        resolve(None, RuntimeArg::Burn, DeviceArg::Metal)
            .expect("burn default"),
        Precision::F32
    );

    // An explicit choice is honored where the backend serves it.
    assert_eq!(
        resolve(
            Some(TtsPrecisionArg::F32),
            RuntimeArg::Candle,
            DeviceArg::Cpu
        )
        .expect("candle f32"),
        Precision::F32
    );
    assert_eq!(
        resolve(
            Some(TtsPrecisionArg::Bf16),
            RuntimeArg::Candle,
            DeviceArg::Metal
        )
        .expect("candle metal bf16"),
        Precision::Bf16
    );
    assert_eq!(
        resolve(Some(TtsPrecisionArg::F32), RuntimeArg::Burn, DeviceArg::Cpu)
            .expect("burn f32"),
        Precision::F32
    );

    // Where bf16 does not exist it is rejected rather than downgraded.
    assert!(
        resolve(
            Some(TtsPrecisionArg::Bf16),
            RuntimeArg::Candle,
            DeviceArg::Cpu
        )
        .is_err()
    );
    assert!(
        resolve(
            Some(TtsPrecisionArg::Bf16),
            RuntimeArg::Burn,
            DeviceArg::Metal
        )
        .is_err()
    );
}

#[test]
fn the_output_extension_picks_the_encoder() {
    let target = |path: &str, encoder| output_target(Path::new(path), encoder);

    // What the built-in encoders write, they write — case-insensitively.
    assert!(matches!(
        target("a.wav", AudioEncoderArg::Auto).expect("wav"),
        OutputTarget::Native(encode::Format::Wav)
    ));
    assert!(matches!(
        target("a.FLAC", AudioEncoderArg::Auto).expect("flac"),
        OutputTarget::Native(encode::Format::Flac)
    ));
    // Anything else goes to ffmpeg on its own …
    assert!(matches!(
        target("a.mp3", AudioEncoderArg::Auto).expect("mp3"),
        OutputTarget::Ffmpeg
    ));
    // … unless the built-in encoder was demanded, which fails before a single
    // frame is generated.
    assert!(target("a.mp3", AudioEncoderArg::Builtin).is_err());
    assert!(target("a", AudioEncoderArg::Builtin).is_err());
    // And ffmpeg can be demanded for a format the built-in one also writes.
    assert!(matches!(
        target("a.wav", AudioEncoderArg::Ffmpeg).expect("ffmpeg wav"),
        OutputTarget::Ffmpeg
    ));
}

#[test]
fn the_format_reported_is_the_extension() {
    assert_eq!(format_name(Path::new("a/b.WAV")), "wav");
    assert_eq!(format_name(Path::new("a/b.mp3")), "mp3");
    assert_eq!(format_name(Path::new("a/b")), "");
}
