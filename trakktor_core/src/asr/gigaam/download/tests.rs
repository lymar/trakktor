use super::{resolve_model, verify_md5};

#[test]
fn verify_md5_accepts_known_digest() {
    let dir = tempfile::tempdir().unwrap();
    let file = dir.path().join("data.bin");
    std::fs::write(&file, b"abc").unwrap();
    // md5("abc") = 900150983cd24fb0d6963f7d28e17f72
    verify_md5(&file, "900150983cd24fb0d6963f7d28e17f72").unwrap();
    assert!(verify_md5(&file, "00000000000000000000000000000000").is_err());
}

#[test]
fn resolve_unknown_name_errors() {
    let dir = tempfile::tempdir().unwrap();
    let err = resolve_model(dir.path(), "no_such_model", &mut |_, _, _| {})
        .unwrap_err();
    assert!(format!("{err}").contains("unknown model"));
}

#[test]
fn resolve_local_ckpt_by_stem() {
    // A local file named like a known model resolves without a download.
    let dir = tempfile::tempdir().unwrap();
    let ckpt = dir.path().join("v3_ctc.ckpt");
    std::fs::write(&ckpt, b"not real weights").unwrap();
    let resolved =
        resolve_model(dir.path(), ckpt.to_str().unwrap(), &mut |_, _, _| {})
            .unwrap();
    assert_eq!(resolved.config.model_name, "v3_ctc");
    assert_eq!(resolved.ckpt, ckpt);
}
