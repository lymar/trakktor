use super::resolve_model;

fn no_progress() -> impl FnMut(&str, u64, Option<u64>) { |_, _, _| {} }

#[test]
fn unknown_name_lists_known() {
    let tmp = tempfile::tempdir().unwrap();
    let err = resolve_model(tmp.path(), "nope", &mut no_progress())
        .unwrap_err()
        .to_string();
    assert!(err.contains("unknown model `nope`"), "{err}");
    assert!(err.contains("small-streaming-ru"), "{err}");
}

#[test]
fn local_dir_needs_full_bundle() {
    let tmp = tempfile::tempdir().unwrap();
    let dir = tmp.path().join("m");
    std::fs::create_dir(&dir).unwrap();
    std::fs::write(dir.join("encoder.onnx"), b"x").unwrap();
    let err =
        resolve_model(tmp.path(), dir.to_str().unwrap(), &mut no_progress())
            .unwrap_err()
            .to_string();
    assert!(err.contains("has no `decoder.onnx`"), "{err}");
}

#[test]
fn local_dir_with_bundle_resolves() {
    let tmp = tempfile::tempdir().unwrap();
    let dir = tmp.path().join("m");
    std::fs::create_dir(&dir).unwrap();
    for f in ["encoder.onnx", "decoder.onnx", "joiner.onnx", "tokens.txt"] {
        std::fs::write(dir.join(f), b"x").unwrap();
    }
    let resolved =
        resolve_model(tmp.path(), dir.to_str().unwrap(), &mut no_progress())
            .unwrap();
    assert_eq!(resolved.dir, dir);
    assert!(resolved.spec.is_none());
}
