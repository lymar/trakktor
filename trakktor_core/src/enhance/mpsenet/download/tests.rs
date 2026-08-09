//! Model resolution, without a network.

use super::*;

/// A progress sink that is never called, because nothing here downloads.
fn no_progress(_: &str, _: u64, _: Option<u64>) {
    unreachable!("resolution should not reach the network here")
}

#[test]
fn every_published_name_has_a_checkpoint() {
    for name in KNOWN_MODELS {
        let checkpoint = find(name).expect("a published checkpoint");
        assert_eq!(&checkpoint.name, name);
        assert!(checkpoint.size > 0);
        assert_eq!(checkpoint.blake3.len(), 64);
        assert!(download_size(name) > 0);
    }
    assert!(KNOWN_MODELS.contains(&DEFAULT_MODEL));
}

#[test]
fn the_two_checkpoints_are_distinct() {
    let dns = find("dns").expect("dns");
    let vb = find("vb").expect("vb");
    assert_ne!(dns.blake3, vb.blake3);
    assert_ne!(dns.file, vb.file);
}

#[test]
fn an_unknown_name_names_the_known_ones() {
    let dir = std::env::temp_dir().join("trakktor-mpsenet-unknown");
    let error = resolve_model(&dir, "no-such-model", &mut no_progress)
        .expect_err("an unknown model is an error");
    let message = error.to_string();
    for name in KNOWN_MODELS {
        assert!(message.contains(name), "{message}");
    }
}

#[test]
fn a_directory_without_weights_says_so() {
    let dir = std::env::temp_dir().join("trakktor-mpsenet-empty");
    std::fs::create_dir_all(&dir).expect("a scratch directory");
    let error = resolve_model(
        &std::env::temp_dir(),
        &dir.display().to_string(),
        &mut no_progress,
    )
    .expect_err("a directory with no weights is an error");
    assert!(error.to_string().contains(WEIGHTS_FILE), "{error}");
    let _ = std::fs::remove_dir(&dir);
}

#[test]
fn each_model_gets_its_own_directory() {
    let root = std::path::Path::new("/models");
    assert_ne!(model_dir(root, "dns"), model_dir(root, "vb"));
    assert!(model_dir(root, "dns").ends_with("enhance/mpsenet/dns"));
}
