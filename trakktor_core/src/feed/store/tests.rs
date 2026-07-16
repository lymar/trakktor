use super::*;

// A valid-looking uid (64 lowercase hex chars).
const UID_A: &str =
    "aabbccddeeff00112233445566778899aabbccddeeff00112233445566778899";
const UID_B: &str =
    "0011223344556677889900112233445566778899001122334455667788990011";

#[test]
fn validate_uid_accepts_64_lower_hex() {
    assert!(validate_uid(UID_A).is_ok());
}

#[test]
fn validate_uid_rejects_bad_values() {
    assert!(validate_uid("").is_err());
    assert!(validate_uid("abc").is_err());
    // Uppercase is rejected (must be lowercase).
    assert!(validate_uid(&UID_A.to_uppercase()).is_err());
    // Wrong length.
    assert!(validate_uid(&UID_A[..63]).is_err());
    // Non-hex character.
    let mut bad = UID_A.to_string();
    bad.replace_range(0..1, "g");
    assert!(validate_uid(&bad).is_err());
}

#[test]
fn mark_read_is_idempotent_and_counts() {
    let dir = tempfile::tempdir().unwrap();
    let store = ReadStore::new(dir.path());

    // Nothing is read initially; the store is not created by `is_read`.
    assert!(!store.is_read(UID_A).unwrap());
    assert!(!dir.path().join("feed").exists());

    let s1 = store.mark_read(&[UID_A.to_string()]).unwrap();
    assert_eq!(s1.marked, 1);
    assert_eq!(s1.already_read, 0);
    assert!(store.is_read(UID_A).unwrap());

    // Re-marking is a no-op.
    let s2 = store.mark_read(&[UID_A.to_string()]).unwrap();
    assert_eq!(s2.marked, 0);
    assert_eq!(s2.already_read, 1);
}

#[test]
fn mark_read_dedups_repeated_arguments() {
    let dir = tempfile::tempdir().unwrap();
    let store = ReadStore::new(dir.path());
    let summary = store
        .mark_read(&[UID_A.to_string(), UID_A.to_string(), UID_B.to_string()])
        .unwrap();
    assert_eq!(summary.marked, 2);
    assert_eq!(summary.already_read, 1);
}

#[test]
fn mark_read_validates_before_writing() {
    let dir = tempfile::tempdir().unwrap();
    let store = ReadStore::new(dir.path());
    // A bad uid in the batch fails the whole call without side effects.
    let err = store
        .mark_read(&[UID_A.to_string(), "bad".to_string()])
        .unwrap_err();
    assert!(matches!(err, FeedError::InvalidUid(_)));
    assert!(!store.is_read(UID_A).unwrap());
}

#[test]
fn shard_path_uses_documented_layout() {
    let dir = tempfile::tempdir().unwrap();
    let store = ReadStore::new(dir.path());
    let expected = dir.path().join("feed").join("aa").join("b");
    assert_eq!(
        store.shard_path(
            "aab0000000000000000000000000000000000000000000000000000000000000"
        ),
        expected
    );
}
