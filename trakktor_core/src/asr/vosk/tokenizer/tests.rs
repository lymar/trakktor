use super::Tokenizer;

const TABLE: &str =
    "<blk> 0\n<sos/eos> 1\n<unk> 2\nе 3\n▁с 4\nт 5\n▁в 6\n▁ 7\n#0 8\n";

#[test]
fn parses_and_decodes() {
    let t = Tokenizer::parse(TABLE).unwrap();
    assert_eq!(t.len(), 9);
    assert_eq!(t.unk_id(), Some(2));
    assert_eq!(t.id_to_str(4), "▁с");
    // "▁с" "е" "▁в" "т" -> "се вт"; the leading marker is dropped.
    assert_eq!(t.decode(&[4, 3, 6, 5]), "се вт");
    assert_eq!(t.decode(&[]), "");
    // A bare "▁" piece renders as a space separator.
    assert_eq!(t.decode(&[3, 7, 3]), "е е");
}

#[test]
fn rejects_sparse_ids() {
    let err = Tokenizer::parse("<blk> 0\nx 2\n").unwrap_err();
    assert!(err.to_string().contains("dense"));
}

#[test]
fn space_piece_parses() {
    // The piece may itself be a space (rsplit keeps it intact).
    let t = Tokenizer::parse("  0\nx 1\n").unwrap();
    assert_eq!(t.id_to_str(0), " ");
}
