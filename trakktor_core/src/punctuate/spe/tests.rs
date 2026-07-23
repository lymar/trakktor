//! Unit tests for the SentencePiece `ModelProto` reader, on hand-encoded
//! protobufs.

use super::*;

fn varint(mut v: u64, out: &mut Vec<u8>) {
    loop {
        let mut byte = (v & 0x7f) as u8;
        v >>= 7;
        if v != 0 {
            byte |= 0x80;
        }
        out.push(byte);
        if v == 0 {
            break;
        }
    }
}

fn tag(field: u64, wire: u64, out: &mut Vec<u8>) {
    varint((field << 3) | wire, out);
}

fn len_delim(field: u64, data: &[u8], out: &mut Vec<u8>) {
    tag(field, 2, out);
    varint(data.len() as u64, out);
    out.extend_from_slice(data);
}

fn piece_msg(piece: &str, score: f32, kind: u64) -> Vec<u8> {
    let mut m = Vec::new();
    len_delim(1, piece.as_bytes(), &mut m);
    tag(2, 5, &mut m);
    m.extend_from_slice(&score.to_le_bytes());
    tag(3, 0, &mut m);
    varint(kind, &mut m);
    m
}

/// Builds a `ModelProto` with the given pieces and a normalizer char map.
fn model_proto(pieces: &[(&str, f32, u64)], charsmap: &[u8]) -> Vec<u8> {
    let mut model = Vec::new();
    for &(piece, score, kind) in pieces {
        len_delim(1, &piece_msg(piece, score, kind), &mut model);
    }
    // normalizer_spec (field 3) with precompiled_charsmap (field 2).
    let mut ns = Vec::new();
    len_delim(2, charsmap, &mut ns);
    len_delim(3, &ns, &mut model);
    model
}

#[test]
fn parses_pieces_unk_and_charsmap() {
    // Types: 2 = UNKNOWN, 1 = NORMAL.
    let bytes = model_proto(
        &[("<unk>", 0.0, 2), ("▁a", -1.5, 1), ("b", -2.0, 1)],
        &[1, 2, 3, 4],
    );
    let parsed = parse(&bytes).unwrap();
    assert_eq!(
        parsed.pieces,
        vec![
            ("<unk>".to_string(), 0.0),
            ("▁a".to_string(), -1.5),
            ("b".to_string(), -2.0),
        ]
    );
    assert_eq!(parsed.unk_id, 0);
    assert_eq!(parsed.precompiled_charsmap, vec![1, 2, 3, 4]);
}

#[test]
fn unk_id_tracks_the_unknown_piece() {
    // The unknown piece is not first here (id 1).
    let bytes = model_proto(&[("a", -1.0, 1), ("<unk>", 0.0, 2)], &[]);
    let parsed = parse(&bytes).unwrap();
    assert_eq!(parsed.unk_id, 1);
    assert!(parsed.precompiled_charsmap.is_empty());
}

#[test]
fn rejects_a_model_without_an_unknown_piece() {
    let bytes = model_proto(&[("a", -1.0, 1)], &[]);
    assert!(parse(&bytes).is_err());
}

#[test]
fn skips_unknown_fields() {
    // Prepend an unknown varint field (field 9, wire 0) and an unknown
    // length-delimited field (field 10, wire 2); both must be skipped.
    let mut bytes = Vec::new();
    tag(9, 0, &mut bytes);
    varint(12345, &mut bytes);
    len_delim(10, b"ignored", &mut bytes);
    bytes.extend(model_proto(&[("<unk>", 0.0, 2), ("x", -1.0, 1)], &[9]));
    let parsed = parse(&bytes).unwrap();
    assert_eq!(parsed.pieces.len(), 2);
    assert_eq!(parsed.unk_id, 0);
    assert_eq!(parsed.precompiled_charsmap, vec![9]);
}
