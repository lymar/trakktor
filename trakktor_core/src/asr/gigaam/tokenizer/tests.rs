use super::{SPACE_MARKER, Tokenizer};

fn sp(pieces: &[&str]) -> Tokenizer {
    Tokenizer::sentencepiece(
        pieces.iter().map(|s| s.to_string()).collect(),
        0, // <unk> at id 0
    )
}

fn charwise(chars: &[&str]) -> Tokenizer {
    Tokenizer::charwise(chars.iter().map(|s| s.to_string()).collect())
}

#[test]
fn charwise_decode_joins() {
    let t = charwise(&[" ", "a", "b", "c"]);
    assert_eq!(t.decode(&[1, 2, 0, 3]), "ab c");
    assert_eq!(t.blank_id(), 4);
}

// SentencePiece decode cases below mirror the reference
// `SentencePieceProcessor.decode` behavior verified against the library:
// leading `▁` dropped, interior `▁` → space, `<unk>` → " ⁇ ".
const M: char = SPACE_MARKER;

#[test]
fn sp_drops_leading_space_marker() {
    // ['▁', 'да', ','] -> 'да,'
    let t = sp(&["<unk>", &M.to_string(), "да", ","]);
    assert_eq!(t.decode(&[1, 2, 3]), "да,");
}

#[test]
fn sp_interior_markers_become_spaces() {
    // ['да', '▁', '▁', 'ты'] -> 'да  ты'
    let m = M.to_string();
    let t = sp(&["<unk>", "да", &m, "ты"]);
    assert_eq!(t.decode(&[1, 2, 2, 3]), "да  ты");
}

#[test]
fn sp_trailing_marker_kept() {
    // ['▁', 'да', '▁'] -> 'да '
    let m = M.to_string();
    let t = sp(&["<unk>", &m, "да"]);
    assert_eq!(t.decode(&[1, 2, 1]), "да ");
}

#[test]
fn sp_unknown_expands_to_surface() {
    // ['<unk>', '▁', 'да'] -> ' ⁇  да'
    let m = M.to_string();
    let t = sp(&["<unk>", &m, "да"]);
    assert_eq!(t.decode(&[0, 1, 2]), " \u{2047}  да");
}

#[test]
fn sp_prefixed_piece() {
    // A single '▁слово' piece decodes to 'слово' (leading marker dropped).
    let t = sp(&["<unk>", &format!("{M}слово")]);
    assert_eq!(t.decode(&[1]), "слово");
}
