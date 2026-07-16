use super::*;

fn dims_with_vocab(n_vocab: usize, n_mels: usize) -> ModelDims {
    ModelDims {
        n_mels,
        n_audio_ctx: 1500,
        n_audio_state: 384,
        n_audio_head: 6,
        n_audio_layer: 4,
        n_vocab,
        n_text_ctx: 448,
        n_text_state: 384,
        n_text_head: 6,
        n_text_layer: 4,
    }
}

#[test]
fn multilinguality_and_language_count_derive_from_vocab_size() {
    // English-only models (gpt2 vocabulary).
    let dims = dims_with_vocab(51864, 80);
    assert!(!dims.is_multilingual());
    assert_eq!(dims.num_languages(), 99);

    // Multilingual models.
    let dims = dims_with_vocab(51865, 80);
    assert!(dims.is_multilingual());
    assert_eq!(dims.num_languages(), 99);

    // large-v3 and turbo vocabularies add Cantonese.
    let dims = dims_with_vocab(51866, 128);
    assert!(dims.is_multilingual());
    assert_eq!(dims.num_languages(), 100);
}

#[test]
fn mel_bands_map_from_dims() {
    let bands = dims_with_vocab(51865, 80).mel_bands().unwrap();
    assert_eq!(bands, MelBands::Mel80);
    let bands = dims_with_vocab(51866, 128).mel_bands().unwrap();
    assert_eq!(bands, MelBands::Mel128);
    assert!(matches!(
        dims_with_vocab(51865, 96).mel_bands(),
        Err(WhisperError::InvalidModel(_))
    ));
}

#[test]
fn alignment_heads_come_from_the_table_or_the_default() {
    let tiny = alignment_heads("tiny").unwrap();
    assert_eq!(tiny, &[(2, 2), (3, 0), (3, 2), (3, 3), (3, 4), (3, 5)]);
    assert_eq!(alignment_heads("turbo"), alignment_heads("large-v3-turbo"));
    assert!(alignment_heads("unknown-model").is_none());

    // The fallback: every head of the last half of the decoder layers.
    let dims = dims_with_vocab(51865, 80);
    let default = dims.default_alignment_heads();
    assert_eq!(default.len(), 2 * 6);
    assert_eq!(default.first(), Some(&(2, 0)));
    assert_eq!(default.last(), Some(&(3, 5)));
}

#[test]
fn logits_index_row_major() {
    // 2 sequences x 3 positions x 4 vocabulary entries.
    let data: Vec<f32> = (0..24).map(|v| v as f32).collect();
    let logits = Logits::new(2, 3, 4, data);
    assert_eq!(logits.row(0, 0), &[0.0, 1.0, 2.0, 3.0]);
    assert_eq!(logits.row(1, 0), &[12.0, 13.0, 14.0, 15.0]);
    assert_eq!(logits.last_position(0), &[8.0, 9.0, 10.0, 11.0]);
    assert_eq!(logits.last_position(1), &[20.0, 21.0, 22.0, 23.0]);
}

#[test]
fn cross_qk_head_slices_are_contiguous() {
    // 2 layers x 2 heads x 1 token x 3 frames.
    let data: Vec<f32> = (0..12).map(|v| v as f32).collect();
    let qk = CrossQk::new(2, 2, 1, 3, data);
    assert_eq!(qk.head(0, 0), &[0.0, 1.0, 2.0]);
    assert_eq!(qk.head(0, 1), &[3.0, 4.0, 5.0]);
    assert_eq!(qk.head(1, 0), &[6.0, 7.0, 8.0]);
    assert_eq!(qk.head(1, 1), &[9.0, 10.0, 11.0]);
}
