use super::{DitConfig, ShapeIndex, VocoderConfig};

/// The shapes a published checkpoint has, as far as the geometry cares.
fn dit_shapes() -> ShapeIndex {
    let mut shapes = ShapeIndex::new();
    let mut put = |name: &str, shape: &[usize]| {
        shapes.insert(name.to_owned(), shape.to_vec());
    };
    put("transformer.proj_out.weight", &[100, 1024]);
    put("transformer.text_embed.text_embed.weight", &[2546, 512]);
    put(
        "transformer.transformer_blocks.0.ff.ff.0.0.weight",
        &[2048, 1024],
    );
    put(
        "transformer.text_embed.text_blocks.0.pwconv1.weight",
        &[1024, 512],
    );
    put("transformer.time_embed.time_mlp.0.weight", &[1024, 256]);
    put(
        "transformer.input_embed.conv_pos_embed.conv1d.0.weight",
        &[1024, 64, 31],
    );
    for block in 0..22 {
        put(
            &format!("transformer.transformer_blocks.{block}.attn.to_q.weight"),
            &[1024, 1024],
        );
    }
    for block in 0..4 {
        put(
            &format!(
                "transformer.text_embed.text_blocks.{block}.dwconv.weight"
            ),
            &[512, 1, 7],
        );
    }
    shapes
}

#[test]
fn the_published_geometry_is_derived_from_the_tensors() {
    let cfg = DitConfig::derive(&dit_shapes()).expect("geometry");
    assert_eq!(
        cfg,
        DitConfig {
            dim: 1024,
            depth: 22,
            heads: 16,
            ff_inner: 2048,
            mel_channels: 100,
            text_dim: 512,
            text_ff_inner: 1024,
            text_conv_layers: 4,
            vocab_size: 2545,
            time_dim: 256,
            conv_pos_kernel: 31,
            conv_pos_groups: 16,
        }
    );
}

#[test]
fn a_missing_tensor_is_named_in_the_error() {
    let mut shapes = dit_shapes();
    shapes.remove("transformer.proj_out.weight");
    let error = DitConfig::derive(&shapes).expect_err("should fail");
    assert!(
        error.to_string().contains("transformer.proj_out.weight"),
        "{error}"
    );
}

#[test]
fn the_vocoder_geometry_comes_from_its_own_tensors() {
    let mut shapes = ShapeIndex::new();
    shapes.insert("backbone.embed.weight".into(), vec![512, 100, 7]);
    shapes.insert("backbone.convnext.0.pwconv1.weight".into(), vec![1536, 512]);
    shapes.insert("head.out.weight".into(), vec![1026, 512]);
    for block in 0..8 {
        shapes.insert(format!("backbone.convnext.{block}.gamma"), vec![512]);
    }
    let cfg = VocoderConfig::derive(&shapes).expect("geometry");
    assert_eq!(
        cfg,
        VocoderConfig {
            mel_channels: 100,
            dim: 512,
            ff_inner: 1536,
            layers: 8,
            n_fft: 1024,
        }
    );
}
