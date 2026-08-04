//! Golden-trace parity for the burn runtime, against the same reference dump
//! the candle tests read (see [`vl::tests`](crate::ocr::vl::tests)).
//!
//! Everything here runs on the ndarray backend in f32 — the precision the
//! parity contract is stated in. The layers are the domain's: the tower's
//! tensors strictly, then the token sequence exactly. The prompt and its
//! positions need no second test — they are computed by the shared driver,
//! outside any runtime.

use burn::{
    backend::ndarray::{NdArray, NdArrayDevice},
    tensor::{Tensor, TensorData},
};
use tokenizers::Tokenizer;

use super::{
    Weights, load_cpu,
    vision::{Projector, Tower},
};
use crate::ocr::vl::{
    config::{ImageConfig, ModelConfig, TOKENIZER_FILE},
    generate::{Limits, Reader, Task},
    tests::{blob, checkpoint, picture, relative, trace},
};

type B = NdArray<f32>;

#[test]
#[ignore = "needs ~/.trakktor/ocr/vl and tmp/ocr/vl/golden"]
fn the_vision_tower_and_projector_match_the_reference() {
    let prepared = picture();
    let dir = checkpoint();
    let cfg = ModelConfig::load(&dir).unwrap();
    let weights = Weights::open(&dir).unwrap();
    let device = NdArrayDevice::Cpu;
    let tower = Tower::<B>::load(&weights, &device, &cfg.vision).unwrap();
    let projector =
        Projector::<B>::load(&weights, &device, &cfg.vision, cfg.hidden_size)
            .unwrap();

    // The **reference's own** patches go in, not ours, as in the candle test:
    // fed our patches, this would measure the resampling, not the tower.
    let theirs = blob("pixel_values");
    let pixels = Tensor::<B, 2>::from_data(
        TensorData::new(
            theirs.clone(),
            [prepared.patches(), theirs.len() / prepared.patches()],
        ),
        &device,
    );

    let ours = tower.forward(pixels, prepared.grid);
    let tower_error =
        relative(&ours.to_data().to_vec::<f32>().unwrap(), &blob("tower"));
    assert!(tower_error < 1e-3, "vision tower differs by {tower_error}");

    let ours = projector.forward(ours, prepared.grid);
    let projector_error =
        relative(&ours.to_data().to_vec::<f32>().unwrap(), &blob("projector"));
    assert!(
        projector_error < 1e-3,
        "projector differs by {projector_error}"
    );
}

#[test]
#[ignore = "needs ~/.trakktor/ocr/vl and tmp/ocr/vl/golden"]
fn the_generated_tokens_match_the_reference() {
    let prepared = picture();
    let dir = checkpoint();
    let cfg = ModelConfig::load(&dir).unwrap();
    let image_cfg = ImageConfig::load(&dir).unwrap();
    let tokenizer = Tokenizer::from_file(dir.join(TOKENIZER_FILE)).unwrap();
    let mut reader = Reader::new(
        load_cpu(&dir, cfg).unwrap(),
        tokenizer,
        ModelConfig::load(&dir).unwrap(),
        image_cfg,
    )
    .unwrap();
    let trace = trace();

    let answer = reader
        .read(&prepared, Task::Ocr, &Limits::default())
        .unwrap();
    assert_eq!(answer.text, trace["text"].as_str().unwrap().trim());
    assert!(!answer.truncated, "the answer was cut short");
    assert!(
        answer.score > 0.8,
        "mean token probability {}",
        answer.score
    );
}
