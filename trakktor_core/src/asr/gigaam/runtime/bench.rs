//! Ad-hoc timing to locate the per-chunk bottleneck. `#[ignore]`.

use std::time::Instant;

use candle_core::{Device, IndexOp};

use super::{GigaamModel, Precision};
use crate::asr::gigaam::config::config_for;

fn ckpt(name: &str) -> std::path::PathBuf {
    let home = std::env::var("HOME").expect("HOME");
    std::path::PathBuf::from(home)
        .join(".cache/gigaam")
        .join(format!("{name}.ckpt"))
}

#[test]
#[ignore = "bench: TRAKKTOR_BENCH_DEVICE=metal cargo test --release ... -- \
            --ignored --nocapture"]
fn bench_chunk() {
    let device = match std::env::var("TRAKKTOR_BENCH_DEVICE").as_deref() {
        Ok("metal") => Device::new_metal(0).unwrap(),
        _ => Device::Cpu,
    };
    let precision = match std::env::var("TRAKKTOR_BENCH_PREC").as_deref() {
        Ok("f16") => Precision::F16,
        _ => Precision::F32,
    };
    println!("device={device:?} precision={precision:?}");
    let config = config_for("v3_ctc").unwrap();
    let model =
        GigaamModel::load(&ckpt("v3_ctc"), &config, device, precision).unwrap();

    // A ~22 s chunk of white-ish noise (content does not matter for timing).
    let n = 22 * 16000;
    let chunk: Vec<f32> =
        (0..n).map(|i| (i as f32 * 0.001).sin() * 0.1).collect();

    // Warm up (first call compiles Metal kernels).
    let mel = model.feature().log_mel(&chunk);
    let enc = model.encode(&mel).unwrap();
    let _ = model
        .ctc_logits(&enc)
        .unwrap()
        .argmax(candle_core::D::Minus1);

    let reps = 5;
    let t = Instant::now();
    for _ in 0..reps {
        let _ = model.feature().log_mel(&chunk);
    }
    println!(
        "feature: {:.1} ms/chunk",
        t.elapsed().as_secs_f64() * 1000.0 / reps as f64
    );

    let mel = model.feature().log_mel(&chunk);
    let t = Instant::now();
    for _ in 0..reps {
        let enc = model.encode(&mel).unwrap();
        // force completion
        let _ = enc
            .i(0)
            .and_then(|r| r.sum_all())
            .and_then(|s| s.to_scalar::<f32>());
    }
    println!(
        "encode(+sync): {:.1} ms/chunk",
        t.elapsed().as_secs_f64() * 1000.0 / reps as f64
    );

    let t = Instant::now();
    for _ in 0..reps {
        let enc = model.encode(&mel).unwrap();
        let logits = model.ctc_logits(&enc).unwrap();
        let _ = logits
            .argmax(candle_core::D::Minus1)
            .unwrap()
            .to_vec1::<u32>();
    }
    println!(
        "encode+ctc+argmax: {:.1} ms/chunk",
        t.elapsed().as_secs_f64() * 1000.0 / reps as f64
    );
}
