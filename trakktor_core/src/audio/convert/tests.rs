use super::*;

#[test]
fn u8_is_offset_binary() {
    assert_eq!(u8_to_s16(0x80), 0);
    assert_eq!(u8_to_s16(0x00), -32768);
    assert_eq!(u8_to_s16(0xff), 32512); // 127 << 8
}

#[test]
fn widening_scalings_are_exact() {
    assert_eq!(s16_to_f32(-32768), -1.0);
    assert_eq!(s16_to_f32(16384), 0.5);
    assert_eq!(s32_to_f32(i32::MIN), -1.0);
    assert_eq!(s32_to_f32(1 << 30), 0.5);
}

#[test]
fn f32_quantizer_basics() {
    assert_eq!(f32_to_s16(0.0), 0);
    assert_eq!(f32_to_s16(0.5), 16384);
    assert_eq!(f32_to_s16(-0.5), -16384);
    // Full scale saturates: 1.0 * 32768 clips to 32767.
    assert_eq!(f32_to_s16(1.0), 32767);
    assert_eq!(f32_to_s16(-1.0), -32768);
    assert_eq!(f32_to_s16(2.0), 32767);
    assert_eq!(f32_to_s16(-2.0), -32768);
    assert_eq!(f32_to_s16(f32::NAN), 0);
}

/// Half-LSB ties: the NEON rule rounds them up (toward +∞), the C rule to
/// even. This is the sole value class where the two reference paths differ.
#[test]
fn f32_quantizer_tie_semantics() {
    let tie = |k: i32| (k as f32 + 0.5) / 32768.0;
    // 2.5 → NEON 3, lrintf 2 (even).
    assert_eq!(f32_to_s16(tie(2)), 3);
    assert_eq!(f32_to_s16_lrint(tie(2)), 2);
    // 1.5 → both 2.
    assert_eq!(f32_to_s16(tie(1)), 2);
    assert_eq!(f32_to_s16_lrint(tie(1)), 2);
    // −2.5 → NEON −2 (toward +∞), lrintf −2 (even): equal here.
    assert_eq!(f32_to_s16(tie(-3)), -2);
    assert_eq!(f32_to_s16_lrint(tie(-3)), -2);
    // −3.5 → NEON −3, lrintf −4 (even).
    assert_eq!(f32_to_s16(tie(-4)), -3);
    assert_eq!(f32_to_s16_lrint(tie(-4)), -4);
    // Off the tie both agree.
    for k in [-5, -1, 0, 7] {
        let x = (k as f32 + 0.25) / 32768.0;
        assert_eq!(f32_to_s16(x), f32_to_s16_lrint(x));
    }
}

#[test]
fn f64_quantizer_rounds_half_even() {
    assert_eq!(f64_to_s16(2.5 / 32768.0), 2);
    assert_eq!(f64_to_s16(1.5 / 32768.0), 2);
    assert_eq!(f64_to_s16(1.0), 32767);
    assert_eq!(f64_to_s16(-1.0), -32768);
}
