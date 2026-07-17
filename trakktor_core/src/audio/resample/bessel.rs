//! Modified Bessel function of the first kind, order zero.
//!
//! Ports `av_bessel_i0` from ffmpeg's libavutil (`mathematics.c`), which the
//! filter-bank construction uses for the Kaiser window. The minimax rational
//! approximations follow Blair and Edwards, Chalk River Report AECL-4928,
//! 1974 (via the Boost project, as in the original). The polynomial branch
//! (|x| ≤ 15, always taken for the default Kaiser β = 9) is pure arithmetic
//! and therefore reproducible bit for bit on any platform.

const P1: [f64; 15] = [
    -2.2335582639474375249e+15,
    -5.5050369673018427753e+14,
    -3.2940087627407749166e+13,
    -8.4925101247114157499e+11,
    -1.1912746104985237192e+10,
    -1.0313066708737980747e+08,
    -5.9545626019847898221e+05,
    -2.4125195876041896775e+03,
    -7.0935347449210549190e+00,
    -1.5453977791786851041e-02,
    -2.5172644670688975051e-05,
    -3.0517226450451067446e-08,
    -2.6843448573468483278e-11,
    -1.5982226675653184646e-14,
    -5.2487866627945699800e-18,
];

const Q1: [f64; 6] = [
    -2.2335582639474375245e+15,
    7.8858692566751002988e+12,
    -1.2207067397808979846e+10,
    1.0377081058062166144e+07,
    -4.8527560179962773045e+03,
    1.0,
];

const P2: [f64; 7] = [
    -2.2210262233306573296e-04,
    1.3067392038106924055e-02,
    -4.4700805721174453923e-01,
    5.5674518371240761397e+00,
    -2.3517945679239481621e+01,
    3.1611322818701131207e+01,
    -9.6090021968656180000e+00,
];

const Q2: [f64; 8] = [
    -5.5194330231005480228e-04,
    3.2547697594819615062e-02,
    -1.1151759188741312645e+00,
    1.3982595353892851542e+01,
    -6.0228002066743340583e+01,
    8.5539563258012929600e+01,
    -3.1446690275135491500e+01,
    1.0,
];

/// Horner evaluation with separate multiply and add steps, exactly as the
/// original `eval_poly` (the two statements must not be fused into an fma).
fn eval_poly(coeff: &[f64], x: f64) -> f64 {
    let mut sum = coeff[coeff.len() - 1];
    for &c in coeff[..coeff.len() - 1].iter().rev() {
        sum *= x;
        sum += c;
    }
    sum
}

/// `I₀(x)`, bit-compatible with the reference `av_bessel_i0`.
pub fn bessel_i0(x: f64) -> f64 {
    if x == 0.0 {
        return 1.0;
    }
    let x = x.abs();
    if x <= 15.0 {
        let y = x * x;
        eval_poly(&P1, y) / eval_poly(&Q1, y)
    } else {
        // Unreachable with the default Kaiser β = 9 (arguments stay ≤ β);
        // kept for completeness. `exp` comes from the platform libm here.
        let y = 1.0 / x - 1.0 / 15.0;
        let r = eval_poly(&P2, y) / eval_poly(&Q2, y);
        let factor = x.exp() / x.sqrt();
        factor * r
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn known_values() {
        assert_eq!(bessel_i0(0.0), 1.0);
        // I0(1) ≈ 1.2660658777520083
        assert!((bessel_i0(1.0) - 1.2660658777520083).abs() < 1e-15);
        // I0(9) ≈ 1093.588354511375 — the top of the default Kaiser range.
        assert!((bessel_i0(9.0) - 1093.588354511375).abs() < 1e-9);
        assert_eq!(bessel_i0(-2.0), bessel_i0(2.0));
    }
}
