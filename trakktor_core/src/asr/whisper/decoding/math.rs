//! Numeric primitives of the decoding policy, over plain f32 rows.
//!
//! Sums accumulate in f64 for stability; inputs may contain `-inf` from
//! suppression masks, and an all-`-inf` row propagates as NaN, matching the
//! reference (comparisons against NaN are false, so downstream rules simply
//! do not trigger).

/// Numerically stable log-softmax.
pub(crate) fn log_softmax(row: &[f32]) -> Vec<f32> {
    let max = row.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    if !max.is_finite() {
        return vec![f32::NAN; row.len()];
    }
    let sum: f64 = row.iter().map(|&x| f64::from(x - max).exp()).sum();
    let log_sum = max as f64 + sum.ln();
    row.iter()
        .map(|&x| (f64::from(x) - log_sum) as f32)
        .collect()
}

/// Numerically stable softmax.
pub(crate) fn softmax(row: &[f32]) -> Vec<f32> {
    let max = row.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    if !max.is_finite() {
        return vec![f32::NAN; row.len()];
    }
    let exps: Vec<f64> =
        row.iter().map(|&x| f64::from(x - max).exp()).collect();
    let sum: f64 = exps.iter().sum();
    exps.iter().map(|&e| (e / sum) as f32).collect()
}

/// Numerically stable log-sum-exp.
pub(crate) fn logsumexp(xs: &[f32]) -> f32 {
    let max = xs.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    if !max.is_finite() {
        return max; // -inf for an empty or fully suppressed slice
    }
    let sum: f64 = xs.iter().map(|&x| f64::from(x - max).exp()).sum();
    (max as f64 + sum.ln()) as f32
}

/// Index of the first maximum.
pub(crate) fn argmax(xs: &[f32]) -> usize {
    let mut best = 0;
    let mut best_value = f32::NEG_INFINITY;
    for (i, &x) in xs.iter().enumerate() {
        if x > best_value {
            best_value = x;
            best = i;
        }
    }
    best
}

/// The `k` largest values with their indices, in descending value order;
/// ties resolve to the lower index.
pub(crate) fn topk(xs: &[f32], k: usize) -> Vec<(f32, usize)> {
    // Descending by value, ascending by index on ties.
    let better = |a: &(f32, usize), b: &(f32, usize)| {
        a.0 > b.0 || (a.0 == b.0 && a.1 < b.1)
    };
    let mut best: Vec<(f32, usize)> = Vec::with_capacity(k + 1);
    for (i, &x) in xs.iter().enumerate() {
        let candidate = (x, i);
        if best.len() == k &&
            !better(&candidate, best.last().expect("k > 0 candidates"))
        {
            continue;
        }
        let position = best
            .iter()
            .position(|kept| better(&candidate, kept))
            .unwrap_or(best.len());
        best.insert(position, candidate);
        best.truncate(k);
    }
    best
}

/// Samples an index from `softmax(logits)`, consuming one uniform draw.
pub(crate) fn sample_categorical(
    logits: &[f32],
    rng: &mut impl rand::Rng,
) -> usize {
    let probs = softmax(logits);
    let draw: f32 = rng.random();
    let mut cumulative = 0.0f32;
    for (i, &p) in probs.iter().enumerate() {
        if p.is_nan() {
            break;
        }
        cumulative += p;
        if draw < cumulative {
            return i;
        }
    }
    // Rounding left the draw above the accumulated mass: pick the last
    // index with non-zero probability.
    probs
        .iter()
        .rposition(|&p| p > 0.0)
        .unwrap_or(probs.len().saturating_sub(1))
}
