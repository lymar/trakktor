//! Turning logits into a code.
//!
//! Two levels sample independently — the talker for codebook 0 and the code
//! predictor for the residual codebooks — but both go through the same steps,
//! in the order the reference applies them: penalize what has already been
//! said, forbid what must not be said, scale by temperature, keep the most
//! likely handful, and draw.
//!
//! Greedy decoding skips the drawing and takes the maximum. It is the mode the
//! runtimes are compared in, because it is the only one whose output does not
//! depend on a random stream.

use rand::{Rng, SeedableRng, rngs::StdRng};

/// The picking rule for one level.
#[derive(Debug, Clone, Copy)]
pub enum Rule {
    /// Take the most likely code.
    Greedy,
    /// Draw from the `top_k` most likely codes after temperature scaling.
    TopK { top_k: usize, temperature: f32 },
}

/// Draws codes from logits, carrying the random stream across a whole run so a
/// seed reproduces it.
pub struct Sampler {
    rng: StdRng,
}

impl Sampler {
    /// Creates a sampler whose stream is fixed by `seed`.
    #[must_use]
    pub fn new(seed: u64) -> Self {
        Self {
            rng: StdRng::seed_from_u64(seed),
        }
    }

    /// Picks one code from `logits`, which this call may modify in place.
    ///
    /// `penalize` lists codes already produced, discouraged by
    /// `repetition_penalty`; `forbid` lists codes that must never be picked.
    pub fn pick(
        &mut self,
        logits: &mut [f32],
        rule: Rule,
        penalize: &[u32],
        repetition_penalty: f32,
        forbid: &[u32],
    ) -> u32 {
        // A code already produced is pushed toward zero from whichever side it
        // sits on, exactly as the reference does it.
        if repetition_penalty != 1.0 {
            for &code in penalize {
                if let Some(score) = logits.get_mut(code as usize) {
                    *score = if *score > 0.0 {
                        *score / repetition_penalty
                    } else {
                        *score * repetition_penalty
                    };
                }
            }
        }
        for &code in forbid {
            if let Some(score) = logits.get_mut(code as usize) {
                *score = f32::NEG_INFINITY;
            }
        }

        let (top_k, temperature) = match rule {
            Rule::Greedy => return argmax(logits),
            Rule::TopK { top_k, temperature } => (top_k, temperature),
        };

        if temperature > 0.0 && temperature != 1.0 {
            for score in logits.iter_mut() {
                *score /= temperature;
            }
        }

        // Keep only the most likely candidates; everything else is out of the
        // running before the distribution is formed.
        let kept = keep_top_k(logits, top_k);
        let probabilities = softmax(&kept);
        self.draw(&kept, &probabilities)
    }

    /// Draws an index from `probabilities` by walking the cumulative
    /// distribution.
    fn draw(
        &mut self,
        candidates: &[(u32, f32)],
        probabilities: &[f32],
    ) -> u32 {
        let threshold: f32 = self.rng.random_range(0.0..1.0);
        let mut cumulative = 0.0f32;
        for (candidate, probability) in candidates.iter().zip(probabilities) {
            cumulative += probability;
            if threshold < cumulative {
                return candidate.0;
            }
        }
        // Rounding can leave the threshold just past the end; the last
        // candidate is the right answer there.
        candidates.last().map_or(0, |&(code, _)| code)
    }
}

/// The index of the largest score, ties going to the lowest index.
fn argmax(logits: &[f32]) -> u32 {
    let mut best = 0usize;
    let mut best_score = f32::NEG_INFINITY;
    for (index, &score) in logits.iter().enumerate() {
        if score > best_score {
            best_score = score;
            best = index;
        }
    }
    best as u32
}

/// Selects the `top_k` highest-scoring codes, largest first.
fn keep_top_k(logits: &[f32], top_k: usize) -> Vec<(u32, f32)> {
    let mut scored: Vec<(u32, f32)> = logits
        .iter()
        .enumerate()
        .filter(|(_, score)| score.is_finite())
        .map(|(index, &score)| (index as u32, score))
        .collect();
    // Descending by score; a stable tie-break on the code keeps runs
    // reproducible.
    scored.sort_by(|a, b| {
        b.1.partial_cmp(&a.1)
            .unwrap_or(std::cmp::Ordering::Equal)
            .then(a.0.cmp(&b.0))
    });
    let keep = top_k.clamp(1, scored.len().max(1));
    scored.truncate(keep);
    scored
}

/// Normalizes candidate scores into probabilities.
fn softmax(candidates: &[(u32, f32)]) -> Vec<f32> {
    let max = candidates
        .iter()
        .map(|&(_, score)| score)
        .fold(f32::NEG_INFINITY, f32::max);
    let exponentiated: Vec<f32> = candidates
        .iter()
        .map(|&(_, score)| (score - max).exp())
        .collect();
    let total: f32 = exponentiated.iter().sum();
    if total <= 0.0 {
        return vec![1.0 / candidates.len().max(1) as f32; candidates.len()];
    }
    exponentiated.iter().map(|value| value / total).collect()
}

#[cfg(test)]
mod tests;
