//! The Gaussian noise the enhancer needs, and the reason it needs its own.
//!
//! The generative half of this project draws normal noise **twice per chunk** —
//! once to mix into the flow model's starting point (how much is
//! `--temperature`), and once as the waveform generator's input, which is a
//! noise-driven vocoder and has nothing else to start from. Upstream draws both
//! from torch's global generator, which means two things at once:
//!
//! - **the reference is not reproducible** unless its seed is set, and its
//!   command line never sets one. Two runs of upstream on the same file give
//!   two different files;
//! - **no port can match it.** Reproducing torch's Philox stream would be
//!   reproducing an implementation detail of another project, and it would
//!   still not survive that project changing it.
//!
//! So this port draws its own, from a stream that is fixed by the seed and by
//! nothing else — same seed, same machine or another, same samples. The
//! reference's own numbers can be fed in instead, which is how the parity tests
//! measure the network rather than the generator.
//!
//! The generator is SplitMix64 with a Box–Muller transform on top: a few lines,
//! no dependency, and the only property that matters here is that it is the
//! same everywhere.

/// Where the enhancer's Gaussian draws come from.
#[derive(Debug, Clone)]
pub enum Source {
    /// A stream fixed by a seed, which is what a run uses.
    Stream(Noise),
    /// Draws recorded elsewhere, handed out in order. This is how parity with
    /// the reference is established at all: fed the reference's own numbers,
    /// the two pipelines can be compared sample by sample, and what is measured
    /// is the network rather than the generator.
    Scripted(Vec<Vec<f32>>),
}

impl Source {
    /// A stream fixed by `seed`.
    #[must_use]
    pub fn seeded(seed: u64) -> Self { Self::Stream(Noise::new(seed)) }

    /// The next `count` values.
    ///
    /// A scripted source hands out its blocks in the order they were recorded,
    /// trimmed or zero-filled to the length asked for, and zeros once it runs
    /// out — so a mismatch shows up as a wrong answer rather than a panic in
    /// the middle of a run.
    #[must_use]
    pub fn draw(&mut self, count: usize) -> Vec<f32> {
        match self {
            Self::Stream(noise) => noise.sample(count),
            Self::Scripted(blocks) => {
                let mut block = if blocks.is_empty() {
                    Vec::new()
                } else {
                    blocks.remove(0)
                };
                block.resize(count, 0.0);
                block
            },
        }
    }
}

/// A reproducible source of standard normal values.
#[derive(Debug, Clone)]
pub struct Noise {
    state: u64,
    /// The second value of the last Box–Muller pair, kept for the next call.
    spare: Option<f32>,
}

impl Noise {
    /// A stream fixed by `seed`.
    #[must_use]
    pub fn new(seed: u64) -> Self {
        Self {
            // An odd, well-mixed starting state: SplitMix64 is weak from a
            // small one, and zero is the worst of them.
            state: seed.wrapping_mul(0x9E37_79B9_7F4A_7C15) ^
                0xDA3E_39CB_94B9_5BDB,
            spare: None,
        }
    }

    /// `count` standard normal values.
    #[must_use]
    pub fn sample(&mut self, count: usize) -> Vec<f32> {
        (0..count).map(|_| self.next_normal()).collect()
    }

    /// One standard normal value.
    fn next_normal(&mut self) -> f32 {
        if let Some(spare) = self.spare.take() {
            return spare;
        }
        // Box–Muller: a radius from one uniform and an angle from another. The
        // first is taken in (0, 1] so that its logarithm is finite.
        let u1 = self.next_open_unit();
        let u2 = self.next_unit();
        let radius = (-2.0 * u1.ln()).sqrt();
        let angle = std::f64::consts::TAU * u2;
        self.spare = Some((radius * angle.sin()) as f32);
        (radius * angle.cos()) as f32
    }

    /// A uniform in `[0, 1)`.
    fn next_unit(&mut self) -> f64 {
        // The top 53 bits are the ones with full entropy in this generator.
        (self.next_u64() >> 11) as f64 / (1u64 << 53) as f64
    }

    /// A uniform in `(0, 1]`.
    fn next_open_unit(&mut self) -> f64 { 1.0 - self.next_unit() }

    /// The next state of SplitMix64.
    fn next_u64(&mut self) -> u64 {
        self.state = self.state.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.state;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }
}

#[cfg(test)]
mod tests;
