//! The flow model's time grid and stepping rule — the part of the solver that
//! is arithmetic on the host and therefore shared by the runtimes.
//!
//! The tensor arithmetic of a step is not here: `ψ + dt · v` lives wherever ψ
//! does. What is here is everything that decides **when** the velocity field is
//! asked, which is the part the two runtimes must not be allowed to disagree
//! about.
//!
//! # The time grid is not uniform, and that is the whole trick
//!
//! The generation runs from `t = 0` to `t = 1`, but the trajectory is not
//! travelled evenly: upstream warps the grid with
//!
//! ```text
//! h(t) = (aᵗ − 1) / (a − 1)
//! ```
//!
//! choosing `a` so that `h(1/n) = 1/2` — half the distance is covered in the
//! first `1/n` of the time, with `n` = [`TIME_MAPPING_DIVISOR`]. The reference
//! finds `a` numerically at every call with `scipy.optimize.fsolve`; it is a
//! constant, and it is found here by bisection, which needs no dependency and
//! is exact to the last bit of an `f64`.

use super::config::TIME_MAPPING_DIVISOR;

/// Which integrator the solver steps with.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum Method {
    /// One evaluation per step.
    Euler,
    /// Two evaluations per step — the reference's default.
    #[default]
    Midpoint,
    /// Four evaluations per step.
    Rk4,
}

impl Method {
    /// The name this method reports under.
    #[must_use]
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Euler => "euler",
            Self::Midpoint => "midpoint",
            Self::Rk4 => "rk4",
        }
    }

    /// Evaluations of the velocity field one step costs.
    #[must_use]
    pub fn evaluations(self) -> usize {
        match self {
            Self::Euler => 1,
            Self::Midpoint => 2,
            Self::Rk4 => 4,
        }
    }

    /// Steps a budget of `nfe` evaluations buys.
    ///
    /// The reference divides and truncates, so an odd budget under the midpoint
    /// rule simply buys one evaluation less — and a budget of one falls back to
    /// Euler rather than taking no step at all.
    #[must_use]
    pub fn steps(self, nfe: usize) -> usize { nfe / self.evaluations() }

    /// The method actually used for a budget of `nfe`, which is Euler when the
    /// budget cannot pay for a single step of anything else.
    #[must_use]
    pub fn resolve(self, nfe: usize) -> Self {
        if nfe == 1 && self != Self::Euler {
            Self::Euler
        } else {
            self
        }
    }
}

/// The warped times the solver visits, `steps + 1` of them, from 0 to 1.
#[must_use]
pub fn schedule(steps: usize) -> Vec<f64> {
    let base = time_base(TIME_MAPPING_DIVISOR);
    (0..=steps)
        .map(|index| {
            let even = index as f64 / steps as f64;
            warp(even, base)
        })
        .collect()
}

/// `h(t) = (aᵗ − 1) / (a − 1)`.
fn warp(t: f64, base: f64) -> f64 { (base.powf(t) - 1.0) / (base - 1.0) }

/// The `a` for which half the trajectory is covered by `t = 1/divisor`.
///
/// `h(1/n, ·)` falls monotonically from 1 as `a → 0` to `1/n` at `a = 1`, so
/// the root is unique in `(0, 1)` and a bisection finds it without a solver.
fn time_base(divisor: f64) -> f64 {
    let target = 1.0 / divisor;
    let value = |base: f64| warp(target, base) - 0.5;
    let (mut low, mut high) = (f64::MIN_POSITIVE, 1.0 - f64::EPSILON);
    for _ in 0..200 {
        let middle = 0.5 * (low + high);
        if value(middle) > 0.0 {
            low = middle;
        } else {
            high = middle;
        }
    }
    0.5 * (low + high)
}

#[cfg(test)]
mod tests;
