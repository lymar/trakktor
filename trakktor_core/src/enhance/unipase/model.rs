//! The seam between the driver and a runtime.
//!
//! The driver owns everything that is not a tensor operation — decoding the
//! file, cutting it into windows, deciding which frames are holes, stitching
//! the windows back together, matching the original loudness, writing the
//! result. A runtime owns exactly one operation, and it takes and returns
//! plain host values:
//!
//! - [`enhance_window`](EnhanceModel::enhance_window): 16 kHz samples in, 16
//!   kHz samples out.
//!
//! Keeping the seam this coarse is deliberate. Everything inside one window is
//! a single chain of matrix multiplications with no host decision in the
//! middle, so there is nothing to gain from a finer split and one synchronous
//! round trip per stage to lose.

use super::error::UnipaseError;

/// One loaded pipeline, on one runtime and one device.
pub trait EnhanceModel: Send {
    /// Enhances one window of 16 kHz mono audio.
    ///
    /// `lost` marks the frames the packet-loss detector found, one flag per 320
    /// samples; an empty slice disables concealment for this window. The result
    /// is `frames × 320` samples long, where `frames` is what the window aligns
    /// to — that is, as long as the aligned window, not necessarily as long as
    /// what was passed in.
    ///
    /// # Errors
    ///
    /// Returns [`UnipaseError::Compute`] when the tensor backend fails.
    fn enhance_window(
        &mut self,
        samples: &[f32],
        lost: &[bool],
    ) -> Result<Vec<f32>, UnipaseError>;

    /// Which runtime this is, for the run's report.
    fn runtime(&self) -> Runtime;
}

/// The tensor backend a run computes on.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum Runtime {
    /// The candle runtime.
    #[default]
    Candle,
    /// The burn runtime.
    Burn,
}

impl Runtime {
    /// The name this runtime reports under.
    #[must_use]
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Candle => "candle",
            Self::Burn => "burn",
        }
    }
}

/// Compute precision of a run.
///
/// The published checkpoints are `f32` throughout and the reference offers no
/// half-precision path, so [`F32`](Precision::F32) is both the default and the
/// only mode parity is claimed in. [`F16`](Precision::F16) is a candle-only
/// speed option, and is a different computation of the same model rather than
/// the same one faster.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum Precision {
    /// Full precision (`f32`), the default.
    #[default]
    F32,
    /// Half precision (`f16`).
    F16,
}

impl Precision {
    /// The name this precision reports under.
    #[must_use]
    pub fn as_str(self) -> &'static str {
        match self {
            Self::F32 => "f32",
            Self::F16 => "f16",
        }
    }
}
