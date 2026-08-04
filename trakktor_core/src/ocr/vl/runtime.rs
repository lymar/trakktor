//! The candle-backed runtime: everything on the tensor side of the seam.
//!
//! [`CandleModel`] owns the three networks — the [`vision`] tower with its
//! projector and the [`ernie`] decoder — and implements the driver's
//! [`VlModel`] contract: prime once per picture, then one step per token.
//! The generation state (the preallocated key/value cache and the rotary
//! rows, both sized for the whole answer up front) lives here, behind the
//! seam.

pub mod ernie;
pub mod vision;

use std::path::Path;

use candle_core::{DType, Device, Tensor};
use candle_nn::{VarBuilder, ops::softmax_last_dim};

use self::{
    ernie::{Cache, Decoder},
    vision::{Projector, Tower},
};
use super::{
    config::{ModelConfig, WEIGHTS_FILE},
    generate::VlModel,
    image::Prepared,
};
use crate::ocr::error::OcrError;

/// The networks running on candle.
pub struct CandleModel {
    tower: Tower,
    projector: Projector,
    decoder: Decoder,
    cfg: ModelConfig,
    device: Device,
    dtype: DType,
    session: Option<Session>,
}

/// One generation's state: the cache and the rotary rows, sized once for the
/// whole answer. How far the decode has advanced is the cache's fill.
struct Session {
    cache: Cache,
    cos: Tensor,
    sin: Tensor,
}

/// Loads the checkpoint's weights onto `device` at `dtype`.
///
/// # Errors
///
/// Returns [`OcrError`] when the weights are missing, malformed, or do not
/// match the declared geometry.
pub fn load(
    dir: &Path,
    cfg: ModelConfig,
    device: Device,
    dtype: DType,
) -> Result<CandleModel, OcrError> {
    let weights = dir.join(WEIGHTS_FILE);
    // SAFETY: the weights are memory-mapped and must not change while the
    // model is loaded; they are ours, under the model directory, and verified
    // against a digest when they were fetched.
    let vb = unsafe {
        VarBuilder::from_mmaped_safetensors(&[weights], dtype, &device)?
    };
    let tower = Tower::load(&cfg.vision, vb.pp("visual"))?;
    let projector =
        Projector::load(&cfg.vision, cfg.hidden_size, vb.pp("mlp_AR"))?;
    let decoder = Decoder::load(&cfg, vb)?;
    Ok(CandleModel {
        tower,
        projector,
        decoder,
        cfg,
        device,
        dtype,
        session: None,
    })
}

impl CandleModel {
    /// The probability row of the position the logits score. The softmax
    /// stays on the device; the row comes back in one transfer, and the wait
    /// is what a decode step's time is made of.
    fn probabilities(&self, logits: &Tensor) -> Result<Vec<f32>, OcrError> {
        Ok(
            softmax_last_dim(&logits.to_dtype(DType::F32)?)?
                .to_vec1::<f32>()?,
        )
    }
}

impl VlModel for CandleModel {
    fn prime(
        &mut self,
        picture: &Prepared,
        tokens: &[u32],
        image_at: usize,
        positions: &[[i64; 3]],
    ) -> Result<Vec<f32>, OcrError> {
        self.session = None;
        let places = picture.tokens(self.cfg.vision.spatial_merge_size);

        // 1. The picture, once: patches → tower → projector.
        let pixels = Tensor::from_vec(
            picture.pixels.clone(),
            (picture.patches(), picture.pixels.len() / picture.patches()),
            &self.device,
        )?
        .to_dtype(self.dtype)?;
        let features = self.tower.forward(&pixels, picture.grid)?;
        let vision = self.projector.forward(&features, picture.grid)?;

        // 2. The picture's tokens take the placeholders' places.
        let embedded = self.decoder.embed(tokens, &self.device)?;
        let mut parts = Vec::with_capacity(3);
        if image_at > 0 {
            parts.push(embedded.narrow(1, 0, image_at)?);
        }
        parts.push(vision.unsqueeze(0)?);
        let after = image_at + places;
        if after < tokens.len() {
            parts.push(embedded.narrow(1, after, tokens.len() - after)?);
        }
        let inputs = Tensor::cat(&parts, 1)?.contiguous()?;

        // 3. Prefill. The cache and the rotary rows are sized once, up front,
        // for every position the answer can reach.
        let prompt = tokens.len();
        let (cos, sin) =
            self.decoder.tables(positions, &self.device, self.dtype)?;
        let mut cache =
            Cache::new(&self.cfg, positions.len(), self.dtype, &self.device)?;
        let logits = self.decoder.forward(
            &inputs,
            &cos.narrow(0, 0, prompt)?,
            &sin.narrow(0, 0, prompt)?,
            &mut cache,
        )?;

        let row = self.probabilities(&logits)?;
        self.session = Some(Session { cache, cos, sin });
        Ok(row)
    }

    fn step(&mut self, token: u32) -> Result<Vec<f32>, OcrError> {
        let Some(session) = self.session.as_mut() else {
            return Err(OcrError::Runtime(
                "a decode step before the generation was primed".into(),
            ));
        };
        let embedded = self.decoder.embed(&[token], &self.device)?;
        // The next row of the precomputed tables is exactly the cache's fill.
        let row = session.cache.len();
        let logits = self.decoder.forward(
            &embedded,
            &session.cos.narrow(0, row, 1)?,
            &session.sin.narrow(0, row, 1)?,
            &mut session.cache,
        )?;
        self.probabilities(&logits)
    }
}
