//! The unwarper: a port of `UVDoc`.
//!
//! What it produces is not a picture but a **backward map**: a two-channel
//! field the size of the page, where the value at an output pixel says which
//! point of the photograph that pixel comes from. Everything follows from
//! that. The straightened page is one bilinear gather through the field; the
//! way back — from a box found on the straightened page to where it sits on
//! the photograph — is a lookup in the same field, with no inverse to solve.
//!
//! The network itself is small and old-fashioned: a five-by-five convolutional
//! stem, three stages of dilated residual blocks, and then six parallel
//! branches whose dilations run from one to eighteen. That fan is the whole
//! idea — on a grid of 45x31 a dilation of eighteen spans the page, so one
//! layer can relate a corner to the opposite margin, which is what deciding a
//! page's shape needs.
//!
//! Four details are worth stating because each is silent when wrong:
//!
//! * **The field is computed at a fixed 712x488** whatever the page is. The
//!   graph has no dynamic input; the page is scaled into that box, and the
//!   field that comes out is scaled back up to the page.
//! * **A residual block numbers its shortcut first.** In a block that halves
//!   the map, the first convolution of the three is the shortcut and not the
//!   first of the pair — the dataflow says so plainly, and the module names in
//!   the exported graph suggest the opposite.
//! * **The bridge convolutions are three-by-three**, not five-by-five like
//!   everything before them, and their padding equals their dilation rather
//!   than twice it.
//! * **The head pads by reflection**, not with zeros. A page whose content runs
//!   to the margin would otherwise get a fold of black reflected into the
//!   field, and the straightened page would bend at the edges.
//!
//! The published weights also carry a second head — a three-channel one that
//! predicts positions in space rather than on the page. The exported graph
//! does not use it and neither does this port.

#[cfg(test)]
mod tests;

use candle_core::{Device, Tensor};

use crate::ocr::{
    error::OcrError,
    paddle::{
        image::{CHANNELS, Page},
        net::{BatchNorm, Conv, Loader, relu},
    },
};

/// The box the page is scaled into before the field is computed.
pub const INPUT_HEIGHT: usize = 712;
pub const INPUT_WIDTH: usize = 488;

/// The field the network emits, before it is enlarged to the page.
pub const FIELD_HEIGHT: usize = 45;
pub const FIELD_WIDTH: usize = 31;

/// The kernel of everything but the bridge.
const KERNEL: usize = 5;
/// The kernel of the bridge.
const BRIDGE_KERNEL: usize = 3;

/// The stages of the trunk: `(in, out, dilation, downsample)` per block.
const STAGES: [&[(usize, usize, usize, bool)]; 3] = [
    &[(32, 32, 1, false), (32, 32, 3, false), (32, 32, 3, false)],
    &[
        (32, 64, 1, true),
        (64, 64, 3, false),
        (64, 64, 3, false),
        (64, 64, 3, false),
    ],
    &[
        (64, 128, 1, true),
        (128, 128, 3, false),
        (128, 128, 3, false),
        (128, 128, 3, false),
        (128, 128, 3, false),
        (128, 128, 3, false),
    ],
];

/// The six branches of the bridge, each a list of dilations.
const BRIDGE: [&[usize]; 6] =
    [&[1], &[2], &[5], &[8, 3, 2], &[12, 7, 4], &[18, 12, 6]];

/// The width of the trunk's output and of every bridge branch.
const WIDE: usize = 128;

/// A convolution with batch normalization, and an activation the caller
/// decides on.
#[derive(Debug)]
struct ConvBn {
    conv: Conv,
    norm: BatchNorm,
}

impl ConvBn {
    fn load(
        loader: &Loader,
        at: &mut usize,
        dims: [usize; 4],
        stride: usize,
        padding: usize,
        dilation: usize,
    ) -> Result<Self, OcrError> {
        let index = *at;
        *at += 1;
        Ok(Self {
            conv: Conv::load(
                loader,
                &format!("conv2d_{index}"),
                dims,
                stride,
                padding,
                1,
            )?
            .dilated(dilation),
            norm: BatchNorm::load(
                loader,
                &format!("batch_norm2d_{index}"),
                dims[0],
            )?,
        })
    }

    fn forward(&self, x: &Tensor) -> Result<Tensor, OcrError> {
        self.norm.forward(&self.conv.forward(x)?)
    }
}

/// A residual block: a shortcut, and a pair of convolutions across it.
#[derive(Debug)]
struct Residual {
    /// Present only where the block changes the shape it is asked to add to.
    shortcut: Option<ConvBn>,
    start: ConvBn,
    final_: ConvBn,
}

impl Residual {
    fn load(
        loader: &Loader,
        at: &mut usize,
        (inputs, outputs, dilation, downsample): (usize, usize, usize, bool),
    ) -> Result<Self, OcrError> {
        let padding = dilation * 2;
        let stride = if downsample { 2 } else { 1 };
        // The shortcut is numbered before the pair, so it is read first.
        let shortcut = if downsample {
            Some(ConvBn::load(
                loader,
                at,
                [outputs, inputs, KERNEL, KERNEL],
                stride,
                KERNEL / 2,
                1,
            )?)
        } else {
            None
        };
        Ok(Self {
            shortcut,
            start: ConvBn::load(
                loader,
                at,
                [outputs, inputs, KERNEL, KERNEL],
                stride,
                padding,
                dilation,
            )?,
            final_: ConvBn::load(
                loader,
                at,
                [outputs, outputs, KERNEL, KERNEL],
                1,
                padding,
                dilation,
            )?,
        })
    }

    fn forward(&self, x: &Tensor) -> Result<Tensor, OcrError> {
        let residual = match &self.shortcut {
            None => x.clone(),
            Some(shortcut) => shortcut.forward(x)?,
        };
        let y = relu(&self.start.forward(x)?)?;
        relu(&self.final_.forward(&y)?.add(&residual)?)
    }
}

/// The unwarper.
#[derive(Debug)]
pub struct Unwarper {
    device: Device,
    stem: Vec<ConvBn>,
    trunk: Vec<Residual>,
    bridge: Vec<Vec<ConvBn>>,
    connector: ConvBn,
    reduce: ConvBn,
    prelu: Tensor,
    project: Conv,
}

impl Unwarper {
    pub fn load(loader: &Loader) -> Result<Self, OcrError> {
        let mut at = 0usize;
        let stem = vec![
            ConvBn::load(loader, &mut at, [32, 3, KERNEL, KERNEL], 2, 2, 1)?,
            ConvBn::load(loader, &mut at, [32, 32, KERNEL, KERNEL], 2, 2, 1)?,
        ];

        let mut trunk = Vec::new();
        for stage in STAGES {
            for block in stage.iter() {
                trunk.push(Residual::load(loader, &mut at, *block)?);
            }
        }

        let mut bridge = Vec::with_capacity(BRIDGE.len());
        for branch in BRIDGE {
            let mut blocks = Vec::with_capacity(branch.len());
            for dilation in branch.iter() {
                blocks.push(ConvBn::load(
                    loader,
                    &mut at,
                    [WIDE, WIDE, BRIDGE_KERNEL, BRIDGE_KERNEL],
                    1,
                    *dilation,
                    *dilation,
                )?);
            }
            bridge.push(blocks);
        }

        let connector = ConvBn::load(
            loader,
            &mut at,
            [WIDE, WIDE * BRIDGE.len(), 1, 1],
            1,
            0,
            1,
        )?;
        // The head's two convolutions pad by reflection, which the port does
        // itself, so both are loaded with no padding of their own.
        let reduce =
            ConvBn::load(loader, &mut at, [32, WIDE, KERNEL, KERNEL], 1, 0, 1)?;
        let prelu = loader.get("p_re_lu_0.w_0", &[1])?;
        let project = Conv::load(
            loader,
            &format!("conv2d_{at}"),
            [2, 32, KERNEL, KERNEL],
            1,
            0,
            1,
        )?;

        Ok(Self {
            device: loader.device().clone(),
            stem,
            trunk,
            bridge,
            connector,
            reduce,
            prelu,
            project,
        })
    }

    /// The device the weights live on.
    pub fn device(&self) -> &Device { &self.device }

    /// The backward map for one page, as `2 * 45 * 31` values: all of the
    /// horizontal channel, then all of the vertical one.
    ///
    /// The values are normalized to `-1..=1` over the photograph — the
    /// convention `grid_sample` reads them in — and they are *not* clamped:
    /// a page shot with the table in frame produces a field that points
    /// outside itself, and that is how the sheet gets cut out of the frame.
    pub fn field(&self, page: &Page) -> Result<Vec<f32>, OcrError> {
        let input = self.input_tensor(page)?;
        let field = self.forward(&input)?;
        Ok(field.flatten_all()?.to_vec1::<f32>()?)
    }

    fn forward(&self, x: &Tensor) -> Result<Tensor, OcrError> {
        let mut y = x.clone();
        for layer in &self.stem {
            y = relu(&layer.forward(&y)?)?;
        }
        for block in &self.trunk {
            y = block.forward(&y)?;
        }

        let mut branches = Vec::with_capacity(self.bridge.len());
        for branch in &self.bridge {
            let mut z = y.clone();
            for block in branch {
                z = relu(&block.forward(&z)?)?;
            }
            branches.push(z);
        }
        let fused = Tensor::cat(&branches, 1)?;

        let z = relu(&self.connector.forward(&fused)?)?;
        let z = reflect_pad(&z, KERNEL / 2)?;
        let z = prelu(&self.reduce.forward(&z)?, &self.prelu)?;
        let z = reflect_pad(&z, KERNEL / 2)?;
        Ok(self.project.forward(&z)?)
    }

    /// Scales the page into the fixed input box and normalizes it.
    ///
    /// The scaling is bilinear with the corners pinned — Paddle's
    /// `align_corners=True` — which is a different mapping from the
    /// half-pixel one everything else in the domain resizes with, and using
    /// the wrong one shifts the whole field by half a cell of the input.
    /// Values are the bytes over 255, in the page's own channel order.
    fn input_tensor(&self, page: &Page) -> Result<Tensor, OcrError> {
        let (width, height) = (page.width as usize, page.height as usize);
        if width < 2 || height < 2 {
            return Err(OcrError::Runtime(
                "the unwarper was handed a page with no area".into(),
            ));
        }
        let mut data = vec![0f32; CHANNELS * INPUT_HEIGHT * INPUT_WIDTH];
        let row = width * CHANNELS;
        let scale_y = (height - 1) as f32 / (INPUT_HEIGHT - 1) as f32;
        let scale_x = (width - 1) as f32 / (INPUT_WIDTH - 1) as f32;
        for oy in 0..INPUT_HEIGHT {
            let sy = oy as f32 * scale_y;
            let y0 = sy.floor() as usize;
            let y1 = (y0 + 1).min(height - 1);
            let fy = sy - y0 as f32;
            for ox in 0..INPUT_WIDTH {
                let sx = ox as f32 * scale_x;
                let x0 = sx.floor() as usize;
                let x1 = (x0 + 1).min(width - 1);
                let fx = sx - x0 as f32;
                for c in 0..CHANNELS {
                    let at = |y: usize, x: usize| {
                        f32::from(page.bgr[y * row + x * CHANNELS + c])
                    };
                    let top = at(y0, x0) * (1.0 - fx) + at(y0, x1) * fx;
                    let bottom = at(y1, x0) * (1.0 - fx) + at(y1, x1) * fx;
                    data[(c * INPUT_HEIGHT + oy) * INPUT_WIDTH + ox] =
                        (top * (1.0 - fy) + bottom * fy) / 255.0;
                }
            }
        }
        Ok(Tensor::from_vec(
            data,
            (1, CHANNELS, INPUT_HEIGHT, INPUT_WIDTH),
            &self.device,
        )?)
    }
}

/// Mirrors `pad` rows and columns outward at every edge, without repeating the
/// edge itself.
fn reflect_pad(x: &Tensor, pad: usize) -> Result<Tensor, OcrError> {
    if pad == 0 {
        return Ok(x.clone());
    }
    let mut y = if x.is_contiguous() {
        x.clone()
    } else {
        x.contiguous()?
    };
    for axis in [2usize, 3usize] {
        let size = y.dim(axis)?;
        if size <= pad {
            return Err(OcrError::Runtime(format!(
                "cannot reflect {pad} across an axis only {size} wide"
            )));
        }
        let mut index: Vec<u32> = (1..=pad).rev().map(|i| i as u32).collect();
        index.extend(0..size as u32);
        index.extend((2..=pad + 1).map(|i| (size - i) as u32));
        let count = index.len();
        let index = Tensor::from_vec(index, count, y.device())?;
        y = y.index_select(&index, axis)?.contiguous()?;
    }
    Ok(y)
}

/// The published activation of the head's first convolution: one slope for
/// every channel, so it is a scalar multiply on the negative half.
fn prelu(x: &Tensor, weight: &Tensor) -> Result<Tensor, OcrError> {
    let slope = weight.flatten_all()?.to_vec1::<f32>()?[0] as f64;
    let positive = x.relu()?;
    let negative = x.neg()?.relu()?;
    Ok(positive.sub(&negative.affine(slope, 0.0)?)?)
}
