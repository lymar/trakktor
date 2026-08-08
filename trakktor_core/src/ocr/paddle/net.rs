//! The handful of layers PP-OCR networks are built from, on candle.
//!
//! The exported graphs are deliberately plain — convolutions, batch
//! normalization, two activations, a squeeze-and-excitation gate and a pair of
//! learned scalars — so this module is small and shared by the detector, the
//! recognizer and the orientation classifier.
//!
//! Two details are worth stating once, because getting either wrong produces a
//! result that looks plausible:
//!
//! - **Batch-norm sub-names.** A published `batch_norm*` parameter set is `w_0`
//!   = scale, `b_0` = bias, `w_1` = running mean, `w_2` = running variance. The
//!   mean and the variance are *not* `w_0`/`w_1`.
//! - **Two hard-sigmoid slopes.** The backbone's squeeze-and-excitation uses
//!   Paddle's default slope, which is the float nearest `0.1666667` — not the
//!   float nearest one sixth. The neck's uses `0.2`. They are different gates
//!   in different places and the graph carries both constants explicitly.

use candle_core::{D, DType, Device, Tensor};

use crate::ocr::{
    error::OcrError,
    paddle::artifact::{Artifact, RawTensor},
};

/// Reads weights out of a published artifact onto a device.
pub struct Loader<'a> {
    artifact: &'a Artifact,
    device: &'a Device,
    /// Published names grouped by the name they are copies of, each group in
    /// declaration order. See [`Loader::copy`].
    copies: std::collections::HashMap<String, Vec<String>>,
}

impl<'a> Loader<'a> {
    pub fn new(artifact: &'a Artifact, device: &'a Device) -> Self {
        let mut copies: std::collections::HashMap<String, Vec<String>> =
            std::collections::HashMap::new();
        for name in artifact.names() {
            let base = base_name(name);
            if base != name {
                copies
                    .entry(base.to_string())
                    .or_default()
                    .push(name.to_string());
            }
        }
        for group in copies.values_mut() {
            group.sort_by_key(|name| copy_indices(name));
        }
        Self {
            artifact,
            device,
            copies,
        }
    }

    /// One copy of a weight that a repeated module owns.
    ///
    /// A module the exporter duplicated — the six decoder layers of the layout
    /// detector, say — declares the *same* parameter name six times and tells
    /// the copies apart only by a `_deepcopy_<n>` suffix chain the exporter
    /// appends. The suffixes rise with declaration order, and declaration order
    /// is layer order, so a group sorted by them is indexable by layer. That is
    /// the whole of the mapping; nothing else in the artifact records it.
    pub fn copy(
        &self,
        base: &str,
        copy: usize,
        dims: &[usize],
    ) -> Result<Tensor, OcrError> {
        let group = self.copies.get(base).ok_or_else(|| {
            OcrError::Artifact(format!("no weight is a copy of `{base}`"))
        })?;
        let name = group.get(copy).ok_or_else(|| {
            OcrError::Artifact(format!(
                "`{base}` has {} copies, not {}",
                group.len(),
                copy + 1
            ))
        })?;
        let raw = self.artifact.shaped(name, dims)?;
        Ok(Tensor::from_slice(&raw.data, dims, self.device)?)
    }

    /// How many copies of a weight the artifact carries.
    pub fn copy_count(&self, base: &str) -> usize {
        self.copies.get(base).map_or(0, Vec::len)
    }

    pub fn artifact(&self) -> &Artifact { self.artifact }

    pub fn device(&self) -> &Device { self.device }

    /// The name the file stores a weight under.
    ///
    /// Most exports name a parameter after the operator that owns it
    /// (`conv2d_0.w_0`); some — the text-line orientation classifiers among
    /// them — append the parameter's declaration index to every name
    /// (`conv2d_0.w_0_deepcopy_0`). That index says nothing a reader needs and
    /// nothing else distinguishes the two spellings, so a name is looked up
    /// plain first and then against each index the file could have handed out.
    /// A name that matches neither comes back unchanged, so a genuinely
    /// missing weight is reported as the caller asked for it.
    fn resolve(&self, name: &str) -> String {
        if self.artifact.has(name) {
            return name.to_string();
        }
        // A name the exporter suffixed. When it did so once there is nothing to
        // choose between; when it did so several times the caller wants
        // [`Loader::copy`] and asking for the bare name is a mistake, so the
        // ambiguous case is left to fail as a missing weight.
        match self.copies.get(name).map(Vec::as_slice) {
            Some([only]) => only.clone(),
            _ => name.to_string(),
        }
    }

    /// Whether the artifact carries this weight, under either spelling.
    pub fn has(&self, name: &str) -> bool {
        self.artifact.has(&self.resolve(name))
    }

    /// Reads a weight of a known shape without moving it onto the device.
    pub fn raw(
        &self,
        name: &str,
        dims: &[usize],
    ) -> Result<&RawTensor, OcrError> {
        self.artifact.shaped(&self.resolve(name), dims)
    }

    /// The shape a weight has in the file, for the few layers whose width is
    /// only known from the model.
    pub fn dims(&self, name: &str) -> Result<&[usize], OcrError> {
        Ok(self.artifact.tensor(&self.resolve(name))?.dims.as_slice())
    }

    /// Loads a weight of a known shape.
    pub fn get(&self, name: &str, dims: &[usize]) -> Result<Tensor, OcrError> {
        let raw = self.raw(name, dims)?;
        Ok(Tensor::from_slice(&raw.data, dims, self.device)?)
    }

    /// Loads a weight whose shape is only known from the file.
    pub fn get_any(&self, name: &str) -> Result<Tensor, OcrError> {
        let raw = self.artifact.tensor(&self.resolve(name))?;
        Ok(Tensor::from_slice(
            &raw.data,
            raw.dims.as_slice(),
            self.device,
        )?)
    }

    /// Loads a single learned scalar (`[1]`).
    pub fn scalar(&self, name: &str) -> Result<f32, OcrError> {
        Ok(self.raw(name, &[1])?.data[0])
    }
}

/// A published name with every `_deepcopy_<n>` the exporter appended stripped
/// off.
fn base_name(name: &str) -> &str {
    let mut base = name;
    while let Some(cut) = base.rfind("_deepcopy_") {
        let (head, tail) = base.split_at(cut);
        if tail["_deepcopy_".len()..]
            .chars()
            .all(|c| c.is_ascii_digit()) &&
            tail.len() > "_deepcopy_".len()
        {
            base = head;
        } else {
            break;
        }
    }
    base
}

/// The suffix numbers of a copied name, outermost last — the key its group is
/// ordered by.
fn copy_indices(name: &str) -> Vec<u64> {
    let mut indices = Vec::new();
    let mut rest = name;
    while let Some(cut) = rest.rfind("_deepcopy_") {
        let (head, tail) = rest.split_at(cut);
        let digits = &tail["_deepcopy_".len()..];
        match digits.parse::<u64>() {
            Ok(index) => {
                indices.push(index);
                rest = head;
            },
            Err(_) => break,
        }
    }
    indices.reverse();
    indices
}

/// A convolution with an optional bias, as the graph applies it: the bias is a
/// separate per-channel add rather than a conv argument.
#[derive(Debug)]
pub struct Conv {
    weight: Tensor,
    bias: Option<Tensor>,
    stride: usize,
    padding: usize,
    groups: usize,
}

impl Conv {
    /// Loads `<name>.w_0` and, when the layer has one, `<name>.b_0`.
    pub fn load(
        loader: &Loader,
        name: &str,
        dims: [usize; 4],
        stride: usize,
        padding: usize,
        groups: usize,
    ) -> Result<Self, OcrError> {
        let weight = loader.get(&format!("{name}.w_0"), &dims)?;
        let bias_name = format!("{name}.b_0");
        let bias = if loader.has(&bias_name) {
            Some(loader.get(&bias_name, &[dims[0]])?)
        } else {
            None
        };
        Ok(Self {
            weight,
            bias,
            stride,
            padding,
            groups,
        })
    }

    pub fn forward(&self, x: &Tensor) -> Result<Tensor, OcrError> {
        let y =
            x.conv2d(&self.weight, self.padding, self.stride, 1, self.groups)?;
        Ok(add_channel_bias(&y, self.bias.as_ref())?)
    }
}

/// A transposed convolution, used only by the detection head to bring the
/// probability map back to the input resolution. The published weight is
/// `[in, out, kh, kw]`, the same layout candle expects.
#[derive(Debug)]
pub struct ConvTranspose {
    weight: Tensor,
    bias: Option<Tensor>,
    stride: usize,
}

impl ConvTranspose {
    pub fn load(
        loader: &Loader,
        name: &str,
        dims: [usize; 4],
        stride: usize,
    ) -> Result<Self, OcrError> {
        let weight = loader.get(&format!("{name}.w_0"), &dims)?;
        let bias_name = format!("{name}.b_0");
        let bias = if loader.has(&bias_name) {
            Some(loader.get(&bias_name, &[dims[1]])?)
        } else {
            None
        };
        Ok(Self {
            weight,
            bias,
            stride,
        })
    }

    pub fn forward(&self, x: &Tensor) -> Result<Tensor, OcrError> {
        let y = x.conv_transpose2d(&self.weight, 0, 0, self.stride, 1)?;
        Ok(add_channel_bias(&y, self.bias.as_ref())?)
    }
}

fn add_channel_bias(
    y: &Tensor,
    bias: Option<&Tensor>,
) -> Result<Tensor, candle_core::Error> {
    match bias {
        None => Ok(y.clone()),
        Some(bias) => {
            let channels = bias.dim(0)?;
            y.broadcast_add(&bias.reshape((1, channels, 1, 1))?)
        },
    }
}

/// Inference-time batch normalization, kept as the per-channel scale and shift
/// it collapses to.
#[derive(Debug)]
pub struct BatchNorm {
    scale: Tensor,
    shift: Tensor,
}

/// The epsilon every published graph carries (the float nearest `1e-5`).
const BATCH_NORM_EPS: f32 = 1e-5;

impl BatchNorm {
    pub fn load(
        loader: &Loader,
        name: &str,
        channels: usize,
    ) -> Result<Self, OcrError> {
        let gamma = loader.raw(&format!("{name}.w_0"), &[channels])?;
        let beta = loader.raw(&format!("{name}.b_0"), &[channels])?;
        let mean = loader.raw(&format!("{name}.w_1"), &[channels])?;
        let var = loader.raw(&format!("{name}.w_2"), &[channels])?;

        let mut scale = Vec::with_capacity(channels);
        let mut shift = Vec::with_capacity(channels);
        for c in 0..channels {
            let s = gamma.data[c] / (var.data[c] + BATCH_NORM_EPS).sqrt();
            scale.push(s);
            shift.push(beta.data[c] - mean.data[c] * s);
        }
        Ok(Self {
            scale: Tensor::from_vec(
                scale,
                (1, channels, 1, 1),
                loader.device(),
            )?,
            shift: Tensor::from_vec(
                shift,
                (1, channels, 1, 1),
                loader.device(),
            )?,
        })
    }

    pub fn forward(&self, x: &Tensor) -> Result<Tensor, OcrError> {
        Ok(x.broadcast_mul(&self.scale)?.broadcast_add(&self.shift)?)
    }
}

/// A convolution and the inference-time normalization behind it, addressed by
/// the number the export gave them rather than by name.
///
/// The graphs of the PP-HGNetV2 family number every convolution `conv2d_{n}` in
/// declaration order — which is also execution order — and its normalization
/// `batch_norm2d_{n + offset}`, with an offset that is a constant of the
/// artifact: zero in the server detector, eighty in the layout model. A port
/// therefore walks the structure with a counter instead of spelling out eighty
/// names.
#[derive(Debug)]
pub struct ConvBn {
    conv: Conv,
    /// Set when the layer strides its two axes by different amounts, which a
    /// convolution cannot; see [`ConvBn::load_axes`].
    keep: Option<(usize, usize)>,
    norm: BatchNorm,
}

impl ConvBn {
    /// A layer that strides both axes alike.
    pub fn load(
        loader: &Loader,
        index: usize,
        bn_offset: usize,
        dims: [usize; 4],
        stride: usize,
        padding: usize,
        groups: usize,
    ) -> Result<Self, OcrError> {
        Self::load_axes(
            loader,
            index,
            bn_offset,
            dims,
            (stride, stride),
            padding,
            groups,
        )
    }

    /// A layer whose two axes stride by different amounts — what the backbone
    /// does when it reads a text line rather than a page, spending the height
    /// and keeping the length.
    ///
    /// candle takes one stride for both axes, so an uneven pair convolves at
    /// stride one and drops the rows and columns a strided convolution would
    /// never have computed. The values that remain are the same ones, at the
    /// price of computing those that go.
    #[allow(clippy::too_many_arguments)]
    pub fn load_axes(
        loader: &Loader,
        index: usize,
        bn_offset: usize,
        dims: [usize; 4],
        stride: (usize, usize),
        padding: usize,
        groups: usize,
    ) -> Result<Self, OcrError> {
        let even = stride.0 == stride.1;
        Ok(Self {
            conv: Conv::load(
                loader,
                &format!("conv2d_{index}"),
                dims,
                if even { stride.0 } else { 1 },
                padding,
                groups,
            )?,
            keep: (!even).then_some(stride),
            norm: BatchNorm::load(
                loader,
                &format!("batch_norm2d_{}", index + bn_offset),
                dims[0],
            )?,
        })
    }

    pub fn forward(&self, x: &Tensor) -> Result<Tensor, OcrError> {
        let y = self.conv.forward(x)?;
        let y = match self.keep {
            None => y,
            Some(stride) => subsample(&y, stride)?,
        };
        self.norm.forward(&y)
    }
}

/// Pads the right and bottom edges with zeros.
///
/// This is what Paddle's `SAME` comes to for an even kernel at unit stride, and
/// candle pads symmetrically, so a layer that needs it does it by hand. The
/// activations feeding such a layer are non-negative, so zero is also the
/// identity for a max pool reading the same padded tensor.
pub fn pad_end(x: &Tensor, by: usize) -> Result<Tensor, OcrError> {
    Ok(x.pad_with_zeros(2, 0, by)?.pad_with_zeros(3, 0, by)?)
}

/// A fully connected layer.
///
/// The published weight is `[in, out]` — the transpose of what most frameworks
/// store — and the graph applies it as `x @ W + b`, so it is used exactly as it
/// comes off disk. Handing it to a layer that expects `[out, in]` fails
/// loudly on a rectangular weight and, worse, silently succeeds on a square
/// one.
#[derive(Debug, Clone)]
pub struct Linear {
    weight: Tensor,
    bias: Option<Tensor>,
}

impl Linear {
    /// Builds the layer from tensors a caller read itself — the way a module
    /// the exporter duplicated has to, since its weights are addressed by copy
    /// rather than by name.
    pub fn from_parts(weight: Tensor, bias: Option<Tensor>) -> Self {
        Self { weight, bias }
    }

    pub fn load(
        loader: &Loader,
        name: &str,
        in_features: usize,
        out_features: usize,
    ) -> Result<Self, OcrError> {
        let weight =
            loader.get(&format!("{name}.w_0"), &[in_features, out_features])?;
        let bias_name = format!("{name}.b_0");
        let bias = if loader.has(&bias_name) {
            Some(loader.get(&bias_name, &[out_features])?)
        } else {
            None
        };
        Ok(Self { weight, bias })
    }

    /// Applies the layer to the last axis of `x`, whatever comes before it.
    pub fn forward(&self, x: &Tensor) -> Result<Tensor, OcrError> {
        let y = x.broadcast_matmul(&self.weight)?;
        match &self.bias {
            None => Ok(y),
            Some(bias) => Ok(y.broadcast_add(bias)?),
        }
    }
}

/// Layer normalization over the last axis.
///
/// The variance is the biased one (divided by the width, not by the width less
/// one) and the epsilon sits inside the square root, which is what every
/// published graph does — but *which* epsilon differs between layers of the
/// same network, so it is a parameter rather than a constant here.
#[derive(Debug, Clone)]
pub struct LayerNorm {
    scale: Tensor,
    shift: Tensor,
    eps: f64,
}

impl LayerNorm {
    /// Builds the layer from tensors a caller read itself.
    pub fn from_parts(scale: Tensor, shift: Tensor, eps: f64) -> Self {
        Self { scale, shift, eps }
    }

    pub fn load(
        loader: &Loader,
        name: &str,
        width: usize,
        eps: f64,
    ) -> Result<Self, OcrError> {
        Ok(Self {
            scale: loader.get(&format!("{name}.w_0"), &[width])?,
            shift: loader.get(&format!("{name}.b_0"), &[width])?,
            eps,
        })
    }

    pub fn forward(&self, x: &Tensor) -> Result<Tensor, OcrError> {
        let mean = x.mean_keepdim(D::Minus1)?;
        let centered = x.broadcast_sub(&mean)?;
        let variance = centered.sqr()?.mean_keepdim(D::Minus1)?;
        let normed = centered.broadcast_div(&(variance + self.eps)?.sqrt()?)?;
        Ok(normed
            .broadcast_mul(&self.scale)?
            .broadcast_add(&self.shift)?)
    }
}

/// The two learned scalars every rep-layer of the backbone ends with
/// (`scale * x + bias`). They are far from the identity in a trained model, so
/// they cannot be optimized away.
#[derive(Debug)]
pub struct Affine {
    scale: f64,
    bias: f64,
}

impl Affine {
    pub fn load(loader: &Loader, index: usize) -> Result<Self, OcrError> {
        let name = format!("learnable_affine_block_{index}");
        Ok(Self {
            scale: f64::from(loader.scalar(&format!("{name}.w_0"))?),
            bias: f64::from(loader.scalar(&format!("{name}.w_1"))?),
        })
    }

    pub fn forward(&self, x: &Tensor) -> Result<Tensor, OcrError> {
        Ok(x.affine(self.scale, self.bias)?)
    }
}

/// Paddle's default hard-sigmoid slope: the float nearest `0.1666667`, which
/// is *not* the float nearest one sixth.
pub const HARD_SIGMOID_SLOPE: f64 = 0.166_666_7;
/// The slope the neck's squeeze-and-excitation gate carries instead.
pub const NECK_HARD_SIGMOID_SLOPE: f64 = 0.2;
pub const HARD_SIGMOID_OFFSET: f64 = 0.5;

/// `x * clamp(x + 3, 0, 6) / 6`.
pub fn hardswish(x: &Tensor) -> Result<Tensor, OcrError> {
    let gate = x
        .affine(1.0, 3.0)?
        .clamp(0f32, 6f32)?
        .affine(1.0 / 6.0, 0.0)?;
    Ok((x * gate)?)
}

/// `clamp(slope * x + offset, 0, 1)`.
pub fn hardsigmoid(x: &Tensor, slope: f64) -> Result<Tensor, OcrError> {
    Ok(x.affine(slope, HARD_SIGMOID_OFFSET)?.clamp(0f32, 1f32)?)
}

pub fn relu(x: &Tensor) -> Result<Tensor, OcrError> { Ok(x.relu()?) }

/// `x * sigmoid(x)`. The recognizer's neck uses this where the backbone around
/// it uses the piecewise-linear [`hardswish`]; they are not interchangeable.
pub fn swish(x: &Tensor) -> Result<Tensor, OcrError> { Ok(x.silu()?) }

/// A squeeze-and-excitation gate: global average, two 1x1 convolutions, a hard
/// sigmoid, and a channel-wise scaling of the input.
#[derive(Debug)]
pub struct SqueezeExcite {
    down: Conv,
    up: Conv,
    slope: f64,
}

impl SqueezeExcite {
    /// `down`/`up` name the two convolutions; `channels` is the gate's width
    /// and `reduced` the squeezed width.
    pub fn load(
        loader: &Loader,
        down: &str,
        up: &str,
        channels: usize,
        reduced: usize,
        slope: f64,
    ) -> Result<Self, OcrError> {
        Ok(Self {
            down: Conv::load(loader, down, [reduced, channels, 1, 1], 1, 0, 1)?,
            up: Conv::load(loader, up, [channels, reduced, 1, 1], 1, 0, 1)?,
            slope,
        })
    }

    pub fn forward(&self, x: &Tensor) -> Result<Tensor, OcrError> {
        let pooled = x.mean_keepdim(3)?.mean_keepdim(2)?;
        let gate = self.down.forward(&pooled)?;
        let gate = relu(&gate)?;
        let gate = self.up.forward(&gate)?;
        let gate = hardsigmoid(&gate, self.slope)?;
        Ok(x.broadcast_mul(&gate)?)
    }
}

/// Nearest-neighbour upsampling by an integer factor, replicating pixels the
/// way the exported graph does (no half-pixel centring).
pub fn upsample(x: &Tensor, factor: usize) -> Result<Tensor, OcrError> {
    let (_, _, h, w) = x.dims4()?;
    Ok(x.upsample_nearest2d(h * factor, w * factor)?)
}

/// Keeps every `stride.0`-th row and every `stride.1`-th column.
///
/// A convolution takes one stride for both axes, but the networks that read
/// text lines stride along one axis at a time — halving the height of a crop
/// that is already only 48 pixels tall while leaving its length alone. Such a
/// layer convolves at stride one and then drops the rows (or columns) a
/// per-axis stride would never have computed: the values that remain are the
/// same ones, at the price of computing those that go. Every layer that needs
/// this is depthwise, where that price is small.
pub fn subsample(
    x: &Tensor,
    stride: (usize, usize),
) -> Result<Tensor, OcrError> {
    let mut y = if x.is_contiguous() {
        x.clone()
    } else {
        x.contiguous()?
    };
    for (axis, stride) in [(2, stride.0), (3, stride.1)] {
        if stride > 1 {
            let kept: Vec<u32> =
                (0..y.dim(axis)? as u32).step_by(stride).collect();
            let count = kept.len();
            let kept = Tensor::from_vec(kept, count, x.device())?;
            y = y.index_select(&kept, axis)?;
        }
    }
    Ok(y)
}

/// Builds an NCHW input tensor from row-major `f32` channel data.
pub fn input_tensor(
    data: Vec<f32>,
    channels: usize,
    height: usize,
    width: usize,
    device: &Device,
) -> Result<Tensor, OcrError> {
    Ok(
        Tensor::from_vec(data, (1, channels, height, width), device)?
            .to_dtype(DType::F32)?,
    )
}
