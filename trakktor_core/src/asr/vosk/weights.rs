//! Weight extraction from the ONNX exports and geometry inference.
//!
//! A model directory holds the icefall export triple `encoder.onnx`,
//! `decoder.onnx`, `joiner.onnx` (fp32) plus `tokens.txt`. This module turns
//! the three graphs into a flat map of **canonically named** `f32` tensors —
//! the original PyTorch `state_dict` paths — plus a [`ZipformerConfig`]
//! describing the geometry, so the runtimes never look at ONNX again.
//!
//! Most initializers keep their `state_dict` names in the export. The
//! exceptions, re-derived from the graph structure:
//!
//! - **Linear (`MatMul`) weights**: the exporter pre-transposes them to `[in,
//!   out]` and renames them (`onnx::MatMul_*`). Every such weight is matched
//!   back through its `Add`-bias consumer, whose bias initializer keeps the
//!   clean `<path>.bias` name; the weight is stored transposed back as
//!   `<path>.weight`. The only bias-less linear, `linear_pos`, is matched by
//!   execution order (one per encoder layer, in layer order).
//! - **`SimpleDownsample` weights**: the exporter folds
//!   `softmax(downsample.bias)` into a `[ds, 1, 1]` constant feeding a `Mul →
//!   ReduceSum` pair; the folded softmax is stored as
//!   `<stack>.downsample.weights` (that is exactly the factor the forward pass
//!   uses).
//! - **Causal conv edge scales**: `chunkwise_conv_scale[0]`/`[1]` survive as
//!   two `[C, K]` constants inside the scale expression next to each
//!   `chunkwise_conv`; they are re-assembled into the `[2, C, K]` parameter.
//!
//! The mapping is validated end to end by golden-trace tests and was
//! cross-checked bit for bit against a published PyTorch checkpoint of the
//! line.

#[cfg(test)]
mod tests;

use std::collections::HashMap;

use super::{
    error::VoskError,
    onnx::{OnnxModel, OnnxTensor},
};

/// A runtime-neutral tensor: shape and row-major `f32` data.
#[derive(Debug, Clone)]
pub struct RawTensor {
    pub dims: Vec<usize>,
    pub data: Vec<f32>,
}

/// Per-stack encoder geometry.
#[derive(Debug, Clone)]
pub struct StackConfig {
    /// Number of encoder layers in the stack.
    pub num_layers: usize,
    /// Embedding dimension of the stack.
    pub encoder_dim: usize,
    /// Temporal downsampling factor of the stack (1 = none).
    pub downsample: usize,
    pub num_heads: usize,
    pub query_head_dim: usize,
    pub pos_head_dim: usize,
    pub value_head_dim: usize,
    /// Hidden width of the nonlin-attention module (`3·dim/4`).
    pub nonlin_hidden: usize,
    /// Depthwise-convolution kernel of the two conv modules.
    pub cnn_kernel: usize,
}

impl StackConfig {
    /// Canonical name prefix of layer `l` of stack `s` (downsampled stacks
    /// nest their layers under an inner `encoder`).
    pub fn layer_prefix(s: usize, downsample: usize, l: usize) -> String {
        if downsample == 1 {
            format!("encoder.encoders.{s}.layers.{l}")
        } else {
            format!("encoder.encoders.{s}.encoder.layers.{l}")
        }
    }
}

/// Streaming (causal) runtime parameters, from the export metadata.
#[derive(Debug, Clone)]
pub struct StreamingConfig {
    /// Feature frames consumed per step (window).
    pub window_frames: usize,
    /// Feature frames the window advances per step (`decode_chunk_len`).
    pub shift_frames: usize,
    /// Attention left-context length per stack, in that stack's frames.
    pub left_context: Vec<usize>,
}

/// The complete model geometry, inferred from weight shapes (plus export
/// metadata for the streaming parameters).
#[derive(Debug, Clone)]
pub struct ZipformerConfig {
    pub stacks: Vec<StackConfig>,
    /// Dimension of the relative positional encoding.
    pub pos_dim: usize,
    /// Feature dimension the encoder consumes (mel bins).
    pub feature_dim: usize,
    /// Output dimension of the encoder stacks (`max(encoder_dim)`).
    pub encoder_out_dim: usize,
    /// Joint dimension; the encoder output is already projected to it.
    pub joiner_dim: usize,
    /// Vocabulary size (blank id 0 included).
    pub vocab_size: usize,
    /// Decoder context length (tokens fed to the stateless decoder).
    pub context_size: usize,
    /// Decoder embedding width.
    pub decoder_dim: usize,
    /// Present exactly for causal (streaming) models.
    pub streaming: Option<StreamingConfig>,
}

/// The extracted model: geometry plus the canonical tensor map.
#[derive(Debug)]
pub struct ModelWeights {
    pub config: ZipformerConfig,
    tensors: HashMap<String, RawTensor>,
}

impl ModelWeights {
    /// The tensor under a canonical name, shape-checked.
    pub fn get(
        &self,
        name: &str,
        dims: &[usize],
    ) -> Result<&RawTensor, VoskError> {
        let t = self.tensors.get(name).ok_or_else(|| {
            VoskError::InvalidModel(format!("missing weight `{name}`"))
        })?;
        if t.dims != dims {
            return Err(VoskError::InvalidModel(format!(
                "weight `{name}`: shape {:?}, expected {dims:?}",
                t.dims
            )));
        }
        Ok(t)
    }

    /// The tensor under a canonical name, any shape.
    pub fn get_any(&self, name: &str) -> Result<&RawTensor, VoskError> {
        self.tensors.get(name).ok_or_else(|| {
            VoskError::InvalidModel(format!("missing weight `{name}`"))
        })
    }

    /// Whether a canonical name is present.
    pub fn contains(&self, name: &str) -> bool {
        self.tensors.contains_key(name)
    }

    /// Iterates the canonical names (diagnostics and tests).
    pub fn names(&self) -> impl Iterator<Item = &str> {
        self.tensors.keys().map(String::as_str)
    }
}

/// Reads and maps a model directory's ONNX triple.
pub fn load_dir(dir: &std::path::Path) -> Result<ModelWeights, VoskError> {
    let read = |file: &str| -> Result<Vec<u8>, VoskError> {
        let path = dir.join(file);
        std::fs::read(&path).map_err(|e| {
            VoskError::InvalidModel(format!("reading {}: {e}", path.display()))
        })
    };
    let encoder = OnnxModel::parse(&read("encoder.onnx")?, "encoder.onnx")?;
    let decoder = OnnxModel::parse(&read("decoder.onnx")?, "decoder.onnx")?;
    let joiner = OnnxModel::parse(&read("joiner.onnx")?, "joiner.onnx")?;
    extract(&encoder, &decoder, &joiner)
}

/// Maps the three parsed graphs into canonical tensors plus geometry.
pub fn extract(
    encoder: &OnnxModel,
    decoder: &OnnxModel,
    joiner: &OnnxModel,
) -> Result<ModelWeights, VoskError> {
    // Old exports of the line label themselves plain "zipformer"; the real
    // gate is the structural validation below, which fails loudly on any
    // other architecture.
    if let Some(kind) = encoder.metadata("model_type") &&
        kind != "zipformer2" &&
        kind != "zipformer"
    {
        return Err(VoskError::InvalidModel(format!(
            "unsupported model type `{kind}` (expected zipformer2)"
        )));
    }

    let mut tensors: HashMap<String, RawTensor> = HashMap::new();
    // Decoder/joiner exports keep every weight cleanly named (2-D `Gemm`s are
    // not folded); the encoder needs the graph-based recovery.
    for model in [decoder, joiner] {
        for t in &model.initializers {
            if !t.name.starts_with("onnx::") {
                insert(&mut tensors, &t.name, t.dims.clone(), t.data.clone())?;
            }
        }
    }
    map_encoder(encoder, &mut tensors)?;

    let config = infer_config(encoder, &tensors)?;
    validate(&config, &tensors)?;
    Ok(ModelWeights { config, tensors })
}

fn insert(
    map: &mut HashMap<String, RawTensor>,
    name: &str,
    dims: Vec<usize>,
    data: Vec<f32>,
) -> Result<(), VoskError> {
    if map
        .insert(name.to_string(), RawTensor { dims, data })
        .is_some()
    {
        return Err(VoskError::InvalidModel(format!(
            "duplicate weight `{name}`"
        )));
    }
    Ok(())
}

/// Transposes a `[rows, cols]` matrix.
fn transpose(t: &OnnxTensor) -> Result<RawTensor, VoskError> {
    let [rows, cols] = t.dims[..] else {
        return Err(VoskError::InvalidModel(format!(
            "weight `{}`: expected a matrix, got {:?}",
            t.name, t.dims
        )));
    };
    let mut out = vec![0.0f32; t.data.len()];
    for r in 0..rows {
        for c in 0..cols {
            out[c * rows + r] = t.data[r * cols + c];
        }
    }
    Ok(RawTensor {
        dims: vec![cols, rows],
        data: out,
    })
}

/// Recovers the encoder graph's tensors into canonical names.
fn map_encoder(
    encoder: &OnnxModel,
    tensors: &mut HashMap<String, RawTensor>,
) -> Result<(), VoskError> {
    let inits: HashMap<&str, &OnnxTensor> = encoder
        .initializers
        .iter()
        .map(|t| (t.name.as_str(), t))
        .collect();
    let mut consumers: HashMap<&str, Vec<usize>> = HashMap::new();
    for (i, n) in encoder.nodes.iter().enumerate() {
        for input in &n.inputs {
            consumers.entry(input).or_default().push(i);
        }
    }
    let mut produced_by: HashMap<&str, usize> = HashMap::new();
    for (i, n) in encoder.nodes.iter().enumerate() {
        for output in &n.outputs {
            produced_by.insert(output, i);
        }
    }

    // Clean names pass through as-is.
    for t in &encoder.initializers {
        if !t.name.starts_with("onnx::") {
            insert(tensors, &t.name, t.dims.clone(), t.data.clone())?;
        }
    }

    // Linear weights: MatMul with a folded (renamed, transposed) weight,
    // matched through the bias of its Add consumer. Bias-less ones are
    // collected for the linear_pos pass.
    let mut biasless: Vec<&OnnxTensor> = Vec::new();
    for node in &encoder.nodes {
        if node.op_type != "MatMul" || node.inputs.len() != 2 {
            continue;
        }
        let Some(&w) = inits.get(node.inputs[1].as_str()) else {
            continue;
        };
        if !w.name.starts_with("onnx::") {
            continue;
        }
        let mut bias: Option<&str> = None;
        for &c in consumers
            .get(node.outputs[0].as_str())
            .into_iter()
            .flatten()
        {
            let cn = &encoder.nodes[c];
            if cn.op_type != "Add" {
                continue;
            }
            let other = cn
                .inputs
                .iter()
                .find(|i| *i != &node.outputs[0])
                .map(String::as_str);
            if let Some(name) = other &&
                inits.contains_key(name) &&
                !name.starts_with("onnx::") &&
                name.ends_with(".bias")
            {
                bias = Some(name);
                break;
            }
        }
        match bias {
            Some(bias) => {
                let path = &bias[..bias.len() - ".bias".len()];
                insert_tensor(
                    tensors,
                    &format!("{path}.weight"),
                    transpose(w)?,
                )?;
            },
            None => biasless.push(w),
        }
    }

    // linear_pos: the only bias-less linear; one per encoder layer, assigned
    // in execution order, which is layer order.
    let mut layers: Vec<String> = tensors
        .keys()
        .filter_map(|k| {
            k.strip_suffix(".self_attn_weights.in_proj.bias")
                .map(str::to_string)
        })
        .collect();
    layers.sort_by_key(|p| layer_sort_key(p));
    if layers.len() != biasless.len() {
        return Err(VoskError::InvalidModel(format!(
            "{} bias-less linear weights for {} encoder layers",
            biasless.len(),
            layers.len()
        )));
    }
    for (layer, w) in layers.iter().zip(biasless) {
        insert_tensor(
            tensors,
            &format!("{layer}.self_attn_weights.linear_pos.weight"),
            transpose(w)?,
        )?;
    }

    // SimpleDownsample weights, stored canonically as the softmax the
    // forward pass multiplies by. Newer exports fold `softmax(bias)` into
    // [ds, 1, 1] constants (in execution order: downsampled stacks in stack
    // order, then the output downsample); older ones keep the clean
    // `downsample.bias` with an in-graph softmax.
    let n_stacks = 1 + tensors
        .keys()
        .filter_map(|k| {
            k.strip_prefix("encoder.encoders.")
                .and_then(|r| r.split('.').next())
                .and_then(|s| s.parse::<usize>().ok())
        })
        .max()
        .unwrap_or(0);
    let clean_biases: Vec<String> = tensors
        .keys()
        .filter(|k| {
            k.ends_with(".downsample.bias") ||
                *k == "encoder.downsample_output.bias"
        })
        .cloned()
        .collect();
    if clean_biases.is_empty() {
        let mut folded: Vec<Vec<f32>> = Vec::new();
        for node in &encoder.nodes {
            if node.op_type != "Mul" {
                continue;
            }
            for input in &node.inputs {
                let Some(&t) = inits.get(input.as_str()) else {
                    continue;
                };
                if t.name.starts_with("onnx::") &&
                    t.dims.len() == 3 &&
                    t.dims[1] == 1 &&
                    t.dims[2] == 1
                {
                    folded.push(t.data.clone());
                }
            }
        }
        if folded.len() != n_stacks {
            return Err(VoskError::InvalidModel(format!(
                "{} downsample weights for {n_stacks} stacks",
                folded.len()
            )));
        }
        // Stack 0 never downsamples; its slot is the output downsample.
        for (i, data) in folded.into_iter().enumerate() {
            let name = if i + 1 < n_stacks {
                format!("encoder.encoders.{}.downsample.weights", i + 1)
            } else {
                "encoder.downsample_output.weights".to_string()
            };
            let dims = vec![data.len()];
            insert(tensors, &name, dims, data)?;
        }
    } else {
        for bias_name in clean_biases {
            let bias = tensors[&bias_name].data.clone();
            let max = bias.iter().copied().fold(f32::NEG_INFINITY, f32::max);
            let exps: Vec<f32> = bias.iter().map(|v| (v - max).exp()).collect();
            let sum: f32 = exps.iter().sum();
            let weights: Vec<f32> = exps.iter().map(|v| v / sum).collect();
            let name = bias_name
                .strip_suffix(".bias")
                .expect("filtered on suffix")
                .to_string() +
                ".weights";
            let dims = vec![weights.len()];
            insert(tensors, &name, dims, weights)?;
        }
    }

    // Causal conv edge scales: for each chunkwise conv, the two [C, K]
    // constants inside the scale expression of its Mul consumer, in
    // execution order (left edge first).
    let chunkwise: Vec<String> = tensors
        .keys()
        .filter_map(|k| {
            k.strip_suffix(".chunkwise_conv.weight").map(str::to_string)
        })
        .collect();
    for base in chunkwise {
        let weight_name = format!("{base}.chunkwise_conv.weight");
        let conv = consumers
            .get(weight_name.as_str())
            .into_iter()
            .flatten()
            .map(|&i| &encoder.nodes[i])
            .find(|n| n.op_type == "Conv")
            .ok_or_else(|| {
                VoskError::InvalidModel(format!(
                    "`{weight_name}` has no Conv consumer"
                ))
            })?;
        let mul = consumers
            .get(conv.outputs[0].as_str())
            .into_iter()
            .flatten()
            .map(|&i| &encoder.nodes[i])
            .find(|n| n.op_type == "Mul")
            .ok_or_else(|| {
                VoskError::InvalidModel(format!(
                    "`{weight_name}`: conv output has no scale Mul"
                ))
            })?;
        let scale_in = mul
            .inputs
            .iter()
            .find(|i| *i != &conv.outputs[0])
            .ok_or_else(|| {
                VoskError::InvalidModel(format!(
                    "`{weight_name}`: malformed scale Mul"
                ))
            })?;
        let edges = collect_scale_edges(
            encoder,
            &inits,
            &produced_by,
            &consumers,
            scale_in,
        );
        let [left, right] = edges[..] else {
            return Err(VoskError::InvalidModel(format!(
                "`{base}`: expected 2 edge-scale constants, found {}",
                edges.len()
            )));
        };
        if left.dims != right.dims {
            return Err(VoskError::InvalidModel(format!(
                "`{base}`: edge-scale shapes differ"
            )));
        }
        let mut data = left.data.clone();
        data.extend_from_slice(&right.data);
        let dims = vec![2, left.dims[0], left.dims[1]];
        insert(tensors, &format!("{base}.chunkwise_conv_scale"), dims, data)?;
    }

    Ok(())
}

fn insert_tensor(
    map: &mut HashMap<String, RawTensor>,
    name: &str,
    t: RawTensor,
) -> Result<(), VoskError> {
    insert(map, name, t.dims, t.data)
}

/// Walks the constant expression under a scale `Mul` input and returns the
/// 2-D `f32` constants inside it, ordered by their first consumer's position
/// (execution order — the left edge is computed first).
fn collect_scale_edges<'a>(
    encoder: &'a OnnxModel,
    inits: &HashMap<&str, &'a OnnxTensor>,
    produced_by: &HashMap<&str, usize>,
    consumers: &HashMap<&str, Vec<usize>>,
    root: &str,
) -> Vec<&'a OnnxTensor> {
    // Ops that can appear inside the folded constant expression; anything
    // else ends the walk (never reached in practice — the expression is a
    // slice/pad/add chain).
    const CONST_OPS: &[&str] = &[
        "Slice",
        "Concat",
        "Pad",
        "Add",
        "ConstantOfShape",
        "Unsqueeze",
        "Squeeze",
        "Cast",
        "Constant",
        "Reshape",
        "Expand",
        "Shape",
        "Gather",
    ];
    let mut found: Vec<(usize, &OnnxTensor)> = Vec::new();
    let mut seen: std::collections::HashSet<&str> =
        std::collections::HashSet::new();
    let mut queue: Vec<&str> = vec![root];
    while let Some(v) = queue.pop() {
        if !seen.insert(v) {
            continue;
        }
        if let Some(&t) = inits.get(v) {
            if t.dims.len() == 2 {
                let first_use = consumers
                    .get(v)
                    .and_then(|c| c.iter().min())
                    .copied()
                    .unwrap_or(usize::MAX);
                found.push((first_use, t));
            }
            continue;
        }
        if let Some(&p) = produced_by.get(v) {
            let node = &encoder.nodes[p];
            if CONST_OPS.contains(&node.op_type.as_str()) {
                queue.extend(node.inputs.iter().map(String::as_str));
            }
        }
    }
    found.sort_by_key(|(first_use, _)| *first_use);
    found.into_iter().map(|(_, t)| t).collect()
}

/// Sort key `(stack, layer)` of a canonical layer prefix.
fn layer_sort_key(prefix: &str) -> (usize, usize) {
    // encoder.encoders.<s>.layers.<l> or
    // encoder.encoders.<s>.encoder.layers.<l>
    let parts: Vec<&str> = prefix.split('.').collect();
    let s = parts.get(2).and_then(|p| p.parse().ok()).unwrap_or(0);
    let l = match parts.get(3) {
        Some(&"layers") => parts.get(4),
        _ => parts.get(5),
    }
    .and_then(|p| p.parse().ok())
    .unwrap_or(0);
    (s, l)
}

/// Infers the model geometry from tensor shapes and export metadata.
fn infer_config(
    encoder: &OnnxModel,
    tensors: &HashMap<String, RawTensor>,
) -> Result<ZipformerConfig, VoskError> {
    let bad = |m: String| VoskError::InvalidModel(m);
    let dim_of = |name: &str, axis: usize| -> Result<usize, VoskError> {
        tensors
            .get(name)
            .and_then(|t| t.dims.get(axis).copied())
            .ok_or_else(|| bad(format!("missing weight `{name}`")))
    };

    // Stacks and layers by name scan.
    let mut n_stacks = 0usize;
    let mut layers_per_stack: HashMap<usize, usize> = HashMap::new();
    for name in tensors.keys() {
        let Some(rest) = name.strip_prefix("encoder.encoders.") else {
            continue;
        };
        let mut parts = rest.split('.');
        let Some(s) = parts.next().and_then(|p| p.parse::<usize>().ok()) else {
            continue;
        };
        n_stacks = n_stacks.max(s + 1);
        let mut parts = parts.peekable();
        if parts.peek() == Some(&"encoder") {
            parts.next();
        }
        if parts.next() == Some("layers") &&
            let Some(l) = parts.next().and_then(|p| p.parse::<usize>().ok())
        {
            let entry = layers_per_stack.entry(s).or_default();
            *entry = (*entry).max(l + 1);
        }
    }
    if n_stacks == 0 {
        return Err(bad("no encoder stacks found".into()));
    }

    // The reference recipe fixes the positional head width; everything else
    // is derived and cross-checked, so an incompatible export fails loudly.
    const POS_HEAD_DIM: usize = 4;

    let mut stacks = Vec::with_capacity(n_stacks);
    for s in 0..n_stacks {
        let num_layers = *layers_per_stack
            .get(&s)
            .ok_or_else(|| bad(format!("stack {s}: no layers")))?;
        let downsample = if s == 0 {
            1
        } else {
            dim_of(&format!("encoder.encoders.{s}.downsample.weights"), 0)?
        };
        let p = StackConfig::layer_prefix(s, downsample, 0);
        let encoder_dim = dim_of(&format!("{p}.norm.bias"), 0)?;
        let pos_out =
            dim_of(&format!("{p}.self_attn_weights.linear_pos.weight"), 0)?;
        if pos_out % POS_HEAD_DIM != 0 {
            return Err(bad(format!(
                "stack {s}: linear_pos width {pos_out} is not a multiple of \
                 the positional head width {POS_HEAD_DIM}"
            )));
        }
        let num_heads = pos_out / POS_HEAD_DIM;
        let in_proj_out =
            dim_of(&format!("{p}.self_attn_weights.in_proj.bias"), 0)?;
        let per_head = in_proj_out
            .checked_div(num_heads)
            .filter(|_| in_proj_out % num_heads == 0)
            .ok_or_else(|| {
                bad(format!("stack {s}: in_proj width {in_proj_out}"))
            })?;
        if !(per_head - POS_HEAD_DIM).is_multiple_of(2) {
            return Err(bad(format!(
                "stack {s}: cannot split in_proj width {in_proj_out} into \
                 {num_heads} heads"
            )));
        }
        let query_head_dim = (per_head - POS_HEAD_DIM) / 2;
        let value_out = dim_of(&format!("{p}.self_attn1.in_proj.bias"), 0)?;
        if value_out % num_heads != 0 {
            return Err(bad(format!(
                "stack {s}: value width {value_out} for {num_heads} heads"
            )));
        }
        let value_head_dim = value_out / num_heads;
        let nonlin3 = dim_of(&format!("{p}.nonlin_attention.in_proj.bias"), 0)?;
        if nonlin3 % 3 != 0 {
            return Err(bad(format!(
                "stack {s}: nonlin-attention width {nonlin3}"
            )));
        }
        let causal = tensors.contains_key(&format!(
            "{p}.conv_module1.depthwise_conv.chunkwise_conv.weight"
        ));
        let cnn_kernel = if causal {
            dim_of(
                &format!(
                    "{p}.conv_module1.depthwise_conv.chunkwise_conv.weight"
                ),
                2,
            )?
        } else {
            dim_of(&format!("{p}.conv_module1.depthwise_conv.weight"), 2)?
        };
        stacks.push(StackConfig {
            num_layers,
            encoder_dim,
            downsample,
            num_heads,
            query_head_dim,
            pos_head_dim: POS_HEAD_DIM,
            value_head_dim,
            nonlin_hidden: nonlin3 / 3,
            cnn_kernel,
        });
    }

    let p0 = StackConfig::layer_prefix(0, 1, 0);
    let pos_dim =
        dim_of(&format!("{p0}.self_attn_weights.linear_pos.weight"), 1)?;
    let encoder_out_dim =
        stacks.iter().map(|s| s.encoder_dim).max().unwrap_or(0);

    // The line's feature width is fixed; the embed output linear confirms it
    // (in = 128 · (((feat − 1) / 2 − 1) / 2), which floors and is thus not
    // uniquely invertible).
    let feature_dim = 80usize;
    let embed_in = dim_of("encoder_embed.out.weight", 1)?;
    let embed_channels = dim_of("encoder_embed.conv.7.weight", 0)?;
    let out_width = (((feature_dim - 1) / 2) - 1) / 2;
    if embed_in != embed_channels * out_width {
        return Err(bad(format!(
            "embed output width {embed_in} does not match {feature_dim} \
             feature bins ({embed_channels}·{out_width})"
        )));
    }

    let joiner_dim = dim_of("encoder_proj.bias", 0)?;
    let vocab_size = dim_of("output_linear.weight", 0)?;
    let decoder_dim = dim_of("decoder.embedding.weight", 1)?;
    let context_size = if tensors.contains_key("decoder.conv.weight") {
        dim_of("decoder.conv.weight", 2)?
    } else {
        1
    };

    let causal = tensors
        .keys()
        .any(|k| k.ends_with(".chunkwise_conv.weight"));
    let streaming = if causal {
        let meta_usize = |key: &str| -> Result<usize, VoskError> {
            encoder
                .metadata(key)
                .and_then(|v| v.parse().ok())
                .ok_or_else(|| {
                    bad(format!("streaming export without metadata `{key}`"))
                })
        };
        let left: Vec<usize> = encoder
            .metadata("left_context_len")
            .unwrap_or("")
            .split(',')
            .filter_map(|v| v.trim().parse().ok())
            .collect();
        if left.len() != n_stacks {
            return Err(bad(format!(
                "streaming metadata left_context_len has {} entries for \
                 {n_stacks} stacks",
                left.len()
            )));
        }
        Some(StreamingConfig {
            window_frames: meta_usize("T")?,
            shift_frames: meta_usize("decode_chunk_len")?,
            left_context: left,
        })
    } else {
        None
    };

    Ok(ZipformerConfig {
        stacks,
        pos_dim,
        feature_dim,
        encoder_out_dim,
        joiner_dim,
        vocab_size,
        context_size,
        decoder_dim,
        streaming,
    })
}

/// Cross-checks the inferred geometry against every tensor the runtimes will
/// read, so shape errors surface at load, not mid-forward.
fn validate(
    config: &ZipformerConfig,
    tensors: &HashMap<String, RawTensor>,
) -> Result<(), VoskError> {
    let get = |name: &str, dims: &[usize]| -> Result<(), VoskError> {
        let t = tensors.get(name).ok_or_else(|| {
            VoskError::InvalidModel(format!("missing weight `{name}`"))
        })?;
        if t.dims != dims {
            return Err(VoskError::InvalidModel(format!(
                "weight `{name}`: shape {:?}, expected {dims:?}",
                t.dims
            )));
        }
        Ok(())
    };

    // Embed front end.
    let f = config.feature_dim;
    let out_width = (((f - 1) / 2) - 1) / 2;
    get("encoder_embed.conv.0.weight", &[8, 1, 3, 3])?;
    get("encoder_embed.conv.4.weight", &[32, 8, 3, 3])?;
    get("encoder_embed.conv.7.weight", &[128, 32, 3, 3])?;
    get(
        "encoder_embed.convnext.depthwise_conv.weight",
        &[128, 1, 7, 7],
    )?;
    get(
        "encoder_embed.convnext.pointwise_conv1.weight",
        &[384, 128, 1, 1],
    )?;
    get(
        "encoder_embed.convnext.pointwise_conv2.weight",
        &[128, 384, 1, 1],
    )?;
    let d0 = config.stacks[0].encoder_dim;
    get("encoder_embed.out.weight", &[d0, 128 * out_width])?;
    get("encoder_embed.out_norm.bias", &[d0])?;
    get("encoder_embed.out_norm.log_scale", &[])?;

    for (s, stack) in config.stacks.iter().enumerate() {
        let d = stack.encoder_dim;
        let h = stack.num_heads;
        let (q, p, v) = (
            stack.query_head_dim,
            stack.pos_head_dim,
            stack.value_head_dim,
        );
        let hidden = stack.nonlin_hidden;
        let k = stack.cnn_kernel;
        if s > 0 {
            get(
                &format!("encoder.encoders.{s}.downsample.weights"),
                &[stack.downsample],
            )?;
            get(
                &format!("encoder.encoders.{s}.out_combiner.bypass_scale"),
                &[d],
            )?;
        }
        for l in 0..stack.num_layers {
            let pre = StackConfig::layer_prefix(s, stack.downsample, l);
            let lin = |suffix: &str, o: usize, i: usize| {
                [
                    (format!("{pre}.{suffix}.weight"), vec![o, i]),
                    (format!("{pre}.{suffix}.bias"), vec![o]),
                ]
            };
            let mut checks: Vec<(String, Vec<usize>)> = Vec::new();
            checks.extend(lin("self_attn_weights.in_proj", (2 * q + p) * h, d));
            checks.push((
                format!("{pre}.self_attn_weights.linear_pos.weight"),
                vec![p * h, config.pos_dim],
            ));
            checks.extend(lin("self_attn1.in_proj", h * v, d));
            checks.extend(lin("self_attn1.out_proj", d, h * v));
            checks.extend(lin("self_attn2.in_proj", h * v, d));
            checks.extend(lin("self_attn2.out_proj", d, h * v));
            checks.extend(lin("nonlin_attention.in_proj", 3 * hidden, d));
            checks.extend(lin("nonlin_attention.out_proj", d, hidden));
            for name in ["feed_forward1", "feed_forward2", "feed_forward3"] {
                let ff_dim = tensors
                    .get(&format!("{pre}.{name}.in_proj.bias"))
                    .map(|t| t.dims[0])
                    .unwrap_or(0);
                checks.extend(lin(&format!("{name}.in_proj"), ff_dim, d));
                checks.extend(lin(&format!("{name}.out_proj"), d, ff_dim));
            }
            for conv in ["conv_module1", "conv_module2"] {
                checks.extend(lin(&format!("{conv}.in_proj"), 2 * d, d));
                checks.extend(lin(&format!("{conv}.out_proj"), d, d));
                if config.streaming.is_some() {
                    let half = k / 2 + 1;
                    checks.push((
                        format!(
                            "{pre}.{conv}.depthwise_conv.causal_conv.weight"
                        ),
                        vec![d, 1, half],
                    ));
                    checks.push((
                        format!("{pre}.{conv}.depthwise_conv.causal_conv.bias"),
                        vec![d],
                    ));
                    checks.push((
                        format!(
                            "{pre}.{conv}.depthwise_conv.chunkwise_conv.weight"
                        ),
                        vec![d, 1, k],
                    ));
                    checks.push((
                        format!(
                            "{pre}.{conv}.depthwise_conv.chunkwise_conv.bias"
                        ),
                        vec![d],
                    ));
                } else {
                    checks.push((
                        format!("{pre}.{conv}.depthwise_conv.weight"),
                        vec![d, 1, k],
                    ));
                    checks.push((
                        format!("{pre}.{conv}.depthwise_conv.bias"),
                        vec![d],
                    ));
                }
            }
            checks.push((format!("{pre}.norm.bias"), vec![d]));
            checks.push((format!("{pre}.norm.log_scale"), vec![]));
            checks.push((format!("{pre}.bypass.bypass_scale"), vec![d]));
            checks.push((format!("{pre}.bypass_mid.bypass_scale"), vec![d]));
            for (name, dims) in checks {
                get(&name, &dims)?;
            }
        }
    }

    get("encoder.downsample_output.weights", &[2])?;
    get(
        "encoder_proj.weight",
        &[config.joiner_dim, config.encoder_out_dim],
    )?;
    get("encoder_proj.bias", &[config.joiner_dim])?;

    // Decoder and joiner.
    get(
        "decoder.embedding.weight",
        &[config.vocab_size, config.decoder_dim],
    )?;
    if config.context_size > 1 {
        let groups_width = tensors
            .get("decoder.conv.weight")
            .map(|t| t.dims[1])
            .unwrap_or(0);
        get(
            "decoder.conv.weight",
            &[config.decoder_dim, groups_width, config.context_size],
        )?;
    }
    get(
        "decoder_proj.weight",
        &[config.joiner_dim, config.decoder_dim],
    )?;
    get("decoder_proj.bias", &[config.joiner_dim])?;
    get(
        "output_linear.weight",
        &[config.vocab_size, config.joiner_dim],
    )?;
    get("output_linear.bias", &[config.vocab_size])?;
    Ok(())
}
