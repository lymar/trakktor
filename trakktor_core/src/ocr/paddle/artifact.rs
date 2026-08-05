//! Reading PaddleOCR's published inference artifacts without PaddlePaddle.
//!
//! A model directory holds three files that together describe one network:
//!
//! - `inference.pdiparams` — the weights, as a bare concatenation of records
//!   with no names, no index and no magic. Each record is `u32` stream version,
//!   `u64` LoD level count, `u32` stream version, `i32` descriptor length, a
//!   protobuf `TensorDesc` (data type + dims) and then the raw row-major
//!   payload.
//! - `inference.json` — the computation graph in Paddle's PIR JSON form. The
//!   weights file carries no names, so the names come from here: every
//!   parameter is an op with the compressed name `p`, and its declared shape
//!   sits on the op's single result.
//! - `config.json` — pre/post-processing parameters and, for a recognizer, the
//!   character dictionary (see [`super::config`]).
//!
//! **The record order is the parameter names sorted lexicographically**, not
//! the order the graph declares them in. That is what the writer does
//! (`sorted(...)` over the name map before combining the tensors), and it is
//! the whole of the name-to-weight mapping: nothing else in the file ties a
//! record to a name. Sorting is by bytes; every published name is ASCII, so
//! Rust's `str` ordering matches the writer's Python one exactly.
//!
//! Two properties of the exported graphs matter to the callers of this module
//! and are checked here rather than rediscovered downstream: a detector's
//! sigmoid and a recognizer's softmax are **inside** the graph (the network
//! emits probabilities, not logits), and the declared input/output dims tell
//! the runtimes what geometry the artifact expects.

#[cfg(test)]
mod tests;

use std::{collections::HashMap, fs, path::Path};

use crate::ocr::error::OcrError;

/// A runtime-neutral tensor: shape and row-major `f32` data.
#[derive(Debug, Clone)]
pub struct RawTensor {
    pub dims: Vec<usize>,
    pub data: Vec<f32>,
}

impl RawTensor {
    /// Number of elements.
    pub fn len(&self) -> usize { self.data.len() }

    pub fn is_empty(&self) -> bool { self.data.is_empty() }
}

/// One network read from a model directory: its weights by name, plus the
/// input/output geometry the graph declares (`-1` marks a dynamic dimension).
#[derive(Debug)]
pub struct Artifact {
    tensors: HashMap<String, RawTensor>,
    pub input_dims: Vec<i64>,
    pub output_dims: Vec<i64>,
}

/// The two file names every published model directory carries.
pub const GRAPH_FILE: &str = "inference.json";
pub const WEIGHTS_FILE: &str = "inference.pdiparams";

impl Artifact {
    /// Reads `inference.json` + `inference.pdiparams` from `dir`.
    pub fn load(dir: &Path) -> Result<Self, OcrError> {
        let graph = read_json(&dir.join(GRAPH_FILE))?;
        let params = parameters(&graph)?;
        let (input_dims, output_dims) = graph_io(&graph)?;

        let bytes = read_file(&dir.join(WEIGHTS_FILE))?;
        let records = records(&bytes)?;

        let mut names: Vec<&Parameter> = params.iter().collect();
        names.sort_unstable_by(|a, b| a.name.cmp(&b.name));
        for pair in names.windows(2) {
            if pair[0].name == pair[1].name {
                return Err(artifact(format!(
                    "the graph declares parameter `{}` twice",
                    pair[0].name
                )));
            }
        }
        if names.len() != records.len() {
            return Err(artifact(format!(
                "the graph declares {} parameters but the weights file holds \
                 {} tensors",
                names.len(),
                records.len()
            )));
        }

        let mut tensors = HashMap::with_capacity(names.len());
        for (param, record) in names.iter().zip(&records) {
            if record.dims != param.dims {
                return Err(artifact(format!(
                    "parameter `{}` is {:?} in the graph but {:?} in the \
                     weights file",
                    param.name, param.dims, record.dims
                )));
            }
            let payload = &bytes[record.payload.clone()];
            let data = match record.data_type {
                DATA_TYPE_BOOL => payload
                    .iter()
                    .map(|b| if *b == 0 { 0.0 } else { 1.0 })
                    .collect(),
                _ => f32_le(payload),
            };
            tensors.insert(
                param.name.clone(),
                RawTensor {
                    dims: record.dims.clone(),
                    data,
                },
            );
        }

        Ok(Self {
            tensors,
            input_dims,
            output_dims,
        })
    }

    /// Looks a weight up by its published name.
    pub fn tensor(&self, name: &str) -> Result<&RawTensor, OcrError> {
        self.tensors
            .get(name)
            .ok_or_else(|| artifact(format!("weight `{name}` is missing")))
    }

    /// Looks a weight up and checks its shape, so a mismatch surfaces as a
    /// named model error instead of a confusing failure deeper in a net.
    pub fn shaped(
        &self,
        name: &str,
        dims: &[usize],
    ) -> Result<&RawTensor, OcrError> {
        let tensor = self.tensor(name)?;
        if tensor.dims != dims {
            return Err(artifact(format!(
                "weight `{name}` is {:?}, expected {dims:?}",
                tensor.dims
            )));
        }
        Ok(tensor)
    }

    /// Whether a weight of this name is present.
    pub fn has(&self, name: &str) -> bool { self.tensors.contains_key(name) }

    /// Every weight name the file carries, in no particular order.
    pub fn names(&self) -> impl Iterator<Item = &str> {
        self.tensors.keys().map(String::as_str)
    }

    /// Number of weights read.
    pub fn len(&self) -> usize { self.tensors.len() }

    pub fn is_empty(&self) -> bool { self.tensors.is_empty() }

    /// The last declared output dimension — the class count of a recognizer
    /// or classifier head.
    pub fn output_classes(&self) -> Option<usize> {
        match self.output_dims.last() {
            Some(&n) if n > 0 => Some(n as usize),
            _ => None,
        }
    }
}

fn artifact(message: String) -> OcrError { OcrError::Artifact(message) }

fn read_file(path: &Path) -> Result<Vec<u8>, OcrError> {
    fs::read(path).map_err(|source| OcrError::ModelFile {
        path: path.display().to_string(),
        source,
    })
}

fn read_json(path: &Path) -> Result<serde_json::Value, OcrError> {
    let bytes = read_file(path)?;
    serde_json::from_slice(&bytes).map_err(|e| {
        artifact(format!("{} is not valid JSON: {e}", path.display()))
    })
}

/// A parameter as the graph declares it.
struct Parameter {
    name: String,
    dims: Vec<usize>,
}

/// One record of the weights file: what the descriptor said, and where the
/// payload lives inside the file.
struct Record {
    dims: Vec<usize>,
    data_type: u64,
    payload: std::ops::Range<usize>,
}

/// The single block of ops the exported programs carry.
fn ops(graph: &serde_json::Value) -> Result<&Vec<serde_json::Value>, OcrError> {
    let magic = graph
        .get("base_code")
        .and_then(|c| c.get("magic"))
        .and_then(|m| m.as_str());
    if magic != Some("pir") {
        return Err(artifact(format!(
            "not a Paddle PIR program (magic {magic:?})"
        )));
    }
    // The version is deliberately not checked: the published artifacts say 1,
    // while a program saved by a current Paddle build says 4, and both parse.
    let regions = graph
        .pointer("/program/regions")
        .and_then(|r| r.as_array())
        .ok_or_else(|| artifact("the program has no regions".into()))?;
    let [region] = regions.as_slice() else {
        return Err(artifact(format!(
            "expected one region, found {}",
            regions.len()
        )));
    };
    let blocks = region
        .get("blocks")
        .and_then(|b| b.as_array())
        .ok_or_else(|| artifact("the region has no blocks".into()))?;
    let [block] = blocks.as_slice() else {
        return Err(artifact(format!(
            "expected one block, found {}",
            blocks.len()
        )));
    };
    block
        .get("ops")
        .and_then(|o| o.as_array())
        .ok_or_else(|| artifact("the block has no ops".into()))
}

/// Collects the parameter declarations. A parameter op is compressed: its
/// name is the only string in the positional attribute array, and its result
/// is a bare object rather than the array every other op carries.
fn parameters(graph: &serde_json::Value) -> Result<Vec<Parameter>, OcrError> {
    let mut params = Vec::new();
    for op in ops(graph)? {
        if op.get("#").and_then(|n| n.as_str()) != Some("p") {
            continue;
        }
        let name = op
            .get("A")
            .and_then(|a| a.as_array())
            .and_then(|a| a.iter().find_map(|v| v.as_str()))
            .ok_or_else(|| artifact("a parameter op carries no name".into()))?;
        let dims = result_dims(op.get("O"))?;
        let mut shape = Vec::with_capacity(dims.len());
        for dim in &dims {
            if *dim <= 0 {
                return Err(artifact(format!(
                    "parameter `{name}` has a dynamic shape {dims:?}"
                )));
            }
            shape.push(*dim as usize);
        }
        params.push(Parameter {
            name: name.to_string(),
            dims: shape,
        });
    }
    if params.is_empty() {
        return Err(artifact("the graph declares no parameters".into()));
    }
    Ok(params)
}

/// Dims of a result value, from either shape a result takes (`p` gives an
/// object, every other op an array).
fn result_dims(
    result: Option<&serde_json::Value>,
) -> Result<Vec<i64>, OcrError> {
    let value = match result {
        Some(serde_json::Value::Array(items)) => items
            .first()
            .ok_or_else(|| artifact("an op has an empty result list".into()))?,
        Some(object @ serde_json::Value::Object(_)) => object,
        _ => return Err(artifact("an op has no result".into())),
    };
    let tensor_type = value
        .get("TT")
        .ok_or_else(|| artifact("a result carries no type".into()))?;
    if tensor_type.get("#").and_then(|n| n.as_str()) != Some("0.t_dtensor") {
        return Err(artifact("a result is not a dense tensor".into()));
    }
    let dims = tensor_type
        .pointer("/D/1")
        .and_then(|d| d.as_array())
        .ok_or_else(|| artifact("a tensor type carries no dims".into()))?;
    dims.iter()
        .map(|d| {
            d.as_i64()
                .ok_or_else(|| artifact("a dim is not an integer".into()))
        })
        .collect()
}

/// The declared input and output geometry: the `data` op's `shape` attribute
/// and the `fetch` op's result dims.
fn graph_io(
    graph: &serde_json::Value,
) -> Result<(Vec<i64>, Vec<i64>), OcrError> {
    let ops = ops(graph)?;
    let input = ops
        .iter()
        .find(|op| op.get("#").and_then(|n| n.as_str()) == Some("1.data"))
        .and_then(|op| op.get("A"))
        .and_then(|a| a.as_array())
        .and_then(|attrs| {
            attrs.iter().find(|attr| {
                attr.get("N").and_then(|n| n.as_str()) == Some("shape")
            })
        })
        .and_then(|attr| attr.pointer("/AT/D"))
        .and_then(|d| d.as_array())
        .ok_or_else(|| artifact("the graph declares no input".into()))?
        .iter()
        .map(|d| {
            d.as_i64().ok_or_else(|| {
                artifact("an input dim is not an integer".into())
            })
        })
        .collect::<Result<Vec<_>, _>>()?;

    let fetch = ops
        .iter()
        .find(|op| op.get("#").and_then(|n| n.as_str()) == Some("1.fetch"))
        .ok_or_else(|| artifact("the graph fetches no result".into()))?;
    let output = result_dims(fetch.get("O"))?;

    Ok((input, output))
}

/// Walks the weights file into records. The file is self-delimiting and
/// carries no length or checksum, so the only integrity signal is that the
/// walk consumes it exactly — a truncated download otherwise looks like a
/// valid shorter file.
fn records(bytes: &[u8]) -> Result<Vec<Record>, OcrError> {
    /// Descriptors seen in published models are 4..12 bytes; the cap keeps a
    /// corrupt length from allocating wildly.
    const MAX_DESC: usize = 1024;

    let mut records = Vec::new();
    let mut at = 0usize;
    while at < bytes.len() {
        let version = read_u32(bytes, &mut at)?;
        let lod_levels = read_u64(bytes, &mut at)?;
        let desc_version = read_u32(bytes, &mut at)?;
        if version != 0 || desc_version != 0 {
            return Err(artifact(format!(
                "unsupported tensor stream version {version}/{desc_version}"
            )));
        }
        if lod_levels != 0 {
            return Err(artifact(
                "the weights file carries level-of-detail data, which no \
                 published OCR model uses"
                    .into(),
            ));
        }
        let desc_size = read_u32(bytes, &mut at)? as usize;
        if desc_size == 0 || desc_size > MAX_DESC {
            return Err(artifact(format!(
                "tensor descriptor length {desc_size} is out of range"
            )));
        }
        let end = at
            .checked_add(desc_size)
            .filter(|end| *end <= bytes.len())
            .ok_or_else(|| artifact("the weights file is truncated".into()))?;
        let (data_type, dims) = tensor_desc(&bytes[at..end])?;
        at = end;

        let width = match data_type {
            DATA_TYPE_FP32 => 4,
            DATA_TYPE_BOOL => 1,
            other => {
                return Err(artifact(format!(
                    "tensor data type {other} is neither float32 nor bool; \
                     this reader only handles what the published models carry"
                )));
            },
        };
        let numel: usize = dims.iter().product();
        let payload = numel.checked_mul(width).ok_or_else(|| {
            artifact("a tensor is too large to address".into())
        })?;
        let end = at
            .checked_add(payload)
            .filter(|end| *end <= bytes.len())
            .ok_or_else(|| artifact("the weights file is truncated".into()))?;
        records.push(Record {
            dims,
            data_type,
            payload: at..end,
        });
        at = end;
    }
    Ok(records)
}

/// `VarType::TensorDesc.data_type` for float32 and for bool. The enum is not
/// contiguous, so no arithmetic may be done on it.
///
/// Almost every published weight is float32; the layout detector carries one
/// `bool` tensor — the mask of valid anchor positions, folded into the graph
/// because that model was exported at a fixed input size. A bool is read as
/// 0.0/1.0, which is how the graph uses it (it multiplies the memory).
const DATA_TYPE_FP32: u64 = 5;
const DATA_TYPE_BOOL: u64 = 0;

/// Parses the `VarType::TensorDesc` protobuf: field 1 is the data type, field
/// 2 is a repeated dim. Proto2 leaves the repeated field unpacked, but the
/// packed encoding is accepted so a future writer does not break the reader.
fn tensor_desc(bytes: &[u8]) -> Result<(u64, Vec<usize>), OcrError> {
    let mut at = 0usize;
    let mut data_type = None;
    let mut dims = Vec::new();
    while at < bytes.len() {
        let key = varint(bytes, &mut at)?;
        match (key >> 3, key & 7) {
            (1, 0) => data_type = Some(varint(bytes, &mut at)?),
            (2, 0) => dims.push(dim(varint(bytes, &mut at)?)?),
            (2, 2) => {
                let len = varint(bytes, &mut at)? as usize;
                let end = at.checked_add(len).filter(|e| *e <= bytes.len());
                let end = end.ok_or_else(|| {
                    artifact("a tensor descriptor is truncated".into())
                })?;
                while at < end {
                    dims.push(dim(varint(bytes, &mut at)?)?);
                }
            },
            (field, wire) => {
                return Err(artifact(format!(
                    "unexpected field {field} (wire type {wire}) in a tensor \
                     descriptor"
                )));
            },
        }
    }
    let data_type = data_type
        .ok_or_else(|| artifact("a tensor has no data type".into()))?;
    if dims.is_empty() {
        return Err(artifact("a tensor has no dims".into()));
    }
    Ok((data_type, dims))
}

/// Dims are `int64` on the wire but varints are unsigned; a negative (or
/// absurd) dimension means the descriptor is not what this reader expects.
fn dim(raw: u64) -> Result<usize, OcrError> {
    let value = raw as i64;
    if value <= 0 {
        return Err(artifact(format!("a tensor dim is {value}")));
    }
    Ok(value as usize)
}

fn varint(bytes: &[u8], at: &mut usize) -> Result<u64, OcrError> {
    let mut value = 0u64;
    let mut shift = 0u32;
    loop {
        let byte = *bytes.get(*at).ok_or_else(|| {
            artifact("a tensor descriptor ends mid-number".into())
        })?;
        *at += 1;
        value |= u64::from(byte & 0x7f) << shift;
        if byte & 0x80 == 0 {
            return Ok(value);
        }
        shift += 7;
        if shift >= 64 {
            return Err(artifact(
                "a number in a tensor descriptor overflows".into(),
            ));
        }
    }
}

fn read_u32(bytes: &[u8], at: &mut usize) -> Result<u32, OcrError> {
    let end = *at + 4;
    let slice = bytes
        .get(*at..end)
        .ok_or_else(|| artifact("the weights file is truncated".into()))?;
    *at = end;
    Ok(u32::from_le_bytes(slice.try_into().expect("4 bytes")))
}

fn read_u64(bytes: &[u8], at: &mut usize) -> Result<u64, OcrError> {
    let end = *at + 8;
    let slice = bytes
        .get(*at..end)
        .ok_or_else(|| artifact("the weights file is truncated".into()))?;
    *at = end;
    Ok(u64::from_le_bytes(slice.try_into().expect("8 bytes")))
}

/// Payload offsets are not aligned — the descriptors before them are of odd
/// length — so the floats are decoded one at a time rather than cast.
fn f32_le(bytes: &[u8]) -> Vec<f32> {
    bytes
        .chunks_exact(4)
        .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect()
}
