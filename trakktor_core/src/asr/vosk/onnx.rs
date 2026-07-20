//! A minimal reader for the ONNX protobuf container.
//!
//! The models of this line ship their weights as ONNX graph exports, and
//! trakktor uses those files purely as a **weight container**: this module
//! parses just enough of the protobuf wire format to extract the named `f32`
//! initializer tensors, the graph's node list (needed to re-derive the
//! original parameter names of exporter-folded weights, see
//! [`weights`](super::weights)), and the model metadata. No graph execution,
//! no ONNX runtime, and no protobuf dependency — the wire format subset is
//! decoded by hand.
//!
//! Field numbers follow the ONNX protobuf schema (`onnx.proto3`): only the
//! handful of fields used here are decoded, everything else is skipped by
//! wire type.

#[cfg(test)]
mod tests;

use super::error::VoskError;

/// An `f32` initializer tensor: name, shape, row-major data.
#[derive(Debug, Clone)]
pub struct OnnxTensor {
    pub name: String,
    pub dims: Vec<usize>,
    pub data: Vec<f32>,
}

/// One graph node: operator, optional name, and its value names.
#[derive(Debug, Clone)]
pub struct OnnxNode {
    pub op_type: String,
    pub inputs: Vec<String>,
    pub outputs: Vec<String>,
}

/// The parts of an ONNX model file the engine consumes.
#[derive(Debug, Default)]
pub struct OnnxModel {
    /// `metadata_props` key/value pairs.
    pub metadata: Vec<(String, String)>,
    /// `f32` initializers, in file order (other dtypes are dropped).
    pub initializers: Vec<OnnxTensor>,
    /// Graph nodes in file order (the exporter emits execution order).
    pub nodes: Vec<OnnxNode>,
}

impl OnnxModel {
    /// Parses an ONNX file's bytes.
    pub fn parse(bytes: &[u8], label: &str) -> Result<Self, VoskError> {
        let mut model = OnnxModel::default();
        let mut r = Reader::new(bytes, label);
        while !r.at_end() {
            let (field, wire) = r.key()?;
            match field {
                // GraphProto graph = 7
                7 => {
                    let graph = r.bytes(wire)?;
                    parse_graph(&mut Reader::new(graph, label), &mut model)?;
                },
                // StringStringEntryProto metadata_props = 14
                14 => {
                    let entry = r.bytes(wire)?;
                    model
                        .metadata
                        .push(parse_string_pair(Reader::new(entry, label))?);
                },
                _ => r.skip(wire)?,
            }
        }
        Ok(model)
    }

    /// The value of a metadata key, if present.
    pub fn metadata(&self, key: &str) -> Option<&str> {
        self.metadata
            .iter()
            .find(|(k, _)| k == key)
            .map(|(_, v)| v.as_str())
    }
}

/// GraphProto: node = 1, initializer = 5.
fn parse_graph(r: &mut Reader, model: &mut OnnxModel) -> Result<(), VoskError> {
    while !r.at_end() {
        let (field, wire) = r.key()?;
        match field {
            1 => {
                let node = r.bytes(wire)?;
                model.nodes.push(parse_node(Reader::new(node, r.label))?);
            },
            5 => {
                let tensor = r.bytes(wire)?;
                if let Some(t) = parse_tensor(Reader::new(tensor, r.label))? {
                    model.initializers.push(t);
                }
            },
            _ => r.skip(wire)?,
        }
    }
    Ok(())
}

/// NodeProto: input = 1, output = 2, op_type = 4.
fn parse_node(mut r: Reader) -> Result<OnnxNode, VoskError> {
    let mut node = OnnxNode {
        op_type: String::new(),
        inputs: Vec::new(),
        outputs: Vec::new(),
    };
    while !r.at_end() {
        let (field, wire) = r.key()?;
        match field {
            1 => node.inputs.push(r.string(wire)?),
            2 => node.outputs.push(r.string(wire)?),
            4 => node.op_type = r.string(wire)?,
            _ => r.skip(wire)?,
        }
    }
    Ok(node)
}

/// TensorProto data_type value for f32.
const DTYPE_F32: u64 = 1;

/// TensorProto: dims = 1, data_type = 2, float_data = 4, name = 8,
/// raw_data = 9, external data location = 13. Returns `None` for non-f32
/// tensors (shape/index constants the engine never reads).
fn parse_tensor(mut r: Reader) -> Result<Option<OnnxTensor>, VoskError> {
    let mut dims: Vec<usize> = Vec::new();
    let mut data_type: u64 = 0;
    let mut name = String::new();
    let mut floats: Vec<f32> = Vec::new();
    let mut raw: Option<Vec<u8>> = None;
    while !r.at_end() {
        let (field, wire) = r.key()?;
        match field {
            1 => match wire {
                0 => dims.push(r.varint()? as usize),
                _ => {
                    // Packed varints.
                    let packed = r.bytes(wire)?;
                    let mut pr = Reader::new(packed, r.label);
                    while !pr.at_end() {
                        dims.push(pr.varint()? as usize);
                    }
                },
            },
            2 => data_type = r.varint()?,
            4 => match wire {
                5 => floats.push(f32::from_bits(r.fixed32()?)),
                _ => {
                    let packed = r.bytes(wire)?;
                    for c in packed.chunks_exact(4) {
                        floats
                            .push(f32::from_le_bytes([c[0], c[1], c[2], c[3]]));
                    }
                },
            },
            8 => name = r.string(wire)?,
            9 => raw = Some(r.bytes(wire)?.to_vec()),
            13 => {
                return Err(r.err("external tensor data is not supported"));
            },
            _ => r.skip(wire)?,
        }
    }
    if data_type != DTYPE_F32 {
        return Ok(None);
    }
    let data = match raw {
        Some(bytes) => {
            if bytes.len() % 4 != 0 {
                return Err(r.err(&format!(
                    "tensor `{name}`: raw data length {} is not a multiple of \
                     4",
                    bytes.len()
                )));
            }
            bytes
                .chunks_exact(4)
                .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
                .collect()
        },
        None => floats,
    };
    let expected: usize = dims.iter().product();
    if data.len() != expected {
        return Err(r.err(&format!(
            "tensor `{name}`: {} values for shape {dims:?}",
            data.len()
        )));
    }
    Ok(Some(OnnxTensor { name, dims, data }))
}

/// StringStringEntryProto: key = 1, value = 2.
fn parse_string_pair(mut r: Reader) -> Result<(String, String), VoskError> {
    let (mut key, mut value) = (String::new(), String::new());
    while !r.at_end() {
        let (field, wire) = r.key()?;
        match field {
            1 => key = r.string(wire)?,
            2 => value = r.string(wire)?,
            _ => r.skip(wire)?,
        }
    }
    Ok((key, value))
}

/// A protobuf wire-format cursor over a byte slice.
struct Reader<'a> {
    buf: &'a [u8],
    pos: usize,
    label: &'a str,
}

impl<'a> Reader<'a> {
    fn new(buf: &'a [u8], label: &'a str) -> Self {
        Self { buf, pos: 0, label }
    }

    fn err(&self, message: &str) -> VoskError {
        VoskError::InvalidModel(format!("{}: {message}", self.label))
    }

    fn at_end(&self) -> bool { self.pos >= self.buf.len() }

    /// Reads a field key, returning `(field_number, wire_type)`.
    fn key(&mut self) -> Result<(u64, u8), VoskError> {
        let key = self.varint()?;
        Ok((key >> 3, (key & 7) as u8))
    }

    fn varint(&mut self) -> Result<u64, VoskError> {
        let mut value: u64 = 0;
        for shift in (0..64).step_by(7) {
            let byte = *self
                .buf
                .get(self.pos)
                .ok_or_else(|| self.err("truncated varint"))?;
            self.pos += 1;
            value |= u64::from(byte & 0x7f) << shift;
            if byte & 0x80 == 0 {
                return Ok(value);
            }
        }
        Err(self.err("varint too long"))
    }

    fn fixed32(&mut self) -> Result<u32, VoskError> {
        let end = self.pos + 4;
        let bytes = self
            .buf
            .get(self.pos..end)
            .ok_or_else(|| self.err("truncated fixed32"))?;
        self.pos = end;
        Ok(u32::from_le_bytes([bytes[0], bytes[1], bytes[2], bytes[3]]))
    }

    /// Reads a length-delimited field's bytes; rejects other wire types.
    fn bytes(&mut self, wire: u8) -> Result<&'a [u8], VoskError> {
        if wire != 2 {
            return Err(
                self.err(&format!("expected length-delimited, got {wire}"))
            );
        }
        let len = self.varint()? as usize;
        let end = self.pos + len;
        let bytes = self
            .buf
            .get(self.pos..end)
            .ok_or_else(|| self.err("truncated field"))?;
        self.pos = end;
        Ok(bytes)
    }

    fn string(&mut self, wire: u8) -> Result<String, VoskError> {
        let bytes = self.bytes(wire)?;
        String::from_utf8(bytes.to_vec())
            .map_err(|_| self.err("invalid utf-8 string"))
    }

    fn skip(&mut self, wire: u8) -> Result<(), VoskError> {
        match wire {
            0 => {
                self.varint()?;
            },
            1 => {
                let end = self.pos + 8;
                if end > self.buf.len() {
                    return Err(self.err("truncated fixed64"));
                }
                self.pos = end;
            },
            2 => {
                self.bytes(2)?;
            },
            5 => {
                self.fixed32()?;
            },
            other => {
                return Err(self.err(&format!("wire type {other}")));
            },
        }
        Ok(())
    }
}
