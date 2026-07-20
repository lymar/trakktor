use super::OnnxModel;

/// Encodes a varint.
fn varint(mut v: u64, out: &mut Vec<u8>) {
    loop {
        let byte = (v & 0x7f) as u8;
        v >>= 7;
        if v == 0 {
            out.push(byte);
            break;
        }
        out.push(byte | 0x80);
    }
}

/// Encodes a length-delimited field.
fn field_bytes(field: u64, payload: &[u8], out: &mut Vec<u8>) {
    varint(field << 3 | 2, out);
    varint(payload.len() as u64, out);
    out.extend_from_slice(payload);
}

/// Encodes a varint field.
fn field_varint(field: u64, value: u64, out: &mut Vec<u8>) {
    varint(field << 3, out);
    varint(value, out);
}

fn tensor_f32(name: &str, dims: &[u64], data: &[f32]) -> Vec<u8> {
    let mut t = Vec::new();
    for &d in dims {
        field_varint(1, d, &mut t);
    }
    field_varint(2, 1, &mut t); // data_type = FLOAT
    field_bytes(8, name.as_bytes(), &mut t);
    let raw: Vec<u8> = data.iter().flat_map(|f| f.to_le_bytes()).collect();
    field_bytes(9, &raw, &mut t);
    t
}

fn tensor_i64(name: &str) -> Vec<u8> {
    let mut t = Vec::new();
    field_varint(1, 2, &mut t);
    field_varint(2, 7, &mut t); // data_type = INT64
    field_bytes(8, name.as_bytes(), &mut t);
    field_bytes(9, &1i64.to_le_bytes(), &mut t);
    t
}

#[test]
fn parses_synthetic_model() {
    let mut node = Vec::new();
    field_bytes(1, b"x", &mut node);
    field_bytes(1, b"w", &mut node);
    field_bytes(2, b"y", &mut node);
    field_bytes(4, b"MatMul", &mut node);

    let mut graph = Vec::new();
    field_bytes(1, &node, &mut graph);
    field_bytes(
        5,
        &tensor_f32("w", &[2, 3], &[1., 2., 3., 4., 5., 6.]),
        &mut graph,
    );
    field_bytes(5, &tensor_i64("starts"), &mut graph);

    let mut meta = Vec::new();
    field_bytes(1, b"model_type", &mut meta);
    field_bytes(2, b"zipformer2", &mut meta);

    let mut model = Vec::new();
    field_varint(1, 8, &mut model); // ir_version, skipped
    field_bytes(7, &graph, &mut model);
    field_bytes(14, &meta, &mut model);

    let parsed = OnnxModel::parse(&model, "test").unwrap();
    assert_eq!(parsed.metadata("model_type"), Some("zipformer2"));
    assert_eq!(parsed.nodes.len(), 1);
    assert_eq!(parsed.nodes[0].op_type, "MatMul");
    assert_eq!(parsed.nodes[0].inputs, ["x", "w"]);
    assert_eq!(parsed.nodes[0].outputs, ["y"]);
    // The int64 tensor is dropped; the f32 one is kept with its shape.
    assert_eq!(parsed.initializers.len(), 1);
    assert_eq!(parsed.initializers[0].name, "w");
    assert_eq!(parsed.initializers[0].dims, [2, 3]);
    assert_eq!(parsed.initializers[0].data, [1., 2., 3., 4., 5., 6.]);
}

#[test]
fn rejects_truncated() {
    let mut model = Vec::new();
    field_bytes(7, &[0x0a, 0xff], &mut model); // graph with truncated field
    assert!(OnnxModel::parse(&model, "test").is_err());
}
