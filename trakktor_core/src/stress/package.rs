//! A minimal reader for the reference's `torch.package` archive.
//!
//! The published accentor is not a `state_dict` but a `torch.package`: a ZIP
//! holding the reference's own Python sources, two TorchScript modules
//! (`.data/ts_code/{0,1}/data.pkl`), an object pickle with the lookup tables
//! (`accentor_models/accentor`), and the tensors as flat storages
//! (`.data/<n>.storage`). candle's own `.pth` reader assumes a flat state
//! dictionary and a different storage layout, so this module walks the archive
//! itself — reusing candle's pickle machinery ([`Stack`]/[`Object`]) for the
//! opcode level, which is the part worth not rewriting.
//!
//! It is used **once**, when a downloaded model is converted into the layout
//! the runtimes read (`download.rs`); nothing on the hot path touches it.

use std::{collections::HashMap, io::Read, path::Path};

use candle_core::pickle::{Object, Stack};

use super::error::StressError;

/// How a storage's bytes are to be read.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) enum Element {
    /// `torch.FloatStorage` — 4 bytes, little-endian f32.
    F32,
    /// `torch.CharStorage` — one signed byte.
    I8,
    /// `torch.LongStorage` — 8 bytes, little-endian i64.
    I64,
}

impl Element {
    fn size(self) -> usize {
        match self {
            Self::F32 => 4,
            Self::I8 => 1,
            Self::I64 => 8,
        }
    }
}

/// A tensor as the archive describes it: which storage holds it, how to read
/// the bytes, and what shape they form.
#[derive(Debug, Clone)]
pub(super) struct StoredTensor {
    pub element: Element,
    pub shape: Vec<usize>,
    /// Element offset into the storage.
    pub offset: usize,
    /// The storage file name, relative to the archive's `.data/` directory.
    pub storage: String,
}

impl StoredTensor {
    /// Number of elements the shape holds.
    pub fn len(&self) -> usize { self.shape.iter().product() }
}

/// An opened `torch.package` archive.
pub(super) struct Package {
    zip: zip::ZipArchive<std::io::BufReader<std::fs::File>>,
    /// The single top-level directory every entry lives under.
    root: String,
}

/// Wraps a failure to make sense of the archive.
fn broken(path: &Path, detail: impl std::fmt::Display) -> StressError {
    StressError::ModelDownload(format!("{}: {detail}", path.display()))
}

impl Package {
    /// Opens `path` and locates the archive's single root directory.
    pub fn open(path: &Path) -> Result<Self, StressError> {
        let file = std::fs::File::open(path).map_err(|e| broken(path, e))?;
        let zip = zip::ZipArchive::new(std::io::BufReader::new(file))
            .map_err(|e| broken(path, e))?;
        let roots: std::collections::BTreeSet<&str> = zip
            .file_names()
            .filter_map(|name| name.split('/').next())
            .collect();
        let [root] = roots.into_iter().collect::<Vec<_>>()[..] else {
            return Err(broken(
                path,
                "expected a torch.package archive with one root directory",
            ));
        };
        let root = root.to_owned();
        Ok(Self { zip, root })
    }

    /// Reads an entry, named relative to the archive root.
    fn entry(&mut self, name: &str) -> Result<Vec<u8>, StressError> {
        let full = format!("{}/{name}", self.root);
        let mut file = self
            .zip
            .by_name(&full)
            .map_err(|e| StressError::ModelDownload(format!("{full}: {e}")))?;
        let mut bytes = Vec::with_capacity(file.size() as usize);
        file.read_to_end(&mut bytes)
            .map_err(|e| StressError::ModelDownload(format!("{full}: {e}")))?;
        Ok(bytes)
    }

    /// Parses an entry as a pickle.
    pub fn pickle(&mut self, name: &str) -> Result<Object, StressError> {
        let bytes = self.entry(name)?;
        let mut stack = Stack::empty();
        let mut reader = std::io::Cursor::new(bytes);
        stack
            .read_loop(&mut reader)
            .map_err(|e| StressError::ModelDownload(format!("{name}: {e}")))?;
        stack
            .finalize()
            .map_err(|e| StressError::ModelDownload(format!("{name}: {e}")))
    }

    /// The TorchScript modules the archive carries, as entry names relative to
    /// the root, in archive order.
    pub fn script_modules(&self) -> Vec<String> {
        let prefix = format!("{}/.data/ts_code/", self.root);
        let mut names: Vec<String> = self
            .zip
            .file_names()
            .filter(|name| {
                name.starts_with(&prefix) && name.ends_with("/data.pkl")
            })
            .map(|name| name[self.root.len() + 1..].to_owned())
            .collect();
        names.sort();
        names
    }

    /// Reads a tensor's bytes and returns them as f32 values.
    ///
    /// `I8` storages are widened to f32 as signed bytes; the caller applies the
    /// quantization scale.
    pub fn values(
        &mut self,
        tensor: &StoredTensor,
    ) -> Result<Vec<f32>, StressError> {
        let bytes = self.raw(tensor)?;
        Ok(match tensor.element {
            Element::F32 => bytes
                .chunks_exact(4)
                .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
                .collect(),
            Element::I8 => bytes.iter().map(|&b| f32::from(b as i8)).collect(),
            Element::I64 => bytes
                .chunks_exact(8)
                .map(|c| {
                    i64::from_le_bytes(c.try_into().expect("8 bytes")) as f32
                })
                .collect(),
        })
    }

    /// Reads a tensor's bytes verbatim — for a storage kept in its stored form.
    pub fn raw(
        &mut self,
        tensor: &StoredTensor,
    ) -> Result<Vec<u8>, StressError> {
        let size = tensor.element.size();
        let start = tensor.offset * size;
        let wanted = tensor.len() * size;
        let bytes = self.entry(&format!(".data/{}", tensor.storage))?;
        if bytes.len() < start + wanted {
            return Err(StressError::ModelDownload(format!(
                "{}: holds {} bytes, need {}",
                tensor.storage,
                bytes.len(),
                start + wanted
            )));
        }
        Ok(bytes[start..start + wanted].to_vec())
    }
}

/// The attribute list of a pickled module (`NEWOBJ` + `BUILD` over its
/// `__dict__`). Plain dictionaries are deliberately **not** modules: the
/// reference stores its 126 523-entry n-gram table as one, and walking into it
/// as if it were a submodule would be both wrong and slow.
fn module_attrs(object: &Object) -> Option<&Vec<(Object, Object)>> {
    match object {
        Object::Build { args, .. } => match args.as_ref() {
            Object::Dict(items) => Some(items),
            _ => None,
        },
        _ => None,
    }
}

/// Unwraps TorchScript's `restore_type_tag(value, "Dict[str, int]")` wrapper,
/// which it puts around every typed container it serializes.
fn untagged(object: &Object) -> &Object {
    let Object::Reduce { callable, args } = object else {
        return object;
    };
    let Object::Class {
        module_name,
        class_name,
    } = callable.as_ref()
    else {
        return object;
    };
    if module_name != "torch.jit._pickle" || class_name != "restore_type_tag" {
        return object;
    }
    match args.as_ref() {
        Object::Tuple(args) => args.first().unwrap_or(object),
        _ => object,
    }
}

/// Looks a named attribute up in a module or a dictionary.
pub(super) fn field<'a>(object: &'a Object, name: &str) -> Option<&'a Object> {
    let items = match untagged(object) {
        Object::Build { args, .. } => match args.as_ref() {
            Object::Dict(items) => items,
            _ => return None,
        },
        Object::Dict(items) => items,
        _ => return None,
    };
    items.iter().find_map(|(key, value)| match key {
        Object::Unicode(key) if key == name => Some(untagged(value)),
        _ => None,
    })
}

/// Follows a chain of attribute names.
pub(super) fn path<'a>(
    object: &'a Object,
    names: &[&str],
) -> Option<&'a Object> {
    names.iter().try_fold(object, |at, name| field(at, name))
}

/// Every tensor of a pickled module tree, keyed by its dotted attribute path
/// (`bert.encoder.layer.0.attention.self.query.weight`) — the same names
/// PyTorch's own `named_parameters` produces.
pub(super) fn tensors(object: &Object) -> HashMap<String, StoredTensor> {
    let mut out = HashMap::new();
    collect_tensors(object, "", &mut out);
    out
}

/// Walks a module tree, recording every tensor under its dotted path.
fn collect_tensors(
    object: &Object,
    prefix: &str,
    out: &mut HashMap<String, StoredTensor>,
) {
    let Some(items) = module_attrs(object) else {
        return;
    };
    for (key, value) in items {
        let Object::Unicode(name) = key else { continue };
        let path = if prefix.is_empty() {
            name.clone()
        } else {
            format!("{prefix}.{name}")
        };
        if let Some(tensor) = stored_tensor(value) {
            out.insert(path, tensor);
        } else {
            collect_tensors(value, &path, out);
        }
    }
}

/// Recognizes `torch._utils._rebuild_tensor_v2(storage, offset, size, stride,
/// …)` and reads out what it takes to find the bytes.
fn stored_tensor(object: &Object) -> Option<StoredTensor> {
    let Object::Reduce { callable, args } = object else {
        return None;
    };
    match callable.as_ref() {
        Object::Class {
            module_name,
            class_name,
        } if module_name == "torch._utils" &&
            class_name == "_rebuild_tensor_v2" => {},
        _ => return None,
    }
    let Object::Tuple(args) = args.as_ref() else {
        return None;
    };
    let [storage, offset, size, ..] = &args[..] else {
        return None;
    };

    let Object::PersistentLoad(id) = storage else {
        return None;
    };
    let Object::Tuple(id) = id.as_ref() else {
        return None;
    };
    let [_, class, file, ..] = &id[..] else {
        return None;
    };
    let Object::Class { class_name, .. } = class else {
        return None;
    };
    let element = match class_name.as_str() {
        "FloatStorage" => Element::F32,
        "CharStorage" => Element::I8,
        "LongStorage" => Element::I64,
        _ => return None,
    };
    let Object::Unicode(file) = file else {
        return None;
    };

    let offset = match offset {
        Object::Int(value) if *value >= 0 => *value as usize,
        Object::Long(value) if *value >= 0 => *value as usize,
        _ => return None,
    };
    let Object::Tuple(size) = size else {
        return None;
    };
    let shape = size
        .iter()
        .map(|dim| match dim {
            Object::Int(value) if *value >= 0 => Some(*value as usize),
            Object::Long(value) if *value >= 0 => Some(*value as usize),
            _ => None,
        })
        .collect::<Option<Vec<usize>>>()?;

    Some(StoredTensor {
        element,
        shape,
        offset,
        storage: file.clone(),
    })
}

/// Reads a pickled `dict[str, int]`.
pub(super) fn string_int_map(object: &Object) -> Option<Vec<(String, i64)>> {
    let Object::Dict(items) = untagged(object) else {
        return None;
    };
    items
        .iter()
        .map(|(key, value)| {
            let Object::Unicode(key) = key else {
                return None;
            };
            let value = match value {
                Object::Int(value) => i64::from(*value),
                Object::Long(value) => *value,
                _ => return None,
            };
            Some((key.clone(), value))
        })
        .collect()
}

/// Reads a pickled `dict[str, tuple[int, int]]` — the exception table.
pub(super) fn string_pair_map(
    object: &Object,
) -> Option<Vec<(String, i64, i64)>> {
    let Object::Dict(items) = untagged(object) else {
        return None;
    };
    items
        .iter()
        .map(|(key, value)| {
            let Object::Unicode(key) = key else {
                return None;
            };
            let Object::Tuple(pair) = value else {
                return None;
            };
            let [first, second] = &pair[..] else {
                return None;
            };
            let number = |object: &Object| match object {
                Object::Int(value) => Some(i64::from(*value)),
                Object::Long(value) => Some(*value),
                _ => None,
            };
            Some((key.clone(), number(first)?, number(second)?))
        })
        .collect()
}

/// Reads a pickled `dict[str, list[str]]` — the homograph table.
pub(super) fn string_list_map(
    object: &Object,
) -> Option<Vec<(String, Vec<String>)>> {
    let Object::Dict(items) = untagged(object) else {
        return None;
    };
    items
        .iter()
        .map(|(key, value)| {
            let Object::Unicode(key) = key else {
                return None;
            };
            let Object::List(values) = untagged(value) else {
                return None;
            };
            let values = values
                .iter()
                .map(|value| match value {
                    Object::Unicode(value) => Some(value.clone()),
                    _ => None,
                })
                .collect::<Option<Vec<String>>>()?;
            Some((key.clone(), values))
        })
        .collect()
}

/// Reads a scalar tensor's single f32 value — the quantization constants are
/// stored as zero-dimensional tensors.
pub(super) fn scalar(
    package: &mut Package,
    tensors: &HashMap<String, StoredTensor>,
    key: &str,
) -> Result<f32, StressError> {
    let tensor = tensors.get(key).ok_or_else(|| {
        StressError::ModelDownload(format!("{key}: not in the archive"))
    })?;
    let values = package.values(tensor)?;
    values.first().copied().ok_or_else(|| {
        StressError::ModelDownload(format!("{key}: empty scalar"))
    })
}
