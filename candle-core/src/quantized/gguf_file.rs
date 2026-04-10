//! Support for the [GGUF file format](https://github.com/philpax/ggml/blob/gguf-spec/docs/gguf.md).
//!
//! Spec: https://github.com/ggml-org/ggml/blob/master/docs/gguf.md

use super::{GgmlDType, QTensor};
use crate::{Context, Device, Result};
use byteorder::{LittleEndian, ReadBytesExt, WriteBytesExt};
use std::collections::HashMap;
use std::fs::File;
use std::io::{Read, Seek, SeekFrom};
use std::path::{Path, PathBuf};

pub const DEFAULT_ALIGNMENT: u64 = 32;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Magic {
    Gguf,
}

impl TryFrom<u32> for Magic {
    type Error = crate::Error;
    fn try_from(value: u32) -> Result<Self> {
        let magic = match value {
            0x46554747 | 0x47475546 => Self::Gguf,
            _ => crate::bail!("unknown magic 0x{value:08x}"),
        };
        Ok(magic)
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum VersionedMagic {
    GgufV1,
    GgufV2,
    GgufV3,
}

impl VersionedMagic {
    fn read<R: std::io::Read>(reader: &mut R) -> Result<Self> {
        let magic = reader.read_u32::<LittleEndian>()?;
        let magic = Magic::try_from(magic)?;
        let version = reader.read_u32::<LittleEndian>()?;
        let versioned_magic = match (magic, version) {
            (Magic::Gguf, 1) => Self::GgufV1,
            (Magic::Gguf, 2) => Self::GgufV2,
            (Magic::Gguf, 3) => Self::GgufV3,
            _ => crate::bail!("gguf: unsupported magic/version {magic:?}/{version}"),
        };
        Ok(versioned_magic)
    }
}

#[derive(Debug)]
pub struct TensorInfo {
    pub ggml_dtype: GgmlDType,
    pub shape: crate::Shape,
    pub offset: u64,
}

impl TensorInfo {
    pub fn read<R: std::io::Seek + std::io::Read>(
        &self,
        reader: &mut R,
        tensor_data_offset: u64,
        device: &Device,
    ) -> Result<QTensor> {
        let tensor_elems = self.shape.elem_count();
        let block_size = self.ggml_dtype.block_size();
        if !tensor_elems.is_multiple_of(block_size) {
            crate::bail!(
            "the number of elements {tensor_elems} is not divisible by the block size {block_size}"
        )
        }
        let size_in_bytes = tensor_elems / block_size * self.ggml_dtype.type_size();
        let mut raw_data = vec![0u8; size_in_bytes];
        reader.seek(std::io::SeekFrom::Start(tensor_data_offset + self.offset))?;
        reader.read_exact(&mut raw_data)?;
        super::ggml_file::qtensor_from_ggml(
            self.ggml_dtype,
            &raw_data,
            self.shape.dims().to_vec(),
            device,
        )
    }
}

#[derive(Debug)]
pub struct Content {
    pub magic: VersionedMagic,
    pub metadata: HashMap<String, Value>,
    pub tensor_infos: HashMap<String, TensorInfo>,
    pub tensor_data_offset: u64,
}

fn read_string<R: std::io::Read>(reader: &mut R, magic: &VersionedMagic) -> Result<String> {
    let len = match magic {
        VersionedMagic::GgufV1 => reader.read_u32::<LittleEndian>()? as usize,
        VersionedMagic::GgufV2 | VersionedMagic::GgufV3 => {
            reader.read_u64::<LittleEndian>()? as usize
        }
    };
    let mut v = vec![0u8; len];
    reader.read_exact(&mut v)?;
    // GGUF strings are supposed to be non-null terminated but in practice this happens.
    while let Some(0) = v.last() {
        v.pop();
    }
    // GGUF strings are utf8 encoded but there are cases that don't seem to be valid.
    Ok(String::from_utf8_lossy(&v).into_owned())
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum ValueType {
    // The value is a 8-bit unsigned integer.
    U8,
    // The value is a 8-bit signed integer.
    I8,
    // The value is a 16-bit unsigned little-endian integer.
    U16,
    // The value is a 16-bit signed little-endian integer.
    I16,
    // The value is a 32-bit unsigned little-endian integer.
    U32,
    // The value is a 32-bit signed little-endian integer.
    I32,
    // The value is a 64-bit unsigned little-endian integer.
    U64,
    // The value is a 64-bit signed little-endian integer.
    I64,
    // The value is a 32-bit IEEE754 floating point number.
    F32,
    // The value is a 64-bit IEEE754 floating point number.
    F64,
    // The value is a boolean.
    // 1-byte value where 0 is false and 1 is true.
    // Anything else is invalid, and should be treated as either the model being invalid or the reader being buggy.
    Bool,
    // The value is a UTF-8 non-null-terminated string, with length prepended.
    String,
    // The value is an array of other values, with the length and type prepended.
    // Arrays can be nested, and the length of the array is the number of elements in the array, not the number of bytes.
    Array,
}

#[derive(Debug, Clone)]
pub enum Value {
    U8(u8),
    I8(i8),
    U16(u16),
    I16(i16),
    U32(u32),
    I32(i32),
    U64(u64),
    I64(i64),
    F32(f32),
    F64(f64),
    Bool(bool),
    String(String),
    Array(Vec<Value>),
}

impl Value {
    pub fn value_type(&self) -> ValueType {
        match self {
            Self::U8(_) => ValueType::U8,
            Self::I8(_) => ValueType::I8,
            Self::U16(_) => ValueType::U16,
            Self::I16(_) => ValueType::I16,
            Self::U32(_) => ValueType::U32,
            Self::I32(_) => ValueType::I32,
            Self::U64(_) => ValueType::U64,
            Self::I64(_) => ValueType::I64,
            Self::F32(_) => ValueType::F32,
            Self::F64(_) => ValueType::F64,
            Self::Bool(_) => ValueType::Bool,
            Self::String(_) => ValueType::String,
            Self::Array(_) => ValueType::Array,
        }
    }

    pub fn to_u8(&self) -> Result<u8> {
        match self {
            Self::U8(v) => Ok(*v),
            v => crate::bail!("not a u8 {v:?}"),
        }
    }

    pub fn to_i8(&self) -> Result<i8> {
        match self {
            Self::I8(v) => Ok(*v),
            v => crate::bail!("not a i8 {v:?}"),
        }
    }

    pub fn to_u16(&self) -> Result<u16> {
        match self {
            Self::U16(v) => Ok(*v),
            v => crate::bail!("not a u16 {v:?}"),
        }
    }

    pub fn to_i16(&self) -> Result<i16> {
        match self {
            Self::I16(v) => Ok(*v),
            v => crate::bail!("not a i16 {v:?}"),
        }
    }

    pub fn to_u32(&self) -> Result<u32> {
        match self {
            Self::U32(v) => Ok(*v),
            v => crate::bail!("not a u32 {v:?}"),
        }
    }

    pub fn to_i32(&self) -> Result<i32> {
        match self {
            Self::I32(v) => Ok(*v),
            v => crate::bail!("not a i32 {v:?}"),
        }
    }

    /// This will also automatically upcast any integral types which will not truncate.
    pub fn to_u64(&self) -> Result<u64> {
        match self {
            Self::U64(v) => Ok(*v),
            // Autoupcast cases here
            Self::U8(v) => Ok(*v as u64),
            Self::U16(v) => Ok(*v as u64),
            Self::U32(v) => Ok(*v as u64),
            Self::Bool(v) => Ok(*v as u64),
            v => crate::bail!("not a u64 or upcastable to u64 {v:?}"),
        }
    }

    pub fn to_i64(&self) -> Result<i64> {
        match self {
            Self::I64(v) => Ok(*v),
            v => crate::bail!("not a i64 {v:?}"),
        }
    }

    pub fn to_f32(&self) -> Result<f32> {
        match self {
            Self::F32(v) => Ok(*v),
            v => crate::bail!("not a f32 {v:?}"),
        }
    }

    pub fn to_f64(&self) -> Result<f64> {
        match self {
            Self::F64(v) => Ok(*v),
            v => crate::bail!("not a f64 {v:?}"),
        }
    }

    pub fn to_bool(&self) -> Result<bool> {
        match self {
            Self::Bool(v) => Ok(*v),
            v => crate::bail!("not a bool {v:?}"),
        }
    }

    pub fn to_vec(&self) -> Result<&Vec<Value>> {
        match self {
            Self::Array(v) => Ok(v),
            v => crate::bail!("not a vec {v:?}"),
        }
    }

    pub fn to_string(&self) -> Result<&String> {
        match self {
            Self::String(v) => Ok(v),
            v => crate::bail!("not a string {v:?}"),
        }
    }

    fn read<R: std::io::Read>(
        reader: &mut R,
        value_type: ValueType,
        magic: &VersionedMagic,
    ) -> Result<Self> {
        let v = match value_type {
            ValueType::U8 => Self::U8(reader.read_u8()?),
            ValueType::I8 => Self::I8(reader.read_i8()?),
            ValueType::U16 => Self::U16(reader.read_u16::<LittleEndian>()?),
            ValueType::I16 => Self::I16(reader.read_i16::<LittleEndian>()?),
            ValueType::U32 => Self::U32(reader.read_u32::<LittleEndian>()?),
            ValueType::I32 => Self::I32(reader.read_i32::<LittleEndian>()?),
            ValueType::U64 => Self::U64(reader.read_u64::<LittleEndian>()?),
            ValueType::I64 => Self::I64(reader.read_i64::<LittleEndian>()?),
            ValueType::F32 => Self::F32(reader.read_f32::<LittleEndian>()?),
            ValueType::F64 => Self::F64(reader.read_f64::<LittleEndian>()?),
            ValueType::Bool => match reader.read_u8()? {
                0 => Self::Bool(false),
                1 => Self::Bool(true),
                b => crate::bail!("unexpected bool value {b}"),
            },
            ValueType::String => Self::String(read_string(reader, magic)?),
            ValueType::Array => {
                let value_type = reader.read_u32::<LittleEndian>()?;
                let value_type = ValueType::from_u32(value_type)?;
                let len = match magic {
                    VersionedMagic::GgufV1 => reader.read_u32::<LittleEndian>()? as usize,
                    VersionedMagic::GgufV2 | VersionedMagic::GgufV3 => {
                        reader.read_u64::<LittleEndian>()? as usize
                    }
                };
                let mut vs = Vec::with_capacity(len);
                for _ in 0..len {
                    vs.push(Value::read(reader, value_type, magic)?)
                }
                Self::Array(vs)
            }
        };
        Ok(v)
    }

    fn write<W: std::io::Write>(&self, w: &mut W) -> Result<()> {
        match self {
            &Self::U8(v) => w.write_u8(v)?,
            &Self::I8(v) => w.write_i8(v)?,
            &Self::U16(v) => w.write_u16::<LittleEndian>(v)?,
            &Self::I16(v) => w.write_i16::<LittleEndian>(v)?,
            &Self::U32(v) => w.write_u32::<LittleEndian>(v)?,
            &Self::I32(v) => w.write_i32::<LittleEndian>(v)?,
            &Self::U64(v) => w.write_u64::<LittleEndian>(v)?,
            &Self::I64(v) => w.write_i64::<LittleEndian>(v)?,
            &Self::F32(v) => w.write_f32::<LittleEndian>(v)?,
            &Self::F64(v) => w.write_f64::<LittleEndian>(v)?,
            &Self::Bool(v) => w.write_u8(u8::from(v))?,
            Self::String(v) => write_string(w, v.as_str())?,
            Self::Array(v) => {
                // The `Value` type does not enforce that all the values in an Array have the same
                // type.
                let value_type = if v.is_empty() {
                    // Doesn't matter, the array is empty.
                    ValueType::U32
                } else {
                    let value_type: std::collections::HashSet<_> =
                        v.iter().map(|elem| elem.value_type()).collect();
                    if value_type.len() != 1 {
                        crate::bail!("multiple value-types in the same array {value_type:?}")
                    }
                    value_type.into_iter().next().context("empty value_type")?
                };
                w.write_u32::<LittleEndian>(value_type.to_u32())?;
                w.write_u64::<LittleEndian>(v.len() as u64)?;
                for elem in v.iter() {
                    elem.write(w)?
                }
            }
        }
        Ok(())
    }
}

impl ValueType {
    fn from_u32(v: u32) -> Result<Self> {
        let v = match v {
            0 => Self::U8,
            1 => Self::I8,
            2 => Self::U16,
            3 => Self::I16,
            4 => Self::U32,
            5 => Self::I32,
            6 => Self::F32,
            7 => Self::Bool,
            8 => Self::String,
            9 => Self::Array,
            10 => Self::U64,
            11 => Self::I64,
            12 => Self::F64,
            v => crate::bail!("unrecognized value-type {v:#08x}"),
        };
        Ok(v)
    }

    fn to_u32(self) -> u32 {
        match self {
            Self::U8 => 0,
            Self::I8 => 1,
            Self::U16 => 2,
            Self::I16 => 3,
            Self::U32 => 4,
            Self::I32 => 5,
            Self::F32 => 6,
            Self::Bool => 7,
            Self::String => 8,
            Self::Array => 9,
            Self::U64 => 10,
            Self::I64 => 11,
            Self::F64 => 12,
        }
    }
}

impl Content {
    pub fn read<R: std::io::Seek + std::io::Read>(reader: &mut R) -> Result<Self> {
        let magic = VersionedMagic::read(reader)?;

        let tensor_count = match magic {
            VersionedMagic::GgufV1 => reader.read_u32::<LittleEndian>()? as usize,
            VersionedMagic::GgufV2 | VersionedMagic::GgufV3 => {
                reader.read_u64::<LittleEndian>()? as usize
            }
        };
        let metadata_kv_count = match magic {
            VersionedMagic::GgufV1 => reader.read_u32::<LittleEndian>()? as usize,
            VersionedMagic::GgufV2 | VersionedMagic::GgufV3 => {
                reader.read_u64::<LittleEndian>()? as usize
            }
        };

        let mut metadata = HashMap::new();
        for _idx in 0..metadata_kv_count {
            let key = read_string(reader, &magic)?;
            let value_type = reader.read_u32::<LittleEndian>()?;
            let value_type = ValueType::from_u32(value_type)?;
            let value = Value::read(reader, value_type, &magic)?;
            metadata.insert(key, value);
        }
        let mut tensor_infos = HashMap::new();
        for _idx in 0..tensor_count {
            let tensor_name = read_string(reader, &magic)?;
            let n_dimensions = reader.read_u32::<LittleEndian>()?;

            let mut dimensions: Vec<usize> = match magic {
                VersionedMagic::GgufV1 => {
                    let mut dimensions = vec![0; n_dimensions as usize];
                    reader.read_u32_into::<LittleEndian>(&mut dimensions)?;
                    dimensions.into_iter().map(|c| c as usize).collect()
                }
                VersionedMagic::GgufV2 | VersionedMagic::GgufV3 => {
                    let mut dimensions = vec![0; n_dimensions as usize];
                    reader.read_u64_into::<LittleEndian>(&mut dimensions)?;
                    dimensions.into_iter().map(|c| c as usize).collect()
                }
            };

            dimensions.reverse();
            let ggml_dtype = reader.read_u32::<LittleEndian>()?;
            let ggml_dtype = GgmlDType::from_u32(ggml_dtype)?;
            let offset = reader.read_u64::<LittleEndian>()?;
            tensor_infos.insert(
                tensor_name,
                TensorInfo {
                    shape: crate::Shape::from(dimensions),
                    offset,
                    ggml_dtype,
                },
            );
        }
        let position = reader.stream_position()?;
        let alignment = match metadata.get("general.alignment") {
            Some(Value::U8(v)) => *v as u64,
            Some(Value::U16(v)) => *v as u64,
            Some(Value::U32(v)) => *v as u64,
            Some(Value::I8(v)) if *v >= 0 => *v as u64,
            Some(Value::I16(v)) if *v >= 0 => *v as u64,
            Some(Value::I32(v)) if *v >= 0 => *v as u64,
            _ => DEFAULT_ALIGNMENT,
        };
        let tensor_data_offset = position.div_ceil(alignment) * alignment;
        Ok(Self {
            magic,
            metadata,
            tensor_infos,
            tensor_data_offset,
        })
    }

    pub fn tensor<R: std::io::Seek + std::io::Read>(
        &self,
        reader: &mut R,
        name: &str,
        device: &Device,
    ) -> Result<QTensor> {
        let tensor_info = match self.tensor_infos.get(name) {
            Some(tensor_info) => tensor_info,
            None => crate::bail!("cannot find tensor info for {name}"),
        };
        tensor_info.read(reader, self.tensor_data_offset, device)
    }

    /// Open a possibly multi-file GGUF given the path to the *first* part.
    ///
    /// Detects the `<base>-<NNNNN>-of-<MMMMM>.gguf` split pattern. If the path
    /// matches, all sibling parts are opened, their metadata + tensor infos are
    /// merged into a single `Content`, and a [`MultiFileReader`] that virtually
    /// concatenates them is returned. Tensor offsets in the merged `Content`
    /// point into the virtual address space, with `tensor_data_offset = 0`, so
    /// downstream loaders can treat the result like a normal single-file GGUF.
    ///
    /// If the path is not split, this falls back to a regular [`Content::read`]
    /// over the single file.
    ///
    /// Layout (per llama.cpp's `gguf-split`):
    /// - Part 0 holds the full model metadata (no `general.*` keys in others).
    ///   Each part has `split.no`, `split.count`, `split.tensors.count`.
    /// - Each part is a complete, self-contained GGUF whose tensor offsets are
    ///   relative to *that part's* `tensor_data_offset`.
    /// - The merged tensor count must equal `split.tensors.count`.
    pub fn read_split_files(first_path: impl AsRef<Path>) -> Result<(Self, MultiFileReader)> {
        let first_path = first_path.as_ref();
        let parts = match split_sibling_paths(first_path)? {
            Some(parts) => parts,
            None => {
                // Single-file GGUF — wrap a single File in a MultiFileReader so
                // callers get a uniform reader type.
                let file = File::open(first_path)
                    .map_err(|e| crate::Error::Io(e).with_path(first_path))?;
                let mut reader = MultiFileReader::new(vec![file])?;
                let ct = Content::read(&mut reader)?;
                return Ok((ct, reader));
            }
        };

        // Open every part and read its header.
        let mut files: Vec<File> = Vec::with_capacity(parts.len());
        let mut headers: Vec<Content> = Vec::with_capacity(parts.len());
        for p in &parts {
            let mut f = File::open(p).map_err(|e| crate::Error::Io(e).with_path(p))?;
            let header = Content::read(&mut f)?;
            files.push(f);
            headers.push(header);
        }

        // Sort by `split.no` so files are stitched in the correct order even
        // if the directory listing returned them out of order.
        let mut order: Vec<usize> = (0..files.len()).collect();
        order.sort_by_key(|&i| {
            headers[i]
                .metadata
                .get("split.no")
                .and_then(|v| v.to_u64().ok())
                .unwrap_or(i as u64)
        });

        // Validate split.count and total tensor count.
        let expected_count = headers
            .first()
            .and_then(|h| h.metadata.get("split.count"))
            .and_then(|v| v.to_u64().ok())
            .unwrap_or(parts.len() as u64) as usize;
        if expected_count != parts.len() {
            crate::bail!(
                "split.count = {} but found {} files on disk",
                expected_count,
                parts.len()
            );
        }
        let expected_tensors = headers
            .first()
            .and_then(|h| h.metadata.get("split.tensors.count"))
            .and_then(|v| v.to_u64().ok())
            .unwrap_or(0) as usize;

        // Reorder files + headers, compute virtual file boundaries.
        let mut sorted_files: Vec<File> = Vec::with_capacity(order.len());
        let mut sorted_headers: Vec<Content> = Vec::with_capacity(order.len());
        for &i in &order {
            // Move ownership: take from the original Vecs in `order` order.
            // We use `mem::take` because we can't move out of an indexed Vec.
            sorted_files.push(files[i].try_clone().map_err(crate::Error::Io)?);
            // Re-read headers from the cloned file: cheaper to just clone the
            // already-parsed Content via field-by-field copy. Build a new one.
            let h = &headers[i];
            sorted_headers.push(Content {
                magic: h.magic,
                metadata: h.metadata.clone(),
                tensor_infos: h
                    .tensor_infos
                    .iter()
                    .map(|(k, v)| {
                        (
                            k.clone(),
                            TensorInfo {
                                ggml_dtype: v.ggml_dtype,
                                shape: v.shape.clone(),
                                offset: v.offset,
                            },
                        )
                    })
                    .collect(),
                tensor_data_offset: h.tensor_data_offset,
            });
        }

        // Build the virtual address space by stacking files end-to-end.
        let reader = MultiFileReader::new(sorted_files)?;

        // Merge metadata from part 0 (it's the only one that has the full
        // model config). Subsequent parts only have split keys.
        let mut metadata = sorted_headers[0].metadata.clone();
        // Drop split keys — they describe storage layout, not the model.
        metadata.remove("split.no");
        metadata.remove("split.count");
        metadata.remove("split.tensors.count");

        // Walk every part's tensor_infos and rebase their offsets into the
        // virtual address space. The new offset is:
        //   virt_offset = file_virtual_start + part.tensor_data_offset + local_offset
        let mut tensor_infos: HashMap<String, TensorInfo> = HashMap::new();
        for (idx, header) in sorted_headers.iter().enumerate() {
            let file_start = reader.file_starts[idx];
            for (name, info) in header.tensor_infos.iter() {
                let new_offset = file_start + header.tensor_data_offset + info.offset;
                tensor_infos.insert(
                    name.clone(),
                    TensorInfo {
                        ggml_dtype: info.ggml_dtype,
                        shape: info.shape.clone(),
                        offset: new_offset,
                    },
                );
            }
        }
        if expected_tensors != 0 && tensor_infos.len() != expected_tensors {
            crate::bail!(
                "split.tensors.count = {} but only {} unique tensor infos found across {} parts",
                expected_tensors,
                tensor_infos.len(),
                parts.len()
            );
        }

        let magic = sorted_headers[0].magic;
        let merged = Content {
            magic,
            metadata,
            tensor_infos,
            // We baked the per-file tensor_data_offsets into each tensor's
            // offset above, so the merged Content reads from offset 0.
            tensor_data_offset: 0,
        };
        Ok((merged, reader))
    }
}

// ---------------------------------------------------------------------------
// Split file discovery
// ---------------------------------------------------------------------------

/// If `path` matches the `<base>-<NNNNN>-of-<MMMMM>.gguf` pattern, returns
/// the ordered list of all sibling parts (including `path` itself). Returns
/// `None` if the filename does not look like a split. Errors if any sibling
/// referenced by the `-of-<MMMMM>` count is missing.
fn split_sibling_paths(path: &Path) -> Result<Option<Vec<PathBuf>>> {
    let file_name = match path.file_name().and_then(|s| s.to_str()) {
        Some(s) => s,
        None => return Ok(None),
    };
    // Strip the .gguf extension if present.
    let stem = match file_name.strip_suffix(".gguf") {
        Some(s) => s,
        None => return Ok(None),
    };
    // Look for the trailing `-NNNNN-of-MMMMM` segment.
    // Example: "Foo-Q4_1-00001-of-00003".
    let dash_idx = match stem.rfind("-of-") {
        Some(i) => i,
        None => return Ok(None),
    };
    let (head, tail) = stem.split_at(dash_idx);
    let total_str = &tail["-of-".len()..];
    let total: usize = match total_str.parse() {
        Ok(n) if n > 0 => n,
        _ => return Ok(None),
    };
    // Now `head` should end with "-NNNNN".
    let part_dash_idx = match head.rfind('-') {
        Some(i) => i,
        None => return Ok(None),
    };
    let (base, part_str) = head.split_at(part_dash_idx);
    let part_str = &part_str[1..]; // skip leading '-'
    if part_str.parse::<usize>().is_err() {
        return Ok(None);
    }
    let width = part_str.len(); // 5 in practice but accept any width
    let parent = path.parent().unwrap_or_else(|| Path::new(""));
    let mut paths = Vec::with_capacity(total);
    for i in 1..=total {
        let name = format!("{base}-{i:0width$}-of-{total:0width$}.gguf");
        let p = parent.join(name);
        if !p.exists() {
            crate::bail!("missing split GGUF part: {}", p.display());
        }
        paths.push(p);
    }
    Ok(Some(paths))
}

// ---------------------------------------------------------------------------
// MultiFileReader: virtual concatenation of N GGUF parts
// ---------------------------------------------------------------------------

/// `Read + Seek` adapter that virtually concatenates multiple files into one
/// flat address space. Used by [`Content::read_split_files`] to expose a
/// multi-file GGUF as if it were a single contiguous file.
///
/// Coordinates:
/// - `file_starts[i]` is the cumulative virtual byte offset where file `i`
///   begins. `file_starts[0]` is always 0; `file_starts[N]` is the total size.
/// - `pos` is the current virtual cursor position.
///
/// Reads that straddle a file boundary are correctly split across the two
/// files. Single-file callers see no overhead — the only indirection is one
/// indexed lookup per `seek`/`read`.
pub struct MultiFileReader {
    files: Vec<File>,
    /// Cumulative starts: `file_starts[i]` is the virtual position of the
    /// first byte of file `i`. Length = `files.len() + 1`, with the last
    /// element being the total virtual size.
    pub file_starts: Vec<u64>,
    pos: u64,
}

impl MultiFileReader {
    pub fn new(files: Vec<File>) -> Result<Self> {
        let mut file_starts = Vec::with_capacity(files.len() + 1);
        let mut acc: u64 = 0;
        file_starts.push(0u64);
        for f in &files {
            let len = f
                .metadata()
                .map_err(crate::Error::Io)?
                .len();
            acc = acc.checked_add(len).ok_or_else(|| {
                crate::Error::Msg("MultiFileReader: cumulative file size overflow".into())
            })?;
            file_starts.push(acc);
        }
        Ok(Self {
            files,
            file_starts,
            pos: 0,
        })
    }

    /// Total virtual size in bytes.
    pub fn total_size(&self) -> u64 {
        *self.file_starts.last().unwrap_or(&0)
    }

    /// Find the index of the file containing virtual position `pos`. Returns
    /// `files.len()` when `pos` is at or past the end.
    fn locate(&self, pos: u64) -> usize {
        // Linear scan is fine — split GGUFs rarely exceed a handful of parts.
        for i in 0..self.files.len() {
            if pos < self.file_starts[i + 1] {
                return i;
            }
        }
        self.files.len()
    }
}

impl Read for MultiFileReader {
    fn read(&mut self, buf: &mut [u8]) -> std::io::Result<usize> {
        if buf.is_empty() {
            return Ok(0);
        }
        let total = self.total_size();
        if self.pos >= total {
            return Ok(0);
        }
        let mut written = 0usize;
        while written < buf.len() && self.pos < total {
            let idx = self.locate(self.pos);
            if idx >= self.files.len() {
                break;
            }
            let local_pos = self.pos - self.file_starts[idx];
            let file_end = self.file_starts[idx + 1];
            // Bytes available in the current file from `pos`.
            let avail = (file_end - self.pos) as usize;
            let want = (buf.len() - written).min(avail);
            // Seek the underlying file and read.
            self.files[idx].seek(SeekFrom::Start(local_pos))?;
            let n = self.files[idx].read(&mut buf[written..written + want])?;
            if n == 0 {
                break;
            }
            written += n;
            self.pos += n as u64;
        }
        Ok(written)
    }
}

impl Seek for MultiFileReader {
    fn seek(&mut self, pos: SeekFrom) -> std::io::Result<u64> {
        let total = self.total_size();
        let new_pos: i128 = match pos {
            SeekFrom::Start(p) => p as i128,
            SeekFrom::Current(d) => self.pos as i128 + d as i128,
            SeekFrom::End(d) => total as i128 + d as i128,
        };
        if new_pos < 0 {
            return Err(std::io::Error::new(
                std::io::ErrorKind::InvalidInput,
                "MultiFileReader: seek before start",
            ));
        }
        self.pos = new_pos as u64;
        Ok(self.pos)
    }
}

fn write_string<W: std::io::Write>(w: &mut W, str: &str) -> Result<()> {
    let bytes = str.as_bytes();
    w.write_u64::<LittleEndian>(bytes.len() as u64)?;
    w.write_all(bytes)?;
    Ok(())
}

pub fn write<W: std::io::Seek + std::io::Write>(
    w: &mut W,
    metadata: &[(&str, &Value)],
    tensors: &[(&str, &QTensor)],
) -> Result<()> {
    w.write_u32::<LittleEndian>(0x46554747)?;
    w.write_u32::<LittleEndian>(2)?; // version 2.
    w.write_u64::<LittleEndian>(tensors.len() as u64)?;
    w.write_u64::<LittleEndian>(metadata.len() as u64)?;
    for (name, value) in metadata.iter() {
        write_string(w, name)?;
        w.write_u32::<LittleEndian>(value.value_type().to_u32())?;
        value.write(w)?;
    }
    let mut offset = 0usize;
    let mut offsets = Vec::with_capacity(tensors.len());
    for (name, tensor) in tensors.iter() {
        write_string(w, name)?;
        let dims = tensor.shape().dims();
        w.write_u32::<LittleEndian>(dims.len() as u32)?;
        for &dim in dims.iter().rev() {
            w.write_u64::<LittleEndian>(dim as u64)?;
        }
        w.write_u32::<LittleEndian>(tensor.dtype().to_u32())?;
        w.write_u64::<LittleEndian>(offset as u64)?;
        offsets.push(offset);
        let size_in_bytes = tensor.storage_size_in_bytes();
        let padding = 31 - (31 + size_in_bytes) % 32;
        offset += size_in_bytes + padding;
    }
    let pos = w.stream_position()? as usize;
    let padding = 31 - (31 + pos) % 32;
    w.write_all(&vec![0u8; padding])?;
    let tensor_start_pos = w.stream_position()? as usize;
    for (offset, (_name, tensor)) in offsets.iter().zip(tensors.iter()) {
        let pos = w.stream_position()? as usize;
        if tensor_start_pos + offset != pos {
            crate::bail!(
                "internal error, unexpected current position {tensor_start_pos} {offset} {pos}"
            )
        }
        let data = tensor.data()?;
        let size_in_bytes = data.len();
        w.write_all(&data)?;
        let padding = 31 - (31 + size_in_bytes) % 32;
        w.write_all(&vec![0u8; padding])?;
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Write;

    /// Hand-rolled temporary directory under `std::env::temp_dir()`. The dir
    /// is cleaned up when dropped. Used to avoid pulling in a `tempfile` dep
    /// just for tests.
    struct TmpDir(PathBuf);
    impl TmpDir {
        fn new(label: &str) -> Self {
            let mut p = std::env::temp_dir();
            p.push(format!(
                "candle-gguf-test-{label}-{}-{}",
                std::process::id(),
                std::time::SystemTime::now()
                    .duration_since(std::time::UNIX_EPOCH)
                    .map(|d| d.as_nanos())
                    .unwrap_or(0),
            ));
            std::fs::create_dir_all(&p).unwrap();
            Self(p)
        }
        fn path(&self) -> &Path {
            &self.0
        }
    }
    impl Drop for TmpDir {
        fn drop(&mut self) {
            let _ = std::fs::remove_dir_all(&self.0);
        }
    }

    /// `split_sibling_paths` should return `None` for a regular single-file
    /// GGUF and a sorted list of every part for a real split filename.
    #[test]
    fn test_split_sibling_paths_single_file() {
        // The file doesn't exist, but split detection runs purely on the
        // filename — non-split paths return None without touching disk.
        let p = Path::new("/this/path/does/not/exist/Foo-Q4_0.gguf");
        let r = split_sibling_paths(p).unwrap();
        assert!(r.is_none(), "single-file gguf should not be detected as split");
    }

    #[test]
    fn test_split_sibling_paths_detects_split_pattern() {
        let dir = TmpDir::new("split-detect");
        for i in 1..=3usize {
            let name = format!("Foo-Q4_1-{i:05}-of-00003.gguf");
            std::fs::write(dir.path().join(&name), b"").unwrap();
        }
        let first = dir.path().join("Foo-Q4_1-00001-of-00003.gguf");
        let parts = split_sibling_paths(&first).unwrap().expect("split detected");
        assert_eq!(parts.len(), 3);
        for (i, p) in parts.iter().enumerate() {
            let want = format!("Foo-Q4_1-{:05}-of-00003.gguf", i + 1);
            assert_eq!(p.file_name().unwrap().to_str().unwrap(), want);
        }
    }

    #[test]
    fn test_split_sibling_paths_missing_part() {
        let dir = TmpDir::new("split-missing");
        // Only write parts 1 and 3 of a 3-part split — part 2 is missing.
        for i in &[1usize, 3] {
            let name = format!("Foo-Q4_1-{i:05}-of-00003.gguf");
            std::fs::write(dir.path().join(&name), b"").unwrap();
        }
        let first = dir.path().join("Foo-Q4_1-00001-of-00003.gguf");
        let err = split_sibling_paths(&first).expect_err("missing part should error");
        let msg = format!("{err}");
        assert!(
            msg.contains("00002-of-00003"),
            "expected missing-part error, got: {msg}"
        );
    }

    /// MultiFileReader must read across file boundaries seamlessly. Construct
    /// three small temp files holding distinct byte patterns, then verify that
    /// `read_exact` and `seek` produce the right bytes regardless of which
    /// file the requested range straddles.
    #[test]
    fn test_multi_file_reader_concatenates_and_seeks() {
        let dir = TmpDir::new("multi-reader");
        // file0: 10 bytes of 0xAA, file1: 5 bytes of 0xBB, file2: 7 bytes of 0xCC
        let file_specs: [(usize, u8); 3] = [(10, 0xAA), (5, 0xBB), (7, 0xCC)];
        let mut paths = Vec::new();
        for (i, (n, byte)) in file_specs.iter().enumerate() {
            let p = dir.path().join(format!("part{i}"));
            let mut f = std::fs::File::create(&p).unwrap();
            f.write_all(&vec![*byte; *n]).unwrap();
            paths.push(p);
        }
        let files: Vec<File> = paths.iter().map(|p| File::open(p).unwrap()).collect();
        let mut reader = MultiFileReader::new(files).unwrap();
        // Total size = 10 + 5 + 7 = 22.
        assert_eq!(reader.total_size(), 22);
        assert_eq!(reader.file_starts, vec![0, 10, 15, 22]);

        // Read all 22 bytes sequentially.
        let mut all = vec![0u8; 22];
        std::io::Read::read_exact(&mut reader, &mut all).unwrap();
        let expect: Vec<u8> = (0..22)
            .map(|i| {
                if i < 10 {
                    0xAA
                } else if i < 15 {
                    0xBB
                } else {
                    0xCC
                }
            })
            .collect();
        assert_eq!(all, expect);

        // Seek into the middle of file0 and read across the file0→file1 boundary.
        std::io::Seek::seek(&mut reader, SeekFrom::Start(8)).unwrap();
        let mut buf = vec![0u8; 4]; // bytes [8..12) = [AA, AA, BB, BB]
        std::io::Read::read_exact(&mut reader, &mut buf).unwrap();
        assert_eq!(buf, vec![0xAA, 0xAA, 0xBB, 0xBB]);

        // Seek to file2 directly and read 3 bytes.
        std::io::Seek::seek(&mut reader, SeekFrom::Start(15)).unwrap();
        let mut buf = vec![0u8; 3];
        std::io::Read::read_exact(&mut reader, &mut buf).unwrap();
        assert_eq!(buf, vec![0xCC, 0xCC, 0xCC]);

        // Reading past the end returns 0 (EOF).
        std::io::Seek::seek(&mut reader, SeekFrom::Start(22)).unwrap();
        let mut buf = vec![0u8; 4];
        let n = std::io::Read::read(&mut reader, &mut buf).unwrap();
        assert_eq!(n, 0);
    }
}
