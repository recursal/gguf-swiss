mod metadata;
mod primitives;

use std::io::Read;

use anyhow::{bail, Context, Error};

use crate::{
    read::{
        metadata::read_metadata_entry,
        primitives::{read_string, read_u32, read_u64},
    },
    Header, TensorDimensions, TensorInfo, TensorType, MAGIC_NUMBER,
};

/// Read the header of a GGUF file reader.
///
/// This function will also validate the magic number, and supported version.
pub fn read_header(reader: &mut impl Read) -> Result<Header, Error> {
    // Validate we're reading a GGUF model
    let mut magic_bytes = [0u8; 4];
    reader.read_exact(&mut magic_bytes)?;
    if magic_bytes != MAGIC_NUMBER {
        bail!("magic number doesn't match");
    }

    // Validate we're reading a supported version
    // Currently, only 2 and 3 are supported. These versions are mostly identical except for that
    // version 3 can hypothetically contain big-endian values (but has no mechanism for indicating
    // this, so for now we just ignore this).
    // Version 1 has 32-bit counts, and will require more work to support if desired.
    let gguf_version = read_u32(reader)?;
    if gguf_version != 3 && gguf_version != 2 {
        bail!("unsupported gguf version: {}", gguf_version);
    }

    // Read header data
    let tensor_count = read_u64(reader)?;
    let metadata_kv_count = read_u64(reader)?;

    // Protection against unreasonably large values
    if tensor_count > 1024 {
        bail!("excessive tensor count");
    }
    if metadata_kv_count > 1024 {
        bail!("excessive metadata count");
    }

    // Read metadata KVs
    let mut metadata = Vec::new();
    for _ in 0..metadata_kv_count {
        let (key, value) = read_metadata_entry(reader).context("failed to read metadata entry")?;
        metadata.push((key, value));
    }

    // Read tensor info
    let mut tensors = Vec::new();
    for _ in 0..tensor_count {
        let value = read_tensor_info(reader).context("failed to read tensor info")?;
        tensors.push(value);
    }

    let value = Header { metadata, tensors };
    Ok(value)
}

fn read_tensor_info(reader: &mut impl Read) -> Result<TensorInfo, Error> {
    let name = read_string(reader)?;
    let name = String::from_utf8(name).context("name not valid utf-8")?;

    // Read the tensor dimensions
    let dimensions_count = read_u32(reader)? as usize;

    if dimensions_count > 4 {
        bail!("invalid tensor: too many dimensions")
    }

    let mut dimensions = TensorDimensions([0, 0, 0, 0]);
    for i in 0..dimensions_count {
        dimensions.0[i] = read_u64(reader)?;
    }

    // Read the tensor type
    let tensor_type_index = read_u32(reader)?;
    let tensor_type =
        TensorType::from_u32(tensor_type_index).context("invalid tensor: invalid type")?;

    // Read the tensor offset
    let offset = read_u64(reader)?;

    let value = TensorInfo {
        name,
        tensor_type,
        dimensions,
        offset,
    };
    Ok(value)
}
