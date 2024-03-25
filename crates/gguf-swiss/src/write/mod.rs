mod primitives;

use std::io::Write;

use anyhow::{bail, Error};

use crate::{
    write::primitives::{write_string, write_u32, write_u64},
    Header, MetadataType, MetadataValue, TensorInfo, MAGIC_NUMBER,
};

pub fn write_header(writer: &mut impl Write, header: &Header) -> Result<(), Error> {
    // Write magic number and version
    writer.write_all(&MAGIC_NUMBER)?;
    write_u32(writer, 3)?;

    // Placeholder header sizes
    write_u64(writer, header.tensors.len() as u64)?;
    write_u64(writer, header.metadata.len() as u64)?;

    for (key, value) in &header.metadata {
        write_metadata_entry(writer, key, value)?;
    }

    for (key, value) in &header.tensors {
        write_tensor_info(writer, key, value)?;
    }

    Ok(())
}

pub fn write_metadata_entry(
    writer: &mut impl Write,
    key: &str,
    value: &MetadataValue,
) -> Result<(), Error> {
    write_string(writer, key)?;

    // TODO: Support different value types
    let MetadataValue::String(value) = value else {
        bail!("unsupported metadata value to write")
    };

    write_u32(writer, MetadataType::String as u32)?;
    write_string(writer, value)?;

    Ok(())
}

pub fn write_tensor_info(
    writer: &mut impl Write,
    key: &str,
    value: &TensorInfo,
) -> Result<(), Error> {
    write_string(writer, key)?;

    write_u32(writer, value.dimensions.count() as u32)?;
    for i in 0..value.dimensions.count() {
        write_u64(writer, value.dimensions.0[i])?;
    }

    write_u32(writer, value.tensor_type as u32)?;
    write_u64(writer, value.offset)?;

    Ok(())
}
