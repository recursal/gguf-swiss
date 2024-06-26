mod metadata;
mod primitives;

use std::io::Write;

use anyhow::Error;

use crate::{
    write::{
        metadata::write_metadata_entry,
        primitives::{write_string, write_u32, write_u64},
    },
    Header, TensorInfo, MAGIC_NUMBER,
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

    for value in &header.tensors {
        write_tensor_info(writer, value)?;
    }

    Ok(())
}

fn write_tensor_info(writer: &mut impl Write, value: &TensorInfo) -> Result<(), Error> {
    write_string(writer, value.name.as_bytes())?;

    write_u32(writer, value.dimensions.count() as u32)?;
    for i in 0..value.dimensions.count() {
        write_u64(writer, value.dimensions.0[i])?;
    }

    write_u32(writer, value.tensor_type as u32)?;
    write_u64(writer, value.offset)?;

    Ok(())
}
