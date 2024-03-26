use std::io::Write;

use anyhow::{bail, Error};

use crate::{
    write::primitives::{
        write_bool, write_f32, write_f64, write_i16, write_i32, write_i64, write_i8, write_string,
        write_u16, write_u32, write_u64, write_u8,
    },
    MetadataValue,
};

pub fn write_metadata_entry(
    writer: &mut impl Write,
    key: &str,
    value: &MetadataValue,
) -> Result<(), Error> {
    write_string(writer, key)?;

    // Write the type
    let ty = value.ty();
    write_u32(writer, ty as u32)?;

    // Write the specific value
    match value {
        MetadataValue::UInt8(value) => write_u8(writer, *value)?,
        MetadataValue::Int8(value) => write_i8(writer, *value)?,
        MetadataValue::UInt16(value) => write_u16(writer, *value)?,
        MetadataValue::Int16(value) => write_i16(writer, *value)?,
        MetadataValue::UInt32(value) => write_u32(writer, *value)?,
        MetadataValue::Int32(value) => write_i32(writer, *value)?,
        MetadataValue::Float32(value) => write_f32(writer, *value)?,
        MetadataValue::Bool(value) => write_bool(writer, *value)?,
        MetadataValue::String(value) => write_string(writer, value)?,
        MetadataValue::Array(_value) => bail!("unsupported value type"),
        MetadataValue::UInt64(value) => write_u64(writer, *value)?,
        MetadataValue::Int64(value) => write_i64(writer, *value)?,
        MetadataValue::Float64(value) => write_f64(writer, *value)?,
    }

    Ok(())
}
