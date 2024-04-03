use std::io::Write;

use anyhow::{bail, Error};

use crate::{
    write::primitives::{
        write_bool, write_f32, write_f64, write_i16, write_i32, write_i64, write_i8, write_string,
        write_u16, write_u32, write_u64, write_u8,
    },
    MetadataArray, MetadataValue,
};

pub fn write_metadata_entry(
    writer: &mut impl Write,
    key: &str,
    value: &MetadataValue,
) -> Result<(), Error> {
    write_string(writer, key.as_bytes())?;

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
        MetadataValue::Array(value) => write_array(writer, value)?,
        MetadataValue::UInt64(value) => write_u64(writer, *value)?,
        MetadataValue::Int64(value) => write_i64(writer, *value)?,
        MetadataValue::Float64(value) => write_f64(writer, *value)?,
    }

    Ok(())
}

fn write_array(writer: &mut impl Write, value: &MetadataArray) -> Result<(), Error> {
    // Write the type
    let ty = value.ty();
    write_u32(writer, ty as u32)?;

    // Write the array itself
    match value {
        MetadataArray::UInt8(value) => array_inner(writer, write_u8, value)?,
        MetadataArray::Int8(value) => array_inner(writer, write_i8, value)?,
        MetadataArray::UInt16(value) => array_inner(writer, write_u16, value)?,
        MetadataArray::Int16(value) => array_inner(writer, write_i16, value)?,
        MetadataArray::UInt32(value) => array_inner(writer, write_u32, value)?,
        MetadataArray::Int32(value) => array_inner(writer, write_i32, value)?,
        MetadataArray::Float32(value) => array_inner(writer, write_f32, value)?,
        MetadataArray::Bool(value) => array_inner(writer, write_bool, value)?,
        MetadataArray::String(value) => {
            let value: Vec<_> = value.iter().map(|v| v.as_slice()).collect();
            array_inner(writer, write_string, &value)?
        }
        MetadataArray::Array(_) => {
            // TODO: Implement this
            bail!("array of array writing unsupported")
        }
        MetadataArray::UInt64(value) => array_inner(writer, write_u64, value)?,
        MetadataArray::Int64(value) => array_inner(writer, write_i64, value)?,
        MetadataArray::Float64(value) => array_inner(writer, write_f64, value)?,
    }

    Ok(())
}

fn array_inner<T, W, F>(writer: &mut W, mut write: F, value: &[T]) -> Result<(), Error>
where
    T: Copy,
    W: Write,
    F: FnMut(&mut W, T) -> Result<(), Error>,
{
    // Write length
    write_u64(writer, value.len() as u64)?;

    // Write every entry
    for value in value {
        write(writer, *value)?
    }

    Ok(())
}
