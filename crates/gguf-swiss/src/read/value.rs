use std::io::Read;

use anyhow::{bail, Context, Error};

use crate::read::primitives::read_bool;
use crate::{
    read::primitives::{
        read_f32, read_f64, read_i16, read_i32, read_i64, read_i8, read_string, read_u16, read_u32,
        read_u64, read_u8,
    },
    MetadataArray, MetadataType, MetadataValue,
};

pub fn read_metadata_value(reader: &mut impl Read) -> Result<MetadataValue, Error> {
    let type_index = read_u32(reader)?;
    let ty = MetadataType::from_u32(type_index).context("invalid type")?;

    let value = match ty {
        MetadataType::UInt8 => MetadataValue::UInt8(read_u8(reader)?),
        MetadataType::Int8 => MetadataValue::Int8(read_i8(reader)?),
        MetadataType::UInt16 => MetadataValue::UInt16(read_u16(reader)?),
        MetadataType::Int16 => MetadataValue::Int16(read_i16(reader)?),
        MetadataType::UInt32 => MetadataValue::UInt32(read_u32(reader)?),
        MetadataType::Int32 => MetadataValue::Int32(read_i32(reader)?),
        MetadataType::Float32 => MetadataValue::Float32(read_f32(reader)?),
        MetadataType::Bool => MetadataValue::Bool(read_u8(reader)? != 0),
        MetadataType::String => MetadataValue::String(read_string(reader)?),
        MetadataType::Array => MetadataValue::Array(read_metadata_array(reader, 0)?),
        MetadataType::UInt64 => MetadataValue::UInt64(read_u64(reader)?),
        MetadataType::Int64 => MetadataValue::Int64(read_i64(reader)?),
        MetadataType::Float64 => MetadataValue::Float64(read_f64(reader)?),
    };

    Ok(value)
}

fn read_metadata_array(reader: &mut impl Read, depth: usize) -> Result<MetadataArray, Error> {
    if depth > 2 {
        bail!("excessive metadata depth");
    }

    let type_index = read_u32(reader)?;
    let ty = MetadataType::from_u32(type_index).context("invalid type")?;

    let length = read_u64(reader)?;

    // This is a very large value, but it's necessary for some vocabs
    if length > 524288 {
        bail!("excessive array size");
    }

    let value = match ty {
        MetadataType::UInt8 => MetadataArray::UInt8(array_inner(length, reader, read_u8)?),
        MetadataType::Int8 => MetadataArray::Int8(array_inner(length, reader, read_i8)?),
        MetadataType::UInt16 => MetadataArray::UInt16(array_inner(length, reader, read_u16)?),
        MetadataType::Int16 => MetadataArray::Int16(array_inner(length, reader, read_i16)?),
        MetadataType::UInt32 => MetadataArray::UInt32(array_inner(length, reader, read_u32)?),
        MetadataType::Int32 => MetadataArray::Int32(array_inner(length, reader, read_i32)?),
        MetadataType::Float32 => MetadataArray::Float32(array_inner(length, reader, read_f32)?),
        MetadataType::Bool => MetadataArray::Bool(array_inner(length, reader, read_bool)?),
        MetadataType::String => MetadataArray::String(array_inner(length, reader, read_string)?),
        MetadataType::Array => {
            let read_value = |reader: &mut _| read_metadata_array(reader, depth + 1);
            let array = array_inner(length, reader, read_value)?;
            MetadataArray::Array(array)
        }
        MetadataType::UInt64 => MetadataArray::UInt64(array_inner(length, reader, read_u64)?),
        MetadataType::Int64 => MetadataArray::Int64(array_inner(length, reader, read_i64)?),
        MetadataType::Float64 => MetadataArray::Float64(array_inner(length, reader, read_f64)?),
    };

    Ok(value)
}

fn array_inner<T, F, R>(length: u64, reader: &mut R, mut read: F) -> Result<Vec<T>, Error>
where
    F: FnMut(&mut R) -> Result<T, Error>,
    R: Read,
{
    let mut data = Vec::with_capacity(length as usize);

    for _ in 0..length {
        let value = read(reader)?;
        data.push(value);
    }

    Ok(data)
}
