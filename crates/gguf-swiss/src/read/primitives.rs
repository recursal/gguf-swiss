use std::io::Read;

use anyhow::{bail, Context, Error};

pub fn read_u8(reader: &mut impl Read) -> Result<u8, Error> {
    let mut bytes = [0u8; 1];

    reader.read_exact(&mut bytes)?;

    Ok(bytes[0])
}

pub fn read_i8(reader: &mut impl Read) -> Result<i8, Error> {
    let mut bytes = [0u8; 1];

    reader.read_exact(&mut bytes)?;

    Ok(i8::from_le_bytes(bytes))
}

pub fn read_u16(reader: &mut impl Read) -> Result<u16, Error> {
    let mut bytes = [0u8; 2];

    reader.read_exact(&mut bytes)?;
    let value = u16::from_le_bytes(bytes);

    Ok(value)
}

pub fn read_i16(reader: &mut impl Read) -> Result<i16, Error> {
    let mut bytes = [0u8; 2];

    reader.read_exact(&mut bytes)?;
    let value = i16::from_le_bytes(bytes);

    Ok(value)
}

pub fn read_u32(reader: &mut impl Read) -> Result<u32, Error> {
    let mut bytes = [0u8; 4];

    reader.read_exact(&mut bytes)?;
    let value = u32::from_le_bytes(bytes);

    Ok(value)
}

pub fn read_i32(reader: &mut impl Read) -> Result<i32, Error> {
    let mut bytes = [0u8; 4];

    reader.read_exact(&mut bytes)?;
    let value = i32::from_le_bytes(bytes);

    Ok(value)
}

pub fn read_u64(reader: &mut impl Read) -> Result<u64, Error> {
    let mut bytes = [0u8; 8];

    reader.read_exact(&mut bytes)?;
    let value = u64::from_le_bytes(bytes);

    Ok(value)
}

pub fn read_i64(reader: &mut impl Read) -> Result<i64, Error> {
    let mut bytes = [0u8; 8];

    reader.read_exact(&mut bytes)?;
    let value = i64::from_le_bytes(bytes);

    Ok(value)
}

pub fn read_f32(reader: &mut impl Read) -> Result<f32, Error> {
    let mut bytes = [0u8; 4];

    reader.read_exact(&mut bytes)?;
    let value = f32::from_le_bytes(bytes);

    Ok(value)
}

pub fn read_f64(reader: &mut impl Read) -> Result<f64, Error> {
    let mut bytes = [0u8; 8];

    reader.read_exact(&mut bytes)?;
    let value = f64::from_le_bytes(bytes);

    Ok(value)
}

pub fn read_string(reader: &mut impl Read) -> Result<String, Error> {
    let length = read_u64(reader)? as usize;

    if length > 65535 {
        bail!("invalid string: too long");
    }

    let mut bytes = vec![0u8; length];
    reader.read_exact(&mut bytes)?;

    let value = String::from_utf8(bytes).context("invalid string: not ascii")?;
    Ok(value)
}
