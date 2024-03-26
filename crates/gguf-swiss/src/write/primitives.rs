use std::io::Write;

use anyhow::{bail, Error};

pub fn write_u8(writer: &mut impl Write, value: u8) -> Result<(), Error> {
    let bytes = value.to_le_bytes();
    writer.write_all(&bytes)?;

    Ok(())
}

pub fn write_i8(writer: &mut impl Write, value: i8) -> Result<(), Error> {
    let bytes = value.to_le_bytes();
    writer.write_all(&bytes)?;

    Ok(())
}

pub fn write_u16(writer: &mut impl Write, value: u16) -> Result<(), Error> {
    let bytes = value.to_le_bytes();
    writer.write_all(&bytes)?;

    Ok(())
}

pub fn write_i16(writer: &mut impl Write, value: i16) -> Result<(), Error> {
    let bytes = value.to_le_bytes();
    writer.write_all(&bytes)?;

    Ok(())
}

pub fn write_u32(writer: &mut impl Write, value: u32) -> Result<(), Error> {
    let bytes = value.to_le_bytes();
    writer.write_all(&bytes)?;

    Ok(())
}

pub fn write_i32(writer: &mut impl Write, value: i32) -> Result<(), Error> {
    let bytes = value.to_le_bytes();
    writer.write_all(&bytes)?;

    Ok(())
}

pub fn write_u64(writer: &mut impl Write, value: u64) -> Result<(), Error> {
    let bytes = value.to_le_bytes();
    writer.write_all(&bytes)?;

    Ok(())
}

pub fn write_i64(writer: &mut impl Write, value: i64) -> Result<(), Error> {
    let bytes = value.to_le_bytes();
    writer.write_all(&bytes)?;

    Ok(())
}

pub fn write_f32(writer: &mut impl Write, value: f32) -> Result<(), Error> {
    let bytes = value.to_le_bytes();
    writer.write_all(&bytes)?;

    Ok(())
}

pub fn write_f64(writer: &mut impl Write, value: f64) -> Result<(), Error> {
    let bytes = value.to_le_bytes();
    writer.write_all(&bytes)?;

    Ok(())
}

pub fn write_bool(writer: &mut impl Write, value: bool) -> Result<(), Error> {
    write_u8(writer, value as u8)
}

pub fn write_string(writer: &mut impl Write, data: &str) -> Result<(), Error> {
    if data.bytes().len() > 65535 {
        bail!("invalid string: too long");
    }
    if !data.is_ascii() {
        bail!("invalid string: not ascii");
    }

    write_u64(writer, data.bytes().len() as u64)?;
    writer.write_all(data.as_bytes())?;

    Ok(())
}
