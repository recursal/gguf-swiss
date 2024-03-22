use std::io::Write;

use anyhow::{bail, Error};

pub fn write_u32(writer: &mut impl Write, value: u32) -> Result<(), Error> {
    let bytes = value.to_le_bytes();
    writer.write_all(&bytes)?;

    Ok(())
}

pub fn write_u64(writer: &mut impl Write, value: u64) -> Result<(), Error> {
    let bytes = value.to_le_bytes();
    writer.write_all(&bytes)?;

    Ok(())
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
