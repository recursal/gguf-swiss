use std::{
    collections::HashMap,
    fs::File,
    io::{Read, Seek, SeekFrom},
};

use anyhow::{Context, Error};
use serde_json::Value;

// TODO: This module should not export anything other than a source processing function.
//  Specifics about the safetensors format aren't relevant for the core processing code.

pub fn read_header(file: &mut File) -> Result<StHeader, Error> {
    file.seek(SeekFrom::Start(0))?;

    // Read the header
    let mut header = [0u8; 8];
    file.read_exact(&mut header)?;
    let size = u64::from_le_bytes(header);

    let mut bytes = vec![0u8; size as usize];
    file.read_exact(&mut bytes)?;
    let text = std::str::from_utf8(&bytes)?;

    // Parse in the header
    let entries: HashMap<String, Value> = serde_json::from_str(text)?;

    // Convert to tensor entries
    let entries: Result<_, _> = entries
        .into_iter()
        .filter(|(name, _)| !name.starts_with("__"))
        .map(|(name, entry)| parse_entry(name, entry))
        .collect();
    let entries = entries?;

    let start = file.stream_position()?;

    let value = StHeader {
        entries,
        data_start: start,
    };
    Ok(value)
}

fn parse_entry(name: String, entry: Value) -> Result<(String, StTensorInfo), Error> {
    let data_type = entry
        .get("dtype")
        .context("missing key dtype")?
        .as_str()
        .context("invalid type")?;
    let data_offsets = entry
        .get("data_offsets")
        .context("missing key data_offsets")?
        .as_array()
        .context("invalid type")?;

    let start: u64 = data_offsets
        .get(0)
        .context("invalid data_offsets")?
        .as_u64()
        .context("invalid data_offsets")?;
    let end: u64 = data_offsets
        .get(1)
        .context("invalid data_offsets")?
        .as_u64()
        .context("invalid data_offsets")?;

    let shape_value = entry.get("shape").context("missing key shape")?;
    let data_shape = serde_json::from_value(shape_value.clone())?;

    let entry = StTensorInfo {
        data_type: data_type.to_string(),
        data_offsets: [start, end],
        data_shape,
    };
    Ok((name, entry))
}

#[derive(Debug)]
pub struct StHeader {
    pub entries: HashMap<String, StTensorInfo>,
    pub data_start: u64,
}

#[derive(Debug)]
pub struct StTensorInfo {
    pub data_type: String,
    pub data_offsets: [u64; 2],
    pub data_shape: Vec<u64>,
}
