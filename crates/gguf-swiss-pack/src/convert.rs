use std::{
    fs::File,
    io::{Read, Seek, SeekFrom, Write},
};

use anyhow::{Context, Error};
use gguf_swiss::{align_offset, Header, MetadataValue, TensorDimensions, TensorInfo, TensorType};

use crate::safetensors::{self, StHeader};

pub fn convert(
    output: &mut File,
    tensors_source_file: &mut File,
    info: &ConvertInfo,
) -> Result<(), Error> {
    // Generate and write the GGUF header
    write_header(output, info)?;
    let data_start = write_padding(output)?;

    // Handle all tensor conversion tasks
    let source_header = safetensors::read_header(tensors_source_file)?;
    for tensor in &info.tensors {
        convert_tensor(
            output,
            data_start,
            tensors_source_file,
            &source_header,
            &tensor,
        )?;
    }

    Ok(())
}

fn write_header(target: &mut File, info: &ConvertInfo) -> Result<(), Error> {
    println!("writing header");

    let mut header = Header::default();

    header.metadata = info.metadata.clone();
    apply_header_tensors(&mut header, info)?;

    // Write the prepared header
    gguf_swiss::write_header(target, &header)?;

    Ok(())
}

fn apply_header_tensors(header: &mut Header, info: &ConvertInfo) -> Result<(), Error> {
    for tensor in &info.tensors {
        // Record the tensor in the metadata
        let info = TensorInfo {
            name: tensor.name.clone(),
            tensor_type: TensorType::F16,
            dimensions: tensor.dimensions,
            offset: tensor.offset,
        };
        header.tensors.push(info);
    }

    Ok(())
}

fn write_padding(target: &mut File) -> Result<u64, Error> {
    let current = target.stream_position()?;

    let padded = align_offset(current);
    let padding = padded - current;

    if padding != 0 {
        let padding = vec![0u8; padding as usize];
        target.write_all(&padding)?;
    }

    Ok(padded)
}

fn convert_tensor(
    target: &mut File,
    data_start: u64,
    source_file: &mut File,
    source_header: &StHeader,
    tensor: &ConvertTensorInfo,
) -> Result<(), Error> {
    println!("converting tensor {:?}", tensor.name);

    let scalars = read_source_scalars(source_file, source_header, tensor)?;
    write_scalars(target, data_start, tensor, scalars)?;

    Ok(())
}

fn read_source_scalars(
    source_file: &mut File,
    source_header: &StHeader,
    tensor: &ConvertTensorInfo,
) -> Result<Vec<f32>, Error> {
    let source_info = &source_header
        .entries
        .get(&tensor.source)
        // TODO: This error should probably be more informative
        .context("unable to find source tensor")?;

    let source_dimensions = TensorDimensions::from_width_last(&source_info.data_shape)?;
    let scalars_len = tensor.dimensions.total();

    assert_eq!(source_info.data_type, "BF16");
    assert_eq!(source_dimensions.total(), scalars_len);

    // Prepare buffer for the source data
    let mut data = vec![0u8; scalars_len as usize * 2];

    // Validate the source data is the correct size
    let data_expected = source_info.data_offsets[1] - source_info.data_offsets[0];
    assert_eq!(data.len(), data_expected as usize);

    // Read the raw source data
    let start = source_header.data_start + source_info.data_offsets[0];
    source_file.seek(SeekFrom::Start(start))?;
    source_file.read_exact(&mut data)?;

    // Convert to f32
    // TODO: One at a time is inefficient
    let mut scalars = vec![0f32; scalars_len as usize];
    for (i, source_bytes) in data.chunks(2).enumerate() {
        let half_value = half::bf16::from_le_bytes([source_bytes[0], source_bytes[1]]);
        scalars[i] = half_value.to_f32();
    }

    Ok(scalars)
}

fn write_scalars(
    target: &mut File,
    data_start: u64,
    tensor: &ConvertTensorInfo,
    scalars: Vec<f32>,
) -> Result<(), Error> {
    // Convert to target format (just f16 for now)
    // TODO: One at a time is inefficient
    let scalars_len = tensor.dimensions.total();
    let mut target_data = vec![0u8; scalars_len as usize * 2];
    for (i, target_bytes) in target_data.chunks_mut(2).enumerate() {
        let value = half::f16::from_f32(scalars[i]);
        target_bytes.copy_from_slice(&value.to_le_bytes());
    }

    // Pad if necessary
    let position = write_padding(target)?;

    // We should now be at the correct position, but make sure
    assert_eq!(position, data_start + tensor.offset);

    // Write the converted data
    target.write_all(&target_data)?;

    Ok(())
}

#[derive(Default)]
pub struct ConvertInfo {
    pub metadata: Vec<(String, MetadataValue)>,
    pub tensors: Vec<ConvertTensorInfo>,
}

pub struct ConvertTensorInfo {
    pub name: String,
    pub source: String,
    pub dimensions: TensorDimensions,
    pub offset: u64,
}
