use std::{
    collections::HashMap,
    fs::File,
    io::{Read, Seek, SeekFrom, Write},
    path::Path,
};

use anyhow::{bail, Context, Error};
use gguf_swiss::{align_offset, TensorDimensions, TensorInfo, TensorType};
use serde::Deserialize;
use toml::Table;

use crate::{
    safetensors::{self, StHeader},
    tasks::{PackTask, ProcessContext},
};

pub struct ConvertSafetensorsTask {
    manifest: ConvertSafetensorsManifest,
    tensors: Vec<ConvertTensorInfo>,
}

impl ConvertSafetensorsTask {
    pub fn new(manifest: &Table) -> Result<Self, Error> {
        let manifest = manifest
            .clone()
            .try_into()
            .context("failed to parse manifest")?;

        let value = Self {
            manifest,
            tensors: Vec::new(),
        };
        Ok(value)
    }
}

impl PackTask for ConvertSafetensorsTask {
    fn process(&mut self, ctx: &mut ProcessContext) -> Result<(), Error> {
        let mut next_offset = 0;

        for (name, value) in &self.manifest.tensors {
            if value.ty != "F16" {
                bail!("target tensor types other than F16 not supported for conversion currently");
            }

            // Convert from manifest dimensions to gguf-swiss
            let mut dimensions = TensorDimensions::default();
            for (i, value) in value.dimensions.iter().enumerate() {
                dimensions.0[i] = *value;
            }

            // Record a conversion task
            let scalars = dimensions.total();
            let tensor_info = ConvertTensorInfo {
                name: name.clone(),
                source: value.source.clone(),
                dimensions,
                offset: next_offset,
            };
            self.tensors.push(tensor_info);

            // Record tensor entry
            let value = TensorInfo {
                name: name.clone(),
                tensor_type: TensorType::F16,
                dimensions,
                offset: next_offset,
            };
            ctx.tensors.push(value);

            // Figure out the next offset
            next_offset += scalars * 2;
            next_offset = align_offset(next_offset);
        }

        Ok(())
    }

    fn write_tensors(&mut self, source_root: &Path, output: &mut File) -> Result<(), Error> {
        let data_start = write_padding(output)?;

        // Open tensors source file
        let tensors_source_file_path = source_root.join(&self.manifest.source);
        let mut tensors_source_file =
            File::open(tensors_source_file_path).context("failed to open tensors source")?;

        // Handle all tensor conversion tasks
        let source_header = safetensors::read_header(&mut tensors_source_file)?;
        for tensor in &self.tensors {
            convert_tensor(
                output,
                data_start,
                &mut tensors_source_file,
                &source_header,
                &tensor,
            )?;
        }

        Ok(())
    }
}

#[derive(Deserialize, Debug)]
struct ConvertSafetensorsManifest {
    pub source: String,
    pub tensors: HashMap<String, TensorManifest>,
}

#[derive(Deserialize, Debug)]
pub struct TensorManifest {
    pub source: String,

    #[serde(rename = "type")]
    pub ty: String,

    pub dimensions: Vec<u64>,
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
    assert_eq!(source_dimensions, tensor.dimensions);

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

pub struct ConvertTensorInfo {
    pub name: String,
    pub source: String,
    pub dimensions: TensorDimensions,
    pub offset: u64,
}
