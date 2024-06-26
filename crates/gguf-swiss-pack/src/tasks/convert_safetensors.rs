use std::{
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

    fn expand_tensors(&self) -> Result<Vec<(String, String, TensorManifest)>, Error> {
        let mut values = Vec::new();

        for (name, value) in &self.manifest.tensors {
            if let Some(mac) = name.strip_prefix('$') {
                let (start_s, end_s) = mac
                    .split_once("..")
                    .context("unable to split expansion macro")?;
                let start: usize = start_s.parse().context("unable to parse expansion macro")?;
                let end: usize = end_s.parse().context("unable to parse expansion macro")?;

                let table = value.as_table().context("expansion table must be table")?;
                for (name, value) in table {
                    // Parse the manifest
                    let manifest: TensorManifest = value
                        .clone()
                        .try_into()
                        .context("failed to parse manifest")?;

                    for i in start..end {
                        let is = i.to_string();
                        let target_name = name.replace('$', &is);
                        let source_name = manifest.source.replace('$', &is);

                        values.push((target_name, source_name, manifest.clone()));
                    }
                }

                continue;
            }

            // Parse the manifest
            let manifest: TensorManifest = value
                .clone()
                .try_into()
                .context("failed to parse manifest")?;
            values.push((name.clone(), manifest.source.clone(), manifest));
        }

        Ok(values)
    }

    fn prepare_tensor(
        &mut self,
        ctx: &mut ProcessContext,
        next_offset: &mut u64,
        manifest: &TensorManifest,
        target_name: String,
        source_name: String,
    ) -> Result<(), Error> {
        let (tensor_type, scalar_size) = match manifest.ty.as_str() {
            "F16" => (TensorType::F16, 2),
            "F32" => (TensorType::F32, 4),
            _ => {
                bail!("target tensor types other than F16 or F32 not supported for conversion currently");
            }
        };

        // Convert from manifest dimensions to gguf-swiss
        let mut dimensions = TensorDimensions::default();
        for (i, value) in manifest.dimensions.iter().enumerate() {
            dimensions.0[i] = *value;
        }

        // Record a conversion task
        let scalars = dimensions.total();
        let tensor_info = ConvertTensorInfo {
            name: target_name.clone(),
            source: source_name,
            dimensions,
            offset: *next_offset,
            ty: tensor_type,
        };
        self.tensors.push(tensor_info);

        // Record tensor entry
        let value = TensorInfo {
            name: target_name,
            tensor_type,
            dimensions,
            offset: *next_offset,
        };
        ctx.tensors.push(value);

        // Figure out the next offset
        *next_offset += scalars * scalar_size;
        *next_offset = align_offset(*next_offset);

        Ok(())
    }
}

impl PackTask for ConvertSafetensorsTask {
    fn process(&mut self, ctx: &mut ProcessContext) -> Result<(), Error> {
        let mut next_offset = 0;

        let expanded = self.expand_tensors()?;

        // Process expanded tensors
        for (target_name, source_name, value) in expanded {
            self.prepare_tensor(ctx, &mut next_offset, &value, target_name, source_name)?;
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
                tensor,
            )?;
        }

        Ok(())
    }
}

#[derive(Deserialize, Debug)]
struct ConvertSafetensorsManifest {
    pub source: String,
    pub tensors: Table,
}

#[derive(Deserialize, Debug, Clone)]
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
    // Convert to target format
    let data = match tensor.ty {
        TensorType::F16 => encode_values_f16(&scalars),
        TensorType::F32 => encode_values_f32(&scalars),
        _ => bail!("unexpected tensor type"),
    };

    // Pad if necessary
    let position = write_padding(target)?;

    // We should now be at the correct position, but make sure
    assert_eq!(position, data_start + tensor.offset);

    // Write the converted data
    target.write_all(&data)?;

    Ok(())
}

fn encode_values_f16(scalars: &[f32]) -> Vec<u8> {
    // TODO: One at a time is inefficient

    let mut data = vec![0u8; scalars.len() * 2];

    for (i, bytes) in data.chunks_mut(2).enumerate() {
        let value = half::f16::from_f32(scalars[i]);
        bytes.copy_from_slice(&value.to_le_bytes());
    }

    data
}

fn encode_values_f32(scalars: &[f32]) -> Vec<u8> {
    // TODO: One at a time is inefficient

    let mut data = vec![0u8; scalars.len() * 4];

    for (i, bytes) in data.chunks_mut(4).enumerate() {
        let value = scalars[i];
        bytes.copy_from_slice(&value.to_le_bytes());
    }

    data
}

pub struct ConvertTensorInfo {
    pub name: String,
    pub source: String,
    pub dimensions: TensorDimensions,
    pub offset: u64,
    pub ty: TensorType,
}
