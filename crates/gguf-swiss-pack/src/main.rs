mod manifest;
mod safetensors;

use std::{
    fs::File,
    io::{Read, Seek, SeekFrom, Write},
    path::PathBuf,
};

use anyhow::{bail, Context, Error};
use clap::Parser;
use gguf_swiss::{align_offset, Header, TensorDimensions, TensorInfo, TensorType};

use crate::{
    manifest::{read_manifest, Manifest},
    safetensors::StHeader,
};

fn main() -> Result<(), Error> {
    let args = Args::parse();

    println!("reading manifest");
    let manifest_path = PathBuf::from(args.manifest);
    let manifest = read_manifest(&manifest_path).context("failed to load packaging manifest")?;

    // Validate model path
    let model_path = PathBuf::from(args.model);
    if !model_path.is_dir() {
        bail!("model path is not a valid directory");
    }

    let mut target = File::create(&args.output)?;
    let mut tensor_tasks = Vec::new();

    // Generate and write the GGUF header
    write_header(&mut target, &manifest, &mut tensor_tasks)?;
    let data_start = write_padding(&mut target)?;

    // Read input safetensors header
    // TODO: Support more than one source, and different formats
    let source = &manifest.tensors.sources[0];
    let source_path = model_path.join(source);
    let mut source_file = File::open(source_path).context("failed to open source file")?;
    let source_header = safetensors::read_header(&mut source_file)?;

    // Handle all tensor conversion tasks
    for task in tensor_tasks {
        run_tensor_task(
            &mut target,
            data_start,
            &mut source_file,
            &source_header,
            &task,
        )?;
    }

    Ok(())
}

fn write_header(
    target: &mut File,
    manifest: &Manifest,
    tensor_tasks: &mut Vec<TensorTask>,
) -> Result<(), Error> {
    println!("writing header");

    let mut header = Header::default();

    convert_metadata(&mut header, manifest)?;
    convert_tensors(&mut header, manifest, tensor_tasks)?;

    // Write the prepared header
    gguf_swiss::write_header(target, &header)?;

    Ok(())
}

/// Convert explicit manifest metadata to header.
fn convert_metadata(header: &mut Header, manifest: &Manifest) -> Result<(), Error> {
    for (key, value) in &manifest.metadata {
        let Some(value) = value.as_str() else {
            bail!("unsupported metadata value for {:?}", key)
        };

        header.metadata.insert(key.clone(), value.to_string());
    }

    Ok(())
}

/// Convert tensor entries, and prepare conversion tasks.
fn convert_tensors(
    header: &mut Header,
    manifest: &Manifest,
    tensor_tasks: &mut Vec<TensorTask>,
) -> Result<(), Error> {
    let mut next_offset = 0;

    for (name, tensor_manifest) in &manifest.tensors.entries {
        if tensor_manifest.tensor_type != "F16" {
            bail!("target tensor types other than F16 not supported for conversion currently");
        }

        // Convert from manifest dimensions to gguf-swiss
        let mut dimensions = TensorDimensions::default();
        for (i, value) in tensor_manifest.dimensions.iter().enumerate() {
            dimensions.0[i] = *value;
        }

        // Record the tensor in the metadata
        let info = TensorInfo {
            tensor_type: TensorType::F16,
            dimensions,
            offset: next_offset,
        };
        header.tensors.insert(name.clone(), info);

        // Record a conversion task
        let scalars = dimensions.total();
        let task = TensorTask {
            name: name.clone(),
            source: tensor_manifest.source.clone(),
            offset: next_offset,
            scalars,
        };
        tensor_tasks.push(task);

        // Figure out the next offset
        next_offset += scalars * 2;
        next_offset = align_offset(next_offset);
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

fn run_tensor_task(
    target: &mut File,
    data_start: u64,
    source_file: &mut File,
    source_header: &StHeader,
    task: &TensorTask,
) -> Result<(), Error> {
    println!("converting tensor {}", task.name);

    let scalars = read_source_scalars(source_file, source_header, task)?;

    // Convert to target format (just f16 for now)
    // TODO: One at a time is inefficient
    let mut target_data = vec![0u8; task.scalars as usize * 2];
    for (i, target_bytes) in target_data.chunks_mut(2).enumerate() {
        let value = half::f16::from_f32(scalars[i]);
        target_bytes.copy_from_slice(&value.to_le_bytes());
    }

    // Pad if necessary
    let position = write_padding(target)?;

    // We should now be at the correct position, but make sure
    assert_eq!(position, data_start + task.offset);

    // Write the converted data
    target.write_all(&target_data)?;

    Ok(())
}

fn read_source_scalars(
    source_file: &mut File,
    source_header: &StHeader,
    task: &TensorTask,
) -> Result<Vec<f32>, Error> {
    let source_info = &source_header
        .entries
        .get(&task.source)
        // TODO: This error should probably be more informative
        .context("unable to find source tensor")?;

    let dimensions = TensorDimensions::from_width_last(&source_info.data_shape)?;

    assert_eq!(source_info.data_type, "BF16");
    assert_eq!(dimensions.total(), task.scalars);

    // Prepare buffer for the source data
    let mut data = vec![0u8; task.scalars as usize * 2];

    // Validate the source data is the correct size
    let data_expected = source_info.data_offsets[1] - source_info.data_offsets[0];
    assert_eq!(data.len(), data_expected as usize);

    // Read the raw source data
    let start = source_header.data_start + source_info.data_offsets[0];
    source_file.seek(SeekFrom::Start(start))?;
    source_file.read_exact(&mut data)?;

    // Convert to f32
    // TODO: One at a time is inefficient
    let mut scalars = vec![0f32; task.scalars as usize];
    for (i, source_bytes) in data.chunks(2).enumerate() {
        let half_value = half::bf16::from_le_bytes([source_bytes[0], source_bytes[1]]);
        scalars[i] = half_value.to_f32();
    }

    Ok(scalars)
}

/// GGUF Swiss Army Knife, model packaging utility.
///
/// Currently only supports RWKV5 in safetensors format.
#[derive(Parser, Debug)]
#[command(version, about)]
struct Args {
    /// Path to the packaging manifest to read.
    #[arg(long)]
    manifest: String,

    /// Path to the model directory to read.
    #[arg(long)]
    model: String,

    /// Path to the output file.
    output: String,
}

struct TensorTask {
    name: String,
    source: String,
    offset: u64,
    scalars: u64,
}
