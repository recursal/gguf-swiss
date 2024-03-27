mod convert;
mod manifest;
mod safetensors;

use std::{fs::File, path::PathBuf};

use anyhow::{bail, Context, Error};
use clap::Parser;
use gguf_swiss::{align_offset, MetadataValue, TensorDimensions};

use crate::{
    convert::{ConvertInfo, ConvertTensorInfo},
    manifest::{read_manifest, Manifest},
};

fn main() -> Result<(), Error> {
    let args = Args::parse();

    let manifest_path = PathBuf::from(args.manifest);
    let model_path = PathBuf::from(args.model);

    // Validate paths
    if !manifest_path.is_file() {
        bail!("manifest path is not a valid file");
    }
    if !model_path.is_dir() {
        bail!("model path is not a valid directory");
    }

    println!("loading manifest");
    let manifest = read_manifest(&manifest_path).context("failed to load packaging manifest")?;

    // Prepare output file
    let mut output = File::create(&args.output)?;

    // Open tensors source file
    // TODO: Support more than one source
    let tensors_source_file_path = model_path.join(&manifest.tensors.sources[0]);
    let mut tensors_source_file =
        File::open(tensors_source_file_path).context("failed to open tensors source")?;

    // Prepare conversion info, gathering and validating the information necessary to perform
    // conversion
    println!("preparing conversion");
    let mut info = ConvertInfo::default();
    prepare_metadata(&mut info, &manifest)?;
    prepare_tensors(&mut info, &manifest)?;

    // Perform conversion
    convert::convert(&mut output, &mut tensors_source_file, &info)?;

    Ok(())
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

fn prepare_metadata(info: &mut ConvertInfo, manifest: &Manifest) -> Result<(), Error> {
    for (key, value) in &manifest.metadata {
        let Some(value) = value.as_str() else {
            bail!("unsupported metadata value for {:?}", key)
        };

        let value = MetadataValue::String(value.to_string());
        info.metadata.push((key.clone(), value));
    }

    Ok(())
}

fn prepare_tensors(info: &mut ConvertInfo, manifest: &Manifest) -> Result<(), Error> {
    let mut next_offset = 0;

    for (name, value) in &manifest.tensors.entries {
        if value.tensor_type != "F16" {
            bail!("target tensor types other than F16 not supported for conversion currently");
        }

        // Convert from manifest dimensions to gguf-swiss
        let mut dimensions = TensorDimensions::default();
        for (i, value) in value.dimensions.iter().enumerate() {
            dimensions.0[i] = *value;
        }

        // Record a conversion task
        let scalars = dimensions.total();
        let task = ConvertTensorInfo {
            name: name.clone(),
            source: value.source.clone(),
            dimensions,
            offset: next_offset,
        };
        info.tensors.push(task);

        // Figure out the next offset
        next_offset += scalars * 2;
        next_offset = align_offset(next_offset);
    }

    Ok(())
}
