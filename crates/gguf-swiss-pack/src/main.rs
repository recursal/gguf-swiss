mod manifest;
mod safetensors;
mod tensors;

use std::{collections::HashMap, fs::File, path::PathBuf};

use anyhow::{bail, Context, Error};
use clap::Parser;
use gguf_swiss::{
    align_offset, Header, MetadataArray, MetadataValue, TensorDimensions, TensorInfo, TensorType,
};
use serde_json::Value;

use crate::{
    manifest::{read_manifest, Manifest},
    tensors::ConvertTensorInfo,
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
    let mut metadata = prepare_metadata(&manifest)?;
    let tensors = prepare_tensors(&manifest)?;

    // Load tokenizer
    println!("loading tokenizer");
    let vocab_file_path = model_path.join(&manifest.tokenizer.source);
    let vocab_file = File::open(&vocab_file_path).context("failed to open tokenizer source")?;
    let vocab_map: HashMap<String, u64> =
        serde_json::from_reader(vocab_file).context("failed to parse tokenizer source")?;

    // Convert vocab to flat list
    let max_index = vocab_map
        .values()
        .cloned()
        .max()
        .context("no entries in vocab")?;
    let mut vocab = vec![String::new(); max_index as usize];
    for (token, index) in vocab_map {
        vocab[index as usize - 1] = token;
    }

    // Insert tokenizer into metadata
    metadata.push((
        "tokenizer.ggml.model".to_string(),
        MetadataValue::String("rwkv".to_string()),
    ));
    metadata.push((
        "tokenizer.ggml.tokens".to_string(),
        MetadataValue::Array(MetadataArray::String(vocab)),
    ));

    // Generate and write the GGUF header
    write_header(&mut output, metadata, &tensors)?;

    // Perform tensor conversion
    println!("converting tensors");
    tensors::convert(&mut output, &mut tensors_source_file, &tensors)?;

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

fn prepare_metadata(manifest: &Manifest) -> Result<Vec<(String, MetadataValue)>, Error> {
    let mut metadata = Vec::new();

    for (key, value) in &manifest.metadata {
        let value = match value {
            Value::String(value) => MetadataValue::String(value.clone()),
            Value::Number(value) => {
                // TODO: We need a better solution for different numeric types, which kind it is can
                //  be important for GGUF
                let Some(value) = value.as_u64() else {
                    bail!("unsupported numeric metadata value for {:?}", key);
                };

                MetadataValue::UInt64(value)
            }
            _ => bail!("unsupported metadata value for {:?}", key),
        };

        metadata.push((key.clone(), value));
    }

    Ok(metadata)
}

fn prepare_tensors(manifest: &Manifest) -> Result<Vec<ConvertTensorInfo>, Error> {
    let mut tensors = Vec::new();
    let mut next_offset = 0;

    for (name, value) in &manifest.tensors.entries {
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
        let task = ConvertTensorInfo {
            name: name.clone(),
            source: value.source.clone(),
            dimensions,
            offset: next_offset,
        };
        tensors.push(task);

        // Figure out the next offset
        next_offset += scalars * 2;
        next_offset = align_offset(next_offset);
    }

    Ok(tensors)
}

fn write_header(
    target: &mut File,
    metadata: Vec<(String, MetadataValue)>,
    tensors: &[ConvertTensorInfo],
) -> Result<(), Error> {
    println!("writing header");

    let mut header = Header::default();

    header.metadata = metadata;
    apply_header_tensors(&mut header, tensors)?;

    // Write the prepared header
    gguf_swiss::write_header(target, &header)?;

    Ok(())
}

fn apply_header_tensors(header: &mut Header, tensors: &[ConvertTensorInfo]) -> Result<(), Error> {
    for tensor in tensors {
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
