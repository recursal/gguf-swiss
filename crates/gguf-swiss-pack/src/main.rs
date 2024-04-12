mod manifest;
mod safetensors;
mod tasks;

use std::{fs::File, path::PathBuf};

use anyhow::{bail, Context, Error};
use clap::Parser;
use gguf_swiss::{Header, MetadataValue, TensorInfo};

use crate::manifest::Manifest;

fn main() -> Result<(), Error> {
    let args = Args::parse();

    let manifest_path = PathBuf::from(args.manifest);
    let source_path = PathBuf::from(args.source);
    let output = PathBuf::from(args.output);

    // Validate paths
    if !manifest_path.is_file() {
        bail!("manifest path is not a valid file");
    }
    if !source_path.is_dir() {
        bail!("model path is not a valid directory");
    }

    // Load the manifest that describes how to perform conversion
    println!("loading manifest");
    let manifest =
        manifest::read_manifest(&manifest_path).context("failed to load packaging manifest")?;

    // Perform conversion
    convert_from_manifest(&manifest, &source_path, &output).context("failed to convert")?;

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

    /// Path to the directory to read source files from.
    #[arg(long)]
    source: String,

    /// Path to the output file.
    output: String,
}

fn convert_from_manifest(
    manifest: &Manifest,
    source_path: &PathBuf,
    output: &PathBuf,
) -> Result<(), Error> {
    // Load and process tasks
    let mut tasks = tasks::load(&manifest.tasks)?;
    let (metadata, tensors) = tasks::process(&mut tasks, source_path.to_path_buf())?;

    // Prepare output file, and write the GGUF header
    let mut output = File::create(output)?;
    write_header(&mut output, metadata, tensors)?;

    // Perform tensor conversion
    tasks::write_tensors(&mut tasks, source_path, &mut output)?;

    Ok(())
}

fn write_header(
    target: &mut File,
    metadata: Vec<(String, MetadataValue)>,
    tensors: Vec<TensorInfo>,
) -> Result<(), Error> {
    println!("writing header");

    let mut header = Header::default();

    header.metadata = metadata;
    header.tensors = tensors;

    // Write the prepared header
    gguf_swiss::write_header(target, &header)?;

    Ok(())
}
