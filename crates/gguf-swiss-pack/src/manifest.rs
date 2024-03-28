use std::{collections::HashMap, fs::File, path::Path};

use anyhow::{bail, Context, Error};
use serde::Deserialize;
use serde_json::Value;

pub fn read_manifest(path: &Path) -> Result<Manifest, Error> {
    // Try opening the manifest
    let manifest_file = File::open(path).context("failed to open")?;
    let manifest_value: Value =
        serde_json::from_reader(manifest_file).context("failed to parse")?;

    // Validate matching manifest version
    if manifest_value.get("manifest_version") != Some(&Value::from(0)) {
        bail!("not a manifest or unsupported version");
    }

    let value: Manifest = serde_json::from_value(manifest_value).context("failed to decode")?;

    Ok(value)
}

#[derive(Deserialize, Debug)]
pub struct Manifest {
    /// Metadata to be inserted, in addition to generated values.
    pub metadata: HashMap<String, Value>,

    /// Tokenizer source data to be processed.
    pub tokenizer: TokenizerManifest,

    /// Tensors source data to be processed.
    pub tensors: TensorsManifest,
}

#[derive(Deserialize, Debug)]
pub struct TokenizerManifest {
    pub source: String,

    #[serde(rename = "type")]
    pub ty: String,
}

#[derive(Deserialize, Debug)]
pub struct TensorsManifest {
    pub sources: Vec<String>,

    pub entries: HashMap<String, TensorManifest>,
}

#[derive(Deserialize, Debug)]
pub struct TensorManifest {
    pub source: String,

    #[serde(rename = "type")]
    pub ty: String,

    // TODO: Infer dimensions from source.
    pub dimensions: Vec<u64>,
}
