use std::{collections::HashMap, path::Path};

use anyhow::{bail, Context, Error};
use serde::Deserialize;
use toml::{Table, Value};

pub fn read_manifest(path: &Path) -> Result<Manifest, Error> {
    // Try opening the manifest
    let manifest_str = std::fs::read_to_string(path).context("failed to open")?;
    let manifest_value: Value = toml::from_str(&manifest_str).context("failed to parse")?;

    // Validate matching manifest version
    if manifest_value.get("manifest_version") != Some(&Value::from(0)) {
        bail!("not a manifest or unsupported version");
    }

    let value: Manifest = manifest_value.try_into().context("failed to decode")?;

    Ok(value)
}

#[derive(Deserialize, Debug)]
pub struct Manifest {
    pub tasks: HashMap<String, Table>,
}
