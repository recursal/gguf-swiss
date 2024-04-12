use anyhow::{Context, Error};
use serde::Deserialize;
use toml::Table;

use crate::tasks::{PackTask, ProcessContext};

pub struct AddModelCardTask {
    manifest: AddModelCardManifest,
}

impl AddModelCardTask {
    pub fn new(manifest: &Table) -> Result<Self, Error> {
        let manifest = manifest
            .clone()
            .try_into()
            .context("failed to parse manifest")?;

        let value = Self { manifest };
        Ok(value)
    }
}

impl PackTask for AddModelCardTask {
    fn process(&mut self, ctx: &mut ProcessContext) -> Result<(), Error> {
        let m = &self.manifest;

        ctx.push_metadata_str("general.name", &m.name);
        ctx.push_metadata_str("general.author", &m.author);
        ctx.push_metadata_str("general.description", &m.description);
        ctx.push_metadata_str("general.license", &m.license);
        ctx.push_metadata_str("general.architecture", &m.architecture);

        Ok(())
    }
}

#[derive(Deserialize, Debug)]
struct AddModelCardManifest {
    pub name: String,

    pub author: String,

    pub description: String,

    pub license: String,

    pub architecture: String,
}
