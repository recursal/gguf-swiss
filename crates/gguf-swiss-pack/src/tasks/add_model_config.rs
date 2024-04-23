use anyhow::{Context, Error};
use serde::Deserialize;
use toml::Table;

use crate::tasks::{PackTask, ProcessContext};

pub struct AddModelConfigTask {
    manifest: AddLlamaConfigManifest,
}

impl AddModelConfigTask {
    pub fn new(manifest: &Table) -> Result<Self, Error> {
        let manifest = manifest
            .clone()
            .try_into()
            .context("failed to parse manifest")?;

        let value = Self { manifest };
        Ok(value)
    }
}

impl PackTask for AddModelConfigTask {
    fn process(&mut self, ctx: &mut ProcessContext) -> Result<(), Error> {
        let m = &self.manifest;

        ctx.push_metadata_str("general.architecture", &m.architecture);
        let k = |k: &str| format!("{}.{}", m.architecture, k);

        ctx.push_metadata_u32(k("context_length"), m.context_length);
        ctx.push_metadata_u32(k("embedding_length"), m.embedding_length);
        ctx.push_metadata_u32(k("block_count"), m.block_count);
        ctx.push_metadata_u32(k("feed_forward_length"), m.feed_forward_length);
        ctx.push_metadata_u32(k("attention.head_count"), m.attention_head_count);
        ctx.push_metadata_f32(k("attention.layer_norm_epsilon"), m.layer_norm_epsilon);

        // TODO: Configurable, these are placeholders necessary for RWKV to load right now
        ctx.push_metadata_u32(k("ssm.state_size"), 1);
        ctx.push_metadata_u32(k("ssm.inner_size"), 1);

        Ok(())
    }
}

#[derive(Deserialize, Debug)]
struct AddLlamaConfigManifest {
    pub architecture: String,

    pub context_length: u32,

    pub embedding_length: u32,

    pub block_count: u32,

    pub feed_forward_length: u32,

    pub attention_head_count: u32,

    pub layer_norm_epsilon: f32,
}
