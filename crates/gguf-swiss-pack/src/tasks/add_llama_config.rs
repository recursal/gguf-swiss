use anyhow::{Context, Error};
use serde::Deserialize;
use toml::Table;

use crate::tasks::{PackTask, ProcessContext};

pub struct AddLlamaConfigTask {
    manifest: AddLlamaConfigManifest,
}

impl AddLlamaConfigTask {
    pub fn new(manifest: &Table) -> Result<Self, Error> {
        let manifest = manifest
            .clone()
            .try_into()
            .context("failed to parse manifest")?;

        let value = Self { manifest };
        Ok(value)
    }
}

impl PackTask for AddLlamaConfigTask {
    fn process(&mut self, ctx: &mut ProcessContext) -> Result<(), Error> {
        let m = &self.manifest;

        // TODO: prefix needs to be configurable

        ctx.push_metadata_u64("rwkv5.context_length", m.context_length);
        ctx.push_metadata_u64("rwkv5.embedding_length", m.embedding_length);
        ctx.push_metadata_u64("rwkv5.block_count", m.block_count);
        ctx.push_metadata_u64("rwkv5.feed_forward_length", m.feed_forward_length);
        ctx.push_metadata_u64("rwkv5.head_count", m.head_count);
        ctx.push_metadata_u64("rwkv5.head_count_kv", m.head_count_kv);
        ctx.push_metadata_f64("rwkv5.layer_norm_epsilon", m.layer_norm_epsilon);

        Ok(())
    }
}

#[derive(Deserialize, Debug)]
struct AddLlamaConfigManifest {
    pub context_length: u64,

    pub embedding_length: u64,

    pub block_count: u64,

    pub feed_forward_length: u64,

    pub head_count: u64,

    pub head_count_kv: u64,

    pub layer_norm_epsilon: f64,
}
