use anyhow::{bail, Context, Error};
use gguf_swiss::{MetadataArray, MetadataValue};
use serde::Deserialize;
use toml::Table;

use crate::tasks::{PackTask, ProcessContext};

pub struct ConvertRwkvTokenizerTask {
    manifest: ConvertRwkvTokenizerManifest,
}

impl ConvertRwkvTokenizerTask {
    pub fn new(manifest: &Table) -> Result<Self, Error> {
        let manifest = manifest
            .clone()
            .try_into()
            .context("failed to parse manifest")?;

        let value = Self { manifest };
        Ok(value)
    }
}

impl PackTask for ConvertRwkvTokenizerTask {
    fn process(&mut self, ctx: &mut ProcessContext) -> Result<(), Error> {
        // Load tokenizer/vocab file
        let vocab_file_path = ctx.source_root().join(&self.manifest.source);
        let vocab_raw =
            std::fs::read_to_string(&vocab_file_path).context("failed to open vocab")?;
        let vocab = parse_vocab(&vocab_raw)?;

        // Insert tokenizer into metadata
        ctx.push_metadata_str("tokenizer.ggml.model", "rwkv");
        ctx.push_metadata_value(
            "tokenizer.ggml.tokens",
            MetadataValue::Array(MetadataArray::String(vocab)),
        );

        // TODO: Correct for token count mismatch, some models have additional placeholder tokens

        Ok(())
    }
}

#[derive(Deserialize, Debug)]
struct ConvertRwkvTokenizerManifest {
    pub source: String,

    #[allow(dead_code)]
    pub token_count: u64,
}

pub fn parse_vocab(raw: &str) -> Result<Vec<Vec<u8>>, Error> {
    let mut vocab = Vec::new();

    for line in raw.lines() {
        let token = parse_vocab_line(line)?;

        if token.is_empty() {
            bail!("empty tokens not allowed");
        }

        vocab.push(token);
    }

    Ok(vocab)
}

fn parse_vocab_line(line: &str) -> Result<Vec<u8>, Error> {
    // Trim start and end quotations
    if !(line.starts_with("b'") || line.starts_with("b\""))
        || !(line.ends_with("'") || line.ends_with("\""))
    {
        bail!("invalid tokenizer format");
    }

    let trimmed = &line[2..line.len() - 1];

    // For the time being, we intentionally do *not* unescape the string.
    // The string will be unescaped inside the llama.cpp tokenizer implementation, the tokens
    // themselves currently have to be valid UTF-8, and vocabs including *partial* codepoints
    // aren't.
    // This issue affects RWKV's vocab for example.

    Ok(trimmed.as_bytes().to_vec())
}
