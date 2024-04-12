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
        let mut vocab = parse_vocab(&vocab_raw)?;

        // Generate a token type map, to tell llama.cpp which ones need to actually be matched in
        // which ways. Defaulting to 1, which is "normal".
        let mut token_type = vec![1u32; vocab.len()];

        // Overwrite the type of token 0 for RWKV as 3 (control token)
        token_type[0] = 3;

        // Extend tokens if necessary, models can provide additional placeholder tokens, mark these
        // as 5 (unused)
        let remainder = self.manifest.token_count as i32 - vocab.len() as i32;
        if remainder < 0 {
            bail!("\"token_count\" less than provided in the vocab");
        }
        for i in 0..remainder {
            // Every token has to be unique for llama.cpp
            vocab.push(format!("<unused {}>", i).into_bytes());
            token_type.push(5);
        }

        // Insert tokenizer into metadata
        ctx.push_metadata_str("tokenizer.ggml.model", "rwkv");

        let vocab_value = MetadataValue::Array(MetadataArray::String(vocab));
        ctx.push_metadata_value("tokenizer.ggml.tokens", vocab_value);

        let vocab_value = MetadataValue::Array(MetadataArray::UInt32(token_type));
        ctx.push_metadata_value("tokenizer.ggml.token_type", vocab_value);

        Ok(())
    }
}

#[derive(Deserialize, Debug)]
struct ConvertRwkvTokenizerManifest {
    pub source: String,

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
