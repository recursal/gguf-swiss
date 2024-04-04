use anyhow::{bail, Error};

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
