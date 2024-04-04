use anyhow::{bail, Error};

pub fn parse_vocab(raw: &str) -> Result<Vec<Vec<u8>>, Error> {
    let mut vocab = Vec::new();

    for line in raw.lines() {
        let mut token = parse_vocab_line(line)?;

        if token.is_empty() {
            bail!("empty tokens not allowed");
        }

        llamacpp_byte_workaround(&mut token);

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
    let unescaped = unescape(trimmed)?;

    Ok(unescaped)
}

fn unescape(value: &str) -> Result<Vec<u8>, Error> {
    // The original string can contain unicode codepoints escaped as multiple \xNN sequences. This
    // means it's non-trivial to work on the string one codepoint at a time.
    // Additionally, tokens can contain partial codepoints.
    // The only solution we have left is to deal with this as byte arrays.
    let mut output = Vec::new();

    let mut escaping = false;
    let mut hexing = 0; // abracadabra
    let mut hex_value = String::new();

    for c in value.chars() {
        if hexing != 0 {
            hex_value.push(c);
            hexing -= 1;

            // If we completed parsing a single hex escape sequence
            if hexing == 0 {
                let value = u8::from_str_radix(&hex_value, 16)?;
                output.push(value);
                hex_value.clear();
            }

            continue;
        }

        if escaping {
            match c {
                '\\' => output.push(b'\\'),
                '\'' => output.push(b'\''),
                '\"' => output.push(b'\"'),
                't' => output.push(b'\t'),
                'r' => output.push(b'\r'),
                'n' => output.push(b'\n'),
                'x' => hexing = 2,
                _ => bail!("unknown escape sequence"),
            }

            escaping = false;
            continue;
        }

        if c == '\\' {
            escaping = true;
            continue;
        }

        // Convert this codepoint to bytes
        let mut bytes = [0; 4];
        let slice = c.encode_utf8(&mut bytes);
        output.extend_from_slice(slice.as_bytes());
    }

    Ok(output)
}

/// llama.cpp doesn't handle byte tokens that are not valid UTF-8 correctly, for now we encode these
/// using raw escape codes. This is very fragile and makes assumptions about the tokenizer to avoid
/// these going into the input, but for now this is the only way to solve this.
fn llamacpp_byte_workaround(token: &mut Vec<u8>) {
    if token.len() != 1 {
        return;
    }
    let value = token[0];

    if value == 0 {
        *token = b"\\x00".to_vec();
    }

    if value >= 0b1000_0000 {
        let formatted = format!("\\x{:02x}", value);
        *token = formatted.into_bytes();
    }
}
