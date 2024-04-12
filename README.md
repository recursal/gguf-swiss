# GGUF Swiss Army Knife

A small set of command line utilities and libraries for working with GGUF files.

## Binaries

### gguf-swiss-info

Model information reader utility.
Reads a GGUF file, and outputs a summary of its metadata and tensors to stdout, in markdown format.

#### Usage

```
$ gguf-swiss-info model.gguf
```

### gguf-swiss-pack

> This project is in very early development, and not ready to be used in most situations.
> Expect the manifest file format to change frequently, and bugs in model output.
>
> The development goal of this utility is to convert RWKV models to GGUF format for llama.cpp.
> However, this utility can be extended to convert tensors from any safetensors format model.

Model packaging utility.
Converts model files to GGUF.

#### Usage

```
$ gguf-swiss-pack \
    --manifest manifest.toml \
    --source /source/root/directory \
    model.gguf
```

Example manifest files included in `/data`.

## Safety

An effort has been made to avoid unsafe code and unsafe dependencies.
The core GGUF reader/writer library is entirely safe Rust code.
