# GGUF Swiss Army Knife

A small set of command line utilities and libraries for working with GGUF files.

## Binaries

### gguf-swiss-info

Model information printer utility.
Reads a GGUF file, and outputs a summary of its metadata and tensors to stdout.

### gguf-swiss-pack

> This project is in very early development, and not ready to be used in most situations.
> Expect the manifest file format to change frequently, and bugs in model output.
> 
> The development goal of this utility is to convert RWKV models to GGUF format for llama.cpp.
> However, this utility can convert tensors from any safetensors format model.

Model packaging utility.
Converts model files to GGUF.
