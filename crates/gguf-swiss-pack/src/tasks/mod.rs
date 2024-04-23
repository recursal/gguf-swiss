use std::{
    collections::HashMap,
    fs::File,
    path::{Path, PathBuf},
};

use anyhow::{bail, Context, Error};
use gguf_swiss::{MetadataValue, TensorInfo};
use toml::Table;

use crate::tasks::{
    add_model_card::AddModelCardTask, add_model_config::AddModelConfigTask,
    convert_rwkv_tokenizer::ConvertRwkvTokenizerTask, convert_safetensors::ConvertSafetensorsTask,
};

mod add_model_card;
mod add_model_config;
mod convert_rwkv_tokenizer;
mod convert_safetensors;

pub fn load(manifest: &HashMap<String, Table>) -> Result<Vec<TaskEntry>, Error> {
    println!("loading tasks");

    let mut tasks = Vec::new();

    for (key, value) in manifest {
        let task =
            load_task(key, value).with_context(|| format!("failed to load task \"{}\"", key))?;

        tasks.push(task);
    }

    Ok(tasks)
}

fn load_task(key: &str, manifest: &Table) -> Result<TaskEntry, Error> {
    let name = manifest.get("task").context("missing key \"task\"")?;
    let name = name.as_str().context("\"task\" not string")?;
    let name = name.to_string();

    println!("loading task \"{}\" -> \"{}\"", key, name);

    let task: Box<dyn PackTask> = match name.as_str() {
        "add-model-card" => Box::new(AddModelCardTask::new(manifest)?),
        "add-model-config" => Box::new(AddModelConfigTask::new(manifest)?),
        "convert-rwkv-tokenizer" => Box::new(ConvertRwkvTokenizerTask::new(manifest)?),
        "convert-safetensors" => Box::new(ConvertSafetensorsTask::new(manifest)?),
        value => bail!("unknown task type \"{}\"", value),
    };

    let value = TaskEntry { name, task };
    Ok(value)
}

pub fn process(
    tasks: &mut [TaskEntry],
    source_root: PathBuf,
) -> Result<(Vec<(String, MetadataValue)>, Vec<TensorInfo>), Error> {
    println!("processing tasks");

    let mut ctx = ProcessContext {
        source_root,
        metadata: Vec::new(),
        tensors: Vec::new(),
    };

    for entry in tasks {
        println!("processing \"{}\"", entry.name);
        entry.task.process(&mut ctx)?;
    }

    Ok((ctx.metadata, ctx.tensors))
}

pub fn write_tensors(
    tasks: &mut [TaskEntry],
    source_root: &Path,
    output: &mut File,
) -> Result<(), Error> {
    println!("writing tensors");

    for entry in tasks {
        entry.task.write_tensors(source_root, output)?;
    }

    Ok(())
}

pub struct TaskEntry {
    name: String,
    task: Box<dyn PackTask>,
}

trait PackTask {
    fn process(&mut self, _ctx: &mut ProcessContext) -> Result<(), Error> {
        Ok(())
    }

    fn write_tensors(&mut self, _source_root: &Path, _output: &mut File) -> Result<(), Error> {
        Ok(())
    }
}

struct ProcessContext {
    source_root: PathBuf,
    metadata: Vec<(String, MetadataValue)>,
    tensors: Vec<TensorInfo>,
}

impl ProcessContext {
    fn source_root(&self) -> &Path {
        &self.source_root
    }

    fn push_metadata_str(&mut self, key: impl ToString, value: &str) {
        let value = value.as_bytes().to_vec();
        self.push_metadata_value(key, MetadataValue::String(value));
    }

    fn push_metadata_u32(&mut self, key: impl ToString, value: u32) {
        self.push_metadata_value(key, MetadataValue::UInt32(value));
    }

    #[allow(dead_code)]
    fn push_metadata_u64(&mut self, key: impl ToString, value: u64) {
        self.push_metadata_value(key, MetadataValue::UInt64(value));
    }

    fn push_metadata_f32(&mut self, key: impl ToString, value: f32) {
        self.push_metadata_value(key, MetadataValue::Float32(value));
    }

    fn push_metadata_value(&mut self, key: impl ToString, value: MetadataValue) {
        self.metadata.push((key.to_string(), value));
    }
}
