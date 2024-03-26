mod dimensions;
mod metadata;
mod read;
mod write;

use std::collections::HashMap;

pub use crate::{
    dimensions::TensorDimensions,
    metadata::{MetadataArray, MetadataType, MetadataValue},
    read::read_header,
    write::write_header,
};

const MAGIC_NUMBER: [u8; 4] = [0x47, 0x47, 0x55, 0x46];

#[derive(Debug, Default, Clone)]
pub struct Header {
    pub metadata: HashMap<String, MetadataValue>,
    pub tensors: HashMap<String, TensorInfo>,
}

/// Info about a tensor inside a GGUF file.
#[derive(Debug, Clone)]
pub struct TensorInfo {
    /// The encoding of the tensor's values.
    pub tensor_type: TensorType,

    /// The dimensions of the tensor.
    pub dimensions: TensorDimensions,

    /// Offset of the tensor's values, relative to the start of the tensor data.
    pub offset: u64,
}

#[allow(non_camel_case_types)]
#[repr(u32)]
#[derive(Debug, Clone, Copy)]
pub enum TensorType {
    F32 = 0,
    F16 = 1,
    Q4_0 = 2,
    Q4_1 = 3,
    Q5_0 = 6,
    Q5_1 = 7,
    Q8_0 = 8,
    Q8_1 = 9,
    Q2_K = 10,
    Q3_K = 11,
    Q4_K = 12,
    Q5_K = 13,
    Q6_K = 14,
    Q8_K = 15,
    I8 = 16,
    I16 = 17,
    I32 = 18,
    Count = 19,
}

impl TensorType {
    fn from_u32(value: u32) -> Option<Self> {
        let value = match value {
            0 => Self::F32,
            1 => Self::F16,
            2 => Self::Q4_0,
            3 => Self::Q4_1,
            6 => Self::Q5_0,
            7 => Self::Q5_1,
            8 => Self::Q8_0,
            9 => Self::Q8_1,
            10 => Self::Q2_K,
            11 => Self::Q3_K,
            12 => Self::Q4_K,
            13 => Self::Q5_K,
            14 => Self::Q6_K,
            15 => Self::Q8_K,
            16 => Self::I8,
            17 => Self::I16,
            18 => Self::I32,
            19 => Self::Count,
            _ => return None,
        };

        Some(value)
    }
}

/// Align an offset value to the next aligned value.
///
/// Just to be safe, this is hardcoded to align to 32 for now, which is the GGUF default if nothing
/// else is specified.
pub fn align_offset(offset: u64) -> u64 {
    return offset + (32 - (offset % 32)) % 32;
}
