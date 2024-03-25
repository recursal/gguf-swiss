mod read;
mod write;

use anyhow::{bail, Error};
use std::{
    collections::HashMap,
    fmt::{Display, Formatter},
};

pub use crate::{read::read_header, write::write_header};

const MAGIC_NUMBER: [u8; 4] = [0x47, 0x47, 0x55, 0x46];

#[derive(Debug, Default, Clone)]
pub struct Header {
    pub metadata: HashMap<String, MetadataValue>,
    pub tensors: HashMap<String, TensorInfo>,
}

#[derive(Debug, Clone)]
pub enum MetadataValue {
    UInt8(u8),
    Int8(i8),
    UInt16(u16),
    Int16(i16),
    UInt32(u32),
    Int32(i32),
    Float32(f32),
    Bool(bool),
    String(String),
    Array(MetadataArray),
    UInt64(u64),
    Int64(i64),
    Float64(f64),
}

#[derive(Debug, Clone)]
pub enum MetadataArray {
    UInt8(Vec<u8>),
    Int8(Vec<i8>),
    UInt16(Vec<u16>),
    Int16(Vec<i16>),
    UInt32(Vec<u32>),
    Int32(Vec<i32>),
    Float32(Vec<f32>),
    Bool(Vec<bool>),
    String(Vec<String>),
    Array(Vec<MetadataArray>),
    UInt64(Vec<u64>),
    Int64(Vec<i64>),
    Float64(Vec<f64>),
}

#[repr(u32)]
#[derive(Debug, Clone, Copy)]
pub enum MetadataType {
    UInt8 = 0,
    Int8 = 1,
    UInt16 = 2,
    Int16 = 3,
    UInt32 = 4,
    Int32 = 5,
    Float32 = 6,
    Bool = 7,
    String = 8,
    Array = 9,
    UInt64 = 10,
    Int64 = 11,
    Float64 = 12,
}

impl MetadataType {
    fn from_u32(value: u32) -> Option<Self> {
        let value = match value {
            0 => Self::UInt8,
            1 => Self::Int8,
            2 => Self::UInt16,
            3 => Self::Int16,
            4 => Self::UInt32,
            5 => Self::Int32,
            6 => Self::Float32,
            7 => Self::Bool,
            8 => Self::String,
            9 => Self::Array,
            10 => Self::UInt64,
            11 => Self::Int64,
            12 => Self::Float64,
            _ => return None,
        };

        Some(value)
    }
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

/// Stack-friendly encoding of GGUF tensor dimensions.
///
/// Limited to 4 dimensions, which is currently the maximum supported.
/// All not used dimensions will be zero, which tensor dimensions normally cannot be.
///
/// The order of dimensions in GGUF is `Width x Height x Channel x Batch`.
#[derive(Default, Debug, Clone, Copy)]
pub struct TensorDimensions(pub [u64; 4]);

impl TensorDimensions {
    /// Create new dimensions from values with the width last.
    ///
    /// GGUF dimensions are width-first, but for example safetensors are width-last.
    /// This lets you convert them trivially.
    pub fn from_width_last(source: &[u64]) -> Result<Self, Error> {
        if source.len() > 4 {
            bail!("source dimensions too long")
        }

        let mut value = TensorDimensions::default();

        for (i, v) in source.iter().rev().enumerate() {
            value.0[i] = *v;
        }

        Ok(value)
    }

    pub fn count(&self) -> usize {
        self.0.iter().position(|v| *v == 0).unwrap_or(4)
    }

    /// Amount of scalars in total.
    pub fn total(&self) -> u64 {
        let mut value = self.0[0];

        for i in 1..self.count() {
            value *= self.0[i];
        }

        value
    }
}

impl Display for TensorDimensions {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        write!(f, "[")?;
        for i in 0..self.count() {
            write!(f, "{}", self.0[i])?;

            if i != self.count() - 1 {
                write!(f, ", ")?;
            }
        }
        write!(f, "]")?;
        Ok(())
    }
}

/// Align an offset value to the next aligned value.
///
/// Just to be safe, this is hardcoded to align to 32 for now, which is the GGUF default if nothing
/// else is specified.
pub fn align_offset(offset: u64) -> u64 {
    return offset + (32 - (offset % 32)) % 32;
}

#[cfg(test)]
mod tests {
    use crate::TensorDimensions;

    #[test]
    fn tensor_dimensions_count() {
        let dimensions = TensorDimensions([1, 2, 0, 0]);
        assert_eq!(dimensions.count(), 2);

        let dimensions = TensorDimensions([1, 2, 3, 4]);
        assert_eq!(dimensions.count(), 4);

        let dimensions = TensorDimensions([0, 0, 0, 0]);
        assert_eq!(dimensions.count(), 0);
    }

    #[test]
    fn tensor_dimensions_total() {
        let dimensions = TensorDimensions([2, 4, 0, 0]);
        assert_eq!(dimensions.total(), 8);

        let dimensions = TensorDimensions([1, 2, 3, 4]);
        assert_eq!(dimensions.total(), 24);

        let dimensions = TensorDimensions([0, 0, 0, 0]);
        assert_eq!(dimensions.total(), 0);
    }
}
