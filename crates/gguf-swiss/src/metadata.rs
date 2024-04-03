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
    pub fn from_u32(value: u32) -> Option<Self> {
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
    String(Vec<u8>),
    Array(MetadataArray),
    UInt64(u64),
    Int64(i64),
    Float64(f64),
}

impl MetadataValue {
    pub fn ty(&self) -> MetadataType {
        match self {
            Self::UInt8(_) => MetadataType::UInt8,
            Self::Int8(_) => MetadataType::Int8,
            Self::UInt16(_) => MetadataType::UInt16,
            Self::Int16(_) => MetadataType::Int16,
            Self::UInt32(_) => MetadataType::UInt32,
            Self::Int32(_) => MetadataType::Int32,
            Self::Float32(_) => MetadataType::Float32,
            Self::Bool(_) => MetadataType::Bool,
            Self::String(_) => MetadataType::String,
            Self::Array(_) => MetadataType::Array,
            Self::UInt64(_) => MetadataType::UInt64,
            Self::Int64(_) => MetadataType::Int64,
            Self::Float64(_) => MetadataType::Float64,
        }
    }
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
    String(Vec<Vec<u8>>),
    Array(Vec<MetadataArray>),
    UInt64(Vec<u64>),
    Int64(Vec<i64>),
    Float64(Vec<f64>),
}

impl MetadataArray {
    pub fn ty(&self) -> MetadataType {
        match self {
            Self::UInt8(_) => MetadataType::UInt8,
            Self::Int8(_) => MetadataType::Int8,
            Self::UInt16(_) => MetadataType::UInt16,
            Self::Int16(_) => MetadataType::Int16,
            Self::UInt32(_) => MetadataType::UInt32,
            Self::Int32(_) => MetadataType::Int32,
            Self::Float32(_) => MetadataType::Float32,
            Self::Bool(_) => MetadataType::Bool,
            Self::String(_) => MetadataType::String,
            Self::Array(_) => MetadataType::Array,
            Self::UInt64(_) => MetadataType::UInt64,
            Self::Int64(_) => MetadataType::Int64,
            Self::Float64(_) => MetadataType::Float64,
        }
    }

    pub fn len(&self) -> usize {
        match self {
            Self::UInt8(v) => v.len(),
            Self::Int8(v) => v.len(),
            Self::UInt16(v) => v.len(),
            Self::Int16(v) => v.len(),
            Self::UInt32(v) => v.len(),
            Self::Int32(v) => v.len(),
            Self::Float32(v) => v.len(),
            Self::Bool(v) => v.len(),
            Self::String(v) => v.len(),
            Self::Array(v) => v.len(),
            Self::UInt64(v) => v.len(),
            Self::Int64(v) => v.len(),
            Self::Float64(v) => v.len(),
        }
    }
}
