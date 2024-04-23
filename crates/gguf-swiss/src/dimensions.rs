use std::fmt::{Display, Formatter};

use anyhow::{bail, Error};

/// Stack-friendly encoding of GGUF tensor dimensions.
///
/// Limited to 4 dimensions, which is currently the maximum supported.
/// All not used dimensions will be zero, which tensor dimensions normally cannot be.
///
/// The order of dimensions in GGUF is `Width x Height x Channel x Batch`.
#[derive(Default, Debug, PartialEq, Clone, Copy)]
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
