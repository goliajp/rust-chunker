/// What can go wrong in chunkedrs.
///
/// Splitting itself never fails. Errors come from choosing an encoding by
/// name, from vectors handed back to [`Semantic::finish`](crate::Semantic::finish),
/// and from the embedding call of [`Chunker::split_semantic`](crate::Chunker::split_semantic).
#[derive(Debug)]
#[non_exhaustive]
pub enum Error {
    /// The model name maps to no encoding this build has: tiktoken does not
    /// know it, or its vocabulary feature is not enabled.
    UnknownModel(String),
    /// No encoding of this name in this build: a typo, or a vocabulary
    /// feature that is not enabled.
    UnknownEncoding(String),
    /// The vectors handed to [`Semantic::finish`](crate::Semantic::finish)
    /// are not one vector of `dim` values per input.
    Vectors {
        /// Number of inputs the vectors were meant for.
        inputs: usize,
        /// Number of `f32` values received.
        values: usize,
        /// Dimension given with them.
        dim: usize,
    },
    /// The embedding call failed.
    #[cfg(feature = "semantic")]
    Embed(embedrs::Error),
}

impl std::fmt::Display for Error {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Error::UnknownModel(name) => write!(
                f,
                "no encoding for model {name:?} in this build (unknown, or its vocab feature is off)"
            ),
            Error::UnknownEncoding(name) => write!(
                f,
                "no encoding {name:?} in this build (unknown, or its vocab feature is off)"
            ),
            Error::Vectors {
                inputs,
                values,
                dim,
            } => write!(
                f,
                "expected {inputs} vectors of {dim} values, got {values} values"
            ),
            #[cfg(feature = "semantic")]
            Error::Embed(e) => write!(f, "embedding error: {e}"),
        }
    }
}

impl std::error::Error for Error {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            #[cfg(feature = "semantic")]
            Error::Embed(e) => Some(e),
            _ => None,
        }
    }
}

#[cfg(feature = "semantic")]
impl From<embedrs::Error> for Error {
    fn from(e: embedrs::Error) -> Self {
        Error::Embed(e)
    }
}

/// Result type for chunkedrs operations.
pub type Result<T> = std::result::Result<T, Error>;
