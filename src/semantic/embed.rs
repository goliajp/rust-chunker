//! Semantic splitting with the embedding done through `embedrs::Embed`
//! (feature `semantic`).

use embedrs::{Embed, InputType};

use super::Semantic;
use crate::{Chunk, Chunker, Error, Result};

impl<'a> Semantic<'a> {
    /// Embed [`inputs`](Self::inputs) with `embedder` in one call and
    /// [`finish`](Self::finish).
    ///
    /// Any [`Embed`] works: an `embedrs::Client` for a remote provider
    /// (which splits the call into requests the provider accepts and sends
    /// them concurrently), a local model, or your own type.
    ///
    /// # Errors
    ///
    /// [`Error::Embed`] when the embedding call fails, [`Error::Vectors`]
    /// when the embedder returns the wrong number of vectors.
    pub async fn embed(self, embedder: &dyn Embed) -> Result<Vec<Chunk<'a>>> {
        if self.inputs.is_empty() {
            return self.finish(&[], 0);
        }
        let result = embedder
            .embed_texts(&self.inputs, Some(InputType::Clustering))
            .await?;
        self.finish(result.as_flat(), result.dim())
    }
}

impl Chunker {
    /// Split `text` where neighbouring sentences stop being similar, with
    /// [`Breakpoints::Percentile`](crate::Breakpoints::Percentile)`(95.0)`.
    /// `chunker.semantic(text).breakpoints(..).embed(embedder)` picks
    /// another rule.
    ///
    /// ```rust,no_run
    /// # async fn run() -> chunkedrs::Result<()> {
    /// let client = embedrs::local();
    /// let chunks = chunkedrs::Chunker::new()
    ///     .max_tokens(256)
    ///     .split_semantic("your long text here...", &client)
    ///     .await?;
    /// # Ok(())
    /// # }
    /// ```
    ///
    /// # Errors
    ///
    /// As [`Semantic::embed`].
    pub async fn split_semantic<'a>(
        &self,
        text: &'a str,
        embedder: &dyn Embed,
    ) -> Result<Vec<Chunk<'a>>> {
        self.semantic(text).embed(embedder).await
    }

    /// [`split_semantic`](Self::split_semantic) for many texts, with one
    /// embedding call for all of their inputs, so the embedder gets full
    /// batches. The result is in input order.
    ///
    /// # Errors
    ///
    /// As [`Semantic::embed`]; one failed call fails the batch.
    pub async fn split_semantic_batch<'a, S: AsRef<str> + Sync>(
        &self,
        texts: &'a [S],
        embedder: &dyn Embed,
    ) -> Result<Vec<Vec<Chunk<'a>>>> {
        let jobs: Vec<Semantic<'a>> = texts.iter().map(|t| self.semantic(t.as_ref())).collect();
        let inputs: Vec<&str> = jobs.iter().flat_map(|j| j.inputs.iter().copied()).collect();
        if inputs.is_empty() {
            return jobs.into_iter().map(|j| j.finish(&[], 0)).collect();
        }
        let result = embedder
            .embed_texts(&inputs, Some(InputType::Clustering))
            .await?;
        let (flat, dim) = (result.as_flat(), result.dim());
        if flat.len() != inputs.len() * dim {
            return Err(Error::Vectors {
                inputs: inputs.len(),
                values: flat.len(),
                dim,
            });
        }
        let mut at = 0;
        jobs.into_iter()
            .map(|j| {
                let n = j.inputs.len() * dim;
                let part = &flat[at..at + n];
                at += n;
                j.finish(part, dim)
            })
            .collect()
    }
}
