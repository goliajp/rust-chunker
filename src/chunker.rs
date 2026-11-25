use crate::Chunk;
use crate::builder::{Strategy, resolve, run};

/// A configured splitter to reuse across documents.
///
/// [`chunk()`](crate::chunk) configures one split of one text. A `Chunker`
/// holds the same settings, with the encoder resolved once, and splits any
/// number of texts, one at a time or as a batch on several threads.
///
/// ```rust
/// let chunker = chunkedrs::Chunker::new().max_tokens(64).markdown();
///
/// let one = chunker.split("# Title\n\nSome text.\n");
/// assert_eq!(one[0].section(), Some("# Title"));
///
/// let docs = ["First document.", "Second document."];
/// let all = chunker.split_batch(&docs);
/// assert_eq!(all.len(), 2);
/// assert_eq!(all[1][0].content, "Second document.");
/// ```
#[derive(Clone, Copy)]
pub struct Chunker {
    encoder: &'static tiktoken::CoreBpe,
    max_tokens: usize,
    overlap: usize,
    strategy: Strategy,
}

impl std::fmt::Debug for Chunker {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Chunker")
            .field("max_tokens", &self.max_tokens)
            .field("overlap", &self.overlap)
            .field("strategy", &self.strategy)
            .finish_non_exhaustive()
    }
}

impl Default for Chunker {
    fn default() -> Self {
        Self::new()
    }
}

impl Chunker {
    /// Recursive splitting, 512 max tokens, no overlap, `o200k_base`: the
    /// same defaults as [`chunk()`](crate::chunk).
    pub fn new() -> Self {
        Chunker {
            encoder: resolve(None, None),
            max_tokens: 512,
            overlap: 0,
            strategy: Strategy::Recursive,
        }
    }

    /// Maximum tokens per chunk; see [`ChunkBuilder::max_tokens`](crate::ChunkBuilder::max_tokens).
    #[must_use]
    pub fn max_tokens(mut self, n: usize) -> Self {
        self.max_tokens = n.max(1);
        self
    }

    /// Tokens shared by consecutive chunks; see [`ChunkBuilder::overlap`](crate::ChunkBuilder::overlap).
    #[must_use]
    pub fn overlap(mut self, tokens: usize) -> Self {
        self.overlap = tokens;
        self
    }

    /// The encoder for a model name, resolved now; see
    /// [`ChunkBuilder::model`](crate::ChunkBuilder::model) for the fallback.
    #[must_use]
    pub fn model(mut self, model: &str) -> Self {
        self.encoder = resolve(None, Some(model));
        self
    }

    /// The encoder for an encoding name, resolved now; see
    /// [`ChunkBuilder::encoding`](crate::ChunkBuilder::encoding) for the fallback.
    #[must_use]
    pub fn encoding(mut self, encoding: &str) -> Self {
        self.encoder = resolve(Some(encoding), None);
        self
    }

    /// Split on paragraphs, lines, sentences, clauses and words (the default).
    #[must_use]
    pub fn recursive(mut self) -> Self {
        self.strategy = Strategy::Recursive;
        self
    }

    /// Markdown-aware splitting; see [`ChunkBuilder::markdown`](crate::ChunkBuilder::markdown).
    #[must_use]
    pub fn markdown(mut self) -> Self {
        self.strategy = Strategy::Markdown;
        self
    }

    /// Code-aware splitting; see [`ChunkBuilder::code`](crate::ChunkBuilder::code).
    #[must_use]
    pub fn code(mut self) -> Self {
        self.strategy = Strategy::Code;
        self
    }

    /// HTML-aware splitting; see [`ChunkBuilder::html`](crate::ChunkBuilder::html).
    #[must_use]
    pub fn html(mut self) -> Self {
        self.strategy = Strategy::Html;
        self
    }

    /// Split one text.
    pub fn split(&self, text: &str) -> Vec<Chunk> {
        run(
            text,
            self.encoder,
            self.strategy,
            self.max_tokens,
            self.overlap,
        )
    }

    /// Split many texts on up to [`std::thread::available_parallelism`]
    /// threads. The result is in input order, and each entry equals what
    /// [`split`](Self::split) returns for that text.
    ///
    /// tiktoken also encodes a single text of 32 KiB or more on several
    /// threads. If the batch holds large documents and the process already
    /// keeps every core busy, set `TIKTOKEN_THREADS=1` or call
    /// [`tiktoken::set_parallelism`] to keep tiktoken on the calling thread.
    pub fn split_batch<S: AsRef<str> + Sync>(&self, texts: &[S]) -> Vec<Vec<Chunk>> {
        let threads = std::thread::available_parallelism()
            .map_or(1, |n| n.get())
            .min(texts.len());
        if threads <= 1 {
            return texts.iter().map(|t| self.split(t.as_ref())).collect();
        }
        let per_thread = texts.len().div_ceil(threads);
        std::thread::scope(|scope| {
            let handles: Vec<_> = texts
                .chunks(per_thread)
                .map(|part| {
                    scope.spawn(move || {
                        part.iter()
                            .map(|t| self.split(t.as_ref()))
                            .collect::<Vec<_>>()
                    })
                })
                .collect();
            handles
                .into_iter()
                .flat_map(|h| h.join().unwrap_or_else(|e| std::panic::resume_unwind(e)))
                .collect()
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn split_matches_the_builder() {
        let text = "One sentence here. Another one there.\n\nA second paragraph.";
        for max in [3, 8, 512] {
            assert_eq!(
                Chunker::new().max_tokens(max).overlap(1).split(text),
                crate::chunk(text).max_tokens(max).overlap(1).split()
            );
        }
    }

    #[test]
    fn batch_keeps_input_order() {
        let docs: Vec<String> = (0..37)
            .map(|i| format!("Document {i}. ").repeat(i + 1))
            .collect();
        let chunker = Chunker::new().max_tokens(16);
        let batch = chunker.split_batch(&docs);
        assert_eq!(batch.len(), docs.len());
        for (doc, chunks) in docs.iter().zip(&batch) {
            assert_eq!(chunks, &chunker.split(doc));
        }
        assert!(chunker.split_batch::<&str>(&[]).is_empty());
    }
}
