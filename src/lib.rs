//! # chunkedrs
//!
//! AI-native text chunking — split long documents into token-accurate pieces for
//! embedding and retrieval. Built on [tiktoken](https://crates.io/crates/tiktoken)
//! for precise token counting.
//!
//! ## Design: 用就要好用
//!
//! Three strategies, each done right:
//!
//! | Strategy | Use case | Speed |
//! |----------|----------|-------|
//! | **Recursive** (default) | General text — paragraphs, sentences, words | Fastest |
//! | **Markdown** | Documents with `#` headers — preserves section metadata | Fast |
//! | **Semantic** | High-quality RAG — splits at meaning boundaries via embeddings | Slower (API calls) |
//!
//! ## Quick start
//!
//! ```rust
//! // split with defaults: recursive, 512 max tokens, no overlap
//! let chunks = chunkedrs::chunk("your long text here...").split();
//! for chunk in &chunks {
//!     println!("[{}] {} tokens", chunk.index, chunk.token_count);
//! }
//! ```
//!
//! ## Token-accurate splitting
//!
//! ```rust
//! let chunks = chunkedrs::chunk("your long text here...")
//!     .max_tokens(256)
//!     .overlap(50)
//!     .model("gpt-4o")
//!     .split();
//!
//! // every chunk is guaranteed to have <= 256 tokens
//! assert!(chunks.iter().all(|c| c.token_count <= 256));
//! ```
//!
//! ## Markdown-aware splitting
//!
//! ```rust
//! let markdown = "# Intro\n\nSome text.\n\n## Details\n\nMore text here.\n";
//! let chunks = chunkedrs::chunk(markdown).markdown().split();
//!
//! // each chunk knows which section it belongs to
//! assert_eq!(chunks[0].section.as_deref(), Some("# Intro"));
//! ```
//!
//! ## Semantic splitting
//!
//! With the `semantic` feature enabled, split at meaning boundaries using embeddings:
//!
//! ```rust,ignore
//! let client = embedrs::openai("sk-...");
//! let chunks = chunkedrs::chunk("your long text here...")
//!     .semantic(&client)
//!     .split_async()
//!     .await?;
//! ```

mod chunk;
mod markdown;
pub(crate) mod recursive;
#[cfg(feature = "semantic")]
mod semantic;

pub use chunk::Chunk;

/// find byte offset of a substring within the parent string using pointer arithmetic
pub(crate) fn byte_offset_of(sub: &str, parent: &str) -> usize {
    let sub_ptr = sub.as_ptr() as usize;
    let parent_ptr = parent.as_ptr() as usize;
    debug_assert!(
        sub_ptr >= parent_ptr && sub_ptr <= parent_ptr + parent.len(),
        "substring pointer is not within parent string bounds"
    );
    sub_ptr.saturating_sub(parent_ptr)
}

/// Error types for chunkedrs operations.
#[derive(Debug)]
#[non_exhaustive]
pub enum Error {
    /// Embedding error during semantic chunking.
    #[cfg(feature = "semantic")]
    Embed(embedrs::Error),
}

impl std::fmt::Display for Error {
    #[allow(unused_variables)]
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match *self {
            #[cfg(feature = "semantic")]
            Error::Embed(ref e) => write!(f, "embedding error: {e}"),
            // without semantic feature, Error is uninhabited but non_exhaustive
            // keeps the type valid for future expansion
            #[cfg(not(feature = "semantic"))]
            _ => unreachable!("Error is uninhabited without semantic feature"),
        }
    }
}

impl std::error::Error for Error {}

/// Result type for chunkedrs operations.
pub type Result<T> = std::result::Result<T, Error>;

/// Create a chunk builder for the given text.
///
/// This is the main entry point. Call `.split()` to get chunks with the default
/// strategy (recursive), or chain builder methods to customize:
///
/// ```rust
/// let chunks = chunkedrs::chunk("hello world").split();
/// assert_eq!(chunks.len(), 1);
/// assert_eq!(chunks[0].content, "hello world");
/// ```
pub fn chunk(text: &str) -> ChunkBuilder<'_> {
    ChunkBuilder {
        text,
        max_tokens: 512,
        overlap: 0,
        model_name: None,
        encoding_name: None,
        strategy: Strategy::Recursive,
        #[cfg(feature = "semantic")]
        semantic_client: None,
        #[cfg(feature = "semantic")]
        semantic_threshold: 0.5,
    }
}

/// Strategy for splitting text.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Strategy {
    Recursive,
    Markdown,
    #[cfg(feature = "semantic")]
    Semantic,
}

/// Builder for configuring text chunking.
///
/// Created by [`chunk()`]. Chain methods to configure, then call [`.split()`](ChunkBuilder::split)
/// (sync) or [`.split_async()`](ChunkBuilder::split_async) (semantic).
pub struct ChunkBuilder<'a> {
    text: &'a str,
    max_tokens: usize,
    overlap: usize,
    model_name: Option<&'a str>,
    encoding_name: Option<&'a str>,
    strategy: Strategy,
    #[cfg(feature = "semantic")]
    semantic_client: Option<&'a embedrs::Client>,
    #[cfg(feature = "semantic")]
    semantic_threshold: f64,
}

impl<'a> ChunkBuilder<'a> {
    /// Set the maximum number of tokens per chunk. Default: 512.
    ///
    /// ```rust
    /// let chunks = chunkedrs::chunk("hello world").max_tokens(256).split();
    /// assert!(chunks.iter().all(|c| c.token_count <= 256));
    /// ```
    pub fn max_tokens(mut self, n: usize) -> Self {
        self.max_tokens = n.max(1);
        self
    }

    /// Set the number of overlapping tokens between consecutive chunks. Default: 0.
    ///
    /// Overlap ensures context is not lost at chunk boundaries — important for
    /// retrieval quality in RAG pipelines.
    ///
    /// ```rust
    /// let chunks = chunkedrs::chunk("hello world").overlap(50).split();
    /// ```
    pub fn overlap(mut self, tokens: usize) -> Self {
        self.overlap = tokens;
        self
    }

    /// Set the model name to auto-select the correct tokenizer encoding.
    ///
    /// Uses [`tiktoken::encoding_for_model`] to find the right encoding.
    /// Default: `o200k_base` (GPT-4o, GPT-4-turbo).
    ///
    /// This is independent of [`.encoding()`](ChunkBuilder::encoding). If both are
    /// set, `encoding` takes precedence.
    ///
    /// If the model name is not recognized, falls back to `o200k_base` silently.
    ///
    /// ```rust
    /// let chunks = chunkedrs::chunk("hello world").model("gpt-4o").split();
    /// ```
    pub fn model(mut self, model: &'a str) -> Self {
        self.model_name = Some(model);
        self
    }

    /// Set the tiktoken encoding name directly.
    ///
    /// Use this when you know the exact encoding (e.g. `"cl100k_base"`, `"o200k_base"`).
    /// Takes precedence over [`.model()`](ChunkBuilder::model) if both are set.
    ///
    /// If the encoding name is not recognized, falls back to `o200k_base` silently.
    ///
    /// ```rust
    /// let chunks = chunkedrs::chunk("hello world").encoding("cl100k_base").split();
    /// ```
    pub fn encoding(mut self, encoding: &'a str) -> Self {
        self.encoding_name = Some(encoding);
        self
    }

    /// Use markdown-aware splitting.
    ///
    /// Splits at `#` header boundaries first, then applies recursive splitting
    /// within each section. Each chunk's [`Chunk::section`] field contains the
    /// header it belongs to.
    ///
    /// Note: header lines themselves are stored in `section` metadata, not in
    /// chunk `content`. This means joining all chunk contents will not reproduce
    /// the header lines from the original document.
    ///
    /// ```rust
    /// let md = "# Title\n\nContent here.\n";
    /// let chunks = chunkedrs::chunk(md).markdown().split();
    /// assert_eq!(chunks[0].section.as_deref(), Some("# Title"));
    /// ```
    pub fn markdown(mut self) -> Self {
        self.strategy = Strategy::Markdown;
        self
    }

    /// Use semantic splitting with an embedding client.
    ///
    /// Splits at meaning boundaries by computing cosine similarity between
    /// consecutive sentence embeddings. When similarity drops below the threshold,
    /// a new chunk begins.
    ///
    /// Requires the `semantic` feature and an [`embedrs::Client`].
    /// Must use [`.split_async()`](ChunkBuilder::split_async) instead of `.split()`.
    ///
    /// ```rust,ignore
    /// let client = embedrs::openai("sk-...");
    /// let chunks = chunkedrs::chunk(text)
    ///     .semantic(&client)
    ///     .split_async()
    ///     .await?;
    /// ```
    #[cfg(feature = "semantic")]
    pub fn semantic(mut self, client: &'a embedrs::Client) -> Self {
        self.strategy = Strategy::Semantic;
        self.semantic_client = Some(client);
        self
    }

    /// Set the similarity threshold for semantic splitting. Default: 0.5.
    ///
    /// Lower values create fewer, larger chunks. Higher values create more, smaller chunks.
    /// Only effective when using [`.semantic()`](ChunkBuilder::semantic).
    #[cfg(feature = "semantic")]
    pub fn threshold(mut self, t: f64) -> Self {
        self.semantic_threshold = t;
        self
    }

    /// Split the text synchronously. Works with recursive and markdown strategies.
    ///
    /// Panics if called with the semantic strategy — use
    /// [`.split_async()`](ChunkBuilder::split_async) instead.
    ///
    /// ```rust
    /// let chunks = chunkedrs::chunk("hello world").split();
    /// assert_eq!(chunks[0].content, "hello world");
    /// ```
    pub fn split(self) -> Vec<Chunk> {
        let encoder = self.resolve_encoder();
        match self.strategy {
            Strategy::Recursive => recursive::split_recursive(
                self.text,
                0,
                self.max_tokens,
                self.overlap,
                encoder,
                &None,
            ),
            Strategy::Markdown => {
                markdown::split_markdown(self.text, self.max_tokens, self.overlap, encoder)
            }
            #[cfg(feature = "semantic")]
            Strategy::Semantic => {
                panic!(
                    "semantic strategy requires async: use .split_async().await instead of .split()"
                )
            }
        }
    }

    /// Split the text asynchronously. Required for semantic splitting.
    ///
    /// ```rust,ignore
    /// let chunks = chunkedrs::chunk(text)
    ///     .semantic(&client)
    ///     .split_async()
    ///     .await?;
    /// ```
    #[cfg(feature = "semantic")]
    pub async fn split_async(self) -> Result<Vec<Chunk>> {
        let encoder = self.resolve_encoder();
        match self.strategy {
            Strategy::Semantic => {
                let client = self
                    .semantic_client
                    .expect("semantic() must be called before split_async()");
                semantic::split_semantic(
                    self.text,
                    self.max_tokens,
                    self.overlap,
                    encoder,
                    client,
                    self.semantic_threshold,
                )
                .await
            }
            _ => Ok(self.split()),
        }
    }

    fn resolve_encoder(&self) -> &'static tiktoken::CoreBpe {
        let default = || tiktoken::get_encoding("o200k_base").expect("o200k_base encoding");

        // encoding name takes precedence over model name
        if let Some(name) = self.encoding_name {
            return tiktoken::get_encoding(name).unwrap_or_else(default);
        }

        // try model name
        if let Some(model) = self.model_name {
            return tiktoken::encoding_for_model(model)
                .or_else(|| tiktoken::get_encoding(model))
                .unwrap_or_else(default);
        }

        default()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn chunk_short_text() {
        let chunks = chunk("hello world").split();
        assert_eq!(chunks.len(), 1);
        assert_eq!(chunks[0].content, "hello world");
        assert_eq!(chunks[0].index, 0);
        assert_eq!(chunks[0].start_byte, 0);
        assert_eq!(chunks[0].end_byte, 11);
        assert!(chunks[0].token_count > 0);
    }

    #[test]
    fn chunk_empty_text() {
        let chunks = chunk("").split();
        assert!(chunks.is_empty());
    }

    #[test]
    fn chunk_respects_max_tokens() {
        let text = "The quick brown fox. ".repeat(100);
        let chunks = chunk(&text).max_tokens(20).split();
        for c in &chunks {
            assert!(
                c.token_count <= 20,
                "chunk {} has {} tokens",
                c.index,
                c.token_count
            );
        }
    }

    #[test]
    fn chunk_with_overlap() {
        let text = "Sentence one. Sentence two. Sentence three. Sentence four. Sentence five. Sentence six.";
        let chunks = chunk(text).max_tokens(10).overlap(3).split();
        assert!(chunks.len() >= 2);
    }

    #[test]
    fn chunk_max_tokens_minimum_one() {
        let chunks = chunk("hello").max_tokens(0).split();
        // max_tokens(0) becomes 1
        assert!(!chunks.is_empty());
    }

    #[test]
    fn chunk_with_model() {
        let chunks = chunk("hello world").model("gpt-4o").split();
        assert_eq!(chunks.len(), 1);
    }

    #[test]
    fn chunk_with_encoding() {
        let chunks = chunk("hello world").encoding("cl100k_base").split();
        assert_eq!(chunks.len(), 1);
    }

    #[test]
    fn chunk_markdown_mode() {
        let md = "# Title\n\nSome content.\n\n## Section\n\nMore content.\n";
        let chunks = chunk(md).markdown().split();
        assert!(chunks.len() >= 2);
        assert_eq!(chunks[0].section.as_deref(), Some("# Title"));
    }

    #[test]
    fn chunk_sequential_indices() {
        let text = "Word. ".repeat(200);
        let chunks = chunk(&text).max_tokens(10).split();
        for (i, c) in chunks.iter().enumerate() {
            assert_eq!(c.index, i);
        }
    }

    #[test]
    fn chunk_chinese_text() {
        let text = "这是一段中文文本。它包含多个句子。每个句子都应该被正确分割。更多的内容在这里。还有更多。最后一句话。";
        let chunks = chunk(text).max_tokens(10).split();
        assert!(chunks.len() >= 2);
        for c in &chunks {
            assert!(c.token_count <= 10);
        }
    }

    #[test]
    fn chunk_japanese_text() {
        let text =
            "これは日本語のテキストです。複数の文が含まれています。正しく分割されるべきです。";
        let chunks = chunk(text).max_tokens(10).split();
        assert!(!chunks.is_empty());
        for c in &chunks {
            assert!(c.token_count <= 10);
        }
    }

    #[test]
    fn chunk_preserves_all_content() {
        let text = "First paragraph.\n\nSecond paragraph.\n\nThird paragraph.";
        let chunks = chunk(text).max_tokens(5).split();
        let combined: String = chunks
            .iter()
            .map(|c| c.content.as_str())
            .collect::<Vec<_>>()
            .join("");
        assert!(combined.contains("First"));
        assert!(combined.contains("Second"));
        assert!(combined.contains("Third"));
    }

    #[test]
    fn chunk_large_document() {
        let text = "Lorem ipsum dolor sit amet. ".repeat(1000);
        let chunks = chunk(&text).max_tokens(100).split();
        assert!(chunks.len() >= 10);
        for c in &chunks {
            assert!(c.token_count <= 100);
        }
    }

    #[test]
    fn chunk_single_token_max() {
        let chunks = chunk("hello world foo bar").max_tokens(1).split();
        assert!(chunks.len() >= 4);
        for c in &chunks {
            assert!(c.token_count <= 1);
        }
    }

    #[test]
    fn resolve_encoder_unknown_falls_back() {
        let builder = chunk("test").model("nonexistent-model-xyz");
        let enc = builder.resolve_encoder();
        assert!(enc.count("hello") > 0);
    }

    #[test]
    fn model_and_encoding_are_independent() {
        // encoding takes precedence over model
        // gpt-4o uses o200k_base, but we explicitly set cl100k_base
        let enc_cl100k = chunk("test")
            .model("gpt-4o")
            .encoding("cl100k_base")
            .resolve_encoder();
        let enc_o200k = chunk("test").model("gpt-4o").resolve_encoder();

        // verify they are different encoders by checking that at least one of
        // several test strings produces different token counts
        let test_texts = [
            "hello_world_123_test",
            "foo::bar::baz::qux",
            "αβγδεζηθ",
            "1234567890",
        ];
        let any_different = test_texts
            .iter()
            .any(|t| enc_cl100k.count(t) != enc_o200k.count(t));
        assert!(
            any_different,
            "cl100k_base and o200k_base should produce different token counts for at least one test string"
        );
    }

    #[test]
    fn encoding_only_without_model() {
        let builder = chunk("test").encoding("cl100k_base");
        let enc = builder.resolve_encoder();
        assert!(enc.count("hello") > 0);
    }

    #[test]
    fn model_only_without_encoding() {
        let builder = chunk("test").model("gpt-4o");
        let enc = builder.resolve_encoder();
        assert!(enc.count("hello") > 0);
    }
}
