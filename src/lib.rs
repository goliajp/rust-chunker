//! # chunkedrs
//!
//! Split long documents into chunks that fit a token budget, for embedding and
//! retrieval. Token counts come from [tiktoken](https://crates.io/crates/tiktoken);
//! every chunk carries its exact byte range in the source and its token range
//! in the document's token stream.
//!
//! ## Strategies
//!
//! | Strategy | Splits on |
//! |----------|-----------|
//! | **Recursive** (default) | paragraphs, lines, sentences (CJK marks included), clauses, words, then tokens |
//! | **Markdown** | ATX and setext headers first, keeping the header ancestry on each chunk |
//! | **Code** | blank lines, brackets closing a block at column 0, lines, words |
//! | **HTML** | block-level closing tags, then the recursive ladder |
//! | **Semantic** | sentence groups whose embeddings stay similar (`semantic` feature, network calls) |
//!
//! Each chunk ends at the last boundary of the highest-priority kind that
//! still fits `max_tokens`, and its `token_count` is counted on its own text,
//! so `token_count <= max_tokens` holds for every chunk. The one exception is
//! a single character that alone takes more tokens than the budget.
//!
//! ## Late chunking
//!
//! Every chunk carries `start_token..end_token`: its range in the *document's*
//! token stream. That is what [late chunking] needs: embed the document once,
//! then pool each chunk's vector over its own range, so each chunk embedding
//! carries the context of the whole document rather than only its own text.
//!
//! ```rust
//! let doc = "First sentence here. Second sentence here. Third sentence here.";
//! let chunks = chunkedrs::chunk(doc).max_tokens(8).split();
//!
//! let encoder = tiktoken::get_encoding("o200k_base").unwrap();
//! let document_tokens = encoder.encode(doc);
//!
//! for chunk in &chunks {
//!     // in a real pipeline this indexes the encoder's per-token hidden states
//!     let span = &document_tokens[chunk.start_token..chunk.end_token];
//!     assert!(!span.is_empty());
//! }
//! ```
//!
//! [late chunking]: https://arxiv.org/abs/2409.04701
//!
//! ## CJK text
//!
//! Chinese and Japanese put no space after a sentence mark, so a separator
//! ladder built out of `". "`, `", "` and `" "` matches nothing in them and
//! falls through to cutting mid-word. chunkedrs splits on the marks these
//! scripts actually use, and keeps a closing quote with the sentence it closes:
//!
//! ```rust
//! let zh = "他说「今天天气很好。」然后就出门了。她回答说「确实不错。」于是也跟着出去了。";
//! let chunks = chunkedrs::chunk(zh).max_tokens(12).split();
//!
//! assert_eq!(chunks[0].content, "他说「今天天气很好。」");
//! assert_eq!(chunks[1].content, "然后就出门了。");
//! ```
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
//! ## Many documents
//!
//! [`Chunker`] holds a configuration with its encoder resolved once.
//! [`Chunker::split_batch`] splits a slice of texts on all cores and returns
//! the results in input order.
//!
//! ```rust
//! let chunker = chunkedrs::Chunker::new().max_tokens(256);
//! let docs = ["First document.", "Second document."];
//! let chunks = chunker.split_batch(&docs);
//! assert_eq!(chunks[1][0].content, "Second document.");
//! ```
//!
//! tiktoken encodes a text of 32 KiB or more on several threads. A server
//! that already keeps every core busy can turn that off with
//! `TIKTOKEN_THREADS=1` or [`tiktoken::set_parallelism`]`(1)`.
//!
//! ## Token budget
//!
//! ```rust
//! let chunks = chunkedrs::chunk("your long text here...")
//!     .max_tokens(256)
//!     .overlap(50)
//!     .model("gpt-5.6-terra")
//!     .split();
//!
//! // every chunk is guaranteed to have <= 256 tokens
//! assert!(chunks.iter().all(|c| c.token_count <= 256));
//! ```
//!
//! ## Markdown-aware splitting
//!
//! ```rust
//! let markdown = "# Guide\n\nIntro.\n\n## Install\n\nRun cargo add.\n";
//! let chunks = chunkedrs::chunk(markdown).markdown().split();
//!
//! // each chunk knows the section it belongs to...
//! assert_eq!(chunks[0].section(), Some("# Guide"));
//! // ...and its full ancestry, so nested sections keep their context
//! assert_eq!(chunks[1].section_path, ["# Guide", "## Install"]);
//! ```
//!
//! ## Code and HTML
//!
//! Both are boundary-aware rather than AST-aware: they scan for structural
//! markers and parse nothing, so they apply to any language or markup and add
//! no dependencies. Use a tree-sitter based splitter when you need real syntax.
//!
//! ```rust
//! let src = "fn a() {\n    one();\n}\n\nfn b() {\n    two();\n}\n";
//! let chunks = chunkedrs::chunk(src).code().max_tokens(20).split();
//! assert!(!chunks.is_empty());
//!
//! let page = "<h1>Title</h1><p>First para.</p><p>Second para.</p>";
//! let chunks = chunkedrs::chunk(page).html().max_tokens(10).split();
//! assert!(chunks.len() >= 2);
//! ```
//!
//! ## Vocabulary features
//!
//! Tokenizer vocabularies are the bulk of this crate's compiled size, and most
//! builds use one. They are opt-out: the default carries all 20 encodings, and
//! `default-features = false` keeps only `o200k_base`, the encoder every
//! unresolved name falls back to, so it is never absent.
//!
//! ```toml
//! # everything (default)
//! chunkedrs = "2"
//!
//! # o200k_base only: GPT-4o, GPT-5, o-series
//! chunkedrs = { version = "2", default-features = false }
//!
//! # o200k_base plus the Zhipu family
//! chunkedrs = { version = "2", default-features = false, features = ["vocab-zhipu"] }
//! ```
//!
//! `examples/basic`, release build on macOS arm64: 9,189,376 bytes with every
//! vocabulary, 2,072,256 bytes with `default-features = false`.
//!
//! Features come in vendor groups (`vocab-openai`, `vocab-meta`,
//! `vocab-deepseek`, `vocab-qwen`, `vocab-mistral`, `vocab-moonshot`,
//! `vocab-zhipu`, `vocab-minimax`) and per-vocabulary
//! (`vocab-cl100k_base`, `vocab-llama3`, `vocab-glm5`, …).
//!
//! **One sharp edge.** Asking for a vocabulary this build did not compile in is
//! indistinguishable from a typo: both fall back to `o200k_base` silently, and
//! you get plausible counts for the wrong tokenizer. If you slim the build,
//! make sure the encodings you name are ones you enabled.
//!
//! ## Semantic splitting
//!
//! With the `semantic` feature enabled, split at meaning boundaries using embeddings:
//!
//! ```rust,ignore
//! let client = embedrs::Client::openai("sk-...");
//! let chunks = chunkedrs::chunk("your long text here...")
//!     .semantic(&client)
//!     .split_async()
//!     .await?;
//! ```

mod builder;
mod chunk;
mod chunker;
mod code;
mod html;
mod markdown;
mod plan;
pub(crate) mod recursive;
mod scan;
#[cfg(feature = "semantic")]
mod semantic;

#[cfg(feature = "semantic")]
pub use builder::SemanticChunkBuilder;
pub use builder::{ChunkBuilder, chunk};
pub use chunk::Chunk;
pub use chunker::Chunker;

/// Final pass shared by every strategy: drop chunks with nothing in them and
/// number what is left.
///
/// A separator tier can leave a lone `"\n"` stranded between two cuts, and that
/// chunk would go on to be embedded, stored, and never usefully retrieved. The
/// split is therefore not byte-for-byte reconstructible (the markdown strategy
/// also keeps header lines in metadata), but every surviving chunk's byte
/// range addresses the source exactly.
fn finish(chunks: &mut Vec<Chunk>) {
    chunks.retain(|c| !c.content.trim().is_empty());
    for (i, chunk) in chunks.iter_mut().enumerate() {
        chunk.index = i;
    }
}

/// Byte length of one token, borrowed from the vocabulary. A token may end
/// inside a multi-byte character, so a summed offset is not always a char
/// boundary.
#[inline]
pub(crate) fn token_len(encoder: &tiktoken::CoreBpe, token: u32) -> usize {
    encoder.token_bytes(token).map_or(0, <[u8]>::len)
}

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
    #[cfg(feature = "semantic")]
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match *self {
            Error::Embed(ref e) => write!(f, "embedding error: {e}"),
        }
    }

    /// Without the `semantic` feature nothing in this crate can fail, so
    /// `Error` has no variants. Matching an uninhabited value with no arms is
    /// how you say that to the compiler; no `unreachable!()` required.
    #[cfg(not(feature = "semantic"))]
    fn fmt(&self, _f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match *self {}
    }
}

impl std::error::Error for Error {}

/// Result type for chunkedrs operations.
pub type Result<T> = std::result::Result<T, Error>;
