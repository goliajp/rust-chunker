//! # chunkedrs
//!
//! Split long documents into chunks that fit a token budget, for embedding and
//! retrieval. Token counts come from [tiktoken](https://crates.io/crates/tiktoken);
//! every chunk borrows its text from the document and carries its exact byte
//! range in it and its token range in the document's token stream.
//!
//! ## Strategies
//!
//! | Strategy | Splits on |
//! |----------|-----------|
//! | **Recursive** (default) | paragraphs, lines, sentences (CJK marks included), clauses, words, then tokens |
//! | **Markdown** | ATX and setext headers first, keeping the header ancestry on each chunk |
//! | **Code** | blank lines, brackets closing a block at column 0, lines, words |
//! | **HTML** | block-level closing tags, then the recursive ladder |
//! | **Semantic** | sentence groups whose embeddings stay similar, with vectors from any model |
//!
//! Each chunk ends at the last boundary of the highest-priority kind that
//! still fits `max_tokens`, and its `token_count` is counted on its own text,
//! so `token_count <= max_tokens` holds for every chunk. The one exception is
//! a single character that alone takes more tokens than the budget.
//!
//! ## Quick start
//!
//! ```rust
//! // o200k_base, 512 tokens per chunk, no overlap, recursive
//! let chunks = chunkedrs::split("your long text here...");
//! for chunk in &chunks {
//!     println!("[{}] {} tokens: {}", chunk.index, chunk.token_count, chunk.content);
//! }
//! ```
//!
//! Everything else is set on a [`Chunker`], which can be kept and reused:
//!
//! ```rust
//! let chunker = chunkedrs::Chunker::for_model("gpt-4o")?
//!     .max_tokens(256)
//!     .overlap(32);
//! let chunks = chunker.split("your long text here...");
//! assert!(chunks.iter().all(|c| c.token_count <= 256));
//! # Ok::<(), chunkedrs::Error>(())
//! ```
//!
//! A [`Chunk`] borrows from the text; [`Chunk::to_owned_chunk`] gives an
//! [`OwnedChunk`] to keep after the text is gone.
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
//! let chunks = chunkedrs::Chunker::new().max_tokens(12).split(zh);
//!
//! assert_eq!(chunks[0].content, "他说「今天天气很好。」");
//! assert_eq!(chunks[1].content, "然后就出门了。");
//! ```
//!
//! ## Markdown, code and HTML
//!
//! ```rust
//! let markdown = "# Guide\n\nIntro.\n\n## Install\n\nRun cargo add.\n";
//! let chunks = chunkedrs::Chunker::new().markdown().split(markdown);
//!
//! // each chunk knows the section it belongs to...
//! assert_eq!(chunks[0].section(), Some("# Guide"));
//! // ...and its full ancestry, so nested sections keep their context
//! assert_eq!(chunks[1].section_path, ["# Guide", "## Install"]);
//! ```
//!
//! Code and HTML are boundary-aware rather than AST-aware: they scan for
//! structural markers and parse nothing, so they apply to any language or
//! markup and add no dependencies. Use a tree-sitter based splitter when you
//! need real syntax.
//!
//! ```rust
//! let src = "fn a() {\n    one();\n}\n\nfn b() {\n    two();\n}\n";
//! let chunks = chunkedrs::Chunker::new().code().max_tokens(20).split(src);
//! assert!(!chunks.is_empty());
//!
//! let page = "<h1>Title</h1><p>First para.</p><p>Second para.</p>";
//! let chunks = chunkedrs::Chunker::new().html().max_tokens(10).split(page);
//! assert!(chunks.len() >= 2);
//! ```
//!
//! ## Late chunking
//!
//! Every chunk carries `start_token..end_token`: its range in the *document's*
//! token stream, which [`Chunker::split_with_tokens`] returns with the chunks.
//! That is what [late chunking] needs: embed the document once, then pool
//! each chunk's vector over its own range, so each chunk embedding carries
//! the context of the whole document.
//!
//! ```rust
//! let doc = "First sentence here. Second sentence here. Third sentence here.";
//! let (chunks, ids) = chunkedrs::Chunker::new().max_tokens(8).split_with_tokens(doc);
//! for chunk in &chunks {
//!     // in a real pipeline this indexes the model's per-token hidden states
//!     assert!(!ids[chunk.start_token..chunk.end_token].is_empty());
//! }
//! ```
//!
//! [late chunking]: https://arxiv.org/abs/2409.04701
//!
//! ## Semantic splitting
//!
//! [`Chunker::semantic`] cuts the text into sentences and hands them out to
//! embed with anything; [`Semantic::finish`] takes the vectors back and
//! breaks where neighbours stop being similar. No feature, no dependency, no
//! I/O:
//!
//! ```rust
//! let text = "Cats purr. Cats nap. Stocks fell. Markets slid.";
//! let job = chunkedrs::Chunker::new().semantic(text);
//! let vectors = [1.0, 0.0, 0.9, 0.1, 0.0, 1.0, 0.1, 0.9]; // from your model
//! let chunks = job
//!     .breakpoints(chunkedrs::Breakpoints::Threshold(0.5))
//!     .finish(&vectors, 2)?;
//! assert_eq!(chunks.len(), 2);
//! # Ok::<(), chunkedrs::Error>(())
//! ```
//!
//! With the `semantic` feature, [`Chunker::split_semantic`] does the
//! embedding through any `embedrs::Embed`: a remote provider's client, a
//! local model, or your own type.
//!
//! ```rust,ignore
//! let client = embedrs::local(); // all-MiniLM-L6-v2 on the CPU
//! let chunks = chunkedrs::Chunker::new().split_semantic(text, &client).await?;
//! ```
//!
//! ## Many documents
//!
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
//! ## Vocabulary features
//!
//! Tokenizer vocabularies are the bulk of this crate's compiled size, and most
//! builds use one. They are opt-out: the default carries all 20 encodings, and
//! `default-features = false` keeps only `o200k_base`, the encoding
//! [`Chunker::new`] uses.
//!
//! ```toml
//! # everything (default)
//! chunkedrs = "3"
//!
//! # o200k_base only: GPT-4o, GPT-5, o-series
//! chunkedrs = { version = "3", default-features = false }
//!
//! # o200k_base plus the Zhipu family
//! chunkedrs = { version = "3", default-features = false, features = ["vocab-zhipu"] }
//! ```
//!
//! Features come in vendor groups (`vocab-openai`, `vocab-meta`,
//! `vocab-deepseek`, `vocab-qwen`, `vocab-mistral`, `vocab-moonshot`,
//! `vocab-zhipu`, `vocab-minimax`) and per-vocabulary
//! (`vocab-cl100k_base`, `vocab-llama3`, `vocab-glm5`, …).
//! [`Chunker::for_model`] and [`Chunker::for_encoding`] return an error for a
//! name whose vocabulary this build leaves out.

mod chunk;
mod chunker;
mod error;
mod plan;
mod semantic;
mod strategy;

pub use chunk::{Chunk, OwnedChunk};
pub use chunker::Chunker;
pub use error::{Error, Result};
pub use semantic::{Breakpoints, Semantic};

/// Split `text` with the defaults of [`Chunker::new`]: `o200k_base`, 512
/// tokens per chunk, no overlap, recursive.
///
/// ```rust
/// let chunks = chunkedrs::split("hello world");
/// assert_eq!(chunks.len(), 1);
/// assert_eq!(chunks[0].content, "hello world");
/// ```
#[must_use]
pub fn split(text: &str) -> Vec<Chunk<'_>> {
    Chunker::new().split(text)
}
