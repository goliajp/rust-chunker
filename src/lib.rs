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
//! | **Recursive** (default) | General text — paragraphs, sentences, clauses, words | Fastest |
//! | **Markdown** | Documents with headers — preserves section ancestry | Fast |
//! | **Code** | Source in any language — blank lines, block closers, lines | Fast |
//! | **HTML** | Web pages — block-level tag boundaries | Fast |
//! | **Semantic** | High-quality RAG — splits at meaning boundaries via embeddings | Slower (API calls) |
//!
//! ## Late chunking
//!
//! Every chunk carries `start_token..end_token`: its range in the *document's*
//! token stream. That is what [late chunking] needs — embed the document once,
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
//! ## Token-accurate splitting
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
//! builds use one. They are opt-out: the default carries all 17 encodings, and
//! `default-features = false` keeps only `o200k_base` — the encoder every
//! unresolved name falls back to, so it is never absent.
//!
//! ```toml
//! # everything (default)
//! chunkedrs = "2"
//!
//! # o200k_base only — GPT-4o, GPT-5, o-series
//! chunkedrs = { version = "2", default-features = false }
//!
//! # o200k_base plus the Zhipu family
//! chunkedrs = { version = "2", default-features = false, features = ["vocab-zhipu"] }
//! ```
//!
//! Measured on `examples/basic`, release: **7,100,544 → 2,695,104 bytes**, 62%.
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

mod chunk;
mod code;
mod html;
mod markdown;
pub(crate) mod recursive;
#[cfg(feature = "semantic")]
mod semantic;

pub use chunk::Chunk;

/// Final pass shared by every strategy: drop chunks with nothing in them,
/// number what is left, and locate each one in the document's token stream.
///
/// Dropping whitespace-only chunks matters more than it sounds. A separator
/// tier can leave a lone `"\n"` stranded between two oversized pieces, and that
/// chunk would go on to be embedded, stored, and never usefully retrieved. The
/// split is therefore not byte-for-byte reconstructible — it already was not,
/// since the markdown strategy keeps header lines in metadata — but every
/// surviving chunk's byte range still addresses the source exactly.
fn finish(chunks: &mut Vec<Chunk>, text: &str, encoder: &tiktoken::CoreBpe) {
    chunks.retain(|c| !c.content.trim().is_empty());
    for (i, chunk) in chunks.iter_mut().enumerate() {
        chunk.index = i;
    }
    assign_token_spans(chunks, text, encoder);
}

/// Fill in each chunk's `start_token..end_token` against the whole document.
///
/// The splitters work on slices and cannot see the document's token stream, so
/// this runs once at the end: encode the document, build a prefix sum of token
/// byte lengths, and binary-search each chunk's byte range into token indices.
///
/// Where a chunk boundary falls inside a token — possible whenever a separator
/// sits mid-token — the span widens to the covering range. That is the right
/// direction for the use this enables: [late chunking] pools a chunk's
/// embedding over its token range, and a range that covers the chunk is
/// correct while one that clips it is not.
///
/// [late chunking]: https://arxiv.org/abs/2409.04701
fn assign_token_spans(chunks: &mut [Chunk], text: &str, encoder: &tiktoken::CoreBpe) {
    if chunks.is_empty() {
        return;
    }

    let tokens = encoder.encode(text);
    // byte offset at which each token starts; one extra entry for the end
    let mut starts = Vec::with_capacity(tokens.len() + 1);
    let mut acc = 0usize;
    for &t in &tokens {
        starts.push(acc);
        acc += encoder.decode(&[t]).len();
    }
    starts.push(acc);

    // index of the token containing `byte`, or the token count if past the end
    let token_containing = |byte: usize| -> usize {
        match starts.binary_search(&byte) {
            Ok(i) => i,
            // partition_point-style: the insertion point is one past the
            // token whose range covers this byte
            Err(i) => i.saturating_sub(1),
        }
    };

    for chunk in chunks {
        let start = token_containing(chunk.start_byte).min(tokens.len());
        // `end_byte` is exclusive, so locate the last byte the chunk owns
        let end = if chunk.end_byte > chunk.start_byte {
            (token_containing(chunk.end_byte - 1) + 1).min(tokens.len())
        } else {
            start
        };
        chunk.start_token = start;
        chunk.end_token = end.max(start);
    }
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
    /// how you say that to the compiler — no `unreachable!()` required.
    #[cfg(not(feature = "semantic"))]
    fn fmt(&self, _f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match *self {}
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
    }
}

/// Strategy for splitting text.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Strategy {
    Recursive,
    Markdown,
    Code,
    Html,
}

/// Builder for configuring text chunking.
///
/// Created by [`chunk()`]. Chain methods to configure, then call
/// [`.split()`](ChunkBuilder::split).
///
/// [`.semantic()`](ChunkBuilder::semantic) converts this into a
/// [`SemanticChunkBuilder`], which offers only `.split_async()` — semantic
/// splitting makes network calls, so there is no synchronous `.split()` to
/// call by mistake.
pub struct ChunkBuilder<'a> {
    text: &'a str,
    max_tokens: usize,
    overlap: usize,
    model_name: Option<&'a str>,
    encoding_name: Option<&'a str>,
    strategy: Strategy,
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
    /// Default: `o200k_base` (GPT-4o, GPT-5, o-series).
    ///
    /// Resolves names across OpenAI, Meta (`llama-3.1-70b`), DeepSeek
    /// (`deepseek-v4`), Alibaba (`qwen2.5-72b`), Mistral, Moonshot (`kimi-k2`),
    /// Zhipu (`glm-5`) and MiniMax (`minimax-m2`).
    ///
    /// This is independent of [`.encoding()`](ChunkBuilder::encoding). If both are
    /// set, `encoding` takes precedence.
    ///
    /// If the model name is not recognized, falls back to `o200k_base` silently.
    /// Anthropic and Google publish no tiktoken-compatible vocabulary, so
    /// `claude-*` and `gemini-*` take that fallback — an approximation, not an
    /// exact count.
    ///
    /// A model whose vocabulary was not compiled in takes the same fallback.
    /// See [vocabulary features](crate#vocabulary-features).
    ///
    /// ```rust
    /// let chunks = chunkedrs::chunk("hello world").model("gpt-5.6-terra").split();
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
    /// If the encoding name is not recognized, falls back to `o200k_base`
    /// silently. Since tiktoken 4 a vocabulary that was not compiled in is
    /// *also* unrecognized, and looks identical to a typo — so a slimmed build
    /// that asks for a vocabulary it did not enable gets `o200k_base` counts
    /// without complaint. See [vocabulary features](crate#vocabulary-features).
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
    /// Splits at header boundaries first, then applies recursive splitting
    /// within each section. Each chunk's [`Chunk::section`] field contains the
    /// header it belongs to.
    ///
    /// Headers are ATX (`#` through `######`) or setext (`===` / `---`
    /// underline). Fenced code blocks are skipped, so a `#` comment inside a
    /// shell or python block is not mistaken for a header, and YAML/TOML front
    /// matter is excluded from header detection.
    ///
    /// Note: header lines themselves are stored in `section` metadata, not in
    /// chunk `content`. This means joining all chunk contents will not reproduce
    /// the header lines from the original document.
    ///
    /// ```rust
    /// let md = "# Title\n\nContent here.\n";
    /// let chunks = chunkedrs::chunk(md).markdown().split();
    /// assert_eq!(chunks[0].section(), Some("# Title"));
    /// ```
    pub fn markdown(mut self) -> Self {
        self.strategy = Strategy::Markdown;
        self
    }

    /// Use code-aware splitting.
    ///
    /// Boundary-aware, **not** AST-aware: this splits on blank lines, block
    /// braces at low nesting depth, and dedents, in that order, then falls back
    /// to the ordinary text ladder. It parses nothing and adds no dependencies,
    /// so it works on any language — and it will not track a construct across a
    /// boundary the way a real parser would. If you need AST fidelity, use a
    /// tree-sitter based splitter.
    ///
    /// ```rust
    /// let src = "fn a() {\n    one();\n}\n\nfn b() {\n    two();\n}\n";
    /// let chunks = chunkedrs::chunk(src).code().max_tokens(12).split();
    /// assert!(chunks.len() >= 2);
    /// ```
    pub fn code(mut self) -> Self {
        self.strategy = Strategy::Code;
        self
    }

    /// Use HTML-aware splitting.
    ///
    /// Boundary-aware, **not** DOM-aware: this splits after block-level closing
    /// tags (`</p>`, `</div>`, `</section>`, headings, list items, …) by
    /// scanning bytes. It builds no tree and adds no dependencies. Malformed or
    /// deeply nested markup degrades to the ordinary text ladder rather than
    /// failing.
    ///
    /// ```rust
    /// let html = "<h1>Title</h1><p>First para.</p><p>Second para.</p>";
    /// let chunks = chunkedrs::chunk(html).html().max_tokens(8).split();
    /// assert!(chunks.len() >= 2);
    /// ```
    pub fn html(mut self) -> Self {
        self.strategy = Strategy::Html;
        self
    }

    /// Use semantic splitting with an embedding client.
    ///
    /// Splits at meaning boundaries by computing cosine similarity between
    /// consecutive sentence embeddings. When similarity drops below the
    /// threshold, a new chunk begins.
    ///
    /// Returns a [`SemanticChunkBuilder`], which has no `.split()` — semantic
    /// splitting makes network calls, so the synchronous method simply does not
    /// exist on it. This used to be a runtime panic.
    ///
    /// Requires the `semantic` feature and an [`embedrs::Client`].
    ///
    /// ```rust,ignore
    /// let client = embedrs::Client::openai("sk-...");
    /// let chunks = chunkedrs::chunk(text)
    ///     .semantic(&client)
    ///     .threshold(0.5)
    ///     .split_async()
    ///     .await?;
    /// ```
    #[cfg(feature = "semantic")]
    pub fn semantic(self, client: &'a embedrs::Client) -> SemanticChunkBuilder<'a> {
        SemanticChunkBuilder {
            base: self,
            client,
            threshold: 0.5,
        }
    }

    /// Split the text.
    ///
    /// ```rust
    /// let chunks = chunkedrs::chunk("hello world").split();
    /// assert_eq!(chunks[0].content, "hello world");
    /// ```
    pub fn split(self) -> Vec<Chunk> {
        let encoder = self.resolve_encoder();
        let mut chunks = match self.strategy {
            Strategy::Recursive => recursive::split_recursive(
                self.text,
                0,
                self.max_tokens,
                self.overlap,
                encoder,
                &[],
            ),
            Strategy::Markdown => {
                markdown::split_markdown(self.text, self.max_tokens, self.overlap, encoder)
            }
            Strategy::Code => code::split_code(self.text, self.max_tokens, self.overlap, encoder),
            Strategy::Html => html::split_html(self.text, self.max_tokens, self.overlap, encoder),
        };
        finish(&mut chunks, self.text, encoder);
        chunks
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

/// Builder for semantic splitting, produced by
/// [`ChunkBuilder::semantic`](ChunkBuilder::semantic).
///
/// It deliberately offers no `.split()`. Semantic splitting has to embed the
/// text, which means network calls, so the synchronous entry point does not
/// exist here rather than existing and panicking.
#[cfg(feature = "semantic")]
pub struct SemanticChunkBuilder<'a> {
    base: ChunkBuilder<'a>,
    client: &'a embedrs::Client,
    threshold: f64,
}

#[cfg(feature = "semantic")]
impl<'a> SemanticChunkBuilder<'a> {
    /// Set the similarity threshold. Default: 0.5.
    ///
    /// Lower values create fewer, larger chunks; higher values create more,
    /// smaller ones.
    pub fn threshold(mut self, t: f64) -> Self {
        self.threshold = t;
        self
    }

    /// Set the maximum number of tokens per chunk. Default: 512.
    pub fn max_tokens(mut self, n: usize) -> Self {
        self.base = self.base.max_tokens(n);
        self
    }

    /// Set the number of overlapping tokens between consecutive chunks.
    pub fn overlap(mut self, tokens: usize) -> Self {
        self.base = self.base.overlap(tokens);
        self
    }

    /// Set the model name used to select the tokenizer encoding.
    pub fn model(mut self, model: &'a str) -> Self {
        self.base = self.base.model(model);
        self
    }

    /// Set the tiktoken encoding name directly.
    pub fn encoding(mut self, encoding: &'a str) -> Self {
        self.base = self.base.encoding(encoding);
        self
    }

    /// Split the text, embedding it to find meaning boundaries.
    ///
    /// ```rust,ignore
    /// let chunks = chunkedrs::chunk(text)
    ///     .semantic(&client)
    ///     .split_async()
    ///     .await?;
    /// ```
    pub async fn split_async(self) -> Result<Vec<Chunk>> {
        let encoder = self.base.resolve_encoder();
        let mut chunks = semantic::split_semantic(
            self.base.text,
            self.base.max_tokens,
            self.base.overlap,
            encoder,
            self.client,
            self.threshold,
        )
        .await?;
        finish(&mut chunks, self.base.text, encoder);
        Ok(chunks)
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
        assert_eq!(chunks[0].section(), Some("# Title"));
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

    /// Needs two distinct vocabularies compiled in; a build that names only
    /// `vocab-o200k_base` resolves both sides to the same encoder.
    #[test]
    #[cfg(feature = "vocab-cl100k_base")]
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
