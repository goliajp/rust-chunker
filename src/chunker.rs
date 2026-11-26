use crate::plan::Doc;
use crate::strategy::{self, Strategy};
use crate::{Chunk, Error, Result};

/// A splitter: encoding, token budget, overlap and strategy, set once and
/// used for any number of texts.
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
    pub(crate) encoder: &'static tiktoken::CoreBpe,
    pub(crate) max_tokens: usize,
    pub(crate) overlap: usize,
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
    /// `o200k_base` (GPT-4o, GPT-5, o-series), 512 tokens per chunk, no
    /// overlap, recursive splitting.
    #[must_use]
    pub fn new() -> Self {
        Self::with_encoder(
            tiktoken::get_encoding("o200k_base").expect("o200k_base is always built"),
        )
    }

    /// The defaults of [`new`](Self::new) with an encoder you already hold.
    ///
    /// ```rust
    /// let enc = tiktoken::get_encoding("o200k_base").unwrap();
    /// let chunker = chunkedrs::Chunker::with_encoder(enc);
    /// ```
    #[must_use]
    pub fn with_encoder(encoder: &'static tiktoken::CoreBpe) -> Self {
        Chunker {
            encoder,
            max_tokens: 512,
            overlap: 0,
            strategy: Strategy::Recursive,
        }
    }

    /// The defaults of [`new`](Self::new) with the encoding of a model name,
    /// as tiktoken maps it (an encoding name is accepted too).
    ///
    /// # Errors
    ///
    /// [`Error::UnknownModel`] when tiktoken maps the name to nothing, or to
    /// an encoding whose `vocab-*` feature this build does not enable.
    ///
    /// ```rust
    /// let chunker = chunkedrs::Chunker::for_model("gpt-4o")?;
    /// assert!(chunkedrs::Chunker::for_model("no-such-model").is_err());
    /// # Ok::<(), chunkedrs::Error>(())
    /// ```
    pub fn for_model(model: &str) -> Result<Self> {
        tiktoken::encoding_for_model(model)
            .or_else(|| tiktoken::get_encoding(model))
            .map(Self::with_encoder)
            .ok_or_else(|| Error::UnknownModel(model.to_owned()))
    }

    /// The defaults of [`new`](Self::new) with an encoding by name.
    ///
    /// # Errors
    ///
    /// [`Error::UnknownEncoding`] for a name tiktoken does not know, or one
    /// whose `vocab-*` feature this build does not enable.
    ///
    /// ```rust
    /// let chunker = chunkedrs::Chunker::for_encoding("o200k_base")?;
    /// # Ok::<(), chunkedrs::Error>(())
    /// ```
    pub fn for_encoding(name: &str) -> Result<Self> {
        tiktoken::get_encoding(name)
            .map(Self::with_encoder)
            .ok_or_else(|| Error::UnknownEncoding(name.to_owned()))
    }

    /// Maximum tokens per chunk, counted on the chunk's own text. Default
    /// 512; 0 is taken as 1.
    ///
    /// Every chunk holds at most this many tokens. The one exception is a
    /// single character that alone needs more tokens than the budget.
    #[must_use]
    pub fn max_tokens(mut self, n: usize) -> Self {
        self.max_tokens = n.max(1);
        self
    }

    /// About this many tokens of the end of each chunk start the next one.
    /// Default 0.
    ///
    /// The overlap starts on a char boundary and is at most `max_tokens - 1`
    /// tokens, so every chunk adds new text; a chunk never starts before the
    /// previous one.
    #[must_use]
    pub fn overlap(mut self, tokens: usize) -> Self {
        self.overlap = tokens;
        self
    }

    /// Split on paragraphs, lines, sentences (CJK marks included), clauses
    /// and words, then tokens. The default.
    #[must_use]
    pub fn recursive(mut self) -> Self {
        self.strategy = Strategy::Recursive;
        self
    }

    /// Split at ATX and setext headers first, and record each chunk's header
    /// ancestry in [`Chunk::section_path`]. Fenced code blocks and front
    /// matter are not read as headers. Header lines are not part of any
    /// chunk's text.
    #[must_use]
    pub fn markdown(mut self) -> Self {
        self.strategy = Strategy::Markdown;
        self
    }

    /// Split source code at blank lines, then at brackets closing a block at
    /// column 0, then lines and words. Boundary-aware, not a parser: it
    /// works for any language and reads no grammar.
    #[must_use]
    pub fn code(mut self) -> Self {
        self.strategy = Strategy::Code;
        self
    }

    /// Split HTML after block-level closing tags (`</p>`, `</li>`, `</tr>`,
    /// ...), then by the recursive ladder. Boundary-aware, not a DOM parser.
    #[must_use]
    pub fn html(mut self) -> Self {
        self.strategy = Strategy::Html;
        self
    }

    /// The encoder this chunker counts with.
    #[must_use]
    pub fn encoder(&self) -> &'static tiktoken::CoreBpe {
        self.encoder
    }

    /// Split one text.
    ///
    /// ```rust
    /// let chunks = chunkedrs::Chunker::new().max_tokens(8).split("One. Two. Three. Four.");
    /// assert!(chunks.iter().all(|c| c.token_count <= 8));
    /// ```
    #[must_use]
    pub fn split<'a>(&self, text: &'a str) -> Vec<Chunk<'a>> {
        self.split_doc(&Doc::new(text, self.encoder))
    }

    /// Split one text and return its token ids too, the ids
    /// `start_token..end_token` index. For late chunking: run the
    /// embedding model over these ids once, then pool each chunk's span.
    ///
    /// ```rust
    /// let text = "First sentence here. Second sentence here.";
    /// let chunker = chunkedrs::Chunker::new().max_tokens(5);
    /// let (chunks, ids) = chunker.split_with_tokens(text);
    /// assert_eq!(ids, chunker.encoder().encode(text));
    /// for c in &chunks {
    ///     assert!(!ids[c.start_token..c.end_token].is_empty());
    /// }
    /// ```
    #[must_use]
    pub fn split_with_tokens<'a>(&self, text: &'a str) -> (Vec<Chunk<'a>>, Vec<u32>) {
        let doc = Doc::new(text, self.encoder);
        let chunks = self.split_doc(&doc);
        (chunks, doc.tokens)
    }

    /// Split many texts on up to [`std::thread::available_parallelism`]
    /// threads. The result is in input order, and each entry equals what
    /// [`split`](Self::split) returns for that text.
    ///
    /// tiktoken also encodes a single text of 32 KiB or more on several
    /// threads. If the batch holds large documents and the process already
    /// keeps every core busy, set `TIKTOKEN_THREADS=1` or call
    /// [`tiktoken::set_parallelism`] to keep tiktoken on the calling thread.
    #[must_use]
    pub fn split_batch<'a, S: AsRef<str> + Sync>(&self, texts: &'a [S]) -> Vec<Vec<Chunk<'a>>> {
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

    pub(crate) fn split_doc<'a>(&self, doc: &Doc<'a>) -> Vec<Chunk<'a>> {
        let (max, overlap) = (self.max_tokens, self.overlap);
        let mut chunks = match self.strategy {
            Strategy::Recursive => strategy::recursive::split_recursive(doc, max, overlap),
            Strategy::Markdown => strategy::markdown::split_markdown(doc, max, overlap),
            Strategy::Code => strategy::code::split_code(doc, max, overlap),
            Strategy::Html => strategy::html::split_html(doc, max, overlap),
        };
        finish(&mut chunks);
        chunks
    }
}

/// Final pass shared by every strategy: drop chunks with nothing in them and
/// number what is left.
///
/// A separator tier can leave a lone `"\n"` stranded between two cuts, and that
/// chunk would go on to be embedded, stored, and never usefully retrieved. The
/// chunks therefore need not cover every byte of the text (the markdown
/// strategy also keeps header lines out of them), but every surviving chunk's
/// byte range addresses the text exactly.
pub(crate) fn finish(chunks: &mut Vec<Chunk<'_>>) {
    chunks.retain(|c| !c.content.trim().is_empty());
    for (i, chunk) in chunks.iter_mut().enumerate() {
        chunk.index = i;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn short_text_is_one_chunk() {
        let chunks = Chunker::new().split("hello world");
        assert_eq!(chunks.len(), 1);
        assert_eq!(chunks[0].content, "hello world");
        assert_eq!(
            (chunks[0].index, chunks[0].start_byte, chunks[0].end_byte),
            (0, 0, 11)
        );
        assert!(chunks[0].token_count > 0);
        assert!(Chunker::new().split("").is_empty());
    }

    #[test]
    fn budgets_hold() {
        let text = "The quick brown fox. ".repeat(100);
        for c in Chunker::new().max_tokens(20).split(&text) {
            assert!(
                c.token_count <= 20,
                "chunk {} has {} tokens",
                c.index,
                c.token_count
            );
        }
        let chunks = Chunker::new().max_tokens(0).split("hello");
        assert!(!chunks.is_empty(), "max_tokens(0) becomes 1");
        let chunks = Chunker::new().max_tokens(1).split("hello world foo bar");
        assert!(chunks.len() >= 4 && chunks.iter().all(|c| c.token_count <= 1));
    }

    #[test]
    fn indices_are_sequential() {
        let text = "Word. ".repeat(200);
        for (i, c) in Chunker::new()
            .max_tokens(10)
            .split(&text)
            .iter()
            .enumerate()
        {
            assert_eq!(c.index, i);
        }
    }

    #[test]
    fn cjk_text_splits_within_budget() {
        for text in [
            "这是一段中文文本。它包含多个句子。每个句子都应该被正确分割。更多的内容在这里。还有更多。最后一句话。",
            "これは日本語のテキストです。複数の文が含まれています。正しく分割されるべきです。",
        ] {
            let chunks = Chunker::new().max_tokens(10).split(text);
            assert!(chunks.len() >= 2);
            assert!(chunks.iter().all(|c| c.token_count <= 10));
        }
    }

    #[test]
    fn names_resolve_or_fail() {
        assert!(Chunker::for_model("gpt-4o").is_ok());
        assert!(Chunker::for_encoding("o200k_base").is_ok());
        assert!(matches!(
            Chunker::for_model("nonexistent-model-xyz"),
            Err(Error::UnknownModel(n)) if n == "nonexistent-model-xyz"
        ));
        assert!(matches!(
            Chunker::for_encoding("o201k"),
            Err(Error::UnknownEncoding(n)) if n == "o201k"
        ));
    }

    /// Needs two distinct vocabularies compiled in.
    #[test]
    #[cfg(feature = "vocab-cl100k_base")]
    fn encodings_differ() {
        let cl = Chunker::for_encoding("cl100k_base").unwrap().encoder();
        let o2 = Chunker::for_model("gpt-4o").unwrap().encoder();
        assert!(
            ["foo::bar::baz::qux", "αβγδεζηθ", "1234567890"]
                .iter()
                .any(|t| cl.count(t) != o2.count(t))
        );
    }

    #[test]
    fn with_tokens_returns_the_document_ids() {
        let text = "One sentence here. Another one there.\n\nA second paragraph.";
        let chunker = Chunker::new().max_tokens(4).overlap(1);
        let (chunks, ids) = chunker.split_with_tokens(text);
        assert_eq!(chunks, chunker.split(text));
        assert_eq!(ids, chunker.encoder().encode(text));
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
