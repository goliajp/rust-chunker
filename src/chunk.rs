/// A piece of text produced by splitting a larger document.
///
/// Every chunk knows where it came from — byte offsets *and* token offsets into
/// the original text, its position in the sequence, and its own token count.
/// This metadata enables accurate citation, deduplication, context-window
/// budgeting, and late chunking.
///
/// # Byte offsets vs token spans
///
/// `start_byte..end_byte` addresses the source text: slicing the original
/// document with that range reproduces `content` exactly.
///
/// `start_token..end_token` addresses the document's *token stream* — the range
/// of `encoder.encode(whole_document)` that covers this chunk. That is what
/// [late chunking] needs: embed the document once, then mean-pool each chunk's
/// vector over its own token range, so every chunk embedding carries the
/// context of the whole document.
///
/// # `token_count` is not `end_token - start_token`
///
/// `token_count` is a fresh count of `content` on its own — the number you
/// budget a context window against, and the number `max_tokens` bounds. The
/// span width is this chunk's footprint in the document stream. BPE merges
/// across boundaries, so re-tokenizing a fragment in isolation does not always
/// reproduce its slice of the whole; and where a chunk boundary falls inside a
/// token, the span widens to cover it. Both numbers are real and they answer
/// different questions.
///
/// [late chunking]: https://arxiv.org/abs/2409.04701
///
/// ```
/// let chunks = chunkedrs::chunk("Hello world. Goodbye world.").split();
/// for chunk in &chunks {
///     println!(
///         "[{}] bytes {}..{}, tokens {}..{} ({} tokens)",
///         chunk.index,
///         chunk.start_byte,
///         chunk.end_byte,
///         chunk.start_token,
///         chunk.end_token,
///         chunk.token_count,
///     );
/// }
/// ```
#[derive(Debug, Clone, Default, PartialEq, Eq)]
#[non_exhaustive]
pub struct Chunk {
    /// The text content of this chunk.
    pub content: String,

    /// Zero-based position in the chunk sequence.
    pub index: usize,

    /// Byte offset of the first character in the original text.
    pub start_byte: usize,

    /// Byte offset one past the last character in the original text.
    pub end_byte: usize,

    /// Index of the first token of this chunk in the document's token stream.
    pub start_token: usize,

    /// Index one past this chunk's last token in the document's token stream.
    pub end_token: usize,

    /// Number of tokens in `content`, counted on its own.
    pub token_count: usize,

    /// Header ancestry from the markdown strategy, outermost first — e.g.
    /// `["# Guide", "## Installation", "### From source"]`.
    ///
    /// Empty for content before the first header and for every non-markdown
    /// strategy. Use [`section`](Chunk::section) for just the deepest header.
    pub section_path: Vec<String>,
}

impl Chunk {
    /// Start an empty chunk carrying `content`.
    ///
    /// `Chunk` is `#[non_exhaustive]`, so downstream crates cannot build one
    /// with a struct literal — new metadata can be added without a major
    /// version. This constructor and the `with_*` methods are the way to make
    /// one by hand, for tests and for adapters that re-materialise chunks from
    /// storage.
    ///
    /// ```
    /// let c = chunkedrs::Chunk::new("hello").with_bytes(0, 5);
    /// assert_eq!(c.end_byte, 5);
    /// ```
    pub fn new(content: impl Into<String>) -> Self {
        Self {
            content: content.into(),
            ..Self::default()
        }
    }

    /// Set the position in the chunk sequence.
    #[must_use]
    pub fn with_index(mut self, index: usize) -> Self {
        self.index = index;
        self
    }

    /// Set the byte range in the source document.
    #[must_use]
    pub fn with_bytes(mut self, start: usize, end: usize) -> Self {
        self.start_byte = start;
        self.end_byte = end;
        self
    }

    /// Set the token range in the document's token stream.
    #[must_use]
    pub fn with_tokens(mut self, start: usize, end: usize) -> Self {
        self.start_token = start;
        self.end_token = end;
        self
    }

    /// Set the number of tokens in `content`.
    #[must_use]
    pub fn with_token_count(mut self, count: usize) -> Self {
        self.token_count = count;
        self
    }

    /// Set the header ancestry, outermost first.
    #[must_use]
    pub fn with_section_path(mut self, path: Vec<String>) -> Self {
        self.section_path = path;
        self
    }

    /// The deepest section header this chunk sits under, if any.
    ///
    /// Equivalent to `section_path.last()`.
    ///
    /// ```
    /// let md = "# Title\n\nBody.\n";
    /// let chunks = chunkedrs::chunk(md).markdown().split();
    /// assert_eq!(chunks[0].section(), Some("# Title"));
    /// ```
    #[inline]
    pub fn section(&self) -> Option<&str> {
        self.section_path.last().map(String::as_str)
    }

    /// The byte length of the content.
    #[inline]
    pub fn len(&self) -> usize {
        self.content.len()
    }

    /// Whether the content is empty.
    #[inline]
    pub fn is_empty(&self) -> bool {
        self.content.is_empty()
    }
}

impl std::fmt::Display for Chunk {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.content)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn chunk_display() {
        assert_eq!(format!("{}", Chunk::new("hello")), "hello");
    }

    #[test]
    fn chunk_len_and_is_empty() {
        let c = Chunk::new("abc");
        assert_eq!(c.len(), 3);
        assert!(!c.is_empty());

        let empty = Chunk::new("");
        assert_eq!(empty.len(), 0);
        assert!(empty.is_empty());
    }

    #[test]
    fn chunk_clone_and_eq() {
        let c = Chunk::new("test")
            .with_index(1)
            .with_bytes(10, 14)
            .with_tokens(3, 4)
            .with_token_count(1)
            .with_section_path(vec!["## Intro".into()]);
        assert_eq!(c, c.clone());
    }

    #[test]
    fn section_is_the_deepest_path_entry() {
        let c = Chunk::new("x").with_section_path(vec!["# A".into(), "## B".into()]);
        assert_eq!(c.section(), Some("## B"));

        assert_eq!(Chunk::new("x").section(), None);
    }

    #[test]
    fn builder_defaults_are_zero() {
        let c = Chunk::new("x");
        assert_eq!(c.index, 0);
        assert_eq!((c.start_byte, c.end_byte), (0, 0));
        assert_eq!((c.start_token, c.end_token), (0, 0));
        assert_eq!(c.token_count, 0);
        assert!(c.section_path.is_empty());
    }
}
