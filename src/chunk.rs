use std::ops::Range;

/// One chunk of a document, borrowed from it.
///
/// `content` is a slice of the text that was split, so a chunk costs no copy
/// of its text; [`to_owned_chunk`](Chunk::to_owned_chunk) gives an
/// [`OwnedChunk`] to store or send elsewhere.
///
/// # Byte offsets vs token spans
///
/// `start_byte..end_byte` addresses the source text: `&text[start_byte..end_byte]`
/// is `content`, the same bytes at the same address.
///
/// `start_token..end_token` addresses the document's *token stream*, the range
/// of `encoder.encode(whole_document)` that covers this chunk (see
/// [`Chunker::split_with_tokens`](crate::Chunker::split_with_tokens)). That is
/// what [late chunking] needs: embed the document once, then pool each
/// chunk's vector over its own token range, so every chunk embedding carries
/// the context of the whole document.
///
/// # `token_count` is not `end_token - start_token`
///
/// `token_count` is the count of `content` on its own: the number you budget
/// a context window against, and the number `max_tokens` bounds. The span
/// width is this chunk's footprint in the document stream. BPE merges across
/// boundaries, so a fragment counted on its own does not always reproduce its
/// slice of the whole; and where a chunk boundary falls inside a token, the
/// span widens to cover it. Both numbers are real and they answer different
/// questions.
///
/// [late chunking]: https://arxiv.org/abs/2409.04701
///
/// ```
/// let chunks = chunkedrs::split("Hello world. Goodbye world.");
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
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
#[non_exhaustive]
pub struct Chunk<'a> {
    /// The text of this chunk, a slice of the document.
    pub content: &'a str,

    /// Zero-based position in the chunk sequence.
    pub index: usize,

    /// Byte offset of the first character in the document.
    pub start_byte: usize,

    /// Byte offset one past the last character in the document.
    pub end_byte: usize,

    /// Index of the first token of this chunk in the document's token stream.
    pub start_token: usize,

    /// Index one past this chunk's last token in the document's token stream.
    pub end_token: usize,

    /// Number of tokens in `content`, counted on its own.
    pub token_count: usize,

    /// Header lines from the markdown strategy, outermost first, e.g.
    /// `["# Guide", "## Installation", "### From source"]`; slices of the
    /// document.
    ///
    /// Empty for content before the first header and for every other
    /// strategy. Use [`section`](Chunk::section) for just the deepest header.
    pub section_path: Vec<&'a str>,
}

impl<'a> Chunk<'a> {
    /// The innermost markdown header this chunk sits under, if any.
    ///
    /// ```
    /// let md = "# Guide\n\nIntro.\n\n## Install\n\nRun it.\n";
    /// let chunks = chunkedrs::Chunker::new().markdown().split(md);
    /// assert_eq!(chunks[1].section(), Some("## Install"));
    /// ```
    #[inline]
    #[must_use]
    pub fn section(&self) -> Option<&'a str> {
        self.section_path.last().copied()
    }

    /// `start_byte..end_byte`.
    #[inline]
    #[must_use]
    pub fn byte_range(&self) -> Range<usize> {
        self.start_byte..self.end_byte
    }

    /// `start_token..end_token`.
    #[inline]
    #[must_use]
    pub fn token_range(&self) -> Range<usize> {
        self.start_token..self.end_token
    }

    /// The same chunk with its text and section path copied, free of the
    /// document's lifetime.
    #[must_use]
    pub fn to_owned_chunk(&self) -> OwnedChunk {
        OwnedChunk {
            content: self.content.to_owned(),
            index: self.index,
            start_byte: self.start_byte,
            end_byte: self.end_byte,
            start_token: self.start_token,
            end_token: self.end_token,
            token_count: self.token_count,
            section_path: self.section_path.iter().map(|&s| s.to_owned()).collect(),
        }
    }
}

impl std::fmt::Display for Chunk<'_> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(self.content)
    }
}

impl From<Chunk<'_>> for OwnedChunk {
    fn from(c: Chunk<'_>) -> Self {
        c.to_owned_chunk()
    }
}

/// A [`Chunk`] that owns its text, for storing chunks past the document's
/// lifetime. The fields mean what they mean on [`Chunk`].
///
/// ```
/// let owned: Vec<chunkedrs::OwnedChunk> = {
///     let doc = String::from("A short document.");
///     chunkedrs::split(&doc).iter().map(|c| c.to_owned_chunk()).collect()
/// };
/// assert_eq!(owned[0].content, "A short document.");
/// ```
#[derive(Debug, Clone, Default, PartialEq, Eq, Hash)]
#[non_exhaustive]
pub struct OwnedChunk {
    /// The text of this chunk.
    pub content: String,
    /// Zero-based position in the chunk sequence.
    pub index: usize,
    /// Byte offset of the first character in the document.
    pub start_byte: usize,
    /// Byte offset one past the last character in the document.
    pub end_byte: usize,
    /// Index of the first token of this chunk in the document's token stream.
    pub start_token: usize,
    /// Index one past this chunk's last token in the document's token stream.
    pub end_token: usize,
    /// Number of tokens in `content`, counted on its own.
    pub token_count: usize,
    /// Header lines from the markdown strategy, outermost first.
    pub section_path: Vec<String>,
}

impl OwnedChunk {
    /// The innermost markdown header this chunk sits under, if any.
    #[inline]
    #[must_use]
    pub fn section(&self) -> Option<&str> {
        self.section_path.last().map(String::as_str)
    }
}

impl std::fmt::Display for OwnedChunk {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(&self.content)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn owned_round_trip() {
        let doc = "# A\n\nbody text\n";
        let chunks = crate::Chunker::new().markdown().split(doc);
        let c = &chunks[0];
        let o = c.to_owned_chunk();
        assert_eq!(o.content, c.content);
        assert_eq!(o.section(), c.section());
        assert_eq!(o.section_path, ["# A"]);
        assert_eq!(&doc[o.start_byte..o.end_byte], o.content);
        assert_eq!(OwnedChunk::from(c.clone()), o);
        assert_eq!(format!("{c}"), format!("{o}"));
        assert_eq!(c.byte_range(), c.start_byte..c.end_byte);
        assert_eq!(c.token_range(), c.start_token..c.end_token);
    }
}
