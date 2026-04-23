/// A piece of text produced by splitting a larger document.
///
/// Every chunk knows where it came from — byte offsets into the original text,
/// its position in the sequence, and its token count. This metadata enables
/// accurate citation, deduplication, and context-window budgeting in RAG pipelines.
///
/// ```
/// # let chunks = vec![chunkedrs::Chunk {
/// #     content: "Hello world".into(),
/// #     index: 0,
/// #     start_byte: 0,
/// #     end_byte: 11,
/// #     token_count: 2,
/// #     section: None,
/// # }];
/// for chunk in &chunks {
///     println!("[{}] {}..{} ({} tokens)", chunk.index, chunk.start_byte, chunk.end_byte, chunk.token_count);
/// }
/// ```
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Chunk {
    /// The text content of this chunk.
    pub content: String,

    /// Zero-based position in the chunk sequence.
    pub index: usize,

    /// Byte offset of the first character in the original text.
    pub start_byte: usize,

    /// Byte offset one past the last character in the original text.
    pub end_byte: usize,

    /// Number of tokens as counted by the tokenizer.
    pub token_count: usize,

    /// Section header from markdown (e.g. `"## Architecture"`), if markdown-aware splitting was used.
    pub section: Option<String>,
}

impl Chunk {
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
        let c = Chunk {
            content: "hello".into(),
            index: 0,
            start_byte: 0,
            end_byte: 5,
            token_count: 1,
            section: None,
        };
        assert_eq!(format!("{c}"), "hello");
    }

    #[test]
    fn chunk_len_and_is_empty() {
        let c = Chunk {
            content: "abc".into(),
            index: 0,
            start_byte: 0,
            end_byte: 3,
            token_count: 1,
            section: None,
        };
        assert_eq!(c.len(), 3);
        assert!(!c.is_empty());

        let empty = Chunk {
            content: String::new(),
            index: 0,
            start_byte: 0,
            end_byte: 0,
            token_count: 0,
            section: None,
        };
        assert_eq!(empty.len(), 0);
        assert!(empty.is_empty());
    }

    #[test]
    fn chunk_clone_and_eq() {
        let c = Chunk {
            content: "test".into(),
            index: 1,
            start_byte: 10,
            end_byte: 14,
            token_count: 1,
            section: Some("## Intro".into()),
        };
        let c2 = c.clone();
        assert_eq!(c, c2);
    }

    #[test]
    fn chunk_with_section() {
        let c = Chunk {
            content: "content".into(),
            index: 0,
            start_byte: 0,
            end_byte: 7,
            token_count: 1,
            section: Some("# Title".into()),
        };
        assert_eq!(c.section.as_deref(), Some("# Title"));
    }
}
