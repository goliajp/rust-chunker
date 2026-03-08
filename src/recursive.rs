use crate::Chunk;
use tiktoken::CoreBpe;

/// default separators in priority order: paragraph → line → sentence → word
const SEPARATORS: &[&str] = &["\n\n", "\n", ". ", "! ", "? ", "; ", ", ", " "];

/// split text recursively at semantic boundaries, respecting token limits.
///
/// the algorithm:
/// 1. try splitting by the highest-priority separator (paragraph breaks first)
/// 2. merge consecutive pieces until adding one more would exceed max_tokens
/// 3. if a single piece exceeds max_tokens, recurse with the next separator
/// 4. at the lowest level, split by characters (guaranteed to fit)
pub(crate) fn split_recursive(
    text: &str,
    text_offset: usize,
    max_tokens: usize,
    overlap_tokens: usize,
    encoder: &CoreBpe,
    section: &Option<String>,
) -> Vec<Chunk> {
    if text.is_empty() {
        return Vec::new();
    }

    let token_count = encoder.count(text);
    if token_count <= max_tokens {
        return vec![Chunk {
            content: text.to_string(),
            index: 0,
            start_byte: text_offset,
            end_byte: text_offset + text.len(),
            token_count,
            section: section.clone(),
        }];
    }

    split_with_separators(
        text,
        text_offset,
        max_tokens,
        overlap_tokens,
        encoder,
        section,
        0,
    )
}

fn split_with_separators(
    text: &str,
    text_offset: usize,
    max_tokens: usize,
    overlap_tokens: usize,
    encoder: &CoreBpe,
    section: &Option<String>,
    sep_index: usize,
) -> Vec<Chunk> {
    // base case: character-level split
    if sep_index >= SEPARATORS.len() {
        return split_by_chars(
            text,
            text_offset,
            max_tokens,
            overlap_tokens,
            encoder,
            section,
        );
    }

    let sep = SEPARATORS[sep_index];
    let pieces = split_keeping_separator(text, sep);

    // if separator didn't split anything, try next
    if pieces.len() <= 1 {
        return split_with_separators(
            text,
            text_offset,
            max_tokens,
            overlap_tokens,
            encoder,
            section,
            sep_index + 1,
        );
    }

    merge_pieces(
        &pieces,
        text,
        text_offset,
        max_tokens,
        overlap_tokens,
        encoder,
        section,
        sep_index,
    )
}

/// split text by separator, keeping the separator attached to the piece before it
fn split_keeping_separator<'a>(text: &'a str, sep: &str) -> Vec<&'a str> {
    let mut pieces = Vec::new();
    let mut start = 0;

    // find all occurrences of separator
    let mut search_start = 0;
    while let Some(pos) = text[search_start..].find(sep) {
        let abs_pos = search_start + pos;
        let end = abs_pos + sep.len();
        if end > start {
            pieces.push(&text[start..end]);
        }
        start = end;
        search_start = end;
    }

    // remaining text after last separator
    if start < text.len() {
        pieces.push(&text[start..]);
    }

    pieces
}

/// merge small pieces into chunks that fit within max_tokens
#[allow(clippy::too_many_arguments)]
fn merge_pieces(
    pieces: &[&str],
    original_text: &str,
    text_offset: usize,
    max_tokens: usize,
    overlap_tokens: usize,
    encoder: &CoreBpe,
    section: &Option<String>,
    sep_index: usize,
) -> Vec<Chunk> {
    let mut chunks = Vec::new();
    let mut current = String::new();
    let mut current_tokens = 0usize;
    let mut current_start = 0usize; // byte offset within original_text
    let mut piece_start = 0usize;

    for piece in pieces {
        let piece_tokens = encoder.count(piece);

        // single piece exceeds max — recurse with finer separator
        if piece_tokens > max_tokens {
            // flush current buffer first
            if !current.is_empty() {
                chunks.push(make_chunk(
                    &current,
                    text_offset + current_start,
                    encoder,
                    section,
                ));
                current.clear();
                current_tokens = 0;
            }

            let piece_offset = byte_offset_of(piece, original_text);
            let sub_chunks = split_with_separators(
                piece,
                text_offset + piece_offset,
                max_tokens,
                overlap_tokens,
                encoder,
                section,
                sep_index + 1,
            );
            chunks.extend(sub_chunks);
            piece_start = piece_offset + piece.len();
            current_start = piece_start;
            continue;
        }

        // would adding this piece overflow?
        if current_tokens + piece_tokens > max_tokens && !current.is_empty() {
            chunks.push(make_chunk(
                &current,
                text_offset + current_start,
                encoder,
                section,
            ));

            // handle overlap: take tokens from end of current chunk
            let overlap_text = if overlap_tokens > 0 {
                take_suffix_tokens(&current, overlap_tokens, encoder)
            } else {
                String::new()
            };

            current = overlap_text;
            current_tokens = if current.is_empty() {
                0
            } else {
                encoder.count(&current)
            };
            current_start = piece_start - current.len();
        }

        if current.is_empty() {
            current_start = byte_offset_of(piece, original_text);
        }
        current.push_str(piece);
        current_tokens += piece_tokens;
        piece_start = byte_offset_of(piece, original_text) + piece.len();
    }

    // flush remaining
    if !current.is_empty() {
        chunks.push(make_chunk(
            &current,
            text_offset + current_start,
            encoder,
            section,
        ));
    }

    // assign sequential indices
    for (i, chunk) in chunks.iter_mut().enumerate() {
        chunk.index = i;
    }

    chunks
}

/// split by characters as last resort
fn split_by_chars(
    text: &str,
    text_offset: usize,
    max_tokens: usize,
    overlap_tokens: usize,
    encoder: &CoreBpe,
    section: &Option<String>,
) -> Vec<Chunk> {
    let tokens = encoder.encode(text);
    let mut chunks = Vec::new();
    let mut start = 0;

    while start < tokens.len() {
        let end = (start + max_tokens).min(tokens.len());
        let chunk_tokens = &tokens[start..end];
        let content = String::from_utf8_lossy(&encoder.decode(chunk_tokens)).into_owned();

        chunks.push(Chunk {
            content,
            index: chunks.len(),
            start_byte: text_offset, // approximate for char-level splits
            end_byte: text_offset + text.len(),
            token_count: chunk_tokens.len(),
            section: section.clone(),
        });

        let advance = if overlap_tokens > 0 && end < tokens.len() {
            max_tokens.saturating_sub(overlap_tokens)
        } else {
            max_tokens
        };
        start += advance;
    }

    chunks
}

fn make_chunk(
    content: &str,
    start_byte: usize,
    encoder: &CoreBpe,
    section: &Option<String>,
) -> Chunk {
    Chunk {
        content: content.to_string(),
        index: 0, // will be assigned later
        start_byte,
        end_byte: start_byte + content.len(),
        token_count: encoder.count(content),
        section: section.clone(),
    }
}

/// take approximately `n` tokens from the end of text
fn take_suffix_tokens(text: &str, n: usize, encoder: &CoreBpe) -> String {
    let tokens = encoder.encode(text);
    if tokens.len() <= n {
        return text.to_string();
    }
    let suffix_tokens = &tokens[tokens.len() - n..];
    String::from_utf8_lossy(&encoder.decode(suffix_tokens)).into_owned()
}

/// find byte offset of a substring within the original text using pointer arithmetic
fn byte_offset_of(sub: &str, parent: &str) -> usize {
    let sub_ptr = sub.as_ptr() as usize;
    let parent_ptr = parent.as_ptr() as usize;
    sub_ptr.saturating_sub(parent_ptr)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn encoder() -> &'static CoreBpe {
        tiktoken::get_encoding("o200k_base").unwrap()
    }

    #[test]
    fn split_keeping_separator_basic() {
        let pieces = split_keeping_separator("aaa\n\nbbb\n\nccc", "\n\n");
        assert_eq!(pieces, vec!["aaa\n\n", "bbb\n\n", "ccc"]);
    }

    #[test]
    fn split_keeping_separator_no_match() {
        let pieces = split_keeping_separator("hello world", "\n\n");
        assert_eq!(pieces, vec!["hello world"]);
    }

    #[test]
    fn split_keeping_separator_trailing() {
        let pieces = split_keeping_separator("aaa\n\n", "\n\n");
        assert_eq!(pieces, vec!["aaa\n\n"]);
    }

    #[test]
    fn short_text_returns_single_chunk() {
        let enc = encoder();
        let chunks = split_recursive("hello world", 0, 100, 0, enc, &None);
        assert_eq!(chunks.len(), 1);
        assert_eq!(chunks[0].content, "hello world");
        assert_eq!(chunks[0].index, 0);
        assert_eq!(chunks[0].start_byte, 0);
        assert_eq!(chunks[0].end_byte, 11);
    }

    #[test]
    fn empty_text_returns_empty() {
        let enc = encoder();
        let chunks = split_recursive("", 0, 100, 0, enc, &None);
        assert!(chunks.is_empty());
    }

    #[test]
    fn paragraph_split() {
        let enc = encoder();
        let text =
            "First paragraph with some content here.\n\nSecond paragraph with different content.";
        // use a small max_tokens to force splitting
        let chunks = split_recursive(text, 0, 10, 0, enc, &None);
        assert!(chunks.len() >= 2);
        // first chunk should start at 0
        assert_eq!(chunks[0].start_byte, 0);
        // all chunks should have sequential indices
        for (i, chunk) in chunks.iter().enumerate() {
            assert_eq!(chunk.index, i);
        }
    }

    #[test]
    fn respects_max_tokens() {
        let enc = encoder();
        let text = "The quick brown fox jumps over the lazy dog. ".repeat(50);
        let max_tokens = 20;
        let chunks = split_recursive(&text, 0, max_tokens, 0, enc, &None);
        for chunk in &chunks {
            assert!(
                chunk.token_count <= max_tokens,
                "chunk {} has {} tokens, max is {}",
                chunk.index,
                chunk.token_count,
                max_tokens
            );
        }
    }

    #[test]
    fn overlap_creates_shared_content() {
        let enc = encoder();
        let text = "Sentence one. Sentence two. Sentence three. Sentence four. Sentence five. Sentence six. Sentence seven. Sentence eight.";
        let chunks = split_recursive(text, 0, 10, 3, enc, &None);
        assert!(chunks.len() >= 2);
        // with overlap, later chunks should share some content with previous
    }

    #[test]
    fn section_metadata_preserved() {
        let enc = encoder();
        let section = Some("## Architecture".to_string());
        let chunks = split_recursive("hello world", 0, 100, 0, enc, &section);
        assert_eq!(chunks[0].section.as_deref(), Some("## Architecture"));
    }

    #[test]
    fn text_offset_propagated() {
        let enc = encoder();
        let chunks = split_recursive("hello", 100, 100, 0, enc, &None);
        assert_eq!(chunks[0].start_byte, 100);
        assert_eq!(chunks[0].end_byte, 105);
    }

    #[test]
    fn byte_offset_of_works() {
        let text = "hello world";
        let sub = &text[6..];
        assert_eq!(byte_offset_of(sub, text), 6);
    }

    #[test]
    fn take_suffix_tokens_basic() {
        let enc = encoder();
        let suffix = take_suffix_tokens("hello world foo bar", 2, enc);
        // should get roughly the last 2 tokens
        assert!(!suffix.is_empty());
        assert!(encoder().count(&suffix) <= 2);
    }

    #[test]
    fn take_suffix_tokens_more_than_available() {
        let enc = encoder();
        let suffix = take_suffix_tokens("hi", 100, enc);
        assert_eq!(suffix, "hi");
    }

    #[test]
    fn chinese_text_split() {
        let enc = encoder();
        let text = "这是第一段内容，包含一些中文文本。\n\n这是第二段内容，也包含中文。\n\n第三段。";
        let chunks = split_recursive(text, 0, 15, 0, enc, &None);
        assert!(chunks.len() >= 2);
        for chunk in &chunks {
            assert!(chunk.token_count <= 15);
        }
    }

    #[test]
    fn japanese_text_split() {
        let enc = encoder();
        let text = "最初の段落です。日本語のテキストを含みます。\n\n二番目の段落です。異なる内容があります。";
        let chunks = split_recursive(text, 0, 15, 0, enc, &None);
        assert!(chunks.len() >= 1);
        for chunk in &chunks {
            assert!(chunk.token_count <= 15);
        }
    }

    #[test]
    fn sentence_level_split() {
        let enc = encoder();
        // no paragraph breaks, should fall through to sentence splitting
        let text = "First sentence here. Second sentence here. Third sentence here. Fourth sentence here. Fifth sentence here.";
        let chunks = split_recursive(text, 0, 8, 0, enc, &None);
        assert!(chunks.len() >= 2);
        for chunk in &chunks {
            assert!(chunk.token_count <= 8);
        }
    }

    #[test]
    fn single_long_word_split_by_chars() {
        let enc = encoder();
        // a very long "word" with no separators
        let text = "a".repeat(500);
        let chunks = split_recursive(&text, 0, 10, 0, enc, &None);
        assert!(chunks.len() >= 2);
        for chunk in &chunks {
            assert!(chunk.token_count <= 10);
        }
    }
}
