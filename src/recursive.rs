use crate::Chunk;
use crate::byte_offset_of;
use tiktoken::CoreBpe;

/// default separators in priority order: paragraph → line → sentence → word
const SEPARATORS: &[&str] = &["\n\n", "\n", ". ", "! ", "? ", "; ", ", ", " "];

/// split text recursively at semantic boundaries, respecting token limits.
///
/// the algorithm:
/// 1. try splitting by the highest-priority separator (paragraph breaks first)
/// 2. merge consecutive pieces until adding one more would exceed max_tokens
/// 3. if a single piece exceeds max_tokens, recurse with the next separator
/// 4. at the lowest level, split by tokens (guaranteed to fit)
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

    // clamp overlap to guarantee forward progress (must be < max_tokens)
    let safe_overlap = overlap_tokens.min(max_tokens.saturating_sub(1));

    split_with_separators(
        text,
        text_offset,
        max_tokens,
        safe_overlap,
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
    // base case: token-level split
    if sep_index >= SEPARATORS.len() {
        return split_by_tokens(
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
    let mut piece_end = 0usize;

    for piece in pieces {
        let piece_tokens = encoder.count(piece);
        let piece_offset = byte_offset_of(piece, original_text);

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
            piece_end = piece_offset + piece.len();
            current_start = piece_end;
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
            current_start = piece_end - current.len();
        }

        if current.is_empty() {
            current_start = piece_offset;
        }
        current.push_str(piece);
        current_tokens += piece_tokens;
        piece_end = piece_offset + piece.len();
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

    // post-verify: re-split any chunks exceeding max_tokens.
    // BPE tokenization is non-additive, so the sum of piece token counts
    // may underestimate the actual merged token count. this pass catches
    // the rare edge case and guarantees the max_tokens invariant.
    // uses split_by_tokens (not split_recursive) to avoid infinite recursion.
    let needs_resplit = chunks.iter().any(|c| c.token_count > max_tokens);
    let mut final_chunks = if needs_resplit {
        let mut verified = Vec::new();
        for chunk in chunks {
            if chunk.token_count > max_tokens {
                let sub = split_by_tokens(
                    &chunk.content,
                    chunk.start_byte,
                    max_tokens,
                    0, // no overlap in post-verify to guarantee termination
                    encoder,
                    section,
                );
                verified.extend(sub);
            } else {
                verified.push(chunk);
            }
        }
        verified
    } else {
        chunks
    };

    // assign sequential indices
    for (i, chunk) in final_chunks.iter_mut().enumerate() {
        chunk.index = i;
    }

    final_chunks
}

/// split by tokens as last resort — precise byte offsets per token chunk
fn split_by_tokens(
    text: &str,
    text_offset: usize,
    max_tokens: usize,
    overlap_tokens: usize,
    encoder: &CoreBpe,
    section: &Option<String>,
) -> Vec<Chunk> {
    let tokens = encoder.encode(text);
    let mut chunks = Vec::new();

    // compute cumulative byte offsets via prefix sum for O(1) per-chunk lookup
    let token_byte_lens: Vec<usize> = tokens.iter().map(|&t| encoder.decode(&[t]).len()).collect();
    let mut prefix_sums = Vec::with_capacity(token_byte_lens.len() + 1);
    prefix_sums.push(0usize);
    for &len in &token_byte_lens {
        prefix_sums.push(prefix_sums.last().unwrap() + len);
    }

    let mut start = 0;

    while start < tokens.len() {
        let end = (start + max_tokens).min(tokens.len());

        let byte_start = prefix_sums[start];
        let byte_end = prefix_sums[end];

        // prefer slicing the original text to preserve valid UTF-8 without lossy conversion
        let content = text
            .get(byte_start..byte_end)
            .map(|s| s.to_string())
            .unwrap_or_else(|| {
                String::from_utf8_lossy(&encoder.decode(&tokens[start..end])).into_owned()
            });

        chunks.push(Chunk {
            content,
            index: chunks.len(),
            start_byte: text_offset + byte_start,
            end_byte: text_offset + byte_end,
            token_count: end - start,
            section: section.clone(),
        });

        // guarantee forward progress even with large overlap
        let advance = if overlap_tokens > 0 && end < tokens.len() {
            max_tokens.saturating_sub(overlap_tokens).max(1)
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

/// take approximately `n` tokens from the end of text, preserving valid UTF-8.
/// returns a substring of the original text (no lossy conversion).
fn take_suffix_tokens(text: &str, n: usize, encoder: &CoreBpe) -> String {
    let tokens = encoder.encode(text);
    if tokens.len() <= n {
        return text.to_string();
    }
    let prefix_byte_len = encoder.decode(&tokens[..tokens.len() - n]).len();
    // find next char boundary (BPE tokens may split multi-byte chars)
    let mut start = prefix_byte_len;
    while start < text.len() && !text.is_char_boundary(start) {
        start += 1;
    }
    text[start..].to_string()
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
        let chunks = split_recursive(text, 0, 10, 0, enc, &None);
        assert!(chunks.len() >= 2);
        assert_eq!(chunks[0].start_byte, 0);
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
        let text = "Alpha bravo charlie delta echo foxtrot golf hotel india juliet kilo lima mike november oscar papa.";
        let chunks = split_recursive(text, 0, 8, 2, enc, &None);
        assert!(chunks.len() >= 2);
        // with overlap, chunk N+1 should start with some text from the end of chunk N
        for i in 0..chunks.len() - 1 {
            let c1 = &chunks[i].content;
            let c2 = &chunks[i + 1].content;
            let c2_first_word = c2.split_whitespace().next().unwrap_or("");
            if !c2_first_word.is_empty() {
                assert!(
                    c1.contains(c2_first_word),
                    "chunk {}'s first word '{}' should appear in chunk {}: '{}'",
                    i + 1,
                    c2_first_word,
                    i,
                    c1
                );
            }
        }
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
    fn take_suffix_tokens_preserves_utf8() {
        let enc = encoder();
        let text = "こんにちは世界テスト";
        let suffix = take_suffix_tokens(text, 2, enc);
        // must be valid UTF-8 (no replacement characters)
        assert!(!suffix.contains('\u{FFFD}'));
        assert!(!suffix.is_empty());
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
        assert!(!chunks.is_empty());
        for chunk in &chunks {
            assert!(chunk.token_count <= 15);
        }
    }

    #[test]
    fn sentence_level_split() {
        let enc = encoder();
        let text = "First sentence here. Second sentence here. Third sentence here. Fourth sentence here. Fifth sentence here.";
        let chunks = split_recursive(text, 0, 8, 0, enc, &None);
        assert!(chunks.len() >= 2);
        for chunk in &chunks {
            assert!(chunk.token_count <= 8);
        }
    }

    #[test]
    fn single_long_word_split_by_tokens() {
        let enc = encoder();
        let text = "a".repeat(500);
        let chunks = split_recursive(&text, 0, 10, 0, enc, &None);
        assert!(chunks.len() >= 2);
        for chunk in &chunks {
            assert!(chunk.token_count <= 10);
        }
    }

    // --- new tests for bug fixes ---

    #[test]
    fn overlap_equal_to_max_tokens_does_not_hang() {
        let enc = encoder();
        let text = "hello world foo bar baz qux quux corge";
        // overlap == max_tokens should be clamped, not infinite loop
        let chunks = split_recursive(text, 0, 3, 3, enc, &None);
        assert!(!chunks.is_empty());
        for c in &chunks {
            assert!(c.token_count <= 3);
        }
    }

    #[test]
    fn overlap_exceeds_max_tokens_does_not_hang() {
        let enc = encoder();
        let text = "hello world foo bar baz qux quux corge";
        let chunks = split_recursive(text, 0, 3, 100, enc, &None);
        assert!(!chunks.is_empty());
        for c in &chunks {
            assert!(c.token_count <= 3);
        }
    }

    #[test]
    fn byte_offsets_match_content_no_overlap() {
        let enc = encoder();
        let text = "First paragraph here.\n\nSecond paragraph here.\n\nThird paragraph here.";
        let chunks = split_recursive(text, 0, 8, 0, enc, &None);
        assert!(chunks.len() >= 2);
        for chunk in &chunks {
            let extracted = &text[chunk.start_byte..chunk.end_byte];
            assert_eq!(
                extracted, chunk.content,
                "chunk {} byte offset mismatch: expected '{}', got '{}'",
                chunk.index, chunk.content, extracted
            );
        }
    }

    #[test]
    fn byte_offsets_match_content_with_overlap() {
        let enc = encoder();
        let text = "Alpha bravo charlie. Delta echo foxtrot. Golf hotel india.";
        let chunks = split_recursive(text, 0, 6, 2, enc, &None);
        assert!(chunks.len() >= 2);
        for chunk in &chunks {
            let extracted = &text[chunk.start_byte..chunk.end_byte];
            assert_eq!(
                extracted, chunk.content,
                "chunk {} byte offset mismatch with overlap",
                chunk.index
            );
        }
    }

    #[test]
    fn token_split_has_distinct_byte_offsets() {
        let enc = encoder();
        // long string with no separators — forces token-level split
        let text = "a".repeat(100);
        let chunks = split_recursive(&text, 0, 5, 0, enc, &None);
        assert!(chunks.len() >= 2);
        for i in 1..chunks.len() {
            assert!(
                chunks[i].start_byte > chunks[i - 1].start_byte,
                "chunk {} start ({}) should be after chunk {} start ({})",
                i,
                chunks[i].start_byte,
                i - 1,
                chunks[i - 1].start_byte
            );
        }
    }

    #[test]
    fn token_split_with_overlap() {
        let enc = encoder();
        let text = "a".repeat(100);
        let chunks = split_recursive(&text, 0, 10, 3, enc, &None);
        assert!(chunks.len() >= 2);
        for c in &chunks {
            assert!(c.token_count <= 10);
        }
        // overlapping chunks should have overlapping byte ranges
        for i in 1..chunks.len() {
            assert!(
                chunks[i].start_byte < chunks[i - 1].end_byte,
                "overlap should cause byte range overlap between chunk {} and {}",
                i - 1,
                i
            );
        }
    }

    #[test]
    fn max_tokens_guarantee_strict() {
        let enc = encoder();
        // various texts that might cause BPE boundary issues
        let texts = [
            "ab cd ef gh ij kl mn op qr st uv wx yz ".repeat(20),
            "The quick brown fox. ".repeat(100),
            "Hello! World? Yes. No! Maybe? ".repeat(50),
        ];
        for text in &texts {
            let chunks = split_recursive(text, 0, 7, 0, enc, &None);
            for chunk in &chunks {
                let actual = enc.count(&chunk.content);
                assert!(
                    actual <= 7,
                    "chunk {} has {} actual tokens, content: '{}'",
                    chunk.index,
                    actual,
                    &chunk.content[..chunk.content.len().min(50)]
                );
                assert_eq!(
                    chunk.token_count, actual,
                    "stored token_count must match actual"
                );
            }
        }
    }
}
