use crate::Chunk;
use crate::byte_offset_of;
use tiktoken::CoreBpe;

/// Separator tiers in priority order: paragraph → line → sentence → clause → word.
///
/// Within a tier every separator is a candidate and the *earliest* match wins,
/// so a document mixing scripts splits on whichever mark comes first instead of
/// letting one script's punctuation starve the other's.
///
/// The CJK marks matter more than they look. Chinese and Japanese do not put a
/// space after a sentence mark, so a ladder whose every entry ends in an ASCII
/// space (`". "`, `"! "`, `", "`, `" "`) matches nothing at all in CJK prose —
/// it fell through every tier to the token-level fallback and cut mid-word.
pub(crate) const TEXT_TIERS: &[&[&str]] = &[
    // paragraph
    &["\r\n\r\n", "\n\n"],
    // line
    &["\r\n", "\n"],
    // sentence-final
    &[
        ". ", "! ", "? ", // ASCII, space-delimited
        "。", "！", "？", "．", "｡", // CJK full/half-width terminals
        "…", "‥", // ellipses
    ],
    // clause
    &[
        "; ", ", ", ":", // ASCII
        "；", "，", "、", "：", "･", // CJK
    ],
    // word
    &[" ", "\t", "\u{3000}"],
];

/// Marks that close something the previous sentence opened. When a separator
/// match is followed by a run of these, they belong to the piece that just
/// ended — otherwise a chunk begins with an orphaned `」` or `)`.
///
/// Horizontal whitespace is absorbed for the same reason: a chunk should not
/// open with the space that trailed the previous sentence.
const TRAILING_ABSORB: &[char] = &[
    '」', '』', '）', '》', '〉', '】', '｝', '〕', '｣', '〞', '＞', // CJK closers
    '”', '’', '"', '\'', ')', ']', '}', '>', // ASCII / typographic closers
    '！', '？', '。', '．', // repeated terminals: "？！", "。。。"
    ' ', '\t', '\u{3000}', // trailing horizontal space
];

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
    section_path: &[String],
) -> Vec<Chunk> {
    split_with_tiers(
        text,
        text_offset,
        max_tokens,
        overlap_tokens,
        encoder,
        section_path,
        TEXT_TIERS,
    )
}

/// Everything the descent needs that does not change as it recurses.
///
/// Bundling these is not cosmetic: the descent passes them through four
/// functions, and threading them individually is what earned the old
/// `too_many_arguments` allow.
#[derive(Clone, Copy)]
pub(crate) struct Ctx<'a> {
    pub max_tokens: usize,
    pub overlap_tokens: usize,
    pub encoder: &'a CoreBpe,
    pub section_path: &'a [String],
    /// Separator tiers, tried outermost first. Each strategy brings its own —
    /// prose splits on sentence marks, code on block boundaries.
    pub tiers: &'static [&'static [&'static str]],
}

/// Split with an explicit separator ladder.
pub(crate) fn split_with_tiers(
    text: &str,
    text_offset: usize,
    max_tokens: usize,
    overlap_tokens: usize,
    encoder: &CoreBpe,
    section_path: &[String],
    tiers: &'static [&'static [&'static str]],
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
            // token spans are filled in by a document-level pass in lib.rs,
            // which is the only place that can see the whole token stream
            start_token: 0,
            end_token: 0,
            token_count,
            section_path: section_path.to_vec(),
        }];
    }

    let ctx = Ctx {
        max_tokens,
        // clamp overlap to guarantee forward progress (must be < max_tokens)
        overlap_tokens: overlap_tokens.min(max_tokens.saturating_sub(1)),
        encoder,
        section_path,
        tiers,
    };

    descend(text, text_offset, ctx, 0)
}

fn descend(text: &str, text_offset: usize, ctx: Ctx<'_>, tier: usize) -> Vec<Chunk> {
    // base case: token-level split
    if tier >= ctx.tiers.len() {
        return split_by_tokens(
            text,
            text_offset,
            ctx.max_tokens,
            ctx.overlap_tokens,
            ctx.encoder,
            ctx.section_path,
        );
    }

    let pieces = split_at_any(text, ctx.tiers[tier]);

    // if this tier didn't split anything, try the next
    if pieces.len() <= 1 {
        return descend(text, text_offset, ctx, tier + 1);
    }

    merge_pieces(&pieces, text, text_offset, ctx, tier + 1)
}

/// Split text at any separator in `seps`, keeping the separator attached to the
/// piece before it.
///
/// The earliest match wins; ties go to the longest separator so `"\r\n"` is not
/// shadowed by `"\n"`. Any run of [`TRAILING_ABSORB`] following the match is
/// pulled back into the same piece.
pub(crate) fn split_at_any<'a>(text: &'a str, seps: &[&str]) -> Vec<&'a str> {
    let mut pieces = Vec::new();
    let mut start = 0;
    let mut search = 0;

    while search < text.len() {
        let hit = seps
            .iter()
            .filter_map(|s| text[search..].find(s).map(|p| (search + p, s.len())))
            .min_by_key(|&(pos, len)| (pos, std::cmp::Reverse(len)));

        let Some((pos, len)) = hit else { break };

        let end = absorb_trailing(text, pos + len);
        if end > start {
            pieces.push(&text[start..end]);
        }
        start = end;
        search = end;
    }

    // remaining text after the last separator
    if start < text.len() {
        pieces.push(&text[start..]);
    }

    pieces
}

/// Extend `from` over any run of closing marks and horizontal whitespace, so
/// they stay with the sentence they belong to rather than opening the next one.
fn absorb_trailing(text: &str, from: usize) -> usize {
    let mut end = from;
    while let Some(c) = text[end..].chars().next() {
        if TRAILING_ABSORB.contains(&c) {
            end += c.len_utf8();
        } else {
            break;
        }
    }
    end
}

/// Merge pieces into chunks that fit within `max_tokens`.
///
/// `next_tier` is where an oversized piece goes to be split further. Strategies
/// that produce their own pieces (HTML block tags, say) pass `0` to hand the
/// remainder to the whole ladder.
pub(crate) fn merge_pieces(
    pieces: &[&str],
    original_text: &str,
    text_offset: usize,
    ctx: Ctx<'_>,
    next_tier: usize,
) -> Vec<Chunk> {
    let Ctx {
        max_tokens,
        overlap_tokens,
        encoder,
        section_path,
        ..
    } = ctx;

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
                    section_path,
                ));
                current.clear();
                current_tokens = 0;
            }

            let sub_chunks = descend(piece, text_offset + piece_offset, ctx, next_tier);
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
                section_path,
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
            section_path,
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
                    section_path,
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
    section_path: &[String],
) -> Vec<Chunk> {
    let tokens = encoder.encode(text);
    let mut chunks = Vec::new();

    // Cumulative byte offsets, snapped back to character boundaries.
    //
    // A single character can span several tokens (rare CJK and emoji fall back
    // to byte-level tokens), so a raw token boundary is not necessarily a
    // character boundary. Snapping *back* rather than forward keeps each slice
    // a subset of its token window, which is what preserves the `max_tokens`
    // guarantee; the bytes trimmed here are picked up by the next chunk, so the
    // split stays lossless.
    let mut boundaries = Vec::with_capacity(tokens.len() + 1);
    boundaries.push(0usize);
    let mut acc = 0usize;
    for &t in &tokens {
        acc += encoder.decode(&[t]).len();
        boundaries.push(floor_char_boundary(text, acc));
    }

    let mut start = 0;

    while start < tokens.len() {
        let end = (start + max_tokens).min(tokens.len());

        let byte_start = boundaries[start];
        let byte_end = boundaries[end];

        // Skip windows that collapse to nothing — every token in them was a
        // partial byte sequence of a character the next window will emit whole.
        if byte_end > byte_start {
            let content = text[byte_start..byte_end].to_string();
            // Re-count rather than trusting `end - start`: the boundary snap can
            // trim bytes, and a stored count that disagrees with the content
            // would be a lie in the one field users budget against.
            let token_count = encoder.count(&content);

            chunks.push(Chunk {
                content,
                index: chunks.len(),
                start_byte: text_offset + byte_start,
                end_byte: text_offset + byte_end,
                start_token: 0,
                end_token: 0,
                token_count,
                section_path: section_path.to_vec(),
            });
        }

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

/// Largest character boundary at or below `i`.
fn floor_char_boundary(text: &str, i: usize) -> usize {
    let mut i = i.min(text.len());
    while i > 0 && !text.is_char_boundary(i) {
        i -= 1;
    }
    i
}

fn make_chunk(
    content: &str,
    start_byte: usize,
    encoder: &CoreBpe,
    section_path: &[String],
) -> Chunk {
    Chunk {
        content: content.to_string(),
        index: 0, // will be assigned later
        start_byte,
        end_byte: start_byte + content.len(),
        start_token: 0,
        end_token: 0,
        token_count: encoder.count(content),
        section_path: section_path.to_vec(),
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
    fn split_at_any_basic() {
        let pieces = split_at_any("aaa\n\nbbb\n\nccc", &["\n\n"]);
        assert_eq!(pieces, vec!["aaa\n\n", "bbb\n\n", "ccc"]);
    }

    #[test]
    fn split_at_any_no_match() {
        let pieces = split_at_any("hello world", &["\n\n"]);
        assert_eq!(pieces, vec!["hello world"]);
    }

    #[test]
    fn split_at_any_trailing() {
        let pieces = split_at_any("aaa\n\n", &["\n\n"]);
        assert_eq!(pieces, vec!["aaa\n\n"]);
    }

    #[test]
    fn split_at_any_takes_earliest_match_across_separators() {
        // '?' comes first even though '.' is listed first — earliest wins, so
        // one script's punctuation cannot starve another's.
        let pieces = split_at_any("who? me. you!", &[". ", "! ", "? "]);
        assert_eq!(pieces, vec!["who? ", "me. ", "you!"]);
    }

    #[test]
    fn split_at_any_prefers_longer_separator_on_tie() {
        let pieces = split_at_any("a\r\nb", &["\n", "\r\n"]);
        assert_eq!(pieces, vec!["a\r\n", "b"]);
    }

    #[test]
    fn split_at_any_splits_cjk_without_spaces() {
        let pieces = split_at_any("第一句。第二句。第三句。", &["。"]);
        assert_eq!(pieces, vec!["第一句。", "第二句。", "第三句。"]);
    }

    #[test]
    fn split_at_any_absorbs_closing_marks() {
        let pieces = split_at_any("他说「你好。」然后走了。", &["。"]);
        assert_eq!(pieces, vec!["他说「你好。」", "然后走了。"]);
    }

    #[test]
    fn split_at_any_absorbs_repeated_terminals() {
        let pieces = split_at_any("真的？！那好吧。", &["。", "？"]);
        assert_eq!(pieces, vec!["真的？！", "那好吧。"]);
    }

    #[test]
    fn absorb_trailing_stops_at_ordinary_text() {
        assert_eq!(absorb_trailing("」）abc", 0), "」）".len());
        assert_eq!(absorb_trailing("abc", 0), 0);
    }

    #[test]
    fn floor_char_boundary_snaps_back() {
        let s = "日本語"; // 3 bytes per char
        assert_eq!(floor_char_boundary(s, 0), 0);
        assert_eq!(floor_char_boundary(s, 1), 0);
        assert_eq!(floor_char_boundary(s, 2), 0);
        assert_eq!(floor_char_boundary(s, 3), 3);
        assert_eq!(floor_char_boundary(s, 4), 3);
        assert_eq!(floor_char_boundary(s, 99), s.len());
    }

    #[test]
    fn short_text_returns_single_chunk() {
        let enc = encoder();
        let chunks = split_recursive("hello world", 0, 100, 0, enc, &[]);
        assert_eq!(chunks.len(), 1);
        assert_eq!(chunks[0].content, "hello world");
        assert_eq!(chunks[0].index, 0);
        assert_eq!(chunks[0].start_byte, 0);
        assert_eq!(chunks[0].end_byte, 11);
    }

    #[test]
    fn empty_text_returns_empty() {
        let enc = encoder();
        let chunks = split_recursive("", 0, 100, 0, enc, &[]);
        assert!(chunks.is_empty());
    }

    #[test]
    fn paragraph_split() {
        let enc = encoder();
        let text =
            "First paragraph with some content here.\n\nSecond paragraph with different content.";
        let chunks = split_recursive(text, 0, 10, 0, enc, &[]);
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
        let chunks = split_recursive(&text, 0, max_tokens, 0, enc, &[]);
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
        let chunks = split_recursive(text, 0, 8, 2, enc, &[]);
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
        let path = vec!["# Design".to_string(), "## Architecture".to_string()];
        let chunks = split_recursive("hello world", 0, 100, 0, enc, &path);
        assert_eq!(chunks[0].section_path, path);
        assert_eq!(chunks[0].section(), Some("## Architecture"));
    }

    #[test]
    fn text_offset_propagated() {
        let enc = encoder();
        let chunks = split_recursive("hello", 100, 100, 0, enc, &[]);
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
        let chunks = split_recursive(text, 0, 15, 0, enc, &[]);
        assert!(chunks.len() >= 2);
        for chunk in &chunks {
            assert!(chunk.token_count <= 15);
        }
    }

    #[test]
    fn japanese_text_split() {
        let enc = encoder();
        let text = "最初の段落です。日本語のテキストを含みます。\n\n二番目の段落です。異なる内容があります。";
        let chunks = split_recursive(text, 0, 15, 0, enc, &[]);
        assert!(!chunks.is_empty());
        for chunk in &chunks {
            assert!(chunk.token_count <= 15);
        }
    }

    #[test]
    fn sentence_level_split() {
        let enc = encoder();
        let text = "First sentence here. Second sentence here. Third sentence here. Fourth sentence here. Fifth sentence here.";
        let chunks = split_recursive(text, 0, 8, 0, enc, &[]);
        assert!(chunks.len() >= 2);
        for chunk in &chunks {
            assert!(chunk.token_count <= 8);
        }
    }

    #[test]
    fn single_long_word_split_by_tokens() {
        let enc = encoder();
        let text = "a".repeat(500);
        let chunks = split_recursive(&text, 0, 10, 0, enc, &[]);
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
        let chunks = split_recursive(text, 0, 3, 3, enc, &[]);
        assert!(!chunks.is_empty());
        for c in &chunks {
            assert!(c.token_count <= 3);
        }
    }

    #[test]
    fn overlap_exceeds_max_tokens_does_not_hang() {
        let enc = encoder();
        let text = "hello world foo bar baz qux quux corge";
        let chunks = split_recursive(text, 0, 3, 100, enc, &[]);
        assert!(!chunks.is_empty());
        for c in &chunks {
            assert!(c.token_count <= 3);
        }
    }

    #[test]
    fn byte_offsets_match_content_no_overlap() {
        let enc = encoder();
        let text = "First paragraph here.\n\nSecond paragraph here.\n\nThird paragraph here.";
        let chunks = split_recursive(text, 0, 8, 0, enc, &[]);
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
        let chunks = split_recursive(text, 0, 6, 2, enc, &[]);
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
        let chunks = split_recursive(&text, 0, 5, 0, enc, &[]);
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
        let chunks = split_recursive(&text, 0, 10, 3, enc, &[]);
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
            let chunks = split_recursive(text, 0, 7, 0, enc, &[]);
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
