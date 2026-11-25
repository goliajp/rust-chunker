use crate::Chunk;
use crate::plan::{Doc, plan, tiers};

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

/// Split the whole document on [`TEXT_TIERS`]: paragraph, line, sentence,
/// clause, word, and token boundaries as the last resort.
pub(crate) fn split_recursive(doc: &Doc<'_>, max_tokens: usize, overlap: usize) -> Vec<Chunk> {
    plan(
        doc,
        0,
        doc.text.len(),
        &tiers(TEXT_TIERS),
        max_tokens,
        overlap,
    )
    .into_iter()
    .map(|p| doc.chunk(p, &[]))
    .collect()
}

/// Split text at any separator in `seps`, keeping the separator attached to the
/// piece before it.
///
/// The earliest match wins; ties go to the longest separator so `"\r\n"` is not
/// shadowed by `"\n"`. Any run of [`TRAILING_ABSORB`] following the match is
/// pulled back into the same piece.
#[cfg(any(test, feature = "semantic"))]
pub(crate) fn split_at_any<'a>(text: &'a str, seps: &[&str]) -> Vec<&'a str> {
    let bytes = text.as_bytes();
    // one forward pass: only bytes that can start a separator are tried
    let mut lead = [false; 256];
    for sep in seps {
        lead[sep.as_bytes()[0] as usize] = true;
    }

    let mut pieces = Vec::new();
    let mut start = 0;
    let mut i = 0;
    while i < bytes.len() {
        if !lead[bytes[i] as usize] {
            i += 1;
            continue;
        }
        let rest = &bytes[i..];
        let Some(len) = seps
            .iter()
            .filter(|s| rest.starts_with(s.as_bytes()))
            .map(|s| s.len())
            .max()
        else {
            i += 1;
            continue;
        };
        let end = absorb_trailing(text, i + len);
        if end > start {
            pieces.push(&text[start..end]);
        }
        start = end;
        i = end;
    }

    // remaining text after the last separator
    if start < text.len() {
        pieces.push(&text[start..]);
    }

    pieces
}

/// Extend `from` over any run of closing marks and horizontal whitespace, so
/// they stay with the sentence they belong to rather than opening the next one.
#[inline]
pub(crate) fn absorb_trailing(text: &str, from: usize) -> usize {
    let bytes = text.as_bytes();
    let mut end = from;
    while let Some(&b) = bytes.get(end) {
        let len = match b {
            // the ASCII members of TRAILING_ABSORB
            b' ' | b'\t' | b'"' | b'\'' | b')' | b']' | b'}' | b'>' => 1,
            // every other member is a three-byte character starting with one of these
            0xE2 | 0xE3 | 0xEF => match text[end..].chars().next() {
                Some(c) if TRAILING_ABSORB.contains(&c) => 3,
                _ => 0,
            },
            _ => 0,
        };
        if len == 0 {
            break;
        }
        end += len;
    }
    end
}

#[cfg(test)]
mod tests {
    use super::*;

    fn encoder() -> &'static tiktoken::CoreBpe {
        tiktoken::get_encoding("o200k_base").unwrap()
    }

    fn split_recursive(
        text: &str,
        _offset: usize,
        max: usize,
        overlap: usize,
        enc: &tiktoken::CoreBpe,
        _path: &[String],
    ) -> Vec<Chunk> {
        let mut chunks = super::split_recursive(&Doc::new(text, enc), max, overlap);
        crate::finish(&mut chunks);
        chunks
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

    /// The straightforward version: `find` every separator from the cursor and
    /// keep the earliest, longest match.
    fn split_at_any_reference<'a>(text: &'a str, seps: &[&str]) -> Vec<&'a str> {
        let (mut pieces, mut start, mut search) = (Vec::new(), 0, 0);
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
        if start < text.len() {
            pieces.push(&text[start..]);
        }
        pieces
    }

    #[test]
    fn split_at_any_matches_reference() {
        let alphabet = [
            "a", "bc", " ", ". ", "\n", "\n\n", "\r\n", "。", "」", "，", "…", "? ", "\t", "語",
            "\n}\n", "end", ")", "\u{3000}",
        ];
        let mut x: u64 = 0x2545_f491;
        for _ in 0..2000 {
            let mut text = String::new();
            for _ in 0..40 {
                x ^= x << 13;
                x ^= x >> 7;
                x ^= x << 17;
                text.push_str(alphabet[(x % alphabet.len() as u64) as usize]);
            }
            for tier in TEXT_TIERS.iter().chain(crate::code::CODE_TIERS) {
                assert_eq!(
                    split_at_any(&text, tier),
                    split_at_any_reference(&text, tier),
                    "{text:?}"
                );
            }
        }
    }

    #[test]
    fn absorb_fast_path_covers_every_mark() {
        for &c in TRAILING_ABSORB {
            let len = c.len_utf8();
            let text = format!("{c}x");
            assert_eq!(absorb_trailing(&text, 0), len, "{c:?}");
        }
        for c in (0u32..0x3100)
            .chain(0xFF00..0xFFF0)
            .filter_map(char::from_u32)
        {
            let text = format!("{c}x");
            let want = if TRAILING_ABSORB.contains(&c) {
                c.len_utf8()
            } else {
                0
            };
            assert_eq!(absorb_trailing(&text, 0), want, "{c:?}");
        }
    }

    #[test]
    fn absorb_trailing_stops_at_ordinary_text() {
        assert_eq!(absorb_trailing("」）abc", 0), "」）".len());
        assert_eq!(absorb_trailing("abc", 0), 0);
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
