//! Boundary-quality tests for CJK text.
//!
//! The crate's existing CJK tests only ever asserted `token_count <= max`, which
//! every possible splitter satisfies — including one that cuts mid-word. These
//! tests assert where the cuts actually land, which is the thing users care
//! about and the thing that was broken: the separator ladder required a
//! trailing ASCII space on every sentence and clause mark, so Chinese and
//! Japanese prose fell straight through to token-level hard cuts.

/// Sentence-final marks used across Chinese and Japanese.
const CJK_SENTENCE_ENDS: &[char] = &['。', '！', '？', '．', '…'];
/// Clause marks — legal chunk starts only when nothing better exists.
const CJK_CLAUSE_MARKS: &[char] = &['，', '、', '；', '：', ',', ';'];

fn assert_no_chunk_starts_with_punctuation(chunks: &[chunkedrs::Chunk], label: &str) {
    for c in chunks {
        let first = c.content.trim_start().chars().next();
        if let Some(ch) = first {
            assert!(
                !CJK_CLAUSE_MARKS.contains(&ch) && !CJK_SENTENCE_ENDS.contains(&ch),
                "{label}: chunk {} starts with punctuation {ch:?} — the cut landed \
                 mid-sentence: {:?}",
                c.index,
                c.content
            );
        }
    }
}

#[test]
fn chinese_prose_splits_on_sentence_marks() {
    let zh = "这是第一句话，讲的是分块。这是第二句话，讲的是检索。这是第三句话，讲的是嵌入。这是第四句话，讲的是排序。";
    let chunks = chunkedrs::chunk(zh).max_tokens(12).split();

    assert!(chunks.len() >= 2, "expected multiple chunks");
    assert_no_chunk_starts_with_punctuation(&chunks, "zh");

    // Every chunk but possibly the last should close on a sentence mark.
    for c in &chunks[..chunks.len() - 1] {
        let last = c.content.trim_end().chars().last().unwrap();
        assert!(
            CJK_SENTENCE_ENDS.contains(&last) || CJK_CLAUSE_MARKS.contains(&last),
            "zh: chunk {} ends on {last:?}, not a sentence or clause mark: {:?}",
            c.index,
            c.content
        );
    }
}

#[test]
fn japanese_prose_splits_on_sentence_marks() {
    let ja =
        "これは最初の文です。これは二番目の文です。これは三番目の文です。これは四番目の文です。";
    let chunks = chunkedrs::chunk(ja).max_tokens(12).split();

    assert!(chunks.len() >= 2, "expected multiple chunks");
    assert_no_chunk_starts_with_punctuation(&chunks, "ja");

    for c in &chunks[..chunks.len() - 1] {
        let last = c.content.trim_end().chars().last().unwrap();
        assert!(
            CJK_SENTENCE_ENDS.contains(&last) || CJK_CLAUSE_MARKS.contains(&last),
            "ja: chunk {} ends on {last:?}: {:?}",
            c.index,
            c.content
        );
    }
}

/// When a single sentence will not fit, the clause mark is the next boundary
/// down — not an arbitrary token offset.
#[test]
fn chinese_falls_back_to_clause_marks() {
    let zh = "分块策略的选择取决于文档结构，检索目标，以及下游模型的上下文窗口大小，这几个因素互相牵制。";
    let chunks = chunkedrs::chunk(zh).max_tokens(10).split();

    assert!(chunks.len() >= 2);
    assert_no_chunk_starts_with_punctuation(&chunks, "zh-clause");
}

/// A document containing both scripts must not have one script's separators
/// starve the other's.
#[test]
fn mixed_script_document_splits_both_scripts() {
    let mixed = "This is an English sentence. And a second one here.\n\n这是一句中文。这是第二句中文。这是第三句。";
    let chunks = chunkedrs::chunk(mixed).max_tokens(12).split();

    assert!(chunks.len() >= 3);
    assert_no_chunk_starts_with_punctuation(&chunks, "mixed");
}

/// A closing bracket or quote belongs to the sentence it closes, not to the
/// next one.
#[test]
fn cjk_closing_punctuation_stays_with_its_sentence() {
    let zh = "他说「今天天气很好。」然后就出门了。她回答说「确实不错。」于是也跟着出去了。";
    let chunks = chunkedrs::chunk(zh).max_tokens(12).split();

    for c in &chunks {
        let first = c.content.trim_start().chars().next().unwrap();
        assert!(
            !['」', '』', '）', '》', '”'].contains(&first),
            "chunk {} starts with a closing mark {first:?} — it was orphaned from \
             its sentence: {:?}",
            c.index,
            c.content
        );
    }
}

/// Token-level splitting is the last resort, and it must still cut on
/// character boundaries — never mid-codepoint.
#[test]
fn token_level_split_never_produces_replacement_chars() {
    // No separators at all, so this can only be split by tokens.
    let dense = "日本語日本語日本語日本語日本語日本語日本語日本語日本語日本語".repeat(3);
    for max in [1usize, 2, 3, 5, 8] {
        let chunks = chunkedrs::chunk(&dense).max_tokens(max).split();
        for c in &chunks {
            assert!(
                !c.content.contains('\u{FFFD}'),
                "max_tokens={max}: chunk {} contains a replacement character: {:?}",
                c.index,
                c.content
            );
        }
        let rejoined: String = chunks.iter().map(|c| c.content.as_str()).collect();
        assert_eq!(
            rejoined, dense,
            "max_tokens={max}: token-level split must be lossless"
        );
    }
}

/// Byte offsets must still address the original text exactly after the
/// separator work.
#[test]
fn cjk_byte_offsets_address_original_text() {
    let zh = "第一句在这里。第二句在这里。第三句在这里。第四句在这里。第五句在这里。";
    let chunks = chunkedrs::chunk(zh).max_tokens(8).split();
    for c in &chunks {
        assert_eq!(
            &zh[c.start_byte..c.end_byte],
            c.content,
            "chunk {} byte range does not match its content",
            c.index
        );
    }
}
