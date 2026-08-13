//! Tests for the v2 `Chunk` metadata surface: token spans and section paths.

/// Token spans locate a chunk in the *document's* token stream, which is what
/// late chunking needs: embed the whole document once, then pool each chunk's
/// vector over its own token range.
#[test]
fn token_spans_are_monotonic_and_cover_the_document() {
    let text = "Alpha beta gamma. Delta epsilon zeta. Eta theta iota. Kappa lambda mu.";
    let chunks = chunkedrs::chunk(text).max_tokens(6).split();
    assert!(chunks.len() >= 2);

    let total = tiktoken::get_encoding("o200k_base").unwrap().count(text);

    assert_eq!(chunks[0].start_token, 0, "first chunk starts at token 0");
    assert_eq!(
        chunks.last().unwrap().end_token,
        total,
        "last chunk ends at the document's final token"
    );

    for c in &chunks {
        assert!(
            c.start_token < c.end_token,
            "chunk {} has an empty token span {}..{}",
            c.index,
            c.start_token,
            c.end_token
        );
        assert!(c.end_token <= total, "chunk {} runs past the end", c.index);
    }

    // Spans advance and never leave a hole. They may overlap by the single
    // token that straddles a chunk boundary — see
    // `spans_overlap_by_the_token_that_straddles_a_boundary`.
    for w in chunks.windows(2) {
        assert!(
            w[1].start_token <= w[0].end_token,
            "gap between chunk {} (ends {}) and {} (starts {})",
            w[0].index,
            w[0].end_token,
            w[1].index,
            w[1].start_token
        );
        assert!(
            w[0].start_token <= w[1].start_token && w[0].end_token <= w[1].end_token,
            "spans must advance monotonically"
        );
    }
}

/// A chunk boundary does not have to land on a token boundary: a separator can
/// sit inside a token, as `". "` does when the tokenizer merges the space into
/// the following word. The span then widens to cover the straddling token, so
/// consecutive spans overlap by exactly one.
///
/// Covering is the correct direction — a late-chunking pool over a range that
/// includes one extra token is fine, one that clips the chunk's own text is not.
#[test]
fn spans_overlap_by_the_token_that_straddles_a_boundary() {
    let text = "Alpha beta gamma. Delta epsilon zeta. Eta theta iota. Kappa lambda mu.";
    let chunks = chunkedrs::chunk(text).max_tokens(6).split();

    let straddling = chunks
        .windows(2)
        .filter(|w| w[1].start_token < w[0].end_token)
        .count();
    assert!(
        straddling > 0,
        "this fixture is meant to exercise the straddling case"
    );

    for w in chunks.windows(2) {
        assert!(
            w[0].end_token - w[1].start_token <= 1,
            "at most one token may be shared across a boundary, chunk {} shares {}",
            w[0].index,
            w[0].end_token - w[1].start_token
        );
    }
}

/// The token span must cover the chunk's text, so a late-chunking pool over
/// that range sees at least the chunk's own tokens.
#[test]
fn token_span_covers_the_chunk_content() {
    let text = "First sentence here. Second sentence here. Third sentence here.";
    let enc = tiktoken::get_encoding("o200k_base").unwrap();
    let doc_tokens = enc.encode(text);

    for c in chunkedrs::chunk(text).max_tokens(5).split() {
        let span = enc.decode(&doc_tokens[c.start_token..c.end_token]);
        let span = String::from_utf8_lossy(&span);
        assert!(
            span.contains(c.content.trim()),
            "chunk {} content {:?} is not inside its token span {:?}",
            c.index,
            c.content,
            span
        );
    }
}

/// `token_count` and the span width answer different questions and are allowed
/// to disagree: BPE is not additive, so re-tokenizing a chunk in isolation can
/// differ from its slice of the document stream. Both must be present.
#[test]
fn token_count_is_measured_on_the_chunk_itself() {
    let enc = tiktoken::get_encoding("o200k_base").unwrap();
    let text = "Paragraph one here.\n\nParagraph two here.\n\nParagraph three here.";
    for c in chunkedrs::chunk(text).max_tokens(8).split() {
        assert_eq!(
            c.token_count,
            enc.count(&c.content),
            "chunk {} token_count must equal a fresh count of its own content",
            c.index
        );
    }
}

#[test]
fn token_spans_are_present_for_cjk() {
    let zh = "第一句在这里。第二句在这里。第三句在这里。第四句在这里。";
    let total = tiktoken::get_encoding("o200k_base").unwrap().count(zh);
    let chunks = chunkedrs::chunk(zh).max_tokens(8).split();
    assert_eq!(chunks[0].start_token, 0);
    assert_eq!(chunks.last().unwrap().end_token, total);
}

// --- section paths -------------------------------------------------------

/// A nested header knows its ancestry, not just its own line.
#[test]
fn section_path_carries_header_ancestry() {
    let md = "\
# Guide

Intro text.

## Installation

Install text.

### From source

Source text.

## Usage

Usage text.
";
    let chunks = chunkedrs::chunk(md).markdown().split();

    let find = |needle: &str| {
        chunks
            .iter()
            .find(|c| c.content.contains(needle))
            .unwrap_or_else(|| panic!("no chunk containing {needle:?}"))
    };

    assert_eq!(find("Intro text").section_path, vec!["# Guide"]);
    assert_eq!(
        find("Install text").section_path,
        vec!["# Guide", "## Installation"]
    );
    assert_eq!(
        find("Source text").section_path,
        vec!["# Guide", "## Installation", "### From source"],
        "a level-3 header must know both ancestors"
    );
    assert_eq!(
        find("Usage text").section_path,
        vec!["# Guide", "## Usage"],
        "dropping back to level 2 must pop the level-3 header"
    );
}

/// `section()` remains the ergonomic accessor for the deepest header.
#[test]
fn section_returns_the_deepest_header() {
    let md = "# Top\n\nA.\n\n## Nested\n\nB.\n";
    let chunks = chunkedrs::chunk(md).markdown().split();

    let nested = chunks.iter().find(|c| c.content.contains('B')).unwrap();
    assert_eq!(nested.section(), Some("## Nested"));
    assert_eq!(
        nested.section_path.last().map(String::as_str),
        nested.section()
    );
}

/// Content before any header has an empty path, not a phantom one.
#[test]
fn preamble_has_no_section_path() {
    let md = "Preamble here.\n\n# First\n\nBody.\n";
    let chunks = chunkedrs::chunk(md).markdown().split();
    let preamble = chunks
        .iter()
        .find(|c| c.content.contains("Preamble"))
        .unwrap();
    assert!(preamble.section_path.is_empty());
    assert_eq!(preamble.section(), None);
}

/// Setext headers participate in the hierarchy — `===` is level 1, `---` is 2.
#[test]
fn setext_headers_have_levels() {
    let md = "\
Title
=====

Intro.

Subsection
----------

Body.
";
    let chunks = chunkedrs::chunk(md).markdown().split();
    let body = chunks.iter().find(|c| c.content.contains("Body")).unwrap();
    assert_eq!(body.section_path, vec!["Title", "Subsection"]);
}

/// A document that skips a level (h1 -> h3) must not panic or invent an h2.
#[test]
fn skipped_header_levels_do_not_panic() {
    let md = "# One\n\nA.\n\n### Three\n\nB.\n\n## Two\n\nC.\n";
    let chunks = chunkedrs::chunk(md).markdown().split();
    let b = chunks.iter().find(|c| c.content.contains('B')).unwrap();
    assert_eq!(b.section_path, vec!["# One", "### Three"]);
    let c = chunks.iter().find(|c| c.content.contains('C')).unwrap();
    assert_eq!(c.section_path, vec!["# One", "## Two"]);
}

/// Non-markdown strategies leave the path empty rather than guessing.
#[test]
fn recursive_strategy_has_no_section_path() {
    for c in chunkedrs::chunk("plain text here").split() {
        assert!(c.section_path.is_empty());
        assert_eq!(c.section(), None);
    }
}

/// `Chunk` is `#[non_exhaustive]`, so downstream code needs a way to build one
/// — for tests, and for adapters that re-materialise chunks from storage.
#[test]
fn chunk_can_be_constructed_downstream() {
    let c = chunkedrs::Chunk::new("hello")
        .with_index(3)
        .with_bytes(10, 15)
        .with_tokens(2, 4)
        .with_section_path(vec!["# Title".to_string()]);

    assert_eq!(c.content, "hello");
    assert_eq!(c.index, 3);
    assert_eq!((c.start_byte, c.end_byte), (10, 15));
    assert_eq!((c.start_token, c.end_token), (2, 4));
    assert_eq!(c.section(), Some("# Title"));
}
