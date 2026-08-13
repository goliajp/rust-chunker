use crate::Chunk;
use crate::recursive::{Ctx, TEXT_TIERS, merge_pieces};
use tiktoken::CoreBpe;

/// Tags whose closing marks the end of a block of content.
///
/// Splitting after these keeps a paragraph, heading, list item or table row
/// whole. Inline tags (`<a>`, `<em>`, `<code>`, …) are deliberately absent —
/// cutting there would break a sentence mid-phrase.
const BLOCK_TAGS: &[&str] = &[
    "p",
    "div",
    "section",
    "article",
    "aside",
    "header",
    "footer",
    "main",
    "nav",
    "figure",
    "figcaption",
    "blockquote",
    "pre",
    "h1",
    "h2",
    "h3",
    "h4",
    "h5",
    "h6",
    "li",
    "ul",
    "ol",
    "dl",
    "dt",
    "dd",
    "table",
    "thead",
    "tbody",
    "tfoot",
    "tr",
    "td",
    "th",
    "form",
    "fieldset",
    "address",
    "details",
    "summary",
    "hgroup",
    "video",
    "audio",
];

/// Split HTML after block-level closing tags, then apply the ordinary text
/// ladder within whatever is still too large.
///
/// This is **boundary-aware, not DOM-aware**: it scans bytes for `</tag>` and
/// builds no tree. Malformed or unclosed markup simply yields fewer boundaries
/// and degrades to the text ladder rather than failing.
pub(crate) fn split_html(
    text: &str,
    max_tokens: usize,
    overlap_tokens: usize,
    encoder: &CoreBpe,
) -> Vec<Chunk> {
    if text.is_empty() {
        return Vec::new();
    }

    let token_count = encoder.count(text);
    if token_count <= max_tokens {
        return vec![
            Chunk::new(text)
                .with_bytes(0, text.len())
                .with_token_count(token_count),
        ];
    }

    let pieces = split_at_block_tags(text);

    let ctx = Ctx {
        max_tokens,
        overlap_tokens: overlap_tokens.min(max_tokens.saturating_sub(1)),
        encoder,
        section_path: &[],
        tiers: TEXT_TIERS,
    };

    // `0` hands any oversized block to the whole text ladder.
    let mut chunks = merge_pieces(&pieces, text, 0, ctx, 0);
    for (i, chunk) in chunks.iter_mut().enumerate() {
        chunk.index = i;
    }
    chunks
}

/// Cut `text` after every block-level closing tag.
///
/// Tag names are matched case-insensitively — HTML is, and uppercase tags still
/// appear in hand-written and generated markup.
fn split_at_block_tags(text: &str) -> Vec<&str> {
    let bytes = text.as_bytes();
    let mut pieces = Vec::new();
    let mut start = 0;
    let mut i = 0;

    while i < bytes.len() {
        // looking for "</"
        if bytes[i] != b'<' || bytes.get(i + 1) != Some(&b'/') {
            i += 1;
            continue;
        }

        let name_start = i + 2;
        let mut j = name_start;
        while j < bytes.len() && bytes[j].is_ascii_alphanumeric() {
            j += 1;
        }
        // the tag must actually close here
        if j == name_start || bytes.get(j) != Some(&b'>') {
            i += 1;
            continue;
        }

        let name = &text[name_start..j];
        if BLOCK_TAGS.iter().any(|t| t.eq_ignore_ascii_case(name)) {
            let end = j + 1;
            if end > start {
                pieces.push(&text[start..end]);
            }
            start = end;
            i = end;
        } else {
            i = j + 1;
        }
    }

    if start < text.len() {
        pieces.push(&text[start..]);
    }

    pieces
}

#[cfg(test)]
mod tests {
    use super::*;

    fn encoder() -> &'static CoreBpe {
        tiktoken::get_encoding("o200k_base").unwrap()
    }

    #[test]
    fn split_at_block_tags_basic() {
        let html = "<p>one</p><p>two</p><p>three</p>";
        assert_eq!(
            split_at_block_tags(html),
            vec!["<p>one</p>", "<p>two</p>", "<p>three</p>"]
        );
    }

    #[test]
    fn split_at_block_tags_ignores_inline_tags() {
        let html = "<p>a <em>b</em> <a href=\"#\">c</a></p><p>d</p>";
        assert_eq!(
            split_at_block_tags(html),
            vec!["<p>a <em>b</em> <a href=\"#\">c</a></p>", "<p>d</p>"],
            "inline closings must not create boundaries"
        );
    }

    #[test]
    fn split_at_block_tags_is_case_insensitive() {
        let html = "<P>one</P><DIV>two</DIV>";
        assert_eq!(
            split_at_block_tags(html),
            vec!["<P>one</P>", "<DIV>two</DIV>"]
        );
    }

    #[test]
    fn split_at_block_tags_handles_unclosed_markup() {
        let html = "<p>dangling";
        assert_eq!(split_at_block_tags(html), vec!["<p>dangling"]);
    }

    #[test]
    fn split_at_block_tags_ignores_malformed_closers() {
        // "</ p>" and "</>" are not closing tags
        let html = "<p>a</ p>b</>c</p>";
        assert_eq!(split_at_block_tags(html), vec!["<p>a</ p>b</>c</p>"]);
    }

    #[test]
    fn headings_and_list_items_are_boundaries() {
        let html = "<h1>Title</h1><ul><li>one</li><li>two</li></ul>";
        let pieces = split_at_block_tags(html);
        assert!(pieces.contains(&"<h1>Title</h1>"));
        assert!(pieces.iter().any(|p| p.contains("<li>one</li>")));
    }

    #[test]
    fn split_html_respects_token_budget_and_is_lossless() {
        let html = "<h1>Title</h1><p>First paragraph here.</p><p>Second paragraph here.</p><p>Third one.</p>";
        let chunks = split_html(html, 10, 0, encoder());
        assert!(chunks.len() >= 2);
        for c in &chunks {
            assert!(c.token_count <= 10, "chunk {} over budget", c.index);
            assert_eq!(&html[c.start_byte..c.end_byte], c.content);
        }
        let rejoined: String = chunks.iter().map(|c| c.content.as_str()).collect();
        assert_eq!(rejoined, html);
    }

    #[test]
    fn split_html_merges_small_blocks() {
        let html = "<p>a</p><p>b</p><p>c</p>";
        let chunks = split_html(html, 100, 0, encoder());
        assert_eq!(chunks.len(), 1, "small blocks should merge into one chunk");
    }

    #[test]
    fn split_html_empty() {
        assert!(split_html("", 10, 0, encoder()).is_empty());
    }

    #[test]
    fn split_html_indices_are_sequential() {
        let html =
            "<p>First paragraph here.</p><p>Second paragraph here.</p><p>Third paragraph.</p>";
        for (i, c) in split_html(html, 8, 0, encoder()).iter().enumerate() {
            assert_eq!(c.index, i);
        }
    }
}
