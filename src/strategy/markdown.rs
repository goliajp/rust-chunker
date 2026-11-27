use crate::Chunk;
use crate::plan::{Doc, plan, tiers};
use crate::strategy::recursive::TEXT_TIERS;

/// split markdown text by headers first, then apply recursive splitting within each section.
///
/// headers are ATX (`#` through `######`) or setext (`===` / `---` underline).
/// fenced code blocks are skipped entirely, and YAML/TOML front matter is
/// excluded from header detection — a `#` inside a shell block is a comment,
/// and front matter's closing `---` is not a setext rule.
/// each resulting chunk carries the section header in its `section` metadata field.
///
/// note: header lines are stored in metadata only, not in chunk `content`.
/// joining all chunk contents will not reproduce header lines from the original document.
pub(crate) fn split_markdown<'a>(
    doc: &Doc<'a>,
    max_tokens: usize,
    overlap: usize,
) -> Vec<Chunk<'a>> {
    let ladder = tiers(TEXT_TIERS);
    let mut chunks = Vec::new();
    for section in extract_sections(doc.text, &doc.newlines) {
        let start = section.byte_offset;
        let end = start + section.content.len();
        for piece in plan(doc, start, end, &ladder, max_tokens, overlap) {
            chunks.push(doc.chunk(piece, &section.path));
        }
    }
    chunks
}

/// An open fenced code block: the marker character and how many of it opened
/// the fence. A fence can only be closed by a run of the same character that is
/// at least as long, which is how documents nest markdown inside markdown.
#[derive(Clone, Copy)]
struct Fence {
    marker: u8,
    len: usize,
}

/// CommonMark allows up to three spaces of indentation before a block marker;
/// four or more makes it an indented code block instead.
const MAX_BLOCK_INDENT: usize = 3;

/// A run of body text together with the header ancestry it sits under.
struct Section<'a> {
    /// Headers from outermost to innermost, e.g.
    /// `["# Guide", "## Installation"]`. Empty before the first header.
    path: Vec<&'a str>,
    /// The body between this section's header and the next, a slice of the
    /// document.
    content: &'a str,
    byte_offset: usize,
}

/// The lines of `text`, each with its `\n`, as `split_inclusive('\n')` gives
/// them; `newlines` are the offsets of every `\n` in `text`.
struct Lines<'a, 'n> {
    text: &'a str,
    newlines: std::slice::Iter<'n, usize>,
    at: usize,
}

impl<'a> Iterator for Lines<'a, '_> {
    type Item = &'a str;

    #[inline]
    fn next(&mut self) -> Option<&'a str> {
        if self.at >= self.text.len() {
            return None;
        }
        let end = self.newlines.next().map_or(self.text.len(), |&p| p + 1);
        let line = &self.text[self.at..end];
        self.at = end;
        Some(line)
    }
}

/// extract markdown sections with their header ancestry
fn extract_sections<'a>(text: &'a str, newlines: &[usize]) -> Vec<Section<'a>> {
    let front_matter_end = front_matter_end(text);

    let mut sections: Vec<Section> = Vec::new();
    // (level, header) from outermost inwards — a header of level L pops every
    // entry at level >= L, so `## Usage` after `### From source` correctly
    // drops back to sitting under `# Guide`.
    let mut stack: Vec<(usize, &str)> = Vec::new();
    // byte range of the body collected since the last header
    let mut current: Option<(usize, usize)> = None;
    let mut fence: Option<Fence> = None;
    let mut lines = Lines {
        text,
        newlines: newlines.iter(),
        at: 0,
    }
    .peekable();

    while let Some(line) = lines.next() {
        let body = line.trim_start();
        let indent = line.len() - body.len();
        let indented_enough = indent <= MAX_BLOCK_INDENT;

        // --- fenced code blocks -------------------------------------------
        // Inside a fence nothing is markup. This is the whole point: `#` opens
        // a comment in shell, python, ruby, yaml and toml, so without this the
        // majority of technical documents grow phantom sections.
        if let Some(open) = fence {
            if indented_enough && closes_fence(body, open) {
                fence = None;
            }
            push_content(line, text, &mut current);
            continue;
        }
        if indented_enough && let Some(open) = opens_fence(body) {
            fence = Some(open);
            push_content(line, text, &mut current);
            continue;
        }

        // --- headers -------------------------------------------------------
        // Front matter is metadata, not prose: its closing `---` follows an
        // ordinary `key: value` line and would otherwise read as a setext rule.
        let in_front_matter = offset(line, text) < front_matter_end;

        let mut header: Option<(usize, &str)> = None;

        if !in_front_matter && indented_enough {
            if let Some(level) = atx_level(body) {
                header = Some((level, body.trim_end()));
            } else if !body.trim().is_empty()
                && let Some(level) = lines.peek().and_then(|next| setext_level(next))
            {
                // `Title` followed by `====` (level 1) or `----` (level 2);
                // the underline is part of the header
                header = Some((level, body.trim_end()));
                lines.next();
            }
        }

        match header {
            Some((level, text_of_header)) => {
                // flush previous section
                if let Some((start, end)) = current.take() {
                    sections.push(Section {
                        path: stack.iter().map(|&(_, h)| h).collect(),
                        content: &text[start..end],
                        byte_offset: start,
                    });
                }
                stack.retain(|(l, _)| *l < level);
                stack.push((level, text_of_header));
            }
            None => push_content(line, text, &mut current),
        }
    }

    // flush last section
    if let Some((start, end)) = current {
        sections.push(Section {
            path: stack.iter().map(|&(_, h)| h).collect(),
            content: &text[start..end],
            byte_offset: start,
        });
    }

    sections
}

/// Byte offset of `line`, a slice of `text`, in `text`.
fn offset(line: &str, text: &str) -> usize {
    line.as_ptr() as usize - text.as_ptr() as usize
}

fn push_content(line: &str, text: &str, current: &mut Option<(usize, usize)>) {
    let at = offset(line, text);
    let start = current.map_or(at, |(start, _)| start);
    *current = Some((start, at + line.len()));
}

/// Nesting level of an ATX header: one to six `#` followed by a space.
fn atx_level(line: &str) -> Option<usize> {
    let hashes = line.bytes().take_while(|&b| b == b'#').count();
    ((1..=6).contains(&hashes) && line.as_bytes().get(hashes) == Some(&b' ')).then_some(hashes)
}

/// Nesting level of a setext underline: `=` is level 1, `-` is level 2.
fn setext_level(line: &str) -> Option<usize> {
    let body = line.trim_start();
    if line.len() - body.len() > MAX_BLOCK_INDENT {
        return None;
    }
    let body = body.trim_end();
    if body.is_empty() {
        return None;
    }
    if body.bytes().all(|b| b == b'=') {
        Some(1)
    } else if body.bytes().all(|b| b == b'-') {
        Some(2)
    } else {
        None
    }
}

/// Opening fence: a run of three or more backticks or tildes.
fn opens_fence(body: &str) -> Option<Fence> {
    let marker = match body.as_bytes().first() {
        Some(&b @ (b'`' | b'~')) => b,
        _ => return None,
    };
    let len = body.bytes().take_while(|&b| b == marker).count();
    if len < 3 {
        return None;
    }
    // A backtick fence's info string may not contain a backtick, otherwise
    // inline code spans would open blocks.
    if marker == b'`' && body[len..].contains('`') {
        return None;
    }
    Some(Fence { marker, len })
}

/// Closing fence: a run of the same marker, at least as long as the opener,
/// with nothing but whitespace after it.
fn closes_fence(body: &str, open: Fence) -> bool {
    let len = body.bytes().take_while(|&b| b == open.marker).count();
    len >= open.len && body[len..].trim().is_empty()
}

/// Byte offset of the first line after YAML (`---`) or TOML (`+++`) front
/// matter. Returns 0 when the document has none.
fn front_matter_end(text: &str) -> usize {
    let mut lines = text.split_inclusive('\n');
    let Some(first) = lines.next() else {
        return 0;
    };
    let delim = match first.trim_end() {
        "---" => "---",
        "+++" => "+++",
        _ => return 0,
    };
    let mut at = first.len();
    for line in lines {
        at += line.len();
        if line.trim_end() == delim {
            return at;
        }
    }
    0
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::plan::newlines;

    fn encoder() -> &'static tiktoken::CoreBpe {
        tiktoken::get_encoding("o200k_base").unwrap()
    }

    fn split_markdown<'a>(
        text: &'a str,
        max: usize,
        overlap: usize,
        enc: &'static tiktoken::CoreBpe,
    ) -> Vec<Chunk<'a>> {
        let mut chunks = super::split_markdown(&Doc::new(text, enc), max, overlap);
        crate::chunker::finish(&mut chunks);
        chunks
    }

    #[test]
    fn is_header_valid() {
        assert!(atx_level("# Title").is_some());
        assert!(atx_level("## Subtitle").is_some());
        assert!(atx_level("### H3").is_some());
        assert!(atx_level("###### H6").is_some());
    }

    #[test]
    fn is_header_invalid() {
        assert!(atx_level("#NoSpace").is_none());
        assert!(atx_level("####### Too many").is_none());
        assert!(atx_level("Not a header").is_none());
        assert!(atx_level("").is_none());
    }

    #[test]
    fn opens_fence_backticks_and_tildes() {
        assert!(opens_fence("```").is_some());
        assert!(opens_fence("```rust").is_some());
        assert!(opens_fence("~~~").is_some());
        assert!(opens_fence("````markdown").is_some());
        assert!(opens_fence("``").is_none(), "two backticks is not a fence");
        assert!(opens_fence("text").is_none());
        assert!(
            opens_fence("```a`b").is_none(),
            "a backtick in the info string means this is not a fence opener"
        );
    }

    #[test]
    fn closes_fence_requires_same_marker_and_length() {
        let open = opens_fence("````").unwrap();
        assert!(!closes_fence("```", open), "shorter run must not close");
        assert!(closes_fence("````", open));
        assert!(closes_fence("`````", open), "longer run closes");
        assert!(
            !closes_fence("~~~~", open),
            "different marker must not close"
        );
        assert!(
            !closes_fence("````rust", open),
            "a closing fence carries no info string"
        );
    }

    #[test]
    fn is_setext_underline_variants() {
        assert!(setext_level("===\n").is_some());
        assert!(setext_level("---\n").is_some());
        assert!(setext_level("=\n").is_some());
        assert!(setext_level("\n").is_none());
        assert!(setext_level("=-=\n").is_none(), "must be a uniform run");
        assert!(setext_level("|---|---|\n").is_none(), "table separator");
        assert!(
            setext_level("    ---\n").is_none(),
            "four spaces makes it an indented code block"
        );
        assert_eq!(setext_level("===\n"), Some(1), "`=` is an h1 underline");
        assert_eq!(setext_level("---\n"), Some(2), "`-` is an h2 underline");
    }

    #[test]
    fn atx_level_counts_hashes() {
        assert_eq!(atx_level("# a"), Some(1));
        assert_eq!(atx_level("### a"), Some(3));
        assert_eq!(atx_level("###### a"), Some(6));
        assert_eq!(atx_level("####### a"), None, "seven is too many");
        assert_eq!(atx_level("#a"), None, "a space is required");
    }

    #[test]
    fn front_matter_end_detects_yaml_and_toml() {
        let yaml = "---\ntitle: x\n---\nbody\n";
        assert_eq!(front_matter_end(yaml), yaml.find("body").unwrap());

        let toml = "+++\ntitle = 'x'\n+++\nbody\n";
        assert_eq!(front_matter_end(toml), toml.find("body").unwrap());

        assert_eq!(front_matter_end("# Title\nbody\n"), 0);
        assert_eq!(
            front_matter_end("---\ntitle: x\n"),
            0,
            "an unterminated block is not front matter"
        );
    }

    #[test]
    fn extract_sections_skips_fenced_content() {
        let text = "# Title\ntext\n```\n# fake\n```\nmore\n";
        let sections = extract_sections(text, &newlines(text));
        assert_eq!(sections.len(), 1);
        assert_eq!(sections[0].path.last().copied(), Some("# Title"));
        assert!(
            sections[0].content.contains("# fake"),
            "content is preserved"
        );
    }

    #[test]
    fn extract_sections_setext() {
        let text = "Title\n=====\nbody\n";
        let sections = extract_sections(text, &newlines(text));
        assert_eq!(sections.len(), 1);
        assert_eq!(sections[0].path.last().copied(), Some("Title"));
        assert_eq!(sections[0].content, "body\n");
    }

    #[test]
    fn extract_sections_basic() {
        let text = "# Title\nSome intro.\n## Section A\nContent A.\n## Section B\nContent B.\n";
        let sections = extract_sections(text, &newlines(text));
        assert_eq!(sections.len(), 3);
        assert_eq!(sections[0].path.last().copied(), Some("# Title"));
        assert!(sections[0].content.contains("Some intro."));
        assert_eq!(sections[1].path.last().copied(), Some("## Section A"));
        assert_eq!(sections[2].path.last().copied(), Some("## Section B"));
    }

    #[test]
    fn extract_sections_no_headers() {
        let text = "Just plain text\nwith lines\nand more.";
        let sections = extract_sections(text, &newlines(text));
        assert_eq!(sections.len(), 1);
        assert_eq!(sections[0].path.last(), None);
        assert_eq!(sections[0].content, text);
    }

    #[test]
    fn split_markdown_basic() {
        let enc = encoder();
        let text = "# Introduction\n\nSome introductory text here.\n\n## Details\n\nDetailed content goes here with more words.\n";
        let chunks = split_markdown(text, 100, 0, enc);
        assert!(chunks.len() >= 2);
        assert_eq!(chunks[0].section(), Some("# Introduction"));
    }

    #[test]
    fn split_markdown_large_section_splits_further() {
        let enc = encoder();
        let long_content = "Word. ".repeat(200);
        let text = format!("# Big Section\n\n{long_content}");
        let chunks = split_markdown(&text, 20, 0, enc);
        assert!(chunks.len() >= 2);
        for chunk in &chunks {
            assert!(chunk.token_count <= 20);
            assert_eq!(chunk.section(), Some("# Big Section"));
        }
    }

    #[test]
    fn split_markdown_sequential_indices() {
        let enc = encoder();
        let text = "# A\n\nContent A.\n\n# B\n\nContent B.\n\n# C\n\nContent C.\n";
        let chunks = split_markdown(text, 100, 0, enc);
        for (i, chunk) in chunks.iter().enumerate() {
            assert_eq!(chunk.index, i);
        }
    }

    #[test]
    fn split_markdown_no_headers() {
        let enc = encoder();
        let text = "Just plain text without any headers.";
        let chunks = split_markdown(text, 100, 0, enc);
        assert_eq!(chunks.len(), 1);
        assert_eq!(chunks[0].section(), None);
    }

    #[test]
    fn split_markdown_empty() {
        let enc = encoder();
        let chunks = split_markdown("", 100, 0, enc);
        assert!(chunks.is_empty());
    }

    #[test]
    fn split_markdown_header_only() {
        let enc = encoder();
        let text = "# Title\n";
        let chunks = split_markdown(text, 100, 0, enc);
        // header with no content produces no chunks
        assert!(chunks.is_empty());
    }

    #[test]
    fn split_markdown_preserves_content() {
        let enc = encoder();
        let text = "# Title\n\nHello world.\n";
        let chunks = split_markdown(text, 100, 0, enc);
        assert_eq!(chunks.len(), 1);
        assert!(chunks[0].content.contains("Hello world."));
    }

    #[test]
    fn split_markdown_with_overlap() {
        let enc = encoder();
        let long_content = "Word. ".repeat(200);
        let text = format!("# Section\n\n{long_content}");
        let chunks = split_markdown(&text, 20, 5, enc);
        assert!(chunks.len() >= 2);
        for chunk in &chunks {
            assert!(chunk.token_count <= 20);
            assert_eq!(chunk.section(), Some("# Section"));
        }
    }

    #[test]
    fn preamble_before_first_header() {
        let enc = encoder();
        let text = "Preamble text.\n\n# First Header\n\nContent.";
        let chunks = split_markdown(text, 100, 0, enc);
        assert!(chunks.len() >= 2);
        assert_eq!(chunks[0].section(), None); // preamble has no header
        assert_eq!(chunks[1].section(), Some("# First Header"));
    }
}
