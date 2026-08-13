use crate::Chunk;
use crate::byte_offset_of;
use crate::recursive::split_recursive;
use tiktoken::CoreBpe;

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
pub(crate) fn split_markdown(
    text: &str,
    max_tokens: usize,
    overlap_tokens: usize,
    encoder: &CoreBpe,
) -> Vec<Chunk> {
    let sections = extract_sections(text);
    let mut all_chunks = Vec::new();

    for (header, content, byte_offset) in &sections {
        let section = header.clone();
        let sub_chunks = split_recursive(
            content,
            *byte_offset,
            max_tokens,
            overlap_tokens,
            encoder,
            &section,
        );
        all_chunks.extend(sub_chunks);
    }

    // reassign sequential indices
    for (i, chunk) in all_chunks.iter_mut().enumerate() {
        chunk.index = i;
    }

    all_chunks
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

/// extract markdown sections: (header, content, byte_offset)
fn extract_sections(text: &str) -> Vec<(Option<String>, String, usize)> {
    let lines: Vec<&str> = text.split_inclusive('\n').collect();
    let front_matter_end = front_matter_end(&lines);

    let mut sections: Vec<(Option<String>, String, usize)> = Vec::new();
    let mut current_header: Option<String> = None;
    let mut current_content = String::new();
    let mut current_offset = 0usize;
    let mut fence: Option<Fence> = None;
    let mut skip_next = false;

    for (i, line) in lines.iter().enumerate() {
        // the underline of a setext header, already consumed with its title
        if skip_next {
            skip_next = false;
            continue;
        }

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
            push_content(line, text, &mut current_content, &mut current_offset);
            continue;
        }
        if indented_enough && let Some(open) = opens_fence(body) {
            fence = Some(open);
            push_content(line, text, &mut current_content, &mut current_offset);
            continue;
        }

        // --- headers -------------------------------------------------------
        // Front matter is metadata, not prose: its closing `---` follows an
        // ordinary `key: value` line and would otherwise read as a setext rule.
        let in_front_matter = i < front_matter_end;

        let mut header: Option<String> = None;
        let mut header_end = line.len();

        if !in_front_matter && indented_enough {
            if is_header(body) {
                header = Some(body.trim_end().to_string());
            } else if !body.trim().is_empty()
                && lines
                    .get(i + 1)
                    .is_some_and(|next| is_setext_underline(next))
            {
                // `Title` followed by `====` or `----`
                header = Some(body.trim_end().to_string());
                header_end = line.len() + lines[i + 1].len();
                skip_next = true;
            }
        }

        match header {
            Some(h) => {
                // flush previous section
                if !current_content.is_empty() {
                    sections.push((
                        current_header.clone(),
                        std::mem::take(&mut current_content),
                        current_offset,
                    ));
                }
                current_header = Some(h);
                current_offset = byte_offset_of(line, text) + header_end;
            }
            None => push_content(line, text, &mut current_content, &mut current_offset),
        }
    }

    // flush last section
    if !current_content.is_empty() {
        sections.push((current_header, current_content, current_offset));
    }

    sections
}

fn push_content(line: &str, text: &str, content: &mut String, offset: &mut usize) {
    if content.is_empty() {
        *offset = byte_offset_of(line, text);
    }
    content.push_str(line);
}

/// An ATX header: one to six `#` followed by a space.
fn is_header(line: &str) -> bool {
    let hashes = line.bytes().take_while(|&b| b == b'#').count();
    (1..=6).contains(&hashes) && line.as_bytes().get(hashes) == Some(&b' ')
}

/// A setext underline: a run of only `=` or only `-`.
fn is_setext_underline(line: &str) -> bool {
    let body = line.trim_start();
    if line.len() - body.len() > MAX_BLOCK_INDENT {
        return false;
    }
    let body = body.trim_end();
    !body.is_empty() && (body.bytes().all(|b| b == b'=') || body.bytes().all(|b| b == b'-'))
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

/// Index of the first line after YAML (`---`) or TOML (`+++`) front matter.
/// Returns 0 when the document has none.
fn front_matter_end(lines: &[&str]) -> usize {
    let Some(first) = lines.first() else {
        return 0;
    };
    let delim = match first.trim_end() {
        "---" => "---",
        "+++" => "+++",
        _ => return 0,
    };
    lines
        .iter()
        .position(|l| l.trim_end() == delim && !std::ptr::eq(*l, *first))
        .map_or(0, |close| close + 1)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn encoder() -> &'static CoreBpe {
        tiktoken::get_encoding("o200k_base").unwrap()
    }

    #[test]
    fn is_header_valid() {
        assert!(is_header("# Title"));
        assert!(is_header("## Subtitle"));
        assert!(is_header("### H3"));
        assert!(is_header("###### H6"));
    }

    #[test]
    fn is_header_invalid() {
        assert!(!is_header("#NoSpace"));
        assert!(!is_header("####### Too many"));
        assert!(!is_header("Not a header"));
        assert!(!is_header(""));
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
        assert!(is_setext_underline("===\n"));
        assert!(is_setext_underline("---\n"));
        assert!(is_setext_underline("=\n"));
        assert!(!is_setext_underline("\n"));
        assert!(!is_setext_underline("=-=\n"), "must be a uniform run");
        assert!(!is_setext_underline("|---|---|\n"), "table separator");
        assert!(
            !is_setext_underline("    ---\n"),
            "four spaces makes it an indented code block"
        );
    }

    #[test]
    fn front_matter_end_detects_yaml_and_toml() {
        let yaml: Vec<&str> = "---\ntitle: x\n---\nbody\n".split_inclusive('\n').collect();
        assert_eq!(front_matter_end(&yaml), 3);

        let toml: Vec<&str> = "+++\ntitle = 'x'\n+++\nbody\n"
            .split_inclusive('\n')
            .collect();
        assert_eq!(front_matter_end(&toml), 3);

        let none: Vec<&str> = "# Title\nbody\n".split_inclusive('\n').collect();
        assert_eq!(front_matter_end(&none), 0);

        let unterminated: Vec<&str> = "---\ntitle: x\n".split_inclusive('\n').collect();
        assert_eq!(
            front_matter_end(&unterminated),
            0,
            "an unterminated block is not front matter"
        );
    }

    #[test]
    fn extract_sections_skips_fenced_content() {
        let text = "# Title\ntext\n```\n# fake\n```\nmore\n";
        let sections = extract_sections(text);
        assert_eq!(sections.len(), 1);
        assert_eq!(sections[0].0.as_deref(), Some("# Title"));
        assert!(sections[0].1.contains("# fake"), "content is preserved");
    }

    #[test]
    fn extract_sections_setext() {
        let text = "Title\n=====\nbody\n";
        let sections = extract_sections(text);
        assert_eq!(sections.len(), 1);
        assert_eq!(sections[0].0.as_deref(), Some("Title"));
        assert_eq!(sections[0].1, "body\n");
    }

    #[test]
    fn extract_sections_basic() {
        let text = "# Title\nSome intro.\n## Section A\nContent A.\n## Section B\nContent B.\n";
        let sections = extract_sections(text);
        assert_eq!(sections.len(), 3);
        assert_eq!(sections[0].0.as_deref(), Some("# Title"));
        assert!(sections[0].1.contains("Some intro."));
        assert_eq!(sections[1].0.as_deref(), Some("## Section A"));
        assert_eq!(sections[2].0.as_deref(), Some("## Section B"));
    }

    #[test]
    fn extract_sections_no_headers() {
        let text = "Just plain text\nwith lines\nand more.";
        let sections = extract_sections(text);
        assert_eq!(sections.len(), 1);
        assert_eq!(sections[0].0, None);
        assert_eq!(sections[0].1, text);
    }

    #[test]
    fn split_markdown_basic() {
        let enc = encoder();
        let text = "# Introduction\n\nSome introductory text here.\n\n## Details\n\nDetailed content goes here with more words.\n";
        let chunks = split_markdown(text, 100, 0, enc);
        assert!(chunks.len() >= 2);
        assert_eq!(chunks[0].section.as_deref(), Some("# Introduction"));
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
            assert_eq!(chunk.section.as_deref(), Some("# Big Section"));
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
        assert_eq!(chunks[0].section, None);
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
            assert_eq!(chunk.section.as_deref(), Some("# Section"));
        }
    }

    #[test]
    fn preamble_before_first_header() {
        let enc = encoder();
        let text = "Preamble text.\n\n# First Header\n\nContent.";
        let chunks = split_markdown(text, 100, 0, enc);
        assert!(chunks.len() >= 2);
        assert_eq!(chunks[0].section, None); // preamble has no header
        assert_eq!(chunks[1].section.as_deref(), Some("# First Header"));
    }
}
