use crate::Chunk;
use crate::byte_offset_of;
use crate::recursive::split_recursive;
use tiktoken::CoreBpe;

/// split markdown text by headers first, then apply recursive splitting within each section.
///
/// headers are detected by lines starting with `#` (up to 6 levels).
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

/// extract markdown sections: (header, content, byte_offset)
fn extract_sections(text: &str) -> Vec<(Option<String>, String, usize)> {
    let mut sections: Vec<(Option<String>, String, usize)> = Vec::new();
    let mut current_header: Option<String> = None;
    let mut current_content = String::new();
    let mut current_offset = 0usize;

    for line in text.split_inclusive('\n') {
        let trimmed = line.trim_start();
        if is_header(trimmed) {
            // flush previous section
            if !current_content.is_empty() {
                sections.push((current_header.clone(), current_content, current_offset));
            }
            current_header = Some(trimmed.trim_end().to_string());
            current_content = String::new();
            let line_offset = byte_offset_of(line, text);
            current_offset = line_offset + line.len();
        } else {
            if current_content.is_empty() {
                current_offset = byte_offset_of(line, text);
            }
            current_content.push_str(line);
        }
    }

    // flush last section
    if !current_content.is_empty() {
        sections.push((current_header, current_content, current_offset));
    }

    sections
}

fn is_header(line: &str) -> bool {
    let hashes = line.bytes().take_while(|&b| b == b'#').count();
    (1..=6).contains(&hashes) && line.as_bytes().get(hashes) == Some(&b' ')
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
