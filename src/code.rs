use crate::Chunk;
use crate::recursive::split_with_tiers;
use tiktoken::CoreBpe;

/// Separator tiers for source code, outermost first.
///
/// This is deliberately **boundary-aware, not AST-aware**. It parses nothing
/// and brings no grammar dependencies, so it applies to any language; the
/// tradeoff is that it reads punctuation, not structure. A `}` inside a string
/// literal looks like a block close to it. For AST fidelity, use a tree-sitter
/// based splitter.
///
/// What it does exploit is that essentially every language in common use
/// separates top-level definitions with a blank line and closes blocks with a
/// dedented bracket on its own line — which is enough to keep whole functions
/// together most of the time.
///
/// The prose tiers are deliberately absent below the line tier: splitting code
/// on `". "` cuts inside string literals and doc comments, which is worse than
/// splitting on a word boundary.
const CODE_TIERS: &[&[&str]] = &[
    // blank lines — top-level definitions in nearly every language
    &["\r\n\r\n", "\n\n"],
    // a bracket closing a block at column 0, alone on its line
    &[
        "\n}\n", "\n};\n", "\n};\r\n", "\n}\r\n", // C family, Rust, Go, Java, JS
        "\n)\n", "\n];\n", "\n]\n",   // call and array literals
        "\nend\n", // Ruby, Lua, Elixir
    ],
    // line
    &["\r\n", "\n"],
    // word — the prose tiers are skipped on purpose
    &[" ", "\t"],
];

/// Split source code at boundary markers, respecting token limits.
pub(crate) fn split_code(
    text: &str,
    max_tokens: usize,
    overlap_tokens: usize,
    encoder: &CoreBpe,
) -> Vec<Chunk> {
    let mut chunks = split_with_tiers(
        text,
        0,
        max_tokens,
        overlap_tokens,
        encoder,
        &[],
        CODE_TIERS,
    );
    for (i, chunk) in chunks.iter_mut().enumerate() {
        chunk.index = i;
    }
    chunks
}

#[cfg(test)]
mod tests {
    use super::*;

    fn encoder() -> &'static CoreBpe {
        tiktoken::get_encoding("o200k_base").unwrap()
    }

    #[test]
    fn splits_rust_functions_at_blank_lines() {
        let src = "\
fn alpha() {
    println!(\"one\");
}

fn bravo() {
    println!(\"two\");
}

fn charlie() {
    println!(\"three\");
}
";
        let chunks = split_code(src, 14, 0, encoder());
        assert!(chunks.len() >= 2);
        for c in &chunks {
            assert!(c.token_count <= 14);
        }
        // each chunk should start at a definition, not mid-body
        for c in &chunks {
            let first = c.content.trim_start();
            assert!(
                !first.starts_with("println!"),
                "chunk {} opens mid-body: {:?}",
                c.index,
                c.content
            );
        }
    }

    #[test]
    fn splits_on_block_close_when_no_blank_lines() {
        let src = "\
fn alpha() {
    one();
    two();
}
fn bravo() {
    three();
    four();
}
";
        let chunks = split_code(src, 12, 0, encoder());
        assert!(chunks.len() >= 2);
        assert!(
            chunks[0].content.contains('}'),
            "the closing brace belongs to the block it closes: {:?}",
            chunks[0].content
        );
    }

    #[test]
    fn python_indentation_falls_back_to_lines() {
        let src = "\
def alpha():
    return 1

def bravo():
    return 2
";
        let chunks = split_code(src, 8, 0, encoder());
        assert!(!chunks.is_empty());
        for c in &chunks {
            assert!(c.token_count <= 8);
        }
    }

    #[test]
    fn is_lossless_and_offsets_are_exact() {
        let src = "fn a() {\n    one();\n}\n\nfn b() {\n    two();\n}\n";
        let chunks = split_code(src, 10, 0, encoder());
        let rejoined: String = chunks.iter().map(|c| c.content.as_str()).collect();
        assert_eq!(rejoined, src, "splitting must not lose or duplicate bytes");
        for c in &chunks {
            assert_eq!(&src[c.start_byte..c.end_byte], c.content);
        }
    }

    #[test]
    fn empty_input() {
        assert!(split_code("", 10, 0, encoder()).is_empty());
    }

    #[test]
    fn short_input_is_one_chunk() {
        let chunks = split_code("let x = 1;", 100, 0, encoder());
        assert_eq!(chunks.len(), 1);
        assert_eq!(chunks[0].content, "let x = 1;");
    }

    #[test]
    fn indices_are_sequential() {
        let src = "fn a() {\n    one();\n}\n\nfn b() {\n    two();\n}\n\nfn c() {\n    x();\n}\n";
        for (i, c) in split_code(src, 8, 0, encoder()).iter().enumerate() {
            assert_eq!(c.index, i);
        }
    }
}
