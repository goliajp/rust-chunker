// markdown-aware chunking with section metadata
//
// splits at header boundaries first, then recursively within each section.
// each chunk carries the section header it belongs to.
//
// run: cargo run -p chunkedrs --example markdown

fn main() {
    let document = "\
# Introduction

Chunkedrs is an AI-native text chunking library for Rust. It splits long documents \
into token-accurate pieces suitable for embedding and retrieval pipelines.

The library supports three strategies: recursive (default), markdown-aware, and \
semantic (via embeddings).

## Recursive Strategy

The recursive strategy splits text at natural boundaries in priority order: \
paragraphs, lines, sentences, words, and finally characters. It merges consecutive \
pieces until the token budget is reached.

This is the fastest strategy and works well for unstructured text like articles, \
emails, and chat transcripts.

## Markdown Strategy

The markdown strategy detects header lines (lines starting with #) and splits \
sections independently. Each resulting chunk knows which section it belongs to \
via the section metadata field.

This is ideal for technical documentation, READMEs, and wiki pages where \
preserving document structure matters for retrieval quality.

## Token Accuracy

Every chunk is guaranteed to contain no more than the configured maximum number \
of tokens. The library uses tiktoken for precise BPE token counting, supporting \
all mainstream LLM tokenizers including GPT-4o, Claude, Llama, and DeepSeek.
";

    let chunks = chunkedrs::chunk(document).markdown().max_tokens(40).split();

    println!(
        "input: {} bytes, split into {} chunks\n",
        document.len(),
        chunks.len()
    );

    for chunk in &chunks {
        let section = chunk.section.as_deref().unwrap_or("(no header)");
        println!(
            "chunk {} | section: {:<25} | tokens: {:>2} | bytes: {}..{}",
            chunk.index, section, chunk.token_count, chunk.start_byte, chunk.end_byte,
        );
        // show first line of content as preview
        let first_line = chunk.content.lines().next().unwrap_or("");
        let preview: String = first_line.chars().take(70).collect();
        let ellipsis = if first_line.len() > 70 { "..." } else { "" };
        println!("  \"{preview}{ellipsis}\"\n");
    }

    // verify every chunk has section metadata (except preamble before first header)
    let with_section = chunks.iter().filter(|c| c.section.is_some()).count();
    println!(
        "{} of {} chunks have section metadata",
        with_section,
        chunks.len()
    );
}
