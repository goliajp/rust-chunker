# chunkedrs

[![Crates.io](https://img.shields.io/crates/v/chunkedrs?style=flat-square&logo=rust)](https://crates.io/crates/chunkedrs)
[![docs.rs](https://img.shields.io/docsrs/chunkedrs?style=flat-square&logo=docs.rs)](https://docs.rs/chunkedrs)
[![License](https://img.shields.io/crates/l/chunkedrs?style=flat-square)](LICENSE)
[![Downloads](https://img.shields.io/crates/d/chunkedrs?style=flat-square)](https://crates.io/crates/chunkedrs)
[![MSRV](https://img.shields.io/badge/MSRV-1.94-blue?style=flat-square)](https://www.rust-lang.org)

**English** | [简体中文](README.zh-CN.md) | [日本語](README.ja.md)

Token-accurate text chunking for RAG pipelines — recursive, markdown-aware, and semantic splitting. Built on [tiktoken](https://crates.io/crates/tiktoken), the fastest pure-Rust BPE tokenizer.

## Highlights

- **Token-accurate** — every chunk is guaranteed within your token budget, not character-approximate
- **3 strategies** — recursive (fast, general), markdown-aware (preserves headers), semantic (embedding-based breakpoints)
- **Rich metadata** — byte offsets, token counts, and section headers on every chunk
- **Overlap** — configurable token overlap between chunks for retrieval context preservation
- **Any tokenizer** — auto-detect from model name (`gpt-4o`, `claude`, `llama`) or specify encoding directly
- **Built on tiktoken** — the fastest pure-Rust BPE tokenizer with 9 encodings across all major LLMs

## Why chunkedrs?

RAG pipelines need text split into chunks that fit model context windows. Naive splitting (by character count or fixed size) breaks mid-word, mid-sentence, or mid-paragraph — destroying meaning and hurting retrieval quality.

chunkedrs splits at **semantic boundaries** (paragraphs, sentences, words) while enforcing **exact token limits**. No chunk ever exceeds `max_tokens`.

| Feature | chunkedrs | text-splitter | Manual |
|---------|-----------|---------------|--------|
| Token-accurate limits | Yes (tiktoken) | Character-based | No |
| Recursive splitting | Yes | Yes | DIY |
| Markdown-aware | Yes (section metadata) | No | DIY |
| Semantic splitting | Yes (via embedrs) | No | DIY |
| Byte offsets | Yes | No | DIY |
| Token count per chunk | Yes | No | DIY |
| Overlap support | Token-level | Character-level | DIY |
| Tokenizer selection | Model name or encoding | N/A | N/A |

## Strategies

| Strategy | Use case | Speed |
|----------|----------|-------|
| **Recursive** (default) | General text — paragraphs, sentences, words | Fastest |
| **Markdown** | Documents with `#` headers — preserves section metadata | Fast |
| **Semantic** | High-quality RAG — splits at meaning boundaries via embeddings | Slower (API calls) |

## Quick start

Add to your `Cargo.toml`:

```toml
[dependencies]
chunkedrs = "1"
```

Split text with defaults (recursive, 512 max tokens, no overlap):

```rust
use chunkedrs::Chunk;

let chunks: Vec<Chunk> = chunkedrs::chunk("your long text here...").split();
for chunk in &chunks {
    println!("[{}] {} tokens (bytes {}..{})", chunk.index, chunk.token_count, chunk.start_byte, chunk.end_byte);
}
// Output:
// [0] 4 tokens (bytes 0..21)
```

## Token-accurate splitting

```rust
let chunks = chunkedrs::chunk("your long text here...")
    .max_tokens(256)
    .overlap(50)
    .model("gpt-4o")
    .split();

// every chunk is guaranteed to have <= 256 tokens
assert!(chunks.iter().all(|c| c.token_count <= 256));
```

## Markdown-aware splitting

```rust
let markdown = "# Intro\n\nSome text.\n\n## Details\n\nMore text here.\n";
let chunks = chunkedrs::chunk(markdown).markdown().split();

// each chunk knows which section it belongs to
assert_eq!(chunks[0].section.as_deref(), Some("# Intro"));
```

## Semantic splitting

With the `semantic` feature enabled, split at meaning boundaries using embeddings:

```toml
[dependencies]
chunkedrs = { version = "1", features = ["semantic"] }
```

```rust,ignore
let client = embedrs::openai("sk-...");
let chunks = chunkedrs::chunk("your long text here...")
    .semantic(&client)
    .threshold(0.5)
    .split_async()
    .await?;
```

## Chunk metadata

Every `Chunk` carries rich metadata:

```rust
pub struct Chunk {
    pub content: String,         // the text
    pub index: usize,            // position in sequence
    pub start_byte: usize,       // byte offset in original text
    pub end_byte: usize,         // byte offset (exclusive)
    pub token_count: usize,      // exact token count
    pub section: Option<String>, // markdown header (if applicable)
}
```

## Overlap

Token overlap between consecutive chunks preserves context at boundaries — critical for retrieval quality:

```rust
let chunks = chunkedrs::chunk("your long text here...")
    .max_tokens(256)
    .overlap(50)
    .split();
```

## Tokenizer selection

```rust
// auto-detect from model name
let chunks = chunkedrs::chunk(text).model("gpt-4o").split();

// or specify encoding directly
let chunks = chunkedrs::chunk(text).encoding("cl100k_base").split();

// default: o200k_base (GPT-4o, GPT-4-turbo)
```

<!-- ECOSYSTEM BEGIN (synced by claws/opensource/scripts/sync-ecosystem.py — edit ecosystem.toml, not this block) -->

## Ecosystem

[tiktoken](https://crates.io/crates/tiktoken) · [@goliapkg/tiktoken-wasm](https://www.npmjs.com/package/@goliapkg/tiktoken-wasm) · [instructors](https://crates.io/crates/instructors) · **chunkedrs** · [embedrs](https://crates.io/crates/embedrs)

<!-- ECOSYSTEM END -->

## License

MIT
