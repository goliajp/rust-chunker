# chunkedrs

[![Crates.io](https://img.shields.io/crates/v/chunkedrs?style=flat-square&logo=rust)](https://crates.io/crates/chunkedrs)
[![docs.rs](https://img.shields.io/docsrs/chunkedrs?style=flat-square&logo=docs.rs)](https://docs.rs/chunkedrs)
[![License](https://img.shields.io/crates/l/chunkedrs?style=flat-square)](LICENSE)

**English** | [简体中文](README.zh-CN.md) | [日本語](README.ja.md)

AI-native text chunking for Rust — split long documents into token-accurate pieces for embedding and retrieval.

Built on [tiktoken](https://crates.io/crates/tiktoken) for precise token counting. Every chunk is guaranteed to respect your token budget.

## Why chunkedrs?

RAG pipelines need text split into chunks that fit model context windows. Naive splitting (by character count or fixed size) breaks mid-word, mid-sentence, or mid-paragraph — destroying meaning and hurting retrieval quality.

chunkedrs splits at **semantic boundaries** (paragraphs → sentences → words) while enforcing **exact token limits**. No chunk ever exceeds `max_tokens`.

## Strategies

| Strategy | Use case | Speed |
|----------|----------|-------|
| **Recursive** (default) | General text — paragraphs, sentences, words | Fastest |
| **Markdown** | Documents with `#` headers — preserves section metadata | Fast |
| **Semantic** | High-quality RAG — splits at meaning boundaries via embeddings | Slower (API calls) |

## Quick start

```rust
// split with defaults: recursive, 512 max tokens, no overlap
let chunks = chunkedrs::chunk("your long text here...").split();
for chunk in &chunks {
    println!("[{}] {} tokens", chunk.index, chunk.token_count);
}
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

## License

MIT
