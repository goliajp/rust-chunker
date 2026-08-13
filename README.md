# chunkedrs

[![Crates.io](https://img.shields.io/crates/v/chunkedrs?style=flat-square&logo=rust)](https://crates.io/crates/chunkedrs)
[![docs.rs](https://img.shields.io/docsrs/chunkedrs?style=flat-square&logo=docs.rs)](https://docs.rs/chunkedrs)
[![License](https://img.shields.io/crates/l/chunkedrs?style=flat-square)](LICENSE)
[![Downloads](https://img.shields.io/crates/d/chunkedrs?style=flat-square)](https://crates.io/crates/chunkedrs)

**English** | [简体中文](README.zh-CN.md) | [日本語](README.ja.md)

Token-accurate text chunking for RAG pipelines — recursive, markdown-aware, and semantic splitting. Built on [tiktoken](https://crates.io/crates/tiktoken), the fastest pure-Rust BPE tokenizer.

## Highlights

- **Token-accurate** — every chunk is guaranteed within your token budget, not character-approximate
- **3 strategies** — recursive (fast, general), markdown-aware (preserves headers), semantic (embedding-based breakpoints)
- **CJK-aware** — Chinese and Japanese split on `。`, `！`, `？` and clause marks, not on the ASCII space they do not use
- **Rich metadata** — byte offsets, token counts, and section headers on every chunk
- **Overlap** — configurable token overlap between chunks
- **Any tokenizer** — auto-detect from model name or specify one of 17 encodings directly
- **Built on tiktoken 4** — the fastest pure-Rust BPE tokenizer, covering 10 providers

## Why chunkedrs?

RAG pipelines need text split into chunks that fit model context windows. Naive splitting (by character count or fixed size) breaks mid-word, mid-sentence, or mid-paragraph — destroying meaning and hurting retrieval quality.

chunkedrs splits at **semantic boundaries** (paragraphs, sentences, clauses, words) while enforcing **exact token limits**. No chunk ever exceeds `max_tokens`.

### Compared to [text-splitter](https://crates.io/crates/text-splitter)

Both are good at this; they aim at different defaults. text-splitter is the broader toolkit — it has tree-sitter code splitting, which chunkedrs does not.

| | chunkedrs | text-splitter 0.32 |
|---|---|---|
| Default sizing | Tokens, always | Characters; token sizing is opt-in |
| Tokenizer | tiktoken, built in | `tiktoken-rs` or `tokenizers`, via feature |
| Markdown | Headers as chunk metadata | CommonMark-structured splitting |
| Code (tree-sitter) | No | Yes |
| Embedding-based semantic splitting | Yes (via [embedrs](https://crates.io/crates/embedrs)) | No |
| Byte offsets | Yes | Yes |
| Overlap | Tokens | Follows the configured sizer |

Reach for chunkedrs when you want token accuracy without configuring it, section metadata on every chunk, or embedding-based breakpoints. Reach for text-splitter when you need to chunk source code.

## Strategies

| Strategy | Use case | Speed |
|----------|----------|-------|
| **Recursive** (default) | General text — paragraphs, sentences, clauses, words | Fastest |
| **Markdown** | Documents with headers — preserves section metadata | Fast |
| **Semantic** | High-quality RAG — splits at meaning boundaries via embeddings | Slower (API calls) |

## Quick start

Add to your `Cargo.toml`:

```toml
[dependencies]
chunkedrs = "1.1"
```

Split text with defaults (recursive, 512 max tokens, no overlap):

```rust
use chunkedrs::Chunk;

let chunks: Vec<Chunk> = chunkedrs::chunk("your long text here...").split();
for chunk in &chunks {
    println!("[{}] {} tokens (bytes {}..{})", chunk.index, chunk.token_count, chunk.start_byte, chunk.end_byte);
}
// Output:
// [0] 5 tokens (bytes 0..22)
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

## CJK text

Chinese and Japanese do not put a space after a sentence mark, so a splitter
whose separators all end in an ASCII space finds no boundary at all and cuts
mid-word. chunkedrs splits on the marks these scripts actually use, and keeps
closing quotes with the sentence they close:

```rust
let zh = "他说「今天天气很好。」然后就出门了。她回答说「确实不错。」于是也跟着出去了。";
let chunks = chunkedrs::chunk(zh).max_tokens(12).split();

assert_eq!(chunks[0].content, "他说「今天天气很好。」");
assert_eq!(chunks[1].content, "然后就出门了。");
```

When a whole sentence will not fit, the clause marks (`，`、`、`、`；`) are the
next boundary down, and only then does it fall back to token offsets.

## Semantic splitting

With the `semantic` feature enabled, split at meaning boundaries using embeddings:

```toml
[dependencies]
chunkedrs = { version = "1.1", features = ["semantic"] }
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

Token overlap between consecutive chunks carries context across boundaries:

```rust
let chunks = chunkedrs::chunk("your long text here...")
    .max_tokens(256)
    .overlap(50)
    .split();
```

Whether overlap helps is workload-dependent — recent retrieval evaluations
report little measurable benefit against a real cost in index size. It is off
by default; measure before turning it on.

## Tokenizer selection

```rust
// auto-detect from model name
let chunks = chunkedrs::chunk(text).model("gpt-4o").split();

// or specify one of the 17 encodings directly
let chunks = chunkedrs::chunk(text).encoding("cl100k_base").split();

// default: o200k_base (GPT-4o, GPT-5, o-series)
```

Model names resolve across OpenAI, Meta (`llama-3.1-70b`), DeepSeek
(`deepseek-v4`), Alibaba (`qwen2.5-72b`), Mistral, Moonshot (`kimi-k2`), Zhipu
(`glm-5`) and MiniMax (`minimax-m2`).

Anthropic and Google do not publish tiktoken-compatible vocabularies, so
`claude-*` and `gemini-*` do **not** resolve — an unrecognised name falls back
to `o200k_base`, which is an approximation rather than an exact count. Use
`.encoding()` if you need to pin that behaviour explicitly.

<!-- ECOSYSTEM BEGIN (generated — edit ecosystem.toml, not this block) -->

## Ecosystem

[tiktoken](https://crates.io/crates/tiktoken) · [@goliapkg/tiktoken-wasm](https://www.npmjs.com/package/@goliapkg/tiktoken-wasm) · [instructors](https://crates.io/crates/instructors) · **chunkedrs** · [embedrs](https://crates.io/crates/embedrs)

<!-- ECOSYSTEM END -->

## License

MIT
