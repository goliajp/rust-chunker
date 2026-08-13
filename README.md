# chunkedrs

[![Crates.io](https://img.shields.io/crates/v/chunkedrs?style=flat-square&logo=rust)](https://crates.io/crates/chunkedrs)
[![docs.rs](https://img.shields.io/docsrs/chunkedrs?style=flat-square&logo=docs.rs)](https://docs.rs/chunkedrs)
[![License](https://img.shields.io/crates/l/chunkedrs?style=flat-square)](LICENSE)
[![Downloads](https://img.shields.io/crates/d/chunkedrs?style=flat-square)](https://crates.io/crates/chunkedrs)

**English** | [简体中文](README.zh-CN.md) | [日本語](README.ja.md)

Token-accurate text chunking for RAG pipelines — recursive, markdown-aware, and semantic splitting. Built on [tiktoken](https://crates.io/crates/tiktoken), the fastest pure-Rust BPE tokenizer.

## Highlights

- **Token-accurate** — every chunk is guaranteed within your token budget, not character-approximate
- **5 strategies** — recursive, markdown, code, HTML, and semantic (embedding-based)
- **Token spans** — every chunk knows its range in the document's token stream, which is what [late chunking](#late-chunking) needs
- **CJK-aware** — Chinese and Japanese split on `。`, `！`, `？` and clause marks, not on the ASCII space they do not use
- **Section hierarchy** — a `### From source` chunk knows it lives under `## Installation` under `# Guide`
- **Rich metadata** — byte offsets, token offsets, token counts, header ancestry
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
| Markdown | Header ancestry as chunk metadata | CommonMark-structured splitting |
| Code | Boundary-aware, no dependencies | **Tree-sitter, real AST** |
| HTML | Boundary-aware, no dependencies | No |
| Token spans (for late chunking) | Yes | No |
| Embedding-based semantic splitting | Yes (via [embedrs](https://crates.io/crates/embedrs)) | No |
| Byte offsets | Yes | Yes |
| Overlap | Tokens | Follows the configured sizer |
| Dependencies | 1 (`tiktoken`) | Grammar crate per language for code |

Reach for chunkedrs when you want token accuracy without configuring it, header ancestry on every chunk, token spans for late chunking, or embedding-based breakpoints. **Reach for text-splitter when you need real AST-aware code splitting** — chunkedrs's `code()` reads punctuation, not syntax, and says so.

## Strategies

| Strategy | Use case | Speed |
|----------|----------|-------|
| **Recursive** (default) | General text — paragraphs, sentences, clauses, words | Fastest |
| **Markdown** | Documents with headers — preserves section ancestry | Fast |
| **Code** | Source in any language — blank lines, block closers, lines | Fast |
| **HTML** | Web pages — block-level tag boundaries | Fast |
| **Semantic** | High-quality RAG — splits at meaning boundaries via embeddings | Slower (API calls) |

## Quick start

Add to your `Cargo.toml`:

```toml
[dependencies]
chunkedrs = "2"
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
let markdown = "# Guide\n\nIntro.\n\n## Install\n\nRun cargo add.\n";
let chunks = chunkedrs::chunk(markdown).markdown().split();

// each chunk knows the section it belongs to...
assert_eq!(chunks[0].section(), Some("# Guide"));

// ...and its full ancestry, so nested sections keep their context
assert_eq!(chunks[1].section_path, ["# Guide", "## Install"]);
```

Headers are ATX (`#`) or setext (`===` / `---`). Fenced code blocks are skipped, so a `#` comment inside a `bash` block is a comment; YAML and TOML front matter is excluded from header detection.

## Code and HTML

```rust
let src = "fn a() {\n    one();\n}\n\nfn b() {\n    two();\n}\n";
let chunks = chunkedrs::chunk(src).code().max_tokens(20).split();

let page = "<h1>Title</h1><p>First para.</p><p>Second para.</p>";
let chunks = chunkedrs::chunk(page).html().max_tokens(10).split();
```

Both are **boundary-aware, not AST-aware**. `code()` splits on blank lines, then on brackets closing a block at column 0, then on lines — and deliberately skips the prose separators, because splitting code on `". "` cuts inside string literals. `html()` scans for block-level closing tags (`</p>`, `</li>`, `</section>`, …) case-insensitively and degrades to the text ladder on malformed markup.

Neither parses anything. That is the tradeoff: they work on any language and any markup and add no dependencies, but a `}` inside a string literal looks like a block close. When you need AST fidelity, use a tree-sitter based splitter such as [text-splitter](https://crates.io/crates/text-splitter)'s `CodeSplitter`.

## Late chunking

Embedding each chunk in isolation loses the surrounding context — a chunk that says "it costs $40" no longer knows what "it" was. [Late chunking](https://arxiv.org/abs/2409.04701) inverts the order: embed the whole document once, then pool each chunk's vector over its own slice of the token stream.

The slice is the part chunkedrs supplies:

```rust
let doc = "First sentence here. Second sentence here. Third sentence here.";
let chunks = chunkedrs::chunk(doc).max_tokens(8).split();

let encoder = tiktoken::get_encoding("o200k_base").unwrap();
let document_tokens = encoder.encode(doc);

for chunk in &chunks {
    // in a real pipeline this indexes the encoder's per-token hidden states
    let span = &document_tokens[chunk.start_token..chunk.end_token];
    assert!(!span.is_empty());
}
```

No re-tokenizing, no guessing where a chunk landed. See [`examples/token_spans.rs`](examples/token_spans.rs).

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
chunkedrs = { version = "2", features = ["semantic"] }
```

```rust,ignore
let client = embedrs::openai("sk-...");
let chunks = chunkedrs::chunk("your long text here...")
    .semantic(&client)
    .threshold(0.5)
    .split_async()
    .await?;
```

`.semantic()` returns a builder with no `.split()` at all — semantic splitting makes network calls, so the synchronous method does not exist to be called by mistake.

## Chunk metadata

```rust
#[non_exhaustive]
pub struct Chunk {
    pub content: String,           // the text
    pub index: usize,              // position in sequence
    pub start_byte: usize,         // byte offset in original text
    pub end_byte: usize,           // byte offset (exclusive)
    pub start_token: usize,        // token offset in the document's token stream
    pub end_token: usize,          // token offset (exclusive)
    pub token_count: usize,        // tokens in this chunk, counted on its own
    pub section_path: Vec<String>, // header ancestry, outermost first
}

impl Chunk {
    pub fn section(&self) -> Option<&str>;  // the deepest header
}
```

`token_count` and `end_token - start_token` answer different questions and may differ. `token_count` is a fresh count of the chunk alone — what `max_tokens` bounds and what you budget a context window against. The span is the chunk's footprint in the document stream; where a separator sits inside a token, it widens to cover it, so consecutive spans can share one token.

`Chunk` is `#[non_exhaustive]`, so future metadata will not be another major version. Build one with `Chunk::new(...).with_bytes(..).with_tokens(..)`.

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

## Vocabulary features

Tokenizer vocabularies are the bulk of the compiled size, and most builds use
one. They are opt-out: the default carries all 17 encodings, and
`default-features = false` keeps only `o200k_base` — the encoder every
unresolved name falls back to, so it is never absent.

```toml
# everything (default)
chunkedrs = "2"

# o200k_base only — GPT-4o, GPT-5, o-series
chunkedrs = { version = "2", default-features = false }

# o200k_base plus the Zhipu family
chunkedrs = { version = "2", default-features = false, features = ["vocab-zhipu"] }
```

Measured on `examples/basic`, release build:

| | size |
|---|---:|
| all vocabularies (default) | 7,100,544 |
| `default-features = false` | 2,695,104 |
| | **−62%** |

Vendor groups: `vocab-openai`, `vocab-meta`, `vocab-deepseek`, `vocab-qwen`,
`vocab-mistral`, `vocab-moonshot`, `vocab-zhipu`, `vocab-minimax`. Individual
vocabularies: `vocab-cl100k_base`, `vocab-llama3`, `vocab-glm5`, and so on.

**One sharp edge.** Asking for a vocabulary this build did not compile in is
indistinguishable from a typo — both fall back to `o200k_base` silently, so you
get plausible counts for the wrong tokenizer. If you slim the build, check that
the encodings you name are ones you enabled.

Anthropic and Google do not publish tiktoken-compatible vocabularies, so
`claude-*` and `gemini-*` do **not** resolve — an unrecognised name falls back
to `o200k_base`, which is an approximation rather than an exact count. Use
`.encoding()` if you need to pin that behaviour explicitly.

## Upgrading from 1.x

Two mechanical renames and one behaviour note.

```rust
// 1.x
chunk.section.as_deref()          // Option<&str>
Chunk { content, index, .. }      // struct literal

// 2.0
chunk.section()                   // Option<&str> — same value, now a method
chunk.section_path                // Vec<String> — the full ancestry
Chunk::new(content).with_index(i) // Chunk is #[non_exhaustive]
```

`.semantic(&client)` now returns a `SemanticChunkBuilder`. If you were calling `.split()` on it, that was a runtime panic; it is now a compile error, and `.split_async()` is what you want.

Chunk boundaries move relative to 1.0.x — see the [CHANGELOG](CHANGELOG.md) for why. Re-chunk and re-embed; byte offsets remain exact against the source.

<!-- ECOSYSTEM BEGIN (generated — edit ecosystem.toml, not this block) -->

## Ecosystem

[tiktoken](https://crates.io/crates/tiktoken) · [@goliapkg/tiktoken-wasm](https://www.npmjs.com/package/@goliapkg/tiktoken-wasm) · [instructors](https://crates.io/crates/instructors) · **chunkedrs** · [embedrs](https://crates.io/crates/embedrs)

<!-- ECOSYSTEM END -->

## License

MIT
