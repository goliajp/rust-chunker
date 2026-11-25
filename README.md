# chunkedrs

[![Crates.io](https://img.shields.io/crates/v/chunkedrs?style=flat-square&logo=rust)](https://crates.io/crates/chunkedrs)
[![docs.rs](https://img.shields.io/docsrs/chunkedrs?style=flat-square&logo=docs.rs)](https://docs.rs/chunkedrs)
[![CI](https://img.shields.io/github/actions/workflow/status/goliajp/rust-chunker/ci.yml?branch=develop&style=flat-square&logo=github&label=ci)](https://github.com/goliajp/rust-chunker/actions/workflows/ci.yml)
[![License](https://img.shields.io/crates/l/chunkedrs?style=flat-square)](#license)

**English** | [简体中文](README.zh-CN.md) | [日本語](README.ja.md)

> **Upgrading from 2.1?** No API change. Chunk boundaries move, so re-chunk stored indexes: tiktoken 5 gives Llama 4, Qwen 3.5 to 3.8, Mistral Large 4 and `deepseek-flash` their own vocabularies (their token counts change), and the new planner ends some chunks at a different boundary than 2.1 did. Documents of 32 KiB and more are now encoded on several threads; `TIKTOKEN_THREADS=1` turns that off. Details: [CHANGELOG](CHANGELOG.md#upgrading-from-21).

chunkedrs splits long documents into chunks that fit a token budget, for embedding and retrieval. It counts tokens with [tiktoken](https://crates.io/crates/tiktoken) (20 encodings: OpenAI, Llama, DeepSeek, Qwen, Mistral, Kimi, GLM, MiniMax) and gives every chunk its exact byte range in the source and its token range in the document's token stream. It has no dependency besides tiktoken; the `semantic` feature adds [embedrs](https://crates.io/crates/embedrs).

On an Apple M4 Pro, one thread, `o200k_base`, 512 tokens per chunk: a 200 KB English book splits in 0.35 ms. text-splitter 0.33.6 takes 1.19 ms on the same tokenizer and 34.8 ms with tiktoken-rs 0.12, the tokenizer its `tiktoken-rs` feature uses.

## Highlights

- **Five strategies**: recursive (paragraph, line, sentence, clause, word), markdown with header ancestry, code and HTML by boundary markers, and semantic splitting by embedding similarity.
- **Token budget held exactly**: `token_count` is counted on the chunk's own text and is at most `max_tokens`. The one exception is a single character that alone takes more tokens than the budget.
- **Offsets**: `&doc[start_byte..end_byte] == content` for every chunk, and `start_token..end_token` is the chunk's range in `encode(doc)`, which [late chunking](#late-chunking) needs.
- **Chinese and Japanese**: sentences end at `。！？` and clauses at `，、；：` with no space after them; a closing quote stays with its sentence.
- **Batches**: `Chunker` resolves the encoder once; `split_batch` spreads a slice of documents over all cores.

## Quick start

```toml
[dependencies]
chunkedrs = "2"
```

```rust
// recursive, 512 tokens, no overlap, o200k_base
let chunks = chunkedrs::chunk("your long text here...").split();
for c in &chunks {
    println!("[{}] {} tokens, bytes {}..{}", c.index, c.token_count, c.start_byte, c.end_byte);
}

// a model name picks the encoding
let chunks = chunkedrs::chunk("your long text here...")
    .max_tokens(256)
    .overlap(32)
    .model("gpt-5.6-terra")
    .split();
assert!(chunks.iter().all(|c| c.token_count <= 256));

// one configuration, many documents
let chunker = chunkedrs::Chunker::new().max_tokens(256).markdown();
let docs = ["# A\n\nFirst.", "# B\n\nSecond."];
let all = chunker.split_batch(&docs);
assert_eq!(all[1][0].section(), Some("# B"));
```

## Performance

Milliseconds per document, median of 9 rounds after warmup. Apple M4 Pro Mac mini, rustc 1.100, release with LTO, `TIKTOKEN_THREADS=1`, `o200k_base`, no overlap; load average 1.3 to 2.0 during the run, 2026-11-25. text-splitter gets the same `max_tokens` as its capacity and its defaults otherwise; its code splitter uses tree-sitter-rust 0.24.

| corpus | strategy | max_tokens | one `encode` | chunkedrs | text-splitter + tiktoken 5.5 | text-splitter + tiktoken-rs 0.12 |
|---|---|--:|--:|--:|--:|--:|
| book_en.txt | recursive | 512 | 0.116 | **0.351** | 1.191 | 34.82 |
| book_en.txt | recursive | 128 | 0.116 | **0.414** | 2.489 | 64.62 |
| zh_xiyouji.md | recursive | 512 | 0.146 | **0.420** | 0.478 | 20.17 |
| zh_xiyouji.md | markdown | 512 | 0.146 | **0.482** | 1.026 | 40.66 |
| zh_xiyouji.md | recursive | 128 | 0.157 | **0.532** | 1.441 | 30.67 |
| zh_xiyouji.md | markdown | 128 | 0.157 | **0.628** | 1.959 | 43.74 |
| md_rust_book.md | recursive | 512 | 0.100 | **0.306** | 0.738 | 22.32 |
| md_rust_book.md | markdown | 512 | 0.100 | **0.401** | 1.191 | 25.26 |
| md_rust_book.md | recursive | 128 | 0.099 | **0.368** | 0.938 | 21.78 |
| md_rust_book.md | markdown | 128 | 0.099 | **0.503** | 1.779 | 33.41 |
| code_rust.txt | recursive | 512 | 0.060 | **0.199** | 0.323 | 11.69 |
| code_rust.txt | code | 512 | 0.060 | **0.207** | 6.633 | 34.28 |
| code_rust.txt | recursive | 128 | 0.060 | **0.243** | 0.525 | 15.08 |
| code_rust.txt | code | 128 | 0.060 | **0.280** | 7.745 | 61.49 |

`book_en.txt` is 200,000 bytes of *Pride and Prejudice*, `zh_xiyouji.md` 178,622 bytes of *Journey to the West* chapters 1-9 in markdown (Wikisource), `md_rust_book.md` 174,237 bytes of *The Rust Programming Language* chapters 3-6, `code_rust.txt` 100,000 bytes of Rust source. All are in [`bench-compare/data`](bench-compare/data).

- Every chunk of every row above was recounted with tiktoken: none is over budget, for chunkedrs or text-splitter.
- chunkedrs takes 2.9 to 5.1 times one `encode` of the document. The document is encoded once (one `encode`), each token's byte length is looked up (about half an `encode`), and each chunk is counted once on its own (about one more `encode`, because BPE does not add up across a cut); finding boundaries and building chunks is the rest.
- `cl100k_base` gives the same picture (0.19 to 0.72 ms for chunkedrs). With tiktoken's default threads, the whole-document encode of these 100 to 200 KB texts runs on several threads and chunkedrs takes 0.17 to 0.45 ms.
- `Chunker::split_batch` on 64 documents (the four corpora 16 times, 10.4 MB): 28.2 ms one after another, 3.9 ms as a batch.

Reproduce with `TIKTOKEN_THREADS=1 cargo run --release -p bench-compare`.

## How it splits

The document is encoded once. From the start of a chunk, the byte offset after `max_tokens` document tokens bounds where the chunk can end. Inside that window the chunk ends at the last boundary of the highest-priority kind present:

| Strategy | Boundaries, highest priority first |
|---|---|
| recursive (default) | blank line; line break; `. ! ?` followed by a space, `。！？．｡…‥`; `; , :` and `；，、：･`; space, tab, ideographic space |
| `markdown()` | header lines first (ATX `#` and setext `===` / `---`, not inside fenced code or front matter), then the recursive boundaries inside each section |
| `code()` | blank line; `}` `};` `)` `]` `];` `end` alone on a line at column 0; line break; space, tab |
| `html()` | after a block-level closing tag (`</p>`, `</li>`, `</section>`, `</h1>`, ...), then the recursive boundaries |
| `semantic()` | between sentence groups whose embeddings fall below the similarity threshold, then the recursive boundaries |

When no boundary falls inside the window, the chunk ends at a token boundary, moved back to a character boundary. The chosen text is then counted on its own: BPE merges across a cut, so a slice can take one token more or less than its share of the document's tokens. In the rare case it is over budget, the window shrinks and the chunk is chosen again.

Code and HTML splitting read punctuation and tags, not syntax: they parse nothing and add no dependency, and a `}` inside a string literal looks like a block close to them. For syntax-aware code splitting, use text-splitter's tree-sitter `CodeSplitter`.

## Markdown

```rust
let md = "# Guide\n\nIntro.\n\n## Install\n\nRun cargo add.\n";
let chunks = chunkedrs::chunk(md).markdown().split();

assert_eq!(chunks[0].section(), Some("# Guide"));
assert_eq!(chunks[1].section_path, ["# Guide", "## Install"]);
```

Header lines go into `section_path`, not into `content`, so joining every chunk's content does not reproduce them.

## Chinese and Japanese

```rust
let zh = "他说「今天天气很好。」然后就出门了。她回答说「确实不错。」于是也跟着出去了。";
let chunks = chunkedrs::chunk(zh).max_tokens(12).split();

assert_eq!(chunks[0].content, "他说「今天天气很好。」");
assert_eq!(chunks[1].content, "然后就出门了。");
```

A splitter whose separators all end in an ASCII space finds no sentence boundary in this text and cuts inside a word.

## Late chunking

[Late chunking](https://arxiv.org/abs/2409.04701) embeds the whole document once and pools each chunk's vector over the chunk's own tokens. `start_token..end_token` is that range in `encode(doc)`:

```rust
let doc = "First sentence here. Second sentence here. Third sentence here.";
let chunks = chunkedrs::chunk(doc).max_tokens(8).split();

let encoder = tiktoken::get_encoding("o200k_base").unwrap();
let document_tokens = encoder.encode(doc);
for chunk in &chunks {
    // in a real pipeline this indexes the model's per-token hidden states
    let span = &document_tokens[chunk.start_token..chunk.end_token];
    assert!(!span.is_empty());
}
```

Where a boundary falls inside a token (`". "` before a word, where the tokenizer joins the space to the word), the span covers that token, so two consecutive spans can share one token. `token_count` can therefore differ from `end_token - start_token`. See [`examples/token_spans.rs`](examples/token_spans.rs).

## Semantic splitting

```toml
chunkedrs = { version = "2", features = ["semantic"] }
```

```rust,ignore
let client = embedrs::Client::openai("sk-...");
let chunks = chunkedrs::chunk("your long text here...")
    .semantic(&client)
    .threshold(0.5)
    .split_async()
    .await?;
```

Sentences are embedded in batches of the provider's request limit. `.semantic()` returns a builder that has only `split_async()`, so the network call cannot be made from a synchronous `split()` by mistake.

## Chunk

```rust
#[non_exhaustive]
pub struct Chunk {
    pub content: String,           // the text, equal to &doc[start_byte..end_byte]
    pub index: usize,              // position in the sequence
    pub start_byte: usize,
    pub end_byte: usize,           // exclusive
    pub start_token: usize,        // range in encode(doc)
    pub end_token: usize,          // exclusive
    pub token_count: usize,        // tokens of content counted on its own
    pub section_path: Vec<String>, // markdown header ancestry, outermost first
}
```

Chunks that contain only whitespace are dropped. `Chunk` is `#[non_exhaustive]`; build one by hand with `Chunk::new(..).with_bytes(..).with_tokens(..)`.

Overlap (`.overlap(n)`) starts each chunk about `n` tokens before the previous one ended, at a character boundary, and is off by default. Whether it helps retrieval depends on the workload; measure before you turn it on.

## Encodings and features

`.model("...")` resolves model names across OpenAI, Meta, DeepSeek, Qwen, Mistral, Moonshot, Zhipu and MiniMax; `.encoding("cl100k_base")` names an encoding directly and wins over `.model()`. An unknown name falls back to `o200k_base` without an error. Anthropic and Google publish no tiktoken-compatible vocabulary, so `claude-*` and `gemini-*` take that fallback, which approximates their counts.

Vocabularies are opt-out features, as in tiktoken. The default compiles in all 20 encodings; `default-features = false` keeps only `o200k_base`, the fallback encoder.

```toml
chunkedrs = "2"                                                   # all vocabularies
chunkedrs = { version = "2", default-features = false }           # o200k_base only
chunkedrs = { version = "2", default-features = false, features = ["vocab-qwen"] }
```

`examples/basic`, release build on macOS arm64: 9,189,376 bytes with all vocabularies, 2,072,256 bytes with `default-features = false` (77% smaller).

Vendor groups: `vocab-openai`, `vocab-meta`, `vocab-deepseek`, `vocab-qwen`, `vocab-mistral`, `vocab-moonshot`, `vocab-zhipu`, `vocab-minimax`. Single vocabularies: `vocab-cl100k_base`, `vocab-llama4`, `vocab-qwen3_5`, `vocab-mistral_v15`, `vocab-glm5` and the rest of tiktoken's list. A vocabulary that is not compiled in resolves like an unknown name, to `o200k_base`, so check that the encodings you name are enabled.

## Compared with text-splitter

| | chunkedrs | text-splitter 0.33 |
|---|---|---|
| Size unit | tokens (tiktoken) | characters by default; tokens through tiktoken-rs, tokenizers or your own `ChunkSizer` |
| Markdown | header ancestry on every chunk | CommonMark structure (pulldown-cmark) |
| Code | boundary markers, any language, no dependency | tree-sitter grammar per language |
| HTML | yes | no |
| Token spans in the document | yes | no |
| Embedding-based splitting | yes (`semantic`) | no |
| Capacity range (`min..max`), trimming | no | yes |
| Unicode segmentation (graphemes, words, sentences via ICU) | no | yes |

## Build

```bash
cargo test --workspace --all-targets
cargo clippy --workspace --all-targets --all-features -- -D warnings
CHUNKEDRS_PROPTEST_CASES=5000 cargo test --test properties   # longer property run
cargo run --release -p bench-compare                          # the tables above
```

## Related crates

[tiktoken](https://crates.io/crates/tiktoken) · [@goliapkg/tiktoken-wasm](https://www.npmjs.com/package/@goliapkg/tiktoken-wasm) · [instructors](https://crates.io/crates/instructors) · [embedrs](https://crates.io/crates/embedrs)

## License

Licensed under either of [MIT](LICENSE-MIT) or [Apache-2.0](LICENSE-APACHE), at your option.

The vocabularies compiled in through tiktoken's `vocab-*` features keep their publishers' licenses, some with conditions on use (Llama, Mistral, DeepSeek, Kimi, MiniMax). tiktoken's [LICENSE-3RD-PARTY](https://github.com/goliajp/rust-tiktoken/blob/develop/tiktoken/LICENSE-3RD-PARTY) lists them; a build without a vocabulary's feature does not contain that vocabulary.
