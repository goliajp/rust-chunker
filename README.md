# chunkedrs

[![Crates.io](https://img.shields.io/crates/v/chunkedrs?style=flat-square&logo=rust)](https://crates.io/crates/chunkedrs)
[![docs.rs](https://img.shields.io/docsrs/chunkedrs?style=flat-square&logo=docs.rs)](https://docs.rs/chunkedrs)
[![CI](https://img.shields.io/github/actions/workflow/status/goliajp/rust-chunker/ci.yml?branch=develop&style=flat-square&logo=github&label=ci)](https://github.com/goliajp/rust-chunker/actions/workflows/ci.yml)
[![License](https://img.shields.io/crates/l/chunkedrs?style=flat-square)](#license)

**English** | [简体中文](README.zh-CN.md) | [日本語](README.ja.md)

> **Upgrading from 2.x?** The API changed; the chunks did not. Everything is configured on `Chunker` now, chunks borrow their text from the document (`OwnedChunk` when you need to keep them), and an unknown model or encoding name is an error instead of a silent `o200k_base`. Recursive, markdown, code and HTML splitting give the same chunks as 2.2. Semantic splitting takes any `embedrs::Embed`. Migration table: [CHANGELOG](CHANGELOG.md#upgrading-from-2x).

chunkedrs splits long documents into chunks that fit a token budget, for embedding and retrieval. It counts tokens with [tiktoken](https://crates.io/crates/tiktoken) (20 encodings: OpenAI, Llama, DeepSeek, Qwen, Mistral, Kimi, GLM, MiniMax). Every chunk is a slice of the document with its exact byte range and its token range in the document's token stream. It has no dependency besides tiktoken; the `semantic` feature adds [embedrs](https://crates.io/crates/embedrs), which itself has none.

On an Apple M6, one thread, `o200k_base`, 512 tokens per chunk: a 200 KB English book splits in 0.13 ms, 1.5 times one `encode` of it. chunkedrs 3.1 took 0.14 ms, 3.0 0.16 ms and 2.2 0.25 ms; text-splitter 0.33.6 takes 0.93 ms on the same tokenizer and 24.6 ms with tiktoken-rs 0.12, the tokenizer its `tiktoken-rs` feature uses.

## Highlights

- **Five strategies**: recursive (paragraph, line, sentence, clause, word), markdown with header ancestry, code and HTML by boundary markers, and semantic splitting by embedding similarity.
- **Token budget held exactly**: `token_count` is counted on the chunk's own text and is at most `max_tokens`. The one exception is a single character that alone takes more tokens than the budget.
- **Borrowed, with offsets**: `content` is `&doc[start_byte..end_byte]`, no copy; `start_token..end_token` is the chunk's range in the document's token ids, which [late chunking](#late-chunking) needs.
- **Chinese and Japanese**: sentences end at `。！？` and clauses at `，、；：` with no space after them; a closing quote stays with its sentence.
- **Semantic splitting with any model**: the chunker hands out the sentences and takes the vectors back, so it works with a remote API, a local model or a cache, and needs no runtime.
- **Batches**: `split_batch` spreads a slice of documents over all cores.

## Quick start

```toml
[dependencies]
chunkedrs = "3"
```

```rust
// recursive, 512 tokens, no overlap, o200k_base
let chunks = chunkedrs::split("your long text here...");
for c in &chunks {
    println!("[{}] {} tokens, bytes {}..{}", c.index, c.token_count, c.start_byte, c.end_byte);
}

// a model name picks the encoding; an unknown name is an error
let chunker = chunkedrs::Chunker::for_model("gpt-5.6-terra")?
    .max_tokens(256)
    .overlap(32);
let chunks = chunker.split("your long text here...");
assert!(chunks.iter().all(|c| c.token_count <= 256));

// one configuration, many documents
let chunker = chunkedrs::Chunker::new().max_tokens(256).markdown();
let docs = ["# A\n\nFirst.", "# B\n\nSecond."];
let all = chunker.split_batch(&docs);
assert_eq!(all[1][0].section(), Some("# B"));

// chunks borrow the document; keep them past it as OwnedChunk
let kept: Vec<chunkedrs::OwnedChunk> = chunks.iter().map(|c| c.to_owned_chunk()).collect();
```

## Performance

Milliseconds per document, median of 9 rounds after warmup, then the median of three runs. Apple M6 Mac mini, rustc 1.100, release with LTO, `TIKTOKEN_THREADS=1`, `o200k_base`, no overlap, 2026-11-27; the load average stayed between 1.1 and 2.2. The columns of each row were measured one right after another. chunkedrs 3.1.0, 3.0.0 and 2.2.0 are the previous releases, on the same tiktoken 5.7. text-splitter gets the same `max_tokens` as its capacity and its defaults otherwise; its code splitter uses tree-sitter-rust 0.24.

| corpus | strategy | max_tokens | one `encode` | chunkedrs | chunkedrs 3.1.0 | chunkedrs 3.0.0 | chunkedrs 2.2.0 | text-splitter + tiktoken 5.7 | text-splitter + tiktoken-rs 0.12 |
|---|---|--:|--:|--:|--:|--:|--:|--:|--:|
| book_en.txt | recursive | 512 | 0.089 | **0.130** | 0.140 | 0.160 | 0.254 | 0.933 | 24.64 |
| book_en.txt | recursive | 128 | 0.092 | **0.164** | 0.182 | 0.202 | 0.300 | 1.981 | 46.01 |
| zh_xiyouji.md | recursive | 512 | 0.093 | **0.140** | 0.142 | 0.162 | 0.263 | 0.361 | 14.88 |
| zh_xiyouji.md | markdown | 512 | 0.093 | **0.147** | 0.156 | 0.176 | 0.283 | 0.800 | 30.03 |
| zh_xiyouji.md | recursive | 128 | 0.093 | **0.209** | 0.224 | 0.243 | 0.358 | 1.100 | 22.54 |
| zh_xiyouji.md | markdown | 128 | 0.093 | **0.220** | 0.243 | 0.261 | 0.392 | 1.512 | 32.42 |
| ja_neko.txt | recursive | 512 | 0.054 | **0.093** | 0.103 | 0.120 | 0.173 | 0.586 | 35.72 |
| ja_neko.txt | recursive | 128 | 0.054 | **0.157** | 0.179 | 0.197 | 0.243 | 1.542 | 49.89 |
| md_rust_book.md | recursive | 512 | 0.079 | **0.115** | 0.126 | 0.145 | 0.226 | 0.586 | 15.97 |
| md_rust_book.md | markdown | 512 | 0.079 | **0.145** | 0.160 | 0.183 | 0.299 | 0.921 | 18.21 |
| md_rust_book.md | recursive | 128 | 0.079 | **0.150** | 0.161 | 0.182 | 0.260 | 0.695 | 15.74 |
| md_rust_book.md | markdown | 128 | 0.079 | **0.187** | 0.212 | 0.234 | 0.360 | 1.366 | 24.14 |
| code_rust.txt | recursive | 512 | 0.048 | **0.075** | 0.087 | 0.099 | 0.149 | 0.256 | 8.41 |
| code_rust.txt | code | 512 | 0.048 | **0.076** | 0.093 | 0.104 | 0.155 | 4.866 | 24.65 |
| code_rust.txt | recursive | 128 | 0.048 | **0.107** | 0.123 | 0.136 | 0.178 | 0.390 | 10.96 |
| code_rust.txt | code | 128 | 0.048 | **0.115** | 0.149 | 0.161 | 0.203 | 5.696 | 44.34 |

`book_en.txt` is 200,000 bytes of *Pride and Prejudice*, `zh_xiyouji.md` 178,622 bytes of *Journey to the West* chapters 1-9 in markdown (Wikisource), `ja_neko.txt` the first 180,000 bytes of Natsume Sōseki's *I Am a Cat* (Aozora Bunko, ruby removed), `md_rust_book.md` 174,237 bytes of *The Rust Programming Language* chapters 3-6, `code_rust.txt` 100,000 bytes of Rust source. All are in [`bench-compare/data`](bench-compare/data).

- Every chunk of every row above was recounted with tiktoken: none is over budget, for chunkedrs or text-splitter.
- At 512 tokens chunkedrs takes 1.5 to 1.8 times one `encode` of the document, whatever the strategy, and 1.8 to 2.9 times at 128. It encodes the document once, with each token's end offset from the encoder, and counts each chunk by reusing the document's ids between the chunk's first and last tiktoken cut position, encoding only the two ends again; finding boundaries and building chunks is the rest.
- Against 3.1.0: 1 to 23% less time; least on Chinese with the recursive strategy at 512 tokens (1%), whose time is mostly the encode itself, most on code. Against 2.2.0: 35 to 52% less. All four give the same chunks.
- `cl100k_base` gives a similar picture: 1.4 to 1.9 times one `encode` at 512 tokens (3.1.0: up to 2.07 times, on Japanese and on Markdown with the markdown strategy) and 2 to 24% less time than 3.1.0. With tiktoken's default threads, the whole-document encode of these texts runs on several threads and chunkedrs takes 0.06 to 0.18 ms (3.1.0: 0.07 to 0.21 ms).
- `Chunker::split_batch` on 80 documents (the five corpora 16 times, 13.3 MB): 7.4 ms one after another, 2.3 ms as a batch.

Reproduce with `TIKTOKEN_THREADS=1 cargo run --release -p bench-compare`.

## How it splits

The document is encoded once. From the start of a chunk, the byte offset after `max_tokens` document tokens bounds where the chunk can end. Inside that window the chunk ends at the last boundary of the highest-priority kind present:

| Strategy | Boundaries, highest priority first |
|---|---|
| recursive (default) | blank line; line break; `. ! ?` followed by a space, `。！？．｡…‥`; `; , :` and `；，、：･`; space, tab, ideographic space |
| `markdown()` | header lines first (ATX `#` and setext `===` / `---`, not inside fenced code or front matter), then the recursive boundaries inside each section |
| `code()` | blank line; `}` `};` `)` `]` `];` `end` alone on a line at column 0; line break; space, tab |
| `html()` | after a block-level closing tag (`</p>`, `</li>`, `</section>`, `</h1>`, ...), then the recursive boundaries |
| `semantic()` | between sentence groups whose embeddings stop being similar, then the recursive boundaries |

When no boundary falls inside the window, the chunk ends at a token boundary, moved back to a character boundary. The chosen text is then counted as if it were encoded on its own: BPE merges across a cut, so a slice can take one token more or less than its share of the document's tokens. That count reuses the document's ids between the first and the last position where tiktoken can cut the chunk without changing its tokens, and encodes only the two ends again. In the rare case the chunk is over budget, the window shrinks and the chunk is chosen again.

Code and HTML splitting read punctuation and tags, not syntax: they parse nothing and add no dependency, and a `}` inside a string literal looks like a block close to them. For syntax-aware code splitting, use text-splitter's tree-sitter `CodeSplitter`.

## Markdown

```rust
let md = "# Guide\n\nIntro.\n\n## Install\n\nRun cargo add.\n";
let chunks = chunkedrs::Chunker::new().markdown().split(md);

assert_eq!(chunks[0].section(), Some("# Guide"));
assert_eq!(chunks[1].section_path, ["# Guide", "## Install"]);
```

Header lines go into `section_path`, not into `content`, so joining every chunk's content does not reproduce them. To embed a chunk with its headings, put them in front yourself: `format!("{}\n\n{}", c.section_path.join(" > "), c.content)`.

## Chinese and Japanese

```rust
let zh = "他说「今天天气很好。」然后就出门了。她回答说「确实不错。」于是也跟着出去了。";
let chunks = chunkedrs::Chunker::new().max_tokens(12).split(zh);

assert_eq!(chunks[0].content, "他说「今天天气很好。」");
assert_eq!(chunks[1].content, "然后就出门了。");
```

A splitter whose separators all end in an ASCII space finds no sentence boundary in this text and cuts inside a word.

## Late chunking

[Late chunking](https://arxiv.org/abs/2409.04701) embeds the whole document once and pools each chunk's vector over the chunk's own tokens. `split_with_tokens` returns the document's ids with the chunks, and `start_token..end_token` indexes them:

```rust
let doc = "First sentence here. Second sentence here. Third sentence here.";
let (chunks, ids) = chunkedrs::Chunker::new().max_tokens(8).split_with_tokens(doc);
for chunk in &chunks {
    // in a real pipeline this indexes the model's per-token hidden states
    let span = &ids[chunk.start_token..chunk.end_token];
    assert!(!span.is_empty());
}
```

Where a boundary falls inside a token (`". "` before a word, where the tokenizer joins the space to the word), the span covers that token, so two consecutive spans can share one token. `token_count` can therefore differ from `end_token - start_token`. See [`examples/token_spans.rs`](examples/token_spans.rs).

## Semantic splitting

`Chunker::semantic(text)` cuts the text into sentences, none longer than `max_tokens`, and hands them out as `inputs()`, slices of the text. Embed them with anything and give the vectors back; the chunker breaks where neighbouring sentences stop being similar and splits each group within the budget. No feature, no dependency, no I/O:

```rust
let text = "Cats purr. Cats nap. Stocks fell. Markets slid.";
let job = chunkedrs::Chunker::new().semantic(text);
let vectors: Vec<f32> = my_model(job.inputs());   // one vector per input, back to back
let chunks = job.finish(&vectors, 384)?;
```

With the `semantic` feature, any `embedrs::Embed` does the embedding: a remote provider's `Client` (requests sized to the provider and sent concurrently), a local model on the CPU, or your own type.

```toml
chunkedrs = { version = "3", features = ["semantic"] }
embedrs = { version = "0.7", features = ["local"] }
```

```rust,ignore
let client = embedrs::local();   // all-MiniLM-L6-v2, downloaded once
let chunks = chunkedrs::Chunker::new()
    .max_tokens(256)
    .split_semantic(text, &client)
    .await?;

// many documents, one embedding call for all their sentences
let all = chunker.split_semantic_batch(&docs, &client).await?;
```

By default a group ends where the cosine distance between neighbours is above the 95th percentile of the document's distances (`Breakpoints::Percentile(95.0)`), which adapts to the model. `semantic(text).breakpoints(Breakpoints::Threshold(0.5))` breaks at a fixed similarity instead. The bundled local models are English models; for Chinese or Japanese use a multilingual embedding model.

## Chunk

```rust
#[non_exhaustive]
pub struct Chunk<'a> {
    pub content: &'a str,          // &doc[start_byte..end_byte]
    pub index: usize,              // position in the sequence
    pub start_byte: usize,
    pub end_byte: usize,           // exclusive
    pub start_token: usize,        // range in the document's ids
    pub end_token: usize,          // exclusive
    pub token_count: usize,        // tokens of content counted on its own
    pub section_path: Vec<&'a str>, // markdown header ancestry, outermost first
}
```

Chunks that contain only whitespace are dropped. `to_owned_chunk()` gives an `OwnedChunk` with the same fields as `String`s, for storing chunks or sending them to another thread after the document is gone.

Overlap (`.overlap(n)`) starts each chunk about `n` tokens before the previous one ended, at a character boundary, and is off by default. Whether it helps retrieval depends on the workload; measure before you turn it on.

## Encodings and features

`Chunker::for_model("...")` resolves model names across OpenAI, Meta, DeepSeek, Qwen, Mistral, Moonshot, Zhipu and MiniMax; `Chunker::for_encoding("cl100k_base")` names an encoding directly. Both return `Error::UnknownModel` / `Error::UnknownEncoding` for a name tiktoken does not know or whose vocabulary is not compiled in. Anthropic and Google publish no tiktoken-compatible vocabulary; `Chunker::new()` (`o200k_base`) approximates their counts.

Vocabularies are opt-out features, as in tiktoken. The default compiles in all 20 encodings; `default-features = false` keeps only `o200k_base`, the encoding `Chunker::new()` uses.

```toml
chunkedrs = "3"                                                   # all vocabularies
chunkedrs = { version = "3", default-features = false }           # o200k_base only
chunkedrs = { version = "3", default-features = false, features = ["vocab-qwen"] }
```

A release build of a program that picks its encoding by name at run time (macOS arm64): 9,055,648 bytes with all vocabularies, 1,938,480 bytes with `default-features = false` (79% smaller).

Vendor groups: `vocab-openai`, `vocab-meta`, `vocab-deepseek`, `vocab-qwen`, `vocab-mistral`, `vocab-moonshot`, `vocab-zhipu`, `vocab-minimax`. Single vocabularies: `vocab-cl100k_base`, `vocab-llama4`, `vocab-qwen3_5`, `vocab-mistral_v15`, `vocab-glm5` and the rest of tiktoken's list.

## Compared with text-splitter

| | chunkedrs | text-splitter 0.33 |
|---|---|---|
| Size unit | tokens (tiktoken) | characters by default; tokens through tiktoken-rs, tokenizers or your own `ChunkSizer` |
| Markdown | header ancestry on every chunk | CommonMark structure (pulldown-cmark) |
| Code | boundary markers, any language, no dependency | tree-sitter grammar per language |
| HTML | yes | no |
| Token spans in the document | yes | no |
| Embedding-based splitting | yes, any embedder | no |
| Capacity range (`min..max`), trimming | no | yes |
| Unicode segmentation (graphemes, words, sentences via ICU) | no | yes |

## Build

```bash
cargo test --workspace --all-targets
cargo test --all-features             # semantic tests run all-MiniLM-L6-v2 on the CPU (91 MB, downloaded once)
cargo clippy --workspace --all-targets --all-features -- -D warnings
CHUNKEDRS_PROPTEST_CASES=5000 cargo test --test properties   # longer property run
cargo run --release -p bench-compare                          # the tables above
```

## Related crates

[tiktoken](https://crates.io/crates/tiktoken) · [@goliapkg/tiktoken-wasm](https://www.npmjs.com/package/@goliapkg/tiktoken-wasm) · [instructors](https://crates.io/crates/instructors) · [embedrs](https://crates.io/crates/embedrs)

## License

Licensed under either of [MIT](LICENSE-MIT) or [Apache-2.0](LICENSE-APACHE), at your option.

The vocabularies compiled in through tiktoken's `vocab-*` features keep their publishers' licenses, some with conditions on use (Llama, Mistral, DeepSeek, Kimi, MiniMax). tiktoken's [LICENSE-3RD-PARTY](https://github.com/goliajp/rust-tiktoken/blob/develop/tiktoken/LICENSE-3RD-PARTY) lists them; a build without a vocabulary's feature does not contain that vocabulary.
