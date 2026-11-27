# chunkedrs

[![Crates.io](https://img.shields.io/crates/v/chunkedrs?style=flat-square&logo=rust)](https://crates.io/crates/chunkedrs)
[![docs.rs](https://img.shields.io/docsrs/chunkedrs?style=flat-square&logo=docs.rs)](https://docs.rs/chunkedrs)
[![CI](https://img.shields.io/github/actions/workflow/status/goliajp/rust-chunker/ci.yml?branch=develop&style=flat-square&logo=github&label=ci)](https://github.com/goliajp/rust-chunker/actions/workflows/ci.yml)
[![License](https://img.shields.io/crates/l/chunkedrs?style=flat-square)](#license)

**English** | [简体中文](README.zh-CN.md) | [日本語](README.ja.md)

> **Upgrading from 2.x?** The API changed; the chunks did not. Everything is configured on `Chunker` now, chunks borrow their text from the document (`OwnedChunk` when you need to keep them), and an unknown model or encoding name is an error instead of a silent `o200k_base`. Recursive, markdown, code and HTML splitting give the same chunks as 2.2. Semantic splitting takes any `embedrs::Embed`. Migration table: [CHANGELOG](CHANGELOG.md#upgrading-from-2x).

chunkedrs splits long documents into chunks that fit a token budget, for embedding and retrieval. It counts tokens with [tiktoken](https://crates.io/crates/tiktoken) (20 encodings: OpenAI, Llama, DeepSeek, Qwen, Mistral, Kimi, GLM, MiniMax). Every chunk is a slice of the document with its exact byte range and its token range in the document's token stream. It has no dependency besides tiktoken; the `semantic` feature adds [embedrs](https://crates.io/crates/embedrs), which itself has none.

On an Apple M4 Pro, one thread, `o200k_base`, 512 tokens per chunk: a 200 KB English book splits in 0.19 ms, 1.6 times one `encode` of it. chunkedrs 3.0 took 0.23 ms and 2.2 took 0.35 ms; text-splitter 0.33.6 takes 1.19 ms on the same tokenizer and 34.8 ms with tiktoken-rs 0.12, the tokenizer its `tiktoken-rs` feature uses.

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

Milliseconds per document, median of 9 rounds after warmup. Apple M4 Pro Mac mini, rustc 1.100, release with LTO, `TIKTOKEN_THREADS=1`, `o200k_base`, no overlap, 2026-11-27. Other jobs shared the machine (load average 1.2 to 5.8 during the run), so the columns of each row were measured one right after another. chunkedrs 3.0.0 and 2.2.0 are the two previous releases, on the same tiktoken 5.7. text-splitter gets the same `max_tokens` as its capacity and its defaults otherwise; its code splitter uses tree-sitter-rust 0.24.

| corpus | strategy | max_tokens | one `encode` | chunkedrs 3.1 | chunkedrs 3.0.0 | chunkedrs 2.2.0 | text-splitter + tiktoken 5.7 | text-splitter + tiktoken-rs 0.12 |
|---|---|--:|--:|--:|--:|--:|--:|--:|
| book_en.txt | recursive | 512 | 0.117 | **0.189** | 0.226 | 0.354 | 1.187 | 34.80 |
| book_en.txt | recursive | 128 | 0.116 | **0.251** | 0.308 | 0.428 | 2.456 | 64.88 |
| zh_xiyouji.md | recursive | 512 | 0.149 | **0.233** | 0.269 | 0.453 | 0.495 | 20.06 |
| zh_xiyouji.md | markdown | 512 | 0.149 | **0.253** | 0.286 | 0.517 | 1.035 | 40.27 |
| zh_xiyouji.md | recursive | 128 | 0.146 | **0.345** | 0.389 | 0.622 | 1.433 | 29.88 |
| zh_xiyouji.md | markdown | 128 | 0.146 | **0.373** | 0.417 | 0.629 | 1.980 | 42.86 |
| ja_neko.txt | recursive | 512 | 0.076 | **0.148** | 0.184 | 0.271 | 0.762 | 46.18 |
| ja_neko.txt | recursive | 128 | 0.078 | **0.276** | 0.307 | 0.392 | 2.017 | 65.24 |
| md_rust_book.md | recursive | 512 | 0.101 | **0.169** | 0.206 | 0.312 | 0.735 | 22.20 |
| md_rust_book.md | markdown | 512 | 0.101 | **0.222** | 0.257 | 0.416 | 1.188 | 25.19 |
| md_rust_book.md | recursive | 128 | 0.101 | **0.218** | 0.266 | 0.361 | 0.920 | 21.74 |
| md_rust_book.md | markdown | 128 | 0.101 | **0.296** | 0.336 | 0.514 | 1.773 | 33.53 |
| code_rust.txt | recursive | 512 | 0.060 | **0.113** | 0.137 | 0.202 | 0.321 | 11.57 |
| code_rust.txt | code | 512 | 0.060 | **0.120** | 0.144 | 0.207 | 6.769 | 34.45 |
| code_rust.txt | recursive | 128 | 0.060 | **0.162** | 0.187 | 0.244 | 0.510 | 15.15 |
| code_rust.txt | code | 128 | 0.060 | **0.202** | 0.232 | 0.288 | 7.707 | 61.53 |

`book_en.txt` is 200,000 bytes of *Pride and Prejudice*, `zh_xiyouji.md` 178,622 bytes of *Journey to the West* chapters 1-9 in markdown (Wikisource), `ja_neko.txt` the first 180,000 bytes of Natsume Sōseki's *I Am a Cat* (Aozora Bunko, ruby removed), `md_rust_book.md` 174,237 bytes of *The Rust Programming Language* chapters 3-6, `code_rust.txt` 100,000 bytes of Rust source. All are in [`bench-compare/data`](bench-compare/data).

- Every chunk of every row above was recounted with tiktoken: none is over budget, for chunkedrs or text-splitter.
- chunkedrs takes 1.6 to 2.0 times one `encode` of the document with the recursive strategy at 512 tokens, and up to 3.5 times at 128. It encodes the document once, with each token's end offset from the encoder, and counts each chunk by reusing the document's ids between the chunk's first and last tiktoken cut position, encoding only the two ends again; finding boundaries and building chunks is the rest.
- Against 3.0.0: 10 to 20% less time on every row; against 2.2.0: 30 to 51% less. All three give the same chunks.
- `cl100k_base` gives a similar picture: 0.11 to 0.48 ms, up to 22% less than 3.0.0. With tiktoken's default threads, the whole-document encode of these texts runs on several threads and chunkedrs takes 0.08 to 0.24 ms (3.0.0: 0.11 to 0.30 ms).
- The Chinese rows vary the most between runs of the same build. In five more runs of only that corpus, the recursive strategy at 128 tokens took 15% less to 9% more time than 3.0.0 with `o200k_base` (median 2% less) and 0.4% to 19% less with `cl100k_base` (median 8% less); at 512 tokens 0.4% to 15% less with `o200k_base` (median 6% less).
- `Chunker::split_batch` on 80 documents (the five corpora 16 times, 13.3 MB): 18.2 ms one after another, 2.5 ms as a batch.

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
