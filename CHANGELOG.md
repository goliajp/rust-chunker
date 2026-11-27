# Changelog

All notable changes to this crate will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/)
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [3.1.0] - 2026-11-27

### Changed

- Faster, with the same chunks. The document's token ends come from
  tiktoken 5.7's `encode_with_ends_into`, which the encoder fills as it
  goes, instead of a second pass over the ids that looks up each token's
  length; markdown sections are found by scanning the text itself instead
  of a vector of its lines. On an Apple M4 Pro, one thread, `o200k_base`:
  10 to 20% less time than 3.0.0 on every row of the README table, and the
  recursive strategy at 512 tokens takes 1.6 to 2.0 times one `encode` of
  the document (3.0.0: 1.8 to 2.4 times), for example 200 KB of English
  0.226 ms to 0.189 ms and 180 KB of Japanese 0.184 ms to 0.148 ms.
- Requires tiktoken 5.7.

### Added

- A test that compares every strategy with 3.0.0: byte ranges, token
  spans, token counts and section paths on the five `bench-compare`
  corpora (budgets 8 to 512, with and without overlap) and 2,000 generated
  documents, for each compiled encoding among `o200k_base`, `cl100k_base`,
  `qwen3_5`, `llama3` and `glm5`.

## [3.0.0] - 2026-11-27

### Upgrading from 2.x

3.0 changes the API; the chunks themselves do not change. Recursive,
markdown, code and HTML splitting give the same byte ranges, token spans,
token counts and section paths as 2.2 for the same encoding, budget and
overlap (a test compares the two on every strategy, five encodings and
2,000 generated documents). Semantic splitting changed; see below.

| 2.x | 3.0 |
|---|---|
| `chunkedrs::chunk(t).split()` | `chunkedrs::split(t)` |
| `chunk(t).max_tokens(n).overlap(o).split()` | `Chunker::new().max_tokens(n).overlap(o).split(t)` |
| `chunk(t).model(m)`: an unknown name fell back to `o200k_base` | `Chunker::for_model(m)?`: an unknown name, or one whose vocabulary feature is off, is `Error::UnknownModel`; `.unwrap_or_else(\|_\| Chunker::new())` keeps the old fallback |
| `chunk(t).encoding(e)` | `Chunker::for_encoding(e)?` or `Chunker::with_encoder(enc)` |
| `chunk(t).markdown()` / `.code()` / `.html()` | the same methods on `Chunker` |
| `Chunker::model(m)` / `Chunker::encoding(e)` | `Chunker::for_model(m)?` / `Chunker::for_encoding(e)?` |
| `Chunk` owns its text | `Chunk<'a>` borrows it: `content: &'a str`, `section_path: Vec<&'a str>`; `chunk.to_owned_chunk()` gives an `OwnedChunk` with the 2.x field types |
| `Chunk::new(..).with_bytes(..)` | removed; `OwnedChunk::default()` and set the fields |
| encode the document again for late chunking | `Chunker::split_with_tokens(t)` returns the ids with the chunks |
| `chunk(t).semantic(&client).threshold(0.5).split_async().await?` | `Chunker::new().semantic(t).breakpoints(Breakpoints::Threshold(0.5)).embed(&client).await?` |
| embedrs 0.6 `Client` | anything that implements `embedrs::Embed` 0.7 |

`ChunkBuilder` and `SemanticChunkBuilder` are gone.

### Added

- `Chunk<'a>` borrows from the text: no copy of each chunk's text or
  header path. `OwnedChunk` and `Chunk::to_owned_chunk` for chunks that
  outlive the text; `Chunk::byte_range` and `Chunk::token_range`.
- `chunkedrs::split(text)` for the defaults.
- `Chunker::for_model`, `Chunker::for_encoding` and `Chunker::with_encoder`;
  `Error::UnknownModel` and `Error::UnknownEncoding`.
- `Chunker::split_with_tokens`: the chunks and the document's token ids.
- Semantic splitting without I/O: `Chunker::semantic(text)` gives the
  sentences to embed (`Semantic::inputs`, slices of the text, none longer
  than `max_tokens`); `Semantic::finish(vectors, dim)` takes the vectors of
  any model back. No feature and no dependency needed.
- Feature `semantic`: `Semantic::embed`, `Chunker::split_semantic` and
  `Chunker::split_semantic_batch` over any `embedrs::Embed`, remote or
  local. The batch form makes one embedding call for every text's
  sentences.
- `Breakpoints::Percentile(p)` (the default, 95) and
  `Breakpoints::Threshold(t)`.
- A Japanese corpus in `bench-compare` and a test there that compares
  every strategy with 2.2.0.

### Changed

- Faster: each chunk's count reuses the document's token ids between the
  first and the last position where tiktoken 5.6 can cut the chunk
  (`CoreBpe::next_cut`, `prev_cut`) and encodes only the two ends again.
  The counts are the same as encoding the whole chunk. On an Apple M4 Pro,
  one thread, `o200k_base`: 15 to 39% less time than 2.2 on every row of
  the README table, a 200 KB English book at 512 tokens 0.37 ms to 0.23 ms,
  180 KB of Japanese 0.27 ms to 0.20 ms. Separator matchers are built once
  per split instead of once per markdown section.
- Requires tiktoken 5.6. The `semantic` feature depends on embedrs 0.7
  with its default features off, which has no dependencies.
- Semantic splitting breaks at the 95th percentile of the document's
  neighbour distances by default instead of at a fixed cosine similarity of
  0.5, whose meaning depends on the model. A sentence longer than
  `max_tokens` is cut before it is embedded, so no input is longer than a
  chunk.
- Authors and copyright holder: GOLIA K.K. and DoraCawl.

### Removed

- `chunk()`, `ChunkBuilder`, `SemanticChunkBuilder`, `Chunk::new` and the
  `Chunk::with_*` setters.

## [2.2.0] - 2026-11-25

### Upgrading from 2.1

No API change; code that builds against 2.1 builds against this release.
Chunk boundaries move, so re-chunk and re-embed stored indexes:

- **tiktoken 5.5.** Token counts change for the model names tiktoken 5.0
  moved to their own vocabularies: `llama-4*` and `muse-*` now use `llama4`,
  `qwen3.5*` to `qwen3.8*` and `qwen4*` use `qwen3_5`, `mistral-large-4*` and
  `mistral-large-latest` use `mistral_v15`, `deepseek-flash` uses
  `deepseek_v4`. `gpt-6*` resolves to `o200k_base`. 2.1 resolved these names to
  older vocabularies or to the `o200k_base` fallback.
- **New planner.** A chunk ends at the last boundary of the highest-priority
  kind that fits, measured on one encode of the whole document. Boundaries
  are the same as 2.1 on most text and differ where 2.1 summed the token
  counts of separate pieces.
- **Overlap.** Applies between every pair of consecutive chunks of a section
  and never repeats or rewinds a chunk (2.1 could emit a chunk made only of
  the previous chunk's tail, or start a chunk before the previous one).
- **Threads.** tiktoken 5.2 and later encode a text of 32 KiB or more on
  several threads by default. chunkedrs encodes each document once, so
  documents of that size use them. `TIKTOKEN_THREADS=1` or
  `tiktoken::set_parallelism(1)` keeps tiktoken on the calling thread.

### Added

- `Chunker`: a configuration with its encoder resolved once, reusable across
  texts; `Chunker::split_batch` splits a slice of texts on all cores with
  `std::thread::scope` and returns the results in input order.
- Features `vocab-llama4`, `vocab-qwen3_5` and `vocab-mistral_v15`, forwarded
  to tiktoken.
- `bench-compare/`, a harness that times chunkedrs against text-splitter on
  four public corpora (not published).

### Changed

- Faster. On an Apple M4 Pro, one thread, `o200k_base`, 512 tokens per
  chunk, compared with the 2.1 code on tiktoken 5.4: a 200 KB English book
  10.9 ms to 0.35 ms, 179 KB of Chinese 4.2 ms to 0.42 ms (markdown 2.2 ms to
  0.48 ms), 100 KB of Rust source 2.0 ms to 0.20 ms. The document is encoded
  once, each chunk is counted once, separators are found without rescanning,
  and token lengths come from tiktoken's `token_bytes` instead of decoding
  every token. The README has the full table against text-splitter.
- Requires tiktoken 5.5.
- License: `MIT OR Apache-2.0` (was `MIT`).

### Fixed

- Token-level cuts held the budget only per token window: a character that
  spans several tokens could push a chunk over `max_tokens`. A chunk is now
  over budget only when it is a single character that alone needs more
  tokens than `max_tokens`.
- Semantic splitting: a whitespace-only sentence inside a group shifted the
  byte offsets of the chunks after it, so `start_byte..end_byte` no longer
  addressed the source.
- Semantic splitting sent all sentences in one embedding request, which failed
  with `InputTooLarge` once a document had more sentences than the provider
  accepts per request (96 for Cohere, 2048 for OpenAI). Requests are now split
  to the provider's batch size.

## Upgrading from 1.x to 2.0: migration

Two mechanical renames and one API shape change.

```rust
// 1.x
chunk.section.as_deref()          // Option<&str>
Chunk { content, index, .. }      // struct literal

// 2.0
chunk.section()                   // Option<&str> — same value, now a method
chunk.section_path                // Vec<String> — the full header ancestry
Chunk::new(content).with_index(i) // Chunk is now #[non_exhaustive]
```

`.semantic(&client)` returns a `SemanticChunkBuilder`. If you were calling
`.split()` on a semantic builder, that was a runtime panic in 1.x; it is now a
compile error, and `.split_async()` is the method you want.

If you are coming from 1.0.x rather than 1.1.0, read the 1.1.0 note below as
well — chunk boundaries and token counts both moved there.

## [2.1.1] - 2026-08-13

Documentation only. No code changes.

### Fixed

- **Every semantic-splitting example called a constructor that does not
  exist.** `embedrs::openai("sk-...")` is not an item in embedrs — the
  constructor is `embedrs::Client::openai(...)`. Five occurrences: the module
  docs, the `semantic()` doc, and all three READMEs. The blocks are marked
  `rust,ignore`, so no doctest ever compiled them.
- "covering 10 providers" is now 11, after tiktoken 4.1.1 added Voyage.

## [2.1.0] - 2026-08-13

### Changed

- **`embedrs` floor raised to 0.6** (the `semantic` feature). 0.6 is the
  release that moved to tiktoken 4.1.1, so an older one would pull a second
  4.x into a `semantic` + `cost-tracking` build. Note 0.6 also changes the
  Voyage default to `voyage-4-large`.
- **`tiktoken` floor raised to 4.1.1.** Nothing here needs it — this crate uses
  the encodings, not the price table. It is the floor the rest of the family is
  on, and a shared floor keeps a `semantic` build (which pulls embedrs) from
  resolving two different 4.x.
- Documentation examples name a current model (`gpt-5.6-terra`) rather than
  `gpt-4o`. No behavior change — `gpt-4o` still resolves, and the tests that
  exercise the `gpt-4o` → `o200k_base` mapping still name it deliberately.

### Testing

- `advertised_models_resolve` pins the 2026 GPT-5.x SKUs (`gpt-5.6-sol` /
  `-terra` / `-luna`, `gpt-5.5`, `gpt-5.4`, `o4-mini`). They resolve through
  the `gpt-5` prefix rather than entries of their own, which is by design but
  is exactly the kind of thing that silently stops holding — and `.model()`
  falls back to `o200k_base` without an error, so the failure mode is wrong
  token counts rather than a crash.

## [2.0.0] - 2026-08-13

### Added

- **Vocabulary features, forwarded from tiktoken 4.** Tokenizer vocabularies
  are the bulk of the compiled size and most builds use one of them. They are
  now opt-out: `default` carries all 17 encodings as before, and
  `default-features = false` keeps only `o200k_base`.

  ```toml
  chunkedrs = { version = "2", default-features = false, features = ["vocab-zhipu"] }
  ```

  Measured on `examples/basic`, release: **7,100,544 → 2,695,104 bytes**, −62%.

  `vocab-o200k_base` is enabled unconditionally by the dependency declaration
  rather than being one of the optional features. It is the encoder every
  unresolved name falls back to, so `resolve_encoder` cannot fail to find one —
  that is now a structural guarantee instead of an assumption, and
  `fallback_encoding_is_always_available` pins it.

  One sharp edge, documented in the READMEs and on `.encoding()`: a vocabulary
  that was not compiled in is indistinguishable from a typo — both fall back to
  `o200k_base` silently. CI gained a `feature-matrix` job that runs the tests
  across four vocabulary configurations, because a slim build fails at run time
  rather than at compile time and only tests catch it.

- **Token spans on every chunk** — `start_token` and `end_token` give the
  chunk's range in the *document's* token stream. This is the entry ticket for
  [late chunking](https://arxiv.org/abs/2409.04701): embed the document once,
  then pool each chunk's vector over its own range, so every chunk embedding
  carries whole-document context instead of only its own text. The splitter
  already computed these boundaries internally and discarded them.

  They are deliberately not the same number as `token_count`. `token_count` is
  a fresh count of the chunk's own text — what `max_tokens` bounds and what you
  budget a context window against. The span is the chunk's footprint in the
  document stream. BPE merges across boundaries, and where a separator sits
  inside a token the span widens to cover it, so consecutive spans can share
  exactly one token. Both numbers are real and they answer different questions.

- **Section hierarchy** — `section_path: Vec<String>` carries the full header
  ancestry, so a `### From source` chunk knows it sits under `## Installation`
  under `# Guide`. Previously a chunk knew only the one header line above it.
  Setext headers carry levels too (`===` is 1, `---` is 2).

- **`.code()`** — splits source in any language on blank lines, then on
  brackets closing a block at column 0, then on lines. The prose separators are
  deliberately skipped below the line tier: splitting code on `". "` cuts
  inside string literals and doc comments.

- **`.html()`** — splits after block-level closing tags (`</p>`, `</li>`,
  `</section>`, headings, table rows, …), matched case-insensitively. Inline
  tags are not boundaries, so a sentence is not cut mid-phrase.

  Both are **boundary-aware, not AST/DOM-aware**, and both add zero
  dependencies. Not pulling in tree-sitter is a decision, not an oversight: it
  would mean a grammar crate per language in a crate whose entire dependency
  surface is `tiktoken`. Where AST fidelity matters, the docs point at
  tree-sitter based splitters by name.

- **`Chunk::new` and `with_*` setters**, so downstream can still construct a
  `Chunk` now that it is `#[non_exhaustive]` — for tests, and for adapters that
  re-materialise chunks from storage.

### Changed — breaking

- **`Chunk` is `#[non_exhaustive]`.** Adding a metadata field used to be a
  major version every single time; now it is not. This is the last such break.
- **`section: Option<String>` → `section_path: Vec<String>`**, with a
  `section()` method returning the deepest entry. One source of truth plus an
  ergonomic accessor, rather than two fields that can disagree.
- **`.semantic()` returns `SemanticChunkBuilder`**, which has no `.split()`.
  Semantic splitting makes network calls, so the synchronous entry point does
  not exist on it rather than existing and panicking. `.threshold()` moves to
  this type; `.max_tokens()`, `.overlap()`, `.model()` and `.encoding()` are
  available on both.
- **Whitespace-only chunks are dropped.** A separator tier could strand a lone
  `"\n"` between two oversized pieces; that chunk would then be embedded,
  stored, and never usefully retrieved. Splitting is therefore not
  byte-for-byte reconstructible — it already was not, since markdown keeps
  header lines in metadata — but every surviving chunk's byte range still
  addresses the source exactly.

### Changed — dependencies

The floors themselves (`tiktoken` `"4"`, `embedrs` `"0.5"`) arrive in 1.1.0 —
both lines are on the current majors. What 2.0 changes is *how* tiktoken is
depended on: `default-features = false` plus the `vocab-*` passthrough above,
so callers can choose their vocabularies. 1.x keeps the plain `"4"` and
therefore all of them.

With chunkedrs and embedrs both on tiktoken 4, a project enabling
`chunkedrs/semantic` and `embedrs/cost-tracking` together resolves to a single
tiktoken — verified with a scratch consumer depending on both.

### Changed — internal

- The recursive descent's shared state moves into a `Ctx`, retiring the
  `too_many_arguments` allow and letting each strategy supply its own separator
  ladder. Chunk numbering and the token-span pass are now done once at the top
  level rather than in each strategy.

### Docs

- New `examples/token_spans.rs` (late chunking) and `examples/code_and_html.rs`.
- The comparison table now names tree-sitter code splitting as something
  text-splitter does and chunkedrs does not.

## Upgrading to 1.1.0 — chunk boundaries change

1.1.0 adds no API, but it changes **where the cuts land and what
`token_count` reports**, for three independent reasons:

- the tokenizer floor moved to 4, which corrects token counts for any text
  containing a blank line (see below);
- CJK text now splits on sentence and clause marks instead of falling through
  to token offsets;
- markdown documents containing fenced code blocks no longer grow phantom
  sections.

All three are corrections — the previous output was wrong — but any stored
chunk ids, cached token counts, or embeddings keyed to chunk boundaries should
be recomputed after upgrading. Byte offsets remain exact against the source
text, so re-chunking and re-embedding is the whole of the migration.

## [1.1.0] - 2026-08-13

### Fixed

- **Token counts were wrong for any text containing a blank line.** The
  `tiktoken` floor was `"3.1"` and resolved to 3.5.1, which predates the
  3.6.0 fix for a pre-tokenizer bug that split canonical newline runs into
  single-newline tokens. `"\n\n"` is this crate's highest-priority separator,
  so every paragraph break in every document was overcounted by one token —
  directly undermining the token-accuracy guarantee that is the crate's
  headline claim. Measured on the previously locked version:

  ```text
  "word\n\nnext"              -> 4 tokens, ids [1801, 198, 198, 7311]
  "alpha\n\nbravo\n\ncharlie" -> 9 tokens (canonical: 7)
  ```

  The floor is now `"4"`, which carries that fix plus 3.8.0's fix to the o200k
  punctuation rule. `tests/tokenizer_contract.rs` pins both so a downgrade
  cannot silently reintroduce them.

  tiktoken 4.0.0 changed no token ids, no encoding behaviour and no function
  signatures — its major is entirely about vocabularies moving behind
  default-on features, which this release does not opt out of. So it carries
  every encoding exactly as before, and no chunk boundary moves because of it.
  (2.0 forwards those features so callers can slim the build; on the 1.x line
  the dependency is a plain `"4"`.)

- **Chinese and Japanese were split mid-word.** Every entry in the separator
  ladder ended in an ASCII space (`". "`, `"! "`, `", "`, `" "`). CJK scripts
  do not use one, so CJK prose matched nothing in any tier and fell straight
  through to the token-level fallback:

  ```text
  before, max_tokens=12:
    [0] "这是第一句话，讲的是分块。这是第二句话"
    [1] "，讲的是检索。这是第三句话，讲的是"      <- opens on a comma
    [2] "嵌入。这是第四句话，讲的是排序"
    [3] "。"                                      <- a lone full stop

  after:
    [0] "这是第一句话，讲的是分块。"
    [1] "这是第二句话，讲的是检索。"
  ```

  The existing CJK tests passed throughout, because they only asserted
  `token_count <= max` — a bar any splitter clears, including one that cuts
  mid-word. `tests/cjk_boundaries.rs` asserts where the cuts land instead.

- **`#` comments inside fenced code blocks became section headers.** Header
  detection was a bare per-line `#` check with no document model, so a
  `# install the tool` line in a `bash` block became an H1, taking both the
  section metadata and the split point with it. Since `#` opens a comment in
  shell, python, ruby, yaml and toml, this affected most technical documents.

- **Token-level splitting could emit `U+FFFD`.** The last-resort splitter
  sliced at raw token boundaries; a character spanning several tokens put the
  slice mid-codepoint, which fell back to a lossy decode. Boundaries now snap
  back to character boundaries — which also keeps each slice inside its token
  window, preserving the `max_tokens` guarantee — and `token_count` is
  re-counted from the content rather than assumed.

### Added

- **CJK sentence and clause separators**: `。`, `！`, `？`, `．`, `｡`, `…`, `‥`
  at the sentence tier and `；`, `，`, `、`, `：`, `･` at the clause tier.
  A run of closing marks (`」`, `』`, `）`, `”`, …) following a separator is
  absorbed into the piece that just ended, so no chunk opens with an orphaned
  bracket.
- **Setext headers** (`Title` over `====` / `----`) in the markdown strategy.
  These are CommonMark and common in older documents; they were previously
  invisible.
- **YAML (`---`) and TOML (`+++`) front matter** is excluded from header
  detection. This is not optional once setext works: front matter's closing
  delimiter follows an ordinary `key: value` line and would otherwise promote
  it to an H2. The front matter itself is still preserved as content.
- Six encodings arrive with the tokenizer upgrade: `kimi_k2`, `kimi_k3`,
  `glm4`, `glm5`, `minimax_m2`, `deepseek_v4` — 17 in total, across 10
  providers. `.model()` resolves `kimi-k2*`, `glm-4*` / `glm-5*`, `minimax*`
  and `deepseek-v4*`.

### Changed

- Separators are now **tiers** (paragraph / line / sentence / clause / word)
  and the earliest match within a tier wins. Previously each separator was its
  own rung, so `"! "` outranked `"? "` for no reason, and a document mixing
  scripts let one script's punctuation starve the other's.
- `semantic.rs` had a second, independent, also-ASCII-only sentence terminator
  table; it now delegates to the same tier. Its own test previously documented
  the bug in a comment — CJK semantic chunking was degenerate by construction.
- Cosine similarity delegates to `embedrs::similarity::cosine_similarity`
  (8-lane, autovectorizing as of embedrs 0.4) instead of a local scalar copy.
- ATX and setext headers honour CommonMark's three-space indent limit.
- `embedrs` floor `"0.3"` → `"0.5"` for the `semantic` feature. 0.5 is the first
  embedrs release on tiktoken 4; against anything older, a build enabling both
  `chunkedrs/semantic` and `embedrs/cost-tracking` would compile two majors of
  tiktoken.
- `encode` is 5–49x faster via the tiktoken 3.8 rewrite; this crate is
  tokenizer-bound.

### Docs

- The README comparison table claimed text-splitter had no markdown support,
  no byte offsets, and character-level overlap only. As of its 0.32.0 it has
  a CommonMark `MarkdownSplitter`, a tree-sitter `CodeSplitter`,
  `ChunkCharIndex` offsets, and sizer-relative overlap. The table is replaced
  with an honest side-by-side that names what text-splitter does better.
- The MSRV badge still read 1.94 after 1.0.4 removed the pin. Removed.
- The highlights claimed model auto-detection for `claude` and `llama`.
  Neither resolves — Anthropic publishes no tiktoken-compatible vocabulary at
  all, and the Meta prefix is `llama-`, not `llama`. Unrecognised names fall
  back to `o200k_base` silently, so this was invisible at runtime. Documented
  as a fallback rather than advertised as a feature.
- "9 encodings" → 17.
- All three READMEs' quick-start output comments reported token counts and
  byte ranges that did not match what the code produces; they appear to have
  been copied between languages. Recomputed.
- The overlap section claimed overlap is "critical for retrieval quality".
  Recent retrieval evaluations do not support that; reworded to describe the
  tradeoff and note that it is off by default.

## [1.0.4] - 2026-06-07

### Changed
- Drop `rust-version = "1.94"` pin from `Cargo.toml`. The crate now follows
  whatever stable rustc is current, rather than declaring an MSRV. Transitive
  deps (`tiktoken`, `embedrs`) still declare their own. No source changes.

## [1.0.3] - 2026-04-24

### Changed
- Smoke-test release via the new repo's GitHub Actions publish workflow.
  No code changes.

## [1.0.2] - 2026-04-24

### Changed
- Moved to its own repository, `goliajp/rust-chunker`. No code changes;
  `repository` URL updated, `tiktoken` and `embedrs` come from crates.io.

## [1.0.1] - 2026-04

- Maintenance release.

## [1.0.0] - 2026-04

- Initial public release.
