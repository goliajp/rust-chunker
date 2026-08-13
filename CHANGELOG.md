# Changelog

All notable changes to this crate will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/)
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

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
- Migrated from `goliajp/airs` mono-repo to standalone `goliajp/rust-chunker`.
  No code changes; `repository` URL updated. `tiktoken` and `embedrs` deps
  now resolved via crates.io instead of workspace path.

## [1.0.1] - 2026-04

- Previous release (from `goliajp/airs`).

## [1.0.0] - 2026-04

- Initial public release.
