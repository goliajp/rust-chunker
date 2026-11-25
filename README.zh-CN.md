# chunkedrs

[![Crates.io](https://img.shields.io/crates/v/chunkedrs?style=flat-square&logo=rust)](https://crates.io/crates/chunkedrs)
[![docs.rs](https://img.shields.io/docsrs/chunkedrs?style=flat-square&logo=docs.rs)](https://docs.rs/chunkedrs)
[![CI](https://img.shields.io/github/actions/workflow/status/goliajp/rust-chunker/ci.yml?branch=develop&style=flat-square&logo=github&label=ci)](https://github.com/goliajp/rust-chunker/actions/workflows/ci.yml)
[![License](https://img.shields.io/crates/l/chunkedrs?style=flat-square)](#许可证)

[English](README.md) | **简体中文** | [日本語](README.ja.md)

> **从 2.1 升级**：API 没有变化，但分块的边界会变，已经建好的索引要重新分块。tiktoken 5 给 Llama 4、Qwen 3.5〜3.8、Mistral Large 4 和 `deepseek-flash` 换成了它们实际使用的词表，这些模型的 token 数会变；新的分块算法也会让一部分分块在和 2.1 不同的位置结束。32 KiB 及以上的文档现在用多个线程编码，设置 `TIKTOKEN_THREADS=1` 可以关掉。详见 [CHANGELOG](CHANGELOG.md#upgrading-from-21)。

chunkedrs 把长文档切成不超过 token 上限的分块，用于向量化和检索。token 数由 [tiktoken](https://crates.io/crates/tiktoken) 计算，支持 20 种编码（OpenAI、Llama、DeepSeek、Qwen、Mistral、Kimi、GLM、MiniMax）。每个分块都带着它在原文中的字节范围，以及它在整篇文档 token 序列中的位置。除 tiktoken 外没有其他依赖，开启 `semantic` feature 时会加上 [embedrs](https://crates.io/crates/embedrs)。

在 Apple M4 Pro 上单线程运行，编码 `o200k_base`，每块 512 token：切分一本 200 KB 的英文书用 0.35 ms。同样条件下，text-splitter 0.33.6 用同一个分词器需要 1.19 ms，用 tiktoken-rs 0.12（它的 `tiktoken-rs` feature 所用的分词器）需要 34.8 ms。

## 特点

- **五种策略**：递归（段落、行、句子、分句、词）、带标题层级的 Markdown、按边界标记切分的代码和 HTML，以及按向量相似度切分的语义模式
- **严格遵守 token 上限**：`token_count` 是对分块自身文本单独计数的结果，不超过 `max_tokens`。唯一的例外是单个字符本身就超过上限的情况
- **位置信息**：对每个分块都有 `&doc[start_byte..end_byte] == content`；`start_token..end_token` 是这个分块在 `encode(doc)` 中的范围，[late chunking](#late-chunking) 需要的正是它
- **中文和日文**：句子在 `。！？` 处结束，分句在 `，、；：` 处结束，后面不需要空格；右引号留在它所属的句子里
- **批量处理**：`Chunker` 只解析一次编码器，`split_batch` 把一批文档分配到所有 CPU 核上处理

## 快速上手

```toml
[dependencies]
chunkedrs = "2"
```

```rust
// 默认：递归切分，每块 512 token，不重叠，编码 o200k_base
let chunks = chunkedrs::chunk("your long text here...").split();
for c in &chunks {
    println!("[{}] {} tokens, bytes {}..{}", c.index, c.token_count, c.start_byte, c.end_byte);
}

// 用模型名选择编码
let chunks = chunkedrs::chunk("your long text here...")
    .max_tokens(256)
    .overlap(32)
    .model("gpt-5.6-terra")
    .split();
assert!(chunks.iter().all(|c| c.token_count <= 256));

// 一份配置处理多篇文档
let chunker = chunkedrs::Chunker::new().max_tokens(256).markdown();
let docs = ["# A\n\nFirst.", "# B\n\nSecond."];
let all = chunker.split_batch(&docs);
assert_eq!(all[1][0].section(), Some("# B"));
```

## 性能

表中是切分一篇文档的毫秒数，取预热后 9 轮的中位数。测试机为 Apple M4 Pro Mac mini，rustc 1.100，release 构建并开启 LTO，`TIKTOKEN_THREADS=1`，编码 `o200k_base`，不重叠；测试期间负载为 1.3〜2.0，测于 2026-11-25。text-splitter 的容量设为同样的 `max_tokens`，其余用默认值；它的代码切分器用 tree-sitter-rust 0.24。

| 语料 | 策略 | max_tokens | 一次 `encode` | chunkedrs | text-splitter + tiktoken 5.5 | text-splitter + tiktoken-rs 0.12 |
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

`book_en.txt` 是《傲慢与偏见》的 200,000 字节，`zh_xiyouji.md` 是 Wikisource 上《西游记》第一至九回整理成的 Markdown，共 178,622 字节，`md_rust_book.md` 是《The Rust Programming Language》第 3〜6 章，共 174,237 字节，`code_rust.txt` 是 100,000 字节的 Rust 源码。文件都在 [`bench-compare/data`](bench-compare/data)。

- 上表所有分块都用 tiktoken 重新数过，chunkedrs 和 text-splitter 都没有超出上限的分块
- chunkedrs 的耗时是对整篇文档做一次 `encode` 的 2.9〜5.1 倍。其中整篇文档编码一次（一次 `encode`），查出每个 token 的字节长度（约半次 `encode`），每个分块再单独计数一次（约一次 `encode`，因为 BPE 的结果在切点两侧不能直接相加）；剩下的是找边界和构造分块
- `cl100k_base` 的结果相近，chunkedrs 为 0.19〜0.72 ms。使用 tiktoken 默认的多线程时，这些 100〜200 KB 的文档整篇编码会用多个线程，chunkedrs 为 0.17〜0.45 ms
- 用 `Chunker::split_batch` 处理 64 篇文档（四份语料各 16 次，共 10.4 MB）：逐篇处理 28.2 ms，批量处理 3.9 ms

复现命令：`TIKTOKEN_THREADS=1 cargo run --release -p bench-compare`。

## 切分方式

整篇文档只编码一次。从一个分块的起点往后数 `max_tokens` 个文档 token，得到这个分块最远能到的字节位置。分块在这个范围内、优先级最高的那一类边界的最后一处结束：

| 策略 | 边界，按优先级从高到低 |
|---|---|
| 递归（默认）| 空行；换行；后跟空格的 `. ! ?` 以及 `。！？．｡…‥`；`; , :` 和 `；，、：･`；空格、制表符、全角空格 |
| `markdown()` | 先按标题行切（ATX `#` 和 setext `===` / `---`，代码块和 front matter 里的不算），每一节内部再用递归的边界 |
| `code()` | 空行；位于第 0 列、独占一行的 `}` `};` `)` `]` `];` `end`；换行；空格、制表符 |
| `html()` | 块级元素的结束标签之后（`</p>`、`</li>`、`</section>`、`</h1>` 等），再用递归的边界 |
| `semantic()` | 相邻句子的向量相似度低于阈值处，再用递归的边界 |

范围内没有任何边界时，分块在 token 边界处结束，并退回到字符边界。选定的文本随后单独计数一次：BPE 会跨越切点合并，一段文本单独编码时，可能比它在整篇文档中所占的 token 多一个或少一个。少数情况下超出上限，就缩小范围重新选择。

代码和 HTML 的切分只看标点和标签，不做语法分析，也不增加依赖；因此字符串字面量里的 `}` 也会被当成代码块的结束。需要按语法切分代码时，请用 text-splitter 基于 tree-sitter 的 `CodeSplitter`。

## Markdown

```rust
let md = "# Guide\n\nIntro.\n\n## Install\n\nRun cargo add.\n";
let chunks = chunkedrs::chunk(md).markdown().split();

assert_eq!(chunks[0].section(), Some("# Guide"));
assert_eq!(chunks[1].section_path, ["# Guide", "## Install"]);
```

标题行放在 `section_path` 里，不在 `content` 里，所以把所有分块的内容拼起来不会得到这些标题行。

## 中文和日文

```rust
let zh = "他说「今天天气很好。」然后就出门了。她回答说「确实不错。」于是也跟着出去了。";
let chunks = chunkedrs::chunk(zh).max_tokens(12).split();

assert_eq!(chunks[0].content, "他说「今天天气很好。」");
assert_eq!(chunks[1].content, "然后就出门了。");
```

如果切分器的分隔符都以 ASCII 空格结尾，在这段文字里一个句子边界也找不到，只能在词中间切开。

## Late chunking

[Late chunking](https://arxiv.org/abs/2409.04701) 先对整篇文档做一次向量化，再在每个分块自己的 token 范围上做池化，得到分块的向量。`start_token..end_token` 就是这个分块在 `encode(doc)` 中的范围：

```rust
let doc = "First sentence here. Second sentence here. Third sentence here.";
let chunks = chunkedrs::chunk(doc).max_tokens(8).split();

let encoder = tiktoken::get_encoding("o200k_base").unwrap();
let document_tokens = encoder.encode(doc);
for chunk in &chunks {
    // 实际使用时，这里取的是模型每个 token 的隐藏状态
    let span = &document_tokens[chunk.start_token..chunk.end_token];
    assert!(!span.is_empty());
}
```

边界落在一个 token 内部时（例如 `". "` 后面的空格被分词器并入了下一个词），范围会把这个 token 包含进来，所以相邻两个分块的范围可能共用一个 token，`token_count` 也可能和 `end_token - start_token` 不同。示例见 [`examples/token_spans.rs`](examples/token_spans.rs)。

## 语义切分

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

句子按服务商单次请求的上限分批向量化。`.semantic()` 返回的构建器只有 `split_async()`，不会在同步的 `split()` 里误发网络请求。

## Chunk

```rust
#[non_exhaustive]
pub struct Chunk {
    pub content: String,           // 文本，等于 &doc[start_byte..end_byte]
    pub index: usize,              // 在序列中的位置
    pub start_byte: usize,
    pub end_byte: usize,           // 不含
    pub start_token: usize,        // 在 encode(doc) 中的范围
    pub end_token: usize,          // 不含
    pub token_count: usize,        // content 单独计数的 token 数
    pub section_path: Vec<String>, // Markdown 标题层级，从外到内
}
```

只含空白字符的分块会被丢弃。`Chunk` 带有 `#[non_exhaustive]`，手动构造请用 `Chunk::new(..).with_bytes(..).with_tokens(..)`。

重叠（`.overlap(n)`）让每个分块从上一个分块结束前大约 `n` 个 token 处开始，起点落在字符边界上，默认关闭。重叠对检索效果有没有帮助取决于具体场景，开启前请先测一下。

## 编码和 feature

`.model("...")` 能解析 OpenAI、Meta、DeepSeek、Qwen、Mistral、Moonshot、智谱和 MiniMax 的模型名；`.encoding("cl100k_base")` 直接指定编码，优先于 `.model()`。不认识的名字不会报错，而是退回 `o200k_base`。Anthropic 和 Google 没有公开与 tiktoken 兼容的词表，`claude-*` 和 `gemini-*` 也会退回 `o200k_base`，得到的是近似的 token 数。

和 tiktoken 一样，词表是默认开启、可以关闭的 feature。默认编译全部 20 种编码；`default-features = false` 只保留 `o200k_base`，也就是退回时使用的编码。

```toml
chunkedrs = "2"                                                   # 全部词表
chunkedrs = { version = "2", default-features = false }           # 只有 o200k_base
chunkedrs = { version = "2", default-features = false, features = ["vocab-qwen"] }
```

以 `examples/basic` 为例，在 macOS arm64 上做 release 构建：包含全部词表时为 9,189,376 字节，`default-features = false` 时为 2,072,256 字节，小 77%。

按厂商分组：`vocab-openai`、`vocab-meta`、`vocab-deepseek`、`vocab-qwen`、`vocab-mistral`、`vocab-moonshot`、`vocab-zhipu`、`vocab-minimax`。单个词表：`vocab-cl100k_base`、`vocab-llama4`、`vocab-qwen3_5`、`vocab-mistral_v15`、`vocab-glm5`，以及 tiktoken 列出的其他词表。没有编译进来的词表和不认识的名字一样，会退回 `o200k_base`，所以请确认你指定的编码已经开启。

## 和 text-splitter 的比较

| | chunkedrs | text-splitter 0.33 |
|---|---|---|
| 大小单位 | token（tiktoken）| 默认是字符；可以通过 tiktoken-rs、tokenizers 或自定义的 `ChunkSizer` 按 token 计 |
| Markdown | 每个分块带标题层级 | 按 CommonMark 结构切分（pulldown-cmark）|
| 代码 | 按边界标记，适用于任何语言，无依赖 | 每种语言一个 tree-sitter 语法 |
| HTML | 支持 | 不支持 |
| 分块在文档 token 序列中的范围 | 有 | 没有 |
| 按向量相似度切分 | 有（`semantic`）| 没有 |
| 大小区间（`min..max`）、去除首尾空白 | 没有 | 有 |
| Unicode 分段（字素、词、句，基于 ICU）| 没有 | 有 |

## 构建

```bash
cargo test --workspace --all-targets
cargo clippy --workspace --all-targets --all-features -- -D warnings
CHUNKEDRS_PROPTEST_CASES=5000 cargo test --test properties   # 运行更多的属性测试用例
cargo run --release -p bench-compare                          # 生成上面的性能表
```

## 相关 crate

[tiktoken](https://crates.io/crates/tiktoken) · [@goliapkg/tiktoken-wasm](https://www.npmjs.com/package/@goliapkg/tiktoken-wasm) · [instructors](https://crates.io/crates/instructors) · [embedrs](https://crates.io/crates/embedrs)

## 许可证

可任选 [MIT](LICENSE-MIT) 或 [Apache-2.0](LICENSE-APACHE) 许可证。

通过 tiktoken 的 `vocab-*` feature 编译进来的词表沿用各自发布方的许可证，其中一些对使用有条件（Llama、Mistral、DeepSeek、Kimi、MiniMax）。清单见 tiktoken 的 [LICENSE-3RD-PARTY](https://github.com/goliajp/rust-tiktoken/blob/develop/tiktoken/LICENSE-3RD-PARTY)。没有开启某个词表的 feature，构建产物里就不含这份词表。
