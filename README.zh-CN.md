# chunkedrs

[![Crates.io](https://img.shields.io/crates/v/chunkedrs?style=flat-square&logo=rust)](https://crates.io/crates/chunkedrs)
[![docs.rs](https://img.shields.io/docsrs/chunkedrs?style=flat-square&logo=docs.rs)](https://docs.rs/chunkedrs)
[![CI](https://img.shields.io/github/actions/workflow/status/goliajp/rust-chunker/ci.yml?branch=develop&style=flat-square&logo=github&label=ci)](https://github.com/goliajp/rust-chunker/actions/workflows/ci.yml)
[![License](https://img.shields.io/crates/l/chunkedrs?style=flat-square)](#许可证)

[English](README.md) | **简体中文** | [日本語](README.ja.md)

> **从 2.x 升级**：API 变了，切出来的分块没变。所有设置都改在 `Chunker` 上；分块直接借用文档里的文本，需要长期保存时转成 `OwnedChunk`；不认识的模型名或编码名现在会报错，不再悄悄换成 `o200k_base`。递归、Markdown、代码和 HTML 策略切出的分块和 2.2 完全相同。语义切分可以接任何实现了 `embedrs::Embed` 的向量化后端。迁移对照表见 [CHANGELOG](CHANGELOG.md#upgrading-from-2x)。

chunkedrs 把长文档切成不超过 token 上限的分块，用于向量化和检索。token 数由 [tiktoken](https://crates.io/crates/tiktoken) 计算，支持 20 种编码（OpenAI、Llama、DeepSeek、Qwen、Mistral、Kimi、GLM、MiniMax）。每个分块都是原文的一个切片，带着它在原文中的字节范围，以及它在整篇文档 token 序列中的位置。除 tiktoken 外没有其他依赖；开启 `semantic` feature 时会加上 [embedrs](https://crates.io/crates/embedrs)，它本身也没有依赖。

在 Apple M6 上单线程运行，编码 `o200k_base`，每块 512 token：切分一本 200 KB 的英文书用 0.13 ms，是对整本书做一次 `encode` 的 1.5 倍。chunkedrs 3.1 需要 0.14 ms，3.0 需要 0.16 ms，2.2 需要 0.25 ms；text-splitter 0.33.6 用同一个分词器需要 0.93 ms，用 tiktoken-rs 0.12（它的 `tiktoken-rs` feature 所用的分词器）需要 24.6 ms。

## 特点

- **五种策略**：递归（段落、行、句子、分句、词）、带标题层级的 Markdown、按边界标记切分的代码和 HTML，以及按向量相似度切分的语义模式
- **严格遵守 token 上限**：`token_count` 是对分块自身文本单独计数的结果，不超过 `max_tokens`。唯一的例外是单个字符本身就超过上限的情况
- **借用原文，带位置信息**：`content` 就是 `&doc[start_byte..end_byte]`，不复制；`start_token..end_token` 是这个分块在整篇文档 token id 中的范围，[late chunking](#late-chunking) 需要的正是它
- **中文和日文**：句子在 `。！？` 处结束，分句在 `，、；：` 处结束，后面不需要空格；右引号留在它所属的句子里
- **语义切分可以用任何模型**：chunkedrs 把句子交出去，再收回向量，所以远程 API、本地模型或缓存都能用，也不需要异步运行时
- **批量处理**：`split_batch` 把一批文档分配到所有 CPU 核上处理

## 快速上手

```toml
[dependencies]
chunkedrs = "3"
```

```rust
// 递归策略，每块 512 token，不重叠，o200k_base
let chunks = chunkedrs::split("your long text here...");
for c in &chunks {
    println!("[{}] {} tokens, bytes {}..{}", c.index, c.token_count, c.start_byte, c.end_byte);
}

// 用模型名选择编码；不认识的名字会返回错误
let chunker = chunkedrs::Chunker::for_model("gpt-5.6-terra")?
    .max_tokens(256)
    .overlap(32);
let chunks = chunker.split("your long text here...");
assert!(chunks.iter().all(|c| c.token_count <= 256));

// 一份配置处理多篇文档
let chunker = chunkedrs::Chunker::new().max_tokens(256).markdown();
let docs = ["# A\n\nFirst.", "# B\n\nSecond."];
let all = chunker.split_batch(&docs);
assert_eq!(all[1][0].section(), Some("# B"));

// 分块借用原文；文档释放后还要用，就转成 OwnedChunk
let kept: Vec<chunkedrs::OwnedChunk> = chunks.iter().map(|c| c.to_owned_chunk()).collect();
```

## 性能

表中是切分一篇文档所用的毫秒数：先取预热后 9 轮的中位数，再取 3 次测量的中位数。测试环境：Apple M6 的 Mac mini，rustc 1.100，开启 LTO 的 release 构建，`TIKTOKEN_THREADS=1`，`o200k_base`，不重叠，2026-11-27。测试期间平均负载在 1.1 到 2.2 之间。同一行的各列是紧接着依次测的。chunkedrs 3.1.0、3.0.0 和 2.2.0 是之前的版本，用的是同一个 tiktoken 5.7。text-splitter 的容量设为同样的 `max_tokens`，其余用默认值；它的代码切分用 tree-sitter-rust 0.24。

| 语料 | 策略 | max_tokens | 一次 `encode` | chunkedrs | chunkedrs 3.1.0 | chunkedrs 3.0.0 | chunkedrs 2.2.0 | text-splitter + tiktoken 5.7 | text-splitter + tiktoken-rs 0.12 |
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

`book_en.txt` 是《傲慢与偏见》的 200,000 字节；`zh_xiyouji.md` 是 Wikisource 上《西游记》第 1 到 9 回，整理成 Markdown，共 178,622 字节；`ja_neko.txt` 是青空文库中夏目漱石《我是猫》的前 180,000 字节，去掉了注音；`md_rust_book.md` 是《The Rust Programming Language》第 3 到 6 章，共 174,237 字节；`code_rust.txt` 是 100,000 字节的 Rust 源码。这些文件都在 [`bench-compare/data`](bench-compare/data) 里。

- 上表每一行的每个分块都用 tiktoken 重新数过，chunkedrs 和 text-splitter 都没有超出上限的分块
- 每块 512 token 时，无论哪种策略，chunkedrs 用时都是对整篇文档做一次 `encode` 的 1.5 到 1.8 倍，每块 128 token 时是 1.8 到 2.9 倍。它把文档编码一次，每个 token 的结束位置由编码器直接给出；数每个分块的 token 时，在分块中第一个和最后一个 tiktoken 切点之间直接用文档的 id，只重新编码两端；其余时间用于找边界和生成分块
- 和 3.1.0 相比少用 1% 到 23% 的时间。差得最少的是中文用 recursive 策略、每块 512 token 的那一行（1%），这一行的时间大部分就是编码本身；差得最多的是代码。和 2.2.0 相比少用 35% 到 52%。四个版本切出的分块相同
- 换成 `cl100k_base` 结果相近：每块 512 token 时是一次 `encode` 的 1.4 到 1.9 倍（3.1.0 在日文上、以及用 markdown 策略切 Markdown 时最多 2.07 倍），比 3.1.0 少用 2% 到 24%。使用 tiktoken 默认的多线程时，这些文档的整篇编码会分到多个线程上，chunkedrs 用时 0.06 到 0.18 ms（3.1.0 是 0.07 到 0.21 ms）
- 用 `Chunker::split_batch` 处理 80 篇文档（5 份语料各 16 次，共 13.3 MB）：逐篇处理 7.4 ms，批量处理 2.3 ms

复现方法：`TIKTOKEN_THREADS=1 cargo run --release -p bench-compare`。

## 切分方式

整篇文档只编码一次。从一个分块的起点往后数 `max_tokens` 个文档 token，所到的字节位置就是这个分块最远能结束的地方。在这个范围内，分块在出现的边界中优先级最高的那一类的最后一个位置结束：

| 策略 | 边界（优先级从高到低）|
|---|---|
| 递归（默认）| 空行；换行；后面跟空格的 `. ! ?`，以及 `。！？．｡…‥`；`; , :` 和 `；，、：･`；空格、制表符、全角空格 |
| `markdown()` | 先按标题行切（ATX 的 `#` 和 setext 的 `===` / `---`，代码块和 front matter 里的不算），每一节内部再用递归策略的边界 |
| `code()` | 空行；在第 0 列单独占一行的 `}` `};` `)` `]` `];` `end`；换行；空格、制表符 |
| `html()` | 块级元素的结束标签之后（`</p>`、`</li>`、`</section>`、`</h1>` 等），再用递归策略的边界 |
| `semantic()` | 相邻句子的向量不再相似的地方，再用递归策略的边界 |

范围内一个边界都没有时，分块在 token 边界结束，并退到字符边界上。选出的文本随后按单独编码的情况重新计数：BPE 会跨过切点合并，所以单独编码一段文本，得到的 token 数可能比它在整篇文档中所占的多一个或少一个。这次计数复用整篇文档的 token id：在分块中第一个和最后一个「tiktoken 保证切开后 token 不变」的位置之间，直接用文档的 id，只把两端的几个字节重新编码。极少数情况下分块超出上限，就缩小范围重新选。

代码和 HTML 策略只看标点和标签，不做语法分析，也不增加依赖；字符串字面量里的 `}` 也会被当成代码块的结尾。需要按语法切分代码时，请用 text-splitter 基于 tree-sitter 的 `CodeSplitter`。

## Markdown

```rust
let md = "# Guide\n\nIntro.\n\n## Install\n\nRun cargo add.\n";
let chunks = chunkedrs::Chunker::new().markdown().split(md);

assert_eq!(chunks[0].section(), Some("# Guide"));
assert_eq!(chunks[1].section_path, ["# Guide", "## Install"]);
```

标题行放在 `section_path` 里，不在 `content` 里，所以把所有分块的内容拼起来不会得到这些标题行。想把标题和分块一起向量化，自己拼在前面即可：`format!("{}\n\n{}", c.section_path.join(" > "), c.content)`。

## 中文和日文

```rust
let zh = "他说「今天天气很好。」然后就出门了。她回答说「确实不错。」于是也跟着出去了。";
let chunks = chunkedrs::Chunker::new().max_tokens(12).split(zh);

assert_eq!(chunks[0].content, "他说「今天天气很好。」");
assert_eq!(chunks[1].content, "然后就出门了。");
```

如果切分器的分隔符都以 ASCII 空格结尾，在这段文字里一个句子边界也找不到，只能在词中间切开。

## Late chunking

[Late chunking](https://arxiv.org/abs/2409.04701) 先对整篇文档做一次向量化，再在每个分块自己的 token 范围上做池化，得到分块的向量。`split_with_tokens` 在返回分块的同时返回整篇文档的 token id，`start_token..end_token` 就是这些 id 中的下标范围：

```rust
let doc = "First sentence here. Second sentence here. Third sentence here.";
let (chunks, ids) = chunkedrs::Chunker::new().max_tokens(8).split_with_tokens(doc);
for chunk in &chunks {
    // 实际使用时，这里取的是模型每个 token 的隐藏状态
    let span = &ids[chunk.start_token..chunk.end_token];
    assert!(!span.is_empty());
}
```

边界落在一个 token 内部时（例如 `". "` 后面的空格被分词器并入了下一个词），范围会把这个 token 包含进来，所以相邻两个分块的范围可能共用一个 token，`token_count` 也可能和 `end_token - start_token` 不同。示例见 [`examples/token_spans.rs`](examples/token_spans.rs)。

## 语义切分

`Chunker::semantic(text)` 把文本切成句子（每句都不超过 `max_tokens`），通过 `inputs()` 交给你，它们都是原文的切片。用任何方式把它们向量化，再把向量交回来；chunkedrs 在相邻句子不再相似的地方断开，并把每一组按上限切好。不需要 feature，没有依赖，也不做 I/O：

```rust
let text = "Cats purr. Cats nap. Stocks fell. Markets slid.";
let job = chunkedrs::Chunker::new().semantic(text);
let vectors: Vec<f32> = my_model(job.inputs());   // 每个输入一个向量，依次首尾相接
let chunks = job.finish(&vectors, 384)?;
```

开启 `semantic` feature 后，任何实现了 `embedrs::Embed` 的类型都可以负责向量化：远程服务商的 `Client`（按服务商的上限拆分请求并并发发送）、在 CPU 上运行的本地模型，或者你自己的类型。

```toml
chunkedrs = { version = "3", features = ["semantic"] }
embedrs = { version = "0.7", features = ["local"] }
```

```rust,ignore
let client = embedrs::local();   // all-MiniLM-L6-v2，首次使用时下载
let chunks = chunkedrs::Chunker::new()
    .max_tokens(256)
    .split_semantic(text, &client)
    .await?;

// 多篇文档的句子合成一次向量化调用
let all = chunker.split_semantic_batch(&docs, &client).await?;
```

默认在相邻句子的余弦距离高于这篇文档所有相邻距离的第 95 百分位数时断开（`Breakpoints::Percentile(95.0)`），这样阈值会随模型自动调整。用 `semantic(text).breakpoints(Breakpoints::Threshold(0.5))` 可以改成按固定的相似度断开。embedrs 自带的本地模型都是英文模型，中文和日文请用多语言的向量模型。

## Chunk

```rust
#[non_exhaustive]
pub struct Chunk<'a> {
    pub content: &'a str,          // &doc[start_byte..end_byte]
    pub index: usize,              // 在序列中的位置
    pub start_byte: usize,
    pub end_byte: usize,           // 不含
    pub start_token: usize,        // 在文档 token id 中的范围
    pub end_token: usize,          // 不含
    pub token_count: usize,        // content 单独计数的 token 数
    pub section_path: Vec<&'a str>, // Markdown 标题层级，从外到内
}
```

只含空白字符的分块会被丢弃。`to_owned_chunk()` 得到字段相同、文本为 `String` 的 `OwnedChunk`，文档释放后仍要保存分块或把分块交给别的线程时使用。

重叠（`.overlap(n)`）让每个分块从上一个分块结束前大约 `n` 个 token 处开始，起点落在字符边界上，默认关闭。重叠对检索效果有没有帮助取决于具体场景，开启前请先测一下。

## 编码和 feature

`Chunker::for_model("...")` 能解析 OpenAI、Meta、DeepSeek、Qwen、Mistral、Moonshot、智谱和 MiniMax 的模型名；`Chunker::for_encoding("cl100k_base")` 直接指定编码。tiktoken 不认识的名字，或者词表没有编译进来的名字，分别返回 `Error::UnknownModel` 和 `Error::UnknownEncoding`。Anthropic 和 Google 没有公开与 tiktoken 兼容的词表，用 `Chunker::new()`（`o200k_base`）可以得到近似的 token 数。

和 tiktoken 一样，词表是默认开启、可以关闭的 feature。默认编译全部 20 种编码；`default-features = false` 只保留 `o200k_base`，也就是 `Chunker::new()` 使用的编码。

```toml
chunkedrs = "3"                                                   # 全部词表
chunkedrs = { version = "3", default-features = false }           # 只有 o200k_base
chunkedrs = { version = "3", default-features = false, features = ["vocab-qwen"] }
```

一个在运行时按名字选择编码的程序，在 macOS arm64 上做 release 构建：包含全部词表时为 9,055,648 字节，`default-features = false` 时为 1,938,480 字节，小 79%。

按厂商分组：`vocab-openai`、`vocab-meta`、`vocab-deepseek`、`vocab-qwen`、`vocab-mistral`、`vocab-moonshot`、`vocab-zhipu`、`vocab-minimax`。单个词表：`vocab-cl100k_base`、`vocab-llama4`、`vocab-qwen3_5`、`vocab-mistral_v15`、`vocab-glm5`，以及 tiktoken 列出的其他词表。

## 和 text-splitter 的比较

| | chunkedrs | text-splitter 0.33 |
|---|---|---|
| 大小单位 | token（tiktoken）| 默认是字符；可以通过 tiktoken-rs、tokenizers 或自定义的 `ChunkSizer` 按 token 计 |
| Markdown | 每个分块带标题层级 | 按 CommonMark 结构切分（pulldown-cmark）|
| 代码 | 按边界标记，适用于任何语言，无依赖 | 每种语言一个 tree-sitter 语法 |
| HTML | 支持 | 不支持 |
| 分块在文档 token 序列中的范围 | 有 | 没有 |
| 按向量相似度切分 | 有，任何向量化后端 | 没有 |
| 大小区间（`min..max`）、去除首尾空白 | 没有 | 有 |
| Unicode 分段（字素、词、句，基于 ICU）| 没有 | 有 |

## 构建

```bash
cargo test --workspace --all-targets
cargo test --all-features             # 语义切分的测试在 CPU 上运行 all-MiniLM-L6-v2（91 MB，只下载一次）
cargo clippy --workspace --all-targets --all-features -- -D warnings
CHUNKEDRS_PROPTEST_CASES=5000 cargo test --test properties   # 运行更多的属性测试用例
cargo run --release -p bench-compare                          # 生成上面的性能表
```

## 相关 crate

[tiktoken](https://crates.io/crates/tiktoken) · [@goliapkg/tiktoken-wasm](https://www.npmjs.com/package/@goliapkg/tiktoken-wasm) · [instructors](https://crates.io/crates/instructors) · [embedrs](https://crates.io/crates/embedrs)

## 许可证

可任选 [MIT](LICENSE-MIT) 或 [Apache-2.0](LICENSE-APACHE) 许可证。

通过 tiktoken 的 `vocab-*` feature 编译进来的词表沿用各自发布方的许可证，其中一些对使用有条件（Llama、Mistral、DeepSeek、Kimi、MiniMax）。清单见 tiktoken 的 [LICENSE-3RD-PARTY](https://github.com/goliajp/rust-tiktoken/blob/develop/tiktoken/LICENSE-3RD-PARTY)。没有开启某个词表的 feature，构建产物里就不含这份词表。
