# chunkedrs

[![Crates.io](https://img.shields.io/crates/v/chunkedrs?style=flat-square&logo=rust)](https://crates.io/crates/chunkedrs)
[![docs.rs](https://img.shields.io/docsrs/chunkedrs?style=flat-square&logo=docs.rs)](https://docs.rs/chunkedrs)
[![License](https://img.shields.io/crates/l/chunkedrs?style=flat-square)](LICENSE)
[![Downloads](https://img.shields.io/crates/d/chunkedrs?style=flat-square)](https://crates.io/crates/chunkedrs)

[English](README.md) | **简体中文** | [日本語](README.ja.md)

面向 RAG 管道的 token 精确文本分块 — 支持递归、Markdown 感知和语义分割。基于 [tiktoken](https://crates.io/crates/tiktoken)，最快的纯 Rust BPE 分词器。

## 特性亮点

- **Token 精确** — 每个分块严格保证不超过 token 上限，而非近似的字符数估算
- **3 种策略** — 递归（快速通用）、Markdown 感知（保留标题结构）、语义分割（基于 embedding 的断点检测）
- **中日文感知** — 中文和日文按 `。`、`！`、`？` 及子句标点切分，而不是按它们根本不用的 ASCII 空格
- **丰富元数据** — 每个分块附带字节偏移、token 计数和章节标题
- **重叠支持** — 可配置的 token 级重叠
- **任意分词器** — 从模型名称自动检测，或直接指定 17 种编码之一
- **基于 tiktoken 4** — 最快的纯 Rust BPE 分词器，覆盖 10 家厂商

## 为什么选择 chunkedrs？

RAG 管道需要将文本分割成适合模型上下文窗口的片段。简单的按字符数或固定大小分割会在词中间、句子中间甚至段落中间断开 — 破坏语义，降低检索质量。

chunkedrs 在**语义边界**（段落、句子、子句、单词）处分割，同时保证**精确的 token 限制**。没有任何分块会超过 `max_tokens`。

### 与 [text-splitter](https://crates.io/crates/text-splitter) 的对比

两者都做得不错，只是默认取向不同。text-splitter 是更宽的工具箱 —— 它有基于 tree-sitter 的代码分割，chunkedrs 没有。

| | chunkedrs | text-splitter 0.32 |
|---|---|---|
| 默认计量单位 | 恒为 token | 字符；token 计量需另行配置 |
| 分词器 | 内置 tiktoken | 通过 feature 接 `tiktoken-rs` 或 `tokenizers` |
| Markdown | 标题作为分块元数据 | 按 CommonMark 结构分割 |
| 代码（tree-sitter） | 无 | 有 |
| 基于 embedding 的语义分割 | 有（经 [embedrs](https://crates.io/crates/embedrs)） | 无 |
| 字节偏移 | 有 | 有 |
| 重叠 | 按 token | 跟随所配置的计量单位 |

想要「不用配置就是 token 精确」、每块都带章节元数据、或基于 embedding 的断点检测，选 chunkedrs。需要切分源代码，选 text-splitter。

## 分割策略

| 策略 | 适用场景 | 速度 |
|------|---------|------|
| **递归分割**（默认） | 通用文本 — 按段落、句子、子句、单词 | 最快 |
| **Markdown** | 含标题的文档 — 保留章节元数据 | 快 |
| **语义分割** | 高质量 RAG — 基于 embedding 在语义边界分割 | 较慢（需 API 调用） |

## 快速开始

添加到 `Cargo.toml`：

```toml
[dependencies]
chunkedrs = "1.1"
```

使用默认配置分割文本（递归、512 最大 token、无重叠）：

```rust
use chunkedrs::Chunk;

let chunks: Vec<Chunk> = chunkedrs::chunk("你的长文本...").split();
for chunk in &chunks {
    println!("[{}] {} tokens (bytes {}..{})", chunk.index, chunk.token_count, chunk.start_byte, chunk.end_byte);
}
// 输出:
// [0] 4 tokens (bytes 0..18)
```

## Token 精确分割

```rust
let chunks = chunkedrs::chunk("你的长文本...")
    .max_tokens(256)
    .overlap(50)
    .model("gpt-4o")
    .split();

// 每个分块保证 <= 256 tokens
assert!(chunks.iter().all(|c| c.token_count <= 256));
```

## Markdown 感知分割

```rust
let markdown = "# 介绍\n\n一些文本。\n\n## 详情\n\n更多内容。\n";
let chunks = chunkedrs::chunk(markdown).markdown().split();

// 每个分块知道它属于哪个章节
assert_eq!(chunks[0].section.as_deref(), Some("# 介绍"));
```

## 中日文文本

中文和日文在句号后面不加空格，所以「每个分隔符都以 ASCII 空格结尾」的切分器
在这类文本里一个边界都找不到，只能从词中间硬切。chunkedrs 按这两种文字真正
使用的标点切分，并且让收尾引号跟着它所收尾的那一句：

```rust
let zh = "他说「今天天气很好。」然后就出门了。她回答说「确实不错。」于是也跟着出去了。";
let chunks = chunkedrs::chunk(zh).max_tokens(12).split();

assert_eq!(chunks[0].content, "他说「今天天气很好。」");
assert_eq!(chunks[1].content, "然后就出门了。");
```

整句放不下时，子句标点（`，`、`、`、`；`）是下一级边界，再往下才退到 token 偏移。

## 语义分割

启用 `semantic` feature 后，基于 embedding 在语义边界分割：

```toml
[dependencies]
chunkedrs = { version = "1.1", features = ["semantic"] }
```

```rust,ignore
let client = embedrs::openai("sk-...");
let chunks = chunkedrs::chunk("你的长文本...")
    .semantic(&client)
    .threshold(0.5)
    .split_async()
    .await?;
```

## 分块元数据

每个 `Chunk` 携带丰富的元数据：

```rust
pub struct Chunk {
    pub content: String,         // 文本内容
    pub index: usize,            // 在序列中的位置
    pub start_byte: usize,       // 原文中的字节偏移
    pub end_byte: usize,         // 字节偏移（不含）
    pub token_count: usize,      // 精确的 token 数
    pub section: Option<String>, // markdown 标题（如适用）
}
```

## 重叠

连续分块之间的 token 重叠可把上下文带过边界：

```rust
let chunks = chunkedrs::chunk("你的长文本...")
    .max_tokens(256)
    .overlap(50)
    .split();
```

重叠是否有用取决于具体工况 —— 近期的检索评测报告显示收益不明显，而索引体积
的代价是实打实的。默认关闭；先量再开。

## 分词器选择

```rust
// 从模型名称自动检测
let chunks = chunkedrs::chunk(text).model("gpt-4o").split();

// 或直接指定 17 种编码之一
let chunks = chunkedrs::chunk(text).encoding("cl100k_base").split();

// 默认：o200k_base（GPT-4o, GPT-5, o 系列）
```

模型名覆盖 OpenAI、Meta（`llama-3.1-70b`）、DeepSeek（`deepseek-v4`）、
阿里（`qwen2.5-72b`）、Mistral、月之暗面（`kimi-k2`）、智谱（`glm-5`）、
MiniMax（`minimax-m2`）。

Anthropic 和 Google 没有公开 tiktoken 兼容的词表，所以 `claude-*` 和
`gemini-*` **不会**匹配 —— 无法识别的名字会回落到 `o200k_base`，那是估算
而非精确计数。需要显式固定这一行为时请用 `.encoding()`。

<!-- ECOSYSTEM BEGIN (generated — edit ecosystem.toml, not this block) -->

## 生态系统

[tiktoken](https://crates.io/crates/tiktoken) · [@goliapkg/tiktoken-wasm](https://www.npmjs.com/package/@goliapkg/tiktoken-wasm) · [instructors](https://crates.io/crates/instructors) · **chunkedrs** · [embedrs](https://crates.io/crates/embedrs)

<!-- ECOSYSTEM END -->

## 许可证

MIT
