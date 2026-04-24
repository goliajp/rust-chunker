# chunkedrs

[![Crates.io](https://img.shields.io/crates/v/chunkedrs?style=flat-square&logo=rust)](https://crates.io/crates/chunkedrs)
[![docs.rs](https://img.shields.io/docsrs/chunkedrs?style=flat-square&logo=docs.rs)](https://docs.rs/chunkedrs)
[![License](https://img.shields.io/crates/l/chunkedrs?style=flat-square)](LICENSE)
[![Downloads](https://img.shields.io/crates/d/chunkedrs?style=flat-square)](https://crates.io/crates/chunkedrs)
[![MSRV](https://img.shields.io/badge/MSRV-1.94-blue?style=flat-square)](https://www.rust-lang.org)

[English](README.md) | **简体中文** | [日本語](README.ja.md)

面向 RAG 管道的 token 精确文本分块 — 支持递归、Markdown 感知和语义分割。基于 [tiktoken](https://crates.io/crates/tiktoken)，最快的纯 Rust BPE 分词器。

## 特性亮点

- **Token 精确** — 每个分块严格保证不超过 token 上限，而非近似的字符数估算
- **3 种策略** — 递归（快速通用）、Markdown 感知（保留标题结构）、语义分割（基于 embedding 的断点检测）
- **丰富元数据** — 每个分块附带字节偏移、token 计数和章节标题
- **重叠支持** — 可配置的 token 级重叠，为检索保留边界上下文
- **任意分词器** — 从模型名称自动检测（`gpt-4o`、`claude`、`llama`）或直接指定编码
- **基于 tiktoken** — 最快的纯 Rust BPE 分词器，支持所有主流 LLM 的 9 种编码

## 为什么选择 chunkedrs？

RAG 管道需要将文本分割成适合模型上下文窗口的片段。简单的按字符数或固定大小分割会在词中间、句子中间甚至段落中间断开 — 破坏语义，降低检索质量。

chunkedrs 在**语义边界**（段落、句子、单词）处分割，同时保证**精确的 token 限制**。没有任何分块会超过 `max_tokens`。

| 特性 | chunkedrs | text-splitter | 手动实现 |
|------|-----------|---------------|---------|
| Token 精确限制 | 是（tiktoken） | 基于字符数 | 否 |
| 递归分割 | 是 | 是 | 自行实现 |
| Markdown 感知 | 是（含章节元数据） | 否 | 自行实现 |
| 语义分割 | 是（基于 embedrs） | 否 | 自行实现 |
| 字节偏移 | 是 | 否 | 自行实现 |
| 每块 token 计数 | 是 | 否 | 自行实现 |
| 重叠支持 | Token 级别 | 字符级别 | 自行实现 |
| 分词器选择 | 模型名称或编码 | 不适用 | 不适用 |

## 分割策略

| 策略 | 适用场景 | 速度 |
|------|---------|------|
| **递归分割**（默认） | 通用文本 — 按段落、句子、单词 | 最快 |
| **Markdown** | 含 `#` 标题的文档 — 保留章节元数据 | 快 |
| **语义分割** | 高质量 RAG — 基于 embedding 在语义边界分割 | 较慢（需 API 调用） |

## 快速开始

添加到 `Cargo.toml`：

```toml
[dependencies]
chunkedrs = "1"
```

使用默认配置分割文本（递归、512 最大 token、无重叠）：

```rust
use chunkedrs::Chunk;

let chunks: Vec<Chunk> = chunkedrs::chunk("你的长文本...").split();
for chunk in &chunks {
    println!("[{}] {} tokens (bytes {}..{})", chunk.index, chunk.token_count, chunk.start_byte, chunk.end_byte);
}
// 输出:
// [0] 7 tokens (bytes 0..21)
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

## 语义分割

启用 `semantic` feature 后，基于 embedding 在语义边界分割：

```toml
[dependencies]
chunkedrs = { version = "1", features = ["semantic"] }
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

连续分块之间的 token 重叠可在边界处保留上下文 — 这对检索质量至关重要：

```rust
let chunks = chunkedrs::chunk("你的长文本...")
    .max_tokens(256)
    .overlap(50)
    .split();
```

## 分词器选择

```rust
// 从模型名称自动检测
let chunks = chunkedrs::chunk(text).model("gpt-4o").split();

// 或直接指定编码
let chunks = chunkedrs::chunk(text).encoding("cl100k_base").split();

// 默认：o200k_base（GPT-4o, GPT-4-turbo）
```

<!-- ECOSYSTEM BEGIN (synced by claws/opensource/scripts/sync-ecosystem.py — edit ecosystem.toml, not this block) -->

## 生态系统

GOLIA Rust AI 基础设施系列的一部分 —— 各自独立仓维护，通过 crates.io 组合：

| Crate / Package | 仓库 | 说明 |
|---|---|---|
| [tiktoken](https://crates.io/crates/tiktoken) | [rust-tiktoken](https://github.com/goliajp/rust-tiktoken) | 高性能 BPE 分词器 —— 9 套编码、57 个模型、多厂商价格 |
| [@goliapkg/tiktoken-wasm](https://www.npmjs.com/package/@goliapkg/tiktoken-wasm) | [rust-tiktoken](https://github.com/goliajp/rust-tiktoken) | tiktoken 的 WASM 绑定 —— 浏览器 / Node.js |
| [instructors](https://crates.io/crates/instructors) | [rust-instructor](https://github.com/goliajp/rust-instructor) | 类型安全的 LLM 结构化输出提取 |
| [embedrs](https://crates.io/crates/embedrs) | [rust-embeddings](https://github.com/goliajp/rust-embeddings) | 统一 embedding —— 云 API + 本地推理，一套接口 |
| **chunkedrs**（本 crate） | [rust-chunker](https://github.com/goliajp/rust-chunker) | AI 原生文本分块 —— 递归、Markdown 感知、语义 |

<!-- ECOSYSTEM END -->

## 许可证

MIT
