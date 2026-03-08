# chunkedrs

面向 AI 的 Rust 文本分块工具 — 将长文档精确分割为 token 级别的片段，用于 embedding 和检索。

基于 [tiktoken](https://crates.io/crates/tiktoken) 精确计数 token。每个分块严格保证不超过你设定的 token 上限。

## 为什么需要 chunkedrs？

RAG 管道需要将文本分割成适合模型上下文窗口的片段。简单的按字符数或固定大小分割会在词中间、句子中间甚至段落中间断开 — 破坏语义，降低检索质量。

chunkedrs 在**语义边界**（段落 → 句子 → 单词）处分割，同时保证**精确的 token 限制**。没有任何分块会超过 `max_tokens`。

## 分割策略

| 策略 | 适用场景 | 速度 |
|------|---------|------|
| **递归分割**（默认） | 通用文本 — 按段落、句子、单词 | 最快 |
| **Markdown** | 含 `#` 标题的文档 — 保留章节元数据 | 快 |
| **语义分割** | 高质量 RAG — 基于 embedding 在语义边界分割 | 较慢（需 API 调用） |

## 快速开始

```rust
// 默认：递归分割，512 最大 token，无重叠
let chunks = chunkedrs::chunk("你的长文本...").split();
for chunk in &chunks {
    println!("[{}] {} tokens", chunk.index, chunk.token_count);
}
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

## 许可证

MIT
