# chunkedrs

[![Crates.io](https://img.shields.io/crates/v/chunkedrs?style=flat-square&logo=rust)](https://crates.io/crates/chunkedrs)
[![docs.rs](https://img.shields.io/docsrs/chunkedrs?style=flat-square&logo=docs.rs)](https://docs.rs/chunkedrs)
[![License](https://img.shields.io/crates/l/chunkedrs?style=flat-square)](LICENSE)
[![Downloads](https://img.shields.io/crates/d/chunkedrs?style=flat-square)](https://crates.io/crates/chunkedrs)

[English](README.md) | **简体中文** | [日本語](README.ja.md)

面向 RAG 管道的 token 精确文本分块 — 支持递归、Markdown 感知和语义分割。基于 [tiktoken](https://crates.io/crates/tiktoken)，最快的纯 Rust BPE 分词器。

## 特性亮点

- **Token 精确** — 每个分块严格保证不超过 token 上限，而非近似的字符数估算
- **5 种策略** — 递归、Markdown、代码、HTML、语义（基于 embedding）
- **Token 区间** — 每个分块都知道自己在整篇文档 token 流中的位置，这正是 [late chunking](#late-chunking) 需要的
- **中日文感知** — 中文和日文按 `。`、`！`、`？` 及子句标点切分，而不是按它们根本不用的 ASCII 空格
- **章节层级** — `### 从源码安装` 的分块知道自己在 `## 安装` 之下、`# 指南` 之下
- **丰富元数据** — 字节偏移、token 偏移、token 计数、标题祖先链
- **重叠支持** — 可配置的 token 级重叠
- **任意分词器** — 从模型名称自动检测，或直接指定 17 种编码之一
- **基于 tiktoken 4** — 最快的纯 Rust BPE 分词器，覆盖 11 家厂商

## 为什么选择 chunkedrs？

RAG 管道需要将文本分割成适合模型上下文窗口的片段。简单的按字符数或固定大小分割会在词中间、句子中间甚至段落中间断开 — 破坏语义，降低检索质量。

chunkedrs 在**语义边界**（段落、句子、子句、单词）处分割，同时保证**精确的 token 限制**。没有任何分块会超过 `max_tokens`。

### 与 [text-splitter](https://crates.io/crates/text-splitter) 的对比

两者都做得不错，只是默认取向不同。text-splitter 是更宽的工具箱 —— 它有基于 tree-sitter 的代码分割，chunkedrs 没有。

| | chunkedrs | text-splitter 0.32 |
|---|---|---|
| 默认计量单位 | 恒为 token | 字符；token 计量需另行配置 |
| 分词器 | 内置 tiktoken | 通过 feature 接 `tiktoken-rs` 或 `tokenizers` |
| Markdown | 标题祖先链作为分块元数据 | 按 CommonMark 结构分割 |
| 代码 | 边界感知，零依赖 | **tree-sitter,真 AST** |
| HTML | 边界感知，零依赖 | 无 |
| Token 区间（late chunking 用） | 有 | 无 |
| 基于 embedding 的语义分割 | 有（经 [embedrs](https://crates.io/crates/embedrs)） | 无 |
| 字节偏移 | 有 | 有 |
| 重叠 | 按 token | 跟随所配置的计量单位 |
| 依赖数 | 1（`tiktoken`） | 代码分割需每语言一个 grammar crate |

想要「不用配置就是 token 精确」、每块都带标题祖先链、late chunking 用的 token 区间、或基于 embedding 的断点检测，选 chunkedrs。**需要真正的 AST 级代码分割，选 text-splitter** —— chunkedrs 的 `code()` 读的是标点不是语法，这一点它自己写明了。

## 分割策略

| 策略 | 适用场景 | 速度 |
|------|---------|------|
| **递归分割**（默认） | 通用文本 — 按段落、句子、子句、单词 | 最快 |
| **Markdown** | 含标题的文档 — 保留章节祖先链 | 快 |
| **代码** | 任意语言源码 — 空行、块结束符、行 | 快 |
| **HTML** | 网页 — 块级标签边界 | 快 |
| **语义分割** | 高质量 RAG — 基于 embedding 在语义边界分割 | 较慢（需 API 调用） |

## 快速开始

添加到 `Cargo.toml`：

```toml
[dependencies]
chunkedrs = "2"
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
    .model("gpt-5.6-terra")
    .split();

// 每个分块保证 <= 256 tokens
assert!(chunks.iter().all(|c| c.token_count <= 256));
```

## Markdown 感知分割

```rust
let markdown = "# 指南\n\n介绍。\n\n## 安装\n\n执行 cargo add。\n";
let chunks = chunkedrs::chunk(markdown).markdown().split();

// 每个分块知道它属于哪个章节……
assert_eq!(chunks[0].section(), Some("# 指南"));

// ……以及完整的祖先链，嵌套章节因此保留上下文
assert_eq!(chunks[1].section_path, ["# 指南", "## 安装"]);
```

标题支持 ATX（`#`）和 setext（`===` / `---`）。围栏代码块整体跳过，所以 `bash` 块里的 `#` 注释就是注释；YAML 和 TOML front matter 不参与标题识别。

## 代码与 HTML

```rust
let src = "fn a() {\n    one();\n}\n\nfn b() {\n    two();\n}\n";
let chunks = chunkedrs::chunk(src).code().max_tokens(20).split();

let page = "<h1>标题</h1><p>第一段。</p><p>第二段。</p>";
let chunks = chunkedrs::chunk(page).html().max_tokens(10).split();
```

两者都是**边界感知,而非 AST 感知**。`code()` 依次按空行、列 0 上闭合块的括号、行来切,并刻意跳过散文分隔符 —— 因为按 `". "` 切代码会切进字符串字面量。`html()` 大小写不敏感地扫描块级闭合标签（`</p>`、`</li>`、`</section>` 等）,遇到畸形标记则退化到普通文本阶梯。

两者都不做解析。这就是取舍:它们适用于任何语言、任何标记,且不引入任何依赖,但字符串字面量里的 `}` 在它看来就是块结束。需要 AST 保真度时,请用基于 tree-sitter 的分割器,例如 [text-splitter](https://crates.io/crates/text-splitter) 的 `CodeSplitter`。

## Late chunking

孤立地嵌入每个分块会丢失周围的上下文 —— 一个写着「它要 40 美元」的分块已经不知道「它」是什么了。[Late chunking](https://arxiv.org/abs/2409.04701) 把顺序反过来:先把整篇文档嵌入一次,再按每个分块自己那段 token 区间做池化。

chunkedrs 提供的正是那段区间:

```rust
let doc = "First sentence here. Second sentence here. Third sentence here.";
let chunks = chunkedrs::chunk(doc).max_tokens(8).split();

let encoder = tiktoken::get_encoding("o200k_base").unwrap();
let document_tokens = encoder.encode(doc);

for chunk in &chunks {
    // 真实管道里这里索引的是编码器的逐 token 隐状态
    let span = &document_tokens[chunk.start_token..chunk.end_token];
    assert!(!span.is_empty());
}
```

不用重新分词,不用猜分块落在哪。见 [`examples/token_spans.rs`](examples/token_spans.rs)。

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
let client = embedrs::Client::openai("sk-...");
let chunks = chunkedrs::chunk("你的长文本...")
    .semantic(&client)
    .threshold(0.5)
    .split_async()
    .await?;
```

## 分块元数据

```rust
#[non_exhaustive]
pub struct Chunk {
    pub content: String,           // 文本内容
    pub index: usize,              // 在序列中的位置
    pub start_byte: usize,         // 原文中的字节偏移
    pub end_byte: usize,           // 字节偏移（不含）
    pub start_token: usize,        // 在文档 token 流中的偏移
    pub end_token: usize,          // token 偏移（不含）
    pub token_count: usize,        // 本分块单独计数的 token 数
    pub section_path: Vec<String>, // 标题祖先链，由外到内
}

impl Chunk {
    pub fn section(&self) -> Option<&str>;  // 最深一级标题
}
```

`token_count` 和 `end_token - start_token` 回答的是两个不同问题，可以不相等。`token_count` 是对分块本身的重新计数 —— `max_tokens` 约束的是它，你拿去算上下文窗口预算的也是它。区间则是这个分块在文档 token 流中的占位；当分隔符落在某个 token 内部时，区间会向外扩展去覆盖它，因此相邻区间可能共享一个 token。

`Chunk` 标了 `#[non_exhaustive]`，以后再加元数据不会再是一次 major。手工构造用 `Chunk::new(...).with_bytes(..).with_tokens(..)`。

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
let chunks = chunkedrs::chunk(text).model("gpt-5.6-terra").split();

// 或直接指定 17 种编码之一
let chunks = chunkedrs::chunk(text).encoding("cl100k_base").split();

// 默认：o200k_base（GPT-4o, GPT-5, o 系列）
```

模型名覆盖 OpenAI、Meta（`llama-3.1-70b`）、DeepSeek（`deepseek-v4`）、
阿里（`qwen2.5-72b`）、Mistral、月之暗面（`kimi-k2`）、智谱（`glm-5`）、
MiniMax（`minimax-m2`）。

## 词表 feature

分词器词表占了编译体积的绝大部分，而大多数项目只用其中一两个。词表是
**opt-out** 的：默认带全部 17 种编码，`default-features = false` 只保留
`o200k_base` —— 它是所有无法解析的名字最终回落到的编码器，因此永远不会缺席。

```toml
# 全部（默认）
chunkedrs = "2"

# 只要 o200k_base —— GPT-4o、GPT-5、o 系列
chunkedrs = { version = "2", default-features = false }

# o200k_base 加上智谱系列
chunkedrs = { version = "2", default-features = false, features = ["vocab-zhipu"] }
```

实测 `examples/basic`，release 构建：

| | 体积 |
|---|---:|
| 全部词表（默认） | 7,100,544 |
| `default-features = false` | 2,695,104 |
| | **−62%** |

按厂商分组：`vocab-openai`、`vocab-meta`、`vocab-deepseek`、`vocab-qwen`、
`vocab-mistral`、`vocab-moonshot`、`vocab-zhipu`、`vocab-minimax`。
也可按单个词表启用：`vocab-cl100k_base`、`vocab-llama3`、`vocab-glm5` 等。

**一个尖角。** 请求一个本次构建没有编进来的词表，和把名字拼错是分不出来的 ——
两者都会静默回落到 `o200k_base`，于是你拿到的是一个看起来合理、但来自错误
分词器的计数。精简构建时，请确认你点名的编码就是你启用过的那些。

Anthropic 和 Google 没有公开 tiktoken 兼容的词表，所以 `claude-*` 和
`gemini-*` **不会**匹配 —— 无法识别的名字会回落到 `o200k_base`，那是估算
而非精确计数。需要显式固定这一行为时请用 `.encoding()`。

## 从 1.x 升级

两处机械改名，加一处 API 形态变化。

```rust
// 1.x
chunk.section.as_deref()          // Option<&str>
Chunk { content, index, .. }      // 结构体字面量

// 2.0
chunk.section()                   // Option<&str> —— 同一个值，现在是方法
chunk.section_path                // Vec<String> —— 完整祖先链
Chunk::new(content).with_index(i) // Chunk 已标 #[non_exhaustive]
```

`.semantic(&client)` 现在返回 `SemanticChunkBuilder`。如果你之前在它上面调 `.split()`，那在 1.x 里是运行时 panic；现在它是编译错误，你要的是 `.split_async()`。

相对 1.0.x，分块边界会变 —— 原因见 [CHANGELOG](CHANGELOG.md)。重新分块、重新嵌入即可；字节偏移对原文依然精确。

<!-- ECOSYSTEM BEGIN (generated — edit ecosystem.toml, not this block) -->

## 生态系统

[tiktoken](https://crates.io/crates/tiktoken) · [@goliapkg/tiktoken-wasm](https://www.npmjs.com/package/@goliapkg/tiktoken-wasm) · [instructors](https://crates.io/crates/instructors) · **chunkedrs** · [embedrs](https://crates.io/crates/embedrs)

<!-- ECOSYSTEM END -->

## 许可证

MIT
