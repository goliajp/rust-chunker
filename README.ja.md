# chunkedrs

[![Crates.io](https://img.shields.io/crates/v/chunkedrs?style=flat-square&logo=rust)](https://crates.io/crates/chunkedrs)
[![docs.rs](https://img.shields.io/docsrs/chunkedrs?style=flat-square&logo=docs.rs)](https://docs.rs/chunkedrs)
[![License](https://img.shields.io/crates/l/chunkedrs?style=flat-square)](LICENSE)
[![Downloads](https://img.shields.io/crates/d/chunkedrs?style=flat-square)](https://crates.io/crates/chunkedrs)

[English](README.md) | [简体中文](README.zh-CN.md) | **日本語**

RAG パイプライン向けのトークン精度テキストチャンキング — 再帰、Markdown 対応、セマンティック分割をサポート。最速の純 Rust BPE トークナイザー [tiktoken](https://crates.io/crates/tiktoken) を基盤に構築。

## 特徴

- **トークン精度** — すべてのチャンクがトークン上限内であることを保証（文字数近似ではない）
- **3 つの戦略** — 再帰（高速・汎用）、Markdown 対応（ヘッダー構造を保持）、セマンティック（embedding ベースのブレークポイント検出）
- **CJK 対応** — 日本語・中国語が使わない ASCII 空白ではなく、`。`・`！`・`？` と読点で分割
- **豊富なメタデータ** — バイトオフセット、トークン数、セクションヘッダーを各チャンクに付与
- **オーバーラップ** — 設定可能なトークンオーバーラップ
- **任意のトークナイザー** — モデル名から自動検出、または 17 種のエンコーディングを直接指定
- **tiktoken 4 基盤** — 10 プロバイダーを網羅する最速の純 Rust BPE トークナイザー

## なぜ chunkedrs？

RAG パイプラインでは、テキストをモデルのコンテキストウィンドウに収まるチャンクに分割する必要があります。文字数や固定サイズでの単純な分割は、単語、文、段落の途中で切断され、意味が壊れ、検索品質が低下します。

chunkedrs は**意味的な境界**（段落、文、節、単語）で分割しつつ、**正確なトークン制限**を保証します。`max_tokens` を超えるチャンクは生成されません。

### [text-splitter](https://crates.io/crates/text-splitter) との比較

どちらも良いライブラリで、狙うデフォルトが違います。text-splitter はより広いツールキットで、tree-sitter によるコード分割を備えています（chunkedrs にはありません）。

| | chunkedrs | text-splitter 0.32 |
|---|---|---|
| デフォルトの計量単位 | 常にトークン | 文字。トークン計量は別途設定 |
| トークナイザー | tiktoken を内蔵 | feature 経由で `tiktoken-rs` / `tokenizers` |
| Markdown | ヘッダーをチャンクのメタデータに | CommonMark 構造で分割 |
| コード（tree-sitter） | なし | あり |
| embedding によるセマンティック分割 | あり（[embedrs](https://crates.io/crates/embedrs) 経由） | なし |
| バイトオフセット | あり | あり |
| オーバーラップ | トークン単位 | 設定した計量単位に従う |

設定なしでトークン精度が欲しい、全チャンクにセクション情報が欲しい、embedding によるブレークポイントが欲しい場合は chunkedrs。ソースコードを分割したい場合は text-splitter。

## 分割戦略

| 戦略 | ユースケース | 速度 |
|------|------------|------|
| **再帰分割**（デフォルト） | 一般テキスト — 段落、文、節、単語で分割 | 最速 |
| **Markdown** | ヘッダー付きドキュメント — セクション情報を保持 | 高速 |
| **セマンティック** | 高品質 RAG — embedding で意味境界を検出 | 低速（API 呼出） |

## クイックスタート

`Cargo.toml` に追加：

```toml
[dependencies]
chunkedrs = "1.1"
```

デフォルト設定で分割（再帰、最大 512 トークン、オーバーラップなし）：

```rust
use chunkedrs::Chunk;

let chunks: Vec<Chunk> = chunkedrs::chunk("長いテキスト...").split();
for chunk in &chunks {
    println!("[{}] {} tokens (bytes {}..{})", chunk.index, chunk.token_count, chunk.start_byte, chunk.end_byte);
}
// 出力:
// [0] 6 tokens (bytes 0..21)
```

## トークン精度の分割

```rust
let chunks = chunkedrs::chunk("長いテキスト...")
    .max_tokens(256)
    .overlap(50)
    .model("gpt-4o")
    .split();

// すべてのチャンクが <= 256 トークンであることを保証
assert!(chunks.iter().all(|c| c.token_count <= 256));
```

## Markdown 対応分割

```rust
let markdown = "# はじめに\n\nテキスト。\n\n## 詳細\n\n追加コンテンツ。\n";
let chunks = chunkedrs::chunk(markdown).markdown().split();

// 各チャンクが所属セクションを認識
assert_eq!(chunks[0].section.as_deref(), Some("# はじめに"));
```

## 日本語・中国語テキスト

日本語と中国語は句点のあとに空白を置かないため、区切り文字がすべて ASCII 空白で終わる分割器は境界を一つも見つけられず、単語の途中で切ってしまいます。chunkedrs はこれらの文字体系が実際に使う記号で分割し、閉じ括弧はそれが閉じる文と一緒に保ちます：

```rust
let ja = "彼は「今日はいい天気ですね。」と言いました。彼女は「本当にそうですね。」と答えました。";
let chunks = chunkedrs::chunk(ja).max_tokens(14).split();

assert_eq!(chunks[0].content, "彼は「今日はいい天気ですね。」と言いました。");
assert_eq!(chunks[1].content, "彼女は「本当にそうですね。」と答えました。");
```

一文が収まらない場合は読点（`、`・`，`・`；`）が次の境界となり、そこで初めてトークンオフセットにフォールバックします。

## セマンティック分割

`semantic` feature を有効にすると、embedding を使って意味境界で分割：

```toml
[dependencies]
chunkedrs = { version = "1.1", features = ["semantic"] }
```

```rust,ignore
let client = embedrs::openai("sk-...");
let chunks = chunkedrs::chunk("長いテキスト...")
    .semantic(&client)
    .threshold(0.5)
    .split_async()
    .await?;
```

## チャンクメタデータ

各 `Chunk` にはメタデータが付与されます：

```rust
pub struct Chunk {
    pub content: String,         // テキスト内容
    pub index: usize,            // シーケンス内の位置
    pub start_byte: usize,       // 原文中のバイトオフセット
    pub end_byte: usize,         // バイトオフセット（排他）
    pub token_count: usize,      // 正確なトークン数
    pub section: Option<String>, // markdown ヘッダー（該当時）
}
```

## オーバーラップ

連続チャンク間のトークンオーバーラップは、境界を越えてコンテキストを運びます：

```rust
let chunks = chunkedrs::chunk("長いテキスト...")
    .max_tokens(256)
    .overlap(50)
    .split();
```

オーバーラップが効くかどうかはワークロード次第です。近年の検索評価では、計測可能な効果はほとんど報告されない一方、インデックスサイズのコストは実在します。デフォルトは無効です。有効にする前に計測してください。

## トークナイザーの選択

```rust
// モデル名から自動検出
let chunks = chunkedrs::chunk(text).model("gpt-4o").split();

// 17 種のエンコーディングを直接指定
let chunks = chunkedrs::chunk(text).encoding("cl100k_base").split();

// デフォルト：o200k_base（GPT-4o, GPT-5, o シリーズ）
```

モデル名は OpenAI、Meta（`llama-3.1-70b`）、DeepSeek（`deepseek-v4`）、Alibaba（`qwen2.5-72b`）、Mistral、Moonshot（`kimi-k2`）、Zhipu（`glm-5`）、MiniMax（`minimax-m2`）に対応します。

Anthropic と Google は tiktoken 互換の語彙を公開していないため、`claude-*` と `gemini-*` は **解決されません**。認識されない名前は `o200k_base` にフォールバックし、正確なカウントではなく近似値になります。この挙動を明示的に固定したい場合は `.encoding()` を使ってください。

<!-- ECOSYSTEM BEGIN (generated — edit ecosystem.toml, not this block) -->

## エコシステム

[tiktoken](https://crates.io/crates/tiktoken) · [@goliapkg/tiktoken-wasm](https://www.npmjs.com/package/@goliapkg/tiktoken-wasm) · [instructors](https://crates.io/crates/instructors) · **chunkedrs** · [embedrs](https://crates.io/crates/embedrs)

<!-- ECOSYSTEM END -->

## ライセンス

MIT
