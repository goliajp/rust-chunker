# chunkedrs

[![Crates.io](https://img.shields.io/crates/v/chunkedrs?style=flat-square&logo=rust)](https://crates.io/crates/chunkedrs)
[![docs.rs](https://img.shields.io/docsrs/chunkedrs?style=flat-square&logo=docs.rs)](https://docs.rs/chunkedrs)
[![License](https://img.shields.io/crates/l/chunkedrs?style=flat-square)](LICENSE)
[![Downloads](https://img.shields.io/crates/d/chunkedrs?style=flat-square)](https://crates.io/crates/chunkedrs)
[![MSRV](https://img.shields.io/badge/MSRV-1.94-blue?style=flat-square)](https://www.rust-lang.org)

[English](README.md) | [简体中文](README.zh-CN.md) | **日本語**

RAG パイプライン向けのトークン精度テキストチャンキング — 再帰、Markdown 対応、セマンティック分割をサポート。最速の純 Rust BPE トークナイザー [tiktoken](https://crates.io/crates/tiktoken) を基盤に構築。

## 特徴

- **トークン精度** — すべてのチャンクがトークン上限内であることを保証（文字数近似ではない）
- **3 つの戦略** — 再帰（高速・汎用）、Markdown 対応（ヘッダー構造を保持）、セマンティック（embedding ベースのブレークポイント検出）
- **豊富なメタデータ** — バイトオフセット、トークン数、セクションヘッダーを各チャンクに付与
- **オーバーラップ** — 検索コンテキスト保持のための設定可能なトークンオーバーラップ
- **任意のトークナイザー** — モデル名から自動検出（`gpt-4o`、`claude`、`llama`）またはエンコーディングを直接指定
- **tiktoken 基盤** — 主要 LLM 全 9 エンコーディングに対応する最速の純 Rust BPE トークナイザー

## なぜ chunkedrs？

RAG パイプラインでは、テキストをモデルのコンテキストウィンドウに収まるチャンクに分割する必要があります。文字数や固定サイズでの単純な分割は、単語、文、段落の途中で切断され、意味が壊れ、検索品質が低下します。

chunkedrs は**意味的な境界**（段落、文、単語）で分割しつつ、**正確なトークン制限**を保証します。`max_tokens` を超えるチャンクは生成されません。

| 機能 | chunkedrs | text-splitter | 手動実装 |
|------|-----------|---------------|---------|
| トークン精度の制限 | あり（tiktoken） | 文字数ベース | なし |
| 再帰分割 | あり | あり | 自前実装 |
| Markdown 対応 | あり（セクション情報付き） | なし | 自前実装 |
| セマンティック分割 | あり（embedrs 経由） | なし | 自前実装 |
| バイトオフセット | あり | なし | 自前実装 |
| チャンク毎トークン数 | あり | なし | 自前実装 |
| オーバーラップ | トークンレベル | 文字レベル | 自前実装 |
| トークナイザー選択 | モデル名またはエンコーディング | 非対応 | 非対応 |

## 分割戦略

| 戦略 | ユースケース | 速度 |
|------|------------|------|
| **再帰分割**（デフォルト） | 一般テキスト — 段落、文、単語で分割 | 最速 |
| **Markdown** | `#` ヘッダー付きドキュメント — セクション情報を保持 | 高速 |
| **セマンティック** | 高品質 RAG — embedding で意味境界を検出 | 低速（API 呼出） |

## クイックスタート

`Cargo.toml` に追加：

```toml
[dependencies]
chunkedrs = "1"
```

デフォルト設定で分割（再帰、最大 512 トークン、オーバーラップなし）：

```rust
use chunkedrs::Chunk;

let chunks: Vec<Chunk> = chunkedrs::chunk("長いテキスト...").split();
for chunk in &chunks {
    println!("[{}] {} tokens (bytes {}..{})", chunk.index, chunk.token_count, chunk.start_byte, chunk.end_byte);
}
// 出力:
// [0] 5 tokens (bytes 0..22)
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

## セマンティック分割

`semantic` feature を有効にすると、embedding を使って意味境界で分割：

```toml
[dependencies]
chunkedrs = { version = "1", features = ["semantic"] }
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

連続チャンク間のトークンオーバーラップにより、境界でのコンテキストが保持されます — 検索品質に不可欠：

```rust
let chunks = chunkedrs::chunk("長いテキスト...")
    .max_tokens(256)
    .overlap(50)
    .split();
```

## トークナイザーの選択

```rust
// モデル名から自動検出
let chunks = chunkedrs::chunk(text).model("gpt-4o").split();

// エンコーディングを直接指定
let chunks = chunkedrs::chunk(text).encoding("cl100k_base").split();

// デフォルト：o200k_base（GPT-4o, GPT-4-turbo）
```

<!-- ECOSYSTEM BEGIN (synced by claws/opensource/scripts/sync-ecosystem.py — edit ecosystem.toml, not this block) -->

## エコシステム

GOLIA の Rust AI インフラ系列の一員 — 各々独立したリポジトリで、crates.io 経由で組み合わせ:

| Crate / Package | リポジトリ | 説明 |
|---|---|---|
| [tiktoken](https://crates.io/crates/tiktoken) | [rust-tiktoken](https://github.com/goliajp/rust-tiktoken) | 高性能 BPE トークナイザー — 9 エンコーディング、57 モデル、各社料金 |
| [@goliapkg/tiktoken-wasm](https://www.npmjs.com/package/@goliapkg/tiktoken-wasm) | [rust-tiktoken](https://github.com/goliajp/rust-tiktoken) | tiktoken の WASM バインディング — ブラウザ / Node.js |
| [instructors](https://crates.io/crates/instructors) | [rust-instructor](https://github.com/goliajp/rust-instructor) | LLM からの型安全な構造化出力抽出 |
| [embedrs](https://crates.io/crates/embedrs) | [rust-embeddings](https://github.com/goliajp/rust-embeddings) | 統一 embedding — クラウド API + ローカル推論、単一インターフェース |
| **chunkedrs**（本 crate） | [rust-chunker](https://github.com/goliajp/rust-chunker) | AI ネイティブテキストチャンキング — 再帰、Markdown 対応、セマンティック |

<!-- ECOSYSTEM END -->

## ライセンス

MIT
