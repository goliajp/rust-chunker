# chunkedrs

AI ネイティブなテキストチャンキング — 長いドキュメントをトークン精度で分割し、embedding と検索に最適化。

[tiktoken](https://crates.io/crates/tiktoken) による正確なトークンカウント。すべてのチャンクがトークン上限を厳守します。

## なぜ chunkedrs？

RAG パイプラインでは、テキストをモデルのコンテキストウィンドウに収まるチャンクに分割する必要があります。文字数や固定サイズでの単純な分割は、単語、文、段落の途中で切断され、意味が壊れ、検索品質が低下します。

chunkedrs は**意味的な境界**（段落 → 文 → 単語）で分割しつつ、**正確なトークン制限**を保証します。`max_tokens` を超えるチャンクは生成されません。

## 分割戦略

| 戦略 | ユースケース | 速度 |
|------|------------|------|
| **再帰分割**（デフォルト） | 一般テキスト — 段落、文、単語で分割 | 最速 |
| **Markdown** | `#` ヘッダー付きドキュメント — セクション情報を保持 | 高速 |
| **セマンティック** | 高品質 RAG — embedding で意味境界を検出 | 低速（API 呼出） |

## クイックスタート

```rust
// デフォルト：再帰分割、最大 512 トークン、オーバーラップなし
let chunks = chunkedrs::chunk("長いテキスト...").split();
for chunk in &chunks {
    println!("[{}] {} tokens", chunk.index, chunk.token_count);
}
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

## ライセンス

MIT
