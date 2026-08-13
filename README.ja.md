# chunkedrs

[![Crates.io](https://img.shields.io/crates/v/chunkedrs?style=flat-square&logo=rust)](https://crates.io/crates/chunkedrs)
[![docs.rs](https://img.shields.io/docsrs/chunkedrs?style=flat-square&logo=docs.rs)](https://docs.rs/chunkedrs)
[![License](https://img.shields.io/crates/l/chunkedrs?style=flat-square)](LICENSE)
[![Downloads](https://img.shields.io/crates/d/chunkedrs?style=flat-square)](https://crates.io/crates/chunkedrs)

[English](README.md) | [简体中文](README.zh-CN.md) | **日本語**

RAG パイプライン向けのトークン精度テキストチャンキング — 再帰、Markdown 対応、セマンティック分割をサポート。最速の純 Rust BPE トークナイザー [tiktoken](https://crates.io/crates/tiktoken) を基盤に構築。

## 特徴

- **トークン精度** — すべてのチャンクがトークン上限内であることを保証（文字数近似ではない）
- **5 つの戦略** — 再帰、Markdown、コード、HTML、セマンティック（embedding ベース）
- **トークン区間** — 各チャンクが文書全体のトークン列における自分の範囲を知っている。[late chunking](#late-chunking) が必要とするのはこれ
- **CJK 対応** — 日本語・中国語が使わない ASCII 空白ではなく、`。`・`！`・`？` と読点で分割
- **セクション階層** — `### ソースから` のチャンクは自分が `## インストール` の下、`# ガイド` の下にいることを知っている
- **豊富なメタデータ** — バイトオフセット、トークンオフセット、トークン数、見出しの祖先チェーン
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
| Markdown | 見出しの祖先チェーンをメタデータに | CommonMark 構造で分割 |
| コード | 境界ベース、依存ゼロ | **tree-sitter、本物の AST** |
| HTML | 境界ベース、依存ゼロ | なし |
| トークン区間（late chunking 用） | あり | なし |
| embedding によるセマンティック分割 | あり（[embedrs](https://crates.io/crates/embedrs) 経由） | なし |
| バイトオフセット | あり | あり |
| オーバーラップ | トークン単位 | 設定した計量単位に従う |
| 依存クレート数 | 1（`tiktoken`） | コード分割は言語ごとに grammar クレート |

設定なしでトークン精度が欲しい、全チャンクに見出しの祖先チェーンが欲しい、late chunking 用のトークン区間が欲しい、embedding によるブレークポイントが欲しい場合は chunkedrs。**本物の AST ベースのコード分割が必要なら text-splitter** — chunkedrs の `code()` が読むのは構文ではなく約物であり、そのことを自ら明記しています。

## 分割戦略

| 戦略 | ユースケース | 速度 |
|------|------------|------|
| **再帰分割**（デフォルト） | 一般テキスト — 段落、文、節、単語で分割 | 最速 |
| **Markdown** | ヘッダー付きドキュメント — セクションの祖先チェーンを保持 | 高速 |
| **コード** | 任意言語のソース — 空行、ブロック終端、行 | 高速 |
| **HTML** | ウェブページ — ブロックレベルタグの境界 | 高速 |
| **セマンティック** | 高品質 RAG — embedding で意味境界を検出 | 低速（API 呼出） |

## クイックスタート

`Cargo.toml` に追加：

```toml
[dependencies]
chunkedrs = "2"
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
let markdown = "# ガイド\n\nはじめに。\n\n## インストール\n\ncargo add を実行。\n";
let chunks = chunkedrs::chunk(markdown).markdown().split();

// 各チャンクが所属セクションを認識し……
assert_eq!(chunks[0].section(), Some("# ガイド"));

// ……祖先チェーン全体も保持するので、入れ子のセクションが文脈を失わない
assert_eq!(chunks[1].section_path, ["# ガイド", "## インストール"]);
```

見出しは ATX（`#`）と setext（`===` / `---`）に対応。フェンス付きコードブロックは丸ごとスキップされるため、`bash` ブロック内の `#` はコメントのままです。YAML / TOML の front matter は見出し判定から除外されます。

## コードと HTML

```rust
let src = "fn a() {\n    one();\n}\n\nfn b() {\n    two();\n}\n";
let chunks = chunkedrs::chunk(src).code().max_tokens(20).split();

let page = "<h1>タイトル</h1><p>第一段落。</p><p>第二段落。</p>";
let chunks = chunkedrs::chunk(page).html().max_tokens(10).split();
```

どちらも **境界ベースであり、AST ベースではありません**。`code()` は空行 → 桁 0 でブロックを閉じる括弧 → 行 の順で分割し、散文用の区切りは意図的に使いません（`". "` でコードを切ると文字列リテラルの内部を切ってしまうため）。`html()` はブロックレベルの終了タグ（`</p>`、`</li>`、`</section>` など）を大文字小文字を区別せずに走査し、壊れたマークアップでは通常のテキスト階層に退化します。

いずれも解析は行いません。これがトレードオフです。任意の言語・任意のマークアップで動作し依存を一切増やしませんが、文字列リテラル内の `}` はブロック終端に見えます。AST の忠実さが必要な場合は、[text-splitter](https://crates.io/crates/text-splitter) の `CodeSplitter` など tree-sitter ベースの分割器をお使いください。

## Late chunking

各チャンクを単独で埋め込むと周囲の文脈が失われます — 「それは 40 ドルです」というチャンクは、もう「それ」が何だったかを知りません。[Late chunking](https://arxiv.org/abs/2409.04701) は順序を逆にします。まず文書全体を一度埋め込み、そのあと各チャンク自身のトークン区間でプーリングします。

chunkedrs が提供するのはその区間です：

```rust
let doc = "First sentence here. Second sentence here. Third sentence here.";
let chunks = chunkedrs::chunk(doc).max_tokens(8).split();

let encoder = tiktoken::get_encoding("o200k_base").unwrap();
let document_tokens = encoder.encode(doc);

for chunk in &chunks {
    // 実際のパイプラインではエンコーダーのトークン単位の隠れ状態を参照する
    let span = &document_tokens[chunk.start_token..chunk.end_token];
    assert!(!span.is_empty());
}
```

再トークン化も、チャンクの着地点の推測も不要です。[`examples/token_spans.rs`](examples/token_spans.rs) を参照。

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

```rust
#[non_exhaustive]
pub struct Chunk {
    pub content: String,           // テキスト内容
    pub index: usize,              // シーケンス内の位置
    pub start_byte: usize,         // 原文中のバイトオフセット
    pub end_byte: usize,           // バイトオフセット（排他）
    pub start_token: usize,        // 文書のトークン列におけるオフセット
    pub end_token: usize,          // トークンオフセット（排他）
    pub token_count: usize,        // このチャンク単独でのトークン数
    pub section_path: Vec<String>, // 見出しの祖先チェーン、外側から順に
}

impl Chunk {
    pub fn section(&self) -> Option<&str>;  // 最も深い見出し
}
```

`token_count` と `end_token - start_token` は別の問いに答えるもので、一致しないことがあります。`token_count` はチャンク自身を数え直した値 — `max_tokens` が制約するのも、コンテキストウィンドウの予算に使うのもこちらです。区間のほうは文書のトークン列におけるこのチャンクの占有範囲で、区切り文字がトークンの内部に落ちる場合はそれを覆うように広がります。そのため隣接する区間が 1 トークンを共有することがあります。

`Chunk` は `#[non_exhaustive]` なので、今後メタデータを追加してもメジャーバージョンにはなりません。手で組み立てる場合は `Chunk::new(...).with_bytes(..).with_tokens(..)` を使ってください。

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

## 語彙 feature

トークナイザーの語彙データはコンパイル後サイズの大半を占めますが、実際に使うのは通常ひとつかふたつです。語彙は **opt-out** です。デフォルトは 17 種すべてを含み、`default-features = false` では `o200k_base` だけが残ります — 解決できなかった名前が最終的にフォールバックする先なので、これが欠けることはありません。

```toml
# すべて（デフォルト）
chunkedrs = "2"

# o200k_base のみ — GPT-4o、GPT-5、o シリーズ
chunkedrs = { version = "2", default-features = false }

# o200k_base に Zhipu 系を追加
chunkedrs = { version = "2", default-features = false, features = ["vocab-zhipu"] }
```

`examples/basic` の release ビルドでの実測値：

| | サイズ |
|---|---:|
| 全語彙（デフォルト） | 7,100,544 |
| `default-features = false` | 2,695,104 |
| | **−62%** |

ベンダー単位のグループ：`vocab-openai`、`vocab-meta`、`vocab-deepseek`、`vocab-qwen`、`vocab-mistral`、`vocab-moonshot`、`vocab-zhipu`、`vocab-minimax`。語彙単位（`vocab-cl100k_base`、`vocab-llama3`、`vocab-glm5` など）でも指定できます。

**落とし穴がひとつ。** このビルドに含まれていない語彙を指名することと、名前を打ち間違えることは区別できません。どちらも黙って `o200k_base` にフォールバックするため、「もっともらしいが別のトークナイザーの」カウントが返ります。ビルドを絞る場合は、指名する encoding が有効化したものかどうかを確認してください。

Anthropic と Google は tiktoken 互換の語彙を公開していないため、`claude-*` と `gemini-*` は **解決されません**。認識されない名前は `o200k_base` にフォールバックし、正確なカウントではなく近似値になります。この挙動を明示的に固定したい場合は `.encoding()` を使ってください。

## 1.x からの移行

機械的な改名が 2 つと、API 形状の変更が 1 つです。

```rust
// 1.x
chunk.section.as_deref()          // Option<&str>
Chunk { content, index, .. }      // 構造体リテラル

// 2.0
chunk.section()                   // Option<&str> — 同じ値、メソッドになった
chunk.section_path                // Vec<String> — 祖先チェーン全体
Chunk::new(content).with_index(i) // Chunk は #[non_exhaustive]
```

`.semantic(&client)` は `SemanticChunkBuilder` を返すようになりました。これに対して `.split()` を呼んでいた場合、1.x では実行時 panic でしたが、いまはコンパイルエラーです。必要なのは `.split_async()` です。

1.0.x と比べてチャンク境界は変わります — 理由は [CHANGELOG](CHANGELOG.md) を参照してください。再チャンク化と再埋め込みを行ってください。バイトオフセットは原文に対して正確なままです。

<!-- ECOSYSTEM BEGIN (generated — edit ecosystem.toml, not this block) -->

## エコシステム

[tiktoken](https://crates.io/crates/tiktoken) · [@goliapkg/tiktoken-wasm](https://www.npmjs.com/package/@goliapkg/tiktoken-wasm) · [instructors](https://crates.io/crates/instructors) · **chunkedrs** · [embedrs](https://crates.io/crates/embedrs)

<!-- ECOSYSTEM END -->

## ライセンス

MIT
