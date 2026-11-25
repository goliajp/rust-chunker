# chunkedrs

[![Crates.io](https://img.shields.io/crates/v/chunkedrs?style=flat-square&logo=rust)](https://crates.io/crates/chunkedrs)
[![docs.rs](https://img.shields.io/docsrs/chunkedrs?style=flat-square&logo=docs.rs)](https://docs.rs/chunkedrs)
[![CI](https://img.shields.io/github/actions/workflow/status/goliajp/rust-chunker/ci.yml?branch=develop&style=flat-square&logo=github&label=ci)](https://github.com/goliajp/rust-chunker/actions/workflows/ci.yml)
[![License](https://img.shields.io/crates/l/chunkedrs?style=flat-square)](#ライセンス)

[English](README.md) | [简体中文](README.zh-CN.md) | **日本語**

> **2.1 からの更新**：API の変更はありませんが、チャンクの境界が変わるため、作成済みのインデックスはチャンク分割をやり直してください。tiktoken 5 で Llama 4、Qwen 3.5〜3.8、Mistral Large 4、`deepseek-flash` が実際の語彙を使うようになり、これらのモデルのトークン数が変わります。新しい分割アルゴリズムでも、一部のチャンクが 2.1 と異なる位置で終わります。32 KiB 以上の文書は複数のスレッドでエンコードされるようになりました。`TIKTOKEN_THREADS=1` で無効にできます。詳細は [CHANGELOG](CHANGELOG.md#upgrading-from-21) を参照してください。

chunkedrs は、長い文書をトークン数の上限に収まるチャンクに分割するライブラリです。埋め込みと検索での利用を想定しています。トークン数は [tiktoken](https://crates.io/crates/tiktoken) で数え、20 種類のエンコーディング（OpenAI、Llama、DeepSeek、Qwen、Mistral、Kimi、GLM、MiniMax）に対応します。各チャンクは、元の文書でのバイト範囲と、文書全体のトークン列での範囲を持ちます。依存は tiktoken だけで、`semantic` feature を有効にすると [embedrs](https://crates.io/crates/embedrs) が加わります。

Apple M4 Pro のシングルスレッド、`o200k_base`、1 チャンク 512 トークンの条件で、200 KB の英語の本を 0.35 ms で分割します。同じ条件で text-splitter 0.33.6 は、同じトークナイザーで 1.19 ms、tiktoken-rs 0.12（`tiktoken-rs` feature が使うトークナイザー）で 34.8 ms かかります。

## 特徴

- **5 つの分割方式**：再帰（段落、行、文、節、単語）、見出しの階層を保持する Markdown、境界の記号で分けるコードと HTML、埋め込みの類似度で分けるセマンティック分割
- **トークン上限を厳密に守る**：`token_count` はチャンク自身のテキストを単独で数えた値で、`max_tokens` を超えません。例外は、1 文字だけで上限を超える場合のみです
- **位置情報**：すべてのチャンクで `&doc[start_byte..end_byte] == content` が成り立ちます。`start_token..end_token` は `encode(doc)` の中でのチャンクの範囲で、[late chunking](#late-chunking) に必要なのはこの範囲です
- **日本語と中国語**：空白が続かない `。！？` で文を、`，、；：` で節を区切ります。閉じかぎ括弧はその文の側に残ります
- **一括処理**：`Chunker` はエンコーダーを一度だけ解決し、`split_batch` は複数の文書をすべての CPU コアに分けて処理します

## クイックスタート

```toml
[dependencies]
chunkedrs = "2"
```

```rust
// 既定：再帰分割、512 トークン、オーバーラップなし、o200k_base
let chunks = chunkedrs::chunk("your long text here...").split();
for c in &chunks {
    println!("[{}] {} tokens, bytes {}..{}", c.index, c.token_count, c.start_byte, c.end_byte);
}

// モデル名からエンコーディングを選ぶ
let chunks = chunkedrs::chunk("your long text here...")
    .max_tokens(256)
    .overlap(32)
    .model("gpt-5.6-terra")
    .split();
assert!(chunks.iter().all(|c| c.token_count <= 256));

// 1 つの設定で複数の文書を処理する
let chunker = chunkedrs::Chunker::new().max_tokens(256).markdown();
let docs = ["# A\n\nFirst.", "# B\n\nSecond."];
let all = chunker.split_batch(&docs);
assert_eq!(all[1][0].section(), Some("# B"));
```

## 性能

数値は文書 1 つの分割にかかるミリ秒で、ウォームアップ後 9 ラウンドの中央値です。Apple M4 Pro の Mac mini、rustc 1.100、LTO を有効にした release ビルド、`TIKTOKEN_THREADS=1`、`o200k_base`、オーバーラップなしで測定しました。測定中のロードアベレージは 1.3〜2.0、測定日は 2026-11-25 です。text-splitter には容量として同じ `max_tokens` を渡し、ほかは既定値です。コード分割には tree-sitter-rust 0.24 を使っています。

| コーパス | 方式 | max_tokens | `encode` 1 回 | chunkedrs | text-splitter + tiktoken 5.5 | text-splitter + tiktoken-rs 0.12 |
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

`book_en.txt` は『高慢と偏見』の 200,000 バイト、`zh_xiyouji.md` は Wikisource の『西遊記』第 1〜9 回を Markdown にした 178,622 バイト、`md_rust_book.md` は『The Rust Programming Language』第 3〜6 章の 174,237 バイト、`code_rust.txt` は 100,000 バイトの Rust のソースコードです。いずれも [`bench-compare/data`](bench-compare/data) にあります。

- 上の表のすべてのチャンクを tiktoken で数え直しました。chunkedrs にも text-splitter にも上限を超えるチャンクはありません
- chunkedrs の時間は、文書全体を 1 回 `encode` する時間の 2.9〜5.1 倍です。内訳は、文書全体のエンコード（`encode` 1 回分）、各トークンのバイト長の取得（`encode` の約半分）、各チャンクの単独での計数（`encode` 約 1 回分。BPE の結果は切れ目の前後で足し算にならないため）で、残りが境界探しとチャンクの組み立てです
- `cl100k_base` でもほぼ同じで、chunkedrs は 0.19〜0.72 ms です。tiktoken の既定のマルチスレッドを使うと、100〜200 KB のこれらの文書は複数のスレッドでエンコードされ、chunkedrs は 0.17〜0.45 ms になります
- `Chunker::split_batch` で 64 文書（4 つのコーパスを各 16 回、合計 10.4 MB）を処理すると、1 つずつでは 28.2 ms、一括では 3.9 ms です

再現するには `TIKTOKEN_THREADS=1 cargo run --release -p bench-compare` を実行してください。

## 分割の仕組み

文書全体を一度だけエンコードします。チャンクの開始位置から文書のトークンを `max_tokens` 個数えた位置が、そのチャンクが終われる最も遠いバイト位置です。チャンクはその範囲の中で、存在する境界のうち優先度が最も高い種類の最後の位置で終わります。

| 方式 | 境界（優先度の高い順）|
|---|---|
| 再帰（既定）| 空行；改行；空白が続く `. ! ?` と `。！？．｡…‥`；`; , :` と `；，、：･`；空白、タブ、全角空白 |
| `markdown()` | まず見出し行（ATX の `#` と setext の `===` / `---`。コードブロックと front matter の中は除く）、各セクションの中は再帰の境界 |
| `code()` | 空行；0 桁目に単独で置かれた `}` `};` `)` `]` `];` `end`；改行；空白、タブ |
| `html()` | ブロック要素の終了タグの後（`</p>`、`</li>`、`</section>`、`</h1>` など）、その後は再帰の境界 |
| `semantic()` | 隣り合う文の埋め込みの類似度がしきい値を下回る位置、その後は再帰の境界 |

範囲内に境界が一つもない場合は、トークンの境界で終わり、文字の境界まで戻します。選んだテキストはその後に単独で数えます。BPE は切れ目をまたいで結合するため、単独でエンコードしたテキストは、文書全体の中で占めるトークン数より 1 つ多くなることも少なくなることもあります。まれに上限を超えた場合は、範囲を縮めて選び直します。

コードと HTML の分割は記号とタグだけを見て、構文解析はしません。依存も増えませんが、文字列リテラルの中の `}` もブロックの終わりとして扱います。構文に沿ってコードを分割したい場合は、text-splitter の tree-sitter ベースの `CodeSplitter` を使ってください。

## Markdown

```rust
let md = "# Guide\n\nIntro.\n\n## Install\n\nRun cargo add.\n";
let chunks = chunkedrs::chunk(md).markdown().split();

assert_eq!(chunks[0].section(), Some("# Guide"));
assert_eq!(chunks[1].section_path, ["# Guide", "## Install"]);
```

見出し行は `content` ではなく `section_path` に入ります。そのため、全チャンクの内容をつなげても見出し行は復元されません。

## 日本語と中国語

```rust
let zh = "他说「今天天气很好。」然后就出门了。她回答说「确实不错。」于是也跟着出去了。";
let chunks = chunkedrs::chunk(zh).max_tokens(12).split();

assert_eq!(chunks[0].content, "他说「今天天气很好。」");
assert_eq!(chunks[1].content, "然后就出门了。");
```

区切り文字がすべて ASCII の空白で終わる分割器では、この文章に文の境界が一つも見つからず、単語の途中で切ることになります。

## Late chunking

[Late chunking](https://arxiv.org/abs/2409.04701) では、文書全体を一度埋め込み、各チャンク自身のトークン範囲でプーリングしてチャンクのベクトルを得ます。`start_token..end_token` が `encode(doc)` の中でのその範囲です。

```rust
let doc = "First sentence here. Second sentence here. Third sentence here.";
let chunks = chunkedrs::chunk(doc).max_tokens(8).split();

let encoder = tiktoken::get_encoding("o200k_base").unwrap();
let document_tokens = encoder.encode(doc);
for chunk in &chunks {
    // 実際の処理では、モデルのトークンごとの隠れ状態をここで取り出す
    let span = &document_tokens[chunk.start_token..chunk.end_token];
    assert!(!span.is_empty());
}
```

境界がトークンの内部に入る場合（`". "` の空白が次の単語と同じトークンになる場合など）、範囲はそのトークンを含むように広がります。そのため隣り合う 2 つのチャンクの範囲が 1 トークン重なることがあり、`token_count` と `end_token - start_token` が一致しないこともあります。例は [`examples/token_spans.rs`](examples/token_spans.rs) にあります。

## セマンティック分割

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

文は、プロバイダーが 1 回のリクエストで受け付ける件数ごとにまとめて埋め込みます。`.semantic()` が返すビルダーには `split_async()` しかないため、同期の `split()` から誤ってネットワーク呼び出しをすることはありません。

## Chunk

```rust
#[non_exhaustive]
pub struct Chunk {
    pub content: String,           // テキスト。&doc[start_byte..end_byte] と等しい
    pub index: usize,              // 並びの中の位置
    pub start_byte: usize,
    pub end_byte: usize,           // この位置を含まない
    pub start_token: usize,        // encode(doc) の中での範囲
    pub end_token: usize,          // この位置を含まない
    pub token_count: usize,        // content を単独で数えたトークン数
    pub section_path: Vec<String>, // Markdown の見出しの階層、外側から順に
}
```

空白だけのチャンクは除かれます。`Chunk` には `#[non_exhaustive]` が付いているため、手で作る場合は `Chunk::new(..).with_bytes(..).with_tokens(..)` を使ってください。

オーバーラップ（`.overlap(n)`）を設定すると、各チャンクは前のチャンクの終わりからおよそ `n` トークン前の文字境界で始まります。既定では無効です。検索の精度が上がるかどうかは用途によるため、有効にする前に測定してください。

## エンコーディングと feature

`.model("...")` は OpenAI、Meta、DeepSeek、Qwen、Mistral、Moonshot、Zhipu、MiniMax のモデル名を解決します。`.encoding("cl100k_base")` はエンコーディングを直接指定し、`.model()` より優先されます。知らない名前はエラーにならず `o200k_base` になります。Anthropic と Google は tiktoken 互換の語彙を公開していないため、`claude-*` と `gemini-*` も `o200k_base` になり、トークン数は近似値です。

語彙は tiktoken と同じく、既定で有効で無効にもできる feature です。既定では 20 種類すべてを組み込みます。`default-features = false` にすると、名前が解決できないときに使う `o200k_base` だけが残ります。

```toml
chunkedrs = "2"                                                   # すべての語彙
chunkedrs = { version = "2", default-features = false }           # o200k_base のみ
chunkedrs = { version = "2", default-features = false, features = ["vocab-qwen"] }
```

`examples/basic` を macOS arm64 で release ビルドすると、すべての語彙を含む場合は 9,189,376 バイト、`default-features = false` の場合は 2,072,256 バイトで、77% 小さくなります。

ベンダー単位：`vocab-openai`、`vocab-meta`、`vocab-deepseek`、`vocab-qwen`、`vocab-mistral`、`vocab-moonshot`、`vocab-zhipu`、`vocab-minimax`。語彙単位：`vocab-cl100k_base`、`vocab-llama4`、`vocab-qwen3_5`、`vocab-mistral_v15`、`vocab-glm5` のほか、tiktoken の一覧にあるもの。組み込まれていない語彙は知らない名前と同じく `o200k_base` になるため、指定するエンコーディングが有効になっているか確認してください。

## text-splitter との比較

| | chunkedrs | text-splitter 0.33 |
|---|---|---|
| サイズの単位 | トークン（tiktoken）| 既定は文字数。tiktoken-rs、tokenizers、独自の `ChunkSizer` でトークン数にもできる |
| Markdown | 各チャンクに見出しの階層 | CommonMark の構造で分割（pulldown-cmark）|
| コード | 境界の記号で分割、言語を問わず依存なし | 言語ごとの tree-sitter 文法 |
| HTML | 対応 | 非対応 |
| 文書のトークン列での範囲 | あり | なし |
| 埋め込みによる分割 | あり（`semantic`）| なし |
| サイズの範囲指定（`min..max`）、前後の空白の除去 | なし | あり |
| Unicode の分割（書記素、単語、文。ICU による）| なし | あり |

## ビルド

```bash
cargo test --workspace --all-targets
cargo clippy --workspace --all-targets --all-features -- -D warnings
CHUNKEDRS_PROPTEST_CASES=5000 cargo test --test properties   # プロパティテストのケースを増やす
cargo run --release -p bench-compare                          # 上の性能表を再現する
```

## 関連クレート

[tiktoken](https://crates.io/crates/tiktoken) · [@goliapkg/tiktoken-wasm](https://www.npmjs.com/package/@goliapkg/tiktoken-wasm) · [instructors](https://crates.io/crates/instructors) · [embedrs](https://crates.io/crates/embedrs)

## ライセンス

[MIT](LICENSE-MIT) と [Apache-2.0](LICENSE-APACHE) のいずれかを選んで利用できます。

tiktoken の `vocab-*` feature で組み込まれる語彙には、それぞれの公開元のライセンスが適用されます。一部は利用に条件があります（Llama、Mistral、DeepSeek、Kimi、MiniMax）。一覧は tiktoken の [LICENSE-3RD-PARTY](https://github.com/goliajp/rust-tiktoken/blob/develop/tiktoken/LICENSE-3RD-PARTY) にあります。語彙の feature を有効にしないビルドには、その語彙は含まれません。
