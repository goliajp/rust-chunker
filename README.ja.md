# chunkedrs

[![Crates.io](https://img.shields.io/crates/v/chunkedrs?style=flat-square&logo=rust)](https://crates.io/crates/chunkedrs)
[![docs.rs](https://img.shields.io/docsrs/chunkedrs?style=flat-square&logo=docs.rs)](https://docs.rs/chunkedrs)
[![CI](https://img.shields.io/github/actions/workflow/status/goliajp/rust-chunker/ci.yml?branch=develop&style=flat-square&logo=github&label=ci)](https://github.com/goliajp/rust-chunker/actions/workflows/ci.yml)
[![License](https://img.shields.io/crates/l/chunkedrs?style=flat-square)](#ライセンス)

[English](README.md) | [简体中文](README.zh-CN.md) | **日本語**

> **2.x からの更新**：API は変わりましたが、分割結果は変わりません。設定はすべて `Chunker` で行い、チャンクは文書のテキストを借用します（保存しておく場合は `OwnedChunk` に変換します）。知らないモデル名やエンコーディング名は、黙って `o200k_base` になるのではなくエラーになります。再帰、Markdown、コード、HTML の分割結果は 2.2 と同じです。セマンティック分割は `embedrs::Embed` を実装したものなら何でも使えます。移行の対応表は [CHANGELOG](CHANGELOG.md#upgrading-from-2x) にあります。

chunkedrs は、長い文書をトークン数の上限に収まるチャンクに分割するライブラリです。埋め込みと検索での利用を想定しています。トークン数は [tiktoken](https://crates.io/crates/tiktoken) で数え、20 種類のエンコーディング（OpenAI、Llama、DeepSeek、Qwen、Mistral、Kimi、GLM、MiniMax）に対応します。各チャンクは元の文書のスライスで、文書でのバイト範囲と、文書全体のトークン列での範囲を持ちます。依存は tiktoken だけです。`semantic` feature を有効にすると [embedrs](https://crates.io/crates/embedrs) が加わりますが、embedrs 自体も依存を持ちません。

Apple M6 のシングルスレッド、`o200k_base`、1 チャンク 512 トークンの条件で、200 KB の英語の本を 0.13 ms で分割します。本全体を 1 回 `encode` する時間の 1.5 倍です。chunkedrs 3.1 では 0.14 ms、3.0 では 0.16 ms、2.2 では 0.25 ms でした。text-splitter 0.33.6 は、同じトークナイザーで 0.93 ms、tiktoken-rs 0.12（`tiktoken-rs` feature が使うトークナイザー）で 24.6 ms かかります。

## 特徴

- **5 つの分割方式**：再帰（段落、行、文、節、単語）、見出しの階層を保持する Markdown、境界の記号で分けるコードと HTML、埋め込みの類似度で分けるセマンティック分割
- **トークン上限を厳密に守る**：`token_count` はチャンク自身のテキストを単独で数えた値で、`max_tokens` を超えません。例外は、1 文字だけで上限を超える場合のみです
- **借用と位置情報**：`content` は `&doc[start_byte..end_byte]` そのもので、コピーしません。`start_token..end_token` は文書全体のトークン ID の中でのチャンクの範囲で、[late chunking](#late-chunking) に必要なのはこの範囲です
- **日本語と中国語**：空白が続かない `。！？` で文を、`，、；：` で節を区切ります。閉じかぎ括弧はその文の側に残ります
- **どのモデルでもセマンティック分割**：文を渡し、ベクトルを受け取る形なので、外部 API、ローカルモデル、キャッシュのどれでも使えます。非同期ランタイムも不要です
- **一括処理**：`split_batch` は複数の文書をすべての CPU コアに分けて処理します

## クイックスタート

```toml
[dependencies]
chunkedrs = "3"
```

```rust
// 再帰、512 トークン、オーバーラップなし、o200k_base
let chunks = chunkedrs::split("your long text here...");
for c in &chunks {
    println!("[{}] {} tokens, bytes {}..{}", c.index, c.token_count, c.start_byte, c.end_byte);
}

// モデル名でエンコーディングを選ぶ。知らない名前はエラーになる
let chunker = chunkedrs::Chunker::for_model("gpt-5.6-terra")?
    .max_tokens(256)
    .overlap(32);
let chunks = chunker.split("your long text here...");
assert!(chunks.iter().all(|c| c.token_count <= 256));

// 1 つの設定で複数の文書を処理する
let chunker = chunkedrs::Chunker::new().max_tokens(256).markdown();
let docs = ["# A\n\nFirst.", "# B\n\nSecond."];
let all = chunker.split_batch(&docs);
assert_eq!(all[1][0].section(), Some("# B"));

// チャンクは文書を借用する。文書より長く持つなら OwnedChunk にする
let kept: Vec<chunkedrs::OwnedChunk> = chunks.iter().map(|c| c.to_owned_chunk()).collect();
```

## 性能

数値は文書 1 つの分割にかかるミリ秒で、ウォームアップ後 9 ラウンドの中央値を、さらに 3 回の測定の中央値にしたものです。Apple M6 の Mac mini、rustc 1.100、LTO を有効にした release ビルド、`TIKTOKEN_THREADS=1`、`o200k_base`、オーバーラップなし、測定日は 2026-11-27 です。測定中のロードアベレージは 1.1〜2.2 でした。各行の列は続けて測っています。chunkedrs 3.1.0、3.0.0、2.2.0 はそれ以前のリリースで、同じ tiktoken 5.7 を使っています。text-splitter には容量として同じ `max_tokens` を渡し、ほかは既定値です。コード分割には tree-sitter-rust 0.24 を使っています。

| コーパス | 方式 | max_tokens | 1 回の `encode` | chunkedrs | chunkedrs 3.1.0 | chunkedrs 3.0.0 | chunkedrs 2.2.0 | text-splitter + tiktoken 5.7 | text-splitter + tiktoken-rs 0.12 |
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

`book_en.txt` は『高慢と偏見』の 200,000 バイト、`zh_xiyouji.md` は Wikisource の『西遊記』第 1〜9 回を Markdown にした 178,622 バイト、`ja_neko.txt` は青空文庫の夏目漱石『吾輩は猫である』の先頭 180,000 バイト（ルビを除去）、`md_rust_book.md` は『The Rust Programming Language』第 3〜6 章の 174,237 バイト、`code_rust.txt` は 100,000 バイトの Rust のソースコードです。いずれも [`bench-compare/data`](bench-compare/data) にあります。

- 上の表のすべてのチャンクを tiktoken で数え直しました。chunkedrs にも text-splitter にも上限を超えるチャンクはありません
- 512 トークンのとき、chunkedrs の時間はどの方式でも文書全体を 1 回 `encode` する時間の 1.5〜1.8 倍、128 トークンでは 1.8〜2.9 倍です。文書を 1 回エンコードし、各トークンの終端位置はエンコーダーがそのまま返します。各チャンクのトークン数は、チャンク内で最初と最後の tiktoken のカット位置の間は文書の ID を再利用し、両端だけをエンコードし直して数えます。残りが境界探しとチャンクの組み立てです
- 3.1.0 と比べると 1〜23% 短くなります。差が最も小さいのは中国語の recursive 方式の 512 トークン（1%）で、時間のほとんどがエンコードそのものです。最も大きいのはコードです。2.2.0 と比べると 35〜52% 短くなります。4 つのバージョンの分割結果は同じです
- `cl100k_base` でもほぼ同じで、512 トークンでは 1 回の `encode` の 1.4〜1.9 倍（3.1.0 は日本語と、Markdown を markdown 方式で分けたときに最大 2.07 倍）、3.1.0 より 2〜24% 短くなります。tiktoken の既定のマルチスレッドを使うと、これらの文書の全体のエンコードが複数のスレッドで行われ、chunkedrs は 0.06〜0.18 ms（3.1.0 は 0.07〜0.21 ms）になります
- `Chunker::split_batch` で 80 文書（5 つのコーパスを各 16 回、合計 13.3 MB）を処理すると、1 つずつでは 7.4 ms、一括では 2.3 ms です

再現するには `TIKTOKEN_THREADS=1 cargo run --release -p bench-compare` を実行してください。

## 分割の仕組み

文書全体を一度だけエンコードします。チャンクの開始位置から文書のトークンを `max_tokens` 個数えた位置が、そのチャンクが終われる最も遠いバイト位置です。チャンクはその範囲の中で、存在する境界のうち優先度が最も高い種類の最後の位置で終わります。

| 方式 | 境界（優先度の高い順）|
|---|---|
| 再帰（既定）| 空行；改行；空白が続く `. ! ?` と `。！？．｡…‥`；`; , :` と `；，、：･`；空白、タブ、全角空白 |
| `markdown()` | まず見出し行（ATX の `#` と setext の `===` / `---`。コードブロックと front matter の中は除く）、各セクションの中は再帰の境界 |
| `code()` | 空行；0 桁目に単独で置かれた `}` `};` `)` `]` `];` `end`；改行；空白、タブ |
| `html()` | ブロック要素の終了タグの後（`</p>`、`</li>`、`</section>`、`</h1>` など）、その後は再帰の境界 |
| `semantic()` | 隣り合う文の埋め込みが似なくなる位置、その後は再帰の境界 |

範囲内に境界が一つもない場合は、トークンの境界で終わり、文字の境界まで戻します。選んだテキストはその後、単独でエンコードした場合のトークン数を数えます。BPE は切れ目をまたいで結合するため、単独でエンコードしたテキストは、文書全体の中で占めるトークン数より 1 つ多くなることも少なくなることもあります。この計数では文書のトークン ID を再利用します。チャンクの中で「tiktoken が切ってもトークンが変わらないと保証する位置」の最初と最後の間は文書の ID をそのまま使い、両端の数バイトだけをエンコードし直します。まれに上限を超えた場合は、範囲を縮めて選び直します。

コードと HTML の分割は記号とタグだけを見て、構文解析はしません。依存も増えませんが、文字列リテラルの中の `}` もブロックの終わりとして扱います。構文に沿ってコードを分割したい場合は、text-splitter の tree-sitter ベースの `CodeSplitter` を使ってください。

## Markdown

```rust
let md = "# Guide\n\nIntro.\n\n## Install\n\nRun cargo add.\n";
let chunks = chunkedrs::Chunker::new().markdown().split(md);

assert_eq!(chunks[0].section(), Some("# Guide"));
assert_eq!(chunks[1].section_path, ["# Guide", "## Install"]);
```

見出し行は `content` ではなく `section_path` に入ります。そのため、全チャンクの内容をつなげても見出し行は復元されません。見出しと一緒に埋め込みたい場合は、`format!("{}\n\n{}", c.section_path.join(" > "), c.content)` のように前に付けてください。

## 日本語と中国語

```rust
let zh = "他说「今天天气很好。」然后就出门了。她回答说「确实不错。」于是也跟着出去了。";
let chunks = chunkedrs::Chunker::new().max_tokens(12).split(zh);

assert_eq!(chunks[0].content, "他说「今天天气很好。」");
assert_eq!(chunks[1].content, "然后就出门了。");
```

区切り文字がすべて ASCII の空白で終わる分割器では、この文章に文の境界が一つも見つからず、単語の途中で切ることになります。

## Late chunking

[Late chunking](https://arxiv.org/abs/2409.04701) では、文書全体を一度埋め込み、各チャンク自身のトークン範囲でプーリングしてチャンクのベクトルを得ます。`split_with_tokens` はチャンクと一緒に文書のトークン ID を返し、`start_token..end_token` はその ID の中での範囲です。

```rust
let doc = "First sentence here. Second sentence here. Third sentence here.";
let (chunks, ids) = chunkedrs::Chunker::new().max_tokens(8).split_with_tokens(doc);
for chunk in &chunks {
    // 実際の処理では、モデルのトークンごとの隠れ状態をここで取り出す
    let span = &ids[chunk.start_token..chunk.end_token];
    assert!(!span.is_empty());
}
```

境界がトークンの内部に入る場合（`". "` の空白が次の単語と同じトークンになる場合など）、範囲はそのトークンを含むように広がります。そのため隣り合う 2 つのチャンクの範囲が 1 トークン重なることがあり、`token_count` と `end_token - start_token` が一致しないこともあります。例は [`examples/token_spans.rs`](examples/token_spans.rs) にあります。

## セマンティック分割

`Chunker::semantic(text)` はテキストを文に分け（どの文も `max_tokens` 以下）、`inputs()` として渡します。どれも元のテキストのスライスです。好きな方法で埋め込み、ベクトルを返すと、隣り合う文が似なくなる位置で区切り、各グループを上限に収まるように分割します。feature も依存も I/O も要りません。

```rust
let text = "Cats purr. Cats nap. Stocks fell. Markets slid.";
let job = chunkedrs::Chunker::new().semantic(text);
let vectors: Vec<f32> = my_model(job.inputs());   // 入力ごとに 1 つのベクトルを順に並べたもの
let chunks = job.finish(&vectors, 384)?;
```

`semantic` feature を有効にすると、`embedrs::Embed` を実装したものなら何でも埋め込みに使えます。外部プロバイダーの `Client`（プロバイダーの上限に合わせてリクエストを分け、並行して送ります）、CPU で動くローカルモデル、独自の型のどれでもかまいません。

```toml
chunkedrs = { version = "3", features = ["semantic"] }
embedrs = { version = "0.7", features = ["local"] }
```

```rust,ignore
let client = embedrs::local();   // all-MiniLM-L6-v2。初回だけダウンロードする
let chunks = chunkedrs::Chunker::new()
    .max_tokens(256)
    .split_semantic(text, &client)
    .await?;

// 複数の文書の文を 1 回の埋め込み呼び出しにまとめる
let all = chunker.split_semantic_batch(&docs, &client).await?;
```

既定では、隣り合う文のコサイン距離が、その文書の隣接距離の 95 パーセンタイルを超えたところで区切ります（`Breakpoints::Percentile(95.0)`）。このためしきい値はモデルに合わせて変わります。`semantic(text).breakpoints(Breakpoints::Threshold(0.5))` とすると、固定の類似度で区切ります。embedrs に付属するローカルモデルは英語のモデルなので、日本語や中国語には多言語の埋め込みモデルを使ってください。

## Chunk

```rust
#[non_exhaustive]
pub struct Chunk<'a> {
    pub content: &'a str,          // &doc[start_byte..end_byte]
    pub index: usize,              // 並びの中の位置
    pub start_byte: usize,
    pub end_byte: usize,           // この位置を含まない
    pub start_token: usize,        // 文書のトークン ID の中での範囲
    pub end_token: usize,          // この位置を含まない
    pub token_count: usize,        // content を単独で数えたトークン数
    pub section_path: Vec<&'a str>, // Markdown の見出しの階層、外側から順に
}
```

空白だけのチャンクは除かれます。`to_owned_chunk()` は同じフィールドを `String` で持つ `OwnedChunk` を返します。文書を解放した後もチャンクを保存する場合や、別のスレッドに渡す場合に使います。

オーバーラップ（`.overlap(n)`）を設定すると、各チャンクは前のチャンクの終わりからおよそ `n` トークン前の文字境界で始まります。既定では無効です。検索の精度が上がるかどうかは用途によるため、有効にする前に測定してください。

## エンコーディングと feature

`Chunker::for_model("...")` は OpenAI、Meta、DeepSeek、Qwen、Mistral、Moonshot、Zhipu、MiniMax のモデル名を解決します。`Chunker::for_encoding("cl100k_base")` はエンコーディングを直接指定します。tiktoken が知らない名前や、語彙が組み込まれていない名前には、それぞれ `Error::UnknownModel` と `Error::UnknownEncoding` を返します。Anthropic と Google は tiktoken 互換の語彙を公開していないため、`Chunker::new()`（`o200k_base`）で近似値を得てください。

語彙は tiktoken と同じく、既定で有効で無効にもできる feature です。既定では 20 種類すべてを組み込みます。`default-features = false` にすると、`Chunker::new()` が使う `o200k_base` だけが残ります。

```toml
chunkedrs = "3"                                                   # すべての語彙
chunkedrs = { version = "3", default-features = false }           # o200k_base のみ
chunkedrs = { version = "3", default-features = false, features = ["vocab-qwen"] }
```

実行時に名前でエンコーディングを選ぶプログラムを macOS arm64 で release ビルドすると、すべての語彙を含む場合は 9,055,648 バイト、`default-features = false` の場合は 1,938,480 バイトで、79% 小さくなります。

ベンダー単位：`vocab-openai`、`vocab-meta`、`vocab-deepseek`、`vocab-qwen`、`vocab-mistral`、`vocab-moonshot`、`vocab-zhipu`、`vocab-minimax`。語彙単位：`vocab-cl100k_base`、`vocab-llama4`、`vocab-qwen3_5`、`vocab-mistral_v15`、`vocab-glm5` のほか、tiktoken の一覧にあるもの。

## text-splitter との比較

| | chunkedrs | text-splitter 0.33 |
|---|---|---|
| サイズの単位 | トークン（tiktoken）| 既定は文字数。tiktoken-rs、tokenizers、独自の `ChunkSizer` でトークン数にもできる |
| Markdown | 各チャンクに見出しの階層 | CommonMark の構造で分割（pulldown-cmark）|
| コード | 境界の記号で分割、言語を問わず依存なし | 言語ごとの tree-sitter 文法 |
| HTML | 対応 | 非対応 |
| 文書のトークン列での範囲 | あり | なし |
| 埋め込みによる分割 | あり、どの埋め込み実装でも | なし |
| サイズの範囲指定（`min..max`）、前後の空白の除去 | なし | あり |
| Unicode の分割（書記素、単語、文。ICU による）| なし | あり |

## ビルド

```bash
cargo test --workspace --all-targets
cargo test --all-features             # セマンティック分割のテストは all-MiniLM-L6-v2 を CPU で動かす（91 MB、初回のみダウンロード）
cargo clippy --workspace --all-targets --all-features -- -D warnings
CHUNKEDRS_PROPTEST_CASES=5000 cargo test --test properties   # プロパティテストのケースを増やす
cargo run --release -p bench-compare                          # 上の性能表を再現する
```

## 関連クレート

[tiktoken](https://crates.io/crates/tiktoken) · [@goliapkg/tiktoken-wasm](https://www.npmjs.com/package/@goliapkg/tiktoken-wasm) · [instructors](https://crates.io/crates/instructors) · [embedrs](https://crates.io/crates/embedrs)

## ライセンス

[MIT](LICENSE-MIT) と [Apache-2.0](LICENSE-APACHE) のいずれかを選んで利用できます。

tiktoken の `vocab-*` feature で組み込まれる語彙には、それぞれの公開元のライセンスが適用されます。一部は利用に条件があります（Llama、Mistral、DeepSeek、Kimi、MiniMax）。一覧は tiktoken の [LICENSE-3RD-PARTY](https://github.com/goliajp/rust-tiktoken/blob/develop/tiktoken/LICENSE-3RD-PARTY) にあります。語彙の feature を有効にしないビルドには、その語彙は含まれません。
