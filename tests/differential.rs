//! This release splits exactly as 3.0.0 did: same byte ranges, token spans,
//! token counts and section paths, for every strategy, compiled encoding,
//! budget and overlap tried here.

type Row = (usize, usize, usize, usize, usize, Vec<String>);

#[derive(Clone, Copy, Debug)]
enum Strategy {
    Recursive,
    Markdown,
    Code,
    Html,
}

const STRATEGIES: [Strategy; 4] = [
    Strategy::Recursive,
    Strategy::Markdown,
    Strategy::Code,
    Strategy::Html,
];

const ENCODINGS: [&str; 5] = ["o200k_base", "cl100k_base", "qwen3_5", "llama3", "glm5"];

/// The encodings of `ENCODINGS` this build has.
fn encodings() -> impl Iterator<Item = &'static str> {
    ENCODINGS
        .into_iter()
        .filter(|e| tiktoken::get_encoding(e).is_some())
}

fn now(text: &str, st: Strategy, enc: &str, max: usize, overlap: usize) -> Vec<Row> {
    let c = chunkedrs::Chunker::for_encoding(enc)
        .unwrap()
        .max_tokens(max)
        .overlap(overlap);
    let c = match st {
        Strategy::Recursive => c,
        Strategy::Markdown => c.markdown(),
        Strategy::Code => c.code(),
        Strategy::Html => c.html(),
    };
    c.split(text)
        .into_iter()
        .map(|k| {
            assert_eq!(&text[k.start_byte..k.end_byte], k.content);
            let path = k.section_path.iter().map(|s| s.to_string()).collect();
            (
                k.start_byte,
                k.end_byte,
                k.start_token,
                k.end_token,
                k.token_count,
                path,
            )
        })
        .collect()
}

fn before(text: &str, st: Strategy, enc: &str, max: usize, overlap: usize) -> Vec<Row> {
    let c = chunkedrs_3::Chunker::for_encoding(enc)
        .unwrap()
        .max_tokens(max)
        .overlap(overlap);
    let c = match st {
        Strategy::Recursive => c,
        Strategy::Markdown => c.markdown(),
        Strategy::Code => c.code(),
        Strategy::Html => c.html(),
    };
    c.split(text)
        .into_iter()
        .map(|k| {
            let path = k.section_path.iter().map(|s| s.to_string()).collect();
            (
                k.start_byte,
                k.end_byte,
                k.start_token,
                k.end_token,
                k.token_count,
                path,
            )
        })
        .collect()
}

fn same(label: &str, text: &str, st: Strategy, enc: &str, max: usize, overlap: usize) {
    let (a, b) = (
        now(text, st, enc, max, overlap),
        before(text, st, enc, max, overlap),
    );
    if a != b {
        let i = a
            .iter()
            .zip(&b)
            .position(|(x, y)| x != y)
            .unwrap_or(a.len().min(b.len()));
        panic!(
            "{label} {st:?} {enc} max={max} overlap={overlap}: chunk {i} differs: {:?} vs 3.0.0 {:?}",
            a.get(i),
            b.get(i)
        );
    }
}

fn corpus(name: &str) -> String {
    std::fs::read_to_string(format!(
        "{}/bench-compare/data/{name}",
        env!("CARGO_MANIFEST_DIR")
    ))
    .unwrap()
}

const CORPORA: [&str; 5] = [
    "book_en.txt",
    "zh_xiyouji.md",
    "ja_neko.txt",
    "md_rust_book.md",
    "code_rust.txt",
];

#[test]
fn corpora_split_as_3_0() {
    for name in CORPORA {
        let text = corpus(name);
        let short = &text[..text.floor_char_boundary(20_000)];
        for enc in encodings() {
            for st in STRATEGIES {
                for max in [64, 128, 512] {
                    for overlap in [0, max / 4] {
                        same(name, &text, st, enc, max, overlap);
                    }
                }
                // small budgets on a slice, to keep the run short
                for max in [8, 16] {
                    for overlap in [0, max / 4] {
                        same(name, short, st, enc, max, overlap);
                    }
                }
            }
        }
    }
}

struct Rng(u64);

impl Rng {
    fn next(&mut self) -> u64 {
        self.0 ^= self.0 << 13;
        self.0 ^= self.0 >> 7;
        self.0 ^= self.0 << 17;
        self.0
    }
    fn pick<'a>(&mut self, xs: &[&'a str]) -> &'a str {
        xs[(self.next() % xs.len() as u64) as usize]
    }
}

const PIECES: &[&str] = &[
    "The quick brown fox",
    " jumps",
    " over the lazy dog",
    ". ",
    "! ",
    "? ",
    ", ",
    "; ",
    ": ",
    "\n",
    "\n\n",
    "\r\n",
    "  ",
    "\t",
    "# Heading\n",
    "## Sub\n",
    "Title\n=====\n",
    "```\n# not a header\n```\n",
    "---\nkey: v\n---\n",
    "fn main() {\n    let x = 1;\n}\n",
    "<p>para</p>",
    "</div>",
    "<br>",
    "你好世界。",
    "今天天气很好，",
    "「引用。」",
    "吾輩は猫である。",
    "名前はまだ無い、",
    "カタカナ・テスト",
    "１２３",
    "2026",
    "x=1+2",
    "https://example.com/a?b=c",
    "😀",
    "é",
    "naïve",
    "    indented",
];

#[test]
fn generated_documents_split_as_3_0() {
    let encs: Vec<&str> = encodings().collect();
    let mut rng = Rng(0x2545_f491_4f6c_dd1d);
    for case in 0..2000 {
        let n = 1 + (rng.next() % 60) as usize;
        let text: String = (0..n).map(|_| rng.pick(PIECES)).collect();
        let st = STRATEGIES[case % 4];
        let enc = encs[case % encs.len()];
        let max = [1, 2, 3, 5, 8, 16, 40][(rng.next() % 7) as usize];
        let overlap = (rng.next() % (max as u64 + 1)) as usize;
        same(
            &format!("case {case}: {text:?}"),
            &text,
            st,
            enc,
            max,
            overlap,
        );
    }
}
