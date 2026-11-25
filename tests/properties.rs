//! Seeded property tests: random documents, every strategy, a range of sizes.
//!
//! The generator is a fixed xorshift, so a failure prints a seed that
//! reproduces it. `CHUNKEDRS_PROPTEST_CASES` raises the number of documents
//! (default 120 per strategy); `CHUNKEDRS_PROPTEST_SEED` replays one.

use chunkedrs::Chunk;

struct Rng(u64);

impl Rng {
    fn next(&mut self) -> u64 {
        // xorshift64*
        self.0 ^= self.0 >> 12;
        self.0 ^= self.0 << 25;
        self.0 ^= self.0 >> 27;
        self.0.wrapping_mul(0x2545_f491_4f6c_dd1d)
    }

    fn below(&mut self, n: usize) -> usize {
        (self.next() % n as u64) as usize
    }
}

/// Fragments chosen to hit every separator tier, the closing-mark absorption,
/// characters that take several tokens, and runs with no separator at all.
const FRAGMENTS: &[&str] = &[
    "the",
    "quick",
    "brown",
    "fox",
    "jumps",
    "over",
    "lazy",
    "dog",
    "12345",
    "x",
    " ",
    " ",
    " ",
    "  ",
    "\t",
    "\u{3000}",
    ". ",
    "! ",
    "? ",
    ", ",
    "; ",
    ":",
    "...",
    "…",
    "‥",
    "\n",
    "\n",
    "\n\n",
    "\r\n",
    "\r\n\r\n",
    "\n\n\n",
    "你好",
    "世界",
    "分词",
    "。",
    "，",
    "、",
    "：",
    "「",
    "」",
    "（",
    "）",
    "日本語",
    "です",
    "！",
    "？",
    "👍🏽",
    "🧑‍💻",
    "𠀀",
    "é",
    "# ",
    "## ",
    "### ",
    "```\n",
    "~~~\n",
    "---\n",
    "===\n",
    "- ",
    "+++\n",
    "{\n",
    "\n}\n",
    "\n};\n",
    "fn x() ",
    "(",
    ")",
    "\nend\n",
    "<p>",
    "</p>",
    "<div>",
    "</div>",
    "</H1>",
    "</li>",
    "<br>",
    "\"",
    "'",
    "”",
    "]",
    "\r",
];

fn document(rng: &mut Rng) -> String {
    let len = 1 + rng.below(400);
    let mut s = String::new();
    for _ in 0..len {
        match rng.below(40) {
            // a run with no separator, long enough to force a token-level cut
            0 => {
                let unit = ["a", "xyz", "語", "🙂", "0"][rng.below(5)];
                s.push_str(&unit.repeat(1 + rng.below(120)));
            }
            _ => s.push_str(FRAGMENTS[rng.below(FRAGMENTS.len())]),
        }
    }
    s
}

#[derive(Clone, Copy, Debug)]
enum Strategy {
    Recursive,
    Markdown,
    Code,
    Html,
}

fn split(text: &str, strategy: Strategy, max: usize, overlap: usize, enc: &str) -> Vec<Chunk> {
    let b = chunkedrs::chunk(text)
        .max_tokens(max)
        .overlap(overlap)
        .encoding(enc);
    match strategy {
        Strategy::Recursive => b.split(),
        Strategy::Markdown => b.markdown().split(),
        Strategy::Code => b.code().split(),
        Strategy::Html => b.html().split(),
    }
}

fn is_underline(line: &str) -> bool {
    let t = line.trim();
    !t.is_empty() && (t.bytes().all(|b| b == b'=') || t.bytes().all(|b| b == b'-'))
}

fn is_atx(line: &str) -> bool {
    let body = line.trim_start();
    let hashes = body.bytes().take_while(|&b| b == b'#').count();
    line.len() - body.len() <= 3 && (1..=6).contains(&hashes) && body[hashes..].starts_with(' ')
}

/// What a markdown split may leave out of every chunk: header lines, setext
/// titles with their underlines, and blank lines.
fn is_header_residue(gap: &str) -> bool {
    let lines: Vec<&str> = gap.split_inclusive('\n').collect();
    (0..lines.len()).all(|i| {
        let l = lines[i];
        l.trim().is_empty()
            || is_atx(l)
            || is_underline(l)
            || lines.get(i + 1).is_some_and(|n| is_underline(n))
    })
}

fn check(text: &str, strategy: Strategy, max: usize, overlap: usize, enc_name: &str) {
    let enc = tiktoken::get_encoding(enc_name).unwrap();
    let chunks = split(text, strategy, max, overlap, enc_name);
    let ctx = || format!("{strategy:?} max={max} overlap={overlap} enc={enc_name}");

    let doc_tokens = enc.encode(text);
    // byte offset at which each document token starts, plus the end
    let mut token_starts = vec![0usize];
    for &t in &doc_tokens {
        let last = *token_starts.last().unwrap();
        token_starts.push(last + enc.decode(&[t]).len());
    }

    for (i, c) in chunks.iter().enumerate() {
        assert_eq!(c.index, i, "{}: index", ctx());
        assert!(
            !c.content.trim().is_empty(),
            "{}: chunk {i} is whitespace only",
            ctx()
        );

        // exact byte range
        assert_eq!(
            &text[c.start_byte..c.end_byte],
            c.content,
            "{}: chunk {i} bytes {}..{}",
            ctx(),
            c.start_byte,
            c.end_byte
        );

        // budget: a single character that takes more tokens than the budget
        // cannot be split further, and is the only allowed excess
        let actual = enc.count(&c.content);
        assert_eq!(c.token_count, actual, "{}: chunk {i} stored count", ctx());
        assert!(
            actual <= max || c.content.chars().count() == 1,
            "{}: chunk {i} has {actual} tokens: {:?}",
            ctx(),
            c.content
        );

        // the token span covers the chunk's bytes
        assert!(c.start_token <= c.end_token, "{}: chunk {i} span", ctx());
        assert!(
            c.end_token <= doc_tokens.len(),
            "{}: chunk {i} span end",
            ctx()
        );
        assert!(
            token_starts[c.start_token] <= c.start_byte && token_starts[c.end_token] >= c.end_byte,
            "{}: chunk {i} span {}..{} does not cover bytes {}..{}",
            ctx(),
            c.start_token,
            c.end_token,
            c.start_byte,
            c.end_byte
        );
    }

    for w in chunks.windows(2) {
        let (a, b) = (&w[0], &w[1]);
        assert!(
            a.start_byte < b.start_byte && a.end_byte < b.end_byte,
            "{}: chunks {} {}..{} {:?} and {} {}..{} {:?} do not advance",
            ctx(),
            a.index,
            a.start_byte,
            a.end_byte,
            a.content,
            b.index,
            b.start_byte,
            b.end_byte,
            b.content
        );
        assert!(
            a.start_token <= b.start_token && a.end_token <= b.end_token,
            "{}: spans of chunks {} and {} are not monotonic",
            ctx(),
            a.index,
            b.index
        );
        if overlap == 0 {
            assert!(
                a.end_byte <= b.start_byte,
                "{}: chunks {} and {} overlap without overlap set",
                ctx(),
                a.index,
                b.index
            );
        }
    }

    // reassembly: what no chunk covers is whitespace, or header lines for markdown
    let mut covered = 0usize;
    let mut gaps = Vec::new();
    for c in &chunks {
        if c.start_byte > covered {
            gaps.push(&text[covered..c.start_byte]);
        }
        covered = covered.max(c.end_byte);
    }
    gaps.push(&text[covered..]);
    for gap in gaps {
        let ok = match strategy {
            Strategy::Markdown => is_header_residue(gap),
            _ => gap.trim().is_empty(),
        };
        assert!(ok, "{}: uncovered text {gap:?}", ctx());
    }
}

fn cases() -> u64 {
    std::env::var("CHUNKEDRS_PROPTEST_CASES")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(120)
}

fn run(strategy: Strategy, salt: u64, with_overlap: bool) {
    let encodings: &[&str] = &[
        "o200k_base",
        #[cfg(feature = "vocab-cl100k_base")]
        "cl100k_base",
    ];
    let seeds: Vec<u64> = match std::env::var("CHUNKEDRS_PROPTEST_SEED") {
        Ok(s) => vec![s.parse().expect("seed")],
        Err(_) => (1..=cases()).map(|i| i * 0x9e37_79b9 + salt).collect(),
    };
    for seed in seeds {
        let mut rng = Rng(seed);
        let text = document(&mut rng);
        let max = [1, 2, 3, 5, 8, 16, 40, 128, 512][rng.below(9)];
        let overlap = if with_overlap && rng.below(2) == 0 {
            rng.below(max + 2)
        } else {
            0
        };
        let enc = encodings[rng.below(encodings.len())];
        let result = std::panic::catch_unwind(|| check(&text, strategy, max, overlap, enc));
        if let Err(e) = result {
            eprintln!("seed {seed} failed; text = {text:?}");
            for c in split(&text, strategy, max, overlap, enc) {
                eprintln!(
                    "  {}..{} {} {:?}",
                    c.start_byte, c.end_byte, c.token_count, c.content
                );
            }
            std::panic::resume_unwind(e);
        }
    }
}

#[test]
fn recursive_holds_invariants() {
    run(Strategy::Recursive, 1, false);
}

#[test]
fn markdown_holds_invariants() {
    run(Strategy::Markdown, 2, false);
}

#[test]
fn code_holds_invariants() {
    run(Strategy::Code, 3, false);
}

#[test]
fn html_holds_invariants() {
    run(Strategy::Html, 4, false);
}
