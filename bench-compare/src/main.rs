//! chunkedrs vs text-splitter on the corpora in `data/`.
//!
//!   cargo run --release -p bench-compare [corpus-prefix]
//!
//! Environment: `ENCS=o200k_base,cl100k_base` picks encodings, `QUICK=1` runs
//! each candidate once (a smoke run, no timing), `TIKTOKEN_THREADS=1` keeps
//! tiktoken on the calling thread (tiktoken-rs has no threads of its own).
//!
//! Every candidate's output is checked once with tiktoken: chunks over the
//! budget, the largest chunk, and how much of the document the chunks cover.

use std::time::Instant;
use text_splitter::{ChunkConfig, ChunkSizer, CodeSplitter, MarkdownSplitter, TextSplitter};

type Enc = &'static tiktoken::CoreBpe;
type Candidate = Box<dyn Fn(&str) -> Vec<String>>;

/// text-splitter sized by this repository's tokenizer instead of tiktoken-rs.
struct Tiktoken(Enc);

impl ChunkSizer for Tiktoken {
    fn size(&self, chunk: &str) -> usize {
        self.0.count(chunk)
    }
}

fn quick() -> bool {
    std::env::var_os("QUICK").is_some()
}

/// Median milliseconds per call over 9 rounds of about 150 ms each.
fn median_ms(text: &str, f: &dyn Fn(&str) -> Vec<String>) -> f64 {
    let t0 = Instant::now();
    std::hint::black_box(f(text));
    let first = t0.elapsed().as_secs_f64();
    if quick() {
        return first * 1e3;
    }
    std::hint::black_box(f(text));
    let n = ((0.15 / first.max(1e-6)) as usize).clamp(1, 5000);
    let mut rounds: Vec<f64> = (0..9)
        .map(|_| {
            let s = Instant::now();
            for _ in 0..n {
                std::hint::black_box(f(text));
            }
            s.elapsed().as_secs_f64() * 1e3 / n as f64
        })
        .collect();
    rounds.sort_by(f64::total_cmp);
    rounds[4]
}

fn check(text: &str, chunks: &[String], enc: Enc, max: usize) -> String {
    let counts: Vec<usize> = chunks.iter().map(|c| enc.count(c)).collect();
    let over = counts.iter().filter(|&&c| c > max).count();
    let largest = counts.iter().copied().max().unwrap_or(0);
    let bytes: usize = chunks.iter().map(String::len).sum();
    format!(
        "chunks={:>5} over={:>2} largest={:>4} bytes/doc={:.3}",
        chunks.len(),
        over,
        largest,
        bytes as f64 / text.len() as f64
    )
}

fn chunked<'a>(text: &'a str, enc: &'static str, max: usize) -> chunkedrs::ChunkBuilder<'a> {
    chunkedrs::chunk(text).encoding(enc).max_tokens(max)
}

fn contents(chunks: Vec<chunkedrs::Chunk>) -> Vec<String> {
    chunks.into_iter().map(|c| c.content).collect()
}

fn candidates(corpus: &str, enc_name: &'static str, max: usize) -> Vec<(&'static str, Candidate)> {
    let enc: Enc = tiktoken::get_encoding(enc_name).unwrap();
    let rs: &'static tiktoken_rs::CoreBPE = Box::leak(Box::new(match enc_name {
        "o200k_base" => tiktoken_rs::o200k_base().unwrap(),
        _ => tiktoken_rs::cl100k_base().unwrap(),
    }));

    let mut out: Vec<(&'static str, Candidate)> = vec![
        (
            "floor: one tiktoken encode",
            Box::new(move |t| vec![enc.encode(t).len().to_string()]),
        ),
        (
            "chunkedrs",
            Box::new(move |t| contents(chunked(t, enc_name, max).split())),
        ),
        ("text-splitter + tiktoken-rs", {
            let s = TextSplitter::new(ChunkConfig::new(max).with_sizer(rs));
            Box::new(move |t| s.chunks(t).map(str::to_owned).collect())
        }),
        ("text-splitter + tiktoken", {
            let s = TextSplitter::new(ChunkConfig::new(max).with_sizer(Tiktoken(enc)));
            Box::new(move |t| s.chunks(t).map(str::to_owned).collect())
        }),
    ];
    if corpus.ends_with(".md") {
        out.push((
            "chunkedrs markdown",
            Box::new(move |t| contents(chunked(t, enc_name, max).markdown().split())),
        ));
        out.push(("text-splitter markdown + tiktoken-rs", {
            let s = MarkdownSplitter::new(ChunkConfig::new(max).with_sizer(rs));
            Box::new(move |t| s.chunks(t).map(str::to_owned).collect())
        }));
        out.push(("text-splitter markdown + tiktoken", {
            let s = MarkdownSplitter::new(ChunkConfig::new(max).with_sizer(Tiktoken(enc)));
            Box::new(move |t| s.chunks(t).map(str::to_owned).collect())
        }));
    }
    if corpus.starts_with("code") {
        out.push((
            "chunkedrs code",
            Box::new(move |t| contents(chunked(t, enc_name, max).code().split())),
        ));
        out.push(("text-splitter code(rust) + tiktoken-rs", {
            let cfg = ChunkConfig::new(max).with_sizer(rs);
            let s = CodeSplitter::new(tree_sitter_rust::LANGUAGE, cfg).unwrap();
            Box::new(move |t| s.chunks(t).map(str::to_owned).collect())
        }));
        out.push(("text-splitter code(rust) + tiktoken", {
            let cfg = ChunkConfig::new(max).with_sizer(Tiktoken(enc));
            let s = CodeSplitter::new(tree_sitter_rust::LANGUAGE, cfg).unwrap();
            Box::new(move |t| s.chunks(t).map(str::to_owned).collect())
        }));
    }
    out
}

fn main() {
    let dir = concat!(env!("CARGO_MANIFEST_DIR"), "/data");
    let only = std::env::args().nth(1);
    let encs = std::env::var("ENCS").unwrap_or_else(|_| "o200k_base,cl100k_base".into());
    let threads = std::env::var("TIKTOKEN_THREADS").unwrap_or_else(|_| "default".into());
    println!("# TIKTOKEN_THREADS={threads}");

    let corpora = [
        "book_en.txt",
        "zh_xiyouji.md",
        "md_rust_book.md",
        "code_rust.txt",
    ];
    for enc_name in ["o200k_base", "cl100k_base"] {
        if !encs.split(',').any(|e| e == enc_name) {
            continue;
        }
        let enc: Enc = tiktoken::get_encoding(enc_name).unwrap();
        for corpus in corpora {
            if only.as_deref().is_some_and(|o| !corpus.starts_with(o)) {
                continue;
            }
            let text = std::fs::read_to_string(format!("{dir}/{corpus}")).unwrap();
            for max in [512usize, 128] {
                println!(
                    "\n## {corpus} ({} B) {enc_name} max_tokens={max}",
                    text.len()
                );
                for (name, f) in candidates(corpus, enc_name, max) {
                    let report = if name.starts_with("floor") {
                        String::new()
                    } else {
                        check(&text, &f(&text), enc, max)
                    };
                    let ms = median_ms(&text, f.as_ref());
                    println!("{name:<40} {ms:>9.3} ms  {report}");
                }
            }
        }
    }
}
