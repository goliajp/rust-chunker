// code and html strategies — boundary-aware splitting without a parser
//
// run: cargo run --example code_and_html
//
// Both strategies scan for structural markers rather than building a tree.
// That keeps the dependency surface at zero and works on any language or any
// markup, at the cost of not tracking constructs the way a real parser would.
// When AST fidelity matters, reach for a tree-sitter based splitter instead.

fn main() {
    code();
    println!();
    html();
}

fn code() {
    let src = "\
fn parse_header(line: &str) -> Option<usize> {
    let hashes = line.bytes().take_while(|&b| b == b'#').count();
    (1..=6).contains(&hashes).then_some(hashes)
}

fn is_fence(line: &str) -> bool {
    line.starts_with(\"```\") || line.starts_with(\"~~~\")
}

fn main() {
    println!(\"{:?}\", parse_header(\"## Title\"));
}
";

    println!("=== code() — splits on blank lines, then dedented block closers ===\n");
    for chunk in chunkedrs::chunk(src).code().max_tokens(40).split() {
        println!(
            "--- chunk {} ({} tokens) ---",
            chunk.index, chunk.token_count
        );
        println!("{}", chunk.content.trim_end());
    }
}

fn html() {
    let page = "\
<article>\
<h1>Chunking strategies</h1>\
<p>Recursive splitting is the default and handles most prose.</p>\
<p>Markdown-aware splitting preserves section metadata on every chunk.</p>\
<ul><li>Token accurate</li><li>Byte offsets</li><li>Token spans</li></ul>\
<p>Semantic splitting uses embeddings to find meaning boundaries.</p>\
</article>";

    println!("=== html() — splits after block-level closing tags ===\n");
    for chunk in chunkedrs::chunk(page).html().max_tokens(20).split() {
        println!(
            "--- chunk {} ({} tokens) ---",
            chunk.index, chunk.token_count
        );
        println!("{}", chunk.content);
    }
}
