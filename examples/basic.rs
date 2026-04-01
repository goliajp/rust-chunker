// basic recursive chunking with token counts and byte offsets
//
// run: cargo run -p chunkedrs --example basic

fn main() {
    let text = "\
Rust is a systems programming language focused on safety, speed, and concurrency. \
It achieves memory safety without a garbage collector, using a system of ownership \
with a set of rules that the compiler checks at compile time.

The language was originally designed by Graydon Hoare at Mozilla Research, with \
contributions from many others. It has gained significant popularity in recent years, \
especially for systems programming, web assembly, and embedded development.

One of Rust's key features is its type system and ownership model. Every value in \
Rust has a single owner, and when the owner goes out of scope, the value is dropped. \
This eliminates many common bugs like use-after-free and double-free errors.

Concurrency in Rust is particularly elegant. The ownership system prevents data races \
at compile time, making it possible to write concurrent code that is both safe and fast. \
The standard library provides channels, mutexes, and atomic types for synchronization.";

    // split with a small token limit to demonstrate multiple chunks
    let chunks = chunkedrs::chunk(text).max_tokens(50).split();

    println!(
        "input: {} bytes, split into {} chunks\n",
        text.len(),
        chunks.len()
    );

    for chunk in &chunks {
        println!(
            "chunk {} | tokens: {:>2} | bytes: {}..{} | len: {} bytes",
            chunk.index,
            chunk.token_count,
            chunk.start_byte,
            chunk.end_byte,
            chunk.len(),
        );
        // show a preview of the content (first 80 chars)
        let preview: String = chunk.content.chars().take(80).collect();
        let ellipsis = if chunk.content.len() > 80 { "..." } else { "" };
        println!("  \"{preview}{ellipsis}\"\n");
    }

    // verify the token budget is respected
    assert!(
        chunks.iter().all(|c| c.token_count <= 50),
        "every chunk must have <= 50 tokens"
    );
    println!("all chunks are within the 50-token budget");
}
