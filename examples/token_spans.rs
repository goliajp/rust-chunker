// token spans — the metadata that makes late chunking possible
//
// run: cargo run --example token_spans
//
// Ordinary RAG embeds each chunk in isolation, so a chunk that says "it costs
// $40" loses track of what "it" was. Late chunking (arXiv 2409.04701) fixes
// that by inverting the order: embed the whole document once, then pool each
// chunk's vector over its own slice of the token stream. Every chunk embedding
// then carries the context of the document around it.
//
// The part chunkedrs supplies is the slice. `start_token..end_token` locates
// each chunk in `encoder.encode(whole_document)` — no re-tokenizing, no
// guessing where a chunk landed.

fn main() {
    let doc = "\
The Rust compiler enforces memory safety at compile time. It does this through \
an ownership system rather than a garbage collector.

That system has three rules. Each value has a single owner. There can be many \
immutable borrows or one mutable borrow. Borrows must not outlive the owner.

The cost is a learning curve. The benefit is that data races become compile \
errors instead of production incidents.";

    let chunks = chunkedrs::chunk(doc).max_tokens(40).split();

    // this is the tokenization a late-chunking pipeline would feed its encoder
    let encoder = tiktoken::get_encoding("o200k_base").unwrap();
    let document_tokens = encoder.encode(doc);

    println!(
        "document: {} bytes, {} tokens, {} chunks\n",
        doc.len(),
        document_tokens.len(),
        chunks.len()
    );

    for chunk in &chunks {
        println!(
            "chunk {} | bytes {:>3}..{:<3} | tokens {:>3}..{:<3} | own count: {}",
            chunk.index,
            chunk.start_byte,
            chunk.end_byte,
            chunk.start_token,
            chunk.end_token,
            chunk.token_count,
        );

        // In a real pipeline this slice indexes into the encoder's per-token
        // hidden states, and you mean-pool them into the chunk's vector.
        let span = &document_tokens[chunk.start_token..chunk.end_token];
        println!("           span width {} tokens", span.len());
    }

    println!();

    // The two numbers are not the same thing, and the difference is the point:
    //   token_count            — a fresh count of the chunk alone. This is what
    //                            `max_tokens` bounds and what you budget a
    //                            context window against.
    //   end_token - start_token — the chunk's footprint in the document stream.
    //                            Where a separator sits inside a token, the span
    //                            widens to cover it.
    for chunk in &chunks {
        let span_width = chunk.end_token - chunk.start_token;
        if span_width != chunk.token_count {
            println!(
                "chunk {}: span is {} tokens but the chunk alone counts {} — \
                 the boundary fell inside a token",
                chunk.index, span_width, chunk.token_count,
            );
        }
    }

    // Spans never leave a hole, so pooling over all of them sees every token.
    let covered = chunks.first().map(|c| c.start_token).unwrap_or(0)
        ..chunks.last().map(|c| c.end_token).unwrap_or(0);
    println!(
        "\ncovered token range: {}..{} of {}",
        covered.start,
        covered.end,
        document_tokens.len()
    );
    assert_eq!(covered.start, 0);
    assert_eq!(covered.end, document_tokens.len());
}
