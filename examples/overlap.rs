// overlap between chunks for RAG context preservation
//
// when splitting text for retrieval-augmented generation, overlap ensures that
// information near chunk boundaries is not lost. each chunk shares some tokens
// with its neighbors, so a query matching the boundary region will find relevant
// context in at least one chunk.
//
// run: cargo run -p chunkedrs --example overlap

fn main() {
    let text = "\
The solar system formed approximately 4.6 billion years ago from the gravitational \
collapse of a giant molecular cloud. The vast majority of the system's mass is in \
the Sun, with most of the remaining mass contained in Jupiter. The four inner planets \
— Mercury, Venus, Earth, and Mars — are terrestrial planets composed primarily of \
rock and metal. The four outer planets are giant planets, being substantially more \
massive than the terrestrials. The two largest, Jupiter and Saturn, are gas giants \
composed mainly of hydrogen and helium. The two outermost planets, Uranus and Neptune, \
are ice giants composed largely of substances with relatively high melting points \
compared with hydrogen and helium. All eight planets have roughly circular orbits \
that lie near the plane of Earth's orbit, called the ecliptic.";

    let max_tokens = 40;
    let overlap_tokens = 10;

    // split without overlap
    let no_overlap = chunkedrs::chunk(text).max_tokens(max_tokens).split();

    // split with overlap
    let with_overlap = chunkedrs::chunk(text)
        .max_tokens(max_tokens)
        .overlap(overlap_tokens)
        .split();

    println!("max_tokens: {max_tokens}, overlap: {overlap_tokens}\n");
    println!(
        "without overlap: {} chunks | with overlap: {} chunks\n",
        no_overlap.len(),
        with_overlap.len()
    );

    // show overlapping chunks and highlight shared text
    println!("--- chunks with overlap ---\n");
    for chunk in &with_overlap {
        println!(
            "chunk {} | tokens: {:>2} | bytes: {}..{}",
            chunk.index, chunk.token_count, chunk.start_byte, chunk.end_byte,
        );
        let preview: String = chunk.content.chars().take(80).collect();
        let ellipsis = if chunk.content.len() > 80 { "..." } else { "" };
        println!("  \"{preview}{ellipsis}\"\n");
    }

    // demonstrate that consecutive chunks share content at boundaries
    println!("--- overlap verification ---\n");
    for i in 0..with_overlap.len().saturating_sub(1) {
        let current = &with_overlap[i];
        let next = &with_overlap[i + 1];

        // check if byte ranges overlap
        if next.start_byte < current.end_byte {
            let shared_bytes = current.end_byte - next.start_byte;
            println!(
                "chunks {} and {} share {} bytes (bytes {}..{})",
                i,
                i + 1,
                shared_bytes,
                next.start_byte,
                current.end_byte,
            );
            // extract the shared text from the original document
            let shared = &text[next.start_byte..current.end_byte];
            let preview: String = shared.chars().take(60).collect();
            println!("  shared: \"{preview}\"\n");
        }
    }
}
