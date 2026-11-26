//! The token count of one planned chunk, as if it were encoded on its own.
//!
//! BPE is not additive across an arbitrary cut, which is why the planner
//! counts every chunk again. But at a cut tiktoken reports
//! ([`CoreBpe::next_cut`](tiktoken::CoreBpe::next_cut)), encoding the two
//! sides separately gives the ids of the whole, and whether a position is a
//! cut depends only on the bytes around it, so a cut of the chunk is also a
//! cut of the document at the same place. With `h` the chunk's first cut and
//! `t` its last,
//!
//! ```text
//! count(chunk) = count(chunk[..h]) + (document tokens between h and t) + count(chunk[t..])
//! ```
//!
//! and only the two ends are encoded again: the partial word or clause the
//! chunk boundary cut out of a piece. The answer is exactly the count of the
//! whole chunk, so the planner's choices do not change.

use super::Doc;

/// `count(&doc.text[s..e])`, reusing the document's ids between the chunk's
/// first and last cut.
pub(crate) fn count(doc: &Doc<'_>, s: usize, e: usize) -> usize {
    let text = &doc.text[s..e];
    let enc = doc.encoder;
    let Some(h) = enc.next_cut(text, 1) else {
        return enc.count(text);
    };
    // cuts found in `text[h..]` are cuts of `text`: same bytes around them
    let t = enc
        .prev_cut(&text[h..], text.len() - h)
        .map_or(h, |t| h + t);
    let middle = doc.boundary(s + t) - doc.boundary(s + h);
    enc.count(&text[..h]) + middle + enc.count(&text[t..])
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Every chunk-shaped slice of real text counts the same both ways.
    #[test]
    fn local_count_equals_a_whole_count() {
        let texts = [
            include_str!("../../bench-compare/data/book_en.txt"),
            include_str!("../../bench-compare/data/zh_xiyouji.md"),
            include_str!("../../bench-compare/data/ja_neko.txt"),
            include_str!("../../bench-compare/data/code_rust.txt"),
        ];
        for name in ["o200k_base", "cl100k_base"] {
            let enc = tiktoken::get_encoding(name).unwrap();
            for text in texts {
                let text = &text[..text.floor_char_boundary(60_000)];
                let doc = Doc::new(text, enc);
                let mut checked = 0;
                let mut s = 0;
                // starts and lengths that do not line up with anything
                for len in [7usize, 64, 333, 1500, 4097].iter().cycle().take(400) {
                    let e = text.floor_char_boundary(s + len);
                    if e <= s {
                        break;
                    }
                    assert_eq!(count(&doc, s, e), enc.count(&text[s..e]), "{name} {s}..{e}");
                    checked += 1;
                    s = text.ceil_char_boundary(s + len / 3 + 1);
                }
                assert!(checked > 100);
            }
        }
    }
}
