//! Token-first chunk planner shared by every strategy.
//!
//! The document is encoded once. From a chunk's start, the byte offset after
//! `max_tokens` document tokens bounds where the chunk may end; within that
//! window the chunk ends at the last cut of the highest-priority separator
//! tier that has one, or at a token boundary when no tier does. The chosen
//! text is then counted on its own once, because BPE is not additive across a
//! cut, and the window shrinks in the rare case that count is over budget.

use crate::Chunk;
use crate::scan::Scan;
use tiktoken::CoreBpe;

/// The document, its encoder, and where each of its tokens starts.
pub(crate) struct Doc<'a> {
    pub text: &'a str,
    pub encoder: &'a CoreBpe,
    /// `bounds[i]` is the byte offset at which token `i` starts; the last
    /// entry is the text length. A token can end inside a multi-byte
    /// character, so an entry is not always a char boundary.
    bounds: Vec<usize>,
}

impl<'a> Doc<'a> {
    pub fn new(text: &'a str, encoder: &'a CoreBpe) -> Self {
        let tokens = encoder.encode(text);
        let mut bounds = Vec::with_capacity(tokens.len() + 1);
        let mut acc = 0usize;
        bounds.push(0);
        for &t in &tokens {
            acc += crate::token_len(encoder, t);
            bounds.push(acc);
        }
        Doc {
            text,
            encoder,
            bounds,
        }
    }

    fn tokens(&self) -> usize {
        self.bounds.len() - 1
    }

    /// Index of the token that contains `byte`, or the token count at the end.
    fn token_at(&self, byte: usize) -> usize {
        self.bounds
            .partition_point(|&b| b <= byte)
            .saturating_sub(1)
    }

    /// Index of the first token boundary at or after `byte`.
    fn boundary_at_or_after(&self, byte: usize) -> usize {
        self.bounds.partition_point(|&b| b < byte)
    }

    /// The token range covering `start..end`, widened where a boundary falls
    /// inside a token.
    fn span(&self, start: usize, end: usize) -> (usize, usize) {
        let n = self.tokens();
        let first = self.token_at(start).min(n);
        let last = if end > start {
            self.boundary_at_or_after(end).min(n)
        } else {
            first
        };
        (first, last.max(first))
    }

    /// A chunk for `piece`, with its token span in the document.
    pub fn chunk(&self, piece: Piece, section_path: &[String]) -> Chunk {
        let (start_token, end_token) = self.span(piece.start, piece.end);
        Chunk {
            content: self.text[piece.start..piece.end].to_string(),
            index: 0,
            start_byte: piece.start,
            end_byte: piece.end,
            start_token,
            end_token,
            token_count: piece.tokens,
            section_path: section_path.to_vec(),
        }
    }
}

/// One planned chunk: a byte range of the document and its own token count.
#[derive(Clone, Copy, Debug)]
pub(crate) struct Piece {
    pub start: usize,
    pub end: usize,
    pub tokens: usize,
}

/// A separator tier: separators searched in the text, or cut positions a
/// strategy found itself (absolute byte offsets, ascending).
pub(crate) enum Tier {
    Seps(&'static [&'static str]),
    Cuts(Vec<usize>),
}

pub(crate) fn tiers(seps: &'static [&'static [&'static str]]) -> Vec<Tier> {
    seps.iter().map(|s| Tier::Seps(s)).collect()
}

fn floor_char_boundary(text: &str, i: usize) -> usize {
    let mut i = i.min(text.len());
    while !text.is_char_boundary(i) {
        i -= 1;
    }
    i
}

fn next_char_boundary(text: &str, i: usize) -> usize {
    let mut i = (i + 1).min(text.len());
    while !text.is_char_boundary(i) {
        i += 1;
    }
    i
}

struct Planner<'d> {
    doc: &'d Doc<'d>,
    scans: Vec<Scan>,
    hi: usize,
    max: usize,
}

impl Planner<'_> {
    /// The highest-priority cut in `(floor, lim]`, the latest of its tier.
    fn best_cut(&mut self, floor: usize, lim: usize) -> Option<usize> {
        let (text, hi) = (self.doc.text, self.hi);
        self.scans
            .iter_mut()
            .find_map(|scan| scan.last_in(text, hi, floor, lim))
    }

    /// End and token count of the chunk starting at `s` that must end after
    /// `floor`. Over budget only when it is one character past `floor`.
    fn choose(&mut self, s: usize, floor: usize) -> (usize, usize) {
        let doc = self.doc;
        let text = doc.text;
        let one = next_char_boundary(text, floor);
        let t = doc.token_at(s);
        let mut lim = doc.bounds[(t + self.max).min(doc.tokens())].min(self.hi);
        loop {
            let e = if lim >= self.hi {
                self.hi
            } else {
                self.best_cut(floor, lim).unwrap_or_else(|| {
                    let e = floor_char_boundary(text, lim);
                    if e <= floor { one } else { e }
                })
            };
            let n = doc.encoder.count(&text[s..e]);
            if n <= self.max || e <= one {
                return (e, n);
            }
            // shrink the window by the excess, measured in document tokens
            let back = doc.boundary_at_or_after(e).saturating_sub(n - self.max);
            let shrunk = doc.bounds[back];
            lim = if shrunk < e {
                shrunk
            } else {
                floor_char_boundary(text, e - 1)
            }
            .max(one);
        }
    }

    /// Start of the chunk after one that ran `s..e`: about `overlap` tokens
    /// before `e`, on a char boundary, and after `s`.
    fn overlap_start(&self, s: usize, e: usize, overlap: usize) -> usize {
        let doc = self.doc;
        let text = doc.text;
        let back = doc.boundary_at_or_after(e).saturating_sub(overlap);
        let mut start = doc.bounds[back];
        while start < text.len() && !text.is_char_boundary(start) {
            start += 1;
        }
        start.max(next_char_boundary(text, s)).min(e)
    }
}

/// Plan chunks over `lo..hi` of the document.
pub(crate) fn plan(
    doc: &Doc<'_>,
    lo: usize,
    hi: usize,
    tiers: &[Tier],
    max: usize,
    overlap: usize,
) -> Vec<Piece> {
    let mut out = Vec::new();
    if lo >= hi {
        return out;
    }
    let scans = tiers
        .iter()
        .map(|t| match t {
            Tier::Seps(seps) => Scan::seps(seps, lo),
            Tier::Cuts(cuts) => Scan::fixed(cuts, lo, hi),
        })
        .collect();
    let mut planner = Planner {
        doc,
        scans,
        hi,
        max,
    };
    // overlap below the budget, so every chunk adds at least one new token
    let overlap = overlap.min(max.saturating_sub(1));

    let mut s = lo;
    let mut prev_end = lo;
    while prev_end < hi {
        let (mut e, mut n) = planner.choose(s, prev_end);
        if n > max && s < prev_end {
            // the overlap leaves no room for new text: start without it
            s = prev_end;
            (e, n) = planner.choose(s, prev_end);
        }
        out.push(Piece {
            start: s,
            end: e,
            tokens: n,
        });
        prev_end = e;
        s = if overlap > 0 && e < hi {
            planner.overlap_start(s, e, overlap)
        } else {
            e
        };
    }
    out
}
