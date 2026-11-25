//! Token-first chunk planner shared by every strategy.
//!
//! The document is encoded once. From a chunk's start, the byte offset after
//! `max_tokens` document tokens bounds where the chunk may end; within that
//! window the chunk ends at the last cut of the highest-priority separator
//! tier that has one, or at a token boundary when no tier does. The chosen
//! text is then counted on its own once, because BPE is not additive across a
//! cut, and the window shrinks in the rare case that count is over budget.

use crate::Chunk;
use crate::recursive::absorb_trailing;
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

/// Cut positions of every tier whose separators all contain a newline, found
/// up front with one newline search; the other tiers are searched lazily in
/// the window of the chunk being planned.
struct Cuts {
    eager: Vec<Option<Vec<usize>>>,
}

impl Cuts {
    fn new(text: &str, lo: usize, hi: usize, tiers: &[Tier]) -> Self {
        let eager = tiers
            .iter()
            .map(|t| match t {
                Tier::Cuts(c) => Some(c.iter().copied().filter(|&c| c > lo && c < hi).collect()),
                Tier::Seps(seps) if seps.iter().all(|s| s.contains('\n')) => {
                    Some(newline_cuts(&text[lo..hi], seps, lo))
                }
                Tier::Seps(_) => None,
            })
            .collect();
        Cuts { eager }
    }
}

/// Offset of a separator's first newline; at most 1 for every tier in use.
fn newline_offset(sep: &str) -> usize {
    sep.find('\n').unwrap_or(0)
}

/// Piece ends of `split_at_any(text, seps)` without testing every byte: every
/// match contains a newline, so only the bytes around each newline are tried.
fn newline_cuts(text: &str, seps: &[&str], base: usize) -> Vec<usize> {
    let bytes = text.as_bytes();
    let mut cuts = Vec::new();
    let mut resume = 0usize;
    let mut from = 0usize;
    while let Some(p) = text[from..].find('\n').map(|p| p + from) {
        from = p + 1;
        // earliest start, then longest separator
        let mut hit: Option<(usize, usize)> = None;
        for sep in seps {
            let q = newline_offset(sep);
            let Some(c) = p.checked_sub(q) else { continue };
            if c < resume || !bytes[c..].starts_with(sep.as_bytes()) {
                continue;
            }
            let better = match hit {
                None => true,
                Some((hc, hl)) => c < hc || (c == hc && sep.len() > hl),
            };
            if better {
                hit = Some((c, sep.len()));
            }
        }
        if let Some((c, len)) = hit {
            let end = absorb_trailing(text, c + len);
            if end < text.len() {
                cuts.push(base + end);
            }
            resume = end;
            from = from.max(end);
        }
    }
    cuts
}

/// The last piece end of `split_at_any(&text[from..], seps)` that lies in
/// `(floor, lim]`.
fn last_cut(text: &str, from: usize, floor: usize, lim: usize, seps: &[&str]) -> Option<usize> {
    let bytes = text.as_bytes();
    let mut lead = [false; 256];
    for sep in seps {
        lead[sep.as_bytes()[0] as usize] = true;
    }
    let mut best = None;
    let mut i = from;
    while i < lim {
        if !lead[bytes[i] as usize] {
            i += 1;
            continue;
        }
        let rest = &bytes[i..];
        let Some(len) = seps
            .iter()
            .filter(|s| rest.starts_with(s.as_bytes()))
            .map(|s| s.len())
            .max()
        else {
            i += 1;
            continue;
        };
        let end = absorb_trailing(text, i + len);
        if end > lim {
            break;
        }
        if end > floor {
            best = Some(end);
        }
        i = end;
    }
    best
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

struct Planner<'d, 't> {
    doc: &'d Doc<'d>,
    tiers: &'t [Tier],
    cuts: Cuts,
    hi: usize,
    max: usize,
}

impl Planner<'_, '_> {
    /// The highest-priority cut in `(floor, lim]`, the latest of its tier.
    fn best_cut(&self, s: usize, floor: usize, lim: usize) -> Option<usize> {
        let text = self.doc.text;
        self.tiers
            .iter()
            .zip(&self.cuts.eager)
            .find_map(|(tier, eager)| match (tier, eager) {
                (_, Some(cuts)) => {
                    let i = cuts.partition_point(|&c| c <= lim);
                    (i > 0 && cuts[i - 1] > floor).then(|| cuts[i - 1])
                }
                (Tier::Seps(seps), None) => last_cut(text, s, floor, lim, seps),
                (Tier::Cuts(_), None) => None,
            })
    }

    /// End and token count of the chunk starting at `s` that must end after
    /// `floor`. Over budget only when it is one character past `floor`.
    fn choose(&self, s: usize, floor: usize) -> (usize, usize) {
        let doc = self.doc;
        let text = doc.text;
        let one = next_char_boundary(text, floor);
        let t = doc.token_at(s);
        let mut lim = doc.bounds[(t + self.max).min(doc.tokens())].min(self.hi);
        loop {
            let e = if lim >= self.hi {
                self.hi
            } else {
                self.best_cut(s, floor, lim).unwrap_or_else(|| {
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
    let planner = Planner {
        doc,
        tiers,
        cuts: Cuts::new(doc.text, lo, hi, tiers),
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::recursive::{TEXT_TIERS, split_at_any};

    /// Piece ends of the plain splitter, minus the text end.
    fn reference(text: &str, seps: &[&str]) -> Vec<usize> {
        let mut ends = Vec::new();
        let mut at = 0;
        for p in split_at_any(text, seps) {
            at += p.len();
            if at < text.len() {
                ends.push(at);
            }
        }
        ends
    }

    #[test]
    fn newline_cuts_match_the_plain_splitter() {
        let alphabet = [
            "a", " ", "\n", "\n\n", "\r\n", "\r", "」", "\n}\n", "\nend\n", "\n};\r\n", "}", "\t",
        ];
        let mut x: u64 = 0x9e37_79b9;
        for _ in 0..3000 {
            let mut text = String::new();
            for _ in 0..30 {
                x ^= x << 13;
                x ^= x >> 7;
                x ^= x << 17;
                text.push_str(alphabet[(x % alphabet.len() as u64) as usize]);
            }
            for tier in TEXT_TIERS.iter().chain(crate::code::CODE_TIERS) {
                if tier.iter().all(|s| s.contains('\n')) {
                    assert_eq!(
                        newline_cuts(&text, tier, 0),
                        reference(&text, tier),
                        "{text:?}"
                    );
                }
            }
        }
    }

    #[test]
    fn newline_offsets_are_at_most_one() {
        for tier in TEXT_TIERS.iter().chain(crate::code::CODE_TIERS) {
            for sep in tier.iter().filter(|s| s.contains('\n')) {
                assert!(newline_offset(sep) <= 1, "{sep:?}");
            }
        }
    }

    #[test]
    fn last_cut_matches_the_plain_splitter() {
        let text = "One. Two! Three? Four. Five";
        let seps = TEXT_TIERS[2];
        let ends = reference(text, seps);
        for lim in 0..=text.len() {
            let want = ends.iter().copied().rfind(|&e| e <= lim);
            assert_eq!(last_cut(text, 0, 0, lim, seps), want, "lim {lim}");
        }
    }
}
