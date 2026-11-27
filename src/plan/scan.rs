//! Separator cuts, found by one forward pass per tier over the planned range
//! and only as far as the planner has asked for.

use crate::strategy::recursive::absorb_trailing;

/// Separators grouped by their first byte, longest first in each group, so
/// the first that matches is the longest.
pub(crate) struct Matcher {
    first: [u8; 256],
    groups: Vec<Vec<&'static [u8]>>,
    /// Every separator starts with `\n`, or with `\r\n`: the scan can jump
    /// from one newline to the next.
    newline: bool,
}

impl Matcher {
    pub(crate) fn new(seps: &[&'static str]) -> Self {
        let mut first = [0u8; 256];
        let mut groups: Vec<Vec<&'static [u8]>> = Vec::new();
        for sep in seps {
            let b = sep.as_bytes()[0] as usize;
            if first[b] == 0 {
                groups.push(Vec::new());
                first[b] = groups.len() as u8;
            }
            groups[first[b] as usize - 1].push(sep.as_bytes());
        }
        for g in &mut groups {
            g.sort_by_key(|s| std::cmp::Reverse(s.len()));
        }
        let newline = seps
            .iter()
            .all(|s| s.starts_with('\n') || s.starts_with("\r\n"));
        Matcher {
            first,
            groups,
            newline,
        }
    }

    /// Length of the longest separator that starts at `i`.
    #[inline]
    fn at(&self, bytes: &[u8], i: usize) -> Option<usize> {
        let g = self.first[bytes[i] as usize];
        if g == 0 {
            return None;
        }
        let rest = &bytes[i..];
        // a byte loop: separators are a few bytes, too short for a memcmp call
        self.groups[g as usize - 1]
            .iter()
            .find(|s| rest.len() >= s.len() && rest.iter().zip(s.iter()).all(|(a, b)| a == b))
            .map(|s| s.len())
    }
}

/// The cuts of one tier: given by the strategy, or found on demand.
///
/// Newline tiers are scanned forward, newline to newline. Other tiers have a
/// match every few dozen bytes in prose, so they are searched backwards from
/// the window's end, which stops at the first match instead of reading the
/// whole window.
pub(crate) enum Scan<'m> {
    Fixed(Vec<usize>),
    Back(&'m Matcher),
    Seps {
        matcher: &'m Matcher,
        cuts: Vec<usize>,
        /// Bytes before this offset have been scanned.
        pos: usize,
        /// The document's newlines, and the index of the first one not
        /// passed yet.
        newlines: &'m [usize],
        next: usize,
    },
}

impl<'m> Scan<'m> {
    pub fn fixed(cuts: &[usize], lo: usize, hi: usize) -> Self {
        Scan::Fixed(cuts.iter().copied().filter(|&c| c > lo && c < hi).collect())
    }

    pub fn seps(matcher: &'m Matcher, lo: usize, newlines: &'m [usize]) -> Self {
        if !matcher.newline {
            return Scan::Back(matcher);
        }
        Scan::Seps {
            matcher,
            cuts: Vec::new(),
            pos: lo,
            // a section of a longer document starts its walk at its own start
            next: newlines.partition_point(|&p| p < lo),
            newlines,
        }
    }

    /// The last cut in `(floor, lim]` of the range `..hi` of `text`: the end
    /// of a piece `split_at_any` would produce on the range, separator and
    /// trailing closers included.
    ///
    /// `from` is the start of the chunk being planned. A tier that has not
    /// been scanned that far starts there: the chunk start is itself a cut,
    /// and the bytes before it are never asked about again.
    pub fn last_in(
        &mut self,
        text: &str,
        hi: usize,
        from: usize,
        floor: usize,
        lim: usize,
    ) -> Option<usize> {
        let cuts = match self {
            Scan::Back(matcher) => return last_before(matcher, &text[..hi], from, floor, lim),
            Scan::Fixed(cuts) => cuts,
            Scan::Seps {
                matcher,
                cuts,
                pos,
                newlines,
                next,
            } => {
                *pos = (*pos).max(from);
                if *pos < lim {
                    let mut lines = Lines {
                        all: newlines,
                        next: *next,
                    };
                    *pos = extend(matcher, &text[..hi], *pos, lim, cuts, &mut lines);
                    *next = lines.next;
                }
                cuts
            }
        };
        let i = cuts.partition_point(|&c| c <= lim);
        (i > 0 && cuts[i - 1] > floor).then(|| cuts[i - 1])
    }
}

/// The last piece end in `(floor, lim]` of a match that starts at or after
/// `from`, found by walking back from `lim`.
fn last_before(m: &Matcher, text: &str, from: usize, floor: usize, lim: usize) -> Option<usize> {
    let bytes = text.as_bytes();
    let mut i = lim.min(bytes.len());
    while i > from {
        i -= 1;
        if m.first[bytes[i] as usize] == 0 {
            continue;
        }
        let Some(len) = m.at(bytes, i) else { continue };
        let end = absorb_trailing(text, i + len);
        if end <= floor {
            return None;
        }
        if end <= lim && end < bytes.len() {
            return Some(end);
        }
    }
    None
}

/// A cursor over the document's newline offsets.
struct Lines<'a> {
    all: &'a [usize],
    next: usize,
}

impl Lines<'_> {
    /// The first newline at or after `pos`.
    #[inline]
    fn at_or_after(&mut self, pos: usize) -> Option<usize> {
        // `next` only moves forward, so the steps add up to one pass
        while self.all.get(self.next).is_some_and(|&p| p < pos) {
            self.next += 1;
        }
        self.all.get(self.next).copied()
    }
}

/// Offsets of every `\n` in `text`.
pub(crate) fn newlines(text: &str) -> Vec<usize> {
    let mut out = Vec::new();
    let (blocks, tail) = text.as_bytes().as_chunks::<32>();
    let mut base = 0;
    for block in blocks {
        // the test and the mask are written so that they vectorise
        let any = block.iter().fold(0u8, |a, &b| a | u8::from(b == b'\n'));
        if any != 0 {
            let mut mask = block
                .iter()
                .enumerate()
                .fold(0u32, |m, (j, &b)| m | (u32::from(b == b'\n') << j));
            while mask != 0 {
                out.push(base + mask.trailing_zeros() as usize);
                mask &= mask - 1;
            }
        }
        base += 32;
    }
    let tail = tail.iter().enumerate();
    out.extend(tail.filter(|&(_, &b)| b == b'\n').map(|(j, _)| base + j));
    out
}

/// Scan `text` from `pos` until a match starting at or after `until` would be
/// needed, appending piece ends below `text.len()`; returns the new position.
/// A newline tier steps through `lines`, the newlines of a text that starts
/// with `text`.
fn extend(
    m: &Matcher,
    text: &str,
    mut pos: usize,
    until: usize,
    cuts: &mut Vec<usize>,
    lines: &mut Lines<'_>,
) -> usize {
    let bytes = text.as_bytes();
    let n = bytes.len();
    while pos < until.min(n) {
        let hit = if m.newline {
            let Some(p) = lines.at_or_after(pos).filter(|&p| p < n) else {
                return n;
            };
            // a `\r\n` separator starts one byte before its newline
            let crlf = (p > pos && bytes[p - 1] == b'\r')
                .then(|| m.at(bytes, p - 1).map(|len| (p - 1, len)))
                .flatten();
            match crlf.or_else(|| m.at(bytes, p).map(|len| (p, len))) {
                Some(hit) => Some(hit),
                None => {
                    pos = p + 1;
                    None
                }
            }
        } else if m.first[bytes[pos] as usize] == 0 {
            pos += 1;
            None
        } else {
            match m.at(bytes, pos) {
                Some(len) => Some((pos, len)),
                None => {
                    pos += 1;
                    None
                }
            }
        };
        if let Some((start, len)) = hit {
            let end = absorb_trailing(text, start + len);
            if end < n {
                cuts.push(end);
            }
            pos = end;
        }
    }
    pos
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::strategy::code::CODE_TIERS;
    use crate::strategy::recursive::{TEXT_TIERS, split_at_any};

    /// Piece ends of the plain splitter on `text[lo..]`, minus the end.
    fn reference(text: &str, lo: usize, seps: &[&str]) -> Vec<usize> {
        let mut ends = Vec::new();
        let mut at = lo;
        for p in split_at_any(&text[lo..], seps) {
            at += p.len();
            if at < text.len() {
                ends.push(at);
            }
        }
        ends
    }

    #[test]
    fn newline_tiers_qualify_for_the_newline_scan() {
        for tier in TEXT_TIERS.iter().chain(CODE_TIERS) {
            let newline = tier.iter().all(|s| s.contains('\n'));
            assert_eq!(Matcher::new(tier).newline, newline, "{tier:?}");
        }
    }

    #[test]
    fn cuts_match_the_plain_splitter() {
        let alphabet = [
            "a", " ", "\n", "\n\n", "\r\n", "\r", "」", "\n}\n", "\nend\n", "\n};\r\n", "}", "\t",
            ". ", "。", "，", "…", "語", ", ", "? ", "）", "\u{3000}",
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
            for tier in TEXT_TIERS.iter().chain(CODE_TIERS) {
                let want = reference(&text, 0, tier);
                // ask in small steps, as the planner does
                let matcher = Matcher::new(tier);
                let nl = newlines(&text);
                let mut scan = Scan::seps(&matcher, 0, &nl);
                let mut got = Vec::new();
                for lim in 1..=text.len() {
                    if let Some(c) = scan.last_in(&text, text.len(), 0, 0, lim)
                        && got.last() != Some(&c)
                    {
                        got.push(c);
                    }
                }
                assert_eq!(got, want, "{text:?} {tier:?}");
            }
        }
    }

    #[test]
    fn newlines_are_every_newline() {
        let mut x: u64 = 0x2545_f491;
        for len in 0..300 {
            let text: String = (0..len)
                .map(|_| {
                    x ^= x << 13;
                    x ^= x >> 7;
                    x ^= x << 17;
                    ['a', '\n', '語', ' '][(x % 4) as usize]
                })
                .collect();
            let want: Vec<usize> = text.match_indices('\n').map(|(i, _)| i).collect();
            assert_eq!(newlines(&text), want, "{text:?}");
        }
    }

    #[test]
    fn last_in_respects_the_window() {
        let text = "One. Two! Three? Four. Five";
        let ends = reference(text, 0, TEXT_TIERS[2]);
        for lim in 0..=text.len() {
            for floor in 0..lim {
                let want = ends.iter().copied().rfind(|&e| e <= lim && e > floor);
                let matcher = Matcher::new(TEXT_TIERS[2]);
                let mut scan = Scan::seps(&matcher, 0, &[]);
                assert_eq!(scan.last_in(text, text.len(), 0, floor, lim), want);
            }
        }
    }
}
