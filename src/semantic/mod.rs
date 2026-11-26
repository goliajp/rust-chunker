//! Semantic splitting: break where neighbouring sentences stop being similar.
//!
//! The chunker does no I/O here. [`Chunker::semantic`] cuts the text into
//! units (sentences, none longer than the token budget) and hands them out
//! as [`Semantic::inputs`]; the caller embeds them with whatever it uses and
//! gives the vectors back to [`Semantic::finish`]. With the `semantic`
//! feature, [`Semantic::embed`] and [`Chunker::split_semantic`] do the
//! embedding through any [`embedrs::Embed`].

#[cfg(feature = "semantic")]
mod embed;

use crate::plan::{Doc, Tier, plan, tiers};
use crate::strategy::recursive::{TEXT_TIERS, split_at_any};
use crate::{Chunk, Chunker, Error, Result};

/// Where to break between two neighbouring units.
#[derive(Debug, Clone, Copy, PartialEq)]
#[non_exhaustive]
pub enum Breakpoints {
    /// Break where the cosine distance between neighbours is above this
    /// percentile (0 to 100) of all the document's neighbour distances. It
    /// adapts to the model and the document. The default, at 95.
    Percentile(f32),
    /// Break where the cosine similarity between neighbours is below this
    /// value. What a good value is depends on the model.
    Threshold(f32),
}

impl Default for Breakpoints {
    fn default() -> Self {
        Breakpoints::Percentile(95.0)
    }
}

/// A text prepared for semantic splitting: its units are cut, and the
/// chunker waits for their vectors. Made by [`Chunker::semantic`].
///
/// ```rust
/// let text = "Cats purr. Cats nap. Stocks fell. Markets slid.";
/// let job = chunkedrs::Chunker::new().semantic(text);
/// assert_eq!(job.inputs(), ["Cats purr. ", "Cats nap. ", "Stocks fell. ", "Markets slid."]);
///
/// // vectors from any embedding model, one per input, in order
/// let vectors = [1.0, 0.0, 0.9, 0.1, 0.0, 1.0, 0.1, 0.9];
/// let chunks = job
///     .breakpoints(chunkedrs::Breakpoints::Threshold(0.5))
///     .finish(&vectors, 2)?;
/// assert_eq!(chunks[0].content, "Cats purr. Cats nap. ");
/// assert_eq!(chunks[1].content, "Stocks fell. Markets slid.");
/// # Ok::<(), chunkedrs::Error>(())
/// ```
pub struct Semantic<'a> {
    chunker: Chunker,
    doc: Doc<'a>,
    /// Byte ranges of the units in the text.
    units: Vec<(usize, usize)>,
    inputs: Vec<&'a str>,
    rule: Breakpoints,
}

impl std::fmt::Debug for Semantic<'_> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Semantic")
            .field("inputs", &self.inputs.len())
            .field("rule", &self.rule)
            .finish_non_exhaustive()
    }
}

impl Chunker {
    /// Prepare `text` for semantic splitting; see [`Semantic`].
    #[must_use]
    pub fn semantic<'a>(&self, text: &'a str) -> Semantic<'a> {
        let doc = Doc::new(text, self.encoder);
        let units = units(&doc, self.max_tokens);
        let inputs = units.iter().map(|&(s, e)| &text[s..e]).collect();
        Semantic {
            chunker: *self,
            doc,
            units,
            inputs,
            rule: Breakpoints::default(),
        }
    }
}

impl<'a> Semantic<'a> {
    /// The rule for where to break. Default [`Breakpoints::Percentile`]`(95.0)`.
    #[must_use]
    pub fn breakpoints(mut self, rule: Breakpoints) -> Self {
        self.rule = rule;
        self
    }

    /// The texts to embed, in order: the text's sentences, each cut further
    /// if it is longer than the chunker's `max_tokens`. Slices of the text.
    #[must_use]
    pub fn inputs(&self) -> &[&'a str] {
        &self.inputs
    }

    /// Split with the vectors of [`inputs`](Self::inputs): `vectors` holds
    /// one vector of `dim` values per input, in order, back to back (what
    /// `embedrs::EmbedResult::as_flat` gives).
    ///
    /// Units between two breakpoints form a group, and each group is split
    /// like the recursive strategy does, so every chunk holds the token
    /// budget.
    ///
    /// # Errors
    ///
    /// [`Error::Vectors`] when `vectors` is not `inputs().len()` vectors of
    /// `dim` values.
    pub fn finish(self, vectors: &[f32], dim: usize) -> Result<Vec<Chunk<'a>>> {
        let n = self.units.len();
        if vectors.len() != n * dim || (n > 0 && dim == 0) {
            return Err(Error::Vectors {
                inputs: n,
                values: vectors.len(),
                dim,
            });
        }
        let (max, overlap) = (self.chunker.max_tokens, self.chunker.overlap);
        if n <= 1 {
            return Ok(self.chunker.recursive().split_doc(&self.doc));
        }
        let sims: Vec<f32> = vectors
            .chunks_exact(dim)
            .collect::<Vec<_>>()
            .windows(2)
            .map(|w| cosine(w[0], w[1]))
            .collect();
        let breaks = breakpoints(&sims, self.rule);
        let ladder = tiers(TEXT_TIERS);
        let mut chunks = Vec::new();
        let mut first = 0;
        for end in breaks.into_iter().chain(std::iter::once(n)) {
            let (lo, hi) = (self.units[first].0, self.units[end - 1].1);
            chunks.extend(
                plan(&self.doc, lo, hi, &ladder, max, overlap)
                    .into_iter()
                    .map(|p| self.doc.chunk(p, &[])),
            );
            first = end;
        }
        crate::chunker::finish(&mut chunks);
        Ok(chunks)
    }
}

/// Indices `i` (`0 < i < sims.len() + 1`) where a new group starts, given
/// the similarity of each unit to the next.
fn breakpoints(sims: &[f32], rule: Breakpoints) -> Vec<usize> {
    let breaks: Box<dyn Fn(f32) -> bool> = match rule {
        Breakpoints::Threshold(t) => Box::new(move |s| s < t),
        Breakpoints::Percentile(p) => {
            let cut = percentile(sims.iter().map(|s| 1.0 - s).collect(), p);
            Box::new(move |s| 1.0 - s > cut)
        }
    };
    sims.iter()
        .enumerate()
        .filter(|&(_, &s)| breaks(s))
        .map(|(i, _)| i + 1)
        .collect()
}

/// The `p`-th percentile of `xs`, linearly interpolated between ranks.
fn percentile(mut xs: Vec<f32>, p: f32) -> f32 {
    xs.sort_by(f32::total_cmp);
    let rank = (p.clamp(0.0, 100.0) / 100.0) * (xs.len() - 1) as f32;
    let (lo, hi) = (rank.floor() as usize, rank.ceil() as usize);
    xs[lo] + (xs[hi] - xs[lo]) * (rank - lo as f32)
}

fn cosine(a: &[f32], b: &[f32]) -> f32 {
    let (mut dot, mut na, mut nb) = (0.0f32, 0.0f32, 0.0f32);
    for (x, y) in a.iter().zip(b) {
        dot += x * y;
        na += x * x;
        nb += y * y;
    }
    if na == 0.0 || nb == 0.0 {
        return 0.0;
    }
    dot / (na.sqrt() * nb.sqrt())
}

/// The units to embed: sentences that are not blank, and a sentence longer
/// than `max` document tokens planned into pieces that fit.
fn units(doc: &Doc<'_>, max: usize) -> Vec<(usize, usize)> {
    let text = doc.text;
    let seps: Vec<&str> = TEXT_TIERS[SENTENCE_TIER]
        .iter()
        .copied()
        // a newline ends a sentence too; the line tier ranks above the
        // sentence tier in the recursive ladder, so the sentence tier has none
        .chain([".\n", "!\n", "?\n", "\n"])
        .collect();
    let base = text.as_ptr() as usize;
    let mut out = Vec::new();
    let mut fixed: Option<Vec<Tier>> = None;
    for s in split_at_any(text, &seps) {
        if s.trim().is_empty() {
            continue;
        }
        let start = s.as_ptr() as usize - base;
        let end = start + s.len();
        if doc.span_width(start, end) <= max {
            out.push((start, end));
            continue;
        }
        let ladder = fixed.get_or_insert_with(|| tiers(TEXT_TIERS));
        out.extend(
            plan(doc, start, end, ladder, max, 0)
                .into_iter()
                .filter(|p| !text[p.start..p.end].trim().is_empty())
                .map(|p| (p.start, p.end)),
        );
    }
    out
}

/// Paragraph, line, then sentence in [`TEXT_TIERS`].
const SENTENCE_TIER: usize = 2;

#[cfg(test)]
mod tests;
