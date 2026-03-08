use crate::Chunk;
use crate::recursive::split_recursive;
use tiktoken::CoreBpe;

/// split text at semantic boundaries using embedding similarity.
///
/// algorithm:
/// 1. split text into sentences
/// 2. embed each sentence
/// 3. compute cosine similarity between consecutive sentence embeddings
/// 4. find breakpoints where similarity drops below the threshold
/// 5. merge sentences between breakpoints into chunks
/// 6. if a chunk exceeds max_tokens, apply recursive splitting
pub(crate) async fn split_semantic(
    text: &str,
    max_tokens: usize,
    overlap_tokens: usize,
    encoder: &CoreBpe,
    client: &embedrs::Client,
    threshold: f64,
) -> Result<Vec<Chunk>, crate::Error> {
    if text.is_empty() {
        return Ok(Vec::new());
    }

    let sentences = split_sentences(text);
    if sentences.len() <= 1 {
        let chunks = split_recursive(text, 0, max_tokens, overlap_tokens, encoder, &None);
        return Ok(chunks);
    }

    // embed all sentences
    let sentence_texts: Vec<String> = sentences.iter().map(|s| s.text.to_string()).collect();
    let embed_result = client
        .embed(sentence_texts)
        .await
        .map_err(crate::Error::Embed)?;

    // find breakpoints by cosine similarity between consecutive sentences
    let breakpoints = find_breakpoints(&embed_result.embeddings, threshold);

    // merge sentences into groups at breakpoints
    let groups = merge_at_breakpoints(&sentences, &breakpoints);

    let mut all_chunks = Vec::new();
    for group in &groups {
        let content: String = group.iter().map(|s| s.text).collect::<Vec<_>>().join("");
        let group_offset = group.first().map(|s| s.byte_offset).unwrap_or(0);
        let sub_chunks = split_recursive(
            &content,
            group_offset,
            max_tokens,
            overlap_tokens,
            encoder,
            &None,
        );
        all_chunks.extend(sub_chunks);
    }

    for (i, chunk) in all_chunks.iter_mut().enumerate() {
        chunk.index = i;
    }

    Ok(all_chunks)
}

struct Sentence<'a> {
    text: &'a str,
    byte_offset: usize,
}

/// split text into sentences (by `. `, `! `, `? `, or newlines)
fn split_sentences(text: &str) -> Vec<Sentence<'_>> {
    let mut sentences = Vec::new();
    let mut start = 0;

    let terminators = [". ", "! ", "? ", ".\n", "!\n", "?\n"];

    let mut i = 0;
    while i < text.len() {
        let mut found = false;
        for term in &terminators {
            if text[i..].starts_with(term) {
                let end = i + term.len();
                let s = &text[start..end];
                if !s.trim().is_empty() {
                    sentences.push(Sentence {
                        text: s,
                        byte_offset: start,
                    });
                }
                start = end;
                i = end;
                found = true;
                break;
            }
        }
        if !found {
            // advance by one character, not one byte (multi-byte UTF-8 safety)
            i += text[i..].chars().next().map_or(1, |c| c.len_utf8());
        }
    }

    // remaining text
    if start < text.len() {
        let s = &text[start..];
        if !s.trim().is_empty() {
            sentences.push(Sentence {
                text: s,
                byte_offset: start,
            });
        }
    }

    sentences
}

/// find indices where cosine similarity between consecutive embeddings drops below threshold
fn find_breakpoints(embeddings: &[Vec<f32>], threshold: f64) -> Vec<usize> {
    let mut breakpoints = Vec::new();
    for i in 1..embeddings.len() {
        let sim = cosine_similarity(&embeddings[i - 1], &embeddings[i]);
        if sim < threshold {
            breakpoints.push(i);
        }
    }
    breakpoints
}

fn cosine_similarity(a: &[f32], b: &[f32]) -> f64 {
    let mut dot = 0.0f64;
    let mut norm_a = 0.0f64;
    let mut norm_b = 0.0f64;
    for (x, y) in a.iter().zip(b.iter()) {
        let x = *x as f64;
        let y = *y as f64;
        dot += x * y;
        norm_a += x * x;
        norm_b += y * y;
    }
    let denom = norm_a.sqrt() * norm_b.sqrt();
    if denom == 0.0 { 0.0 } else { dot / denom }
}

/// group sentences by breakpoints
fn merge_at_breakpoints<'a>(
    sentences: &'a [Sentence<'a>],
    breakpoints: &[usize],
) -> Vec<Vec<&'a Sentence<'a>>> {
    let mut groups: Vec<Vec<&Sentence<'a>>> = Vec::new();
    let mut current_group: Vec<&Sentence<'a>> = Vec::new();
    let break_set: std::collections::HashSet<usize> = breakpoints.iter().copied().collect();

    for (i, sentence) in sentences.iter().enumerate() {
        if break_set.contains(&i) && !current_group.is_empty() {
            groups.push(current_group);
            current_group = Vec::new();
        }
        current_group.push(sentence);
    }

    if !current_group.is_empty() {
        groups.push(current_group);
    }

    groups
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn split_sentences_basic() {
        let sentences = split_sentences("First sentence. Second sentence. Third.");
        assert_eq!(sentences.len(), 3);
        assert_eq!(sentences[0].text, "First sentence. ");
        assert_eq!(sentences[1].text, "Second sentence. ");
        assert_eq!(sentences[2].text, "Third.");
    }

    #[test]
    fn split_sentences_empty() {
        let sentences = split_sentences("");
        assert!(sentences.is_empty());
    }

    #[test]
    fn split_sentences_no_terminators() {
        let sentences = split_sentences("no terminators here");
        assert_eq!(sentences.len(), 1);
        assert_eq!(sentences[0].text, "no terminators here");
    }

    #[test]
    fn split_sentences_with_exclamation() {
        let sentences = split_sentences("Hello! How are you? Fine.");
        assert_eq!(sentences.len(), 3);
    }

    #[test]
    fn split_sentences_preserves_offset() {
        let sentences = split_sentences("First. Second.");
        assert_eq!(sentences[0].byte_offset, 0);
        assert_eq!(sentences[1].byte_offset, 7);
    }

    #[test]
    fn find_breakpoints_high_threshold() {
        // all similarities will be below 1.0, so all indices should be breakpoints
        let embeddings = vec![
            vec![1.0, 0.0, 0.0],
            vec![0.0, 1.0, 0.0],
            vec![0.0, 0.0, 1.0],
        ];
        let bp = find_breakpoints(&embeddings, 0.5);
        assert_eq!(bp, vec![1, 2]);
    }

    #[test]
    fn find_breakpoints_low_threshold() {
        let embeddings = vec![
            vec![1.0, 0.0, 0.0],
            vec![0.9, 0.1, 0.0],
            vec![0.8, 0.2, 0.0],
        ];
        let bp = find_breakpoints(&embeddings, 0.1);
        // similarities are high, no breakpoints
        assert!(bp.is_empty());
    }

    #[test]
    fn cosine_similarity_identical() {
        let sim = cosine_similarity(&[1.0, 0.0], &[1.0, 0.0]);
        assert!((sim - 1.0).abs() < 1e-6);
    }

    #[test]
    fn cosine_similarity_orthogonal() {
        let sim = cosine_similarity(&[1.0, 0.0], &[0.0, 1.0]);
        assert!(sim.abs() < 1e-6);
    }

    #[test]
    fn merge_at_breakpoints_basic() {
        let s1 = Sentence {
            text: "a",
            byte_offset: 0,
        };
        let s2 = Sentence {
            text: "b",
            byte_offset: 1,
        };
        let s3 = Sentence {
            text: "c",
            byte_offset: 2,
        };
        let sentences = vec![s1, s2, s3];
        let groups = merge_at_breakpoints(&sentences, &[2]);
        assert_eq!(groups.len(), 2);
        assert_eq!(groups[0].len(), 2); // a, b
        assert_eq!(groups[1].len(), 1); // c
    }

    #[test]
    fn split_sentences_cjk_no_panic() {
        // must not panic on multi-byte UTF-8 characters (regression test)
        let sentences = split_sentences("这是第一句。第二句在这里。最后一句。");
        // CJK full-width period is not in the ASCII terminator list,
        // so this returns 1 sentence — but must not panic
        assert!(!sentences.is_empty());
        for s in &sentences {
            assert!(!s.text.contains('\u{FFFD}'));
        }
    }

    #[test]
    fn split_sentences_cjk_with_ascii_terminators() {
        // CJK text with ASCII-style terminators works correctly
        let sentences = split_sentences("这是第一句. 第二句在这里. 最后一句.");
        assert_eq!(sentences.len(), 3);
    }

    #[test]
    fn split_sentences_japanese_no_panic() {
        // must not panic on Japanese multi-byte characters
        let sentences = split_sentences("最初の文です。次の文です。最後。");
        assert!(!sentences.is_empty());
    }

    #[test]
    fn merge_at_breakpoints_no_breaks() {
        let s1 = Sentence {
            text: "a",
            byte_offset: 0,
        };
        let s2 = Sentence {
            text: "b",
            byte_offset: 1,
        };
        let sentences = vec![s1, s2];
        let groups = merge_at_breakpoints(&sentences, &[]);
        assert_eq!(groups.len(), 1);
        assert_eq!(groups[0].len(), 2);
    }
}
