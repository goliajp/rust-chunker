use super::*;

fn chunker() -> Chunker {
    Chunker::new()
}

#[test]
fn units_are_sentences_slices_of_the_text() {
    let text = "First sentence. Second sentence. Third.";
    let job = chunker().semantic(text);
    assert_eq!(
        job.inputs(),
        ["First sentence. ", "Second sentence. ", "Third."]
    );
    for (&input, &(s, e)) in job.inputs().iter().zip(&job.units) {
        assert_eq!(&text[s..e], input);
        assert_eq!(input.as_ptr(), text[s..].as_ptr());
    }
}

#[test]
fn units_skip_blanks_and_split_at_cjk_marks_and_lines() {
    let job = chunker().semantic("这是第一句。第二句在这里。\n \nLine one\nLine two");
    assert_eq!(
        job.inputs(),
        ["这是第一句。", "第二句在这里。", "Line one\n", "Line two"]
    );
    assert!(chunker().semantic("").inputs().is_empty());
    assert!(chunker().semantic(" \n\t ").inputs().is_empty());
}

#[test]
fn long_sentences_are_cut_to_the_budget() {
    let text = "word ".repeat(300);
    let c = chunker().max_tokens(20);
    let job = c.semantic(&text);
    assert!(job.inputs().len() >= 15);
    for input in job.inputs() {
        assert!(c.encoder().count(input) <= 20);
    }
}

#[test]
fn threshold_breaks_below_it() {
    assert_eq!(
        breakpoints(&[0.9, 0.2, 0.8, 0.1], Breakpoints::Threshold(0.5)),
        [2, 4]
    );
    assert!(breakpoints(&[0.9, 0.8], Breakpoints::Threshold(0.1)).is_empty());
}

#[test]
fn percentile_breaks_at_the_largest_distances() {
    // distances 0.1 0.8 0.2 0.9 0.05: the 60th percentile is 0.44
    let sims = [0.9, 0.2, 0.8, 0.1, 0.95];
    assert_eq!(breakpoints(&sims, Breakpoints::Percentile(60.0)), [2, 4]);
    assert_eq!(
        breakpoints(&sims, Breakpoints::Percentile(100.0)),
        Vec::<usize>::new()
    );
    assert_eq!(percentile(vec![3.0, 1.0, 2.0], 50.0), 2.0);
    assert_eq!(percentile(vec![1.0, 2.0], 25.0), 1.25);
}

#[test]
fn cosine_of_basic_vectors() {
    assert!((cosine(&[1.0, 0.0], &[1.0, 0.0]) - 1.0).abs() < 1e-6);
    assert!(cosine(&[1.0, 0.0], &[0.0, 1.0]).abs() < 1e-6);
    assert_eq!(cosine(&[0.0, 0.0], &[1.0, 0.0]), 0.0);
}

#[test]
fn groups_follow_the_vectors_and_keep_offsets() {
    let text = "Alpha one.\n \nBravo two. Charlie three.\n\t\nDelta four.";
    let job = chunker().semantic(text);
    assert_eq!(job.inputs().len(), 4);
    // first two alike, last two alike
    let v = [1.0, 0.0, 1.0, 0.1, 0.0, 1.0, 0.1, 1.0];
    let chunks = job
        .breakpoints(Breakpoints::Threshold(0.5))
        .finish(&v, 2)
        .unwrap();
    assert_eq!(chunks.len(), 2);
    assert_eq!(chunks[0].content, "Alpha one.\n \nBravo two. ");
    assert_eq!(chunks[1].content, "Charlie three.\n\t\nDelta four.");
    for c in &chunks {
        assert_eq!(&text[c.start_byte..c.end_byte], c.content);
    }
    assert_eq!((chunks[0].index, chunks[1].index), (0, 1));
}

#[test]
fn groups_hold_the_budget() {
    let text = "Same topic sentence here. ".repeat(80);
    let c = chunker().max_tokens(30);
    let job = c.semantic(&text);
    let v = vec![1.0f32; job.inputs().len() * 3];
    let chunks = job.finish(&v, 3).unwrap();
    assert!(chunks.len() > 1);
    for ch in &chunks {
        assert!(c.encoder().count(ch.content) <= 30);
    }
}

#[test]
fn wrong_vectors_are_an_error() {
    let job = chunker().semantic("One. Two. Three.");
    assert!(matches!(
        job.finish(&[1.0; 5], 2),
        Err(Error::Vectors {
            inputs: 3,
            values: 5,
            dim: 2
        })
    ));
    let job = chunker().semantic("One. Two.");
    assert!(matches!(job.finish(&[], 0), Err(Error::Vectors { .. })));
}

#[test]
fn one_unit_or_none_splits_recursively() {
    let job = chunker().semantic("just one sentence");
    let chunks = job.finish(&[0.5, 0.5], 2).unwrap();
    assert_eq!(chunks.len(), 1);
    assert_eq!(chunks[0].content, "just one sentence");
    assert!(chunker().semantic("").finish(&[], 0).unwrap().is_empty());
}
