//! Semantic splitting end to end, with no API key: a request built by embedrs
//! and answered with a fixed body, and a real model run on the CPU.
#![cfg(feature = "semantic")]

use chunkedrs::{Breakpoints, Chunker};
use embedrs::transport::HttpResponse;

const COOKING: &[&str] = &[
    "Preheat the oven to two hundred degrees before you start the dough.",
    "Knead the bread dough for ten minutes until it is smooth and elastic.",
    "Let the dough rise in a warm kitchen until it doubles in size.",
    "Bake the loaf until the crust is golden brown and sounds hollow.",
    "Brush the warm bread with melted butter and a pinch of salt.",
    "Slice the bread only after it has cooled on a wire rack.",
];

const ORBITS: &[&str] = &[
    "A satellite in low Earth orbit circles the planet every ninety minutes.",
    "Orbital velocity depends on the altitude of the spacecraft above Earth.",
    "Geostationary satellites orbit at thirty six thousand kilometres up.",
    "Kepler's laws describe the elliptical orbits of planets around the Sun.",
    "A rocket burn at perigee raises the apogee of the spacecraft's orbit.",
    "Space debris in orbit threatens satellites and the space station.",
];

const TAXES: &[&str] = &[
    "File your income tax return before the April deadline to avoid penalties.",
    "Deductible business expenses reduce the taxable income you report.",
    "Keep receipts for charitable donations you claim as tax deductions.",
    "Capital gains on investments are taxed when you sell the shares.",
    "The tax authority may audit returns that report unusual deductions.",
    "Estimated tax payments are due each quarter for self employed workers.",
];

fn document() -> (String, Vec<usize>) {
    let mut text = String::new();
    let mut topic_starts = Vec::new();
    for topic in [COOKING, ORBITS, TAXES] {
        topic_starts.push(text.len());
        for s in topic {
            text.push_str(s);
            text.push(' ');
        }
    }
    (text, topic_starts)
}

/// The OpenAI wire format answered by hand: the request goes through
/// embedrs's own builder and the body through its own parser.
#[test]
fn prepared_request_and_a_fixed_response() {
    let text = "Cats purr. Cats nap. Stocks fell. Markets slid.";
    let job = Chunker::new()
        .semantic(text)
        .breakpoints(Breakpoints::Threshold(0.5));
    let client = embedrs::Client::openai("no-key-needed");
    let prepared = client.prepare(job.inputs()).unwrap();
    assert!(prepared.request().url.ends_with("/embeddings"));

    let vectors = [[1.0, 0.0], [0.9, 0.1], [0.0, 1.0], [0.1, 0.9]];
    let data: Vec<String> = vectors
        .iter()
        .enumerate()
        .map(|(i, v)| format!(r#"{{"index":{i},"embedding":[{},{}]}}"#, v[0], v[1]))
        .collect();
    let body = format!(r#"{{"data":[{}]}}"#, data.join(","));
    let result = prepared
        .finish(HttpResponse::new(200, vec![], body.into_bytes()))
        .unwrap();

    let chunks = job.finish(result.as_flat(), result.dim()).unwrap();
    let texts: Vec<&str> = chunks.iter().map(|c| c.content).collect();
    assert_eq!(
        texts,
        ["Cats purr. Cats nap. ", "Stocks fell. Markets slid."]
    );
}

/// all-MiniLM-L6-v2 on the CPU, downloaded once into the Hugging Face cache.
/// Three topics of six sentences: the two largest neighbour distances are
/// the topic changes, so the 90th percentile breaks there and nowhere else.
#[tokio::test]
async fn local_model_breaks_at_topic_changes() {
    let (text, starts) = document();
    let client = embedrs::local();
    let chunker = Chunker::new();

    let chunks = chunker
        .semantic(&text)
        .breakpoints(Breakpoints::Percentile(90.0))
        .embed(&client)
        .await
        .expect("the local model runs; its weights download on first use");
    let found: Vec<usize> = chunks.iter().map(|c| c.start_byte).collect();
    assert_eq!(found, starts, "chunks: {chunks:#?}");
    for c in &chunks {
        assert_eq!(&text[c.start_byte..c.end_byte], c.content);
    }

    // the batch form agrees with one call per text
    let docs = [text.as_str(), "One short line."];
    let batch = chunker.split_semantic_batch(&docs, &client).await.unwrap();
    for (doc, got) in docs.iter().zip(&batch) {
        assert_eq!(got, &chunker.split_semantic(doc, &client).await.unwrap());
    }
}
