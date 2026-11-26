//! Contract tests for the tokenizer chunkedrs is built on.
//!
//! chunkedrs's headline guarantee is token accuracy, and its highest-priority
//! separator is `"\n\n"` — so canonical newline tokenization is not an
//! implementation detail of the dependency, it is load-bearing for this crate.
//! tiktoken 3.6.0 fixed a pre-tokenizer bug that split canonical newline runs
//! into single-newline tokens, overcounting every paragraph break by one token.
//! These tests pin that behaviour so a dependency downgrade cannot reintroduce it.

/// Vocabularies are features as of tiktoken 4, so only the ones this build
/// actually compiled in can be asserted against. `o200k_base` is always
/// present: it is the encoding `Chunker::new` uses.
const ENCODINGS_UNDER_TEST: &[&str] = &[
    "o200k_base",
    #[cfg(feature = "vocab-cl100k_base")]
    "cl100k_base",
];

/// `"\n\n"` must merge into a single token, not two `"\n"` tokens.
///
/// Before tiktoken 3.6.0 this counted 4 on both cl100k_base and o200k_base.
#[test]
fn canonical_double_newline_is_one_token() {
    for name in ENCODINGS_UNDER_TEST {
        let enc = tiktoken::get_encoding(name).expect(name);
        assert_eq!(
            enc.count("word\n\nnext"),
            3,
            "{name}: 'word\\n\\nnext' must be 3 tokens (word, \\n\\n, next), got ids {:?}",
            enc.encode("word\n\nnext")
        );
    }
}

/// A CRLF pair must not split into separate `\r` and `\n` tokens.
#[test]
fn canonical_crlf_is_not_split() {
    for name in ENCODINGS_UNDER_TEST {
        let enc = tiktoken::get_encoding(name).expect(name);
        let ids = enc.encode("a\r\nb");
        assert!(
            ids.len() <= 3,
            "{name}: 'a\\r\\nb' should not split CRLF into two tokens, got ids {ids:?}"
        );
    }
}

/// Paragraph-separated prose must not be overcounted.
///
/// This is the shape chunkedrs actually feeds the tokenizer: its top-priority
/// separator splits on `"\n\n"`, so every paragraph boundary in every document
/// hits this path.
#[test]
fn paragraph_breaks_are_not_overcounted() {
    let enc = tiktoken::get_encoding("o200k_base").unwrap();
    let joined = "alpha\n\nbravo\n\ncharlie";
    let parts_sum: usize = ["alpha", "bravo", "charlie"]
        .iter()
        .map(|p| enc.count(p))
        .sum();
    // three words + two paragraph-break tokens
    assert_eq!(
        enc.count(joined),
        parts_sum + 2,
        "each paragraph break must cost exactly one token, got ids {:?}",
        enc.encode(joined)
    );
}

/// The encodings chunkedrs advertises must actually resolve.
///
/// These are the families the README names. A model tiktoken stops mapping
/// would make `Chunker::for_model` return an error for it.
#[test]
fn advertised_models_resolve() {
    for model in [
        "gpt-4o",
        "gpt-5",
        // the 2026 GPT-5.x SKUs resolve through the "gpt-5" prefix rather than
        // an entry of their own. That is by design, but it is exactly the kind
        // of thing that silently stops being true, and the failure mode here
        // is wrong token counts, not an error.
        "gpt-5.6-sol",
        "gpt-5.6-terra",
        "gpt-5.6-luna",
        "gpt-5.5",
        "gpt-5.4",
        "o3",
        "o4-mini",
        "llama-3.1-70b",
        "qwen2.5-72b",
        "deepseek-v4",
        "kimi-k2",
        "glm-5",
        "minimax-m2",
        "mistral-large",
    ] {
        assert!(
            tiktoken::model_to_encoding(model).is_some(),
            "model '{model}' must map to an encoding"
        );
        // a slim build leaves most vocabularies out, and then it is an error
        let built = tiktoken::encoding_for_model(model).is_some();
        assert_eq!(
            chunkedrs::Chunker::for_model(model).is_ok(),
            built,
            "{model}"
        );
    }
}

/// Models that tiktoken 5.0 moved to vocabularies of their own. Chunk
/// boundaries for these changed with the 5.x upgrade, so the mapping is pinned.
#[test]
fn models_remapped_in_tiktoken_5() {
    for (model, encoding) in [
        ("llama-4-maverick", "llama4"),
        ("qwen3.5-plus", "qwen3_5"),
        ("mistral-large-latest", "mistral_v15"),
        ("deepseek-flash", "deepseek_v4"),
        ("gpt-6", "o200k_base"),
    ] {
        assert_eq!(
            tiktoken::model_to_encoding(model),
            Some(encoding),
            "model '{model}'"
        );
    }
}

/// Encodings the crate exposes by name must all be reachable.
///
/// As of tiktoken 4 `list_encodings()` reports only the vocabularies compiled
/// in, so this is a contract about *this build*, not the catalogue.
#[test]
fn advertised_encodings_resolve() {
    for name in tiktoken::list_encodings() {
        assert!(
            tiktoken::get_encoding(name).is_some(),
            "listed encoding '{name}' must be constructible"
        );
    }
}

/// The full catalogue, when this build asked for all of it.
#[test]
#[cfg(feature = "vocabs-all")]
fn full_catalogue_is_present() {
    assert!(
        tiktoken::list_encodings().len() >= 20,
        "expected the tiktoken 5 encoding catalogue (20+), got {:?}",
        tiktoken::list_encodings()
    );
    for name in ["llama4", "qwen3_5", "mistral_v15"] {
        assert!(
            tiktoken::get_encoding(name).is_some(),
            "{name} must be compiled in"
        );
    }
}

/// `o200k_base` is the encoding `Chunker::new` uses, so it must be present in
/// *every* build regardless of which vocab features are on.
/// The dependency declaration enables it unconditionally; this pins that.
#[test]
fn default_encoding_is_always_available() {
    assert!(
        tiktoken::get_encoding("o200k_base").is_some(),
        "o200k_base must survive `default-features = false`"
    );
    assert!(chunkedrs::Chunker::new().split("hello world")[0].token_count > 0);
}
