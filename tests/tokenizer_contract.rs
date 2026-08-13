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
/// present — it is the fallback encoder chunkedrs resolves to.
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
/// `.model()` falls back to `o200k_base` silently on an unknown name, so a
/// missing mapping is invisible at runtime — it just quietly produces the
/// wrong token counts. These are the families the README names.
#[test]
fn advertised_models_resolve() {
    for model in [
        "gpt-4o",
        "gpt-5",
        "o3",
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
            "model '{model}' must map to an encoding, not fall back silently"
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
        tiktoken::list_encodings().len() >= 17,
        "expected the 2026-08 encoding catalogue (17+), got {}",
        tiktoken::list_encodings().len()
    );
}

/// `o200k_base` is the encoder every unresolved name falls back to, so it must
/// be present in *every* build regardless of which vocab features are on.
/// The dependency declaration enables it unconditionally; this pins that.
#[test]
fn fallback_encoding_is_always_available() {
    assert!(
        tiktoken::get_encoding("o200k_base").is_some(),
        "o200k_base must survive `default-features = false`"
    );
    assert!(chunkedrs::chunk("hello world").split()[0].token_count > 0);
}
