//! Where each strategy may cut: separator tiers, or cut positions found by
//! scanning the text. All of them plan through `crate::plan`.

pub(crate) mod code;
pub(crate) mod html;
pub(crate) mod markdown;
pub(crate) mod recursive;

/// Which strategy a [`Chunker`](crate::Chunker) splits with.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum Strategy {
    Recursive,
    Markdown,
    Code,
    Html,
}
