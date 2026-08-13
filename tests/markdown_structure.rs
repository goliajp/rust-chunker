//! Structural tests for markdown-aware splitting.
//!
//! `is_header` was a bare per-line `#` check with no notion of document
//! structure, so anything that *looks* like a header out of context was treated
//! as one — most damagingly the comment lines inside fenced code blocks, which
//! appear in essentially every technical document.

fn sections(chunks: &[chunkedrs::Chunk]) -> Vec<Option<&str>> {
    chunks.iter().map(|c| c.section.as_deref()).collect()
}

/// A `#` comment inside a fenced code block is a comment, not a header.
#[test]
fn hash_comment_inside_code_fence_is_not_a_header() {
    let md = "\
# Real Header

Intro text.

```python
# this is a comment, not a header
print('hi')
```

After the fence.
";
    let chunks = chunkedrs::chunk(md).markdown().split();

    assert!(
        !sections(&chunks).contains(&Some("# this is a comment, not a header")),
        "a comment inside a code fence was promoted to a section header: {:?}",
        sections(&chunks)
    );
    assert!(
        sections(&chunks)
            .iter()
            .all(|s| *s == Some("# Real Header")),
        "everything here belongs to the one real header: {:?}",
        sections(&chunks)
    );
}

/// Shell scripts are the worst case — nearly every line starts with `#`.
#[test]
fn shell_fence_comments_are_not_headers() {
    let md = "\
## Installation

```bash
# install the tool
cargo install chunkedrs
## not a header either
```

Done.
";
    let chunks = chunkedrs::chunk(md).markdown().split();
    assert!(
        sections(&chunks)
            .iter()
            .all(|s| *s == Some("## Installation")),
        "shell comments leaked into section metadata: {:?}",
        sections(&chunks)
    );
}

/// Tilde fences are equally valid CommonMark.
#[test]
fn tilde_fences_are_respected() {
    let md = "\
# Title

~~~
# not a header
~~~

Body.
";
    let chunks = chunkedrs::chunk(md).markdown().split();
    assert!(
        sections(&chunks).iter().all(|s| *s == Some("# Title")),
        "tilde fence not honoured: {:?}",
        sections(&chunks)
    );
}

/// A fence longer than three backticks opens a block that shorter runs cannot
/// close — this is how documents embed markdown inside markdown.
#[test]
fn longer_fences_are_not_closed_by_shorter_runs() {
    let md = "\
# Outer

````markdown
```
# still inside the outer fence
```
````

Tail.
";
    let chunks = chunkedrs::chunk(md).markdown().split();
    assert!(
        sections(&chunks).iter().all(|s| *s == Some("# Outer")),
        "a shorter run closed a longer fence: {:?}",
        sections(&chunks)
    );
}

/// Fences may be indented up to three spaces and still be fences.
#[test]
fn indented_fences_are_respected() {
    let md = "\
# Title

   ```
   # not a header
   ```

Body.
";
    let chunks = chunkedrs::chunk(md).markdown().split();
    assert!(
        sections(&chunks).iter().all(|s| *s == Some("# Title")),
        "indented fence not honoured: {:?}",
        sections(&chunks)
    );
}

/// An unterminated fence must not panic, and must not resume header detection.
#[test]
fn unclosed_fence_does_not_resume_header_detection() {
    let md = "\
# Title

```
# never closed
";
    let chunks = chunkedrs::chunk(md).markdown().split();
    assert!(
        sections(&chunks).iter().all(|s| *s == Some("# Title")),
        "header detection resumed inside an unclosed fence: {:?}",
        sections(&chunks)
    );
}

/// Real headers after a fence still work — the state machine must actually
/// close.
#[test]
fn headers_after_a_closed_fence_still_register() {
    let md = "\
# First

```
# fake
```

## Second

Content under second.
";
    let chunks = chunkedrs::chunk(md).markdown().split();
    let s = sections(&chunks);
    assert!(
        s.contains(&Some("## Second")),
        "the real header after the fence was missed: {s:?}"
    );
    assert!(
        !s.contains(&Some("# fake")),
        "the fenced line was still treated as a header: {s:?}"
    );
}

/// Setext headers are CommonMark and appear throughout older documents.
#[test]
fn setext_headers_are_recognised() {
    let md = "\
Document Title
==============

Some intro.

A Subsection
------------

More text.
";
    let chunks = chunkedrs::chunk(md).markdown().split();
    let s = sections(&chunks);
    assert!(
        s.contains(&Some("Document Title")),
        "setext H1 not recognised: {s:?}"
    );
    assert!(
        s.contains(&Some("A Subsection")),
        "setext H2 not recognised: {s:?}"
    );
}

/// YAML front matter must not be mistaken for a setext header — its closing
/// `---` follows an ordinary `key: value` line.
#[test]
fn yaml_front_matter_is_not_a_setext_header() {
    let md = "\
---
title: My Document
author: Someone
---

# Real Header

Body text.
";
    let chunks = chunkedrs::chunk(md).markdown().split();
    let s = sections(&chunks);
    assert!(
        !s.contains(&Some("author: Someone")),
        "front matter closing delimiter was read as a setext header: {s:?}"
    );
    assert!(
        s.contains(&Some("# Real Header")),
        "the real header after front matter was missed: {s:?}"
    );
}

/// Byte offsets must still address the original document after all of this.
#[test]
fn markdown_byte_offsets_address_original_text() {
    let md = "\
# Title

Body one.

```rust
# not a header
let x = 1;
```

## Second

Body two.
";
    for c in chunkedrs::chunk(md).markdown().split() {
        assert_eq!(
            &md[c.start_byte..c.end_byte],
            c.content,
            "chunk {} byte range does not match its content",
            c.index
        );
    }
}
