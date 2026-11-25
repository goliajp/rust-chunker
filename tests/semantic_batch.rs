//! Semantic splitting of a document with more sentences than an embedding
//! provider accepts in one request.
//!
//! The endpoint is a local server speaking the OpenAI embeddings protocol and
//! enforcing OpenAI's limit of 2048 inputs per request.
#![cfg(feature = "semantic")]

use std::io::{BufRead, BufReader, Read, Write};
use std::net::{TcpListener, TcpStream};
use std::sync::{Arc, Mutex};

const PROVIDER_LIMIT: usize = 2048;

fn serve(stream: TcpStream, seen: Arc<Mutex<Vec<usize>>>) {
    let mut reader = BufReader::new(stream.try_clone().unwrap());
    let mut stream = stream;
    loop {
        let mut length = None;
        let mut line = String::new();
        loop {
            line.clear();
            if reader.read_line(&mut line).unwrap_or(0) == 0 {
                return;
            }
            if line == "\r\n" {
                break;
            }
            let lower = line.to_ascii_lowercase();
            if let Some(v) = lower.strip_prefix("content-length:") {
                length = v.trim().parse::<usize>().ok();
            }
        }
        let mut body = vec![0; length.unwrap_or(0)];
        reader.read_exact(&mut body).unwrap();
        let body = String::from_utf8(body).unwrap();

        // inputs carry no quotes or brackets, so counting separators is exact
        let start = body.find("\"input\":[").unwrap();
        let end = start + body[start..].find(']').unwrap();
        let inputs = body[start..end].matches("\",\"").count() + 1;
        seen.lock().unwrap().push(inputs);

        let (status, json) = if inputs > PROVIDER_LIMIT {
            (
                "400 Bad Request",
                "{\"error\":\"too many inputs\"}".to_string(),
            )
        } else {
            let data = vec!["{\"embedding\":[1.0,0.0]}"; inputs].join(",");
            (
                "200 OK",
                format!("{{\"data\":[{data}],\"model\":\"m\",\"usage\":{{\"total_tokens\":1}}}}"),
            )
        };
        let reply = format!(
            "HTTP/1.1 {status}\r\ncontent-type: application/json\r\ncontent-length: {}\r\n\r\n{json}",
            json.len()
        );
        if stream.write_all(reply.as_bytes()).is_err() {
            return;
        }
    }
}

#[tokio::test]
async fn long_documents_are_embedded_in_provider_sized_batches() {
    let listener = TcpListener::bind("127.0.0.1:0").unwrap();
    let addr = listener.local_addr().unwrap();
    let seen = Arc::new(Mutex::new(Vec::new()));
    let log = Arc::clone(&seen);
    std::thread::spawn(move || {
        for stream in listener.incoming().flatten() {
            let log = Arc::clone(&log);
            std::thread::spawn(move || serve(stream, log));
        }
    });

    let sentences = PROVIDER_LIMIT + 300;
    let text: String = (0..sentences).map(|i| format!("Line {i}. ")).collect();
    let client = embedrs::Client::openai_compatible("test", format!("http://{addr}/v1"));

    let chunks = chunkedrs::chunk(&text)
        .semantic(&client)
        .max_tokens(256)
        .split_async()
        .await
        .expect("every request stays within the provider limit");

    let seen = seen.lock().unwrap().clone();
    assert!(seen.len() >= 2, "requests: {seen:?}");
    assert!(
        seen.iter().all(|&n| n <= PROVIDER_LIMIT),
        "requests: {seen:?}"
    );
    assert_eq!(seen.iter().sum::<usize>(), sentences);
    assert!(chunks.iter().all(|c| c.token_count <= 256));
    for c in &chunks {
        assert_eq!(&text[c.start_byte..c.end_byte], c.content);
    }
}
