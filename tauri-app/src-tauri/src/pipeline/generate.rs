//! Stage 5: LLM generation + citation extraction.
//!
//! Supports two backends:
//!   - Anthropic native API (ANTHROPIC_API_KEY)
//!   - OpenRouter OpenAI-compatible API (OPENROUTER_API_KEY)

use crate::models::{AnswerReport, Citation, RetrievalResult};
use anyhow::{bail, Result};
use serde::{Deserialize, Serialize};
use std::time::Instant;

const MAX_TOKENS: u32 = 4096;

// ---------------------------------------------------------------------------
// Public entry point
// ---------------------------------------------------------------------------

pub async fn generate_answer(
    question: &str,
    retrieval: &RetrievalResult,
    doc_id: &str,
    api_key: &str,
    via_openrouter: bool,
    model: &str,
) -> Result<AnswerReport> {
    let t0 = Instant::now();
    let context = build_context(retrieval);

    let system = "You are a precise legal document analyst. \
        Answer the user's question based only on the provided contract excerpts. \
        Be specific and cite the relevant sections. \
        If the contract does not address the question, say so clearly.";

    let user = format!(
        "CONTRACT EXCERPTS:\n{context}\n\n\
        QUESTION: {question}\n\n\
        Provide a clear, direct answer. After your answer, list the key sections you relied on \
        in this format:\nSOURCES:\n- [Section ID]: brief quote"
    );

    let raw = if via_openrouter {
        call_openrouter(system, &user, api_key, model).await?
    } else {
        call_anthropic(system, &user, api_key, model).await?
    };

    let (answer, citations) = parse_response(&raw, retrieval);

    Ok(AnswerReport {
        question: question.to_string(),
        answer,
        citations,
        doc_id: doc_id.to_string(),
        elapsed_seconds: t0.elapsed().as_secs_f64(),
    })
}

// ---------------------------------------------------------------------------
// Anthropic native API
// ---------------------------------------------------------------------------

#[derive(Serialize)]
struct AnthropicRequest<'a> {
    model: &'a str,
    max_tokens: u32,
    system: &'a str,
    messages: Vec<AnthropicMessage<'a>>,
}

#[derive(Serialize)]
struct AnthropicMessage<'a> {
    role: &'a str,
    content: &'a str,
}

#[derive(Deserialize)]
struct AnthropicResponse {
    content: Vec<AnthropicContent>,
}

#[derive(Deserialize)]
struct AnthropicContent {
    text: String,
}

async fn call_anthropic(system: &str, user: &str, api_key: &str, model: &str) -> Result<String> {
    let client = reqwest::Client::new();
    let body = AnthropicRequest {
        model,
        max_tokens: MAX_TOKENS,
        system,
        messages: vec![AnthropicMessage { role: "user", content: user }],
    };

    let resp = client
        .post("https://api.anthropic.com/v1/messages")
        .header("x-api-key", api_key)
        .header("anthropic-version", "2023-06-01")
        .json(&body)
        .send()
        .await?;

    if !resp.status().is_success() {
        let status = resp.status();
        let text = resp.text().await.unwrap_or_default();
        bail!("Anthropic API error {status}: {text}");
    }

    let parsed: AnthropicResponse = resp.json().await?;
    Ok(parsed.content.into_iter().next().map(|c| c.text).unwrap_or_default())
}

// ---------------------------------------------------------------------------
// OpenRouter (OpenAI-compatible chat completions)
// ---------------------------------------------------------------------------

#[derive(Serialize)]
struct OaiRequest<'a> {
    model: &'a str,
    max_tokens: u32,
    messages: Vec<OaiMessage<'a>>,
}

#[derive(Serialize)]
struct OaiMessage<'a> {
    role: &'a str,
    content: &'a str,
}

#[derive(Deserialize)]
struct OaiResponse {
    choices: Vec<OaiChoice>,
}

#[derive(Deserialize)]
struct OaiChoice {
    message: OaiResponseMessage,
}

#[derive(Deserialize)]
struct OaiResponseMessage {
    content: String,
}

async fn call_openrouter(system: &str, user: &str, api_key: &str, model: &str) -> Result<String> {
    let client = reqwest::Client::new();
    let body = OaiRequest {
        model,
        max_tokens: MAX_TOKENS,
        messages: vec![
            OaiMessage { role: "system", content: system },
            OaiMessage { role: "user", content: user },
        ],
    };

    let resp = client
        .post("https://openrouter.ai/api/v1/chat/completions")
        .bearer_auth(api_key)
        .header("HTTP-Referer", "https://legalkit.app")
        .header("X-Title", "LegalKit")
        .json(&body)
        .send()
        .await?;

    if !resp.status().is_success() {
        let status = resp.status();
        let text = resp.text().await.unwrap_or_default();
        bail!("OpenRouter API error {status}: {text}");
    }

    let parsed: OaiResponse = resp.json().await?;
    Ok(parsed
        .choices
        .into_iter()
        .next()
        .map(|c| c.message.content)
        .unwrap_or_default())
}

// ---------------------------------------------------------------------------
// Shared helpers
// ---------------------------------------------------------------------------

/// Clean up stray markdown formatting from LLM output.
fn sanitize_markdown(text: &str) -> String {
    let mut s = text.to_string();

    // Remove trailing unclosed bold/italic markers
    // e.g. "some text **" → "some text"
    for suffix in &["**", "__", "*", "_", "```"] {
        let trimmed = s.trim_end();
        if trimmed.ends_with(suffix) {
            let count = s.matches(suffix).count();
            if count % 2 != 0 {
                s = trimmed.trim_end_matches(suffix).trim_end().to_string();
            }
        }
    }

    // Remove leading "## " or "# " if the entire response is one heading
    // (some models wrap their whole answer in a heading)
    let lines: Vec<&str> = s.lines().collect();
    if lines.len() <= 2 {
        if let Some(first) = lines.first() {
            if first.starts_with("## ") || first.starts_with("# ") {
                // Convert to bold instead of heading for single-line answers
                let heading_text = first.trim_start_matches('#').trim();
                if lines.len() == 1 {
                    s = format!("**{heading_text}**");
                }
            }
        }
    }

    s
}

fn build_context(retrieval: &RetrievalResult) -> String {
    let all_chunks = retrieval
        .ranked_chunks
        .iter()
        .chain(retrieval.cross_ref_chunks.iter())
        .chain(retrieval.definition_chunks.iter());

    let mut parts: Vec<String> = Vec::new();
    let mut seen: std::collections::HashSet<String> = Default::default();

    for rc in all_chunks {
        let chunk = &rc.chunk;
        if !seen.insert(chunk.node_id.clone()) {
            continue;
        }
        let path = chunk.node_path.join(" > ");
        parts.push(format!("[{path}]\n{}", chunk.text));
    }

    parts.join("\n\n---\n\n")
}

fn parse_response(raw: &str, retrieval: &RetrievalResult) -> (String, Vec<Citation>) {
    let (answer_raw, sources_part) = if let Some(idx) = raw.find("SOURCES:") {
        (raw[..idx].trim().to_string(), Some(&raw[idx + 8..]))
    } else {
        (raw.trim().to_string(), None)
    };

    // Clean up common markdown artifacts from LLM output
    let answer_part = sanitize_markdown(&answer_raw);

    let mut citations: Vec<Citation> = Vec::new();

    if let Some(sources) = sources_part {
        for line in sources.lines() {
            let line = line.trim().trim_start_matches('-').trim();
            if line.is_empty() || !line.starts_with('[') {
                continue;
            }
            if let Some(close) = line.find(']') {
                let node_id = line[1..close].trim().to_string();
                let quote = line[close + 1..].trim_start_matches(':').trim().to_string();

                let chunk_opt = retrieval
                    .ranked_chunks
                    .iter()
                    .chain(retrieval.cross_ref_chunks.iter())
                    .find(|rc| rc.chunk.node_id == node_id)
                    .map(|rc| &rc.chunk);

                citations.push(Citation {
                    node_id: node_id.clone(),
                    node_path: chunk_opt
                        .map(|c| c.node_path.clone())
                        .unwrap_or_else(|| vec![node_id]),
                    quote: quote.chars().take(300).collect(),
                    char_start: chunk_opt.map(|c| c.char_start).unwrap_or(0),
                    char_end: chunk_opt.map(|c| c.char_end).unwrap_or(0),
                    confidence: 0.9,
                });
            }
        }
    }

    (answer_part, citations)
}
