//! Stage 3: Embedding via OpenAI API or OpenRouter (OpenAI-compatible).

use anyhow::{bail, Result};
use serde::{Deserialize, Serialize};

const BATCH_SIZE: usize = 50;
const MAX_CHARS: usize = 30_000;

// Model names differ by provider
const OPENAI_MODEL: &str = "text-embedding-3-large";
const OPENROUTER_MODEL: &str = "openai/text-embedding-3-large";

const OPENAI_BASE: &str = "https://api.openai.com/v1";
const OPENROUTER_BASE: &str = "https://openrouter.ai/api/v1";

#[derive(Serialize)]
struct EmbedRequest<'a> {
    model: &'a str,
    input: Vec<&'a str>,
}

#[derive(Deserialize)]
struct EmbedResponse {
    data: Vec<EmbedData>,
}

#[derive(Deserialize)]
struct EmbedData {
    embedding: Vec<f32>,
    index: usize,
}

pub async fn embed_texts(texts: &[String], api_key: &str, via_openrouter: bool) -> Result<Vec<Vec<f32>>> {
    if texts.is_empty() {
        return Ok(vec![]);
    }

    let (base_url, model) = if via_openrouter {
        (OPENROUTER_BASE, OPENROUTER_MODEL)
    } else {
        (OPENAI_BASE, OPENAI_MODEL)
    };

    let client = reqwest::Client::new();
    let mut all_embeddings: Vec<Option<Vec<f32>>> = vec![None; texts.len()];

    for batch_start in (0..texts.len()).step_by(BATCH_SIZE) {
        let batch_end = (batch_start + BATCH_SIZE).min(texts.len());
        let batch: Vec<&str> = texts[batch_start..batch_end]
            .iter()
            .map(|t| {
                let t = t.trim();
                if t.is_empty() { "empty" } else { &t[..t.len().min(MAX_CHARS)] }
            })
            .collect();

        let body = EmbedRequest { model, input: batch };

        let mut req = client
            .post(format!("{base_url}/embeddings"))
            .bearer_auth(api_key)
            .json(&body);

        // OpenRouter requires these headers to track usage
        if via_openrouter {
            req = req
                .header("HTTP-Referer", "https://legalkit.app")
                .header("X-Title", "LegalKit");
        }

        let resp = req.send().await?;

        if !resp.status().is_success() {
            let status = resp.status();
            let text = resp.text().await.unwrap_or_default();
            bail!("Embeddings API error {status}: {text}");
        }

        let parsed: EmbedResponse = resp.json().await?;
        for item in parsed.data {
            all_embeddings[batch_start + item.index] = Some(item.embedding);
        }
    }

    let dim = all_embeddings
        .iter()
        .find_map(|e| e.as_ref())
        .map(|e| e.len())
        .unwrap_or(3072);

    Ok(all_embeddings
        .into_iter()
        .map(|e| e.unwrap_or_else(|| vec![0.0f32; dim]))
        .collect())
}

pub async fn embed_query(query: &str, api_key: &str, via_openrouter: bool) -> Result<Vec<f32>> {
    let mut result = embed_texts(&[query.to_string()], api_key, via_openrouter).await?;
    result.pop().ok_or_else(|| anyhow::anyhow!("Empty embedding response"))
}
