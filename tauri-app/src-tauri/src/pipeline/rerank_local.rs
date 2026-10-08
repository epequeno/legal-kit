//! Local cross-encoder reranker via Candle (XLM-RoBERTa-large).
//!
//! Loads the fine-tuned legal-reranker model and scores (query, passage) pairs,
//! returning a relevance score in [0, 1] (sigmoid of the single logit).
//!
//! All passages for a query are batched into a single forward pass to minimise
//! inference latency. Apple Accelerate BLAS (via the `accelerate` candle feature)
//! is used for fast CPU matrix operations.

use anyhow::{bail, Result};
use candle_core::{DType, Device, Tensor};
use candle_nn::VarBuilder;
use candle_transformers::models::xlm_roberta::{
    Config as XLMRobertaConfig, XLMRobertaForSequenceClassification,
};
use std::path::Path;
use std::sync::Mutex;
use tokenizers::Tokenizer;

static MODEL: Mutex<Option<RerankerModel>> = Mutex::new(None);

struct RerankerModel {
    model: XLMRobertaForSequenceClassification,
    tokenizer: Tokenizer,
    device: Device,
}

pub fn ensure_loaded(model_dir: &Path) -> Result<()> {
    let mut guard = MODEL.lock().map_err(|e| anyhow::anyhow!("Lock error: {e}"))?;
    if guard.is_some() {
        return Ok(());
    }

    // XLM-RoBERTa layer-norm is not implemented in Candle's Metal backend.
    // CPU + Apple Accelerate BLAS is fast enough for batched reranking.
    let device = Device::Cpu;
    println!("[rerank_local] Loading model from {}", model_dir.display());

    // Tokenizer
    let tokenizer_path = model_dir.join("tokenizer.json");
    if !tokenizer_path.exists() {
        bail!("tokenizer.json not found in {}", model_dir.display());
    }
    let tokenizer = Tokenizer::from_file(&tokenizer_path)
        .map_err(|e| anyhow::anyhow!("Failed to load tokenizer: {e}"))?;

    // Config
    let config_str = std::fs::read_to_string(model_dir.join("config.json"))?;
    let config: XLMRobertaConfig = serde_json::from_str(&config_str)?;

    // Weights — remap gamma→weight, beta→bias
    let weights_path = model_dir.join("model.safetensors");
    if !weights_path.exists() {
        bail!("model.safetensors not found in {}", model_dir.display());
    }
    let raw = candle_core::safetensors::load(&weights_path, &device)?;
    let mut tensors = std::collections::HashMap::new();
    for (name, tensor) in raw {
        let new_name = name
            .replace("LayerNorm.gamma", "LayerNorm.weight")
            .replace("LayerNorm.beta", "LayerNorm.bias");
        tensors.insert(new_name, tensor);
    }

    let vb = VarBuilder::from_tensors(tensors, DType::F32, &device);
    let model = XLMRobertaForSequenceClassification::new(1, &config, vb)?;

    println!("[rerank_local] Model loaded (XLM-RoBERTa-large, {} layers, CPU+Accelerate)", config.num_hidden_layers);

    *guard = Some(RerankerModel { model, tokenizer, device });
    Ok(())
}

pub fn is_loaded() -> bool {
    MODEL.lock().map(|g| g.is_some()).unwrap_or(false)
}

/// Rerank passages for a query — all passages in a single batched forward pass.
///
/// Returns the original indices sorted by descending relevance score, paired
/// with their scores (sigmoid of the cross-encoder logit).
pub fn rerank(query: &str, passages: &[(String, usize)]) -> Result<Vec<(usize, f32)>> {
    if passages.is_empty() {
        return Ok(vec![]);
    }

    let guard = MODEL.lock().map_err(|e| anyhow::anyhow!("Lock error: {e}"))?;
    let rm = guard
        .as_ref()
        .ok_or_else(|| anyhow::anyhow!("Reranker model not loaded"))?;

    let max_len = 512usize;
    let batch_size = passages.len();

    // Tokenize all (query, passage) pairs
    let pairs: Vec<(&str, &str)> = passages
        .iter()
        .map(|(text, _)| (query, text.as_str()))
        .collect();

    let encodings = rm
        .tokenizer
        .encode_batch(pairs, true)
        .map_err(|e| anyhow::anyhow!("Tokenization error: {e}"))?;

    // Find the max length (capped at max_len) for padding
    let padded_len = encodings
        .iter()
        .map(|e| e.get_ids().len().min(max_len))
        .max()
        .unwrap_or(1);

    let pad_id = 1u32; // XLM-RoBERTa pad token

    let mut input_ids_flat = Vec::with_capacity(batch_size * padded_len);
    let mut token_type_ids_flat = Vec::with_capacity(batch_size * padded_len);
    let mut attention_mask_flat = Vec::with_capacity(batch_size * padded_len);

    for enc in &encodings {
        let ids = enc.get_ids();
        let type_ids = enc.get_type_ids();
        let seq_len = ids.len().min(max_len);

        // Actual tokens
        input_ids_flat.extend_from_slice(&ids[..seq_len]);
        token_type_ids_flat.extend_from_slice(&type_ids[..seq_len]);
        attention_mask_flat.extend(vec![1u32; seq_len]);

        // Padding
        let pad_count = padded_len - seq_len;
        input_ids_flat.extend(vec![pad_id; pad_count]);
        token_type_ids_flat.extend(vec![0u32; pad_count]);
        attention_mask_flat.extend(vec![0u32; pad_count]);
    }

    let input_ids = Tensor::from_vec(input_ids_flat, (batch_size, padded_len), &rm.device)?;
    let token_type_ids = Tensor::from_vec(token_type_ids_flat, (batch_size, padded_len), &rm.device)?;
    let attention_mask = Tensor::from_vec(attention_mask_flat, (batch_size, padded_len), &rm.device)?;

    // Single batched forward pass → [batch_size, 1]
    let logits = rm.model.forward(&input_ids, &attention_mask, &token_type_ids)?;
    let logits_vec: Vec<f32> = logits.squeeze(1)?.to_vec1()?;

    // Sigmoid → relevance scores, zip with original indices
    let mut scored: Vec<(usize, f32)> = logits_vec
        .into_iter()
        .zip(passages.iter().map(|(_, orig_idx)| *orig_idx))
        .map(|(logit, orig_idx)| (orig_idx, sigmoid(logit)))
        .collect();

    scored.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
    Ok(scored)
}

/// Score a single (query, passage) pair → relevance in [0, 1].
/// Convenience wrapper around `rerank` for single pairs.
pub fn score_pair(query: &str, passage: &str) -> Result<f32> {
    let result = rerank(query, &[(passage.to_string(), 0)])?;
    Ok(result.first().map(|(_, s)| *s).unwrap_or(0.0))
}

#[inline]
fn sigmoid(x: f32) -> f32 {
    1.0 / (1.0 + (-x).exp())
}
