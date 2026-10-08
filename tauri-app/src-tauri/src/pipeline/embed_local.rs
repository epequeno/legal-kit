//! Local embedding inference via Candle (BERT / BGE models).
//!
//! Loads a sentence-transformers model from disk and runs inference
//! on Apple Silicon via Metal, or CPU fallback.

use anyhow::{bail, Result};
use candle_core::{DType, Device, Tensor};
use candle_nn::VarBuilder;
use candle_transformers::models::bert::{BertModel, Config as BertConfig};
use std::path::Path;
use std::sync::Mutex;
use tokenizers::Tokenizer;

/// Cached model + tokenizer for reuse across calls.
static MODEL: Mutex<Option<EmbeddingModel>> = Mutex::new(None);

struct EmbeddingModel {
    model: BertModel,
    tokenizer: Tokenizer,
    device: Device,
    dim: usize,
}

/// Load the model from disk (idempotent — only loads once).
pub fn ensure_loaded(model_dir: &Path) -> Result<()> {
    let mut guard = MODEL.lock().map_err(|e| anyhow::anyhow!("Lock error: {e}"))?;
    if guard.is_some() {
        return Ok(());
    }

    let device = get_device();
    println!("[embed_local] Loading model from {}", model_dir.display());
    println!("[embed_local] Device: {:?}", device);

    // Load tokenizer
    let tokenizer_path = model_dir.join("tokenizer.json");
    if !tokenizer_path.exists() {
        bail!("tokenizer.json not found in {}", model_dir.display());
    }
    let tokenizer = Tokenizer::from_file(&tokenizer_path)
        .map_err(|e| anyhow::anyhow!("Failed to load tokenizer: {e}"))?;

    // Load model config
    let config_path = model_dir.join("config.json");
    if !config_path.exists() {
        bail!("config.json not found in {}", model_dir.display());
    }
    let config_str = std::fs::read_to_string(&config_path)?;
    let config: BertConfig = serde_json::from_str(&config_str)?;
    let dim = config.hidden_size;

    // Load weights — remap old-style LayerNorm names (gamma/beta → weight/bias)
    let weights_path = model_dir.join("model.safetensors");
    if !weights_path.exists() {
        bail!("model.safetensors not found in {}", model_dir.display());
    }

    let vb = {
        let tensors = candle_core::safetensors::load(&weights_path, &device)?;

        // Remap tensor names: gamma→weight, beta→bias
        let remapped: std::collections::HashMap<String, Tensor> = tensors
            .into_iter()
            .map(|(name, tensor)| {
                let new_name = name
                    .replace("LayerNorm.gamma", "LayerNorm.weight")
                    .replace("LayerNorm.beta", "LayerNorm.bias");
                (new_name, tensor)
            })
            .collect();

        VarBuilder::from_tensors(remapped, DType::F32, &device)
    };
    let model = BertModel::load(vb, &config)?;

    println!("[embed_local] Model loaded: dim={dim}");

    *guard = Some(EmbeddingModel {
        model,
        tokenizer,
        device,
        dim,
    });
    Ok(())
}

/// Check if the model is loaded.
pub fn is_loaded() -> bool {
    MODEL.lock().map(|g| g.is_some()).unwrap_or(false)
}

/// Embedding dimension of the loaded model.
pub fn dim() -> usize {
    MODEL
        .lock()
        .ok()
        .and_then(|g| g.as_ref().map(|m| m.dim))
        .unwrap_or(768)
}

/// Embed a batch of texts. Returns Vec<Vec<f32>> of shape [n_texts][dim].
pub fn embed_texts(texts: &[String]) -> Result<Vec<Vec<f32>>> {
    let guard = MODEL.lock().map_err(|e| anyhow::anyhow!("Lock error: {e}"))?;
    let em = guard
        .as_ref()
        .ok_or_else(|| anyhow::anyhow!("Embedding model not loaded — call ensure_loaded first"))?;

    let max_len = 256usize; // BGE models use 256 max
    let mut all_embeddings: Vec<Vec<f32>> = Vec::with_capacity(texts.len());

    // Process in batches to limit memory
    let batch_size = 32;
    for batch_start in (0..texts.len()).step_by(batch_size) {
        let batch_end = (batch_start + batch_size).min(texts.len());
        let batch = &texts[batch_start..batch_end];

        let embeddings = encode_batch(em, batch, max_len)?;
        all_embeddings.extend(embeddings);
    }

    Ok(all_embeddings)
}

/// Embed a single query.
pub fn embed_query(query: &str) -> Result<Vec<f32>> {
    let mut result = embed_texts(&[query.to_string()])?;
    result
        .pop()
        .ok_or_else(|| anyhow::anyhow!("Empty embedding result"))
}

// ---------------------------------------------------------------------------
// Internal
// ---------------------------------------------------------------------------

fn encode_batch(em: &EmbeddingModel, texts: &[String], max_len: usize) -> Result<Vec<Vec<f32>>> {
    let n = texts.len();

    // Tokenize all texts
    let encodings = em
        .tokenizer
        .encode_batch(texts.to_vec(), true)
        .map_err(|e| anyhow::anyhow!("Tokenization error: {e}"))?;

    // Pad to uniform length
    let seq_len = encodings
        .iter()
        .map(|e| e.get_ids().len().min(max_len))
        .max()
        .unwrap_or(1);

    let mut input_ids: Vec<u32> = Vec::with_capacity(n * seq_len);
    let mut attention_mask: Vec<f32> = Vec::with_capacity(n * seq_len);
    let mut token_type_ids: Vec<u32> = Vec::with_capacity(n * seq_len);

    for enc in &encodings {
        let ids = enc.get_ids();
        let len = ids.len().min(seq_len);

        // Copy tokens (truncated to seq_len)
        input_ids.extend_from_slice(&ids[..len]);
        attention_mask.extend(std::iter::repeat(1.0f32).take(len));
        token_type_ids.extend(std::iter::repeat(0u32).take(len));

        // Pad to seq_len
        let pad = seq_len - len;
        if pad > 0 {
            input_ids.extend(std::iter::repeat(0u32).take(pad));
            attention_mask.extend(std::iter::repeat(0.0f32).take(pad));
            token_type_ids.extend(std::iter::repeat(0u32).take(pad));
        }
    }

    // Build tensors
    let input_ids = Tensor::from_vec(input_ids, (n, seq_len), &em.device)?;
    let token_type_ids = Tensor::from_vec(token_type_ids, (n, seq_len), &em.device)?;
    let attention_mask_tensor =
        Tensor::from_vec(attention_mask.clone(), (n, seq_len), &em.device)?;

    // Forward pass
    let output = em.model.forward(&input_ids, &token_type_ids, None)?;
    // output shape: [batch, seq_len, hidden_size]

    // Mean pooling: sum(output * mask) / sum(mask)
    let mask = attention_mask_tensor
        .unsqueeze(2)?
        .broadcast_as(output.shape())?;
    let masked = output.mul(&mask)?;
    let summed = masked.sum(1)?; // [batch, hidden_size]
    let counts = mask.sum(1)?; // [batch, hidden_size]
    let pooled = summed.div(&counts)?;

    // Normalize to unit length
    let norms = pooled
        .sqr()?
        .sum_keepdim(1)?
        .sqrt()?
        .broadcast_as(pooled.shape())?;
    let normalized = pooled.div(&norms.clamp(1e-10, f64::MAX)?)?;

    // Extract to Vec<Vec<f32>>
    let flat: Vec<f32> = normalized.to_vec2()?
        .into_iter()
        .flatten()
        .collect();

    let dim = em.dim;
    Ok(flat.chunks(dim).map(|c| c.to_vec()).collect())
}

fn get_device() -> Device {
    // BGE-base layer-norm is not implemented in Candle's Metal backend
    // when the `accelerate` feature is also enabled. Use CPU + Accelerate BLAS.
    Device::Cpu
}
