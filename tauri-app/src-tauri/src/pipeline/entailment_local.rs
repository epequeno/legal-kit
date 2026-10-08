//! Local citation verification via Candle (DeBERTa v2/v3).
//!
//! Loads the legal-entailment-deberta-v3-large model and classifies
//! (passage, claim) pairs into: entailed, partially_entailed, contradicted, neutral.

use anyhow::{bail, Result};
use candle_core::{DType, Device, Tensor};
use candle_nn::VarBuilder;
use candle_transformers::models::debertav2::{Config as DebertaConfig, DebertaV2Model};
use std::path::Path;
use std::sync::Mutex;
use tokenizers::Tokenizer;

const LABELS: &[&str] = &["entailed", "partially_entailed", "contradicted", "neutral"];

static MODEL: Mutex<Option<EntailmentModel>> = Mutex::new(None);

struct EntailmentModel {
    model: DebertaV2Model,
    classifier_weight: Tensor,
    classifier_bias: Tensor,
    tokenizer: Tokenizer,
    device: Device,
}

/// Result of a single entailment check.
#[derive(Debug, Clone)]
pub struct EntailmentResult {
    pub label: String,
    pub label_index: usize,
    pub confidence: f32,
    pub scores: [f32; 4], // softmax over all 4 classes
}

pub fn ensure_loaded(model_dir: &Path) -> Result<()> {
    let mut guard = MODEL.lock().map_err(|e| anyhow::anyhow!("Lock error: {e}"))?;
    if guard.is_some() {
        return Ok(());
    }

    let device = get_device();
    println!("[entailment_local] Loading model from {}", model_dir.display());
    println!("[entailment_local] Device: {:?}", device);

    // Load tokenizer
    let tokenizer_path = model_dir.join("tokenizer.json");
    if !tokenizer_path.exists() {
        bail!("tokenizer.json not found in {}", model_dir.display());
    }
    let tokenizer = Tokenizer::from_file(&tokenizer_path)
        .map_err(|e| anyhow::anyhow!("Failed to load tokenizer: {e}"))?;

    // Load config
    let config_path = model_dir.join("config.json");
    let config_str = std::fs::read_to_string(&config_path)?;
    let config: DebertaConfig = serde_json::from_str(&config_str)?;

    // Load weights with tensor name remapping
    let weights_path = model_dir.join("model.safetensors");
    if !weights_path.exists() {
        bail!("model.safetensors not found in {}", model_dir.display());
    }

    let tensors = candle_core::safetensors::load(&weights_path, &device)?;

    // Remap: strip "deberta." prefix, gamma→weight, beta→bias
    let mut remapped: std::collections::HashMap<String, Tensor> = std::collections::HashMap::new();
    let mut classifier_weight: Option<Tensor> = None;
    let mut classifier_bias: Option<Tensor> = None;

    for (name, tensor) in tensors {
        if name == "classifier.weight" {
            classifier_weight = Some(tensor);
            continue;
        }
        if name == "classifier.bias" {
            classifier_bias = Some(tensor);
            continue;
        }
        if name == "pooler.dense.weight" || name == "pooler.dense.bias" {
            // Skip pooler — we don't need it for classification
            continue;
        }

        // Strip "deberta." prefix
        let new_name = name
            .strip_prefix("deberta.")
            .unwrap_or(&name)
            .to_string()
            .replace("LayerNorm.gamma", "LayerNorm.weight")
            .replace("LayerNorm.beta", "LayerNorm.bias");

        remapped.insert(new_name, tensor);
    }

    let classifier_weight =
        classifier_weight.ok_or_else(|| anyhow::anyhow!("Missing classifier.weight"))?;
    let classifier_bias =
        classifier_bias.ok_or_else(|| anyhow::anyhow!("Missing classifier.bias"))?;

    let vb = VarBuilder::from_tensors(remapped, DType::F32, &device);
    let model = DebertaV2Model::load(vb, &config)?;

    println!("[entailment_local] Model loaded: {} labels", LABELS.len());

    *guard = Some(EntailmentModel {
        model,
        classifier_weight,
        classifier_bias,
        tokenizer,
        device,
    });
    Ok(())
}

pub fn is_loaded() -> bool {
    MODEL.lock().map(|g| g.is_some()).unwrap_or(false)
}

/// Classify a batch of (passage, claim) pairs in a single forward pass.
/// Returns one EntailmentResult per pair.
pub fn verify_batch(pairs: &[(&str, &str)]) -> Result<Vec<EntailmentResult>> {
    if pairs.is_empty() {
        return Ok(vec![]);
    }

    let guard = MODEL.lock().map_err(|e| anyhow::anyhow!("Lock error: {e}"))?;
    let em = guard
        .as_ref()
        .ok_or_else(|| anyhow::anyhow!("Entailment model not loaded"))?;

    let max_len = 512usize;
    let n = pairs.len();

    // Tokenize all pairs at once
    let pair_vec: Vec<(&str, &str)> = pairs.to_vec();
    let encodings = em
        .tokenizer
        .encode_batch(pair_vec, true)
        .map_err(|e| anyhow::anyhow!("Tokenization error: {e}"))?;

    // Determine padded sequence length
    let padded_len = encodings
        .iter()
        .map(|e| e.get_ids().len().min(max_len))
        .max()
        .unwrap_or(1);

    let mut input_ids_flat  = vec![0u32; n * padded_len];
    let mut type_ids_flat   = vec![0u32; n * padded_len];
    let mut mask_flat       = vec![0u32; n * padded_len];

    for (i, enc) in encodings.iter().enumerate() {
        let ids      = enc.get_ids();
        let type_ids = enc.get_type_ids();
        let len = ids.len().min(max_len);
        let base = i * padded_len;
        input_ids_flat[base..base + len].copy_from_slice(&ids[..len]);
        type_ids_flat[base..base + len].copy_from_slice(&type_ids[..len]);
        for j in 0..len {
            mask_flat[base + j] = 1;
        }
    }

    let input_ids     = Tensor::from_vec(input_ids_flat,  (n, padded_len), &em.device)?;
    let token_type_ids = Tensor::from_vec(type_ids_flat,  (n, padded_len), &em.device)?;
    let attention_mask = Tensor::from_vec(mask_flat,      (n, padded_len), &em.device)?;

    // Single batched forward pass → [n, seq_len, hidden_size]
    let output = em
        .model
        .forward(&input_ids, Some(token_type_ids), Some(attention_mask))?;

    // CLS token at position 0 for each sequence → [n, hidden_size]
    // .contiguous() required — narrow() produces a non-contiguous view
    let cls = output.narrow(1, 0, 1)?.squeeze(1)?.contiguous()?;

    // Classifier head: [n, hidden_size] @ [hidden_size, num_labels]^T + bias → [n, num_labels]
    let logits = cls
        .matmul(&em.classifier_weight.t()?)?
        .broadcast_add(&em.classifier_bias)?;

    // Per-row softmax and decode
    let logits_2d: Vec<Vec<f32>> = logits.to_vec2()?;
    let results = logits_2d
        .iter()
        .map(|row| {
            let max_l = row.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
            let exp: Vec<f32> = row.iter().map(|x| (x - max_l).exp()).collect();
            let sum: f32 = exp.iter().sum();
            let probs: Vec<f32> = exp.iter().map(|x| x / sum).collect();
            let (best_idx, &best_prob) = probs
                .iter()
                .enumerate()
                .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
                .unwrap();
            EntailmentResult {
                label: LABELS[best_idx].to_string(),
                label_index: best_idx,
                confidence: best_prob,
                scores: [
                    probs.get(0).copied().unwrap_or(0.0),
                    probs.get(1).copied().unwrap_or(0.0),
                    probs.get(2).copied().unwrap_or(0.0),
                    probs.get(3).copied().unwrap_or(0.0),
                ],
            }
        })
        .collect();

    Ok(results)
}

/// Classify a single (passage, claim) pair. Convenience wrapper around verify_batch.
pub fn verify(passage: &str, claim: &str) -> Result<EntailmentResult> {
    verify_batch(&[(passage, claim)])?
        .into_iter()
        .next()
        .ok_or_else(|| anyhow::anyhow!("Empty batch result"))
}

/// Verify citations in a single batched forward pass.
/// Caps at 8 citations — enough for any real answer, avoids runaway latency.
pub fn verify_citations(
    citations: &mut Vec<crate::models::Citation>,
    chunks: &[crate::models::Chunk],
) {
    const MAX_CITATIONS: usize = 8;

    // Gather (chunk_text, quote, citation_index) for valid citations
    let pairs: Vec<(String, String, usize)> = citations
        .iter()
        .enumerate()
        .take(MAX_CITATIONS)
        .filter_map(|(i, cit)| {
            let chunk_text = chunks
                .iter()
                .find(|c| c.node_id == cit.node_id)
                .map(|c| c.text.clone())
                .unwrap_or_default();
            if chunk_text.is_empty() || cit.quote.is_empty() {
                None
            } else {
                Some((chunk_text, cit.quote.clone(), i))
            }
        })
        .collect();

    if pairs.is_empty() {
        return;
    }

    // Single batched forward pass
    let borrow: Vec<(&str, &str)> = pairs
        .iter()
        .map(|(p, q, _)| (p.as_str(), q.as_str()))
        .collect();

    match verify_batch(&borrow) {
        Ok(results) => {
            for (result, (_, _, idx)) in results.iter().zip(pairs.iter()) {
                citations[*idx].confidence = match result.label.as_str() {
                    "entailed"          => result.confidence,
                    "partially_entailed"=> result.confidence * 0.7,
                    "contradicted"      => 1.0 - result.confidence,
                    "neutral"           => 0.3,
                    _                   => citations[*idx].confidence,
                };
            }
        }
        Err(e) => {
            eprintln!("[entailment_local] Batch verification failed: {e}");
        }
    }
}

fn get_device() -> Device {
    // DeBERTa v2 layer-norm is not implemented in Candle's Metal backend.
    // CPU + Apple Accelerate BLAS is fast enough for citation verification
    // (typically 3–10 pairs per query, ~100–300ms total).
    Device::Cpu
}
