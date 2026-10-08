//! Persistent settings stored at ~/.legalkit/settings.json
//!
//! Environment variables always take priority over saved settings.
//! If both are set, the env var wins.

use anyhow::Result;
use serde::{Deserialize, Serialize};
use std::path::PathBuf;

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct Settings {
    #[serde(default)]
    pub openrouter_key: String,
    #[serde(default)]
    pub openrouter_model: String,
    #[serde(default)]
    pub openai_key: String,
    #[serde(default)]
    pub anthropic_key: String,
    #[serde(default)]
    pub anthropic_model: String,
}

fn settings_path() -> Result<PathBuf> {
    let base = dirs::home_dir()
        .ok_or_else(|| anyhow::anyhow!("Cannot locate home directory"))?;
    let dir = base.join(".legalkit");
    std::fs::create_dir_all(&dir)?;
    Ok(dir.join("settings.json"))
}

pub fn load() -> Settings {
    settings_path()
        .ok()
        .and_then(|p| std::fs::read_to_string(p).ok())
        .and_then(|json| serde_json::from_str(&json).ok())
        .unwrap_or_default()
}

pub fn save(settings: &Settings) -> Result<()> {
    let path = settings_path()?;
    let json = serde_json::to_string_pretty(settings)?;
    std::fs::write(path, json)?;
    Ok(())
}

/// Merge saved settings with environment variables.
/// Env vars take priority — if set, they override the saved value.
pub fn merge_with_env(saved: &Settings) -> Settings {
    Settings {
        openrouter_key: env_or(&saved.openrouter_key, "OPENROUTER_API_KEY"),
        openrouter_model: env_or(&saved.openrouter_model, "OPENROUTER_MODEL"),
        openai_key: env_or(&saved.openai_key, "OPENAI_API_KEY"),
        anthropic_key: env_or(&saved.anthropic_key, "ANTHROPIC_API_KEY"),
        anthropic_model: env_or(&saved.anthropic_model, "ANTHROPIC_MODEL"),
    }
}

fn env_or(saved: &str, var: &str) -> String {
    match std::env::var(var) {
        Ok(v) if !v.is_empty() => v,
        _ => saved.to_string(),
    }
}

/// Delete all LegalKit data: documents, settings, everything.
pub fn clear_all_data() -> Result<()> {
    let base = dirs::home_dir()
        .ok_or_else(|| anyhow::anyhow!("Cannot locate home directory"))?;
    let dir = base.join(".legalkit");
    if dir.exists() {
        std::fs::remove_dir_all(&dir)?;
    }
    Ok(())
}
