#![cfg_attr(not(debug_assertions), windows_subsystem = "windows")]

mod commands;

use commands::AppConfig;
use tokio::sync::Mutex;

fn main() {
    // Load persisted settings, then overlay env vars (env vars win)
    let saved = legalkit_lib::settings::load();
    let merged = legalkit_lib::settings::merge_with_env(&saved);

    let config = AppConfig {
        openrouter_key: merged.openrouter_key,
        openrouter_model: merged.openrouter_model,
        openai_key: merged.openai_key,
        anthropic_key: merged.anthropic_key,
        anthropic_model: merged.anthropic_model,
    };

    // Try to load the local embedding model at startup
    {
        // Look for the model relative to the executable, then in the repo models/ dir
        let model_paths = vec![
            std::env::current_exe()
                .ok()
                .and_then(|p| p.parent().map(|d| d.join("models/legal-embeddings/model"))),
            Some(std::path::PathBuf::from("models/legal-embeddings/model")),
            Some(dirs::home_dir().unwrap_or_default().join(".legalkit/models/legal-embeddings/model")),
            // Repo-relative path (dev mode)
            Some(std::path::PathBuf::from(
                concat!(env!("CARGO_MANIFEST_DIR"), "/../../models/legal-embeddings/model"),
            )),
        ];

        for path_opt in model_paths {
            if let Some(path) = path_opt {
                if path.join("model.safetensors").exists() {
                    match legalkit_lib::pipeline::embed_local::ensure_loaded(&path) {
                        Ok(()) => {
                            println!("[LegalKit] Local embedding model loaded from {}", path.display());
                            break;
                        }
                        Err(e) => {
                            eprintln!("[LegalKit] Failed to load embedding model from {}: {e}", path.display());
                        }
                    }
                }
            }
        }

        if !legalkit_lib::pipeline::embed_local::is_loaded() {
            println!("[LegalKit] No local embedding model found — using API for embeddings");
        }
    }

    // Try to load the local entailment model
    {
        let model_paths = vec![
            Some(std::path::PathBuf::from(
                concat!(env!("CARGO_MANIFEST_DIR"), "/../../models/legal-entailment/model"),
            )),
            Some(dirs::home_dir().unwrap_or_default().join(".legalkit/models/legal-entailment/model")),
        ];

        for path_opt in model_paths {
            if let Some(path) = path_opt {
                if path.join("model.safetensors").exists() {
                    match legalkit_lib::pipeline::entailment_local::ensure_loaded(&path) {
                        Ok(()) => {
                            println!("[LegalKit] Local entailment model loaded from {}", path.display());
                            break;
                        }
                        Err(e) => {
                            eprintln!("[LegalKit] Failed to load entailment model from {}: {e}", path.display());
                        }
                    }
                }
            }
        }

        if !legalkit_lib::pipeline::entailment_local::is_loaded() {
            println!("[LegalKit] No local entailment model found — citation verification disabled");
        }
    }

    // Try to load the local reranker model
    {
        let model_paths = vec![
            Some(std::path::PathBuf::from(
                concat!(env!("CARGO_MANIFEST_DIR"), "/../../models/legal-reranker/final"),
            )),
            Some(dirs::home_dir().unwrap_or_default().join(".legalkit/models/legal-reranker/final")),
        ];

        for path_opt in model_paths {
            if let Some(path) = path_opt {
                if path.join("model.safetensors").exists() {
                    match legalkit_lib::pipeline::rerank_local::ensure_loaded(&path) {
                        Ok(()) => {
                            println!("[LegalKit] Local reranker model loaded from {}", path.display());
                            break;
                        }
                        Err(e) => {
                            eprintln!("[LegalKit] Failed to load reranker from {}: {e}", path.display());
                        }
                    }
                }
            }
        }

        if !legalkit_lib::pipeline::rerank_local::is_loaded() {
            println!("[LegalKit] No local reranker found — retrieval order unchanged");
        }
    }

    tauri::Builder::default()
        .plugin(tauri_plugin_dialog::init())
        .plugin(tauri_plugin_shell::init())
        .manage(Mutex::new(config))
        .invoke_handler(tauri::generate_handler![
            commands::ingest_document,
            commands::ask_question,
            commands::get_document_outline,
            commands::list_documents,
            commands::delete_document,
            commands::set_api_keys,
            commands::get_api_key_status,
            commands::load_chat_history,
            commands::clear_chat_history,
            commands::save_settings,
            commands::get_settings,
            commands::clear_all_data,
        ])
        .run(tauri::generate_context!())
        .expect("error while running tauri application");
}
