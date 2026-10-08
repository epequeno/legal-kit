"""Embedding generation via OpenAI-compatible API.

Uses OpenRouter by default (supports OpenAI embedding models),
falls back to direct OpenAI if OPENAI_API_KEY is set.

Also supports local sentence-transformers models for self-hosted deployment.
"""

from __future__ import annotations

import os

import numpy as np
import openai


_client: openai.OpenAI | None = None
_local_model = None  # Cached sentence-transformers model
_local_model_path: str | None = None


def _get_client() -> openai.OpenAI:
    global _client
    if _client is None:
        if os.environ.get("OPENAI_API_KEY"):
            _client = openai.OpenAI()
        elif os.environ.get("OPENROUTER_API_KEY"):
            _client = openai.OpenAI(
                base_url="https://openrouter.ai/api/v1",
                api_key=os.environ["OPENROUTER_API_KEY"],
            )
        else:
            raise RuntimeError(
                "No embedding API key found. Set OPENAI_API_KEY or OPENROUTER_API_KEY."
            )
    return _client


def _default_model() -> str:
    """Pick the right model name depending on provider."""
    if os.environ.get("OPENAI_API_KEY"):
        return "text-embedding-3-large"
    return "openai/text-embedding-3-large"


def embed_texts(
    texts: list[str],
    model: str | None = None,
    batch_size: int = 50,
) -> np.ndarray:
    """Embed a list of texts using OpenAI's embedding API.

    Returns numpy array of shape (len(texts), embedding_dim).
    """
    import time as _time

    model = model or _default_model()
    client = _get_client()
    all_embeddings = []

    for i in range(0, len(texts), batch_size):
        batch = texts[i : i + batch_size]
        # Truncate very long texts to avoid API limits (~8191 tokens ≈ 30K chars)
        batch = [t[:30000] if len(t) > 30000 else t for t in batch]
        # Skip empty texts
        batch = [t if t.strip() else "empty" for t in batch]

        for attempt in range(3):
            try:
                response = client.embeddings.create(
                    model=model,
                    input=batch,
                )
                batch_embeddings = [item.embedding for item in response.data]
                all_embeddings.extend(batch_embeddings)
                break
            except Exception as e:
                if attempt < 2:
                    wait = 2 ** attempt
                    _time.sleep(wait)
                else:
                    # Last resort: embed texts one at a time
                    for single in batch:
                        try:
                            _time.sleep(0.5)
                            resp = client.embeddings.create(
                                model=model,
                                input=[single[:10000]],
                            )
                            all_embeddings.append(resp.data[0].embedding)
                        except Exception:
                            # Use zero vector as fallback
                            dim = len(all_embeddings[0]) if all_embeddings else 3072
                            all_embeddings.append([0.0] * dim)

        # Small delay between batches to avoid rate limits
        if i + batch_size < len(texts):
            _time.sleep(0.1)

    return np.array(all_embeddings, dtype=np.float32)


def embed_query(
    query: str,
    model: str | None = None,
) -> np.ndarray:
    """Embed a single query string. Returns 1-D numpy array."""
    result = embed_texts([query], model=model)
    return result[0]


# ---------------------------------------------------------------------------
# Local embedding model support
# ---------------------------------------------------------------------------


def load_local_embedding_model(model_path: str, device: str = "cpu") -> None:
    """Load a sentence-transformers model for local embedding.
    
    Caches it module-level so subsequent calls use the cached model.
    
    Args:
        model_path: Path to sentence-transformers model directory.
        device: Device to load model on ('cpu', 'cuda', 'mps').
    
    Raises:
        RuntimeError: If model loading fails.
    """
    global _local_model, _local_model_path
    
    try:
        from sentence_transformers import SentenceTransformer
        # Try loading directly; if pooling config version mismatch, fall back
        # to loading as a plain transformers model with mean pooling.
        try:
            _local_model = SentenceTransformer(model_path, device=device)
        except TypeError:
            # sentence-transformers version mismatch on pooling config.
            # Load the base transformer and wrap it.
            from transformers import AutoTokenizer, AutoModel
            import torch

            class _MeanPoolingModel:
                """Minimal sentence-transformer compatible wrapper using mean pooling."""
                def __init__(self, model_path: str, device: str):
                    self._tokenizer = AutoTokenizer.from_pretrained(model_path)
                    self._model = AutoModel.from_pretrained(model_path)
                    self._device = torch.device(device if device != "cuda" or torch.cuda.is_available() else "cpu")
                    self._model = self._model.to(self._device)
                    self._model.eval()

                def encode(self, sentences: list[str], batch_size: int = 32,
                           show_progress_bar: bool = False,
                           convert_to_numpy: bool = True, **kwargs) -> np.ndarray:
                    import torch
                    all_embeddings = []
                    for i in range(0, len(sentences), batch_size):
                        batch = sentences[i:i + batch_size]
                        inputs = self._tokenizer(batch, padding=True, truncation=True,
                                                 max_length=256, return_tensors="pt")
                        inputs = {k: v.to(self._device) for k, v in inputs.items()}
                        with torch.no_grad():
                            outputs = self._model(**inputs)
                        # Mean pooling over token embeddings
                        mask = inputs["attention_mask"].unsqueeze(-1).float()
                        embeddings = (outputs.last_hidden_state * mask).sum(1) / mask.sum(1)
                        all_embeddings.append(embeddings.cpu().numpy())
                    return np.concatenate(all_embeddings, axis=0)

            _local_model = _MeanPoolingModel(model_path, device)
        _local_model_path = model_path
    except Exception as e:
        raise RuntimeError(f"Failed to load local embedding model: {e}") from e


def embed_texts_local(
    texts: list[str],
    model_path: str | None = None,
    batch_size: int = 32,
    device: str = "cpu",
) -> np.ndarray:
    """Embed texts using a local sentence-transformers model.
    
    Loads from model_path on first call; uses cache on subsequent calls.
    Falls back to API if model_path is None and no model is cached.
    
    Args:
        texts: List of strings to embed.
        model_path: Path to model directory (None = use cached or fall back to API).
        batch_size: Batch size for encoding.
        device: Device to load model on ('cpu', 'cuda', 'mps').
    
    Returns:
        Numpy array of shape (len(texts), embedding_dim).
    """
    global _local_model, _local_model_path
    
    # Load model if needed
    if _local_model is None or (model_path and model_path != _local_model_path):
        if model_path:
            load_local_embedding_model(model_path, device)
        else:
            # No local model available, fall back to API
            return embed_texts(texts)
    
    # Truncate very long texts and handle empty strings
    texts = [t[:10000] if len(t) > 10000 else t for t in texts]
    texts = [t if t.strip() else "empty" for t in texts]
    
    # Encode with local model
    embeddings = _local_model.encode(
        texts,
        batch_size=batch_size,
        show_progress_bar=False,
        convert_to_numpy=True,
    )
    
    return embeddings.astype(np.float32)


def embed_query_local(query: str, model_path: str | None = None) -> np.ndarray:
    """Embed a single query with local model.
    
    Args:
        query: Query string to embed.
        model_path: Path to model directory (None = use cached).
    
    Returns:
        1-D numpy array.
    """
    return embed_texts_local([query], model_path=model_path)[0]
