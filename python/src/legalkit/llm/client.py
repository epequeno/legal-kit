"""LLM client — supports Anthropic direct or OpenRouter (OpenAI-compatible)."""

from __future__ import annotations

import os
from dataclasses import dataclass
from enum import Enum


class LLMProvider(Enum):
    ANTHROPIC = "anthropic"
    OPENROUTER = "openrouter"


def _detect_provider() -> LLMProvider:
    """Auto-detect which LLM provider to use based on available keys."""
    if os.environ.get("ANTHROPIC_API_KEY"):
        return LLMProvider.ANTHROPIC
    if os.environ.get("OPENROUTER_API_KEY"):
        return LLMProvider.OPENROUTER
    raise RuntimeError(
        "No LLM API key found. Set ANTHROPIC_API_KEY or OPENROUTER_API_KEY."
    )


@dataclass
class LLMConfig:
    """Configuration for LLM calls."""

    generation_model: str = "claude-sonnet-4-20250514"
    context_model: str = "claude-3-haiku-20240307"
    max_tokens: int = 4096
    temperature: float = 0.0


# OpenRouter model name mapping
_OPENROUTER_MODELS = {
    "claude-sonnet-4-20250514": "anthropic/claude-sonnet-4",
    "claude-3-haiku-20240307": "anthropic/claude-3-haiku",
}


class LLMClient:
    """LLM client supporting Anthropic API or OpenRouter.

    Auto-detects provider from environment variables.
    """

    def __init__(self, config: LLMConfig | None = None):
        self.config = config or LLMConfig()
        self._provider = _detect_provider()

    def _resolve_model(self, model: str) -> str:
        """Map model name for the current provider."""
        if self._provider == LLMProvider.OPENROUTER:
            return _OPENROUTER_MODELS.get(model, f"anthropic/{model}")
        return model

    def generate(
        self,
        system: str,
        user: str,
        *,
        model: str | None = None,
        max_tokens: int | None = None,
        temperature: float | None = None,
    ) -> str:
        """Generate a completion. Returns the text response."""
        resolved_model = self._resolve_model(model or self.config.generation_model)
        temp = temperature if temperature is not None else self.config.temperature
        tokens = max_tokens or self.config.max_tokens

        if self._provider == LLMProvider.ANTHROPIC:
            return self._generate_anthropic(system, user, resolved_model, tokens, temp)
        else:
            return self._generate_openrouter(system, user, resolved_model, tokens, temp)

    def _generate_anthropic(
        self, system: str, user: str, model: str, max_tokens: int, temperature: float,
    ) -> str:
        import anthropic
        client = anthropic.Anthropic()
        response = client.messages.create(
            model=model,
            max_tokens=max_tokens,
            temperature=temperature,
            system=system,
            messages=[{"role": "user", "content": user}],
        )
        return response.content[0].text

    def _generate_openrouter(
        self, system: str, user: str, model: str, max_tokens: int, temperature: float,
    ) -> str:
        import openai
        client = openai.OpenAI(
            base_url="https://openrouter.ai/api/v1",
            api_key=os.environ["OPENROUTER_API_KEY"],
        )
        response = client.chat.completions.create(
            model=model,
            max_tokens=max_tokens,
            temperature=temperature,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
        )
        return response.choices[0].message.content

    def expand_query(
        self,
        query: str,
        n: int = 3,
    ) -> list[str]:
        """Generate alternative phrasings of a retrieval query.

        Uses legal terminology substitution and structural variation to
        broaden recall against legal document text.

        Args:
            query: The original user question.
            n: Number of alternative phrasings to generate.

        Returns:
            List of alternative query strings (may be shorter than n if
            the model produces fewer, or if parsing finds fewer lines).
        """
        from legalkit.llm.prompts import QUERY_EXPANSION_SYSTEM, QUERY_EXPANSION_USER

        system = QUERY_EXPANSION_SYSTEM.format(n=n)
        user = QUERY_EXPANSION_USER.format(query=query, n=n)

        raw = self.generate(
            system=system,
            user=user,
            model=self.config.context_model,
            max_tokens=300,
        )

        # Parse one phrasing per non-empty line
        phrasings = [line.strip() for line in raw.splitlines() if line.strip()]
        return phrasings[:n]

    def contextualize_chunk(
        self,
        document: str,
        chunk_path: str,
        chunk_text: str,
    ) -> str:
        """Generate contextual description for a chunk (Anthropic-style).

        Args:
            document: Full (or truncated) document text used as context.
            chunk_path: Human-readable structural path, e.g. "Article IV > §4.2(a)".
            chunk_text: The raw chunk text to contextualize.

        Returns:
            2–3 sentence context string ready to prepend before embedding.
        """
        from legalkit.llm.prompts import CONTEXTUALIZE_SYSTEM, CONTEXTUALIZE_USER

        user = CONTEXTUALIZE_USER.format(
            document=document,
            chunk_path=chunk_path,
            chunk_text=chunk_text,
        )
        return self.generate(
            system=CONTEXTUALIZE_SYSTEM,
            user=user,
            model=self.config.context_model,
            max_tokens=150,
        )
