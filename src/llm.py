"""Unified OpenAI-compatible client wrapper for Assay.

Connects to Ollama (or other OpenAI-compatible endpoints) using settings
defined in .env:
  LLM_BASE_URL: e.g. http://localhost:11434/v1
  LLM_API_KEY: e.g. ollama
  GENERATION_MODEL: e.g. qwen2.5-coder:7b
  EMBEDDING_MODEL: e.g. nomic-embed-text
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union
import openai

# Load environment variables from .env if present
def _load_env():
    env_file = Path(__file__).resolve().parent.parent / ".env"
    if env_file.exists():
        with open(env_file, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith("#") and "=" in line:
                    key, val = line.split("=", 1)
                    key = key.strip()
                    val = val.strip().strip("\"'")
                    if key not in os.environ:
                        os.environ[key] = val

_load_env()

LLM_BASE_URL = os.getenv("LLM_BASE_URL", "http://localhost:11434/v1")
LLM_API_KEY = os.getenv("LLM_API_KEY", "ollama")
GENERATION_MODEL = os.getenv("GENERATION_MODEL", "qwen2.5-coder:7b")
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "nomic-embed-text")


class LLMClient:
    """OpenAI-compatible client for local inference via Ollama."""

    def __init__(
        self,
        base_url: Optional[str] = None,
        api_key: Optional[str] = None,
        generation_model: Optional[str] = None,
        embedding_model: Optional[str] = None,
    ):
        self.base_url = base_url or os.getenv("LLM_BASE_URL", LLM_BASE_URL)
        self.api_key = api_key or os.getenv("LLM_API_KEY", LLM_API_KEY)
        self.generation_model = generation_model or os.getenv("GENERATION_MODEL", GENERATION_MODEL)
        self.embedding_model = embedding_model or os.getenv("EMBEDDING_MODEL", EMBEDDING_MODEL)

        self.client = openai.OpenAI(
            base_url=self.base_url,
            api_key=self.api_key,
        )

    def generate(
        self,
        messages: List[Dict[str, str]],
        model: Optional[str] = None,
        temperature: float = 0.0,
        num_ctx: int = 8192,
        **kwargs: Any,
    ) -> Tuple[str, Dict[str, Any]]:
        """Generate a completion enforcing temperature=0.0 and num_ctx=8192.
        
        Returns:
            Tuple of (response_text, telemetry_dict)
        """
        target_model = model or self.generation_model
        extra_body = kwargs.pop("extra_body", {})
        # Ensure num_ctx is always enforced to prevent DDL truncation
        options = extra_body.get("options", {})
        options.setdefault("num_ctx", num_ctx)
        extra_body["options"] = options

        response = self.client.chat.completions.create(
            model=target_model,
            messages=messages,
            temperature=temperature,
            extra_body=extra_body,
            **kwargs,
        )

        content = response.choices[0].message.content or ""
        usage = {
            "prompt_tokens": getattr(response.usage, "prompt_tokens", 0) if response.usage else 0,
            "completion_tokens": getattr(response.usage, "completion_tokens", 0) if response.usage else 0,
            "total_tokens": getattr(response.usage, "total_tokens", 0) if response.usage else 0,
        }
        return content, usage

    def get_embedding(
        self,
        input_texts: Union[str, List[str]],
        model: Optional[str] = None,
    ) -> List[List[float]]:
        """Compute dense vector embeddings using nomic-embed-text."""
        target_model = model or self.embedding_model
        if isinstance(input_texts, str):
            texts = [input_texts]
        else:
            texts = input_texts

        response = self.client.embeddings.create(
            model=target_model,
            input=texts,
        )
        return [item.embedding for item in response.data]


# Global shared instance
default_client = LLMClient()
