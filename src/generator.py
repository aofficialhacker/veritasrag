"""
generator.py  -  Phase 5, Generation (Section 4.3).
===================================================
Produces the final answer from a grounded prompt. Three interchangeable
backends are supported and can be switched at runtime (this is the Modular-RAG
idea from Section 4.6 in miniature):

    * "gemini"     - Google gemini-2.5-flash over REST (cloud, default)
    * "ollama"     - a local model served by Ollama (free, offline)
    * "extractive" - no LLM at all: returns the top passages verbatim so the
                     app still works if neither backend is reachable.

The backend is selected from GeneratorConfig but can be overridden per call, so
the Streamlit sidebar can flip it live. Every backend degrades gracefully:
if Gemini or Ollama errors out, we fall back to extractive rather than crash.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional

import requests

from .config import GENERATOR
from .prompt_builder import build_prompt, build_ungrounded_prompt
from .retriever import RetrievedPassage


@dataclass
class GenerationResult:
    answer: str
    backend_used: str
    grounded: bool
    note: str = ""


# --------------------------------------------------------------------------
# Backend calls
# --------------------------------------------------------------------------
def _call_gemini(prompt: str, timeout: int = 60) -> str:
    url = (
        f"https://generativelanguage.googleapis.com/v1beta/models/"
        f"{GENERATOR.gemini_model}:generateContent"
    )
    headers = {
        "Content-Type": "application/json",
        "X-goog-api-key": GENERATOR.gemini_api_key,
    }
    body = {
        "contents": [{"parts": [{"text": prompt}]}],
        "generationConfig": {
            "temperature": GENERATOR.temperature,
            "maxOutputTokens": GENERATOR.max_output_tokens,
        },
    }
    resp = requests.post(url, headers=headers, json=body, timeout=timeout)
    resp.raise_for_status()
    data = resp.json()
    candidates = data.get("candidates", [])
    if not candidates:
        raise RuntimeError(f"Gemini returned no candidates: {data}")
    parts = candidates[0].get("content", {}).get("parts", [])
    return "".join(part.get("text", "") for part in parts).strip()


def _call_ollama(prompt: str, timeout: int = 120) -> str:
    url = f"{GENERATOR.ollama_host}/api/generate"
    body = {
        "model": GENERATOR.ollama_model,
        "prompt": prompt,
        "stream": False,
        "options": {"temperature": GENERATOR.temperature},
    }
    resp = requests.post(url, json=body, timeout=timeout)
    resp.raise_for_status()
    return resp.json().get("response", "").strip()


def _extractive_answer(passages: List[RetrievedPassage], max_passages: int = 3) -> str:
    """No-LLM fallback: stitch the top passages into a readable, cited answer."""
    if not passages:
        return "I don't have enough information in the provided documents to answer that."
    lines = [
        "Based on the most relevant retrieved passages (extractive mode, no LLM):",
        "",
    ]
    for i, p in enumerate(passages[:max_passages], start=1):
        snippet = p.chunk.text.strip()
        if len(snippet) > 400:
            snippet = snippet[:400].rsplit(" ", 1)[0] + "..."
        lines.append(f"[{i}] {snippet}  (source: {p.citation})")
    return "\n\n".join(lines)


# --------------------------------------------------------------------------
# Public API
# --------------------------------------------------------------------------
def health_check(backend: str) -> tuple[bool, str]:
    """Cheap probe used by the UI to show a green/red backend status dot."""
    try:
        if backend == "gemini":
            if not GENERATOR.gemini_api_key:
                return False, "No GEMINI_API_KEY set"
            _call_gemini("Reply with OK.", timeout=15)
            return True, "Gemini reachable"
        if backend == "ollama":
            resp = requests.get(f"{GENERATOR.ollama_host}/api/tags", timeout=5)
            resp.raise_for_status()
            models = [m.get("name", "") for m in resp.json().get("models", [])]
            return True, f"Ollama reachable ({', '.join(models) or 'no models pulled'})"
        if backend == "extractive":
            return True, "Extractive mode always available"
    except Exception as exc:  # noqa: BLE001 - surfaced to the UI
        return False, str(exc)
    return False, "Unknown backend"


def generate(
    query: str,
    passages: List[RetrievedPassage],
    backend: Optional[str] = None,
    grounded: bool = True,
) -> GenerationResult:
    """
    Produce an answer. If `grounded` is False, the retrieved passages are ignored
    and the model answers from parametric memory (used by the 'without RAG'
    demonstration toggle).
    """
    backend = backend or GENERATOR.backend
    prompt = build_prompt(query, passages) if grounded else build_ungrounded_prompt(query)

    # Extractive backend cannot answer ungrounded questions.
    if backend == "extractive":
        if not grounded:
            return GenerationResult(
                answer="(Extractive mode has no language model, so it cannot answer "
                "without retrieved context.)",
                backend_used="extractive",
                grounded=False,
            )
        return GenerationResult(_extractive_answer(passages), "extractive", True)

    try:
        if backend == "gemini":
            answer = _call_gemini(prompt)
        elif backend == "ollama":
            answer = _call_ollama(prompt)
        else:
            raise ValueError(f"Unknown backend: {backend}")
        return GenerationResult(answer=answer, backend_used=backend, grounded=grounded)
    except Exception as exc:  # noqa: BLE001
        # Graceful degradation: never crash a demo.
        if grounded:
            fallback = _extractive_answer(passages)
            return GenerationResult(
                answer=fallback,
                backend_used="extractive",
                grounded=True,
                note=f"{backend} unavailable ({exc}); fell back to extractive mode.",
            )
        return GenerationResult(
            answer=f"({backend} unavailable: {exc})",
            backend_used="extractive",
            grounded=False,
            note=str(exc),
        )
