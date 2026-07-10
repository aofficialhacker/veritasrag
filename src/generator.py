"""
generator.py  -  Phase 5, Generation (Section 4.3).
===================================================
Produces the final answer from a grounded prompt. Three interchangeable
backends are supported and can be switched at runtime (this is the Modular-RAG
idea from Section 4.6 in miniature):

    * "gemini"     - Google Gemini via the Interactions API (google-genai SDK)
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
from google import genai

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
def _call_gemini(prompt: str, timeout: int = 90) -> str:
    """Generate an answer via the Gemini Interactions API (google-genai SDK)."""
    client = genai.Client(
        api_key=GENERATOR.gemini_api_key,
        http_options={"timeout": timeout * 1000},  # SDK timeout is milliseconds
    )
    generation_config = {
        "temperature": GENERATOR.temperature,
        "max_output_tokens": GENERATOR.max_output_tokens,
    }
    # Gemini 3 models "think" by default, which adds seconds of latency (and can
    # blow past short timeouts). We don't need deep reasoning to stitch a grounded
    # answer, so ask for the low thinking level to keep responses fast.
    if "gemini-3" in GENERATOR.gemini_model:
        generation_config["thinking_level"] = "low"
    interaction = client.interactions.create(
        model=GENERATOR.gemini_model,
        input=prompt,
        generation_config=generation_config,
    )
    text = (interaction.output_text or "").strip()
    if not text:
        raise RuntimeError(f"Gemini returned no text (interaction id={interaction.id})")
    return text


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
            _call_gemini("Reply with OK.", timeout=60)
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
