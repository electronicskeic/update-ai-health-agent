from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class LLMConfig:
    api_key: str
    model: str


def get_llm_config() -> LLMConfig | None:
    api_key = (os.getenv("GEMINI_API_KEY") or "").strip()
    if not api_key:
        return None
    model = (os.getenv("GEMINI_MODEL") or "gemini-1.5-flash").strip()
    return LLMConfig(api_key=api_key, model=model)


def chat_with_llm(*, system_prompt: str, messages: list[dict[str, Any]], model: str) -> str:
    """
    Uses the Google Generative AI Python SDK.
    """
    import google.generativeai as genai  # type: ignore

    api_key = (os.getenv("GEMINI_API_KEY") or "").strip()
    if not api_key:
        raise RuntimeError("GEMINI_API_KEY is not set.")

    genai.configure(api_key=api_key)

    history = []
    for msg in messages:
        role = "model" if msg["role"] == "assistant" else "user"
        history.append({"role": role, "parts": [msg["content"]]})

    try:
        m = genai.GenerativeModel(
            model_name=model,
            system_instruction=system_prompt,
        )
        response = m.generate_content(
            contents=history,
            generation_config=genai.GenerationConfig(temperature=0.4)
        )
        return (response.text or "").strip()
    except Exception as e:
        print(f"Gemini Error: {e}")
        return ""

