"""LLM client with automatic provider fallback:

  PRIMARY   → Groq (llama-3.3-70b-versatile)       — free, 14,400 req/day
  SECONDARY → OpenRouter (meta-llama/llama-3.3-70b) — kicks in on Groq 429
  FALLBACK  → Gemini 2.5 Flash                       — when both Groq & OR are exhausted

Every component (router, rewriter, SQL agent, synthesizer) calls
complete() or complete_json() here and never knows which provider
actually served the request. Swapping providers or adding a new one
is a one-file change (and a config.py field).
"""
from __future__ import annotations
import json
import logging
from typing import Any

from config import settings

logger = logging.getLogger(__name__)

# ── Lazy client singletons ────────────────────────────────────────────────────

_groq_client = None
_openrouter_client = None
_gemini_client = None


def _get_groq():
    global _groq_client
    if _groq_client is None:
        if not settings.groq_api_key:
            raise RuntimeError(
                "GROQ_API_KEY not set. Get a free key at https://console.groq.com "
                "and add it to your .env file."
            )
        from groq import Groq
        _groq_client = Groq(api_key=settings.groq_api_key)
    return _groq_client


def _get_openrouter():
    global _openrouter_client
    if _openrouter_client is None:
        if not settings.openrouter_api_key:
            return None
        from openai import OpenAI
        _openrouter_client = OpenAI(
            api_key=settings.openrouter_api_key,
            base_url=settings.openrouter_base_url,
        )
    return _openrouter_client


def _get_gemini():
    global _gemini_client
    if _gemini_client is None:
        if not settings.gemini_api_key:
            return None          # Gemini not configured — that's OK
        from google import genai
        _gemini_client = genai.Client(api_key=settings.gemini_api_key)
    return _gemini_client


# ── Provider implementations ──────────────────────────────────────────────────

def _groq_complete(system: str, user: str, max_tokens: int, temperature: float) -> str:
    client = _get_groq()
    resp = client.chat.completions.create(
        model=settings.groq_model,
        messages=[
            {"role": "system", "content": system},
            {"role": "user",   "content": user},
        ],
        max_tokens=max_tokens,
        temperature=temperature,
    )
    return resp.choices[0].message.content or ""


def _groq_complete_json(system: str, user: str, max_tokens: int, temperature: float) -> str:
    """Groq supports OpenAI-style response_format for guaranteed JSON output."""
    client = _get_groq()
    resp = client.chat.completions.create(
        model=settings.groq_model,
        messages=[
            {"role": "system", "content": system},
            {"role": "user",   "content": user},
        ],
        max_tokens=max_tokens,
        temperature=temperature,
        response_format={"type": "json_object"},
    )
    finish = resp.choices[0].finish_reason
    text = resp.choices[0].message.content or ""
    return text, finish


def _openrouter_complete(system: str, user: str, max_tokens: int, temperature: float) -> str:
    """OpenRouter uses an OpenAI-compatible API."""
    client = _get_openrouter()
    if client is None:
        raise RuntimeError("OpenRouter not configured (OPENROUTER_API_KEY missing).")
    resp = client.chat.completions.create(
        model=settings.openrouter_model,
        messages=[
            {"role": "system", "content": system},
            {"role": "user",   "content": user},
        ],
        max_tokens=max_tokens,
        temperature=temperature,
    )
    return resp.choices[0].message.content or ""


def _openrouter_complete_json(system: str, user: str, max_tokens: int, temperature: float) -> str:
    client = _get_openrouter()
    if client is None:
        raise RuntimeError("OpenRouter not configured (OPENROUTER_API_KEY missing).")
    resp = client.chat.completions.create(
        model=settings.openrouter_model,
        messages=[
            {"role": "system", "content": system},
            {"role": "user",   "content": user},
        ],
        max_tokens=max_tokens,
        temperature=temperature,
        response_format={"type": "json_object"},
    )
    finish = resp.choices[0].finish_reason
    text = resp.choices[0].message.content or ""
    return text, finish


def _gemini_complete(system: str, user: str, max_tokens: int, temperature: float) -> str:
    client = _get_gemini()
    if client is None:
        raise RuntimeError("Gemini not configured (GEMINI_API_KEY missing) and Groq also failed.")
    from google.genai import types
    resp = client.models.generate_content(
        model=settings.llm_model,
        contents=user,
        config=types.GenerateContentConfig(
            system_instruction=system,
            max_output_tokens=max_tokens,
            temperature=temperature,
        ),
    )
    return resp.text or ""


def _gemini_complete_json(system: str, user: str, max_tokens: int, temperature: float):
    client = _get_gemini()
    if client is None:
        raise RuntimeError("Gemini not configured (GEMINI_API_KEY missing) and Groq also failed.")
    from google.genai import types
    resp = client.models.generate_content(
        model=settings.llm_model,
        contents=user,
        config=types.GenerateContentConfig(
            system_instruction=system,
            max_output_tokens=max_tokens,
            temperature=temperature,
            response_mime_type="application/json",
        ),
    )
    finish = str(
        resp.candidates[0].finish_reason if resp.candidates else "STOP"
    )
    return resp.text or "", finish


def _is_rate_limit(exc: Exception) -> bool:
    """Detect 429/402 from any provider without importing provider-specific types."""
    msg = str(exc).lower()
    return "429" in msg or "402" in msg or "rate_limit" in msg or "resource_exhausted" in msg or "quota" in msg or "insufficient" in msg or "payment" in msg


# ── Public API ────────────────────────────────────────────────────────────────

def complete(
    system: str,
    user: str,
    max_tokens: int = 1024,
    temperature: float = 0.0,
) -> str:
    """Plain-text completion. Tries Groq → OpenRouter → Gemini, falling back on 429."""
    try:
        result = _groq_complete(system, user, max_tokens, temperature)
        logger.debug("complete() served by Groq")
        return result
    except Exception as e:
        if _is_rate_limit(e):
            logger.warning("Groq rate-limited, trying OpenRouter. (%s)", e)
        else:
            raise

    try:
        result = _openrouter_complete(system, user, max_tokens, temperature)
        logger.debug("complete() served by OpenRouter")
        return result
    except Exception as e:
        if _is_rate_limit(e):
            logger.warning("OpenRouter rate-limited, falling back to Gemini. (%s)", e)
            return _gemini_complete(system, user, max_tokens, temperature)
        raise


def complete_json(
    system: str,
    user: str,
    max_tokens: int = 1024,
    temperature: float = 0.0,
) -> dict[str, Any]:
    """
    JSON-mode completion. Tries Groq → OpenRouter → Gemini, falling back on 429.
    All three providers support JSON mode so malformed output is rare.
    Retries once with doubled token budget if output was truncated.
    """
    try:
        text, finish = _groq_complete_json(system, user, max_tokens, temperature)
        logger.debug("complete_json() served by Groq (finish=%s)", finish)
        return _parse(text, finish, system, user, max_tokens, temperature, provider="groq")
    except Exception as e:
        if _is_rate_limit(e):
            logger.warning("Groq rate-limited, trying OpenRouter. (%s)", e)
        else:
            raise

    try:
        text, finish = _openrouter_complete_json(system, user, max_tokens, temperature)
        logger.debug("complete_json() served by OpenRouter (finish=%s)", finish)
        return _parse(text, finish, system, user, max_tokens, temperature, provider="openrouter")
    except Exception as e:
        if _is_rate_limit(e):
            logger.warning("OpenRouter rate-limited, falling back to Gemini. (%s)", e)
            return _complete_json_gemini(system, user, max_tokens, temperature)
        raise


def _complete_json_gemini(
    system: str, user: str, max_tokens: int, temperature: float
) -> dict[str, Any]:
    text, finish = _gemini_complete_json(system, user, max_tokens, temperature)
    logger.debug("complete_json() served by Gemini (finish=%s)", finish)
    return _parse(text, finish, system, user, max_tokens, temperature, provider="gemini")


def _parse(
    raw: str,
    finish: str,
    system: str,
    user: str,
    max_tokens: int,
    temperature: float,
    provider: str,
    _retried: bool = False,
) -> dict[str, Any]:
    cleaned = raw.strip().strip("```").strip()
    if cleaned.lower().startswith("json"):
        cleaned = cleaned[4:].strip()
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError as e:
        if _retried:
            logger.error("[%s] JSON parse failed twice. Raw: %r", provider, raw)
            raise ValueError(
                f"Model ({provider}) did not return valid JSON after retry. "
                f"Raw response: {raw[:200]!r}. Original error: {e}"
            ) from e

        truncated = "MAX_TOKENS" in str(finish).upper()
        if truncated:
            new_budget = max_tokens * 2
            logger.warning("[%s] Truncated at %d tokens, retrying at %d.", provider, max_tokens, new_budget)
            retry_user = user
        else:
            new_budget = max_tokens
            logger.warning("[%s] Bad JSON (not truncation), retrying once.", provider)
            retry_user = user + "\n\nIMPORTANT: Return ONLY a valid JSON object. No other text."

        if provider in ("groq", "openrouter"):
            text2, finish2 = _groq_complete_json(system, retry_user, new_budget, temperature) if provider == "groq" \
                else _openrouter_complete_json(system, retry_user, new_budget, temperature)
        else:
            text2, finish2 = _gemini_complete_json(system, retry_user, new_budget, temperature)

        return _parse(text2, finish2, system, user, new_budget, temperature, provider, _retried=True)
