"""Minimal OpenAI-compatible chat client (stdlib only).

The same client talks to:
* Ollama (local, private, default)  - http://localhost:11434/v1
* llama.cpp server / vLLM (local)   - any OpenAI-compatible base URL
* OpenAI (cloud, opt-in)            - https://api.openai.com/v1
"""
from __future__ import annotations

import json
import os
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from typing import Any

# Small local models with reliable tool calling, in order of preference.
PREFERRED_OLLAMA_MODELS = ("qwen2.5", "qwen3", "llama3.1", "llama3.2", "mistral-nemo", "mistral", "command-r")


class LLMError(Exception):
    pass


@dataclass
class ToolCall:
    id: str
    name: str
    arguments: dict[str, Any]


@dataclass
class LLMReply:
    content: str
    tool_calls: list[ToolCall] = field(default_factory=list)
    raw_message: dict[str, Any] = field(default_factory=dict)


def _http_json(url: str, payload: dict | None = None, headers: dict | None = None, timeout: float = 120) -> dict:
    data = json.dumps(payload).encode() if payload is not None else None
    req = urllib.request.Request(url, data=data, method="POST" if data else "GET",
                                 headers={"Content-Type": "application/json", **(headers or {})})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return json.loads(resp.read().decode())
    except urllib.error.HTTPError as exc:
        raise LLMError(f"HTTP {exc.code}: {exc.read().decode(errors='replace')[:300]}") from exc
    except (urllib.error.URLError, TimeoutError, ConnectionError, OSError) as exc:
        raise LLMError(f"cannot reach LLM at {url}: {exc}") from exc


class ChatClient:
    def __init__(self, base_url: str, model: str, api_key: str | None = None, label: str = "") -> None:
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.api_key = api_key
        self.label = label or base_url
        self.is_local = any(h in self.base_url for h in ("localhost", "127.0.0.1", "[::1]"))

    def chat(self, messages: list[dict], tools: list[dict]) -> LLMReply:
        headers = {"Authorization": f"Bearer {self.api_key}"} if self.api_key else {}
        body = _http_json(f"{self.base_url}/chat/completions", {
            "model": self.model, "messages": messages, "tools": tools, "temperature": 0,
        }, headers)
        try:
            msg = body["choices"][0]["message"]
        except (KeyError, IndexError) as exc:
            raise LLMError(f"unexpected response: {str(body)[:300]}") from exc
        calls = []
        for i, tc in enumerate(msg.get("tool_calls") or []):
            fn = tc.get("function", {})
            args = fn.get("arguments") or {}
            if isinstance(args, str):
                try:
                    args = json.loads(args) if args.strip() else {}
                except json.JSONDecodeError:
                    args = {"__invalid_json__": args}
            calls.append(ToolCall(tc.get("id") or f"call_{i}", fn.get("name", ""), args))
        return LLMReply(msg.get("content") or "", calls, msg)


def ollama_models(base: str) -> list[str]:
    try:
        tags = _http_json(f"{base.rstrip('/')}/api/tags", timeout=2)
    except LLMError:
        return []
    return [m["name"] for m in tags.get("models", [])]


def pick_ollama_model(models: list[str]) -> str | None:
    for pref in PREFERRED_OLLAMA_MODELS:
        for m in models:
            if m.startswith(pref):
                return m
    return models[0] if models else None


def build_client(backend: str, model: str | None) -> tuple[ChatClient | None, str]:
    """Returns (client or None for offline mode, human-readable description)."""
    ollama = os.getenv("OLLAMA_HOST", "http://localhost:11434")
    if not ollama.startswith("http"):
        ollama = f"http://{ollama}"

    if backend in ("auto", "ollama"):
        models = ollama_models(ollama)
        chosen = model if model and (model in models or backend == "ollama") else pick_ollama_model(models)
        if chosen:
            return ChatClient(f"{ollama}/v1", chosen, label=f"Ollama ({chosen}, local)"), f"local LLM: Ollama/{chosen}"
        if backend == "ollama":
            return None, f"Ollama not reachable at {ollama} or no models pulled -> offline engine"
    if backend == "openai-compatible":
        base = os.getenv("HWQA_LLM_BASE_URL", "http://localhost:8080/v1")
        return ChatClient(base, model or os.getenv("HWQA_LLM_MODEL", "local-model"),
                          os.getenv("HWQA_LLM_API_KEY")), f"OpenAI-compatible server at {base}"
    if backend == "openai":
        key = os.getenv("OPENAI_API_KEY")
        if not key:
            return None, "OPENAI_API_KEY not set -> offline engine"
        m = model or "gpt-4o-mini"
        return ChatClient("https://api.openai.com/v1", m, key, f"OpenAI ({m}, cloud)"), \
            f"cloud LLM: OpenAI/{m} (hardware data will leave this machine)"
    return None, "offline deterministic engine (no LLM)"
