"""Conversation orchestrator: LLM tool-calling loop with an offline fallback.

Anti-hallucination layers
1. The system prompt forbids stating facts that are not in tool output.
2. Tools return explicit ``unavailable`` entries, so "I don't know" has data behind it.
3. Tools pre-compute derived numbers (e.g. RAM upgrade headroom) so the model doesn't do arithmetic.
4. A grounding check flags numbers in the answer that appear in no tool result.
5. With no LLM (or if it fails) the deterministic engine answers from templates.
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import Any, Callable

from .llm import ChatClient, LLMError
from .offline import OfflineEngine
from .tools import ToolRegistry

SYSTEM_PROMPT = """You are a hardware Q&A assistant running locally on the user's Linux computer.

Rules:
1. NEVER guess hardware facts. Call the provided tools to read them, and state only facts that appear in tool results in this conversation.
2. If a tool reports an item under "unavailable" or returns an "error", say plainly that you cannot determine it, give the reason, and how to fix it if the reason says so.
3. Prefer pre-computed fields (e.g. "upgrade.max_addable_now_human", "*_human") over doing your own arithmetic or unit conversion.
4. Answer in plain language, 1-4 short sentences, with units. Do not mention tool names or JSON.
5. For follow-up questions ("and the GPU?", "how fast is it?") use the conversation to infer the topic, and call the relevant tool if you don't already have that data.
6. You are read-only: you cannot change settings, install drivers or run commands other than the tools.
7. If the question is not about this computer's hardware, say you can only answer hardware questions about this machine."""

MAX_TOOL_ROUNDS = 6
MAX_HISTORY_MESSAGES = 30
_NUM_WITH_UNIT = re.compile(r"(\d+(?:\.\d+)?)\s*(tb|gb|gib|mb|mib|ghz|mhz|°c|%|w\b|mt/s|cores?|threads?|slots?)",
                            re.IGNORECASE)


@dataclass
class Answer:
    text: str
    tools_used: list[str] = field(default_factory=list)
    engine: str = "offline"
    ungrounded: list[str] = field(default_factory=list)


def _numbers_in(obj: Any) -> set[float]:
    found: set[float] = set()
    if isinstance(obj, bool):
        return found
    if isinstance(obj, (int, float)):
        found.add(float(obj))
    elif isinstance(obj, str):
        found.update(float(x) for x in re.findall(r"\d+(?:\.\d+)?", obj))
    elif isinstance(obj, dict):
        for v in obj.values():
            found |= _numbers_in(v)
    elif isinstance(obj, list):
        for v in obj:
            found |= _numbers_in(v)
    return found


def ungrounded_numbers(answer: str, tool_results: list[dict]) -> list[str]:
    """Numbers-with-units in the answer that can't be traced to any tool value (any unit scale)."""
    facts = set()
    for r in tool_results:
        facts |= _numbers_in(r)
    scaled = {f / s for f in facts for s in (1, 1000, 1024, 1000**2, 1024**2, 1000**3, 1024**3, 1000**4, 1024**4)}
    bad = []
    for m in _NUM_WITH_UNIT.finditer(answer):
        value = float(m.group(1))
        if value <= 1:
            continue
        if not any(abs(value - f) <= max(0.06 * f, 0.051) for f in scaled if f > 0):
            bad.append(m.group(0))
    return bad


class Assistant:
    def __init__(self, registry: ToolRegistry, llm: ChatClient | None = None,
                 on_tool: Callable[[str, dict, dict], None] | None = None) -> None:
        self.registry = registry
        self.llm = llm
        self.on_tool = on_tool
        self.offline = OfflineEngine(registry)
        self.history: list[dict] = []
        self.session_results: list[dict] = []     # every tool result gathered this session

    def reset(self) -> None:
        self.history.clear()
        self.session_results.clear()
        self.offline.last_intents = []
        self.registry.clear_cache()

    # ------------------------------------------------------------------ public
    def ask(self, question: str) -> Answer:
        question = question.strip()
        if not question:
            return Answer("Please type a question about your hardware.")
        if self.llm:
            try:
                return self._ask_llm(question)
            except LLMError as exc:
                ans = self._ask_offline(question)
                ans.text += f"\n(LLM unavailable: {exc}; answered by the offline engine.)"
                return ans
        return self._ask_offline(question)

    # ------------------------------------------------------------------ offline
    def _ask_offline(self, question: str) -> Answer:
        def hook(name: str, result: dict) -> None:
            self.session_results.append(result)
            if self.on_tool:
                self.on_tool(name, {}, result)

        text, used = self.offline.answer(question, hook)
        self.history += [{"role": "user", "content": question}, {"role": "assistant", "content": text}]
        return Answer(text, used, "offline")

    # ------------------------------------------------------------------ LLM
    def _trimmed_history(self) -> list[dict]:
        hist = self.history[-MAX_HISTORY_MESSAGES:]
        while hist and hist[0]["role"] != "user":       # never start mid tool-exchange
            hist = hist[1:]
        return hist

    def _ask_llm(self, question: str) -> Answer:
        assert self.llm is not None
        turn: list[dict] = [{"role": "user", "content": question}]
        used, turn_results = [], []
        tools = self.registry.specs()
        for _ in range(MAX_TOOL_ROUNDS):
            messages = [{"role": "system", "content": SYSTEM_PROMPT}, *self._trimmed_history(), *turn]
            reply = self.llm.chat(messages, tools)
            if not reply.tool_calls:
                text = reply.content.strip() or "I couldn't produce an answer."
                bad = ungrounded_numbers(text, self.session_results + turn_results)
                if bad:
                    text += ("\n(Note: I couldn't verify these figures against your system data: "
                             + ", ".join(bad) + ". Treat them with caution.)")
                turn.append({"role": "assistant", "content": text})
                self.history += turn
                self.session_results += turn_results
                return Answer(text, used, f"llm:{self.llm.model}", bad)

            turn.append({"role": "assistant", "content": reply.content or "",
                         "tool_calls": [{"id": c.id, "type": "function",
                                         "function": {"name": c.name, "arguments": json.dumps(c.arguments)}}
                                        for c in reply.tool_calls]})
            for call in reply.tool_calls:
                result = self.registry.call(call.name, call.arguments)
                if call.name not in used:
                    used.append(call.name)
                turn_results.append(result)
                if self.on_tool:
                    self.on_tool(call.name, call.arguments, result)
                turn.append({"role": "tool", "tool_call_id": call.id,
                             "content": json.dumps(result, ensure_ascii=False)})
        # The model kept calling tools: answer deterministically instead of looping forever.
        self.session_results += turn_results
        return self._ask_offline(question)
