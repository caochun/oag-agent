from __future__ import annotations

import json
import logging
import time

from openai import OpenAI

log = logging.getLogger(__name__)


def _repair_json(text: str) -> str:
    """Attempt to repair truncated JSON by closing open structures."""
    stack: list[str] = []
    in_string = False
    escape_next = False

    for ch in text:
        if escape_next:
            escape_next = False
            continue
        if ch == "\\":
            if in_string:
                escape_next = True
            continue
        if ch == '"' and not escape_next:
            in_string = not in_string
            continue
        if in_string:
            continue
        if ch in ("{", "["):
            stack.append(ch)
        elif ch == "}" and stack and stack[-1] == "{":
            stack.pop()
        elif ch == "]" and stack and stack[-1] == "[":
            stack.pop()

    if in_string:
        text += '"'

    closers = {"[": "]", "{": "}"}
    for opener in reversed(stack):
        text += closers[opener]

    return text


class DistillerLLM:
    """OpenAI-compatible LLM client with usage tracking and retry."""

    def __init__(self, config: dict):
        self.client = OpenAI(
            api_key=config.get("api_key", "sk-placeholder"),
            base_url=config.get("api_url", "http://localhost:8090/v1"),
        )
        self.model = config.get("model", "qwen3.5-plus")
        self.usage = {"prompt_tokens": 0, "completion_tokens": 0, "calls": 0}
        self._call_log: list[dict] = []

    def call(
        self,
        system: str,
        user: str,
        *,
        json_mode: bool = False,
        max_retries: int = 3,
        temperature: float = 0.1,
    ) -> str:
        kwargs: dict = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            "temperature": temperature,
        }
        if json_mode:
            kwargs["response_format"] = {"type": "json_object"}

        for attempt in range(max_retries):
            try:
                t0 = time.time()
                resp = self.client.chat.completions.create(**kwargs)
                elapsed = time.time() - t0

                choice = resp.choices[0]
                text = choice.message.content or ""

                if resp.usage:
                    self.usage["prompt_tokens"] += resp.usage.prompt_tokens
                    self.usage["completion_tokens"] += resp.usage.completion_tokens
                self.usage["calls"] += 1

                self._call_log.append({
                    "prompt_chars": len(system) + len(user),
                    "completion_chars": len(text),
                    "elapsed_s": round(elapsed, 1),
                    "prompt_tokens": resp.usage.prompt_tokens if resp.usage else 0,
                    "completion_tokens": resp.usage.completion_tokens if resp.usage else 0,
                })

                log.info(
                    "LLM call #%d: %d prompt chars → %d completion chars (%.1fs)",
                    self.usage["calls"],
                    len(system) + len(user),
                    len(text),
                    elapsed,
                )
                return text

            except Exception as e:
                if attempt < max_retries - 1:
                    wait = 2 ** (attempt + 1)
                    log.warning("LLM call failed (%s), retrying in %ds...", e, wait)
                    time.sleep(wait)
                else:
                    raise

        raise RuntimeError("LLM call failed after all retries")

    def call_json(self, system: str, user: str, **kwargs) -> dict:
        text = self.call(system, user, json_mode=True, **kwargs)
        text = text.strip()
        if text.startswith("```"):
            text = text.split("\n", 1)[1] if "\n" in text else text[3:]
            if text.endswith("```"):
                text = text[:-3]
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            repaired = _repair_json(text)
            return json.loads(repaired)

    def usage_summary(self) -> str:
        u = self.usage
        return (
            f"LLM usage: {u['calls']} calls, "
            f"{u['prompt_tokens']:,} prompt tokens, "
            f"{u['completion_tokens']:,} completion tokens"
        )

    @property
    def call_log(self) -> list[dict]:
        return list(self._call_log)
