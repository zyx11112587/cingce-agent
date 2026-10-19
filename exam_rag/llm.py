from __future__ import annotations

import json
import urllib.error
import urllib.request
from typing import Any

from .config import Settings


class LLMError(RuntimeError):
    pass


class OpenAICompatibleClient:
    """Minimal OpenAI-compatible chat client using only the standard library."""

    def __init__(self, settings: Settings):
        self.settings = settings

    @property
    def enabled(self) -> bool:
        return self.settings.llm_enabled

    def _endpoint(self) -> str:
        base = self.settings.llm_base_url.rstrip("/")
        if base.endswith("/chat/completions"):
            return base
        if base.endswith("/v1"):
            return f"{base}/chat/completions"
        return f"{base}/v1/chat/completions"

    def chat(
        self,
        messages: list[dict[str, str]],
        temperature: float = 0.1,
        max_tokens: int = 1800,
    ) -> str:
        if not self.enabled:
            raise LLMError("LLM API is not configured")

        payload = {
            "model": self.settings.llm_model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
        }
        headers = {
            "Content-Type": "application/json",
            "Accept": "application/json",
        }
        if self.settings.llm_api_key:
            headers["Authorization"] = f"Bearer {self.settings.llm_api_key}"

        request = urllib.request.Request(
            self._endpoint(),
            data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
            headers=headers,
            method="POST",
        )
        try:
            with urllib.request.urlopen(
                request,
                timeout=self.settings.llm_timeout,
            ) as response:
                data = json.loads(response.read().decode("utf-8"))
        except urllib.error.HTTPError as exc:
            body = exc.read().decode("utf-8", errors="replace")
            raise LLMError(f"LLM HTTP {exc.code}: {body[:500]}") from exc
        except (urllib.error.URLError, TimeoutError, json.JSONDecodeError) as exc:
            raise LLMError(f"LLM request failed: {exc}") from exc

        choices = data.get("choices") or []
        if not choices:
            raise LLMError(f"LLM response has no choices: {data}")
        message = choices[0].get("message") or {}
        content = message.get("content", "")
        if isinstance(content, list):
            content = "".join(
                item.get("text", "") if isinstance(item, dict) else str(item)
                for item in content
            )
        if not str(content).strip():
            raise LLMError("LLM returned empty content")
        return str(content).strip()


def build_messages(
    system_prompt: str,
    context: str,
    question: str,
    task_type: str,
) -> list[dict[str, str]]:
    user_prompt = (
        "【检索证据】\n"
        f"{context}\n\n"
        "【用户问题】\n"
        f"{question}\n\n"
        "【任务类型】\n"
        f"{task_type}"
    )
    return [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]


def load_system_prompt(root) -> str:
    prompt_path = root / "prompts" / "qa_prompt.txt"
    if prompt_path.exists():
        return prompt_path.read_text(encoding="utf-8").strip()
    return (
        "你是考公真题、政治理论与申论资料问答助手。只能依据用户提供的检索证据回答。"
        "证据不足时明确说明，不得编造题目、答案、年份、地区、法条或解析。"
        "回答时标注证据来源和页码。"
    )
