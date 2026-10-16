from __future__ import annotations

import re
from typing import Any

from .config import Settings
from .index import IndexStore
from .llm import LLMError, OpenAICompatibleClient, build_messages, load_system_prompt
from .models import SearchHit


def _clip(text: str, limit: int) -> str:
    text = re.sub(r"\s+", " ", (text or "").strip())
    if len(text) <= limit:
        return text
    return text[: max(0, limit - 1)].rstrip() + "…"


def _question_for_hit(hit: SearchHit) -> str:
    if not hit.question:
        return hit.chunk.text
    options = "\n".join(
        f"{letter}. {value}"
        for letter, value in sorted(hit.question.options.items())
    )
    return "\n".join(
        part
        for part in [
            hit.question.stem,
            options,
            f"答案：{hit.question.answer}" if hit.question.answer else "",
            hit.question.explanation,
        ]
        if part
    )


def render_context(hits: list[SearchHit], max_chars: int = 18000) -> str:
    blocks: list[str] = []
    total = 0
    for index, hit in enumerate(hits, start=1):
        question = hit.question
        parts = [
            f"[证据{index}]",
            f"类型：{hit.chunk.chunk_type}",
            f"来源：{hit.chunk.source_rel}",
            f"标题：{hit.chunk.title}",
            f"年份：{hit.chunk.year}",
            f"科目：{hit.chunk.subject}",
        ]
        if question:
            parts.extend(
                [
                    f"题目：{question.stem}",
                    "选项：" + "；".join(
                        f"{letter}. {text}"
                        for letter, text in sorted(question.options.items())
                    ),
                    f"答案：{question.answer}",
                    f"解析：{question.explanation}",
                ]
            )
            if question.images:
                parts.append(
                    "图片：" + "；".join(
                        f"{image.alt or image.image_id} ({image.source_path})"
                        for image in question.images
                    )
                )
        else:
            parts.append(f"内容：{hit.chunk.text}")
        block = "\n".join(part for part in parts if part)
        if total + len(block) > max_chars:
            break
        blocks.append(block)
        total += len(block)
    return "\n\n".join(blocks)


def extractive_answer(hits: list[SearchHit], mode: str) -> str:
    if not hits:
        return "当前资料中未检索到足够依据。"
    lines: list[str] = []
    if mode == "explain":
        top = hits[0]
        if top.question:
            lines.extend(
                [
                    "## 题目",
                    top.question.stem or "未识别到题干",
                    "",
                    "## 选项",
                    "\n".join(
                        f"- {letter}. {text}"
                        for letter, text in sorted(top.question.options.items())
                    )
                    or "未识别到选项",
                    "",
                    "## 答案",
                    top.question.answer or "未识别到答案",
                    "",
                    "## 解析",
                    top.question.explanation or "未识别到解析",
                ]
            )
        else:
            lines.append(top.chunk.text)
    else:
        lines.append(
            "根据当前题库检索到以下相关证据。未配置或调用失败时，"
            "系统只展示原文摘录，不生成额外结论。"
        )
        for index, hit in enumerate(hits, start=1):
            text = _question_for_hit(hit)
            lines.append(f"\n### 证据 {index}\n{_clip(text, 600)}")
            lines.append(f"来源：`{hit.chunk.source_rel}`")
    return "\n".join(lines).strip()


class RAGService:
    def __init__(self, settings: Settings):
        self.settings = settings
        self.store = IndexStore(settings.db_path)
        self.store.initialize()
        self.llm = OpenAICompatibleClient(settings)
        self.system_prompt = load_system_prompt(settings.root)

    def search(
        self,
        query: str,
        mode: str = "search",
        top_k: int | None = None,
        filters: dict[str, Any] | None = None,
    ) -> list[dict[str, Any]]:
        hits = self.store.search(
            query,
            limit=top_k or self.settings.default_top_k,
            filters=filters,
            mode=mode,
        )
        return [self._public_hit(hit) for hit in hits]

    def answer(
        self,
        question: str,
        mode: str = "knowledge",
        top_k: int | None = None,
        filters: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        hits = self.store.search(
            question,
            limit=top_k or self.settings.default_top_k,
            filters=filters,
            mode=mode,
        )
        citations = self._citations(hits)
        if not hits:
            return {
                "answer": extractive_answer(hits, mode),
                "citations": [],
                "hits": [],
                "llm_used": False,
                "llm_error": "",
                "confidence": "low",
            }

        answer = ""
        llm_error = ""
        llm_used = False
        if self.llm.enabled:
            try:
                answer = self.llm.chat(
                    build_messages(
                        self.system_prompt,
                        render_context(hits),
                        question,
                        mode,
                    )
                )
                llm_used = True
            except LLMError as exc:
                llm_error = str(exc)
        if not answer:
            answer = extractive_answer(hits, mode)

        return {
            "answer": answer,
            "citations": citations,
            "hits": [self._public_hit(hit) for hit in hits],
            "llm_used": llm_used,
            "llm_error": llm_error,
            "confidence": "high" if len(hits) >= 2 else "medium",
        }

    def _citations(self, hits: list[SearchHit]) -> list[dict[str, Any]]:
        result: list[dict[str, Any]] = []
        seen: set[str] = set()
        for hit in hits:
            key = hit.chunk.chunk_id
            if key in seen:
                continue
            seen.add(key)
            result.append(
                {
                    "chunk_id": hit.chunk.chunk_id,
                    "qid": hit.chunk.qid,
                    "title": hit.chunk.title,
                    "source_rel": hit.chunk.source_rel,
                    "source_path": hit.chunk.source_path,
                }
            )
        return result

    def _public_hit(self, hit: SearchHit) -> dict[str, Any]:
        payload = hit.to_dict()
        payload["chunk"]["text"] = hit.chunk.text
        if hit.question:
            payload["question"]["explanation"] = _clip(
                hit.question.explanation,
                3000,
            )
        return payload
