from __future__ import annotations

import re
from typing import Any

from .config import Settings
from .embeddings import EmbeddingError, get_embedding_provider
from .index import IndexStore
from .llm import LLMError, OpenAICompatibleClient, build_messages, load_system_prompt
from .models import SearchHit
from .vector_store import VectorStoreError, get_vector_store


def _clip(text: str, limit: int) -> str:
    text = re.sub(r"\s+", " ", (text or "").strip())
    if len(text) <= limit:
        return text
    return text[: max(0, limit - 3)].rstrip() + "..."


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


def _source_label(hit: SearchHit) -> str:
    chunk = hit.chunk
    if chunk.page_start:
        page = (
            f"第 {chunk.page_start} 页"
            if chunk.page_start == chunk.page_end
            else f"第 {chunk.page_start}-{chunk.page_end} 页"
        )
        return f"{chunk.source_rel} ({page})"
    return chunk.source_rel


def render_context(hits: list[SearchHit], max_chars: int = 18000) -> str:
    blocks: list[str] = []
    total = 0
    for index, hit in enumerate(hits, start=1):
        question = hit.question
        chunk = hit.chunk
        parts = [
            f"[证据{index}]",
            f"类型：{chunk.chunk_type}",
            f"来源：{_source_label(hit)}",
            f"标题：{chunk.title}",
            f"文档类型：{chunk.doc_type or chunk.subject}",
            f"年份：{chunk.year}",
            f"科目：{chunk.subject}",
            f"检索方式：{hit.retrieval}",
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
            parts.append(f"内容：{chunk.text}")
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
            "根据当前题库和资料检索到以下证据。"
            "未配置或调用失败时，系统只展示原文摘录，不生成额外结论。"
        )
        for index, hit in enumerate(hits, start=1):
            text = _question_for_hit(hit)
            lines.append(f"\n### 证据 {index}\n{_clip(text, 600)}")
            lines.append(f"来源：`{_source_label(hit)}`")
    return "\n".join(lines).strip()


def _matches_filters(hit: SearchHit, filters: dict[str, Any]) -> bool:
    for key, value in filters.items():
        if value in (None, ""):
            continue
        actual = getattr(hit.chunk, key, "")
        if str(value).lower() not in str(actual).lower():
            return False
    return True


class RAGService:
    def __init__(self, settings: Settings):
        self.settings = settings
        self.store = IndexStore(settings.db_path)
        self.store.initialize()
        self.llm = OpenAICompatibleClient(settings)
        self.embedding_provider = get_embedding_provider(settings)
        self.vector_store = get_vector_store(settings.vector_dir)
        self.system_prompt = load_system_prompt(settings.root)

    def retrieve_hits(
        self,
        query: str,
        mode: str = "search",
        top_k: int | None = None,
        filters: dict[str, Any] | None = None,
    ) -> tuple[list[SearchHit], str]:
        requested = max(1, top_k or self.settings.default_top_k)
        filters = filters or {}
        candidate_limit = max(requested * 8, 40)
        bm25_hits = self.store.search(
            query,
            limit=candidate_limit,
            filters=filters,
            mode=mode,
        )

        vector_error = ""
        vector_hits: list[SearchHit] = []
        if self.embedding_provider:
            try:
                if not self.vector_store.ready:
                    return (
                        [
                            self._with_bm25_rank(hit, rank)
                            for rank, hit in enumerate(bm25_hits, start=1)
                        ][:requested],
                        vector_error,
                    )
                query_vector = self.embedding_provider.embed([query])[0]
                matches = self.vector_store.search(query_vector, candidate_limit)
                vector_id_map = self.store.chunk_ids_for_vector_ids(
                    [match.vector_id for match in matches]
                )
                ordered_matches = []
                for match in matches:
                    chunk_id = vector_id_map.get(match.vector_id)
                    if chunk_id:
                        ordered_matches.append((chunk_id, match.score))
                ids = [chunk_id for chunk_id, _ in ordered_matches]
                scores = {
                    chunk_id: score
                    for chunk_id, score in ordered_matches
                }
                looks = self.store.get_hits_by_chunk_ids(ids, scores=scores)
                vector_hits = [
                    hit for hit in looks if _matches_filters(hit, filters)
                ]
            except (EmbeddingError, VectorStoreError, ValueError, OSError) as exc:
                vector_error = str(exc)

        if not vector_hits:
            return (
                [
                    self._with_bm25_rank(hit, rank)
                    for rank, hit in enumerate(bm25_hits, start=1)
                ][:requested],
                vector_error,
            )

        merged: dict[str, SearchHit] = {}
        for rank, hit in enumerate(bm25_hits, start=1):
            hit.retrieval = {
                "bm25_rank": rank,
                "bm25_score": round(hit.score, 4),
            }
            merged[hit.chunk.chunk_id] = hit
        for rank, hit in enumerate(vector_hits, start=1):
            hit.retrieval = {
                "vector_rank": rank,
                "vector_score": round(hit.score, 4),
            }
            existing = merged.get(hit.chunk.chunk_id)
            if existing:
                existing.retrieval.update(hit.retrieval)
            else:
                merged[hit.chunk.chunk_id] = hit

        rrf_k = max(1, self.settings.rrf_k)
        for hit in merged.values():
            score = 0.0
            bm25_rank = hit.retrieval.get("bm25_rank")
            vector_rank = hit.retrieval.get("vector_rank")
            if bm25_rank:
                score += self.settings.bm25_weight / (rrf_k + bm25_rank)
            if vector_rank:
                score += self.settings.vector_weight / (rrf_k + vector_rank)
            hit.score = score

        ranked = sorted(
            merged.values(),
            key=lambda item: (-item.score, item.chunk.chunk_id),
        )
        return ranked[:requested], vector_error

    def _with_bm25_rank(self, hit: SearchHit, rank: int) -> SearchHit:
        hit.retrieval = {
            "bm25_rank": rank,
            "bm25_score": round(hit.score, 4),
        }
        return hit

    def search(
        self,
        query: str,
        mode: str = "search",
        top_k: int | None = None,
        filters: dict[str, Any] | None = None,
    ) -> list[dict[str, Any]]:
        hits, _ = self.retrieve_hits(
            query,
            mode=mode,
            top_k=top_k,
            filters=filters,
        )
        return [self._public_hit(hit) for hit in hits]

    def answer(
        self,
        question: str,
        mode: str = "knowledge",
        top_k: int | None = None,
        filters: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        hits, vector_error = self.retrieve_hits(
            question,
            mode=mode,
            top_k=top_k,
            filters=filters,
        )
        citations = self._citations(hits)
        if not hits:
            return {
                "answer": extractive_answer(hits, mode),
                "citations": [],
                "hits": [],
                "llm_used": False,
                "llm_error": "",
                "vector_error": vector_error,
                "retrieval_mode": "bm25",
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

        retrieval_mode = (
            "hybrid"
            if any("vector_rank" in hit.retrieval for hit in hits)
            else "bm25"
        )
        return {
            "answer": answer,
            "citations": citations,
            "hits": [self._public_hit(hit) for hit in hits],
            "llm_used": llm_used,
            "llm_error": llm_error,
            "vector_error": vector_error,
            "retrieval_mode": retrieval_mode,
            "confidence": "high" if len(hits) >= 2 else "medium",
        }

    def _citations(self, hits: list[SearchHit]) -> list[dict[str, Any]]:
        result: list[dict[str, Any]] = []
        seen: set[str] = set()
        for hit in hits:
            if hit.chunk.chunk_id in seen:
                continue
            seen.add(hit.chunk.chunk_id)
            result.append(
                {
                    "chunk_id": hit.chunk.chunk_id,
                    "qid": hit.chunk.qid,
                    "title": hit.chunk.title,
                    "source_rel": hit.chunk.source_rel,
                    "source_path": hit.chunk.source_path,
                    "page_start": hit.chunk.page_start,
                    "page_end": hit.chunk.page_end,
                }
            )
        return result

    def _public_hit(self, hit: SearchHit) -> dict[str, Any]:
        payload = hit.to_dict()
        if hit.question:
            payload["question"]["explanation"] = _clip(
                hit.question.explanation,
                3000,
            )
        return payload
