from __future__ import annotations

import json
import urllib.error
import urllib.request
from typing import Callable, Sequence

from .config import Settings


class EmbeddingError(RuntimeError):
    pass


ProgressCallback = Callable[[int, int], None]


class EmbeddingProvider:
    name = "unknown"

    def embed(
        self,
        texts: Sequence[str],
        progress: ProgressCallback | None = None,
    ) -> list[list[float]]:
        raise NotImplementedError


class OpenAIEmbeddingProvider(EmbeddingProvider):
    name = "openai-compatible"

    def __init__(self, settings: Settings):
        self.settings = settings
        self.batch_size = max(1, settings.embedding_batch_size)

    def _endpoint(self) -> str:
        base = self.settings.embedding_base_url.rstrip("/")
        if base.endswith("/embeddings"):
            return base
        if base.endswith("/v1"):
            return f"{base}/embeddings"
        return f"{base}/v1/embeddings"

    def embed(
        self,
        texts: Sequence[str],
        progress: ProgressCallback | None = None,
    ) -> list[list[float]]:
        if not self.settings.embedding_model:
            raise EmbeddingError("EXAM_RAG_EMBEDDING_MODEL is not configured")
        vectors: list[list[float]] = []
        total = len(texts)
        for start in range(0, total, self.batch_size):
            batch = list(texts[start:start + self.batch_size])
            payload = {
                "model": self.settings.embedding_model,
                "input": batch,
            }
            headers = {
                "Content-Type": "application/json",
                "Accept": "application/json",
            }
            if self.settings.embedding_api_key:
                headers["Authorization"] = f"Bearer {self.settings.embedding_api_key}"
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
                    result = json.loads(response.read().decode("utf-8"))
            except urllib.error.HTTPError as exc:
                body = exc.read().decode("utf-8", errors="replace")
                raise EmbeddingError(
                    f"Embedding HTTP {exc.code}: {body[:500]}"
                ) from exc
            except (urllib.error.URLError, TimeoutError, json.JSONDecodeError) as exc:
                raise EmbeddingError(f"Embedding request failed: {exc}") from exc

            data = result.get("data") or []
            if len(data) != len(batch):
                raise EmbeddingError(
                    f"Embedding API returned {len(data)} vectors for {len(batch)} texts"
                )
            ordered = sorted(data, key=lambda item: item.get("index", 0))
            vectors.extend(
                [float(value) for value in item.get("embedding", [])]
                for item in ordered
            )
            if progress:
                progress(min(start + len(batch), total), total)
        return vectors


class SentenceTransformerProvider(EmbeddingProvider):
    name = "sentence-transformers"

    def __init__(self, settings: Settings):
        self.settings = settings
        self._model = None
        self.batch_size = max(1, settings.embedding_batch_size)

    def _get_model(self):
        if self._model is None:
            try:
                from sentence_transformers import SentenceTransformer
            except ImportError as exc:
                raise EmbeddingError(
                    "Local embeddings require sentence-transformers. "
                    "Install it with: pip install sentence-transformers"
                ) from exc
            self._model = SentenceTransformer(self.settings.embedding_model)
        return self._model

    def embed(
        self,
        texts: Sequence[str],
        progress: ProgressCallback | None = None,
    ) -> list[list[float]]:
        if not self.settings.embedding_model:
            raise EmbeddingError("EXAM_RAG_EMBEDDING_MODEL is not configured")
        model = self._get_model()
        vectors: list[list[float]] = []
        total = len(texts)
        for start in range(0, total, self.batch_size):
            batch = list(texts[start:start + self.batch_size])
            encoded = model.encode(
                batch,
                normalize_embeddings=True,
                show_progress_bar=False,
            )
            vectors.extend(
                [float(value) for value in vector]
                for vector in encoded
            )
            if progress:
                progress(min(start + len(batch), total), total)
        return vectors


def get_embedding_provider(settings: Settings) -> EmbeddingProvider | None:
    provider = settings.embedding_provider.lower()
    if provider == "none":
        return None
    if provider in {"api", "openai", "openai-compatible"}:
        if settings.embedding_base_url and settings.embedding_model:
            return OpenAIEmbeddingProvider(settings)
        return None
    if provider in {"sentence_transformers", "sentence-transformers", "local"}:
        if settings.embedding_model:
            return SentenceTransformerProvider(settings)
        return None
    if provider == "auto":
        if settings.embedding_base_url and settings.embedding_model:
            return OpenAIEmbeddingProvider(settings)
        if settings.embedding_model:
            return SentenceTransformerProvider(settings)
    return None
