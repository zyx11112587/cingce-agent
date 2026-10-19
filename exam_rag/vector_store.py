from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Sequence


@dataclass
class VectorMatch:
    vector_id: int
    score: float


class VectorStoreError(RuntimeError):
    pass


def _normalise(vectors):
    import numpy as np

    array = np.asarray(vectors, dtype="float32")
    if array.ndim == 1:
        array = array.reshape(1, -1)
    norms = np.linalg.norm(array, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return array / norms


class NumpyVectorStore:
    backend = "numpy"

    def __init__(self, directory: Path):
        self.directory = directory
        self.path = directory / "index.npy"
        self._vectors = None

    def clear(self) -> None:
        for name in ("index.npy", "index.faiss"):
            path = self.directory / name
            if path.exists():
                path.unlink()
        self._vectors = None

    def build(self, vectors: Sequence[Sequence[float]]) -> int:
        if not vectors:
            self.clear()
            return 0
        import numpy as np

        self.directory.mkdir(parents=True, exist_ok=True)
        array = _normalise(vectors)
        np.save(self.path, array)
        self._vectors = array
        return int(array.shape[0])

    def _load(self):
        if self._vectors is not None:
            return self._vectors
        if not self.path.exists():
            return None
        import numpy as np

        self._vectors = np.load(self.path, allow_pickle=False)
        return self._vectors

    @property
    def count(self) -> int:
        vectors = self._load()
        return int(vectors.shape[0]) if vectors is not None else 0

    @property
    def ready(self) -> bool:
        return self.count > 0

    def search(
        self,
        query_vector: Sequence[float],
        top_k: int,
    ) -> list[VectorMatch]:
        import numpy as np

        vectors = self._load()
        if vectors is None or len(vectors) == 0:
            return []
        query = _normalise([query_vector])[0]
        if query.shape[0] != vectors.shape[1]:
            raise VectorStoreError(
                f"Embedding dimension mismatch: query={query.shape[0]}, "
                f"index={vectors.shape[1]}"
            )
        scores = vectors @ query
        count = min(max(1, top_k), len(scores))
        ids = np.argpartition(scores, -count)[-count:]
        ids = ids[np.argsort(scores[ids])[::-1]]
        return [
            VectorMatch(vector_id=int(vector_id), score=float(scores[vector_id]))
            for vector_id in ids
        ]


class FaissVectorStore:
    backend = "faiss"

    def __init__(self, directory: Path):
        self.directory = directory
        self.path = directory / "index.faiss"
        self._index = None

    def clear(self) -> None:
        for name in ("index.npy", "index.faiss"):
            path = self.directory / name
            if path.exists():
                path.unlink()
        self._index = None

    def build(self, vectors: Sequence[Sequence[float]]) -> int:
        if not vectors:
            self.clear()
            return 0
        import faiss

        self.directory.mkdir(parents=True, exist_ok=True)
        array = _normalise(vectors)
        index = faiss.IndexFlatIP(array.shape[1])
        index.add(array)
        faiss.write_index(index, str(self.path))
        self._index = index
        return int(index.ntotal)

    def _load(self):
        if self._index is not None:
            return self._index
        if not self.path.exists():
            return None
        import faiss

        self._index = faiss.read_index(str(self.path))
        return self._index

    @property
    def count(self) -> int:
        index = self._load()
        return int(index.ntotal) if index is not None else 0

    @property
    def ready(self) -> bool:
        return self.count > 0

    def search(
        self,
        query_vector: Sequence[float],
        top_k: int,
    ) -> list[VectorMatch]:
        import numpy as np

        index = self._load()
        if index is None or index.ntotal == 0:
            return []
        query = _normalise([query_vector])
        if query.shape[1] != index.d:
            raise VectorStoreError(
                f"Embedding dimension mismatch: query={query.shape[1]}, index={index.d}"
            )
        scores, ids = index.search(query, min(max(1, top_k), index.ntotal))
        return [
            VectorMatch(vector_id=int(vector_id), score=float(score))
            for vector_id, score in zip(ids[0], scores[0])
            if vector_id >= 0
        ]


def get_vector_store(directory: Path):
    directory = directory.resolve()
    try:
        import faiss  # noqa: F401

        return FaissVectorStore(directory)
    except ImportError:
        return NumpyVectorStore(directory)
