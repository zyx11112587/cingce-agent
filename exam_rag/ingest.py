from __future__ import annotations

from collections import OrderedDict
from pathlib import Path
from typing import Callable, Iterable

from .config import Settings
from .document_loader import iter_document_files, load_document
from .embeddings import EmbeddingError, get_embedding_provider
from .index import IndexStore
from .models import Chunk, ImageRef, Question
from .parser import (
    explanation_chunk,
    parse_source_file,
    question_chunk,
)
from .vector_store import get_vector_store


def _dedupe(items, key):
    result = OrderedDict()
    for item in items:
        result[key(item)] = item
    return list(result.values())


def _iter_inputs(input_dirs: Iterable[Path]):
    seen: set[Path] = set()
    seen_files: set[Path] = set()
    for input_dir in input_dirs:
        input_dir = input_dir.resolve()
        if input_dir in seen or not input_dir.exists():
            continue
        seen.add(input_dir)
        for path in iter_document_files(input_dir):
            resolved = path.resolve()
            if resolved in seen_files:
                continue
            seen_files.add(resolved)
            yield input_dir, path


def ingest_directories(
    input_dirs: Iterable[Path],
    db_path: Path,
    settings: Settings | None = None,
    skip_vectors: bool = False,
    progress: Callable[[int, int, int, int], None] | None = None,
) -> dict[str, object]:
    settings = settings or Settings.from_env()
    input_dirs = tuple(Path(path).resolve() for path in input_dirs)
    files = list(_iter_inputs(input_dirs))
    if not files:
        raise FileNotFoundError(
            "No supported files found. "
            "Put PDF/DOCX/Markdown/TXT under one of the configured input directories."
        )

    questions: list[Question] = []
    chunks: list[Chunk] = []
    images: list[ImageRef] = []
    for index, (input_dir, path) in enumerate(files, start=1):
        suffix = path.suffix.lower()
        if suffix in {".md", ".markdown"}:
            parsed_questions, generic_chunks = parse_source_file(path, input_dir)
            questions.extend(parsed_questions)
            chunks.extend(generic_chunks)
            for question in parsed_questions:
                chunks.append(question_chunk(question))
                chunks.append(explanation_chunk(question))
                images.extend(question.images)
            if not parsed_questions and not generic_chunks:
                chunks.extend(load_document(path, input_dir))
        else:
            chunks.extend(load_document(path, input_dir))
        if progress and (index == 1 or index % 25 == 0 or index == len(files)):
            progress(index, len(files), len(questions), len(chunks))

    questions = _dedupe(questions, lambda item: item.record_id)
    chunks = _dedupe(chunks, lambda item: item.chunk_id)
    images = _dedupe(images, lambda item: (item.record_id, item.image_id))

    store = IndexStore(db_path)
    store.reset()
    store.replace_all(questions, chunks, images)

    vector_count = 0
    vector_backend = "disabled"
    vector_error = ""
    vector_store = get_vector_store(settings.vector_dir)
    if skip_vectors:
        vector_store.clear()
    else:
        provider = get_embedding_provider(settings)
        if provider is None:
            vector_store.clear()
            vector_backend = "not-configured"
        else:
            vector_store.clear()
            try:
                vectors = provider.embed(
                    [chunk.text for chunk in chunks],
                    progress=(
                        lambda done, total: print(
                            f"[vector] {done}/{total} chunks embedded",
                            flush=True,
                        )
                    ),
                )
                if len(vectors) != len(chunks):
                    raise ValueError(
                        f"Embedding count mismatch: {len(vectors)} vectors "
                        f"for {len(chunks)} chunks"
                    )
                store.set_vector_mapping([chunk.chunk_id for chunk in chunks])
                vector_count = vector_store.build(vectors)
                vector_backend = getattr(vector_store, "backend", "unknown")
            except (EmbeddingError, ValueError, OSError) as exc:
                vector_error = str(exc)
                store.set_vector_mapping([])
                vector_store.clear()
                vector_backend = "error"

    return {
        "files": len(files),
        "questions": len(questions),
        "chunks": len(chunks),
        "images": len(images),
        "vectors": vector_count,
        "vector_backend": vector_backend,
        "vector_error": vector_error,
    }


def ingest_directory(
    input_dir: Path,
    db_path: Path,
    progress: Callable[[int, int, int, int], None] | None = None,
    settings: Settings | None = None,
    skip_vectors: bool = False,
) -> dict[str, object]:
    return ingest_directories(
        [input_dir],
        db_path,
        settings=settings,
        skip_vectors=skip_vectors,
        progress=progress,
    )
