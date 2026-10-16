from __future__ import annotations

from collections import OrderedDict
from pathlib import Path

from .index import IndexStore
from .models import Chunk, ImageRef, Question
from .parser import (
    explanation_chunk,
    iter_markdown_files,
    parse_source_file,
    question_chunk,
)


def _dedupe(items, key):
    result = OrderedDict()
    for item in items:
        result[key(item)] = item
    return list(result.values())


def ingest_directory(
    input_dir: Path,
    db_path: Path,
    progress=None,
) -> dict[str, int]:
    input_dir = input_dir.resolve()
    if not input_dir.exists():
        raise FileNotFoundError(f"Input directory does not exist: {input_dir}")

    questions: list[Question] = []
    chunks: list[Chunk] = []
    images: list[ImageRef] = []
    files = list(iter_markdown_files(input_dir))
    for index, path in enumerate(files, start=1):
        parsed_questions, generic_chunks = parse_source_file(path, input_dir)
        questions.extend(parsed_questions)
        chunks.extend(generic_chunks)
        for question in parsed_questions:
            chunks.append(question_chunk(question))
            chunks.append(explanation_chunk(question))
            images.extend(question.images)
        if progress and (index == 1 or index % 25 == 0 or index == len(files)):
            progress(index, len(files), len(questions), len(chunks))

    questions = _dedupe(questions, lambda item: item.record_id)
    chunks = _dedupe(chunks, lambda item: item.chunk_id)
    images = _dedupe(images, lambda item: (item.record_id, item.image_id))

    store = IndexStore(db_path)
    store.reset()
    store.replace_all(questions, chunks, images)
    return {
        "files": len(files),
        "questions": len(questions),
        "chunks": len(chunks),
        "images": len(images),
    }
