from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any


@dataclass
class ImageRef:
    image_id: str
    source_path: str
    record_id: str = ""
    alt: str = ""
    anchor: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "image_id": self.image_id,
            "source_path": self.source_path,
            "alt": self.alt,
            "anchor": self.anchor,
            "url": f"/api/image/{self.image_id}",
        }


@dataclass
class Question:
    qid: str
    record_id: str
    number: str
    stem: str
    options: dict[str, str]
    answer: str
    explanation: str
    title: str
    year: str
    region: str
    subject: str
    question_type: str
    source_path: str
    source_rel: str
    raw_text: str
    images: list[ImageRef] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "qid": self.qid,
            "record_id": self.record_id,
            "number": self.number,
            "stem": self.stem,
            "options": self.options,
            "answer": self.answer,
            "explanation": self.explanation,
            "title": self.title,
            "year": self.year,
            "region": self.region,
            "subject": self.subject,
            "question_type": self.question_type,
            "source_path": self.source_path,
            "source_rel": self.source_rel,
            "images": [image.to_dict() for image in self.images],
        }


@dataclass
class Chunk:
    chunk_id: str
    qid: str
    record_id: str
    chunk_type: str
    text: str
    title: str
    year: str
    region: str
    subject: str
    question_type: str
    source_path: str
    source_rel: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "chunk_id": self.chunk_id,
            "qid": self.qid,
            "record_id": self.record_id,
            "chunk_type": self.chunk_type,
            "text": self.text,
            "title": self.title,
            "year": self.year,
            "region": self.region,
            "subject": self.subject,
            "question_type": self.question_type,
            "source_path": self.source_path,
            "source_rel": self.source_rel,
        }


@dataclass
class SearchHit:
    score: float
    chunk: Chunk
    question: Question | None
    images: list[ImageRef]

    def to_dict(self) -> dict[str, Any]:
        return {
            "score": round(self.score, 4),
            "chunk": self.chunk.to_dict(),
            "question": self.question.to_dict() if self.question else None,
            "images": [image.to_dict() for image in self.images],
        }


def json_dumps(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"))
