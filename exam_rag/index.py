from __future__ import annotations

import json
import re
import sqlite3
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterator

from .models import Chunk, ImageRef, Question, SearchHit


SEARCH_TOKEN_RE = re.compile(r"[\u4e00-\u9fff]{2,}|[A-Za-z0-9_]{2,}")
QID_LOOKUP_RE = re.compile(r"\bqid\s*[:：]?\s*([A-Za-z0-9_-]+)", re.IGNORECASE)
QUERY_STOPWORD_RE = re.compile(
    r"是什么|什么是|如何|怎样|为什么|有哪些|包括|请|简述|谈谈|说明|理解|介绍|的|了|吗|呢|[?？]"
)


def _json_load(value: str, default: Any) -> Any:
    try:
        return json.loads(value or "")
    except (TypeError, ValueError, json.JSONDecodeError):
        return default


def _search_terms(query: str) -> list[str]:
    normalized = QUERY_STOPWORD_RE.sub(" ", query)
    segments = [
        part.strip()
        for part in re.split(r"[\s,，。；;：:、]+", normalized)
        if part.strip()
    ]
    terms: list[str] = []
    for segment in [normalized, *segments]:
        terms.extend(
            match.group(0).strip()
            for match in SEARCH_TOKEN_RE.finditer(segment)
        )
    result: list[str] = []
    for term in terms:
        if len(term) > 80:
            term = term[:80]
        if term and term not in result:
            result.append(term)
    return result


def _safe_fts_query(terms: list[str]) -> str:
    # FTS5 trigram supports substring matching for phrases of three or more
    # characters. Keep each term quoted so punctuation cannot become syntax.
    return " OR ".join(
        '"' + term.replace('"', '""') + '"'
        for term in terms
        if len(term) >= 3
    )


class IndexStore:
    def __init__(self, db_path: Path):
        self.db_path = db_path
        self.db_path.parent.mkdir(parents=True, exist_ok=True)

    @contextmanager
    def _connect(self) -> Iterator[sqlite3.Connection]:
        connection = sqlite3.connect(self.db_path, timeout=30)
        try:
            connection.row_factory = sqlite3.Row
            connection.execute("PRAGMA foreign_keys = ON")
            connection.execute("PRAGMA journal_mode = WAL")
            connection.execute("PRAGMA synchronous = NORMAL")
            yield connection
            connection.commit()
        except Exception:
            connection.rollback()
            raise
        finally:
            connection.close()

    def initialize(self, reset: bool = False) -> None:
        with self._connect() as connection:
            if reset:
                for statement in (
                    "DROP TABLE IF EXISTS search_fts",
                    "DROP TABLE IF EXISTS vector_map",
                    "DROP TABLE IF EXISTS images",
                    "DROP TABLE IF EXISTS chunks",
                    "DROP TABLE IF EXISTS questions",
                ):
                    connection.execute(statement)
            connection.executescript(
                """
                CREATE TABLE IF NOT EXISTS questions (
                    record_id TEXT PRIMARY KEY,
                    qid TEXT NOT NULL DEFAULT '',
                    number TEXT NOT NULL DEFAULT '',
                    stem TEXT NOT NULL DEFAULT '',
                    options_json TEXT NOT NULL DEFAULT '{}',
                    answer TEXT NOT NULL DEFAULT '',
                    explanation TEXT NOT NULL DEFAULT '',
                    title TEXT NOT NULL DEFAULT '',
                    year TEXT NOT NULL DEFAULT '',
                    region TEXT NOT NULL DEFAULT '',
                    subject TEXT NOT NULL DEFAULT '',
                    question_type TEXT NOT NULL DEFAULT '',
                    source_path TEXT NOT NULL DEFAULT '',
                    source_rel TEXT NOT NULL DEFAULT '',
                    raw_text TEXT NOT NULL DEFAULT ''
                );

                CREATE TABLE IF NOT EXISTS chunks (
                    chunk_id TEXT PRIMARY KEY,
                    record_id TEXT NOT NULL DEFAULT '',
                    qid TEXT NOT NULL DEFAULT '',
                    chunk_type TEXT NOT NULL DEFAULT 'knowledge',
                    text TEXT NOT NULL DEFAULT '',
                    title TEXT NOT NULL DEFAULT '',
                    year TEXT NOT NULL DEFAULT '',
                    region TEXT NOT NULL DEFAULT '',
                    subject TEXT NOT NULL DEFAULT '',
                    question_type TEXT NOT NULL DEFAULT '',
                    source_path TEXT NOT NULL DEFAULT '',
                    source_rel TEXT NOT NULL DEFAULT '',
                    source_format TEXT NOT NULL DEFAULT 'md',
                    doc_type TEXT NOT NULL DEFAULT '',
                    page_start INTEGER NOT NULL DEFAULT 0,
                    page_end INTEGER NOT NULL DEFAULT 0
                );

                CREATE INDEX IF NOT EXISTS idx_chunks_qid
                    ON chunks(qid);
                CREATE INDEX IF NOT EXISTS idx_chunks_record_id
                    ON chunks(record_id);
                CREATE INDEX IF NOT EXISTS idx_chunks_filters
                    ON chunks(year, subject, region, chunk_type);

                CREATE TABLE IF NOT EXISTS images (
                    record_id TEXT NOT NULL DEFAULT '',
                    image_id TEXT NOT NULL DEFAULT '',
                    qid TEXT NOT NULL DEFAULT '',
                    source_path TEXT NOT NULL DEFAULT '',
                    alt TEXT NOT NULL DEFAULT '',
                    anchor TEXT NOT NULL DEFAULT '',
                    PRIMARY KEY (record_id, image_id)
                );

                CREATE INDEX IF NOT EXISTS idx_images_qid
                    ON images(qid);

                CREATE TABLE IF NOT EXISTS vector_map (
                    vector_id INTEGER PRIMARY KEY,
                    chunk_id TEXT UNIQUE NOT NULL
                );

                CREATE VIRTUAL TABLE IF NOT EXISTS search_fts USING fts5(
                    chunk_id UNINDEXED,
                    text,
                    tokenize='trigram'
                );
                """
            )
            migrations = {
                "questions": {
                    "record_id": "TEXT NOT NULL DEFAULT ''",
                },
                "chunks": {
                    "record_id": "TEXT NOT NULL DEFAULT ''",
                    "source_format": "TEXT NOT NULL DEFAULT 'md'",
                    "doc_type": "TEXT NOT NULL DEFAULT ''",
                    "page_start": "INTEGER NOT NULL DEFAULT 0",
                    "page_end": "INTEGER NOT NULL DEFAULT 0",
                },
                "images": {
                    "record_id": "TEXT NOT NULL DEFAULT ''",
                },
            }
            for table, columns in migrations.items():
                existing = {
                    row["name"]
                    for row in connection.execute(f"PRAGMA table_info({table})")
                }
                for column, definition in columns.items():
                    if column not in existing:
                        connection.execute(
                            f"ALTER TABLE {table} ADD COLUMN {column} {definition}"
                        )

    def reset(self) -> None:
        self.initialize(reset=True)

    def replace_all(
        self,
        questions: list[Question],
        chunks: list[Chunk],
        images: list[ImageRef],
    ) -> None:
        """Replace the index in one transaction.

        Questions and image references use the qid relationship, while every
        chunk is inserted into both the structured table and FTS5.
        """
        with self._connect() as connection:
            if not images:
                images = [
                    image
                    for question in questions
                    for image in question.images
                ]
            qid_by_record = {
                question.record_id: question.qid
                for question in questions
            }
            connection.execute("DELETE FROM search_fts")
            connection.execute("DELETE FROM vector_map")
            connection.execute("DELETE FROM images")
            connection.execute("DELETE FROM chunks")
            connection.execute("DELETE FROM questions")

            for question in questions:
                connection.execute(
                    """
                    INSERT OR REPLACE INTO questions (
                        record_id, qid, number, stem, options_json, answer, explanation,
                        title, year, region, subject, question_type,
                        source_path, source_rel, raw_text
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        question.record_id,
                        question.qid,
                        question.number,
                        question.stem,
                        json.dumps(question.options, ensure_ascii=False),
                        question.answer,
                        question.explanation,
                        question.title,
                        question.year,
                        question.region,
                        question.subject,
                        question.question_type,
                        question.source_path,
                        question.source_rel,
                        question.raw_text,
                    ),
                )
            for image in images:
                connection.execute(
                    """
                    INSERT OR REPLACE INTO images (
                        record_id, image_id, qid, source_path, alt, anchor
                    ) VALUES (?, ?, ?, ?, ?, ?)
                    """,
                    (
                        image.record_id,
                        image.image_id,
                        qid_by_record.get(image.record_id, ""),
                        image.source_path,
                        image.alt,
                        image.anchor,
                    ),
                )

            for chunk in chunks:
                connection.execute(
                    """
                    INSERT OR REPLACE INTO chunks (
                        chunk_id, record_id, qid, chunk_type, text, title, year, region,
                        subject, question_type, source_path, source_rel,
                        source_format, doc_type, page_start, page_end
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        chunk.chunk_id,
                        chunk.record_id,
                        chunk.qid,
                        chunk.chunk_type,
                        chunk.text,
                        chunk.title,
                        chunk.year,
                        chunk.region,
                        chunk.subject,
                        chunk.question_type,
                        chunk.source_path,
                        chunk.source_rel,
                        chunk.source_format,
                        chunk.doc_type,
                        chunk.page_start,
                        chunk.page_end,
                    ),
                )
                connection.execute(
                    "INSERT INTO search_fts (chunk_id, text) VALUES (?, ?)",
                    (chunk.chunk_id, chunk.text),
                )
            connection.execute("PRAGMA optimize")

    def _chunk_from_row(self, row: sqlite3.Row | None) -> Chunk | None:
        if row is None:
            return None
        return Chunk(
            chunk_id=row["chunk_id"],
            qid=row["qid"],
            record_id=row["record_id"],
            chunk_type=row["chunk_type"],
            text=row["text"],
            title=row["title"],
            year=row["year"],
            region=row["region"],
            subject=row["subject"],
            question_type=row["question_type"],
            source_path=row["source_path"],
            source_rel=row["source_rel"],
            source_format=row["source_format"],
            doc_type=row["doc_type"],
            page_start=row["page_start"],
            page_end=row["page_end"],
        )

    def _question_from_row(self, row: sqlite3.Row | None) -> Question | None:
        if row is None:
            return None
        return Question(
            qid=row["qid"],
            record_id=row["record_id"],
            number=row["number"],
            stem=row["stem"],
            options=_json_load(row["options_json"], {}),
            answer=row["answer"],
            explanation=row["explanation"],
            title=row["title"],
            year=row["year"],
            region=row["region"],
            subject=row["subject"],
            question_type=row["question_type"],
            source_path=row["source_path"],
            source_rel=row["source_rel"],
            raw_text=row["raw_text"],
            images=[],
        )

    def get_question(self, record_id: str) -> Question | None:
        with self._connect() as connection:
            row = connection.execute(
                "SELECT * FROM questions WHERE record_id = ?",
                (record_id,),
            ).fetchone()
            question = self._question_from_row(row)
            if question:
                question.images = self._images_for_record(
                    connection,
                    question.record_id,
                )
            return question

    def find_question_by_qid(self, qid: str) -> Question | None:
        with self._connect() as connection:
            row = connection.execute(
                "SELECT * FROM questions WHERE qid = ? ORDER BY rowid LIMIT 1",
                (qid,),
            ).fetchone()
            question = self._question_from_row(row)
            if question:
                question.images = self._images_for_record(
                    connection,
                    question.record_id,
                )
            return question

    def _images_for_record(
        self,
        connection: sqlite3.Connection,
        record_id: str,
    ) -> list[ImageRef]:
        rows = connection.execute(
            "SELECT * FROM images WHERE record_id = ? ORDER BY image_id",
            (record_id,),
        ).fetchall()
        return [
            ImageRef(
                image_id=row["image_id"],
                source_path=row["source_path"],
                alt=row["alt"],
                anchor=row["anchor"],
            )
            for row in rows
        ]

    def get_image(self, image_id: str) -> ImageRef | None:
        with self._connect() as connection:
            row = connection.execute(
                "SELECT * FROM images WHERE image_id = ? ORDER BY rowid LIMIT 1",
                (image_id,),
            ).fetchone()
            if not row:
                return None
            return ImageRef(
                image_id=row["image_id"],
                source_path=row["source_path"],
                alt=row["alt"],
                anchor=row["anchor"],
            )

    def set_vector_mapping(self, chunk_ids: list[str]) -> None:
        with self._connect() as connection:
            connection.execute("DELETE FROM vector_map")
            connection.executemany(
                "INSERT INTO vector_map (vector_id, chunk_id) VALUES (?, ?)",
                list(enumerate(chunk_ids)),
            )

    def chunk_ids_for_vector_ids(self, vector_ids: list[int]) -> dict[int, str]:
        if not vector_ids:
            return {}
        placeholders = ",".join("?" for _ in vector_ids)
        with self._connect() as connection:
            rows = connection.execute(
                f"""
                SELECT vector_id, chunk_id
                FROM vector_map
                WHERE vector_id IN ({placeholders})
                """,
                vector_ids,
            ).fetchall()
        return {
            int(row["vector_id"]): row["chunk_id"]
            for row in rows
        }

    def _hit_for_chunk(
        self,
        connection: sqlite3.Connection,
        chunk: Chunk,
        score: float,
        retrieval: dict[str, Any] | None = None,
    ) -> SearchHit:
        question = (
            self._question_from_row(
                connection.execute(
                    "SELECT * FROM questions WHERE record_id = ?",
                    (chunk.record_id,),
                ).fetchone()
            )
            if chunk.record_id
            else None
        )
        images = (
            self._images_for_record(connection, question.record_id)
            if question
            else []
        )
        return SearchHit(
            score=score,
            chunk=chunk,
            question=question,
            images=images,
            retrieval=retrieval or {},
        )

    def get_hits_by_chunk_ids(
        self,
        chunk_ids: list[str],
        scores: dict[str, float] | None = None,
        retrievals: dict[str, dict[str, Any]] | None = None,
    ) -> list[SearchHit]:
        if not chunk_ids:
            return []
        scores = scores or {}
        retrievals = retrievals or {}
        result: list[SearchHit] = []
        with self._connect() as connection:
            for chunk_id in chunk_ids:
                row = connection.execute(
                    "SELECT * FROM chunks WHERE chunk_id = ?",
                    (chunk_id,),
                ).fetchone()
                chunk = self._chunk_from_row(row)
                if chunk:
                    result.append(
                        self._hit_for_chunk(
                            connection,
                            chunk,
                            scores.get(chunk_id, 0.0),
                            retrievals.get(chunk_id),
                        )
                    )
        return result

    def _direct_hit(
        self,
        connection: sqlite3.Connection,
        query: str,
    ) -> SearchHit | None:
        qid_match = QID_LOOKUP_RE.search(query)
        qid = qid_match.group(1) if qid_match else query.strip()
        if not qid or len(qid) > 128:
            return None
        row = connection.execute(
            "SELECT * FROM questions WHERE qid = ?",
            (qid,),
        ).fetchone()
        if not row:
            return None
        question = self._question_from_row(row)
        if not question:
            return None
        question.images = self._images_for_record(
            connection,
            question.record_id,
        )
        return SearchHit(
            score=1000.0,
            chunk=Chunk(
                chunk_id=f"direct:{question.qid}",
                qid=question.qid,
                record_id=question.record_id,
                chunk_type="question",
                text=question.stem,
                title=question.title,
                year=question.year,
                region=question.region,
                subject=question.subject,
                question_type=question.question_type,
                source_path=question.source_path,
                source_rel=question.source_rel,
            ),
            question=question,
            images=question.images,
        )

    def _filter_sql(
        self,
        filters: dict[str, Any],
        alias: str = "c",
    ) -> tuple[str, list[Any]]:
        clauses: list[str] = []
        params: list[Any] = []
        for key in (
            "year",
            "subject",
            "region",
            "question_type",
            "chunk_type",
            "doc_type",
            "source_format",
        ):
            value = filters.get(key)
            if value in (None, ""):
                continue
            if key == "chunk_type":
                clauses.append(f"{alias}.{key} = ?")
                params.append(str(value))
            else:
                clauses.append(f"{alias}.{key} LIKE ?")
                params.append(f"%{value}%")
        return (" AND " + " AND ".join(clauses) if clauses else ""), params

    def _candidate_rows(
        self,
        connection: sqlite3.Connection,
        query: str,
        limit: int,
        filters: dict[str, Any],
    ) -> list[tuple[str, float]]:
        terms = _search_terms(query)
        fts_query = _safe_fts_query(terms)
        filter_sql, filter_params = self._filter_sql(filters)
        rows: list[tuple[str, float]] = []

        if fts_query:
            try:
                rows = [
                    (row["chunk_id"], float(row["rank"] or 0.0))
                    for row in connection.execute(
                        f"""
                        SELECT search_fts.chunk_id, bm25(search_fts) AS rank
                        FROM search_fts
                        JOIN chunks c ON c.chunk_id = search_fts.chunk_id
                        WHERE search_fts MATCH ?
                        {filter_sql}
                        ORDER BY rank
                        LIMIT ?
                        """,
                        (fts_query, *filter_params, limit),
                    ).fetchall()
                ]
            except sqlite3.OperationalError:
                rows = []

        if not rows:
            # Short queries and punctuation-only terms fall back to a scan.
            # This path is intentionally conservative because normal Chinese
            # queries are handled by FTS5 trigram.
            needle = query.strip().replace("%", "").replace("_", "")
            if needle:
                rows = [
                    (row["chunk_id"], 0.0)
                    for row in connection.execute(
                        f"""
                        SELECT c.chunk_id
                        FROM chunks c
                        WHERE c.text LIKE ?
                        {filter_sql}
                        LIMIT ?
                        """,
                        (f"%{needle}%", *filter_params, limit),
                    ).fetchall()
                ]
        return rows

    def search(
        self,
        query: str,
        limit: int = 5,
        filters: dict[str, Any] | None = None,
        mode: str = "search",
    ) -> list[SearchHit]:
        query = (query or "").strip()
        if not query:
            return []
        filters = filters or {}
        with self._connect() as connection:
            direct = self._direct_hit(connection, query)
            if direct:
                return [direct]

            rows = self._candidate_rows(
                connection,
                query,
                max(limit * 8, 40),
                filters,
            )
            hits: list[SearchHit] = []
            exact = re.sub(r"\s+", "", query)
            for chunk_id, rank in rows:
                chunk_row = connection.execute(
                    "SELECT * FROM chunks WHERE chunk_id = ?",
                    (chunk_id,),
                ).fetchone()
                chunk = self._chunk_from_row(chunk_row)
                if not chunk:
                    continue

                score = max(0.01, -rank)
                compact_text = re.sub(r"\s+", "", chunk.text)
                if exact and exact in compact_text:
                    score += 5.0
                if mode == "search":
                    score += 1.2 if chunk.chunk_type == "question" else 0.0
                    score += 0.5 if chunk.chunk_type == "document" else 0.0
                elif mode == "knowledge":
                    score += 1.5 if chunk.chunk_type == "explanation" else 0.0
                    score += 0.8 if chunk.chunk_type == "document" else 0.0
                elif mode == "explain":
                    score += 2.0 if chunk.chunk_type in {"question", "explanation"} else 0.0

                question = (
                    self._question_from_row(
                        connection.execute(
                            "SELECT * FROM questions WHERE record_id = ?",
                            (chunk.record_id,),
                        ).fetchone()
                    )
                    if chunk.record_id
                    else None
                )
                images = (
                    self._images_for_record(connection, question.record_id)
                    if question
                    else []
                )
                hits.append(
                    SearchHit(
                        score=score,
                        chunk=chunk,
                        question=question,
                        images=images,
                    )
                )

            hits.sort(key=lambda item: (-item.score, item.chunk.chunk_id))
            return hits[:limit]

    def stats(self) -> dict[str, int]:
        self.initialize()
        with self._connect() as connection:
            questions = connection.execute(
                "SELECT COUNT(*) AS count FROM questions"
            ).fetchone()["count"]
            chunks = connection.execute(
                "SELECT COUNT(*) AS count FROM chunks"
            ).fetchone()["count"]
            images = connection.execute(
                "SELECT COUNT(*) AS count FROM images"
            ).fetchone()["count"]
            files = connection.execute(
                "SELECT COUNT(DISTINCT source_rel) AS count FROM chunks"
            ).fetchone()["count"]
            vector_count = connection.execute(
                "SELECT COUNT(*) AS count FROM vector_map"
            ).fetchone()["count"]
            return {
                "questions": int(questions),
                "chunks": int(chunks),
                "images": int(images),
                "files": int(files),
                "vectors": int(vector_count),
            }
