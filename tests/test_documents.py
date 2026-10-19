from __future__ import annotations

import tempfile
import unittest
import sqlite3
from pathlib import Path

from exam_rag.config import Settings
from exam_rag.document_loader import load_document
from exam_rag.ingest import ingest_directories
from exam_rag.index import IndexStore
from exam_rag.rag import RAGService
from exam_rag.vector_store import NumpyVectorStore, get_vector_store


class DummyEmbeddingProvider:
    def embed(self, texts, progress=None):
        return [
            [1.0, 0.0] if "中国式现代化" in text else [0.0, 1.0]
            for text in texts
        ]


class DocumentAndVectorTest(unittest.TestCase):
    def test_text_document_loading(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            path = root / "data" / "shenlun" / "示例材料.txt"
            path.parent.mkdir(parents=True)
            path.write_text(
                "申论材料\n\n高质量发展是全面建设社会主义现代化国家的首要任务。\n\n"
                "提炼观点时，应先找总起句，再找分论点。",
                encoding="utf-8",
            )
            chunks = load_document(path, root)
        self.assertGreaterEqual(len(chunks), 1)
        self.assertEqual(chunks[0].subject, "申论")
        self.assertEqual(chunks[0].doc_type, "申论")
        self.assertEqual(chunks[0].source_format, "txt")

    def test_hybrid_ingest_falls_back_to_bm25_without_embedding(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            input_dir = root / "data" / "politics"
            input_dir.mkdir(parents=True)
            (input_dir / "政治理论.txt").write_text(
                "中国式现代化是中国共产党领导的社会主义现代化。",
                encoding="utf-8",
            )
            settings = Settings(
                root=root,
                input_dir=input_dir,
                db_path=root / "index.sqlite3",
                assets_dir=root / "assets",
                host="127.0.0.1",
                port=8000,
                llm_base_url="",
                llm_api_key="",
                llm_model="",
                llm_timeout=5,
                default_top_k=3,
                input_dirs=(input_dir,),
                embedding_provider="none",
                vector_dir=root / "vectors",
            )
            result = ingest_directories(
                [input_dir],
                settings.db_path,
                settings=settings,
                skip_vectors=True,
            )
            self.assertGreater(result["chunks"], 0)
            service = RAGService(settings)
            hits = service.search("中国式现代化")
            self.assertGreaterEqual(len(hits), 1)
            answer = service.answer("中国式现代化是什么？", top_k=2)
            self.assertEqual(answer["retrieval_mode"], "bm25")
            self.assertIn("中国式现代化", answer["answer"])

    def test_numpy_vector_store_round_trip(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            store = NumpyVectorStore(Path(directory))
            count = store.build([[1.0, 0.0], [0.0, 1.0]])
            self.assertEqual(count, 2)
            matches = store.search([1.0, 0.0], top_k=1)
            self.assertEqual(matches[0].vector_id, 0)
            self.assertGreater(matches[0].score, 0.9)

    def test_rrf_merges_bm25_and_vector(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            politics = root / "data" / "politics"
            shenlun = root / "data" / "shenlun"
            politics.mkdir(parents=True)
            shenlun.mkdir(parents=True)
            (politics / "a.txt").write_text(
                "中国式现代化是中国共产党领导的社会主义现代化。",
                encoding="utf-8",
            )
            (shenlun / "b.txt").write_text(
                "申论材料分析需要提炼总分论点。",
                encoding="utf-8",
            )
            settings = Settings(
                root=root,
                input_dir=politics,
                db_path=root / "index.sqlite3",
                assets_dir=root / "assets",
                host="127.0.0.1",
                port=8000,
                llm_base_url="",
                llm_api_key="",
                llm_model="",
                llm_timeout=5,
                default_top_k=3,
                input_dirs=(politics, shenlun),
                embedding_provider="none",
                vector_dir=root / "vectors",
            )
            ingest_directories(
                [politics, shenlun],
                settings.db_path,
                settings=settings,
                skip_vectors=True,
            )
            connection = sqlite3.connect(settings.db_path)
            rows = connection.execute(
                "SELECT chunk_id, text FROM chunks ORDER BY chunk_id"
            ).fetchall()
            connection.close()
            ids = [row[0] for row in rows]
            vectors = DummyEmbeddingProvider().embed([row[1] for row in rows])
            IndexStore(settings.db_path).set_vector_mapping(ids)
            get_vector_store(settings.vector_dir).build(vectors)

            service = RAGService(settings)
            service.embedding_provider = DummyEmbeddingProvider()
            hits, error = service.retrieve_hits("中国式现代化是什么？", top_k=2)
            self.assertEqual(error, "")
            self.assertIn("vector_rank", hits[0].retrieval)
            self.assertIn("bm25_rank", hits[0].retrieval)


if __name__ == "__main__":
    unittest.main()
