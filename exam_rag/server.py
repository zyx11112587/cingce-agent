from __future__ import annotations

import json
import mimetypes
import urllib.parse
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any

from .config import Settings
from .index import IndexStore
from .rag import RAGService


class ExamRAGRequestHandler(BaseHTTPRequestHandler):
    server_version = "ExamRAG/0.1"

    @property
    def service(self) -> RAGService:
        return self.server.service  # type: ignore[attr-defined]

    def log_message(self, format: str, *args: Any) -> None:
        print(f"[http] {self.address_string()} - {format % args}")

    def _send_bytes(
        self,
        body: bytes,
        content_type: str,
        status: int = HTTPStatus.OK,
    ) -> None:
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def _send_json(
        self,
        payload: Any,
        status: int = HTTPStatus.OK,
    ) -> None:
        body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        self._send_bytes(body, "application/json; charset=utf-8", status)

    def _send_error_json(self, message: str, status: int) -> None:
        self._send_json({"error": message}, status)

    def _query(self) -> dict[str, list[str]]:
        parsed = urllib.parse.urlparse(self.path)
        return urllib.parse.parse_qs(parsed.query)

    def _filters(self, params: dict[str, list[str]]) -> dict[str, str]:
        filters: dict[str, str] = {}
        for key in ("year", "subject", "region", "question_type", "chunk_type"):
            value = (params.get(key) or [""])[0].strip()
            if value:
                filters[key] = value
        return filters

    def _top_k(self, params: dict[str, list[str]], default: int) -> int:
        try:
            value = int((params.get("top_k") or [str(default)])[0])
        except (TypeError, ValueError):
            value = default
        return max(1, min(value, 20))

    def do_OPTIONS(self) -> None:
        self.send_response(HTTPStatus.NO_CONTENT)
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        self.end_headers()

    def do_GET(self) -> None:
        parsed = urllib.parse.urlparse(self.path)
        path = urllib.parse.unquote(parsed.path)
        params = self._query()

        if path in ("/", "/index.html"):
            html_path = Path(__file__).parent / "web" / "index.html"
            body = html_path.read_bytes()
            self._send_bytes(body, "text/html; charset=utf-8")
            return

        if path == "/api/health":
            self._send_json(
                {
                    "status": "ok",
                    "llm_enabled": self.service.llm.enabled,
                    "index": str(self.service.settings.db_path),
                }
            )
            return

        if path == "/api/stats":
            self._send_json(self.service.store.stats())
            return

        if path == "/api/search":
            query = (params.get("q") or [""])[0].strip()
            if not query:
                self._send_error_json("Missing query parameter q", HTTPStatus.BAD_REQUEST)
                return
            mode = (params.get("mode") or ["search"])[0]
            top_k = self._top_k(params, self.service.settings.default_top_k)
            self._send_json(
                {
                    "query": query,
                    "mode": mode,
                    "hits": self.service.search(
                        query,
                        mode=mode,
                        top_k=top_k,
                        filters=self._filters(params),
                    ),
                }
            )
            return

        if path.startswith("/api/question/"):
            qid = path.removeprefix("/api/question/").strip()
            if not qid:
                self._send_error_json("Missing qid", HTTPStatus.BAD_REQUEST)
                return
            question = (
                self.service.store.get_question(qid)
                or self.service.store.find_question_by_qid(qid)
            )
            if not question:
                self._send_error_json("Question not found", HTTPStatus.NOT_FOUND)
                return
            self._send_json(question.to_dict())
            return

        if path.startswith("/api/image/"):
            image_id = path.removeprefix("/api/image/").strip()
            image = self.service.store.get_image(image_id)
            if not image:
                self._send_error_json("Image not found", HTTPStatus.NOT_FOUND)
                return
            image_path = Path(image.source_path)
            if not image_path.exists() or not image_path.is_file():
                self._send_error_json("Image file is missing", HTTPStatus.NOT_FOUND)
                return
            content_type = mimetypes.guess_type(image_path.name)[0] or "application/octet-stream"
            self._send_bytes(image_path.read_bytes(), content_type)
            return

        self._send_error_json("Not found", HTTPStatus.NOT_FOUND)

    def do_POST(self) -> None:
        parsed = urllib.parse.urlparse(self.path)
        path = urllib.parse.unquote(parsed.path)
        if path != "/api/qa":
            self._send_error_json("Not found", HTTPStatus.NOT_FOUND)
            return

        try:
            length = int(self.headers.get("Content-Length", "0"))
            payload = json.loads(self.rfile.read(length).decode("utf-8"))
        except (ValueError, json.JSONDecodeError):
            self._send_error_json("Invalid JSON body", HTTPStatus.BAD_REQUEST)
            return

        question = str(payload.get("question") or "").strip()
        if not question:
            self._send_error_json("Missing question", HTTPStatus.BAD_REQUEST)
            return
        mode = str(payload.get("mode") or "knowledge").strip()
        if mode not in {"search", "knowledge", "explain"}:
            mode = "knowledge"
        try:
            top_k = int(payload.get("top_k") or self.service.settings.default_top_k)
        except (TypeError, ValueError):
            top_k = self.service.settings.default_top_k
        top_k = max(1, min(top_k, 20))
        filters = payload.get("filters") or {}
        if not isinstance(filters, dict):
            filters = {}
        self._send_json(
            self.service.answer(
                question,
                mode=mode,
                top_k=top_k,
                filters=filters,
            )
        )


class ExamRAGHTTPServer(ThreadingHTTPServer):
    def __init__(self, server_address, handler_class, service: RAGService):
        super().__init__(server_address, handler_class)
        self.service = service


def run_server(
    settings: Settings,
    host: str | None = None,
    port: int | None = None,
) -> None:
    store = IndexStore(settings.db_path)
    store.initialize()
    service = RAGService(settings)
    server = ExamRAGHTTPServer(
        (host or settings.host, port or settings.port),
        ExamRAGRequestHandler,
        service,
    )
    actual_host, actual_port = server.server_address[:2]
    print(f"Exam RAG server: http://{actual_host}:{actual_port}")
    print(f"Index: {settings.db_path}")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nStopping server...")
    finally:
        server.server_close()
