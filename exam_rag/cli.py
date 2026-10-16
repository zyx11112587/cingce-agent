from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

from .config import Settings
from .index import IndexStore
from .ingest import ingest_directory
from .rag import RAGService
from .server import run_server


def _print_json(payload) -> None:
    print(json.dumps(payload, ensure_ascii=False, indent=2))


def _add_common_query_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("query", help="问题或检索词")
    parser.add_argument("--top-k", type=int, default=None)
    parser.add_argument("--year", default="")
    parser.add_argument("--subject", default="")
    parser.add_argument("--region", default="")
    parser.add_argument("--json", action="store_true")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="exam-rag",
        description="Local civil-service exam RAG system",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    ingest = subparsers.add_parser("ingest", help="Parse Markdown and rebuild the index")
    ingest.add_argument("--input", default="", help="Markdown root directory")
    ingest.add_argument("--db", default="", help="SQLite index path")

    search = subparsers.add_parser("search", help="Search question and explanation chunks")
    _add_common_query_args(search)
    search.add_argument(
        "--mode",
        choices=("search", "knowledge", "explain"),
        default="search",
    )

    ask = subparsers.add_parser("ask", help="Ask a RAG question")
    _add_common_query_args(ask)
    ask.add_argument(
        "--mode",
        choices=("search", "knowledge", "explain"),
        default="knowledge",
    )

    explain = subparsers.add_parser("explain", help="Query a question and its explanation")
    _add_common_query_args(explain)

    serve = subparsers.add_parser("serve", help="Start the local HTTP API and web UI")
    serve.add_argument("--host", default="")
    serve.add_argument("--port", type=int, default=0)

    subparsers.add_parser("stats", help="Show index statistics")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    settings = Settings.from_env()

    if args.command == "ingest":
        input_dir = Path(args.input).resolve() if args.input else settings.input_dir
        db_path = Path(args.db).resolve() if args.db else settings.db_path
        started = time.time()

        def progress(index: int, total: int, questions: int, chunks: int) -> None:
            print(
                f"[ingest] {index}/{total} files, "
                f"{questions} questions, {chunks} chunks",
                flush=True,
            )

        print(f"Input: {input_dir}")
        print(f"Database: {db_path}")
        result = ingest_directory(input_dir, db_path, progress=progress)
        result["elapsed_seconds"] = round(time.time() - started, 2)
        _print_json(result)
        return 0

    if args.command == "stats":
        _print_json(IndexStore(settings.db_path).stats())
        return 0

    if args.command == "serve":
        run_server(
            settings,
            host=args.host or None,
            port=args.port or None,
        )
        return 0

    service = RAGService(settings)
    filters = {
        "year": args.year,
        "subject": args.subject,
        "region": args.region,
    }
    if args.command == "search":
        result = service.search(
            args.query,
            mode=args.mode,
            top_k=args.top_k,
            filters=filters,
        )
        _print_json({"query": args.query, "hits": result})
        return 0

    mode = "explain" if args.command == "explain" else args.mode
    result = service.answer(
        args.query,
        mode=mode,
        top_k=args.top_k,
        filters=filters,
    )
    if args.json:
        _print_json(result)
    else:
        print(result["answer"])
        if result.get("citations"):
            print("\n来源：")
            for citation in result["citations"]:
                print(f"- {citation['source_rel']}  qid={citation['qid']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
