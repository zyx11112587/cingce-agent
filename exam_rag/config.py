from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _load_dotenv(path: Path) -> None:
    """Load a small .env file without adding a third-party dependency."""
    if not path.exists():
        return
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip().strip("'\"")
        if key and key not in os.environ:
            os.environ[key] = value


def _env(*names: str, default: str = "") -> str:
    for name in names:
        value = os.getenv(name)
        if value:
            return value
    return default


@dataclass(frozen=True)
class Settings:
    root: Path
    input_dir: Path
    db_path: Path
    assets_dir: Path
    host: str
    port: int
    llm_base_url: str
    llm_api_key: str
    llm_model: str
    llm_timeout: int
    default_top_k: int

    @property
    def llm_enabled(self) -> bool:
        return bool(self.llm_base_url and self.llm_model)

    @classmethod
    def from_env(cls, root: Path | None = None) -> "Settings":
        project_root = (root or PROJECT_ROOT).resolve()
        _load_dotenv(project_root / ".env")

        default_input = project_root / "xingcezhenti"
        if not default_input.exists():
            default_input = project_root / "data" / "raw"

        input_dir = Path(
            _env("EXAM_RAG_INPUT_DIR", default=str(default_input))
        ).expanduser()
        db_path = Path(
            _env(
                "EXAM_RAG_DB",
                default=str(project_root / "data" / "index" / "exam.sqlite3"),
            )
        ).expanduser()
        assets_dir = Path(
            _env(
                "EXAM_RAG_ASSETS",
                default=str(project_root / "data" / "assets"),
            )
        ).expanduser()

        return cls(
            root=project_root,
            input_dir=input_dir.resolve(),
            db_path=db_path.resolve(),
            assets_dir=assets_dir.resolve(),
            host=_env("EXAM_RAG_HOST", default="127.0.0.1"),
            port=int(_env("EXAM_RAG_PORT", default="8000")),
            llm_base_url=_env(
                "EXAM_RAG_LLM_BASE_URL",
                "OPENAI_BASE_URL",
                "OPENAI_API_BASE",
                default="",
            ).rstrip("/"),
            llm_api_key=_env(
                "EXAM_RAG_LLM_API_KEY",
                "OPENAI_API_KEY",
                default="",
            ),
            llm_model=_env(
                "EXAM_RAG_LLM_MODEL",
                "OPENAI_MODEL",
                default="",
            ),
            llm_timeout=int(_env("EXAM_RAG_LLM_TIMEOUT", default="120")),
            default_top_k=int(_env("EXAM_RAG_TOP_K", default="5")),
        )
