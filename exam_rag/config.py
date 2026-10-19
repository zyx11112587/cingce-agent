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
    input_dirs: tuple[Path, ...] = ()
    embedding_provider: str = "auto"
    embedding_base_url: str = ""
    embedding_api_key: str = ""
    embedding_model: str = ""
    embedding_batch_size: int = 32
    vector_dir: Path = PROJECT_ROOT / "data" / "index" / "vectors"
    bm25_weight: float = 0.5
    vector_weight: float = 0.5
    rrf_k: int = 60

    @property
    def llm_enabled(self) -> bool:
        return bool(self.llm_base_url and self.llm_model)

    @property
    def effective_input_dirs(self) -> tuple[Path, ...]:
        return self.input_dirs or (self.input_dir,)

    @classmethod
    def from_env(cls, root: Path | None = None) -> "Settings":
        project_root = (root or PROJECT_ROOT).resolve()
        _load_dotenv(project_root / ".env")

        explicit_inputs = _env("EXAM_RAG_INPUT_DIRS", default="")
        if explicit_inputs:
            input_dirs = tuple(
                Path(part.strip()).expanduser().resolve()
                for part in explicit_inputs.split(os.pathsep)
                if part.strip()
            )
        else:
            singular_input = _env("EXAM_RAG_INPUT_DIR", default="")
            if singular_input:
                input_dirs = (Path(singular_input).expanduser().resolve(),)
            else:
                candidates = (
                    project_root / "xingcezhenti",
                    project_root / "data" / "raw",
                    project_root / "data" / "politics",
                    project_root / "data" / "shenlun",
                    project_root / "data" / "materials",
                )
                input_dirs = tuple(path.resolve() for path in candidates if path.exists())
                if not input_dirs:
                    input_dirs = ((project_root / "data" / "raw").resolve(),)
        input_dir = input_dirs[0]
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
            input_dirs=input_dirs,
            embedding_provider=_env(
                "EXAM_RAG_EMBEDDING_PROVIDER",
                default="auto",
            ).lower(),
            embedding_base_url=_env(
                "EXAM_RAG_EMBEDDING_BASE_URL",
                "EMBEDDING_BASE_URL",
                default="",
            ).rstrip("/"),
            embedding_api_key=_env(
                "EXAM_RAG_EMBEDDING_API_KEY",
                "EMBEDDING_API_KEY",
                default="",
            ),
            embedding_model=_env(
                "EXAM_RAG_EMBEDDING_MODEL",
                "EMBEDDING_MODEL",
                default="",
            ),
            embedding_batch_size=int(
                _env("EXAM_RAG_EMBEDDING_BATCH_SIZE", default="32")
            ),
            vector_dir=Path(
                _env(
                    "EXAM_RAG_VECTOR_DIR",
                    default=str(project_root / "data" / "index" / "vectors"),
                )
            ).expanduser().resolve(),
            bm25_weight=float(_env("EXAM_RAG_BM25_WEIGHT", default="0.5")),
            vector_weight=float(_env("EXAM_RAG_VECTOR_WEIGHT", default="0.5")),
            rrf_k=int(_env("EXAM_RAG_RRF_K", default="60")),
        )
