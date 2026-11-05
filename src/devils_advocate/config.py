"""Settings: built-in defaults, then DA_* environment variables, then command-line flags."""

from __future__ import annotations

import os
from dataclasses import dataclass, fields
from pathlib import Path

ENV_PREFIX = "DA_"


@dataclass(frozen=True, slots=True)
class Config:
    base_url: str = "http://127.0.0.1:11434"
    model: str = "qwen3.5:9b"
    embed_model: str = "nomic-embed-text"
    api_key: str | None = None
    data_dir: Path = Path.home() / ".devils-advocate"
    temperature: float = 0.2
    request_timeout: float = 600.0
    # Embedding calls are short; a hung one is retried rather than waited out.
    embed_timeout: float = 60.0
    # Passed through to the endpoint. "none" switches off the thinking phase of
    # reasoning models on Ollama, which cuts a structured call from ~15 s to ~1 s.
    # Set DA_REASONING_EFFORT="" to send nothing.
    reasoning_effort: str | None = "none"
    # Earliest decision year that `devil index` puts into the index.
    since: int = 2000
    # Passages shown to the model per claim, and claims attacked per argument.
    top_k: int = 8
    max_claims: int = 4

    @property
    def endpoint(self) -> str:
        url = self.base_url.rstrip("/")
        return url[: -len("/v1")] if url.endswith("/v1") else url

    @property
    def corpus_path(self) -> Path:
        return self.data_dir / "bge.parquet"

    @property
    def db_path(self) -> Path:
        return self.data_dir / "index.sqlite"


def load(**overrides: object) -> Config:
    """Resolve the configuration. `None` overrides are ignored so unset flags fall through."""
    defaults = Config()
    values: dict[str, object] = {}
    for f in fields(Config):
        raw = os.environ.get(ENV_PREFIX + f.name.upper())
        if raw:
            values[f.name] = _coerce(getattr(defaults, f.name), raw)
        elif raw == "" and "None" in str(f.type):
            values[f.name] = None
    for name, value in overrides.items():
        if value is not None:
            values[name] = _coerce(getattr(defaults, name), value)
    return Config(**values)


def _coerce(default: object, value: object) -> object:
    if isinstance(default, Path):
        return Path(str(value)).expanduser()
    if isinstance(default, bool):
        return str(value).lower() in {"1", "true", "yes", "on"}
    if isinstance(default, int):
        return int(value)
    if isinstance(default, float):
        return float(value)
    return value
