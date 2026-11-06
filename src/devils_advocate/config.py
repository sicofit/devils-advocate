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
    # Embedding calls are short; a hung one is cut off rather than waited out
    # (`devil embed` tries the batch again, a search does not).
    embed_timeout: float = 60.0
    # Passed through to the endpoint. "none" switches off the thinking phase of
    # reasoning models on Ollama, which cuts a structured call from ~15 s to ~1 s.
    # Set DA_REASONING_EFFORT=omit to send nothing, for servers that reject it.
    reasoning_effort: str | None = "none"
    # Earliest decision year that `devil index` puts into the index.
    since: int = 2000
    # Passages shown to the model per claim, and claims attacked per argument.
    top_k: int = 8
    max_claims: int = 4

    @property
    def endpoint(self) -> str:
        """The server root: "localhost:11434/v1/" becomes "http://localhost:11434"."""
        url = self.base_url.strip().rstrip("/")
        if "://" not in url:
            url = "http://" + url
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
        variable = ENV_PREFIX + f.name.upper()
        raw = os.environ.get(variable)
        if raw is None:
            continue
        raw = raw.strip()
        if "None" in str(f.type) and raw.lower() in {"", "omit"}:
            values[f.name] = None
        elif raw:
            values[f.name] = _coerce(getattr(defaults, f.name), raw, variable)
    for name, value in overrides.items():
        if value is not None:
            values[name] = _coerce(getattr(defaults, name), value, "--" + name.replace("_", "-"))
    config = Config(**values)
    _check(config)
    return config


# The smallest and largest value each number may take, and what it means.
LIMITS: dict[str, tuple[float, float, str]] = {
    "temperature": (0.0, 2.0, "between 0 and 2"),
    "request_timeout": (1.0, 86_400.0, "between 1 and 86400 (seconds)"),
    "embed_timeout": (1.0, 86_400.0, "between 1 and 86400 (seconds)"),
    "since": (1875, 2100, "a year between 1875 and 2100"),
    "top_k": (1, 50, "between 1 and 50"),
    "max_claims": (1, 10, "between 1 and 10"),
}


def _check(config: Config) -> None:
    for name, (low, high, meaning) in LIMITS.items():
        value = getattr(config, name)
        if not low <= value <= high:
            raise SystemExit(f"{ENV_PREFIX}{name.upper()} must be {meaning}, not {value!r}")


def _coerce(default: object, value: object, source: str) -> object:
    if isinstance(default, Path):
        if not str(value).strip():
            raise SystemExit(f"{source} must name a directory, not an empty string")
        return Path(str(value)).expanduser()
    if isinstance(default, bool):
        return str(value).lower() in {"1", "true", "yes", "on"}
    try:
        if isinstance(default, int):
            return int(value)
        if isinstance(default, float):
            return float(value)
    except ValueError:
        kind = "a whole number" if isinstance(default, int) else "a number"
        raise SystemExit(f"{source} must be {kind}, not {value!r}") from None
    return value
