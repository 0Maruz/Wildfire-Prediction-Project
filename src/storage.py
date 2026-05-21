"""URI-portable I/O for the pipeline.

Every caller in the pipeline should route I/O through this module instead of
touching ``open``, ``pd.read_parquet``, ``joblib.load``, ``json.load`` etc.
directly. This is the seam that lets the same code read from local disk today
and from S3 / GCS / Azure tomorrow by swapping nothing but the URI.

Design notes:

- Local paths work without extra dependencies — falls back to plain ``open`` /
  ``pd.read_parquet`` / ``joblib.load``.
- Cloud URIs (``s3://``, ``gs://``, ``az://``, ``r2://``, ``b2://``, ...) lazy-
  import ``fsspec`` plus the matching backend (``s3fs``, ``gcsfs``, ``adlfs``).
  No import overhead until you actually touch a remote URI.
- Tables (CSV / Parquet) delegate to ``io_utils`` for local paths so the
  existing CSV ↔ Parquet sibling-format fallback keeps working unchanged.
- ``write_*`` ensures the parent directory exists for local paths. Object
  stores don't have directories, so it's a no-op for remote URIs.
"""

from __future__ import annotations

import json
import os
from typing import Any, Iterable
from urllib.parse import urlparse

import pandas as pd

import io_utils

# Schemes treated as "remote" — routed through fsspec.
# ``file://`` is parsed as remote-shaped but we strip it back to a local path.
_REMOTE_SCHEMES = {
    "s3", "s3a",
    "gs", "gcs",
    "az", "abfs", "abfss", "adl",
    "r2", "b2",
    "http", "https", "ftp",
    "hdfs",
}


def _scheme(uri: str) -> str:
    if "://" not in uri:
        return ""
    return urlparse(uri).scheme.lower()


def is_remote(uri: str) -> bool:
    """True if the URI should be routed through fsspec rather than the local FS."""
    return _scheme(uri) in _REMOTE_SCHEMES


def _local_path(uri: str) -> str:
    """Strip a leading ``file://`` if present; pass everything else through unchanged."""
    if uri.startswith("file://"):
        return uri[len("file://") :]
    return uri


def _fs(uri: str):
    """Return the fsspec filesystem for a remote URI. Lazily imports fsspec."""
    try:
        import fsspec
    except ImportError as e:  # pragma: no cover
        raise RuntimeError(
            f"Reading remote URI {uri!r} requires fsspec. "
            f"Install with: pip install fsspec  "
            f"(plus the matching backend: s3fs / gcsfs / adlfs / ...)"
        ) from e
    return fsspec.filesystem(_scheme(uri))


# ─────────────────────────────────────────────
# Existence + parent-dir helpers
# ─────────────────────────────────────────────

def exists(uri: str) -> bool:
    if is_remote(uri):
        return _fs(uri).exists(uri)
    return os.path.exists(_local_path(uri))


def mkdirs_for(uri: str) -> None:
    """Ensure the parent directory of ``uri`` exists. No-op for remote URIs."""
    if is_remote(uri):
        return
    parent = os.path.dirname(_local_path(uri))
    if parent:
        os.makedirs(parent, exist_ok=True)


# ─────────────────────────────────────────────
# Bytes / text / JSON
# ─────────────────────────────────────────────

def read_bytes(uri: str) -> bytes:
    if is_remote(uri):
        with _fs(uri).open(uri, "rb") as f:
            return f.read()
    with open(_local_path(uri), "rb") as f:
        return f.read()


def write_bytes(data: bytes, uri: str) -> None:
    mkdirs_for(uri)
    if is_remote(uri):
        with _fs(uri).open(uri, "wb") as f:
            f.write(data)
        return
    with open(_local_path(uri), "wb") as f:
        f.write(data)


def read_text(uri: str, encoding: str = "utf-8") -> str:
    return read_bytes(uri).decode(encoding)


def write_text(text: str, uri: str, encoding: str = "utf-8") -> None:
    write_bytes(text.encode(encoding), uri)


def read_json(uri: str) -> Any:
    return json.loads(read_text(uri))


def write_json(obj: Any, uri: str, *, indent: int | None = 2, default=None) -> None:
    write_text(json.dumps(obj, indent=indent, default=default), uri)


# ─────────────────────────────────────────────
# Pickle / joblib artifacts (models)
# ─────────────────────────────────────────────

def read_pickle(uri: str) -> Any:
    """Load a joblib/pickle artifact from a local path or remote URI."""
    import joblib
    if is_remote(uri):
        with _fs(uri).open(uri, "rb") as f:
            return joblib.load(f)
    return joblib.load(_local_path(uri))


def write_pickle(obj: Any, uri: str, *, compress: int = 0) -> None:
    """Save an object via joblib (sklearn-compatible) to a local path or remote URI."""
    import joblib
    mkdirs_for(uri)
    if is_remote(uri):
        with _fs(uri).open(uri, "wb") as f:
            joblib.dump(obj, f, compress=compress)
        return
    joblib.dump(obj, _local_path(uri), compress=compress)


# ─────────────────────────────────────────────
# Tables (CSV / Parquet)
# ─────────────────────────────────────────────

def read_table(uri: str, **kwargs) -> pd.DataFrame:
    """Read a CSV or Parquet table.

    Local paths delegate to :func:`io_utils.read_table` and keep the CSV ↔
    Parquet sibling-format fallback. Remote URIs are passed straight to
    pandas (which uses pyarrow / fsspec under the hood) with no sibling
    fallback — explicit URIs only.
    """
    if is_remote(uri):
        ext = os.path.splitext(uri)[1].lower()
        if ext == ".parquet":
            return pd.read_parquet(uri, **kwargs)
        return pd.read_csv(uri, **kwargs)
    return io_utils.read_table(_local_path(uri), **kwargs)


def write_table(df: pd.DataFrame, uri: str, **kwargs) -> None:
    """Write a CSV or Parquet table based on the URI extension."""
    mkdirs_for(uri)
    if is_remote(uri):
        ext = os.path.splitext(uri)[1].lower()
        if ext == ".parquet":
            df.to_parquet(uri, index=False, **kwargs)
        else:
            df.to_csv(uri, index=False, **kwargs)
        return
    io_utils.write_table(df, _local_path(uri), **kwargs)


def resolve_existing(uri: str):
    """Return the actual URI that exists (local CSV ↔ Parquet fallback included).

    For remote URIs this is a pass-through — returns the URI if it exists,
    else ``None``. No sibling-format fallback for remote.
    """
    if is_remote(uri):
        return uri if exists(uri) else None
    return io_utils.resolve_existing(_local_path(uri))


def list_tables(paths_or_globs: Iterable[str]) -> list[str]:
    """Resolve a mix of files / dirs / globs into concrete table file paths.

    Local-only for now. Cloud listing should call ``_fs(uri).glob(...)`` with
    a known prefix — left out of the public API until we actually need it.
    """
    return io_utils.list_tables([_local_path(p) for p in paths_or_globs])
