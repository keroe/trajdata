"""Utilities for robust cache file I/O.

Provides atomic writes (write-to-temp-then-rename) and safe loads
with corruption detection for dill and pickle serialized cache files.
"""

import os
import pickle
import tempfile
import warnings
from pathlib import Path
from typing import Any, Callable, IO

import dill


class CacheCorruptionError(Exception):
    """Raised when a cache file cannot be deserialized (likely truncated/corrupted)."""

    def __init__(self, path: Path, original_error: Exception) -> None:
        self.path = path
        self.original_error = original_error
        super().__init__(
            f"Corrupted cache file: {path} "
            f"({type(original_error).__name__}: {original_error})"
        )


def atomic_write(target_path: Path, write_fn: Callable[[IO[bytes]], None]) -> None:
    """Write to a temporary file in the same directory, then atomically replace target.

    Uses os.replace() which is atomic on POSIX when source and destination are on
    the same filesystem. This ensures a killed process can never leave a half-written
    cache file -- the old file stays intact until the new one is fully written.

    Args:
        target_path: Final destination path for the file.
        write_fn: Callable that receives an open binary file handle and writes to it.
    """
    target_path = Path(target_path)
    target_dir = target_path.parent
    target_dir.mkdir(parents=True, exist_ok=True)

    # Create temp file in same directory to guarantee same filesystem for atomic replace
    fd, tmp_path_str = tempfile.mkstemp(
        dir=target_dir, suffix=".tmp", prefix=f".{target_path.name}."
    )
    tmp_path = Path(tmp_path_str)
    try:
        with os.fdopen(fd, "wb") as f:
            write_fn(f)
        # Flush to disk before renaming
        os.replace(tmp_path, target_path)
    except BaseException:
        # Clean up temp file on any failure (including KeyboardInterrupt)
        tmp_path.unlink(missing_ok=True)
        raise


def safe_dill_dump(obj: Any, target_path: Path) -> None:
    """Atomically serialize an object with dill.

    Writes to a temp file first, then renames to target_path.
    If the process is killed mid-write, target_path is never corrupted.
    """
    atomic_write(target_path, lambda f: dill.dump(obj, f))


def safe_pickle_dump(obj: Any, target_path: Path) -> None:
    """Atomically serialize an object with pickle.

    Writes to a temp file first, then renames to target_path.
    If the process is killed mid-write, target_path is never corrupted.
    """
    atomic_write(target_path, lambda f: pickle.dump(obj, f))


def safe_dill_load(path: Path) -> Any:
    """Load a dill-serialized object with corruption detection.

    Args:
        path: Path to the dill file.

    Returns:
        The deserialized object.

    Raises:
        CacheCorruptionError: If the file is corrupted or truncated.
        FileNotFoundError: If the file does not exist.
    """
    if not path.exists():
        raise FileNotFoundError(f"Cache file not found: {path}")

    try:
        with open(path, "rb") as f:
            return dill.load(f)
    except (EOFError, pickle.UnpicklingError) as e:
        raise CacheCorruptionError(path, e) from e


def safe_pickle_load(path: Path) -> Any:
    """Load a pickle-serialized object with corruption detection.

    Args:
        path: Path to the pickle file.

    Returns:
        The deserialized object.

    Raises:
        CacheCorruptionError: If the file is corrupted or truncated.
        FileNotFoundError: If the file does not exist.
    """
    if not path.exists():
        raise FileNotFoundError(f"Cache file not found: {path}")

    try:
        with open(path, "rb") as f:
            return pickle.load(f)
    except (EOFError, pickle.UnpicklingError) as e:
        raise CacheCorruptionError(path, e) from e


def delete_corrupted_file(path: Path, warn: bool = True) -> None:
    """Delete a corrupted cache file and optionally warn.

    Args:
        path: Path to the corrupted file.
        warn: Whether to emit a warning about the deletion.
    """
    if path.exists():
        path.unlink()
        if warn:
            warnings.warn(
                f"Deleted corrupted cache file: {path}. "
                "It will be regenerated on next dataset initialization.",
                UserWarning,
            )
