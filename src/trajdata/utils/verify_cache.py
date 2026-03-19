#!/usr/bin/env python3
"""Verify and optionally repair a trajdata cache directory.

Usage:
    python -m trajdata.utils.verify_cache /path/to/cache
    python -m trajdata.utils.verify_cache /path/to/cache --fix
    python -m trajdata.utils.verify_cache /path/to/cache --fix --verbose

This walks through all cached files (.dill, .pkl, .pb) and attempts to load
each one. Corrupted files (truncated, unpicklable) are reported. With --fix,
they are deleted so that trajdata will regenerate only those specific files on
the next dataset initialization -- instead of requiring a full rebuild.
"""

import argparse
import os
import pickle
import sys
from pathlib import Path
from typing import Iterator, Tuple

import dill
from tqdm import tqdm

# File extensions we know how to verify
_VERIFIABLE_EXTENSIONS = {".dill", ".pkl", ".pb"}


def _iter_cache_files(cache_dir: Path) -> Iterator[Tuple[Path, str]]:
    """Yield (path, file_type) for every verifiable file under cache_dir.

    Uses os.walk to stream results incrementally instead of collecting
    everything upfront with rglob, so the first file is yielded immediately.
    """
    for dirpath, _dirnames, filenames in os.walk(cache_dir):
        for filename in filenames:
            ext = os.path.splitext(filename)[1]
            if ext in _VERIFIABLE_EXTENSIONS:
                yield Path(dirpath) / filename, ext.lstrip(".")


def verify_dill_file(path: Path) -> bool:
    """Attempt to load a dill file. Returns True if valid, False if corrupted."""
    try:
        with open(path, "rb") as f:
            dill.load(f)
        return True
    except (EOFError, pickle.UnpicklingError, Exception) as e:
        if isinstance(e, (EOFError, pickle.UnpicklingError)):
            return False
        # Other exceptions (e.g. ModuleNotFoundError) also indicate corruption
        # in the context of cache verification
        return False


def verify_pickle_file(path: Path) -> bool:
    """Attempt to load a pickle file. Returns True if valid, False if corrupted."""
    try:
        with open(path, "rb") as f:
            pickle.load(f)
        return True
    except (EOFError, pickle.UnpicklingError):
        return False


def verify_protobuf_file(path: Path) -> bool:
    """Check that a protobuf file is non-empty and has content."""
    try:
        size = path.stat().st_size
        return size > 0
    except OSError:
        return False


def verify_cache(
    cache_dir: Path,
    fix: bool = False,
    verbose: bool = False,
) -> tuple:
    """Walk a trajdata cache directory and verify all serialized files.

    Args:
        cache_dir: Root cache directory (e.g. ~/.unified_data_cache).
        fix: If True, delete corrupted files.
        verbose: Print status for every file, not just corrupted ones.

    Returns:
        Tuple of (total_files, corrupted_files, fixed_files).
    """
    total = 0
    corrupted = 0
    fixed = 0

    if not cache_dir.exists():
        print(f"ERROR: Cache directory does not exist: {cache_dir}")
        return 0, 0, 0

    corrupted_paths = []

    pbar = tqdm(
        _iter_cache_files(cache_dir),
        desc="Verifying cache",
        unit=" files",
        dynamic_ncols=True,
    )

    for file_path, file_type in pbar:
        total += 1
        pbar.set_postfix(checked=total, corrupt=corrupted, refresh=False)

        if file_type == "dill":
            valid = verify_dill_file(file_path)
        elif file_type == "pkl":
            valid = verify_pickle_file(file_path)
        elif file_type == "pb":
            valid = verify_protobuf_file(file_path)
        else:
            continue

        if valid:
            if verbose:
                rel_path = file_path.relative_to(cache_dir)
                tqdm.write(f"  OK       {rel_path}")
        else:
            corrupted += 1
            pbar.set_postfix(checked=total, corrupt=corrupted, refresh=True)
            rel_path = file_path.relative_to(cache_dir)
            size = file_path.stat().st_size if file_path.exists() else 0
            tqdm.write(f"  CORRUPT  {rel_path}  (size: {size} bytes)")

            if fix:
                file_path.unlink(missing_ok=True)
                fixed += 1
                tqdm.write(
                    f"           -> DELETED (will regenerate on next init)"
                )
            corrupted_paths.append(rel_path)

    pbar.close()

    return total, corrupted, fixed


def main():
    parser = argparse.ArgumentParser(
        description="Verify and optionally repair a trajdata cache directory.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python -m trajdata.utils.verify_cache ~/.unified_data_cache
  python -m trajdata.utils.verify_cache /storage/.unified_cache --fix
  python -m trajdata.utils.verify_cache /storage/.unified_cache --fix --verbose
        """,
    )
    parser.add_argument(
        "cache_dir",
        type=Path,
        help="Path to the trajdata cache directory.",
    )
    parser.add_argument(
        "--fix",
        action="store_true",
        help="Delete corrupted files so they can be regenerated.",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print status for every file, not just corrupted ones.",
    )

    args = parser.parse_args()

    total, corrupted, fixed = verify_cache(
        args.cache_dir.expanduser().resolve(),
        fix=args.fix,
        verbose=args.verbose,
    )

    print()
    print(f"Results: {total} files checked, {corrupted} corrupted", end="")
    if args.fix:
        print(f", {fixed} deleted for regeneration", end="")
    print()

    if corrupted > 0 and not args.fix:
        print()
        print("Run with --fix to delete corrupted files.")
        print("After fixing, re-initialize your dataset to regenerate only the")
        print("missing files (NOT a full rebuild).")

    sys.exit(1 if corrupted > 0 and not args.fix else 0)


if __name__ == "__main__":
    main()
