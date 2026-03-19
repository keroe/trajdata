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
import pickle
import sys
from pathlib import Path

import dill


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

    # Collect all serialized files
    dill_files = sorted(cache_dir.rglob("*.dill"))
    pkl_files = sorted(cache_dir.rglob("*.pkl"))
    pb_files = sorted(cache_dir.rglob("*.pb"))

    all_files = (
        [(f, "dill") for f in dill_files]
        + [(f, "pkl") for f in pkl_files]
        + [(f, "pb") for f in pb_files]
    )

    if not all_files:
        print(f"No cache files found in {cache_dir}")
        return 0, 0, 0

    print(f"Verifying {len(all_files)} cache files in {cache_dir}...")
    print(f"  {len(dill_files)} .dill files")
    print(f"  {len(pkl_files)} .pkl files")
    print(f"  {len(pb_files)} .pb files")
    print()

    for file_path, file_type in all_files:
        total += 1

        if file_type == "dill":
            valid = verify_dill_file(file_path)
        elif file_type == "pkl":
            valid = verify_pickle_file(file_path)
        elif file_type == "pb":
            valid = verify_protobuf_file(file_path)
        else:
            continue

        rel_path = file_path.relative_to(cache_dir)

        if valid:
            if verbose:
                print(f"  OK   {rel_path}")
        else:
            corrupted += 1
            size = file_path.stat().st_size if file_path.exists() else 0
            print(f"  CORRUPT  {rel_path}  (size: {size} bytes)")

            if fix:
                file_path.unlink(missing_ok=True)
                fixed += 1
                print(f"           -> DELETED (will regenerate on next init)")

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
