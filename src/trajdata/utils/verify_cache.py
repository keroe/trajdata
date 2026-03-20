#!/usr/bin/env python3
"""Verify and optionally repair a trajdata cache directory.

Usage:
    python -m trajdata.utils.verify_cache /path/to/cache
    python -m trajdata.utils.verify_cache /path/to/cache --fix
    python -m trajdata.utils.verify_cache /path/to/cache --fix --workers 16

This walks through all cached files (.dill, .pkl, .pb) and attempts to load
each one. Corrupted files (truncated, unpicklable) are reported. With --fix,
they are deleted so that trajdata will regenerate only those specific files on
the next dataset initialization -- instead of requiring a full rebuild.
"""

import argparse
import os
import pickle
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Iterator, List, Optional, Tuple

import dill
from tqdm import tqdm

# File extensions we know how to verify
_VERIFIABLE_EXTENSIONS = {".dill", ".pkl", ".pb"}


def _iter_cache_files(cache_dir: Path) -> Iterator[Tuple[str, str]]:
    """Yield (path_str, file_type) for every verifiable file under cache_dir.

    Uses os.walk to stream results incrementally instead of collecting
    everything upfront with rglob, so the first file is yielded immediately.

    Returns strings (not Path objects) for cheap pickling across process boundaries.
    """
    for dirpath, _dirnames, filenames in os.walk(cache_dir):
        for filename in filenames:
            ext = os.path.splitext(filename)[1]
            if ext in _VERIFIABLE_EXTENSIONS:
                yield os.path.join(dirpath, filename), ext.lstrip(".")


def _verify_one(path_str: str, file_type: str) -> Tuple[str, str, bool, int]:
    """Verify a single cache file. Runs in a worker process.

    Returns:
        (path_str, file_type, is_valid, file_size_bytes)
    """
    try:
        size = os.path.getsize(path_str)
    except OSError:
        return path_str, file_type, False, 0

    if file_type == "dill":
        valid = _verify_dill(path_str)
    elif file_type == "pkl":
        valid = _verify_pickle(path_str)
    elif file_type == "pb":
        valid = size > 0
    else:
        valid = True

    return path_str, file_type, valid, size


def _verify_dill(path_str: str) -> bool:
    try:
        with open(path_str, "rb") as f:
            dill.load(f)
        return True
    except Exception:
        return False


def _verify_pickle(path_str: str) -> bool:
    try:
        with open(path_str, "rb") as f:
            pickle.load(f)
        return True
    except Exception:
        return False


def verify_cache(
    cache_dir: Path,
    fix: bool = False,
    verbose: bool = False,
    num_workers: int = 1,
) -> Tuple[int, int, int]:
    """Walk a trajdata cache directory and verify all serialized files.

    Args:
        cache_dir: Root cache directory (e.g. ~/.unified_data_cache).
        fix: If True, delete corrupted files.
        verbose: Print status for every file, not just corrupted ones.
        num_workers: Number of parallel worker processes for verification.

    Returns:
        Tuple of (total_files, corrupted_files, fixed_files).
    """
    total = 0
    corrupted = 0
    fixed = 0

    if not cache_dir.exists():
        print(f"ERROR: Cache directory does not exist: {cache_dir}")
        return 0, 0, 0

    cache_dir_str = str(cache_dir)

    if num_workers <= 1:
        # Single-process: stream and verify one at a time
        pbar = tqdm(
            _iter_cache_files(cache_dir),
            desc="Verifying cache",
            unit=" files",
            dynamic_ncols=True,
        )
        for path_str, file_type in pbar:
            total += 1
            pbar.set_postfix(checked=total, corrupt=corrupted, refresh=False)

            _, _, valid, size = _verify_one(path_str, file_type)
            _handle_result(
                path_str, valid, size, cache_dir_str, fix, verbose, pbar,
            )
            if not valid:
                corrupted += 1
                if fix:
                    fixed += 1
        pbar.close()
    else:
        # Multi-process: submit batches of files to a worker pool.
        # We feed files to the pool as os.walk discovers them and
        # process results as they complete.
        _BATCH = 256  # submit in chunks to reduce IPC overhead

        with ProcessPoolExecutor(max_workers=num_workers) as pool:
            pbar = tqdm(
                desc="Verifying cache",
                unit=" files",
                dynamic_ncols=True,
            )

            pending = []
            file_iter = _iter_cache_files(cache_dir)
            exhausted = False

            while not exhausted or pending:
                # Submit up to _BATCH new tasks
                while not exhausted and len(pending) < num_workers * 4:
                    batch: List[Tuple[str, str]] = []
                    for _ in range(_BATCH):
                        item = next(file_iter, None)
                        if item is None:
                            exhausted = True
                            break
                        batch.append(item)
                    for path_str, file_type in batch:
                        fut = pool.submit(_verify_one, path_str, file_type)
                        pending.append(fut)

                # Harvest completed futures
                if not pending:
                    break

                done = []
                still_pending = []
                for fut in pending:
                    if fut.done():
                        done.append(fut)
                    else:
                        still_pending.append(fut)

                # If nothing is done yet, block on the first one
                if not done and still_pending:
                    done.append(still_pending.pop(0))
                    done[0].result()  # blocks

                for fut in done:
                    path_str, file_type, valid, size = fut.result()
                    total += 1
                    pbar.update(1)
                    pbar.set_postfix(checked=total, corrupt=corrupted, refresh=False)

                    _handle_result(
                        path_str, valid, size, cache_dir_str, fix, verbose, pbar,
                    )
                    if not valid:
                        corrupted += 1
                        if fix:
                            fixed += 1

                pending = still_pending

            pbar.close()

    return total, corrupted, fixed


def _handle_result(
    path_str: str,
    valid: bool,
    size: int,
    cache_dir_str: str,
    fix: bool,
    verbose: bool,
    pbar: tqdm,
) -> None:
    """Print status and optionally delete a corrupted file."""
    if valid:
        if verbose:
            rel = os.path.relpath(path_str, cache_dir_str)
            tqdm.write(f"  OK       {rel}")
    else:
        rel = os.path.relpath(path_str, cache_dir_str)
        tqdm.write(f"  CORRUPT  {rel}  (size: {size} bytes)")
        if fix:
            try:
                os.unlink(path_str)
            except OSError:
                pass
            tqdm.write(f"           -> DELETED (will regenerate on next init)")


def main():
    default_workers = min(os.cpu_count() or 1, 8)

    parser = argparse.ArgumentParser(
        description="Verify and optionally repair a trajdata cache directory.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python -m trajdata.utils.verify_cache ~/.unified_data_cache
  python -m trajdata.utils.verify_cache /storage/.unified_cache --fix
  python -m trajdata.utils.verify_cache /storage/.unified_cache --fix --workers 16
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
        "--verbose", "-v",
        action="store_true",
        help="Print status for every file, not just corrupted ones.",
    )
    parser.add_argument(
        "--workers", "-j",
        type=int,
        default=default_workers,
        help=f"Number of parallel workers (default: {default_workers}).",
    )

    args = parser.parse_args()

    total, corrupted, fixed = verify_cache(
        args.cache_dir.expanduser().resolve(),
        fix=args.fix,
        verbose=args.verbose,
        num_workers=args.workers,
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
