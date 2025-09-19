#!/usr/bin/env python3
"""
Digital Alexandria — Dedupe Report from Recipe Files (Phase 1 analytics)

Scans a directory of *.recipe JSON files, aggregates block SHA-256 entries,
and computes:
  - total block references
  - unique blocks
  - duplicate references (total - unique)
  - logical bytes (sum of size per reference)
  - physical bytes (sum of size per unique block)
  - compression ratio (logical / physical) and savings

Assumptions & Notes
- Each recipe entry has "file_recipe" with:
    { "block_sequence_number": int,
      "block_sha256_value": str,
      "block_sha256_path": str, ... }
- We stat the .block file path to get its size. That’s the ground truth.
- If a .block is missing, we WARN and (by default) *exclude* its bytes from
  both logical and physical totals (but we still count references).
  You can opt to "assume-chunk-size" via CLI if you prefer.

Python: 3.9+
License: CC BY-SA 4.0
"""

from __future__ import annotations

import argparse
import concurrent.futures as cf
import json
import logging
import os
from pathlib import Path
from typing import Dict, Iterable, Tuple, List

# ---------------------------- Logging --------------------------------------

def setup_logging(log_dir: Path, verbose: bool) -> Path:
    log_dir.mkdir(parents=True, exist_ok=True)
    logfile = log_dir / "da_recipe_dedupe_report.log"

    root = logging.getLogger()
    root.handlers.clear()
    root.setLevel(logging.DEBUG)

    fh = logging.FileHandler(logfile, encoding="utf-8")
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(logging.Formatter(
        "%(asctime)s %(levelname)s %(threadName)s: %(message)s",
        "%Y-%m-%d %H:%M:%S"
    ))
    root.addHandler(fh)

    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO if verbose else logging.WARNING)
    ch.setFormatter(logging.Formatter("%(levelname)s: %(message)s"))
    root.addHandler(ch)

    logging.debug("Logging ready: %s", logfile)
    return logfile

# ---------------------------- Helpers --------------------------------------

def human_bytes(n: int) -> str:
    units = ["B","KiB","MiB","GiB","TiB","PiB","EiB"]
    f = float(n)
    for u in units:
        if abs(f) < 1024.0:
            return f"{f:,.2f} {u}"
        f /= 1024.0
    return f"{f:.2f} ZiB"  # if you somehow get here, congrats?

def find_recipe_files(root: Path) -> Iterable[Path]:
    for p in root.rglob("*.recipe"):
        if p.is_file():
            yield p

def load_recipe(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)

# -------------------------- Core Aggregation -------------------------------

def aggregate_block_refs(recipe_root: Path) -> Tuple[Dict[str, int], Dict[str, str], int, int]:
    """
    Returns:
      counts: sha256 -> reference count across all recipes
      any_path: sha256 -> one observed .block path (for stat)
      recipes_seen: number of recipe files parsed
      bad_recipes: recipes that failed to parse or had no file_recipe
    """
    counts: Dict[str, int] = {}
    any_path: Dict[str, str] = {}
    recipes_seen = 0
    bad_recipes = 0

    for rp in find_recipe_files(recipe_root):
        try:
            data = load_recipe(rp)
            recipes_seen += 1
            file_recipe = data.get("file_recipe")
            if not isinstance(file_recipe, list):
                logging.warning("No file_recipe array in %s", rp)
                bad_recipes += 1
                continue

            for entry in file_recipe:
                h = entry.get("block_sha256_value")
                bpath = entry.get("block_sha256_path")
                if not h:
                    logging.debug("Missing block_sha256_value in %s", rp)
                    continue
                counts[h] = counts.get(h, 0) + 1
                # prefer first usable path we see
                if h not in any_path and isinstance(bpath, str) and bpath:
                    any_path[h] = bpath

        except Exception as e:
            logging.warning("Failed to parse %s: %s", rp, e)
            bad_recipes += 1

    return counts, any_path, recipes_seen, bad_recipes

def stat_unique_block_sizes(
    any_path: Dict[str, str],
    workers: int,
    assume_chunk_size: int | None
) -> Tuple[Dict[str, int], List[str]]:
    """
    Stat each unique block once. If missing and assume_chunk_size is provided,
    use that size; else mark missing and skip its bytes.

    Returns:
      sizes: sha256 -> size in bytes
      missing: list of sha256 that couldn't be sized
    """
    sizes: Dict[str, int] = {}
    missing: List[str] = []

    def _stat_one(item):
        h, p = item
        try:
            st = os.stat(p)
            return (h, st.st_size, None)
        except FileNotFoundError:
            if assume_chunk_size is not None:
                return (h, assume_chunk_size, "assumed")
            return (h, None, "missing")
        except Exception as e:
            logging.debug("Stat error for %s (%s): %s", h, p, e)
            return (h, None, "missing")

    items = list(any_path.items())
    if workers > 1:
        with cf.ThreadPoolExecutor(max_workers=workers, thread_name_prefix="stat") as ex:
            for h, sz, status in ex.map(_stat_one, items):
                if sz is None:
                    missing.append(h)
                else:
                    sizes[h] = sz
    else:
        for it in items:
            h, sz, status = _stat_one(it)
            if sz is None:
                missing.append(h)
            else:
                sizes[h] = sz

    if assume_chunk_size is not None:
        # Missing set is only those truly missing; assumed sizes are already populated.
        pass

    return sizes, missing

# -------------------------- Report Generator -------------------------------

def write_csv(outfile: Path, counts: Dict[str,int], sizes: Dict[str,int]) -> None:
    """
    CSV columns:
      sha256,count,size_bytes,logical_bytes,physical_bytes
    """
    outfile.parent.mkdir(parents=True, exist_ok=True)
    with outfile.open("w", encoding="utf-8") as f:
        f.write("sha256,count,size_bytes,logical_bytes,physical_bytes\n")
        for h, c in counts.items():
            sz = sizes.get(h, 0)
            logical = c * sz
            physical = sz  # per-unique
            f.write(f"{h},{c},{sz},{logical},{physical}\n")

def summarize(counts: Dict[str,int], sizes: Dict[str,int]) -> dict:
    total_refs = sum(counts.values())
    unique = len(counts)
    dups = total_refs - unique

    # Logical bytes = sum over all refs: count(h) * size(h), but skip unknown sizes
    logical = 0
    physical = 0
    sized_hashes = 0
    for h, c in counts.items():
        if h in sizes:
            sz = sizes[h]
            logical += c * sz
            physical += sz
            sized_hashes += 1

    compression = (logical / physical) if physical > 0 else None
    savings = (1 - (physical / logical)) if logical > 0 else None

    return {
        "total_block_references": total_refs,
        "unique_blocks": unique,
        "duplicate_references": dups,
        "sized_unique_blocks": sized_hashes,
        "logical_bytes": logical,
        "physical_bytes": physical,
        "compression_ratio_logical_over_physical": compression,
        "space_savings_fraction": savings,
    }

# ------------------------------ CLI ----------------------------------------

def main():
    ap = argparse.ArgumentParser(
        description="Tabulate unique vs duplicate Digital Alexandria data blocks from .recipe files"
    )
    ap.add_argument("--recipe-root", type=Path, required=True,
                    help="Directory containing *.recipe files (e.g., /gpfs/Log/qr_proto/recipe)")
    ap.add_argument("--csv-out", type=Path, default=Path("./block_tabulation.csv"),
                    help="Where to write per-hash CSV (default: ./block_tabulation.csv)")
    ap.add_argument("--log-dir", type=Path, default=Path("./log"),
                    help="Directory for logs (default: ./log)")
    ap.add_argument("--workers", type=int, default=8,
                    help="Thread pool workers for stat() (default: 8)")
    ap.add_argument("--assume-chunk-size", type=int, default=None,
                    help="If a .block is missing, assume this size (bytes). Omit to exclude from byte totals.")
    ap.add_argument("--verbose", action="store_true", help="More chatty console output")
    args = ap.parse_args()

    setup_logging(args.log_dir, verbose=args.verbose)

    logging.info("Scanning recipes in %s", args.recipe_root)
    counts, any_path, recipes_seen, bad_recipes = aggregate_block_refs(args.recipe_root)
    logging.info("Parsed recipes: %d (bad: %d) | Unique hashes: %d | Total refs: %d",
                 recipes_seen, bad_recipes, len(counts), sum(counts.values()))

    logging.info("Stating unique block files (workers=%d)...", args.workers)
    sizes, missing = stat_unique_block_sizes(any_path, args.workers, args.assume_chunk_size)

    if missing and args.assume_chunk_size is None:
        logging.warning("Blocks missing on disk (sizes excluded from byte totals): %d", len(missing))

    summary = summarize(counts, sizes)

    # Pretty print summary
    print("\n=== Digital Alexandria — Dedupe Summary ===")
    print(f"Recipes parsed:                  {recipes_seen} (bad: {bad_recipes})")
    print(f"Total block references:          {summary['total_block_references']:,}")
    print(f"Unique blocks:                   {summary['unique_blocks']:,}")
    print(f"Duplicate references:            {summary['duplicate_references']:,}")
    print(f"Sized unique blocks:             {summary['sized_unique_blocks']:,}")
    print(f"Logical bytes (sized only):      {summary['logical_bytes']:,} "
          f"({human_bytes(summary['logical_bytes'])})")
    print(f"Physical bytes (sized only):     {summary['physical_bytes']:,} "
          f"({human_bytes(summary['physical_bytes'])})")
    cr = summary["compression_ratio_logical_over_physical"]
    sv = summary["space_savings_fraction"]
    if cr is not None:
        print(f"Compression ratio (L/P):         {cr:,.3f}x")
    else:
        print("Compression ratio (L/P):         n/a (no sized blocks)")
    if sv is not None:
        print(f"Space savings (1 - P/L):         {sv:.2%}")
    else:
        print("Space savings (1 - P/L):         n/a")
    print("==========================================\n")

    # CSV
    write_csv(args.csv_out, counts, sizes)
    logging.info("Per-hash CSV written: %s", args.csv_out)

if __name__ == "__main__":
    main()
