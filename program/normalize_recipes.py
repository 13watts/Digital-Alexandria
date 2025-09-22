#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Digital Alexandria — Recipe Normalizer & Validator
Python 3.9+

Functions
---------
- Validate that .recipe files conform to the simplified schema.
- Normalize legacy/verbose recipes to the simplified schema:
  * file_recipe entries: keep only {block_sequence_number, block_sha256_value}
  * put paths & flags once under modality_options
  * ensure trailing slashes on *_path fields
  * sort blocks by sequence

CLI Modes
---------
- --validate-only            : read-only validation (no writes, exit 1 on any invalid)
- --fix                      : write normalized/fixed recipes (in-place or mirrored)
- (default normalization)    : like --fix, but also accepts explicit roots/flags

Logging
-------
- INFO to stdout
- DEBUG to timestamped logfile in --log-dir

License
-------
Creative Commons Attribution-ShareAlike 4.0 (CC BY-SA 4.0)
"""

from __future__ import annotations

import argparse
import concurrent.futures as cf
import datetime as dt
import json
import logging
import os
import sys
from pathlib import Path
from typing import Dict, Optional, Tuple

PAIRTREE_LEVELS = 32  # two-hex levels for SHA-256 pair-tree


# =============================== Logging ====================================

def setup_logging(log_dir: Path, verbose: bool) -> Path:
    log_dir.mkdir(parents=True, exist_ok=True)
    stamp = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    logfile = log_dir / f"normalize_recipes_{stamp}_{os.getpid()}.log"

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

    ch = logging.StreamHandler(sys.stdout)
    ch.setLevel(logging.INFO if verbose else logging.WARNING)
    ch.setFormatter(logging.Formatter("%(levelname)s: %(message)s"))
    root.addHandler(ch)

    logging.debug("Logging initialized; logfile=%s", logfile)
    return logfile


# ================================ I/O =======================================

def load_json(path: Path) -> Optional[dict]:
    try:
        with path.open("r", encoding="utf-8") as f:
            return json.load(f)
    except Exception as e:
        logging.warning("Failed to read %s: %s", path, e)
        return None


def write_json(path: Path, data: dict) -> bool:
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(path.suffix + ".tmp")
        tmp.write_text(json.dumps(data, indent=2), encoding="utf-8")
        os.replace(tmp, path)
        return True
    except Exception as e:
        logging.error("Failed to write %s: %s", path, e)
        return False


# ============================== Helpers =====================================

def _ensure_trailing_slash(p: Optional[str]) -> Optional[str]:
    if p is None:
        return None
    return p if p.endswith("/") else p + "/"


def infer_pairtree_root(full_path: str) -> Optional[str]:
    """
    Given /root/aa/bb/.../<hash>.ext -> return '/root/'
    Ascend PAIRTREE_LEVELS from the leaf directory.
    """
    try:
        p = Path(full_path)
        parent = p.parent
        for _ in range(PAIRTREE_LEVELS):
            parent = parent.parent
        return _ensure_trailing_slash(str(parent))
    except Exception:
        return None


def infer_roots_from_recipe(old: dict) -> Dict[str, Optional[str]]:
    roots = {"blocks": None, "qr": None, "dna": None, "protein": None}
    fr = old.get("file_recipe")
    if not isinstance(fr, list) or not fr:
        return roots
    first = fr[0]
    bpath = first.get("block_sha256_path")
    if isinstance(bpath, str):
        roots["blocks"] = infer_pairtree_root(bpath)
    mods = first.get("modalities") or {}
    if isinstance(mods.get("qrcode_svg"), str):
        roots["qr"] = infer_pairtree_root(mods["qrcode_svg"])
    if isinstance(mods.get("dna"), str):
        roots["dna"] = infer_pairtree_root(mods["dna"])
    if isinstance(mods.get("protein"), str):
        roots["protein"] = infer_pairtree_root(mods["protein"])
    return roots


def _sorted_blocks_minimal(old: dict) -> list:
    out = []
    for entry in old.get("file_recipe", []):
        if not isinstance(entry, dict):
            continue
        seq = entry.get("block_sequence_number")
        h = entry.get("block_sha256_value")
        # loose fallbacks for truly old shapes
        if seq is None:
            seq = entry.get("sequence")
        if h is None:
            h = entry.get("sha256")
        if seq is None or h is None:
            logging.debug("Skipping malformed block entry: %s", entry)
            continue
        out.append({"block_sequence_number": int(seq), "block_sha256_value": str(h)})
    out.sort(key=lambda e: e["block_sequence_number"])
    return out


# ============================== Schema ======================================

REQUIRED_TOP_KEYS = {
    "process_uuid",
    "source_file_name",
    "source_file_sha256",
    "file_recipe",
    "modality_options",
}

REQUIRED_MOD_KEYS = {
    "data_block",
    "data_block_path",
    "qr_code",
    "qr_path",
    "qr_code_type",
    "dna_recipe",
    "dna_path",
    "protein_recipe",
    "protein_path",
}

ALLOWED_BLOCK_KEYS = {"block_sequence_number", "block_sha256_value"}


def validate_recipe(path: Path, obj: dict) -> Tuple[bool, str]:
    # top-level
    for k in REQUIRED_TOP_KEYS:
        if k not in obj:
            return (False, f"missing key {k}")

    # file_recipe
    fr = obj.get("file_recipe")
    if not isinstance(fr, list) or not fr:
        return (False, "file_recipe missing or empty")
    for i, e in enumerate(fr):
        if not isinstance(e, dict):
            return (False, f"file_recipe[{i}] not an object")
        keys = set(e.keys())
        if keys != ALLOWED_BLOCK_KEYS:
            return (False, f"file_recipe[{i}] invalid keys: {keys}")

    # modality_options
    mod = obj.get("modality_options", {})
    for k in REQUIRED_MOD_KEYS:
        if k not in mod:
            return (False, f"missing modality_options.{k}")
    # QR type sub-keys sanity
    qrt = mod.get("qr_code_type", {})
    if not isinstance(qrt, dict) or "qr_code_type" not in qrt or "qr_code_ecc" not in qrt:
        return (False, "modality_options.qr_code_type missing qr_code_type/qr_code_ecc")

    # trailing slash nicety (warn only)
    for pkey in ["data_block_path", "qr_path", "dna_path", "protein_path"]:
        val = mod.get(pkey)
        if isinstance(val, str) and not val.endswith("/"):
            logging.debug("path missing trailing slash (tolerated): %s", pkey)

    return (True, "ok")


# =========================== Build New Object ===============================

def build_new_recipe(
    old: dict,
    blocks_root: str,
    qr_root: Optional[str],
    dna_root: Optional[str],
    protein_root: Optional[str],
    qr_version: int,
    qr_ecc: str,
    do_qr: bool,
    do_dna: bool,
    do_protein: bool,
) -> dict:
    process_uuid = old.get("process_uuid")
    src_loc = old.get("source_file_location")
    src_name = old.get("source_file_name")
    src_date = old.get("source_file_creation_date")
    src_sha = old.get("source_file_sha256")
    chunk_sz = old.get("chunk_size_bytes")

    file_recipe = _sorted_blocks_minimal(old)

    modality_options = {
        "data_block": True,
        "data_block_path": _ensure_trailing_slash(blocks_root),
        "qr_code": bool(do_qr),
        "qr_path": _ensure_trailing_slash(qr_root) if qr_root else None,
        "qr_code_type": {
            "qr_code_type": f"Model 2, Version {qr_version}",
            "qr_code_ecc": qr_ecc,
        },
        "dna_recipe": bool(do_dna),
        "dna_path": _ensure_trailing_slash(dna_root) if dna_root else None,
        "protein_recipe": bool(do_protein),
        "protein_path": _ensure_trailing_slash(protein_root) if protein_root else None,
    }

    new_obj = {
        "process_uuid": process_uuid,
        "source_file_location": src_loc,
        "source_file_name": src_name,
        "source_file_creation_date": src_date,
        "source_file_sha256": src_sha,
        "modality_options": modality_options,
        "file_recipe": file_recipe,
        "notes": "Normalized to simplified recipe format; paths declared once in modality_options.",
    }
    if chunk_sz is not None:
        new_obj["chunk_size_bytes"] = chunk_sz
    return new_obj


def normalize_one(
    in_path: Path,
    in_root: Path,
    out_root: Path,
    in_place: bool,
    cli_blocks_root: Optional[str],
    cli_qr_root: Optional[str],
    cli_dna_root: Optional[str],
    cli_prot_root: Optional[str],
    qr_version: int,
    qr_ecc: str,
    do_qr: bool,
    do_dna: bool,
    do_protein: bool,
    dry_run: bool,
) -> Tuple[Path, bool]:
    old = load_json(in_path)
    if old is None:
        return (in_path, False)

    inferred = infer_roots_from_recipe(old)
    blocks_root = _ensure_trailing_slash(cli_blocks_root or inferred.get("blocks") or "blocks/")
    qr_root = _ensure_trailing_slash(cli_qr_root or inferred.get("qr") or "qrcode/")
    dna_root = _ensure_trailing_slash(cli_dna_root or inferred.get("dna") or "dna/")
    protein_root = _ensure_trailing_slash(cli_prot_root or inferred.get("protein") or "protein/")

    new_obj = build_new_recipe(
        old,
        blocks_root=blocks_root,
        qr_root=qr_root,
        dna_root=dna_root,
        protein_root=protein_root,
        qr_version=qr_version,
        qr_ecc=qr_ecc,
        do_qr=do_qr,
        do_dna=do_dna,
        do_protein=do_protein,
    )

    rel = in_path.relative_to(in_root)
    out_path = in_path if in_place else (out_root / rel)

    if dry_run:
        logging.info("[dry-run] would write normalized -> %s", out_path)
        return (out_path, True)

    ok = write_json(out_path, new_obj)
    if ok:
        # Re-validate what we wrote (belt-and-suspenders)
        new_loaded = load_json(out_path)
        if new_loaded is None:
            logging.error("Wrote but could not re-read %s", out_path)
            return (out_path, False)
        valid, msg = validate_recipe(out_path, new_loaded)
        if not valid:
            logging.error("Post-write validation FAILED %s: %s", out_path, msg)
            return (out_path, False)
    return (out_path, ok)


# ================================ CLI =======================================

def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(
        description="Normalize & validate Digital Alexandria recipe files"
    )
    ap.add_argument("--input-root", type=Path, required=True,
                    help="Root directory containing .recipe files")
    ap.add_argument("--output-root", type=Path, default=None,
                    help="Root to write normalized recipes (mirrors tree). "
                         "Omit with --in-place to overwrite originals.")
    ap.add_argument("--in-place", action="store_true",
                    help="Overwrite existing .recipe files in place")
    ap.add_argument("--validate-only", action="store_true",
                    help="Validate recipes without writing changes")
    ap.add_argument("--fix", action="store_true",
                    help="Auto-normalize invalid/legacy recipes to new schema")

    # Optional explicit roots (used for normalization/fix)
    ap.add_argument("--data-block-path", type=str, default=None,
                    help="Root path for blocks (e.g., blocks/)")
    ap.add_argument("--qr-path", type=str, default=None,
                    help="Root path for qrcode SVGs")
    ap.add_argument("--dna-path", type=str, default=None,
                    help="Root path for DNA modality")
    ap.add_argument("--protein-path", type=str, default=None,
                    help="Root path for protein modality")

    # Modality flags + QR settings defaults (used for normalization/fix)
    ap.add_argument("--no-qr", action="store_true", help="Declare QR disabled in outputs")
    ap.add_argument("--no-dna", action="store_true", help="Declare DNA disabled in outputs")
    ap.add_argument("--no-protein", action="store_true", help="Declare Protein disabled in outputs")
    ap.add_argument("--qr-version", type=int, default=40, help="QR version to declare (default: 40)")
    ap.add_argument("--qr-ecc", choices=list("LMQH"), default="H", help="QR ECC level (default: H)")

    # Ops
    ap.add_argument("--workers", type=int, default=8, help="Parallel workers (default: 8)")
    ap.add_argument("--log-dir", type=Path, default=Path("./log"), help="Log directory (default: ./log)")
    ap.add_argument("--verbose", action="store_true", help="Chatty console output")
    ap.add_argument("--dry-run", action="store_true", help="Plan only; do not write files")

    args = ap.parse_args()

    # Mutual exclusivity rules:
    # - validate-only without --fix => no writes allowed
    if args.validate_only and not args.fix:
        if args.in_place or args.output_root is not None or args.dry_run:
            raise SystemExit("--validate-only cannot be combined with --in-place/--output-root/--dry-run unless --fix is set")

    # Default output-root when writing and not in-place
    writing = args.fix and not args.validate_only or args.fix and args.validate_only
    if writing and not args.in_place and args.output_root is None:
        args.output_root = args.input_root.parent / (args.input_root.name + "_normalized")

    # in-place vs output-root sanity
    if args.in_place and args.output_root is not None:
        raise SystemExit("--in-place and --output-root are mutually exclusive")

    return args


# ================================ Main ======================================

def main():
    args = parse_args()
    logfile = setup_logging(args.log_dir, verbose=args.verbose)

    in_root = args.input_root
    out_root = args.output_root if args.output_root is not None else in_root
    do_qr = not args.no_qr
    do_dna = not args.no_dna
    do_protein = not args.no_protein

    recipes = [p for p in in_root.rglob("*.recipe") if p.is_file()]
    if not recipes:
        logging.warning("No .recipe files found under %s", in_root)
        return

    logging.info(
        "normalize_recipes starting; recipes=%d | logfile=%s | modes: validate_only=%s fix=%s in_place=%s",
        len(recipes), logfile, args.validate_only, args.fix, args.in_place
    )

    # --- VALIDATE-ONLY (no fix) ---
    if args.validate_only and not args.fix:
        valid = 0
        invalid = 0
        with cf.ThreadPoolExecutor(max_workers=args.workers, thread_name_prefix="validate") as ex:
            for path in recipes:
                obj = load_json(path)
                if obj is None:
                    invalid += 1
                    logging.error("INVALID %s: unreadable", path)
                    continue
                ok, msg = validate_recipe(path, obj)
                if ok:
                    valid += 1
                    logging.debug("VALID %s", path)
                else:
                    invalid += 1
                    logging.error("INVALID %s: %s", path, msg)
        logging.info("Validation complete: valid=%d invalid=%d", valid, invalid)
        sys.exit(0 if invalid == 0 else 1)

    # --- FIX / NORMALIZE PATH ---
    in_place = bool(args.in_place)
    ok_count = 0
    fail_count = 0
    fixed_count = 0
    skipped_valid = 0

    def _fix_or_normalize(rp: Path) -> Tuple[Path, bool, bool]:
        obj = load_json(rp)
        if obj is None:
            return (rp, False, False)
        ok, msg = validate_recipe(rp, obj)
        if ok and args.fix:
            # already valid: either copy through (if mirroring) or skip
            if not in_place and out_root != in_root:
                rel = rp.relative_to(in_root)
                dest = out_root / rel
                if args.dry_run:
                    logging.info("[dry-run] would copy-through valid -> %s", dest)
                    return (dest, True, False)
                # write an identical file (still apply trailing slash normalization if desired? keep as-is)
                success = write_json(dest, obj)
                return (dest, success, False)
            else:
                # in-place & valid: nothing to do
                return (rp, True, False)

        # Either invalid OR (valid but user wants normalized output regardless)
        out_path, success = normalize_one(
            rp, in_root, out_root, in_place,
            args.data_block_path, args.qr_path, args.dna_path, args.protein_path,
            args.qr_version, args.qr_ecc,
            do_qr, do_dna, do_protein,
            args.dry_run,
        )
        return (out_path, success, True)

    with cf.ThreadPoolExecutor(max_workers=args.workers, thread_name_prefix="fix") as ex:
        futures = [ex.submit(_fix_or_normalize, rp) for rp in recipes]
        for fut in cf.as_completed(futures):
            out_path, success, changed = fut.result()
            if success:
                ok_count += 1
                if changed:
                    fixed_count += 1
                else:
                    skipped_valid += 1
            else:
                fail_count += 1
                logging.error("Failed processing -> %s", out_path)

    logging.info(
        "Normalization/Fix complete: ok=%d fixed=%d skipped_valid=%d failed=%d",
        ok_count, fixed_count, skipped_valid, fail_count
    )
    # Non-zero exit if anything failed
    sys.exit(0 if fail_count == 0 else 1)


if __name__ == "__main__":
    main()
