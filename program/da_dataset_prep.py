#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Digital Alexandria — Dataset Prep Phase (Updated)
Python 3.9+

What it does
------------
- Crawl an input directory for source files.
- For each file:
  * Assign a process UUID, compute file SHA-256.
  * Chunk into <=1 KiB blocks; for each block compute SHA-256 and write
    the .block into a deep pair-tree directory under --blocks-root.
  * If a block already exists, skip re-writing and skip modalities.
  * For NEW blocks (only), optionally generate modalities in parallel:
      - QR Code (Model 2, Version 40, ECC H) as .svg
      - DNA sequence (.dna)
      - Protein sequence (.protein)
  * Emit a per-file .recipe JSON with simplified entries:
      - Each block entry includes only sequence + block_sha256_value.
      - modality_options contains root paths and flags once.

Logging
-------
- INFO to console, DEBUG to ./log/ by default.

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
import uuid
from hashlib import sha256
from pathlib import Path
from typing import Iterable, List, Tuple
from zlib import crc32

import segno  # QR code generator

# ----------------------- Defaults (override via CLI) -----------------------

DEFAULT_INPUT_ROOT   = "./source"
DEFAULT_BLOCKS_ROOT  = "./blocks"
DEFAULT_QR_ROOT      = "./qrcode"
DEFAULT_DNA_ROOT     = "./dna"
DEFAULT_PROT_ROOT    = "./protein"
DEFAULT_RECIPE_ROOT  = "./recipe"
DEFAULT_LOG_DIR      = "./log"

# ----------------------------- Utilities ----------------------------------

def setup_logging(log_dir: Path, verbose: bool = True) -> Path:
    log_dir.mkdir(parents=True, exist_ok=True)
    stamp = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    logfile = log_dir / f"dataset_prep_{stamp}_{os.getpid()}.log"

    logger = logging.getLogger()
    logger.handlers.clear()
    logger.setLevel(logging.DEBUG)

    # File handler (DEBUG)
    fh = logging.FileHandler(logfile, encoding="utf-8")
    fh.setLevel(logging.DEBUG)
    ffmt = logging.Formatter(
        "%(asctime)s %(levelname)s %(threadName)s %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    fh.setFormatter(ffmt)
    logger.addHandler(fh)

    # Console handler (INFO/WARN)
    ch = logging.StreamHandler(sys.stdout)
    ch.setLevel(logging.INFO if verbose else logging.WARNING)
    cfmt = logging.Formatter("%(levelname)s: %(message)s")
    ch.setFormatter(cfmt)
    logger.addHandler(ch)

    logging.debug("Logging initialized; logfile=%s", logfile)
    return logfile


def sha256_file(path: Path, bufsize: int = 1024 * 1024) -> str:
    h = sha256()
    with path.open("rb", buffering=0) as f:
        while True:
            b = f.read(bufsize)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def iter_chunks(path: Path, chunk_size: int) -> Iterable[bytes]:
    with path.open("rb", buffering=0) as f:
        while True:
            b = f.read(chunk_size)
            if not b:
                break
            yield b


def pairtree_path(root: Path, hexhash: str, basename_with_ext: str) -> Path:
    """
    Build a deep pair-tree path from a 64-char hex string:
    /root/aa/bb/cc/.../<hexhash>.<ext>
    """
    assert len(hexhash) >= 64, "Expected 64 hex chars for SHA-256"
    parts = [hexhash[i:i+2] for i in range(0, 64, 2)]
    d = root.joinpath(*parts)
    d.mkdir(parents=True, exist_ok=True)
    return d / basename_with_ext


def file_mtime_iso(path: Path) -> str:
    # Linux "creation" is unreliable; use mtime as proxy
    ts = path.stat().st_mtime
    return dt.datetime.fromtimestamp(ts).isoformat(timespec="seconds")


def ensure_trailing_slash(p: Path) -> str:
    s = str(p)
    return s if s.endswith("/") else s + "/"

# --------------------------- Modalities ------------------------------------

DNA_MAP = ("A", "C", "G", "T")
AA16 = ("A","C","D","E","F","G","H","I","K","L","M","N","P","Q","R","S")  # 0..15

def block_to_dna(block: bytes) -> str:
    # 2 bits -> one nucleotide; 8 bits -> 4 nt per byte
    out = []
    for b in block:
        out.append(DNA_MAP[(b >> 6) & 0b11])
        out.append(DNA_MAP[(b >> 4) & 0b11])
        out.append(DNA_MAP[(b >> 2) & 0b11])
        out.append(DNA_MAP[b & 0b11])
    return "".join(out)

def block_to_protein(block: bytes) -> str:
    # 4-bit nibble -> amino-acid from AA16
    out = []
    for b in block:
        out.append(AA16[(b >> 4) & 0x0F])
        out.append(AA16[b & 0x0F])
    return "".join(out)

def make_qr_payload(block_hash_hex: str, block: bytes) -> bytes:
    """
    Compact binary frame:
    [64 ASCII hex chars of sha256]
    [raw block bytes (<=1024)]
    [CRC32 (big-endian 4 bytes) computed over the raw block bytes only]
    """
    crc = crc32(block) & 0xFFFFFFFF
    tail = crc.to_bytes(4, "big")
    return block_hash_hex.encode("ascii") + block + tail

def write_qr_svg(qr_root: Path, hexhash: str, payload: bytes, version: int, ecc: str) -> Path:
    # segno uses versions 1..40, error parameters 'L','M','Q','H'
    qr = segno.make(payload, version=version, error=ecc, mode="byte")
    svg_path = pairtree_path(qr_root, hexhash, f"{hexhash}.svg")
    qr.save(svg_path, scale=1)  # vector output; microscope-friendly
    return svg_path

def write_text_modality(root: Path, hexhash: str, ext: str, content: str) -> Path:
    out = pairtree_path(root, hexhash, f"{hexhash}.{ext}")
    out.write_text(content, encoding="utf-8")
    return out

# ------------------------------- Worker ------------------------------------

def process_block(
    block_bytes: bytes,
    seq: int,
    blocks_root: Path,
    qr_root: Path,
    dna_root: Path,
    prot_root: Path,
    qr_version: int,
    qr_ecc: str,
    do_qr: bool,
    do_dna: bool,
    do_prot: bool,
) -> Tuple[int, str, bool]:
    """
    Process a single block:
      - compute sha256
      - write .block if absent
      - if newly written, optionally generate QR/DNA/Protein
    Returns: (sequence_number, block_sha256_hex, wrote_block_bool)
    """
    b_hash = sha256(block_bytes).hexdigest()
    block_path = pairtree_path(blocks_root, b_hash, f"{b_hash}.block")

    wrote_block = False
    if not block_path.exists():
        with open(block_path, "wb", buffering=0) as f:
            f.write(block_bytes)
        wrote_block = True
        logging.debug("Wrote block #%d -> %s", seq, block_path)

        # Only generate modalities if requested; otherwise keep it classy and quiet.
        if do_qr:
            try:
                payload = make_qr_payload(b_hash, block_bytes)
                write_qr_svg(qr_root, b_hash, payload, qr_version, qr_ecc)
            except Exception:
                logging.exception("QR generation failed for block %s", b_hash)
        if do_dna:
            try:
                dna_seq = block_to_dna(block_bytes)
                write_text_modality(dna_root, b_hash, "dna", dna_seq)
            except Exception:
                logging.exception("DNA generation failed for block %s", b_hash)
        if do_prot:
            try:
                prot_seq = block_to_protein(block_bytes)
                write_text_modality(prot_root, b_hash, "protein", prot_seq)
            except Exception:
                logging.exception("Protein generation failed for block %s", b_hash)
    else:
        logging.debug("Block %s exists; skipping write & modalities", b_hash)

    return (seq, b_hash, wrote_block)

# ------------------------------- Per-file ----------------------------------

def process_one_file(
    src_path: Path,
    input_root: Path,
    blocks_root: Path,
    qr_root: Path,
    dna_root: Path,
    prot_root: Path,
    recipe_root: Path,
    chunk_size: int,
    file_uuid: str,
    qr_version: int,
    qr_ecc: str,
    do_qr: bool,
    do_dna: bool,
    do_prot: bool,
    modality_pool: cf.Executor,
) -> None:
    rel = src_path.relative_to(input_root)

    logging.info("Processing: %s (UUID=%s)", src_path, file_uuid)

    try:
        file_hash = sha256_file(src_path)
        created_iso = file_mtime_iso(src_path)
        logging.debug("File SHA-256: %s | mtime: %s", file_hash, created_iso)
    except Exception as e:
        logging.exception("Failed to hash/inspect %s: %s", src_path, e)
        return

    # Chunk and schedule blocks
    futures: List[cf.Future] = []
    seq = 0
    for chunk in iter_chunks(src_path, chunk_size):
        fut = modality_pool.submit(
            process_block,
            chunk, seq,
            blocks_root, qr_root, dna_root, prot_root,
            qr_version, qr_ecc,
            do_qr, do_dna, do_prot
        )
        futures.append(fut)
        seq += 1

    # Collect
    results: List[Tuple[int, str, bool]] = []
    for fut in cf.as_completed(futures):
        try:
            results.append(fut.result())
        except Exception as e:
            logging.exception("Block task failed for %s: %s", src_path, e)

    # Sort by sequence
    results.sort(key=lambda t: t[0])

    # Build recipe JSON path and parent
    recipe_out = recipe_root / rel.with_suffix(rel.suffix + ".recipe")
    recipe_out.parent.mkdir(parents=True, exist_ok=True)

    # Construct simplified recipe object
    recipe = {
        "process_uuid": file_uuid,
        "source_file_location": str(src_path.parent),
        "source_file_name": src_path.name,
        "source_file_creation_date": created_iso,
        "source_file_sha256": file_hash,
        "modality_options": {
            "data_block": True,
            "data_block_path": ensure_trailing_slash(blocks_root),
            "qr_code": bool(do_qr),
            "qr_path": ensure_trailing_slash(qr_root),
            "qr_code_type": {
                "qr_code_type": f"Model 2, Version {qr_version}",
                "qr_code_ecc": qr_ecc,
            },
            "dna_recipe": bool(do_dna),
            "dna_path": ensure_trailing_slash(dna_root),
            "protein_recipe": bool(do_prot),
            "protein_path": ensure_trailing_slash(prot_root),
        },
        "file_recipe": [
            {
                "block_sequence_number": seq_i,
                "block_sha256_value": sha_i,
            }
            for (seq_i, sha_i, _wrote) in results
        ],
        "chunk_size_bytes": chunk_size,
        "notes": "Blocks are addressed by SHA-256; modality files (if enabled) can be derived from paths + hash.",
    }

    recipe_out.write_text(json.dumps(recipe, indent=2), encoding="utf-8")
    logging.info("Recipe written: %s (blocks=%d)", recipe_out, len(results))

# ------------------------------- Discovery ---------------------------------

def discover_files(root: Path) -> Iterable[Path]:
    for p in root.rglob("*"):
        if p.is_file():
            yield p

# ---------------------------------- CLI ------------------------------------

def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(
        description="Digital Alexandria — Dataset Prep (Phase 1, updated with modality switches and simplified recipe)"
    )
    ap.add_argument("--input-root", default=DEFAULT_INPUT_ROOT, type=Path)
    ap.add_argument("--blocks-root", default=DEFAULT_BLOCKS_ROOT, type=Path)
    ap.add_argument("--qrcode-root", default=DEFAULT_QR_ROOT, type=Path)
    ap.add_argument("--dna-root", default=DEFAULT_DNA_ROOT, type=Path)
    ap.add_argument("--protein-root", default=DEFAULT_PROT_ROOT, type=Path)
    ap.add_argument("--recipe-root", default=DEFAULT_RECIPE_ROOT, type=Path)
    ap.add_argument("--log-dir", default=DEFAULT_LOG_DIR, type=Path)

    ap.add_argument("--chunk-size", type=int, default=1024, help="Block size in bytes (<=1024 recommended)")
    ap.add_argument("--file-workers", type=int, default=4, help="Parallel source files processed at a time")
    ap.add_argument("--modality-workers", type=int, default=8, help="Parallel workers for block write + modalities")
    ap.add_argument("--qr-version", type=int, default=40, help="QR version (fixed 40 per spec)")
    ap.add_argument("--qr-ecc", choices=list("LMQH"), default="H", help="QR error correction level")

    # New: switches to disable modalities
    ap.add_argument("--no-qr", action="store_true", help="Disable QR code modality")
    ap.add_argument("--no-dna", action="store_true", help="Disable DNA modality")
    ap.add_argument("--no-protein", action="store_true", help="Disable Protein modality")

    ap.add_argument("--verbose", action="store_true", help="More chatty console")
    return ap.parse_args()

# --------------------------------- Main ------------------------------------

def main():
    args = parse_args()
    logfile = setup_logging(args.log_dir, verbose=True)

    # Ensure roots exist (even if modalities disabled, paths still published in recipe)
    for d in (args.blocks_root, args.qrcode_root, args.dna_root, args.protein_root, args.recipe_root):
        d.mkdir(parents=True, exist_ok=True)

    files = list(discover_files(args.input_root))
    if not files:
        logging.warning("No files found under %s", args.input_root)
        return

    do_qr = not args.no_qr
    do_dna = not args.no_dna
    do_prot = not args.no_protein

    logging.info(
        "Starting dataset prep; files=%d | logfile=%s | modalities: QR=%s DNA=%s PROT=%s",
        len(files), logfile, do_qr, do_dna, do_prot
    )

    with cf.ThreadPoolExecutor(max_workers=args.modality_workers, thread_name_prefix="modality") as modality_pool:
        def _handle(src: Path):
            try:
                process_one_file(
                    src,
                    args.input_root,
                    args.blocks_root,
                    args.qrcode_root,
                    args.dna_root,
                    args.protein_root,
                    args.recipe_root,
                    args.chunk_size,
                    str(uuid.uuid4()),
                    args.qr_version,
                    args.qr_ecc,
                    do_qr,
                    do_dna,
                    do_prot,
                    modality_pool,
                )
            except Exception:
                logging.exception("File task failed for %s", src)

        with cf.ThreadPoolExecutor(max_workers=args.file_workers, thread_name_prefix="file") as file_pool:
            list(file_pool.map(_handle, files))

    logging.info("All done. If you disabled modalities, your recipe still references the root paths for re-hydration.")

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        logging.warning("Interrupted; exiting.")
