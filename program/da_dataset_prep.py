#!/usr/bin/env python3
"""
Digital Alexandria — Dataset Prep Phase
- Crawl an input directory for source files
- For each file: assign process UUID, compute file SHA-256
- Chunk into <=1 KiB blocks; for each block compute SHA-256, write .block in pair-tree
- If the .block already exists, skip writing and skip modalities
- For NEW blocks, asynchronously generate three modalities:
    * QR Code (Model 2, Version 40, ECC H) -> .svg
      Payload = [hash-hex ASCII (64)] + [raw block bytes (<=1024)] + [CRC32 (4 bytes, big-endian)]
    * DNA sequence -> .dna  (2-bit -> A/C/G/T)
    * Protein sequence -> .protein (nibble -> 16 aa subset)
- Emit a per-source-file .recipe JSON with block sequence and paths + modality options.
- Logging: INFO to stdout, DEBUG to file in ./log/
NOTE: Written for Python 3.9
License: Creative Commons Attribution-ShareAlike 4.0 (CC BY-SA 4.0)
"""

from __future__ import annotations

import argparse
import concurrent.futures as cf
import dataclasses
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

    # Console handler (INFO)
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
    # Linux doesn't have birth time reliably; use mtime as "creation" proxy
    ts = path.stat().st_mtime
    return dt.datetime.fromtimestamp(ts).isoformat(timespec="seconds")


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
    qr.save(svg_path, scale=1)  # raw SVG with default sizing; microscope will love the vectors
    return svg_path

def write_text_modality(root: Path, hexhash: str, ext: str, content: str) -> Path:
    out = pairtree_path(root, hexhash, f"{hexhash}.{ext}")
    out.write_text(content, encoding="utf-8")
    return out

# ----------------------------- Data Model ----------------------------------

@dataclasses.dataclass
class BlockRecord:
    seq: int
    sha256: str
    block_path: str  # .block file
    wrote_block: bool
    modalities: dict  # paths for svg/dna/protein when created

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
    generate_modalities: bool,
) -> BlockRecord:
    b_hash = sha256(block_bytes).hexdigest()
    block_path = pairtree_path(blocks_root, b_hash, f"{b_hash}.block")
    modalities = {}
    if block_path.exists():
        logging.debug("Block %s exists; skipping write & modalities", b_hash)
        return BlockRecord(seq, b_hash, str(block_path), wrote_block=False, modalities=modalities)

    # Write the .block
    with open(block_path, "wb", buffering=0) as f:
        f.write(block_bytes)
    logging.debug("Wrote block #%d -> %s", seq, block_path)

    if generate_modalities:
        try:
            # QR
            payload = make_qr_payload(b_hash, block_bytes)
            svg_path = write_qr_svg(qr_root, b_hash, payload, qr_version, qr_ecc)
            modalities["qrcode_svg"] = str(svg_path)

            # DNA
            dna_seq = block_to_dna(block_bytes)
            dna_path = write_text_modality(dna_root, b_hash, "dna", dna_seq)
            modalities["dna"] = str(dna_path)

            # Protein
            prot_seq = block_to_protein(block_bytes)
            prot_path = write_text_modality(prot_root, b_hash, "protein", prot_seq)
            modalities["protein"] = str(prot_path)

            logging.debug("Modalities generated for block %s", b_hash)
        except Exception as e:
            # If one modality fails, we log but keep going; the .block is already durable.
            logging.exception("Modality generation failed for block %s: %s", b_hash, e)

    return BlockRecord(seq, b_hash, str(block_path), wrote_block=True, modalities=modalities)

# ------------------------------- Main Flow ---------------------------------

def process_one_file(
    src_path: Path,
    input_root: Path,
    blocks_root: Path,
    qr_root: Path,
    dna_root: Path,
    prot_root: Path,
    recipe_root: Path,
    chunk_size: int,
    file_uuid: str | None,
    qr_version: int,
    qr_ecc: str,
    modality_pool: cf.Executor,
) -> None:
    rel = src_path.relative_to(input_root)
    file_uuid = file_uuid or str(uuid.uuid4())

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
        # schedule block processing (write + modalities)
        fut = modality_pool.submit(
            process_block,
            chunk, seq,
            blocks_root, qr_root, dna_root, prot_root,
            qr_version, qr_ecc,
            True  # generate modalities only when a new block is written (function handles skip)
        )
        futures.append(fut)
        seq += 1

    # Collect results in order of sequence
    blocks: List[BlockRecord] = []
    for fut in cf.as_completed(futures):
        try:
            rec: BlockRecord = fut.result()
            blocks.append(rec)
        except Exception as e:
            logging.exception("Block task failed for %s: %s", src_path, e)

    # Re-sort by sequence (as_completed is unordered)
    blocks.sort(key=lambda r: r.seq)

    # Build recipe JSON path
    recipe_out = recipe_root / rel.with_suffix(rel.suffix + ".recipe")
    recipe_out.parent.mkdir(parents=True, exist_ok=True)

    # Construct recipe object
    recipe = {
        "process_uuid": file_uuid,
        "source_file_location": str(src_path.parent),
        "source_file_name": src_path.name,
        "source_file_creation_date": created_iso,
        "source_file_sha256": file_hash,
        "file_recipe": [
            {
                "block_sequence_number": r.seq,
                "block_sha256_value": r.sha256,
                "block_sha256_path": r.block_path,
                # Include modality paths if they were created in this run
                "modalities": r.modalities,
            }
            for r in blocks
        ],
        "modality_options": {
            "data_block": True,
            "qr_code": True,
            "qr_code_type": "Model 2, Version 40",
            "qr_code_ecc": qr_ecc,
            "dna_recipe": True,
            "protein_recipe": True,
        },
        "chunk_size_bytes": chunk_size,
        "notes": "If a .block pre-existed, it was not re-written and no modalities were regenerated.",
    }

    recipe_out.write_text(json.dumps(recipe, indent=2), encoding="utf-8")
    logging.info("Recipe written: %s (blocks=%d)", recipe_out, len(blocks))


def discover_files(root: Path) -> Iterable[Path]:
    for p in root.rglob("*"):
        if p.is_file():
            yield p


def main():
    ap = argparse.ArgumentParser(
        description="Digital Alexandria — Dataset Prep (Phase 1)"
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

    ap.add_argument("--verbose", action="store_true", help="More chatty console")
    args = ap.parse_args()

    logfile = setup_logging(args.log_dir, verbose=True)

    # Ensure roots exist
    for d in (args.blocks_root, args.qrcode_root, args.dna_root, args.protein_root, args.recipe_root):
        d.mkdir(parents=True, exist_ok=True)

    files = list(discover_files(args.input_root))
    if not files:
        logging.warning("No files found under %s", args.input_root)
        return

    logging.info("Starting dataset prep; files=%d | logfile=%s", len(files), logfile)

    # Dedicated pool for per-file orchestration and a shared pool for blocks+modalities
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
                    None,
                    args.qr_version,
                    args.qr_ecc,
                    modality_pool,
                )
            except Exception as e:
                logging.exception("File task failed: %s", e)

        with cf.ThreadPoolExecutor(max_workers=args.file_workers, thread_name_prefix="file") as file_pool:
            list(file_pool.map(_handle, files))

    logging.info("All done. Consider hydrating metrics & reports before Phase 2. 🧪")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        logging.warning("Interrupted; exiting.")
