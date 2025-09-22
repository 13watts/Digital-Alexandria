# Digital-Alexandria
Digital Alexandria is an open-source initiative to create a modern “Library of Alexandria” for the digital age. The system preserves all possible ≤1 KiB data blocks, each uniquely named by its SHA-256 hash, and stores them in a globally distributed, media-agnostic archive. Beyond raw binary blocks, Digital Alexandria encodes data into multiple durable modalities — including QR codes, DNA sequences, and protein strings — to maximize resilience against technological obsolescence.

Using extreme deduplication, pair-tree storage layouts, and recipe-based file rehydration, the project aims to reduce storage overhead while ensuring long-term accessibility. The ultimate goal is an evergreen, AI-ready archive that outlasts hardware lifecycles, supports multi-site replication, and enables future generations to reconstruct humanity’s digital record regardless of the media available to them.

# Current State
## Digital Alexandria — Prototype Phasing

Digital Alexandria is an open-source initiative to design a next-generation archival system that preserves humanity’s digital knowledge for centuries. The prototype is structured in three phases, each building on the last to ensure extreme deduplication, multi-modal encoding, and long-term accessibility.

### Phase 1 — Dataset Preparation
- Crawl a source directory of files.
- Generate process UUIDs and file-level SHA-256 checksums.
- Chunk each file into ≤1 KiB blocks, compute per-block SHA-256, and store in a pair-tree structure.
- For new blocks, generate additional modalities:
  - Write a per-file .recipe JSON with block sequence, metadata, and modality options.
  - (optional) QR Code (Version 40, ECC H) containing hash, block data, and CRC-32.
  - (optional) DNA sequence (2-bit → A/C/G/T).
  - (optional) Protein sequence (nibble → deterministic 16-amino subset).

### Phase 2 — Analysis & Deduplication Metrics
- Crawl .recipe files to tabulate unique vs. duplicate blocks.
- Calculate logical vs. physical size, compression ratios, and dedupe effectiveness.
- Export per-block CSV for downstream analysis or visualization.

### Future Phase — Integration & Archival Pipeline
- Combine prepared blocks and recipes into a scalable archival system.
- Integrate with storage backends (object stores, tape HSM, or experimental DNA/protein media).
- Provide monitoring, reporting, and APIs for long-term preservation and retrieval.
#
# Why This Matters
- Extreme Deduplication: Store every unique ≤1 KiB block once, reuse across all files.
- Multi-modal Encodings: Preserve data in multiple forms (digital, DNA, protein, QR) to resist obsolescence.
- Evergreen Architecture: Pair-tree + recipes allow interoperability across future storage mediums.
- Open Standards: Licensed under CC BY-SA 4.0 so the community can extend, remix, and improve.

>*Prototype tools are written in Python 3.9+, with concurrency and modular design to scale across HPC or distributed storage environments.*
