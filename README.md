# Redrob Candidate Ranker

Two-stage hybrid retrieval pipeline for the Intelligent Candidate Discovery & Ranking Challenge.

## Architecture

```
100K candidates
    ↓
[Pre-filter] Honeypot detection + hard business rules → ~70-85K
    ↓
[Stage 1] BM25 (bm25s) + Semantic (nomic-embed-text-v1.5) fused via RRF → top 1000
    ↓
[Stage 2] Cross-encoder re-ranking (mxbai-rerank-xsmall-v1) → scored top 1000
    ↓
[Stage 3] Structured signal scoring (availability, notice, location, github, career quality)
    ↓
Top 100 with per-candidate reasoning
```

## Key Design Decisions

- **nomic-embed-text-v1.5**: 8192 token window avoids chunking full candidate profiles
- **bm25s**: Scipy-backed sparse retrieval, orders of magnitude faster than Python BM25
- **RRF fusion**: Ordinal rank fusion avoids score normalization across incompatible scales
- **Structured signals**: Cross-encoder scores text fit; structured multipliers adjust for
  availability, notice period, location, GitHub activity, and career quality (product vs consulting)
- **Honeypot detection**: Pre-filters impossible profiles before they contaminate retrieval
- **Calibrated thresholds**: All signal weights derived from actual 100K dataset percentile stats

## Setup

```bash
uv sync
```

## Precompute (artifacts already committed — skip this)

```bash
uv run python precompute.py --candidates candidates.jsonl.gz
```

## Rank

```bash
uv run python rank.py --candidates /path/to/candidates.jsonl.gz --out team_xxx.csv
```

## Validate

```bash
uv run python validate_submission.py team_xxx.csv
```

## Environment

- Python 3.11
- CPU only, no GPU required
- RAM: ~1.8 GB peak
- Runtime: ~16 seconds (with precomputed artifacts)
- OS tested: Ubuntu 22.04 (WSL2)