# Redrob Candidate Ranker

Two-stage hybrid retrieval pipeline for the Intelligent Candidate Discovery & Ranking Challenge.

## Architecture
1. Hard filters + honeypot detection
2. BM25 (bm25s) + Semantic (nomic-embed-text-v1.5) retrieval fused via RRF → top 1000
3. Cross-encoder re-ranking (mxbai-rerank-xsmall-v1) → top 1000 scored
4. Structured signal scoring (availability, notice period, location, github, career quality)
5. Top 100 exported with per-candidate reasoning

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
uv run python rank.py --candidates candidates.jsonl.gz --out team_xxx.csv
```

## Environment
- Python 3.11
- CPU only, no GPU required
- RAM: ~1.8GB peak
- Runtime: ~16 seconds