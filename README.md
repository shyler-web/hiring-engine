# Candidate Ranker — Intelligent Candidate Discovery & Ranking Challenge
 Hybrid retrieval pipeline for large-scale candidate ranking on CPU.

## Architecture

```
100K candidates
    ↓
[Pre-filter] Honeypot detection + hard business rules → ~5.5K survivors
    ↓
[STEP 1] Candidate Document Builder
    ↓
[STEP 2] Offline Precomputation
    ↓       
[Stage 3] BM25 (bm25s) + Semantic (nomic-embed-text-v1.5) fused via RRF → top 500
    ↓
[Stage 4] Cross-encoder reranking (mxbai-rerank-xsmall-v1, INT8 ONNX) → scored top 500
    ↓
[Stage 5] Structured signal scoring (YoE fit, notice period, location, GitHub, career quality)
    ↓
[STEP 6] Final Rank + Reasoning
    ↓
Top 100 with per-candidate reasoning
```

## Key Design Decisions

- **Honeypot detection** — Pre-filters impossible profiles (timeline contradictions, fictional companies, impossible skill durations) before they contaminate retrieval
- **nomic-embed-text-v1.5** — 8192 token window avoids chunking full candidate profiles
- **bm25s** — Scipy-backed sparse retrieval, significantly faster than pure Python BM25
- **RRF fusion** — Ordinal rank fusion avoids score normalization across incompatible scales
- **INT8 ONNX reranker** — mxbai-rerank-xsmall-v1 exported via torch.onnx and quantized with onnxruntime.quantization. ~3x faster on CPU with minimal accuracy loss
- **Structured signals** — Cross-encoder scores text fit; multipliers adjust for YoE fit, notice period, location, GitHub activity, and career quality (FAANG/product vs consulting). All signal weights derived from actual dataset percentile statistics
---

## Setup

### Option A — pip (verified in sandbox)
```bash
pip install .
```

### Option B — uv (recommended, exact version pinning via uv.lock)
```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
uv sync
```

---

## Ranking — Single Command

### With pip
```bash
CUDA_VISIBLE_DEVICES="" python rank.py \
    --candidates candidates.jsonl.gz \
    --out team_InferenceEngine.csv
```

### With uv
```bash
CUDA_VISIBLE_DEVICES="" uv run python rank.py \
    --candidates candidates.jsonl.gz \
    --out team_InferenceEngine.csv
```

Completes in ~160 seconds on 16-core CPU . Zero network calls.

---

## Validate Output
```bash
python scripts/validate_submission.py team_InferenceEngine.csv
```

---

## Sandbox — 100 Candidate Sample

Sample candidates and artifacts are committed for quick verification:
```bash
CUDA_VISIBLE_DEVICES="" python rank.py \
    --candidates sample_candidates.jsonl \
    --artifacts sample_artifacts \
    --out team_InferenceEngine_submission_sample.csv
```

Full sandbox notebook: [Google Colab](https://colab.research.google.com/drive/1lUH0R8_eM5-wQqr87dlcPSq5xp2lBtLB?usp=sharing)

---

## Pre-computation (already done — artifacts committed to repo)

> ⚠️ Judges skip this entire section. All artifacts are precomputed and committed.

To reproduce from scratch (authors only):

```bash
# Step 1 — Analyse JD templates (produces jd_templates.pkl)
uv run python scripts/analyze_jd_templates.py --candidates candidates.jsonl.gz

# Step 2 — Build template summaries (depends on Step 1, produces jd_templates_enhanced.pkl)
uv run python scripts/build_template_summaries.py

# Step 3 — Compute skill duration percentiles (produces skill_duration_percentiles.pkl)
uv run python scripts/compute_skill_duration_percentiles.py --candidates candidates.jsonl.gz

# Step 4 — Build BM25 index and embeddings (depends on Steps 2 and 3)
uv run python precompute.py --candidates candidates.jsonl.gz

# Step 5 — Export and quantize reranker to INT8 ONNX (authors only)
# Requires: huggingface-cli download mixedbread-ai/mxbai-rerank-xsmall-v1
uv run python scripts/export_onnx.py
```

---

## Compute Environment

| Constraint | Limit | Actual |
|---|---|---|
| Runtime | ≤ 5 minutes | ~140 seconds |
| Memory | ≤ 16 GB RAM | ~1.8 GB peak |
| GPU | CPU only | Disabled via CUDA_VISIBLE_DEVICES="" |
| Network | Off during ranking | Zero external calls |
| Disk | ≤ 5 GB artifacts | ~500 MB |

- Python 3.12.3
- OS: Ubuntu 24.04 LTS (WSL2 on Windows)
- Hardware: HP Victus, 16-core CPU, 16GB RAM
- ONNX inference: 12 intra-op threads + 2 inter-op threads
