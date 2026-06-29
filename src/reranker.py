"""
reranker.py — Cross-encoder reranking via raw ONNXRuntime INT8.

No optimum dependency. Uses onnxruntime.InferenceSession directly.

Load priority:
    1. artifacts/reranker_optimum/model_int8.onnx  ← fast (build with export_onnx.py)
    2. PyTorch fallback via sentence-transformers   ← slow, only if ONNX missing

Constraints satisfied:
    - CPU only (CPUExecutionProvider)
    - No network calls at rank time
    - 2000 candidates @ batch_size=64 → ~50-60s on 16-core CPU
"""

import math
import os
from pathlib import Path

import numpy as np
from tqdm import tqdm
from transformers import AutoTokenizer


# ── Core config ───────────────────────────────────────────────────────────────
_TOTAL_CORES = int(os.cpu_count() or 8)
# Leave ~4 cores free for WSL2 / VS Code / OS I/O overhead
# intra = parallelism inside one op (matmul threads) — use most cores here
# inter = parallelism across independent graph nodes — 2 is enough
_INTRA_THREADS = max(1, _TOTAL_CORES - 4)   # e.g. 12 on a 16-core machine
_INTER_THREADS = 2

# INT8 ONNX path is faster at batch_size=64 regardless of what rank.py passes.
# We override batch_size internally for the ONNX path only.
# rank.py can pass batch_size=32 — it won't matter for ONNX.
_ONNX_BATCH_SIZE = 64

JD_QUERY = """
Senior AI Engineer, founding team, 5 to 9 years experience at product companies.
Core requirement: production systems for embedding-based retrieval and ranking.
Must have shipped to real users at scale, not just prototypes or research.
Vector databases and search infrastructure:
Pinecone, Weaviate, Qdrant, Milvus, FAISS, Elasticsearch, OpenSearch, pgvector.
Hybrid search combining dense and sparse retrieval. BM25. Approximate nearest neighbor.
Embedding models and fine-tuning:
sentence-transformers, BGE, E5, nomic-embed, OpenAI embeddings.
Fine-tuning with LoRA, QLoRA, PEFT. Hugging Face transformers ecosystem.
Ranking and evaluation:
Learning-to-rank: XGBoost, LightGBM. Ranking metrics: NDCG, MRR, MAP, P@K.
A/B testing for ranking systems. Offline-online correlation analysis.
Recommendation systems, search relevance, information retrieval.
LLM integration:
RAG pipelines, retrieval-augmented generation, LLM re-ranking.
Haystack, LlamaIndex frameworks. Prompt engineering for retrieval.
Engineering:
Strong Python. Production ML engineering, not pure research.
MLOps, model serving, inference optimization.
Strong preference for:
Product company background (not consulting-only careers).
Indian product startups: Swiggy, Zomato, Flipkart, Razorpay, CRED, Meesho, PhonePe.
AI-native companies: Sarvam AI, Mad Street Den, Observe.AI, Haptik, Krutrim.
FAANG experience: Google, Meta, Amazon, Microsoft, Netflix.
Located in India, preferably Pune or Noida. Short notice period.
"""


# ── Public: loader ────────────────────────────────────────────────────────────

def load_reranker(artifacts_dir: str = "artifacts") -> dict:
    """
    Returns a ranker dict consumed by rerank().

    Tries artifacts/reranker_optimum/model_int8.onnx first.
    Falls back to PyTorch MiniLM if ONNX is missing.
    If you see the fallback warning, run export_onnx.py first.
    """
    onnx_path = Path(artifacts_dir) / "reranker_optimum" / "model_int8.onnx"
    tok_dir   = Path(artifacts_dir) / "reranker_optimum"

    if onnx_path.exists():
        import onnxruntime as ort

        print(f"[reranker] Loading INT8 ONNX: {onnx_path}")
        print(f"[reranker] CPU cores — intra={_INTRA_THREADS}, inter={_INTER_THREADS}")
        print(f"[reranker] Effective batch size for ONNX: {_ONNX_BATCH_SIZE}")

        sess_opts = ort.SessionOptions()
        sess_opts.intra_op_num_threads  = _INTRA_THREADS
        sess_opts.inter_op_num_threads  = _INTER_THREADS
        # ORT_ENABLE_ALL: constant folding + node fusion + memory rewrites
        # Safe for inference-only; meaningfully reduces latency on BERT models
        sess_opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL

        session   = ort.InferenceSession(
            str(onnx_path),
            sess_options=sess_opts,
            providers=["CPUExecutionProvider"],
        )
        tokenizer = AutoTokenizer.from_pretrained(str(tok_dir))

        # Detect whether model uses token_type_ids
        # (mxbai does; some newer models don't)
        input_names = {inp.name for inp in session.get_inputs()}
        has_tti     = "token_type_ids" in input_names
        print(f"[reranker] ONNX inputs: {input_names}")

        return {
            "type":      "onnx_raw",
            "session":   session,
            "tokenizer": tokenizer,
            "has_tti":   has_tti,
        }

    # ── Fallback ──────────────────────────────────────────────────────────────
    print("[reranker] ⚠️  INT8 ONNX not found — falling back to PyTorch.")
    print("[reranker] Run export_onnx.py once to build the fast path.")
    print(f"[reranker] Expected path: {onnx_path}")
    from sentence_transformers import CrossEncoder
    model = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2", max_length=512)
    return {"type": "pytorch", "model": model}


# ── Public: rerank ────────────────────────────────────────────────────────────

def rerank(ranker: dict, top_ids: list, candidate_docs: dict, batch_size: int = 32) -> list:
    """
    Score every (JD_QUERY, candidate_doc) pair and return sorted results.

    Args:
        ranker:         dict from load_reranker()
        top_ids:        list of candidate IDs (ordered by retrieval score)
        candidate_docs: dict  candidate_id → plain-text doc string
        batch_size:     passed through to PyTorch path only.
                        ONNX path always uses _ONNX_BATCH_SIZE (64) internally.

    Returns:
        List of (candidate_id, sigmoid_score) sorted descending by score.
    """
    valid_ids = [cid for cid in top_ids if cid in candidate_docs]
    if not valid_ids:
        print("[reranker] ⚠️  No valid candidates found in candidate_docs.")
        return []

    pairs = [(JD_QUERY, candidate_docs[cid]) for cid in valid_ids]
    print(f"[reranker] Scoring {len(pairs):,} pairs...")

    if ranker["type"] == "onnx_raw":
        scores = _score_onnx(ranker, pairs)          # uses _ONNX_BATCH_SIZE internally
    else:
        scores = _score_pytorch(ranker, pairs, batch_size)

    # Sigmoid: maps raw logit → (0, 1). Standard for cross-encoder rerankers.
    normalized = [1.0 / (1.0 + math.exp(-float(s))) for s in scores]
    ranked     = sorted(zip(valid_ids, normalized), key=lambda x: x[1], reverse=True)

    print(f"[reranker] Top score   : {ranked[0][1]:.4f}")
    print(f"[reranker] Bottom score: {ranked[-1][1]:.4f}")
    return ranked


# ── Internal: ONNX inference ──────────────────────────────────────────────────

def _score_onnx(ranker: dict, pairs: list) -> list:
    """
    Batched inference through the INT8 ONNX session.

    Tokenization produces numpy int64 arrays directly (return_tensors="np"),
    avoiding a torch→numpy copy. Three arrays for BERT-class models:
        input_ids       — token IDs: [CLS] query tokens [SEP] doc tokens [SEP]
        attention_mask  — 1 for real tokens, 0 for padding
        token_type_ids  — 0 for query segment, 1 for doc segment
    """
    session   = ranker["session"]
    tokenizer = ranker["tokenizer"]
    has_tti   = ranker["has_tti"]
    scores    = []

    for i in tqdm(range(0, len(pairs), _ONNX_BATCH_SIZE), desc="ONNX INT8"):
        batch         = pairs[i : i + _ONNX_BATCH_SIZE]
        queries, docs = zip(*batch)

        enc = tokenizer(
            list(queries),
            list(docs),
            truncation=True,
            padding=True,
            max_length=512,
            return_tensors="np",      # numpy directly — no torch needed
        )

        ort_inputs = {
            "input_ids":      enc["input_ids"].astype(np.int64),
            "attention_mask": enc["attention_mask"].astype(np.int64),
        }
        if has_tti:
            ort_inputs["token_type_ids"] = enc["token_type_ids"].astype(np.int64)

        # session.run → list of np arrays; [0] = logits shape (batch_size, num_labels)
        # For rerankers num_labels=1, so flatten gives one score per pair
        logits = session.run(["logits"], ort_inputs)[0]
        scores.extend(logits.flatten().tolist())

    return scores


# ── Internal: PyTorch fallback ────────────────────────────────────────────────

def _score_pytorch(ranker: dict, pairs: list, batch_size: int) -> list:
    raw = ranker["model"].predict(pairs, batch_size=batch_size, show_progress_bar=True)
    return raw.tolist()


# ── Called by precompute.py (shim) ────────────────────────────────────────────

def export_to_onnx(artifacts_dir: str = "artifacts"):
    """
    Shim so precompute.py can call reranker.export_to_onnx().
    Real logic is in export_onnx.py.
    """
    import subprocess, sys
    print("[reranker] Delegating ONNX export to export_onnx.py ...")
    subprocess.run([sys.executable, "export_onnx.py"], check=True)