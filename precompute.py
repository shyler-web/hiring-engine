"""
precompute.py — Build and save all artifacts needed by rank.py.

Run ONCE offline before submission. Can take 30-90 minutes for 100K candidates.
Artifacts are committed to the repo so judges only need to run rank.py.

Usage:
    uv run python precompute.py --candidates candidates.jsonl.gz
    uv run python precompute.py --candidates sample_candidates.jsonl  # for testing
"""

import argparse
import gzip
import json
import pickle
from pathlib import Path

import numpy as np
import bm25s
from tqdm import tqdm
from sentence_transformers import SentenceTransformer

from src.filters import filter_candidates
from src.candidate_doc import build_candidate_doc
from src.reranker import JD_QUERY

ARTIFACTS_DIR = Path("artifacts")


def load_candidates(path: str) -> list:
    """Load candidates from .jsonl or .jsonl.gz file."""
    candidates = []
    opener = gzip.open if path.endswith(".gz") else open
    mode = "rt"

    with opener(path, mode, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    candidates.append(json.loads(line))
                except json.JSONDecodeError:
                    continue

    print(f"[precompute] Loaded {len(candidates):,} candidates from {path}")
    return candidates


def load_candidates_json_array(path: str) -> list:
    """Load candidates from a JSON array file (sample format)."""
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    print(f"[precompute] Loaded {len(data):,} candidates from JSON array {path}")
    return data


def main(candidates_path: str):
    ARTIFACTS_DIR.mkdir(exist_ok=True)

    # --- Load candidates ---
    # Handle both .jsonl/.jsonl.gz and .json array formats
    if candidates_path.endswith(".json") and not candidates_path.endswith(".jsonl"):
        candidates = load_candidates_json_array(candidates_path)
    else:
        candidates = load_candidates(candidates_path)

    # --- Filter ---
    print("\n[precompute] Filtering candidates...")
    candidates = filter_candidates(candidates)

    if len(candidates) == 0:
        raise ValueError("All candidates were filtered out. Check filters.py.")

    # --- Build documents ---
    print(f"\n[precompute] Building candidate documents ({len(candidates):,} candidates)...")
    docs = []
    candidate_ids = []

    for c in tqdm(candidates, desc="Building docs"):
        doc = build_candidate_doc(c)
        docs.append(doc)
        candidate_ids.append(c['candidate_id'])

    candidate_ids_arr = np.array(candidate_ids)

    # Save candidate_ids — must be aligned with both BM25 index and embeddings
    np.save(ARTIFACTS_DIR / "candidate_ids.npy", candidate_ids_arr)
    print(f"[precompute] Saved candidate_ids.npy ({len(candidate_ids_arr):,} entries)")

    # --- Build BM25 index ---
    print("\n[precompute] Building BM25 index...")
    tokenized_corpus = bm25s.tokenize(docs, show_progress=True)
    retriever = bm25s.BM25()
    retriever.index(tokenized_corpus)

    bm25_dir = str(ARTIFACTS_DIR / "bm25_index")
    Path(bm25_dir).mkdir(exist_ok=True)
    retriever.save(bm25_dir)
    print(f"[precompute] Saved BM25 index to {bm25_dir}/")

    # Save tokenized JD query for BM25 at rank time
    jd_tokens = bm25s.tokenize([JD_QUERY], show_progress=False)
    with open(ARTIFACTS_DIR / "jd_query_tokens.pkl", "wb") as f:
        pickle.dump(jd_tokens, f)
    print("[precompute] Saved JD query tokens")

    # --- Build semantic embeddings ---
    print("\n[precompute] Loading nomic-embed-text-v1.5 model...")
    print("  (First run will download ~500MB model — cached after that)")
    model = SentenceTransformer(
        "nomic-ai/nomic-embed-text-v1.5",
        trust_remote_code=True
    )

    print(f"[precompute] Encoding {len(docs):,} candidate documents...")
    print("  (This is the slow step — ~30-60 min for 100K on CPU)")
    embeddings = model.encode(
        docs,
        batch_size=64,
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=False   # we normalize at retrieval time
    )

    np.save(ARTIFACTS_DIR / "embeddings.npy", embeddings)
    emb_size_mb = embeddings.nbytes / (1024 * 1024)
    print(f"[precompute] Saved embeddings.npy — shape: {embeddings.shape}, size: {emb_size_mb:.0f}MB")

    # --- Encode JD query ---
    print("\n[precompute] Encoding JD query...")
    jd_embedding = model.encode(
        [JD_QUERY],
        convert_to_numpy=True,
        normalize_embeddings=False
    )
    np.save(ARTIFACTS_DIR / "jd_embedding.npy", jd_embedding[0])
    print("[precompute] Saved jd_embedding.npy")

    # --- Summary ---
    print("\n" + "="*50)
    print("PRECOMPUTATION COMPLETE")
    print("="*50)
    print(f"  Candidates indexed: {len(candidate_ids_arr):,}")
    print(f"  Embedding shape:    {embeddings.shape}")
    print(f"  Embedding size:     {emb_size_mb:.0f} MB")
    print(f"  Artifacts saved to: {ARTIFACTS_DIR}/")
    print("\nFiles created:")
    for f in sorted(ARTIFACTS_DIR.rglob("*")):
        if f.is_file():
            size_kb = f.stat().st_size / 1024
            print(f"  {f}  ({size_kb:.0f} KB)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Precompute BM25 index and semantic embeddings for candidate ranking."
    )
    parser.add_argument(
        "--candidates",
        required=True,
        help="Path to candidates file (.jsonl, .jsonl.gz, or .json array)"
    )
    args = parser.parse_args()
    main(args.candidates)