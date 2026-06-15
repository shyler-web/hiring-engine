"""
Run ONCE offline before submission.
Builds and saves: BM25 index, candidate embeddings, JD embedding.

Usage:
    uv run python precompute.py --candidates candidates.jsonl.gz
"""

import argparse
import gzip
import json
import pickle
import numpy as np
from pathlib import Path
from tqdm import tqdm

import bm25s
from sentence_transformers import SentenceTransformer

from src.filters import filter_candidates
from src.candidate_doc import build_candidate_doc
from src.reranker import JD_QUERY

ARTIFACTS_DIR = Path("artifacts")
ARTIFACTS_DIR.mkdir(exist_ok=True)


def load_candidates(path):
    candidates = []
    opener = gzip.open if path.endswith(".gz") else open
    with opener(path, "rt", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                candidates.append(json.loads(line))
    return candidates


def main(candidates_path):
    print("Loading candidates...")
    candidates = load_candidates(candidates_path)
    print(f"Loaded {len(candidates)} candidates")

    print("Filtering candidates...")
    candidates = filter_candidates(candidates)

    print("Building candidate documents...")
    docs = [build_candidate_doc(c) for c in tqdm(candidates)]
    candidate_ids = np.array([c['candidate_id'] for c in candidates])

    # Save candidate_ids aligned with docs
    np.save(ARTIFACTS_DIR / "candidate_ids.npy", candidate_ids)

    # Build BM25 index
    print("Building BM25 index...")
    tokenized = bm25s.tokenize(docs)
    retriever = bm25s.BM25()
    retriever.index(tokenized)
    retriever.save(str(ARTIFACTS_DIR / "bm25_index"))

    # Save tokenized JD query for BM25
    jd_tokens = bm25s.tokenize([JD_QUERY])
    with open(ARTIFACTS_DIR / "jd_query_tokens.pkl", "wb") as f:
        pickle.dump(jd_tokens, f)

    # Build semantic embeddings
    print("Loading embedding model...")
    model = SentenceTransformer(
        "nomic-ai/nomic-embed-text-v1.5",
        trust_remote_code=True
    )

    print("Encoding candidate documents (this takes a while)...")
    embeddings = model.encode(
        docs,
        batch_size=64,
        show_progress_bar=True,
        convert_to_numpy=True
    )
    np.save(ARTIFACTS_DIR / "embeddings.npy", embeddings)

    # Encode JD query
    print("Encoding JD query...")
    jd_embedding = model.encode([JD_QUERY], convert_to_numpy=True)
    np.save(ARTIFACTS_DIR / "jd_embedding.npy", jd_embedding[0])

    print("Precomputation complete. Artifacts saved to artifacts/")
    print(f"  embeddings.npy: {embeddings.shape}")
    print(f"  candidate_ids.npy: {len(candidate_ids)}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--candidates", required=True)
    args = parser.parse_args()
    main(args.candidates)