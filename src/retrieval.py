"""
retrieval.py — Hybrid BM25 + semantic retrieval with Reciprocal Rank Fusion.

Loads precomputed artifacts from disk and runs fast retrieval at rank time.
"""

import pickle
import numpy as np
import faiss
import bm25s


def load_artifacts(artifacts_dir: str = "artifacts"):
    """Load all precomputed artifacts from disk."""
    embeddings = np.load(f"{artifacts_dir}/embeddings.npy").astype("float32")
    candidate_ids = np.load(
        f"{artifacts_dir}/candidate_ids.npy", allow_pickle=True
    )
    jd_embedding = np.load(f"{artifacts_dir}/jd_embedding.npy").astype("float32")

    retriever = bm25s.BM25.load(f"{artifacts_dir}/bm25_index", load_corpus=False)

    with open(f"{artifacts_dir}/jd_query_tokens.pkl", "rb") as f:
        jd_tokens = pickle.load(f)

    print(f"[retrieval] Loaded {len(candidate_ids):,} candidate embeddings")
    return embeddings, candidate_ids, jd_embedding, retriever, jd_tokens


def bm25_search(
    retriever: bm25s.BM25,
    jd_tokens,
    candidate_ids: np.ndarray,
    k: int = 1000
) -> list:
    """BM25 lexical search. Returns list of candidate_ids ranked by BM25 score."""
    actual_k = min(k, len(candidate_ids))
    results, _ = retriever.retrieve(jd_tokens, k=actual_k)
    return [str(candidate_ids[i]) for i in results[0]]


def semantic_search(
    embeddings: np.ndarray,
    jd_embedding: np.ndarray,
    candidate_ids: np.ndarray,
    k: int = 1000
) -> list:
    """
    Semantic search using FAISS IndexFlatIP (cosine similarity after L2 normalization).
    Returns list of candidate_ids ranked by cosine similarity.
    """
    actual_k = min(k, len(embeddings))

    # Normalize for cosine similarity
    emb_copy = embeddings.copy()
    faiss.normalize_L2(emb_copy)

    index = faiss.IndexFlatIP(emb_copy.shape[1])
    index.add(emb_copy)

    jd_vec = jd_embedding.reshape(1, -1).copy()
    faiss.normalize_L2(jd_vec)

    _, indices = index.search(jd_vec, actual_k)
    return [str(candidate_ids[i]) for i in indices[0]]


def reciprocal_rank_fusion(ranked_lists: list, k: int = 60) -> list:
    """
    Merge multiple ranked lists using Reciprocal Rank Fusion.
    
    k=60 is the standard constant from the original Cormack et al. 2009 paper.
    It controls how much top ranks are amplified vs lower ranks.
    
    Returns list of (candidate_id, rrf_score) sorted descending.
    """
    scores: dict = {}
    for ranked_list in ranked_lists:
        for rank, doc_id in enumerate(ranked_list):
            if doc_id not in scores:
                scores[doc_id] = 0.0
            scores[doc_id] += 1.0 / (k + rank + 1)

    return sorted(scores.items(), key=lambda x: x[1], reverse=True)


def hybrid_retrieve(artifacts_dir: str = "artifacts", top_k: int = 1000) -> list:
    """
    Full hybrid retrieval pipeline.
    Returns list of top_k candidate_ids ranked by RRF fusion score.
    """
    embeddings, candidate_ids, jd_embedding, retriever, jd_tokens = \
        load_artifacts(artifacts_dir)

    print("[retrieval] Running BM25 search...")
    bm25_ids = bm25_search(retriever, jd_tokens, candidate_ids, k=top_k)

    print("[retrieval] Running semantic search...")
    semantic_ids = semantic_search(embeddings, jd_embedding, candidate_ids, k=top_k)

    print("[retrieval] Fusing results with RRF...")
    fused = reciprocal_rank_fusion([bm25_ids, semantic_ids])

    actual_top_k = min(top_k, len(fused))
    top_ids = [cid for cid, _ in fused[:actual_top_k]]

    print(f"[retrieval] Retrieved {len(top_ids):,} candidates after RRF fusion")
    return top_ids