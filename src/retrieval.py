import numpy as np
import faiss
import bm25s
import pickle


def load_artifacts(artifacts_dir="artifacts"):
    embeddings = np.load(f"{artifacts_dir}/embeddings.npy").astype("float32")
    candidate_ids = np.load(
        f"{artifacts_dir}/candidate_ids.npy", allow_pickle=True
    )
    jd_embedding = np.load(
        f"{artifacts_dir}/jd_embedding.npy"
    ).astype("float32")

    retriever = bm25s.BM25.load(f"{artifacts_dir}/bm25_index")

    with open(f"{artifacts_dir}/jd_query_tokens.pkl", "rb") as f:
        jd_tokens = pickle.load(f)

    return embeddings, candidate_ids, jd_embedding, retriever, jd_tokens


def semantic_search(embeddings, jd_embedding, candidate_ids, k=1000):
    actual_k = min(k, len(embeddings))
    faiss.normalize_L2(embeddings)
    index = faiss.IndexFlatIP(embeddings.shape[1])
    index.add(embeddings)

    jd_vec = jd_embedding.reshape(1, -1).copy()
    faiss.normalize_L2(jd_vec)

    _, indices = index.search(jd_vec, actual_k)
    return [candidate_ids[i] for i in indices[0]]


def bm25_search(retriever, jd_tokens, candidate_ids, k=1000):
    actual_k = min(k, len(candidate_ids))
    results, _ = retriever.retrieve(jd_tokens, k=actual_k)
    return [candidate_ids[i] for i in results[0]]


def reciprocal_rank_fusion(ranked_lists, k=60):
    scores = {}
    for ranked_list in ranked_lists:
        for rank, doc_id in enumerate(ranked_list):
            if doc_id not in scores:
                scores[doc_id] = 0.0
            scores[doc_id] += 1.0 / (k + rank + 1)
    return sorted(scores.items(), key=lambda x: x[1], reverse=True)


def hybrid_retrieve(artifacts_dir="artifacts", top_k=1000):
    embeddings, candidate_ids, jd_embedding, retriever, jd_tokens = \
        load_artifacts(artifacts_dir)

    print("Running BM25 search...")
    bm25_ids = bm25_search(retriever, jd_tokens, candidate_ids, k=top_k)

    print("Running semantic search...")
    semantic_ids = semantic_search(
        embeddings, jd_embedding, candidate_ids, k=top_k
    )

    print("Fusing with RRF...")
    fused = reciprocal_rank_fusion([bm25_ids, semantic_ids])
    top_ids = [cid for cid, _ in fused[:top_k]]

    print(f"Retrieved {len(top_ids)} candidates after fusion")
    return top_ids