from sentence_transformers import CrossEncoder
import math


JD_QUERY = """
Senior AI Engineer at a product company, 5-9 years experience.
Production experience building embedding-based retrieval systems using
sentence-transformers, BGE, E5, OpenAI embeddings.
Vector databases: Pinecone, Weaviate, Qdrant, Milvus, FAISS, Elasticsearch.
Hybrid search, dense retrieval, sparse retrieval, BM25.
Ranking systems: NDCG, MRR, MAP, learning-to-rank, XGBoost, LightGBM.
LLM integration, fine-tuning: LoRA, QLoRA, PEFT.
Shipped recommendation systems and search systems to real users at scale.
Evaluation frameworks, A/B testing, offline-online correlation.
Strong Python. Product company experience preferred over consulting firms.
India location, Pune or Noida preferred.
"""


def load_reranker():
    return CrossEncoder(
        "mixedbread-ai/mxbai-rerank-xsmall-v1",
        max_length=512
    )


def rerank(model, top_ids, candidate_docs, batch_size=32):
    pairs = [(JD_QUERY, candidate_docs[cid]) for cid in top_ids]

    print(f"Re-ranking {len(pairs)} candidates...")
    raw_scores = model.predict(pairs, batch_size=batch_size, show_progress_bar=True)

    # Normalize logits to 0-1 via sigmoid
    normalized = [1 / (1 + math.exp(-s)) for s in raw_scores]

    ranked = sorted(
        zip(top_ids, normalized),
        key=lambda x: x[1],
        reverse=True
    )
    return ranked  # list of (candidate_id, score)