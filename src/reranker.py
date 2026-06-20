"""
reranker.py — Cross-encoder re-ranking using mxbai-rerank-xsmall-v1.

The JD_QUERY is a distilled version of the job description containing
only technical requirements. Prose, culture notes, and negative examples
are stripped to avoid diluting the embedding signal.

Key insight: The cross-encoder sees both the JD query and candidate doc
together — this captures fit that independent embeddings miss.
"""

import math
from sentence_transformers import CrossEncoder

# Distilled JD query — only technical and role requirements
# Stripped: culture fit prose, location logistics, negative examples section,
#           "things we don't want" sections, company description boilerplate
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


def load_reranker(model_name: str = "mixedbread-ai/mxbai-rerank-xsmall-v1") -> CrossEncoder:
    """Load the cross-encoder model."""
    print(f"[reranker] Loading {model_name}...")
    model = CrossEncoder(model_name, max_length=512)
    print("[reranker] Model loaded")
    return model


def rerank(
    model: CrossEncoder,
    top_ids: list,
    candidate_docs: dict,
    batch_size: int = 32
) -> list:
    """
    Re-rank candidates using cross-encoder.
    
    Returns list of (candidate_id, normalized_score) sorted descending.
    Scores are sigmoid-normalized logits → range (0, 1).
    """
    # Filter to only candidates we have docs for
    valid_ids = [cid for cid in top_ids if cid in candidate_docs]

    if not valid_ids:
        return []

    pairs = [(JD_QUERY, candidate_docs[cid]) for cid in valid_ids]

    print(f"[reranker] Scoring {len(pairs):,} candidates...")
    raw_scores = model.predict(
        pairs,
        batch_size=batch_size,
        show_progress_bar=True
    )

    # Sigmoid normalize: convert logits to 0-1 probability
    normalized = [1.0 / (1.0 + math.exp(-float(s))) for s in raw_scores]

    ranked = sorted(
        zip(valid_ids, normalized),
        key=lambda x: x[1],
        reverse=True
    )

    print(f"[reranker] Top score: {ranked[0][1]:.4f}, Bottom score: {ranked[-1][1]:.4f}")
    return ranked