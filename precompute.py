#!/usr/bin/env python3
# precompute.py — Build and save all artifacts needed by rank.py.
#
# Run ONCE offline before submission. Takes 30‑90 minutes for 100K candidates.
#
# Usage:
#     python precompute.py --candidates candidates.jsonl.gz
#     python precompute.py --candidates sample_candidates.jsonl  # for testing
#
# Artifacts saved to ./artifacts/:
#     candidate_ids.npy        — aligned IDs (same order as embeddings)
#     embeddings.npy           — 8192‑dim dense vectors for each candidate
#     jd_embedding.npy         — JD query embedding
#     bm25_index/              — BM25 index (saved via bm25s)
#     jd_query_tokens.pkl      — tokenized JD query for BM25
#     jd_templates.pkl         — template classification map (from analyze_jd_templates.py)
#     best_narratives.pkl      — semantic quotes + scores for golden candidates

import argparse
import gzip
import json
import pickle
import re
import time
from pathlib import Path

import numpy as np
import bm25s
from tqdm import tqdm
from sentence_transformers import SentenceTransformer

from src.filters import filter_candidates
from src.candidate_doc import build_candidate_doc
from src.reranker import JD_QUERY

# ------------------------------------------------------------------------------
# Constants
# ------------------------------------------------------------------------------

ARTIFACTS_DIR = Path("artifacts")
BATCH_SIZE = 64  # for embedding generation
MODEL_NAME = "nomic-ai/nomic-embed-text-v1.5"

# Golden template classification thresholds (must match rank.py)
GOLDEN_MAX_FREQ = 41


def classify_template(freq: int) -> str:
    if freq <= GOLDEN_MAX_FREQ:
        return 'golden'
    else:
        return 'other'


# ------------------------------------------------------------------------------
# Text helpers (must match analyze_jd_templates.py)
# ------------------------------------------------------------------------------

def normalize_sentence(sent: str) -> str:
    sent = re.sub(r'\d+\.?\d*', 'XX', sent)
    sent = re.sub(r'\b\w+\.ai\b', 'COMPANY', sent)
    sent = re.sub(r'\(.*?\)', '', sent)
    sent = re.sub(r'\s+', ' ', sent).strip()
    return sent


def extract_sentences(text: str) -> list:
    if not text:
        return []
    raw = re.split(r'[.!?\n]', text)
    return [s.strip() for s in raw if len(s.strip()) > 15]


def get_jd_fingerprint(job_desc: str) -> str:
    sents = extract_sentences(job_desc)
    if not sents:
        return ""
    normalized = [normalize_sentence(s) for s in sents]
    return " | ".join(normalized)


# ------------------------------------------------------------------------------
# Loading
# ------------------------------------------------------------------------------

def load_candidates(path: str) -> list:
    """Load candidates from .jsonl or .jsonl.gz."""
    candidates = []
    opener = gzip.open if path.endswith(".gz") else open
    with opener(path, "rt", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    candidates.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
    print(f"[precompute] Loaded {len(candidates):,} candidates from {path}")
    return candidates


def load_jd_templates(artifacts_dir: str) -> dict:
    """
    Load jd_templates.pkl and build:
        - template_counter: fingerprint -> frequency
        - candidate_templates: candidate_id -> fingerprint
    """
    pkl_path = Path(artifacts_dir) / "jd_templates.pkl"
    if not pkl_path.exists():
        print("[precompute] WARNING: jd_templates.pkl not found. "
              "Semantic quote selection will be skipped.")
        return {}

    with open(pkl_path, "rb") as f:
        data = pickle.load(f)

    template_counter = data.get('template_counter', {})
    candidate_templates = data.get('candidate_templates', {})

    print(f"[precompute] Loaded template map: {len(template_counter):,} unique templates, "
          f"{len(candidate_templates):,} candidates mapped")
    return {
        'template_counter': template_counter,
        'candidate_templates': candidate_templates,
    }


# ------------------------------------------------------------------------------
# Semantic quote selection (NEW)
# ------------------------------------------------------------------------------

def select_semantic_quotes(
    candidates: list,
    jd_embedding: np.ndarray,
    model: SentenceTransformer,
    template_data: dict
) -> dict:
    """
    For each golden candidate (template freq <= 41), extract sentences from
    their current JD, encode them, compute cosine similarity to the JD embedding,
    and store the best (quote, score).

    Returns:
        {candidate_id: {'semantic_quote': str, 'semantic_score': float}}
    """
    if not template_data:
        print("[precompute] No template data – skipping semantic quote selection.")
        return {}

    template_counter = template_data.get('template_counter', {})
    candidate_templates = template_data.get('candidate_templates', {})

    # Identify golden candidates
    golden_candidates = []
    for cid, fingerprint in candidate_templates.items():
        freq = template_counter.get(fingerprint, 0)
        if freq <= GOLDEN_MAX_FREQ:
            golden_candidates.append(cid)

    if not golden_candidates:
        print("[precompute] No golden candidates found – skipping semantic quote selection.")
        return {}

    print(f"[precompute] Selecting semantic quotes for {len(golden_candidates):,} golden candidates...")

    # Build a mapping: candidate_id -> candidate
    cand_map = {c['candidate_id']: c for c in candidates}

    results = {}
    # Normalize the JD embedding for cosine similarity
    jd_vec = jd_embedding.reshape(1, -1)
    jd_norm = jd_vec / np.linalg.norm(jd_vec, axis=1, keepdims=True)

    for cid in tqdm(golden_candidates, desc="Semantic quotes"):
        candidate = cand_map.get(cid)
        if not candidate:
            continue

        career = candidate.get('career_history', [])
        if not career:
            continue

        current_job = career[0]
        desc = current_job.get('description', '')
        if not desc:
            continue

        sentences = extract_sentences(desc)
        if not sentences:
            continue

        # Encode all sentences (batch‑size 1 to keep it simple)
        # The model is already loaded in GPU/CPU memory
        try:
            embeds = model.encode(
                sentences,
                convert_to_numpy=True,
                normalize_embeddings=True,  # L2 normalize for cosine similarity
                show_progress_bar=False,
            )
        except Exception as e:
            # Fallback: encode one by one if batch fails
            embeds = []
            for sent in sentences:
                try:
                    e = model.encode(sent, convert_to_numpy=True, normalize_embeddings=True)
                    embeds.append(e)
                except Exception:
                    embeds.append(np.zeros(model.get_sentence_embedding_dimension()))
            embeds = np.array(embeds)

        # Compute cosine similarity (already normalized)
        similarities = embeds @ jd_norm.T  # shape (n_sentences, 1)
        similarities = similarities.flatten()

        # Pick the best sentence (highest similarity)
        best_idx = np.argmax(similarities)
        best_score = float(similarities[best_idx])
        best_quote = sentences[best_idx]

        # Only store if similarity is > 0.3 (otherwise it's basically noise)
        if best_score > 0.3:
            results[cid] = {
                'semantic_quote': best_quote,
                'semantic_score': best_score,
            }

    print(f"[precompute] Extracted semantic quotes for {len(results):,} golden candidates")
    return results


# ------------------------------------------------------------------------------
# Main
# ------------------------------------------------------------------------------

def main(candidates_path: str, artifacts_dir: str = "artifacts"):
    total_start = time.time()
    artifacts_path = Path(artifacts_dir)
    artifacts_path.mkdir(exist_ok=True, parents=True)

    # --- Step 1: Load candidates ---
    t0 = time.time()
    candidates = load_candidates(candidates_path)
    print(f"[precompute] Load time: {time.time()-t0:.1f}s")

    # --- Step 2: Apply filters ---
    t0 = time.time()
    print("\n[precompute] Filtering candidates (honeypots + hard filters)...")
    filtered = filter_candidates(candidates)
    print(f"[precompute] Filter time: {time.time()-t0:.1f}s")
    print(f"[precompute] Candidates after filtering: {len(filtered):,}")

    if len(filtered) == 0:
        raise ValueError("All candidates were filtered out. Check filters.py.")

    # --- Step 3: Load template data (for semantic quote selection) ---
    template_data = load_jd_templates(artifacts_dir)
    # If template data is missing, semantic quote selection will be skipped.

    # --- Step 4: Build candidate documents ---
    t0 = time.time()
    print(f"\n[precompute] Building candidate documents ({len(filtered):,} candidates)...")
    docs = []
    candidate_ids = []
    for c in tqdm(filtered, desc="Building docs"):
        doc = build_candidate_doc(c)
        docs.append(doc)
        candidate_ids.append(c['candidate_id'])

    candidate_ids_arr = np.array(candidate_ids)
    np.save(artifacts_path / "candidate_ids.npy", candidate_ids_arr)
    print(f"[precompute] Saved candidate_ids.npy ({len(candidate_ids_arr):,} entries)")
    print(f"[precompute] Doc build time: {time.time()-t0:.1f}s")

    # --- Step 5: Build BM25 index ---
    t0 = time.time()
    print("\n[precompute] Building BM25 index...")
    tokenized_corpus = bm25s.tokenize(docs, show_progress=True)
    retriever = bm25s.BM25()
    retriever.index(tokenized_corpus)

    bm25_dir = artifacts_path / "bm25_index"
    bm25_dir.mkdir(exist_ok=True)
    retriever.save(str(bm25_dir))
    print(f"[precompute] Saved BM25 index to {bm25_dir}")
    print(f"[precompute] BM25 index time: {time.time()-t0:.1f}s")

    # --- Step 6: Tokenize JD query for BM25 ---
    t0 = time.time()
    jd_tokens = bm25s.tokenize([JD_QUERY], show_progress=False)
    with open(artifacts_path / "jd_query_tokens.pkl", "wb") as f:
        pickle.dump(jd_tokens, f)
    print(f"[precompute] Saved JD query tokens")
    print(f"[precompute] Tokenization time: {time.time()-t0:.1f}s")

    # --- Step 7: Generate embeddings ---
    t0 = time.time()
    print("\n[precompute] Loading nomic-embed-text-v1.5 model...")
    print(" (First run will download ~500MB model – cached after that)")
    model = SentenceTransformer(MODEL_NAME, trust_remote_code=True)

    print(f"[precompute] Encoding {len(docs):,} candidate documents...")
    print(" (This is the slow step – ~30-60 min for 100K on CPU)")
    embeddings = model.encode(
        docs,
        batch_size=BATCH_SIZE,
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=False,  # we normalize at retrieval time
    )

    np.save(artifacts_path / "embeddings.npy", embeddings)
    emb_size_mb = embeddings.nbytes / (1024 * 1024)
    print(f"[precompute] Saved embeddings.npy – shape: {embeddings.shape}, size: {emb_size_mb:.0f}MB")
    print(f"[precompute] Embedding time: {time.time()-t0:.1f}s")

    # --- Step 8: Encode JD query ---
    t0 = time.time()
    print("\n[precompute] Encoding JD query...")
    jd_embedding = model.encode(
        [JD_QUERY],
        convert_to_numpy=True,
        normalize_embeddings=False,
    )
    np.save(artifacts_path / "jd_embedding.npy", jd_embedding[0])
    print(f"[precompute] Saved jd_embedding.npy")
    print(f"[precompute] JD encoding time: {time.time()-t0:.1f}s")

    # --- Step 9: Semantic quote selection (golden candidates only) ---
    t0 = time.time()
    print("\n[precompute] Selecting semantic quotes for golden candidates...")
    # We need to reload the candidate mapping for the filtered candidates
    # (the filtering step may have reordered or dropped candidates)
    # We use the original filtered list to build the quote map.
    best_narratives = select_semantic_quotes(
        filtered,
        jd_embedding[0],
        model,
        template_data
    )
    # Also include template frequency in the output for reasoning
    # (the rank.py will merge this with the template map)
    # We store the quote + score directly.
    with open(artifacts_path / "best_narratives.pkl", "wb") as f:
        pickle.dump(best_narratives, f)
    print(f"[precompute] Saved best_narratives.pkl ({len(best_narratives):,} entries)")
    print(f"[precompute] Semantic quote selection time: {time.time()-t0:.1f}s")

    # --- Step 10: Summary ---
    total_time = time.time() - total_start
    print("\n" + "="*60)
    print("PRECOMPUTATION COMPLETE")
    print("="*60)
    print(f"  Total time:          {total_time:.1f}s ({total_time/60:.1f} min)")
    print(f"  Candidates indexed:  {len(candidate_ids_arr):,}")
    print(f"  Embedding shape:     {embeddings.shape}")
    print(f"  Embedding size:      {emb_size_mb:.0f} MB")
    print(f"  Artifacts saved to:  {artifacts_path}/")
    print("\n  Files created:")
    for f in sorted(artifacts_path.rglob("*")):
        if f.is_file():
            size_kb = f.stat().st_size / 1024
            print(f"    {f} ({size_kb:.0f} KB)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Precompute BM25 index and semantic embeddings for candidate ranking."
    )
    parser.add_argument(
        "--candidates",
        required=True,
        help="Path to candidates file (.jsonl, .jsonl.gz, or .json array)"
    )
    parser.add_argument(
        "--artifacts",
        default="artifacts",
        help="Path to artifacts directory (default: artifacts/)"
    )
    args = parser.parse_args()
    main(args.candidates, args.artifacts)