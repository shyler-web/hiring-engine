"""
rank.py — Main ranking script. Must complete in under 5 minutes on CPU.

Loads precomputed artifacts, runs retrieval + re-ranking + signal scoring,
outputs top 100 candidates with per-candidate reasoning.

Usage:
    uv run python rank.py --candidates candidates.jsonl.gz --out team_xxx.csv
    uv run python rank.py --candidates sample_candidates.jsonl --out test_submission.csv
"""

import argparse
import csv
import gzip
import json
import time
from pathlib import Path

from src.retrieval import hybrid_retrieve
from src.reranker import load_reranker, rerank
from src.signals import compute_final_score
from src.reasoning import generate_reasoning
from src.candidate_doc import build_candidate_doc


def load_candidates(path: str) -> dict:
    """
    Load all candidates into a dict keyed by candidate_id.
    Handles: .jsonl, .jsonl.gz, .json (array format for sample).
    """
    candidates = {}

    # JSON array format (sample_candidates.json)
    if path.endswith(".json") and not path.endswith(".jsonl"):
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
        for c in data:
            candidates[c['candidate_id']] = c
        print(f"[rank] Loaded {len(candidates):,} candidates from JSON array")
        return candidates

    # JSONL or JSONL.GZ
    opener = gzip.open if path.endswith(".gz") else open
    with opener(path, "rt", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    c = json.loads(line)
                    candidates[c['candidate_id']] = c
                except (json.JSONDecodeError, KeyError):
                    continue

    print(f"[rank] Loaded {len(candidates):,} candidates")
    return candidates


def validate_output(rows: list, output_path: str):
    """Basic validation of output before writing."""
    assert len(rows) > 0, "No candidates to output"

    ranks = [r['rank'] for r in rows]
    assert ranks == list(range(1, len(rows) + 1)), "Ranks must be sequential 1..N"

    scores = [r['score'] for r in rows]
    assert all(
        scores[i] >= scores[i+1] for i in range(len(scores)-1)
    ), "Scores must be non-increasing"

    empty_reasoning = [r for r in rows if not r['reasoning'].strip()]
    assert len(empty_reasoning) == 0, f"{len(empty_reasoning)} candidates have empty reasoning"

    print(f"[rank] Validation passed: {len(rows)} rows, sequential ranks, non-increasing scores")


def main(candidates_path: str, output_path: str, artifacts_dir: str = "artifacts"):
    total_start = time.time()

    # --- Step 1: Load candidates ---
    t0 = time.time()
    candidates = load_candidates(candidates_path)
    print(f"[rank] Load time: {time.time()-t0:.1f}s")

    # --- Step 2: Hybrid retrieval (BM25 + semantic + RRF) ---
    t0 = time.time()
    # Cap top_k at corpus size for small test sets
    top_k = min(1000, len(candidates))
    top_ids = hybrid_retrieve(artifacts_dir=artifacts_dir, top_k=top_k)
    print(f"[rank] Retrieval time: {time.time()-t0:.1f}s")

    # --- Step 3: Build docs for top candidates only (fast) ---
    t0 = time.time()
    candidate_docs = {}
    missing = 0
    for cid in top_ids:
        if cid in candidates:
            candidate_docs[cid] = build_candidate_doc(candidates[cid])
        else:
            missing += 1
    if missing > 0:
        print(f"[rank] Warning: {missing} retrieved IDs not found in candidates file")
    print(f"[rank] Doc build time: {time.time()-t0:.1f}s ({len(candidate_docs):,} docs)")

    # --- Step 4: Cross-encoder re-ranking ---
    t0 = time.time()
    reranker = load_reranker()
    ranked_by_ce = rerank(reranker, top_ids, candidate_docs, batch_size=32)
    print(f"[rank] Re-ranking time: {time.time()-t0:.1f}s")

    # --- Step 5: Structured signal scoring ---
    t0 = time.time()
    final_scores = []
    for cid, ce_score in ranked_by_ce:
        if cid not in candidates:
            continue
        candidate = candidates[cid]
        final_score = compute_final_score(ce_score, candidate)
        final_scores.append((cid, final_score))

    # Sort by final score descending
    final_scores.sort(key=lambda x: x[1], reverse=True)

    # Take top 100 (or fewer if corpus is small)
    output_size = min(100, len(final_scores))
    top_results = final_scores[:output_size]
    print(f"[rank] Signal scoring time: {time.time()-t0:.1f}s")

    # --- Step 6: Generate reasoning and write CSV ---
    t0 = time.time()
    rows = []
    for rank_pos, (cid, score) in enumerate(top_results, start=1):
        reasoning = generate_reasoning(candidates[cid])
        rows.append({
            'candidate_id': cid,
            'rank': rank_pos,
            'score': round(score, 6),
            'reasoning': reasoning
        })

    # Validate before writing
    validate_output(rows, output_path)

    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["candidate_id", "rank", "score", "reasoning"]
        )
        writer.writeheader()
        writer.writerows(rows)

    print(f"[rank] Output written: {time.time()-t0:.1f}s")

    # --- Summary ---
    total_time = time.time() - total_start
    print("\n" + "="*50)
    print("RANKING COMPLETE")
    print("="*50)
    print(f"  Total time:         {total_time:.1f}s ({total_time/60:.1f} min)")
    print(f"  Candidates ranked:  {output_size}")
    print(f"  Output file:        {output_path}")
    print(f"\n  Top 5 candidates:")
    for row in rows[:5]:
        print(f"  #{row['rank']:3d} {row['candidate_id']}  score={row['score']:.4f}")
        print(f"       {row['reasoning'][:100]}...")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Rank candidates for the Redrob AI Engineer position."
    )
    parser.add_argument(
        "--candidates",
        required=True,
        help="Path to candidates file (.jsonl, .jsonl.gz, or .json array)"
    )
    parser.add_argument(
        "--out",
        required=True,
        help="Output CSV path (e.g. team_xxx.csv)"
    )
    parser.add_argument(
        "--artifacts",
        default="artifacts",
        help="Path to precomputed artifacts directory (default: artifacts/)"
    )
    args = parser.parse_args()
    main(args.candidates, args.out, args.artifacts)