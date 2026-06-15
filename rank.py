"""
Main ranking script. Must complete in under 5 minutes on CPU.
Loads precomputed artifacts, runs retrieval + reranking + signal scoring.

Usage:
    uv run python rank.py --candidates candidates.jsonl.gz --out team_xxx.csv
"""

import argparse
import csv
import gzip
import json

from src.retrieval import hybrid_retrieve
from src.reranker import load_reranker, rerank
from src.signals import compute_final_score
from src.reasoning import generate_reasoning
from src.candidate_doc import build_candidate_doc


def load_candidates(path):
    candidates = {}
    opener = gzip.open if path.endswith(".gz") else open
    with opener(path, "rt", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                c = json.loads(line)
                candidates[c['candidate_id']] = c
    return candidates


def main(candidates_path, output_path, artifacts_dir="artifacts"):
    print("Loading candidates...")
    candidates = load_candidates(candidates_path)

    print("Running hybrid retrieval...")
    top_k = min(1000, len(candidates))
    top_ids = hybrid_retrieve(artifacts_dir=artifacts_dir, top_k=top_k)

    # Build docs for top 1000 only (fast)
    candidate_docs = {
        cid: build_candidate_doc(candidates[cid])
        for cid in top_ids
        if cid in candidates
    }

    print("Loading cross-encoder...")
    reranker = load_reranker()

    print("Re-ranking top 1000...")
    ranked = rerank(reranker, top_ids, candidate_docs)

    print("Applying structured signal scoring...")
    final_scores = []
    for cid, ce_score in ranked:
        if cid not in candidates:
            continue
        candidate = candidates[cid]
        score = compute_final_score(ce_score, candidate)
        final_scores.append((cid, score))

    final_scores.sort(key=lambda x: x[1], reverse=True)
    output_size = min(100, len(final_scores))
    top_100 = final_scores[:output_size]

    print(f"Writing submission to {output_path}...")
    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["candidate_id", "rank", "score", "reasoning"])
        for rank, (cid, score) in enumerate(top_100, start=1):
            reasoning = generate_reasoning(candidates[cid])
            writer.writerow([cid, rank, round(score, 6), reasoning])

    print("Done.")
    print(f"Top candidate: {top_100[0][0]} with score {top_100[0][1]:.4f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--candidates", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--artifacts", default="artifacts")
    args = parser.parse_args()
    main(args.candidates, args.out, args.artifacts)