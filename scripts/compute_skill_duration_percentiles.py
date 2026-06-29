#!/usr/bin/env python3
"""
compute_skill_duration_percentiles.py

Reads all candidates from candidates.jsonl.gz, extracts duration_months
for every skill, and computes statistical bounds (P95, P99, mean, median, max)
for each skill.

Saves the results to artifacts/skill_duration_percentiles.pkl
"""

import gzip
import json
import pickle
from collections import defaultdict
from pathlib import Path
import numpy as np
from tqdm import tqdm

# If you want to limit to CORE_JD_SKILLS, define them here.
# Otherwise, we compute for all skills (recommended, then you can filter later).
CORE_JD_SKILLS = {
    'faiss', 'pinecone', 'weaviate', 'qdrant', 'milvus', 'elasticsearch',
    'opensearch', 'pgvector', 'vector search', 'semantic search',
    'information retrieval', 'hybrid search', 'bm25',
    'sentence transformers', 'embeddings', 'hugging face transformers',
    'learning to rank', 'recommendation systems', 'lora', 'qlora', 'peft',
    'fine-tuning llms', 'haystack', 'llamaindex', 'pytorch', 'tensorflow',
    'scikit-learn', 'nlp', 'machine learning', 'deep learning', 'python', 'rag',
    'prompt engineering', 'langchain', 'llms'
}


def load_candidates(path: str):
    """Generator that yields candidates one by one from .jsonl or .jsonl.gz"""
    opener = gzip.open if path.endswith(".gz") else open
    with opener(path, "rt", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                try:
                    yield json.loads(line)
                except json.JSONDecodeError:
                    continue


def main(candidates_path: str, output_dir: str = "artifacts"):
    # Dictionary: skill_name -> list of durations (months)
    skill_durations = defaultdict(list)

    print(f"[compute] Reading candidates from {candidates_path}...")
    # Count total candidates for progress bar? We'll just iterate.
    total_candidates = 0
    for c in tqdm(load_candidates(candidates_path), desc="Processing candidates"):
        total_candidates += 1
        for skill in c.get('skills', []):
            name = skill.get('name', '').strip().lower()
            dur = skill.get('duration_months', 0)
            if dur > 0:  # Ignore zero durations (they are suspicious anyway)
                skill_durations[name].append(dur)

    print(f"[compute] Processed {total_candidates} candidates.")
    print(f"[compute] Found {len(skill_durations)} unique skills.")

    # Compute statistics for each skill
    results = {}
    for skill, durations in tqdm(skill_durations.items(), desc="Computing percentiles"):
        arr = np.array(durations)
        results[skill] = {
            'count': len(durations),
            'min': float(np.min(arr)),
            'max': float(np.max(arr)),
            'mean': float(np.mean(arr)),
            'median': float(np.median(arr)),
            'p95': float(np.percentile(arr, 95)),
            'p99': float(np.percentile(arr, 99)),
        }

    # Print top skills (by count) with their P95
    print("\n[compute] === Top 20 skills by frequency ===")
    sorted_skills = sorted(results.items(), key=lambda x: x[1]['count'], reverse=True)
    for skill, stats in sorted_skills[:20]:
        print(f"  {skill:30s} count={stats['count']:6d}  P95={stats['p95']:6.1f}mo  max={stats['max']:6.1f}mo")

    # Save to pickle
    output_path = Path(output_dir) / "skill_duration_percentiles.pkl"
    output_path.parent.mkdir(exist_ok=True)
    with open(output_path, "wb") as f:
        pickle.dump(results, f)
    print(f"\n[compute] Saved to {output_path}")

    # Also print a quick sanity check for LLM-era skills
    print("\n[compute] === LLM-era skills P95 ===")
    llm_skills = ['langchain', 'rag', 'qlora', 'lora', 'peft', 'prompt engineering', 'fine-tuning llms', 'llamaindex']
    for skill in llm_skills:
        if skill in results:
            print(f"  {skill:20s} P95={results[skill]['p95']:.1f}mo  max={results[skill]['max']:.1f}mo")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--candidates", required=True, help="Path to candidates.jsonl.gz")
    parser.add_argument("--output_dir", default="artifacts")
    args = parser.parse_args()
    main(args.candidates, args.output_dir)