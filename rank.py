#!/usr/bin/env python3
# rank.py — Main ranking script. Must complete in under 5 minutes on CPU.
#
# Loads precomputed artifacts, runs hybrid retrieval (BM25 + semantic + RRF),
# cross‑encoder reranking, structured signal scoring, and outputs the top 100
# candidates with per‑candidate reasoning.
#
# Usage:
#     python rank.py --candidates candidates.jsonl.gz --out submission.csv
#

import argparse
import csv
import gzip
import json
import pickle
import time
from pathlib import Path

import numpy as np
from src.retrieval import hybrid_retrieve
from src.reranker import load_reranker, rerank
from src.signals import compute_final_score
from src.reasoning import generate_reasoning
from src.candidate_doc import build_candidate_doc

# ------------------------------------------------------------------------------
# Constants (must match filters.py)
# ------------------------------------------------------------------------------

FICTIONAL_COMPANIES = {
    'pied piper', 'initech', 'wayne enterprises', 'acme corp',
    'stark industries', 'hooli', 'globex inc', 'dunder mifflin'
}

GOLDEN_MAX_FREQ = 41          # Templates with frequency <= 41 are golden
MLADJ_MAX_FREQ = 187          # 42-187 are ml_adj
DATAENG_MAX_FREQ = 900        # 188-900 are data_eng
# Anything above 900 is irrelevant (mass‑produced templates)

def classify_template(freq: int) -> str:
    if freq <= GOLDEN_MAX_FREQ:
        return 'golden'
    elif freq <= MLADJ_MAX_FREQ:
        return 'ml_adj'
    elif freq <= DATAENG_MAX_FREQ:
        return 'data_eng'
    else:
        return 'irrelevant'

def load_template_map(artifacts_dir: str) -> dict:
    """
    Load jd_templates.pkl and build a per‑candidate lookup dict.
    Returns:
        {candidate_id: {
            'fingerprint': str,
            'freq': int,
            'tier': str,
            'norm_sentences': list
        }}
    """
    pkl_path = Path(artifacts_dir) / "jd_templates.pkl"
    if not pkl_path.exists():
        print("[rank] WARNING: jd_templates.pkl not found. Proceeding without template data.")
        return {}

    with open(pkl_path, "rb") as f:
        data = pickle.load(f)

    template_counter = data.get('template_counter', {})
    candidate_templates = data.get('candidate_templates', {})

    result = {}
    for cid, fingerprint in candidate_templates.items():
        freq = template_counter.get(fingerprint, 0)
        tier = classify_template(freq)
        # recover normalized sentences (if any)
        norm_sentences = fingerprint.split(" | ") if fingerprint else []
        result[cid] = {
            'fingerprint': fingerprint,
            'freq': freq,
            'tier': tier,
            'norm_sentences': norm_sentences,
        }

    golden_count = sum(1 for v in result.values() if v['tier'] == 'golden')
    mladj_count  = sum(1 for v in result.values() if v['tier'] == 'ml_adj')
    data_count   = sum(1 for v in result.values() if v['tier'] == 'data_eng')
    irr_count    = sum(1 for v in result.values() if v['tier'] == 'irrelevant')
    print(f"[rank] Template map loaded: {golden_count} golden | {mladj_count} ml_adj | "
          f"{data_count} data_eng | {irr_count} irrelevant")
    return result

def load_best_narratives(artifacts_dir: str) -> dict:
    """Load best_narratives.pkl (semantic quotes + scores)."""
    pkl_path = Path(artifacts_dir) / "best_narratives.pkl"
    if not pkl_path.exists():
        print("[rank] WARNING: best_narratives.pkl not found. Using fallback quote extraction.")
        return {}
    with open(pkl_path, "rb") as f:
        data = pickle.load(f)
    print(f"[rank] Loaded semantic narratives for {len(data):,} candidates")
    return data

def is_mostly_fictional(candidate: dict) -> bool:
    """Return True if >50% of career months are at fictional companies."""
    career = candidate.get('career_history', [])
    if not career:
        return False
    total_months = sum(j.get('duration_months', 0) for j in career)
    if total_months == 0:
        return False
    fictional_months = sum(
        j.get('duration_months', 0) for j in career
        if j.get('company', '').lower() in FICTIONAL_COMPANIES
    )
    return fictional_months / total_months > 0.5

def load_candidates(path: str) -> dict:
    """Load all candidates into a dict keyed by candidate_id."""
    candidates = {}
    if path.endswith(".json") and not path.endswith(".jsonl"):
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
        for c in data:
            candidates[c['candidate_id']] = c
        print(f"[rank] Loaded {len(candidates):,} candidates from JSON array")
        return candidates

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

def validate_output(rows: list):
    """Basic validation of output before writing."""
    assert len(rows) > 0, "No candidates to output"
    ranks = [r['rank'] for r in rows]
    assert ranks == list(range(1, len(rows) + 1)), "Ranks must be sequential 1..N"
    scores = [r['score'] for r in rows]
    assert all(scores[i] >= scores[i+1] for i in range(len(scores)-1)), "Scores must be non-increasing"
    empty_reasoning = [r for r in rows if not r['reasoning'].strip()]
    assert len(empty_reasoning) == 0, f"{len(empty_reasoning)} candidates have empty reasoning"
    print(f"[rank] Validation passed: {len(rows)} rows, sequential ranks, non‑increasing scores")

# ------------------------------------------------------------------------------
# Main
# ------------------------------------------------------------------------------

def main(candidates_path: str, output_path: str, artifacts_dir: str = "artifacts"):
    total_start = time.time()

    # --- Step 1: Load candidates ---
    t0 = time.time()
    candidates = load_candidates(candidates_path)
    print(f"[rank] Load time: {time.time()-t0:.1f}s")

    # --- Step 2: Load artifacts ---
    t0 = time.time()
    template_map = load_template_map(artifacts_dir)
    best_narratives = load_best_narratives(artifacts_dir)
    print(f"[rank] Artifacts load time: {time.time()-t0:.2f}s")

    # --- Step 3: Hybrid retrieval (BM25 + semantic + RRF) ---
    t0 = time.time()
    # Use top_k=2000 (or corpus size if smaller)
    top_k = min(2000, len(candidates))
    top_ids = hybrid_retrieve(artifacts_dir=artifacts_dir, top_k=top_k)
    print(f"[rank] Retrieval time: {time.time()-t0:.1f}s (top_k={top_k})")

    # --- Step 4: Force‑inject missing golden candidates ---
    top_ids_set = set(top_ids)
    golden_ids = [
        cid for cid, info in template_map.items()
        if info['tier'] == 'golden' and cid in candidates
    ]
    injected = 0
    for gid in golden_ids:
        if gid not in top_ids_set:
            top_ids.append(gid)
            top_ids_set.add(gid)
            injected += 1
    if injected:
        print(f"[rank] Force‑injected {injected} golden candidates into retrieval pool")

    # --- Step 5: Safety drop — remove irrelevant and fictional-heavy candidates ---
    pre_filter = len(top_ids)
    filtered_ids = []
    for cid in top_ids:
        # Drop if template is irrelevant
        if cid in template_map and template_map[cid]['tier'] == 'irrelevant':
            continue
        # Drop if mostly fictional
        cand = candidates.get(cid)
        if cand and is_mostly_fictional(cand):
            continue
        filtered_ids.append(cid)
    top_ids = filtered_ids
    print(f"[rank] Safety drop: {pre_filter} → {len(top_ids)} candidates")

    # --- Step 6: Build docs for remaining candidates ---
    t0 = time.time()
    candidate_docs = {}
    missing = 0
    for cid in top_ids:
        if cid in candidates:
            candidate_docs[cid] = build_candidate_doc(candidates[cid])
        else:
            missing += 1
    if missing > 0:
        print(f"[rank] Warning: {missing} retrieved IDs not in candidates file")
    print(f"[rank] Doc build time: {time.time()-t0:.1f}s ({len(candidate_docs):,} docs)")

    # --- Step 7: Cross‑encoder reranking ---
    t0 = time.time()
    reranker = load_reranker()
    ranked_by_ce = rerank(reranker, top_ids, candidate_docs, batch_size=32)
    print(f"[rank] Reranking time: {time.time()-t0:.1f}s")

    # --- Step 8: Structured signal scoring (now returns signal_profile) ---
    t0 = time.time()
    final_results = []  # list of (cid, final_score, signal_profile)
    for cid, ce_score in ranked_by_ce:
        if cid not in candidates:
            continue
        candidate = candidates[cid]
        jd_info = template_map.get(cid)  # may be None
        final_score, signal_profile = compute_final_score(ce_score, candidate, jd_template=jd_info)
        final_results.append((cid, final_score, signal_profile))

    # Sort by final_score descending
    final_results.sort(key=lambda x: x[1], reverse=True)
    output_size = min(100, len(final_results))
    top_results = final_results[:output_size]
    print(f"[rank] Signal scoring time: {time.time()-t0:.1f}s")

    # --- Step 9: Optional top‑10 golden check ---
    top10_tiers = []
    for cid, _, _ in top_results[:10]:
        tier = template_map.get(cid, {}).get('tier', 'unknown')
        top10_tiers.append((cid, tier))
    golden_in_top10 = sum(1 for _, t in top10_tiers if t == 'golden')
    print(f"[rank] Top‑10 template check: {golden_in_top10}/10 are golden")
    for cid, tier in top10_tiers:
        print(f"         {cid}  [{tier}]")

    # --- Step 10: Generate reasoning and write CSV ---
    t0 = time.time()
    rows = []
    for rank_pos, (cid, final_score, signal_profile) in enumerate(top_results, start=1):
        candidate = candidates[cid]
        jd_info = template_map.get(cid)
        # Retrieve semantic quote + score from best_narratives
        narrative_info = best_narratives.get(cid, {})
        semantic_quote = narrative_info.get('semantic_quote', None)
        semantic_score = narrative_info.get('semantic_score', None)

        reasoning = generate_reasoning(
            candidate,
            jd_template=jd_info,
            signal_profile=signal_profile,
            semantic_quote=semantic_quote,
            semantic_score=semantic_score
        )
        rows.append({
            'candidate_id': cid,
            'rank': rank_pos,
            'score': round(final_score, 6),
            'reasoning': reasoning
        })

    validate_output(rows)

    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f, fieldnames=["candidate_id", "rank", "score", "reasoning"]
        )
        writer.writeheader()
        writer.writerows(rows)

    print(f"[rank] Output written: {time.time()-t0:.1f}s")

    total_time = time.time() - total_start
    print("\n" + "="*50)
    print("RANKING COMPLETE")
    print("="*50)
    print(f"  Total time:         {total_time:.1f}s ({total_time/60:.1f} min)")
    print(f"  Candidates ranked:  {output_size}")
    print(f"  Output file:        {output_path}")
    print("\n  Top 5 candidates:")
    for row in rows[:5]:
        print(f"  #{row['rank']:3d} {row['candidate_id']}  score={row['score']:.4f}")
        print(f"       {row['reasoning'][:120]}...")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Rank candidates for the Redrob AI Engineer position."
    )
    parser.add_argument("--candidates", required=True,
                        help="Path to candidates file (.jsonl, .jsonl.gz, or .json array)")
    parser.add_argument("--out", required=True,
                        help="Output CSV path (e.g., submission.csv)")
    parser.add_argument("--artifacts", default="artifacts",
                        help="Path to precomputed artifacts directory (default: artifacts/)")
    args = parser.parse_args()
    main(args.candidates, args.out, args.artifacts)