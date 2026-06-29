#!/usr/bin/env python3
# rank.py — Main ranking script.
#
# Loads precomputed artifacts, runs hybrid retrieval (BM25 + semantic + RRF),
# cross‑encoder reranking, structured signal scoring, and outputs the top 100.
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
# Constants
# ------------------------------------------------------------------------------

FICTIONAL_COMPANIES = {
    'pied piper', 'initech', 'wayne enterprises', 'acme corp',
    'stark industries', 'hooli', 'globex inc', 'dunder mifflin'
}

GOLDEN_MAX_FREQ = 41
MLADJ_MAX_FREQ = 187
DATAENG_MAX_FREQ = 900


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
    """Load jd_templates_enhanced.pkl (fallback to .pkl) and build lookup dict."""
    pkl_path = Path(artifacts_dir) / "jd_templates_enhanced.pkl"
    if not pkl_path.exists():
        pkl_path = Path(artifacts_dir) / "jd_templates.pkl"
        print("[rank] WARNING: No jd_templates_enhanced.pkl found.")
        if not pkl_path.exists():
            print("[rank] WARNING: No jd_templates.pkl found.")
            return {}

    with open(pkl_path, "rb") as f:
        data = pickle.load(f)

    template_counter = data.get('template_counter', {})
    candidate_templates = data.get('candidate_templates', {})
    template_summaries = data.get('template_summaries', {})

    result = {}
    for cid, fingerprint in candidate_templates.items():
        freq = template_counter.get(fingerprint, 0)
        tier = classify_template(freq)
        summary = template_summaries.get(fingerprint, None)
        result[cid] = {
            'fingerprint': fingerprint,
            'freq': freq,
            'tier': tier,
            'summary': summary,
        }
    print(f"[rank] Template map loaded: {sum(1 for v in result.values() if v['tier'] == 'golden')} golden")
    return result


# def is_mostly_fictional(candidate: dict) -> bool:
#     career = candidate.get('career_history', [])
#     if not career:
#         return False
#     total_months = sum(j.get('duration_months', 0) for j in career)
#     if total_months == 0:
#         return False
#     fictional_months = sum(
#         j.get('duration_months', 0) for j in career
#         if j.get('company', '').lower() in FICTIONAL_COMPANIES
#     )
#     return fictional_months / total_months > 0.5


def load_candidates(path: str) -> dict:
    candidates = {}
    opener = gzip.open if path.endswith(".gz") else open
    with opener(path, "rt", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                try:
                    c = json.loads(line)
                    candidates[c['candidate_id']] = c
                except (json.JSONDecodeError, KeyError):
                    continue
    print(f"[rank] Loaded {len(candidates):,} candidates")
    return candidates


def validate_output(rows: list):
    assert len(rows) > 0, "No candidates to output"
    ranks = [r['rank'] for r in rows]
    assert ranks == list(range(1, len(rows) + 1)), "Ranks must be sequential"
    scores = [r['score'] for r in rows]
    assert all(scores[i] >= scores[i+1] for i in range(len(scores)-1)), "Scores must be non-increasing"
    print(f"[rank] Validation passed.")


def load_survivor_ids(artifacts_dir: str):
    """Load candidate_ids.npy and return a set of survivor IDs."""
    path = Path(artifacts_dir) / "candidate_ids.npy"
    if path.exists():
        ids = np.load(path, allow_pickle=True)
        survivor_set = set(ids)
        print(f"[rank] Loaded {len(survivor_set):,} survivor IDs from precompute")
        return survivor_set
    else:
        print("[rank] WARNING: candidate_ids.npy not found. Injecting all golden candidates (riskier).")
        return None


# ------------------------------------------------------------------------------
# Main
# ------------------------------------------------------------------------------

def main(candidates_path: str, output_path: str, artifacts_dir: str = "artifacts"):
    total_start = time.time()

    # --- Step 1: Load candidates ---
    t0 = time.time()
    candidates = load_candidates(candidates_path)
    print(f"[rank] Load time: {time.time()-t0:.1f}s")

    # --- Step 2: Load template map ---
    t0 = time.time()
    template_map = load_template_map(artifacts_dir)
    print(f"[rank] Template map load time: {time.time()-t0:.2f}s")
    print(f"[DEBUG] template_map size: {len(template_map)}")
    if template_map:
        sample_cid = next(iter(template_map))
        sample = template_map[sample_cid]
        print(f"[DEBUG] Sample candidate: {sample_cid}")
        print(f"[DEBUG]   tier: {sample.get('tier')}")
        print(f"[DEBUG]   freq: {sample.get('freq')}")
        print(f"[DEBUG]   summary: {sample.get('summary')[:50] if sample.get('summary') else 'None'}...")
    else:
        print("[DEBUG] template_map is EMPTY! Check load_template_map().")

    # --- Step 3: Load survivor IDs (critical for safe force‑inject) ---
    t0 = time.time()
    survivor_ids = load_survivor_ids(artifacts_dir)
    print(f"[rank] Survivor IDs load time: {time.time()-t0:.2f}s")

    # --- Step 4: Hybrid retrieval ---
    t0 = time.time()
    top_k = min(2000, len(candidates))
    top_ids = hybrid_retrieve(artifacts_dir=artifacts_dir, top_k=top_k)
    print(f"[rank] Retrieval time: {time.time()-t0:.1f}s (top_k={top_k})")

    # --- Step 5: Force‑inject only valid (survived) golden candidates ---
    top_ids_set = set(top_ids)
    golden_ids = [
        cid for cid, info in template_map.items()
        if info.get('tier') == 'golden' and cid in candidates
    ]
    injected = 0
    skipped = 0
    if survivor_ids is not None:
        for gid in golden_ids:
            if gid in survivor_ids:
                if gid not in top_ids_set:
                    top_ids.append(gid)
                    top_ids_set.add(gid)
                    injected += 1
            else:
                skipped += 1
    else:
        # Fallback: inject all golden (riskier, but better than missing them)
        print("[rank] No Golden candidate filtering (survivor_ids not loaded). no golden candidates injection.")
        # for gid in golden_ids:
        #     if gid not in top_ids_set:
        #         top_ids.append(gid)
        #         top_ids_set.add(gid)
        #         injected += 1

    if injected:
        print(f"[rank] Force‑injected {injected} valid golden candidates")
    if skipped:
        print(f"[rank] Skipped {skipped} golden candidates (filtered out by precompute)")

    # # --- Step 6: Safety drop ---
    # pre_filter = len(top_ids)
    # filtered_ids = []s
    # for cid in top_ids:
    #     if cid in template_map and template_map[cid].get('tier') == 'irrelevant':
    #         continue
    #     cand = candidates.get(cid)
    #     # if cand and is_mostly_fictional(cand):
    #     #     continue
    #     filtered_ids.append(cid)
    # top_ids = filtered_ids
    # print(f"[rank] Safety drop: {pre_filter} → {len(top_ids)} candidates")

    # --- Step 7: Build docs for reranker (pass template summary) ---
    t0 = time.time()
    candidate_docs = {}
    for cid in top_ids:
        if cid in candidates:
            summary = template_map.get(cid, {}).get('summary', None) if template_map else print(f"[rank] Template map does not exist for candidate: {cid}")
            candidate_docs[cid] = build_candidate_doc(candidates[cid], template_summary=summary)
    print(f"[rank] Doc build time: {time.time()-t0:.1f}s ({len(candidate_docs):,} docs)")

    # --- Step 8: Cross‑encoder reranking ---
    top_ids = top_ids[:500]  # ⬅️ ADD THIS LINE – cap candidates sent to the reranker
    t0 = time.time()
    reranker = load_reranker(artifacts_dir=artifacts_dir)
    ranked_by_ce = rerank(reranker, top_ids, candidate_docs, batch_size=32)
    print(f"[rank] Reranking time: {time.time()-t0:.1f}s")

    # --- Step 9: Signal scoring ---
    t0 = time.time()
    final_results = []
    for cid, ce_score in ranked_by_ce:
        if cid not in candidates:
            continue
        candidate = candidates[cid]
        jd_info = template_map.get(cid)
        final_score, signal_profile = compute_final_score(ce_score, candidate, jd_template=jd_info)
        final_results.append((cid, final_score, signal_profile))

    final_results.sort(key=lambda x: x[1], reverse=True)
    output_size = min(100, len(final_results))
    top_results = final_results[:output_size]
    print(f"[rank] Signal scoring time: {time.time()-t0:.1f}s")

    # --- Step 10: Top‑10 golden check ---
    print("[rank] Top‑10 template tiers:")
    for cid, _, _ in top_results[:10]:
        tier = template_map.get(cid, {}).get('tier', 'unknown')
        print(f"         {cid}  [{tier}]")

    # --- Step 11: Generate reasoning and write CSV ---
    t0 = time.time()
    rows = []
    for rank_pos, (cid, final_score, signal_profile) in enumerate(top_results, start=1):
        candidate = candidates[cid]
        jd_info = template_map.get(cid)
        template_summary = jd_info.get('summary') if jd_info else None

        reasoning = generate_reasoning(
            candidate,
            jd_template=jd_info,
            signal_profile=signal_profile,
            template_summary=template_summary
        )
        rows.append({
            'candidate_id': cid,
            'rank': rank_pos,
            'score': round(final_score, 6),
            'reasoning': reasoning
        })

    validate_output(rows)
    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["candidate_id", "rank", "score", "reasoning"])
        writer.writeheader()
        writer.writerows(rows)

    print(f"[rank] Output written: {time.time()-t0:.1f}s")
    total_time = time.time() - total_start
    print(f"\n[rank] Total time: {total_time:.1f}s")
    print("[rank] Done.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--candidates", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--artifacts", default="artifacts")
    args = parser.parse_args()
    main(args.candidates, args.out, args.artifacts)