# filters.py - Honeypot detection and hard business filters.

import re
import pickle
import gzip
import json
from pathlib import Path
from collections import Counter
from datetime import datetime, date

# ==============================================================================
# CONSTANTS (from dataset analysis and full_templates_report.txt)
# ==============================================================================

FICTIONAL_COMPANIES = {
    'pied piper', 'initech', 'wayne enterprises', 'acme corp',
    'stark industries', 'hooli', 'globex inc', 'dunder mifflin'
}

PROVEN_COMPANIES = {
    # FAANG / big tech
    'google', 'meta', 'amazon', 'microsoft', 'netflix',
    'apple', 'adobe', 'salesforce', 'linkedin', 'uber',
    # AI-native companies
    'sarvam', 'rephrase', 'aganitha', 'niramai', 'saarthi',
    'mad street den', 'observe.ai', 'krutrim', 'wysa', 'haptik',
    'verloop', 'yellow.ai', 'locobuzz', 'glance',
    # Indian product unicorns
    'swiggy', 'razorpay', 'cred', 'zomato', 'flipkart', 'meesho',
    'nykaa', 'inmobi', 'policybazaar', 'ola', 'zoho', 'vedantu',
    'paytm', 'unacademy', 'pharmeasy', 'upgrad', 'freshworks',
    'phonepe', 'dream11', 'byju'
}

HIGH_SIGNAL_SKILLS = {
    'faiss', 'pinecone', 'weaviate', 'qdrant', 'milvus',
    'bm25', 'elasticsearch', 'opensearch',
    'learning to rank', 'recommendation systems',
    'rag', 'semantic search', 'vector search', 'hybrid search',
    'sentence transformers', 'embeddings', 'information retrieval'
}


# ==============================================================================
# LOAD TEMPLATE FINGERPRINTS (from pre‑computed jd_templates.pkl)
# ==============================================================================

def load_drop_fingerprints(artifacts_dir: str = "artifacts") -> set:
    """
    Load jd_templates.pkl and return a set of fingerprints whose frequency
    is >= 4100 (i.e., Templates #1–15 from full_templates_report.txt).
    If the pickle does not exist, return an empty set (no template drop).
    """
    pkl_path = Path(artifacts_dir) / "jd_templates.pkl"
    if not pkl_path.exists():
        print("[filters] WARNING: jd_templates.pkl not found. No template-based dropping will occur.")
        return set()

    with open(pkl_path, "rb") as f:
        data = pickle.load(f)

    template_counter = data.get('template_counter', Counter())
    drop_fingerprints = {
        fp for fp, count in template_counter.items()
        if count >= 4100   # captures all 15 mass‑produced irrelevant templates
    }
    print(f"[filters] Loaded {len(drop_fingerprints)} drop‑eligible fingerprints "
          f"(frequency >= 4100) from {pkl_path}")
    return drop_fingerprints

DROP_TEMPLATE_FINGERPRINTS = load_drop_fingerprints()


# ==============================================================================
# TEXT NORMALIZATION (must match analyze_jd_templates.py)
# ==============================================================================

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


# ==============================================================================
# HONEYPOT DETECTION (returns number of flags)
# ==============================================================================

def is_honeypot(candidate: dict) -> int:
    flags = 0
    career = candidate.get('career_history', [])
    profile = candidate.get('profile', {})
    signals = candidate.get('redrob_signals', {})
    skills = candidate.get('skills', [])

    # --- Rule 1: expert/advanced skill with 0 months duration ---
    zero_duration_advanced = 0
    for skill in skills:
        if skill.get('proficiency') in ('expert', 'advanced'):
            if skill.get('duration_months', 1) == 0:
                zero_duration_advanced += 1
    if zero_duration_advanced >= 2:
        flags += 1
    if zero_duration_advanced >= 4:
        flags += 1   # extra flag for egregious cases

    # --- Rule 2: career timeline impossible vs years_of_experience ---
    yeo_months = profile.get('years_of_experience', 0) * 12
    total_career_months = sum(j.get('duration_months', 0) for j in career)
    if total_career_months > yeo_months + 18:
        flags += 1

    # --- Rule 3: assessment contradicts expert proficiency ---
    assessments = signals.get('skill_assessment_scores', {})
    skill_proficiency_map = {
        s.get('name', '').lower(): s.get('proficiency', '')
        for s in skills
    }
    contradictions = 0
    for skill_name, score in assessments.items():
        claimed = skill_proficiency_map.get(skill_name.lower(), '')
        if claimed == 'expert' and score < 25:
            contradictions += 1
    if contradictions >= 1:          # tightened from 2 to 1
        flags += 1

    # --- Rule 4: single job tenure > YoE + 6 months ---
    for job in career:
        if job.get('duration_months', 0) > yeo_months + 6:
            flags += 1
            break

    # --- Rule 5: perfect completeness + zero behavioral signals ---
    completeness = signals.get('profile_completeness_score', 0)
    views = signals.get('profile_views_received_30d', -1)
    saves = signals.get('saved_by_recruiters_30d', -1)
    apps = signals.get('applications_submitted_30d', -1)
    if completeness >= 85 and views == 0 and saves == 0 and apps == 0:
        flags += 2   # heavy penalty for a perfectly filled, dead profile

    return flags


# ==============================================================================
# MAIN HARD FILTER
# ==============================================================================

def passes_hard_filters(candidate: dict) -> bool:
    profile = candidate.get('profile', {})
    signals = candidate.get('redrob_signals', {})
    career = candidate.get('career_history', [])
    skills = candidate.get('skills', [])

    # --- Hard Filter A: Minimum experience ---
    yoe = profile.get('years_of_experience', 0)
    if yoe < 3.0:
        return False

    # --- Hard Filter B: Location & relocation (JD explicit) ---
    country = profile.get('country', '')
    willing_to_relocate = signals.get('willing_to_relocate', False)
    if country != 'India' and not willing_to_relocate:
        return False

    # --- Hard Filter C: Fictional companies ---
    current_company = profile.get('current_company', '').lower()
    if current_company in FICTIONAL_COMPANIES:
        return False

    total_months = sum(j.get('duration_months', 0) for j in career)
    if total_months > 0:
        fictional_months = sum(
            j.get('duration_months', 0) for j in career
            if j.get('company', '').lower() in FICTIONAL_COMPANIES
        )
        if fictional_months / total_months > 0.5:
            return False

    # --- Hard Filter D: Template drop (Templates #1–15) ---
    # Only apply if we have drop fingerprints loaded
    if DROP_TEMPLATE_FINGERPRINTS and career:
        current_job = career[0]  # most recent job
        desc = current_job.get('description', '')
        if desc:
            fingerprint = get_jd_fingerprint(desc)
            if fingerprint in DROP_TEMPLATE_FINGERPRINTS:
                # ---- Override 1: Verified Deep Skills ----
                verified_count = 0
                for skill in skills:
                    name = skill.get('name', '').lower()
                    if name in HIGH_SIGNAL_SKILLS:
                        if skill.get('proficiency') in ('advanced', 'expert'):
                            if skill.get('duration_months', 0) >= 12:
                                score = signals.get('skill_assessment_scores', {}).get(name, -1)
                                if score > 50:
                                    verified_count += 1
                if verified_count >= 2:
                    return True   # rescued by deep verified skills

                # ---- Override 2: Proven past companies ----
                for job in career:
                    company = job.get('company', '').lower()
                    if any(p in company for p in PROVEN_COMPANIES):
                        return True   # rescued by past top‑tier company

                # ---- No override applies: drop ----
                return False

    # ---- All checks passed ----
    return True


# ==============================================================================
# FILTER WRAPPER (for use by precompute.py)
# ==============================================================================

def filter_candidates(candidates: list) -> list:
    """
    Apply honeypot detection and hard filters to a list of candidates.
    Returns the list of candidates that survive.
    Also prints a summary of removals.
    """
    passed = []
    honeypot_count = 0
    hard_filter_count = 0
    template_drop_count = 0

    for c in candidates:
        flags = is_honeypot(c)
        if flags > 0:
            honeypot_count += 1
            continue

        # Hard filters (including template drop)
        hard_ok = passes_hard_filters(c)
        if not hard_ok:
            hard_filter_count += 1
            continue

        passed.append(c)

    total = len(candidates)
    print(f"[filter] Total candidates:    {total:,}")
    print(f"[filter] Honeypot removed:    {honeypot_count:,}")
    print(f"[filter] Hard filters removed:{hard_filter_count:,}")
    print(f"[filter] Remaining:           {len(passed):,} ({len(passed)/total*100:.1f}%)")
    return passed


# ==============================================================================
# STANDALONE TESTING MODE
# ==============================================================================

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
    print(f"[test] Loaded {len(candidates):,} candidates from {path}")
    return candidates


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Test filters.py on a candidate file.")
    parser.add_argument("--candidates", required=True, help="Path to candidates.jsonl or .jsonl.gz")
    parser.add_argument("--artifacts", default="artifacts", help="Directory containing jd_templates.pkl")
    args = parser.parse_args()

    # Reload template fingerprints with the user‑specified artifacts dir
    DROP_TEMPLATE_FINGERPRINTS = load_drop_fingerprints(args.artifacts)

    candidates = load_candidates(args.candidates)
    filtered = filter_candidates(candidates)
    print(f"\n[test] Final count: {len(filtered):,} candidates pass all filters.")