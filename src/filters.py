# filters.py - Honeypot detection and hard business filters.
#
# V3 changes:
#   - Fictional company hard drop (current or >50% career)
#   - Mass‑produced template drop (Templates #1–15) with overrides
#   - Founding‑date honeypot (start_date before company_founding_date)
#   - Extreme LLM‑era hard caps (LangChain > 80mo, QLoRA > 80mo, etc.)
#   - All other original honeypot flags retained.

import re
import pickle
import gzip
import json
from pathlib import Path
from collections import Counter
from datetime import datetime

# ==============================================================================
# CONSTANTS (from dataset analysis and full_templates_report.txt)
# ==============================================================================

FICTIONAL_COMPANIES = {
    'pied piper', 'initech', 'wayne enterprises', 'acme corp',
    'stark industries', 'hooli', 'globex inc', 'dunder mifflin'
}

PROVEN_COMPANIES = {
    # FAANG / Global Product Tech
    'google', 'meta', 'amazon', 'microsoft', 'netflix',
    'apple', 'adobe', 'salesforce', 'linkedin', 'uber',
    
    # AI/NLP Native Companies (Cleaned base strings)
    'sarvam', 'rephrase', 'aganitha', 'niramai', 'saarthi',
    'madstreetden', 'observe', 'krutrim', 'wysa', 'haptik',
    'verloop', 'yellow', 'locobuzz', 'glance', 'genpact',
    
    # Indian Tech Product Unicorns & Market Leader Platforms
    'swiggy', 'razorpay', 'cred', 'zomato', 'flipkart', 'meesho',
    'nykaa', 'inmobi', 'policybazaar', 'ola', 'zoho', 'vedantu',
    'paytm', 'unacademy', 'pharmeasy', 'upgrad', 'freshworks',
    'phonepe', 'dream11', 'byju', 'byjus'
}
HIGH_SIGNAL_SKILLS = {
    # Core Search Paradigms & Evaluation (High Priority)
    'rag', 'semantic search', 'vector search', 'hybrid search',
    'learning to rank', 'recommendation systems', 'ranking systems',
    'information retrieval', 'information retrieval systems',
    'content matching', 'search & discovery',
    
    # Vector Indexing & Databases
    'faiss', 'pinecone', 'weaviate', 'qdrant', 'milvus', 'pgvector',
    
    # Traditional & Keyword Inverted Indices
    'bm25', 'elasticsearch', 'opensearch',
    
    # Machine Learning Core & Deep Learning Foundational Frameworks
    'python', 'pytorch', 'tensorflow', 'scikit-learn', 'machine learning', 'deep learning',
    
    # Neural Representations, Encoders, & Fine-Tuning
    'sentence transformers', 'embeddings', 'vector representations', 'text encoders',
    'hugging face transformers', 'llms', 'prompt engineering', 'fine-tuning llms',
    'peft', 'lora', 'qlora', 'haystack', 'model adaptation',
    
    # System Level Architecture Signals (The rarest entries in the skill file)
    'search backend', 'search infrastructure', 'indexing algorithms', 'workflow orchestration'
}

# Extreme hard caps for LLM-era skills – anything above these is physically
# impossible (e.g., LangChain released Oct 2022, max ~45mo as of 2026).
LLM_EXTREME_CAPS = {
    # Mathematically bounded by open-source release calendars (As of June 2026)
    'qlora': 37,                  # May 2023
    'peft': 42,                   # Dec 2022
    'llamaindex': 43,             # Nov 2022
    'langchain': 44,              # Oct 2022
    'rag': 44,                    # Post-ChatGPT production framework era
    'qdrant': 56,                 # Oct 2021
    'lora': 60,                   # June 2021
    'opensearch': 62,             # April 2021
    'pinecone': 64,               # Early 2021
    'pgvector': 66,               # Early 2021
    
    # Industry Adoption & Modern Generative Adaptation Bounding Caps
    'prompt engineering': 48,
    'fine-tuning llms': 48,
    'llms': 48,
    'weaviate': 72,
    'haystack': 78,
    'sentence transformers': 80,
    'milvus': 80,
    'hugging face transformers': 90,
    
    # Deep Learning Systems Bounding Caps
    'faiss': 111,                 # March 2017
    'pytorch': 115,               # Sept 2016
    'tensorflow': 127,            # Nov 2015
    
    # Legacy / Conceptual Upper Limits (Caps at 9 years to match the JD's 5-9 year max range)
    'scikit-learn': 108,
    'elasticsearch': 108,
    'bm25': 108,
    'embeddings': 108,
    'semantic search': 108,
    'vector search': 108,
    'information retrieval': 108,
    'information retrieval systems': 108,
    'search backend': 108,
    'ranking systems': 108,
    'learning to rank': 108,
    'nlp': 108,
    'machine learning': 108,
    'deep learning': 108,
    'data science': 108
}


# Exact founding dates for young AI-native startups (year, month)
FOUNDING_YEARS_MONTHS = {
    'sarvam ai': (2023, 7),
    'krutrim': (2023, 12),
    'rephrase.ai': (2019, 5),
    'aganitha': (2017, 9),
    'niramai': (2016, 6),
    'saarthi.ai': (2017, 10),
    'mad street den': (2013, 9),
    'observe.ai': (2017, 9),
    'wysa': (2015, 12),
    'haptik': (2013, 8),
    'verloop.io': (2015, 6),
    'yellow.ai': (2016, 9),
    'locobuzz': (2015, 4),
    'glance': (2019, 9),
}


# ==============================================================================
# LOAD TEMPLATE FINGERPRINTS (from pre‑computed jd_templates_enhanced.pkl)
# ==============================================================================

def load_drop_fingerprints(artifacts_dir: str = "artifacts") -> set:
    """
    Load jd_templates_enhanced.pkl and return fingerprints with frequency >= 4100
    (Templates #1–15 from full_templates_report.txt).
    """
    pkl_path = Path(artifacts_dir) / "jd_templates_enhanced.pkl"
    if not pkl_path.exists():
        print("[filters] WARNING: jd_templates_enhanced.pkl not found. No template-based dropping.")
        return set()

    with open(pkl_path, "rb") as f:
        data = pickle.load(f)

    template_counter = data.get('template_counter', Counter())
    drop_fingerprints = {
        fp for fp, count in template_counter.items()
        if count >= 4100
    }
    print(f"[filters] Loaded {len(drop_fingerprints)} drop‑eligible fingerprints.")
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
# DATE HELPERS
# ==============================================================================

def parse_date(date_str: str):
    if not date_str:
        return None
    try:
        return datetime.strptime(date_str, "%Y-%m-%d")
    except ValueError:
        return None


# ==============================================================================
# HONEYPOT DETECTION (returns True if candidate is definitely fake)
# ==============================================================================

def is_definitely_fake(candidate: dict) -> bool:
    """
    Checks for absolute, undeniable impossibilities.
    Returns True immediately on any violation.
    """
    career = candidate.get('career_history', [])
    skills = candidate.get('skills', [])

    # --- 1. Extreme LLM-era skill duration ---
    for skill in skills:
        name = skill.get('name', '').lower()
        if name in LLM_EXTREME_CAPS:
            dur = skill.get('duration_months', 0)
            if dur > LLM_EXTREME_CAPS[name]:
                print(f"[filters] LLM cap violation: {name} = {dur}mo")
                return True

    # --- 2. Founding-date violation (started before company existed) ---
    for job in career:
        company = job.get('company', '').lower()
        if company not in FOUNDING_YEARS_MONTHS:
            continue
        start = parse_date(job.get('start_date'))
        if start is None:
            continue
        year, month = FOUNDING_YEARS_MONTHS[company]
        founding = datetime(year, month, 1)
        if start < founding:
            print(f"[filters] Founding violation: {company} started {start} before {founding}")
            return True

    return False


# ==============================================================================
# HONEYPOT FLAGS (suspicious but not absolute)
# ==============================================================================

def honeypot_flags(candidate: dict) -> int:
    """
    Returns a count of suspicious flags. If > 0, candidate is dropped.
    """
    flags = 0
    career = candidate.get('career_history', [])
    profile = candidate.get('profile', {})
    signals = candidate.get('redrob_signals', {})
    skills = candidate.get('skills', [])

    # Rule 1: expert/advanced skill with 0 months
    zero_advanced = sum(
        1 for s in skills
        if s.get('proficiency') in ('expert', 'advanced')
        and s.get('duration_months', 0) == 0
    )
    if zero_advanced >= 2:
        flags += 1
    if zero_advanced >= 4:
        flags += 1

    # Rule 2: total career months > YoE + 18
    yeo_months = profile.get('years_of_experience', 0) * 12
    total_career = sum(j.get('duration_months', 0) for j in career)
    if total_career > yeo_months + 18:
        flags += 1

    # Rule 3: assessment contradiction (expert but score < 25)
    assessments = signals.get('skill_assessment_scores', {})
    skill_prof = {s.get('name', '').lower(): s.get('proficiency') for s in skills}
    contradictions = 0
    for skill_name, score in assessments.items():
        if skill_prof.get(skill_name.lower()) == 'expert' and score < 25:
            contradictions += 1
    if contradictions >= 1:
        flags += 1

    # Rule 4: single job tenure > YoE + 6
    for job in career:
        if job.get('duration_months', 0) > yeo_months + 6:
            flags += 1
            break

    # Rule 5: perfect completeness + zero engagement
    completeness = signals.get('profile_completeness_score', 0)
    if (completeness >= 85 and
        signals.get('profile_views_received_30d', 0) == 0 and
        signals.get('saved_by_recruiters_30d', 0) == 0 and
        signals.get('applications_submitted_30d', 0) == 0):
        flags += 2

    return flags


# ==============================================================================
# MAIN HARD FILTER
# ==============================================================================

def passes_hard_filters(candidate: dict) -> bool:
    profile = candidate.get('profile', {})
    signals = candidate.get('redrob_signals', {})
    career = candidate.get('career_history', [])
    skills = candidate.get('skills', [])

    # --- 1. Minimum experience ---
    if profile.get('years_of_experience', 0) < 3.0:
        return False

    # --- 2. Location & relocation ---
    if profile.get('country') != 'India' and not signals.get('willing_to_relocate', False):
        return False

    # --- 3. Fictional companies ---
    if profile.get('current_company', '').lower() in FICTIONAL_COMPANIES:
        return False
    for job in career:
        if job.get('company', '').lower() in FICTIONAL_COMPANIES:
            return False

    # --- 4. Template drop (Templates #1–15) with overrides ---
    if DROP_TEMPLATE_FINGERPRINTS and career:
        current_job = career[0]
        desc = current_job.get('description', '')
        if desc:
            fingerprint = get_jd_fingerprint(desc)
            if fingerprint in DROP_TEMPLATE_FINGERPRINTS:
                # Override 1: Verified Deep Skills (≥2, assessment > 50)
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
                    return True   # rescued

                # Override 2: Proven past company
                for job in career:
                    if any(p in job.get('company', '').lower() for p in PROVEN_COMPANIES):
                        return True   # rescued

                # No override applies → drop
                return False

    # --- All checks passed ---
    return True


# ==============================================================================
# HELPER FUNCTIONS FOR RESCUES
# ==============================================================================

def _has_verified_skills(candidate: dict) -> bool:
    """Return True if candidate has >= 2 verified deep skills (advanced/expert, >=12mo, score >50)."""
    skills = candidate.get('skills', [])
    signals = candidate.get('redrob_signals', {})
    verified_count = 0
    for skill in skills:
        name = skill.get('name', '').lower()
        if name in HIGH_SIGNAL_SKILLS:
            if skill.get('proficiency') in ('advanced', 'expert'):
                if skill.get('duration_months', 0) >= 12:
                    score = signals.get('skill_assessment_scores', {}).get(name, -1)
                    if score > 50:
                        verified_count += 1
    return verified_count >= 2
# ==============================================================================
# FILTER WRAPPER
# ==============================================================================

def filter_candidates(candidates: list) -> list:
    passed = []
    fake_count = 0
    flag_count = 0
    hard_filter_count = 0

    for c in candidates:
        # Absolute impossibilities
        if is_definitely_fake(c):
            fake_count += 1
            continue

       #-- Suspicious flags (Honeypot) ---
        flags = honeypot_flags(c)
        keep_candidate = True
        
        if flags > 0:
            # Rule: 
            # - 1 flag: Give benefit of doubt (keep)
            # - 2 flags: Keep ONLY if rescued (verified skills OR proven past)
            # - 3+ flags: Drop immediately (even if rescued)
            if flags >= 3:
                keep_candidate = False
            elif flags == 2:
                # Check for rescue
                if not (_has_verified_skills(c) ):
                    keep_candidate = False
            # flags == 1: keep_candidate remains True (no rescue needed)
        
        if not keep_candidate:
            flag_count += 1
            continue

        # Hard filters (including template drop with overrides)
        if not passes_hard_filters(c):
            hard_filter_count += 1
            continue

        passed.append(c)

    total = len(candidates)
    print(f"[filter] Total candidates:     {total:,}")
    print(f"[filter] Absolute fakes:       {fake_count:,}")
    print(f"[filter] Honeypot flags:       {flag_count:,}")
    print(f"[filter] Hard filters removed: {hard_filter_count:,}")
    print(f"[filter] Remaining:            {len(passed):,} ({len(passed)/total*100:.1f}%)")
    return passed


# ==============================================================================
# STANDALONE TESTING MODE
# ==============================================================================

def load_candidates(path: str) -> list:
    candidates = []
    opener = gzip.open if path.endswith(".gz") else open
    with opener(path, "rt", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                try:
                    candidates.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
    print(f"[test] Loaded {len(candidates):,} candidates from {path}")
    return candidates


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--candidates", required=True)
    parser.add_argument("--artifacts", default="artifacts")
    args = parser.parse_args()

    DROP_TEMPLATE_FINGERPRINTS = load_drop_fingerprints(args.artifacts)
    candidates = load_candidates(args.candidates)
    filtered = filter_candidates(candidates)
    print(f"\n[test] Final count: {len(filtered):,} candidates pass all filters.")