# signals.py - Structured signal scoring, calibrated to real 100K dataset distributions.
#
# Set DEBUG = True to enable diagnostic prints for the first few candidates.

DEBUG = True   # Set to False before final submission

from datetime import date, datetime
from pathlib import Path
import pickle

# ------------------------------------------------------------------------------
# Constants
# ------------------------------------------------------------------------------

CONSULTING_FIRMS = {
    'infosys', 'wipro', 'tcs', 'capgemini', 'hcl', 'mindtree',
    'accenture', 'cognizant', 'tech mahindra', 'mphasis', 'hexaware',
    'l&t infotech', 'ltimindtree', 'genpact'
}

PRODUCT_INDUSTRIES = {
    'software', 'ai/ml', 'fintech', 'e-commerce', 'saas', 'food delivery',
    'transportation', 'edtech', 'healthtech', 'healthtech ai',
    'conversational ai', 'ai services', 'voice ai', 'adtech',
    'insurance tech', 'gaming', 'internet'
}

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

HIGH_SIGNAL_SKILLS = CORE_JD_SKILLS

FAANG_LIST = {
    'google', 'meta', 'amazon', 'microsoft', 'netflix',
    'apple', 'adobe', 'salesforce', 'linkedin', 'uber'
}

AI_NATIVE_LIST = {
    'sarvam', 'rephrase', 'aganitha', 'niramai', 'saarthi',
    'mad street den', 'observe.ai', 'krutrim', 'wysa', 'haptik',
    'verloop', 'yellow.ai', 'locobuzz', 'glance'
}

UNICORN_LIST = {
    'swiggy', 'razorpay', 'cred', 'zomato', 'flipkart', 'meesho',
    'nykaa', 'inmobi', 'policybazaar', 'ola', 'zoho', 'vedantu',
    'paytm', 'unacademy', 'pharmeasy', 'upgrad', 'freshworks',
    'phonepe', 'dream11', 'byju'
}

# ------------------------------------------------------------------------------
# LLM-era Hard Caps
# ------------------------------------------------------------------------------

LLM_HARD_CAPS = {
    'qlora': 37,
    'peft': 42,
    'llamaindex': 43,
    'langchain': 44,
    'rag': 44,
    'qdrant': 56,
    'lora': 60,
    'opensearch': 62,
    'pinecone': 64,
    'pgvector': 66,
    'prompt engineering': 48,
    'fine-tuning llms': 48,
    'llms': 48,
    'weaviate': 72,
    'haystack': 78,
    'sentence transformers': 80,
    'milvus': 80,
    'hugging face transformers': 90,
    'faiss': 111,
    'pytorch': 115,
    'tensorflow': 127,
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

# ------------------------------------------------------------------------------
# Load assessment weights
# ------------------------------------------------------------------------------

def load_assessment_weights(artifacts_dir: str = "artifacts") -> dict:
    pkl_path = Path(artifacts_dir) / "skill_duration_percentiles.pkl"
    if not pkl_path.exists():
        print("[signals] WARNING: skill_duration_percentiles.pkl not found. Using flat +0.04 bonus.")
        return {}
    with open(pkl_path, "rb") as f:
        data = pickle.load(f)
    return {skill: stats['count'] for skill, stats in data.items()}

ASSESSMENT_WEIGHTS = load_assessment_weights()

# ------------------------------------------------------------------------------
# Parsing helper
# ------------------------------------------------------------------------------

def parse_date(date_str: str) -> date:
    return datetime.strptime(date_str, "%Y-%m-%d").date()

# ------------------------------------------------------------------------------
# Scoring functions
# ------------------------------------------------------------------------------

def experience_fit_score(profile: dict) -> float:
    yoe = profile.get('years_of_experience', 0)
    if 6 <= yoe <= 8:
        return 1.0
    if 5 <= yoe < 6 or 8 < yoe <= 9:
        return 0.95
    if 4 <= yoe < 5:
        return 0.80
    if 9 < yoe <= 10:
        return 0.85
    if 10 <= yoe < 11:
        return 0.75
    if 11 <= yoe < 12:
        return 0.70
    if yoe > 12:
        return 0.50
    if 3 <= yoe < 4:
        return 0.50
    return 0.40

def availability_score(signals: dict) -> float:
    FROZEN_DATE = date(2026, 5, 27)
    score = 1.0
    if signals.get('open_to_work_flag', False):
        score *= 1.25
    last_active = signals.get('last_active_date')
    if last_active:
        days_inactive = (FROZEN_DATE - parse_date(last_active)).days
        if days_inactive <= 7:
            score *= 1.10
        elif days_inactive <= 30:
            score *= 1.05
        elif days_inactive <= 60:
            score *= 1.00
        elif days_inactive <= 120:
            score *= 0.95
        elif days_inactive <= 180:
            score *= 0.90
        else:
            score *= 0.85
    rr = signals.get('recruiter_response_rate', 0.0)
    if rr >= 0.75:
        score *= 1.06
    elif rr >= 0.62:
        score *= 1.03
    elif rr < 0.20:
        score *= 0.95
    return score

def notice_period_score(signals: dict) -> float:
    days = signals.get('notice_period_days', 90)
    if days <= 30:
        return 1.15
    elif days <= 60:
        return 1.05
    elif days <= 90:
        return 1.00
    elif days <= 120:
        return 0.85
    else:
        return 0.70

def location_score(profile: dict, signals: dict) -> float:
    country = profile.get('country', '')
    location = profile.get('location', '').lower()
    if country == 'India':
        if 'pune' in location or 'noida' in location:
            return 1.10
        elif 'delhi' in location or 'ncr' in location:
            return 1.08
        elif 'bangalore' in location or 'bengaluru' in location:
            return 1.00
        elif 'hyderabad' in location or 'mumbai' in location or 'chennai' in location:
            return 0.97
        else:
            return 0.93
    else:
        if signals.get('willing_to_relocate', False):
            return 0.75
        else:
            return 0.45

def github_score(signals: dict) -> float:
    g = signals.get('github_activity_score', -1)
    if g == -1:
        return 1.00
    if g >= 52:
        return 1.15
    if g >= 42:
        return 1.12
    if g >= 29:
        return 1.06
    if g >= 14:
        return 1.00
    return 0.96

def career_quality_score(career_history: list) -> float:
    if not career_history:
        return 0.70
    total_months = sum(j.get('duration_months', 0) for j in career_history)
    if total_months == 0:
        return 0.70
    weights = {
        'faang': 1.35,
        'ai_native': 1.30,
        'product': 1.15,
        'consulting': 0.85,
        'other': 1.00
    }
    weighted_sum = 0.0
    for job in career_history:
        company = job.get('company', '').lower()
        industry = job.get('industry', '').lower()
        months = job.get('duration_months', 0)
        if any(f in company for f in FAANG_LIST):
            typ = 'faang'
        elif any(a in company for a in AI_NATIVE_LIST):
            typ = 'ai_native'
        elif any(c in company for c in CONSULTING_FIRMS):
            typ = 'consulting'
        elif (any(u in company for u in UNICORN_LIST) or industry in PRODUCT_INDUSTRIES):
            typ = 'product'
        else:
            typ = 'other'
        weighted_sum += weights[typ] * months
    return weighted_sum / total_months

def skill_depth_score(candidate: dict, assessment_weights: dict = None) -> float:
    skills = candidate.get('skills', [])
    assessments = candidate.get('redrob_signals', {}).get('skill_assessment_scores', {})
    deep_count = 0
    assessment_bonus = 1.0
    for skill in skills:
        name = skill.get('name', '').lower()
        if name in CORE_JD_SKILLS:
            if skill.get('proficiency') in ('advanced', 'expert'):
                if skill.get('duration_months', 0) >= 12:
                    deep_count += 1
            score = assessments.get(name, -1)
            if score > 50:
                if assessment_weights:
                    n = assessment_weights.get(name, 0)
                    bonus = 0.06 if n < 350 else (0.04 if n < 700 else 0.02)
                else:
                    bonus = 0.04
                assessment_bonus += bonus
    depth_bonus = 1.15 if deep_count >= 6 else (1.10 if deep_count >= 4 else (1.05 if deep_count >= 2 else (1.00 if deep_count >= 1 else 0.85)))
    assessment_bonus = min(assessment_bonus, 1.25)
    return depth_bonus * assessment_bonus

def recruiter_market_signal(signals: dict) -> float:
    saved = signals.get('saved_by_recruiters_30d', 0)
    views = signals.get('profile_views_received_30d', 0)
    score = 1.0
    if saved >= 20:
        score *= 1.10
    elif saved >= 15:
        score *= 1.06
    elif saved >= 11:
        score *= 1.04
    if views >= 86:
        score *= 1.05
    elif views >= 68:
        score *= 1.03
    return score

# ------------------------------------------------------------------------------
# Critical: jd_template_multiplier (with debug prints)
# ------------------------------------------------------------------------------

_DEBUG_COUNT = 0

def jd_template_multiplier(jd_template: dict) -> float:
    """Returns the multiplier based on the template tier."""
    global _DEBUG_COUNT
    if jd_template is None:
        if DEBUG and _DEBUG_COUNT < 5:
            print(f"[DEBUG] jd_template_multiplier: jd_template is None → returning 1.00")
            _DEBUG_COUNT += 1
        return 1.00

    tier = jd_template.get('tier', '')
    if tier == 'golden':
        multiplier = 1.50
    elif tier == 'ml_adj':
        multiplier = 1.00
    elif tier == 'data_eng':
        multiplier = 0.95
    elif tier == 'irrelevant':
        multiplier = 0.5
    else:
        multiplier = 1

    # if DEBUG and _DEBUG_COUNT < 5:
    #     print(f"[DEBUG] jd_template_multiplier: tier='{tier}' → multiplier={multiplier}")
    #     _DEBUG_COUNT += 1
    return multiplier

# ------------------------------------------------------------------------------
# Soft penalties
# ------------------------------------------------------------------------------

def job_hopping_penalty(career_history: list) -> float:
    if not career_history:
        return 1.00
    total_months = sum(j.get('duration_months', 0) for j in career_history)
    num_jobs = len(career_history)
    avg_tenure = total_months / num_jobs if num_jobs > 0 else 0
    if avg_tenure < 18:
        return 0.85
    return 1.00

def career_gap_penalty(profile: dict, career_history: list) -> float:
    yoe_months = profile.get('years_of_experience', 0) * 12
    total_months = sum(j.get('duration_months', 0) for j in career_history)
    gap = yoe_months - total_months
    if gap > 24:
        return 0.90
    return 1.00

def llm_era_penalty(skills: list) -> float:
    penalty = 1.0
    for skill in skills:
        name = skill.get('name', '').lower()
        if name in LLM_HARD_CAPS:
            dur = skill.get('duration_months', 0)
            cap = LLM_HARD_CAPS[name]
            if dur > cap:
                penalty *= 0.85
    return penalty

def domain_mismatch_penalty(summary: str) -> float:
    if not summary:
        print("no summary found for domain_mismatch")
        return 1.0
    mismatch_phrases = [
        "exclusively building computer vision",
        "zero professional NLP",
        "zero information retrieval",
        "entirely locked into Computer Vision",
    ]
    lower = summary.lower()
    for phrase in mismatch_phrases:
        if phrase in lower:
            return 0.70
    return 1.0
# ------------------------------------------------------------------------------
# Main entry point
# ------------------------------------------------------------------------------

# _DEBUG_SCORE_COUNT = 0

def compute_final_score(ce_score: float, candidate: dict, jd_template: dict = None,template_summary: str = None) -> tuple:
    # if candidate.get('candidate_id') in ['CAND_0082086']:  # the ID from your top 10
    #     print(f"[FINAL DEBUG] candidate: {candidate['candidate_id']}, jd_template is None? {jd_template is None}")
    #     if jd_template:
    #         print(f"[FINAL DEBUG] tier = {jd_template.get('tier')}")
    # global _DEBUG_SCORE_COUNT
    profile = candidate.get('profile', {})
    signals = candidate.get('redrob_signals', {})
    career = candidate.get('career_history', [])
    skills = candidate.get('skills', [])

    exp = experience_fit_score(profile)
    avail = availability_score(signals)
    notice = notice_period_score(signals)
    loc = location_score(profile, signals)
    gh = github_score(signals)
    career_q = career_quality_score(career)
    skill_d = skill_depth_score(candidate, assessment_weights=ASSESSMENT_WEIGHTS)
    recruiter = recruiter_market_signal(signals)
    jd_mult = jd_template_multiplier(jd_template)

    job_hop = job_hopping_penalty(career)
    gap = career_gap_penalty(profile, career)
    llm = llm_era_penalty(skills)
    domain_penalty = domain_mismatch_penalty(template_summary)
    structured_multiplier = (
        exp * avail * notice * loc * gh * career_q * skill_d *
        recruiter * jd_mult * job_hop * gap * llm * domain_penalty
    )

    # structured_multiplier = (
    #     exp * avail * notice * loc * gh * career_q * skill_d *
    #     recruiter * jd_mult * job_hop * gap * llm
    # )

    final_score = ce_score * structured_multiplier

    # if DEBUG and _DEBUG_SCORE_COUNT < 5:
    #     print(f"[DEBUG] compute_final_score for candidate {candidate.get('candidate_id', '?')}:")
    #     print(f"  jd_mult = {jd_mult:.3f}")
    #     print(f"  structured_multiplier = {structured_multiplier:.4f}")
    #     print(f"  final_score = {final_score:.4f}")
    #     _DEBUG_SCORE_COUNT += 1

    profile_dict = {
        'ce_score': ce_score,
        'experience_fit': exp,
        'availability': avail,
        'notice_period': notice,
        'location': loc,
        'github': gh,
        'career_quality': career_q,
        'skill_depth': skill_d,
        'recruiter_market': recruiter,
        'jd_template_multiplier': jd_mult,
        'job_hopping_penalty': job_hop,
        'career_gap_penalty': gap,
        'llm_era_penalty': llm,
        'domain_mismatch_penalty': domain_penalty,
        'combined_multiplier': structured_multiplier,
    }
    return final_score, profile_dict