# signals.py - Structured signal scoring, calibrated to real 100K dataset distributions.
#
# V2 changes:
#   - compute_final_score() now returns (final_score, signal_profile) where
#     signal_profile is a dict of all component multipliers.
#   - This allows rank.py to pass the breakdown to reasoning.py for transparent
#     narrative generation.
#   - Added jd_template_multiplier() that applies 1.30x for golden templates,
#     1.05x for ml_adj, 0.95x for data_eng, 0.70x for irrelevant.

from datetime import date, datetime

# ------------------------------------------------------------------------------
# Constants (from dataset analysis)
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
}

# Same as HIGH_SIGNAL_SKILLS in filters.py – used for skill depth scoring.
HIGH_SIGNAL_SKILLS = CORE_JD_SKILLS  # we use the same set for simplicity

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
# Parsing helper
# ------------------------------------------------------------------------------

def parse_date(date_str: str) -> date:
    return datetime.strptime(date_str, "%Y-%m-%d").date()

# ------------------------------------------------------------------------------
# Individual scoring functions
# ------------------------------------------------------------------------------

def experience_fit_score(profile: dict) -> float:
    """
    JD wants 5-9 years. Dataset mean is 7.17.
    Returns:
        1.0  : 6-8 years (peak)
        0.95 : 5-6 or 8-9
        0.85 : 4-5 or 9-12
        0.70 : 3-4
        0.60 : 9-12 (was 0.85, tightened)
        0.35 : >12 (heavy penalty)
        0.40 : <3
    """
    yoe = profile.get('years_of_experience', 0)
    if 6 <= yoe <= 8:
        return 1.0
    if 5 <= yoe < 6 or 8 < yoe <= 9:
        return 0.95
    if 4 <= yoe < 5:
        return 0.85
    if 9 < yoe <= 12:
        return 0.60
    if yoe > 12:
        return 0.35
    if 3 <= yoe < 4:
        return 0.70
    return 0.40  # <3 years

def availability_score(signals: dict) -> float:
    """
    Combines open_to_work, last activity recency, and recruiter response.
    Calibrated: only 35.3% are open → 1.25x boost.
    """
    today = date.today()
    score = 1.0

    # Open to work – rare, strong signal
    if signals.get('open_to_work_flag', False):
        score *= 1.25

    # Recency – active in last 30 days is good
    last_active = signals.get('last_active_date')
    if last_active:
        days_inactive = (today - parse_date(last_active)).days
        if days_inactive <= 30:
            score *= 1.05
        elif days_inactive > 180:
            score *= 0.92

    # Recruiter response rate – mean=0.44, p75=0.62
    rr = signals.get('recruiter_response_rate', 0.0)
    if rr >= 0.75:
        score *= 1.06
    elif rr >= 0.62:
        score *= 1.03
    elif rr < 0.20:
        score *= 0.95

    return score

def notice_period_score(signals: dict) -> float:
    """
    JD prefers <30 days. Median is 90 days.
    """
    days = signals.get('notice_period_days', 90)
    if days == 0:
        return 1.20
    elif days <= 30:
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
    """
    JD prefers Pune or Noida. India is acceptable.
    Outside India only if willing to relocate.
    """
    country = profile.get('country', '')
    location = profile.get('location', '').lower()

    if country == 'India':
        if 'pune' in location:
            return 1.10
        elif 'noida' in location or 'delhi' in location or 'ncr' in location:
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
    """
    64.6% have -1 (no GitHub) – slight penalty.
    Valid scores: mean=29, p75=42, p90=51.7.
    """
    g = signals.get('github_activity_score', -1)
    if g == -1:
        return 0.92
    if g >= 52:
        return 1.20
    if g >= 42:
        return 1.12
    if g >= 29:
        return 1.06
    if g >= 14:
        return 1.00
    return 0.96

def career_quality_score(career_history: list) -> float:
    """
    Weighted by company type:
      FAANG        → 1.35x (very rare)
      AI-Native    → 1.30x
      Product (unicorn) → 1.15x
      Consulting   → 0.85x
      Other        → 1.00x
    Months are weighted proportionally.
    """
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

        # Determine type
        if any(f in company for f in FAANG_LIST):
            typ = 'faang'
        elif any(a in company for a in AI_NATIVE_LIST):
            typ = 'ai_native'
        elif any(c in company for c in CONSULTING_FIRMS):
            typ = 'consulting'
        elif (any(u in company for u in UNICORN_LIST) or
              industry in PRODUCT_INDUSTRIES):
            typ = 'product'
        else:
            typ = 'other'

        weighted_sum += weights[typ] * months

    return weighted_sum / total_months

def skill_depth_score(candidate: dict) -> float:
    """
    Count advanced/expert CORE_JD_SKILLS with ≥12 months.
    Also add assessment bonus for scores > 50 (threshold lowered from 60).
    """
    skills = candidate.get('skills', [])
    assessments = candidate.get('redrob_signals', {}).get('skill_assessment_scores', {})

    # Count deep skills
    deep_count = 0
    assessment_bonus = 1.0
    for skill in skills:
        name = skill.get('name', '').lower()
        if name in CORE_JD_SKILLS:
            if skill.get('proficiency') in ('advanced', 'expert'):
                if skill.get('duration_months', 0) >= 12:
                    deep_count += 1
            # Assessment bonus for any core skill with score > 50
            score = assessments.get(name, -1)
            if score > 50:
                assessment_bonus += 0.04  # small boost per skill

    # Depth bonus
    if deep_count >= 6:
        depth_bonus = 1.15
    elif deep_count >= 4:
        depth_bonus = 1.10
    elif deep_count >= 2:
        depth_bonus = 1.05
    elif deep_count >= 1:
        depth_bonus = 1.00
    else:
        depth_bonus = 0.85

    # Cap assessment bonus at 1.25
    assessment_bonus = min(assessment_bonus, 1.25)

    return depth_bonus * assessment_bonus

def recruiter_market_signal(signals: dict) -> float:
    """
    External validation: saved_by_recruiters_30d (mean=7.66, p75=11, p90=15)
    and profile_views_received_30d (mean=47.99, p75=68, p90=86).
    """
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

def jd_template_multiplier(jd_template: dict) -> float:
    """
    Multipliers based on template tier (from full_templates_report.txt).
      'golden'   → 1.30
      'ml_adj'   → 1.05
      'data_eng' → 0.95
      'irrelevant' → 0.70
      None       → 1.00
    """
    if jd_template is None:
        return 1.00
    tier = jd_template.get('tier', '')
    if tier == 'golden':
        return 1.30
    elif tier == 'ml_adj':
        return 1.05
    elif tier == 'data_eng':
        return 0.95
    elif tier == 'irrelevant':
        return 0.70
    else:
        return 1.00

# ------------------------------------------------------------------------------
# Main entry point
# ------------------------------------------------------------------------------

def compute_final_score(ce_score: float, candidate: dict, jd_template: dict = None) -> tuple:
    """
    Compute final score and return both the score and a signal profile dict.

    Args:
        ce_score: cross-encoder logit (sigmoid-normalized, 0-1)
        candidate: full candidate JSON
        jd_template: dict from template map (or None)

    Returns:
        (final_score, signal_profile)
        - final_score: float
        - signal_profile: dict with keys:
            'ce_score',
            'experience_fit',
            'availability',
            'notice_period',
            'location',
            'github',
            'career_quality',
            'skill_depth',
            'recruiter_market',
            'jd_template_multiplier',
            'combined_multiplier'
    """
    profile = candidate.get('profile', {})
    signals = candidate.get('redrob_signals', {})
    career = candidate.get('career_history', [])

    # Compute each component
    exp = experience_fit_score(profile)
    avail = availability_score(signals)
    notice = notice_period_score(signals)
    loc = location_score(profile, signals)
    gh = github_score(signals)
    career_q = career_quality_score(career)
    skill_d = skill_depth_score(candidate)
    recruiter = recruiter_market_signal(signals)
    jd_mult = jd_template_multiplier(jd_template)

    # Combine all structured multipliers
    structured_multiplier = (
        exp * avail * notice * loc * gh * career_q * skill_d * recruiter * jd_mult
    )

    final_score = ce_score * structured_multiplier

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
        'combined_multiplier': structured_multiplier,
    }

    return final_score, profile_dict