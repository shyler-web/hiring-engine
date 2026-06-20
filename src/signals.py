"""
signals.py — Structured signal scoring calibrated to real 100K dataset distributions.

All thresholds are derived from actual percentile stats, not guesses:

years_of_experience:  mean=7.17, p25=3.9, p50=6.8, p75=9.9
notice_period_days:   mean=87, median=90, p25=60, p75=120, p90=150
                      Values cluster at 0, 30, 60, 90, 120, 150 (discrete)
recruiter_response:   mean=0.44, p25=0.25, p50=0.44, p75=0.62, p90=0.73
github_activity:      64.6% are -1 (no github). Valid range: 0-96.9, mean=29, p75=42
last_active:          date range 2025-09 → 2026-05, most active in 2026
open_to_work:         only 35,339 / 100,000 = 35.3% are open
offer_acceptance:     59.6% are -1. Valid mean=0.47
interview_completion: mean=0.62, p25=0.48, p75=0.76
saved_by_recruiters:  mean=7.66, p75=11, p90=15, p99=28

Company type scoring based on actual company frequency:
- FAANG (Google, Meta, Amazon etc): 7-14 people total — extremely rare, gold signal
- AI-native (Sarvam AI, Mad Street Den etc): 25-79 people — rare, strong signal
- Indian product unicorns (Swiggy, Razorpay, Zomato etc): 2800-3000 — good signal
- Consulting (Infosys, Wipro, TCS etc): 23,000+ each — negative signal
- Neutral companies: mid-range

JD-specific thresholds:
- YoE sweet spot: 5-9 years (JD explicit)
- Notice period: <30 preferred, 30-60 acceptable, 60-90 tolerable, 90+ penalty
- Salary range hint from data: mean salary_max = 19.84 LPA, p95 = 33.8 LPA
"""

from datetime import date, datetime

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

# Tier-1 AI/search skills from our JD analysis
CORE_JD_SKILLS = {
    'faiss', 'pinecone', 'weaviate', 'qdrant', 'milvus', 'elasticsearch',
    'opensearch', 'pgvector', 'vector search', 'semantic search',
    'information retrieval', 'hybrid search', 'bm25',
    'sentence transformers', 'sentence-transformers', 'embeddings',
    'hugging face transformers', 'learning to rank', 'recommendation systems',
    'lora', 'qlora', 'peft', 'fine-tuning llms', 'haystack', 'llamaindex',
    'pytorch', 'tensorflow', 'scikit-learn', 'nlp', 'machine learning',
    'deep learning', 'python', 'rag',
}


def parse_date(date_str: str) -> date:
    return datetime.strptime(date_str, "%Y-%m-%d").date()


def experience_fit_score(profile: dict) -> float:
    """
    JD wants 5-9 years. Dataset mean is 7.17.
    Score based on how close to sweet spot.
    """
    yoe = profile['years_of_experience']

    if 5 <= yoe <= 9:
        # Peak zone — full score, slight bonus for center
        if 6 <= yoe <= 8:
            return 1.0
        return 0.95
    elif 4 <= yoe < 5:
        return 0.85   # slightly junior but acceptable
    elif 9 < yoe <= 12:
        return 0.85   # slightly senior, may expect more money/title
    elif 3 <= yoe < 4:
        return 0.70
    elif yoe > 12:
        return 0.75   # overqualified concern
    else:
        return 0.50   # < 3 years


def availability_score(signals: dict) -> float:
    """
    Combines open_to_work, recency of activity, and recruiter responsiveness.
    
    Calibrated to dataset:
    - Only 35.3% are open_to_work — being open is a strong differentiator
    - Last active: most candidates are active 2026-01 to 2026-05
    - recruiter_response_rate: mean=0.44, p75=0.62
    """
    today = date.today()
    score = 1.0

    # Open to work — 35.3% signal
    if signals['open_to_work_flag']:
        score *= 1.25   # strong positive — only 1/3 of pool
    else:
        score *= 0.72   # penalize but don't eliminate

    # Activity recency
    days_inactive = (today - parse_date(signals['last_active_date'])).days
    if days_inactive <= 14:
        score *= 1.15   # very recently active
    elif days_inactive <= 30:
        score *= 1.10
    elif days_inactive <= 60:
        score *= 1.0    # baseline — most candidates fall here
    elif days_inactive <= 90:
        score *= 0.90
    elif days_inactive <= 120:
        score *= 0.80
    else:
        score *= 0.60   # > 4 months inactive

    # Recruiter response rate (mean=0.44, p75=0.62)
    rr = signals['recruiter_response_rate']
    if rr >= 0.70:      # above p75
        score *= 1.12
    elif rr >= 0.44:    # above mean
        score *= 1.05
    elif rr >= 0.25:    # above p25
        score *= 0.98
    else:               # below p25
        score *= 0.88

    return score


def notice_period_score(signals: dict) -> float:
    """
    Dataset: median=90 days. Values cluster at 0, 30, 60, 90, 120, 150.
    JD says: <30 preferred, buyout up to 30, 30+ harder, 90+ painful.
    
    Since median is 90 and JD wants low notice:
    - 0-30 days: top 10-15% of candidates → big boost
    - 30-60 days: good
    - 60-90 days: median — neutral
    - 90-120: penalty
    - 120-150: heavy penalty
    """
    days = signals['notice_period_days']

    if days == 0:
        return 1.20   # immediately joinable
    elif days <= 30:
        return 1.15   # well below median — positive differentiator
    elif days <= 60:
        return 1.05   # below median — slight positive
    elif days <= 90:
        return 1.0    # at median — baseline
    elif days <= 120:
        return 0.85   # above median — penalty
    else:
        return 0.70   # 150 days = p90 — strong penalty


def location_score(profile: dict, signals: dict) -> float:
    """
    JD: Pune or Noida preferred. India is acceptable.
    Outside India only if willing to relocate.
    
    Dataset: 71,196 NOT willing to relocate.
    """
    country = profile['country']
    location = profile.get('location', '').lower()

    if country == 'India':
        if 'pune' in location:
            return 1.10    # JD explicitly prefers Pune
        elif 'noida' in location or 'delhi' in location or 'ncr' in location:
            return 1.08    # JD explicitly prefers Noida
        elif 'bangalore' in location or 'bengaluru' in location:
            return 1.0     # major tech hub, easy relocation
        elif 'hyderabad' in location or 'mumbai' in location or 'chennai' in location:
            return 0.97
        else:
            return 0.93    # tier 2/3 city — may need relocation
    else:
        if signals['willing_to_relocate']:
            return 0.75    # willing to come to India — possible but logistics
        else:
            return 0.45    # outside India, not relocating — near-disqualifier


def github_score(signals: dict) -> float:
    """
    64.6% of candidates have github_activity_score = -1 (no GitHub linked).
    Valid scores range 0-96.9, mean=29, p75=42, p90=51.7
    
    Having GitHub at all is a differentiator (only 35.4% do).
    High GitHub score = external validation = JD explicitly values this.
    """
    g = signals['github_activity_score']

    if g == -1:
        return 0.92    # no github — slight penalty, not disqualifying
                       # 64.6% are in this bucket so can't be too harsh
    elif g >= 52:      # above p90 of valid scores
        return 1.20
    elif g >= 42:      # above p75
        return 1.12
    elif g >= 29:      # above mean
        return 1.06
    elif g >= 14:      # above p25
        return 1.0
    else:
        return 0.96    # has github but low activity


def career_quality_score(career_history: list) -> float:
    """
    Most important structured signal. Calibrated to actual company distribution:
    
    Consulting firms: 23K each in career history = extremely common = negative signal
    Product companies: 2800-3000 for tier-1 Indian unicorns = much less common = positive
    FAANG: 7-13 total = extremely rare = highest signal
    AI-native: 25-79 total = very rare = high signal
    
    JD explicitly disqualifies: "consulting firm–only careers (TCS, Infosys...)"
    """
    if not career_history:
        return 0.70

    total_months = sum(j.get('duration_months', 0) for j in career_history)
    if total_months == 0:
        return 0.70

    consulting_months = 0
    faang_months = 0
    ai_native_months = 0
    product_months = 0

    for job in career_history:
        company_lower = job['company'].lower()
        industry_lower = job['industry'].lower()
        months = job.get('duration_months', 0)

        # Check FAANG first
        if any(f in company_lower for f in [
            'google', 'meta', 'amazon', 'microsoft', 'netflix',
            'apple', 'adobe', 'salesforce', 'linkedin', 'uber'
        ]):
            faang_months += months

        # Check AI-native
        elif any(a in company_lower for a in [
            'sarvam', 'rephrase', 'aganitha', 'niramai', 'saarthi',
            'mad street den', 'observe.ai', 'krutrim', 'wysa', 'haptik',
            'verloop', 'yellow.ai', 'locobuzz', 'glance'
        ]):
            ai_native_months += months

        # Check consulting
        elif any(c in company_lower for c in CONSULTING_FIRMS):
            consulting_months += months

        # Check product by industry
        elif industry_lower in PRODUCT_INDUSTRIES:
            product_months += months

    consulting_ratio = consulting_months / total_months
    faang_ratio = faang_months / total_months
    ai_native_ratio = ai_native_months / total_months
    product_ratio = product_months / total_months

    # FAANG/AI-native experience is the highest signal
    if faang_ratio > 0:
        base = 1.35 + (faang_ratio * 0.15)   # 1.35 - 1.50
    elif ai_native_ratio > 0:
        base = 1.25 + (ai_native_ratio * 0.10)  # 1.25 - 1.35
    elif product_ratio >= 0.6:
        base = 1.15
    elif product_ratio >= 0.3:
        base = 1.05
    else:
        base = 0.90

    # Consulting penalty — JD is explicit about this
    if consulting_ratio >= 0.90:
        base *= 0.45   # entire career consulting = strong disqualifier
    elif consulting_ratio >= 0.70:
        base *= 0.65
    elif consulting_ratio >= 0.50:
        base *= 0.80
    elif consulting_ratio >= 0.30:
        base *= 0.90   # mixed but still some consulting — slight penalty

    return min(base, 1.50)   # cap at 1.5x


def skill_depth_score(candidate: dict) -> float:
    """
    Validates claimed skills against assessment scores.
    
    From data: assessment means for core skills range 54-56.
    Anything >60 is above average, >70 is strong.
    
    Also checks: do they have MULTIPLE tier-1 skills with real duration?
    (Many candidates have 1-2 core skills; having 4+ with duration is rare)
    """
    assessments = candidate['redrob_signals'].get('skill_assessment_scores', {})
    skills = candidate.get('skills', [])

    # Count tier-1 skills with real duration
    core_skills_with_depth = [
        s for s in skills
        if s['name'].lower() in CORE_JD_SKILLS
        and s['proficiency'] in ('advanced', 'expert')
        and s.get('duration_months', 0) >= 12  # at least 1 year of actual use
    ]

    # Base from skill depth (more core skills = more breadth in IR/ML stack)
    depth_count = len(core_skills_with_depth)
    if depth_count >= 6:
        depth_bonus = 1.15
    elif depth_count >= 4:
        depth_bonus = 1.10
    elif depth_count >= 2:
        depth_bonus = 1.05
    elif depth_count >= 1:
        depth_bonus = 1.0
    else:
        depth_bonus = 0.85   # no validated core skills is a red flag

    # Assessment bonus for relevant skills
    assessment_bonus = 1.0
    for skill_name, score in assessments.items():
        if skill_name.lower() in CORE_JD_SKILLS:
            if score >= 70:
                assessment_bonus += 0.06
            elif score >= 60:
                assessment_bonus += 0.03
    assessment_bonus = min(assessment_bonus, 1.25)

    return depth_bonus * assessment_bonus


def recruiter_market_signal(signals: dict) -> float:
    """
    Recruiter market signals tell us how the market already values this candidate.
    
    saved_by_recruiters_30d: mean=7.66, p75=11, p90=15, p99=28
    profile_views_received_30d: mean=47.99, p75=68, p90=86
    
    If many recruiters are already saving this profile, that's external validation.
    """
    saved = signals.get('saved_by_recruiters_30d', 0)
    views = signals.get('profile_views_received_30d', 0)

    score = 1.0

    # saved_by_recruiters: p75=11, p90=15, p99=28
    if saved >= 20:     # top ~2%
        score *= 1.10
    elif saved >= 15:   # p90
        score *= 1.06
    elif saved >= 11:   # p75
        score *= 1.03
    elif saved <= 2:    # very low market interest
        score *= 0.95

    # profile_views: p75=68, p90=86
    if views >= 100:
        score *= 1.04
    elif views >= 68:   # p75
        score *= 1.02

    return score


def compute_final_score(ce_score: float, candidate: dict) -> float:
    """
    Combine cross-encoder score with structured signals.
    
    Weights:
    - Cross-encoder already captures text fit (JD ↔ candidate doc)
    - Structured signals multiply to adjust for availability, career quality etc.
    
    ce_score: sigmoid-normalized cross-encoder output (0-1)
    """
    import math

    profile = candidate['profile']
    signals = candidate['redrob_signals']
    career = candidate.get('career_history', [])

    # All multipliers composed together
    structured_multiplier = (
        experience_fit_score(profile)
        * availability_score(signals)
        * notice_period_score(signals)
        * location_score(profile, signals)
        * github_score(signals)
        * career_quality_score(career)
        * skill_depth_score(candidate)
        * recruiter_market_signal(signals)
    )

    return ce_score * structured_multiplier