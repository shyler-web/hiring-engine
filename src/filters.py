"""
filters.py — Honeypot detection and hard business filters.

Key insights from 100K data analysis:
- Fake companies in dataset: Pied Piper, Initech, Wayne Enterprises, Acme Corp,
  Stark Industries, Hooli, Globex Inc, Dunder Mifflin — these are fictional but
  VALID candidates work at them. Do NOT filter by company name.
- Real consulting firms: Infosys, Wipro, TCS, Capgemini, HCL, Mindtree,
  Accenture, Cognizant, Tech Mahindra, Mphasis (~23K entries each in career history)
- 64.6% of candidates have github_activity_score = -1 (no GitHub linked)
- 59.6% have offer_acceptance_rate = -1 (no offer history)
- Only 35,339 / 100,000 are open_to_work
- Mean YoE = 7.17, median = 6.80
- Notice period mean = 87 days, median = 90 days (majority are 60-90-120-150)
- Most irrelevant titles appear ~18-19K times each (Business Analyst, Graphic Designer etc)
"""

from datetime import date, datetime

# Real consulting/services firms from actual dataset
CONSULTING_FIRMS = {
    'infosys', 'wipro', 'tcs', 'capgemini', 'hcl', 'mindtree',
    'accenture', 'cognizant', 'tech mahindra', 'mphasis', 'hexaware',
    'l&t infotech', 'ltimindtree', 'genpact'
}

# Titles that are completely irrelevant to AI/ML engineering
# From data: each appears ~5700-5830 times as current title
HARD_IRRELEVANT_TITLES = {
    'business analyst', 'graphic designer', 'mechanical engineer',
    'accountant', 'project manager', 'customer support',
    'operations manager', 'content writer', 'sales executive',
    'civil engineer', 'marketing manager', 'hr manager',
}

# Industries with AI/ML relevance — from actual dataset industry list
TECH_INDUSTRIES = {
    'software', 'ai/ml', 'fintech', 'e-commerce', 'saas',
    'food delivery', 'transportation', 'edtech', 'healthtech',
    'healthtech ai', 'conversational ai', 'ai services', 'voice ai',
    'adtech', 'insurance tech', 'gaming', 'internet', 'media',
    'consumer electronics'
}


def parse_date(date_str: str) -> date:
    return datetime.strptime(date_str, "%Y-%m-%d").date()


def is_honeypot(candidate: dict) -> bool:
    """
    Detect honeypot candidates with impossible/fabricated profiles.
    Returns True if candidate should be discarded.
    
    Honeypot signals based on schema analysis:
    1. Expert/advanced skill with duration_months = 0
    2. Career timeline mathematically impossible vs years_of_experience
    3. Assessment score contradicts claimed expert proficiency
    4. Suspiciously perfect profile with impossible combinations
    """
    flags = 0

    # --- Rule 1: expert/advanced skill with 0 months duration ---
    # Legitimate candidates with 5+ years in AI will have real duration on core skills
    zero_duration_advanced = 0
    for skill in candidate.get('skills', []):
        if skill['proficiency'] in ('expert', 'advanced'):
            if skill.get('duration_months', 1) == 0:
                zero_duration_advanced += 1
    if zero_duration_advanced >= 2:
        flags += 1
    if zero_duration_advanced >= 4:
        flags += 1  # double flag for egregious cases

    # --- Rule 2: Career timeline impossibility ---
    # Allow 18 months buffer for legitimate job overlaps during transitions
    career = candidate.get('career_history', [])
    total_career_months = sum(j.get('duration_months', 0) for j in career)
    yoe_months = candidate['profile']['years_of_experience'] * 12
    if total_career_months > yoe_months + 18:
        flags += 1

    # --- Rule 3: Assessment contradicts proficiency ---
    # An "expert" who scores <25 on their own skill assessment is suspicious
    assessments = candidate['redrob_signals'].get('skill_assessment_scores', {})
    skill_proficiency_map = {
        s['name'].lower(): s['proficiency']
        for s in candidate.get('skills', [])
    }
    contradictions = 0
    for skill_name, score in assessments.items():
        claimed = skill_proficiency_map.get(skill_name.lower(), '')
        if claimed == 'expert' and score < 25:
            contradictions += 1
    if contradictions >= 2:
        flags += 1

    # --- Rule 4: Impossibly short tenure at very old companies ---
    # e.g. claims 10 years at a company that only has 3 year history in dataset
    # We can't check founding dates, but we can check single-role > YoE
    for job in career:
        if job.get('duration_months', 0) > yoe_months + 6:
            flags += 1
            break

    # --- Rule 5: Perfect completeness + zero behavioral signals ---
    # Real active candidates have some profile views, applications etc.
    signals = candidate['redrob_signals']
    if (signals.get('profile_completeness_score', 0) >= 95 and
            signals.get('profile_views_received_30d', 0) == 0 and
            signals.get('applications_submitted_30d', 0) == 0 and
            signals.get('saved_by_recruiters_30d', 0) == 0):
        flags += 1

    return flags >= 2


def passes_hard_filters(candidate: dict) -> bool:
    """
    Hard business rules. Eliminate candidates who cannot possibly be hired.
    Conservative — only eliminate when very confident.
    """
    profile = candidate['profile']
    signals = candidate['redrob_signals']
    today = date.today()

    # --- Filter 1: Must be India-based or willing to relocate ---
    # 71,196 are NOT willing to relocate, so this matters
    if profile['country'] != 'India' and not signals['willing_to_relocate']:
        return False

    # --- Filter 2: Minimum experience floor ---
    # JD says 5-9 years, we allow 3+ to not be too aggressive
    # Mean YoE in dataset is 7.17 so this removes genuine juniors
    if profile['years_of_experience'] < 3:
        return False

    # --- Filter 3: Completely dead + not looking ---
    # open_to_work=False AND inactive for 6+ months = not reachable
    # 64,661 are not open_to_work — don't eliminate all, just truly dead ones
    days_inactive = (today - parse_date(signals['last_active_date'])).days
    if not signals['open_to_work_flag'] and days_inactive > 180:
        return False

    # --- Filter 4: Completely irrelevant career with zero tech exposure ---
    # Only eliminate if current title is irrelevant AND career has no tech industry at all
    title_lower = profile['current_title'].lower()
    if title_lower in HARD_IRRELEVANT_TITLES:
        all_industries = {j['industry'].lower() for j in candidate.get('career_history', [])}
        has_any_tech = any(
            ind in TECH_INDUSTRIES or 'tech' in ind or 'software' in ind or 'ai' in ind
            for ind in all_industries
        )
        # Also check if they have any AI skills listed
        has_ai_skills = any(
            s['name'].lower() in {
                'machine learning', 'python', 'nlp', 'deep learning',
                'pytorch', 'tensorflow', 'embeddings', 'rag', 'llms'
            }
            for s in candidate.get('skills', [])
            if s['proficiency'] in ('advanced', 'expert')
        )
        if not has_any_tech and not has_ai_skills:
            return False

    return True


def filter_candidates(candidates: list) -> list:
    """
    Apply honeypot detection and hard filters.
    Returns filtered list with stats printed.
    """
    passed = []
    honeypot_count = 0
    hard_filter_count = 0

    for c in candidates:
        if is_honeypot(c):
            honeypot_count += 1
            continue
        if not passes_hard_filters(c):
            hard_filter_count += 1
            continue
        passed.append(c)

    total = len(candidates)
    print(f"[filter] Total input:       {total:,}")
    print(f"[filter] Honeypots removed: {honeypot_count:,}")
    print(f"[filter] Hard filtered:     {hard_filter_count:,}")
    print(f"[filter] Remaining:         {len(passed):,} ({len(passed)/total*100:.1f}%)")
    return passed