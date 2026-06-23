"""
filters.py — Honeypot detection and hard business filters.

V2 changes vs V1:
  - Added fictional company filter (>50% career months at Dunder Mifflin etc.)
  - Added DROP_TEMPLATES support: candidates whose current JD fingerprint
    maps to Templates #1-15 (irrelevant) are dropped early.
    NOTE: This is called from rank.py after template_map is loaded,
    not inside filter_candidates() itself, because filter_candidates()
    runs before template_map is available. The fictional company check
    IS included here since it only needs the candidate dict.

Key insights from 100K data analysis:
- Fake companies in dataset: Pied Piper, Initech, Wayne Enterprises, Acme Corp,
  Stark Industries, Hooli, Globex Inc, Dunder Mifflin — these appear as
  employers. Candidates who spent >50% of their career at these are bots.
- Real consulting firms: Infosys, Wipro, TCS, Capgemini, HCL, Mindtree,
  Accenture, Cognizant, Tech Mahindra, Mphasis (~23K entries each)
- 64.6% of candidates have github_activity_score = -1 (no GitHub linked)
- 59.6% have offer_acceptance_rate = -1 (no offer history)
- Only 35,339 / 100,000 are open_to_work
- Mean YoE = 7.17, median = 6.80
- Notice period mean = 87 days, median = 90 days
- Most irrelevant titles appear ~18-19K times each
"""

from datetime import date, datetime

CONSULTING_FIRMS = {
    'infosys', 'wipro', 'tcs', 'capgemini', 'hcl', 'mindtree',
    'accenture', 'cognizant', 'tech mahindra', 'mphasis', 'hexaware',
    'l&t infotech', 'ltimindtree', 'genpact'
}

HARD_IRRELEVANT_TITLES = {
    'business analyst', 'graphic designer', 'mechanical engineer',
    'accountant', 'project manager', 'customer support',
    'operations manager', 'content writer', 'sales executive',
    'civil engineer', 'marketing manager', 'hr manager',
}

TECH_INDUSTRIES = {
    'software', 'ai/ml', 'fintech', 'e-commerce', 'saas',
    'food delivery', 'transportation', 'edtech', 'healthtech',
    'healthtech ai', 'conversational ai', 'ai services', 'voice ai',
    'adtech', 'insurance tech', 'gaming', 'internet', 'media',
    'consumer electronics'
}

# Fictional companies used as synthetic filler in dataset.
# Candidates whose career is >50% at these are bots/honeypots.
FICTIONAL_COMPANIES = {
    'pied piper', 'initech', 'wayne enterprises', 'acme corp',
    'stark industries', 'hooli', 'globex inc', 'dunder mifflin'
}


def parse_date(date_str: str) -> date:
    return datetime.strptime(date_str, "%Y-%m-%d").date()


def _fictional_career_ratio(candidate: dict) -> float:
    """
    Return the fraction of career months spent at fictional companies.
    Used by both is_honeypot() (as a flag) and passes_hard_filters() (as a cut).
    """
    career = candidate.get('career_history', [])
    total = sum(j.get('duration_months', 0) for j in career)
    if total == 0:
        return 0.0
    fictional = sum(
        j.get('duration_months', 0) for j in career
        if j.get('company', '').lower() in FICTIONAL_COMPANIES
    )
    return fictional / total


def is_honeypot(candidate: dict) -> bool:
    """
    Detect honeypot candidates with impossible/fabricated profiles.
    Returns True if candidate should be discarded.

    Rules:
      1. Expert/advanced skill with duration_months = 0
      2. Career timeline mathematically impossible vs years_of_experience
      3. Assessment score contradicts claimed expert proficiency
      4. Single job duration > total YoE
      5. Perfect completeness + zero behavioral signals
      6. >50% of career at fictional companies (new in V2)
    """
    flags = 0

    # --- Rule 1: expert/advanced skill with 0 months duration ---
    zero_duration_advanced = sum(
        1 for s in candidate.get('skills', [])
        if s['proficiency'] in ('expert', 'advanced')
        and s.get('duration_months', 1) == 0
    )
    if zero_duration_advanced >= 2:
        flags += 1
    if zero_duration_advanced >= 4:
        flags += 1  # double flag for egregious cases

    # --- Rule 2: Career timeline impossibility ---
    career = candidate.get('career_history', [])
    total_career_months = sum(j.get('duration_months', 0) for j in career)
    yoe_months = candidate['profile']['years_of_experience'] * 12
    if total_career_months > yoe_months + 18:
        flags += 1

    # --- Rule 3: Assessment contradicts proficiency ---
    assessments = candidate['redrob_signals'].get('skill_assessment_scores', {})
    skill_proficiency_map = {
        s['name'].lower(): s['proficiency']
        for s in candidate.get('skills', [])
    }
    contradictions = sum(
        1 for skill_name, score in assessments.items()
        if skill_proficiency_map.get(skill_name.lower(), '') == 'expert'
        and score < 25
    )
    if contradictions >= 2:
        flags += 1

    # --- Rule 4: Single job duration > total YoE ---
    for job in career:
        if job.get('duration_months', 0) > yoe_months + 6:
            flags += 1
            break

    # --- Rule 5: Perfect completeness + zero behavioral signals ---
    signals = candidate['redrob_signals']
    if (signals.get('profile_completeness_score', 0) >= 95 and
            signals.get('profile_views_received_30d', 0) == 0 and
            signals.get('applications_submitted_30d', 0) == 0 and
            signals.get('saved_by_recruiters_30d', 0) == 0):
        flags += 1

    # --- Rule 6 (NEW): Majority fictional career ---
    # >50% of career at Dunder Mifflin / Pied Piper etc. → almost certainly a bot
    if _fictional_career_ratio(candidate) > 0.5:
        flags += 2  # Strong signal — double flag immediately

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
    if profile['country'] != 'India' and not signals['willing_to_relocate']:
        return False

    # --- Filter 2: Minimum experience floor ---
    # JD says 5-9 years, we allow 3+ to not be too aggressive
    if profile['years_of_experience'] < 3:
        return False

    # --- Filter 3: Completely dead + not looking ---
    days_inactive = (today - parse_date(signals['last_active_date'])).days
    if not signals['open_to_work_flag'] and days_inactive > 180:
        return False

    # --- Filter 4: Completely irrelevant career with zero tech exposure ---
    title_lower = profile['current_title'].lower()
    if title_lower in HARD_IRRELEVANT_TITLES:
        all_industries = {j['industry'].lower() for j in candidate.get('career_history', [])}
        has_any_tech = any(
            ind in TECH_INDUSTRIES or 'tech' in ind or 'software' in ind or 'ai' in ind
            for ind in all_industries
        )
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

    # --- Filter 5 (NEW): Majority fictional career ---
    # Redundant with is_honeypot Rule 6 but acts as a safety net
    # in case honeypot flags didn't fire (e.g. clean behavioral signals on a bot)
    if _fictional_career_ratio(candidate) > 0.5:
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
