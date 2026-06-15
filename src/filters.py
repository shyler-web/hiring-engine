from datetime import date, datetime

CONSULTING_FIRMS = {
    'tcs', 'infosys', 'wipro', 'accenture', 'cognizant',
    'capgemini', 'hcl', 'tech mahindra', 'mphasis', 'hexaware',
    'mindtree', 'l&t infotech', 'ltimindtree'
}

IRRELEVANT_TITLES = {
    'civil engineer', 'accountant', 'graphic designer',
    'hr manager', 'content writer', 'sales executive',
    'financial analyst', 'mechanical engineer'
}

IRRELEVANT_INDUSTRIES = {
    'paper products', 'manufacturing', 'conglomerate',
    'retail', 'real estate', 'agriculture'
}


def parse_date(date_str):
    return datetime.strptime(date_str, "%Y-%m-%d").date()


def is_honeypot(candidate):
    flags = 0

    # Rule 1: expert/advanced skill with 0 months duration
    for skill in candidate['skills']:
        if skill['proficiency'] in ('expert', 'advanced'):
            if skill.get('duration_months', 1) == 0:
                flags += 1

    # Rule 2: career timeline impossibility
    total_career_months = sum(
        j['duration_months'] for j in candidate['career_history']
    )
    yoe_months = candidate['profile']['years_of_experience'] * 12
    if total_career_months > yoe_months + 18:  # 18 month buffer for overlaps
        flags += 1

    # Rule 3: assessment score contradicts claimed proficiency
    assessments = candidate['redrob_signals']['skill_assessment_scores']
    skill_proficiency = {
        s['name']: s['proficiency'] for s in candidate['skills']
    }
    for skill_name, score in assessments.items():
        if skill_proficiency.get(skill_name) == 'expert' and score < 25:
            flags += 1

    return flags >= 2


def passes_hard_filters(candidate):
    profile = candidate['profile']
    signals = candidate['redrob_signals']
    today = date.today()

    # Location filter
    if profile['country'] != 'India' and not signals['willing_to_relocate']:
        return False

    # Experience floor
    if profile['years_of_experience'] < 3:
        return False

    # Completely dead profile
    days_inactive = (today - parse_date(signals['last_active_date'])).days
    if not signals['open_to_work_flag'] and days_inactive > 180:
        return False

    # Entirely irrelevant career with no software exposure
    all_industries = {j['industry'].lower() for j in candidate['career_history']}
    title_lower = profile['current_title'].lower()
    if title_lower in IRRELEVANT_TITLES and not any(
        'software' in ind or 'tech' in ind or 'ai' in ind
        for ind in all_industries
    ):
        return False

    return True


def filter_candidates(candidates):
    passed = []
    honeypots_removed = 0
    hard_filtered = 0

    for c in candidates:
        if is_honeypot(c):
            honeypots_removed += 1
            continue
        if not passes_hard_filters(c):
            hard_filtered += 1
            continue
        passed.append(c)

    print(f"Honeypots removed: {honeypots_removed}")
    print(f"Hard filtered: {hard_filtered}")
    print(f"Remaining: {len(passed)}")
    return passed