from datetime import date, datetime

CONSULTING_FIRMS = {
    'tcs', 'infosys', 'wipro', 'accenture', 'cognizant',
    'capgemini', 'hcl', 'tech mahindra', 'mphasis', 'hexaware',
    'mindtree', 'l&t infotech', 'ltimindtree'
}

PRODUCT_INDUSTRIES = {
    'software', 'ai/ml', 'fintech', 'e-commerce', 'saas',
    'food delivery', 'transportation', 'edtech', 'healthtech'
}

CORE_SKILLS = {
    'faiss', 'pinecone', 'weaviate', 'qdrant', 'elasticsearch',
    'embeddings', 'information retrieval', 'ranking',
    'recommendation systems', 'nlp', 'vector search', 'rag'
}


def parse_date(date_str):
    return datetime.strptime(date_str, "%Y-%m-%d").date()


def availability_score(signals):
    score = 1.0
    today = date.today()

    if signals['open_to_work_flag']:
        score *= 1.2
    else:
        score *= 0.7

    days_inactive = (today - parse_date(signals['last_active_date'])).days
    if days_inactive <= 30:
        score *= 1.15
    elif days_inactive <= 90:
        score *= 1.0
    elif days_inactive <= 180:
        score *= 0.8
    else:
        score *= 0.5

    rr = signals['recruiter_response_rate']
    if rr >= 0.7:
        score *= 1.1
    elif rr >= 0.4:
        score *= 1.0
    else:
        score *= 0.85

    return score


def notice_score(signals):
    days = signals['notice_period_days']
    if days <= 30:
        return 1.0
    elif days <= 60:
        return 0.9
    elif days <= 90:
        return 0.75
    else:
        return 0.6


def location_score(profile, signals):
    country = profile['country']
    location = profile.get('location', '').lower()

    if country == 'India':
        if 'pune' in location or 'noida' in location or 'delhi' in location:
            return 1.0
        return 0.95
    elif signals['willing_to_relocate']:
        return 0.8
    return 0.5


def github_score(signals):
    g = signals['github_activity_score']
    if g == -1:
        return 0.9
    elif g >= 50:
        return 1.15
    elif g >= 20:
        return 1.05
    return 1.0


def career_quality_score(career_history):
    total_months = sum(j['duration_months'] for j in career_history)
    if total_months == 0:
        return 0.7

    product_months = 0
    consulting_months = 0

    for job in career_history:
        company_lower = job['company'].lower()
        is_consulting = any(cf in company_lower for cf in CONSULTING_FIRMS)
        is_product = job['industry'].lower() in PRODUCT_INDUSTRIES

        if is_consulting:
            consulting_months += job['duration_months']
        elif is_product:
            product_months += job['duration_months']

    consulting_ratio = consulting_months / total_months
    product_ratio = product_months / total_months

    if consulting_ratio >= 0.9:
        return 0.4
    elif product_ratio >= 0.6:
        return 1.2
    elif product_ratio >= 0.3:
        return 1.0
    return 0.8


def skill_verification_bonus(candidate):
    assessments = candidate['redrob_signals']['skill_assessment_scores']
    bonus = 1.0
    for skill_name, score in assessments.items():
        if skill_name.lower() in CORE_SKILLS and score >= 60:
            bonus += 0.05
    return min(bonus, 1.25)


def compute_final_score(ce_score, candidate):
    profile = candidate['profile']
    signals = candidate['redrob_signals']
    career = candidate['career_history']

    multiplier = (
        availability_score(signals)
        * notice_score(signals)
        * location_score(profile, signals)
        * github_score(signals)
        * career_quality_score(career)
        * skill_verification_bonus(candidate)
    )

    return ce_score * multiplier