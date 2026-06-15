from datetime import date, datetime

CORE_SKILLS = {
    'faiss', 'pinecone', 'weaviate', 'qdrant', 'elasticsearch',
    'embeddings', 'information retrieval', 'ranking',
    'recommendation systems', 'nlp', 'vector search', 'rag',
    'sentence-transformers', 'bm25', 'hybrid search'
}

PRODUCT_INDUSTRIES = {
    'software', 'ai/ml', 'fintech', 'e-commerce', 'saas',
    'food delivery', 'transportation', 'edtech', 'healthtech'
}

def parse_date(date_str):
    return datetime.strptime(date_str, "%Y-%m-%d").date()

def generate_reasoning(candidate):
    p = candidate['profile']
    s = candidate['redrob_signals']
    career = candidate['career_history']
    skills = candidate['skills']
    today = date.today()

    parts = []

    # Opening: years of experience + current role
    parts.append(
        f"{p['years_of_experience']}yr {p['current_title']} "
        f"at {p['current_company']}"
    )

    # Core technical skills that are verified
    strong_core = [
        sk['name'] for sk in skills
        if sk['proficiency'] in ('expert', 'advanced')
        and sk['name'].lower() in CORE_SKILLS
    ]
    if strong_core:
        parts.append(f"advanced in {', '.join(strong_core[:3])}")

    # Product company signal
    product_jobs = [
        j for j in career
        if j['industry'].lower() in PRODUCT_INDUSTRIES
    ]
    if product_jobs:
        parts.append(
            f"product company background "
            f"({', '.join(set(j['company'] for j in product_jobs[:2]))})"
        )

    # Assessment scores
    assessments = s['skill_assessment_scores']
    verified = {
        k: v for k, v in assessments.items()
        if k.lower() in CORE_SKILLS and v >= 60
    }
    if verified:
        top_verified = sorted(verified.items(), key=lambda x: x[1], reverse=True)[:2]
        parts.append(
            f"platform-verified: "
            f"{', '.join(f'{k} {v:.0f}/100' for k, v in top_verified)}"
        )

    sentence1 = "; ".join(parts) + "."

    # Concerns and availability as second sentence
    concerns = []
    days_inactive = (today - parse_date(s['last_active_date'])).days

    if s['open_to_work_flag'] and s['notice_period_days'] <= 30:
        concerns.append("immediately available and actively looking")
    elif s['notice_period_days'] > 90:
        concerns.append(f"{s['notice_period_days']}-day notice period is a risk")

    if days_inactive > 90:
        concerns.append(f"inactive for {days_inactive} days")

    if s['recruiter_response_rate'] < 0.2:
        concerns.append(
            f"low recruiter response rate ({s['recruiter_response_rate']:.0%})"
        )

    if p['country'] != 'India':
        concerns.append(
            f"based in {p['country']}, "
            f"{'willing' if s['willing_to_relocate'] else 'unwilling'} to relocate"
        )

    sentence2 = ("; ".join(concerns) + ".").capitalize() if concerns else ""

    return (sentence1 + " " + sentence2).strip()