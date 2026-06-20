"""
reasoning.py — Non-templated, archetype-based reasoning generation.

Core problem with previous version: every candidate got the same 5-part
sentence structure regardless of what made them interesting. Stage 4 caught it.

Fix: Assign each candidate to an archetype based on their dominant signal,
then write reasoning from that archetype's perspective. Different archetypes
produce structurally different sentences about different things.

Archetypes:
  A — FAANG/big-tech veteran (rare, 7-14 people in dataset)
  B — AI-native company specialist (Sarvam AI, Mad Street Den, etc.)
  C — Indian unicorn product engineer (Swiggy, Razorpay, Zomato, CRED etc.)
  D — Deep specialist (exceptional skill duration + verified scores)
  E — Strong generalist (breadth across IR/ML stack, product background)
  F — Available fast-mover (open, short notice, recently active)
  G — Borderline (weaker profile that made it through on text fit alone)
"""

from datetime import date, datetime

CORE_JD_SKILLS = {
    'faiss', 'pinecone', 'weaviate', 'qdrant', 'milvus', 'elasticsearch',
    'opensearch', 'pgvector', 'vector search', 'semantic search',
    'information retrieval', 'hybrid search', 'bm25',
    'sentence transformers', 'embeddings', 'hugging face transformers',
    'learning to rank', 'recommendation systems', 'lora', 'qlora', 'peft',
    'fine-tuning llms', 'haystack', 'llamaindex', 'pytorch', 'tensorflow',
    'scikit-learn', 'nlp', 'machine learning', 'deep learning', 'python', 'rag',
}

PRODUCT_INDUSTRIES = {
    'software', 'ai/ml', 'fintech', 'e-commerce', 'saas', 'food delivery',
    'transportation', 'edtech', 'healthtech', 'healthtech ai',
    'conversational ai', 'ai services', 'voice ai', 'adtech',
    'insurance tech', 'gaming', 'internet'
}

CONSULTING_FIRMS = {
    'infosys', 'wipro', 'tcs', 'capgemini', 'hcl', 'mindtree',
    'accenture', 'cognizant', 'tech mahindra', 'mphasis', 'hexaware',
    'l&t infotech', 'ltimindtree', 'genpact'
}

FAANG = {
    'google', 'meta', 'amazon', 'microsoft', 'netflix',
    'apple', 'adobe', 'salesforce', 'linkedin', 'uber'
}

AI_NATIVE = {
    'sarvam', 'rephrase', 'aganitha', 'niramai', 'saarthi',
    'mad street den', 'observe.ai', 'krutrim', 'wysa', 'haptik',
    'verloop', 'yellow.ai', 'locobuzz', 'glance'
}

INDIAN_UNICORNS = {
    'swiggy', 'razorpay', 'cred', 'zomato', 'flipkart', 'meesho',
    'nykaa', 'inmobi', 'policybazaar', 'ola', 'zoho', 'vedantu',
    'paytm', 'unacademy', 'pharmeasy', 'upgrad', 'freshworks',
    'phonepe', 'dream11', 'byju'
}


def parse_date(d: str) -> date:
    return datetime.strptime(d, "%Y-%m-%d").date()


def _get_core_skills(candidate: dict) -> list:
    """Return advanced/expert core skills sorted by duration desc."""
    return sorted(
        [s for s in candidate.get('skills', [])
         if s['name'].lower() in CORE_JD_SKILLS
         and s['proficiency'] in ('advanced', 'expert')
         and s.get('duration_months', 0) > 0],
        key=lambda x: x.get('duration_months', 0),
        reverse=True
    )


def _classify_companies(career: list) -> dict:
    """Returns counts of months at each company tier."""
    result = {'faang': [], 'ai_native': [], 'unicorn': [], 'consulting': [], 'other': []}
    for job in career:
        co = job['company'].lower()
        if any(f in co for f in FAANG):
            result['faang'].append(job['company'])
        elif any(a in co for a in AI_NATIVE):
            result['ai_native'].append(job['company'])
        elif any(u in co for u in INDIAN_UNICORNS):
            result['unicorn'].append(job['company'])
        elif any(c in co for c in CONSULTING_FIRMS):
            result['consulting'].append(job['company'])
        else:
            result['other'].append(job['company'])
    return result


def _best_assessment(candidate: dict) -> tuple:
    """Return (skill_name, score) for highest-scoring core skill assessment."""
    assessments = candidate['redrob_signals'].get('skill_assessment_scores', {})
    best = None
    for k, v in assessments.items():
        if k.lower() in CORE_JD_SKILLS and v >= 60:
            if best is None or v > best[1]:
                best = (k, v)
    return best


def _archetype(candidate: dict, companies: dict) -> str:
    """Assign candidate to reasoning archetype."""
    signals = candidate['redrob_signals']
    today = date.today()
    days_inactive = (today - parse_date(signals['last_active_date'])).days

    if companies['faang']:
        return 'A'
    if companies['ai_native']:
        return 'B'
    if companies['unicorn']:
        # Check if also a fast-mover
        if (signals['open_to_work_flag']
                and signals['notice_period_days'] <= 30
                and days_inactive <= 30):
            return 'F'
        return 'C'

    core_skills = _get_core_skills(candidate)
    if len(core_skills) >= 4:
        best = _best_assessment(candidate)
        if best and best[1] >= 75:
            return 'D'

    if (signals['open_to_work_flag']
            and signals['notice_period_days'] <= 30
            and days_inactive <= 14):
        return 'F'

    if len(core_skills) >= 3:
        return 'E'

    return 'G'


def _reason_A(candidate: dict, companies: dict) -> str:
    """FAANG veteran — lead with where they worked and what they built."""
    p = candidate['profile']
    s = candidate['redrob_signals']
    career = candidate.get('career_history', [])
    today = date.today()

    faang_jobs = [j for j in career if any(f in j['company'].lower() for f in FAANG)]
    faang_co = faang_jobs[0]['company'] if faang_jobs else p['current_company']
    faang_title = faang_jobs[0]['title'] if faang_jobs else p['current_title']

    core = _get_core_skills(candidate)
    skill_str = ''
    if core:
        top = core[:2]
        skill_str = f", with {top[0]['duration_months']}mo hands-on in {top[0]['name']}"
        if len(top) > 1:
            skill_str += f" and {top[1]['duration_months']}mo in {top[1]['name']}"

    sentence1 = (
        f"Former {faang_title} at {faang_co}{skill_str} — "
        f"one of very few candidates in this pool with production-scale experience "
        f"at a top-tier tech company."
    )

    # Concern
    concerns = []
    days_inactive = (today - parse_date(s['last_active_date'])).days
    if p['years_of_experience'] > 13:
        concerns.append(
            f"at {p['years_of_experience']}yr they may be overqualified for a founding-team IC role"
        )
    if s['notice_period_days'] > 60:
        concerns.append(f"{s['notice_period_days']}-day notice")
    if not s['open_to_work_flag'] and days_inactive > 60:
        concerns.append(f"not marked open-to-work, inactive {days_inactive}d")
    if p['country'] != 'India' and not s['willing_to_relocate']:
        concerns.append(f"based in {p['country']}, not willing to relocate")

    sentence2 = ('Risk: ' + '; '.join(concerns) + '.') if concerns else (
        f"Open to work with {s['notice_period_days']}-day notice — likely fielding multiple offers."
    )

    return sentence1 + ' ' + sentence2


def _reason_B(candidate: dict, companies: dict) -> str:
    """AI-native company specialist — domain is right, highlight specific work."""
    p = candidate['profile']
    s = candidate['redrob_signals']
    career = candidate.get('career_history', [])
    today = date.today()

    ai_jobs = [j for j in career if any(a in j['company'].lower() for a in AI_NATIVE)]
    primary_co = ai_jobs[0]['company']
    primary_title = ai_jobs[0]['title']
    primary_desc_snippet = ai_jobs[0]['description'][:120] if ai_jobs[0].get('description') else ''

    core = _get_core_skills(candidate)
    best_skill = core[0] if core else None

    if best_skill:
        sentence1 = (
            f"{primary_title} at {primary_co} — an AI-native company — "
            f"with {best_skill['duration_months']}mo building on {best_skill['name']}; "
            f"domain and stack align closely with this role."
        )
    else:
        sentence1 = (
            f"{primary_title} at {primary_co}; background in applied AI at a "
            f"product-focused AI company directly relevant to this search role."
        )

    # Availability and concern
    days_inactive = (today - parse_date(s['last_active_date'])).days
    parts = []
    if s['open_to_work_flag'] and s['notice_period_days'] <= 30:
        parts.append(f"available in {s['notice_period_days']} days")
    elif s['notice_period_days'] >= 90:
        parts.append(f"notice period is {s['notice_period_days']} days — plan ahead")
    if days_inactive > 60:
        parts.append(f"last active {days_inactive}d ago, may need warm outreach")

    best = _best_assessment(candidate)
    if best and best[1] >= 70:
        parts.append(f"scored {best[1]:.0f}/100 on {best[0]} assessment")

    g = s['github_activity_score']
    if g > 50:
        parts.append(f"GitHub score {g:.0f}/100 shows active external contribution")

    sentence2 = ('; '.join(parts) + '.').capitalize() if parts else ''
    return (sentence1 + ' ' + sentence2).strip()


def _reason_C(candidate: dict, companies: dict) -> str:
    """Indian unicorn engineer — emphasize shipped product at scale."""
    p = candidate['profile']
    s = candidate['redrob_signals']
    career = candidate.get('career_history', [])
    today = date.today()

    unicorn_jobs = [
        j for j in career
        if any(u in j['company'].lower() for u in INDIAN_UNICORNS)
    ]
    companies_str = ', '.join(
        dict.fromkeys(j['company'] for j in unicorn_jobs[:2])
    )

    core = _get_core_skills(candidate)
    best = _best_assessment(candidate)

    if core and len(core) >= 2:
        skills_str = f"{core[0]['name']} ({core[0]['duration_months']}mo) and {core[1]['name']} ({core[1]['duration_months']}mo)"
        sentence1 = (
            f"{p['current_title']} with {p['years_of_experience']}yr at {companies_str}; "
            f"has shipped production {skills_str} — "
            f"the kind of hands-on retrieval experience the JD is looking for."
        )
    elif core:
        sentence1 = (
            f"{p['current_title']} at {companies_str} with {core[0]['duration_months']}mo "
            f"on {core[0]['name']}; product company background in the right domain."
        )
    else:
        sentence1 = (
            f"{p['current_title']} at {companies_str}; "
            f"{p['years_of_experience']}yr at Indian product companies, "
            f"though core IR/search skills are thinner than ideal."
        )

    # Honest second sentence
    parts = []
    days_inactive = (today - parse_date(s['last_active_date'])).days
    if s['notice_period_days'] > 90:
        parts.append(f"{s['notice_period_days']}-day notice is the main friction")
    if not s['open_to_work_flag']:
        parts.append("not currently open to work — passive candidate, needs warm approach")
    elif s['notice_period_days'] <= 30:
        parts.append(f"open and can join in {s['notice_period_days']} days")
    if best:
        parts.append(f"platform score: {best[0]} {best[1]:.0f}/100")
    if days_inactive > 90:
        parts.append(f"inactive for {days_inactive}d")

    sentence2 = ('; '.join(parts) + '.').capitalize() if parts else ''
    return (sentence1 + ' ' + sentence2).strip()


def _reason_D(candidate: dict, companies: dict) -> str:
    """Deep specialist — lead with verified depth on specific rare skills."""
    p = candidate['profile']
    s = candidate['redrob_signals']
    today = date.today()

    core = _get_core_skills(candidate)
    best = _best_assessment(candidate)

    # Find the deepest skill
    deepest = core[0] if core else None
    assessments = s.get('skill_assessment_scores', {})

    if deepest and best:
        sentence1 = (
            f"{deepest['duration_months']}mo of {deepest['name']} experience "
            f"(verified at {best[1]:.0f}/100 on platform assessment) — "
            f"this depth on {deepest['name']} is rare in the candidate pool; "
            f"currently {p['current_title']} at {p['current_company']}."
        )
    elif deepest:
        sentence1 = (
            f"{deepest['duration_months']}mo working with {deepest['name']} "
            f"as a {p['current_title']} at {p['current_company']}; "
            f"that duration on a single core retrieval technology is above average."
        )
    else:
        sentence1 = (
            f"{p['years_of_experience']}yr {p['current_title']} at {p['current_company']}; "
            f"strong technical profile across the IR/ML stack."
        )

    # Second sentence
    days_inactive = (today - parse_date(s['last_active_date'])).days
    parts = []
    if s['open_to_work_flag']:
        parts.append(f"open to work, {s['notice_period_days']}-day notice")
    else:
        parts.append(f"not marked open-to-work")
    if days_inactive > 60:
        parts.append(f"last active {days_inactive}d ago")
    g = s['github_activity_score']
    if g >= 50:
        parts.append(f"GitHub {g:.0f}/100 — active external contribution")
    rr = s['recruiter_response_rate']
    if rr >= 0.75:
        parts.append(f"responds to {rr:.0%} of recruiter messages")
    elif rr < 0.25:
        parts.append(f"low recruiter response rate ({rr:.0%}) — may be hard to reach")

    sentence2 = ('; '.join(parts) + '.').capitalize() if parts else ''
    return (sentence1 + ' ' + sentence2).strip()


def _reason_E(candidate: dict, companies: dict) -> str:
    """Strong generalist — breadth across IR stack is the story."""
    p = candidate['profile']
    s = candidate['redrob_signals']
    career = candidate.get('career_history', [])
    today = date.today()

    core = _get_core_skills(candidate)
    skill_names = [sk['name'] for sk in core[:4]]
    skills_str = ', '.join(skill_names)

    # What makes their breadth interesting?
    has_retrieval = any(
        sk['name'].lower() in {
            'faiss', 'pinecone', 'weaviate', 'qdrant', 'milvus',
            'elasticsearch', 'opensearch', 'vector search', 'bm25'
        }
        for sk in core
    )
    has_embedding = any(
        sk['name'].lower() in {
            'sentence transformers', 'embeddings', 'hugging face transformers'
        }
        for sk in core
    )
    has_ranking = any(
        sk['name'].lower() in {'learning to rank', 'recommendation systems'}
        for sk in core
    )

    if has_retrieval and has_embedding and has_ranking:
        stack_note = "covers retrieval infrastructure, embedding models, and ranking — the full stack this role needs"
    elif has_retrieval and has_embedding:
        stack_note = "covers both retrieval infrastructure and embedding models"
    elif has_retrieval and has_ranking:
        stack_note = "covers retrieval and ranking — two of three core pillars"
    else:
        stack_note = "breadth across search and ML tooling"

    sentence1 = (
        f"{p['current_title']} at {p['current_company']} with {p['years_of_experience']}yr; "
        f"skill set ({skills_str}) {stack_note}."
    )

    days_inactive = (today - parse_date(s['last_active_date'])).days
    best = _best_assessment(candidate)
    parts = []
    if s['notice_period_days'] > 90:
        parts.append(f"{s['notice_period_days']}-day notice is a concern")
    elif s['open_to_work_flag']:
        parts.append(f"actively looking, {s['notice_period_days']}-day notice")
    if best:
        parts.append(f"{best[0]}: {best[1]:.0f}/100 on platform")
    if days_inactive > 90:
        parts.append(f"inactive {days_inactive}d")
    if p['country'] != 'India':
        relocate = 'willing' if s['willing_to_relocate'] else 'not willing'
        parts.append(f"based in {p['country']}, {relocate} to relocate")

    sentence2 = ('; '.join(parts) + '.').capitalize() if parts else ''
    return (sentence1 + ' ' + sentence2).strip()


def _reason_F(candidate: dict, companies: dict) -> str:
    """Fast-mover — availability is the differentiator, lead with it."""
    p = candidate['profile']
    s = candidate['redrob_signals']
    today = date.today()
    days_inactive = (today - parse_date(s['last_active_date'])).days

    core = _get_core_skills(candidate)

    if s['notice_period_days'] == 0:
        avail = "immediately joinable (zero notice period)"
    elif s['notice_period_days'] <= 15:
        avail = f"can join in {s['notice_period_days']} days"
    else:
        avail = f"{s['notice_period_days']}-day notice, actively looking"

    if days_inactive <= 7:
        activity = "active on platform this week"
    elif days_inactive <= 14:
        activity = f"active {days_inactive}d ago"
    else:
        activity = f"last seen {days_inactive}d ago"

    if core:
        skill_note = (
            f"brings {core[0]['duration_months']}mo of {core[0]['name']} "
            + (f"and {core[1]['duration_months']}mo of {core[1]['name']}" if len(core) > 1 else '')
        )
    else:
        skill_note = f"{p['years_of_experience']}yr in ML/AI roles"

    sentence1 = (
        f"{p['current_title']} at {p['current_company']} — {avail}, {activity}; "
        f"{skill_note}."
    )

    # Honest technical concern
    concerns = []
    if len(core) < 2:
        concerns.append("core IR skill depth is thinner than top candidates")
    best = _best_assessment(candidate)
    if not best:
        concerns.append("no platform assessments completed on core skills")
    else:
        concerns.append(f"{best[0]}: {best[1]:.0f}/100")
    g = s['github_activity_score']
    if g > 50:
        concerns.append(f"GitHub active ({g:.0f}/100)")

    sentence2 = ('; '.join(concerns) + '.').capitalize() if concerns else ''
    return (sentence1 + ' ' + sentence2).strip()


def _reason_G(candidate: dict, companies: dict) -> str:
    """Borderline — honest about why they made it and what's missing."""
    p = candidate['profile']
    s = candidate['redrob_signals']
    today = date.today()

    core = _get_core_skills(candidate)
    days_inactive = (today - parse_date(s['last_active_date'])).days

    if core:
        skill_note = (
            f"{core[0]['name']} ({core[0]['duration_months']}mo)"
            + (f", {core[1]['name']} ({core[1]['duration_months']}mo)" if len(core) > 1 else '')
        )
        sentence1 = (
            f"{p['current_title']} at {p['current_company']} "
            f"with {p['years_of_experience']}yr; "
            f"surfaced on text match with {skill_note} but overall profile "
            f"is thinner than candidates ranked higher."
        )
    else:
        sentence1 = (
            f"{p['current_title']} at {p['current_company']}; "
            f"retrieved on semantic similarity but lacks advanced core IR skills — "
            f"borderline inclusion, recommend manual review."
        )

    parts = []
    if s['open_to_work_flag'] and s['notice_period_days'] <= 30:
        parts.append(f"available quickly ({s['notice_period_days']}-day notice)")
    if days_inactive > 90:
        parts.append(f"inactive {days_inactive}d — engagement risk")
    rr = s['recruiter_response_rate']
    if rr < 0.2:
        parts.append(f"low recruiter response ({rr:.0%})")
    if p['country'] != 'India' and not s['willing_to_relocate']:
        parts.append(f"outside India, not relocating")

    sentence2 = ('; '.join(parts) + '.').capitalize() if parts else ''
    return (sentence1 + ' ' + sentence2).strip()


def generate_reasoning(candidate: dict) -> str:
    """
    Entry point. Assigns archetype, calls the right reasoning function.
    Each archetype produces structurally different sentences.
    """
    career = candidate.get('career_history', [])
    companies = _classify_companies(career)
    archetype = _archetype(candidate, companies)

    dispatch = {
        'A': _reason_A,
        'B': _reason_B,
        'C': _reason_C,
        'D': _reason_D,
        'E': _reason_E,
        'F': _reason_F,
        'G': _reason_G,
    }

    return dispatch[archetype](candidate, companies)