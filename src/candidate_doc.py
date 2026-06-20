"""
candidate_doc.py — Builds the text document per candidate for BM25 + embedding.

Key insights from 100K data:

SKILL PROFICIENCY REALITY (from proficiency crosstab):
- Generic skills (html, css, java, etc): almost all beginner/intermediate, 
  advanced/expert counts in single digits → these are noise, IGNORE them
- Core AI skills (embeddings, faiss, qdrant, etc): split ~60% advanced / 35% intermediate / 5% expert
  → advanced and expert are meaningful signals here
- CV/generic ML skills (yolo, cnn, opencv, etc): split ~50/50 intermediate/advanced
  → these are mid-tier signals, not strong JD fit

ENDORSEMENT REALITY (reveals who really has rare skills):
- Core search/retrieval skills: avg endorsements 19-20 (opensearch: 20.47, scikit-learn: 20.13)
- Generic skills: avg endorsements 7.4-7.6 (html, css, java, etc)
- LLM-adjacent skills: avg endorsements 6.2-6.7 (langchain, rag, llms, pinecone)
  → LLM skills have LOW endorsements = many people claim them, few are validated

ASSESSMENT SCORES (reveals discrimination power):
- llamaindex: 56.50, opensearch: 56.15, haystack: 56.06, weaviate: 56.06
- These HIGH-scoring assessments are the rare validated specialists
- Mean assessment scores range 51-56 so anything >60 is above average

TITLE DISTRIBUTION tells us what to look for:
- Only 7 people have title "Senior AI Engineer" (exact JD title)
- 57 "Recommendation Systems Engineer", 58 "Search Engineer"  
- 68 "Machine Learning Engineer", 51 "Applied ML Engineer"
- These ~200 people are GOLD — retrieval must surface them

COMPANY INSIGHTS:
- Swiggy(3019), Razorpay(2926), CRED(2908), Zomato(2883), Flipkart(2882) = product companies
- Sarvam AI, Rephrase.ai, Mad Street Den, Observe.AI, Krutrim = AI-native companies (tiny but gold)
- Google(14), Netflix(13), Amazon(13), Meta(11) = FAANG — very rare, weight heavily
"""

# Skills that are STRONG signals for this JD
# Based on: JD requirements + assessment score discrimination + endorsement patterns
TIER1_SKILLS = {
    # Vector search / retrieval — core JD requirement
    'faiss', 'pinecone', 'weaviate', 'qdrant', 'milvus', 'elasticsearch',
    'opensearch', 'pgvector', 'vector search', 'semantic search',
    'information retrieval', 'hybrid search', 'dense retrieval',
    'sparse retrieval', 'bm25',
    # Embedding models
    'sentence transformers', 'sentence-transformers', 'embeddings',
    'hugging face transformers', 'text encoders',
    # Ranking
    'learning to rank', 'recommendation systems', 'search engineer',
    'ranking systems', 'information retrieval systems',
    # Fine-tuning
    'lora', 'qlora', 'peft', 'fine-tuning llms',
    # Frameworks
    'haystack', 'llamaindex',
    # Core ML
    'pytorch', 'tensorflow', 'scikit-learn', 'nlp', 'machine learning',
    'deep learning', 'python', 'rag',
}

# Skills that are MEDIUM signals — relevant but not core differentiators
TIER2_SKILLS = {
    'mlops', 'mlflow', 'kubeflow', 'prompt engineering', 'langchain',
    'llms', 'reinforcement learning', 'statistical modeling',
    'feature engineering', 'data science',
}

# Skills that are NOISE for this JD — ignore completely in doc
TIER_NOISE = {
    'html', 'css', 'javascript', 'typescript', 'react', 'angular', 'vue.js',
    'next.js', 'redux', 'webpack', 'tailwind', 'figma', 'photoshop',
    'illustrator', 'powerpoint', 'excel', 'accounting', 'sales', 'marketing',
    'seo', 'tally', 'six sigma', 'scrum', 'agile', 'project management',
    'content writing', 'sap', 'salesforce crm', 'java', 'go', 'rust',
    'spring boot', 'django', 'flask', 'fastapi', 'node.js',
    'apache beam', 'apache flink', 'hadoop', 'spark', 'kafka',
    'kubernetes', 'docker', 'terraform', 'ci/cd', 'aws', 'gcp', 'azure',
    'mongodb', 'postgresql', 'redis', 'snowflake', 'bigquery', 'dbt',
    'databricks', 'airflow', 'graphql', 'grpc', 'rest apis', 'microservices',
    'etl', 'data pipelines', 'sql',
    # CV skills — wrong domain for this JD
    'yolo', 'opencv', 'cnn', 'gans', 'diffusion models', 'object detection',
    'image classification', 'computer vision', 'asr', 'speech recognition',
    'tts', 'bentoml', 'time series', 'forecasting', 'weights & biases',
}

# Product companies from actual dataset — stronger signal than industry alone
PRODUCT_COMPANIES = {
    # Indian unicorns/startups — high value for this JD
    'swiggy', 'razorpay', 'cred', 'zomato', 'flipkart', 'meesho', 'nykaa',
    'inmobi', 'byjus', 'policybazaar', 'ola', 'zoho', 'vedantu', 'paytm',
    'unacademy', 'pharmeasy', 'upgard', 'freshworks', 'phonepe', 'dream11',
    # AI-native companies — gold tier
    'sarvam ai', 'rephrase.ai', 'aganitha', 'niramai', 'saarthi.ai',
    'mad street den', 'observe.ai', 'krutrim', 'wysa', 'haptik',
    'verloop.io', 'yellow.ai', 'locobuzz', 'glance', 'genpact ai',
    # FAANG/big tech — very rare in dataset, highest signal
    'google', 'meta', 'amazon', 'microsoft', 'netflix', 'apple',
    'adobe', 'salesforce', 'linkedin', 'uber',
}

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

# Relevant certifications from dataset (only 658 total across all candidates)
RELEVANT_CERTS = {
    'aws certified machine learning specialty',
    'deep learning specialization',
    'google cloud professional ml engineer',
    'nlp specialization',
    'langchain for llm application development',
}


def _classify_company(company_name: str, industry: str) -> str:
    """Returns 'faang', 'ai_native', 'product', 'consulting', or 'other'"""
    company_lower = company_name.lower()

    if any(f in company_lower for f in ['google', 'meta', 'amazon', 'microsoft',
                                          'netflix', 'apple', 'adobe', 'salesforce',
                                          'linkedin', 'uber']):
        return 'faang'

    if any(p in company_lower for p in [
        'sarvam', 'rephrase', 'aganitha', 'niramai', 'saarthi', 'mad street den',
        'observe.ai', 'krutrim', 'wysa', 'haptik', 'verloop', 'yellow.ai',
        'locobuzz', 'glance'
    ]):
        return 'ai_native'

    if any(c in company_lower for c in CONSULTING_FIRMS):
        return 'consulting'

    if industry.lower() in PRODUCT_INDUSTRIES:
        return 'product'

    return 'other'


def build_candidate_doc(candidate: dict) -> str:
    """
    Build a structured text document for BM25 + embedding retrieval.
    
    Document structure prioritizes:
    1. Headline (concentrated signal)
    2. Summary (candidate's own words about what they do)
    3. Career history with company classification (product vs consulting)
    4. Tier-1 skills only with duration (endorsements stripped — noise)
    5. Verified assessments on relevant skills only
    6. Relevant certifications only
    """
    profile = candidate['profile']
    career = candidate.get('career_history', [])
    skills = candidate.get('skills', [])
    education = candidate.get('education', [])
    certifications = candidate.get('certifications', [])
    assessments = candidate['redrob_signals'].get('skill_assessment_scores', {})

    lines = []

    # --- Section 1: Headline ---
    lines.append(profile['headline'])
    lines.append("")

    # --- Section 2: Summary ---
    lines.append(profile['summary'])
    lines.append("")

    # --- Section 3: Career History ---
    # Include company type classification so BM25 can match "product company"
    lines.append("Work Experience:")
    for job in career:
        company_type = _classify_company(job['company'], job['industry'])

        # Build a rich description line
        type_label = {
            'faang': 'FAANG/big tech product company',
            'ai_native': 'AI-native product company',
            'product': 'product company',
            'consulting': 'IT services / consulting firm',
            'other': 'company'
        }[company_type]

        line = (
            f"- {job['title']} at {job['company']} "
            f"({type_label}, {job['industry']}, {job['duration_months']} months): "
            f"{job['description']}"
        )
        lines.append(line)
    lines.append("")

    # --- Section 4: Relevant Skills ---
    # Only tier1 skills, advanced/expert proficiency, with duration
    # Duration is KEY: "Pinecone (88mo, expert)" >> "Pinecone (2mo, advanced)"
    tier1_skills = [
        s for s in skills
        if s['name'].lower() in TIER1_SKILLS
        and s['proficiency'] in ('advanced', 'expert')
        and s.get('duration_months', 0) > 0
    ]

    if tier1_skills:
        # Sort by duration descending — longest experience first
        tier1_skills.sort(key=lambda x: x.get('duration_months', 0), reverse=True)
        skill_parts = [
            f"{s['name']} ({s.get('duration_months', 0)}mo, {s['proficiency']})"
            for s in tier1_skills
        ]
        lines.append(f"Core Skills: {', '.join(skill_parts)}")
        lines.append("")

    # Also include tier2 skills but without proficiency/duration detail
    tier2_skills = [
        s['name'] for s in skills
        if s['name'].lower() in TIER2_SKILLS
        and s['proficiency'] in ('advanced', 'expert')
    ]
    if tier2_skills:
        lines.append(f"Supporting Skills: {', '.join(tier2_skills)}")
        lines.append("")

    # --- Section 5: Platform Assessment Scores ---
    # Only include assessments for tier1 skills — others are noise
    # Assessment mean scores range 51-56, so >60 = above average, >70 = strong
    relevant_assessments = {
        k: v for k, v in assessments.items()
        if k.lower() in TIER1_SKILLS and v >= 50
    }
    if relevant_assessments:
        sorted_assessments = sorted(
            relevant_assessments.items(), key=lambda x: x[1], reverse=True
        )
        parts = [f"{k} {v:.0f}/100" for k, v in sorted_assessments]
        lines.append(f"Platform-Verified Skills: {', '.join(parts)}")
        lines.append("")

    # --- Section 6: Relevant Certifications ---
    relevant_certs = [
        c for c in certifications
        if c['name'].lower() in RELEVANT_CERTS
    ]
    if relevant_certs:
        cert_parts = [f"{c['name']} ({c['issuer']}, {c['year']})" for c in relevant_certs]
        lines.append(f"Certifications: {', '.join(cert_parts)}")
        lines.append("")

    # --- Section 7: Education ---
    if education:
        edu = education[0]
        tier = edu.get('tier', 'unknown')
        # Only mention tier_1 and tier_2 — tier_3/4 are majority (53K+51K) so not a differentiator
        edu_str = f"Education: {edu['degree']} in {edu['field_of_study']} from {edu['institution']}"
        if tier in ('tier_1', 'tier_2'):
            edu_str += f" ({tier} institution)"
        lines.append(edu_str)

    return "\n".join(lines)