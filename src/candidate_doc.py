CORE_SKILLS = {
    'faiss', 'pinecone', 'weaviate', 'qdrant', 'milvus',
    'elasticsearch', 'opensearch', 'embeddings', 'sentence transformers',
    'sentence-transformers', 'information retrieval', 'nlp', 'ranking',
    'recommendation systems', 'vector search', 'dense retrieval',
    'sparse retrieval', 'bm25', 'hybrid search', 'rag',
    'retrieval augmented generation', 'llm', 'transformers',
    'fine-tuning', 'lora', 'qlora', 'peft', 'ndcg', 'learning to rank'
}

PRODUCT_INDUSTRIES = {
    'software', 'ai/ml', 'fintech', 'e-commerce', 'saas',
    'food delivery', 'transportation', 'edtech', 'healthtech'
}


def build_candidate_doc(candidate):
    profile = candidate['profile']
    career = candidate['career_history']
    skills = candidate['skills']
    education = candidate['education']
    assessments = candidate['redrob_signals']['skill_assessment_scores']

    lines = []

    # Headline and summary
    lines.append(profile['headline'])
    lines.append(profile['summary'])
    lines.append("")

    # Career history - most important section
    lines.append("Experience:")
    for job in career:
        company_type = (
            "product company" if job['industry'].lower() in PRODUCT_INDUSTRIES
            else "services company"
        )
        line = (
            f"- {job['title']} at {job['company']} "
            f"({job['industry']}, {job['company_size']} employees, {company_type}, "
            f"{job['duration_months']} months): {job['description']}"
        )
        lines.append(line)
    lines.append("")

    # Only advanced/expert skills with duration
    strong_skills = [
        s for s in skills
        if s['proficiency'] in ('advanced', 'expert')
    ]
    if strong_skills:
        skill_parts = [
            f"{s['name']} ({s.get('duration_months', 0)}mo, {s['proficiency']})"
            for s in strong_skills
        ]
        lines.append(f"Core Skills: {', '.join(skill_parts)}")
        lines.append("")

    # Assessment scores for core skills only
    relevant_assessments = {
        k: v for k, v in assessments.items()
        if k.lower() in CORE_SKILLS
    }
    if relevant_assessments:
        assessment_parts = [
            f"{k}: {v:.0f}/100"
            for k, v in relevant_assessments.items()
        ]
        lines.append(f"Verified Assessments: {', '.join(assessment_parts)}")
        lines.append("")

    # Education
    if education:
        edu = education[0]
        tier = edu.get('tier', 'unknown')
        lines.append(
            f"Education: {edu['degree']} in {edu['field_of_study']} "
            f"from {edu['institution']} ({tier})"
        )

    return "\n".join(lines)