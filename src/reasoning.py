# reasoning.py - Organic, signal‑driven reasoning generation (V6.1).
#
# V6.1: truncate quote at sentence boundary (last '.', '!', '?') before the character limit.

from datetime import date, datetime
from src.signals import CORE_JD_SKILLS

# ------------------------------------------------------------------------------
# Helper: Quote truncation at sentence boundary
# ------------------------------------------------------------------------------

def truncate_at_sentence_boundary(text: str, max_len: int = 120) -> str:
    """Truncate text at the last sentence boundary (., !, ?) before max_len."""
    if len(text) <= max_len:
        return text
    # Examine the substring up to max_len
    sub = text[:max_len]
    # Find the last occurrence of sentence-ending punctuation
    for sep in ('.', '!', '?'):
        last = sub.rfind(sep)
        if last != -1:
            # Include the punctuation character
            return sub[:last + 1]
    # Fallback: hard truncate with ellipsis
    return sub + '...'

# ------------------------------------------------------------------------------
# Other helpers (unchanged from V5/V6)
# ------------------------------------------------------------------------------

def parse_date(date_str: str) -> date:
    return datetime.strptime(date_str, "%Y-%m-%d").date()

def _get_top_core_skills(candidate: dict, n: int = 2) -> list:
    """Return the top n advanced/expert core skills sorted by duration desc."""
    skills = candidate.get('skills', [])
    core = [
        s for s in skills
        if s.get('name', '').lower() in CORE_JD_SKILLS
        and s.get('proficiency') in ('advanced', 'expert')
        and s.get('duration_months', 0) > 0
    ]
    core.sort(key=lambda x: x.get('duration_months', 0), reverse=True)
    return core[:n]

def _get_best_assessment(candidate: dict, threshold: int = 50) -> tuple:
    """Return (skill_name, score) for the highest‑scoring core skill assessment."""
    assessments = candidate.get('redrob_signals', {}).get('skill_assessment_scores', {})
    best_name = None
    best_score = -1
    for name, score in assessments.items():
        if score > threshold and name.lower() in CORE_JD_SKILLS:
            if score > best_score:
                best_score = score
                best_name = name
    return best_name, best_score

def _format_signal_name(key: str) -> str:
    """Consistent signal name mapping."""
    mapping = {
        'experience_fit': 'experience fit (YoE alignment)',
        'availability': 'availability/open‑to‑work',
        'notice_period': 'short notice period',
        'location': 'location alignment',
        'github': 'GitHub activity',
        'career_quality': 'career quality (company pedigree)',
        'skill_depth': 'core IR skill depth',
        'recruiter_market': 'recruiter market validation',
        'jd_template_multiplier': 'JD template tier',
    }
    return mapping.get(key, key.replace('_', ' '))

def _get_archetype_summary(candidate: dict, jd_template: dict) -> str:
    """Generate a short summarized archetype for ranks > 30."""
    tier = jd_template.get('tier', '') if jd_template else ''
    top_skills = _get_top_core_skills(candidate, n=2)
    if tier == 'golden':
        return "builds retrieval/ranking pipelines for product search"
    elif tier == 'ml_adj':
        return "builds ML production pipelines and adjacent systems"
    elif tier == 'data_eng':
        return "builds data infrastructure that supports ML systems"
    else:
        if top_skills:
            return f"works with {top_skills[0]['name']} in production ML contexts"
        else:
            return "has a technical background in software/ML"

# ------------------------------------------------------------------------------
# Main entry point (V6.1 — 2-sentence output)
# ------------------------------------------------------------------------------

def generate_reasoning(
    candidate: dict,
    jd_template: dict = None,
    signal_profile: dict = None,
    semantic_quote: str = None,
    semantic_score: float = None,
    rank: int = None
) -> str:
    """
    Generate a 2-sentence reasoning string for a candidate.

    Sentence 1: Identity + top skill/achievement anchor + key signal differentiator.
    Sentence 2: Combined multiplier summary + one friction or availability note.
    """
    profile = candidate.get('profile', {})
    signals = candidate.get('redrob_signals', {})

    # --- Extract context ---
    title = profile.get('current_title', 'Unknown')
    company = profile.get('current_company', 'Unknown')
    yoe = profile.get('years_of_experience', 0)
    location = profile.get('location', 'Unknown')
    country = profile.get('country', 'Unknown')
    top_skills = _get_top_core_skills(candidate, n=2)
    best_assessment_name, best_assessment_score = _get_best_assessment(candidate, threshold=50)

    # ----------------------------------------------------------------
    # SENTENCE 1: Who they are + what they bring + top differentiator
    # ----------------------------------------------------------------
    base = f"{yoe:.1f}yr {title} at {company}"

    # Skill anchor (short)
    if top_skills:
        skill_anchor = f"{top_skills[0]['name']} ({top_skills[0]['duration_months']}mo)"
        if len(top_skills) > 1:
            skill_anchor += f" and {top_skills[1]['name']} ({top_skills[1]['duration_months']}mo)"
    elif best_assessment_name and best_assessment_score > 0:
        skill_anchor = f"{best_assessment_name} assessment {best_assessment_score:.0f}/100"
    else:
        skill_anchor = "adjacent ML skills"

    # Top signal differentiator
    top_signal_str = ""
    if signal_profile:
        multiplier_keys = [
            'experience_fit', 'availability', 'notice_period', 'location',
            'github', 'career_quality', 'skill_depth', 'recruiter_market',
            'jd_template_multiplier'
        ]
        items = [(k, signal_profile.get(k, 1.0)) for k in multiplier_keys if k in signal_profile]
        items.sort(key=lambda x: x[1], reverse=True)
        if items:
            top_key, top_val = items[0]
            if top_key == 'career_quality':
                top_signal_str = f"{company} pedigree ({top_val:.2f}x)"
            elif top_key == 'availability':
                top_signal_str = f"immediate availability ({top_val:.2f}x)"
            elif top_key == 'jd_template_multiplier':
                top_signal_str = f"strong JD template match ({top_val:.2f}x)"
            else:
                top_signal_str = f"{_format_signal_name(top_key)} ({top_val:.2f}x)"

    # For ranks > 30 use archetype instead of quote
    if rank is not None and rank > 30:
        archetype = _get_archetype_summary(candidate, jd_template)
        s1 = f"{base}; {archetype} with strong {skill_anchor}."
    else:
        # Quote fragment – now truncated at sentence boundary
        if semantic_quote and len(semantic_quote) > 10:
            quote_clip = truncate_at_sentence_boundary(semantic_quote.strip(), max_len=120)
            s1 = f"{base}; \"{quote_clip}\" — deep {skill_anchor}."
        else:
            s1 = f"{base} with deep {skill_anchor}."

    if top_signal_str:
        s1 = s1.rstrip('.') + f"; key signal: {top_signal_str}."

    # ----------------------------------------------------------------
    # SENTENCE 2: Multiplier summary + friction or availability
    # ----------------------------------------------------------------
    s2_parts = []

    # Combined multiplier
    if signal_profile:
        combined = signal_profile.get('combined_multiplier', None)
        if combined:
            # Re‑use the already sorted 'items' list
            top3_names = [_format_signal_name(k) for k, _ in items[:3]]
            s2_parts.append(f"Composite {combined:.2f}x driven by {', '.join(top3_names)}.")

    # Friction or availability (one note only)
    notice = signals.get('notice_period_days', 90)
    open_flag = signals.get('open_to_work_flag', False)
    willing_relocate = signals.get('willing_to_relocate', False)

    if notice > 60:
        s2_parts.append(f"Notice: {notice} days (above JD ≤30 preference).")
    elif country != 'India' and not willing_relocate:
        s2_parts.append(f"Based in {country}, not willing to relocate.")
    elif 'pune' not in location.lower() and 'noida' not in location.lower() and location not in ('Unknown', ''):
        s2_parts.append(f"Location: {location} (not Pune/Noida).")
    elif open_flag and notice <= 30:
        s2_parts.append(f"Open to work, {notice}-day notice.")

    s2 = ' '.join(s2_parts) if s2_parts else "No significant friction."

    return f"{s1} {s2}".replace('  ', ' ').strip()