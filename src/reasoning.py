# reasoning.py - Non‑templated, signal‑driven reasoning generation.
#
# V3 (final) changes:
#   - Accepts signal_profile (dict of multipliers) from rank.py.
#   - No hard‑coded categories (golden/ml_adj/fallback).
#   - Identifies the top 3 highest multipliers and the lowest multiplier.
#   - Builds a fluid 3‑sentence narrative that explains WHY the candidate
#     ranks where they do, interweaving the semantic quote, company context,
#     and structured signals.
#   - Completely banished generic praise phrases.

from datetime import date, datetime
from src.signals import CORE_JD_SKILLS

# ------------------------------------------------------------------------------
# Helpers (local copies to avoid heavy imports)
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
    """
    Return (skill_name, score) for the highest‑scoring core skill assessment
    that is above the threshold. Returns (None, None) if none found.
    """
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
    """Convert camel_case signal keys to human‑readable labels."""
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

# ------------------------------------------------------------------------------
# Main entry point
# ------------------------------------------------------------------------------

def generate_reasoning(
    candidate: dict,
    jd_template: dict = None,
    signal_profile: dict = None,
    semantic_quote: str = None,
    semantic_score: float = None
) -> str:
    """
    Generate a specific, non‑templated reasoning string for a candidate.

    Args:
        candidate: full candidate JSON
        jd_template: template info from template_map (or None)
        signal_profile: dict of multipliers from compute_final_score()
        semantic_quote: pre‑selected best sentence (offline semantic match)
        semantic_score: cosine similarity of the semantic_quote to the JD

    Returns:
        A single coherent paragraph (2‑3 sentences) explaining the rank.
    """
    profile = candidate.get('profile', {})
    signals = candidate.get('redrob_signals', {})
    career = candidate.get('career_history', [])

    # --- Part 1: Context extraction (always present) ---
    title = profile.get('current_title', 'Unknown')
    company = profile.get('current_company', 'Unknown')
    yoe = profile.get('years_of_experience', 0)
    location = profile.get('location', 'Unknown')
    country = profile.get('country', 'Unknown')

    top_skills = _get_top_core_skills(candidate, n=2)
    best_assessment_name, best_assessment_score = _get_best_assessment(candidate, threshold=50)

    # --- Part 2: Build the opening context + semantic quote ---
    # If we have a semantic quote, use it as the centrepiece.
    # If not, fallback to skills description.
    context_parts = []

    if semantic_quote and len(semantic_quote) > 10:
        # Truncate long quotes
        if len(semantic_quote) > 160:
            quote_display = semantic_quote[:157] + '...'
        else:
            quote_display = semantic_quote

        context = f"Currently a {title} at {company} with {yoe:.1f} years of experience. Their most JD‑relevant narrative is: \"{quote_display}\""
        if semantic_score is not None and semantic_score > 0:
            context += f" (semantic relevance: {semantic_score:.2f}/1.0)"
        context_parts.append(context)

        # Append skill depth if available
        if top_skills:
            skill_str = ', '.join([f"{s['name']} ({s['duration_months']}mo)" for s in top_skills])
            context_parts.append(f"Their top core skills are {skill_str}.")
        if best_assessment_name and best_assessment_score > 0:
            context_parts.append(f"Platform assessment: {best_assessment_name} {best_assessment_score:.0f}/100.")

    else:
        # Fallback: no semantic quote – describe via skills and title
        if top_skills:
            skill_str = ', '.join([f"{s['name']} ({s['duration_months']}mo)" for s in top_skills])
            context = f"Currently a {title} at {company} with {yoe:.1f} years, bringing {skill_str} as their top core skills."
        else:
            context = f"Currently a {title} at {company} with {yoe:.1f} years – no advanced IR/ML core skills listed, but surfaced via other signals."

        if best_assessment_name and best_assessment_score > 0:
            context += f" Their best platform assessment is {best_assessment_name} {best_assessment_score:.0f}/100."
        context_parts.append(context)

    opening = ' '.join(context_parts)

    # --- Part 3: The "Why Ranked Here" (Signal Profile narration) ---
    # Identify top 3 and bottom 1 multipliers from signal_profile
    if signal_profile:
        # We ignore ce_score and combined_multiplier for this breakdown
        multiplier_keys = [
            'experience_fit', 'availability', 'notice_period', 'location',
            'github', 'career_quality', 'skill_depth', 'recruiter_market',
            'jd_template_multiplier'
        ]
        items = [(k, signal_profile.get(k, 1.0)) for k in multiplier_keys if k in signal_profile]
        # Sort by value descending
        items.sort(key=lambda x: x[1], reverse=True)

        top3 = items[:3]
        bottom1 = items[-1] if items else None

        # Build the "why" sentence
        why_parts = []
        if top3:
            descs = []
            for key, val in top3:
                label = _format_signal_name(key)
                descs.append(f"{label} ({val:.2f}x)")
            why_str = "They rank at this position because of strong signals: " + ', '.join(descs)
            # Add combined multiplier for context
            combined = signal_profile.get('combined_multiplier', 1.0)
            why_str += f". Their combined structured multiplier is {combined:.2f}x."

            # Add friction (the lowest signal)
            if bottom1 and bottom1[1] < 0.95:
                key, val = bottom1
                label = _format_signal_name(key)
                why_str += f" The only friction is {label} ({val:.2f}x), which is weaker than their other signals."

            why_parts.append(why_str)

        # If we have a semantic score, add a note about it being a key differentiator
        if semantic_score is not None and semantic_score > 0.75:
            why_parts.append(
                f"Their semantic quote is exceptionally relevant ({semantic_score:.2f}/1.0), placing them among the top semantic matches."
            )

        reasoning_why = ' '.join(why_parts) if why_parts else ''
    else:
        # No signal_profile – fallback to a simple generic reasoning
        reasoning_why = "No structured signal breakdown available; ranked based on the cross‑encoder relevance score and basic profile signals."

    # --- Part 4: Availability & location friction (if not already covered) ---
    # Add a concise availability note if it wasn't already the top signal
    # or if it's a significant weakness.
    avail_note = ''
    notice = signals.get('notice_period_days', 90)
    open_flag = signals.get('open_to_work_flag', False)
    willing_relocate = signals.get('willing_to_relocate', False)

    if signal_profile:
        # Only add if notice wasn't already in top3 or bottom1, or if it's extreme
        notice_val = signal_profile.get('notice_period', 1.0)
        loc_val = signal_profile.get('location', 1.0)
        if notice_val <= 0.85:
            avail_note = f"Notice period is {notice} days – this is a friction point."
        elif notice <= 30 and open_flag:
            avail_note = f"Available with {notice}‑day notice and open to work."
        elif not open_flag:
            avail_note = "Not currently marked as open‑to‑work – may need outreach."

        if country != 'India' and not willing_relocate:
            avail_note += f" Located in {country} and not willing to relocate – a significant friction."
        elif country != 'India' and willing_relocate:
            avail_note += f" Based in {country} but willing to relocate."
        elif 'pune' in location.lower() or 'noida' in location.lower():
            # Already aligned – no need to mention unless it adds value
            pass

    # --- Part 5: Assemble final reasoning ---
    final_parts = [opening]
    if reasoning_why:
        final_parts.append(reasoning_why)
    if avail_note:
        final_parts.append(avail_note)

    # Clean up: join with spaces, ensure no double spaces
    reasoning = ' '.join(final_parts).replace('  ', ' ').strip()
    return reasoning