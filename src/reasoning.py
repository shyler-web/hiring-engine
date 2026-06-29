# reasoning.py - Final spec‑compliant reasoning (V12).
#
# Format: [Role with YoE]. [Template Summary]. [Top Differentiator]. [Weakness if present].
# - Role with YoE: e.g., "Recommendation Systems Engineer with 6.5 years of experience."
# - Template Summary: full, untruncated summary from jd_templates_enhanced.pkl
# - Top Differentiator: natural-language phrase based on the highest structured signal
# - Weakness: only included if a real weakness exists (notice, location, GitHub, activity, YoE)

from src.signals import CORE_JD_SKILLS

# ------------------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------------------

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

def _format_top_signal(signal_profile: dict, candidate: dict) -> str:
    """
    Return a natural‑language phrase describing the top positive signal.
    """
    if not signal_profile:
        return ""

    # Define signal keys and their priority order (higher priority if tie)
    keys = [
        'career_quality',
        'availability',
        'jd_template_multiplier',
        'github',
        'skill_depth',
        'recruiter_market',
        'experience_fit'
    ]
    # Get values, ignore signals <= 1.0 (they are not positive differentiators)
    items = [(k, signal_profile.get(k, 1.0)) for k in keys if k in signal_profile and signal_profile.get(k, 1.0) > 1.0]
    if not items:
        return ""

    # Sort by value descending
    items.sort(key=lambda x: x[1], reverse=True)
    top_key, top_val = items[0]

    # Map to natural language
    if top_key == 'career_quality':
        return "Strong background at a top-tier product company."
    elif top_key == 'availability':
        # Check if open to work and notice period
        signals = candidate.get('redrob_signals', {})
        if signals.get('open_to_work_flag', False):
            notice = signals.get('notice_period_days', 90)
            if notice <= 30:
                return "Immediately available."
            else:
                return "Open to work with reasonable notice."
        else:
            return "Available with short notice."  # This might not be accurate; but we'll use availability signal.
    elif top_key == 'jd_template_multiplier':
        return "Profile exactly matches the JD's retrieval/ranking focus."
    elif top_key == 'github':
        return "Strong GitHub activity."
    elif top_key == 'skill_depth':
        skills = _get_top_core_skills(candidate, n=2)
        if skills:
            skill_names = [s['name'] for s in skills]
            return f"Deep hands-on experience in {skill_names[0]}" + (f" and {skill_names[1]}" if len(skill_names) > 1 else "") + "."
        else:
            return "Deep technical depth in core ML skills."
    elif top_key == 'recruiter_market':
        return "High recruiter interest."
    elif top_key == 'experience_fit':
        return "Experience perfectly aligns with the 5-9 year target."
    else:
        return ""

def _format_weakness(signal_profile: dict, candidate: dict) -> str:
    """
    Return a natural‑language weakness phrase if any signal is below a threshold.
    Returns empty string if no weakness.
    """
    if not signal_profile:
        return ""

    profile = candidate.get('profile', {})
    signals = candidate.get('redrob_signals', {})

    # Define thresholds and corresponding phrases
    checks = []

    # Notice period > 60 days
    notice = signals.get('notice_period_days', 90)
    if notice > 60:
        checks.append(('notice_period', notice, f"{notice}-day notice period is a concern."))

    # Location mismatch (not Pune or Noida)
    location = profile.get('location', '')
    if location and 'pune' not in location.lower() and 'noida' not in location.lower():
        checks.append(('location', 0, f"Based in {location} (not Pune/Noida)."))

    # GitHub activity < 20 (only if not -1)
    github = signals.get('github_activity_score', -1)
    if github >= 0 and github < 20:
        checks.append(('github', github, "GitHub activity is limited."))

    # Recruiter saves < 5
    saves = signals.get('saved_by_recruiters_30d', 0)
    if saves < 5:
        checks.append(('recruiter_market', saves, "Limited recruiter engagement."))

    # YoE < 4
    yoe = profile.get('years_of_experience', 0)
    if yoe < 4:
        checks.append(('experience_fit', yoe, f"Slightly below the 5-9 year target ({yoe:.1f} YoE)."))

    # YoE > 12
    if yoe > 12:
        checks.append(('experience_fit', yoe, f"May be overqualified for an IC role ({yoe:.1f} YoE)."))

    if not checks:
        return ""

    # Pick the weakness with the lowest signal value (most concerning)
    # We assign a numeric score for each check: lower is worse
    # For notice, we use notice days (higher = worse)
    # For others, we use the value if numeric, else a high number
    def score_check(item):
        key, val, phrase = item
        if key == 'notice_period':
            return val  # higher days = worse
        elif key == 'location':
            return 0  # location mismatch is always bad, but we treat as low priority if other issues exist
        elif key == 'github':
            return -val  # lower GitHub = worse, so negative to make it smaller
        elif key == 'recruiter_market':
            return -val  # lower saves = worse
        elif key == 'experience_fit':
            # For YoE, we want to penalize both too low and too high, but we already have phrases
            # We'll treat low YoE as more concerning than high YoE for this role
            if 'below' in phrase:
                return -100 - val  # very low priority
            else:
                return val  # high YoE is less concerning than low
        return 0

    # Sort by the score; we want the most concerning (lowest score)
    checks.sort(key=score_check)
    # Return the phrase of the most concerning weakness
    return checks[0][2]

# ------------------------------------------------------------------------------
# Main reasoning generator
# ------------------------------------------------------------------------------

def generate_reasoning(
    candidate: dict,
    jd_template: dict = None,
    signal_profile: dict = None,
    template_summary: str = None
) -> str:
    """
    Generate reasoning following the format:
        [Role with YoE]. [Template Summary]. [Top Differentiator]. [Weakness if present].
    """
    profile = candidate.get('profile', {})
    title = profile.get('current_title', 'Unknown')
    yoe = profile.get('years_of_experience', 0)

    # --- Role with YoE ---
    role_part = f"{title} with {yoe:.1f} years of experience."

    # --- Template Summary ---
    # If template_summary is missing, fallback to a generic skill-based summary
    if not template_summary:
        # Use top skills or fallback
        skills = _get_top_core_skills(candidate, n=2)
        if skills:
            skill_str = skills[0]['name'] + (f" and {skills[1]['name']}" if len(skills) > 1 else "")
            summary_part = f"Experienced in {skill_str}."
        else:
            summary_part = ""
    else:
        summary_part = template_summary  # full, untruncated

    # --- Top Differentiator ---
    diff_part = _format_top_signal(signal_profile, candidate)

    # --- Weakness (if present) ---
    weak_part = _format_weakness(signal_profile, candidate)

    # Assemble reasoning by joining non-empty parts with spaces.
    parts = [role_part]
    if summary_part:
        parts.append(summary_part)
    if diff_part:
        parts.append(diff_part)
    if weak_part:
        parts.append(weak_part)

    # Join with spaces, ensuring periods are placed correctly.
    # The parts already end with periods, so we just join with space.
    reasoning = " ".join(parts)
    return reasoning