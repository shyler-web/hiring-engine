# reasoning.py - Final spec‑compliant reasoning (V10).
# Injects candidate-specific top skills + structured signal breakdown.

from src.signals import CORE_JD_SKILLS

# ------------------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------------------

def _get_top_core_skills(candidate: dict, n: int = 2) -> list:
    skills = candidate.get('skills', [])
    core = [
        s for s in skills
        if s.get('name', '').lower() in CORE_JD_SKILLS
        and s.get('proficiency') in ('advanced', 'expert')
        and s.get('duration_months', 0) > 0
    ]
    core.sort(key=lambda x: x.get('duration_months', 0), reverse=True)
    return core[:n]

def _format_signal_breakdown(signal_profile: dict) -> str:
    """Return a concise breakdown of key structured signals."""
    if not signal_profile:
        return ""
    parts = []
    # Map internal keys to human‑readable labels
    mapping = {
        'experience_fit': 'YoE fit',
        'availability': 'availability',
        'notice_period': 'notice',
        'location': 'location',
        'github': 'GitHub',
        'career_quality': 'career quality',
        'skill_depth': 'skill depth',
        'jd_template_multiplier': 'JD template',
    }
    for key, label in mapping.items():
        val = signal_profile.get(key, 1.0)
        parts.append(f"{label} {val:.2f}x")
    return " | ".join(parts)

def _format_signal_name(key: str) -> str:
    """Convert signal key to a human‑readable label."""
    mapping = {
        'experience_fit': 'YoE fit',
        'github': 'GitHub',
        'skill_depth': 'skill depth',
        'recruiter_market': 'recruiter signal',
    }
    return mapping.get(key, key.replace('_', ' '))

# ------------------------------------------------------------------------------
# Main reasoning generator
# ------------------------------------------------------------------------------

def generate_reasoning(
    candidate: dict,
    jd_template: dict = None,
    signal_profile: dict = None,
    template_summary: str = None
) -> str:
    profile = candidate.get('profile', {})
    signals = candidate.get('redrob_signals', {})

    title = profile.get('current_title', 'Unknown')
    yoe = profile.get('years_of_experience', 0)

    # --- 1. Top core skills (candidate‑specific) ---
    top_skills = _get_top_core_skills(candidate, n=2)
    if top_skills:
        skill_str = f"{top_skills[0]['name']} ({top_skills[0]['duration_months']}mo)"
        if len(top_skills) > 1:
            skill_str += f" + {top_skills[1]['name']} ({top_skills[1]['duration_months']}mo)"
    else:
        skill_str = "adjacent ML skills"

    # --- 2. Role + YoE ---
    role_part = f"{title} with {yoe:.1f} years of experience."

    # --- 3. Template Summary (the archetype) ---
    if template_summary:
        summary_part = template_summary
    else:
        summary_part = ""

    # --- 4. Top signal (differentiator) ---
    top_signal_str = ""
    if signal_profile:
        keys = ['career_quality', 'availability', 'jd_template_multiplier', 'github', 'skill_depth', 'recruiter_market', 'experience_fit']
        items = [(k, signal_profile.get(k, 1.0)) for k in keys if k in signal_profile]
        if items:
            items.sort(key=lambda x: x[1], reverse=True)
            top_key, top_val = items[0]
            if top_key == 'career_quality':
                top_signal_str = f"key: company pedigree ({top_val:.2f}x)"
            elif top_key == 'availability':
                top_signal_str = f"key: availability ({top_val:.2f}x)"
            elif top_key == 'jd_template_multiplier':
                top_signal_str = f"key: JD template match ({top_val:.2f}x)"
            else:
                top_signal_str = f"key: {_format_signal_name(top_key)} ({top_val:.2f}x)"

    # --- 5. Structured signal breakdown (candidate‑specific) ---
    signal_breakdown = _format_signal_breakdown(signal_profile)

    # --- 6. Weakness (if present) ---
    weakness = ""
    notice = signals.get('notice_period_days', 90)
    location = profile.get('location', '')
    if notice > 60:
        weakness = f"friction: {notice}-day notice"
    elif 'pune' not in location.lower() and 'noida' not in location.lower() and location:
        weakness = f"friction: based in {location} (not Pune/Noida)"
    elif signal_profile:
        # Check if any signal is < 0.9 (significant penalty)
        for key in ['experience_fit', 'notice_period', 'location', 'github', 'career_quality']:
            if signal_profile.get(key, 1.0) < 0.9:
                weakness = f"friction: weaker {key.replace('_', ' ')} ({signal_profile[key]:.2f}x)"
                break

    # --- 7. Assemble ---
    parts = [role_part]
    if summary_part:
        parts.append(summary_part)
    if skill_str and "adjacent" not in skill_str:
        parts.append(f"top skills: {skill_str}.")
    if top_signal_str:
        parts.append(top_signal_str)
    if signal_breakdown:
        parts.append(f"signals: {signal_breakdown}.")
    if weakness:
        parts.append(weakness)

    # If no weakness, add a positive availability note
    if not weakness:
        open_flag = signals.get('open_to_work_flag', False)
        if open_flag and notice <= 30:
            parts.append("available with short notice.")
        elif open_flag:
            parts.append("open to work.")
        else:
            parts.append("not marked open-to-work.")

    reasoning = " ".join(parts).replace("  ", " ").strip()
    return reasoning