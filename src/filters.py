"""
filters.py — Stage 0: Honeypot Detection + Hard Filters
Enhanced with per-rule diagnostics, unique candidate tracking, and overlap analysis.
"""

from datetime import date, datetime
from collections import defaultdict

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


# ─────────────────────────────────────────────────────────────────────────────
# HONEYPOT DETECTION  (returns flags dict per candidate)
# ─────────────────────────────────────────────────────────────────────────────

def get_honeypot_flags(candidate):
    """
    Returns a dict:
      {
        'flags': int,
        'triggered_rules': list[str],   # e.g. ['HP_R1', 'HP_R3']
      }
    """
    triggered = []

    # Rule HP_R1: expert/advanced skill with duration_months == 0
    for skill in candidate['skills']:
        if skill['proficiency'] in ('expert', 'advanced'):
            if skill.get('duration_months', 1) == 0:
                triggered.append('HP_R1')
                break   # one flag per candidate for this rule

    # Rule HP_R2: career timeline impossibility
    total_career_months = sum(
        j['duration_months'] for j in candidate['career_history']
    )
    yoe_months = candidate['profile']['years_of_experience'] * 12
    if total_career_months > yoe_months + 18:
        triggered.append('HP_R2')

    # Rule HP_R3: assessment score contradicts claimed proficiency
    assessments = candidate['redrob_signals']['skill_assessment_scores']
    skill_proficiency = {s['name']: s['proficiency'] for s in candidate['skills']}
    for skill_name, score in assessments.items():
        if skill_proficiency.get(skill_name) == 'expert' and score < 25:
            triggered.append('HP_R3')
            break

    # Rule HP_R4: impossible tenure (duration_months > 120 but yoe < 8)
    for job in candidate['career_history']:
        if job['duration_months'] > 120 and candidate['profile']['years_of_experience'] < 8:
            triggered.append('HP_R4')
            break

    return {
        'flags': len(set(triggered)),        # unique rules triggered
        'triggered_rules': list(set(triggered)),
    }


def is_honeypot(candidate):
    result = get_honeypot_flags(candidate)
    return result['flags'] >= 2


# ─────────────────────────────────────────────────────────────────────────────
# HARD FILTERS  (returns filter reason or None)
# ─────────────────────────────────────────────────────────────────────────────

def get_hard_filter_reason(candidate):
    """
    Returns the rule code that first eliminates this candidate, or None if they pass.
    Rule codes: HF_LOCATION, HF_EXPERIENCE, HF_INACTIVE, HF_IRRELEVANT_TITLE
    """
    profile = candidate['profile']
    signals = candidate['redrob_signals']
    today = date.today()

    # HF_LOCATION
    if profile['country'] != 'India' and not signals['willing_to_relocate']:
        return 'HF_LOCATION'

    # HF_EXPERIENCE
    if profile['years_of_experience'] < 3:
        return 'HF_EXPERIENCE'

    # HF_INACTIVE
    days_inactive = (today - parse_date(signals['last_active_date'])).days
    if not signals['open_to_work_flag'] and days_inactive > 180:
        return 'HF_INACTIVE'

    # HF_IRRELEVANT_TITLE
    all_industries = {j['industry'].lower() for j in candidate['career_history']}
    title_lower = profile['current_title'].lower()
    if title_lower in IRRELEVANT_TITLES and not any(
        'software' in ind or 'tech' in ind or 'ai' in ind
        for ind in all_industries
    ):
        return 'HF_IRRELEVANT_TITLE'

    return None


def passes_hard_filters(candidate):
    return get_hard_filter_reason(candidate) is None


# ─────────────────────────────────────────────────────────────────────────────
# MAIN FILTER  with full diagnostics
# ─────────────────────────────────────────────────────────────────────────────

def filter_candidates(candidates, verbose=True):
    """
    Runs Stage 0 on a list of candidates.

    Returns:
      passed          — list of candidates that survive both filters
      report          — dict with detailed per-rule stats
    """

    # Per honeypot rule → set of candidate_ids
    hp_rule_ids = defaultdict(set)   # HP_R1..HP_R4
    # Per hard-filter rule → set of candidate_ids
    hf_rule_ids = defaultdict(set)   # HF_*

    zero_score_ids = []   # honeypots (score = 0, excluded but not deleted)
    passed = []

    for c in candidates:
        cid = c['candidate_id']
        hp = get_honeypot_flags(c)

        # Record every honeypot rule this candidate triggered (even non-honeypots)
        for rule in hp['triggered_rules']:
            hp_rule_ids[rule].add(cid)

        if hp['flags'] >= 2:
            zero_score_ids.append(cid)
            continue   # skip hard filters — already eliminated

        hf_reason = get_hard_filter_reason(c)
        if hf_reason:
            hf_rule_ids[hf_reason].add(cid)
            continue

        passed.append(c)

    # ── Build overlap-aware report ──────────────────────────────────────────

    # Honeypot: candidates that triggered each rule (including those that
    # triggered only 1 rule and were NOT marked honeypot — you asked to see all)
    hp_section = {}
    all_hp_flagged = set()
    for rule in ['HP_R1', 'HP_R2', 'HP_R3', 'HP_R4']:
        ids = hp_rule_ids[rule]
        hp_section[rule] = {
            'count': len(ids),
            'candidate_ids': sorted(ids),
        }
        all_hp_flagged |= ids

    # Hard-filter section
    hf_section = {}
    all_hf_filtered = set()
    for rule in ['HF_LOCATION', 'HF_EXPERIENCE', 'HF_INACTIVE', 'HF_IRRELEVANT_TITLE']:
        ids = hf_rule_ids[rule]
        hf_section[rule] = {
            'count': len(ids),
            'candidate_ids': sorted(ids),
        }
        all_hf_filtered |= ids

    report = {
        'total_input': len(candidates),
        'honeypot_rules': hp_section,
        'honeypots_removed': {
            'count': len(zero_score_ids),
            'candidate_ids': sorted(zero_score_ids),
        },
        'hard_filter_rules': hf_section,
        'hard_filtered_total': {
            'count': len(all_hf_filtered),
            'candidate_ids': sorted(all_hf_filtered),
        },
        'passed': {
            'count': len(passed),
            'candidate_ids': sorted(c['candidate_id'] for c in passed),
        },
    }

    if verbose:
        _print_report(report)

    return passed, report


# ─────────────────────────────────────────────────────────────────────────────
# PRETTY PRINTER
# ─────────────────────────────────────────────────────────────────────────────

_HP_RULE_LABELS = {
    'HP_R1': 'Impossible skill duration (expert/advanced with 0 months)',
    'HP_R2': 'Career timeline impossibility (total months >> YoE)',
    'HP_R3': 'Assessment score contradicts claimed expertise (expert < 25)',
    'HP_R4': 'Impossible tenure (role > 120 months but YoE < 8 yrs)',
}

_HF_RULE_LABELS = {
    'HF_LOCATION':        'Outside India & unwilling to relocate',
    'HF_EXPERIENCE':      'Years of experience < 3',
    'HF_INACTIVE':        'Not open-to-work & inactive > 180 days',
    'HF_IRRELEVANT_TITLE':'Completely irrelevant title with no AI/tech exposure',
}


def _print_report(r):
    sep = "─" * 70

    print(f"\n{'═'*70}")
    print(f"  STAGE 0 FILTER REPORT  |  Input: {r['total_input']} candidates")
    print(f"{'═'*70}\n")

    # ── Honeypot rules ──────────────────────────────────────────────────────
    print("🔴  HONEYPOT DETECTION RULES  (candidates triggering each rule)")
    print("    (A candidate needs ≥2 flags to be marked a honeypot)\n")
    for rule, label in _HP_RULE_LABELS.items():
        data = r['honeypot_rules'][rule]
        ids_str = ', '.join(data['candidate_ids']) if data['candidate_ids'] else '—'
        print(f"  [{rule}]  {label}")
        print(f"           Triggered by {data['count']} candidate(s): {ids_str}")
        print()

    # ── Honeypots removed ───────────────────────────────────────────────────
    hp = r['honeypots_removed']
    ids_str = ', '.join(hp['candidate_ids']) if hp['candidate_ids'] else '—'
    print(sep)
    print(f"  🚫 HONEYPOTS REMOVED (≥2 flags → score=0, excluded)")
    print(f"     Total: {hp['count']} | IDs: {ids_str}\n")

    # ── Hard filter rules ───────────────────────────────────────────────────
    print("🟡  HARD FILTER RULES\n")
    for rule, label in _HF_RULE_LABELS.items():
        data = r['hard_filter_rules'][rule]
        ids_str = ', '.join(data['candidate_ids']) if data['candidate_ids'] else '—'
        print(f"  [{rule}]  {label}")
        print(f"           Filtered: {data['count']} candidate(s): {ids_str}")
        print()

    hf = r['hard_filtered_total']
    ids_str = ', '.join(hf['candidate_ids']) if hf['candidate_ids'] else '—'
    print(sep)
    print(f"  🚫 HARD-FILTERED TOTAL (unique, post-honeypot-removal)")
    print(f"     Total: {hf['count']} | IDs: {ids_str}\n")

    # ── Summary ─────────────────────────────────────────────────────────────
    passed = r['passed']
    ids_str = ', '.join(passed['candidate_ids']) if passed['candidate_ids'] else '—'
    print(f"{'═'*70}")
    print(f"  ✅ PASSED Stage 0: {passed['count']} candidates")
    print(f"     IDs: {ids_str}")
    print(f"{'═'*70}\n")