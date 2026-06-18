#!/usr/bin/env python3
"""
Complete data exploration script for 1 lakh candidate profiles.
Reads candidates.jsonl.gz (or misspelled version) and outputs structured summaries.
"""

import gzip
import json
import os
from collections import Counter, defaultdict
from datetime import datetime
import numpy as np

# ============================================================
# CONFIGURATION – change these if your file path is different
# ============================================================
INPUT_FILE = "candidates.jsonl.gz"   # default – correct spelling
OUTPUT_BASE = "output"

# Fields that are highly cardinal; we generate both top-100 and full lists
HIGH_CARDINALITY = [
    "current_company",
    "current_title",
    "career_history_company",
    "career_history_title",
]

# ============================================================
# HELPER FUNCTIONS
# ============================================================

def ensure_dirs():
    """Create the required output subdirectories and return their paths."""
    subdirs = [
        "00_summary", "01_frequencies", "02_distributions",
        "03_dates", "04_skills_assessment", "05_salary"
    ]
    paths = {}
    for sub in subdirs:
        p = os.path.join(OUTPUT_BASE, sub)
        os.makedirs(p, exist_ok=True)
        paths[sub] = p
    return paths

def save_freq(counter, filepath, top_n=None):
    """
    Write a frequency table (count, value) sorted descending.
    If top_n is given, only write that many rows and add a footer.
    """
    with open(filepath, 'w', encoding='utf-8') as f:
        items = counter.most_common(top_n)
        for val, cnt in items:
            f.write(f"{cnt:8d}  {val}\n")
        if top_n is not None and len(counter) > top_n:
            f.write(f"\n... and {len(counter) - top_n} more values (see _full.txt)\n")

def save_full_and_top(counter, base_path, label):
    """
    For high‑cardinality fields: write both a top‑100 and a full file.
    """
    top_file = f"{base_path}_top100.txt"
    full_file = f"{base_path}_full.txt"
    save_freq(counter, top_file, top_n=100)
    save_freq(counter, full_file, top_n=None)

def parse_date(date_str):
    """Parse YYYY-MM-DD date; return date object or None if invalid."""
    if not date_str or not isinstance(date_str, str):
        return None
    try:
        return datetime.strptime(date_str, "%Y-%m-%d").date()
    except ValueError:
        return None

def safe_float(val, sentinel=None):
    """Convert to float if possible; return np.nan for sentinel values."""
    if val is None:
        return np.nan
    if sentinel is not None and val == sentinel:
        return np.nan
    try:
        return float(val)
    except (ValueError, TypeError):
        return np.nan

def write_numeric_summary(name, vals, file_handle):
    """
    Compute and write min, max, mean, median, percentiles for a numeric list.
    Handles NaN values with numpy functions.
    """
    arr = np.array(vals, dtype=np.float64)
    non_nan = arr[~np.isnan(arr)]
    if len(non_nan) == 0:
        file_handle.write(f"\n=== {name} ===\n  No valid (non‑null) values.\n")
        return

    stats = {
        "min": np.nanmin(arr),
        "max": np.nanmax(arr),
        "mean": np.nanmean(arr),
        "median": np.nanmedian(arr),
        "p25": np.nanpercentile(arr, 25),
        "p50": np.nanpercentile(arr, 50),
        "p75": np.nanpercentile(arr, 75),
        "p90": np.nanpercentile(arr, 90),
        "p95": np.nanpercentile(arr, 95),
        "p99": np.nanpercentile(arr, 99),
        "count_non_null": len(non_nan),
        "count_null": len(arr) - len(non_nan),
    }
    file_handle.write(f"\n=== {name} ===\n")
    file_handle.write(f"  valid N = {stats['count_non_null']:,}  |  missing = {stats['count_null']:,}\n")
    file_handle.write(f"  min = {stats['min']:.2f}  max = {stats['max']:.2f}  mean = {stats['mean']:.2f}  median = {stats['median']:.2f}\n")
    file_handle.write(f"  p25 = {stats['p25']:.2f}  p50 = {stats['p50']:.2f}  p75 = {stats['p75']:.2f}  p90 = {stats['p90']:.2f}  p95 = {stats['p95']:.2f}  p99 = {stats['p99']:.2f}\n")

# ============================================================
# MAIN ANALYSIS
# ============================================================

def analyze():
    print("🚀 Starting analysis...")
    paths = ensure_dirs()

    # ---------- Data containers ----------
    # Numeric fields
    numeric_data = {
        "years_of_experience": [],
        "profile_completeness_score": [],
        "profile_views_received_30d": [],
        "applications_submitted_30d": [],
        "recruiter_response_rate": [],
        "avg_response_time_hours": [],
        "notice_period_days": [],
        "github_activity_score": [],
        "saved_by_recruiters_30d": [],
        "interview_completion_rate": [],
        "offer_acceptance_rate": [],
        "salary_min": [],
        "salary_max": [],
    }

    # Categorical counters
    counters = {
        "current_company": Counter(),
        "current_title": Counter(),
        "current_industry": Counter(),
        "career_history_company": Counter(),
        "career_history_title": Counter(),
        "career_history_industry": Counter(),
        "field_of_study": Counter(),
        "degree": Counter(),
        "certification_name": Counter(),
        "preferred_work_mode": Counter(),
        "education_tier": Counter(),
        "company_size": Counter(),           # from profile
        "career_company_size": Counter(),    # from career_history
        "open_to_work_flag": Counter(),
        "willing_to_relocate": Counter(),
    }

    # Skill-specific
    skill_counter = Counter()                     # frequency of each skill name
    skill_proficiency = defaultdict(Counter)      # skill -> {proficiency: count}
    skill_endorsements = defaultdict(list)        # skill -> list of endorsement counts

    # Skill assessment (redrob_signals)
    assessed_skill_counter = Counter()            # skill -> number of candidates with that assessment
    assessed_skill_scores = defaultdict(list)     # skill -> list of scores (0-100)

    # Dates
    active_dates = []

    # Missing data tracker
    missing_counts = defaultdict(int)
    total_candidates = 0

    # ---------- Read the file ----------
    try:
        with gzip.open(INPUT_FILE, 'rt', encoding='utf-8') as f:
            for line_num, line in enumerate(f, 1):
                line = line.strip()
                if not line:
                    continue
                try:
                    c = json.loads(line)
                except json.JSONDecodeError:
                    print(f"⚠️  Skipping malformed JSON on line {line_num}")
                    continue

                total_candidates += 1
                if total_candidates % 10000 == 0:
                    print(f"  Processed {total_candidates:,} candidates...")

                # ---------- PROFILE ----------
                prof = c.get('profile', {})
                if prof:
                    counters["current_company"][prof.get('current_company', 'MISSING').strip()] += 1
                    counters["current_title"][prof.get('current_title', 'MISSING').strip()] += 1
                    counters["current_industry"][prof.get('current_industry', 'MISSING').strip()] += 1
                    counters["company_size"][prof.get('current_company_size', 'MISSING')] += 1

                    yoe = prof.get('years_of_experience')
                    numeric_data["years_of_experience"].append(safe_float(yoe))
                else:
                    missing_counts["profile"] += 1

                # ---------- REDROB SIGNALS ----------
                rs = c.get('redrob_signals', {})
                if rs:
                    # Numeric fields with possible sentinels
                    numeric_data["profile_completeness_score"].append(safe_float(rs.get('profile_completeness_score')))
                    numeric_data["profile_views_received_30d"].append(safe_float(rs.get('profile_views_received_30d')))
                    numeric_data["applications_submitted_30d"].append(safe_float(rs.get('applications_submitted_30d')))
                    numeric_data["recruiter_response_rate"].append(safe_float(rs.get('recruiter_response_rate')))
                    numeric_data["avg_response_time_hours"].append(safe_float(rs.get('avg_response_time_hours')))
                    numeric_data["notice_period_days"].append(safe_float(rs.get('notice_period_days')))
                    numeric_data["saved_by_recruiters_30d"].append(safe_float(rs.get('saved_by_recruiters_30d')))
                    numeric_data["interview_completion_rate"].append(safe_float(rs.get('interview_completion_rate')))

                    # GitHub score: -1 means no GitHub
                    gh = rs.get('github_activity_score', -1)
                    numeric_data["github_activity_score"].append(safe_float(gh, sentinel=-1))

                    # Offer acceptance rate: -1 means no offer history
                    offer = rs.get('offer_acceptance_rate', -1)
                    numeric_data["offer_acceptance_rate"].append(safe_float(offer, sentinel=-1))

                    # Work mode
                    mode = rs.get('preferred_work_mode', 'MISSING').strip()
                    counters["preferred_work_mode"][mode] += 1

                    # Booleans
                    open_flag = rs.get('open_to_work_flag')
                    if open_flag is not None:
                        counters["open_to_work_flag"][str(open_flag)] += 1
                    else:
                        counters["open_to_work_flag"]['MISSING'] += 1

                    relocate = rs.get('willing_to_relocate')
                    if relocate is not None:
                        counters["willing_to_relocate"][str(relocate)] += 1
                    else:
                        counters["willing_to_relocate"]['MISSING'] += 1

                    # Salary
                    sal = rs.get('expected_salary_range_inr_lpa')
                    if sal and isinstance(sal, dict):
                        numeric_data["salary_min"].append(safe_float(sal.get('min')))
                        numeric_data["salary_max"].append(safe_float(sal.get('max')))
                    else:
                        numeric_data["salary_min"].append(np.nan)
                        numeric_data["salary_max"].append(np.nan)

                    # Skill assessment scores (dict)
                    assessments = rs.get('skill_assessment_scores', {})
                    if isinstance(assessments, dict):
                        for skill_name, score in assessments.items():
                            skill_name = skill_name.strip().lower()
                            if skill_name:
                                assessed_skill_counter[skill_name] += 1
                                assessed_skill_scores[skill_name].append(safe_float(score))

                    # Date
                    date_str = rs.get('last_active_date')
                    dt = parse_date(date_str)
                    if dt:
                        active_dates.append(dt)
                    else:
                        missing_counts["last_active_date"] += 1
                else:
                    missing_counts["redrob_signals"] += 1

                # ---------- CAREER HISTORY ----------
                for job in c.get('career_history', []):
                    if not isinstance(job, dict):
                        continue
                    comp = job.get('company', 'MISSING').strip()
                    title = job.get('title', 'MISSING').strip()
                    ind = job.get('industry', 'MISSING').strip()
                    size = job.get('company_size', 'MISSING')
                    counters["career_history_company"][comp] += 1
                    counters["career_history_title"][title] += 1
                    counters["career_history_industry"][ind] += 1
                    counters["career_company_size"][size] += 1

                # ---------- EDUCATION ----------
                for edu in c.get('education', []):
                    if not isinstance(edu, dict):
                        continue
                    field = edu.get('field_of_study', 'MISSING').strip()
                    degree = edu.get('degree', 'MISSING').strip()
                    tier = edu.get('tier', 'MISSING')
                    counters["field_of_study"][field] += 1
                    counters["degree"][degree] += 1
                    counters["education_tier"][tier] += 1

                # ---------- SKILLS ----------
                for sk in c.get('skills', []):
                    if not isinstance(sk, dict):
                        continue
                    name = sk.get('name', 'MISSING').strip().lower()
                    if not name or name == 'missing':
                        continue
                    prof = sk.get('proficiency', 'unknown').strip().lower()
                    endorse = sk.get('endorsements', 0)
                    if not isinstance(endorse, (int, float)):
                        endorse = 0

                    skill_counter[name] += 1
                    skill_proficiency[name][prof] += 1
                    skill_endorsements[name].append(float(endorse))

                # ---------- CERTIFICATIONS ----------
                for cert in c.get('certifications', []):
                    if not isinstance(cert, dict):
                        continue
                    cert_name = cert.get('name', 'MISSING').strip()
                    if cert_name:
                        counters["certification_name"][cert_name] += 1

        # ---------- END OF FILE LOOP ----------

    except FileNotFoundError:
        print(f"❌ ERROR: Input file '{INPUT_FILE}' not found.")
        print("   Please check the path and filename (you can edit INPUT_FILE at the top of the script).")
        return
    except Exception as e:
        print(f"❌ Unexpected error while reading file: {e}")
        return

    print(f"✅ Finished reading. Total candidates processed: {total_candidates:,}")

    # ============================================================
    # WRITE OUTPUTS
    # ============================================================

    # ---------- 00_summary / dataset_overview.txt ----------
    overview_path = os.path.join(paths["00_summary"], "dataset_overview.txt")
    with open(overview_path, 'w', encoding='utf-8') as f:
        f.write("=== DATASET OVERVIEW ===\n")
        f.write(f"Total candidates (valid JSON lines): {total_candidates:,}\n\n")
        f.write("=== MISSING DATA COUNT (top-level sections) ===\n")
        for field, cnt in sorted(missing_counts.items(), key=lambda x: -x[1]):
            pct = (cnt / total_candidates) * 100 if total_candidates else 0
            f.write(f"  {field}: {cnt:,} ({pct:.1f}%)\n")

        # Also add missing counts for numeric fields (based on NaN count)
        f.write("\n=== MISSING DATA COUNT (numeric fields, incl. -1 sentinels) ===\n")
        for name, vals in numeric_data.items():
            arr = np.array(vals, dtype=np.float64)
            nulls = np.isnan(arr).sum()
            pct = (nulls / total_candidates) * 100 if total_candidates else 0
            f.write(f"  {name}: {nulls:,} ({pct:.1f}%)\n")

    # ---------- 00_summary / numeric_summary.txt ----------
    num_path = os.path.join(paths["00_summary"], "numeric_summary.txt")
    with open(num_path, 'w', encoding='utf-8') as f:
        f.write("=== NUMERIC FIELD STATISTICS ===\n")
        for name, vals in numeric_data.items():
            write_numeric_summary(name, vals, f)

    # ---------- 01_frequencies ----------
    freq_dir = paths["01_frequencies"]

    # High‑cardinality: top100 + full
    save_full_and_top(counters["current_company"], os.path.join(freq_dir, "current_company"), "current_company")
    save_full_and_top(counters["current_title"], os.path.join(freq_dir, "current_title"), "current_title")
    save_full_and_top(counters["career_history_company"], os.path.join(freq_dir, "career_history_company"), "career_history_company")
    save_full_and_top(counters["career_history_title"], os.path.join(freq_dir, "career_history_title"), "career_history_title")

    # Low‑cardinality: single file (full list)
    low_card = [
        ("current_industry", counters["current_industry"]),
        ("career_history_industry", counters["career_history_industry"]),
        ("field_of_study", counters["field_of_study"]),
        ("degree", counters["degree"]),
        ("certification_name", counters["certification_name"]),
        ("preferred_work_mode", counters["preferred_work_mode"]),
    ]
    for name, counter in low_card:
        fpath = os.path.join(freq_dir, f"{name}_freq.txt")
        save_freq(counter, fpath, top_n=None)

    # ---------- 02_distributions ----------
    dist_dir = paths["02_distributions"]

    # Skill proficiency cross‑tab
    prof_path = os.path.join(dist_dir, "skill_proficiency_crosstab.txt")
    with open(prof_path, 'w', encoding='utf-8') as f:
        f.write("=== SKILL PROFICIENCY DISTRIBUTION ===\n")
        # Sort skills by total frequency (most common first)
        for skill, _ in skill_counter.most_common():
            f.write(f"\n--- {skill} ---\n")
            prof_counts = skill_proficiency[skill]
            if not prof_counts:
                f.write("  (no proficiency data)\n")
            else:
                for level in ["beginner", "intermediate", "advanced", "expert"]:
                    cnt = prof_counts.get(level, 0)
                    if cnt > 0:
                        f.write(f"  {level:12s}: {cnt:,}\n")

    # Skill endorsement average
    endorse_path = os.path.join(dist_dir, "skill_endorsement_avg.txt")
    with open(endorse_path, 'w', encoding='utf-8') as f:
        f.write("=== AVERAGE ENDORSEMENTS PER SKILL ===\n")
        f.write("(only skills with at least 5 mentions)\n\n")
        for skill, lst in skill_endorsements.items():
            if len(lst) >= 5:
                avg = np.mean(lst)
                f.write(f"{skill:30s}  avg = {avg:.2f}  (N={len(lst):,})\n")

    # Education tier
    tier_path = os.path.join(dist_dir, "education_tier_freq.txt")
    save_freq(counters["education_tier"], tier_path, top_n=None)

    # Company size (profile level)
    size_path = os.path.join(dist_dir, "company_size_freq.txt")
    save_freq(counters["company_size"], size_path, top_n=None)

    # Booleans
    def write_bool_counter(counter, filename, label):
        path = os.path.join(dist_dir, filename)
        with open(path, 'w', encoding='utf-8') as f:
            f.write(f"=== {label} ===\n")
            for key in ["True", "False", "MISSING"]:
                cnt = counter.get(key, 0)
                f.write(f"  {key:7s}: {cnt:,}\n")

    write_bool_counter(counters["open_to_work_flag"], "open_to_work_flag.txt", "OPEN TO WORK FLAG")
    write_bool_counter(counters["willing_to_relocate"], "willing_to_relocate.txt", "WILLING TO RELOCATE")

    # ---------- 03_dates / last_active_monthly.txt ----------
    date_dir = paths["03_dates"]
    date_path = os.path.join(date_dir, "last_active_monthly.txt")
    if active_dates:
        # Group by year-month
        month_counts = Counter()
        for d in active_dates:
            key = d.strftime("%Y-%m")
            month_counts[key] += 1
        with open(date_path, 'w', encoding='utf-8') as f:
            f.write("=== LAST ACTIVE DATE – MONTHLY FREQUENCY ===\n")
            for ym, cnt in sorted(month_counts.items()):
                f.write(f"  {ym}: {cnt:,}\n")
            f.write(f"\n  Date range: {min(active_dates)}  →  {max(active_dates)}")
    else:
        with open(date_path, 'w', encoding='utf-8') as f:
            f.write("No valid dates found.\n")

    # ---------- 04_skills_assessment ----------
    ass_dir = paths["04_skills_assessment"]

    # Frequency of assessed skills
    ass_freq_path = os.path.join(ass_dir, "assessed_skill_freq.txt")
    save_freq(assessed_skill_counter, ass_freq_path, top_n=None)

    # Average score per assessed skill
    ass_score_path = os.path.join(ass_dir, "assessed_skill_avg_score.txt")
    with open(ass_score_path, 'w', encoding='utf-8') as f:
        f.write("=== AVERAGE ASSESSMENT SCORE PER SKILL ===\n")
        f.write("(only skills assessed by at least 10 candidates)\n\n")
        for skill, scores in assessed_skill_scores.items():
            if len(scores) >= 10:
                avg = np.nanmean(scores)
                f.write(f"{skill:30s}  avg = {avg:.2f}  (N={len(scores):,})\n")

    # ---------- 05_salary / salary_summary.txt ----------
    sal_dir = paths["05_salary"]
    sal_path = os.path.join(sal_dir, "salary_summary.txt")
    with open(sal_path, 'w', encoding='utf-8') as f:
        f.write("=== SALARY EXPECTATIONS (INR LPA) ===\n")
        write_numeric_summary("salary_min (expected minimum)", numeric_data["salary_min"], f)
        write_numeric_summary("salary_max (expected maximum)", numeric_data["salary_max"], f)

    print("\n✅ Analysis complete! All output files are in the 'output/' folder.")
    print(f"   Summary: {overview_path}")
    print(f"   Numeric stats: {num_path}")
    print("   Explore the subdirectories for detailed views.")

# ============================================================
# ENTRY POINT
# ============================================================
if __name__ == "__main__":
    analyze()