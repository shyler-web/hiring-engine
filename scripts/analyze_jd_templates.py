import re
import gzip
import json
from collections import defaultdict, Counter
from tqdm import tqdm
import pickle
from pathlib import Path

def normalize_sentence(sent: str) -> str:
    """Remove numbers, years, and specific placeholders to create a template."""
    sent = re.sub(r'\d+\.?\d*', 'XX', sent)        # Replace numbers with XX
    sent = re.sub(r'\b\w+\.ai\b', 'COMPANY', sent)  # Replace company names
    sent = re.sub(r'\(.*?\)', '', sent)             # Remove parentheses
    sent = re.sub(r'\s+', ' ', sent).strip()
    return sent

def extract_sentences(text: str) -> list:
    if not text:
        return []
    raw = re.split(r'[.!?\n]', text)
    return [s.strip() for s in raw if len(s.strip()) > 15]

def get_jd_fingerprint(job_desc: str) -> str:
    """Create a normalized fingerprint for a job description."""
    sents = extract_sentences(job_desc)
    if not sents:
        return ""
    normalized = [normalize_sentence(s) for s in sents]
    return " | ".join(normalized)

def main(candidates_path: str, output_dir: str = "artifacts"):
    # Load candidates
    candidates = []
    opener = gzip.open if candidates_path.endswith(".gz") else open
    with opener(candidates_path, "rt", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                candidates.append(json.loads(line))
    print(f"[analyze] Loaded {len(candidates):,} candidates.")

    # Track JD templates and their sentence frequencies
    template_counter = Counter()           # template_id -> count
    sentence_freq_in_template = defaultdict(Counter)  # template_id -> {sentence: freq}
    candidate_templates = {}               # candidate_id -> template_id

    for c in tqdm(candidates, desc="Building JD templates"):
        cid = c['candidate_id']
        # Only analyze the CURRENT job (most recent)
        career = c.get('career_history', [])
        if not career:
            continue
        
        current_job = career[0]  # Most recent
        desc = current_job.get('description', '')
        if not desc:
            continue
        
        # Get normalized fingerprint
        fingerprint = get_jd_fingerprint(desc)
        if not fingerprint:
            continue
        
        # Count this template
        template_counter[fingerprint] += 1
        candidate_templates[cid] = fingerprint
        
        # Track sentence frequencies within this template
        raw_sentences = extract_sentences(desc)
        for sent in raw_sentences:
            normalized = normalize_sentence(sent)
            sentence_freq_in_template[fingerprint][normalized] += 1

    # --- Summary Statistics ---
    total_candidates = len(candidates)
    candidates_with_jd = len(candidate_templates)
    unique_templates = len(template_counter)
    
    print(f"\n[analyze] Candidates with valid JDs: {candidates_with_jd:,}/{total_candidates:,}")
    print(f"[analyze] Unique JD templates: {unique_templates:,}")
    
    # Top 10 most common templates
    print("\n[analyze] Top 10 most common JD templates:")
    for template, count in template_counter.most_common(10):
        # Show the first 3 sentences of this template as preview
        preview = " | ".join(template.split(" | ")[:3])
        print(f"  {count:,}x: {preview[:150]}...")
        # Show sentences in this template and their exact frequencies
        sentences = sentence_freq_in_template[template]
        print(f"    Sentences in this template ({len(sentences)} unique):")
        for sent, sent_count in sentences.most_common(5):
            print(f"      {sent_count}x: {sent[:80]}...")

    # --- Save artifacts ---
    output_path = Path(output_dir) / "jd_templates.pkl"
    output_path.parent.mkdir(exist_ok=True)
    with open(output_path, "wb") as f:
        pickle.dump({
            'template_counter': template_counter,
            'sentence_freq_in_template': dict(sentence_freq_in_template),
            'candidate_templates': candidate_templates
        }, f)
    print(f"\n[analyze] Saved JD template analysis to {output_path}")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--candidates", required=True)
    parser.add_argument("--output_dir", default="artifacts")
    args = parser.parse_args()
    main(args.candidates, args.output_dir)