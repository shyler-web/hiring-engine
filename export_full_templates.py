# export_full_templates.py
import pickle
from pathlib import Path

def main():
    print("[export] Loading pickle file...")
    with open("artifacts/jd_templates.pkl", "rb") as f:
        data = pickle.load(f)

    output_path = Path("artifacts/full_templates_report.txt")
    
    print(f"[export] Writing full report to {output_path}...")
    with open(output_path, "w", encoding="utf-8") as out:
        # ============================================================
        # HEADER
        # ============================================================
        out.write("=" * 120 + "\n")
        out.write("FULL JD TEMPLATE ANALYSIS – COMPLETE DUMP\n")
        out.write("=" * 120 + "\n\n")
        
        out.write(f"Total unique templates: {len(data['template_counter']):,}\n")
        out.write(f"Candidates matched to a template: {len(data['candidate_templates']):,}\n")
        out.write(f"Candidates with no JD or empty description: {100000 - len(data['candidate_templates']):,}\n\n")

        # ============================================================
        # SECTION 1: ALL TEMPLATES – Sorted by Frequency (Most Common First)
        # ============================================================
        out.write("-" * 120 + "\n")
        out.write("SECTION 1: ALL TEMPLATES (Sorted by Frequency – Most Common First)\n")
        out.write("-" * 120 + "\n\n")
        
        # Get sorted list
        sorted_templates = data['template_counter'].most_common()
        
        for idx, (template, count) in enumerate(sorted_templates, 1):
            out.write(f"\n{'#' * 120}\n")
            out.write(f"TEMPLATE #{idx} | Frequency: {count:,} candidates\n")
            out.write(f"{'#' * 120}\n")
            
            # Split the template into individual sentences
            sentences = template.split(" | ")
            out.write("FULL TEMPLATE TEXT (all sentences):\n")
            for i, sent in enumerate(sentences, 1):
                out.write(f"  {i}. {sent}\n")
            
            # Now show the sentence frequency breakdown inside this template
            sent_freqs = data['sentence_freq_in_template'].get(template, {})
            if sent_freqs:
                out.write("\nSENTENCE FREQUENCY BREAKDOWN (within this template):\n")
                out.write("(This shows which parts of the template are variable vs fixed)\n")
                # Sort by frequency descending (most fixed first)
                for sent, freq in sorted(sent_freqs.items(), key=lambda x: x[1], reverse=True):
                    out.write(f"  {freq:>6,}x : {sent}\n")
            else:
                out.write("\n(No sentence-level frequencies recorded for this template)\n")
            
            out.write("\n" + "-" * 120 + "\n")

        # ============================================================
        # SECTION 2: UNIQUE TEMPLATES (Frequency = 1) – Full List
        # ============================================================
        out.write("\n\n" + "=" * 120 + "\n")
        out.write("SECTION 2: UNIQUE TEMPLATES (appear exactly 1 time – completely custom JDs)\n")
        out.write("=" * 120 + "\n\n")
        
        unique_templates = [t for t, c in data['template_counter'].items() if c == 1]
        out.write(f"Total unique templates: {len(unique_templates):,}\n\n")
        
        for idx, template in enumerate(unique_templates, 1):
            sentences = template.split(" | ")
            out.write(f"\n--- UNIQUE #{idx} ---\n")
            for sent in sentences:
                out.write(f"  {sent}\n")
        
        # ============================================================
        # SECTION 3: MAPPING OF CANDIDATE_ID -> TEMPLATE (Optional, huge)
        # ============================================================
        # This section will be ~100,000 lines. Uncomment ONLY if you need it.
        # out.write("\n\n" + "=" * 120 + "\n")
        # out.write("SECTION 3: CANDIDATE ID -> TEMPLATE MAPPING\n")
        # out.write("=" * 120 + "\n\n")
        # 
        # for cid, template in data['candidate_templates'].items():
        #     # Truncate template to first 150 chars for readability
        #     preview = template[:150].replace("\n", " ") + "..."
        #     out.write(f"{cid} -> {preview}\n")

    print(f"✅ Full report written to: {output_path}")
    print(f"   File size: {output_path.stat().st_size / (1024*1024):.2f} MB")
    print(f"   Total templates written: {len(sorted_templates):,}")
    print(f"   Unique templates written: {len(unique_templates):,}")

if __name__ == "__main__":
    main()