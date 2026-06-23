# export_templates_summary.py
import pickle
from pathlib import Path

def main():
    # Load the pickle
    with open("artifacts/jd_templates.pkl", "rb") as f:
        data = pickle.load(f)

    # Open a text file for writing
    output_path = Path("artifacts/templates_summary.txt")
    with open(output_path, "w", encoding="utf-8") as out:
        out.write("=" * 80 + "\n")
        out.write("JD TEMPLATE ANALYSIS – HUMAN READABLE REPORT\n")
        out.write("=" * 80 + "\n\n")

        out.write(f"Total unique templates: {len(data['template_counter']):,}\n")
        out.write(f"Candidates matched: {len(data['candidate_templates']):,}\n\n")

        out.write("-" * 80 + "\n")
        out.write("TOP 20 MOST COMMON JD TEMPLATES\n")
        out.write("-" * 80 + "\n\n")

        for idx, (template, count) in enumerate(data['template_counter'].most_common(20), 1):
            out.write(f"\n{idx}. COUNT: {count:,} candidates\n")
            out.write(f"   TEMPLATE (first 300 chars):\n")
            # Show the template as sentences (split by |)
            sentences = template.split(" | ")
            for sent in sentences[:5]:  # Show first 5 sentences
                out.write(f"      - {sent[:150]}...\n")
            
            # Now show which sentences inside this template are the most variable (rare) vs fixed (common)
            sent_freqs = data['sentence_freq_in_template'].get(template, {})
            if sent_freqs:
                out.write(f"   SENTENCE FREQUENCIES INSIDE THIS TEMPLATE:\n")
                # Sort by frequency (most common first)
                for sent, freq in sorted(sent_freqs.items(), key=lambda x: x[1], reverse=True)[:6]:
                    out.write(f"      {freq}x: {sent[:120]}...\n")
            out.write("\n")

        # Also list all unique templates that appear only once (the "golden" ones)
        out.write("\n" + "-" * 80 + "\n")
        out.write("UNIQUE TEMPLATES (appear exactly 1 time) – SAMPLE OF 20\n")
        out.write("-" * 80 + "\n\n")
        
        unique_templates = [t for t, c in data['template_counter'].items() if c == 1]
        out.write(f"Total unique templates: {len(unique_templates):,}\n\n")
        
        for template in unique_templates[:20]:
            sentences = template.split(" | ")
            out.write(f"  • {sentences[0][:120]}...\n")

    print(f"✅ Summary written to: {output_path}")

if __name__ == "__main__":
    main()