"""
run_stage0.py — Standalone runner for Stage 0 diagnostics
Usage (run from project root):
    python run_stage0.py --input candidates.jsonl.gz
    python run_stage0.py --input data/candidates.jsonl.gz --save-report reports/report.json
"""

import gzip
import json
import argparse
import sys
from pathlib import Path

# ---------- FIX: Tell Python where to find src/filters.py ----------
sys.path.insert(0, str(Path(__file__).parent / 'src'))

# Now we can import from src/
from src.filters import filter_candidates


# ---------- FIX: Handles JSONL (100k lines) and JSON arrays ----------
def load_candidates(path: str):
    p = Path(path)
    if not p.exists():
        print(f"[ERROR] File not found: {path}", file=sys.stderr)
        sys.exit(1)

    opener = gzip.open if path.endswith('.gz') else open
    with opener(path, 'rt', encoding='utf-8') as f:
        first_char = f.read(1)
        f.seek(0)

        if first_char == '[':
            # Standard JSON array
            data = json.load(f)
            return data if isinstance(data, list) else data.get('candidates', data)
        else:
            # JSONL: one JSON per line (your 100k file)
            candidates = []
            for line in f:
                line = line.strip()
                if line:
                    candidates.append(json.loads(line))
            return candidates


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', required=True, help='Path to candidates.jsonl.gz')
    parser.add_argument('--save-report', default=None, help='Save report JSON')
    parser.add_argument('--save-passed', default=None, help='Save passed candidates JSON')
    args = parser.parse_args()

    print(f"[+] Loading from: {args.input}")
    candidates = load_candidates(args.input)
    print(f"[+] Loaded {len(candidates)} candidates\n")

    # Run Stage 0 with verbose=True (prints the beautiful table to console)
    passed, report = filter_candidates(candidates, verbose=True)

    # Save files ONLY if the user asked for them
    if args.save_report:
        # Ensure the parent folder exists (e.g., 'reports/')
        Path(args.save_report).parent.mkdir(parents=True, exist_ok=True)
        with open(args.save_report, 'w') as f:
            json.dump(report, f, indent=2)
        print(f"\n[+] Report saved to: {args.save_report}")

    if args.save_passed:
        Path(args.save_passed).parent.mkdir(parents=True, exist_ok=True)
        with open(args.save_passed, 'w') as f:
            json.dump(passed, f, indent=2)
        print(f"[+] Passed candidates saved to: {args.save_passed}")


if __name__ == '__main__':
    main()