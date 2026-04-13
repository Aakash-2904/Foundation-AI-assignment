"""
main.py
-------
CLI orchestrator — ties the scraper and generator together.
I kept this file intentionally thin: all the real logic lives in
scraper.py and resume_generator.py so this is purely wiring.

Usage:
    python main.py --resume path/to/your_resume.pdf --search "ML engineer"

Run fine-tuning first for best results (one-time, ~1-2 hrs):
    python finetune.py
"""

import argparse
import sys
from pathlib import Path

from scraper           import fetch_all_jobs
from resume_generator  import generate_resumes_for_jobs, print_summary, OUTPUT_DIR


def parse_args():
    # I used argparse here so users can control the full run from the command line
    # without touching any source code. The defaults are set to be sensible for ML engineers but can be overridden.
    parser = argparse.ArgumentParser(
        description="Local AI Resume Generator — Remotive + Arbeitnow + Ollama"
    )
    parser.add_argument("--resume",  required=True,
                        help="Path to your existing resume PDF")
    parser.add_argument("--search",  default="machine learning engineer",
                        help="Job search keyword (default: 'machine learning engineer')")
    parser.add_argument("--limit",   type=int, default=5,
                        help="Max jobs per source — default 5 keeps the first run manageable")
    parser.add_argument("--outdir",  default=str(OUTPUT_DIR),
                        help=f"Output directory for generated resumes (default: {OUTPUT_DIR})")
    return parser.parse_args()


def main():
    args = parse_args()

    resume_path = Path(args.resume)
    if not resume_path.exists():
        print(f"[Error] Resume PDF not found: {resume_path}")
        sys.exit(1)

    print("\n" + "=" * 60)
    print("  LOCAL AI RESUME GENERATOR")
    print("  Remotive + Arbeitnow  →  Ollama  →  LaTeX PDF")
    print("=" * 60)
    print(f"  Resume  : {resume_path}")
    print(f"  Search  : {args.search}")
    print(f"  Limit   : {args.limit} jobs/source  ({args.limit * 2} total max)")
    print(f"  Output  : {args.outdir}")
    print("=" * 60 + "\n")

    # Step 1 — scrape jobs from both sources
    jobs = fetch_all_jobs(search=args.search, limit_per_source=args.limit)
    if not jobs:
        print("[Error] No jobs fetched. Check your internet connection.")
        sys.exit(1)

    print(f"[Pipeline] {len(jobs)} jobs fetched. Starting resume generation ...\n")

    # Step 2 — generate one tailored resume per job
    results = generate_resumes_for_jobs(
        jobs           = jobs,
        resume_pdf_path= str(resume_path),
        output_dir     = Path(args.outdir),
    )

    # Step 3 — print a clean summary of what was produced
    print_summary(results)


if __name__ == "__main__":
    main()