# Automated Resume Tailoring Using Large Language Models
### A Comparative Study of Fine-Tuned Specialization and Agentic Generalization

**Authors:** Riya Nitin Taori · Aakash Kumar
**Program:** MS Artificial Intelligence — Khoury College of Computer Sciences, Northeastern University

---

## Overview

This project implements and compares two fully local, privacy-preserving pipelines for automated resume tailoring using large language models. Both methods run entirely on consumer hardware via [Ollama](https://ollama.com) — no cloud APIs, no data leaving your machine.

| | Method 1 | Method 2 |
|---|---|---|
| **Approach** | QLoRA fine-tuning on Phi-3 mini 3.8B | Multi-agent pipeline with Llama 3.2 |
| **Training data** | 3,000+ curated examples across 5 job categories | None — zero-shot prompting only |
| **LaTeX generation** | Model outputs raw LaTeX directly | Python builds LaTeX from structured model output |
| **Best for** | In-distribution roles (ML, DevOps, Full-Stack, Data Science, DL Research) | Any role, including novel or non-technical ones |
| **Batch success rate** | ~60–70% on diverse job sets | 95% (19/20 jobs) |
| **Avg cosine similarity** | ~0.41 on in-distribution roles | 0.2889 overall (σ=0.12) |

The central finding is a **generalization-quality tradeoff**: Method 1 produces higher-quality output for roles within its training distribution, while Method 2 handles the full diversity of modern job postings with consistent structural reliability.

---

## Project Structure

```
.
├── Method - 1/
│   ├── finetune.py          # QLoRA fine-tuning of Phi-3 mini via unsloth
│   ├── main.py              # CLI entry point — scrape + generate in one command
│   ├── resume_generator.py  # Ollama inference + LaTeX assembly + PDF compilation
│   ├── scraper.py           # Job listing fetcher (Remotive + Arbeitnow APIs)
│   ├── my_resume.txt        # Candidate profile used for evaluation
│   └── jobs_output.xlsx     # Pre-scraped job listings (20 jobs, 6 role types)
│
├── Method - 2/
│   ├── agent.py             # Full agentic pipeline — extraction, LaTeX build, metrics
│   ├── my_resume.txt        # Candidate profile (same as Method 1)
│   └── jobs_output.xlsx     # Same pre-scraped job listings for controlled comparison
│
├── requirements.txt
└── README.md
```

---

## Requirements

**Hardware:** A GPU with at least 6GB VRAM is required for Method 1 fine-tuning. Method 2 runs on CPU with 8GB+ RAM.

**Software:**
- Python 3.10+
- [Ollama](https://ollama.com) installed and running locally
- LaTeX distribution for PDF compilation (optional — `.tex` files are always produced)
  - Linux: `sudo apt-get install texlive-latex-extra`
  - macOS: `brew install --cask mactex-no-gui`

Install Python dependencies:
```bash
pip install -r requirements.txt
```

---

## Dataset

The fine-tuning corpus for Method 1 consists of 3,000+ resume instruction-output pairs across five job categories.

The dataset is **publicly available on Kaggle** and is **not included in this repository**:

> Hadi KP. *Resume Data PDF*. Kaggle, 2024.
> https://www.kaggle.com/datasets/hadikp/resume-data-pdf

**To reproduce the fine-tuning run:**
1. Download the CSV from the Kaggle link above
2. Place it at `Method - 1/data/resume_dataset.csv`
3. Run `python finetune.py` — it auto-detects the file

If the file is not present, `finetune.py` falls back to 5 hardcoded examples (one per training category) which are sufficient to verify the pipeline runs without reproducing the full training job.

---

## Method 1: Fine-Tuned Phi-3 + Direct LaTeX Generation

### How it works

`finetune.py` applies QLoRA fine-tuning to `unsloth/Phi-3-mini-4k-instruct-bnb-4bit` using the [unsloth](https://github.com/unslothai/unsloth) library. LoRA adapters are applied to all projection layers (query, key, value, output, gate, up, down) with rank r=16 and alpha=32, targeting less than 0.5% of the base model's total parameters.

The training corpus covers five established job categories with stable, well-defined skill vocabularies:
- Senior Machine Learning Engineer
- Data Scientist (Financial Services)
- Deep Learning Research Engineer
- DevOps Engineer
- Full-Stack Developer

After training, the adapter is exported to Q4\_K\_M GGUF format and registered as an Ollama model named `resume-generator`. At inference time, `resume_generator.py` calls the Ollama API and compiles the output to PDF via `pdflatex`.

### Run Method 1

**Step 1 — Fine-tune the model** (requires GPU, run once):
```bash
cd "Method - 1"
python finetune.py
```

**Step 2 — Generate tailored resumes:**
```bash
python main.py --resume my_resume.txt
python main.py --resume my_resume.txt --search "data scientist" --limit 10
python main.py --resume my_resume.txt --outdir ./output_resumes
```

| Argument | Default | Description |
|---|---|---|
| `--resume` | *(required)* | Path to your resume PDF |
| `--search` | `"machine learning engineer"` | Job search keyword |
| `--limit` | `5` | Max jobs per source (Remotive + Arbeitnow) |
| `--outdir` | `./generated_resumes` | Output folder for `.tex` and `.pdf` files |

---

## Method 2: Agentic Multi-Agent Pipeline

### How it works

`agent.py` decomposes resume tailoring into a sequential four-agent pipeline, all powered by Llama 3.2 via Ollama at temperature 0.2:

1. **Keyword extraction** — identifies the most salient terms in the job description
2. **Semantic matching** — maps candidate skills against role requirements
3. **Resume tailoring** — generates structured `KEY:VALUE` output with pipe-delimited bullet points
4. **Scoring** — produces a 0–100 relevancy score and a list of matched keywords

The `KEY:VALUE` format was chosen over JSON specifically because Llama 3.2 inserts hard line breaks at approximately column 99 in long strings, producing malformed JSON. The `KEY:VALUE` parser operates line-by-line and is immune to this failure mode.

A `build_latex()` function constructs the complete `.tex` document deterministically from the parsed output. A `fallback_latex()` function ensures a valid file is always written — no job is ever skipped entirely.

Three evaluation metrics are computed for every job:
- **Cosine similarity** — semantic overlap between resume and JD via `all-MiniLM-L6-v2` embeddings
- **TF-IDF ATS score** — fraction of top-20 TF-IDF keywords from the JD that appear in the resume (model-free cross-check)
- **LLM match score** — self-reported 0–100 relevancy estimate from the scoring agent

### Run Method 2

**Prerequisites** — pull the model once:
```bash
ollama pull llama3.2
```

**Run the pipeline:**
```bash
cd "Method - 2"
python agent.py --resume my_resume.txt
python agent.py --resume my_resume.txt --max-jobs 10
python agent.py --resume my_resume.txt --model mistral
python agent.py --resume my_resume.txt --excel custom_jobs.xlsx --out-dir ./results
```

| Argument | Default | Description |
|---|---|---|
| `--resume` | *(required)* | Path to your resume `.txt` file |
| `--excel` | `jobs_output.xlsx` | Pre-scraped job listings from the scraper |
| `--max-jobs` | `20` | Max jobs to process in one run |
| `--model` | `llama3.2` | Ollama model name (also works with `mistral`, `phi3`) |
| `--out-dir` | `tailored_resumes_ollama_5` | Output folder for `.tex` files and results Excel |

**Outputs written to the output folder:**
- `job_001_company_title.tex` — one tailored LaTeX resume per job
- `_results.xlsx` — colour-coded summary with all three metrics per job
- `_manifest.json` — machine-readable results for post-processing

---

## Compiling `.tex` Files to PDF

Method 2 always produces `.tex` files. To compile them to PDF:

```bash
# Compile one file
pdflatex tailored_resumes/job_001_lemon_io_senior_fullstack.tex

# Compile all files in the output folder
for f in tailored_resumes/*.tex; do
  pdflatex -output-directory tailored_resumes "$f"
done
```

---

## Key Results

Evaluated across 20 live-scraped job listings spanning 6 role types (AI/ML Engineer, DevOps, Full-Stack, Product Designer, Sales, Copywriter):

| Metric | Method 1 | Method 2 |
|---|---|---|
| Batch success rate | ~60–70% | **95%** (19/20) |
| Avg cosine sim (in-distribution) | **~0.41** | ~0.41 |
| Avg cosine sim (out-of-distribution) | ~0.10 (defaults to ML content) | ~0.14 |
| ATS keyword coverage (in-distribution) | **8/10** | 6/10 |
| Content richness | **8/10** | 7/10 |
| Out-of-distribution generalization | 3/10 | **8/10** |
| Setup complexity | High (GPU + unsloth + GGUF export) | **Low** (pip install only) |

Method 1 outperforms on roles within its five training categories. Method 2 outperforms on everything else and is the appropriate default for diverse or non-technical job sets.

---

## Limitations

- Both methods are constrained to a 4,096-token context window
- Evaluation was conducted on a single candidate profile — results may vary for different backgrounds
- The LLM self-reported match score has near-zero variance (range 80–85) and should not be used as a primary metric; cosine similarity and TF-IDF ATS score are more reliable
- Neither method has been evaluated on actual ATS callback rates

---

## References

Key works cited in the paper:

- Abdin et al. (2024). *Phi-3 Technical Report*. arXiv:2404.14219
- Dagdelen et al. (2024). *Structured information extraction from scientific text with LLMs*. Nature Communications.
- Dettmers et al. (2023). *QLoRA: Efficient finetuning of quantized LLMs*. NeurIPS.
- Dubey et al. (2024). *The Llama 3 herd of models*. arXiv:2407.21783
- Hu et al. (2022). *LoRA: Low-rank adaptation of large language models*. ICLR.
- Reimers & Gurevych (2019). *Sentence-BERT*. EMNLP.
