"""
resume_generator.py
--------------------
Core generation layer — reads the user's resume PDF, builds a per-job prompt,
calls the local Ollama model, and compiles the LaTeX output to PDF.

All fixed fields (name, phone, email, linkedin, technical skills, certifications,
trainings) are parsed directly from the PDF in Python and injected into the
template. The LLM only generates Education, Experience, and Leadership sections.
This eliminates duplication, missing data, and incorrect extraction entirely.

Dependencies:
    pip install pdfplumber pypdf requests
    sudo apt-get install texlive-latex-extra texlive-fonts-recommended
"""

import re
import subprocess
import time
from pathlib import Path
from typing import Optional

import pdfplumber
import requests


# ── Config ───────────────────────────────────────────────────────────────────
OLLAMA_BASE_URL  = "http://localhost:11434"
FINETUNED_MODEL  = "resume-generator"
FALLBACK_MODEL   = "llama3"
OUTPUT_DIR       = Path("./generated_resumes")
# ─────────────────────────────────────────────────────────────────────────────


# ── LaTeX template ────────────────────────────────────────────────────────────
# Placeholders filled by Python, never by the LLM:
#   %%NAME%%     → candidate full name
#   %%CONTACT%%  → phone | email | linkedin line
#   %%SKILLS%%   → full TECHNICAL SKILLS section
#   %%CERTS%%    → full CERTIFICATIONS/TRAININGS section
#   %%BODY%%     → LLM output (Education, Experience, Leadership only)
LATEX_TEMPLATE = r"""\documentclass[10pt, letterpaper]{article}

\usepackage[top=0.5in, bottom=0.5in, left=0.75in, right=0.75in]{geometry}
\usepackage{enumitem}
\usepackage{titlesec}
\usepackage[hidelinks]{hyperref}
\usepackage{microtype}
\usepackage[T1]{fontenc}
\usepackage{lmodern}

\pagestyle{empty}

% Section headers: bold + horizontal rule underneath
\titleformat{\section}{\bfseries\large}{}{0em}{}[\titlerule]
\titlespacing{\section}{0pt}{8pt}{4pt}

% Tight bullets matching the target resume style
\setlist[itemize]{leftmargin=1.5em, itemsep=1pt, parsep=0pt, topsep=2pt}

% Bold company/org left, italic date right; italic title below
\newcommand{\resumeentry}[3]{%
  \noindent\textbf{#1}\hfill\textit{#2}\\
  \textit{#3}\\[-4pt]
}

% Bold institution left, date right; degree/info below
\newcommand{\eduentry}[3]{%
  \noindent\textbf{#1}\hfill#2\\
  #3\\[-4pt]
}

\begin{document}

\begin{center}
{\Large \textbf{%%NAME%%}}\\[3pt]
%%CONTACT%%
\end{center}

\vspace{2pt}

%%BODY%%

%%SKILLS%%

%%CERTS%%

\end{document}
"""


# ── PDF extraction ────────────────────────────────────────────────────────────

def extract_resume_text(pdf_path: str) -> str:
    """I used pdfplumber as the primary extractor because it preserves layout
    better than pypdf — important for multi-column resume formats. pypdf is the fallback."""
    pdf_path = Path(pdf_path)
    if not pdf_path.exists():
        raise FileNotFoundError(f"Resume PDF not found: {pdf_path}")

    text = ""
    try:
        with pdfplumber.open(str(pdf_path)) as pdf:
            for page in pdf.pages:
                t = page.extract_text()
                if t:
                    text += t + "\n"
        if text.strip():
            return text.strip()
    except Exception as e:
        print(f"[Warning] pdfplumber failed ({e}), trying pypdf ...")

    from pypdf import PdfReader
    reader = PdfReader(str(pdf_path))
    for page in reader.pages:
        text += page.extract_text() or ""
    return text.strip()


def parse_profile_fields(text: str) -> dict:
    """
    I parse every fixed field directly from raw PDF text using regex so the LLM
    never touches them. This is the single fix for all duplication and missing-data bugs.

    Fields extracted:
      - name          : first non-empty line
      - phone         : standard US phone pattern
      - email         : standard email pattern (handles trailing slash from pdfplumber)
      - linkedin_handle / linkedin_url : from linkedin.com/in/... pattern
      - skills_lines  : list of 'Category: skill1, skill2' strings (full block)
      - certifications: everything after 'Certifications:' label, multi-line joined
      - trainings     : everything after 'Trainings:' label, multi-line joined
    """
    lines = [l.strip() for l in text.splitlines() if l.strip()]

    # Name — first non-empty line of the resume
    name = lines[0] if lines else "Candidate Name"

    # Phone — matches 617-555-1234 or (617) 555-1234 or 617.555.1234
    phone_m = re.search(r"(\(?\d{3}\)?[\s.\-]\d{3}[\s.\-]\d{4})", text)
    phone   = phone_m.group(1).strip() if phone_m else ""

    # Email — strip any trailing slash pdfplumber sometimes appends
    email_m = re.search(r"[\w.\-+]+@[\w.\-]+\.[a-zA-Z]{2,}", text)
    email   = email_m.group(0).rstrip("/") if email_m else ""

    # LinkedIn — extract handle from linkedin.com/in/<handle>
    li_m             = re.search(r"linkedin\.com/in/([\w\-]+)", text, re.IGNORECASE)
    linkedin_handle  = li_m.group(1) if li_m else ""
    linkedin_url     = f"https://linkedin.com/in/{linkedin_handle}" if linkedin_handle else ""

    # Technical Skills — extract the full block between TECHNICAL SKILLS and the next
    # all-caps section header. Each line is 'Category: skill1, skill2, ...'
    skills_block_m = re.search(
        r"TECHNICAL SKILLS\s*\n(.*?)(?=\n[A-Z][A-Z/ ]{3,}\n|\nCERTIFICATIONS|\nLEADERSHIP|\Z)",
        text, re.DOTALL | re.IGNORECASE
    )
    if skills_block_m:
        raw_skills = skills_block_m.group(1).strip()
        # Each line is one skill category — preserve them as-is
        skills_lines = [l.strip() for l in raw_skills.splitlines() if l.strip()]
    else:
        skills_lines = []

    # Certifications — grab everything after 'Certifications:' up to 'Trainings:'
    # Multi-line: pdfplumber wraps long lines so we join with a space
    cert_m = re.search(
        r"Certifications?\s*[:\-]\s*(.*?)(?=\nTrainings?|\Z)",
        text, re.DOTALL | re.IGNORECASE
    )
    certifications = re.sub(r"\s+", " ", cert_m.group(1)).strip().rstrip(";,") if cert_m else ""

    # Trainings — grab everything after 'Trainings:' up to the next section or end
    train_m = re.search(
        r"Trainings?\s*[:\-]\s*(.*?)(?=\n[A-Z]{2,}[A-Z /]*\n|\Z)",
        text, re.DOTALL | re.IGNORECASE
    )
    trainings = re.sub(r"\s+", " ", train_m.group(1)).strip().rstrip(";,") if train_m else ""

    return {
        "name":            name,
        "phone":           phone,
        "email":           email,
        "linkedin_handle": linkedin_handle,
        "linkedin_url":    linkedin_url,
        "skills_lines":    skills_lines,
        "certifications":  certifications,
        "trainings":       trainings,
    }


def build_contact_line(fields: dict) -> str:
    """Builds the centered contact line as LaTeX — phone | email | linkedin."""
    parts = []
    if fields["phone"]:
        parts.append(fields["phone"])
    if fields["email"]:
        parts.append(rf"\href{{mailto:{fields['email']}}}{{{fields['email']}}}")
    if fields["linkedin_url"]:
        parts.append(
            rf"\href{{{fields['linkedin_url']}}}{{linkedin.com/in/{fields['linkedin_handle']}}}"
        )
    return " | ".join(parts) if parts else ""


def build_skills_block(fields: dict) -> str:
    """
    Builds the TECHNICAL SKILLS section directly from parsed PDF lines.
    Each line looks like 'Category: skill1, skill2' — I bold the category label
    and escape & characters so LaTeX doesn't choke on them.
    """
    if not fields["skills_lines"]:
        return ""

    lines_latex = []
    for line in fields["skills_lines"]:
        # Escape ampersands for LaTeX
        line = line.replace("&", r"\&")
        # Split on first colon to separate category from skills
        if ":" in line:
            category, skills = line.split(":", 1)
            lines_latex.append(
                rf"\noindent\textbf{{{category.strip()}:}} {skills.strip()}\\"
            )
        else:
            lines_latex.append(rf"\noindent {line}\\")

    return r"\section{TECHNICAL SKILLS}" + "\n" + "\n".join(lines_latex)


def build_certs_block(fields: dict) -> str:
    """Builds the CERTIFICATIONS/TRAININGS section directly from parsed PDF data."""
    if not fields["certifications"] and not fields["trainings"]:
        return ""

    lines = [r"\section{CERTIFICATIONS/TRAININGS}"]
    if fields["certifications"]:
        # Escape & in cert text
        cert_text = fields["certifications"].replace("&", r"\&")
        lines.append(rf"\noindent\textbf{{Certifications:}} {cert_text}\\")
    if fields["trainings"]:
        train_text = fields["trainings"].replace("&", r"\&")
        lines.append(rf"\noindent\textbf{{Trainings:}} {train_text}\\")
    return "\n".join(lines)


# ── Ollama ────────────────────────────────────────────────────────────────────

def get_ollama_model() -> str:
    """I query Ollama's /api/tags to check if the fine-tuned model exists —
    if not, I fall back to llama3 automatically so the pipeline always runs."""
    try:
        resp = requests.get(f"{OLLAMA_BASE_URL}/api/tags", timeout=5)
        available = [m["name"].split(":")[0] for m in resp.json().get("models", [])]
        if FINETUNED_MODEL in available:
            print(f"[Generator] Using fine-tuned model: {FINETUNED_MODEL}")
            return FINETUNED_MODEL
        print(f"[Generator] Fine-tuned model not found, falling back to: {FALLBACK_MODEL}")
        return FALLBACK_MODEL
    except Exception:
        return FALLBACK_MODEL


# The LLM is told to generate ONLY Education, Experience, and Leadership.
# Skills and Certifications are already handled by Python above.
SYSTEM_PROMPT = r"""You are an expert resume writer and LaTeX typesetter.

You will generate ONLY three sections of a resume in LaTeX body content.
The document header, technical skills, and certifications are already handled
separately — do NOT include them.

The template already defines \resumeentry and \eduentry. Do NOT output
\documentclass, \usepackage, \begin{document}, \end{document}, or any preamble.

GENERATE EXACTLY THESE THREE SECTIONS IN ORDER:

1. EDUCATION
\section{EDUCATION}
\eduentry{University Name, City, State}{Month Year}{Degree in Field \quad GPA: X.X}
\noindent Relevant Coursework: Course 1, Course 2\\[4pt]
\eduentry{College Name, City, Country}{Month Year}{Degree in Field}
\noindent Honors: description\\

2. PROFESSIONAL EXPERIENCE
\section{PROFESSIONAL EXPERIENCE}
\resumeentry{Company Name, City, Country}{Month Year -- Month Year}{Job Title}
\begin{itemize}
  \item Achievement bullet mirroring job description keywords
  \item Quantified result where numbers exist in the profile
\end{itemize}

3. LEADERSHIP EXPERIENCE (only if present in the candidate profile — skip entirely if not)
\section{LEADERSHIP EXPERIENCE}
\resumeentry{Organization, City, State}{Date Range}{Role Title}
\begin{itemize}
  \item Leadership bullet
\end{itemize}

STRICT RULES:
- Generate ONLY the 3 sections above. Nothing else.
- Do NOT write a name, contact info, skills section, or certifications section.
- Mirror keywords from the job description naturally in experience bullets.
- Quantify achievements wherever the profile provides numbers.
- Escape special characters: \$ for $,  \% for %,  \& for &
- Return ONLY the LaTeX. No explanations. No markdown fences.
"""


def build_prompt(profile_text: str, job: dict) -> str:
    """I truncate both the job description and the profile to stay within
    Ollama's 4096-token context window while keeping the most useful content."""
    tags_str   = ", ".join(job.get("tags", [])[:8])
    jd_snippet = job["description"][:1500]

    return (
        f"### Job Description\n"
        f"Title   : {job['title']}\n"
        f"Company : {job['company']}\n"
        f"Location: {job['location']}\n"
        f"Tags    : {tags_str}\n\n"
        f"{jd_snippet}\n\n"
        f"### Candidate Profile\n"
        f"{profile_text[:2000]}\n\n"
        f"### Task\n"
        f"Generate ONLY the Education, Professional Experience, and Leadership "
        f"sections (if Leadership exists in the profile). "
        f"Do NOT write name, contact, skills, or certifications. Return ONLY the LaTeX."
    )


def call_ollama(model: str, system: str, user_prompt: str, timeout: int = 120) -> str:
    """I set temperature=0.2 — lower than usual because format compliance
    matters more than creativity for LaTeX resume generation."""
    payload = {
        "model":  model,
        "stream": False,
        "options": {
            "temperature": 0.2,
            "top_p":       0.9,
            "num_ctx":     4096,
            "num_predict": 2048,
        },
        "messages": [
            {"role": "system", "content": system},
            {"role": "user",   "content": user_prompt},
        ],
    }
    try:
        resp = requests.post(f"{OLLAMA_BASE_URL}/api/chat", json=payload, timeout=timeout)
        resp.raise_for_status()
        return resp.json()["message"]["content"].strip()
    except requests.exceptions.Timeout:
        print(f"[Ollama] Timed out after {timeout}s")
        return ""
    except Exception as e:
        print(f"[Ollama] Error: {e}")
        return ""


# ── Assembly ──────────────────────────────────────────────────────────────────

def clean_llm_output(raw: str) -> str:
    """Strip markdown fences and any preamble/header the model added by mistake."""
    raw = re.sub(r"```(?:latex|tex)?\n?", "", raw)
    raw = re.sub(r"```", "", raw).strip()
    raw = re.sub(r"\\documentclass.*?\n", "", raw)
    raw = re.sub(r"\\usepackage.*?\n", "", raw)
    raw = re.sub(r"\\pagestyle.*?\n", "", raw)
    raw = re.sub(r"\\begin\{document\}", "", raw)
    raw = re.sub(r"\\end\{document\}", "", raw)
    # Remove any stray header block the model added anyway
    raw = re.sub(r"\\begin\{center\}.*?\\end\{center\}", "", raw, flags=re.DOTALL)
    # Remove any skills or certifications sections the model added despite instructions
    raw = re.sub(r"\\section\{TECHNICAL SKILLS\}.*?(?=\\section|\Z)", "", raw, flags=re.DOTALL)
    raw = re.sub(r"\\section\{CERTIFICATIONS.*?\}.*?(?=\\section|\Z)", "", raw, flags=re.DOTALL)
    return raw.strip()


def assemble_latex(llm_body: str, fields: dict) -> str:
    """
    I assemble the final LaTeX by filling 5 separate placeholders from distinct sources:
      %%NAME%%     — Python-parsed from PDF (never LLM)
      %%CONTACT%%  — Python-built from parsed phone/email/linkedin
      %%BODY%%     — LLM output (Education, Experience, Leadership only)
      %%SKILLS%%   — Python-parsed and formatted from PDF skills block
      %%CERTS%%    — Python-parsed from PDF certifications/trainings
    """
    body = clean_llm_output(llm_body)

    latex = LATEX_TEMPLATE
    latex = latex.replace("%%NAME%%",    fields["name"])
    latex = latex.replace("%%CONTACT%%", build_contact_line(fields))
    latex = latex.replace("%%BODY%%",    body)
    latex = latex.replace("%%SKILLS%%",  build_skills_block(fields))
    latex = latex.replace("%%CERTS%%",   build_certs_block(fields))
    return latex


def has_required_sections(latex: str) -> bool:
    """Lightweight sanity check before wasting time trying to compile bad LaTeX."""
    return all(tag in latex for tag in [r"\documentclass", r"\begin{document}", r"\end{document}"])


# ── Compile ───────────────────────────────────────────────────────────────────

def compile_latex_to_pdf(latex: str, output_path: Path) -> bool:
    """I run pdflatex twice — the second pass resolves cross-references the first
    pass leaves pending. Cleans up .aux/.log/.out on success."""
    tex_path = output_path.with_suffix(".tex")
    tex_path.write_text(latex, encoding="utf-8")

    if subprocess.run(["which", "pdflatex"], capture_output=True).returncode != 0:
        print("[LaTeX] pdflatex not found. Install: sudo apt-get install texlive-latex-extra")
        print(f"[LaTeX] .tex saved: {tex_path}")
        return False

    compile_cmd = [
        "pdflatex", "-interaction=nonstopmode",
        "-output-directory", str(output_path.parent),
        str(tex_path),
    ]

    print(f"[LaTeX] Compiling {tex_path.name} ...")
    for _ in range(2):  # run twice for stable output
        subprocess.run(compile_cmd, capture_output=True, text=True, cwd=str(output_path.parent))

    pdf_path = output_path.with_suffix(".pdf")
    if pdf_path.exists():
        for ext in [".aux", ".log", ".out"]:
            aux = output_path.with_suffix(ext)
            if aux.exists():
                aux.unlink()
        print(f"[LaTeX] ✓ PDF created: {pdf_path}")
        return True

    print(f"[LaTeX] ✗ Compilation failed — .tex saved at {tex_path}")
    return False


def safe_filename(s: str, max_len: int = 40) -> str:
    """Converts a job title or company name into a clean, filesystem-safe string."""
    s = re.sub(r"[^\w\s-]", "", s.lower())
    s = re.sub(r"[\s_-]+", "_", s)
    return s[:max_len]


# ── Main batch loop ───────────────────────────────────────────────────────────

def generate_resumes_for_jobs(
    jobs: list[dict],
    resume_pdf_path: str,
    output_dir: Optional[Path] = None,
) -> list[dict]:
    """Main batch loop — runs every job through extract → prompt → generate → compile.
    Each job is handled independently so one failure doesn't stop the rest of the batch."""
    out_dir = output_dir or OUTPUT_DIR
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"[Generator] Extracting profile from: {resume_pdf_path}")
    profile_text = extract_resume_text(resume_pdf_path)
    print(f"[Generator] Profile extracted ({len(profile_text)} chars)")

    # Parse all fixed fields once — reused across every job in the batch
    fields = parse_profile_fields(profile_text)

    # Print a verification summary so you can confirm extraction before generation starts
    print(f"\n[Parsed Fields]")
    print(f"  Name     : {fields['name']}")
    print(f"  Phone    : {fields['phone']}")
    print(f"  Email    : {fields['email']}")
    print(f"  LinkedIn : {fields['linkedin_handle']}")
    print(f"  Skills   : {len(fields['skills_lines'])} categories found")
    for s in fields["skills_lines"]:
        print(f"             • {s}")
    print(f"  Certs    : {fields['certifications'][:80]}{'...' if len(fields['certifications']) > 80 else ''}")
    print(f"  Trainings: {fields['trainings'][:80]}{'...' if len(fields['trainings']) > 80 else ''}")
    print()

    model   = get_ollama_model()
    results = []

    for i, job in enumerate(jobs, 1):
        title   = job.get("title", "Unknown Role")
        company = job.get("company", "Unknown Co")
        print(f"[{i}/{len(jobs)}] {title} @ {company}  ({job.get('source', '')})")

        raw_resp = call_ollama(model, SYSTEM_PROMPT, build_prompt(profile_text, job))

        if not raw_resp:
            print("         ✗ Empty response, skipping.\n")
            results.append({"job": job, "status": "failed", "tex": None, "pdf": None})
            continue

        # Assemble: Python header + LLM body + Python skills + Python certs
        latex = assemble_latex(raw_resp, fields)

        if not has_required_sections(latex):
            print("         ✗ Invalid LaTeX after assembly, skipping.\n")
            results.append({"job": job, "status": "invalid_latex", "tex": None, "pdf": None})
            continue

        out_path = out_dir / f"{safe_filename(title)}_{safe_filename(company)}"
        out_path.with_suffix(".tex").write_text(latex, encoding="utf-8")

        pdf_ok = compile_latex_to_pdf(latex, out_path)
        results.append({
            "job":    job,
            "status": "ok" if pdf_ok else "tex_only",
            "tex":    str(out_path.with_suffix(".tex")),
            "pdf":    str(out_path.with_suffix(".pdf")) if pdf_ok else None,
        })

        print()
        time.sleep(1)  # small gap between Ollama calls to avoid overloading the local server

    return results


def print_summary(results: list[dict]):
    """Prints a clean batch summary grouped by outcome: PDF created, .tex only, failed."""
    ok       = [r for r in results if r["status"] == "ok"]
    tex_only = [r for r in results if r["status"] == "tex_only"]
    failed   = [r for r in results if r["status"] in ("failed", "invalid_latex")]

    print("\n" + "=" * 60)
    print(f"  BATCH SUMMARY  |  Total: {len(results)}  |  PDF: {len(ok)}  |  "
          f".tex only: {len(tex_only)}  |  Failed: {len(failed)}")
    print("=" * 60)
    for r in ok:
        print(f"  ✓ {r['job']['title']} @ {r['job']['company']}")
        print(f"    → {r['pdf']}")
    for r in tex_only:
        print(f"  ~ {r['job']['title']} @ {r['job']['company']}  (compile manually)")
        print(f"    → {r['tex']}")
    for r in failed:
        print(f"  ✗ {r['job']['title']} @ {r['job']['company']}  — {r['status']}")
    print("=" * 60)