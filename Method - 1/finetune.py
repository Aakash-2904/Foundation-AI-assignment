
# I picked Phi-3 mini instead of llama3 mainly bcz it fits in ~6GB VRAM with 4-bit
# also faster to train and easier to export

import json
import os
import subprocess
from pathlib import Path
from datasets import Dataset


#  Config 
BASE_MODEL = "unsloth/Phi-3-mini-4k-instruct-bnb-4bit"
OUTPUT_DIR = Path("./phi3_resume_lora")
GGUF_DIR = Path("./phi3_resume_gguf")
OLLAMA_MODEL = "resume-generator"
DATASET_PATH = Path("./dataset/resume_dataset.csv") #processed the entire 3K+ resumes into resume_Dataset.csv

MAX_SEQ_LEN = 512 # didnt want too long seq (memory issue)
USE_4BIT = True   # helps reduce VRAM
LORA_R = 16
LORA_ALPHA = 32     
LORA_DROPOUT = 0.05
TRAIN_EPOCHS = 3 # not sure if 3 epochs enough, worked ok in test
BATCH_SIZE = 1 # batching is small bcz GPU limit
GRAD_ACCUM = 4 # using grad accum to kinda simulate bigger batch
# might need to tune this later

LR = 2e-4

def load_kaggle_dataset(path: Path) -> list[dict]:
    if not path.exists():
        print(f"[Dataset] Kaggle CSV not found at '{path}' — using hardcoded fallback samples.")
        return []
 
    df = pd.read_csv(path)
    print(f"[Dataset] Loaded {len(df)} rows from '{path}'")
 
    samples = []
    for _, row in df.iterrows():
        if pd.notna(row.get("instruction")) and pd.notna(row.get("output")):
            samples.append({
                "instruction": str(row["instruction"]),
                "input":       str(row.get("input", "")),
                "output":      str(row["output"]),
            })
 
    print(f"[Dataset] {len(samples)} valid instruction-output pairs extracted.")
    return samples


# I made samples in instruction-input-output format, so model learns how to map job desc to resume
# tried to cover diff ML roles (not perfect tho)
RESUME_SAMPLES = [
    {
        "instruction": (
            "You are an expert resume writer. "
            "Given the job description and the candidate's profile, "
            "generate a complete, ATS-optimised LaTeX resume. "
            "Use the moderncv class. Return ONLY valid LaTeX, no explanation."
        ),
        "input": (
            "Job: Senior ML Engineer at TechCorp.\n"
            "Requirements: Python, PyTorch, distributed training, MLOps, "
            "Kubernetes, 5+ years experience.\n\n"
            "Profile: John Doe, john@example.com, "
            "4 years ML engineer at StartupX (PyTorch, TF, AWS), "
            "MSc Computer Science Stanford, "
            "published 2 papers on transformer optimisation, "
            "skills: Python, PyTorch, TensorFlow, Docker, Git."
        ),
        "output": r"""
\documentclass[11pt,a4paper,sans]{moderncv}
\moderncvstyle{banking}
\moderncvcolor{blue}
\usepackage[scale=0.85]{geometry}

\name{John}{Doe}
\email{john@example.com}
\social[linkedin]{johndoe}

\begin{document}
\makecvtitle

\section{Summary}
Results-driven ML Engineer with 4+ years building production PyTorch and
TensorFlow systems. Deep expertise in distributed training and MLOps.
Published researcher in transformer optimisation.

\section{Experience}
\cventry{2020--Present}{ML Engineer}{StartupX}{Remote}{}{
  \begin{itemize}
    \item Designed and deployed PyTorch-based recommender system serving 2M+ users
    \item Reduced model training time by 40\% using distributed training on AWS
    \item Built MLOps pipelines with Docker, improving release cadence 3x
  \end{itemize}
}

\section{Education}
\cventry{2018--2020}{MSc Computer Science}{Stanford University}{}{GPA 3.9/4.0}{}

\section{Publications}
\cvitem{2021}{\textit{Efficient Attention Mechanisms for Large Transformers}, NeurIPS Workshop}

\section{Skills}
\cvitem{ML/DL}{PyTorch, TensorFlow, Hugging Face, scikit-learn}
\cvitem{MLOps}{Docker, Kubernetes, AWS SageMaker, MLflow, DVC}
\cvitem{Languages}{Python (expert), Bash, SQL}

\end{document}
""",
    },
    {
        "instruction": (
            "You are an expert resume writer. "
            "Given the job description and the candidate's profile, "
            "generate a complete, ATS-optimised LaTeX resume. "
            "Use the moderncv class. Return ONLY valid LaTeX, no explanation."
        ),
        "input": (
            "Job: Data Scientist at FinanceAI.\n"
            "Requirements: Python, pandas, SQL, statistical modelling, "
            "XGBoost, communication skills, finance domain knowledge.\n\n"
            "Profile: Jane Smith, jane@email.com, "
            "3 years data scientist at BankCo (Python, pandas, XGBoost), "
            "BSc Statistics MIT, "
            "built fraud detection model saving $2M annually, "
            "skills: Python, R, SQL, pandas, scikit-learn, Tableau."
        ),
        "output": r"""
\documentclass[11pt,a4paper,sans]{moderncv}
\moderncvstyle{banking}
\moderncvcolor{burgundy}
\usepackage[scale=0.85]{geometry}

\name{Jane}{Smith}
\email{jane@email.com}

\begin{document}
\makecvtitle

\section{Summary}
Quantitative Data Scientist with 3 years in financial services, specialising
in fraud detection and risk modelling using Python and XGBoost.

\section{Experience}
\cventry{2021--Present}{Data Scientist}{BankCo}{New York}{}{
  \begin{itemize}
    \item Built XGBoost fraud detection model achieving 97\% precision, saving \$2M annually
    \item Automated reporting pipelines with pandas and SQL, saving 15 hrs/week
    \item Presented findings to C-suite stakeholders via Tableau dashboards
  \end{itemize}
}

\section{Education}
\cventry{2018--2021}{BSc Statistics}{MIT}{}{GPA 3.8/4.0}{}

\section{Skills}
\cvitem{ML/Stats}{XGBoost, scikit-learn, statsmodels, R}
\cvitem{Data}{Python, pandas, NumPy, SQL, Tableau}

\end{document}
""",
    },
    {
        "instruction": (
            "You are an expert resume writer. "
            "Given the job description and the candidate's profile, "
            "generate a complete, ATS-optimised LaTeX resume. "
            "Use the moderncv class. Return ONLY valid LaTeX, no explanation."
        ),
        "input": (
            "Job: Deep Learning Research Engineer at VisionAI.\n"
            "Requirements: PyTorch, computer vision, CNNs, object detection, "
            "CUDA, C++, research background preferred.\n\n"
            "Profile: Alex Lee, alex@ml.io, "
            "2 years research engineer at VisionLab (PyTorch, OpenCV, CUDA), "
            "PhD student Computer Vision CMU, "
            "CVPR 2023 paper on real-time object detection, "
            "skills: Python, C++, PyTorch, OpenCV, CUDA, TensorRT."
        ),
        "output": r"""
\documentclass[11pt,a4paper,sans]{moderncv}
\moderncvstyle{classic}
\moderncvcolor{green}
\usepackage[scale=0.85]{geometry}

\name{Alex}{Lee}
\email{alex@ml.io}
\social[github]{alexlee-cv}

\begin{document}
\makecvtitle

\section{Summary}
Deep Learning Research Engineer with expertise in real-time computer vision.
CVPR-published researcher in object detection. Proficient in end-to-end pipeline
development from CUDA kernels to production TensorRT deployment.

\section{Experience}
\cventry{2022--Present}{Research Engineer}{VisionLab}{Pittsburgh}{}{
  \begin{itemize}
    \item Developed anchor-free object detector achieving 45 FPS on edge hardware
    \item Optimised inference with TensorRT achieving 3x speedup
    \item Maintained OpenCV-based data augmentation library used by 10+ researchers
  \end{itemize}
}

\section{Education}
\cventry{2021--Present}{PhD Computer Science (Computer Vision)}{CMU}{}{}{}

\section{Publications}
\cvitem{CVPR 2023}{\textit{FastDet: Real-Time Object Detection via Sparse Anchors}}

\section{Skills}
\cvitem{DL/CV}{PyTorch, OpenCV, torchvision, MMDetection, Detectron2}
\cvitem{Performance}{CUDA, TensorRT, ONNX, C++ inference}

\end{document}
""",
    },
     {
        "instruction": (
            "You are an expert resume writer. "
            "Given the job description and the candidate's profile, "
            "generate a complete, ATS-optimised LaTeX resume. "
            "Use the moderncv class. Return ONLY valid LaTeX, no explanation."
        ),
        "input": (
            "Job: Senior DevOps Engineer at CloudOps Inc.\n"
            "Requirements: Kubernetes, Terraform, CI/CD pipelines, AWS, "
            "Docker, Prometheus, Grafana, SRE mindset, 4+ years experience.\n\n"
            "Profile: Sam Rivera, sam@devops.io, "
            "3 years DevOps at ScaleUp (Kubernetes, Terraform, Jenkins, AWS), "
            "BSc Computer Engineering UT Austin, "
            "reduced deployment failures by 70% via blue-green deployments, "
            "skills: Kubernetes, Docker, Terraform, AWS, Jenkins, Prometheus, Python, Bash."
        ),
        "output": r"""
\documentclass[11pt,a4paper,sans]{moderncv}
\moderncvstyle{banking}
\moderncvcolor{blue}
\usepackage[scale=0.85]{geometry}
 
\name{Sam}{Rivera}
\email{sam@devops.io}
\social[github]{samrivera-devops}
 
\begin{document}
\makecvtitle
 
\section{Summary}
Senior DevOps Engineer with 3+ years building resilient CI/CD pipelines and
cloud-native infrastructure on AWS. Reduced deployment failures by 70\% through
blue-green deployments and automated rollback strategies.
 
\section{Experience}
\cventry{2021--Present}{DevOps Engineer}{ScaleUp}{Austin, TX}{}{
  \begin{itemize}
    \item Orchestrated 200+ microservices on Kubernetes, achieving 99.95\% uptime SLA
    \item Automated infrastructure provisioning with Terraform, cutting setup time by 60\%
    \item Built Prometheus and Grafana observability stack, reducing MTTR from 45 to 8 minutes
  \end{itemize}
}
 
\section{Education}
\cventry{2017--2021}{BSc Computer Engineering}{UT Austin}{}{}{}
 
\section{Skills}
\cvitem{Orchestration}{Kubernetes, Helm, Docker, containerd}
\cvitem{IaC}{Terraform, Ansible, CloudFormation}
\cvitem{Cloud}{AWS (EC2, EKS, RDS, S3), IAM}
\cvitem{CI/CD}{Jenkins, GitHub Actions, ArgoCD}
\cvitem{Observability}{Prometheus, Grafana, ELK Stack}
 
\end{document}
""",
    },
     {
        "instruction": (
            "You are an expert resume writer. "
            "Given the job description and the candidate's profile, "
            "generate a complete, ATS-optimised LaTeX resume. "
            "Use the moderncv class. Return ONLY valid LaTeX, no explanation."
        ),
        "input": (
            "Job: Full-Stack Developer at ProductCo.\n"
            "Requirements: React, Node.js, TypeScript, PostgreSQL, REST APIs, "
            "AWS deployment, agile team experience.\n\n"
            "Profile: Maya Chen, maya@dev.io, "
            "2 years full-stack at WebAgency (React, Node.js, PostgreSQL, AWS), "
            "BSc Software Engineering Georgia Tech, "
            "built e-commerce platform serving 50K monthly users, "
            "skills: React, TypeScript, Node.js, PostgreSQL, Redis, Docker, AWS."
        ),
        "output": r"""
\documentclass[11pt,a4paper,sans]{moderncv}
\moderncvstyle{banking}
\moderncvcolor{burgundy}
\usepackage[scale=0.85]{geometry}
 
\name{Maya}{Chen}
\email{maya@dev.io}
\social[github]{mayachen-dev}
 
\begin{document}
\makecvtitle
 
\section{Summary}
Full-Stack Developer with 2 years building React and Node.js applications at
production scale. Delivered e-commerce platform serving 50K monthly users with
sub-200ms REST API response times on AWS.
 
\section{Experience}
\cventry{2022--Present}{Full-Stack Developer}{WebAgency}{Atlanta, GA}{}{
  \begin{itemize}
    \item Built React/TypeScript storefront processing \$2M+ in annual transactions
    \item Designed Node.js REST API backed by PostgreSQL; query latency reduced by 45\%
    \item Deployed containerised stack on AWS ECS with zero-downtime rolling updates
  \end{itemize}
}
 
\section{Education}
\cventry{2018--2022}{BSc Software Engineering}{Georgia Tech}{}{GPA 3.7/4.0}{}
 
\section{Skills}
\cvitem{Frontend}{React, TypeScript, Next.js, Tailwind CSS}
\cvitem{Backend}{Node.js, Express, REST, GraphQL}
\cvitem{Data}{PostgreSQL, Redis, Prisma ORM}
\cvitem{DevOps}{Docker, AWS (ECS, RDS, S3), GitHub Actions}
 
\end{document}
""",
    },
]


def build_dataset() -> Dataset:
# I used Phi-3 chat format here (important for output)
# wrong format was giving weird outputs earlier so fixed it   
 formatted = []
 
    # prefer the full Kaggle dataset if it's available locally
    samples = load_kaggle_dataset(DATASET_PATH)
    if not samples:
        # CSV not found — use the 5 hardcoded fallback examples instead
        samples = RESUME_SAMPLES_FALLBACK
        print(f"[Dataset] Using {len(samples)} hardcoded fallback examples (one per category).")
 
    for s in samples:
        text = (
            f"<|user|>\n"
            f"{s['instruction']}\n\n"
            f"### Job & Profile\n{s['input']}<|end|>\n"
            f"<|assistant|>\n{s['output'].strip()}<|end|>"
        )
        formatted.append({"text": text})
 
    print(f"[Dataset] Final training set size: {len(formatted)} examples")
    return Dataset.from_list(formatted)


def train():
# using unsloth model instead of HF PEFT
# felt faster + uses less VRAM in my runs  
    from trl import SFTTrainer
    from transformers import TrainingArguments

    print("[FineTune] Loading base model ...")
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name     = BASE_MODEL,
        max_seq_length = MAX_SEQ_LEN,
        dtype          = None,      # auto-detects bf16/fp16
        load_in_4bit   = USE_4BIT,
    )

    print("[FineTune] Attaching LoRA adapters ...")
    model = FastLanguageModel.get_peft_model(
        model,
        r               = LORA_R,
        # we target most layers (attention + mlp) so it learns better
        target_modules  = ["q_proj", "k_proj", "v_proj", "o_proj",
                           "gate_proj", "up_proj", "down_proj"],
        lora_alpha      = LORA_ALPHA,
        lora_dropout    = LORA_DROPOUT,
        bias            = "none",
        use_gradient_checkpointing = "unsloth",
        random_state    = 42,
    )

    dataset = build_dataset()
    print(f"[FineTune] Training samples: {len(dataset)}")

    trainer = SFTTrainer(
        model              = model,
        tokenizer          = tokenizer,
        train_dataset      = dataset,
        dataset_text_field = "text",
        max_seq_length     = MAX_SEQ_LEN,
        args = TrainingArguments(
            per_device_train_batch_size = BATCH_SIZE,
            gradient_accumulation_steps = GRAD_ACCUM,
            num_train_epochs            = TRAIN_EPOCHS,
            learning_rate               = LR,
            fp16                        = True,
            logging_steps               = 5,
            output_dir                  = str(OUTPUT_DIR),
            save_strategy               = "epoch",
            report_to                   = "none",
        ),
    )

    print("[FineTune] Starting training ...")
    trainer.train()

    print(f"[FineTune] Saving adapter to {OUTPUT_DIR} ...")
    model.save_pretrained(str(OUTPUT_DIR))
    tokenizer.save_pretrained(str(OUTPUT_DIR))

    return model, tokenizer


def export_to_ollama(model, tokenizer):
    # we are exporting this model into our ollama
    GGUF_DIR.mkdir(parents=True, exist_ok=True)
    gguf_path = GGUF_DIR / "resume-generator.Q4_K_M.gguf"

    print(f"[Export] Saving GGUF to {gguf_path} ...")
    model.save_pretrained_gguf(
        str(GGUF_DIR / "resume-generator"),
        tokenizer,
        quantization_method = "q4_k_m",
    )

    modelfile_path = GGUF_DIR / "Modelfile"
    modelfile_path.write_text(f"""FROM {gguf_path}

SYSTEM \"\"\"
You are an expert resume writer. Given a job description and a candidate profile,
generate a complete, ATS-optimised LaTeX resume using moderncv.
Return ONLY valid LaTeX. No explanations.
\"\"\"

PARAMETER temperature 0.3
PARAMETER top_p 0.9
PARAMETER num_ctx 4096
""")

    print(f"[Export] Registering '{OLLAMA_MODEL}' with Ollama ...")
    result = subprocess.run(
        ["ollama", "create", OLLAMA_MODEL, "-f", str(modelfile_path)],
        capture_output=True, text=True,
    )
    if result.returncode == 0:
        print(f"[Export]Model '{OLLAMA_MODEL}' ready — run: ollama run {OLLAMA_MODEL}")
    else:
        print(f"[Export]Ollama registration failed:\n{result.stderr}")
        print(f"         Manual: ollama create {OLLAMA_MODEL} -f {modelfile_path}")


if __name__ == "__main__":
    print("  Phi-3 mini LoRA Fine-Tune for Resume Generation")
    model, tokenizer = train()
    export_to_ollama(model, tokenizer)
    print(f"\n[Done]  Adapter : {OUTPUT_DIR}")
    print(f"GGUF    : {GGUF_DIR}")
    print(f"Ollama  : ollama run {OLLAMA_MODEL}")
