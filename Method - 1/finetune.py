"""
finetune.py
-----------
One-time fine-tuning step — trains Phi-3 mini with LoRA on resume examples,
exports as GGUF, and registers it with Ollama as 'resume-generator'.

I chose Phi-3 mini (3.8B) over llama3 for fine-tuning because it fits in
~6GB VRAM with 4-bit QLoRA, trains faster, and exports to a lighter GGUF.

Requirements:
    pip install "unsloth[colab-new] @ git+https://github.com/unslothai/unsloth.git"
    pip install --no-deps trl peft accelerate bitsandbytes datasets
    ollama must be running: `ollama serve`
"""

import json
import os
import subprocess
from pathlib import Path
from datasets import Dataset


# ── Config ──────────────────────────────────────────────────────────────────
BASE_MODEL    = "unsloth/Phi-3-mini-4k-instruct-bnb-4bit"
OUTPUT_DIR    = Path("./phi3_resume_lora")
GGUF_DIR      = Path("./phi3_resume_gguf")
OLLAMA_MODEL  = "resume-generator"

MAX_SEQ_LEN   = 512
USE_4BIT      = True   # keeps VRAM under 6GB during training
LORA_R        = 16
LORA_ALPHA    = 32     # I set alpha = 2 * r, which is the standard convention
LORA_DROPOUT  = 0.05
TRAIN_EPOCHS  = 3
BATCH_SIZE    = 1
GRAD_ACCUM    = 4
LR            = 2e-4
# ────────────────────────────────────────────────────────────────────────────


# ── Training samples ─────────────────────────────────────────────────────────
# I structured these as instruction-input-output triples covering different
# ML/DS roles so the model learns the full range of resume types it might see.
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
]


def build_dataset() -> Dataset:
    # I format each sample using Phi-3's exact chat template (<|user|>...<|end|>)
    # because using the wrong template format causes inconsistent generation at inference time.
    formatted = []
    for s in RESUME_SAMPLES:
        text = (
            f"<|user|>\n"
            f"{s['instruction']}\n\n"
            f"### Job & Profile\n{s['input']}<|end|>\n"
            f"<|assistant|>\n{s['output'].strip()}<|end|>"
        )
        formatted.append({"text": text})
    return Dataset.from_list(formatted)


def train():
    # I used unsloth's FastLanguageModel instead of vanilla HF PEFT because
    # it gives ~2x faster training and 60% less VRAM via custom CUDA kernels.
    from unsloth import FastLanguageModel
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
        # I target all projection layers — attention + MLP — for best instruction-following
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
    # I export as Q4_K_M GGUF — best balance of size (~2GB), speed, and quality for 3.8B.
    # Then I write an Ollama Modelfile and call `ollama create` to register it locally.
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
        print(f"[Export] ✓ Model '{OLLAMA_MODEL}' ready — run: ollama run {OLLAMA_MODEL}")
    else:
        print(f"[Export] ✗ Ollama registration failed:\n{result.stderr}")
        print(f"         Manual: ollama create {OLLAMA_MODEL} -f {modelfile_path}")


if __name__ == "__main__":
    print("=" * 60)
    print("  Phi-3 mini LoRA Fine-Tune for Resume Generation")
    print("=" * 60)
    model, tokenizer = train()
    export_to_ollama(model, tokenizer)
    print(f"\n[Done]  Adapter : {OUTPUT_DIR}")
    print(f"        GGUF    : {GGUF_DIR}")
    print(f"        Ollama  : ollama run {OLLAMA_MODEL}")