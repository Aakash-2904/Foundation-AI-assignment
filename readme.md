# Resume Generator using LLM Fine-Tuning + Agent Pipeline

## Overview

In this project, we tried to build a system that can generate **tailored resumes** based on job descriptions.

We explored **2 approaches**:

1. Fine-tuning a small LLM (Phi-3 mini)
2. Agent-based pipeline using embeddings + prompts

The goal was to see which method gives better personalization and output quality.

---

## What we did

* scraped job descriptions from APIs
* extracted info from an existing resume (PDF)
* generated tailored resumes using:

  * fine-tuned model (Method 1)
  * agent pipeline (Method 2)
* compared outputs using similarity scores

---

## Project Structure

```
.
├── method_1/
│   ├── finetune.py
│   ├── main.py
│   ├── resume_generator.py
│   └── scraper.py
│
├── method_2/
│   └── agent.py
│
├── requirements.txt
└── README.md
```

---

## Method 1: Fine-Tuning

We fine-tuned **Phi-3 mini (4-bit)** using LoRA.

### Why this model?

* fits in low VRAM (~6GB)
* faster training
* smaller output model

### What we did

* created instruction-input-output samples
* used LoRA for efficient training
* exported model for inference

---

## Method 2: Agent Pipeline

Instead of training, we:

* used embeddings (MiniLM)
* matched resume + job description
* generated structured outputs using prompts

This is more flexible and easier to debug.

---

## How to Run

### 1. Install dependencies

```
pip install -r requirements.txt
```

---

### 2. Run Method 1

```
cd method_1
python main.py
```

---

### 3. Run Method 2

```
cd method_2
python agent.py
```

---

## Notes

* I kept configs small bcz of GPU limits
* scraping is basic (might break for some sites)
* PDF parsing is not perfect, but works for most resumes
* prompts can still be improved

---

## Future Improvements

* better dataset for fine-tuning
* improve prompt quality
* add more job sources
* better resume formatting
* add evaluation metrics

---

## Conclusion

* fine-tuning works well but needs data + time
* agent approach is easier and more flexible
* both have pros/cons depending on use case

---

## Author

Aakash Kumar
Riya nitin taori
MS AI @ Northeastern University

---
