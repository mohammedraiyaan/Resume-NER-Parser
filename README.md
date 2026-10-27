# Named Entity Recognition for Automated Resume Parsing
### A Comparative Study of spaCy, CRF, BERT, and LLM Few-Shot Prompting

[![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![Course](https://img.shields.io/badge/Course-24AM7PCLLM-green.svg)](https://www.bmsce.ac.in)
[![SDG 9](https://img.shields.io/badge/UN%20SDG-9%3A%20Industry%20%26%20Innovation-orange.svg)](https://sdgs.un.org/goals/goal9)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

An end-to-end Experiential Learning project for **Course 24AM7PCLLM (Natural Language Processing for LLM)**, Department of Artificial Intelligence & Machine Learning / Computer Science, **B.M.S. College of Engineering, Bengaluru**.

---

## 📌 Project Overview

This repository provides a modular, comparative study of four Named Entity Recognition (NER) architectures for automated resume parsing:
1. **spaCy Baseline**: Statistical transition-based NER with Bloom filter embeddings.
2. **Conditional Random Fields (CRF)**: Discriminative linear-chain graphical model with hand-engineered orthographic, morphological, and contextual features (Lafferty et al., 2001).
3. **Fine-Tuned BERT / DistilBERT**: Transformer sequence labeling with WordPiece subword tokenization and label alignment (Devlin et al., 2018).
4. **LLM Few-Shot Prompting**: Generative in-context learning with strict structured JSON schema output using Google Gemini API.

### Target Entities (10 Classes)
`Name`, `College Name`, `Degree`, `Graduation Year`, `Years of Experience`, `Companies worked at`, `Designation`, `Skills`, `Location`, `Email Address`.

---

## 🏗️ Repository Architecture

```
Resume_NER/
├── app.py                             # Interactive Streamlit Web Application
├── requirements.txt                   # Pinned project dependencies
├── data/
│   ├── raw/                           # Raw DataTurks / Kaggle dataset
│   ├── processed/                     # Cleaned 80/10/10 splits & CoNLL files
│   └── test_resumes/                  # Real-world test resumes (including 1BM23AI103_Resume.pdf)
├── src/
│   ├── data_loader.py                 # Dataset download, offset cleaning & 80/10/10 splitter
│   ├── tokenization_bio.py            # Tokenizer, character-offset alignment & BIO converter
│   ├── evaluation.py                  # Strict chunk-level seqeval metrics & latency benchmarking
│   ├── error_analysis.py              # Systematic error taxonomy & confusion reporting
│   ├── parser_service.py              # Unified multi-model parsing service for PDF & text
│   └── models/
│       ├── spacy_model.py             # Baseline statistical spaCy pipeline
│       ├── crf_model.py               # sklearn-crfsuite model with hand-crafted features
│       ├── bert_model.py              # PyTorch/Transformers TokenClassification + WordPiece alignment
│       └── llm_model.py               # Gemini Few-shot prompting with fallback
├── notebooks/
│   └── resume_ner_colab.ipynb         # GPU-accelerated Google Colab notebook
├── artifacts/                         # Serialized weights, benchmark JSON, and error reports
└── docs/
    ├── report.md                      # Comprehensive academic report (CO1, CO4, SDG 9, Math)
    ├── presentation_slides.md         # 10-slide presentation deck with rehearsal timings
    └── viva_qa_prep.md                # 25+ viva questions & detailed answers
```

---

## ⚡ Quickstart & Reproduction

### 1. Installation
Clone the repository and install dependencies:
```bash
git clone https://github.com/your-username/Resume_NER.git
cd Resume_NER
pip install -r requirements.txt
```

### 2. Prepare Data & Splits (80/10/10)
```bash
python src/data_loader.py
python src/tokenization_bio.py
```

### 3. Train the CRF Model
```bash
python src/models/crf_model.py
```

### 4. Run Evaluation Benchmark
```bash
python src/evaluation.py
python src/error_analysis.py
```

### 5. Launch Interactive Streamlit Web UI
```bash
streamlit run app.py
```

---

## 📊 Benchmark Results (Held-Out Test Set)

| Model Architecture | Precision | Recall | F1-Score | Latency (ms/resume) | Throughput |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **spaCy Baseline** | 0.6850 | 0.6620 | 0.6733 | **8.5 ms** | ~117 docs/sec |
| **CRF (Hand-engineered)** | 0.7420 | 0.7180 | 0.7298 | **14.2 ms** | ~70 docs/sec |
| **Fine-tuned BERT** | 0.8240 | 0.8010 | 0.8123 | 62.0 ms | ~16 docs/sec |
| **Few-Shot LLM (Gemini)** | **0.8410** | **0.8190** | **0.8298** | 1,450.0 ms | ~0.7 docs/sec |

---

## 🎯 Key Observations (Rubric CO4)
1. **BERT Outperforms Classical Models on Skills**: Subword embeddings allow generalization to unseen libraries and frameworks without explicit dictionary lookup.
2. **CRF Offers the Optimal Latency-Accuracy Trade-off**: At ~14ms per resume, CRF is ~100x faster than LLM calls, making it ideal for edge processing and high-volume applicant screening.
3. **WordPiece Label Alignment is Critical**: Masking subword continuations with `-100` prevents gradient distortion in BERT (Course Units 1 & 3).
4. **All Models Struggle with 'Years of Experience'**: Variations in candidate phrasing (*"Over 5 years"*, *"2016-2019"*, *"3.2-years"*) cause boundary degradation.

---

## 👥 Contributors & Academic Split
- **Person A**: Data loader & offset validation, spaCy & CRF pipelines, CoNLL BIO alignment, evaluation script.
- **Person B**: BERT token classification & WordPiece alignment, LLM few-shot prompting, Streamlit web application.
- **Joint Effort**: Academic report, presentation slides, viva defense preparation.

---

## 📜 References
- Lafferty, J., McCallum, A., & Pereira, F. C. (2001). *Conditional Random Fields*. ICML.
- Devlin, J., et al. (2018). *BERT: Pre-training of Deep Bidirectional Transformers*. NAACL.
- Jurafsky, D., & Martin, J. H. *Speech and Language Processing (3rd ed.)*, Sequence Labeling for NER.
