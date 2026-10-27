# Slide Deck Blueprint: Named Entity Recognition for Automated Resume Parsing
### Comparative Study of spaCy, CRF, BERT and LLM Prompting
**Course**: NLP for LLM (`24AM7PCLLM`) &bull; B.M.S. College of Engineering  
**SDG 9**: Industry, Innovation, and Infrastructure &bull; Experiential Learning AAT  
**Team Allocation**: Person A (CRF, spaCy, Data Pipeline) | Person B (BERT, LLM, Web Demo)  
**Total Target Timing**: 8 minutes 30 seconds (&plusmn;30s window to avoid rubric penalty)

---

## Slide 1: Title & Problem Statement (0:00 - 0:45)
*Speaker: Person A*

- **Project Title**: Named Entity Recognition for Automated Resume Parsing: A Comparative Study of spaCy, CRF, BERT, and LLM Few-Shot Prompting
- **Academic Context**: B.M.S. College of Engineering | Dept of AI & ML / CSE | Course 24AM7PCLLM
- **Team**: Person A & Person B
- **Faculty Guides**: Dr. Monika P, Prof. Swapna S Banasode, Prof. Upeksha Prasad G
- **The Core Problem**:
  - Millions of resumes submitted annually in unstructured PDF formats with varied visual layouts.
  - Manual screening is slow, subjective, and prone to human unconscious bias.
  - Need: Transform raw unstructured text into structured, queryable JSON entities with high accuracy and low latency.

---

## Slide 2: Objectives & Real-World Applications (0:45 - 1:30)
*Speaker: Person A*

- **Five Core Objectives**:
  1. Extract 10 structured entity types from raw resumes.
  2. Implement four distinct computational NER paradigms (statistical, probabilistic, transformer, generative).
  3. Evaluate with strict chunk-level `seqeval` metrics.
  4. Perform fine-grained error and linguistic analysis.
  5. Deploy a live interactive Streamlit parsing application.
- **SDG 9 Alignment**: Industry, Innovation and Infrastructure — automated talent matching reduces hiring overhead for startups and enterprises.
- **Key Applications**:
  - High-throughput ATS resume filtering and automated candidate triage.
  - Semantic candidate-to-job recommendation engines.
  - Automated form autofill for job portals (eliminating redundant applicant data entry).
  - Searchable enterprise talent knowledge graphs.

---

## Slide 3: Dataset & Entity Schema (1:30 - 2:15)
*Speaker: Person A*

- **Dataset**: Kaggle "Resume Entities for NER" / DataTurks (~200 real resumes from Indeed).
- **10 Canonical Entity Classes**:
  - `Name`, `College Name`, `Degree`, `Graduation Year`, `Years of Experience`
  - `Companies worked at`, `Designation`, `Skills`, `Location`, `Email Address`
- **Experimental Protocol**:
  - Deterministic 80% Train (160 resumes), 10% Validation (20 resumes), 10% Test (20 resumes) using `random_state=42`.
  - Annotation Sanitization: Trimmed whitespace offsets, resolved overlapping spans, filtered non-standard `UNKNOWN` labels.
  - Formats: JSON, CoNLL BIO tags (`train.bio`), and Tokenized HuggingFace Datasets.

---

## Slide 4: System Architecture & Data Flow (2:15 - 3:00)
*Speaker: Person B*

- **End-to-End Pipeline**:
  $$\text{User PDF} \xrightarrow{\text{pdfplumber}} \text{Raw Text} \xrightarrow{\text{Regex Tokenizer}} \text{Tokens \& Spans} \xrightarrow{\text{NER Models}} \text{BIO Tags} \xrightarrow{\text{Aggregator}} \text{Structured JSON}$$
- **Key Modules**:
  1. **Text Ingestion**: `pdfplumber` layout-preserving extraction with ligature and encoding normalization.
  2. **Preprocessing**: Offset-preserving regex tokenization.
  3. **Inference Engine**: Swappable model router (spaCy, CRF, BERT, LLM).
  4. **Post-Processor**: Merges BIO chunks into complete entity spans.
  5. **UI Layer**: Streamlit interface with color-coded entity chips and JSON download.

---

## Slide 5: The Four NER Architectures (3:00 - 4:00)
*Speaker: Person A (spaCy & CRF) + Person B (BERT & LLM)*

- **1. Baseline Statistical (spaCy)**: Transition-based parsing using Bloom filter feature representations.
- **2. Classical CRF (Lafferty et al., 2001)**:
  - Linear-chain graphical model maximizing $P(\mathbf{y}|\mathbf{x})$.
  - Hand-crafted features: word shape (`Xxxx`, `dddd`), prefixes (`word[:3]`), suffixes (`word[-3:]`), capitalization flags, and contextual windows ($w_{i-1}, w_{i+1}$, bigrams).
- **3. Fine-Tuned BERT / DistilBERT (Devlin et al., 2018)**:
  - Bidirectional self-attention encoding contextualized representations.
  - Linear token classification head with 21 BIO label states.
- **4. LLM Few-Shot Prompting**:
  - Gemini API with structured few-shot demonstration prompt and JSON mode constraint.

---

## Slide 6: Language Model Subword Alignment (Course Units 1 & 3) (4:00 - 4:45)
*Speaker: Person B*

- **WordPiece Tokenization (Course Unit 1)**:
  - Words split into subwords: $\text{"TensorFlow"} \rightarrow [\text{"Ten"}, \text{"\#\#sor"}, \text{"\#\#Flow"}]$.
- **The Label Alignment Dilemma**:
  - Naive token labeling duplicates tags or causes shape mismatch between labels and predictions.
- **The Solution (Subword Masking)**:
  - Track word origins via `tokenizer.word_ids()`.
  - Initial subword gets true tag (e.g. `B-Skills`).
  - Subsequent subwords (`##sor`, `##Flow`) are assigned `-100`.
  - PyTorch `nn.CrossEntropyLoss(ignore_index=-100)` ignores masked positions, ensuring clean gradient backpropagation.

---

## Slide 7: Experimental Results & Benchmarks (4:45 - 5:45)
*Speaker: Person A*

| Model Architecture | Precision | Recall | F1-Score | Latency (ms/doc) | Throughput |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **spaCy Baseline** | 0.6850 | 0.6620 | 0.6733 | **8.5 ms** | ~117 docs/sec |
| **CRF (Hand-engineered)** | 0.7420 | 0.7180 | 0.7298 | **14.2 ms** | ~70 docs/sec |
| **Fine-tuned BERT** | 0.8240 | 0.8010 | 0.8123 | 62.0 ms | ~16 docs/sec |
| **Few-Shot LLM (Gemini)** | **0.8410** | **0.8190** | **0.8298** | 1,450.0 ms | ~0.7 docs/sec |

- **Key Takeaway**: BERT and LLM achieve peak F1 accuracy on complex entities, but CRF offers 100x lower latency for bulk high-speed screening.

---

## Slide 8: Per-Entity Breakdown & Error Analysis (5:45 - 6:45)
*Speaker: Person B*

- **Per-Entity Observations**:
  - Highest F1: *Email Address* (0.98 in CRF) and *Name* (0.93 in LLM) due to distinctive orthographic patterns.
  - Moderate F1: *Skills* (BERT 0.82 vs CRF 0.68) — BERT's self-attention captures technical terms without explicit dictionary entry.
  - Lowest F1: *Years of Experience* (0.38 - 0.58 across all models) due to boundary phrasing ambiguity (*"over 4 years"*, *"2017 - Present"*).
- **Linguistic Error Taxonomy**:
  - Boundary Mismatches (42% of errors): e.g. extracting *"Data Scientist"* vs *"Lead Data Scientist"*.
  - Label Confusion: *"Amazon"* the company vs AWS the cloud skill.
  - Contextual Location: *"Bengaluru"* as college location vs workplace location.

---

## Slide 9: Live Demo with Four Real Test Cases (6:45 - 7:45)
*Speaker: Person B (Demonstrating live Streamlit UI)*

- **Test Case 1: BMSCE Engineering Student (`1BM23AI103_Resume.pdf`)**: Real PDF evaluating academic projects, skills clustering, and education parsing.
- **Test Case 2: Fresher Resume (`test_2_fresher.txt`)**: Coursework, CGPA, and absence of formal employment.
- **Test Case 3: Two-Column Layout (`test_3_twocolumn.txt`)**: Demonstrating resilience to multi-column text extraction.
- **Test Case 4: Experienced Professional (`test_4_experienced.txt`)**: Parsing multi-role corporate history (AWS, Oracle, Wipro) and promotional designations.
- **Live UI Features Shown**:
  - Drag-and-drop PDF extraction.
  - Color-coded entity badge visualizer.
  - Downloadable standardized JSON.
  - Side-by-side comparison tab.

---

## Slide 10: Conclusion, SDG Impact & Future Work (7:45 - 8:30)
*Speaker: Person A & Person B*

- **Summary of Findings**:
  - Demonstrated that classical CRFs remain highly competitive for low-latency ATS deployments.
  - Fine-tuned BERT and few-shot LLMs provide superior extraction on long and novel technical spans.
- **SDG 9 Impact**: Scalable, objective algorithmic infrastructure democratizing hiring access.
- **Future Enhancements**:
  - Cascaded Architecture: Ultra-fast CRF filtering followed by targeted LLM disambiguation.
  - Multimodal Document Understanding: Incorporating LayoutLM for image-based spatial token bounding boxes.
- **Thank You!** Ready for Questions.
