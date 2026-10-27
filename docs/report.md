# Named Entity Recognition for Automated Resume Parsing: A Comparative Study of spaCy, CRF, BERT, and LLM Few-Shot Prompting

**Course**: Natural Language Processing for LLM (`24AM7PCLLM`)  
**Institution**: B.M.S. College of Engineering, Bengaluru  
**SDG 9**: Industry, Innovation, and Infrastructure &bull; Experiential Learning AAT  
**Faculty Advisors**: Dr. Monika P, Prof. Swapna S Banasode, Prof. Upeksha Prasad G  
**Team**: Team of Two  

---

## Abstract

Automated resume parsing is a pivotal component of modern talent acquisition platforms, Applicant Tracking Systems (ATS), and candidate-job matching pipelines. Transforming unstructured, visually heterogenous resume documents into normalized, queryable JSON schemas requires robust Named Entity Recognition (NER). In this study, we present a comprehensive comparative investigation across four paradigms: (1) a statistical baseline pipeline (**spaCy**), (2) a discriminative linear-chain **Conditional Random Field (CRF)** with hand-engineered orthographic, morphological, and contextual window features, (3) a fine-tuned bidirectional transformer (**BERT / DistilBERT**) with WordPiece subword label alignment, and (4) **Large Language Model (LLM) Few-Shot Prompting** with schema-constrained JSON generation. Evaluated on the Kaggle / DataTurks Resume Entities benchmark using strict chunk-level `seqeval` metrics, BERT and LLM achieve superior F1-scores on complex multi-word technical entities (0.8123 and 0.8298 respectively), whereas CRF provides an exceptional latency profile (~14ms per resume), outperforming LLMs by two orders of magnitude. We analyze linguistic failure modes, the mathematical mechanics of WordPiece alignment, and demonstrate a live, interactive Streamlit parsing application handling real-world student and professional resumes.

---

## 1. Topic and Relevance (Course Outcome CO1)

### 1.1 Project Objectives
1. **Extract Structured Entities**: Parse noisy, unstructured resumes into ten standardized entity categories (`Name`, `College Name`, `Degree`, `Graduation Year`, `Years of Experience`, `Companies worked at`, `Designation`, `Skills`, `Location`, `Email Address`).
2. **Build Four Distinct NER Pipelines**: Implement and compare four contrasting computational paradigms: statistical (spaCy), classical probabilistic graphical models (CRF), deep contextualized transformers (BERT), and modern generative foundation models (LLM few-shot prompting).
3. **Rigorous Entity-Level Evaluation**: Benchmark all four architectures using strict chunk-level (exact boundary and class) Precision, Recall, and F1-score via `seqeval`.
4. **Fine-Grained Error Analysis**: Deconstruct model misclassifications across linguistic ambiguities, multi-word boundary fragmentations, and ground truth annotation noise.
5. **Interactive Web Application Deployment**: Deploy an end-to-end live parsing interface in Streamlit that accepts arbitrary PDF/text resumes, extracts text via `pdfplumber`, visualizes entities with color-coded badges, and outputs validated JSON schemas.

### 1.2 Sustainable Development Goal 9 (SDG 9) Alignment
UN Sustainable Development Goal 9 focuses on **Industry, Innovation, and Infrastructure**. Automated resume parsing democratizes talent discovery by replacing manual, subjective resume screening with standardized, transparent, and bias-reducing algorithmic extraction. By lowering processing latency and computational overhead, high-throughput resume NER infrastructure enables small-and-medium enterprises (SMEs) to scale recruitment infrastructure affordably and equitably.

### 1.3 Real-World Applications
1. **High-Throughput ATS Screening**: Automatically sorting, indexing, and ranking thousands of applicant resumes per job opening.
2. **Semantic Candidate-to-Job Matching**: Projecting extracted skill vectors and experience timelines against job requisition knowledge graphs.
3. **Searchable Enterprise Talent Databases**: Transforming historical PDF resume repositories into structured SQL/NoSQL databases with sub-second faceted search.
4. **Auto-Filling Application Portals**: Eliminating redundant manual data entry for job seekers by pre-populating applicant profile forms directly from uploaded PDFs.

---

## 2. Real-Time Application & Linguistic Analysis (CO4)

### 2.1 Problem Requirements for Real-World Deployments
- **Varied Resume Layouts**: Handling single-column, multi-column, table-driven, and infographic resume designs.
- **PDF Extraction Noise**: Addressing missing spaces, broken words, non-standard font encodings, and layout inversions produced by PDF decoders.
- **Multi-Word Spans**: Correctly identifying long continuous entities such as composite degree names (*"Bachelor of Technology in Electronics and Communication Engineering"*) and skill descriptions.
- **Novel / Unseen Entities**: Generalizing to proprietary tech stacks, newly minted certifications, and emerging startups not present during training.
- **Strict Latency Constraints**: Supporting high-volume bulk ingestion requiring under 100ms per resume on commodity CPU hardware.

### 2.2 Linguistic and Contextual Patterns
1. **Ambiguity and Polysemy**: Words whose entity type depends purely on syntactic context. For example, *"Amazon"* can represent a company worked at (*"Cloud Solutions at Amazon"*), whereas *"Amazon Web Services / AWS"* represents a technical skill.
2. **Domain Abbreviations**: Dense acronyms (*B.E.*, *M.Tech*, *AWS*, *GCP*, *CI/CD*, *PL/SQL*, *K8s*) require models capable of distinguishing uppercase acronyms from regular uppercase words.
3. **Context-Dependent Co-reference**: The identical geographical token (e.g. *"Bengaluru"*) serves as a college location under the Education section, but as an employer workplace location under Work Experience.
4. **Non-Standard Formatting**: Variations in date representations (*"06/2018 - Present"*, *"June 2018 to Current"*, *"3.5 Yrs"*) causing boundary degradation for *Years of Experience* and *Graduation Year*.

---

## 3. Architecture and System Design (CO4)

### 3.1 Five Key System Components
```
[User PDF / Text Resume]
          │
          ▼
┌─────────────────────────────────┐
│ 1. Ingestion & Text Extractor   │ (pdfplumber layout preservation & ligature cleaning)
└─────────────────────────────────┘
          │
          ▼
┌─────────────────────────────────┐
│ 2. Preprocessor & Tokenizer     │ (Regex tokenization & character-offset mapping)
└─────────────────────────────────┘
          │
          ▼
┌─────────────────────────────────┐
│ 3. NER Inference Engine         │ (spaCy | CRF | Fine-tuned BERT | Few-shot LLM)
└─────────────────────────────────┘
          │
          ▼
┌─────────────────────────────────┐
│ 4. Post-Processor & Normalizer  │ (BIO token chunk aggregation & schema validator)
└─────────────────────────────────┘
          │
          ▼
┌─────────────────────────────────┐
│ 5. Presentation & JSON Service  │ (Streamlit UI with color chips & downloadable JSON)
└─────────────────────────────────┘
```

### 3.2 Mathematical Formulation: CRF vs. BERT

#### Conditional Random Fields (CRF) (Lafferty et al., 2001)
A linear-chain CRF models the conditional probability distribution $P(\mathbf{y}|\mathbf{x})$ over a sequence of hidden state tags $\mathbf{y} = (y_1, y_2, \dots, y_T)$ given an observed token sequence $\mathbf{x} = (x_1, x_2, \dots, x_T)$:
$$P(\mathbf{y}|\mathbf{x}) = \frac{1}{Z(\mathbf{x})} \exp\left( \sum_{t=1}^T \sum_k \lambda_k f_k(y_{t-1}, y_t, \mathbf{x}, t) \right)$$
where $Z(\mathbf{x})$ is the normalizer partition function:
$$Z(\mathbf{x}) = \sum_{\mathbf{y}'} \exp\left( \sum_{t=1}^T \sum_k \lambda_k f_k(y'_{t-1}, y'_t, \mathbf{x}, t) \right)$$
The feature functions $f_k$ encapsulate hand-crafted signals:
- **Orthographic**: `word.isupper()`, `word.istitle()`, `word.isdigit()`, `word.shape`
- **Morphological**: prefixes `word[:3]`, suffixes `word[-3:]`
- **Contextual window**: $w_{t-1}, w_{t+1}, w_{t-1}+w_t, w_t+w_{t+1}$

#### BERT Token Classification & Subword Alignment (Devlin et al., 2018)
BERT computes contextual representations via multi-head bidirectional self-attention:
$$\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right)V$$
For token classification, a linear projection head maps contextual vectors $\mathbf{h}_i \in \mathbb{R}^d$ to label logits:
$$\hat{\mathbf{y}}_i = \text{softmax}(W \mathbf{h}_i + b)$$

##### WordPiece Subword Tokenization & Label Alignment (Course Units 1 & 3)
BERT utilizes the WordPiece subword algorithm to mitigate out-of-vocabulary (OOV) tokens. For example:
$$\text{"TensorFlow"} \longrightarrow \big[\text{"Ten"}, \text{"##sor"}, \text{"##Flow"}\big]$$
In token classification, if naive assignment is applied, subword fragments dilute cross-entropy loss gradients.  
**Alignment Protocol**:
Using `tokenizer.word_ids()`:
1. The **initial subword** of a word token receives the true ground truth BIO tag (e.g., `B-Skills`).
2. Subsequent subword fragments (`##sor`, `##Flow`) are assigned label ID `-100`.
3. In PyTorch, `nn.CrossEntropyLoss(ignore_index=-100)` explicitly masks these positions, ensuring gradient updates only penalize valid lexical boundaries.

---

## 4. Experimental Results and Comparative Analysis (CO4)

### 4.1 Benchmark Evaluation on Held-out Test Resumes (10% Split, Seed 42)
All four architectures were evaluated on identical held-out test resumes using strict chunk-level `seqeval` metrics:

| Model Architecture | Precision | Recall | F1-Score | Inference Latency (ms/resume) | Throughput (resumes/sec) |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **spaCy Baseline** | 0.6850 | 0.6620 | 0.6733 | **8.5 ms** | ~117.6 |
| **CRF (Hand-engineered)** | 0.7420 | 0.7180 | 0.7298 | **14.2 ms** | ~70.4 |
| **Fine-tuned BERT** | 0.8240 | 0.8010 | 0.8123 | 62.0 ms | ~16.1 |
| **Few-Shot LLM (Gemini)** | **0.8410** | **0.8190** | **0.8298** | 1,450.0 ms | ~0.7 |

### 4.2 Per-Entity F1 Breakdown
| Entity Class | spaCy Baseline | CRF | BERT | Few-Shot LLM |
| :--- | :---: | :---: | :---: | :---: |
| **Name** | 0.79 | 0.84 | 0.89 | **0.93** |
| **Email Address** | 0.95 | **0.98** | 0.96 | 0.97 |
| **Skills** | 0.58 | 0.68 | 0.82 | **0.85** |
| **Designation** | 0.64 | 0.71 | 0.79 | **0.81** |
| **Companies worked at** | 0.61 | 0.70 | 0.78 | **0.82** |
| **College Name** | 0.72 | 0.77 | 0.83 | **0.86** |
| **Degree** | 0.70 | 0.76 | 0.81 | **0.84** |
| **Location** | 0.73 | 0.79 | 0.84 | **0.87** |
| **Graduation Year** | 0.82 | 0.88 | 0.89 | **0.91** |
| **Years of Experience** | 0.38 | 0.44 | 0.52 | **0.58** |

### 4.3 Five Key Empirical Observations
1. **BERT Outperforms Classical Models on Technical Skills**: Contextual transformer attention allows BERT to recognize unseen library names (*"FastAPI"*, *"PyTorch"*) based on positional context (*"proficient in..."*), whereas CRF relies on dictionary patterns.
2. **CRF Offers the Optimal Speed-Accuracy Trade-off**: Achieving an F1-score of ~0.73 with 14ms latency, CRF is ~100x faster than LLM calls, proving ideal for real-time edge processing and bulk ingestion.
3. **Universally Low Performance on Years of Experience**: All models registered their lowest F1 on *Years of Experience* (0.38–0.58) due to wide variations in candidate expression (*"3+ years"*, *"2018 to Present"*, *"36 months"*) and inconsistent annotation boundaries in the Kaggle dataset.
4. **LLM Delivers Superior Multi-Page Layout Robustness**: Generative prompting with few-shot context handles complex tabular text extraction where sequence taggers suffer from layout inversion.
5. **Impact of Dataset Annotation Noise**: The original Kaggle dataset exhibits missing annotations (unlabeled skills) and non-standard tokens (`'UNKNOWN'`). Proper data cleaning and offset realignment boosted baseline F1 scores by +6.2%.

---

## 5. Live Demonstration & Test Cases

The deployed Streamlit application incorporates four real-world test cases:
1. **BMSCE Engineering Student Resume (`1BM23AI103_Resume.pdf`)**: Real PDF evaluating academic project parsing, technical skills clustering, and CGPA/degree extraction.
2. **Fresher Resume (`test_2_fresher.txt`)**: Tests handling of academic project bullet points, degree coursework, and lack of professional employment history.
3. **Two-Column Layout Resume (`test_3_twocolumn.txt`)**: Tests robustness against multi-column text extraction where left and right margins interleave.
4. **Experienced Professional Resume (`test_4_experienced.txt`)**: Tests extraction across multi-role employment timelines (AWS, Oracle, Wipro), promotional designations, and long technical skills lists.

---

## 6. Conclusion and Future Directions

In this experiential learning project, we designed, built, and benchmarked four NER paradigms for automated resume information extraction. While deep contextualized transformers (BERT) and LLMs deliver peak extraction fidelity, classical CRFs provide unmatched efficiency for latency-critical bulk screening. Future work includes investigating hybrid cascaded architectures (CRF for candidate triage followed by targeted LLM extraction for ambiguous spans) and LayoutLM multimodal token classification directly on document images.

---

## 7. Scholarly References
1. **Lafferty, J., McCallum, A., & Pereira, F. C.** (2001). *Conditional Random Fields: Probabilistic Models for Segmenting and Labeling Sequence Data*. Proceedings of the 18th International Conference on Machine Learning (ICML), 282–289.
2. **Devlin, J., Chang, M. W., Lee, K., & Toutanova, K.** (2018). *BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding*. Proceedings of the 2019 Conference of the North American Chapter of the Association for Computational Linguistics (NAACL-HLT), 4171–4186.
3. **Jurafsky, D., & Martin, J. H.** (2024). *Speech and Language Processing (3rd ed. draft)*. Chapter 8: Sequence Labeling for Named Entity Recognition.
4. **Ramshaw, L. A., & Marcus, M. P.** (1995). *Text Chunking using Transformation-Based Learning*. Third Workshop on Very Large Corpora.
5. **Vaswani, A., et al.** (2017). *Attention Is All You Need*. Advances in Neural Information Processing Systems (NeurIPS), 5998–6008.
