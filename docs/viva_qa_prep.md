# Viva & Defense Q&A Preparation Guide
### Course: Natural Language Processing for LLM (`24AM7PCLLM`)
**B.M.S. College of Engineering, Bengaluru**  
*Project: Named Entity Recognition for Automated Resume Parsing*  
*Target: Score 80%+ on faculty defense (Dr. Monika P, Prof. Swapna S Banasode, Prof. Upeksha Prasad G)*

---

## Category 1: CRF & Classical Machine Learning for NER

### Q1: What is a Conditional Random Field (CRF), and why is it preferred over an HMM or MEMM for NER?
**Answer**:
A CRF is a discriminative undirected probabilistic graphical model that directly estimates the conditional distribution $P(\mathbf{y}|\mathbf{x})$ over label sequences given the observed input sequence $\mathbf{x}$.
- **Compared to HMMs (Generative)**: HMMs model the joint distribution $P(\mathbf{x}, \mathbf{y}) = P(\mathbf{y})P(\mathbf{x}|\mathbf{y})$ and require strict independence assumptions between observed tokens. CRFs allow arbitrary, overlapping, non-independent features (such as prefixes, capitalization, and surrounding word windows).
- **Compared to MEMMs (Maximum Entropy Markov Models)**: MEMMs normalize locally at each state transition, making them vulnerable to the **Label Bias Problem** (states with low outgoing transition entropy dominate regardless of observation). CRFs normalize globally using a single sequence-level partition function $Z(\mathbf{x})$, effectively overcoming label bias.

### Q2: What exact features did you engineer for your CRF model?
**Answer**:
We implemented four categories of hand-crafted features in `src/models/crf_model.py`:
1. **Orthographic**: `word.isupper()` (acronyms like AWS, B.E.), `word.istitle()` (names, companies), `word.isdigit()` (years, experience), `word.has_at` (emails), `word.has_dot` (degrees like M.Tech).
2. **Morphological**: Prefixes (`word[:2]`, `word[:3]`) and suffixes (`word[-2:]`, `word[-3:]` like *-ing*, *-tion*, *-eer*).
3. **Word Shape**: Regex shape patterns collapsing character sequences (e.g., `Google` $\rightarrow$ `Xxxxx`, `2024` $\rightarrow$ `dddd`).
4. **Contextual Window**: Features of preceding tokens ($w_{i-1}, w_{i-2}$), succeeding tokens ($w_{i+1}, w_{i+2}$), and bigram transition combinations ($w_{i-1}+w_i$ and $w_i+w_{i+1}$).

### Q3: How is inference performed in a linear-chain CRF?
**Answer**:
Inference uses the dynamic programming **Viterbi Algorithm**. It finds the globally optimal state sequence $\hat{\mathbf{y}} = \arg\max_{\mathbf{y}} P(\mathbf{y}|\mathbf{x})$ in $O(T \cdot |S|^2)$ time complexity, where $T$ is the sequence length and $|S|$ is the number of label states.

---

## Category 2: BERT & Transformer Token Classification (Units 1 & 3)

### Q4: Explain WordPiece tokenization and why label alignment is necessary in BERT.
**Answer** (Crucial Course Connection to Unit 1 and Unit 3):
- **WordPiece Tokenization**: BERT's vocabulary consists of frequent word units and subword pieces prefixed by `##`. Out-of-vocabulary or compound words are broken into subwords (e.g. `TensorFlow` $\rightarrow$ `["Ten", "##sor", "##Flow"]`).
- **The Label Alignment Dilemma**: In sequence labeling datasets, annotations are assigned at the word level. If `TensorFlow` has label `B-Skills`, tokenizing it produces three subword tokens.
- **Alignment Solution**: Using HuggingFace's `tokenizer.word_ids()`:
  - The first subword (`Ten`) is assigned the true label `B-Skills`.
  - The continuation subwords (`##sor`, `##Flow`) are assigned label ID `-100`.
  - In PyTorch, `nn.CrossEntropyLoss(ignore_index=-100)` ignores positions with label `-100`, preventing subword noise from distorting backpropagation gradients.

### Q5: How does BERT's bidirectional self-attention benefit resume entity extraction over left-to-right LSTMs?
**Answer**:
In resumes, the meaning of an entity is heavily influenced by both preceding and succeeding text. For example, in the phrase *"admitted to Adithya Institute of Technology in 2012"*, the word *"admitted to"* (left context) signals an educational institution, while *"in 2012"* (right context) signals a temporal qualifier. Bidirectional self-attention computes query-key-value interactions across the full context window simultaneously:
$$\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right)V$$

### Q6: How did you handle resumes exceeding BERT's 512 maximum sequence length?
**Answer**:
Resumes often exceed 1,000 words. We implemented a sliding window chunking mechanism with stride in `src/models/bert_model.py`. The resume is divided into 128-to-256 token chunks, evaluated individually through BERT, and then recombined using word index tracking.

---

## Category 3: Evaluation Metrics & seqeval

### Q7: Why is token-level accuracy inadequate for evaluating NER models? Why did you use seqeval?
**Answer**:
Token-level accuracy is heavily skewed by class imbalance: over 85% of resume tokens are non-entities (`O`). A naive classifier predicting `O` for every token would achieve 85%+ accuracy while extracting zero entities.  
Furthermore, entities are multi-token **chunks** (e.g. `B-Degree I-Degree I-Degree`). If a model extracts `B-Degree I-Degree` but misses the third word, token accuracy credits it with 66% correctness. However, in reality, the extracted entity boundary is incorrect. `seqeval` enforces strict CoNLL chunk-level evaluation: a predicted chunk is considered a True Positive **only if both the boundary span (start and end) and entity class match the ground truth exactly**.

### Q8: What is the difference between Micro, Macro, and Weighted F1-scores?
**Answer**:
- **Micro-F1**: Aggregates total True Positives, False Positives, and False Negatives globally across all classes. Heavily influenced by frequent classes like *Skills* and *Companies*.
- **Macro-F1**: Computes F1 independently for each class and calculates the unweighted mean. Gives equal importance to rare classes (e.g., *Graduation Year*) and common classes.
- **Weighted-F1**: Computes F1 per class and averages them weighted by class support (number of true instances).

---

## Category 4: LLM Prompting & System Engineering

### Q9: What are the trade-offs between fine-tuned BERT and few-shot LLM prompting for resume parsing?
**Answer**:
- **Inference Latency & Cost**: BERT executes locally on CPU in ~62ms at zero API cost; LLM API calls take ~1,450ms (~23x slower) and incur token costs.
- **Generalization**: LLMs excel at zero-shot generalization across bizarre resume layouts, multi-column tables, and unusual phrasing without task-specific training data.
- **Data Privacy**: Local BERT and CRF models run on-premise without exposing sensitive candidate PII (names, phone numbers, emails) to external cloud LLM APIs.

### Q10: How did you enforce valid JSON structure in LLM outputs?
**Answer**:
1. Defined a strict JSON schema in the few-shot system prompt with explicit input-output demonstration pairs.
2. Configured `response_mime_type="application/json"` in the Gemini API configuration.
3. Implemented a regex sanitization wrapper in `src/models/llm_model.py` to strip accidental markdown fences (````json ... ````) and validate required keys.

---

## Category 5: Dataset & Error Analysis

### Q11: What were the primary limitations of the Kaggle / DataTurks dataset?
**Answer**:
The Kaggle dataset contains approximately 200 resumes annotated via crowdsourcing on DataTurks. Its primary limitations are:
1. **Annotation Inconsistencies**: Many technical skills within project descriptions are untagged, while whole paragraphs of job descriptions were occasionally tagged as a single *Skills* entity.
2. **Anomalous Labels**: Presence of non-standard tags like `'UNKNOWN'`.
3. **Offset Inaccuracies**: Trailing spaces in character offsets caused span mismatches, which our `src/data_loader.py` cleaned through offset boundary validation.

### Q12: Why did all models perform worst on 'Years of Experience'?
**Answer**:
Because candidate resumes express experience through widely disparate formats:
- Absolute numbers: *"5 Years 6 Months"*, *"3.2-years"*
- Approximate strings: *"Over 7+ years of IT experience"*
- Date ranges: *"November 2012 to Present"* (requiring arithmetic reasoning rather than sequence labeling)  
This linguistic variance and annotation boundary inconsistency in the ground truth led to F1 scores between 0.38 and 0.58 across all models.
