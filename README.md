# Bayan | بيان

Bayan is a bilingual Arabic/English NLP project developed as part of the SDAIA Academy **Natural Language Processing with Transformers (SDA-AIE-211)** course.

The project is cumulative: each lab builds a component that will later be integrated into the final Bayan citizen-feedback intelligence service.

---

# Current Progress

| Lab | Status | Progress |
|---|---|---|
| Lab 1 — Bilingual Preprocessing & Tokenisation | ✅ Completed | Preprocessing, PII masking, sentence segmentation, tokenizer audit and tokenizer selection completed |
| Lab 2 — Transformer Anatomy | ✅ Completed | Scaled dot-product attention, masking, Multi-Head Attention and transformer analysis completed |
| Lab 3 — Classification, NER & QA | 🟡 In Progress | Classification and NER completed locally; extractive QA implementation and smoke testing completed in Google Colab |
| Lab 4 — Arabic NLP | ⬜ Not Started | — |
| Lab 5 — Semantic Search | ⬜ Not Started | — |
| Lab 6 — Evaluation & Error Analysis | ⬜ Not Started | — |
| Lab 7 — Optimisation & Serving | ⬜ Not Started | — |
| Capstone — Bayan Service | ⬜ Not Started | — |

---

# Work Completed

## Lab 1 — Bilingual Preprocessing and Tokenisation

The first lab focused on creating a deterministic preprocessing pipeline for Arabic and English citizen-feedback text before it is passed to transformer models.

### Completed

- Implemented deterministic text normalisation.
- Implemented PII masking.
- Handled unwanted whitespace and formatting.
- Preserved useful linguistic and sentiment information.
- Implemented Arabic and English sentence segmentation.
- Tested preprocessing behaviour using the provided test suite.
- Audited four transformer tokenizers on the Bayan dataset:
  - mBERT
  - XLM-R
  - CAMeLBERT
  - DistilBERT
- Measured tokenizer fertility separately for Arabic and English.
- Measured token-length distributions and p95 sequence lengths.
- Used corpus evidence to select a suitable bilingual tokenizer/model family.

### Validation

- Preprocessing tests: **25/25 passed**
- PII masking fixture: **100% recall**
- Arabic and English sentence segmentation verified.

### Tokenizer Audit Results

| Tokenizer | Arabic Fertility | English Fertility | Arabic p95 | English p95 |
|---|---:|---:|---:|---:|
| mBERT | 2.14 | 1.51 | 27 | 25 |
| XLM-R | 1.67 | 1.43 | 21 | 23 |
| CAMeLBERT | 1.40 | 2.70 | 20 | 38 |
| DistilBERT | 4.52 | 1.30 | 47 | 21 |

### Tokenizer Decision

**XLM-R** was selected as the bilingual model family because it provided a strong balance between Arabic and English tokenisation quality.

CAMeLBERT produced the lowest Arabic fertility, but its English fertility was considerably higher. DistilBERT performed well on English but fragmented Arabic heavily.

The decision was therefore based on the requirements of a bilingual Arabic/English system rather than selecting the model with the best score for only one language.

---

## Lab 2 — Anatomy of a Transformer

The second lab focused on understanding and implementing the transformer attention mechanism rather than treating the transformer as a black box.

### Completed

- Implemented scaled dot-product attention.
- Calculated attention using Queries, Keys and Values.
- Applied the `1 / sqrt(d_k)` scaling factor before softmax.
- Implemented attention masking.
- Implemented Multi-Head Attention.
- Verified tensor input/output shapes.
- Compared the custom attention implementation with PyTorch.
- Built and verified a causal attention mask.
- Investigated attention leakage into padding tokens.
- Inspected Arabic and English attention behaviour.
- Performed transformer parameter analysis.

### Validation

- Attention unit tests: **2 passed**
- PyTorch numerical equivalence: **True**
- Maximum numerical difference: **0.00000024**
- Multi-Head Attention input/output shape: **[2, 5, 32]**
- Causal mask verified as lower triangular.
- PAD attention without mask: **0.172933**
- PAD attention with mask: **0.000000**

### Key Observation

The padding experiment demonstrated why the attention mask is part of model correctness.

Without the mask, the attention mechanism assigned some probability to padding tokens. After applying the mask, attention to padding tokens became zero.

---

## Lab 3 — Fine-tuning Classification, NER and Extractive QA

**Status: In Progress**

Lab 3 applies pretrained transformer models to specific NLP tasks.

The work has been divided between the local development environment and Google Colab. GPU-dependent work is performed in Google Colab using an NVIDIA Tesla T4.

---

### Lab 3A — Topic Classification

**Status: ✅ Completed Locally**

Completed the classification workflow locally.

### Completed

- Prepared the classification dataset.
- Worked with grouped dataset splitting to reduce data leakage between training and evaluation sets.
- Built the TF-IDF baseline workflow.
- Implemented the topic-classification workflow.
- Used macro-F1 as the main classification metric.
- Worked with the transformer fine-tuning pipeline required by the lab.

---

### Lab 3B — Named Entity Recognition

**Status: ✅ Completed Locally**

Completed the Named Entity Recognition portion locally.

### Completed

- Worked with BIO entity labels.
- Implemented label alignment between original words and tokenizer subwords.
- Assigned the original word label to the first subword.
- Assigned `-100` to continuation subwords.
- Assigned `-100` to special tokens.
- Prepared the alignment logic required for transformer token classification.
- Used entity-level evaluation rather than subword-level accuracy.

The main alignment rule used was:

- First subword → original BIO label
- Continuation subword → `-100`
- Special token → `-100`

This prevents continuation pieces and special tokens from incorrectly contributing to the NER training loss.

---

### Lab 3C — Extractive Question Answering

**Status: ✅ QA Implementation and Smoke Test Completed in Google Colab**

Google Colab was used for the QA work with a Python 3.12 virtual environment and NVIDIA Tesla T4 GPU.

### Implemented

- Implemented constrained answer-span search in:

`src/bayan/models/qa.py`

- Used start logits and end logits to find the strongest answer span.
- Prevented invalid spans where:

`end < start`

- Added a maximum answer-length constraint.
- Restricted answer selection to valid context tokens.
- Implemented null/no-answer handling.
- Compared the best text span against the model's null score.
- Allowed the system to return `None` when the context does not contain an answer.

### QA Unit Tests

The QA unit tests passed successfully:

**2/2 passed**

The tests verify that:

1. The QA system can return `None` for an unanswerable question.
2. The QA system rejects invalid/inverted answer spans.

### Answerable QA Example

Question:

`How long does Service 01 take?`

Expected answer:

`2 business days`

Predicted answer:

`2 business days`

Predicted character span:

`(63, 78)`

This verified that the model could correctly identify and map the predicted token span back to the original context.

### No-Answer QA Example

Question:

`What is the walk-in office address for Service 01?`

Expected answer:

`None`

The model's strongest possible text span was:

`Bayan`

However, the null score was stronger than the best answer-span score.

The system therefore correctly rejected the false text span and returned:

`None`

This demonstrates why explicit null handling is required for extractive QA. Without a null comparison, an extractive model may return a text span even when the requested information does not exist in the context.

### Full QA Smoke Test

The complete QA smoke test was implemented in:

`scripts/qa_smoke.py`

Results:

- **Answerable questions: 9/9 passed**
- **Unanswerable questions: 3/3 returned `None`**
- **QA smoke test: PASS**

The result was also recorded in `BENCHMARKS.md` as:

| Model | Metric | Validation | Frozen test | Train time |
|---|---|---:|---:|---:|
| QA | span/null smoke | 9/9 answerable + 3/3 null | N/A | N/A |

---

# Development Environments

## Local Development

The early labs and lighter development work were completed locally using:

- macOS
- Visual Studio Code
- Python 3.12
- `.venv` virtual environment
- Git
- GitHub
- pytest

The following work was primarily completed locally:

- Lab 1 preprocessing
- Lab 1 tokenizer audit
- Lab 2 transformer attention
- Lab 3A classification
- Lab 3B NER

---

## Google Colab

Google Colab is used for GPU-dependent work.

Current Lab 3 Colab environment:

- Python **3.12.14**
- NVIDIA **Tesla T4**
- Transformers **4.43.4**
- PyTorch
- Hugging Face Transformers
- pytest

The BAYAN GitHub repository is cloned directly into Google Colab so the same project structure can be used locally and in the cloud.

A Python 3.12 virtual environment is created inside the cloned repository because the BAYAN project requires Python 3.12.

---

# Project Approach

The project follows an evidence-based NLP engineering approach.

Instead of selecting models only because they are popular, each decision is supported by tests and measurements.

Examples include:

- Measuring Arabic and English tokenizer fertility before selecting a tokenizer.
- Testing preprocessing deterministically.
- Measuring PII masking recall.
- Comparing custom attention calculations with PyTorch.
- Measuring attention leakage into padding tokens.
- Evaluating NER using entity-level metrics.
- Preventing data leakage through grouped dataset splits.
- Allowing extractive QA to abstain when an answer is not available.
- Using automated unit tests and smoke tests to verify implementation behaviour.

---

# Remaining Work

## Lab 3

Lab 3 is still considered **in progress** until all required Lab 3 artefacts, metrics and repository evidence are consolidated.

Completed so far:

- ✅ Topic Classification
- ✅ Named Entity Recognition
- ✅ QA span selection
- ✅ QA null handling
- ✅ QA unit tests
- ✅ QA smoke test

Remaining Lab 3 work will be completed according to the course requirements before moving fully into Lab 4.

---

## Upcoming Labs

### Lab 4 — Arabic NLP

Planned work includes:

- Arabic-specific normalisation
- Arabic morphology
- Dialect analysis
- Gulf vs MSA evaluation
- Arabic-centric model comparison

### Lab 5 — Semantic Search

Planned work includes:

- Sentence embeddings
- FAISS indexing
- Cross-encoder re-ranking
- Recall@k
- MRR
- Cross-lingual retrieval evaluation

### Lab 6 — Evaluation and Error Analysis

Planned work includes:

- Slice-level evaluation
- Confidence intervals
- Bootstrap comparison
- Behavioural testing
- Error taxonomy
- Model cards

### Lab 7 — Inference Optimisation and Serving

Planned work includes:

- Latency benchmarking
- Dynamic padding
- Sequence-length optimisation
- ONNX export
- INT8 quantisation
- FastAPI serving
- Startup canaries
- Rollback artefacts

---

# Final Goal

The final goal is to assemble all completed lab components into **Bayan**, a bilingual Arabic/English citizen-feedback intelligence service capable of:

- Preprocessing Arabic and English feedback
- Protecting PII
- Classifying feedback topics and sentiment
- Extracting named entities
- Answering extractive questions
- Returning no answer when appropriate
- Finding semantically similar historical cases
- Evaluating performance across language and dialect slices
- Serving predictions efficiently through an API

The project will continue to be updated as the remaining labs are completed.

https://github.com/SDAIAAcademy
