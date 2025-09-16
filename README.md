## 1. Overview

This project implements a **Scalable Question Generation System** that automatically produces high-quality multiple-choice questions (MCQs) from large text documents. Inspired by the *Savaal* research paper, the system is designed not only to generate questions, but also to ensure **scalability, quality control, and structured outputs**.

The pipeline ingests **PDF and plain text documents**, performs preprocessing and semantic chunking, and uses a **large language model (LLM)** to generate conceptual questions with one correct answer and 3–4 plausible distractors. Each question is further validated for schema correctness, answerability, and difficulty level (easy/medium/hard).

Two complementary deliverables are provided:

* A **Jupyter/Colab notebook** that walks through the design in a step-by-step manner, showing the evolution from a simple baseline to more complex implementations.
* A **stand-alone Python module (`sota_mcq_pipeline.py`)** that consolidates the final, production-style pipeline with caching, error handling, and manifest logging.

The final output is a **JSON file (`questions_sota_150.json`)** containing the generated questions along with metadata such as evidence spans, difficulty tags, and runtime statistics.

## 2. Evolution: From Baseline → SOTA

This project was built iteratively, moving from a simple baseline implementation toward a robust, state-of-the-art (SOTA) pipeline. The **notebook** captures this evolution step by step, while the **final `.py` file** integrates all improvements into a single modular system.

### 🔹 Baseline Parsing & Input Handling

* Started with loading text from **PDFs and transcripts** using simple extractors.
* Applied **basic cleaning** (removing page numbers, fixing line breaks) to normalize raw content.
* Established a foundation where documents could be reliably ingested into the pipeline.

### 🔹 Chunking & Scalability

* Introduced **sliding window segmentation** to address context length limits in LLMs.
* For transcripts, implemented **fine-grained segmentation** (max words per segment) to preserve speaker intent.
* For PDFs, leveraged **section-aware splitting** based on headings and structure.
* This step ensured the system could process large documents without memory or context loss.

### 🔹 Initial Question Generation

* Built generation prompts to produce **MCQs with 1 correct + 3 distractors**.
* Focused on **conceptual understanding** rather than verbatim recall.
* Exported results into a JSON structure for easy inspection.

### 🔹 Quality Control & Validation

* Added **schema validation** to enforce JSON correctness (4 choices, correct answer must match).
* Implemented **answerability checks** by asking the LLM to solve the generated MCQs using the source chunk only — filtering out ungrounded questions.
* Introduced **deduplication** via embeddings similarity to remove near-duplicate stems.

### 🔹 Difficulty Management

* Incorporated **Bloom’s Taxonomy–inspired labeling** into the pipeline.
* Difficulty classification was handled via LLM tags, with a **heuristic fallback** for robustness.
* Enabled stratification into **easy / medium / hard**, which was key for balanced outputs.

### 🔹 Final SOTA Pipeline (`sota_mcq_pipeline.py`)

All of the above improvements were consolidated into a single, production-ready module with:

* **Robust loaders** (PyMuPDF, pdfplumber, TXT) + normalization for math/notation.
* **Asynchronous batching** with exponential backoff, caching (SQLite), and token budgeting.
* **Pluggable LLM provider** with mock fallback for local testing.
* **Strict prompts** enforcing JSON schema, evidence spans, and plausible distractors.
* **Comprehensive validation**: schema, answerability, entailment, deduplication.
* **Difficulty tagging + manifest logging**, including counts, runtime, and distribution of difficulty levels.
* **Single JSON export** (`questions_sota_150.json`) with questions and full metadata.

## 3. Final Architecture

The final implementation (`sota_mcq_pipeline.py`) is organized into modular components, each responsible for a key part of the pipeline. Together, they form a robust end-to-end system for scalable MCQ generation .

### 📂 Document Loading & Cleaning

* **Supports**: PDF (via PyMuPDF, pdfplumber) and plain text.
* **Normalization**: Fixes symbols, math notation, subscripts, and dotted leaders.
* **Cleaning**: Removes page numbers, collapses whitespace, applies Unicode fixes.

### ✂️ Segmentation & Chunking

* **Transcript-aware segmentation**: Splits transcripts into small segments (≤ 25 words) to preserve coherence.
* **PDF section splitting**: Detects numbered headings and all-caps lines to respect document structure.
* **Sliding window chunking**: Creates overlapping windows of text (configurable size/stride) to manage LLM context.

### 🤖 LLM Provider & Helpers

* **Pluggable OpenAI client**: Defaults to GPT-4o-mini for generation, with mock fallback for offline runs.
* **Async batching with retries**: Handles API rate limits, exponential backoff, and caching.
* **SQLite cache**: Avoids redundant API calls by storing request–response pairs.

### 📝 Prompt Templates

* **Generation Prompt**: Produces conceptual MCQs with exactly 4 options, 1 correct answer, and an evidence span.
* **Verification Prompt**: Ensures the correct answer is grounded in the source text (answerability check).
* **Deduplication Prompt**: Optionally confirms near-duplicate questions via LLM.
* **Difficulty Prompt**: Labels questions as easy / medium / hard using Bloom’s taxonomy.

### ✅ Validation & Quality Control

* **Schema validation**: Enforces JSON structure and ensures the correct answer matches one of the choices.
* **Answerability check**: Discards ungrounded or unsupported questions.
* **Entailment/contradiction checks**: Detects invalid distractors (if added).
* **Deduplication**: Uses embeddings similarity (configurable threshold) to remove near-duplicate stems.

### 🎚 Difficulty Tagging

* **LLM-based**: Primary difficulty assignment using Bloom’s taxonomy.
* **Heuristic fallback**: Keyword-based detection (e.g., “What is…” → easy, “Why…” → medium).

### 📊 Orchestration & Manifest

* **`run_all()`**: High-level function for Colab/Jupyter or scripts. Takes input file paths, runs the pipeline, and saves a JSON output.
* **Manifest logging**: Records key metadata:

  * Document count, chunk count, generated vs. filtered questions
  * Difficulty distribution
  * Runtime and cost estimates
  * Input file hashes for reproducibility
* **Single JSON export**: Outputs all validated questions and metadata into `questions_sota_150.json`.


## 4. How to Run & Outputs

The system can be executed either in **Colab/Jupyter notebooks** (recommended for interactive exploration) or as a **stand-alone Python module** for streamlined batch runs.

### 🛠️ Installation

Install the required dependencies:

```bash
pip install pymupdf pdfplumber ftfy openai tiktoken numpy pandas scikit-learn
```

Optional (for faster sentence segmentation):

```bash
pip install spacy
python -m spacy download en_core_web_sm
```

### 🔑 API Key Setup

Set your OpenAI API key in your environment:

```python
import os
os.environ["OPENAI_API_KEY"] = "sk-..."
```

### ▶ Run in Notebook / Colab

```python
from sota_mcq_pipeline import run_all

res = run_all(
    input_paths=["/content/notes.pdf", "/content/transcript_1.txt"],
    out_path="/content/questions_sota.json"
)

print(res["counts"])
print(res["difficulty_distribution"])
```

### 💻 Run as Script (CLI, quick smoke test)

If executed directly, the module runs in dev-safe mode with a few chunks:

```bash
python sota_mcq_pipeline.py
```

Sample output metrics:

```
{
  "counts": {
    "documents": 6,
    "chunks": 254,
    "generated": 496,
    "answerability_failed": 16,
    "after_dedup": 468
  },
  "difficulty_distribution": {"easy": 94, "medium": 46, "hard": 10},
  "runtime_seconds": 2.3
}
```

### 📦 Output Files

Running the pipeline produces a **single JSON file** (`questions_sota.json`) with:

* **Question text**
* **4 options** (1 correct, 3 distractors)
* **Correct answer** (must match exactly one option)
* **Evidence span** (≤ 25 words, justifying the correct answer)
* **Difficulty label** (easy / medium / hard)
* **Document source + chunk index** (for traceability)
* **Manifest metadata**: input file hashes, counts, difficulty distribution, runtime, errors

Example JSON entry:

```json
{
  "question": "What is the primary purpose of ...?",
  "choices": ["A", "B", "C", "D"],
  "correct_answer": "A",
  "evidence_span": "short snippet from source text",
  "difficulty": "medium",
  "doc": "notes.pdf",
  "chunk_index": 12
}
```


## 5. Metrics & Evaluation

The pipeline not only generates questions but also produces a **manifest** containing detailed metrics to help evaluate quality, scalability, and performance. This makes the system transparent and reproducible.

### 📊 Metrics Captured

* **Document Count** – number of input files processed.
* **Chunk Count** – total number of text chunks produced after segmentation.
* **Generated Questions** – raw count of MCQs produced by the LLM before filtering.
* **Answerability Failures** – number of questions discarded because they were ungrounded or unverifiable.
* **After Deduplication** – final number of unique, validated questions in the JSON output.
* **Difficulty Distribution** – easy / medium / hard counts, useful for stratification.
* **Runtime (seconds)** – total execution time.
* **Input Integrity** – SHA-256 hashes of input files to ensure reproducibility.
* **Error Logs** – dictionary of errors encountered (e.g., parsing failures, schema mismatches).

### 📝 Example Manifest Summary

```json
{
  "assignment": "Scalable Question Generation System",
  "counts": {
    "documents": 6,
    "chunks": 254,
    "generated": 496,
    "answerability_failed": 16,
    "after_dedup": 468
  },
  "difficulty_distribution": {
    "easy": 94,
    "medium": 46,
    "hard": 10
  },
  "runtime_seconds": 2.35
}
```

### ✅ Evaluation Criteria Alignment

* **Functionality & Quality** – validated via schema enforcement, answerability checks, and difficulty labels.
* **Scalability & Design** – handles large documents with sliding windows, async batching, and caching.
* **Code Quality** – modular `.py` pipeline with guardrails + explanatory notebook.
* **Communication** – metrics and manifest make results easy to interpret and verify.

---

## 6. File Structure & Deliverables

The submission package is organized as a single `.zip` file containing both the exploratory notebook and the final production pipeline. 

### 📂 Folder Layout

```
submission/
 ├── Scalable_Question_Generation_System.ipynb   # Notebook: step-by-step evolution (baseline → SOTA)
 ├── sota_mcq_pipeline.py                        # Final modular Python pipeline (production-ready)
 ├── questions_sota.json                         # Output: generated MCQs with metadata
 ├── README.md                                   # Project overview, evolution, usage instructions
 └── requirements.txt                            # (Optional) package list for reproducibility
```

### 📦 Contents Explained

* **`Scalable_Question_Generation_System.ipynb`**

  * Interactive Colab/Jupyter notebook.
  * Documents the incremental design process, from simple parsing to advanced validation.
  * Contains detailed markdown explanations, test runs, and intermediate outputs.

* **`sota_mcq_pipeline.py`**

  * Final, production-grade pipeline.
  * Includes robust document loaders, semantic chunking, LLM integration, validation, deduplication, difficulty tagging, and manifest logging.
  * Can be imported in notebooks or run directly as a script.

* **`questions_sota.json`**

  * Single consolidated output file.
  * Contains all validated MCQs, each with question, choices, correct answer, evidence span, difficulty label, and metadata.

* **`README.md`**

  * Provides the project overview, system evolution, architecture breakdown, usage instructions, metrics, and file structure.
  * Serves as the main guide for evaluators.

* **`requirements.txt`** 
  * Lists required dependencies (pymupdf, pdfplumber, ftfy, openai, tiktoken, numpy, pandas, scikit-learn, etc.).
  * Ensures reproducibility in fresh environments
