# Scalable Question Generation Pipeline

A modular pipeline for generating multiple-choice questions (MCQs) from PDF and plain-text documents using LLM-assisted generation, validation, deduplication, and metadata logging.

The project includes:

- a notebook version for step-by-step exploration,
- a reusable Python pipeline,
- a command-line wrapper for local runs,
- an offline smoke test that works without an OpenAI API key,
- intermediate and final generated-question artifacts from earlier experiments.

---

## Project Summary

This project explores how large language models can be used to generate structured educational questions from long-form documents.

Given one or more `.txt` or `.pdf` files, the pipeline:

1. loads and cleans the source documents,
2. chunks the text into manageable segments,
3. generates MCQs with four answer choices,
4. validates the generated question schema,
5. optionally checks answerability against the source chunk,
6. removes near-duplicate questions using embeddings,
7. assigns difficulty labels,
8. exports questions and run metadata to JSON.

The implementation is designed to be runnable both with an OpenAI API key and in offline/mock mode for local testing.

---

## Repository Structure

```text
scalable-question-generation/
├── Final Python code/
│   └── sota_mcq_pipeline.py
├── Colab Notebook (.ipynb)/
│   └── Scalable_Question_Generation_System.ipynb
├── scripts/
│   ├── run_pipeline.py
│   └── smoke_test.py
├── sample_inputs/
│   └── sample_text.txt
├── Intermediate Questions/
│   ├── questions.json
│   ├── questions_enhanced.json
│   └── questions_sota.json
├── final Questions (after QC)/
│   └── questions_sota_150.json
├── Approach Walkthrough.mp4
├── requirements.txt
├── LICENSE
└── README.md
````

---

## Core Capabilities

| Capability               |      Status | Description                                                                            |
| ------------------------ | ----------: | -------------------------------------------------------------------------------------- |
| PDF and text loading     | Implemented | Supports `.pdf` and `.txt` inputs using PyMuPDF/pdfplumber/text readers                |
| Text cleaning            | Implemented | Normalizes whitespace, Unicode artifacts, and common notation issues                   |
| Chunking                 | Implemented | Uses transcript-style segmentation and sliding-window chunking                         |
| LLM-based MCQ generation | Implemented | Generates structured MCQs with four options and one correct answer                     |
| Mock/offline mode        | Implemented | Runs without an OpenAI API key for smoke testing                                       |
| Schema validation        | Implemented | Enforces required fields, four choices, correct-answer consistency, and evidence spans |
| Answerability checking   | Implemented | Can verify whether the source chunk supports the generated answer                      |
| Deduplication            | Implemented | Uses embedding similarity to remove near-duplicate question stems                      |
| Difficulty tagging       | Implemented | Labels questions as `easy`, `medium`, or `hard`                                        |
| Manifest logging         | Implemented | Saves counts, runtime, config, input hashes, and error metadata                        |
| CLI wrapper              | Implemented | Provides a cleaner local command-line entry point                                      |
| Notebook walkthrough     |    Included | Shows the project evolution from exploratory notebook to final pipeline                |

---

## Setup

Python 3.10+ is recommended.

Create and activate a virtual environment:

```bash
python -m venv .venv
```

Windows PowerShell:

```powershell
.\.venv\Scripts\Activate.ps1
```

macOS/Linux:

```bash
source .venv/bin/activate
```

Install dependencies:

```bash
python -m pip install --upgrade pip setuptools wheel
pip install -r requirements.txt
```

---

## Quick Offline Smoke Test

The repository includes a small sample input and a smoke test that runs without an OpenAI API key.

```bash
python scripts/smoke_test.py
```

Expected output should look similar to:

```json
{
  "counts": {
    "documents": 1,
    "chunks": 1,
    "generated": 1,
    "answerability_failed": 0,
    "after_dedup": 1
  },
  "difficulty_distribution": {
    "hard": 1
  },
  "runtime_seconds": 0.01,
  "output_path": ".../outputs/smoke_questions.json"
}
```

This verifies that the local pipeline, mock LLM fallback, JSON export, and manifest generation are working.

---

## Run the Pipeline Locally

Use the command-line wrapper:

```bash
python scripts/run_pipeline.py --input sample_inputs/sample_text.txt --output outputs/questions.json --skip-answerability
```

The `--skip-answerability` flag is useful for offline/mock testing because answerability verification is most meaningful when using a real LLM provider.

Example with custom settings:

```bash
python scripts/run_pipeline.py \
  --input sample_inputs/sample_text.txt \
  --output outputs/questions.json \
  --max-chunks 1 \
  --questions-per-chunk 1 \
  --skip-answerability
```

On Windows PowerShell:

```powershell
python scripts\run_pipeline.py --input sample_inputs\sample_text.txt --output outputs\questions.json --max-chunks 1 --questions-per-chunk 1 --skip-answerability
```

---

## Run with OpenAI

Set your OpenAI API key.

Windows PowerShell:

```powershell
$env:OPENAI_API_KEY="your-api-key"
```

macOS/Linux:

```bash
export OPENAI_API_KEY="your-api-key"
```

Then run:

```bash
python scripts/run_pipeline.py --input path/to/document.pdf --output outputs/questions.json
```

When an API key is available, the pipeline uses the configured OpenAI models for generation, verification, embeddings, and difficulty labeling.

---

## Output Format

The pipeline writes a JSON file containing both metadata and generated questions.

Example structure:

```json
{
  "assignment": "Scalable Question Generation System",
  "generated_at": "2026-05-09T01:55:25.657321+00:00",
  "models": {
    "generation": "gpt-4o-mini",
    "verifier": "gpt-4o-mini",
    "embedding": "text-embedding-3-small"
  },
  "config": {},
  "inputs": [
    {
      "path": "sample_inputs/sample_text.txt",
      "sha256": "..."
    }
  ],
  "counts": {
    "documents": 1,
    "chunks": 1,
    "generated": 1,
    "answerability_failed": 0,
    "after_dedup": 1
  },
  "difficulty_distribution": {
    "hard": 1
  },
  "runtime_seconds": 0.01,
  "errors": {},
  "questions": [
    {
      "question": "Mock: What is the purpose of the chunk?",
      "choices": ["A", "B", "C", "D"],
      "correct_answer": "A",
      "evidence_span": "mock evidence from source",
      "difficulty": "hard",
      "doc": "sample_inputs/sample_text.txt",
      "chunk_index": 0
    }
  ]
}
```

Each generated question includes:

* question text,
* four answer choices,
* exact correct answer,
* evidence span,
* difficulty label,
* source document,
* chunk index.

---

## Main Pipeline Module

The core implementation lives in:

```text
Final Python code/sota_mcq_pipeline.py
```

It includes:

* `RunConfig`: configuration object for chunking, model settings, caching, validation, and deduplication,
* document loaders for PDF/TXT files,
* cleaning and normalization utilities,
* chunking functions,
* prompt builders,
* OpenAI/mock LLM provider,
* SQLite prompt cache,
* schema validation,
* answerability verification,
* embedding-based deduplication,
* difficulty tagging,
* `run_all()`: high-level orchestration function.

The `scripts/run_pipeline.py` wrapper is provided so users can run the project without directly importing from a folder with spaces in its name.

---

## Existing Artifacts

This repository also includes previous generated outputs:

```text
Intermediate Questions/
├── questions.json
├── questions_enhanced.json
└── questions_sota.json

final Questions (after QC)/
└── questions_sota_150.json
```

These files represent earlier project outputs and quality-control stages. New local runs write to the ignored `outputs/` directory by default.

---

## What This Project Demonstrates

This project demonstrates:

* LLM pipeline design for educational content generation,
* document ingestion and preprocessing,
* chunking strategies for long-context inputs,
* structured JSON generation,
* validation and quality-control checks,
* answerability verification,
* embedding-based deduplication,
* difficulty labeling,
* metadata/manifest logging,
* local smoke testing with mock model fallback,
* CLI wrapping for reproducible execution.

---

## Known Limitations

* The repository includes legacy folder names from the original notebook/submission workflow, such as `Final Python code/`.
* The offline/mock mode is intended for smoke testing, not question quality evaluation.
* High-quality MCQ generation requires a real LLM API key and source documents with enough conceptual content.
* Answerability verification is most useful with a real LLM provider.
* No formal benchmark against human-authored questions is included.
* The project does not claim benchmarked state-of-the-art performance; it is a modular, scalable MCQ generation pipeline inspired by research-style question-generation workflows.

---

## Suggested Commands

Minimal local verification:

```bash
python scripts/smoke_test.py
```

Run on sample input:

```bash
python scripts/run_pipeline.py --input sample_inputs/sample_text.txt --output outputs/questions.json --skip-answerability
```

Run on your own document with OpenAI enabled:

```bash
python scripts/run_pipeline.py --input path/to/notes.pdf --output outputs/questions.json
```

---

## License

This project is open-source and available under the MIT License.