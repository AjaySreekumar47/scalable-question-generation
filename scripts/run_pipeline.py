import argparse
import importlib.util
import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
PIPELINE_PATH = ROOT / "Final Python code" / "sota_mcq_pipeline.py"


def load_pipeline_module():
    spec = importlib.util.spec_from_file_location("sota_mcq_pipeline", PIPELINE_PATH)
    module = importlib.util.module_from_spec(spec)
    sys.modules["sota_mcq_pipeline"] = module
    spec.loader.exec_module(module)
    return module


def main():
    parser = argparse.ArgumentParser(
        description="Run the scalable MCQ generation pipeline on local input files."
    )
    parser.add_argument(
        "--input",
        nargs="+",
        required=True,
        help="One or more input .txt or .pdf files.",
    )
    parser.add_argument(
        "--output",
        default="outputs/questions.json",
        help="Output JSON path.",
    )
    parser.add_argument(
        "--max-chunks",
        type=int,
        default=3,
        help="Maximum chunks per document for development/testing.",
    )
    parser.add_argument(
        "--questions-per-chunk",
        type=int,
        default=1,
        help="Number of MCQs to generate per chunk.",
    )
    parser.add_argument(
        "--skip-answerability",
        action="store_true",
        help="Skip answerability verification. Useful for offline/mock testing.",
    )

    args = parser.parse_args()

    pipeline = load_pipeline_module()

    output_path = ROOT / args.output
    output_path.parent.mkdir(parents=True, exist_ok=True)

    cfg = pipeline.RunConfig(
        max_chunks=args.max_chunks,
        n_questions_per_chunk=args.questions_per_chunk,
        cache_path=str(ROOT / "outputs" / "mcq_cache.sqlite"),
        answerability_required=not args.skip_answerability,
    )

    result = pipeline.run_all(
        input_paths=[str(Path(p)) for p in args.input],
        out_path=str(output_path),
        cfg=cfg,
    )

    print(
        json.dumps(
            {
                "counts": result.get("counts"),
                "difficulty_distribution": result.get("difficulty_distribution"),
                "runtime_seconds": result.get("runtime_seconds"),
                "output_path": str(output_path),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()