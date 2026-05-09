import importlib.util
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PIPELINE_PATH = ROOT / "Final Python code" / "sota_mcq_pipeline.py"

spec = importlib.util.spec_from_file_location("sota_mcq_pipeline", PIPELINE_PATH)
pipeline = importlib.util.module_from_spec(spec)
sys.modules["sota_mcq_pipeline"] = pipeline
spec.loader.exec_module(pipeline)

input_path = ROOT / "sample_inputs" / "sample_text.txt"
output_path = ROOT / "outputs" / "smoke_questions.json"
cache_path = ROOT / "outputs" / "mcq_cache.sqlite"

output_path.parent.mkdir(exist_ok=True)

cfg = pipeline.RunConfig(
    max_chunks=1,
    n_questions_per_chunk=1,
    cache_path=str(cache_path),
    answerability_required=False,
)

result = pipeline.run_all(
    input_paths=[str(input_path)],
    out_path=str(output_path),
    cfg=cfg,
)

print(json.dumps({
    "counts": result.get("counts"),
    "difficulty_distribution": result.get("difficulty_distribution"),
    "runtime_seconds": result.get("runtime_seconds"),
    "output_path": str(output_path),
}, indent=2))