# oas_pipeline/main.py
import os, time, json
import torch
from .generation    import generate_spec
from .data_io       import find_all_docs
from .validation    import is_valid_json, is_valid_oas, strip_code_fences
from .diffing       import run_oasdiff
from .mlflow_utils  import start_experiment, log_params, log_metrics, log_artifact

DOCS_DIR             = os.getenv("DOCS_DIR", "/content/random-50-docs-5k-8k-range")
GOLD_STANDARD_OAS_DIR= os.getenv("GOLD_OAS_DIR", "/content/truth-openapi3")
LOCAL_GEN_DIR        = os.getenv("OUTPUT_DIR", "/content/generated_specs")
MODEL_NAMES          = os.getenv("MODEL_NAMES", "mistral").split(",")

def process_docs(model_name: str):
    paths = find_all_docs(DOCS_DIR)
    out_dir = os.path.join(LOCAL_GEN_DIR, model_name)
    os.makedirs(out_dir, exist_ok=True)

    # Counters for summary
    stats = {"json_ok": 0.0, "oas_ok": 0.0, "oasdiff_ok": 0.0, "total_time": 0.0}

    for i, p in enumerate(paths, start=1):
        text = open(p, encoding="utf-8").read()
        start = time.time()
        generated = generate_spec(model_name, text)
        elapsed = time.time() - start

        clean = strip_code_fences(generated)
        ok_json = float(is_valid_json(clean))
        ok_oas  = float(is_valid_oas(clean))

        import re

        # Compare to gold spec
        base = os.path.splitext(os.path.basename(p))[0]
        base = re.sub(r'-?index-?', '', base).strip('-')
        gold_name = f"{base}-openapi.json"

        gold_path = os.path.join(GOLD_STANDARD_OAS_DIR, gold_name)
        corr, diff_count, _ = (0.0, 0, [])
        if os.path.exists(gold_path):
            corr, diff_count, _ = run_oasdiff(open(gold_path).read(), clean)

        # Log per‐doc, step‐indexed metrics
        log_metrics({
            "valid_json":        ok_json,
            "valid_oas":         ok_oas,
            "oasdiff_correct":   corr,
            "oasdiff_diff_count":diff_count,
            "generation_time":   elapsed,
            "generation_speed":  len(clean) / elapsed if elapsed else 0,
        }, step=i)

        # Update counters
        stats["json_ok"]     += ok_json
        stats["oas_ok"]      += ok_oas
        stats["oasdiff_ok"]  += corr
        stats["total_time"]  += elapsed

        # Write out the cleaned spec
        out_file = os.path.join(out_dir, f"{os.path.splitext(os.path.basename(p))[0]}.json")
        open(out_file, "w", encoding="utf-8").write(clean)

    # Run-level summary metrics (no step)
    n = len(paths) or 1
    log_metrics({
        "valid_json_percentage":      stats["json_ok"]    / n * 100,
        "valid_oas_percentage":       stats["oas_ok"]     / n * 100,
        "oasdiff_correct_percentage": stats["oasdiff_ok"] / n * 100,
        "average_generation_time":    stats["total_time"] / n,
    })

    print(f"[INFO] {model_name}: "
          f"JSON {stats['json_ok']/n*100:.2f}% | "
          f"OAS {stats['oas_ok']/n*100:.2f}% | "
          f"OASDiff {stats['oasdiff_ok']/n*100:.2f}% | "
          f"AvgTime {stats['total_time']/n:.2f}s")


if __name__=="__main__":
    if torch.cuda.is_available():
        print("GPU:", torch.cuda.get_device_name(0))
    start_experiment("oas_pipeline")
    for m in MODEL_NAMES:
        log_params({"model_name":m})
        process_docs(m)
