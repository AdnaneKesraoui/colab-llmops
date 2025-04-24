# oas_pipeline/diffing.py
import tempfile, subprocess, json, os

def run_oasdiff(expected_spec: str, generated_spec: str):
    # write temps
    f1 = tempfile.NamedTemporaryFile("w", delete=False, suffix=".json", encoding="utf-8")
    f1.write(expected_spec); f1.close()
    f2 = tempfile.NamedTemporaryFile("w", delete=False, suffix=".json", encoding="utf-8")
    f2.write(generated_spec); f2.close()

    try:
        res = subprocess.run(
            ["oasdiff", "diff", f1.name, f2.name, "-f", "json"],
            capture_output=True, text=True, check=False
        )
        code = res.returncode
        diffs = json.loads(res.stdout).get("differences", [])
        correct = (code == 0)
        count   = len(diffs) if not correct else 0
        return float(correct), count, diffs
    except Exception:
        return 0.0, -1, []
    finally:
        os.remove(f1.name); os.remove(f2.name)
