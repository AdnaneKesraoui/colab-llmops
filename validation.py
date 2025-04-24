# oas_pipeline/validation.py
import re, json, jsonschema

def clean_doc_text(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()

def strip_code_fences(text: str) -> str:
    m = re.search(r"```(?:json)?\s*([\s\S]*?)\s*```", text, re.DOTALL|re.IGNORECASE)
    if m:
        return m.group(1).strip()
    m2 = re.search(r"(\{[\s\S]*\})", text)
    return (m2.group(1).strip() if m2 else text.strip())

def is_valid_json(text: str) -> bool:
    try:
        json.loads(text); return True
    except json.JSONDecodeError:
        return False

# oas_pipeline/validation.py
import json
from pathlib import Path

_schema_path = Path(__file__).parent / "openapi_3_0_3_schema.json"
_openapi_schema = json.loads(_schema_path.read_text(encoding="utf-8-sig"))

def is_valid_oas(text: str) -> bool:
    try:
        candidate = json.loads(text)
        jsonschema.validate(candidate, _openapi_schema)
        return True
    except Exception:
        return False

