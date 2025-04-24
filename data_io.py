# oas_pipeline/data_io.py
import os

def find_all_docs(directory: str) -> list[str]:
    docs = []
    for root, _, files in os.walk(directory):
        for f in files:
            docs.append(os.path.join(root, f))
    return docs

def load_system_prompt(path: str = "/content/old_system_prompt.txt") -> str:
    if os.path.exists(path):
        return open(path, "r", encoding="utf-8").read()
    return (
      "You are a specialist in generating OpenAPI 3.0 specifications in JSON format. "
      "Ensure every response includes a complete OpenAPI 3.0 JSON specification."
    )
