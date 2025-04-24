# colab_llmops/data_io.py
import os
from pathlib import Path

def find_all_docs(directory: str) -> list[str]:
    docs = []
    for root, _, files in os.walk(directory):
        for f in files:
            docs.append(os.path.join(root, f))
    return docs

def load_system_prompt() -> str:
    # locate old_system_prompt.txt in the same folder as this file
    pkg_folder = Path(__file__).parent
    prompt_file = pkg_folder / "old_system_prompt.txt"
    if prompt_file.exists():
        return prompt_file.read_text(encoding="utf-8")
    # fallback default
    return (
        "You are a specialist in generating OpenAPI 3.0 specifications in JSON format. "
        "Ensure every response includes a complete OpenAPI 3.0 JSON specification."
    )
