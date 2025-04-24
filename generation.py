# oas_pipeline/generation.py
import os, torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from .validation import clean_doc_text, strip_code_fences
from .data_io import load_system_prompt

def generate_with_model(model_id: str, doc_text: str, max_length=8192, temperature=0.1):
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model     = AutoModelForCausalLM.from_pretrained(
                    model_id,
                    torch_dtype=torch.float16,
                    low_cpu_mem_usage=True
                ).to(device)

    prompt = load_system_prompt() + "\nAPI Documentation:\n" + clean_doc_text(doc_text)
    inputs = tokenizer(prompt, return_tensors="pt").to(device)

    try:
        outputs = model.generate(
            **inputs,
            max_length=max_length,
            temperature=temperature,
            do_sample=True
        )
    except RuntimeError as e:
        torch.cuda.empty_cache()
        raise

    return strip_code_fences(tokenizer.decode(outputs[0], skip_special_tokens=True))

def generate_spec(model_name: str, doc_text: str) -> str:
    if model_name == "mistral":
        return generate_with_model("mistralai/Mistral-7B-Instruct-v0.3", doc_text)
    elif model_name == "mistral_finetuned":
        return generate_with_model("AdnaneIsMe/oas_lora_model_v6_with_eval", doc_text)
    else:
        raise ValueError(f"Unsupported model: {model_name}")
