# colab_llmops/generation.py
import os
import torch
from functools import lru_cache
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    BitsAndBytesConfig
)
from .validation import clean_doc_text, strip_code_fences
from .data_io import load_system_prompt

@lru_cache(maxsize=None)
def _load_model_and_tokenizer(model_id: str):
    """
    Load (and quantize) the model & tokenizer once, caching the result.
    """
    # configure bitsandbytes 4-bit quantization
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",        # NF4 quant type is smallest with good perf
        bnb_4bit_compute_dtype=torch.float16
    )

    # load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_id)

    # load a 4-bit quantized model, automatically placing layers on CPU/GPU
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        quantization_config=bnb_config,
        device_map="auto",              # let Transformers auto‐map layers
        low_cpu_mem_usage=True
    )

    # figure out where the model lives (GPU or CPU)
    device = next(model.parameters()).device
    return model, tokenizer, device

def generate_with_model(
    model_id: str,
    doc_text: str,
    max_length: int = 8192,
    temperature: float = 0.1
) -> str:
    """
    Generate an OpenAPI spec using a specified model.
    Model & tokenizer are loaded & cached in 4-bit quantized form.
    """
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

    # get (possibly quantized) model, tokenizer, and device
    model, tokenizer, device = _load_model_and_tokenizer(model_id)

    prompt = load_system_prompt() + "\nAPI Documentation:\n" + clean_doc_text(doc_text)
    inputs = tokenizer(prompt, return_tensors="pt").to(device)

    try:
        outputs = model.generate(
            **inputs,
            max_length=max_length,
            temperature=temperature,
            do_sample=True
        )
    except RuntimeError:
        torch.cuda.empty_cache()
        raise

    return strip_code_fences(tokenizer.decode(outputs[0], skip_special_tokens=True))

def generate_spec(model_name: str, doc_text: str) -> str:
    """
    Dispatch to zero-shot or finetuned Mistral models.
    """
    if model_name == "mistral":
        return generate_with_model("mistralai/Mistral-7B-Instruct-v0.3", doc_text)
    else:
        raise ValueError(f"Unsupported model: {model_name}")
