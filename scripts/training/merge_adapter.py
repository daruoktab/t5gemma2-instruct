"""
Merge LoRA Adapter ke Base Model
================================
Menggabungkan LoRA adapter weights ke base model untuk:
- Inference lebih cepat (tidak perlu load adapter terpisah)
- Deploy lebih mudah (single model directory)
- Upload ke HuggingFace Hub

Usage:
    python merge_adapter.py
    python merge_adapter.py --push-to-hub username/model-name
"""

import argparse
from typing import Optional, cast, Any
import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from peft import PeftModel

# ============================================================
# Configuration
# ============================================================
BASE_MODEL = "google/t5gemma-2-270m-270m"
ADAPTER_PATH = "../t5gemma2-chat-v1/final"
MERGED_PATH = "../t5gemma2-chat-v1/merged"


def merge_and_save(push_to_hub: Optional[str] = None):
    """Merge LoRA adapter into base model and save."""

    # Load base model di CPU (butuh RAM, bukan VRAM)
    print("Loading base model on CPU...")
    model = AutoModelForSeq2SeqLM.from_pretrained(
        BASE_MODEL,
        torch_dtype=torch.bfloat16,
        device_map="cpu",
        attn_implementation="eager",
    )

    # Load LoRA adapter
    print(f"Loading LoRA adapter from {ADAPTER_PATH}...")
    model = PeftModel.from_pretrained(model, ADAPTER_PATH)

    # Merge
    print("Merging adapter into base model...")
    model = cast(Any, model).merge_and_unload()

    # Save
    print(f"Saving merged model to {MERGED_PATH}...")
    model.save_pretrained(MERGED_PATH)

    # Save tokenizer — load eksplisit sebagai PreTrainedTokenizerFast
    tokenizer = AutoTokenizer.from_pretrained(ADAPTER_PATH)
    # AutoTokenizer.from_pretrained selalu mengembalikan tokenizer (raise jika gagal)
    assert tokenizer is not None, "Gagal load tokenizer dari ADAPTER_PATH"
    tokenizer.save_pretrained(MERGED_PATH)

    print(f"Merged model saved to {MERGED_PATH}")

    # Optional: push to Hub
    if push_to_hub:
        print(f"\nPushing to HuggingFace Hub: {push_to_hub}")
        model.push_to_hub(push_to_hub)
        tokenizer.push_to_hub(push_to_hub)
        print(f"Model pushed to https://huggingface.co/{push_to_hub}")

    print("\nDone!")
    print(f"Untuk inference dengan merged model: python inference.py --merged")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Merge LoRA adapter into base model")
    parser.add_argument(
        "--push-to-hub",
        type=str,
        default=None,
        help="Push merged model to HuggingFace Hub (e.g., 'username/t5gemma2-chat-v1')",
    )
    args = parser.parse_args()

    merge_and_save(push_to_hub=args.push_to_hub)
