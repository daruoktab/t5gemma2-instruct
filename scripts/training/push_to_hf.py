import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from peft import PeftModel
import os

# ============================================================
# Configuration
# ============================================================
BASE_MODEL = "google/t5gemma-2-270m-270m"
# Path adapter terakhir (silakan sesuaikan jika ingin SFT atau DPO)
ADAPTER_PATH = "results/t5gemma2-270m-clean-sft/final_adapter"
REPO_NAME = "daruoktab/t5gemma2-instruct"


def push_model():
    print(f"Loading base model: {BASE_MODEL}...")
    # Load base model di CPU untuk menghemat VRAM saat merge
    model = AutoModelForSeq2SeqLM.from_pretrained(
        BASE_MODEL,
        torch_dtype=torch.bfloat16,
        device_map="cpu",
        attn_implementation="eager",
    )

    print(f"Loading LoRA adapter: {ADAPTER_PATH}...")
    model = PeftModel.from_pretrained(model, ADAPTER_PATH)

    print("Merging adapter into base model (menggabungkan weight)...")
    model = model.merge_and_unload()

    print(f"Loading tokenizer dari: {ADAPTER_PATH}...")
    tokenizer = AutoTokenizer.from_pretrained(ADAPTER_PATH)
    assert tokenizer is not None

    print(f"Mulai push merged model & tokenizer ke Hugging Face: {REPO_NAME}...")

    # Ini akan mem-push semua file model yang dibutuhkan (config, safetensors, tokenizer files)
    model.push_to_hub(REPO_NAME)
    tokenizer.push_to_hub(REPO_NAME)

    print(f"✅ Model berhasil dipush ke: https://huggingface.co/{REPO_NAME}")


if __name__ == "__main__":
    # Pastikan Anda sudah login via `huggingface-cli login` di terminal
    # Atau un-comment baris di bawah ini jika ingin memasukkan token langsung
    # from huggingface_hub import login
    # login("TOKEN_HF_ANDA")

    if not os.path.exists(ADAPTER_PATH):
        print(f"❌ Error: Adapter path {ADAPTER_PATH} tidak ditemukan.")
        print(
            "Silakan edit script ini dan sesuaikan ADAPTER_PATH dengan model yang benar."
        )
    else:
        push_model()
