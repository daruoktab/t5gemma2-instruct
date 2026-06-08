import json
import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from peft import PeftModel

BASE_MODEL_NAME = "models/t5gemma2-270m-task-vector"
SFT_ADAPTER_DIR = "results/t5gemma2-270m-light-sft/final_adapter"
DPO_ADAPTER_DIR = "results/t5gemma2-270m-light-dpo/final_dpo_adapter"
DPO_DATA_FILE = "data/preferences_dpo_light.jsonl"


def generate_response(model, tokenizer, prompt: str) -> str:
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=128,
            do_sample=True,
            temperature=0.7,
            top_p=0.9,
            repetition_penalty=1.2,
        )
    return tokenizer.decode(outputs[0], skip_special_tokens=True)


def main():
    print("=" * 80)
    print("MULAILAH EVALUASI KHUSUS PADA DATA TRAINING DPO")
    print("=" * 80)

    # 1. Ambil 3 contoh data dari dataset DPO
    samples = []
    with open(DPO_DATA_FILE, "r", encoding="utf-8") as f:
        for idx, line in enumerate(f):
            if idx in (
                0,
                1,
                3,
            ):  # Ambil baris ke-1 (Pisang), ke-2 (Dokumen), ke-4 (Al-Kautsar)
                samples.append(json.loads(line))
            if len(samples) >= 3:
                break

    # 2. Load Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_NAME)

    # 3. Load Base Model
    print("\n[1/3] Loading Base Model...")
    base_model = AutoModelForSeq2SeqLM.from_pretrained(
        BASE_MODEL_NAME,
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )
    base_model.eval()

    base_outputs = []
    for s in samples:
        base_outputs.append(generate_response(base_model, tokenizer, s["input"]))

    # 4. Load SFT Model
    print("\n[2/3] Loading SFT Adapter...")
    sft_model = PeftModel.from_pretrained(base_model, SFT_ADAPTER_DIR)
    sft_model.eval()

    sft_outputs = []
    for s in samples:
        sft_outputs.append(generate_response(sft_model, tokenizer, s["input"]))

    # Bongkar SFT adapter secara bersih
    sft_model.unload()

    # 5. Load DPO Model
    print("\n[3/3] Loading DPO Adapter...")
    dpo_model = PeftModel.from_pretrained(base_model, DPO_ADAPTER_DIR)
    dpo_model.eval()

    dpo_outputs = []
    for s in samples:
        dpo_outputs.append(generate_response(dpo_model, tokenizer, s["input"]))

    # Cetak hasil perbandingan premium
    print("\n" + "=" * 80)
    print("HASIL PERBANDINGAN OUTPUT MODEL PADA CONTOH DATA TRAINING")
    print("=" * 80)

    for idx, s in enumerate(samples):
        prompt_preview = s["input"][-250:] if len(s["input"]) > 250 else s["input"]
        print(f"\n📝 CONTOH DATA LATIH #{idx + 1}")
        print(f"🔹 Flaw Type / Rationale: {s['flaw_type']} - {s.get('rationale', '')}")
        print(f"🔹 Prompt (Akhir): ... {prompt_preview.strip()}")
        print("-" * 80)
        print(f"🎯 CHOSEN (Target): {s['chosen']}")
        print(f"❌ REJECTED (Hindari): {s['rejected']}")
        print("-" * 80)
        print(f"🔴 BASE MODEL Output:\n{base_outputs[idx]}")
        print("-" * 80)
        print(f"🟡 SFT MODEL Output:\n{sft_outputs[idx]}")
        print("-" * 80)
        print(f"🟢 DPO MODEL Output:\n{dpo_outputs[idx]}")
        print("=" * 80)


if __name__ == "__main__":
    main()
