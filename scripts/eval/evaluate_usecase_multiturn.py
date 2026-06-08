import json
import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from peft import PeftModel

BASE_MODEL_NAME = "models/t5gemma2-270m-task-vector"
SFT_ADAPTER_DIR = "results/t5gemma2-270m-light-sft/final_adapter"
DPO_ADAPTER_DIR = "results/t5gemma2-270m-light-dpo/final_dpo_adapter"
MERGED_DATA_FILE = "data/t5-gemma-2-chat-instruct-merged.jsonl"


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
    print("MULAILAH EVALUASI MULTI-TURN PADA USECASE UTAMA KITA")
    print("=" * 80)

    # 1. Ambil percakapan asuransi pertama dari dataset merged asli
    with open(MERGED_DATA_FILE, "r", encoding="utf-8") as f:
        first_line = f.readline()
        chat_data = json.loads(first_line)

    # 2. Rekonstruksi seluruh riwayat percakapan kecuali turn asisten terakhir
    # Format percakapan: system:\n... \nuser:\n... \nassistant:\n...
    # Mari kita format secara dinamis sesuai dengan template pelatihan kita
    conversations = chat_data["conversations"]

    formatted_turns = []
    system_prompt = ""
    for turn in conversations[:-1]:  # abaikan asisten terakhir
        role = turn["role"]
        content = turn["content"]
        if role == "system":
            system_prompt = content
        elif role == "user":
            formatted_turns.append(f"user: {content}")
        elif role == "assistant":
            formatted_turns.append(f"assistant: {content}")

    # Gabungkan dengan template yang tepat
    prompt_header = f"system: {system_prompt}"
    prompt_body = "\n\n".join(formatted_turns)
    full_prompt = f"{prompt_header}\n\n{prompt_body}\n\nassistant:\n"

    target_chosen = conversations[-1]["content"]

    print(f"\n📂 Usecase Utama: {chat_data['topik']} ({chat_data['num_turns']} Turns)")
    print(f'🔹 User (Turn Terakhir): "{conversations[-2]["content"]}"')
    print(f'🎯 CHOSEN Target (Ekspektasi): "{target_chosen}"')
    print("-" * 80)

    # 3. Load Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_NAME)

    # 4. Load Base Model
    print("\n[1/3] Loading Base Model...")
    base_model = AutoModelForSeq2SeqLM.from_pretrained(
        BASE_MODEL_NAME,
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )
    base_model.eval()
    base_out = generate_response(base_model, tokenizer, full_prompt)

    # 5. Load SFT Model
    print("\n[2/3] Loading SFT Adapter...")
    sft_model = PeftModel.from_pretrained(base_model, SFT_ADAPTER_DIR)
    sft_model.eval()
    sft_out = generate_response(sft_model, tokenizer, full_prompt)

    # Bongkar SFT
    sft_model.unload()

    # 6. Load DPO Model
    print("\n[3/3] Loading DPO Adapter...")
    dpo_model = PeftModel.from_pretrained(base_model, DPO_ADAPTER_DIR)
    dpo_model.eval()
    dpo_out = generate_response(dpo_model, tokenizer, full_prompt)

    # Cetak Perbandingan final
    print("\n" + "=" * 80)
    print("HASIL EVALUASI MULTI-TURN KUALITATIF (USECASE UTAMA)")
    print("=" * 80)
    print(f"🎯 CHOSEN Target:\n{target_chosen}")
    print("-" * 80)
    print(f"🔴 BASE MODEL Output:\n{base_out}")
    print("-" * 80)
    print(f"🟡 SFT MODEL Output:\n{sft_out}")
    print("-" * 80)
    print(f"🟢 DPO MODEL Output:\n{dpo_out}")
    print("=" * 80)


if __name__ == "__main__":
    main()
