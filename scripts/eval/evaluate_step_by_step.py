import json
import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from peft import PeftModel
from typing import Any, cast

BASE_MODEL_NAME = "models/t5gemma2-270m-task-vector"
SFT_ADAPTER_DIR = "results/t5gemma2-270m-light-sft/final_adapter"
DPO_ADAPTER_DIR = "results/t5gemma2-270m-light-dpo/final_dpo_adapter"
MERGED_DATA_FILE = "data/t5-gemma-2-chat-instruct-merged.jsonl"

SUPPRESS_BLOCK1 = list(range(6, 105))             # <unused0>–<unused98>
SUPPRESS_BLOCK2 = list(range(256002, 262144))     # <unused100>–<unused6241>
SUPPRESS_VISION = [255999, 256000, 256001]        # <end_of_image>, <image_soft_token>
ALL_SUPPRESS_IDS = list(set(SUPPRESS_BLOCK1 + SUPPRESS_BLOCK2 + SUPPRESS_VISION))

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
            suppress_tokens=ALL_SUPPRESS_IDS,
        )
    return tokenizer.decode(outputs[0], skip_special_tokens=True)

def build_prompt_up_to_turn(conversations: list, turn_limit: int) -> tuple[str, str]:
    """
    Membangun prompt multi-turn dari percakapan dari indeks 0 hingga turn_limit (eksklusif asisten turn_limit).
    Mengembalikan: (prompt_input, target_chosen)
    """
    formatted_turns = []
    system_prompt = ""
    
    # turn_limit merepresentasikan turn user yang ingin kita uji (user_turn ke-X)
    # kita kumpulkan percakapan sebelum turn tersebut
    for i in range(turn_limit):
        turn = conversations[i]
        role = turn["role"]
        content = turn["content"]
        if role == "system":
            system_prompt = content
        elif role == "user":
            formatted_turns.append(f"user: {content}")
        elif role == "assistant":
            formatted_turns.append(f"assistant: {content}")
            
    # Tambahkan turn user target yang sedang diuji
    target_user_turn = conversations[turn_limit]
    formatted_turns.append(f"user: {target_user_turn['content']}")
    
    prompt_header = f"system: {system_prompt}"
    prompt_body = "\n\n".join(formatted_turns)
    full_prompt = f"{prompt_header}\n\n{prompt_body}\n\nassistant:\n"
    
    # Target chosen adalah jawaban asisten asli untuk turn ini
    target_chosen = conversations[turn_limit + 1]["content"]
    
    return full_prompt, target_chosen

def main():
    print("=" * 80)
    print("EVALUASI BEHAVIOR MULTI-TURN LANGKAH DEMI LANGKAH (STEP-BY-STEP FLOW)")
    print("=" * 80)

    # 1. Ambil percakapan asuransi pertama
    with open(MERGED_DATA_FILE, "r", encoding="utf-8") as f:
        first_line = f.readline()
        chat_data = json.loads(first_line)

    conversations = chat_data["conversations"]

    # Kita tentukan 4 Milestone krusial dalam percakapan asuransi ini:
    # MILESTONE 1 (Turn awal - Diskusi Keuangan UMR)
    #   User: "Kalau gaji UMR kayak aku, masuk akal nggak kalau ambil swasta juga?"
    #   Indeks di conversations: system (0), user (1), asst (2), user (3), asst (4), user (5), asst (6), user (7)
    # MILESTONE 2 (Task Switch 1 - Minta Ringkasan / Summarize)
    #   User: "Boleh, tolong ringkasin dong."
    # MILESTONE 3 (Task Switch 2 - Minta Translate ke Inggris)
    #   User: "Wah iya bener juga. Kalau pake bahasa Inggris, istilah-istilah itu gimana?"
    # MILESTONE 4 (Turn Akhir - Penutup ucapan terima kasih)
    #   User: "Mantap. Makasih banyak ya penjelasannya!"
    
    milestones: list[dict[str, Any]] = [
        {"name": "M1: Konsultasi Finansial (Awal)", "user_idx": 7},
        {"name": "M2: Task Switch -> Summarize (Tengah)", "user_idx": 13},
        {"name": "M3: Task Switch -> Translate (Tengah-Akhir)", "user_idx": 17},
        {"name": "M4: Penutup & Terima Kasih (Akhir)", "user_idx": 21}
    ]

    # 2. Load Tokenizer & Model
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_NAME)
    
    print("\nLoading Base Model...")
    base_model = AutoModelForSeq2SeqLM.from_pretrained(
        BASE_MODEL_NAME,
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )
    base_model.eval()

    # Evaluasi Base Model
    base_results = []
    for m in milestones:
        prompt, target = build_prompt_up_to_turn(conversations, cast(int, m["user_idx"]))
        out = generate_response(base_model, tokenizer, prompt)
        base_results.append(out)

    # Load SFT Model
    print("\nLoading SFT Adapter...")
    sft_model = PeftModel.from_pretrained(base_model, SFT_ADAPTER_DIR)
    sft_model.eval()
    
    sft_results = []
    for m in milestones:
        prompt, target = build_prompt_up_to_turn(conversations, cast(int, m["user_idx"]))
        out = generate_response(sft_model, tokenizer, prompt)
        sft_results.append(out)
        
    sft_model.unload()

    # Load DPO Model
    print("\nLoading DPO Adapter...")
    dpo_model = PeftModel.from_pretrained(base_model, DPO_ADAPTER_DIR)
    dpo_model.eval()
    
    dpo_results = []
    for m in milestones:
        prompt, target = build_prompt_up_to_turn(conversations, cast(int, m["user_idx"]))
        out = generate_response(dpo_model, tokenizer, prompt)
        dpo_results.append(out)

    # Tampilkan perbandingan langkah demi langkah secara visual
    print("\n" + "=" * 80)
    print("HASIL SIMULASI PERILAKU MODEL BERURUTAN (MILESTONE FLOW)")
    print("=" * 80)

    for i, m in enumerate(milestones):
        prompt, target = build_prompt_up_to_turn(conversations, cast(int, m["user_idx"]))
        user_msg = conversations[cast(int, m["user_idx"])]["content"]
        
        print(f"\n📌 {m['name']}")
        print(f"💬 USER: \"{user_msg}\"")
        print(f"🎯 TARGET CHOSEN (Ekspektasi): {target}")
        print("-" * 80)
        print(f"🔴 BASE MODEL Output:\n{base_results[i]}")
        print("-" * 80)
        print(f"🟡 SFT MODEL Output:\n{sft_results[i]}")
        print("-" * 80)
        print(f"🟢 DPO MODEL Output:\n{dpo_results[i]}")
        print("=" * 80)

if __name__ == "__main__":
    main()
