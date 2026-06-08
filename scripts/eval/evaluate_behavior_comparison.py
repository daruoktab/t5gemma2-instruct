import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from peft import PeftModel
from typing import cast
from transformers import PreTrainedTokenizerBase

BASE_MODEL_NAME = "models/t5gemma2-270m-task-vector"
SFT_ADAPTER_DIR = "results/t5gemma2-270m-sft/checkpoint-460" # Menggunakan checkpoint terakhir karena final_adapter mungkin tidak lengkap/sama

# Harus sama persis dengan system prompt di data training
SYSTEM_PROMPT = (
    "Kamu adalah asisten AI yang helpful, santai, dan ramah. "
    "Gunakan Bahasa Indonesia sebagai bahasa utama. "
    "Switch ke English hanya kalau user memang minta atau konteksnya English. "
    "Boleh casual dan natural — pakai 'aku/kamu' atau 'saya/Anda' sesuai situasi. "
    "Kalau ada task seperti translate, summarize, paraphrase, atau rewrite muncul dalam obrolan, "
    "langsung bantu dengan natural tanpa basa-basi berlebihan. "
    "Jangan terlalu formal kecuali situasinya memang mengharuskan."
)

EVAL_QUERIES = [
    "Siapa presiden Indonesia pertama?",
    "Jelaskan apa itu fotosintesis dengan bahasa sederhana.",
    "Apa perbedaan antara simile dan metafora?",
]


def build_prompt(prompt: str) -> str:
    formatted = "<start_of_turn>user\n"
    formatted += SYSTEM_PROMPT + "\n\n"
    formatted += prompt + "<end_of_turn>\n"
    formatted += "<start_of_turn>model\n"
    return formatted


def generate_response(model, tokenizer, prompt: str) -> str:
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=2048).to(model.device)
    
    end_of_turn_id = tokenizer.convert_tokens_to_ids("<end_of_turn>")
    eos_id = tokenizer.eos_token_id
    stop_ids = [end_of_turn_id, eos_id] if end_of_turn_id != eos_id else [eos_id]

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=256,
            do_sample=True,
            temperature=0.7,
            top_p=0.9,
            repetition_penalty=1.2,
            eos_token_id=stop_ids,
        )
    
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    # Cleanup token spesial dari T5Gemma jika ter-decode
    response = response.replace("<end_of_turn>", "").strip()
    return response


def main():
    print("=" * 80)
    print("MULAILAH EVALUASI BEHAVIOR KUALITATIF")
    print("=" * 80)

    # 1. Load Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(SFT_ADAPTER_DIR)

    # 2. Load Base Model
    print("\n[1/2] Loading Base Model...")
    base_model = AutoModelForSeq2SeqLM.from_pretrained(
        BASE_MODEL_NAME,
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )
    
    if getattr(base_model.config, "decoder_start_token_id", None) is None:
        base_model.config.decoder_start_token_id = tokenizer.bos_token_id

    base_model.eval()

    base_results = {}
    for query in EVAL_QUERIES:
        prompt = build_prompt(query)
        base_results[query] = generate_response(base_model, tokenizer, prompt)

    # 3. Load SFT Model
    print("\n[2/2] Loading SFT Adapter...")
    sft_model = PeftModel.from_pretrained(base_model, SFT_ADAPTER_DIR)
    sft_model.eval()

    sft_results = {}
    for query in EVAL_QUERIES:
        prompt = build_prompt(query)
        sft_results[query] = generate_response(sft_model, tokenizer, prompt)

    # Cetak Perbandingan berdampingan secara premium
    print("\n" + "=" * 80)
    print("HASIL PERBANDINGAN OUTPUT MODEL")
    print("=" * 80)

    for idx, query in enumerate(EVAL_QUERIES, 1):
        print(f'\n📌 QUERY #{idx}: "{query}"')
        print("-" * 80)
        print(f"🔴 BASE MODEL:\n{base_results[query]}")
        print("-" * 80)
        print(f"🟡 SFT MODEL (10% Training):\n{sft_results[query]}")
        print("=" * 80)


if __name__ == "__main__":
    main()
