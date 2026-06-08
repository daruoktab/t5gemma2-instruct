import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, PreTrainedTokenizerBase
from typing import cast
from peft import PeftModel

# Harus sama persis dengan system prompt di data training
SYSTEM_PROMPT = (
    "Kamu adalah asisten AI yang helpful, santai, dan ramah. "
    "Gunakan Bahasa Indonesia sebagai bahasa utama. "
    "Switch ke English hanya kalau user memang minta atau konteksnya English. "
    "Boleh casual dan natural \u2014 pakai 'aku/kamu' atau 'saya/Anda' sesuai situasi. "
    "Kalau ada task seperti translate, summarize, paraphrase, atau rewrite muncul dalam obrolan, "
    "langsung bantu dengan natural tanpa basa-basi berlebihan. "
    "Jangan terlalu formal kecuali situasinya memang mengharuskan."
)

BASE_MODEL = "models/t5gemma2-270m-task-vector"  # base untuk load adapter
ADAPTER_PATH = "results/t5gemma2-270m-sft/final_adapter"


def build_prompt(prompt: str, history: list[tuple[str, str]] | None = None) -> str:
    """Bangun prompt format Gemma 3 yang konsisten dengan data training.
    System prompt dimasukkan ke dalam turn user pertama (Gemma 3 style).
    history: list of (user, assistant) tuples untuk multi-turn.
    """
    formatted = ""
    is_first_user = True

    if history:
        for user_msg, assistant_msg in history:
            formatted += "<start_of_turn>user\n"
            if is_first_user:
                formatted += SYSTEM_PROMPT + "\n\n"
                is_first_user = False
            formatted += user_msg + "<end_of_turn>\n"
            formatted += "<start_of_turn>model\n"
            formatted += assistant_msg + "<end_of_turn>\n"

    formatted += "<start_of_turn>user\n"
    if is_first_user:
        formatted += SYSTEM_PROMPT + "\n\n"
    formatted += prompt + "<end_of_turn>\n"
    formatted += "<start_of_turn>model\n"
    return formatted


def test_inference():
    print(f"Loading base model from {BASE_MODEL}...")
    base_model = AutoModelForSeq2SeqLM.from_pretrained(
        BASE_MODEL,
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )

    print(f"Loading LoRA adapter from {ADAPTER_PATH}...")
    model = PeftModel.from_pretrained(base_model, ADAPTER_PATH)
    model.eval()

    print(f"Loading tokenizer from {ADAPTER_PATH}...")
    tokenizer = AutoTokenizer.from_pretrained(ADAPTER_PATH)
    if tokenizer is None:
        raise ValueError("Failed to load tokenizer")
    tokenizer = cast(PreTrainedTokenizerBase, tokenizer)

    end_of_turn_id = tokenizer.convert_tokens_to_ids("<end_of_turn>")
    eos_id = tokenizer.eos_token_id
    stop_ids = [end_of_turn_id, eos_id] if end_of_turn_id != eos_id else [eos_id]

    prompts = [
        "Halo, siapa kamu?",
        "Tolong buatkan puisi singkat tentang kopi.",
        "1+1 berapa?",
    ]

    print("\n--- SFT ADAPTER INFERENCE RESULTS ---")
    for prompt in prompts:
        formatted_prompt = build_prompt(prompt)

        inputs = tokenizer(
            formatted_prompt, return_tensors="pt", truncation=True, max_length=2048
        ).to(model.device)

        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=256,
                do_sample=True,
                temperature=0.7,
                repetition_penalty=1.2,
                eos_token_id=stop_ids,
            )

        response = cast(str, tokenizer.decode(outputs[0], skip_special_tokens=True))
        response = response.replace("<end_of_turn>", "").strip()

        print(f"\nUser   : {prompt}")
        print(f"Model  : {response}")


if __name__ == "__main__":
    test_inference()
