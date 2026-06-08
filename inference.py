"""
Inference Script untuk Fine-tuned T5Gemma-2-270m-270m
=====================================================
Mendukung:
1. LoRA adapter (load base model + adapter)
2. Merged model (sudah di-merge dengan merge_adapter.py)
3. Preset test examples (EN + ID)
4. Interactive mode

Usage:
    python inference.py                    # Pakai default config
    python inference.py --merged           # Pakai merged model
    python inference.py --no-interactive   # Skip interactive mode
"""

import argparse
import os
import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from peft import PeftModel

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_ROOT_DIR   = os.path.normpath(os.path.join(_SCRIPT_DIR, ".."))

BASE_MODEL   = "google/t5gemma-2-270m-270m"
ADAPTER_PATH = os.path.join(_SCRIPT_DIR, "results", "t5gemma2-270m-sft", "final")
MERGED_PATH  = os.path.join(_SCRIPT_DIR, "results", "t5gemma2-270m-sft", "merged")
MAX_NEW_TOKENS = 256

# Auto-detect checkpoint terbaru jika 'final' belum ada
_adapter_base = os.path.join(_SCRIPT_DIR, "results", "t5gemma2-270m-sft")
if not os.path.exists(ADAPTER_PATH) and os.path.exists(_adapter_base):
    _checkpoints = sorted(
        [d for d in os.listdir(_adapter_base) if d.startswith("checkpoint-")],
        key=lambda x: int(x.split("-")[1]),
    )
    if _checkpoints:
        ADAPTER_PATH = os.path.join(_adapter_base, _checkpoints[-1])
        print(f"  'final' tidak ditemukan, pakai checkpoint: {ADAPTER_PATH}")



def load_model(use_merged: bool = False):
    """Load model - either LoRA adapter or merged model."""

    if use_merged:
        print("Loading merged model...")
        tokenizer = AutoTokenizer.from_pretrained(MERGED_PATH)
        model = AutoModelForSeq2SeqLM.from_pretrained(
            MERGED_PATH,
            torch_dtype=torch.bfloat16,
            device_map="auto",
            attn_implementation="eager",
        )
    else:
        print("Loading base model + LoRA adapter...")
        tokenizer = AutoTokenizer.from_pretrained(ADAPTER_PATH)

        model = AutoModelForSeq2SeqLM.from_pretrained(
            BASE_MODEL,
            torch_dtype=torch.bfloat16,
            device_map="auto",
            attn_implementation="eager",
        )

        print("Applying LoRA adapter...")
        model = PeftModel.from_pretrained(model, ADAPTER_PATH)

    model.eval()
    print(f"Model loaded on: {model.device}")

    if torch.cuda.is_available():
        print(f"GPU Memory: {torch.cuda.memory_allocated() / 1024**3:.2f} GB")

    return model, tokenizer


def generate_response(
    model,
    tokenizer,
    instruction: str,
    max_new_tokens: int = MAX_NEW_TOKENS,
    do_sample: bool = True,
    temperature: float = 0.7,
    top_p: float = 0.9,
    top_k: int = 50,
    repetition_penalty: float = 1.3,  # v2: naik dari 1.1 ke 1.3
    no_repeat_ngram_size: int = 4,
    num_beams: int = 1,
) -> str:
    """
    Generate response dari encoder-decoder model.

    Encoder menerima instruksi/pertanyaan (bidirectional attention),
    Decoder menghasilkan jawaban (autoregressive).
    """
    inputs = tokenizer(
        instruction,
        return_tensors="pt",
        max_length=512,
        truncation=True,
    ).to(model.device)

    gen_kwargs: dict = {
        "max_new_tokens": max_new_tokens,
        "repetition_penalty": repetition_penalty,
        "no_repeat_ngram_size": no_repeat_ngram_size,
    }

    if num_beams > 1:
        gen_kwargs["num_beams"] = num_beams
        gen_kwargs["early_stopping"] = True
        gen_kwargs["do_sample"] = False
    elif do_sample:
        gen_kwargs["do_sample"] = True
        gen_kwargs["temperature"] = temperature
        gen_kwargs["top_p"] = top_p
        gen_kwargs["top_k"] = top_k
    else:
        gen_kwargs["do_sample"] = False

    with torch.no_grad():
        generate_fn = getattr(model, "generate")
        raw_out = generate_fn(**inputs, **gen_kwargs)

    response: str = str(tokenizer.decode(raw_out[0], skip_special_tokens=True))
    return response


# ============================================================
# Test Examples - Tugas yang cocok untuk encoder-decoder
# ============================================================
TEST_PROMPTS = [
    # === Summarization (keunggulan encoder-decoder) ===
    {
        "category": "Summarization (EN)",
        "input": (
            "Summarize the following text: Machine learning is a branch of artificial intelligence "
            "that focuses on building applications that learn from data and improve their accuracy "
            "over time without being programmed to do so. In data science, an algorithm is a sequence "
            "of statistical processing steps. In machine learning, algorithms are trained to find "
            "patterns and features in massive amounts of data in order to make decisions and predictions "
            "based on new data. The better the algorithm, the more accurate the decisions and predictions "
            "will become as it processes more data."
        ),
    },
    {
        "category": "Summarization (ID)",
        "input": (
            "Buatkan ringkasan dari teks berikut: Indonesia adalah negara kepulauan terbesar di dunia "
            "yang terdiri dari lebih dari 17.000 pulau. Terletak di Asia Tenggara, Indonesia memiliki "
            "populasi lebih dari 270 juta jiwa, menjadikannya negara berpenduduk keempat terbesar di "
            "dunia. Indonesia memiliki keragaman budaya, bahasa, dan suku bangsa yang sangat kaya. "
            "Negara ini juga memiliki kekayaan alam yang melimpah, mulai dari hutan hujan tropis, "
            "gunung berapi aktif, hingga terumbu karang yang indah."
        ),
    },

    # === Translation (keunggulan encoder-decoder) ===
    {
        "category": "Translation EN->ID",
        "input": "Translate to Indonesian: The rapid advancement of technology has transformed the way we communicate, work, and live our daily lives.",
    },
    {
        "category": "Translation ID->EN",
        "input": "Terjemahkan ke bahasa Inggris: Pendidikan adalah kunci untuk membangun masa depan yang lebih baik bagi seluruh bangsa Indonesia.",
    },

    # === Question Answering ===
    {
        "category": "QA (EN)",
        "input": "What is photosynthesis and why is it important for life on Earth?",
    },
    {
        "category": "QA (ID)",
        "input": "Apa itu perubahan iklim dan bagaimana dampaknya terhadap Indonesia?",
    },

    # === General Instructions ===
    {
        "category": "Instruction (EN)",
        "input": "Explain the concept of machine learning in simple terms that a 10-year-old could understand.",
    },
    {
        "category": "Instruction (ID)",
        "input": "Jelaskan cara membuat nasi goreng yang enak langkah demi langkah.",
    },

    # === Text Generation / Creative ===
    {
        "category": "Creative (EN)",
        "input": "Write a short paragraph about the importance of reading books.",
    },
    {
        "category": "Creative (ID)",
        "input": "Tulis paragraf singkat tentang pentingnya menjaga lingkungan hidup.",
    },
]


def run_test_examples(model, tokenizer):
    """Jalankan test examples dan tampilkan hasilnya."""
    print("\n" + "=" * 60)
    print("Running test examples...")
    print("=" * 60)

    for i, test in enumerate(TEST_PROMPTS, 1):
        category = test["category"]
        prompt = test["input"]

        print(f"\n{'='*60}")
        print(f"[{i}/{len(TEST_PROMPTS)}] {category}")
        print(f"{'─'*60}")
        print(f"Input: {prompt[:200]}{'...' if len(prompt) > 200 else ''}")
        print(f"{'─'*60}")
        
        # Format the prompt properly
        SYSTEM_PROMPT = "Kamu adalah asisten AI yang helpful, santai, dan ramah. Gunakan Bahasa Indonesia sebagai bahasa utama."
        formatted_prompt = f"<start_of_turn>user\n{SYSTEM_PROMPT}\n\n{prompt}<end_of_turn>\n<start_of_turn>model\n"

        # Gunakan beam search untuk translation/summarization, sampling untuk lainnya
        if "Translation" in category or "Summarization" in category:
            response = generate_response(
                model, tokenizer, formatted_prompt,
                num_beams=4, do_sample=False
            )
        else:
            response = generate_response(
                model, tokenizer, formatted_prompt,
                do_sample=True, temperature=0.7
            )

        print(f"Output: {response}")


def run_interactive(model, tokenizer):
    """
    Interactive multi-turn mode.
    Conversation history di-maintain dan di-concat ke encoder input
    menggunakan format yang sama dengan training data.
    """
    print("\n" + "=" * 60)
    print("Interactive Multi-turn Mode")
    print("Ketik pesan dalam Bahasa Indonesia atau Inggris.")
    print("Commands:")
    print("  'quit' / 'exit'  — keluar")
    print("  'reset'          — hapus history percakapan")
    print("  'beam'           — toggle beam search (bagus untuk translate/summarize)")
    print("  'temp X'         — set temperature (e.g., 'temp 0.5')")
    print("  'history'        — tampilkan history saat ini")
    print("=" * 60)

    use_beam  = False
    temperature = 0.7
    # History: list of (user_msg, assistant_msg) tuples
    history: list[tuple[str, str]] = []
    MAX_HISTORY_TURNS = 5  # Batasi agar tidak overflow encoder

    def build_encoder_input(hist: list[tuple[str, str]], current_user: str) -> str:
        """Menggunakan format Gemma yang sama dengan training."""
        SYSTEM_PROMPT = "Kamu adalah asisten AI yang helpful, santai, dan ramah. Gunakan Bahasa Indonesia sebagai bahasa utama."
        formatted = f"<start_of_turn>user\n{SYSTEM_PROMPT}\n\n"
        
        # Tambahkan history jika ada
        for user_msg, asst_msg in hist:
            formatted += f"{user_msg.strip()}<end_of_turn>\n<start_of_turn>model\n{asst_msg.strip()}<end_of_turn>\n<start_of_turn>user\n"
            
        # Tambahkan input user saat ini
        formatted += f"{current_user.strip()}<end_of_turn>\n<start_of_turn>model\n"
        return formatted

    while True:
        try:
            user_input = input("\nYou: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nGoodbye!")
            break

        if not user_input:
            continue
        if user_input.lower() in ("quit", "exit", "q"):
            print("Goodbye!")
            break
        if user_input.lower() == "reset":
            history.clear()
            print("  ✅ History percakapan dihapus.")
            continue
        if user_input.lower() == "history":
            if not history:
                print("  (History kosong)")
            else:
                for i, (u, a) in enumerate(history, 1):
                    print(f"  [{i}] You: {u[:80]}")
                    print(f"       Model: {a[:80]}")
            continue
        if user_input.lower() == "beam":
            use_beam = not use_beam
            print(f"  Beam search: {'ON (4 beams)' if use_beam else 'OFF (sampling)'}")
            continue
        if user_input.lower().startswith("temp "):
            try:
                temperature = float(user_input.split()[1])
                print(f"  Temperature: {temperature}")
            except (ValueError, IndexError):
                print("  Usage: temp 0.5")
            continue

        # Trim history agar tidak overflow
        window = history[-MAX_HISTORY_TURNS:]
        encoder_input = build_encoder_input(window, user_input)

        if use_beam:
            response = generate_response(
                model, tokenizer, encoder_input,
                num_beams=4, do_sample=False,
            )
        else:
            response = generate_response(
                model, tokenizer, encoder_input,
                do_sample=True, temperature=temperature,
            )

        print(f"Model: {response}")

        # Simpan ke history
        history.append((user_input, response))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="T5Gemma2 Instruction Model Inference")
    parser.add_argument("--merged", action="store_true", help="Use merged model instead of LoRA adapter")
    parser.add_argument("--no-interactive", action="store_true", help="Skip interactive mode")
    parser.add_argument("--no-examples", action="store_true", help="Skip test examples")
    args = parser.parse_args()

    # Load model
    model, tokenizer = load_model(use_merged=args.merged)

    # Run test examples
    if not args.no_examples:
        run_test_examples(model, tokenizer)

    # Interactive mode
    if not args.no_interactive:
        run_interactive(model, tokenizer)
