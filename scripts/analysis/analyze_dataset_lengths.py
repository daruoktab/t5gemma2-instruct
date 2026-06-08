"""
Analyze exact token length distributions of dataset inputs and targets.
"""
import os
import re
import json
import numpy as np
from transformers import AutoTokenizer, PreTrainedTokenizerFast

MODEL_NAME = "google/t5gemma-2-270m-270m" # Tokenizers are identical, using 270m is faster/cached
CHAT_TRAIN_FILE   = "data/chat_train.jsonl"
INDOQA_TRAIN_FILE = "data/indoqa_train.jsonl"

SYSTEM_PROMPT = (
    "Kamu adalah asisten AI yang helpful, santai, dan ramah. "
    "Gunakan Bahasa Indonesia sebagai bahasa utama."
)

def format_encoder_from_raw(raw_input: str) -> str:
    system_match = re.search(r'^system:\s*(.*?)(?=\nuser:)', raw_input, re.DOTALL)
    system = system_match.group(1).strip() if system_match else SYSTEM_PROMPT

    if system_match:
        raw_input = raw_input[system_match.end():].strip()

    parts = re.split(r'\n(user:|assistant:)\s*', '\n' + raw_input)
    formatted = ''
    is_first_user = True

    for i in range(1, len(parts), 2):
        role = parts[i].replace(':', '').strip()
        content = parts[i + 1].strip()
        if not content:
            continue

        if role == 'user':
            formatted += '<start_of_turn>user\n'
            if is_first_user and system:
                formatted += system + '\n\n'
                is_first_user = False
            formatted += content + '<end_of_turn>\n'
        elif role == 'assistant':
            formatted += '<start_of_turn>model\n'
            formatted += content + '<end_of_turn>\n'

    formatted += '<start_of_turn>model\n'
    return formatted

def analyze_file(path, tokenizer):
    if not os.path.exists(path):
        print(f"File {path} not found.")
        return [], []
    
    input_lengths = []
    target_lengths = []

    print(f"Reading and tokenizing {path}...")
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            obj = json.loads(line)
            inp_raw = obj.get("input", "")
            tgt_raw = obj.get("target", "")

            inp_f = format_encoder_from_raw(inp_raw)
            tgt_f = tgt_raw.strip() + "<end_of_turn>"

            # Encode to get token ids
            inp_ids = tokenizer.encode(inp_f, add_special_tokens=False)
            tgt_ids = tokenizer.encode(tgt_f, add_special_tokens=False)

            input_lengths.append(len(inp_ids))
            target_lengths.append(len(tgt_ids))

    return input_lengths, target_lengths

def print_stats(name, lengths):
    if not lengths:
        print(f"No data for {name}")
        return
    arr = np.array(lengths)
    print(f"\nStats for {name}:")
    print(f"  Total samples: {len(arr)}")
    print(f"  Min length:    {arr.min()}")
    print(f"  Max length:    {arr.max()}")
    print(f"  Mean length:   {arr.mean():.2f}")
    print(f"  Median (p50):  {np.percentile(arr, 50)}")
    print(f"  p95 length:    {np.percentile(arr, 95)}")
    print(f"  p99 length:    {np.percentile(arr, 99)}")
    print(f"  p99.9 length:  {np.percentile(arr, 99.9)}")

def main():
    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
    assert isinstance(tokenizer, PreTrainedTokenizerFast)

    # 1. Analyze Chat Train
    chat_in, chat_tgt = analyze_file(CHAT_TRAIN_FILE, tokenizer)
    print_stats("Chat Train Input (Source)", chat_in)
    print_stats("Chat Train Target (Label)", chat_tgt)

    # 2. Analyze IndoQA Train
    qa_in, qa_tgt = analyze_file(INDOQA_TRAIN_FILE, tokenizer)
    print_stats("IndoQA Train Input (Source)", qa_in)
    print_stats("IndoQA Train Target (Label)", qa_tgt)

    # 3. Combined Stats
    combined_in = chat_in + qa_in
    combined_tgt = chat_tgt + qa_tgt
    print("\n" + "="*50)
    print("  COMBINED DATASET STATS (Chat + IndoQA)")
    print("="*50)
    print_stats("Combined Input (Source)", combined_in)
    print_stats("Combined Target (Label)", combined_tgt)

if __name__ == "__main__":
    main()
