import json
import random
import sys
from pathlib import Path
from transformers import AutoTokenizer, PreTrainedTokenizerFast

# Setup paths
SCRIPT_DIR = Path(__file__).resolve().parent
ROOT_DIR = SCRIPT_DIR.parent.parent
DATA_DIR = ROOT_DIR / "data"

INPUT_EDITED = DATA_DIR / "t5-gemma-2-chat-instruct-dataset-edited.jsonl"
INPUT_AGENTIC = DATA_DIR / "generated_prefix_tasks_agentic.jsonl"

OUTPUT_FULL_V2 = DATA_DIR / "t5-gemma-2-chat-instruct-dataset-v2.jsonl"
OUTPUT_TRAIN = DATA_DIR / "chat_train_v2.jsonl"
OUTPUT_VAL = DATA_DIR / "chat_val_v2.jsonl"
TOKENIZER_DIR = DATA_DIR / "tokenizernya-t5gemma2"

# Import helper functions from flatten_conversations_jsonl_to_sft
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from flatten_conversations_jsonl_to_sft import (  # type: ignore
    conversations_to_sft_rows,
    default_system_prompt,
)

def main():
    if not INPUT_EDITED.exists():
        print(f"[ERROR] Edited dataset file not found: {INPUT_EDITED}")
        sys.exit(1)

    if not INPUT_AGENTIC.exists():
        print(f"[ERROR] Agentic dataset file not found: {INPUT_AGENTIC}")
        sys.exit(1)

    if not TOKENIZER_DIR.exists():
        print(f"[ERROR] Tokenizer directory not found: {TOKENIZER_DIR}")
        sys.exit(1)

    # 1. Load tokenizer
    print(f"[INFO] Loading local tokenizer from {TOKENIZER_DIR}...")
    tokenizer = AutoTokenizer.from_pretrained(str(TOKENIZER_DIR), trust_remote_code=True)
    assert isinstance(tokenizer, PreTrainedTokenizerFast)

    # 2. Read conversations
    conversations = []
    
    print(f"[INFO] Reading {INPUT_EDITED}...")
    with INPUT_EDITED.open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            conversations.append(json.loads(line))
    edited_count = len(conversations)
    print(f"[INFO] Loaded {edited_count} edited conversations.")

    print(f"[INFO] Reading {INPUT_AGENTIC}...")
    with INPUT_AGENTIC.open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            conversations.append(json.loads(line))
    agentic_count = len(conversations) - edited_count
    print(f"[INFO] Loaded {agentic_count} agentic prefix-task conversations.")

    total_conversations = len(conversations)
    print(f"[INFO] Total merged conversations: {total_conversations}")

    # 3. Save Full Merged Version V2
    print(f"[INFO] Saving Full Merged Version to {OUTPUT_FULL_V2}...")
    with OUTPUT_FULL_V2.open("w", encoding="utf-8") as f:
        for conv in conversations:
            # Hitung total token untuk percakapan ini
            total_text = " ".join([t["content"] for t in conv["conversations"]])
            conv["tokens"] = len(tokenizer.encode(total_text, add_special_tokens=False))
            
            f.write(json.dumps(conv, ensure_ascii=False) + "\n")
    print(f"[INFO] Successfully saved {total_conversations} full conversations to V2 file.")

    # 4. Shuffle and split for flattening
    random.seed(42)
    random.shuffle(conversations)

    val_size = min(100, total_conversations)
    val_convs = conversations[:val_size]
    train_convs = conversations[val_size:]

    print(f"[INFO] Splitting for flattened SFT: Train={len(train_convs)}, Val={len(val_convs)}")

    fallback_sys = default_system_prompt()

    def process_and_write(convs, output_path):
        flattened_count = 0
        print(f"[INFO] Flattening and tokenizing to {output_path.name}...")
        with output_path.open("w", encoding="utf-8") as out_f:
            for item in convs:
                chat_idx = item["id"]
                messages = item["conversations"]
                
                sft_rows = conversations_to_sft_rows(messages, fallback_sys, chat_idx)
                
                for row in sft_rows:
                    inp_tokens = len(tokenizer.encode(row["input"], add_special_tokens=True))
                    tgt_tokens = len(tokenizer.encode(row["target"], add_special_tokens=False))
                    
                    row["input_tokens"] = inp_tokens
                    row["target_tokens"] = tgt_tokens
                    
                    out_f.write(json.dumps(row, ensure_ascii=False) + "\n")
                    flattened_count += 1
        print(f"       -> Saved {flattened_count} flattened turns to {output_path.name}")

    # 5. Process and write flattened splits
    process_and_write(train_convs, OUTPUT_TRAIN)
    process_and_write(val_convs, OUTPUT_VAL)

    print("[INFO] Done! V2 Dataset Generation and Flattening Completed.")
    print("Files ready to be uploaded to Hugging Face:")
    print(f"1. Full Conv: {OUTPUT_FULL_V2.name}")
    print(f"2. Flattened Train: {OUTPUT_TRAIN.name}")
    print(f"3. Flattened Val: {OUTPUT_VAL.name}")

if __name__ == "__main__":
    main()
