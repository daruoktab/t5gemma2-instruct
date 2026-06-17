import json
import os
import sys
from pathlib import Path

def main():
    print("Loading T5-Gemma-2 tokenizer...")
    try:
        from transformers import AutoTokenizer
    except ImportError:
        print("Error: transformers is not installed.")
        sys.exit(1)
        
    try:
        tokenizer = AutoTokenizer.from_pretrained("google/t5gemma-2-270m-270m", trust_remote_code=True)
    except Exception as e:
        print(f"Error loading tokenizer: {e}")
        sys.exit(1)

    assert tokenizer is not None
    
    # We will use tqdm for progress bar
    try:
        from tqdm import tqdm
    except ImportError:
        def tqdm(iterable, *args, **kwargs):
            return iterable

    files = [
        "data/chat_train.jsonl",
        "data/chat_val.jsonl",
        "data/indoqa_train.jsonl",
        "data/indoqa_val.jsonl",
    ]

    for file_path_str in files:
        file_path = Path(file_path_str)
        if not file_path.exists():
            print(f"Warning: File {file_path_str} does not exist. Skipping.")
            continue
            
        print(f"Processing {file_path_str}...")
        
        # Read all lines
        with open(file_path, "r", encoding="utf-8") as f:
            lines = [line.strip() for line in f if line.strip()]
            
        updated_lines = []
        
        # Batch process tokenization for speed
        batch_size = 500
        for i in tqdm(range(0, len(lines), batch_size), desc=f"Tokenizing {file_path.name}"):
            batch_lines = lines[i : i + batch_size]
            batch_objs = [json.loads(line) for line in batch_lines]
            
            inputs = [obj.get("input", "") for obj in batch_objs]
            targets = [obj.get("target", "") for obj in batch_objs]
            
            # Tokenize batch
            input_encodings = tokenizer(inputs, add_special_tokens=False, verbose=False)["input_ids"]
            target_encodings = tokenizer(targets, add_special_tokens=False, verbose=False)["input_ids"]
            
            for obj, in_enc, tgt_enc in zip(batch_objs, input_encodings, target_encodings):
                obj["input_tokens"] = len(in_enc)
                obj["target_tokens"] = len(tgt_enc)
                updated_lines.append(json.dumps(obj, ensure_ascii=False) + "\n")
                
        # Safe replacement (write to temp, then overwrite)
        temp_file = file_path.with_suffix(".tmp")
        with open(temp_file, "w", encoding="utf-8") as f:
            f.writelines(updated_lines)
            
        if os.path.exists(file_path):
            os.remove(file_path)
        os.rename(temp_file, file_path)
        print(f"Successfully updated {file_path_str}\n")

if __name__ == "__main__":
    main()
