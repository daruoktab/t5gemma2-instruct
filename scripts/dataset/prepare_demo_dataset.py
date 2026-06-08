import os
import json
import random
from transformers import AutoTokenizer

def default_system_prompt():
    return (
        "Kamu adalah asisten AI yang helpful, santai, dan ramah. "
        "Gunakan Bahasa Indonesia sebagai bahasa utama. "
        "Switch ke English hanya kalau user memang minta atau konteksnya English. "
        "Boleh casual dan natural — pakai 'aku/kamu' atau 'saya/Anda' sesuai situasi. "
        "Kalau ada task seperti translate, summarize, paraphrase, atau rewrite muncul dalam obrolan, "
        "langsung bantu dengan natural tanpa basa-basi berlebihan. Jangan terlalu formal kecuali situasinya memang mengharuskan."
    )

def parse_system_and_turns(messages, fallback_system):
    system = fallback_system.strip()
    turns = []
    pending_user = None

    for m in messages:
        role = (m.get("role") or "").strip()
        content = (m.get("content") or "").strip()
        if not content:
            continue
        if role == "system":
            system = content
        elif role == "user":
            pending_user = content
        elif role == "assistant":
            if pending_user is None:
                continue
            turns.append((pending_user, content))
            pending_user = None

    return system, turns

def conversations_to_sft_rows(messages, fallback_system):
    system, turns = parse_system_and_turns(messages, fallback_system)
    rows = []
    for k in range(len(turns)):
        parts = [f"system: {system}"]
        for i in range(k):
            parts.append(f"user: {turns[i][0]}")
            parts.append(f"assistant: {turns[i][1]}")
        parts.append(f"user: {turns[k][0]}")
        inp = "\n".join(parts).strip()
        rows.append((inp, turns[k][1]))
    return rows

def format_encoder_from_raw(raw_input, fallback_system):
    # This formats the encoder inputs into the Gemma chat template format
    import re
    system_match = re.search(r'^system:\s*(.*?)(?=\nuser:)', raw_input, re.DOTALL)
    system = system_match.group(1).strip() if system_match else fallback_system

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

def main():
    MODEL_NAME = "google/t5gemma-2-270m-270m"
    NESTED_FILE = "data/t5-gemma-2-chat-instruct-merged.jsonl"
    
    print(f"Loading tokenizer {MODEL_NAME}...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
    
    fallback_system = default_system_prompt()
    
    valid_threads = []
    
    print(f"Reading {NESTED_FILE} and filtering threads...")
    with open(NESTED_FILE, "r", encoding="utf-8") as f:
        for idx, line in enumerate(f):
            if not line.strip():
                continue
            obj = json.loads(line)
            convs = obj.get("conversations", [])
            if not convs:
                continue
                
            # Convert to SFT turns
            turns = conversations_to_sft_rows(convs, fallback_system)
            
            # Check length constraints for ALL turns in the thread
            thread_valid = True
            max_in = 0
            max_out = 0
            
            for inp, tgt in turns:
                # Format to actual training template
                inp_formatted = format_encoder_from_raw(inp, fallback_system)
                tgt_formatted = tgt.strip() + "<end_of_turn>"
                
                # Tokenize
                inp_ids = tokenizer.encode(inp_formatted, add_special_tokens=False)
                tgt_ids = tokenizer.encode(tgt_formatted, add_special_tokens=False)
                
                max_in = max(max_in, len(inp_ids))
                max_out = max(max_out, len(tgt_ids))
                
                if len(inp_ids) > 4096 or len(tgt_ids) > 1024:
                    thread_valid = False
                    break
            
            if thread_valid:
                valid_threads.append({
                    "id": obj.get("id", idx),
                    "conversations": convs,
                    "max_input_len": max_in,
                    "max_target_len": max_out
                })
                
    print(f"Found {len(valid_threads)} valid threads matching constraints (max_input <= 4096, max_target <= 1024).")
    
    if len(valid_threads) < 110:
        print("Warning: Not enough threads to sample 100 for train and 10 for validation. Sampling all available.")
        
    random.seed(42)
    random.shuffle(valid_threads)
    
    train_threads = valid_threads[:100]
    val_threads = valid_threads[100:110]
    
    # Flatten and save
    train_out_file = "data/chat_train_demo.jsonl"
    val_out_file = "data/chat_val_demo.jsonl"
    
    print(f"Saving train SFT turns to {train_out_file}...")
    n_train_turns = 0
    with open(train_out_file, "w", encoding="utf-8") as f_train:
        for t in train_threads:
            turns = conversations_to_sft_rows(t["conversations"], fallback_system)
            for inp, tgt in turns:
                f_train.write(json.dumps({"input": inp, "target": tgt}, ensure_ascii=False) + "\n")
                n_train_turns += 1
                
    print(f"Saving validation SFT turns to {val_out_file}...")
    n_val_turns = 0
    with open(val_out_file, "w", encoding="utf-8") as f_val:
        for t in val_threads:
            turns = conversations_to_sft_rows(t["conversations"], fallback_system)
            for inp, tgt in turns:
                f_val.write(json.dumps({"input": inp, "target": tgt}, ensure_ascii=False) + "\n")
                n_val_turns += 1
                
    print(f"✅ Created {train_out_file} with {n_train_turns} SFT turns (from 100 threads).")
    print(f"✅ Created {val_out_file} with {n_val_turns} SFT turns (from 10 threads).")

if __name__ == "__main__":
    main()
