import os
import json
from datasets import Dataset, DatasetDict

def load_jsonl_samples(file_path):
    formatted_data = []
    if not os.path.exists(file_path):
        print(f"Warning: File {file_path} not found.")
        return formatted_data
        
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            if not line.strip(): 
                continue
            try:
                row = json.loads(line)
                if 'input' in row and 'target' in row:
                    item = {
                        "input": row['input'],
                        "target": row['target']
                    }
                    if 'chat_idx' in row:
                        item['chat_idx'] = row['chat_idx']
                    if 'turn_idx' in row:
                        item['turn_idx'] = row['turn_idx']
                    if 'input_tokens' in row:
                        item['input_tokens'] = row['input_tokens']
                    if 'target_tokens' in row:
                        item['target_tokens'] = row['target_tokens']
                    formatted_data.append(item)
            except Exception as e:
                print(f"Error parsing line in {file_path}: {e}")
    return formatted_data

def push_sft_dataset(repo_id, config_name, train_file, val_file):
    print(f"\nProcessing SFT dataset for {config_name}...")
    
    train_data = load_jsonl_samples(train_file)
    val_data = load_jsonl_samples(val_file)
    
    if not train_data:
        print(f"Error: No training data found for {train_file}. Aborting push.")
        return
        
    dataset_dict = DatasetDict({
        'train': Dataset.from_list(train_data)
    })
    
    if val_data:
        dataset_dict['validation'] = Dataset.from_list(val_data)
        
    print(f"Pushing SFT config '{config_name}' to HF ({len(train_data)} train, {len(val_data)} val)...")
    dataset_dict.push_to_hub(repo_id, config_name=config_name)
    print(f"✅ Successfully pushed SFT config '{config_name}'!")

def main():
    REPO_ID = "daruokta/t5gemma2-indonesia-chat-formatted"
    
    # 1. Pushing Chat SFT (Flattened/Unrolled)
    push_sft_dataset(
        repo_id=REPO_ID,
        config_name="chat_sft",
        train_file="data/chat_train.jsonl",
        val_file="data/chat_val.jsonl"
    )
    
    # 2. Pushing IndoQA SFT (Flattened)
    push_sft_dataset(
        repo_id=REPO_ID,
        config_name="indoqa_sft",
        train_file="data/indoqa_train.jsonl",
        val_file="data/indoqa_val.jsonl"
    )
    
    print("\n🎉 SFT DATASETS HAVE BEEN SUCCESSFULLY UPLOADED TO HUGGINGFACE!")

if __name__ == "__main__":
    main()
