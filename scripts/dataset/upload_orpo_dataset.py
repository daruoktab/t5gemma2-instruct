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
                item = {
                    "id": row.get('id'),
                    "prompt": row.get('prompt'),
                    "chosen": row.get('chosen'),
                    "rejected": row.get('rejected'),
                    "flaw": row.get('flaw'),
                    "rationale": row.get('rationale')
                }
                formatted_data.append(item)
            except Exception as e:
                print(f"Error parsing line in {file_path}: {e}")
    return formatted_data

def push_orpo_dataset(repo_id, config_name, train_file):
    print(f"\nProcessing ORPO dataset for {config_name}...")
    
    train_data = load_jsonl_samples(train_file)
    
    if not train_data:
        print(f"Error: No training data found for {train_file}. Aborting push.")
        return
        
    dataset_dict = DatasetDict({
        'train': Dataset.from_list(train_data)
    })
    
    print(f"Pushing config '{config_name}' to HF ({len(train_data)} train)...")
    dataset_dict.push_to_hub(repo_id, config_name=config_name)
    print(f"✅ Successfully pushed config '{config_name}'!")

def main():
    REPO_ID = "daruokta/t5gemma2-indonesia-chat-formatted"
    
    push_orpo_dataset(
        repo_id=REPO_ID,
        config_name="chat_orpo",
        train_file="data/orpo_train.jsonl"
    )
    
    print("\n🎉 ORPO DATASET HAS BEEN SUCCESSFULLY UPLOADED TO HUGGINGFACE!")

if __name__ == "__main__":
    main()
