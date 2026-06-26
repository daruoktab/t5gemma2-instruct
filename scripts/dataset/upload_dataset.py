import os
import re
import json
from datasets import Dataset, DatasetDict

def parse_conversations(input_str, target_str):
    messages = []
    # Split input string by role markers. \b ensures we match the whole word at the start of a line or string
    parts = re.split(r'\n?(system|user|assistant):\s*', input_str.strip())
    
    current_role = None
    for part in parts:
        if part in ['system', 'user', 'assistant']:
            current_role = part
        elif current_role:
            content = part.strip()
            if content:
                messages.append({"role": current_role, "content": content})
            current_role = None
            
    # Finally, add the target as the last assistant message
    if target_str:
        messages.append({"role": "assistant", "content": target_str.strip()})
        
    return messages

def convert_jsonl_to_hf_format(file_path):
    formatted_data = []
    if not os.path.exists(file_path):
        print(f"Warning: File {file_path} not found.")
        return formatted_data
        
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            if not line.strip(): continue
            try:
                row = json.loads(line)
                if 'conversations' in row:
                    messages = row['conversations']
                elif 'messages' in row:
                    messages = row['messages']
                else:
                    messages = parse_conversations(row.get('input', ''), row.get('target', ''))
                formatted_data.append({"messages": messages})
            except Exception as e:
                print(f"Error parsing line in {file_path}: {e}")
    return formatted_data

def push_subset(repo_id, config_name, train_file, val_file=None):
    print(f"\nProcessing {config_name}...")
    
    train_data = convert_jsonl_to_hf_format(train_file)
    dataset_dict = DatasetDict({
        'train': Dataset.from_list(train_data)
    })
    
    if val_file and os.path.exists(val_file):
        val_data = convert_jsonl_to_hf_format(val_file)
        dataset_dict['validation'] = Dataset.from_list(val_data)
        
    print(f"Pushing {config_name} to hub ({len(train_data)} train samples)...")
    dataset_dict.push_to_hub(repo_id, config_name=config_name)
    print(f"✅ Successfully pushed {config_name}!")

def main():
    REPO_ID = "daruokta/t5gemma2-indonesia-chat-formatted"
    
    # 1. Chat Multiturn (3000 Curated Synthetic Data)
    push_subset(
        repo_id=REPO_ID, 
        config_name="chat_multiturn", 
        train_file="data/t5-gemma-2-chat-instruct-dataset-v2.jsonl"
    )

    
    # 3. IndoQA 
    # push_subset(
    #     repo_id=REPO_ID, 
    #     config_name="indoqa_documents", 
    #     train_file="../../data/indoqa_train.jsonl",
    #     val_file="../../data/indoqa_val.jsonl"
    # )
    
    print("\n🎉 ALL DATASETS HAVE BEEN SUCCESSFULLY UPLOADED TO HUGGINGFACE!")

if __name__ == "__main__":
    main()
