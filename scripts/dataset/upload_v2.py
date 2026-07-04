import os
import json
from datetime import datetime
from datasets import Dataset, DatasetDict
from huggingface_hub import HfApi, hf_hub_download

REPO_ID = "daruokta/t5gemma2-indonesia-chat-formatted"
api = HfApi()

def load_jsonl_samples(file_path):
    formatted_data = []
    if not os.path.exists(file_path):
        print(f"Warning: File {file_path} not found.")
        return formatted_data
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            if not line.strip(): 
                continue
            row = json.loads(line)
            formatted_data.append(row)
    return formatted_data

def push_config(config_name, train_file, val_file=None):
    print(f"\nProcessing {config_name}...")
    train_data = load_jsonl_samples(train_file)
    dataset_dict = DatasetDict({
        'train': Dataset.from_list(train_data)
    })
    
    if val_file and os.path.exists(val_file):
        val_data = load_jsonl_samples(val_file)
        dataset_dict['validation'] = Dataset.from_list(val_data)
        
    print(f"Pushing '{config_name}' to HF...")
    dataset_dict.push_to_hub(REPO_ID, config_name=config_name)
    print(f"✅ Successfully pushed '{config_name}'!")

def update_readme():
    print("\nUpdating README.md...")
    try:
        readme_path = hf_hub_download(repo_id=REPO_ID, filename="README.md", repo_type="dataset")
        with open(readme_path, "r", encoding="utf-8") as f:
            content = f.read()
        
        today = datetime.now().strftime("%Y-%m-%d")
        changelog = f"\n- **{today} (Update V2)**: Menggabungkan dataset lama (2500 percakapan hasil edit) dengan dataset tambahan baru (500 percakapan tipe agentic/perfect-response). Total 3000 percakapan. File konfigurasi `chat_sft` (flattened) dan `chat_multiturn` (full nested) telah ditimpa/diperbarui dengan data V2 ini."
        
        if "## Changelog" not in content:
            content += "\n\n## Changelog\n" + changelog
        else:
            content = content.replace("## Changelog", f"## Changelog\n{changelog}")
            
        with open("temp_readme.md", "w", encoding="utf-8") as f:
            f.write(content)
            
        api.upload_file(
            path_or_fileobj="temp_readme.md",
            path_in_repo="README.md",
            repo_id=REPO_ID,
            repo_type="dataset"
        )
        os.remove("temp_readme.md")
        print("✅ Successfully updated README.md!")
    except Exception as e:
        print(f"⚠️ Failed to update README.md: {e}")

def main():
    # 1. Pushing Chat SFT (Flattened/Unrolled)
    push_config(
        config_name="chat_sft",
        train_file="d:/Codings/unsloth-porto/t5-gemma-2/instruct/data/chat_train_v2.jsonl",
        val_file="d:/Codings/unsloth-porto/t5-gemma-2/instruct/data/chat_val_v2.jsonl"
    )
    
    # 2. Pushing Chat Multiturn (Full Conversation)
    push_config(
        config_name="chat_multiturn",
        train_file="d:/Codings/unsloth-porto/t5-gemma-2/instruct/data/t5-gemma-2-chat-instruct-dataset-v2.jsonl"
    )
    
    # 3. Update README
    update_readme()

if __name__ == "__main__":
    main()
