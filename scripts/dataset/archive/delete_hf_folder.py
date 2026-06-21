import sys
from huggingface_hub import HfApi

def main():
    REPO_ID = "daruokta/t5gemma2-indonesia-chat-formatted"
    FOLDERS_TO_DELETE = ["chat_full", "chat_seed"]
    
    api = HfApi()
    
    for folder in FOLDERS_TO_DELETE:
        print(f"Attempting to delete folder '{folder}' from Hugging Face dataset '{REPO_ID}'...")
        try:
            api.delete_folder(
                repo_id=REPO_ID,
                path_in_repo=folder,
                repo_type="dataset",
                commit_message=f"Delete unused '{folder}' folder"
            )
            print(f"✅ Successfully deleted folder '{folder}' from Hugging Face!")
        except Exception as e:
            print(f"❌ Error deleting folder '{folder}': {e}", file=sys.stderr)

if __name__ == "__main__":
    main()
