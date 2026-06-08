import os
from huggingface_hub import HfApi

def main():
    repo_id = "daruokta/t5gemma2-indonesia-chat-formatted"
    readme_path = os.path.join(os.path.dirname(__file__), "..", "..", "docs", "HF_README.md")
    
    if not os.path.exists(readme_path):
        print(f"Error: {readme_path} tidak ditemukan.")
        return

    print("Mengupload README.md ke HuggingFace...")
    api = HfApi()
    
    try:
        api.upload_file(
            path_or_fileobj=readme_path,
            path_in_repo="README.md",
            repo_id=repo_id,
            repo_type="dataset",
        )
        print("✅ README.md berhasil diperbarui di HuggingFace!")
    except Exception as e:
        print(f"❌ Gagal mengupload README: {e}")

if __name__ == "__main__":
    main()
