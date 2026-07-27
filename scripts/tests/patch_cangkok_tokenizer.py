"""Fix tokenizer_config.json repo cangkok: pakai dari v6 merged (lengkap dgn added_tokens_decoder)."""
from huggingface_hub import hf_hub_download, HfApi

REPO_CANGKOK = "daruokta/t5gemma-2-4b-4b-instruct-chat-indo-v4-vision-cangkok"
V6_REPO = "daruokta/t5gemma-2-4b-4b-instruct-chat-indo-v4-unsloth"
V6_SUBFOLDER = "merged_bf16"

# 1. Download tokenizer_config.json dari v6 merged (sudah lengkap: added_tokens_decoder + task_prefix_mapping)
print(f"Downloading tokenizer_config.json dari {V6_REPO}/{V6_SUBFOLDER}...")
path = hf_hub_download(repo_id=V6_REPO, filename="tokenizer_config.json", subfolder=V6_SUBFOLDER)
print(f"  ✅ Downloaded: {path}")

# 2. Upload ke repo cangkok (replace yang lama)
print(f"Uploading ke {REPO_CANGKOK}...")
api = HfApi()
api.upload_file(
    path_or_fileobj=path,
    path_in_repo="tokenizer_config.json",
    repo_id=REPO_CANGKOK,
    commit_message="Fix: replace dgn v6 merged tokenizer_config (lengkap added_tokens_decoder + task_prefix_mapping)",
)
print(f"  ✅ Uploaded! Repo cangkok sekarang punya tokenizer_config identik dgn v6 merged.")

