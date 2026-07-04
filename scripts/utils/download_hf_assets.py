from huggingface_hub import hf_hub_download
import shutil
import os

repos = {
    "v4": {
        "repo": "daruokta/t5gemma-2-1b-1b-instruct-chat-indo-v2",
        "latest_checkpoint": "checkpoint-1000"
    },
    "v4_exp": {
        "repo": "daruokta/t5gemma-2-1b-1b-instruct-chat-indo-v2-exp",
        "latest_checkpoint": "checkpoint-1225"
    }
}

os.makedirs("scratch", exist_ok=True)

for name, info in repos.items():
    repo = info["repo"]
    chk = info["latest_checkpoint"]
    print(f"\nDownloading assets for {name} ({repo})...")
    
    # 1. Download eval_samples.txt
    try:
        path = hf_hub_download(repo_id=repo, filename="eval_samples.txt")
        dest = f"scratch/{name}_eval_samples.txt"
        shutil.copy(path, dest)
        print(f"  Downloaded eval_samples.txt -> {dest}")
    except Exception as e:
        print(f"  Error downloading eval_samples.txt: {e}")
        
    # 2. Download training_chart.png
    try:
        path = hf_hub_download(repo_id=repo, filename="training_chart.png")
        dest = f"scratch/{name}_training_chart.png"
        shutil.copy(path, dest)
        print(f"  Downloaded training_chart.png -> {dest}")
    except Exception as e:
        print(f"  Error downloading training_chart.png: {e}")
        
    # 3. Download trainer_state.json from latest checkpoint
    try:
        filename = f"{chk}/trainer_state.json"
        path = hf_hub_download(repo_id=repo, filename=filename)
        dest = f"scratch/{name}_trainer_state.json"
        shutil.copy(path, dest)
        print(f"  Downloaded {filename} -> {dest}")
    except Exception as e:
        print(f"  Error downloading {filename}: {e}")
