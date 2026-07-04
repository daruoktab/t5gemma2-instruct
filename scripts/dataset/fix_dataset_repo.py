import os

file_path = "d:/Codings/unsloth-porto/t5-gemma-2/instruct/working-molab-v6-unsloth.py"

with open(file_path, "r", encoding="utf-8") as f:
    content = f.read()

# 1. Update Config Cell in Molab V6
config_old = """    # Tambahan: Repo ID khusus untuk menyimpan Checkpoint hasil training secara otomatis
    HF_CHECKPOINT_REPO = "daruokta/t5gemma-2-4b-4b-instruct-chat-indo-v4-unsloth"
    HF_CHECKPOINT_REPO_ORPO = "daruokta/t5gemma-2-4b-4b-instruct-chat-indo-v4-unsloth-orpo"
    HF_ORPO_DATASET_REPO = "daruokta/t5gemma2-indonesia-orpo"
    ORPO_BETA = 0.1
    OUTPUT_DIR_ORPO = "results/t5gemma2-orpo"
    CHAT_CONFIG = "chat_sft"
    INDOQA_CONFIG = "indoqa_sft"
"""

config_new = """    # Tambahan: Repo ID khusus untuk menyimpan Checkpoint hasil training secara otomatis
    HF_CHECKPOINT_REPO = "daruokta/t5gemma-2-4b-4b-instruct-chat-indo-v4-unsloth"
    HF_CHECKPOINT_REPO_ORPO = "daruokta/t5gemma-2-4b-4b-instruct-chat-indo-v4-unsloth-orpo"
    ORPO_BETA = 0.1
    OUTPUT_DIR_ORPO = "results/t5gemma2-orpo"
    
    # Dataset Subsets (Configs)
    CHAT_CONFIG = "chat_sft"
    INDOQA_CONFIG = "indoqa_sft"
    ORPO_CONFIG = "orpo"
"""
content = content.replace(config_old, config_new)

# 2. Update ORPO Load Dataset in Molab V6
# Find the line: orpo_dataset = load_dataset(HF_ORPO_DATASET_REPO, split="train")
# But we need to also update the function arguments in @app.cell
cell_args_old = """@app.cell
def _(
    HF_CHECKPOINT_REPO,
    HF_CHECKPOINT_REPO_ORPO,
    HF_ORPO_DATASET_REPO,
    ORPO_BETA,"""

cell_args_new = """@app.cell
def _(
    HF_CHECKPOINT_REPO,
    HF_CHECKPOINT_REPO_ORPO,
    HF_REPO_ID,
    ORPO_CONFIG,
    ORPO_BETA,"""
content = content.replace(cell_args_old, cell_args_new)

load_ds_old = """        # 2. Muat Dataset ORPO
        print(f"\\n[ORPO] Mendownload dataset ORPO dari {HF_ORPO_DATASET_REPO}...")
        try:
            orpo_dataset = load_dataset(HF_ORPO_DATASET_REPO, split="train")"""

load_ds_new = """        # 2. Muat Dataset ORPO
        print(f"\\n[ORPO] Mendownload dataset ORPO dari {HF_REPO_ID} (subset: {ORPO_CONFIG})...")
        try:
            orpo_dataset = load_dataset(HF_REPO_ID, name=ORPO_CONFIG, split="train")"""
content = content.replace(load_ds_old, load_ds_new)

with open(file_path, "w", encoding="utf-8") as f:
    f.write(content)
print("working-molab-v6-unsloth.py dataset repo logic fixed!")

# Now fix generate_orpo_dataset.py (Remove the auto upload so the user can review locally)
orpo_gen_path = "d:/Codings/unsloth-porto/t5-gemma-2/instruct/scripts/dataset/generate_orpo_dataset.py"
with open(orpo_gen_path, "r", encoding="utf-8") as f:
    content_gen = f.read()

import re
# Regex to remove the auto upload try-except block
upload_block = re.search(r"    # Auto upload to Hugging Face Hub\n    try:.*?if __name__ == \"__main__\":", content_gen, re.DOTALL)
if upload_block:
    content_gen = content_gen.replace(upload_block.group(0), "if __name__ == \"__main__\":")
    with open(orpo_gen_path, "w", encoding="utf-8") as f:
        f.write(content_gen)
    print("generate_orpo_dataset.py auto-upload removed!")
else:
    print("Auto-upload block not found in generate_orpo_dataset.py")
