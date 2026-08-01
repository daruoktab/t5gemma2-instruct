"""
Hitung Jumlah Percakapan (Unflattened Rows) & Turn di Seluruh Dataset
"""

import json
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent.parent

def check_file(path: Path):
    if not path.exists():
        return None
    
    count_rows = 0
    total_turns = 0
    has_images = 0
    
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
                count_rows += 1
                convs = obj.get("conversations") or obj.get("messages") or []
                # Hitung turn (pasangan user-assistant)
                turns = sum(1 for m in convs if m.get("role") in ("user", "assistant")) // 2
                total_turns += turns
                if "images" in obj or "image" in obj:
                    has_images += 1
            except Exception:
                pass
                
    return {
        "file": path.name,
        "path": str(path.relative_to(ROOT_DIR)),
        "unflattened_conversations": count_rows,
        "total_user_assistant_pairs": total_turns,
        "has_images_count": has_images
    }

def main():
    sft_dir = ROOT_DIR / "data" / "sft"
    multimodal_dir = ROOT_DIR / "data" / "multimodal"
    synthetic_dir = ROOT_DIR / "data" / "synthetic"
    
    files_to_check = [
        # SFT / Text Datasets
        sft_dir / "t5-gemma-2-chat-instruct-dataset.jsonl",
        sft_dir / "t5-gemma-2-chat-instruct-dataset-edited.jsonl",
        sft_dir / "t5-gemma-2-chat-instruct-dataset-v2.jsonl",
        sft_dir / "chat_train.jsonl",
        sft_dir / "chat_val.jsonl",
        sft_dir / "chat_train_v2.jsonl",
        sft_dir / "chat_val_v2.jsonl",
        
        # Multimodal Datasets
        multimodal_dir / "train_vision.jsonl",
        multimodal_dir / "train_vision_fixed.jsonl",
        
        # Synthetic / Extra Datasets
        synthetic_dir / "generated_prefix_tasks_agentic.jsonl",
        ROOT_DIR / "longform_output_dataset.jsonl",
    ]
    
    print("=========================================================================")
    print("  HASIL PEMERIKSAAN JUMLAH DATASET (UNFLATTENED PERCAKAPAN)              ")
    print("=========================================================================\n")
    
    for p in files_to_check:
        res = check_file(p)
        if res:
            print(f"📄 File: {res['file']}")
            print(f"   Path                       : {res['path']}")
            print(f"   Jumlah Percakapan (Rows)   : {res['unflattened_conversations']:,}")
            print(f"   Total Turns (User-Assistant): {res['total_user_assistant_pairs']:,}")
            if res['has_images_count'] > 0:
                print(f"   Gambar (Multimodal)        : {res['has_images_count']:,}")
            print("-" * 65)

if __name__ == "__main__":
    main()
