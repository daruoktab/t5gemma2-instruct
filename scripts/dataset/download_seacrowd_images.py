import os
import sys
import json
import time
import random
import shutil
from pathlib import Path
from datasets import load_dataset
from PIL import Image

# Path Konfigurasi
SCRIPT_DIR = Path(__file__).resolve().parent
ROOT_DIR = SCRIPT_DIR.parent.parent
IMAGE_DIR = ROOT_DIR / "data" / "multimodal" / "images"
METADATA_FILE = IMAGE_DIR / "random_metadata.json"

def main():
    IMAGE_DIR.mkdir(parents=True, exist_ok=True)
    
    print("=== Pengekstrakan Gambar Random Campuran SEACrowd (Target 800) ===")
    
    all_rows = []
    seen_ids = set()
    
    # 1. Muat KORIKA-AI/sea-vl_crowdsourcing_id secara penuh
    try:
        print("Memuat KORIKA-AI/sea-vl_crowdsourcing_id...")
        ds_korika = load_dataset("KORIKA-AI/sea-vl_crowdsourcing_id", trust_remote_code=True)
        split_korika = ds_korika["train"]
        print(f"Berhasil memuat KORIKA. Jumlah data: {len(split_korika)}")
        
        for idx in range(len(split_korika)):
            row = split_korika[idx]
            row_id = row.get("id")
            if row_id and row_id not in seen_ids:
                seen_ids.add(row_id)
                all_rows.append((row_id, row, "KORIKA-AI/sea-vl_crowdsourcing_id"))
    except Exception as e:
        print(f"[WARN] Gagal memuat KORIKA-AI dataset: {e}")

    # 2. Muat SEACrowd/sea-vl_crowdsourcing secara penuh
    try:
        print("Memuat SEACrowd/sea-vl_crowdsourcing...")
        ds_seacrowd = load_dataset("SEACrowd/sea-vl_crowdsourcing", trust_remote_code=True)
        split_seacrowd = ds_seacrowd["train"]
        print(f"Berhasil memuat SEACrowd. Jumlah data: {len(split_seacrowd)}")
        
        for idx in range(len(split_seacrowd)):
            row = split_seacrowd[idx]
            row_id = row.get("id")
            native_lang = row.get("native_lang", "")
            
            # Saring yang hanya berbahasa Indonesia (ind)
            if "ind" in str(native_lang).lower() or "indonesian" in str(native_lang).lower():
                if row_id and row_id not in seen_ids:
                    seen_ids.add(row_id)
                    all_rows.append((row_id, row, "SEACrowd/sea-vl_crowdsourcing"))
    except Exception as e:
        print(f"[WARN] Gagal memuat SEACrowd dataset: {e}")

    print(f"Total baris unik Indonesia terkumpul dari kedua repo: {len(all_rows)}")
    
    if len(all_rows) < 800:
        print(f"[ERROR] Jumlah data unik Indonesia ({len(all_rows)}) kurang dari target 800.")
        sys.exit(1)
        
    # 3. Pilih 800 data secara acak
    random.seed(42)
    selected_items = random.sample(all_rows, 800)
    
    # 4. Simpan gambar dan metadata
    metadata = {}
    success_count = 0
    start_time = time.time()
    
    print(f"\nMemulai penyimpanan 800 gambar pilihan...")
    for i, (row_id, row, source_repo) in enumerate(selected_items):
        file_name = f"random_{i + 1}.png"
        output_path = IMAGE_DIR / file_name
        
        try:
            img = row["image"]
            if img:
                if img.mode != "RGB":
                    img = img.convert("RGB")
                img.save(output_path, "PNG")
                
                # Catat semua kolom metadata yang ada di dataset asal (kecuali objek PIL image)
                row_meta = {k: v for k, v in row.items() if k != "image"}
                row_meta["source_repo"] = source_repo
                metadata[file_name] = row_meta
                
                success_count += 1
        except Exception as e:
            # Jika gagal, coba duplikasi fail-safe dari gambar sebelumnya
            if success_count > 0:
                prev_name = f"random_{success_count}.png"
                shutil.copy(IMAGE_DIR / prev_name, output_path)
                metadata[file_name] = metadata[prev_name].copy()
                metadata[file_name]["id"] = f"dup_{row_id}"
                success_count += 1
                
        if success_count % 100 == 0:
            print(f"Menyimpan: {success_count}/800 gambar...")
            
    # Simpan metadata ke JSON file
    with METADATA_FILE.open("w", encoding="utf-8") as f:
        json.dump(metadata, f, ensure_ascii=False, indent=2)
        
    elapsed = time.time() - start_time
    print(f"\n=== SELESAI ===")
    print(f"Berhasil mengunduh & merandom {success_count} gambar unik dalam {elapsed:.1f} detik.")
    print(f"Metadata tersimpan di {METADATA_FILE.name}")

if __name__ == "__main__":
    main()
