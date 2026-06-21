"""
Filter OASST2 dataset - Bahasa Indonesia only
============================================================
Jalankan di environment yang punya akses HuggingFace:
    pip install datasets pandas pyarrow

Output:
    oasst2_indonesian_train.parquet
    oasst2_indonesian_validation.parquet
    oasst2_indonesian_full.csv
"""

from datasets import load_dataset
import pandas as pd
from typing import Any, cast

# ─────────────────────────────────────────
# 1. Load dataset
# ─────────────────────────────────────────
print("Loading OpenAssistant/oasst2 ...")
ds = load_dataset("OpenAssistant/oasst2")

# ─────────────────────────────────────────
# 2. Filter bahasa Indonesia (lang == "id")
# ─────────────────────────────────────────
splits = {}
for split_name in ds:
    filtered = ds[split_name].filter(lambda x: x["lang"] == "id")
    splits[split_name] = filtered
    print(f"  [{split_name}] {len(ds[split_name]):,} rows → {len(filtered):,} Indonesian rows")

# ─────────────────────────────────────────
# 3. Simpan ke parquet per split
# ─────────────────────────────────────────
for split_name, split_ds in splits.items():
    out_path = f"oasst2_indonesian_{split_name}.parquet"
    cast(Any, split_ds).to_parquet(out_path)
    print(f"Saved: {out_path}")

# ─────────────────────────────────────────
# 4. Gabungkan semua split → CSV
# ─────────────────────────────────────────
dfs = []
for split_name, split_ds in splits.items():
    df = cast(Any, split_ds).to_pandas()
    df["split"] = split_name
    dfs.append(df)

df_all = pd.concat(dfs, ignore_index=True)
df_all.to_csv("oasst2_indonesian_full.csv", index=False)
print(f"\nGabungan semua split: {len(df_all):,} rows")
print("Saved: oasst2_indonesian_full.csv")

# ─────────────────────────────────────────
# 5. Preview
# ─────────────────────────────────────────
print("\n=== Preview 5 baris pertama ===")
cols = ["role", "lang", "text", "split"]
print(df_all[cols].head().to_string(max_colwidth=80))

# Distribusi role
print("\n=== Distribusi role ===")
print(df_all["role"].value_counts().to_string())

# Optional: buat dataset percakapan (thread) yang hanya ID
# ─────────────────────────────────────────────────────────
print("\n=== Tree IDs yang mengandung ID ===")
id_tree_ids = set(df_all["message_tree_id"].unique())
print(f"  Jumlah message tree unik: {len(id_tree_ids)}")
