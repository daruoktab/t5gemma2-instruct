"""
Download tokenizer.json dari google/gemma-3-4b-it dan bandingkan
dengan tokenizer lokal google/gemma-3-4b-pt (tokenizernya-gemma3).
"""
import json
import os
from huggingface_hub import hf_hub_download

LOCAL_PT = r"d:\Codings\unsloth\t5-gemma-2\instruct\data\tokenizernya-gemma3\tokenizer.json"
IT_REPO = "google/gemma-3-4b-it"

print("Downloading tokenizer.json dari google/gemma-3-4b-it ...")
it_path = hf_hub_download(repo_id=IT_REPO, filename="tokenizer.json")
print(f"  Downloaded to: {it_path}")

print("\nLoading kedua tokenizer...")
with open(LOCAL_PT, "r", encoding="utf-8") as f:
    pt_data = json.load(f)
with open(it_path, "r", encoding="utf-8") as f:
    it_data = json.load(f)

# --- Compare top-level keys ---
print("\n=== Top-Level Keys ===")
pt_keys = set(pt_data.keys())
it_keys = set(it_data.keys())
print(f"  PT keys: {sorted(pt_keys)}")
print(f"  IT keys: {sorted(it_keys)}")
if pt_keys != it_keys:
    print(f"  DIFF: Only in PT: {pt_keys - it_keys}, Only in IT: {it_keys - pt_keys}")

# --- Compare added_tokens ---
print("\n=== Added Tokens ===")
pt_added = pt_data.get("added_tokens", [])
it_added = it_data.get("added_tokens", [])
print(f"  PT: {len(pt_added)} tokens")
print(f"  IT: {len(it_added)} tokens")

pt_added_map = {t["content"]: t for t in pt_added}
it_added_map = {t["content"]: t for t in it_added}

only_pt = set(pt_added_map.keys()) - set(it_added_map.keys())
only_it = set(it_added_map.keys()) - set(pt_added_map.keys())

if only_pt:
    print(f"\n  Hanya di PT ({len(only_pt)}):")
    for name in sorted(only_pt)[:20]:
        print(f"    id {pt_added_map[name]['id']}: {name}")
    if len(only_pt) > 20:
        print(f"    ... dan {len(only_pt)-20} lainnya")

if only_it:
    print(f"\n  Hanya di IT ({len(only_it)}):")
    for name in sorted(only_it)[:20]:
        print(f"    id {it_added_map[name]['id']}: {name}")
    if len(only_it) > 20:
        print(f"    ... dan {len(only_it)-20} lainnya")

# Check ID differences for shared tokens
shared = set(pt_added_map.keys()) & set(it_added_map.keys())
id_diffs = []
attr_diffs = []
for name in sorted(shared):
    pt_t = pt_added_map[name]
    it_t = it_added_map[name]
    if pt_t["id"] != it_t["id"]:
        id_diffs.append((name, pt_t["id"], it_t["id"]))
    # Compare other attributes
    for key in set(list(pt_t.keys()) + list(it_t.keys())):
        if key == "id":
            continue
        if pt_t.get(key) != it_t.get(key):
            attr_diffs.append((name, key, pt_t.get(key), it_t.get(key)))

if id_diffs:
    print(f"\n  Token dengan ID berbeda ({len(id_diffs)}):")
    for name, pt_id, it_id in id_diffs[:20]:
        print(f"    {name}: PT={pt_id}, IT={it_id}")
else:
    print(f"\n  ✅ Semua {len(shared)} shared token memiliki ID identik.")

if attr_diffs:
    print(f"\n  Token dengan atribut berbeda ({len(attr_diffs)}):")
    for name, key, pt_val, it_val in attr_diffs[:20]:
        print(f"    {name}.{key}: PT={pt_val}, IT={it_val}")
else:
    print(f"  ✅ Semua atribut shared token identik.")

# --- Compare BPE vocab ---
print("\n=== BPE Vocab ===")
pt_vocab = pt_data.get("model", {}).get("vocab", {})
it_vocab = it_data.get("model", {}).get("vocab", {})
print(f"  PT vocab size: {len(pt_vocab)}")
print(f"  IT vocab size: {len(it_vocab)}")

pt_vocab_set = set(pt_vocab.keys())
it_vocab_set = set(it_vocab.keys())
v_only_pt = pt_vocab_set - it_vocab_set
v_only_it = it_vocab_set - pt_vocab_set

if v_only_pt or v_only_it:
    print(f"  Hanya di PT: {len(v_only_pt)}")
    print(f"  Hanya di IT: {len(v_only_it)}")
    if v_only_pt:
        print(f"    PT only: {sorted(v_only_pt)[:10]}")
    if v_only_it:
        print(f"    IT only: {sorted(v_only_it)[:10]}")
else:
    print(f"  ✅ BPE vocab identik.")

# Check for value (id) differences
val_diffs = []
for token in pt_vocab_set & it_vocab_set:
    if pt_vocab[token] != it_vocab[token]:
        val_diffs.append((token, pt_vocab[token], it_vocab[token]))
if val_diffs:
    print(f"  Token dengan ID berbeda di vocab ({len(val_diffs)}):")
    for tok, pt_id, it_id in val_diffs[:10]:
        print(f"    '{tok}': PT={pt_id}, IT={it_id}")
else:
    print(f"  ✅ Semua vocab token IDs identik.")

# --- Compare merges ---
print("\n=== Merges ===")
pt_merges = pt_data.get("model", {}).get("merges", [])
it_merges = it_data.get("model", {}).get("merges", [])
print(f"  PT merges: {len(pt_merges)}")
print(f"  IT merges: {len(it_merges)}")
if pt_merges == it_merges:
    print(f"  ✅ Merges identik.")
else:
    print(f"  ❌ Merges BERBEDA!")

# --- Final verdict ---
print("\n" + "="*60)
if not only_pt and not only_it and not id_diffs and not attr_diffs \
   and not v_only_pt and not v_only_it and not val_diffs \
   and pt_merges == it_merges:
    print("  🎉 KESIMPULAN: Tokenizer gemma-3-4b-pt dan gemma-3-4b-it")
    print("     100% IDENTIK!")
else:
    print("  ⚠️  KESIMPULAN: Terdapat perbedaan antara PT dan IT tokenizer.")
print("="*60)
