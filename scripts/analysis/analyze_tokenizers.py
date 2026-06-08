"""
Analisis perbandingan tiga tokenizer: Gemma 3, Gemma 4, dan T5Gemma2.
"""
import json
import os
from collections import defaultdict

BASE = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.normpath(os.path.join(BASE, "..", "..", "data"))

TOKENIZERS = {
    "Gemma 3":   os.path.join(DATA_DIR, "tokenizernya-gemma3",  "tokenizer.json"),
    "Gemma 4":   os.path.join(DATA_DIR, "tokenizernya-gemma4",  "tokenizer.json"),
    "T5Gemma2":  os.path.join(DATA_DIR, "tokenizernya-t5gemma2","tokenizer.json"),
}

def load_tokenizer(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def analyze(name, data):
    print(f"\n{'='*70}")
    print(f"  TOKENIZER: {name}")
    print(f"{'='*70}")

    # --- Added tokens ---
    added = data.get("added_tokens", [])
    print(f"\n[Added Tokens]")
    print(f"  Total added_tokens: {len(added)}")

    unused = [t for t in added if t["content"].startswith("<unused")]
    non_unused = [t for t in added if not t["content"].startswith("<unused")]
    print(f"  Unused tokens: {len(unused)}")
    print(f"  Non-unused added tokens: {len(non_unused)}")

    # Unused token ID ranges
    if unused:
        unused_ids = sorted([t["id"] for t in unused])
        # Detect contiguous blocks
        blocks = []
        block_start = unused_ids[0]
        prev = unused_ids[0]
        for uid in unused_ids[1:]:
            if uid != prev + 1:
                blocks.append((block_start, prev))
                block_start = uid
            prev = uid
        blocks.append((block_start, prev))

        print(f"  Unused token ID blocks ({len(blocks)} blocks):")
        for i, (s, e) in enumerate(blocks):
            count = e - s + 1
            # Get names
            start_name = next(t["content"] for t in unused if t["id"] == s)
            end_name = next(t["content"] for t in unused if t["id"] == e)
            print(f"    Block {i+1}: id {s}–{e}  ({count} tokens)  [{start_name} → {end_name}]")

    # --- Non-unused special tokens ---
    print(f"\n[Non-Unused Added Tokens] ({len(non_unused)} total):")
    for t in sorted(non_unused, key=lambda x: x["id"]):
        flags = []
        if t.get("special"): flags.append("special")
        if t.get("rstrip"):  flags.append("rstrip")
        if t.get("lstrip"):  flags.append("lstrip")
        print(f"    id {t['id']:>7}: {t['content']:<35} {', '.join(flags)}")

    # --- Vocab from model (merges/vocab) ---
    model_section = data.get("model", {})
    vocab = model_section.get("vocab", {})
    merges = model_section.get("merges", [])
    print(f"\n[Model / BPE Info]")
    print(f"  Vocab size (in model.vocab): {len(vocab)}")
    print(f"  Merges count: {len(merges)}")
    print(f"  Model type: {model_section.get('type', 'N/A')}")

    # --- Special token IDs from config ---
    # Check for vision-related tokens
    vision_tokens = [t for t in added if any(k in t["content"] for k in ["image", "vision", "v_patch", "patch"])]
    if vision_tokens:
        print(f"\n[Vision-Related Tokens] ({len(vision_tokens)} total):")
        for t in sorted(vision_tokens, key=lambda x: x["id"]):
            print(f"    id {t['id']:>7}: {t['content']}")

    # Turn tokens
    turn_tokens = [t for t in added if "turn" in t["content"]]
    if turn_tokens:
        print(f"\n[Turn Tokens] ({len(turn_tokens)} total):")
        for t in sorted(turn_tokens, key=lambda x: x["id"]):
            print(f"    id {t['id']:>7}: {t['content']}")

    return {
        "added": added,
        "unused": unused,
        "non_unused": non_unused,
        "vocab_size": len(vocab),
        "merges": len(merges),
    }


def compare_all(results):
    print(f"\n\n{'='*70}")
    print(f"  PERBANDINGAN ANTAR TOKENIZER")
    print(f"{'='*70}")

    names = list(results.keys())

    # Summary table
    print(f"\n{'Metrik':<35}", end="")
    for n in names:
        print(f"{n:>15}", end="")
    print()
    print("-" * (35 + 15 * len(names)))

    metrics = [
        ("Total added_tokens", lambda r: len(r["added"])),
        ("Unused tokens", lambda r: len(r["unused"])),
        ("Non-unused added tokens", lambda r: len(r["non_unused"])),
        ("Vocab size (model.vocab)", lambda r: r["vocab_size"]),
        ("Merges count", lambda r: r["merges"]),
    ]
    for label, fn in metrics:
        print(f"{label:<35}", end="")
        for n in names:
            print(f"{fn(results[n]):>15,}", end="")
        print()

    # --- Diff: Non-unused added tokens ---
    print(f"\n[Perbedaan Non-Unused Added Tokens]")
    all_non_unused = {}
    for n in names:
        for t in results[n]["non_unused"]:
            key = t["content"]
            if key not in all_non_unused:
                all_non_unused[key] = {}
            all_non_unused[key][n] = t["id"]

    # Find tokens that differ
    diff_tokens = []
    for content, mapping in sorted(all_non_unused.items()):
        if len(mapping) < len(names):
            diff_tokens.append((content, mapping, "MISSING"))
        else:
            ids = list(mapping.values())
            if len(set(ids)) > 1:
                diff_tokens.append((content, mapping, "ID_DIFFERS"))

    if diff_tokens:
        for content, mapping, reason in diff_tokens:
            print(f"\n  Token: {content}")
            print(f"  Reason: {reason}")
            for n in names:
                if n in mapping:
                    print(f"    {n:>12}: id {mapping[n]}")
                else:
                    print(f"    {n:>12}: TIDAK ADA")
    else:
        print("  Semua non-unused added tokens identik!")

    # --- Diff: Unused tokens ---
    print(f"\n[Perbedaan Unused Tokens]")
    for n in names:
        unused_contents = set(t["content"] for t in results[n]["unused"])
        for n2 in names:
            if n2 <= n:
                continue
            unused_contents2 = set(t["content"] for t in results[n2]["unused"])
            only_in_1 = unused_contents - unused_contents2
            only_in_2 = unused_contents2 - unused_contents
            if only_in_1 or only_in_2:
                print(f"\n  {n} vs {n2}:")
                if only_in_1:
                    print(f"    Hanya di {n} ({len(only_in_1)}):", sorted(only_in_1)[:10], "..." if len(only_in_1) > 10 else "")
                if only_in_2:
                    print(f"    Hanya di {n2} ({len(only_in_2)}):", sorted(only_in_2)[:10], "..." if len(only_in_2) > 10 else "")
            else:
                print(f"\n  {n} vs {n2}: Unused tokens identik.")

    # --- Vocab diff (BPE) ---
    print(f"\n[Perbedaan BPE Vocab]")
    vocabs = {}
    for n in names:
        with open(TOKENIZERS[n], "r", encoding="utf-8") as f:
            d = json.load(f)
        vocabs[n] = set(d.get("model", {}).get("vocab", {}).keys())

    for n in names:
        for n2 in names:
            if n2 <= n:
                continue
            only1 = vocabs[n] - vocabs[n2]
            only2 = vocabs[n2] - vocabs[n]
            if only1 or only2:
                print(f"\n  {n} vs {n2}:")
                if only1:
                    samples = sorted(only1)[:20]
                    print(f"    Hanya di {n} ({len(only1)}): {samples}{'...' if len(only1) > 20 else ''}")
                if only2:
                    samples = sorted(only2)[:20]
                    print(f"    Hanya di {n2} ({len(only2)}): {samples}{'...' if len(only2) > 20 else ''}")
            else:
                print(f"\n  {n} vs {n2}: BPE vocab identik.")

    # --- Merges diff ---
    print(f"\n[Perbedaan Merges]")
    merge_lists = {}
    for n in names:
        with open(TOKENIZERS[n], "r", encoding="utf-8") as f:
            d = json.load(f)
        merge_lists[n] = d.get("model", {}).get("merges", [])

    for n in names:
        for n2 in names:
            if n2 <= n:
                continue
            m1 = merge_lists[n]
            m2 = merge_lists[n2]
            if m1 == m2:
                print(f"  {n} vs {n2}: Merges identik.")
            else:
                print(f"  {n} vs {n2}: Merges BERBEDA! ({len(m1)} vs {len(m2)})")
                # Show first few differences
                diff_count = 0
                for i, (a, b) in enumerate(zip(m1, m2)):
                    if a != b:
                        if diff_count < 5:
                            print(f"    Merge #{i}: {n}='{a}' vs {n2}='{b}'")
                        diff_count += 1
                if len(m1) != len(m2):
                    print(f"    Length differs: {len(m1)} vs {len(m2)}")
                if diff_count > 5:
                    print(f"    ... dan {diff_count - 5} perbedaan lainnya")


def main():
    results = {}
    for name, path in TOKENIZERS.items():
        if not os.path.exists(path):
            print(f"SKIP: {path} tidak ditemukan.")
            continue
        data = load_tokenizer(path)
        results[name] = analyze(name, data)

    if len(results) > 1:
        compare_all(results)

    print("\n\nSelesai.")


if __name__ == "__main__":
    main()
