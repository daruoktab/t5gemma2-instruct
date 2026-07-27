"""Analisis kualitas dataset train_vision.jsonl — cari masalah unused token di role user."""
import json

with open("data/multimodal/train_vision.jsonl", "r", encoding="utf-8") as f:
    records = [json.loads(l.strip()) for l in f]

print(f"Total records: {len(records)}")

user_unused = []
user_weird_prefix = []
assistant_no_prefix = []
total_user_msgs = 0
total_asst_msgs = 0

for rec in records:
    for m in rec.get("messages", []):
        if m["role"] == "user":
            total_user_msgs += 1
            content = m["content"]
            if "<unused" in content:
                user_unused.append((rec.get("id"), content[:120]))
            # Cek prefix aneh (bukan emoji, bukan huruf)
            first_chars = content[:10]
            if first_chars.startswith("\uf400"):  # boi emoji
                rest = content[1:].lstrip()
                if rest.startswith("<unused"):
                    user_weird_prefix.append((rec.get("id"), repr(content[:40])))
            elif "<unused" in first_chars:
                user_weird_prefix.append((rec.get("id"), repr(content[:40])))
        elif m["role"] == "assistant":
            total_asst_msgs += 1
            content = m["content"]
            if not content.startswith("<unused"):
                assistant_no_prefix.append((rec.get("id"), content[:100]))

print(f"\nTotal user messages: {total_user_msgs}")
print(f"Total assistant messages: {total_asst_msgs}")

print(f"\n=== User dengan <unused> token: {len(user_unused)} ===")
for uid, c in user_unused[:10]:
    print(f"  ID {uid}: {c}")

print(f"\n=== User dengan weird prefix (unused setelah boi): {len(user_weird_prefix)} ===")
for uid, c in user_weird_prefix[:10]:
    print(f"  ID {uid}: {c}")

print(f"\n=== Assistant TANPA prefix <unused>: {len(assistant_no_prefix)} ===")
for uid, c in assistant_no_prefix[:10]:
    print(f"  ID {uid}: {c}")

# Statistik images
img_counts = [len(r.get("images", [])) for r in records]
print(f"\n=== Image stats ===")
print(f"  Min: {min(img_counts)}, Max: {max(img_counts)}, Avg: {sum(img_counts)/len(img_counts):.1f}")
print(f"  >10 images: {sum(1 for c in img_counts if c > 10)}")
print(f"  >4 images: {sum(1 for c in img_counts if c > 4)}")
