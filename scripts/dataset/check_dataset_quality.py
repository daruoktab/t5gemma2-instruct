"""
Cek kualitas dataset per source.
Jalankan: conda activate unsloth && python check_dataset_quality.py
"""
import json
import random
from collections import Counter

DATA = "./data/final_dataset.json"

print("Loading dataset...")
with open(DATA, "r", encoding="utf-8") as f:
    data = json.load(f)

print(f"Total: {len(data)} samples\n")

# Group by source
by_source = {}
for row in data:
    src = row.get("source", "unknown")
    if src not in by_source:
        by_source[src] = []
    by_source[src].append(row)

# Check each source
issues = []

for src in sorted(by_source.keys()):
    rows = by_source[src]
    n = len(rows)

    print("=" * 70)
    print(f"SOURCE: {src} ({n} samples)")
    print("=" * 70)

    # Basic stats
    empty_inputs = sum(1 for r in rows if not r.get("inputs", "").strip())
    empty_targets = sum(1 for r in rows if not r.get("targets", "").strip())
    short_targets = sum(1 for r in rows if len(r.get("targets", "").strip()) < 5)
    very_long_inputs = sum(1 for r in rows if len(r.get("inputs", "")) > 2000)
    very_long_targets = sum(1 for r in rows if len(r.get("targets", "")) > 2000)

    avg_input_len = sum(len(r.get("inputs", "")) for r in rows) / n
    avg_target_len = sum(len(r.get("targets", "")) for r in rows) / n

    # Language distribution
    langs = Counter(r.get("language", "?") for r in rows)

    # Has required columns?
    sample = rows[0]
    has_inputs = "inputs" in sample
    has_targets = "targets" in sample

    print(f"  Columns present: inputs={has_inputs}, targets={has_targets}")
    print(f"  Languages: {dict(langs)}")
    print(f"  Avg input length:  {avg_input_len:.0f} chars")
    print(f"  Avg target length: {avg_target_len:.0f} chars")
    print(f"  Empty inputs:  {empty_inputs}/{n} ({empty_inputs/n*100:.1f}%)")
    print(f"  Empty targets: {empty_targets}/{n} ({empty_targets/n*100:.1f}%)")
    print(f"  Short targets (<5 chars): {short_targets}/{n} ({short_targets/n*100:.1f}%)")
    print(f"  Very long inputs  (>2000): {very_long_inputs}")
    print(f"  Very long targets (>2000): {very_long_targets}")

    # Flag issues
    if empty_inputs > 0:
        issues.append(f"  ⚠️  {src}: {empty_inputs} EMPTY inputs!")
    if empty_targets > 0:
        issues.append(f"  ⚠️  {src}: {empty_targets} EMPTY targets!")
    if short_targets > n * 0.3:
        issues.append(f"  ⚠️  {src}: {short_targets} very short targets (>30%)")

    # Show 3 random samples
    print("\n  --- Random Samples ---")
    samples = random.Random(42).sample(rows, min(3, n))
    for i, s in enumerate(samples, 1):
        inp = s.get("inputs", "")[:200]
        tgt = s.get("targets", "")[:200]
        lang = s.get("language", "?")
        print(f"\n  [{i}] lang={lang}")
        print(f"  INPUT:  {repr(inp)}")
        print(f"  TARGET: {repr(tgt)}")

    # Check for target == input (copying issue)
    same_count = sum(1 for r in rows
                     if r.get("inputs", "").strip() == r.get("targets", "").strip()
                     and len(r.get("inputs", "").strip()) > 0)
    if same_count > 0:
        print(f"\n  ⚠️  {same_count} samples where input == target (copying/echo)")
        issues.append(f"  ⚠️  {src}: {same_count} samples input==target")

    # Check for HTML tags in output
    html_count = sum(1 for r in rows if "<" in r.get("targets", "") and ">" in r.get("targets", ""))
    if html_count > n * 0.1:
        print(f"\n  ℹ️  {html_count} targets contain HTML-like tags")

    print()


# ========================
# SUMMARY
# ========================
print("\n" + "=" * 70)
print("QUALITY SUMMARY")
print("=" * 70)

if issues:
    print("\n⚠️  Issues Found:")
    for iss in issues:
        print(iss)
else:
    print("\n✅ No major issues found!")

# Overall stats
total_empty_in = sum(1 for r in data if not r.get("inputs", "").strip())
total_empty_tgt = sum(1 for r in data if not r.get("targets", "").strip())
total_same = sum(1 for r in data
                 if r.get("inputs", "").strip() == r.get("targets", "").strip()
                 and len(r.get("inputs", "").strip()) > 0)

print("\nOverall:")
print(f"  Total samples:  {len(data)}")
print(f"  Empty inputs:   {total_empty_in}")
print(f"  Empty targets:  {total_empty_tgt}")
print(f"  Input==Target:  {total_same}")
print(f"  Usable:         {len(data) - total_empty_in - total_empty_tgt}")
