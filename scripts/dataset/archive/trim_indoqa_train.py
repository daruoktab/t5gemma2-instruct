"""
Ambil subset acak (deterministik) dari IndoQA train JSONL.

- Menyalin seluruh isi input ke file backup **sekali** jika backup belum ada
  dan jumlah baris input > `--n` (supaya full ~3.3k tidak hilang).
- Menulis `--n` baris ke `--output` (default menimpa `indoqa_train.jsonl`).

Jalankan dari folder instruct:
  python scripts/trim_indoqa_train.py --n 2500
"""

from __future__ import annotations

import argparse
import random
import sys
from pathlib import Path


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    p = argparse.ArgumentParser(description="Trim IndoQA JSONL to N lines with optional full backup")
    p.add_argument("--input", type=Path, default=root / "indoqa_train.jsonl")
    p.add_argument("--output", type=Path, default=root / "indoqa_train.jsonl")
    p.add_argument("--backup", type=Path, default=root / "indoqa_train_full.jsonl")
    p.add_argument("--n", type=int, default=2500)
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    if not args.input.is_file():
        print(f"[ERR] tidak ada: {args.input}", file=sys.stderr)
        sys.exit(1)

    lines: list[str] = []
    with args.input.open("r", encoding="utf-8") as f:
        for line in f:
            s = line.strip()
            if s:
                lines.append(s)

    if len(lines) <= args.n:
        print(f"[WARN] hanya {len(lines)} baris ≤ {args.n}, salin semua ke output")
        sample = lines
    else:
        rng = random.Random(args.seed)
        rng.shuffle(lines)
        sample = lines[: args.n]

    if args.backup and len(lines) > args.n:
        if not args.backup.is_file():
            args.backup.parent.mkdir(parents=True, exist_ok=True)
            with args.backup.open("w", encoding="utf-8") as fb:
                for s in lines:
                    fb.write(s + "\n")
            print(f"[OK] backup full {len(lines)} baris → {args.backup}")
        else:
            print(f"[INFO] backup sudah ada, tidak menimpa: {args.backup}")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as fo:
        for s in sample:
            fo.write(s + "\n")

    print(f"[OK] wrote {len(sample)} baris → {args.output}")


if __name__ == "__main__":
    main()
