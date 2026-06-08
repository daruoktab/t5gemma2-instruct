"""
Gabungkan beberapa JSONL nested (`{"conversations": [...]}`) menjadi satu file.

Urutan baris = urutan file di `--inputs` (base dulu, extra belakang, dll.).
Opsional dedupe by `id` jika ada di objek JSON.

Contoh (dari folder instruct):
  python scripts/merge_nested_conversations_jsonl.py
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    p = argparse.ArgumentParser(description="Merge nested conversation JSONL files")
    p.add_argument(
        "--inputs",
        type=Path,
        nargs="+",
        default=[
            root / "t5-gemma-2-chat-instruct-dataset.jsonl",
            root / "t5-gemma-2-chat-instruct-dataset-extra8970.jsonl",
        ],
        help="Satu atau lebih file JSONL nested",
    )
    p.add_argument(
        "--output",
        type=Path,
        default=root / "t5-gemma-2-chat-instruct-merged.jsonl",
        help="File merged output",
    )
    p.add_argument(
        "--dedupe-id",
        action="store_true",
        help="Lewati baris jika `id` sudah pernah muncul (pertahankan yang pertama)",
    )
    args = p.parse_args()

    seen_ids: set[int] = set()
    n_written = 0
    n_skipped = 0

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as fout:
        for in_path in args.inputs:
            if not in_path.is_file():
                print(f"[WARN] skip (tidak ada): {in_path}", file=sys.stderr)
                continue
            with in_path.open("r", encoding="utf-8") as fin:
                for line in fin:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        obj = json.loads(line)
                    except json.JSONDecodeError as e:
                        print(f"[SKIP] {in_path}: JSON {e}", file=sys.stderr)
                        n_skipped += 1
                        continue
                    if not isinstance(obj.get("conversations"), list):
                        print(f"[SKIP] {in_path}: tanpa conversations[]", file=sys.stderr)
                        n_skipped += 1
                        continue
                    if args.dedupe_id and "id" in obj:
                        oid = obj["id"]
                        if isinstance(oid, int):
                            if oid in seen_ids:
                                n_skipped += 1
                                continue
                            seen_ids.add(oid)
                    fout.write(json.dumps(obj, ensure_ascii=False) + "\n")
                    n_written += 1

    print(f"[OK] merged {n_written} baris → {args.output} (skipped {n_skipped})")


if __name__ == "__main__":
    main()
