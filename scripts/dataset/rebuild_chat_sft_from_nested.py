"""
Dari satu JSONL nested (biasanya hasil merge), bangun ulang `chat_train.jsonl` + `chat_val.jsonl`
dengan split thread (bukan split baris SFT), lalu flatten per turn seperti
`flatten_conversations_jsonl_to_sft.py`.

Jalankan dari folder instruct:
  python scripts/rebuild_chat_sft_from_nested.py
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

# Import dari folder instruct (parent)
ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# SCRIPT_DIR is needed for local imports if ROOT fails, but ROOT should be enough now
SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from flatten_conversations_jsonl_to_sft import (  # noqa: E402  # ty:ignore[unresolved-import]
    conversations_to_sft_rows,
    default_system_prompt,
)


def flatten_objects(
    objects: list[dict],
    fout,
    fallback: str,
) -> int:
    n = 0
    for obj in objects:
        chat_idx = obj.get("id")
        convs = obj["conversations"]
        for row in conversations_to_sft_rows(convs, fallback, chat_idx):
            fout.write(json.dumps(row, ensure_ascii=False) + "\n")
            n += 1
    return n


def main() -> None:
    p = argparse.ArgumentParser(description="Split nested threads → chat_train / chat_val SFT JSONL")
    p.add_argument(
        "--nested",
        type=Path,
        default=ROOT / "data" / "t5-gemma-2-chat-instruct-dataset.jsonl",
        help="JSONL nested dataset",
    )
    p.add_argument("--train-out", type=Path, default=ROOT / "data" / "chat_train.jsonl")
    p.add_argument("--val-out", type=Path, default=ROOT / "data" / "chat_val.jsonl")
    p.add_argument(
        "--val-threads",
        type=int,
        default=40,
        help="Jumlah thread (bukan baris SFT) untuk validation",
    )
    p.add_argument("--seed", type=int, default=42)
    p.add_argument(
        "--system-text",
        type=str,
        default="",
        help="Fallback system (kosong = pakai default dari generate_dataset_deepseek)",
    )
    args = p.parse_args()
    fallback = args.system_text.strip() or default_system_prompt()

    if not args.nested.is_file():
        print(f"[ERR] tidak ada: {args.nested}", file=sys.stderr)
        sys.exit(1)

    threads: list[dict] = []
    with args.nested.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError as e:
                print(f"[SKIP] JSON: {e}", file=sys.stderr)
                continue
            if isinstance(obj.get("conversations"), list):
                threads.append(obj)

    n_all = len(threads)
    if n_all < 2:
        print("[ERR] terlalu sedikit thread", file=sys.stderr)
        sys.exit(1)

    rng = random.Random(args.seed)
    rng.shuffle(threads)
    n_val = min(max(1, args.val_threads), n_all - 1)
    val_objs = threads[:n_val]
    train_objs = threads[n_val:]

    args.train_out.parent.mkdir(parents=True, exist_ok=True)
    with args.train_out.open("w", encoding="utf-8") as f_train:
        n_train_rows = flatten_objects(train_objs, f_train, fallback)
    with args.val_out.open("w", encoding="utf-8") as f_val:
        n_val_rows = flatten_objects(val_objs, f_val, fallback)

    print(
        f"[OK] threads total={n_all} train_threads={len(train_objs)} val_threads={len(val_objs)} "
        f"→ {args.train_out} ({n_train_rows} baris), {args.val_out} ({n_val_rows} baris)"
    )


if __name__ == "__main__":
    main()
