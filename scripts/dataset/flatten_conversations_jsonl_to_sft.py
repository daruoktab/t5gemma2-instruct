"""
Flatten baris JSONL dengan struktur seperti `t5-gemma-2-chat-instruct-dataset.jsonl`
atau `t5-gemma-2-chat-instruct-dataset-extra8970.jsonl` (`{"conversations": [...]}`)
menjadi format trainer (`input` / `target`, turn unrolling per giliran assistant).

Contoh:
  python flatten_conversations_jsonl_to_sft.py \\
    --input t5-gemma-2-chat-instruct-dataset-extra8970.jsonl \\
    --output flattened_extra_sft.jsonl

Alur gabung base+extra lalu `chat_train`/`chat_val` (split per thread):
  python scripts/merge_nested_conversations_jsonl.py
  python scripts/rebuild_chat_sft_from_nested.py
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

# Ensure current directory and root are in sys.path for local/absolute imports
_P = Path(__file__).resolve()
SCRIPT_DIR = str(_P.parent)
ROOT_DIR = str(_P.parents[2])

if SCRIPT_DIR not in sys.path:
    sys.path.insert(0, SCRIPT_DIR)
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)


def default_system_prompt() -> str:
    try:
        from scripts.dataset.generate_dataset_deepseek import SYSTEM_PROMPT

        return SYSTEM_PROMPT.strip()
    except Exception:
        return (
            "Kamu adalah asisten AI yang helpful, santai, dan ramah. "
            "Gunakan Bahasa Indonesia sebagai bahasa utama."
        )


def parse_system_and_turns(
    messages: list[dict],
    fallback_system: str,
) -> tuple[str, list[tuple[str, str]]]:
    """Ekstrak system string dan list (user, assistant) lengkap per giliran."""
    system = fallback_system.strip()
    turns: list[tuple[str, str]] = []
    pending_user: str | None = None

    for m in messages:
        role = (m.get("role") or "").strip()
        content = (m.get("content") or "").strip()
        if not content:
            continue
        if role == "system":
            system = content
        elif role == "user":
            pending_user = content
        elif role == "assistant":
            if pending_user is None:
                continue
            turns.append((pending_user, content))
            pending_user = None

    return system, turns


def conversations_to_sft_rows(
    messages: list[dict],
    fallback_system: str,
    chat_idx: int | None = None,
) -> list[dict]:
    system, turns = parse_system_and_turns(messages, fallback_system)
    rows: list[dict] = []
    for k in range(len(turns)):
        parts: list[str] = [f"system: {system}"]
        for i in range(k):
            parts.append(f"user: {turns[i][0]}")
            parts.append(f"assistant: {turns[i][1]}")
        parts.append(f"user: {turns[k][0]}")
        inp = "\n".join(parts).strip()
        
        row_data: dict[str, Any] = {
            "input": inp,
            "target": turns[k][1],
        }
        if chat_idx is not None:
            row_data["chat_idx"] = chat_idx
        row_data["turn_idx"] = k
        rows.append(row_data)
    return rows


def main() -> None:
    p = argparse.ArgumentParser(description="Flatten conversations JSONL → input/target SFT JSONL")
    p.add_argument("--input", type=Path, required=True, help="JSONL dengan kunci conversations")
    p.add_argument("--output", type=Path, required=True, help="JSONL output input/target")
    p.add_argument(
        "--system-text",
        type=str,
        default="",
        help="Fallback system jika tidak ada di percakapan (default: SYSTEM_PROMPT dari generate_dataset_deepseek)",
    )
    args = p.parse_args()
    fallback = args.system_text.strip() or default_system_prompt()

    n_in = 0
    n_out = 0
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.input.open("r", encoding="utf-8") as fin, args.output.open("w", encoding="utf-8") as fout:
        for line in fin:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError as e:
                print(f"[SKIP] JSON error: {e}", file=sys.stderr)
                continue
            convs = obj.get("conversations")
            if not isinstance(convs, list):
                print("[SKIP] baris tanpa conversations[]", file=sys.stderr)
                continue
            n_in += 1
            chat_idx = obj.get("id")
            for row in conversations_to_sft_rows(convs, fallback, chat_idx):
                fout.write(json.dumps(row, ensure_ascii=False) + "\n")
                n_out += 1

    print(f"[OK] {n_in} baris conversations → {n_out} baris SFT → {args.output}")


if __name__ == "__main__":
    main()
