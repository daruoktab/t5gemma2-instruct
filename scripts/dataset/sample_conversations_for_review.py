"""Sample N random full conversations from nested JSONL (base + extra) for manual review."""
from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parents[1]


def load_conversations(path: Path) -> list[dict]:
    out: list[dict] = []
    if not path.exists():
        print(f"[WARN] missing: {path}", file=sys.stderr)
        return out
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            conv = row.get("conversations")
            if isinstance(conv, list) and len(conv) >= 3:
                out.append(
                    {
                        "source": path.name,
                        "id": row.get("id"),
                        "topik": row.get("topik", ""),
                        "num_turns": len(conv),
                        "conversations": conv,
                    }
                )
    return out


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("-n", type=int, default=20)
    p.add_argument(
        "--base",
        type=Path,
        default=SCRIPT_DIR / "t5-gemma-2-chat-instruct-dataset.jsonl",
    )
    p.add_argument(
        "--extra",
        type=Path,
        default=SCRIPT_DIR / "t5-gemma-2-chat-instruct-dataset-extra8970.jsonl",
    )
    p.add_argument("--out", type=Path, default=SCRIPT_DIR / "_sample_review_20.json")
    args = p.parse_args()

    pool = load_conversations(args.base) + load_conversations(args.extra)
    if len(pool) < args.n:
        print(f"[ERR] only {len(pool)} conversations", file=sys.stderr)
        sys.exit(1)
    rng = random.Random(args.seed)
    sample = rng.sample(pool, args.n)
    args.out.write_text(
        json.dumps(sample, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(f"[OK] wrote {args.n} conversations → {args.out} (pool={len(pool)})")


if __name__ == "__main__":
    main()
