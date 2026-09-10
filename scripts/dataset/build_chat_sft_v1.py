#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
BUILD chat_sft v1 : gabung chat lama (3000 percakapan) + teks baru (2000 percakapan)
==================================================================================
- Pertahankan `chat_idx` lama apa adanya (biar ORPO lama masih bisa trace ke percakapan).
- Percakapan BARU diberi `chat_idx` baru dengan offset 100000 (tidak tabrakan).
- SHUFFLE berdasarkan PERCAKAPAN (bukan row), lalu split 80:20 (4000:1000).
- Tulis parquet train-* + validation-* -> data_repo/.../sft/text/chat_hf/ + PUSH ke HF config chat_sft.
- Simpan chat_idx_map.json (percakapan lama/baku + split) untuk trace ORPO.
"""
from __future__ import annotations

import gc
import json
import random
from pathlib import Path

from datasets import Dataset, Features, Value, load_dataset
from huggingface_hub import HfApi

ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT / "data"
REPO_DIR = ROOT / "data_repo" / "t5gemma2-indonesia-instruct-v1"
CHAT_REPO = "daruokta/t5gemma2-indonesia-chat-formatted"
SYNTH = DATA_DIR / "synthetic" / "generated_conv_agent.jsonl"
REPO_ID = "daruokta/t5gemma2-indonesia-instruct-v1"
SUBFOLDER = "chat/text"
OUT_DIR = REPO_DIR / SUBFOLDER
SEED = 3407
NEW_OFFSET = 100_000
TRAIN_FRACTION = 0.8

FEATURES = Features({
    "input": Value("string"),
    "target": Value("string"),
    "chat_idx": Value("int64"),
    "turn_idx": Value("int64"),
    "input_tokens": Value("int64"),
    "target_tokens": Value("int64"),
    "source": Value("string"),
})


def _tokenizer():
    try:
        from transformers import AutoTokenizer
        return AutoTokenizer.from_pretrained("google/t5gemma-2-4b-4b")
    except Exception:
        return None


class _T:
    def __init__(self, tok):
        self.tok = tok

    def enc(self, s: str) -> int:
        try:
            if self.tok is None:
                return 0
            return len(self.tok.encode(s, add_special_tokens=False))
        except Exception:
            return 0


def _render_input(messages, upto: int) -> str:
    """Render role: content untuk pesan sebelum index `upto`."""
    lines = []
    for m in messages[:upto]:
        role = m.get("role", "user")
        lines.append(f"{role}: {m.get('content', '')}")
    return "\n".join(lines)


def load_old_chat():
    """HF chat_sft train+val -> dict groups by chat_idx. Jaga chat_idx lama."""
    rows = []
    for split in ("train", "validation"):
        for r in load_dataset(CHAT_REPO, "chat_sft", split=split):
            rows.append(r)
    groups = {}
    for r in rows:
        groups.setdefault(int(r["chat_idx"]), []).append({
            "input": r["input"], "target": r["target"],
            "chat_idx": int(r["chat_idx"]), "turn_idx": int(r["turn_idx"]),
            "input_tokens": int(r.get("input_tokens") or 0),
            "target_tokens": int(r.get("target_tokens") or 0),
            "source": "hf:chat_sft",
        })
    print(f"  [LAMA] percakapan={len(groups)} rows={sum(len(v) for v in groups.values())}")
    return groups


def load_new_text(tok):
    """generated_conv_agent teks -> unroll per turn -> rows.
    catatan: field `id` TIDAK unik (34 id diulang utk conv beda) -> pakai counter untuk
    chat_idx unik; simpan `id` asli di map (chat_idx_map.json) utk trace ORPO."""
    groups = {}
    counter = 0
    for line in open(SYNTH, encoding="utf-8"):
        line = line.strip()
        if not line:
            continue
        try:
            o = json.loads(line)
        except Exception:
            continue
        if o.get("category") == "vision_chat":
            continue
        cidx = NEW_OFFSET + counter
        counter += 1
        msgs = o.get("messages", [])
        src = o.get("source", "")
        orig_id = o.get("id", "")
        turns = []
        for i, m in enumerate(msgs):
            if m.get("role") == "assistant":
                ctx = _render_input(msgs, i)
                tgt = m.get("content", "")
                if not tgt.strip() or not ctx.strip():
                    continue
                turns.append({
                    "input": ctx, "target": tgt,
                    "chat_idx": cidx, "turn_idx": len(turns),
                    "input_tokens": tok.enc(ctx), "target_tokens": tok.enc(tgt),
                    "source": src,
                })
        if turns:
            groups[cidx] = turns
            _META[cidx] = {"orig_id": str(orig_id), "source": src}
    print(f"  [BARU] percakapan={len(groups)} rows={sum(len(v) for v in groups.values())}")
    return groups


# peta chat_idx -> orig id / source (dipakai di map file)
_META: dict = {}



def main():
    print("=" * 70)
    print("BUILD chat_sft v1 (gabung lama + baru, shuffle per percakapan, split 80:20)")
    print("=" * 70)
    tok = _T(_tokenizer())
    old_groups = load_old_chat()
    new_groups = load_new_text(tok)

    all_conv = {}
    all_conv.update(old_groups)
    all_conv.update(new_groups)
    conv_keys = list(all_conv.keys())
    print(f"  TOTAL percakapan: {len(conv_keys)} (lama={len(old_groups)} baru={len(new_groups)})")

    # SHUFFLE berbasis percakapan
    random.seed(SEED)
    random.shuffle(conv_keys)
    n_train = int(len(conv_keys) * TRAIN_FRACTION)
    train_keys = set(conv_keys[:n_train])
    val_keys = set(conv_keys[n_train:])
    print(f"  Split: train={len(train_keys)} percakapan | val={len(val_keys)} percakapan")

    def rows_for(keys):
        out = []
        for k in keys:
            out.extend(all_conv[k])
        return out

    train_rows = rows_for(train_keys)
    val_rows = rows_for(val_keys)
    print(f"  Rows: train={len(train_rows)} | val={len(val_rows)}")

    def write_split(rows, split):
        OUT_DIR.mkdir(parents=True, exist_ok=True)
        for f in OUT_DIR.glob(f"{split}-*.parquet"):
            f.unlink()
        ds = Dataset.from_list(rows, features=FEATURES)
        path = OUT_DIR / f"{split}-00000-of-00001.parquet"
        ds.to_parquet(path)
        print(f"  wrote {path} ({len(rows)} rows)")
        gc.collect()

    write_split(train_rows, "train")
    write_split(val_rows, "validation")

    # chat_idx_map.json utk trace ORPO lama -> (chat_idx, split, source)
    audit = {"version": "v1", "seed": SEED, "total_conversations": len(conv_keys),
             "train_conversations": len(train_keys), "validation_conversations": len(val_keys),
             "new_offset": NEW_OFFSET, "split": {"train": TRAIN_FRACTION, "validation": 1 - TRAIN_FRACTION}}
    conv_meta = {}
    for k in sorted(all_conv.keys()):
        src = all_conv[k][0]["source"] if all_conv[k] else ""
        meta = _META.get(k, {})
        conv_meta[str(k)] = {
            "source": src,
            "orig_id": meta.get("orig_id", str(k)),
            "split": "train" if k in train_keys else "validation",
        }
    audit["conversations"] = conv_meta
    with open(REPO_DIR / "chat_idx_map.json", "w", encoding="utf-8") as f:
        json.dump(audit, f, ensure_ascii=False, indent=2)
    print("  wrote chat_idx_map.json (untuk trace ORPO)")

    # PUSH ke HF config chat_sft
    api = HfApi()
    api.create_repo(repo_id=REPO_ID, repo_type="dataset", exist_ok=True, private=False)
    if any(OUT_DIR.glob("*.parquet")):
        api.upload_folder(repo_id=REPO_ID, repo_type="dataset", folder_path=str(OUT_DIR),
                          path_in_repo=SUBFOLDER, commit_message="dataset: chat_sft v1 (lama+baru, 80:20)")
        print(f"  ✅ pushed chat_sft -> {REPO_ID}/{SUBFOLDER}")
    api.upload_file(repo_id=REPO_ID, repo_type="dataset", path_or_fileobj=str(REPO_DIR / "chat_idx_map.json"),
                    path_in_repo="chat_idx_map.json", commit_message="dataset: chat_idx map (trace ORPO)")
    print("  ✅ pushed chat_idx_map.json")
    print("DONE")


if __name__ == "__main__":
    main()
