#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
BUILD sea_general_sft + indoqa_sft  (skema B: + task) — STREAMING
=================================================================
- SEA-Instruct-2602 (Indonesian) — STREAMING (tidak load penuh), filter
  prompt_translated_from_english == False (+ kualitas Excellent opsional).
  Unroll per-turn, chat_idx unik, split 80:20 via HASH deterministik (streaming-safe).
- indoqa — load kecil, per-row = 1 percakapan (turn_idx=0), real shuffle, split 80:20.

Kolom (skema B): input, target, chat_idx, turn_idx, input_tokens, target_tokens, source, task
Ditulis per-CHUNK ke parquet shard (RAM aman), lalu push ke HF.
"""
from __future__ import annotations

import argparse
import ast
import gc
import hashlib
import json
import random
from pathlib import Path

from datasets import Dataset, Features, Value, load_dataset
from huggingface_hub import HfApi

ROOT = Path(__file__).resolve().parents[2]
REPO_DIR = ROOT / "data_repo" / "t5gemma2-indonesia-instruct-v1"
REPO_ID = "daruokta/t5gemma2-indonesia-instruct-v1"
SEA_REPO = "aisingapore/SEA-Instruct-2602"
INDOQA_REPO = "daruokta/t5gemma2-indonesia-chat-formatted"
SEED = 3407

FEATURES = Features({
    "input": Value("string"),
    "target": Value("string"),
    "chat_idx": Value("int64"),
    "turn_idx": Value("int64"),
    "input_tokens": Value("int64"),
    "target_tokens": Value("int64"),
    "source": Value("string"),
    "task": Value("string"),
})


def _hash_split(cidx, frac_train=0.8):
    """Split deterministik 80:20 berbasis hash chat_idx (streaming-safe & adil)."""
    h = hashlib.md5(str(cidx).encode()).digest()[0]
    return "train" if h < int(frac_train * 256) else "validation"


def _hash_keep(idx, ratio):
    """Sampling acak deterministik ~ratio dari stream (tanpa hold semua baris)."""
    h = hashlib.md5(f"sea:{idx}".encode()).digest()[0]
    return (h / 256.0) < ratio


def _render_context(msgs, upto):
    lines = []
    for m in msgs[:upto]:
        role = m.get("role", "user")
        lines.append(f"{role}: {m.get('content', '')}")
    return "\n".join(lines)


def _parse_conversations(raw):
    """conversations bisa berupa string JSON ATAU Python-repr (petik tunggal).
    PENTING: field ini sering disimpan sebagai repr (single-quote), BUKAN JSON valid."""
    if isinstance(raw, str):
        text = raw.strip()
        try:
            raw = json.loads(text)
        except Exception:
            try:
                raw = ast.literal_eval(text)
            except Exception:
                return []
    if not isinstance(raw, list):
        return []
    return raw


def _turns_from_conv(convs, cidx, src, task):
    turns = []
    for i, m in enumerate(convs):
        if m.get("role") == "assistant":
            ctx = _render_context(convs, i)
            tgt = m.get("content", "")
            if not ctx.strip() or not tgt.strip():
                continue
            turns.append({
                "input": ctx, "target": tgt, "chat_idx": cidx,
                "turn_idx": len(turns), "input_tokens": 0, "target_tokens": 0,
                "source": src, "task": task,
            })
    return turns


def _write_shard(rows, out_dir, split, shard_idx):
    path = out_dir / f"{split}-{shard_idx:05d}-of-99999.parquet"
    ds = Dataset.from_list(rows, features=FEATURES)
    ds.to_parquet(path)
    gc.collect()
    return path


# =====================================================================
def build_sea(args):
    print("=" * 70)
    print(f"SEA-Instruct (Indonesian) -> foundation/sea_instruct  [STREAMING]")
    print("=" * 70)
    out = REPO_DIR / "foundation" / "sea_instruct"
    out.mkdir(parents=True, exist_ok=True)
    for f in out.glob("*-*.parquet"):
        f.unlink()

    ds = load_dataset(SEA_REPO, "Indonesian", split="train", streaming=True)
    if args.max_seen and args.max_seen > 0:
        import itertools
        ds = itertools.islice(ds, args.max_seen)
    bufs = {"train": [], "validation": []}
    shard_cnt = {"train": 0, "validation": 0}
    cnt = {"train": 0, "validation": 0, "convs": 0, "rows": 0, "seen": 0}
    ratio = min(1.0, (args.sea_cap or 0) / max(1, args.sea_total_est))
    cidx = 0
    kept = 0
    acc_idx = 0
    for row in ds:
        cnt["seen"] += 1
        # 1) filter translated_from_english == False
        if row.get("prompt_translated_from_english", False):
            continue
        # 2) filter kualitas (opsional)
        if args.min_quality and args.min_quality != "all":
            if row.get("prompt_input_quality") != args.min_quality:
                continue
            if row.get("prompt_is_coherent") is False or row.get("prompt_is_natural") is False:
                continue
        convs = _parse_conversations(row.get("conversations"))
        if not convs:
            continue
        task = row.get("prompt_primary_task", "") or ""
        acc_idx += 1
        if args.sea_cap and not _hash_keep(acc_idx, ratio):
            continue
        turns = _turns_from_conv(convs, cidx, "SEA-Instruct-2602:Indonesian", task)
        cidx += 1
        if not turns:
            continue
        kept += 1
        cnt["convs"] += 1
        cnt["rows"] += len(turns)
        split = _hash_split(cidx)
        bufs[split].extend(turns)
        cnt[split] += len(turns)
        # flush bila buffer penuh
        for sp in ("train", "validation"):
            if len(bufs[sp]) >= args.shard_size:
                _write_shard(bufs[sp], out, sp, shard_cnt[sp])
                shard_cnt[sp] += 1
                bufs[sp] = []
        if cnt["seen"] % 20000 == 0:
            print(f"  … {cnt['seen']:,} row dilihat | conv={cnt['convs']:,} rows={cnt['rows']:,} "
                  f"(train={cnt['train']:,} val={cnt['validation']:,})", flush=True)
    for sp in ("train", "validation"):
        if bufs[sp]:
            _write_shard(bufs[sp], out, sp, shard_cnt[sp])
            shard_cnt[sp] += 1
    print(f"\n  ✅ SEA: conv={cnt['convs']:,} rows={cnt['rows']:,} "
          f"| train={cnt['train']:,} val={cnt['validation']:,} | shard train={shard_cnt['train']} val={shard_cnt['validation']}")
    return cnt


def build_indoqa(args):
    print("=" * 70)
    print("indoqa -> foundation/indoqa  (1 row = 1 percakapan)")
    print("=" * 70)
    out = REPO_DIR / "foundation" / "indoqa"
    out.mkdir(parents=True, exist_ok=True)
    for f in out.glob("*-*.parquet"):
        f.unlink()

    rows = []
    for split in ("train", "validation"):
        for r in load_dataset(INDOQA_REPO, "indoqa_sft", split=split):
            rows.append({"input": r["input"], "target": r["target"],
                         "chat_idx": 0, "turn_idx": 0,
                         "input_tokens": int(r.get("input_tokens") or 0),
                         "target_tokens": int(r.get("target_tokens") or 0),
                         "source": "indoqa", "task": "qa"})
    for i, r in enumerate(rows):
        r["chat_idx"] = i
    random.seed(args.seed)
    random.shuffle(rows)
    if args.indoqa_cap and len(rows) > args.indoqa_cap:
        rows = rows[: args.indoqa_cap]
        for i, r in enumerate(rows):
            r["chat_idx"] = i
    n_train = int(len(rows) * 0.8)
    train_rows, val_rows = rows[:n_train], rows[n_train:]
    print(f"  total rows: {len(rows)} | train={len(train_rows)} val={len(val_rows)}")
    for split, rr in (("train", train_rows), ("validation", val_rows)):
        if rr:
            _write_shard(rr, out, split, 0)
    return {"convs": len(rows), "rows": len(rows), "train": len(train_rows), "validation": len(val_rows)}


def push_config(subfolder):
    api = HfApi()
    api.create_repo(repo_id=REPO_ID, repo_type="dataset", exist_ok=True, private=False)
    folder = REPO_DIR / subfolder
    if any(folder.glob("*.parquet")):
        api.upload_folder(repo_id=REPO_ID, repo_type="dataset", folder_path=str(folder),
                          path_in_repo=subfolder, commit_message=f"dataset: {subfolder}")
        print(f"  ✅ pushed {subfolder}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--shard-size", type=int, default=2000)
    ap.add_argument("--min-quality", type=str, default="Excellent", help="'all' utk nonaktifkan")
    ap.add_argument("--seed", type=int, default=SEED)
    ap.add_argument("--only", type=str, default="sea,indoqa")
    ap.add_argument("--max-seen", type=int, default=0, help="Batasi baris stream (uji cepat)")
    ap.add_argument("--sea-cap", type=int, default=100000, help="Cap jumlah percakapan SEA (0 = semua)")
    ap.add_argument("--sea-total-est", type=int, default=822251, help="Estimasi total percakapan SEA yang lolos filter (utk ratio sampling)")
    ap.add_argument("--indoqa-cap", type=int, default=4000, help="Cap indoqa (angka cantik; 0 = semua)")
    ap.add_argument("--push", action="store_true")
    args = ap.parse_args()
    only = [x.strip() for x in args.only.split(",") if x.strip()]

    if "sea" in only:
        build_sea(args)
        if args.push:
            push_config("foundation/sea_instruct")
    if "indoqa" in only:
        build_indoqa(args)
        if args.push:
            push_config("foundation/indoqa")
    print("DONE")


if __name__ == "__main__":
    main()
