#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
BUILD & UPLOAD MONO-REPO DATASET T5GEMMA-2 INDONESIA (memory-safe)
=================================================================
Membangun SATU repositori dataset HF dari:
  1. HF lama  : daruokta/t5gemma2-indonesia-chat-formatted  (chat_sft, indoqa_sft, chat_orpo)
  2. HF lama  : daruokta/t5gemma2-indonesia-vision-formatted (vision_sft, vision_orpo)
  3. Lokal    : data/synthetic/generated_conv_agent.jsonl   (3000 baru: 2000 teks + 1000 vision)
  4. Lokal    : data/preference/*_add.jsonl                  (tambahan ORPO, opsional)

Struktur output (nested, satu repo -> bisa di-load_dataset(repo, config)):
    sft/text/chat_hf/             -> config chat_sft
    sft/text/indoqa/              -> config indoqa_sft
    sft/text/synthetic_text/      -> config sft_synthetic_text   (BARU 2000)
    sft/vision/hf/                -> config vision_sft
    sft/vision/synthetic_vision/  -> config sft_synthetic_vision (BARU 1000)
    orpo/text/chat_orpo/          -> config chat_orpo
    orpo/vision/vision_orpo/      -> config vision_orpo

KENAPA MEMORY-SAFE:
  - Source di-STREAM (bukan load penuh) dari HF / jsonl.
  - Ditulis per-CHUNK ke parquet shard (bounded memory), lalu dilepas (gc).
  - Vision (gambar) diproses per-conversation dalam shard kecil -> RAM aman.
  - Per-config saja (`--only`), jadi tidak semua data masuk memory sekaligus.
"""
from __future__ import annotations

import argparse
import gc
import hashlib
import itertools
import json
import os
import sys
from pathlib import Path
from typing import Any, Iterable, Iterator, Optional

from datasets import Dataset, Features, Value, Image as HFImage, List, Sequence, load_dataset
from huggingface_hub import HfApi

ROOT_DIR = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT_DIR / "data"
REPO_DIR = ROOT_DIR / "data_repo" / "t5gemma2-indonesia-instruct-v1"

DEFAULT_REPO_ID = "daruokta/t5gemma2-indonesia-instruct-v1"
CHAT_REPO = "daruokta/t5gemma2-indonesia-chat-formatted"
VISION_REPO = "daruokta/t5gemma2-indonesia-vision-formatted"
SYNTHETIC_JSONL = DATA_DIR / "synthetic" / "generated_conv_agent.jsonl"

MSGS_FEATURES = List({
    "role": Value("string"),
    "content": Value("string"),
})

# ----------------------------------------------------------------------
# CONFIG SPECS (sumber -> output nested subfolder + skema)
# ----------------------------------------------------------------------
TEXT_SFT_FEATURES = Features({
    "input": Value("string"),
    "target": Value("string"),
    "chat_idx": Value("int64"),
    "turn_idx": Value("int64"),
    "input_tokens": Value("int64"),
    "target_tokens": Value("int64"),
    "source": Value("string"),
})
INDOQA_FEATURES = Features({
    "input": Value("string"),
    "target": Value("string"),
    "input_tokens": Value("int64"),
    "target_tokens": Value("int64"),
    "source": Value("string"),
})
VISION_SFT_FEATURES = Features({
    "id": Value("string"),
    "images": Sequence(HFImage()),
    "messages": MSGS_FEATURES,
    "source": Value("string"),
})
ORPO_TEXT_FEATURES = Features({
    "id": Value("string"),
    "prompt": Value("string"),
    "chosen": Value("string"),
    "rejected": Value("string"),
    "flaw": Value("string"),
    "rationale": Value("string"),
    "source": Value("string"),
})
ORPO_VISION_FEATURES = Features({
    "id": Value("string"),
    "images": Sequence(HFImage()),
    "prompt": Value("string"),
    "chosen": Value("string"),
    "rejected": Value("string"),
    "flaw": Value("string"),
    "rationale": Value("string"),
    "source": Value("string"),
})

# source: (kind, args) ; kind in {"hf", "file"}
CONFIG_SPECS: dict[str, dict[str, Any]] = {
    "chat_sft": {
        "subfolder": "chat/text",
        "splits": {"train": "train", "validation": "validation"},
        "features": TEXT_SFT_FEATURES,
        "source": ("hf", (CHAT_REPO, "chat_sft")),
        "kind": "hf_text_sft",
    },
    "indoqa_sft": {
        "subfolder": "foundation/indoqa",
        "splits": {"train": "train", "validation": "validation"},
        "features": INDOQA_FEATURES,
        "source": ("hf", (CHAT_REPO, "indoqa_sft")),
        "kind": "hf_text_sft",
        "tags_chat": False,
    },
    "vision_sft": {
        "subfolder": "chat/vision",
        "splits": {"train": "train"},
        "features": VISION_SFT_FEATURES,
        "source": ("hf", (VISION_REPO, "vision_sft")),
        "kind": "hf_vision_sft",
    },
    "chat_orpo": {
        "subfolder": "orpo/text",
        "splits": {"train": "train"},
        "features": ORPO_TEXT_FEATURES,
        "source": ("hf", (CHAT_REPO, "chat_orpo")),
        "kind": "hf_orpo_text",
        "add_files": [DATA_DIR / "preference" / "orpo_chat_add.jsonl"],
    },
    "vision_orpo": {
        "subfolder": "orpo/vision",
        "splits": {"train": "train"},
        "features": ORPO_VISION_FEATURES,
        "source": ("hf", (VISION_REPO, "vision_orpo")),
        "kind": "hf_orpo_vision",
        "add_files": [DATA_DIR / "preference" / "orpo_vision_add.jsonl"],
        "target_size": 1000,
    },
}

GROUP_ORDER = ["sft/text", "sft/vision", "orpo/text", "orpo/vision"]


# ----------------------------------------------------------------------
# SOURCE ITERATORS (streaming — jangan load penuh)
# ----------------------------------------------------------------------
def _iter_hf_rows(repo: str, config: str, split: str) -> Iterator[dict]:
    ds = load_dataset(repo, config, split=split, streaming=True)
    for row in ds:
        yield dict(row)


def _iter_jsonl(path: Path) -> Iterator[dict]:
    if not path.exists():
        return
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except Exception:
                continue


def _is_vision(obj: dict) -> bool:
    if obj.get("category") == "vision_chat":
        return True
    for m in obj.get("messages", []):
        if m.get("role") == "user" and "📷" in m.get("content", ""):
            return True
    return False


def _norm_messages(messages: list) -> list:
    out = []
    for m in messages:
        out.append({"role": m.get("role", "user"), "content": m.get("content", "")})
    return out


def _load_images(paths: list) -> list:
    from PIL import Image as PILImage
    imgs = []
    for p in paths:
        if not isinstance(p, str):
            continue
        if Path(p).is_absolute() and not Path(p).exists():
            continue
        fps = [Path(p) if Path(p).is_absolute() else Path(ROOT_DIR / p)]
        for fp in fps:
            if fp.exists():
                try:
                    imgs.append(PILImage.open(fp).convert("RGB"))
                except Exception:
                    pass
                break
    return imgs


def _rows_for_config(cfg_name: str, split: str) -> Iterator[dict]:
    spec = CONFIG_SPECS[cfg_name]
    kind = spec["kind"]

    if kind in ("hf_text_sft", "hf_vision_sft", "hf_orpo_text", "hf_orpo_vision"):
        _, (repo, config) = spec["source"]
        source_tag = f"hf:{repo}:{config}"
        for row in _iter_hf_rows(repo, config, split):
            out = dict(row)
            out["source"] = source_tag
            if kind == "hf_orpo_text" and not out.get("id"):
                out["id"] = f"{cfg_name}_{len(out)}"
            yield out

    elif kind == "jsonl_synth_text":
        source_tag = f"file:{SYNTHETIC_JSONL.name}"
        for obj in _iter_jsonl(SYNTHETIC_JSONL):
            if _is_vision(obj):
                continue
            yield {
                "id": str(obj.get("id", "")),
                "source": obj.get("source", source_tag),
                "category": obj.get("category", "text_nlu_chat"),
                "num_turns": int(obj.get("num_turns", 0)),
                "num_pairs": int(obj.get("num_pairs", 0)),
                "reviewed": bool(obj.get("reviewed", False)),
                "edited_turns": int(obj.get("edited_turns", 0)),
                "messages": _norm_messages(obj.get("messages", [])),
            }

    elif kind == "jsonl_synth_vision":
        source_tag = f"file:{SYNTHETIC_JSONL.name}"
        for obj in _iter_jsonl(SYNTHETIC_JSONL):
            if not _is_vision(obj):
                continue
            imgs = _load_images(obj.get("images", []))
            yield {
                "id": str(obj.get("id", "")),
                "source": obj.get("source", source_tag),
                "category": obj.get("category", "vision_chat"),
                "num_turns": int(obj.get("num_turns", 0)),
                "messages": _norm_messages(obj.get("messages", [])),
                "images": imgs,
            }


def _iter_orpo_with_additions(cfg_name: str, split: str) -> Iterator[dict]:
    spec = CONFIG_SPECS[cfg_name]
    _, (repo, config) = spec["source"]
    yield from _rows_for_config(cfg_name, split)
    # Tambahan ORPO lokal (opsional, belum tentu ada -> dilewati tanpa error)
    for add_file in spec.get("add_files", []):
        if not Path(add_file).exists():
            print(f"  ℹ️  {cfg_name}: tambahan lokal '{add_file}' belum ada (dilewati).")
            continue
        print(f"  ⊕ {cfg_name}: menyertakan tambahan lokal '{add_file}'.")
        for obj in _iter_jsonl(add_file):
            if not obj:
                continue
            if cfg_name == "vision_orpo":
                imgs = _load_images(obj.get("images", []))
                yield {
                    "id": str(obj.get("id", "")),
                    "images": imgs,
                    "prompt": obj.get("prompt", ""),
                    "chosen": obj.get("chosen", ""),
                    "rejected": obj.get("rejected", ""),
                    "flaw": obj.get("flaw", ""),
                    "rationale": obj.get("rationale", ""),
                    "source": obj.get("source", "local_add"),
                }
            else:
                yield {
                    "id": str(obj.get("id", "")),
                    "prompt": obj.get("prompt", ""),
                    "chosen": obj.get("chosen", ""),
                    "rejected": obj.get("rejected", ""),
                    "flaw": obj.get("flaw", ""),
                    "rationale": obj.get("rationale", ""),
                    "source": obj.get("source", "local_add"),
                }


# ----------------------------------------------------------------------
# CHUNKED PARQUET WRITER (memory-safe)
# ----------------------------------------------------------------------
def _write_parquet_chunked(
    cfg_name: str,
    split: str,
    out_dir: Path,
    features: Features,
    rows_iter: Iterator[dict],
    shard_size: int,
    force: bool,
) -> int:
    subfolder = out_dir / CONFIG_SPECS[cfg_name]["subfolder"]
    subfolder.mkdir(parents=True, exist_ok=True)

    # Bersihkan semua shard lama untuk split ini (kalau manifest belum 'built')
    # -> menghindari shard campuran dari build parsial sebelumnya.
    for f in subfolder.glob(f"{split}-*.parquet"):
        f.unlink()

    chunk = []
    written = 0
    shard_idx = 0
    sample = []

    def flush():
        nonlocal chunk, written, shard_idx, sample
        if not chunk:
            return
        ds = Dataset.from_list(chunk, features=features)
        if not sample:
            sample = list(ds.column_names)
        path = subfolder / f"{split}-{shard_idx:05d}-of-{99999}.parquet"
        ds.to_parquet(path)
        written += len(chunk)
        shard_idx += 1
        print(f"    · shard {shard_idx:02d} ({len(chunk)} rows) -> {path.name}")
        chunk = []
        ds = None
        gc.collect()

    for row in rows_iter:
        chunk.append(row)
        if len(chunk) >= shard_size:
            flush()

    flush()
    print(f"  ✅ {cfg_name}[{split}] -> {written} rows, {shard_idx} shard(s) di {subfolder.name}/")
    return written


# ----------------------------------------------------------------------
# MANIFEST
# ----------------------------------------------------------------------
def _load_manifest() -> dict:
    mf = REPO_DIR / "manifest.json"
    if mf.exists():
        with open(mf, "r", encoding="utf-8") as f:
            return json.load(f)
    return {"repo_id": DEFAULT_REPO_ID, "configs": {}}


def _save_manifest(manifest: dict) -> None:
    REPO_DIR.mkdir(parents=True, exist_ok=True)
    with open(REPO_DIR / "manifest.json", "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)


def _content_hash(cfg_name: str) -> str:
    spec = CONFIG_SPECS[cfg_name]
    h = hashlib.sha256()
    h.update(cfg_name.encode())
    h.update(str(spec["subfolder"]).encode())
    src = spec["source"]
    h.update(json.dumps(src, default=str).encode())
    for add in spec.get("add_files", []):
        h.update(str(add).encode())
    # ikutkan ukuran/mtime sumber lokal sebagai penanda perubahan
    if src[0] == "file":
        p = Path(src[1]) if isinstance(src[1], str) else SYNTHETIC_JSONL
        if p.exists():
            st = p.stat()
            h.update(f"{st.st_size}:{st.st_mtime_ns}".encode())
    return h.hexdigest()[:16]


# ----------------------------------------------------------------------
# MAIN
# ----------------------------------------------------------------------
def main() -> None:
    ap = argparse.ArgumentParser(description="Build & upload mono-repo dataset (memory-safe).")
    ap.add_argument("--repo-id", type=str, default=DEFAULT_REPO_ID)
    ap.add_argument("--only", type=str, default="",
                    help="Comma list config yang diproses (default: semua).")
    ap.add_argument("--shard-size", type=int, default=64,
                    help="Batas baris per chunk (kecil untuk vision, misal 32).")
    ap.add_argument("--push", action="store_true", help="Upload config ke HF setelah build.")
    ap.add_argument("--dry-run", action="store_true",
                    help="Hitung budget & target (tanpa menulis/upload).")
    ap.add_argument("--force", action="store_true", help="Rebuild walau sudah ada.")
    ap.add_argument("--limit", type=int, default=0, help="Batasi baris (0=semua) — untuk uji cepat.")
    ap.add_argument("--token", type=str, default=os.environ.get("HF_TOKEN"))
    args = ap.parse_args()

    manifest = _load_manifest()
    manifest["repo_id"] = args.repo_id
    want = [c.strip() for c in args.only.split(",") if c.strip()] if args.only else list(CONFIG_SPECS.keys())

    if args.dry_run:
        print("=" * 70)
        print(f"🧾 DRY-RUN — Budget dataset mono-repo  {args.repo_id}")
        print("=" * 70)
        for cfg_name in want:
            spec = CONFIG_SPECS[cfg_name]
            cfg_meta = manifest["configs"].setdefault(cfg_name, {"splits": {}, "target_size": None})
            counts = _count_source(cfg_name, manifest)
            tgt = spec.get("target_size")
            extra = " (BARU di SFT)" if cfg_name.startswith("sft_synthetic") else ""
            if cfg_name in ("chat_orpo", "vision_orpo"):
                add = ", +adds" if any(Path(f).exists() for f in spec.get("add_files", [])) else ""
                extra += add
            note = f" (target {tgt})" if tgt else ""
            print(f"  • {cfg_name:<22} -> {spec['subfolder']:<30} {counts} rows{note}{extra}")
        print("\n  Gunakan: python scripts/dataset/build_mono_repo.py --only <cfg> [--push]")
        return

    api: Optional[HfApi] = HfApi(token=args.token) if args.push else None
    if api is not None:
        api.create_repo(repo_id=args.repo_id, repo_type="dataset", exist_ok=True, private=False)

    for cfg_name in want:
        spec = CONFIG_SPECS[cfg_name]
        src_hash = _content_hash(cfg_name)
        m = manifest["configs"].get(cfg_name, {})
        if m.get("built") and m.get("build_sha") == src_hash and not args.force:
            print(f"⏭️  [{cfg_name}] build hash sama — dilewati (pakai --force untuk rebuild).")
        else:
            for split in spec["splits"]:
                if cfg_name in ("chat_orpo", "vision_orpo"):
                    rows_iter = _iter_orpo_with_additions(cfg_name, split)
                else:
                    rows_iter = _rows_for_config(cfg_name, split)
                if args.limit > 0:
                    rows_iter = itertools.islice(rows_iter, args.limit)
                written = _write_parquet_chunked(
                    cfg_name, split, REPO_DIR, spec["features"], rows_iter,
                    args.shard_size, args.force,
                )
                if written >= 0:
                    manifest["configs"].setdefault(cfg_name, {})["splits"][split] = written
            # Hanya tandai 'built' kalau build penuh (bukan --limit untuk uji cepat).
            if args.limit <= 0:
                manifest["configs"].setdefault(cfg_name, {})["built"] = True
                manifest["configs"].setdefault(cfg_name, {})["build_sha"] = src_hash
            _save_manifest(manifest)

        if args.push and api is not None:
            subfolder = REPO_DIR / spec["subfolder"]
            if any(subfolder.glob("*.parquet")):
                print(f"  📤 Upload {cfg_name} -> {args.repo_id}/{spec['subfolder']}")
                api.upload_folder(
                    repo_id=args.repo_id,
                    repo_type="dataset",
                    folder_path=str(subfolder),
                    path_in_repo=spec["subfolder"],
                    commit_message=f"dataset: {cfg_name} ({len(list(subfolder.glob('*.parquet')))} shard)",
                )

    _save_manifest(manifest)

    # Upload README + manifest ke root repo (dataset card & peta sumber)
    if args.push and api is not None:
        for root_file in ("README.md", "manifest.json"):
            if (REPO_DIR / root_file).exists():
                print(f"  📄 Upload {root_file} -> {args.repo_id}/{root_file}")
                api.upload_file(
                    repo_id=args.repo_id,
                    repo_type="dataset",
                    path_or_fileobj=str(REPO_DIR / root_file),
                    path_in_repo=root_file,
                    commit_message=f"dataset: {root_file}",
                )
    print("🎉 Selesai.")


def _count_source(cfg_name: str, manifest: dict) -> dict:
    """Jumlah baris sumber untuk dry-run. Pakai manifest bila sudah ada (hindari stream berat)."""
    m = manifest["configs"].get(cfg_name, {})
    if m.get("splits"):
        return m["splits"]
    spec = CONFIG_SPECS[cfg_name]
    out = {}
    for split in spec["splits"]:
        n = 0
        if cfg_name in ("chat_orpo", "vision_orpo"):
            for _ in _iter_orpo_with_additions(cfg_name, split):
                n += 1
        else:
            for _ in _rows_for_config(cfg_name, split):
                n += 1
        out[split] = n
    return out


if __name__ == "__main__":
    main()
