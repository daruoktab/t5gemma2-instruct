#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
PLAN & GENERATE ORPO BALANCE (agar chat vs vision tidak jomplang)
================================================================
v8 saat ini: chat_orpo=1000, vision_orpo=200  -> vision jauh lebih kecil.
Script ini menghitung budget tambahan yang diperlukan, lalu (opsional)
meng-generate respon 'rejected' dari kandidat SFT agar jumlahnya seimbang.

MODE:
  --plan        (default) hitung & cetak budget, tanpa memanggil API.
  --generate    buat pasangan ORPO tambahan (panggil LLM) dan tulis jsonl.

Sumber kandidat:
  - vision : data/synthetic/generated_conv_agent.jsonl (bagian vision) atau
             data/multimodal/train_vision.jsonl
  - text   : data/synthetic/generated_conv_agent.jsonl (bagian text) atau
             data/sft/chat_train_v2.jsonl

Output:
  - data/preference/orpo_vision_add.jsonl   (tambahan vision_orpo)
  - data/preference/orpo_chat_add.jsonl     (tambahan chat_orpo)

Contoh:
  python scripts/dataset/plan_orpo_balance.py --plan
  python scripts/dataset/plan_orpo_balance.py --generate --target-vision 1000
"""
from __future__ import annotations

import argparse
import json
import os
import random
from pathlib import Path
from typing import Any

ROOT_DIR = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT_DIR / "data"
CHAT_REPO = "daruokta/t5gemma2-indonesia-chat-formatted"
VISION_REPO = "daruokta/t5gemma2-indonesia-vision-formatted"

SYNTH_JSONL = DATA_DIR / "synthetic" / "generated_conv_agent.jsonl"
VIS_SFT_JSONL = DATA_DIR / "multimodal" / "train_vision.jsonl"
CHAT_SFT_JSONL = DATA_DIR / "sft" / "chat_train_v2.jsonl"

CURRENT: dict[str, dict[str, Any]] = {
    "chat_orpo": {"repo": CHAT_REPO, "config": "chat_orpo", "current": 1000, "target": 1000},
    "vision_orpo": {"repo": VISION_REPO, "config": "vision_orpo", "current": 200, "target": 1000},
}
ADD_FILES = {
    "chat_orpo": DATA_DIR / "preference" / "orpo_chat_add.jsonl",
    "vision_orpo": DATA_DIR / "preference" / "orpo_vision_add.jsonl",
}


# ----------------------------------------------------------------------
# Kandidat dari SFT vision/text
# ----------------------------------------------------------------------
def _messages_to_final_tuple(messages: list, obj_images: list | None = None) -> tuple | None:
    """
    Ambil prompt = semua pesan sebelum pesan asisten terakhir; chosen = jawaban asisten terakhir.
    Untuk vision, hitung jumlah \U0001F4F7 di dalam prompt lalu potong obj_images sesuai urutan.
    """
    last_asst = None
    for i, m in enumerate(messages):
        if m.get("role") == "assistant":
            last_asst = i
    if last_asst is None or last_asst == 0:
        return None
    ctx = messages[:last_asst]
    chosen = messages[last_asst].get("content", "").strip()
    if not chosen:
        return None
    lines = []
    n_img = 0
    for m in ctx:
        role = m.get("role", "user")
        content = m.get("content", "")
        if role == "user" and "\U0001F4F7" in content:
            n_img += content.count("\U0001F4F7")
        lines.append(f"{role}: {content}")
    imgs = []
    if obj_images:
        for p in obj_images:
            if n_img <= 0:
                break
            imgs.append(p)
            n_img -= 1
    return "\n".join(lines), chosen, imgs


def _iter_candidates(pool_path: Path, vision: bool) -> list[dict]:
    """Ambil kandidat prompt+chosen (+gambar utk vision) dari jsonl percakapan."""
    if not pool_path.exists():
        print(f"  ⚠️  Kandidat tidak ditemukan: {pool_path}")
        return []
    rows = []
    with open(pool_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except Exception:
                continue
            msgs = obj.get("messages", [])
            is_vis = (obj.get("category") == "vision_chat") or any(
                "\U0001F4F7" in m.get("content", "") for m in msgs if m.get("role") == "user"
            )
            if vision != is_vis:
                continue
            t = _messages_to_final_tuple(msgs, obj.get("images") if vision else None)
            if not t:
                continue
            prompt, chosen, imgs = t
            rows.append({"id": str(obj.get("id", "")), "prompt": prompt, "chosen": chosen, "images": imgs})
    return rows





# ----------------------------------------------------------------------
# PLAN
# ----------------------------------------------------------------------
def plan(target_vision: int, target_chat: int) -> dict:
    print("=" * 70)
    print("📊 BUDGET ORPO (agar tidak jomplang)")
    print("=" * 70)
    need = {}
    for name, spec in CURRENT.items():
        cur = spec["current"]
        tgt = target_vision if name == "vision_orpo" else target_chat
        spec["target"] = tgt
        delta = max(0, tgt - cur)
        need[name] = {"current": cur, "target": tgt, "delta": delta}
        print(f"  • {name:<14} current={cur:<6} target={tgt:<6} perlu={delta:+d}")
    print()

    vis_cand = _iter_candidates(SYNTH_JSONL, vision=True)
    txt_cand = _iter_candidates(SYNTH_JSONL, vision=False)
    print(f"  Kandidat vision tersedia : {len(vis_cand)}")
    print(f"  Kandidat teks tersedia   : {len(txt_cand)}")
    if need["vision_orpo"]["delta"] > len(vis_cand):
        print(f"  ⚠️  Perlu {need['vision_orpo']['delta']} vision, kandidat hanya {len(vis_cand)} -> "
              f"turuni target atau tambah kandidat (mis. train_vision.jsonl).")
    print()
    print("  Dari kandidat di atas, sample acak untuk tiap config (gunakan --generate agar menulis jsonl).")
    return need


# ----------------------------------------------------------------------
# GENERATE (panggil LLM, tulis jsonl, resumable)
# ----------------------------------------------------------------------
def _load_done(path: Path) -> set:
    if not path.exists():
        return set()
    done = set()
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            try:
                obj = json.loads(line)
                done.add(obj.get("id", ""))
            except Exception:
                pass
    return done


def generate(target_vision: int, target_chat: int, max_workers: int, api_key: str | None) -> None:
    # --- pilih kandidat ---
    vis_cand = _iter_candidates(SYNTH_JSONL, vision=True)
    txt_cand = _iter_candidates(SYNTH_JSONL, vision=False)

    for name, add_path, cand, target in (
        ("vision_orpo", ADD_FILES["vision_orpo"], vis_cand, target_vision),
        ("chat_orpo", ADD_FILES["chat_orpo"], txt_cand, target_chat),
    ):
        spec = CURRENT[name]
        need = max(0, target - spec["current"])
        done = _load_done(add_path)
        remaining = [c for c in cand if c["id"] not in done][:need]
        print(f"\n▶ {name}: butuh {need}, kandidat pending {len(remaining)} (sudah ada {len(done)}).")
        if not remaining:
            continue
        _generate_with_api(name, add_path, remaining, max_workers, api_key)


def _generate_with_api(name: str, out_path: Path, rows: list, max_workers: int, api_key: str | None) -> None:
    import concurrent.futures as cf
    import time
    import httpx

    base_url = os.environ.get("API_BASE_URL", "https://api.openmodel.ai/v1")
    model = os.environ.get("API_MODEL", "deepseek-chat")
    if not api_key:
        print("  ⚠️  Tidak ada API key (set API_KEY / OPENMODEL_API_KEY). Melewati generate.")
        return

    flaws = {
        "hallucination": "memberikan fakta yang salah/mengarang detail yang terdengar meyakinkan",
        "repetitive": "mengulang-ulang poin yang sama tanpa menambah informasi baru",
        "vague_and_short": "menjawab sangat singkat, kosong, dan tidak solutif",
        "off_topic": "mengalihkan ke topik lain yang tidak ditanyakan",
        "ignore_instruction": "mengabaikan instruksi format/spesifikasi user",
        "bad_list_formatting": "merusak format daftar/list dengan simbol tidak lazim",
    }

    SYSPROMPT = (
        "Kamu ahli pembuatan dataset preference (ORPO/DPO). Berikan respon 'rejected' yang sengaja"
        " lebih buruk dari 'chosen' sesuai instruksi flaw. Tanpa metateks, langsung sebagai asisten."
    )

    def call(row: dict) -> dict | None:
        prompt, chosen = row["prompt"], row["chosen"]
        imgs = row.get("images", [])
        flaw = random.choice(list(flaws.keys()))
        user_prompt = (
            f"RIWAYAT PERCAKAPAN:\n{prompt}\n\n"
            f"RESPON YANG BAIK (CHOSEN):\n{chosen}\n\n"
            f"INSTRUKSI: buat respon alternatif yang menderita cacat [{flaw}] — {flaws[flaw]}.\n"
        )
        try:
            resp = httpx.post(
                f"{base_url}/chat/completions",
                headers={"Authorization": f"Bearer {api_key}"},
                json={
                    "model": model,
                    "messages": [
                        {"role": "system", "content": SYSPROMPT},
                        {"role": "user", "content": user_prompt},
                    ],
                    "temperature": 0.8,
                    "max_tokens": 512,
                },
                timeout=60,
                verify=False,
            )
            data = resp.json()
            rej = data["choices"][0]["message"]["content"].strip()
            return {
                "id": row["id"],
                "images": imgs,
                "prompt": prompt,
                "chosen": chosen,
                "rejected": rej,
                "flaw": flaw,
                "rationale": f"Respon sengaja dibuat {flaw} agar lebih buruk dari chosen.",
                "source": "generated_add",
            }
        except Exception as e:
            print(f"  ✗ {row['id']} gagal: {str(e)[:80]}")
            return None

    out_path.parent.mkdir(parents=True, exist_ok=True)
    written = 0
    with cf.ThreadPoolExecutor(max_workers=max_workers) as ex:
        for res in ex.map(call, rows):
            if res:
                with open(out_path, "a", encoding="utf-8") as f:
                    f.write(json.dumps(res, ensure_ascii=False) + "\n")
                written += 1
                if written % 20 == 0:
                    print(f"    … {written}/{len(rows)} written")
    print(f"  ✅ {name} tambahan ditulis: {written} (total sekarang ~{written} baru).")


def main() -> None:
    ap = argparse.ArgumentParser(description="Plan/generate ORPO balance.")
    ap.add_argument("--mode", choices=["plan", "generate"], default="plan")
    ap.add_argument("--target-vision", type=int, default=1000)
    ap.add_argument("--target-chat", type=int, default=1000)
    ap.add_argument("--max-workers", type=int, default=8)
    ap.add_argument("--api-key", type=str, default=os.environ.get("API_KEY") or os.environ.get("OPENMODEL_API_KEY"))
    args = ap.parse_args()

    if args.mode == "plan":
        plan(args.target_vision, args.target_chat)
        return
    generate(args.target_vision, args.target_chat, args.max_workers, args.api_key)


if __name__ == "__main__":
    main()
