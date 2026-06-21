"""
[Phase 1.2] Augment IndoQA — Buat Jawaban Lebih Panjang & Explanatory
=======================================================================
Script ini membaca `data/indoqa_train.jsonl` yang memiliki target sangat
pendek (rata-rata 5 kata), lalu meminta LLM untuk menghasilkan jawaban
yang lebih panjang dan explanatory (2-4 kalimat), tanpa mengubah kebenaran
faktual dari konteks yang diberikan.

Output: `data/indoqa_train_augmented.jsonl`
  Format sama: {input, target, input_tokens, target_tokens}

Config via env vars (atau .env di root project):
  API_BASE_URL    — OpenAI-compatible base URL
  API_KEY         — API key
  API_MODEL       — model name

Contoh:
  conda activate unsloth
  python scripts/dataset/augment_indoqa_targets.py --target 2000
  python scripts/dataset/augment_indoqa_targets.py --dry-run
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
import random
from pathlib import Path
from typing import Any

# ─── Load .env ───────────────────────────────────────────────────────────────
SCRIPT_DIR = Path(__file__).resolve().parent
ROOT_DIR   = SCRIPT_DIR.parent.parent
DATA_DIR   = ROOT_DIR / "data"

try:
    from dotenv import load_dotenv
    load_dotenv(ROOT_DIR / ".env")
except ImportError:
    pass

try:
    from openai import OpenAI, RateLimitError, APIConnectionError, APITimeoutError
except ImportError:
    print("Install: uv pip install openai", file=sys.stderr)
    raise SystemExit(1)

# ─── Konfigurasi API ─────────────────────────────────────────────────────────
API_BASE_URL = os.environ.get("API_BASE_URL", "https://api.openmodel.ai/v1")
API_KEY      = os.environ.get("API_KEY") or os.environ.get("OPENMODEL_API_KEY")
API_MODEL    = os.environ.get("API_MODEL", "deepseek-v4-flash")

# ─── Paths ───────────────────────────────────────────────────────────────────
INPUT_FILE       = DATA_DIR / "indoqa_train.jsonl"
INPUT_VAL_FILE   = DATA_DIR / "indoqa_val.jsonl"
OUTPUT_FILE      = DATA_DIR / "indoqa_train_augmented.jsonl"
OUTPUT_VAL_FILE  = DATA_DIR / "indoqa_val_augmented.jsonl"

# ─── Threshold augmentasi ────────────────────────────────────────────────────
# Hanya augment jawaban yang pendek (di bawah threshold kata)
SHORT_TARGET_THRESHOLD = 15   # kata — jawaban < 15 kata akan diaugment
MAX_RETRIES = 3
SLEEP_BETWEEN = 0.8

# ─── Helpers ─────────────────────────────────────────────────────────────────

def parse_json_loose(text: str) -> dict[str, Any]:
    raw = (text or "").strip()
    if raw.startswith("```"):
        raw = re.sub(r"^```(?:json)?\s*", "", raw)
        raw = re.sub(r"\s*```\s*$", "", raw).strip()
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        start = raw.find("{")
        end   = raw.rfind("}")
        if start < 0 or end <= start:
            raise ValueError(f"Bukan JSON: {raw[:100]}")
        return json.loads(raw[start:end + 1])


def extract_context_and_question(input_str: str) -> tuple[str, str]:
    """Ekstrak konteks dan pertanyaan dari format input IndoQA."""
    # Format: "system: ...\nuser: Jawablah ...\n\nKonteks: ...\n\nPertanyaan: ..."
    context = ""
    question = ""

    ctx_match = re.search(r"Konteks:\s*(.+?)(?:\n\nPertanyaan:|\Z)", input_str, re.DOTALL)
    q_match   = re.search(r"Pertanyaan:\s*(.+?)$", input_str, re.DOTALL)

    if ctx_match:
        context  = ctx_match.group(1).strip()
    if q_match:
        question = q_match.group(1).strip()

    return context, question


def build_augmentation_prompt(context: str, question: str, short_answer: str) -> str:
    return f"""Kamu adalah asisten AI yang ahli dalam analisis dokumen Bahasa Indonesia.

Diberikan:
- KONTEKS: paragraf teks bahasa Indonesia
- PERTANYAAN: pertanyaan tentang isi konteks
- JAWABAN SINGKAT: jawaban yang benar tapi terlalu pendek

Tugasmu: Tulis ulang JAWABAN menjadi 2-3 kalimat yang lebih explanatory dan natural dalam Bahasa Indonesia,
TANPA menambahkan fakta di luar konteks yang diberikan. Jawaban harus tetap faktual dan berdasarkan konteks.

KONTEKS:
{context[:800]}

PERTANYAAN:
{question}

JAWABAN SINGKAT (benar tapi terlalu pendek):
{short_answer}

Keluarkan HANYA objek JSON dengan satu kunci:
{{"augmented_answer": "jawaban yang lebih panjang dan explanatory di sini"}}

Format JSON valid saja, tanpa markdown."""


def call_api(client: OpenAI, prompt: str) -> str:
    backoff = [0.0, 2.0, 6.0]
    last_exc: BaseException | None = None
    for delay in backoff:
        if delay:
            time.sleep(delay)
        try:
            completion = client.chat.completions.create(
                model=API_MODEL,
                messages=[{"role": "user", "content": prompt}],
                response_format={"type": "json_object"},
                temperature=0.4,
                max_tokens=512,
            )
            text = (completion.choices[0].message.content or "").strip()
            if not text:
                raise ValueError("Response kosong")
            return text
        except RateLimitError as e:
            last_exc = e
            time.sleep(10)
        except (APIConnectionError, APITimeoutError, OSError) as e:
            last_exc = e
    raise last_exc if last_exc else RuntimeError("API gagal")


def augment_row(client: OpenAI, row: dict) -> dict | None:
    """Augment satu baris IndoQA. Kembalikan None jika gagal atau tidak perlu augment."""
    input_str     = row.get("input", "")
    short_answer  = row.get("target", "").strip()

    # Cek apakah perlu diaugment
    word_count = len(short_answer.split())
    if word_count >= SHORT_TARGET_THRESHOLD:
        return row  # Sudah cukup panjang, langsung return

    # Skip jika jawaban "tidak dapat ditemukan"
    if "tidak dapat" in short_answer.lower() or "tidak ditemukan" in short_answer.lower():
        return row

    context, question = extract_context_and_question(input_str)
    if not context or not question:
        return row  # Tidak bisa parse, skip

    prompt = build_augmentation_prompt(context, question, short_answer)

    for attempt in range(MAX_RETRIES):
        try:
            text = call_api(client, prompt)
            data = parse_json_loose(text)
            augmented = data.get("augmented_answer", "").strip()

            # Validasi: harus lebih panjang dari yang asli
            if augmented and len(augmented.split()) > word_count:
                new_row = dict(row)
                new_row["target"] = augmented
                new_row["target_original"] = short_answer  # simpan yang asli
                new_row["target_tokens"] = len(augmented.split())
                return new_row
            else:
                print(f"    [retry {attempt + 1}] augmented tidak lebih panjang: {augmented[:60]!r}")
        except Exception as e:
            print(f"    [retry {attempt + 1}] error: {e}")
        time.sleep(1.0)

    return row  # Gagal augment, kembalikan yang asli


def process_file(
    client: OpenAI,
    input_path: Path,
    output_path: Path,
    max_augment: int,
    seed: int = 42,
) -> None:
    if not input_path.exists():
        print(f"[WARN] File tidak ditemukan: {input_path}")
        return

    # Load semua rows
    rows: list[dict] = []
    with input_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    rows.append(json.loads(line))
                except json.JSONDecodeError:
                    pass

    print(f"[INFO] {input_path.name}: {len(rows)} baris dimuat")

    # Identifikasi yang perlu diaugment
    short_indices = [
        i for i, r in enumerate(rows)
        if len(r.get("target", "").split()) < SHORT_TARGET_THRESHOLD
        and "tidak dapat" not in r.get("target", "").lower()
    ]
    print(f"[INFO] Baris dengan target pendek (< {SHORT_TARGET_THRESHOLD} kata): {len(short_indices)}")

    # Batasi sesuai target
    rng = random.Random(seed)
    rng.shuffle(short_indices)
    to_augment = short_indices[:max_augment]
    print(f"[INFO] Akan augment: {len(to_augment)} baris → {output_path}")

    # Augment
    to_augment_set = set(to_augment)
    augmented_count = 0

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as out:
        for i, row in enumerate(rows):
            if i in to_augment_set:
                idx_in_queue = to_augment.index(i)
                print(f"  [{idx_in_queue + 1}/{len(to_augment)}] Augment baris {i} | target asli: '{row.get('target', '')[:40]}'...")
                new_row = augment_row(client, row)
                if new_row and new_row.get("target") != row.get("target"):
                    augmented_count += 1
                out.write(json.dumps(new_row or row, ensure_ascii=False) + "\n")
                time.sleep(SLEEP_BETWEEN)
            else:
                out.write(json.dumps(row, ensure_ascii=False) + "\n")

    print(f"[INFO] Selesai: {augmented_count} baris berhasil diaugment → {output_path}")


# ─── Main ────────────────────────────────────────────────────────────────────

def main() -> None:
    global SHORT_TARGET_THRESHOLD, SLEEP_BETWEEN
    parser = argparse.ArgumentParser(description="Augment IndoQA targets menjadi lebih panjang")
    parser.add_argument("--target", type=int, default=2000,
                        help="Max baris yang akan diaugment dari indoqa_train (default: 2000)")
    parser.add_argument("--target-val", type=int, default=500,
                        help="Max baris yang akan diaugment dari indoqa_val (default: 500)")
    parser.add_argument("--threshold", type=int, default=SHORT_TARGET_THRESHOLD,
                        help=f"Augment jawaban dengan < N kata (default: {SHORT_TARGET_THRESHOLD})")
    parser.add_argument("--sleep", type=float, default=SLEEP_BETWEEN)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--dry-run", action="store_true",
                        help="Hanya tampilkan statistik, tanpa augment")
    args = parser.parse_args()

    SHORT_TARGET_THRESHOLD = args.threshold
    SLEEP_BETWEEN = args.sleep

    if args.dry_run:
        # Hanya statistik
        for path, label in [(INPUT_FILE, "train"), (INPUT_VAL_FILE, "val")]:
            if not path.exists():
                print(f"[WARN] {path} tidak ada")
                continue
            rows = [json.loads(l) for l in path.open("r", encoding="utf-8") if l.strip()]
            short = [r for r in rows if len(r.get("target", "").split()) < args.threshold]
            avg_words = sum(len(r.get("target", "").split()) for r in rows) / max(len(rows), 1)
            print(f"\n[DRY-RUN] {label}: {len(rows)} total | {len(short)} target pendek | avg target: {avg_words:.1f} kata")
            print(f"  Contoh pendek:")
            for r in sorted(short, key=lambda x: len(x.get("target", "")))[:5]:
                print(f"    '{r.get('target', '')}' ({len(r.get('target','').split())} kata)")
        return

    if not API_KEY:
        print("[ERROR] Set API_KEY atau OPENMODEL_API_KEY di environment atau .env", file=sys.stderr)
        sys.exit(1)

    client = OpenAI(api_key=API_KEY, base_url=API_BASE_URL)

    print(f"[INFO] API: base_url={API_BASE_URL} | model={API_MODEL}")
    print(f"[INFO] Threshold augmentasi: < {SHORT_TARGET_THRESHOLD} kata")

    process_file(client, INPUT_FILE, OUTPUT_FILE, args.target, args.seed)
    process_file(client, INPUT_VAL_FILE, OUTPUT_VAL_FILE, args.target_val, args.seed)


if __name__ == "__main__":
    main()
