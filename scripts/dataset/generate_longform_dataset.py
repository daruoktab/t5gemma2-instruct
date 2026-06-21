"""
[Phase 3.2] Generate Long-Form Output Dataset Bahasa Indonesia
=============================================================
Script ini menghasilkan dataset instruksi dengan output panjang (200-1000 kata),
untuk melatih model menghasilkan respons yang lebih komprehensif dan mendalam.

Kategori output panjang:
  - Artikel analisis / opini (500-800 kata)
  - Esai formal (400-700 kata)
  - Review produk/film/buku (300-500 kata)
  - Penjelasan konsep mendalam (400-600 kata)
  - Laporan / summary dokumen panjang (300-500 kata)
  - Panduan langkah-demi-langkah detail (400-700 kata)

Output: `data/longform_output_dataset.jsonl`
  Format: {input, target, category, word_count, input_tokens, target_tokens}

Config via env vars (atau .env di root project):
  API_BASE_URL    — OpenAI-compatible base URL
  API_KEY         — API key
  API_MODEL       — model name

Contoh:
  conda activate unsloth
  python scripts/dataset/generate_longform_dataset.py --target 300
  python scripts/dataset/generate_longform_dataset.py --category artikel,panduan --target 100
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
OUTPUT_FILE  = DATA_DIR / "longform_output_dataset.jsonl"

MAX_RETRIES   = 3
SLEEP_BETWEEN = 2.0

SYSTEM_PROMPT = (
    "Kamu adalah asisten AI yang helpful, santai, dan ramah. "
    "Gunakan Bahasa Indonesia sebagai bahasa utama. "
    "Switch ke English hanya kalau user memang minta atau konteksnya English. "
    "Boleh casual dan natural — pakai 'aku/kamu' atau 'saya/Anda' sesuai situasi. "
    "Jangan terlalu formal kecuali situasinya memang mengharuskan."
)

# ─── Kategori & Template ─────────────────────────────────────────────────────

CATEGORIES = {
    "artikel": {
        "description": "Artikel analisis/opini tentang topik teknologi, sosial, budaya, atau sains",
        "min_words": 400,
        "max_words": 700,
        "prompt_template": (
            "Tulis {count} instruksi unik dari user yang meminta penulisan artikel analisis atau opini "
            "dalam Bahasa Indonesia tentang topik yang berbeda-beda (teknologi, lingkungan, budaya, ekonomi, dll).\n"
            "Untuk setiap instruksi, buat juga ARTIKEL LENGKAP sebagai respons asisten ({min_words}-{max_words} kata).\n"
            "Artikel harus punya struktur: pendahuluan, isi beberapa paragraf, dan penutup/kesimpulan.\n"
        ),
    },
    "panduan": {
        "description": "Panduan langkah-demi-langkah untuk keterampilan praktis",
        "min_words": 350,
        "max_words": 600,
        "prompt_template": (
            "Tulis {count} instruksi unik dari user yang meminta panduan atau tutorial step-by-step "
            "tentang keterampilan praktis berbeda-beda dalam Bahasa Indonesia "
            "(memasak, coding, berkebun, desain, keuangan, dll).\n"
            "Untuk setiap instruksi, buat PANDUAN LENGKAP sebagai respons ({min_words}-{max_words} kata) "
            "dengan langkah-langkah yang jelas, tips, dan warning jika ada.\n"
        ),
    },
    "review": {
        "description": "Review mendalam produk, film, buku, atau aplikasi",
        "min_words": 300,
        "max_words": 500,
        "prompt_template": (
            "Tulis {count} instruksi unik dari user yang meminta review atau ulasan mendalam "
            "tentang produk, film, buku, atau aplikasi yang berbeda-beda dalam Bahasa Indonesia.\n"
            "Untuk setiap instruksi, buat REVIEW LENGKAP sebagai respons ({min_words}-{max_words} kata) "
            "dengan penilaian jujur, pro-cons, dan rekomendasi untuk siapa produk/karya tersebut cocok.\n"
        ),
    },
    "penjelasan": {
        "description": "Penjelasan mendalam tentang konsep, fenomena, atau teknologi",
        "min_words": 350,
        "max_words": 600,
        "prompt_template": (
            "Tulis {count} instruksi unik dari user yang bertanya penjelasan mendalam tentang konsep, "
            "fenomena, atau teknologi yang berbeda-beda dalam Bahasa Indonesia.\n"
            "Untuk setiap instruksi, buat PENJELASAN LENGKAP sebagai respons ({min_words}-{max_words} kata) "
            "yang educational, menggunakan analogi jika perlu, dan mudah dipahami.\n"
        ),
    },
    "laporan": {
        "description": "Laporan atau ringkasan eksekutif dari situasi/dokumen",
        "min_words": 300,
        "max_words": 500,
        "prompt_template": (
            "Tulis {count} instruksi unik dari user yang meminta pembuatan laporan, rangkuman eksekutif, "
            "atau analisis situasi tentang berbagai topik dalam Bahasa Indonesia "
            "(bisnis, sosial, proyek, riset, dll).\n"
            "Untuk setiap instruksi, buat LAPORAN LENGKAP sebagai respons ({min_words}-{max_words} kata) "
            "dengan struktur yang profesional.\n"
        ),
    },
}

BATCH_SIZE = 2  # jumlah pasang (instruksi + respons) per API call


def build_generation_prompt(category: str, cat_info: dict) -> str:
    base = cat_info["prompt_template"].format(
        count=BATCH_SIZE,
        min_words=cat_info["min_words"],
        max_words=cat_info["max_words"],
    )
    return f"""{base}
Keluarkan HANYA array JSON dengan {BATCH_SIZE} objek, masing-masing berisi:
- "instruction": string — instruksi dari user (1-3 kalimat, natural)
- "response": string — respons asisten yang panjang ({cat_info['min_words']}-{cat_info['max_words']} kata)
- "category": "{category}"

Pastikan respons BENAR-BENAR panjang ({cat_info['min_words']}-{cat_info['max_words']} kata).
Format: [{{"instruction": "...", "response": "...", "category": "{category}"}}]
Tanpa markdown, hanya JSON array valid."""


def parse_json_loose(text: str) -> Any:
    raw = (text or "").strip()
    if raw.startswith("```"):
        raw = re.sub(r"^```(?:json)?\s*", "", raw)
        raw = re.sub(r"\s*```\s*$", "", raw).strip()
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        start = raw.find("[")
        end   = raw.rfind("]")
        if start >= 0 and end > start:
            return json.loads(raw[start:end + 1])
        start = raw.find("{")
        end   = raw.rfind("}")
        if start >= 0 and end > start:
            result = json.loads(raw[start:end + 1])
            return [result] if isinstance(result, dict) else result
        raise ValueError(f"Tidak bisa parse JSON: {raw[:120]!r}")


def call_api(client: OpenAI, prompt: str) -> str:
    backoff = [0.0, 3.0, 10.0]
    last_exc: BaseException | None = None
    for delay in backoff:
        if delay:
            time.sleep(delay)
        try:
            completion = client.chat.completions.create(
                model=API_MODEL,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.8,
                max_tokens=8192,
            )
            text = (completion.choices[0].message.content or "").strip()
            if not text:
                raise ValueError("Response kosong")
            return text
        except RateLimitError as e:
            last_exc = e
            time.sleep(15)
        except (APIConnectionError, APITimeoutError, OSError) as e:
            last_exc = e
    raise last_exc if last_exc else RuntimeError("API gagal")


def generate_longform_batch(client: OpenAI, category: str, cat_info: dict) -> list[dict]:
    prompt = build_generation_prompt(category, cat_info)

    for attempt in range(MAX_RETRIES):
        try:
            text = call_api(client, prompt)
            parsed = parse_json_loose(text)

            if isinstance(parsed, dict):
                parsed = [parsed]
            if not isinstance(parsed, list):
                print(f"    [retry {attempt + 1}] bukan list")
                continue

            rows = []
            for item in parsed:
                instruction = item.get("instruction", "").strip()
                response    = item.get("response", "").strip()
                if not instruction or not response:
                    continue

                word_count = len(response.split())
                if word_count < cat_info["min_words"] * 0.7:  # toleransi 30%
                    print(f"    [warn] response terlalu pendek: {word_count} kata")
                    continue

                inp    = f"system: {SYSTEM_PROMPT}\nuser: {instruction}"
                target = response

                rows.append({
                    "input": inp,
                    "target": target,
                    "category": category,
                    "word_count": word_count,
                    "input_tokens": len(inp.split()),
                    "target_tokens": word_count,
                })

            if rows:
                return rows
            print(f"    [retry {attempt + 1}] tidak ada row valid")

        except Exception as e:
            print(f"    [retry {attempt + 1}] error: {e}")
        time.sleep(3.0)

    return []


# ─── Main ────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description="Generate long-form output dataset")
    parser.add_argument("--target", type=int, default=300,
                        help="Jumlah baris yang ingin dibuat (default: 300)")
    parser.add_argument("--category", type=str, default="",
                        help=f"Kategori tertentu: {', '.join(CATEGORIES.keys())} (default: semua)")
    parser.add_argument("--sleep", type=float, default=SLEEP_BETWEEN)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", type=Path, default=OUTPUT_FILE)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    active = dict(CATEGORIES)
    if args.category:
        filtered = {k: v for k, v in CATEGORIES.items() if k in args.category.split(",")}
        if not filtered:
            print(f"[ERROR] Kategori tidak valid. Pilih: {', '.join(CATEGORIES.keys())}", file=sys.stderr)
            sys.exit(1)
        active = filtered

    if args.dry_run:
        print("\n[DRY-RUN] Kategori yang akan di-generate:")
        for cat, info in active.items():
            print(f"  {cat}: {info['description']}")
            print(f"         Target: {info['min_words']}-{info['max_words']} kata per respons")
        per_cat = args.target // len(active)
        print(f"\n  ~{per_cat} baris per kategori × {len(active)} kategori = ~{args.target} total")
        return

    if not API_KEY:
        print("[ERROR] Set API_KEY atau OPENMODEL_API_KEY", file=sys.stderr)
        sys.exit(1)

    client = OpenAI(api_key=API_KEY, base_url=API_BASE_URL)
    args.output.parent.mkdir(parents=True, exist_ok=True)

    # Resume
    existing_count = 0
    if args.output.exists():
        with args.output.open("r", encoding="utf-8") as f:
            existing_count = sum(1 for l in f if l.strip())
        print(f"[INFO] Resume: sudah ada {existing_count} baris di {args.output}")

    # Buat plan: distribusi merata antar kategori
    rng = random.Random(args.seed)
    plan: list[str] = []
    cats = list(active.keys())
    while len(plan) < args.target:
        rng.shuffle(cats)
        plan.extend(cats)
    plan = plan[:args.target]
    rng.shuffle(plan)

    produced = existing_count
    total_target = args.target + existing_count

    print(f"[INFO] API: {API_BASE_URL} | model={API_MODEL}")
    print(f"[INFO] Target: {args.target} baris | Output: {args.output}")

    i = 0
    while produced < total_target and i < len(plan):
        category = plan[i]
        i += 1

        print(f"\n[{produced - existing_count + 1}/{args.target}] category={category}")
        rows = generate_longform_batch(client, category, active[category])

        if not rows:
            print("  ✗ Batch gagal")
            time.sleep(args.sleep)
            continue

        with args.output.open("a", encoding="utf-8") as f:
            for row in rows:
                if produced < total_target:
                    f.write(json.dumps(row, ensure_ascii=False) + "\n")
                    produced += 1

        avg_words = sum(r["word_count"] for r in rows) / len(rows)
        print(f"  ✓ {len(rows)} respons | avg {avg_words:.0f} kata (total: {produced})")
        time.sleep(args.sleep)

    print(f"\n[SELESAI] {produced - existing_count} baris long-form → {args.output}")


if __name__ == "__main__":
    main()
