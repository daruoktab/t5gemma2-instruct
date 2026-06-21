"""
[Phase 3.4] Generate IndoQA v2 — Ekspansi Domain (Hukum, Bisnis, Teknis, Berita)
================================================================================
Script ini menghasilkan dataset Grounded QA (seperti IndoQA) tapi dengan fokus
pada domain-domain di luar Wikipedia sejarah/geografi, yaitu:
  - Dokumen Hukum (Pasal, UU, Peraturan)
  - Berita / Jurnalistik
  - Dokumen Bisnis & Keuangan (Laporan keuangan, memo)
  - Teknis & IT (Dokumentasi sistem, panduan instalasi)

Mekanisme:
Meminta LLM untuk membuat sebuah "konteks/paragraf" sintetis yang realistis,
lalu membuat 2-3 pertanyaan berdasarkan konteks tersebut beserta jawabannya.

Output: `data/indoqa_v2_dataset.jsonl`
  Format: {input, target, domain, input_tokens, target_tokens}

Config via env vars (atau .env di root project):
  API_BASE_URL    — OpenAI-compatible base URL
  API_KEY         — API key
  API_MODEL       — model name

Contoh:
  conda activate unsloth
  python scripts/dataset/generate_indoqa_v2.py --target 500
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
OUTPUT_FILE  = DATA_DIR / "indoqa_v2_dataset.jsonl"

MAX_RETRIES   = 3
SLEEP_BETWEEN = 1.5

SYSTEM_PROMPT = (
    "Kamu adalah asisten AI yang ahli dalam menganalisis dokumen."
)

DOMAINS = [
    "Dokumen Hukum (Pasal, UU, Peraturan Pemerintah)",
    "Berita / Jurnalistik Indonesia",
    "Dokumen Bisnis & Keuangan (Laporan tahunan, memo perusahaan, prospektus)",
    "Teknis & IT (Dokumentasi sistem, panduan instalasi, spesifikasi API)"
]

def build_generation_prompt(domain: str) -> str:
    return f"""Buatlah sebuah dataset Grounded QA (Tanya Jawab Berdasarkan Teks) untuk melatih AI.

Domain: {domain}

Langkah-langkah:
1. Buat sebuah KONTEKS/paragraf sintetis yang sangat realistis dan profesional dalam Bahasa Indonesia (sekitar 100-200 kata).
2. Buat 3 PERTANYAAN yang jawabannya PASTI ADA di dalam konteks tersebut.
3. Buat JAWABAN untuk masing-masing pertanyaan (1-3 kalimat yang natural dan jelas).

Keluarkan HANYA JSON objek dengan struktur berikut, tanpa markdown:
{{
  "konteks": "teks paragraf...",
  "qa_pairs": [
    {{"pertanyaan": "...", "jawaban": "..."}},
    {{"pertanyaan": "...", "jawaban": "..."}},
    {{"pertanyaan": "...", "jawaban": "..."}}
  ]
}}"""

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
        if start >= 0 and end > start:
            return json.loads(raw[start:end + 1])
        raise ValueError(f"Tidak bisa parse JSON: {raw[:120]!r}")

def call_api(client: OpenAI, prompt: str) -> str:
    backoff = [0.0, 3.0, 8.0]
    last_exc: BaseException | None = None
    for delay in backoff:
        if delay:
            time.sleep(delay)
        try:
            completion = client.chat.completions.create(
                model=API_MODEL,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.7,
                response_format={"type": "json_object"},
                max_tokens=2048,
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

def generate_indoqa_batch(client: OpenAI, domain: str) -> list[dict]:
    prompt = build_generation_prompt(domain)
    
    for attempt in range(MAX_RETRIES):
        try:
            text = call_api(client, prompt)
            data = parse_json_loose(text)
            
            konteks = data.get("konteks", "").strip()
            qa_pairs = data.get("qa_pairs", [])
            
            if not konteks or not isinstance(qa_pairs, list):
                print(f"    [retry {attempt+1}] format JSON tidak sesuai")
                continue
                
            rows = []
            for qa in qa_pairs:
                q = qa.get("pertanyaan", "").strip()
                a = qa.get("jawaban", "").strip()
                if not q or not a:
                    continue
                    
                # Format IndoQA:
                # system: ...
                # user: Jawablah pertanyaan berikut berdasarkan konteks yang tersedia.
                #
                # Konteks: {konteks}
                #
                # Pertanyaan: {pertanyaan}
                
                inp = (
                    f"system: {SYSTEM_PROMPT}\n"
                    f"user: Jawablah pertanyaan berikut berdasarkan konteks yang tersedia.\n\n"
                    f"Konteks: {konteks}\n\n"
                    f"Pertanyaan: {q}"
                )
                
                rows.append({
                    "input": inp,
                    "target": a,
                    "domain": domain,
                    "input_tokens": len(inp.split()),
                    "target_tokens": len(a.split()),
                })
                
            if rows:
                return rows
            print(f"    [retry {attempt+1}] tidak ada pasangan QA valid")
        except Exception as e:
            print(f"    [retry {attempt+1}] error: {e}")
        time.sleep(2.0)
        
    return []

# ─── Main ────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description="Generate IndoQA v2 dataset")
    parser.add_argument("--target", type=int, default=500,
                        help="Jumlah baris QA baru yang ingin dibuat (default: 500)")
    parser.add_argument("--sleep", type=float, default=SLEEP_BETWEEN)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", type=Path, default=OUTPUT_FILE)
    args = parser.parse_args()
    
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

    produced = existing_count
    total_target = args.target + existing_count
    
    rng = random.Random(args.seed)
    print(f"[INFO] Target: {args.target} baris QA baru | Output: {args.output}")
    
    # Setiap panggilan API idealnya menghasilkan 3 QA pairs
    # Kita butuh sekitar target/3 panggilan API
    api_calls_needed = (args.target // 3) + 5
    
    calls_made = 0
    while produced < total_target and calls_made < api_calls_needed:
        domain = rng.choice(DOMAINS)
        print(f"\n[{produced - existing_count}/{args.target}] domain={domain.split('(')[0].strip()}")
        
        rows = generate_indoqa_batch(client, domain)
        calls_made += 1
        
        if not rows:
            print("  ✗ Batch gagal")
            time.sleep(args.sleep)
            continue
            
        with args.output.open("a", encoding="utf-8") as f:
            for row in rows:
                if produced < total_target:
                    f.write(json.dumps(row, ensure_ascii=False) + "\n")
                    produced += 1
                    
        print(f"  ✓ {len(rows)} QA pairs ditambahkan (total: {produced})")
        time.sleep(args.sleep)
        
    print(f"\n[SELESAI] {produced - existing_count} QA pairs IndoQA v2 baru → {args.output}")

if __name__ == "__main__":
    main()
