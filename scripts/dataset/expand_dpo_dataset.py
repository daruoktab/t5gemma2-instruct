"""
[Phase 3.3] Expand DPO Dataset — dari 100 ke 1,000+ pairs
===========================================================
Script ini memperluas `data/preferences_dpo_light.jsonl` (saat ini 100 pasang)
dengan menghasilkan pasangan DPO baru dari dataset yang ada.

Strategi:
  - 60% dari chat_train.jsonl (sampling acak berbeda dari yang ada)
  - 30% dari indoqa_train.jsonl
  - 10% dari cot_reasoning_dataset.jsonl (jika tersedia)

Flaw types yang digunakan untuk rejected response:
  - echo_user: mengulang pertanyaan tanpa menjawab
  - vague: basa-basi kosong tanpa substansi
  - hallucination: klaim palsu yang terdengar meyakinkan
  - incomplete: jawaban terpotong di tengah
  - off_topic: menjawab hal yang tidak ditanya
  - overlong: menambahkan padding & repetisi berlebihan

Output: `data/preferences_dpo_expanded.jsonl`
  Format: {input, chosen, rejected, flaw_type, rationale, source}

Config via env vars (atau .env di root project):
  API_BASE_URL    — OpenAI-compatible base URL
  API_KEY         — API key
  API_MODEL       — model name

Contoh:
  conda activate unsloth
  python scripts/dataset/expand_dpo_dataset.py --target 900  # tambah 900 (→ total 1000)
  python scripts/dataset/expand_dpo_dataset.py --target 1000 --output data/preferences_dpo_v2.jsonl
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
CHAT_TRAIN_FILE    = DATA_DIR / "chat_train.jsonl"
INDOQA_TRAIN_FILE  = DATA_DIR / "indoqa_train.jsonl"
COT_DATASET_FILE   = DATA_DIR / "cot_reasoning_dataset.jsonl"
DPO_EXISTING_FILE  = DATA_DIR / "preferences_dpo_light.jsonl"
OUTPUT_FILE        = DATA_DIR / "preferences_dpo_expanded.jsonl"

MAX_RETRIES   = 3
SLEEP_BETWEEN = 1.2

FLAW_TYPES = ["echo_user", "vague", "hallucination", "incomplete", "off_topic", "overlong"]

FLAW_DESCRIPTIONS = {
    "echo_user": (
        "rejected harus mengulang atau hampir menyalin kata-kata pertanyaan user "
        "di turn terakhir secara repetitif TANPA memberikan jawaban substantif apapun."
    ),
    "vague": (
        "rejected berisi basa-basi kosong Bahasa Indonesia santai seperti "
        "'Wah pertanyaan yang menarik!' atau 'Tentu saja!' tanpa isi yang menjawab substansi masalah."
    ),
    "hallucination": (
        "rejected menyatakan klaim atau fakta konkret yang terdengar sangat meyakinkan "
        "tapi sebenarnya salah, tidak diverifikasi, atau dibuat-buat (misleading)."
    ),
    "incomplete": (
        "rejected memulai menjawab dengan baik tapi tiba-tiba memotong kalimat "
        "di tengah jalan dengan '...' atau berhenti tanpa kesimpulan."
    ),
    "off_topic": (
        "rejected menjawab topik atau pertanyaan yang BERBEDA dari apa yang ditanyakan user, "
        "seolah salah membaca pertanyaan."
    ),
    "overlong": (
        "rejected memberikan jawaban yang benar tapi mengulang-ulang poin yang sama berkali-kali, "
        "menambahkan padding berlebihan, dan bertele-tele sehingga membosankan dan tidak efisien."
    ),
}


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
        raise ValueError(f"Bukan JSON: {raw[:100]}")


def load_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    rows.append(json.loads(line))
                except json.JSONDecodeError:
                    pass
    return rows


def get_existing_inputs(path: Path) -> set[str]:
    """Ambil set input yang sudah ada di DPO file untuk menghindari duplikat."""
    existing: set[str] = set()
    if not path.exists():
        return existing
    for row in load_jsonl(path):
        inp = row.get("input", "")[:200]
        if inp:
            existing.add(inp)
    return existing


def build_dpo_prompt(inp: str, chosen: str, flaw: str) -> str:
    return f"""Kamu adalah kurator dataset DPO (Direct Preference Optimization) untuk melatih model bahasa Bahasa Indonesia.

INPUT PERCAKAPAN (prompt dari user):
{inp[:1500]}

JAWABAN CHOSEN (berkualitas, gunakan ini sebagai dasar atau tingkatkan sedikit):
{chosen[:800]}

TUGAS:
1. chosen: Gunakan atau sempurnakan JAWABAN CHOSEN agar natural, membantu, dan dalam Bahasa Indonesia yang baik.
2. rejected: Buat jawaban LEBIH BURUK dengan cacat tipe: "{flaw}"
   Deskripsi cacat: {FLAW_DESCRIPTIONS[flaw]}
3. flaw_type: String persis "{flaw}"
4. rationale: Satu kalimat singkat Bahasa Indonesia menjelaskan mengapa rejected buruk.

PENTING:
- chosen harus benar-benar bagus dan informatif
- rejected harus jelas-jelas lebih buruk (bukan hanya sedikit berbeda)
- Keduanya harus dalam konteks yang relevan dengan input

Keluarkan HANYA objek JSON valid:
{{
  "chosen": "...",
  "rejected": "...",
  "flaw_type": "{flaw}",
  "rationale": "..."
}}
Tanpa markdown."""


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
                temperature=0.7,
                max_tokens=2048,
            )
            text = (completion.choices[0].message.content or "").strip()
            if not text:
                raise ValueError("Response kosong")
            return text
        except RateLimitError as e:
            last_exc = e
            time.sleep(12)
        except (APIConnectionError, APITimeoutError, OSError) as e:
            last_exc = e
    raise last_exc if last_exc else RuntimeError("API gagal")


def generate_dpo_pair(client: OpenAI, inp: str, chosen: str, flaw: str, source: str) -> dict | None:
    prompt = build_dpo_prompt(inp, chosen, flaw)

    for attempt in range(MAX_RETRIES):
        try:
            text = call_api(client, prompt)
            data = parse_json_loose(text)

            chosen_new  = data.get("chosen", "").strip()
            rejected    = data.get("rejected", "").strip()
            flaw_type   = data.get("flaw_type", flaw).strip()
            rationale   = data.get("rationale", "").strip()

            if not chosen_new or not rejected:
                print(f"    [retry {attempt + 1}] chosen/rejected kosong")
                continue

            if chosen_new == rejected:
                print(f"    [retry {attempt + 1}] chosen dan rejected identik")
                continue

            return {
                "input": inp,
                "chosen": chosen_new,
                "rejected": rejected,
                "flaw_type": flaw_type,
                "rationale": rationale,
                "source": source,
            }
        except Exception as e:
            print(f"    [retry {attempt + 1}] error: {e}")
        time.sleep(1.5)

    return None


def prepare_candidates(
    rng: random.Random,
    existing_inputs: set[str],
    n_chat: int,
    n_indoqa: int,
    n_cot: int,
) -> list[tuple[str, str, str]]:
    """Siapkan (input, target, source) dari berbagai sumber dataset."""
    candidates: list[tuple[str, str, str]] = []

    # Dari chat_train (ambil baris SFT langsung — sudah unrolled)
    chat_rows = load_jsonl(CHAT_TRAIN_FILE)
    chat_sampled = rng.sample(chat_rows, min(n_chat * 3, len(chat_rows)))
    count_chat = 0
    for row in chat_sampled:
        if count_chat >= n_chat:
            break
        inp    = row.get("input", "").strip()
        target = row.get("target", "").strip()
        if not inp or not target:
            continue
        if inp[:200] in existing_inputs:
            continue
        candidates.append((inp, target, "chat_train"))
        existing_inputs.add(inp[:200])
        count_chat += 1

    # Dari indoqa_train
    qa_rows = load_jsonl(INDOQA_TRAIN_FILE)
    qa_sampled = rng.sample(qa_rows, min(n_indoqa * 3, len(qa_rows)))
    count_qa = 0
    for row in qa_sampled:
        if count_qa >= n_indoqa:
            break
        inp    = row.get("input", "").strip()
        target = row.get("target", "").strip()
        if not inp or not target:
            continue
        if inp[:200] in existing_inputs:
            continue
        candidates.append((inp, target, "indoqa"))
        existing_inputs.add(inp[:200])
        count_qa += 1

    # Dari CoT dataset (jika ada)
    if n_cot > 0 and COT_DATASET_FILE.exists():
        cot_rows = load_jsonl(COT_DATASET_FILE)
        cot_sampled = rng.sample(cot_rows, min(n_cot * 3, len(cot_rows)))
        count_cot = 0
        for row in cot_sampled:
            if count_cot >= n_cot:
                break
            inp    = row.get("input", "").strip()
            target = row.get("target", "").strip()
            if not inp or not target:
                continue
            if inp[:200] in existing_inputs:
                continue
            candidates.append((inp, target, "cot"))
            existing_inputs.add(inp[:200])
            count_cot += 1

    rng.shuffle(candidates)
    return candidates


# ─── Main ────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description="Expand DPO dataset dari 100 ke 1000+ pairs")
    parser.add_argument("--target", type=int, default=900,
                        help="Jumlah pasang DPO baru yang ingin dibuat (default: 900)")
    parser.add_argument("--sleep", type=float, default=SLEEP_BETWEEN)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", type=Path, default=OUTPUT_FILE)
    parser.add_argument("--append-to-existing", action="store_true",
                        help="Append ke preferences_dpo_light.jsonl yang sudah ada")
    parser.add_argument("--dry-run", action="store_true",
                        help="Tampilkan statistik kandidat saja")
    args = parser.parse_args()

    rng = random.Random(args.seed)

    # Tentukan output file
    out_file = args.output
    if args.append_to_existing:
        out_file = DPO_EXISTING_FILE

    # Load existing inputs untuk avoid duplikat
    existing_inputs = get_existing_inputs(DPO_EXISTING_FILE)
    if out_file != DPO_EXISTING_FILE:
        existing_inputs |= get_existing_inputs(out_file)
    print(f"[INFO] Input yang sudah ada di DPO: {len(existing_inputs)} entri")

    # Distribusi sumber: 60% chat, 30% indoqa, 10% cot
    n_chat   = int(args.target * 0.60)
    n_indoqa = int(args.target * 0.30)
    n_cot    = args.target - n_chat - n_indoqa

    candidates = prepare_candidates(rng, existing_inputs, n_chat, n_indoqa, n_cot)

    if args.dry_run:
        by_source: dict[str, int] = {}
        for _, _, src in candidates:
            by_source[src] = by_source.get(src, 0) + 1
        print(f"\n[DRY-RUN] Kandidat tersedia untuk DPO: {len(candidates)}")
        for src, cnt in by_source.items():
            print(f"  {src}: {cnt}")
        print(f"\n[INFO] Target: {args.target} pasang baru | Output: {out_file}")
        return

    if not API_KEY:
        print("[ERROR] Set API_KEY atau OPENMODEL_API_KEY", file=sys.stderr)
        sys.exit(1)

    client = OpenAI(api_key=API_KEY, base_url=API_BASE_URL)
    out_file.parent.mkdir(parents=True, exist_ok=True)

    # Resume
    existing_count = 0
    if out_file.exists():
        with out_file.open("r", encoding="utf-8") as f:
            existing_count = sum(1 for l in f if l.strip())
        print(f"[INFO] Resume: sudah ada {existing_count} baris di {out_file}")

    produced = 0
    flaw_cycle = FLAW_TYPES * ((len(candidates) // len(FLAW_TYPES)) + 1)
    rng.shuffle(flaw_cycle)

    print(f"[INFO] API: {API_BASE_URL} | model={API_MODEL}")
    print(f"[INFO] Target: {args.target} pasang DPO baru | Output: {out_file}")

    for i, (inp, chosen, source) in enumerate(candidates):
        if produced >= args.target:
            break

        flaw = flaw_cycle[i % len(flaw_cycle)]
        print(f"\n[{produced + 1}/{args.target}] source={source} | flaw={flaw}")
        print(f"  input preview: {inp[:80].replace(chr(10), ' ')!r}...")

        pair = generate_dpo_pair(client, inp, chosen, flaw, source)

        if pair is None:
            print("  ✗ Gagal generate pair")
            time.sleep(args.sleep)
            continue

        with out_file.open("a", encoding="utf-8") as f:
            f.write(json.dumps(pair, ensure_ascii=False) + "\n")

        produced += 1
        print(f"  ✓ Pair #{produced} | chosen: {len(pair['chosen'])} char | rejected: {len(pair['rejected'])} char")
        time.sleep(args.sleep)

    print(f"\n[SELESAI] {produced} pasang DPO baru → {out_file}")
    print(f"[INFO] Total DPO di file: {existing_count + produced}")


if __name__ == "__main__":
    main()
