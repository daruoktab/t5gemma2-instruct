"""
Generate dataset preferensi untuk tahap **DPO / ORPO** (setelah SFT), memakai
DeepSeek API dengan pola mirip `generate_dataset_deepseek.py`.

Format keluaran (satu objek JSON per baris):
  - "input"   : prefix chat sama seperti chat_train (system:...\\nuser:...\\nassistant:...\\nuser: TERAKHIR)
  - "chosen"  : jawaban assistant yang dianggap lebih baik
  - "rejected": jawaban dengan cacat terkontrol (echo, vague, halusinasi ringan, tidak aman, off-topic)
  - "flaw_type": label singkat untuk audit
  - "rationale" (opsional): alasan singkat; bisa dibuang sebelum training DPO

Catatan integrasi TRL:
  - Encoder–decoder: field yang dipakai DPOTrainer bisa bernama `prompt` atau pasangan
    `chosen`/`rejected` sebagai teks completion saja — **cek versi TRL + transformers** kamu.
  - Saat training, terapkan `format_to_gemma` yang sama dengan SFT pada bagian prompt,
    dan akhiri chosen/rejected dengan token penutup yang sama dengan SFT (`<end_of_turn>`).

Lingkungan: sama seperti generate_dataset_deepseek (DEEPSEEK_API_KEY, .env, DEEPSEEK_MODEL, dll.)

Contoh:
  python generate_dataset_preferences_deepseek.py --target 500 \\
    --output preferences_dpo_indo.jsonl --seed 42
"""

from __future__ import annotations

import argparse
import json
import os
import random
import re
import sys
import time
from pathlib import Path

# Ensure current directory and root are in sys.path for local/absolute imports
_P = Path(__file__).resolve()
_CURRENT_DIR = str(_P.parent)
_ROOT_DIR = str(_P.parents[2])

if _CURRENT_DIR not in sys.path:
    sys.path.insert(0, _CURRENT_DIR)
if _ROOT_DIR not in sys.path:
    sys.path.insert(0, _ROOT_DIR)
from typing import Any, Literal, Optional

try:
    from pydantic import BaseModel, Field, root_validator
except ImportError:
    print("Install: pip install pydantic", file=sys.stderr)
    raise SystemExit(1) from None

SCRIPT_DIR = Path(__file__).resolve().parent

try:
    from dotenv import load_dotenv

    load_dotenv(SCRIPT_DIR / ".env")
except ImportError:
    pass

from scripts.dataset.generate_dataset_deepseek import (  # noqa: E402
    DEEPSEEK_BASE_URL,
    MODEL_CHAT,
    SYSTEM_PROMPT,
    call_deepseek_raw,
    parse_json_loose,
)

FlawType = Literal["echo_user", "vague", "hallucination", "unsafe", "off_topic"]


class PreferenceRecord(BaseModel):
    """Satu baris preferensi untuk JSONL."""

    input: str = Field(..., min_length=20, max_length=120_000)
    chosen: str = Field(..., min_length=5, max_length=16_000)
    rejected: str = Field(..., min_length=5, max_length=16_000)
    flaw_type: FlawType
    rationale: str = Field("", max_length=800)

    class Config:
        anystr_strip_whitespace = True

    @root_validator(pre=False)
    def chosen_differs_from_rejected(cls, values):
        chosen = values.get("chosen")
        rejected = values.get("rejected")
        if chosen is not None and rejected is not None:
            if chosen.strip() == rejected.strip():
                raise ValueError("chosen dan rejected tidak boleh identik")
        return values


def build_preference_prompt(flaw: FlawType, rng: random.Random) -> str:
    """Instruksi untuk menghasilkan SATU objek JSON preferensi."""
    scenario_hints = [
        "topik keuangan pribadi atau investasi pemula",
        "relasi sosial atau komunikasi di tempat kerja",
        "kesehatan umum non-diagnosis (edukasi ringan)",
        "teknologi sehari-hari (HP, laptop, aplikasi)",
        "pendidikan atau belajar bahasa",
    ]
    hint = rng.choice(scenario_hints)
    flaw_desc = {
        "echo_user": "rejected harus **mengulang atau hampir menyalin** pertanyaan user tanpa jawaban substantif",
        "vague": "rejected hanya basa-basi umum ('Wah menarik nih') tanpa isi yang menjawab",
        "hallucination": "rejected menyatakan fakta konkret yang **plausibel tapi tidak diverifikasi** dan bisa misleading",
        "unsafe": "rejected memberi instruksi berbahaya / melanggar kebijakan (lalu chosen harus menolak dengan aman)",
        "off_topic": "rejected menjawab topik lain yang tidak diminta user",
    }[flaw]
    return f"""Kamu membuat SATU contoh data preferensi (DPO-style) untuk melatih chatbot Bahasa Indonesia.

SYSTEM dataset (wajib sama persis di field input, di awalan system:):
{SYSTEM_PROMPT}

Syarat konten:
- Buat **satu** situasi multi-turn singkat dalam format `input`: minimal ada **satu** pasangan user–assistant sebelum **user** pertanyaan terakhir, lalu user pertanyaan terakhir yang jelas.
- `input` harus berupa teks multi-baris dengan pola persis:
  system: ...\\nuser: ...\\nassistant: ...\\nuser: ... (boleh lebih dari satu blok user/assistant sebelum user terakhir)
- Akhiri `input` dengan **user:** baris terakhir (pertanyaan / permintaan) tanpa menjawabnya di `input`.
- `chosen`: jawaban assistant yang membantu, konsisten dengan system, Bahasa Indonesia natural.
- `rejected`: jawaban yang lebih buruk dengan cacat: {flaw_desc}
- flaw_type harus persis string: "{flaw}"
- rationale: satu kalimat mengapa rejected lebih buruk.

Konteks tema acak (boleh dipakai atau diadaptasi): {hint}

Keluarkan HANYA satu objek JSON valid dengan kunci:
"input", "chosen", "rejected", "flaw_type", "rationale"
Tanpa markdown code fence."""


def _validate_model(model_cls, data: dict[str, Any]):
    if hasattr(model_cls, "model_validate"):
        return getattr(model_cls, "model_validate")(data)
    return getattr(model_cls, "parse_obj")(data)


def _dump_model(model_inst) -> dict[str, Any]:
    if hasattr(model_inst, "model_dump"):
        return getattr(model_inst, "model_dump")()
    return getattr(model_inst, "dict")()


def validate_record(data: dict[str, Any]) -> tuple[PreferenceRecord | None, str]:
    try:
        flaw = data.get("flaw_type")
        if flaw not in ("echo_user", "vague", "hallucination", "unsafe", "off_topic"):
            return None, f"flaw_type tidak valid: {flaw!r}"
        rec = _validate_model(PreferenceRecord, data)
        if rec.chosen.strip() == rec.rejected.strip():
            return None, "chosen == rejected"
        if "system:" not in rec.input.lower():
            return None, "input tanpa system:"
        if not re.search(r"(?i)user:\s*\S", rec.input):
            return None, "input tanpa user:"
        return rec, "ok"
    except Exception as e:
        return None, str(e)


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate DPO-style preference JSONL via DeepSeek")
    parser.add_argument("--output", type=Path, default=SCRIPT_DIR / "preferences_dpo.jsonl")
    parser.add_argument("--target", type=int, default=200, help="Jumlah baris preferensi")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--sleep", type=float, default=1.2)
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Cetak prompt contoh lalu keluar (tanpa API)",
    )
    args = parser.parse_args()

    rng = random.Random(args.seed)
    flaws: list[FlawType] = ["echo_user", "vague", "hallucination", "unsafe", "off_topic"]

    api_key = os.environ.get("DEEPSEEK_API_KEY")
    if not api_key and not args.dry_run:
        print("Set DEEPSEEK_API_KEY atau gunakan --dry-run", file=sys.stderr)
        sys.exit(1)

    print(f"[INFO] API base={DEEPSEEK_BASE_URL} model={MODEL_CHAT}")
    flaw = flaws[0]
    sample_prompt = build_preference_prompt(flaw, rng)
    if args.dry_run:
        print("[DRY-RUN] Contoh prompt (truncated):\n", sample_prompt[:1200], "...\n")
        return

    from openai import OpenAI

    client = OpenAI(api_key=api_key, base_url=DEEPSEEK_BASE_URL)
    args.output.parent.mkdir(parents=True, exist_ok=True)

    existing = 0
    if args.output.exists():
        with args.output.open("r", encoding="utf-8") as f:
            existing = sum(1 for _ in f)
    print(f"[INFO] Baris existing di {args.output}: {existing}")

    produced = 0
    attempts = 0
    max_attempts = args.target * 8
    while produced < args.target and attempts < max_attempts:
        attempts += 1
        flaw = rng.choice(flaws)
        prompt = build_preference_prompt(flaw, rng)
        try:
            text, tok = call_deepseek_raw(
                client,
                prompt,
                max_tokens=8192,
                temperature=0.75,
            )
            data = parse_json_loose(text)
            rec, reason = validate_record(data)
            if rec is None:
                print(f"  [skip] validasi: {reason}")
                time.sleep(args.sleep)
                continue
            line = _dump_model(rec)
            with args.output.open("a", encoding="utf-8") as out:
                out.write(json.dumps(line, ensure_ascii=False) + "\n")
            produced += 1
            print(f"  [{produced}/{args.target}] flaw={rec.flaw_type} ok (~tok={tok})")
        except Exception as e:
            print(f"  [err] {e}")
        time.sleep(args.sleep)

    print(f"[SELESAI] {produced} baris → {args.output} (attempts={attempts})")


if __name__ == "__main__":
    main()
