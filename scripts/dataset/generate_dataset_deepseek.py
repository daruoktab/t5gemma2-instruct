"""
Generate chat instruct dataset entries menggunakan DeepSeek API (OpenAI-compatible).

Format keluaran: **sama seperti** `t5-gemma-2-chat-instruct-dataset.jsonl` — tiap baris
`{"id", "topik", "num_turns", "tokens", "conversations": [{"role","content"}, ...]}`.
Ini **bukan** format `chat_train.jsonl` (hasil *turn unrolling* ke `input`/`target` untuk trainer);
flatten ke `input`/`target` dilakukan di pipeline training terpisah (mis. `flatten_conversations_jsonl_to_sft.py`).

Alur per entri:
  1) Generate topik baru yang belum ada di dataset dasar + yang sudah dihasilkan di run ini.
  2) Generate percakapan multi-turn sesuai spesifikasi (encoder-decoder friendly tasks dalam chat).

Lingkungan:
  - Taruh kunci di file instruct/.env (dimuat otomatis jika python-dotenv terpasang), atau:
  - export DEEPSEEK_API_KEY="sk-..."
  # opsional (default: deepseek-v4-flash — selaras rekomendasi terbaru dokumentasi):
  export DEEPSEEK_MODEL="deepseek-v4-flash"
  # opsional bila sering error content kosong / JSON topik terpotong:
  export DEEPSEEK_TOPIC_MAX_TOKENS="8192"
  export DEEPSEEK_CONV_MAX_TOKENS="8192"
  export DEEPSEEK_TOPIC_TEMPERATURE="0.35"
  export DEEPSEEK_CONV_TEMPERATURE="0.85"

Dependensi:
  pip install openai
  pip install pydantic
  pip install python-dotenv   # opsional, untuk memuat .env

Contoh:
  python generate_dataset_deepseek.py --target 8970 --start-id 1031 \\
    --output t5-gemma-2-chat-instruct-dataset-extra8970.jsonl

  # Berhenti saat id berikutnya akan melewati batas (mis. id global max 10000 termasuk base):
  python generate_dataset_deepseek.py --target 8970 --start-id 1031 --max-id 10000

Resume otomatis: baris yang sudah ada di output diskip berdasarkan id & topik log.

Referensi API (OpenAI-compatible): https://api-docs.deepseek.com/
  - base_url: https://api.deepseek.com
  - Model umum: deepseek-v4-flash, deepseek-v4-pro; alias deepseek-chat / deepseek-reasoner
    dijadwalkan deprecated 2026/07/24 (lihat tabel model di dokumentasi).
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
from typing import Any, Literal

try:
    from openai import APIConnectionError, APITimeoutError, OpenAI, RateLimitError
except ImportError:
    print("Install: pip install openai", file=sys.stderr)
    raise

try:
    from pydantic import AliasChoices, BaseModel, ConfigDict, Field
except ImportError as exc:
    print("Install: pip install pydantic", file=sys.stderr)
    raise SystemExit(1) from exc

# -----------------------------------------------------------------------------
# Konfigurasi default (samakan dengan dataset / full_train_270m)
# -----------------------------------------------------------------------------
SCRIPT_DIR = Path(__file__).resolve().parent

try:
    from dotenv import load_dotenv

    load_dotenv(SCRIPT_DIR / ".env")
except ImportError:
    pass

DEFAULT_BASE_DATASET = SCRIPT_DIR / "../../data/t5-gemma-2-chat-instruct-dataset.jsonl"
DEFAULT_OUTPUT = SCRIPT_DIR / "../../data/t5-gemma-2-chat-instruct-dataset-extra8970.jsonl"
DEFAULT_TOPICS_LOG = SCRIPT_DIR / "../../data/t5-gemma-2-chat-instruct-dataset-extra8970_topics.jsonl"

SYSTEM_PROMPT = (
    "Kamu adalah asisten AI yang helpful, santai, dan ramah. "
    "Gunakan Bahasa Indonesia sebagai bahasa utama. "
    "Switch ke English hanya kalau user memang minta atau konteksnya English. "
    "Boleh casual dan natural — pakai 'aku/kamu' atau 'saya/Anda' sesuai situasi. "
    "Kalau ada task seperti translate, summarize, paraphrase, atau rewrite "
    "muncul dalam obrolan, langsung bantu dengan natural tanpa basa-basi berlebihan. "
    "Jangan terlalu formal kecuali situasinya memang mengharuskan."
)

# Dokumentasi: https://api-docs.deepseek.com/ — format kompatibel OpenAI.
DEEPSEEK_BASE_URL = os.environ.get("DEEPSEEK_BASE_URL", "https://api.deepseek.com")
# Default: deepseek-v4-flash (non-thinking). deepseek-chat masih alias kompatibilitas hingga 2026/07/24.
MODEL_CHAT = os.environ.get("DEEPSEEK_MODEL", "deepseek-v4-flash")

# Total pesan termasuk system: 21..31  => 10..15 pasang user-assistant
MIN_MESSAGES_TOTAL = 21
MAX_MESSAGES_TOTAL = 31

MAX_RETRIES_TOPIC = 5
MAX_RETRIES_CONV = 4
SLEEP_DEFAULT = 1.5

# Generate topik = JSON kecil; batas terlalu rendah → finish_reason=length → message.content kosong.
TOPIC_MAX_TOKENS_BASE = int(os.environ.get("DEEPSEEK_TOPIC_MAX_TOKENS", "8192"))
CONV_MAX_TOKENS = int(os.environ.get("DEEPSEEK_CONV_MAX_TOKENS", "8192"))
# Topik harus ringkas; suhu tinggi sering membuat keluaran verbose → JSON terpotong (finish_reason=length).
TOPIC_TEMPERATURE = float(os.environ.get("DEEPSEEK_TOPIC_TEMPERATURE", "0.35"))
CONV_TEMPERATURE = float(os.environ.get("DEEPSEEK_CONV_TEMPERATURE", "0.85"))

# Jumlah pesan user+assistant sebelum inject system (10–15 pasang => 20–30 pesan)
MIN_BODY_TURNS = MIN_MESSAGES_TOTAL - 1
MAX_BODY_TURNS = MAX_MESSAGES_TOTAL - 1


class TopicGenResponse(BaseModel):
    """Satu objek JSON dari tahap generate topik."""

    model_config = ConfigDict(str_strip_whitespace=True)

    topik: str = Field(min_length=2, max_length=240)
    ringkasan: str = Field(
        min_length=5,
        max_length=1200,
        validation_alias=AliasChoices("ringkasan", "summary"),
    )


class DialogueTurn(BaseModel):
    model_config = ConfigDict(str_strip_whitespace=True)

    role: Literal["user", "assistant"]
    content: str = Field(min_length=1)


class ConversationGenBody(BaseModel):
    conversations: list[DialogueTurn] = Field(
        min_length=MIN_BODY_TURNS,
        max_length=MAX_BODY_TURNS,
    )


def normalize_topic(s: str) -> str:
    return re.sub(r"\s+", " ", (s or "").strip().lower())


def load_existing_topics(path: Path) -> set[str]:
    topics: set[str] = set()
    if not path.exists():
        return topics
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
                t = row.get("topik") or row.get("topic")
                if t:
                    topics.add(normalize_topic(t))
            except json.JSONDecodeError:
                continue
    return topics


def load_existing_ids(path: Path) -> set[int]:
    ids: set[int] = set()
    if not path.exists():
        return ids
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
                if "id" in row:
                    ids.add(int(row["id"]))
            except (json.JSONDecodeError, ValueError, TypeError):
                continue
    return ids


def sample_topics_for_prompt(forbidden: set[str], k: int = 40) -> list[str]:
    """Ambil sampel topik untuk prompt agar tidak memuat seluruh list."""
    pool = list(forbidden)
    if len(pool) <= k:
        return pool
    return random.sample(pool, k)


def build_topic_prompt(sample_existing: list[str]) -> str:
    listed = "\n".join(f"- {t}" for t in sample_existing[:80])
    return f"""Kamu membuat SATU judul topik BARU untuk dataset percakapan chatbot Bahasa Indonesia.

Persyaratan judul topik:
- Pendek dan jelas (3–12 kata), seperti judul chapter / tema diskusi.
- Harus BERBEDA dari semua topik yang mirip dengan daftar berikut (hindari sinonim atau tema yang sama).
- Bidang boleh apa saja: kehidupan sehari-hari, kerja, teknologi, kesehatan, finansial ringan, pendidikan, hobi, dll.
- Jangan gunakan tanda kutip atau emoji di judul.

Contoh topik yang SUDAH ADA (jangan meniru):
{listed}

Keluarkan HANYA satu objek JSON dengan kunci:
- "topik": string judul baru (Bahasa Indonesia)
- "ringkasan": tepat satu kalimat pendek (maksimal ~160 karakter) — konteks apa yang dibahas

Format JSON valid saja, tanpa markdown. Jangan tambahkan teks di luar JSON."""


def build_conversation_prompt(topik: str, ringkasan: str) -> str:
    return f"""Buat SATU dataset percakapan antara User dan Assistant untuk chatbot instruct.

SYSTEM (wajib sama persis untuk semua dataset — jangan diubah):
{SYSTEM_PROMPT}

META:
- Judul topik: {topik}
- Ringkasan semantis: {ringkasan}

Aturan struktur:
1) Output JSON dengan kunci "conversations": array objek {{ "role", "content" }}.
2) Jangan sertakan pesan system — mulai langsung dari pesan "user" pertama (system akan ditambahkan oleh pipeline).
3) Role HARUS bergantian: user, assistant, user, assistant, ... tanpa duplikasi berurutan.
4) Total pesan user+assistant setelah ini akan ditambah 1 system = antara {MIN_MESSAGES_TOTAL} dan {MAX_MESSAGES_TOTAL} inclusive.
   Artinya kamu harus menghasilkan persis {MIN_MESSAGES_TOTAL - 1} sampai {MAX_MESSAGES_TOTAL - 1} pesan (semuanya bergantian user/assistant).
   Ini berarti 10–15 pasang user–assistant setelah system.
5) Bahasa Indonesia utama; English hanya jika user meminta atau konteksnya English.
6) Dalam percakapan, sisipkan secara natural beberapa tugas yang cocok untuk encoder–decoder:
   - ringkas kutipan / percakapan / paragraf yang user tempel
   - jawab pertanyaan berdasarkan teks yang user berikan (QA grounded)
   - terjemahkan sebagian teks (ID↔EN)
   - ubah gaya atau parafrase (formal/informal)
   - ekstraksi poin / checklist dari teks
   Minimal 3 macam tugas berbeda muncul di timeline percakapan (boleh lebih).
7) Pada salah satu turn, user menempel kutipan artikel atau catatan (fiktif tapi realistis, 2–5 kalimat) dalam bubble percakapan.
8) Jawaban assistant harus membantu, tidak boilerplate berlebihan, tidak menyalin ringkasan META secara literal sebagai dialog.
9) Jangan sertakan markdown code fence di JSON; isi content boleh memakai newline dan bullet.

Keluarkan HANYA JSON valid satu objek dengan kunci "conversations"."""


def parse_json_loose(text: str) -> dict[str, Any]:
    """Parse JSON dari API; tangani kosong, markdown fence, atau teks dengan JSON di tengah."""
    raw = (text or "").strip()
    if not raw:
        raise ValueError("response API kosong (cek quota / model / reasoning output)")
    if raw.startswith("```"):
        raw = re.sub(r"^```(?:json)?\s*", "", raw)
        raw = re.sub(r"\s*```\s*$", "", raw).strip()
    try:
        parsed: Any = json.loads(raw)
    except json.JSONDecodeError:
        start = raw.find("{")
        end = raw.rfind("}")
        if start < 0 or end <= start:
            raise ValueError(f"bukan JSON objek (awalan: {raw[:120]!r})") from None
        parsed = json.loads(raw[start : end + 1])
    if not isinstance(parsed, dict):
        raise ValueError("JSON harus berupa objek di root")
    return parsed


def _json_string_value(raw: str, key: str) -> str | None:
    """Ambil nilai string untuk key; mendukung escape JSON minimal dan string tanpa penutup (respons API terpotong)."""
    m = re.search(rf'"{re.escape(key)}"\s*:\s*"', raw)
    if not m:
        return None
    i = m.end()
    out: list[str] = []
    while i < len(raw):
        c = raw[i]
        if c == "\\" and i + 1 < len(raw):
            esc = raw[i + 1]
            out.append(
                {
                    "n": "\n",
                    "r": "\r",
                    "t": "\t",
                    '"': '"',
                    "\\": "\\",
                }.get(esc, esc)
            )
            i += 2
            continue
        if c == '"':
            return "".join(out)
        out.append(c)
        i += 1
    return "".join(out)


def parse_topic_payload(text: str) -> dict[str, Any]:
    """Parse JSON topik; jika terpotong, coba ekstrak topik+ringkasan dari fragmen."""
    try:
        return parse_json_loose(text)
    except (json.JSONDecodeError, ValueError) as e:
        topik = _json_string_value(text, "topik")
        ring = _json_string_value(text, "ringkasan") or _json_string_value(text, "summary")
        if topik and ring:
            return {"topik": topik.strip(), "ringkasan": ring.strip()}
        raise e


def parse_topic_strings(data: dict[str, Any]) -> tuple[str, str]:
    t = TopicGenResponse.model_validate(data)
    return t.topik.strip(), t.ringkasan.strip()


def parse_conversation_turns(data: dict[str, Any]) -> list[dict[str, Any]]:
    body = ConversationGenBody.model_validate(data)
    return [turn.model_dump() for turn in body.conversations]


def call_deepseek_raw(
    client: OpenAI,
    user_prompt: str,
    max_tokens: int = 8192,
    temperature: float = CONV_TEMPERATURE,
) -> tuple[str, int | None]:
    backoff = [0.0, 2.0, 6.0, 14.0]
    last_exc: BaseException | None = None
    for delay in backoff:
        if delay:
            time.sleep(delay)
        try:
            completion = client.chat.completions.create(
                model=MODEL_CHAT,
                messages=[{"role": "user", "content": user_prompt}],
                response_format={"type": "json_object"},
                temperature=temperature,
                max_tokens=max_tokens,
                extra_body={"thinking": {"type": "enabled"}},
                reasoning_effort="low",
            )
            break
        except RateLimitError as e:
            last_exc = e
            continue
        except (APIConnectionError, APITimeoutError, OSError) as e:
            last_exc = e
            continue
    else:
        raise last_exc if last_exc else RuntimeError("API gagal tanpa exception")

    choice = completion.choices[0]
    msg = choice.message
    text = (msg.content or "").strip()
    if not text:
        fr = getattr(choice, "finish_reason", None)
        for attr in ("parsed", "reasoning_content"):
            extra = getattr(msg, attr, None)
            if isinstance(extra, str) and extra.strip():
                text = extra.strip()
                break
        if not text:
            raise ValueError(
                f"content kosong (finish_reason={fr}, max_tokens={max_tokens}); "
                "untuk tahap topik set DEEPSEEK_TOPIC_MAX_TOKENS misalnya 8192"
            )
    usage = completion.usage
    tokens = usage.completion_tokens if usage else None
    return text, tokens


def call_deepseek_json(client: OpenAI, user_prompt: str, max_tokens: int = 8192) -> tuple[dict[str, Any], int | None]:
    """Kompatibilitas: parse JSON mentah menjadi dict."""
    text, tokens = call_deepseek_raw(client, user_prompt, max_tokens=max_tokens)
    return parse_json_loose(text), tokens


def validate_conversations(raw: list[dict[str, Any]]) -> tuple[bool, str]:
    """Validasi setelah inject_system: system + selang-seling user/assistant, panjang rentang."""
    if not raw:
        return False, "kosong"
    if raw[0].get("role") != "system":
        return False, "pertama bukan system"
    if (raw[0].get("content") or "").strip() != SYSTEM_PROMPT.strip():
        return False, "system prompt tidak sama dengan konstanta skrip"
    for i in range(len(raw) - 1):
        if raw[i].get("role") == raw[i + 1].get("role"):
            return False, f"duplikasi role berurutan di indeks {i}"
    for i in range(1, len(raw)):
        want = "user" if i % 2 == 1 else "assistant"
        if raw[i].get("role") != want:
            return False, f"indeks {i} harus {want}, dapat {raw[i].get('role')}"
    if len(raw) < MIN_MESSAGES_TOTAL or len(raw) > MAX_MESSAGES_TOTAL:
        return False, f"jumlah pesan {len(raw)} di luar [{MIN_MESSAGES_TOTAL},{MAX_MESSAGES_TOTAL}]"
    return True, "ok"


def inject_system(conversations: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Paksa system prompt dataset agar konsisten; hapus system model jika ada."""
    body = [dict(x) for x in conversations if x.get("role") != "system"]
    return [{"role": "system", "content": SYSTEM_PROMPT}, *body]


def generate_unique_topic(
    client: OpenAI,
    forbidden_norm: set[str],
    rng: random.Random,
) -> tuple[str, str] | tuple[None, None]:
    for attempt in range(MAX_RETRIES_TOPIC):
        sample = sample_topics_for_prompt(forbidden_norm, k=min(60, len(forbidden_norm)))
        if not sample:
            sample = ["(belum ada topik di dataset — buat topik bebas yang spesifik)"]
        prompt = build_topic_prompt(sample)
        try:
            topic_budget = min(TOPIC_MAX_TOKENS_BASE * (2**attempt), 32_768)
            text, _ = call_deepseek_raw(
                client, prompt, max_tokens=topic_budget, temperature=TOPIC_TEMPERATURE
            )
            data = parse_topic_payload(text)
            topik, ringkasan = parse_topic_strings(data)
            if not topik or not ringkasan:
                continue
            nt = normalize_topic(topik)
            if nt in forbidden_norm:
                continue
            if any(nt in other or other in nt for other in forbidden_norm if len(other) > 10 and len(nt) > 10):
                continue
            return topik, ringkasan
        except Exception as e:
            print(f"    [topic retry {attempt + 1}] {e}")
            time.sleep(1.5)
    return None, None


def generate_conversation(
    client: OpenAI,
    topik: str,
    ringkasan: str,
) -> tuple[list[dict[str, Any]], int | None]:
    prompt = build_conversation_prompt(topik, ringkasan)
    for attempt in range(MAX_RETRIES_CONV):
        try:
            conv_budget = min(CONV_MAX_TOKENS * (2**attempt), 65_536)
            text, tok = call_deepseek_raw(
                client, prompt, max_tokens=conv_budget, temperature=CONV_TEMPERATURE
            )
            data = parse_json_loose(text)
            turns = parse_conversation_turns(data)
            convs = inject_system(turns)
            ok, reason = validate_conversations(convs)
            if ok:
                return convs, tok
            print(f"    [conv invalid] {reason}, retry {attempt + 1}")
        except Exception as e:
            print(f"    [conv retry {attempt + 1}] {e}")
        time.sleep(2.0)
    return [], None


def append_topics_log(path: Path, topik: str, ringkasan: str) -> None:
    rec = {"topik": topik, "ringkasan": ringkasan}
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(rec, ensure_ascii=False) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate dataset rows via DeepSeek")
    parser.add_argument(
        "--base-dataset",
        type=Path,
        default=DEFAULT_BASE_DATASET,
        help="JSONL sumber untuk membaca topik yang sudah ada",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
        help="JSONL keluaran (append)",
    )
    parser.add_argument(
        "--topics-log",
        type=Path,
        default=DEFAULT_TOPICS_LOG,
        help="Log semua topik yang berhasil dibuat",
    )
    parser.add_argument("--target", type=int, default=8970, help="Jumlah baris baru yang ingin ditambahkan")
    parser.add_argument("--start-id", type=int, default=1031, help="ID pertama untuk baris baru")
    parser.add_argument(
        "--max-id",
        type=int,
        default=None,
        metavar="N",
        help="Berhenti sebelum menulis baris dengan id > N (opsional; cocok untuk batas id global mis. 10000)",
    )
    parser.add_argument("--sleep", type=float, default=SLEEP_DEFAULT)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Hanya muat topik & cetak statistik, tanpa API",
    )
    args = parser.parse_args()

    api_key = os.environ.get("DEEPSEEK_API_KEY")
    if not api_key and not args.dry_run:
        print("Set environment variable DEEPSEEK_API_KEY", file=sys.stderr)
        sys.exit(1)

    rng = random.Random(args.seed)

    forbidden = load_existing_topics(args.base_dataset)
    print(f"[INFO] Topik unik dari base dataset: {len(forbidden)} ({args.base_dataset})")
    print(f"[INFO] DeepSeek API: base_url={DEEPSEEK_BASE_URL} model={MODEL_CHAT}")
    print(
        f"[INFO] max_tokens: topik base={TOPIC_MAX_TOKENS_BASE} (per-retry ×2 hingga 32k), "
        f"percakapan base={CONV_MAX_TOKENS} (per-retry ×2 hingga 65k); "
        f"suhu topik={TOPIC_TEMPERATURE}, percakapan={CONV_TEMPERATURE}"
    )
    print("[INFO] Validasi respons: pydantic")

    # Gabungkan topik dari output & log agar tidak bentrok saat resume
    forbidden |= load_existing_topics(args.output)
    if args.topics_log.exists():
        with args.topics_log.open("r", encoding="utf-8") as tf:
            for line in tf:
                line = line.strip()
                if not line:
                    continue
                try:
                    forbidden.add(normalize_topic(json.loads(line).get("topik", "")))
                except json.JSONDecodeError:
                    pass

    existing_ids = load_existing_ids(args.output)
    print(f"[INFO] ID sudah ada di output: {len(existing_ids)} ({args.output})")
    if args.max_id is not None:
        print(f"[INFO] Batas id (--max-id): tidak akan menulis baris dengan id > {args.max_id}")

    if args.dry_run:
        print("[DRY-RUN] Selesai.")
        return

    client = OpenAI(api_key=api_key, base_url=DEEPSEEK_BASE_URL)

    next_id = args.start_id
    produced = 0
    while produced < args.target:
        while next_id in existing_ids:
            next_id += 1

        if args.max_id is not None and next_id > args.max_id:
            print(
                f"\n[INFO] Berhenti: id berikutnya ({next_id}) melewati --max-id {args.max_id} "
                f"({produced} baris baru dalam sesi ini)."
            )
            break

        topik, ringkasan = generate_unique_topic(client, forbidden, rng)
        if not topik or not ringkasan:
            print("[WARN] Gagal mendapatkan topik unik setelah percobaan; tunggu dan lanjut...")
            time.sleep(5)
            continue

        nt = normalize_topic(topik)
        if nt in forbidden:
            continue

        print(f"\n[{produced + 1}/{args.target}] id={next_id} | {topik}")
        convs, ctoks = generate_conversation(client, topik, ringkasan)
        if not convs:
            print("  ✗ Percakapan gagal validasi.")
            time.sleep(args.sleep)
            continue

        entry = {
            "id": next_id,
            "topik": topik,
            "num_turns": len(convs),
            "tokens": ctoks if ctoks is not None else 0,
            "conversations": convs,
        }

        with args.output.open("a", encoding="utf-8") as out:
            out.write(json.dumps(entry, ensure_ascii=False) + "\n")

        append_topics_log(args.topics_log, topik, ringkasan)
        forbidden.add(nt)
        existing_ids.add(next_id)
        produced += 1
        next_id += 1
        print(f"  ✓ {len(convs)} pesan (num_turns={len(convs)}), ~tokens={ctoks}")
        time.sleep(args.sleep)

    print(f"\n[SELESAI] {produced} percakapan baru → {args.output}")


if __name__ == "__main__":
    main()
