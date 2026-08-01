"""
Generate Synthetic Conversational Dataset "EXTRA" — 2026-07-31
===============================================================
Generator percakapan sintetis multi-sumber (pengganti
`generate_conv_from_awesome.py` yang dihapus) + integrasi hasil deep research
katalog `awesome-indonesian-llm-dataset`. SEMUA fitur lama dipertahankan:
schemas Pydantic + validators, prefix token Gemma 2, multi-source HF loader,
sampling acak baris asli, prompt builder, retries, concurrency, output JSONL
dengan lampiran gambar.

Fitur `datasets` 5.0.x yang dimanfaatkan (terverifikasi di env unsloth-env):
  - STREAMING untuk dataset besar (Cendol 12,8jt / LFQA 226K / KORIKA 1,6GB /
    SEA-VL 3,7GB): `load_dataset(..., streaming=True)` + shuffle multi-shard
    (`shuffle(seed, buffer_size, max_buffer_input_shards)`) — TANPA unduh penuh
    ke RAM/disk. Gambar vision dibangun per row yang di-sample lalu diekstrak lokal.
  - Composed split `train+test` untuk menggabungkan split IndoRad-VQA
  - `filter()` pada Dataset/IterableDataset untuk menyaring SEA-VL native_lang=ind
  - (parquet columnar read via pyarrow untuk CVQA metadata tanpa gambar ~5GB)
  - Flag `--quick`: hanya muat dataset ringan (uji cepat tanpa download besar)

DAFTAR SUMBER (100% dataset asli HuggingFace, tanpa dummy/hardcode):

  Text / NLU / QA / Summarization / Paraphrase (lama, dipertahankan):
    1. FineWeb-Edu-25K             -> irfanfadhullah/FineWeb-Edu-25K
    2. OpenWebText-Indonesia-10k   -> irfanfadhullah/OpenWebText-Indonesia-10k
    3. Indonesian Simple Summaries -> irfanfadhullah/indonesian-simple-summaries
    4. Indonesian Wikipedia        -> indonesian-nlp/wikipedia-id (subset 2%)
    5. IndoMMLU                    -> indolem/IndoMMLU
    6. IndoCareer                  -> indolem/IndoCareer
    7. IndoCulture                 -> indolem/IndoCulture
    8. QQPR-Triplets-ID            -> robinsyihab/QQPR-triplets-ID
    9. Cendol Collection v2        -> indonlp/cendol_collection_v2 (subset 1%)

  Text / QA (BARU dari deep research):
   10. LFQA-ID (226K long-form QA)   -> indonesian-nlp/lfqa_id
   11. CVQA Subset Indonesia (VQA budaya; metadata-only via pyarrow, tanpa
       unduh gambar ~5GB)            -> afaji/cvqa

  Vision / VLM (lama, dipertahankan):
   12. KTP VLM Instruct Dataset (780 rows OCR KTP) -> danielsyahputra/ktp-vlm-instruct-dataset

  Vision / VLM (BARU dari deep research):
   13. KORIKA SEA-VL Crowdsourcing ID (3.096 rows gambar + caption Bahasa
       Indonesia asli + skor kualitas annotator) -> KORIKA-AI/sea-vl_crowdsourcing_id
   14. SEACrowd SEA-VL Crowdsourcing (7.010 rows, difilter native_lang=ind)
       -> SEACrowd/sea-vl_crowdsourcing
   15. IndoRad-VQA (2.244 rows VQA radiologi Bahasa Indonesia, train+test)
       -> Lab-IS/IndoRad-VQA

  Catatan gambar:
   - KTP            : URL resolve HF (pola ktp_row_{idx}.jpg), dipertahankan.
   - KORIKA/SEA-VL/IndoRad : gambar PIL di-ekstrak ke data/multimodal/images/
     (path lokal dipakai sebagai referensi di kolom `images` output).
   - CVQA           : metadata soal/opsi/jawaban saja (kategori text QA).

Penggunaan:
  python scripts/dataset/generate_conv_extra_20260731.py --mode text --target 2000 --model step-3.7-flash
  python scripts/dataset/generate_conv_extra_20260731.py --mode vision --target 1000 --model step-3.7-flash
  python scripts/dataset/generate_conv_extra_20260731.py --mode mixed --target 3000 --model step-3.7-flash
  # Kuota per kategori (mengalahkan --mode/--target):
  python scripts/dataset/generate_conv_extra_20260731.py --text-target 2000 --vision-target 1000 --model step-3.7-flash
  python scripts/dataset/generate_conv_extra_20260731.py --quick --text-target 2 --vision-target 2  # uji cepat
  # Exclude row/id yang sudah dipakai: otomatis membaca file output yang ada (resume),
  # bisa ditambah file JSONL lain via --exclude-file:
  python scripts/dataset/generate_conv_extra_20260731.py --quick --limit 2 --exclude-file data/synthetic/run_sebelumnya.jsonl
  # Output default APPEND (menumpuk ke file, id global lanjut) — gunakan --overwrite untuk mulai dari nol:
  python scripts/dataset/generate_conv_extra_20260731.py --overwrite --text-target 2000 --vision-target 1000

Konfigurasi default (sesuai batas API):
  --model step-3.7-flash | --concurrency 5 | --rpm 10 (request/menit) | --tpm 5000000 (token/menit)
"""

import argparse
import asyncio
import io
import json
import os
import random
import re
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional, Tuple, cast

import datasets
from PIL import Image as PILImage
from pydantic import BaseModel, Field, field_validator
from pydantic_ai import Agent, ModelSettings
from pydantic_ai.models.openai import OpenAIChatModel
from pydantic_ai.providers.openai import OpenAIProvider

# Load .env dari root project jika ada
SCRIPT_DIR = Path(__file__).resolve().parent
ROOT_DIR = SCRIPT_DIR.parent.parent
DATA_DIR = ROOT_DIR / "data"
VISION_IMAGE_DIR = DATA_DIR / "multimodal" / "images"

try:
    from dotenv import load_dotenv
    load_dotenv(ROOT_DIR / ".env")
except ImportError:
    pass

# Default System Prompt Gemma
SYSTEM_PROMPT = (
    "Kamu adalah Gemma, asisten AI cerdas berbahasa Indonesia yang dirancang untuk membantu "
    "pengguna dalam berbagai tugas pemrosesan bahasa (NLP), pemahaman visual, maupun percakapan "
    "sehari-hari. Berikan respons yang akurat, terstruktur, ramah, dan natural."
)

# Definisi Prefix Token Murni Gemma 2
VALID_PREFIXES = Literal[
    "<unused1>",  # SUMMARIZE
    "<unused2>",  # TRANSLATE
    "<unused3>",  # NER
    "<unused4>",  # QA
    "<unused5>",  # PARAPHRASE
    "<unused6>",  # GENERAL_CHAT
]

# ─── Robust Pydantic Output Schemas ───────────────────────────────────────────

class TurnMessage(BaseModel):
    role: Literal["user", "assistant"] = Field(description="Role turn pesan: harus 'user' atau 'assistant'")
    prefixes: List[VALID_PREFIXES] = Field(
        default=[],
        description="Khusus role assistant: WAJIB diisi 1 hingga 3 token prefix murni, contoh: ['<unused1>'] atau ['<unused1>', '<unused4>']. Khusus role user: isi list kosong []"
    )
    content: str = Field(description="Isi teks pesan murni TANPA menyertakan token atau teks nama task seperti 'SUMMARIZE' atau 'NER'")

class ConversationOutput(BaseModel):
    conversations: List[TurnMessage] = Field(
        description="Daftar pasangan pesan dialog antara user dan assistant secara selang-seling (user, assistant, user, assistant...)"
    )

    @field_validator("conversations")
    @classmethod
    def check_turns(cls, convs: List[TurnMessage]) -> List[TurnMessage]:
        if len(convs) % 2 != 0:
            raise ValueError("Jumlah pesan harus genap (pasangan user-assistant).")
        for i, m in enumerate(convs):
            expected_role = "user" if i % 2 == 0 else "assistant"
            if m.role.strip() != expected_role:
                raise ValueError(f"Pesan urutan [{i}] harus ber-role '{expected_role}', bukan '{m.role}'")
            if m.role == "assistant":
                # Normalisasi: bersihkan spasi per elemen, buang duplikat token SAMA
                # (pertahankan urutan). Multi-task tetap boleh: ['<unused1>', '<unused2>'].
                if not m.prefixes:
                    m.prefixes = ["<unused4>"]
                else:
                    cleaned = []
                    for p in m.prefixes:
                        p = re.sub(r"\s+", "", str(p))
                        if p and p not in cleaned:
                            cleaned.append(p)
                    m.prefixes = cast(List[VALID_PREFIXES], cleaned) if cleaned else ["<unused4>"]
        return convs


# ─── Text Builder Helpers ──────────────────────────────────────────────────────
# Helper functions untuk mengekstrak teks dari dataset dengan struktur non-standar
# (misal QA datasets dengan kolom terpisah: question + options + answer)

def _normalize_options(options: Any) -> List[str]:
    """Normalisasi kolom options yang bisa berupa list atau string repr."""
    if isinstance(options, list):
        return [str(opt) for opt in options if opt]
    if isinstance(options, str):
        try:
            parsed = eval(options)
            if isinstance(parsed, list):
                return [str(opt) for opt in parsed if opt]
        except Exception:
            pass
        return [options] if options else []
    return []


def _build_indommlu_text(row: Dict[str, Any]) -> str:
    """Bangun teks konteks dari IndoMMLU (educational QA)."""
    question = row.get("context", row.get("question", ""))
    options = _normalize_options(row.get("options", []))
    answer = row.get("answer", "")
    options_text = "\n".join(f"{chr(65 + i)}. {opt}" for i, opt in enumerate(options))
    return f"Soal: {question}\n\nPilihan:\n{options_text}\n\nJawaban: {answer}"


def _build_indocareer_text(row: Dict[str, Any]) -> str:
    """Bangun teks konteks dari IndoCareer (professional QA)."""
    question = row.get("Question", "")
    options = _normalize_options([
        row.get("Option A", ""),
        row.get("Option B", ""),
        row.get("Option C", ""),
        row.get("Option D", ""),
        row.get("Option E", ""),
    ])
    answer = row.get("Answer Key", "")
    options_text = "\n".join(f"{chr(65 + i)}. {opt}" for i, opt in enumerate(options))
    return f"Soal: {question}\n\nPilihan:\n{options_text}\n\nJawaban: {answer}"


def _build_indoculture_text(row: Dict[str, Any]) -> str:
    """Bangun teks konteks dari IndoCulture (cultural QA)."""
    question = row.get("context", row.get("question", ""))
    options = _normalize_options(row.get("options", []))
    answer = row.get("answer", "")
    options_text = "\n".join(f"{chr(65 + i)}. {opt}" for i, opt in enumerate(options))
    return f"Soal: {question}\n\nPilihan:\n{options_text}\n\nJawaban: {answer}"


def _build_simple_summaries_text(row: Dict[str, Any]) -> str:
    """Bangun teks konteks dari Indonesian Simple Summaries."""
    original_text = row.get("translated_text", row.get("original_text", ""))
    summary = row.get("translated_summary", row.get("original_summary", ""))
    return f"Teks Asli:\n{original_text}\n\nRingkasan:\n{summary}"


def _build_qqpr_text(row: Dict[str, Any]) -> str:
    """Bangun teks konteks dari QQPR-triplets-ID (paraphrase/similarity)."""
    query = row.get("query", "")
    pos = _normalize_options(row.get("pos", []))[:3]
    neg = _normalize_options(row.get("neg", []))[:3]
    pos_text = "\n".join(f"- {p}" for p in pos)
    neg_text = "\n".join(f"- {n}" for n in neg)
    return f"Pertanyaan: {query}\n\nParaprase Positif:\n{pos_text}\n\nParaprase Negatif:\n{neg_text}"


def _build_wikipedia_text(row: Dict[str, Any]) -> str:
    """Bangun teks konteks dari Indonesian Wikipedia."""
    title = row.get("title", "")
    text = row.get("text", "")
    if len(text) > 1500:
        text = text[:1500] + "..."
    return f"Judul: {title}\n\n{text}"


def _build_cendol_text(row: Dict[str, Any]) -> str:
    """Bangun teks konteks dari Cendol Collection v2 (instruction-tuning)."""
    input_text = row.get("input", "")
    output_text = row.get("output", "")
    return f"Instruksi: {input_text}\n\nRespon: {output_text}"


# ─── Text Builder Helpers BARU (deep research 2026-07-31) ─────────────────────

def _build_indommlu_csv_text(row: Dict[str, Any]) -> str:
    """Bangun teks konteks dari IndoMMLU CSV asli (kolom: soal/jawaban/kunci/subject)."""
    soal = row.get("soal", "")
    kunci = row.get("kunci", "")
    subject = row.get("subject", "")
    jawaban = row.get("jawaban", "")
    opts = [ln.strip() for ln in str(jawaban).splitlines() if re.match(r"^[A-E]\.", ln.strip())]
    options_text = "\n".join(opts)
    return f"Soal: {soal}\n\nPilihan:\n{options_text}\n\nJawaban: {kunci}\nMata Pelajaran: {subject}"


def _build_lfqa_text(row: Dict[str, Any]) -> str:
    """Bangun teks konteks dari LFQA-ID (long-form QA ala ELI5)."""
    title = row.get("title", "")
    selftext = row.get("selftext", "")
    document = row.get("document", "")
    answers = row.get("answers", "")
    if isinstance(answers, list):
        texts = []
        for a in answers:
            if isinstance(a, dict):
                texts.append(str(a.get("text", a.get("answer", ""))).strip())
            else:
                texts.append(str(a).strip())
        ans = " | ".join(t for t in texts if t)[:1200]
    else:
        ans = str(answers)[:1200]
    doc = str(document)[:1000]
    return f"Pertanyaan: {title}\n{selftext}\n\nKonteks:\n{doc}\n\nJawaban Terbaik:\n{ans}"


def _build_cvqa_id_text(row: Dict[str, Any]) -> str:
    """Bangun teks konteks dari CVQA subset Indonesia (metadata-only)."""
    question = row.get("Question", "")
    options = _normalize_options(row.get("Options", []))
    label = row.get("Label", "")
    category = row.get("Category", "")
    try:
        ans_letter = chr(65 + int(label)) if str(label).isdigit() else str(label)
    except Exception:
        ans_letter = str(label)
    options_text = "\n".join(f"{chr(65 + i)}. {opt}" for i, opt in enumerate(options))
    return f"Soal Visual (VQA): {question}\n\nPilihan:\n{options_text}\n\nJawaban: {ans_letter}\nKategori: {category}"


def _build_sea_vl_text(row: Dict[str, Any]) -> str:
    """Bangun teks konteks dari KORIKA/SEACrowd SEA-VL crowdsourcing (caption BI)."""
    cap_id = row.get("caption_native_lang", "")
    cap_en = row.get("caption", "")
    loc = row.get("culture_relevant_loc", "")
    lang = row.get("native_lang", "")
    return f"Deskripsi (Bahasa Indonesia): {cap_id}\nDeskripsi (English): {cap_en}\nLokasi Budaya: {loc}\nBahasa: {lang}"


def _build_indorad_text(row: Dict[str, Any]) -> str:
    """Bangun teks konteks dari IndoRad-VQA (VQA radiologi Bahasa Indonesia)."""
    question = row.get("question_indonesian", "") or row.get("question", "")
    answer = row.get("answer_indonesian", "") or row.get("answer", "")
    answer_type = row.get("answer_type", "")
    organ = row.get("image_organ", "")
    return f"Pertanyaan Medis (Radiologi): {question}\nJawaban: {answer}\nTipe Jawaban: {answer_type}"


def _build_ktp_text(row: Dict[str, Any]) -> str:
    """Bangun teks konteks dari KTP VLM Instruct Dataset (OCR identitas)."""
    inst = row.get("instruction", "")
    out = row.get("output", "")
    return f"Instruksi Asli: {inst}\nOutput Asli KTP: {out}"


# ─── Image Helpers (dataset vision yang menyimpan PIL/bytes) ──────────────────

def _pil_image_from_value(value: Any) -> Optional[PILImage.Image]:
    """Konversi nilai kolom 'image' (PIL, dict bytes, bytes mentah) menjadi PIL.Image."""
    if value is None:
        return None
    if isinstance(value, PILImage.Image):
        return value
    if isinstance(value, dict) and value.get("bytes"):
        try:
            return PILImage.open(io.BytesIO(value["bytes"])).convert("RGB")
        except Exception:
            return None
    if isinstance(value, (bytes, bytearray)):
        try:
            return PILImage.open(io.BytesIO(bytes(value))).convert("RGB")
        except Exception:
            return None
    return None


def _save_vision_image(value: Any, fname_base: str) -> Optional[str]:
    """Simpan gambar vision (PIL/bytes) ke data/multimodal/images/, kembalikan path lokal."""
    pil = _pil_image_from_value(value)
    if pil is None:
        return None
    VISION_IMAGE_DIR.mkdir(parents=True, exist_ok=True)
    path = VISION_IMAGE_DIR / f"{fname_base}.jpg"
    if not path.exists():
        try:
            pil.save(path, "JPEG", quality=90)
        except Exception:
            return None
    return str(path)


def _make_image_saver(source_key: str):
    """Factory image_builder untuk dataset yang gambarnya PIL/bytes (disimpan lokal).

    - row_idx >= 0 (dataset in-memory): nama file {source_key}_{row_idx}.jpg
    - row_idx < 0 (streaming, row tidak punya index): pakai kolom id/qid sebagai nama.
    """
    def saver(row: Dict[str, Any], row_idx: int) -> Optional[str]:
        if row_idx >= 0:
            base = f"{source_key}_{row_idx}"
        else:
            ident = re.sub(r"[^\w-]+", "_", str(row.get("id", row.get("qid", "stream"))))[:50]
            base = f"{source_key}_{ident}"
        return _save_vision_image(row.get("image"), base)
    return saver


def _ktp_image_url(row: Dict[str, Any], row_idx: int) -> str:
    """Image_builder KTP: URL resolve HF (pola asli ktp_row_{idx}.jpg, dipertahankan)."""
    return f"https://huggingface.co/datasets/danielsyahputra/ktp-vlm-instruct-dataset/resolve/main/images/ktp_row_{row_idx}.jpg"


# ─── Exclude Helper (row/id yang sudah dipakai di JSONL sebelumnya) ─────────────

SOURCE_ROW_RE = re.compile(r"^(?P<name>.+), Row #(?P<idx>\d+)$")
SOURCE_ID_RE = re.compile(r"^(?P<name>.+), id=(?P<id>.+)$")


def load_excluded_sources(paths: List[Path]) -> Tuple[Dict[str, set], Dict[str, set]]:
    """Baca file JSONL output → kumpulkan row (in-memory) / id (streaming) yang sudah dipakai.

    Format `source` di output:
      - in-memory : "{source_name}, Row #{row_idx}"
      - streaming : "{source_name}, id={nilai kolom id_key}"
    """
    excluded_rows: Dict[str, set] = {}
    excluded_ids: Dict[str, set] = {}
    for p in paths:
        path = Path(p)
        if not path.exists():
            print(f"      ⚠️ File exclude tidak ditemukan: {path}")
            continue
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    item = json.loads(line)
                except json.JSONDecodeError:
                    continue
                src = str(item.get("source", ""))
                m = SOURCE_ROW_RE.match(src)
                if m:
                    excluded_rows.setdefault(m.group("name"), set()).add(int(m.group("idx")))
                    continue
                m = SOURCE_ID_RE.match(src)
                if m:
                    excluded_ids.setdefault(m.group("name"), set()).add(m.group("id"))
    return excluded_rows, excluded_ids


# ─── Helper Pure Real Multi-Source HuggingFace Dataset Loader ─────────────────

class PureHuggingFaceDataLoader:
    def __init__(self, quick: bool = False):
        self.text_datasets = {}
        self.vision_datasets = {}
        self.quick = quick
        self.excluded_rows: Dict[str, set] = {}
        self.excluded_ids: Dict[str, set] = {}

    def apply_exclusions(self) -> int:
        """Hapus row/id yang sudah dipakai di JSONL sebelumnya dari pool sampling.

        In-memory : ds.select(index yang belum dipakai)
        Streaming : ds.filter(id_key != id yang sudah dipakai)
        """
        total_removed = 0
        for pool in (self.text_datasets, self.vision_datasets):
            for key, meta in pool.items():
                name = meta["source_name"]
                ex_rows = self.excluded_rows.get(name, set())
                ex_ids = self.excluded_ids.get(name, set())
                if not ex_rows and not ex_ids:
                    continue
                ds = meta["dataset"]
                if meta.get("streaming"):
                    id_key = meta.get("id_key")
                    if ex_ids and id_key:
                        ids = ex_ids
                        meta["dataset"] = ds.filter(
                            lambda r, id_key=id_key, ids=ids: str(r.get(id_key, "")) not in ids
                        )
                        total_removed += len(ids)
                else:
                    keep = [i for i in range(len(ds)) if i not in ex_rows]
                    removed = len(ds) - len(keep)
                    if removed:
                        meta["dataset"] = ds.select(keep)
                        total_removed += removed
        return total_removed

    @staticmethod
    def _load_cvqa_id_metadata() -> Optional[datasets.Dataset]:
        """CVQA subset Indonesia via pyarrow partial-column read (TANPA unduh gambar ~5GB).

        Parquet bersifat columnar: membaca hanya kolom teks/metadata tidak
        mentransfer kolom `image` (bytes gambar ~5GB). Berfungsi karena
        `datasets.load_dataset(columns=...)` tidak didukung di datasets 5.0.1.
        """
        import json as _json
        import urllib.request as _req

        import fsspec
        import pyarrow as pa
        import pyarrow.compute as pc
        import pyarrow.parquet as pq

        try:
            url = "https://datasets-server.huggingface.co/parquet?dataset=afaji/cvqa"
            with _req.urlopen(_req.Request(url, headers={"User-Agent": "generate-extra"}), timeout=60) as resp:
                payload = _json.loads(resp.read().decode("utf-8"))
            file_urls = [f.get("url") for f in payload.get("parquet_files", []) if f.get("url")]
            if not file_urls:
                print("      ⚠️ CVQA: daftar parquet kosong, dilewati.")
                return None

            cols = ["ID", "Subset", "Question", "Options",
                    "Translated Question", "Translated Options", "Label", "Category"]
            # fsspec http filesystem memungkinkan pyarrow membaca parquet remote
            # sekaligus tetap memanfaatkan sifat columnar (hanya kolom teks yang ditransfer)
            fs = fsspec.filesystem("https")
            tables = []
            for fu in file_urls:
                # Baca hanya kolom teks/metadata (columnar parquet) — kolom image TIDAK ditransfer
                tables.append(pq.read_table(fu, filesystem=fs, columns=cols))  # type: ignore[arg-type]
            full = pa.concat_tables(tables)
            mask = pc.match_substring(full["Subset"], "Indonesian")  # type: ignore
            id_table = full.filter(mask)
            if id_table.num_rows == 0:
                print("      ⚠️ CVQA: tidak ada subset Indonesia ditemukan, dilewati.")
                return None
            # Dataset.from_arrow dihapus di datasets 5.x → pakai from_dict (API stabil)
            col_data = {c: id_table[c].to_pylist() for c in id_table.column_names}
            return datasets.Dataset.from_dict(col_data)
        except Exception as e:
            print(f"      ⚠️ CVQA gagal dimuat (metadata-only): {e}")
            return None

    def load_all(self):
        print("📦 Loading multi-source pure real HuggingFace datasets into memory (EXTRA 2026-07-31)...")
        VISION_IMAGE_DIR.mkdir(parents=True, exist_ok=True)

        # 1. FineWeb-Edu-25K
        try:
            print("  --> Loading irfanfadhullah/FineWeb-Edu-25K...")
            ds_fw = datasets.load_dataset("irfanfadhullah/FineWeb-Edu-25K", split="train")
            self.text_datasets["FineWeb-Edu-25K"] = {
                "dataset": ds_fw,
                "text_key": "text_indonesian",
                "task_type": "Educational Concept Explanation & Science QA",
                "tuning_instruction": "Fokus pada gaya penjelasan akademis/edukatif yang mudah dipahami, penguraian konsep sains/teknologi secara terstruktur, analogi, dan penjelasan bertahap.",
                "source_name": "FineWeb-Edu-25K (HuggingFace: irfanfadhullah/FineWeb-Edu-25K)"
            }
            print(f"      ✅ Loaded {len(ds_fw):,} real rows!")
        except Exception as e:
            print(f"      ❌ Failed loading FineWeb-Edu-25K: {e}")

        # 2. OpenWebText-Indonesia-10k
        try:
            print("  --> Loading irfanfadhullah/OpenWebText-Indonesia-10k...")
            ds_ow = datasets.load_dataset("irfanfadhullah/OpenWebText-Indonesia-10k", split="train")
            self.text_datasets["OpenWebText-10k"] = {
                "dataset": ds_ow,
                "text_key": "translated_text",
                "task_type": "News Summarization & Public Policy Discussion",
                "tuning_instruction": "Fokus pada perangkuman berita, diskusi isu sosial/ekonomi, analisis kebijakan publik, serta evaluasi dampak bagi masyarakat.",
                "source_name": "OpenWebText-Indonesia-10k (HuggingFace: irfanfadhullah/OpenWebText-Indonesia-10k)"
            }
            print(f"      ✅ Loaded {len(ds_ow):,} real rows!")
        except Exception as e:
            print(f"      ❌ Failed loading OpenWebText-10k: {e}")

        # 3. Indonesian Simple Summaries
        try:
            print("  --> Loading irfanfadhullah/indonesian-simple-summaries...")
            ds_sum = datasets.load_dataset("irfanfadhullah/indonesian-simple-summaries", split="train")
            self.text_datasets["Indonesian-Simple-Summaries"] = {
                "dataset": ds_sum,
                "text_builder": _build_simple_summaries_text,
                "task_type": "News Summarization & Text Simplification",
                "tuning_instruction": "Fokus pada perangkuman berita, penyederhanaan teks, dan pembelajaran konseptual yang mudah dipahami.",
                "source_name": "Indonesian Simple Summaries (HuggingFace: irfanfadhullah/indonesian-simple-summaries)"
            }
            print(f"      ✅ Loaded {len(ds_sum):,} real rows!")
        except Exception as e:
            print(f"      ❌ Failed loading Indonesian Simple Summaries: {e}")

        # 4. IndoMMLU (via CSV asli di repo — loading script tidak didukung datasets 5.x)
        try:
            print("  --> Loading indolem/IndoMMLU (CSV asli)...")
            mmlu_csv = "https://huggingface.co/datasets/indolem/IndoMMLU/resolve/main/IndoMMLU.csv"
            ds_mmlu = datasets.load_dataset("csv", data_files=mmlu_csv, split="train")
            self.text_datasets["IndoMMLU"] = {
                "dataset": ds_mmlu,
                "text_builder": _build_indommlu_csv_text,
                "task_type": "Educational Concept Explanation & Science QA",
                "tuning_instruction": "Fokus pada gaya penjelasan akademis/edukatif yang mudah dipahami, penguraian konsep sains/teknologi secara terstruktur, analogi, dan penjelasan bertahap.",
                "source_name": "IndoMMLU (HuggingFace: indolem/IndoMMLU, CSV asli)"
            }
            print(f"      ✅ Loaded {len(ds_mmlu):,} real rows!")
        except Exception as e:
            print(f"      ❌ Failed loading IndoMMLU: {e}")

        # 5. IndoCareer (config 'all', split 'test' — dataset tidak punya split train)
        try:
            print("  --> Loading indolem/IndoCareer (all/test)...")
            ds_career = datasets.load_dataset("indolem/IndoCareer", "all", split="test")
            self.text_datasets["IndoCareer"] = {
                "dataset": ds_career,
                "text_builder": _build_indocareer_text,
                "task_type": "Professional QA & Career Counseling",
                "tuning_instruction": "Fokus pada penjelasan konteks profesional, persiapan ujian sertifikasi, konseling karir, dan pengetahuan bidang hukum/keuangan/medis.",
                "source_name": "IndoCareer (HuggingFace: indolem/IndoCareer, all/test)"
            }
            print(f"      ✅ Loaded {len(ds_career):,} real rows!")
        except Exception as e:
            print(f"      ❌ Failed loading IndoCareer: {e}")

        # 6. IndoCulture (split 'test' — dataset tidak punya split train)
        try:
            print("  --> Loading indolem/IndoCulture (test)...")
            ds_culture = datasets.load_dataset("indolem/IndoCulture", split="test")
            self.text_datasets["IndoCulture"] = {
                "dataset": ds_culture,
                "text_builder": _build_indoculture_text,
                "task_type": "Cultural QA & Local Knowledge",
                "tuning_instruction": "Fokus pada pengetahuan kebudayaan Indonesia, adat istiadat daerah, kuliner, seni, dan Kearifan lokal.",
                "source_name": "IndoCulture (HuggingFace: indolem/IndoCulture, test)"
            }
            print(f"      ✅ Loaded {len(ds_culture):,} real rows!")
        except Exception as e:
            print(f"      ❌ Failed loading IndoCulture: {e}")

        # 7. QQPR-Triplets-ID
        try:
            print("  --> Loading robinsyihab/QQPR-triplets-ID...")
            ds_qqpr = datasets.load_dataset("robinsyihab/QQPR-triplets-ID", split="train")
            self.text_datasets["QQPR-Triplets-ID"] = {
                "dataset": ds_qqpr,
                "text_builder": _build_qqpr_text,
                "task_type": "Paraphrase Generation & Text Similarity",
                "tuning_instruction": "Fokus pada generasi paraphrasing, identifikasi kemiripan teks, dan penjelasan perbedaan makna antar kalimat.",
                "source_name": "QQPR-Triplets-ID (HuggingFace: robinsyihab/QQPR-triplets-ID)"
            }
            print(f"      ✅ Loaded {len(ds_qqpr):,} real rows!")
        except Exception as e:
            print(f"      ❌ Failed loading QQPR-Triplets-ID: {e}")

        # 8. Indonesian Wikipedia
        try:
            print("  --> Loading indonesian-nlp/wikipedia-id (subset 2%)...")
            ds_wiki = datasets.load_dataset("indonesian-nlp/wikipedia-id", split="train[:2%]")
            self.text_datasets["Indonesian-Wikipedia"] = {
                "dataset": ds_wiki,
                "text_builder": _build_wikipedia_text,
                "task_type": "General Knowledge & Encyclopedia QA",
                "tuning_instruction": "Fokus pada penjelasan pengetahuan umum, informasi ensiklopedia, fakta sejarah, sains, dan geografi Indonesia.",
                "source_name": "Indonesian Wikipedia (HuggingFace: indonesian-nlp/wikipedia-id)"
            }
            print(f"      ✅ Loaded {len(ds_wiki):,} real rows (subset)!")
        except Exception as e:
            print(f"      ❌ Failed loading Indonesian Wikipedia: {e}")

        # 9. Cendol Collection v2 (STREAMING — 12,8M rows tidak dimuat penuh)
        try:
            if self.quick:
                print("  --> ⏭️ Cendol Collection v2 dilewati (--quick)")
            else:
                print("  --> Loading indonlp/cendol_collection_v2 (streaming, shuffle acak)...")
                ds_cendol = datasets.load_dataset("indonlp/cendol_collection_v2", split="train", streaming=True)
                self.text_datasets["Cendol-Collection-v2"] = {
                    "dataset": ds_cendol,
                    "text_builder": _build_cendol_text,
                    "streaming": True,
                    "streaming_buffer": 1000,
                    "id_key": "prompt_id",
                    "task_type": "General Instruction Following & Knowledge QA",
                    "tuning_instruction": "Fokus pada instruction following, pengetahuan umum, dan percakapan sehari-hari dalam bahasa Indonesia.",
                    "source_name": "Cendol Collection v2 (HuggingFace: indonlp/cendol_collection_v2, streaming)"
                }
                print("      ✅ Streaming siap (12,8M rows, sampling shuffle acak)!")
        except Exception as e:
            print(f"      ❌ Failed loading Cendol Collection v2: {e}")

        # 10. LFQA-ID (BARU: long-form QA Indonesia, 226K — STREAMING)
        try:
            if self.quick:
                print("  --> ⏭️ LFQA-ID dilewati (--quick)")
            else:
                print("  --> Loading indonesian-nlp/lfqa_id (streaming)...")
                ds_lfqa = datasets.load_dataset("indonesian-nlp/lfqa_id", split="train", streaming=True)
                self.text_datasets["LFQA-ID"] = {
                    "dataset": ds_lfqa,
                    "text_builder": _build_lfqa_text,
                    "streaming": True,
                    "streaming_buffer": 1000,
                    "id_key": "q_id",
                    "task_type": "Long-Form QA & Explanatory Dialogue",
                    "tuning_instruction": "Fokus pada jawaban panjang yang mendidik, penjelasan konsep dengan analogi, dan percakapan tanya-jawab mendalam ala forum.",
                    "source_name": "LFQA-ID (HuggingFace: indonesian-nlp/lfqa_id, streaming)"
                }
                print("      ✅ Streaming siap (226K rows)!")
        except Exception as e:
            print(f"      ❌ Failed loading LFQA-ID: {e}")

        # 11. CVQA subset Indonesia (BARU: metadata-only VQA budaya)
        try:
            print("  --> Loading afaji/cvqa (subset Indonesia, metadata-only)...")
            ds_cvqa_id = self._load_cvqa_id_metadata()
            if ds_cvqa_id is not None and len(ds_cvqa_id) > 0:
                self.text_datasets["CVQA-Indonesia"] = {
                    "dataset": ds_cvqa_id,
                    "text_builder": _build_cvqa_id_text,
                    "task_type": "Cultural Visual QA (soal Bahasa Indonesia)",
                    "tuning_instruction": "Fokus pada tanya-jawab berbasis budaya Indonesia: landmark, makanan, pakaian, sejarah, dan kehidupan sehari-hari. Jelaskan alasan jawaban dengan detail.",
                    "source_name": "CVQA Subset Indonesia (HuggingFace: afaji/cvqa, metadata-only tanpa gambar)"
                }
                print(f"      ✅ Loaded {len(ds_cvqa_id):,} real rows (subset Indonesia)!")
            else:
                print("      ⚠️ CVQA subset Indonesia kosong, dilewati.")
        except Exception as e:
            print(f"      ❌ Failed loading CVQA: {e}")

        # 12. KTP VLM Instruct Dataset (Real Vision Data, dipertahankan)
        try:
            print("  --> Loading danielsyahputra/ktp-vlm-instruct-dataset...")
            ds_ktp = datasets.load_dataset("danielsyahputra/ktp-vlm-instruct-dataset", split="train")
            self.vision_datasets["KTP-VLM"] = {
                "dataset": ds_ktp,
                "text_builder": _build_ktp_text,
                "image_builder": _ktp_image_url,
                "is_vision": True,
                "task_type": "Document OCR & KTP Identity Extraction",
                "tuning_instruction": "Fokus pada ekstraksi data identitas KTP (NIK, Nama, Tempat/Tgl Lahir, Alamat, Agama, Status), penjelasan field identitas, dan analisis dokumen visual.",
                "source_name": "KTP VLM Instruct Dataset (HuggingFace: danielsyahputra/ktp-vlm-instruct-dataset)"
            }
            print(f"      ✅ Loaded {len(ds_ktp):,} real rows!")
        except Exception as e:
            print(f"      ❌ Failed loading KTP VLM Dataset: {e}")

        # 13. KORIKA SEA-VL Crowdsourcing ID (BARU: caption Bahasa Indonesia asli + gambar PIL, STREAMING)
        try:
            if self.quick:
                print("  --> ⏭️ KORIKA SEA-VL dilewati (--quick)")
            else:
                print("  --> Loading KORIKA-AI/sea-vl_crowdsourcing_id (streaming, gambar per-sample)...")
                ds_korika = datasets.load_dataset("KORIKA-AI/sea-vl_crowdsourcing_id", split="train", streaming=True)
                self.vision_datasets["SEA-VL-KORIKA-ID"] = {
                    "dataset": ds_korika,
                    "text_builder": _build_sea_vl_text,
                    "image_builder": _make_image_saver("korika_sea_vl"),
                    "is_vision": True,
                    "streaming": True,
                    "streaming_buffer": 64,
                    "id_key": "id",
                    "task_type": "Image Captioning & Cultural Scene Understanding",
                    "tuning_instruction": "Fokus pada deskripsi gambar budaya Asia Tenggara dalam Bahasa Indonesia: bangunan, kuliner, pakaian tradisional, kegiatan sehari-hari, dan landmark. Jelaskan detail visual dengan natural.",
                    "source_name": "KORIKA SEA-VL Crowdsourcing ID (HuggingFace: KORIKA-AI/sea-vl_crowdsourcing_id, streaming)"
                }
                print("      ✅ Streaming siap (3.096 rows, gambar diekstrak per sample)!")
        except Exception as e:
            print(f"      ❌ Failed loading KORIKA SEA-VL: {e}")

        # 14. SEACrowd SEA-VL Crowdsourcing (BARU: difilter native_lang=ind, STREAMING)
        try:
            if self.quick:
                print("  --> ⏭️ SEACrowd SEA-VL dilewati (--quick)")
            else:
                print("  --> Loading SEACrowd/sea-vl_crowdsourcing (streaming, filter native_lang=ind)...")
                ds_svl = datasets.load_dataset("SEACrowd/sea-vl_crowdsourcing", split="train", streaming=True)
                ds_svl = ds_svl.filter(lambda r: "ind" in str(r.get("native_lang", "")).lower())
                self.vision_datasets["SEA-VL-SEACrowd-ID"] = {
                    "dataset": ds_svl,
                    "text_builder": _build_sea_vl_text,
                    "image_builder": _make_image_saver("seacrowd_sea_vl"),
                    "is_vision": True,
                    "streaming": True,
                    "streaming_buffer": 64,
                    "id_key": "id",
                    "task_type": "Image Captioning & Cultural Scene Understanding",
                    "tuning_instruction": "Fokus pada deskripsi gambar budaya Asia Tenggara dalam Bahasa Indonesia: bangunan, kuliner, pakaian tradisional, kegiatan sehari-hari, dan landmark. Jelaskan detail visual dengan natural.",
                    "source_name": "SEACrowd SEA-VL Crowdsourcing (HuggingFace: SEACrowd/sea-vl_crowdsourcing, filter ind, streaming)"
                }
                print("      ✅ Streaming siap (7.010 rows, filter ind, gambar per-sample)!")
        except Exception as e:
            print(f"      ❌ Failed loading SEACrowd SEA-VL: {e}")

        # 15. IndoRad-VQA (BARU: VQA radiologi Bahasa Indonesia, train+test)
        try:
            print("  --> Loading Lab-IS/IndoRad-VQA (train+test)...")
            ds_indorad = datasets.load_dataset("Lab-IS/IndoRad-VQA", split="train+test")
            self.vision_datasets["IndoRad-VQA"] = {
                "dataset": ds_indorad,
                "text_builder": _build_indorad_text,
                "image_builder": _make_image_saver("indorad_vqa"),
                "is_vision": True,
                "task_type": "Medical VQA & Radiology Image Analysis",
                "tuning_instruction": "Fokus pada analisis citra radiologi (X-ray, CT, MRI): identifikasi kelainan, jawaban ya/tidak terstruktur, dan penjelasan medis sederhana dalam Bahasa Indonesia.",
                "source_name": "IndoRad-VQA (HuggingFace: Lab-IS/IndoRad-VQA, train+test)"
            }
            print(f"      ✅ Loaded {len(ds_indorad):,} real rows (train+test)!")
        except Exception as e:
            print(f"      ❌ Failed loading IndoRad-VQA: {e}")

        removed = self.apply_exclusions()
        if removed:
            print(f"🚫 EXCLUDE: {removed} row/id yang sudah ada di JSONL sebelumnya dikeluarkan dari pool sampling.")

        print(f"\n📊 RINGKASAN: {len(self.text_datasets)} text dataset(s), {len(self.vision_datasets)} vision dataset(s)")


def sample_random_real_row(loader: PureHuggingFaceDataLoader, category_mode: str) -> Tuple[Dict[str, Any], int, str, str, Optional[str]]:
    """Sample 1 baris mentah asli secara acak dari HuggingFace dataset."""
    if category_mode == "vision" and loader.vision_datasets:
        dataset_key = random.choice(list(loader.vision_datasets.keys()))
        meta = loader.vision_datasets[dataset_key]
        ds = meta["dataset"]

        # Sampling: streaming (shuffle multi-shard datasets 5.0.x) atau index acak in-memory
        if meta.get("streaming"):
            seed = random.randint(0, 2**31 - 1)
            buf = meta.get("streaming_buffer", 64)
            it = ds.shuffle(seed=seed, buffer_size=buf, max_buffer_input_shards=4)
            row = next(iter(it))
            row_idx = -1
        else:
            row_idx = random.randint(0, len(ds) - 1)
            row = ds[row_idx]

        # Bangun teks konteks dari text_builder (umum) atau instruction/output key (khusus lama)
        if "text_builder" in meta and callable(meta["text_builder"]):
            raw_text = meta["text_builder"](row)
        else:
            inst = row.get(meta.get("instruction_key", "instruction"), "")
            out = row.get(meta.get("output_key", "output"), "")
            raw_text = f"Instruksi Asli: {inst}\nOutput Asli: {out}"

        # Referensi gambar: URL (KTP) atau path lokal hasil ekstrak (KORIKA/SEA-VL/IndoRad)
        image_ref = meta["image_builder"](row, row_idx) if meta.get("image_builder") else None

        if row_idx >= 0:
            source_detail = f"{meta['source_name']}, Row #{row_idx}"
        else:
            id_val = str(row.get(meta.get("id_key", "id"), ""))
            source_detail = f"{meta['source_name']}, id={id_val}"
        return meta, row_idx, raw_text, source_detail, image_ref

    # Default to text
    if not loader.text_datasets:
        raise RuntimeError("No text dataset loaded!")

    dataset_key = random.choice(list(loader.text_datasets.keys()))
    meta = loader.text_datasets[dataset_key]
    ds = meta["dataset"]

    if meta.get("streaming"):
        seed = random.randint(0, 2**31 - 1)
        buf = meta.get("streaming_buffer", 1000)
        it = ds.shuffle(seed=seed, buffer_size=buf, max_buffer_input_shards=4)
        row = next(iter(it))
        row_idx = -1
    else:
        row_idx = random.randint(0, len(ds) - 1)
        row = ds[row_idx]

    if "text_builder" in meta and callable(meta["text_builder"]):
        raw_text = meta["text_builder"](row)
    else:
        raw_text = row.get(meta.get("text_key", ""), "") or str(row)
    if len(raw_text) > 1000:
        raw_text = raw_text[:1000] + "..."

    if row_idx >= 0:
        source_detail = f"{meta['source_name']}, Row #{row_idx}"
    else:
        id_val = str(row.get(meta.get("id_key", "id"), ""))
        source_detail = f"{meta['source_name']}, id={id_val}"
    return meta, row_idx, raw_text, source_detail, None


def build_prompt_for_conversation(meta: Dict[str, Any], raw_context: str, source_detail: str, num_pairs: int) -> str:
    is_vision = bool(meta.get("is_vision", False))
    vision_note = " (Catatan: Ini adalah percakapan visual. Turn user pertama HARUS dimulai dengan token 📷 yang menandai adanya gambar. Seluruh dialog HARUS NYAMBUNG 100% dengan objek/teks asli yang ada di dalam gambar)." if is_vision else ""
    total_msgs = num_pairs * 2
    row_label = source_detail.split("#")[-1] if "#" in source_detail else "baris acak (streaming)"

    prompt = f"""
Buatkan 1 percakapan multi-turn sintetis Bahasa Indonesia antara User dan Gemma (Assistant) sebanyak TEPAT {num_pairs} pasang dialog (total TEPAT {total_msgs} pesan selang-seling).

Konteks / Baris Dataset Asli dari HuggingFace:
- Sumber Metadata Detail: {source_detail}
- Kategori Task: {meta['task_type']}
- Instruksi Tuning Khusus Sumber: {meta['tuning_instruction']}
- Isi Teks Mentah Baris Asli (Row {row_label}):
\"\"\"
{raw_context}
\"\"\"

Panduan Pengisian Field Pydantic:
1. `prefixes`: Khusus role 'assistant', berisi 1 ATAU LEBIH token prefix murni (maksimal 3) yang MENGGAMBARKAN TASK YANG TERKANDUNG dalam pesan yang kamu tulis. LAKUKAN REASONING DULU:
   - Baca isi pesan yang akan kamu tulis, identifikasi task-task yang terkandung (bisa lebih dari satu, misal merangkum sekaligus menerjemahkan).
   - Pilih satu token untuk setiap task yang muncul, gabung TANPA SPASI, dan JANGAN pernah mengulang token yang sama.
   Token tersedia:
   - "<unused1>" SUMMARIZE (perangkuman)
   - "<unused2>" TRANSLATE (terjemahan)
   - "<unused3>" NER (ekstraksi entitas/identitas)
   - "<unused4>" QA (tanya-jawab/penjelasan)
   - "<unused5>" PARAPHRASE (parafrase/kemiripan teks)
   - "<unused6>" GENERAL_CHAT (percakapan umum)
   Contoh BENAR: ['<unused1>'] (satu task), ['<unused1>', '<unused2>'] (rangkum + terjemah -> digabung '<unused1><unused2>').
   Contoh SALAH: ['<unused4>', '<unused4>'] (mengulang token sama — TIDAK BOLEH), '<unused4> <unused4>' (ber-spasi — TIDAK BOLEH).
   Khusus role 'user', isi list kosong [].
2. `content`: Tuliskan isi teks pesan murni TANPA pernah menyertakan kata seperti "SUMMARIZE" atau "NER", DAN TANPA token '<unusedX>' (token prefix HANYA boleh ada di field `prefixes`, jangan pernah menulisnya di dalam konten).
3. Percakapan HARUS 100% NYAMBUNG dengan isi teks mentah asli.{vision_note}
"""
    return prompt.strip()

# ─── Rate Limiter (RPM & TPM) ─────────────────────────────────────────────────

class RateLimiter:
    """Token bucket async sederhana untuk RPM (requests/menit) & TPM (tokens/menit).

    Bucket di-refill kontinu berbasis waktu; request menunggu (polling 0.5s)
    sampai tersedia 1 token RPM dan cukup token TPM. Aman dipakai lintas worker.
    """

    def __init__(self, rpm: int = 10, tpm: int = 5_000_000):
        self.rpm = max(1, rpm)
        self.tpm = max(1, tpm)
        self._tokens = float(self.rpm)
        self._tp_tokens = float(self.tpm)
        self._last_refill = time.monotonic()
        self._lock = asyncio.Lock()

    async def acquire(self, est_tokens: int = 1) -> None:
        while True:
            async with self._lock:
                now = time.monotonic()
                dt = now - self._last_refill
                self._tokens = min(self.rpm, self._tokens + dt * self.rpm / 60.0)
                self._tp_tokens = min(self.tpm, self._tp_tokens + dt * self.tpm / 60.0)
                self._last_refill = now
                if self._tokens >= 1.0 and self._tp_tokens >= est_tokens:
                    self._tokens -= 1.0
                    self._tp_tokens -= est_tokens
                    return
            await asyncio.sleep(0.5)


def _render_draft(convs: List[TurnMessage]) -> str:
    """Render draft percakapan menjadi teks untuk prompt review (stage-2)."""
    lines = []
    for i, m in enumerate(convs, 1):
        if m.role == "user":
            lines.append(f"[{i}] USER: {m.content}")
        else:
            pfx = "".join(m.prefixes) if m.prefixes else "<unused4>"
            lines.append(f"[{i}] ASSISTANT ({pfx}): {m.content}")
    return "\n".join(lines)


def _count_edited_turns(a: List[TurnMessage], b: List[TurnMessage]) -> int:
    """Hitung jumlah turn yang berubah antara draft dan final (untuk metadata)."""
    n = 0
    for ma, mb in zip(a, b):
        if ma.role != mb.role or ma.content != mb.content or ma.prefixes != mb.prefixes:
            n += 1
    return n


def build_review_prompt(
    meta: Dict[str, Any],
    raw_context: str,
    source_detail: str,
    draft_convs: List[TurnMessage],
    num_pairs: int,
) -> str:
    """Prompt stage-2: model menganalisa SELURUH draft & mengembalikan versi final yang diedit."""
    total_msgs = num_pairs * 2
    draft_text = _render_draft(draft_convs)
    return f"""
Berikut adalah DRAFT percakapan multi-turn Bahasa Indonesia antara User dan Gemma (Assistant)
beserta konteks sumber aslinya. Tugasmu: ANALISA seluruh percakapan ini turn demi turn,
lalu kembalikan VERSI FINAL yang sudah diperbaiki.

Konteks Sumber Asli:
- Sumber Metadata Detail: {source_detail}
- Kategori Task: {meta['task_type']}
- Instruksi Tuning Khusus Sumber: {meta['tuning_instruction']}
- Isi Teks Mentah Baris Asli:
\"\"\"
{raw_context}
\"\"\"

DRAFT PERCAKAPAN:
\"\"\"
{draft_text}
\"\"\"

Checklist Evaluasi (periksa SATU PER SATU setiap turn):
1. Prefix turn assistant TEPAT & sesuai isi pesan. Token: <unused1> SUMMARIZE, <unused2> TRANSLATE,
   <unused3> NER, <unused4> QA, <unused5> PARAPHRASE, <unused6> GENERAL_CHAT.
   Boleh 1+ token UNIK tanpa spasi/duplikat jika pesan multi-task (mis. <unused1><unused2>).
2. Konten 100% nyambung dengan isi teks mentah asli (JANGAN menambah fakta di luar data).
3. Bahasa Indonesia natural & gaya sesuai Instruksi Tuning.
4. TIDAK ada kebocoran kata nama task (SUMMARIZE, TRANSLATE, NER, QA, PARAPHRASE, GENERAL_CHAT) di dalam konten.
5. Turn selang-seling user/assistant dengan TEPAT {total_msgs} pesan, role benar, jumlah pasangan tepat {num_pairs}.
6. TIDAK ada token '<unusedX>' di dalam konten turn — token prefix HANYA boleh di field `prefixes`, jangan pernah di dalam konten.

Instruksi:
- Turn yang SUDAH memenuhi checklist: DIPERTAHANKAN IDENTIK (jangan diubah).
- Turn yang melanggar checklist: PERBAIKI (prefix dan/atau konten sesuai kebutuhan).
- Output: seluruh turn final (user & assistant) dalam skema ConversationOutput, selang-seling dari turn 1.
""".strip()


async def generate_single_conversation_pydantic(
    agent: Agent[Any, ConversationOutput],
    loader: PureHuggingFaceDataLoader,
    category_mode: str,
    num_pairs: int,
    max_retries: int = 3,
    limiter: Optional[RateLimiter] = None,
    review: bool = True
) -> Optional[Dict[str, Any]]:
    # Sample baris riil secara acak dari HuggingFace dataset
    meta, row_idx, raw_context, source_detail, dynamic_image_ref = sample_random_real_row(loader, category_mode)

    async def run_with_limit(prompt_text: str) -> ConversationOutput:
        if limiter is not None:
            # Estimasi token: panjang prompt + anggaran output maks (~3K token), 4 char/token latin
            est = max(1, (len(prompt_text) + 12000) // 4)
            await limiter.acquire(est)
        res = await agent.run(prompt_text)
        return res.output

    # ── Stage 1: generate DRAFT ────────────────────────────────────────────────
    draft_convs: Optional[List[TurnMessage]] = None
    for attempt in range(max_retries):
        try:
            prompt = build_prompt_for_conversation(meta, raw_context, source_detail, num_pairs=num_pairs)
            draft_convs = (await run_with_limit(prompt)).conversations
            break
        except Exception as e:
            if attempt == max_retries - 1:
                print(f"[ERROR] Gagal generate draft untuk {source_detail}: {e}", file=sys.stderr)
            await asyncio.sleep(1.5 * (attempt + 1))
    if draft_convs is None:
        return None

    # ── Stage 2: SELF-REVIEW & EDIT (opsional) ─────────────────────────────────
    final_convs: List[TurnMessage] = draft_convs
    edited_turns = 0
    if review:
        review_prompt = build_review_prompt(meta, raw_context, source_detail, draft_convs, num_pairs)
        for attempt in range(max_retries):
            try:
                reviewed = (await run_with_limit(review_prompt)).conversations
                if reviewed:
                    final_convs = reviewed
                    edited_turns = _count_edited_turns(draft_convs, final_convs)
                break
            except Exception as e:
                if attempt == max_retries - 1:
                    print(f"[ERROR] Gagal review untuk {source_detail}: {e}", file=sys.stderr)
                await asyncio.sleep(1.5 * (attempt + 1))

    # ── Format pesan lengkap dari versi FINAL ──────────────────────────────────
    formatted_messages = [{"role": "system", "content": SYSTEM_PROMPT}]

    is_vision = dynamic_image_ref is not None
    for idx, msg in enumerate(final_convs):
        role = msg.role
        content_clean = msg.content.replace("\\n", "\n").strip()

        # Pastikan tidak ada kebocoran teks nama task
        for leak in ["SUMMARIZE", "TRANSLATE", "NER", "QA", "PARAPHRASE", "GENERAL_CHAT"]:
            content_clean = re.sub(rf"^\s*{leak}\s*", "", content_clean, flags=re.IGNORECASE)

        # Bersihkan token prefix yang bocor ke AWAL content (model sering meniru
        # format target dengan menulis '<unused4> <unused4>' di dalam konten)
        content_clean = re.sub(r"^(?:\s*<unused[1-6]>\s*)+", "", content_clean)

        if role == "user":
            if is_vision and idx == 0 and not content_clean.startswith("📷"):
                content_clean = "📷\n" + content_clean
            formatted_messages.append({"role": "user", "content": content_clean})
        else:
            # Assistant turn: gabungkan prefix murni (bisa multi-task, tanpa spasi).
            # Defensif: ekstrak semua token valid, buang spasi internal & duplikat
            tokens = re.findall(r"<unused[1-6]>", "".join(msg.prefixes))
            seen: List[str] = []
            for t in tokens:
                if t not in seen:
                    seen.append(t)
            prefix_str = "".join(seen) if seen else "<unused4>"
            assistant_content = f"{prefix_str} {content_clean}".strip()
            formatted_messages.append({"role": "assistant", "content": assistant_content})

    result = {
        "source": source_detail,
        "category": "vision_chat" if is_vision else "text_nlu_chat",
        "num_turns": len(formatted_messages),
        "num_pairs": len(final_convs) // 2,
        "reviewed": bool(review),
        "edited_turns": edited_turns,
        "messages": formatted_messages
    }
    if is_vision and dynamic_image_ref:
        result["images"] = [dynamic_image_ref]

    return result

async def main_async():
    default_model_name = os.environ.get("STEPFUN_MODEL") or os.environ.get("API_MODEL") or "step-3.7-flash"
    default_base_url = (
        os.environ.get("STEPFUN_BASE_URL") or
        os.environ.get("API_BASE_URL") or
        "https://api.stepfun.ai/step_plan/v1"
    )
    api_key = (
        os.environ.get("STEPFUN_API_KEY") or
        os.environ.get("API_KEY") or
        os.environ.get("OPENROUTER_API_KEY") or
        "sk-dummy"
    )

    parser = argparse.ArgumentParser(description="Generate Synthetic Conversational Dataset EXTRA 2026-07-31 dari Pure Real HuggingFace Datasets via Pydantic AI")
    parser.add_argument("--target", type=int, default=20, help="Jumlah total percakapan yang ingin dibuat (default: 20)")
    parser.add_argument("--limit", type=int, default=0, help="Alias untuk --target jika ingin uji coba sampel kecil")
    parser.add_argument("--mode", type=str, choices=["text", "vision", "mixed"], default="mixed", help="Mode data: 'text', 'vision', 'mixed'")
    parser.add_argument("--output", type=Path, default=DATA_DIR / "synthetic" / "generated_conv_extra_20260731.jsonl", help="Path file output JSONL")
    parser.add_argument("--base-url", type=str, default=default_base_url, help="OpenAI-compatible API Base URL (misal: https://api.stepfun.ai/step_plan/v1)")
    parser.add_argument("--model", type=str, default=default_model_name, help="Model name (misal: step-3.7-flash, step-3.5-flash)")
    parser.add_argument("--concurrency", type=int, default=5, help="Jumlah concurrent API requests")
    parser.add_argument("--rpm", type=int, default=10, help="Rate limit: request per menit")
    parser.add_argument("--tpm", type=int, default=5_000_000, help="Rate limit: token per menit")
    parser.add_argument("--max-tokens", type=int, default=0, help="Batas token output per respons (0 = default model, mis. 2048/4096)")
    parser.add_argument("--quick", action="store_true", help="Mode uji cepat: hanya muat dataset ringan (skip Cendol/LFQA/KORIKA/SEA-VL)")
    parser.add_argument("--text-target", type=int, default=0, help="Kuota percakapan TEXT khusus (0 = ikut --mode)")
    parser.add_argument("--vision-target", type=int, default=0, help="Kuota percakapan VISION khusus (0 = ikut --mode)")
    parser.add_argument("--exclude-file", nargs="*", type=Path, default=[], help="File JSONL tambahan untuk exclude row/id yang sudah dipakai (selain output file)")
    parser.add_argument("--overwrite", action="store_true", help="Mulai dari nol: timpa file output (default: append ke file yang ada)")
    parser.add_argument("--no-review", action="store_true", help="Nonaktifkan stage-2 self-review (hemat 2x request; default: review AKTIF)")

    args = parser.parse_args()
    target_count = args.limit if args.limit > 0 else args.target

    # Kuota per kategori (--text-target/--vision-target) mengalahkan --mode/--target
    use_quota = args.text_target > 0 or args.vision_target > 0
    if use_quota:
        total_count = args.text_target + args.vision_target
        task_cats = ["text"] * args.text_target + ["vision"] * args.vision_target
        random.shuffle(task_cats)
    else:
        total_count = target_count
        task_cats = None

    # 1. Load Pure Real HuggingFace Datasets into Memory
    loader = PureHuggingFaceDataLoader(quick=args.quick)

    # Baca row/id yang sudah dipakai: file output default (resume) + --exclude-file
    # (kecuali --overwrite: mulai dari nol, boleh pakai ulang row lama)
    exclude_paths = list(args.exclude_file)
    if args.output.exists() and not args.overwrite:
        exclude_paths.append(args.output)
    if exclude_paths:
        loader.excluded_rows, loader.excluded_ids = load_excluded_sources(exclude_paths)
        n_excluded = sum(len(v) for v in loader.excluded_rows.values()) + sum(len(v) for v in loader.excluded_ids.values())
        if n_excluded:
            print(f"🚫 {n_excluded} row/id sumber dikecualikan dari {len(exclude_paths)} file JSONL sebelumnya.")

    loader.load_all()

    # 2. Inisialisasi Pydantic AI Model & Agent dengan retries=3
    provider = OpenAIProvider(
        base_url=args.base_url,
        api_key=api_key
    )
    pydantic_model = OpenAIChatModel(args.model, provider=provider)

    # ModelSettings opsional: batasi max_tokens output agar tidak boros TPM/biaya
    model_settings = ModelSettings(max_tokens=args.max_tokens) if args.max_tokens > 0 else None

    agent = Agent(
        model=pydantic_model,
        output_type=ConversationOutput,
        retries=3,
        model_settings=model_settings,
        instructions="Kamu adalah generator percakapan sintetis terstruktur yang mematuhi skema Pydantic. Kamu HANYA mengembalikan token prefix murni seperti ['<unused1>'] tanpa pernah menyertakan teks kata 'SUMMARIZE' atau 'NER'."
    )

    print(f"\n=== Synthetic Dataset Generator EXTRA (Expanded Multi-Source HuggingFace Loader) ===")
    print(f"Mode Generation   : {args.mode.upper()}")
    print(f"Target Percakapan : {total_count}")
    if use_quota:
        print(f"  ├─ TEXT          : {args.text_target}")
        print(f"  └─ VISION        : {args.vision_target}")
    print(f"Base URL          : {args.base_url}")
    print(f"Model             : {args.model}")
    print(f"Output File       : {args.output} ({'OVERWRITE' if args.overwrite else 'APPEND'})")
    print(f"Concurrency       : {args.concurrency}")
    print(f"Rate Limit        : {args.rpm} req/mnt | {args.tpm:,} tok/mnt" + (f" | max_tokens={args.max_tokens}" if args.max_tokens > 0 else ""))
    print(f"Self-Review       : {'OFF' if args.no_review else 'ON'} (draft -> analisa -> edit)")
    print()

    args.output.parent.mkdir(parents=True, exist_ok=True)

    # ID global lintas run (append): mulai dari id terbesar di file yang ada
    existing_max_id = 0
    if args.output.exists() and not args.overwrite:
        try:
            with open(args.output, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        iid = json.loads(line).get("id")
                        if isinstance(iid, int) and iid > existing_max_id:
                            existing_max_id = iid
                    except json.JSONDecodeError:
                        continue
        except Exception:
            pass

    semaphore = asyncio.Semaphore(args.concurrency)
    limiter = RateLimiter(rpm=args.rpm, tpm=args.tpm)
    generated_count = 0
    start_time = time.time()

    write_mode = "w" if args.overwrite else "a"
    write_lock = asyncio.Lock()
    out_file = open(args.output, write_mode, encoding="utf-8")

    async def worker(conv_id: int):
        nonlocal generated_count
        async with semaphore:
            num_pairs = random.choice([3, 4, 5])
            if task_cats is not None:
                # Kuota per kategori: alokasi ditentukan urutan task_cats
                category_mode = task_cats[conv_id - 1]
            else:
                category_mode = args.mode
                if category_mode == "mixed":
                    category_mode = random.choice(["text", "vision"])

            res = await generate_single_conversation_pydantic(agent, loader, category_mode, num_pairs=num_pairs, limiter=limiter, review=not args.no_review)
            if res:
                res["id"] = existing_max_id + conv_id
                # Tulis langsung per item (real-time): progress tersimpan, tahan interupsi
                async with write_lock:
                    out_file.write(json.dumps(res, ensure_ascii=False) + "\n")
                    out_file.flush()
                generated_count += 1
                print(f"[{generated_count}/{total_count}] Generated ID {res['id']} ({res['category']} - {res['source']}) | Pairs: {num_pairs} ({num_pairs*2} msgs)")
                return res
            return None

    try:
        tasks = [worker(i + 1) for i in range(total_count)]
        await asyncio.gather(*tasks)
    finally:
        out_file.close()

    # Total baris di file setelah append
    try:
        with open(args.output, "r", encoding="utf-8") as f:
            total_lines = sum(1 for _ in f)
    except Exception:
        total_lines = generated_count

    elapsed = time.time() - start_time
    print(f"\n[SELESAI] Berhasil membuat {generated_count} percakapan sintetis Pydantic AI dalam {elapsed:.2f} detik.")
    print(f"Disimpan di: {args.output} (total {total_lines:,} baris di file)")

def main():
    asyncio.run(main_async())


if __name__ == "__main__":
    main()
