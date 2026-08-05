"""
Generate Synthetic Conversational Dataset "EXTRA" — v2 (2026-08-02)
====================================================================
Pipeline STANDALONE (API StepFun via pydantic-ai) — fitur PERSIS dengan
MCPB `generate-conv-indonesia` (v2.1.18), tapi dijalankan sendiri tanpa Claude.

Yang dipertahankan & diselaraskan dengan MCPB:
  - RANDOM-ACCESS datasets-server: 1 HTTP call per row, TANPA unduh dataset penuh
    (termasuk Cendol 12,8jt row). Sampling PROPORSIONAL jumlah row.
  - 15 pool sumber: 10 text + 5 vision (IndoRad-VQA split train saja).
  - CVQA & SEACrowd: offset map pyarrow (columnar-read kolom kecil saja) + fallback
    `/rows` rejection sampling (rate-limit safe) — tetap MULTIMODAL.
  - Exclude row/id yang sudah dipakai (dari file output + --exclude-file).
  - Kuota 2:1 (--text-target / --vision-target) dengan pemilihan kategori dinamis
    berbasis rasio (yang paling tertinggal) — persis `get_progress` di MCPB.
  - Distribusi PREFIX <unused1..6> & jumlah pasang 3/4/5 SEIMBANG lintas percakapan
    (hint dihitung dari file output + yang sudah dibuat, dimasukkan ke prompt).
  - Validasi strict: TEPAT 6/8/10 pesan (3-5 pasang), turn assistant min 40
    (vision) / 60 (text) karakter, prefix 1-3 token unik & user wajib [],
    selang-seling, tanpa token <unusedX>/kata task di konten.
  - Sanitasi: strip token <unusedX> di SELURUH konten, prefix digabung dari field
    `prefixes`, 📷 otomatis di user pertama vision.
  - Prompt berkualitas: grounding 2-3 fakta kunci, turn user seperti manusia
    (bukan robot/AI), kedalaman jawaban & anti-repetisi (kalimat sama antar
    turn ditolak), etika KTP.

Arsitektur pydantic-ai:
  - WRITER + REVIEWER (sub-agent): writer membuat draft (temp 0.6, retries=3),
    reviewer (temp 0.1) menilai grounding/kedalaman/repetisi/prefix, lalu loop
    revisi (maks 2x) sampai approved (atau --no-review untuk hemat API).
  - Panjang minimal turn assistant & repetisi kalimat di-cek deterministik setelah
    run; bila gagal, prompt di-append catatan error & diulang (maks 2x).

Penggunaan:
  python scripts/dataset/generate_conv_extra_20260802.py --mode text --target 2000 --model step-3.7-flash
  python scripts/dataset/generate_conv_extra_20260802.py --mode vision --target 1000 --model step-3.7-flash
  python scripts/dataset/generate_conv_extra_20260802.py --text-target 2000 --vision-target 1000 --model step-3.7-flash
  python scripts/dataset/generate_conv_extra_20260802.py --quick --text-target 2 --vision-target 2   # uji cepat
  python scripts/dataset/generate_conv_extra_20260802.py --limit 2 --exclude-file data/synthetic/run_lain.jsonl
  python scripts/dataset/generate_conv_extra_20260802.py --overwrite --text-target 2000 --vision-target 1000

Konfigurasi default (batas API):
  --model step-3.7-flash | --concurrency 5 | --rpm 10 | --tpm 5000000
API key: env STEPFUN_API_KEY / API_KEY / OPENROUTER_API_KEY (jangan hardcode).
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
import urllib.error
import urllib.request
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional, Tuple, cast
from urllib.parse import urlencode

import datasets
from PIL import Image as PILImage
from pydantic import BaseModel, Field, field_validator

from pydantic_ai import Agent, ModelSettings
from pydantic_ai.models.openai import OpenAIChatModel
from pydantic_ai.providers.openai import OpenAIProvider

# Load .env dari root project jika ada (jangan dibuka/diedit manual)
SCRIPT_DIR = Path(__file__).resolve().parent
ROOT_DIR = SCRIPT_DIR.parent.parent
DATA_DIR = ROOT_DIR / "data"
VISION_IMAGE_DIR = DATA_DIR / "multimodal" / "images"

try:
    from dotenv import load_dotenv
    load_dotenv(ROOT_DIR / ".env")
except ImportError:
    pass

# ─── Default System Prompt Gemma ───────────────────────────────────────────────
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

_PREFIX_LABEL = {"<unused1>": "SUMMARIZE", "<unused2>": "TRANSLATE", "<unused3>": "NER",
                 "<unused4>": "QA", "<unused5>": "PARAPHRASE", "<unused6>": "GENERAL_CHAT"}

DEFAULT_OUTPUT = DATA_DIR / "synthetic" / "generated_conv_extra_20260802.jsonl"
DS_API = "https://datasets-server.huggingface.co"


class _DataSourceError(RuntimeError):
    """Error dari datasets-server (bukan error logika sampling)."""


# ─── Robust Pydantic Output Schemas (PERSIS MCPB v2.1.18) ─────────────────────

class TurnMessage(BaseModel):
    role: Literal["user", "assistant"] = Field(description="Role turn pesan: harus 'user' atau 'assistant'")
    prefixes: List[VALID_PREFIXES] = Field(
        default=[],
        description="Khusus role assistant: WAJIB 1-3 token prefix murni unik, contoh: ['<unused1>'] atau ['<unused1>', '<unused4>']. Khusus role user: isi list kosong []"
    )
    content: str = Field(description="Isi teks pesan murni TANPA token <unusedX> atau kata nama task seperti 'SUMMARIZE' atau 'NER'")


class ConversationOutput(BaseModel):
    conversations: List[TurnMessage] = Field(
        description="Pasangan pesan user-assistant selang-seling, total TEPAT 6, 8, atau 10 pesan (3, 4, atau 5 pasang)"
    )

    @field_validator("conversations")
    @classmethod
    def check_turns(cls, convs: List[TurnMessage]) -> List[TurnMessage]:
        n = len(convs)
        if n % 2 != 0:
            raise ValueError("Jumlah pesan harus genap (pasangan user-assistant).")
        if n < 6 or n > 10:
            raise ValueError(
                f"Jumlah pesan harus TEPAT 6, 8, atau 10 (3, 4, atau 5 pasang user-assistant), "
                f"bukan {n} pesan ({n // 2} pasang). PERPANJANG percakapanmu."
            )
        for i, m in enumerate(convs):
            expected_role = "user" if i % 2 == 0 else "assistant"
            if m.role.strip() != expected_role:
                raise ValueError(f"Pesan urutan [{i}] harus ber-role '{expected_role}', bukan '{m.role}'")
            if m.role == "user":
                if m.prefixes:
                    raise ValueError(
                        f"Turn user [{i}] wajib prefixes kosong [] (ditemukan {m.prefixes}). "
                        f"Token prefix HANYA untuk role assistant."
                    )
            else:
                # Normalisasi: strip spasi per elemen, buang duplikat (pertahankan urutan).
                if not m.prefixes:
                    m.prefixes = ["<unused4>"]
                else:
                    cleaned = []
                    for p in m.prefixes:
                        p = re.sub(r"\s+", "", str(p))
                        if p and p not in cleaned:
                            cleaned.append(p)
                    m.prefixes = cast(List[VALID_PREFIXES], cleaned) if cleaned else ["<unused4>"]
                if len(m.prefixes) > 3:
                    raise ValueError(
                        f"Prefix turn assistant [{i}] maksimal 3 token UNIK (ditemukan "
                        f"{len(m.prefixes)}: {m.prefixes}). Batasi hingga 3 task per pesan."
                    )
        return convs


# ─── Sub-agent Review (kompensasi model lemah) ───────────────────────────────

class ReviewIssue(BaseModel):
    turn_index: int = Field(description="Indeks turn 0-based; -1 artinya masalah global (bukan turn tertentu)")
    severity: Literal["error", "warning"] = Field(description="error = wajib diperbaiki, warning = saran")
    problem: str = Field(description="Jelaskan masalahnya secara spesifik")
    suggestion: str = Field(description="Saran perbaikan yang konkret")


class ReviewReport(BaseModel):
    approved: bool = Field(description="true bila percakapan LULUS review, false bila masih ada masalah")
    summary: str = Field(description="Ringkasan penilaian 1-2 kalimat")
    issues: List[ReviewIssue] = Field(default=[], description="Daftar masalah yang ditemukan (kosong = bersih)")


# ─── Text Builder Helpers (per sumber) — PERSIS MCPB ─────────────────────────

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
    question = row.get("context", row.get("question", ""))
    options = _normalize_options(row.get("options", []))
    answer = row.get("answer", "")
    options_text = "\n".join(f"{chr(65 + i)}. {opt}" for i, opt in enumerate(options))
    return f"Soal: {question}\n\nPilihan:\n{options_text}\n\nJawaban: {answer}"


def _build_indocareer_text(row: Dict[str, Any]) -> str:
    question = row.get("Question", "")
    options = _normalize_options([
        row.get("Option A", ""), row.get("Option B", ""), row.get("Option C", ""),
        row.get("Option D", ""), row.get("Option E", ""),
    ])
    answer = row.get("Answer Key", "")
    options_text = "\n".join(f"{chr(65 + i)}. {opt}" for i, opt in enumerate(options))
    return f"Soal: {question}\n\nPilihan:\n{options_text}\n\nJawaban: {answer}"


def _build_indoculture_text(row: Dict[str, Any]) -> str:
    question = row.get("context", row.get("question", ""))
    options = _normalize_options(row.get("options", []))
    answer = row.get("answer", "")
    options_text = "\n".join(f"{chr(65 + i)}. {opt}" for i, opt in enumerate(options))
    return f"Soal: {question}\n\nPilihan:\n{options_text}\n\nJawaban: {answer}"


def _build_simple_summaries_text(row: Dict[str, Any]) -> str:
    original_text = row.get("translated_text", row.get("original_text", ""))
    summary = row.get("translated_summary", row.get("original_summary", ""))
    return f"Teks Asli:\n{original_text}\n\nRingkasan:\n{summary}"


def _build_qqpr_text(row: Dict[str, Any]) -> str:
    query = row.get("query", "")
    pos = _normalize_options(row.get("pos", []))[:3]
    neg = _normalize_options(row.get("neg", []))[:3]
    pos_text = "\n".join(f"- {p}" for p in pos)
    neg_text = "\n".join(f"- {n}" for n in neg)
    return f"Pertanyaan: {query}\n\nParaprase Positif:\n{pos_text}\n\nParaprase Negatif:\n{neg_text}"


def _build_wikipedia_text(row: Dict[str, Any]) -> str:
    title = row.get("title", "")
    text = row.get("text", "")
    if len(text) > 1500:
        text = text[:1500] + "..."
    return f"Judul: {title}\n\n{text}"


def _build_cendol_text(row: Dict[str, Any]) -> str:
    input_text = row.get("input", "")
    output_text = row.get("output", "")
    return f"Instruksi: {input_text}\n\nRespon: {output_text}"


def _build_indommlu_csv_text(row: Dict[str, Any]) -> str:
    soal = row.get("soal", "")
    kunci = row.get("kunci", "")
    subject = row.get("subject", "")
    jawaban = row.get("jawaban", "")
    opts = [ln.strip() for ln in str(jawaban).splitlines() if re.match(r"^[A-E]\.", ln.strip())]
    options_text = "\n".join(opts)
    return f"Soal: {soal}\n\nPilihan:\n{options_text}\n\nJawaban: {kunci}\nMata Pelajaran: {subject}"


def _build_lfqa_text(row: Dict[str, Any]) -> str:
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
    cap_id = row.get("caption_native_lang", "")
    cap_en = row.get("caption", "")
    loc = row.get("culture_relevant_loc", "")
    lang = row.get("native_lang", "")
    return f"Deskripsi (Bahasa Indonesia): {cap_id}\nDeskripsi (English): {cap_en}\nLokasi Budaya: {loc}\nBahasa: {lang}"


def _build_ktp_text(row: Dict[str, Any]) -> str:
    inst = row.get("instruction", "")
    out = row.get("output", "")
    return f"Instruksi Asli: {inst}\nOutput Asli KTP: {out}"


def _build_indorad_text(row: Dict[str, Any]) -> str:
    question = row.get("question_indonesian", "") or row.get("question", "")
    answer = row.get("answer_indonesian", "") or row.get("answer", "")
    answer_type = row.get("answer_type", "")
    return f"Pertanyaan Medis (Radiologi): {question}\nJawaban: {answer}\nTipe Jawaban: {answer_type}"


# ─── Registry Sumber (PERSIS MCPB) ─────────────────────────────────────────────
# access: "rows" (/rows random) | "offsets" (subset via pyarrow + /rows) | "csv"
SOURCES: List[Dict[str, Any]] = [
    # ── TEXT ──
    {"key": "IndoMMLU", "category": "text", "label": "ringan", "access": "csv",
     "dataset": "indolem/IndoMMLU", "config": "default", "split": "train",
     "csv_url": "https://huggingface.co/datasets/indolem/IndoMMLU/resolve/main/IndoMMLU.csv",
     "builder": _build_indommlu_csv_text, "id_key": None, "approx_rows": 14981,
     "source_name": "IndoMMLU (HuggingFace: indolem/IndoMMLU, CSV asli)",
     "task_type": "Educational Concept Explanation & Science QA",
     "tuning_instruction": "Fokus pada gaya penjelasan akademis/edukatif yang mudah dipahami, penguraian konsep sains/teknologi secara terstruktur, analogi, dan penjelasan bertahap."},
    {"key": "IndoCareer", "category": "text", "label": "ringan", "access": "rows",
     "dataset": "indolem/IndoCareer", "config": "all", "split": "test",
     "builder": _build_indocareer_text, "id_key": None,
     "source_name": "IndoCareer (HuggingFace: indolem/IndoCareer, all/test)",
     "task_type": "Professional QA & Career Counseling",
     "tuning_instruction": "Fokus pada penjelasan konteks profesional, persiapan ujian sertifikasi, konseling karir, dan pengetahuan bidang hukum/keuangan/medis."},
    {"key": "IndoCulture", "category": "text", "label": "ringan", "access": "rows",
     "dataset": "indolem/IndoCulture", "config": "default", "split": "test",
     "builder": _build_indoculture_text, "id_key": None,
     "source_name": "IndoCulture (HuggingFace: indolem/IndoCulture, test)",
     "task_type": "Cultural QA & Local Knowledge",
     "tuning_instruction": "Fokus pada pengetahuan kebudayaan Indonesia, adat istiadat daerah, kuliner, seni, dan Kearifan lokal."},
    {"key": "FineWeb-Edu-25K", "category": "text", "label": "berat", "access": "rows",
     "dataset": "irfanfadhullah/FineWeb-Edu-25K", "config": "default", "split": "train",
     "text_key": "text_indonesian", "id_key": None,
     "source_name": "FineWeb-Edu-25K (HuggingFace: irfanfadhullah/FineWeb-Edu-25K)",
     "task_type": "Educational Concept Explanation & Science QA",
     "tuning_instruction": "Fokus pada gaya penjelasan akademis/edukatif yang mudah dipahami, penguraian konsep sains/teknologi secara terstruktur, analogi, dan penjelasan bertahap."},
    {"key": "OpenWebText-10k", "category": "text", "label": "berat", "access": "rows",
     "dataset": "irfanfadhullah/OpenWebText-Indonesia-10k", "config": "default", "split": "train",
     "text_key": "translated_text", "id_key": None,
     "source_name": "OpenWebText-Indonesia-10k (HuggingFace: irfanfadhullah/OpenWebText-Indonesia-10k)",
     "task_type": "News Summarization & Public Policy Discussion",
     "tuning_instruction": "Fokus pada perangkuman berita, diskusi isu sosial/ekonomi, analisis kebijakan publik, serta evaluasi dampak bagi masyarakat."},
    {"key": "Indonesian-Simple-Summaries", "category": "text", "label": "berat", "access": "rows",
     "dataset": "irfanfadhullah/indonesian-simple-summaries", "config": "default", "split": "train",
     "builder": _build_simple_summaries_text, "id_key": None,
     "source_name": "Indonesian Simple Summaries (HuggingFace: irfanfadhullah/indonesian-simple-summaries)",
     "task_type": "News Summarization & Text Simplification",
     "tuning_instruction": "Fokus pada perangkuman berita, penyederhanaan teks, dan pembelajaran konseptual yang mudah dipahami."},
    {"key": "QQPR-Triplets-ID", "category": "text", "label": "berat", "access": "rows",
     "dataset": "robinsyihab/QQPR-triplets-ID", "config": "default", "split": "train",
     "builder": _build_qqpr_text, "id_key": None,
     "source_name": "QQPR-Triplets-ID (HuggingFace: robinsyihab/QQPR-triplets-ID)",
     "task_type": "Paraphrase Generation & Text Similarity",
     "tuning_instruction": "Fokus pada generasi paraphrasing, identifikasi kemiripan teks, dan penjelasan perbedaan makna antar kalimat."},
    {"key": "Indonesian-Wikipedia", "category": "text", "label": "berat", "access": "rows",
     "dataset": "indonesian-nlp/wikipedia-id", "config": "default", "split": "train",
     "builder": _build_wikipedia_text, "id_key": None,
     "source_name": "Indonesian Wikipedia (HuggingFace: indonesian-nlp/wikipedia-id)",
     "task_type": "General Knowledge & Encyclopedia QA",
     "tuning_instruction": "Fokus pada penjelasan pengetahuan umum, informasi ensiklopedia, fakta sejarah, sains, dan geografi Indonesia."},
    {"key": "Cendol-Collection-v2", "category": "text", "label": "berat", "access": "rows",
     "dataset": "indonlp/cendol_collection_v2", "config": "default", "split": "train",
     "builder": _build_cendol_text, "id_key": "prompt_id",
     "source_name": "Cendol Collection v2 (HuggingFace: indonlp/cendol_collection_v2)",
     "task_type": "General Instruction Following & Knowledge QA",
     "tuning_instruction": "Fokus pada instruction following, pengetahuan umum, dan percakapan sehari-hari dalam bahasa Indonesia."},
    {"key": "LFQA-ID", "category": "text", "label": "berat", "access": "rows",
     "dataset": "indonesian-nlp/lfqa_id", "config": "default", "split": "train",
     "builder": _build_lfqa_text, "id_key": "q_id",
     "source_name": "LFQA-ID (HuggingFace: indonesian-nlp/lfqa_id)",
     "task_type": "Long-Form QA & Explanatory Dialogue",
     "tuning_instruction": "Fokus pada jawaban panjang yang mendidik, penjelasan konsep dengan analogi, dan percakapan tanya-jawab mendalam ala forum."},
    # ── VISION ──
    {"key": "CVQA-Indonesia", "category": "vision", "label": "berat", "access": "offsets",
     "dataset": "afaji/cvqa", "config": "default", "split": "test",
     "offsets_cols": ["Subset", "ID"],
     "offsets_keep": lambda s: "indonesian" in str(s).lower(),
     "builder": _build_cvqa_id_text, "id_key": "ID", "image": "src", "approx_rows": 412,
     "source_name": "CVQA Subset Indonesia (HuggingFace: afaji/cvqa, subset Indonesian)",
     "task_type": "Cultural Visual QA (soal Bahasa Indonesia)",
     "tuning_instruction": "Fokus pada tanya-jawab berbasis budaya Indonesia: landmark, makanan, pakaian, sejarah, dan kehidupan sehari-hari. Jelaskan alasan jawaban dengan detail."},
    {"key": "KTP-VLM", "category": "vision", "label": "ringan", "access": "rows",
     "dataset": "danielsyahputra/ktp-vlm-instruct-dataset", "config": "default", "split": "train",
     "builder": _build_ktp_text, "id_key": None, "image": "src",
     "source_name": "KTP VLM Instruct Dataset (HuggingFace: danielsyahputra/ktp-vlm-instruct-dataset)",
     "disclaimer": ("Gambar KTP adalah data SAH milik perusahaan untuk membangun dataset training model bahasa "
                    "— bukan data pribadi orang lain yang kamu akses tanpa izin. Tugasmu hanya mengekstraksi isi KTP "
                    "untuk percakapan sintetis (OCR) dan menyimpannya LOKAL; tidak ada pengiriman data ke luar. "
                    "JANGAN menolak memproses gambar KTP dengan alasan privasi/data pribadi."),
     "task_type": "Document OCR & KTP Identity Extraction",
     "tuning_instruction": "Fokus pada ekstraksi data identitas KTP (NIK, Nama, Tempat/Tgl Lahir, Alamat, Agama, Status), penjelasan field identitas, dan analisis dokumen visual."},
    {"key": "SEA-VL-KORIKA-ID", "category": "vision", "label": "berat", "access": "rows",
     "dataset": "KORIKA-AI/sea-vl_crowdsourcing_id", "config": "default", "split": "train",
     "builder": _build_sea_vl_text, "id_key": None, "image": "src",
     "source_name": "KORIKA SEA-VL Crowdsourcing ID (HuggingFace: KORIKA-AI/sea-vl_crowdsourcing_id)",
     "task_type": "Image Captioning & Cultural Scene Understanding",
     "tuning_instruction": "Fokus pada deskripsi gambar budaya Asia Tenggara dalam Bahasa Indonesia: bangunan, kuliner, pakaian tradisional, kegiatan sehari-hari, dan landmark. Jelaskan detail visual dengan natural."},
    {"key": "SEA-VL-SEACrowd-ID", "category": "vision", "label": "berat", "access": "offsets",
     "dataset": "SEACrowd/sea-vl_crowdsourcing", "config": "default", "split": "train",
     "offsets_cols": ["native_lang", "id"],
     "offsets_keep": lambda s: "ind" in str(s).lower(),
     "builder": _build_sea_vl_text, "id_key": "id", "image": "src", "approx_rows": 7010,
     "source_name": "SEACrowd SEA-VL Crowdsourcing (HuggingFace: SEACrowd/sea-vl_crowdsourcing, filter ind)",
     "task_type": "Image Captioning & Cultural Scene Understanding",
     "tuning_instruction": "Fokus pada deskripsi gambar budaya Asia Tenggara dalam Bahasa Indonesia: bangunan, kuliner, pakaian tradisional, kegiatan sehari-hari, dan landmark. Jelaskan detail visual dengan natural."},
    {"key": "IndoRad-VQA-train", "category": "vision", "label": "berat", "access": "rows",
     "dataset": "Lab-IS/IndoRad-VQA", "config": "default", "split": "train",
     "builder": _build_indorad_text, "id_key": None, "image": "src",
     "source_name": "IndoRad-VQA (HuggingFace: Lab-IS/IndoRad-VQA, train)",
     "task_type": "Medical VQA & Radiology Image Analysis",
     "tuning_instruction": "Fokus pada analisis citra radiologi (X-ray, CT, MRI): identifikasi kelainan, jawaban ya/tidak terstruktur, dan penjelasan medis sederhana dalam Bahasa Indonesia."},
]


# ─── Exclude helper ────────────────────────────────────────────────────────────

SOURCE_ROW_RE = re.compile(r"^(?P<name>.+), Row #(?P<idx>\d+)$")
SOURCE_ID_RE = re.compile(r"^(?P<name>.+), id=(?P<id>.+)$")


def load_excluded_sources(paths: List[Path]) -> Tuple[Dict[str, set], Dict[str, set]]:
    excluded_rows: Dict[str, set] = {}
    excluded_ids: Dict[str, set] = {}
    for p in paths:
        path = Path(p)
        if not path.exists():
            print(f"      ⚠️ File exclude tidak ditemukan: {path}", file=sys.stderr)
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


# ─── Datasets-server (random access) — PERSIS MCPB ────────────────────────────

_COUNT_CACHE: Dict[Tuple, int] = {}
_CSV_CACHE: Dict[str, Any] = {}
_OFFSETS_CACHE: Dict[str, List[Tuple[int, str]]] = {}


def _ds_retry(path_query: str, attempts: int = 3, base_delay: float = 2.5) -> Dict[str, Any]:
    headers = {"User-Agent": "generate-conv-extra-20260802"}
    token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
    if token:
        headers["Authorization"] = f"Bearer {token}"
    url = f"{DS_API}{path_query}"
    last: Optional[Exception] = None
    for i in range(attempts):
        try:
            req = urllib.request.Request(url, headers=headers)
            with urllib.request.urlopen(req, timeout=30) as resp:
                payload = json.loads(resp.read().decode("utf-8"))
            err = payload.get("error")
            if not err:
                return payload
            last = _DataSourceError(str(err))
            if "index is loading" not in str(err).lower():
                break
        except urllib.error.HTTPError as e:
            last = _DataSourceError(f"HTTP {e.code} {e.reason}: {path_query}")
            if e.code != 429 and e.code < 500:
                break
        except (urllib.error.URLError, TimeoutError, OSError) as e:
            last = e
        if i < attempts - 1:
            time.sleep(base_delay * (i + 1))
    raise last if isinstance(last, Exception) else _DataSourceError(f"datasets-server gagal: {path_query}")


def _count_rows(spec: Dict[str, Any]) -> int:
    key = (spec["dataset"], spec["config"], spec["split"])
    if key in _COUNT_CACHE:
        return _COUNT_CACHE[key]
    q = urlencode({"dataset": spec["dataset"], "config": spec["config"]})
    payload = _ds_retry(f"/size?{q}")
    n = next((s.get("num_rows") for s in payload.get("splits", []) if s.get("split") == spec["split"]), None)
    if not n:
        q = urlencode({"dataset": spec["dataset"], "config": spec["config"],
                       "split": spec["split"], "offset": 0, "length": 1})
        n = _ds_retry(f"/rows?{q}").get("num_rows_total")
    if not n:
        raise _DataSourceError(f"/size tidak mengenali split {spec['split']} utk {spec['key']}")
    _COUNT_CACHE[key] = n
    return n


def _fetch_row_at(spec: Dict[str, Any], offset: int) -> Dict[str, Any]:
    q = urlencode({"dataset": spec["dataset"], "config": spec["config"], "split": spec["split"],
                   "offset": offset, "length": 1})
    payload = _ds_retry(f"/rows?{q}")
    rows = payload.get("rows") or []
    if not rows:
        raise _DataSourceError(f"datasets-server mengembalikan row kosong di offset {offset} ({spec['key']})")
    return rows[0]["row"]


def _pick_offset(rng: Any, num_rows: int, excluded: set) -> int:
    for _ in range(40):
        off = rng.randint(0, num_rows - 1)
        if off not in excluded:
            return off
    allowed = [i for i in range(num_rows) if i not in excluded]
    if not allowed:
        raise ValueError("Semua row sumber ini sudah dipakai (excluded).")
    return rng.choice(allowed)


def _source_weight(spec: Dict[str, Any]) -> int:
    if spec["access"] == "rows":
        try:
            return max(1, _count_rows(spec))
        except Exception:
            pass
    return max(1, int(spec.get("approx_rows", 1000) or 1000))


def _pick_source(specs: List[Dict[str, Any]]) -> Dict[str, Any]:
    weights = [_source_weight(s) for s in specs]
    return random.choices(specs, weights=weights, k=1)[0]


def _csv_dataset(spec: Dict[str, Any]) -> Any:
    key = spec["dataset"]
    if key in _CSV_CACHE:
        return _CSV_CACHE[key]
    cache_dir = Path.home() / ".cache" / "generate-conv-indonesia"
    cache_dir.mkdir(parents=True, exist_ok=True)
    path = cache_dir / "IndoMMLU.csv"
    if not path.exists():
        print(f"⬇️ Mengunduh CSV asli {spec['dataset']} → {path}", file=sys.stderr)
        req = urllib.request.Request(spec["csv_url"], headers={"User-Agent": "generate-conv-extra-20260802"})
        with urllib.request.urlopen(req, timeout=120) as resp, open(path, "wb") as f:
            f.write(resp.read())
    ds = datasets.load_dataset("csv", data_files=str(path), split="train")
    _CSV_CACHE[key] = ds
    return ds


def _offsets_for(spec: Dict[str, Any]) -> List[Tuple[int, str]]:
    key = spec["key"]
    if key in _OFFSETS_CACHE:
        return _OFFSETS_CACHE[key]
    ex = ThreadPoolExecutor(max_workers=1)
    try:
        fut = ex.submit(_build_offset_map, spec)
        try:
            result = fut.result(timeout=180)
        except Exception as e:
            raise _DataSourceError(f"{key}: pembacaan parquet columnar timeout/gagal: {e}") from None
    finally:
        ex.shutdown(wait=False)
    print(f"✅ {key}: {len(result)} baris subset (offset map pyarrow, tanpa unduh penuh)", file=sys.stderr)
    _OFFSETS_CACHE[key] = result
    return result


def _build_offset_map(spec: Dict[str, Any]) -> List[Tuple[int, str]]:
    import fsspec
    import pyarrow.parquet as pq

    q = urlencode({"dataset": spec["dataset"]})
    payload = _ds_retry(f"/parquet?{q}", attempts=4, base_delay=3.0)
    files = payload.get("parquet_files", [])
    if not files:
        raise _DataSourceError(f"/parquet kosong untuk {spec['key']}")
    fs = fsspec.filesystem("https")
    cols = spec["offsets_cols"]
    label_col, id_col = cols[0], cols[1]
    keep = spec["offsets_keep"]
    result: List[Tuple[int, str]] = []
    base = 0
    for f in files:
        url = f.get("url")
        if not url:
            continue
        try:
            table = pq.read_table(url, filesystem=fs, columns=cols)  # type: ignore[arg-type]
        except Exception as e:
            raise _DataSourceError(f"{spec['key']}: parquet gagal dibaca columnar: {e}") from e
        vals = table[label_col].to_pylist()
        ids = table[id_col].to_pylist()
        for i, v in enumerate(vals):
            if keep(v):
                result.append((base + i, str(ids[i])))
        base += table.num_rows
    if not result:
        raise _DataSourceError(f"Tidak ada baris subset untuk {spec['key']}")
    return result


def _sample_subset_reject(spec: Dict[str, Any], seed: Optional[int], fallback_err: str = "") -> Dict[str, Any]:
    rng = random.Random(seed) if seed is not None else random
    total = _count_rows(spec)
    keep = spec["offsets_keep"]
    label_col, id_col = spec["offsets_cols"][0], spec["offsets_cols"][1]
    ex_ids = _EXCLUDED_IDS.get(spec["source_name"], set())
    for _ in range(40):
        off = rng.randint(0, total - 1)
        row = _fetch_row_at(spec, off)
        if not keep(row.get(label_col)):
            time.sleep(0.35)
            continue
        idv = str(row.get(id_col, ""))
        if idv and idv in ex_ids:
            time.sleep(0.35)
            continue
        err_note = f" (offset map gagal: {fallback_err}) " if fallback_err else " "
        return _build_sample_result(
            spec, row, off, idv,
            note=f"Subset Indonesia via /rows rejection sampling{err_note}— vision tetap (dengan gambar).")
    raise _DataSourceError(f"{spec['key']}: tidak menemukan baris subset setelah 40 percobaan (fallback).")


def _image_src_from_row(row: Dict[str, Any]) -> Optional[str]:
    img = row.get("image")
    items = img if isinstance(img, list) else [img]
    for item in items:
        if isinstance(item, dict):
            src = item.get("src")
            if isinstance(src, str) and src:
                return src
    return None


def _build_context(spec: Dict[str, Any], row: Dict[str, Any]) -> str:
    if spec.get("builder"):
        raw = spec["builder"](row)
    elif spec.get("text_key"):
        raw = str(row.get(spec["text_key"], "")) or str(row)
    else:
        raw = str(row)
    return str(raw)[:1500]


def _build_sample_result(spec: Dict[str, Any], row: Dict[str, Any],
                         off: Optional[int], idv: Optional[str], note: str) -> Dict[str, Any]:
    raw = _build_context(spec, row)
    image_ref = _image_src_from_row(row)
    if not image_ref and spec["key"] == "KTP-VLM":
        image_ref = _ktp_image_url(row, off if off is not None else 0)
    if idv:
        source_detail = f"{spec['source_name']}, id={idv}"
    else:
        source_detail = f"{spec['source_name']}, Row #{off}"
    return {"spec": spec, "row": row, "source": source_detail, "raw_context": raw,
            "image_ref": image_ref, "note": note}


def _ktp_image_url(row: Dict[str, Any], row_idx: int) -> str:
    return f"https://huggingface.co/datasets/danielsyahputra/ktp-vlm-instruct-dataset/resolve/main/images/ktp_row_{row_idx}.jpg"


def _sample_from_spec(spec: Dict[str, Any], seed: Optional[int]) -> Dict[str, Any]:
    if spec["access"] == "offsets":
        return _sample_offsets(spec, seed)
    rng = random.Random(seed) if seed is not None else random
    name = spec["source_name"]
    ex_rows = _EXCLUDED_ROWS.get(name, set())
    ex_ids = _EXCLUDED_IDS.get(name, set())
    idv: Optional[str] = None
    off: Optional[int] = None
    row: Optional[Dict[str, Any]] = None
    for _ in range(30):
        if spec["access"] == "csv":
            ds = _csv_dataset(spec)
            off = _pick_offset(rng, len(ds), ex_rows)
            row = ds[off]
        else:
            off = _pick_offset(rng, _count_rows(spec), ex_rows)
            row = _fetch_row_at(spec, off)
        id_key = spec.get("id_key")
        if id_key:
            idv = str(row.get(id_key, ""))
            if idv and idv in ex_ids:
                continue
        break
    else:
        raise RuntimeError(f"Tidak ada row tersisa untuk {name} (semua sudah terpakai).")
    if spec["access"] == "csv":
        note = "IndoMMLU via CSV asli lokal (cache sekali), random index."
    else:
        note = "Random-access datasets-server (/rows) — 1 HTTP call, tanpa unduh dataset penuh."
    return _build_sample_result(spec, row, off, idv, note)


def _sample_offsets(spec: Dict[str, Any], seed: Optional[int]) -> Dict[str, Any]:
    rng = random.Random(seed) if seed is not None else random
    try:
        entries = _offsets_for(spec)
        ex_ids = _EXCLUDED_IDS.get(spec["source_name"], set())
        candidates = [(off, idv) for off, idv in entries if idv not in ex_ids]
        if not candidates:
            raise _DataSourceError(f"Semua baris {spec['source_name']} sudah terpakai (excluded).")
        off, idv = rng.choice(candidates)
        row = _fetch_row_at(spec, off)
        return _build_sample_result(spec, row, off, idv,
                                    note="Subset Indonesia via offset map pyarrow (columnar) + /rows — vision dengan gambar.")
    except _DataSourceError as e:
        return _sample_subset_reject(spec, seed, fallback_err=str(e))


# ─── State & distribusi (PERSIS get_progress MCPB) ─────────────────────────────

class GenState:
    """Melacak kuota 2:1 + distribusi prefix & jumlah pasang dari file + yang dibuat."""

    def __init__(self, output_path: Path, text_target: int, vision_target: int):
        self.output_path = Path(output_path)
        self.text_target = max(0, text_target)
        self.vision_target = max(0, vision_target)
        self.prefix_counts: Dict[str, int] = {f"<unused{i}>": 0 for i in range(1, 7)}
        self.pairs_counts: Dict[str, int] = {"3": 0, "4": 0, "5": 0}
        self.text_done = 0
        self.vision_done = 0
        if self.output_path.exists():
            self._seed_from_file()
        self._lock = asyncio.Lock()

    def _seed_from_file(self) -> None:
        with open(self.output_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    item = json.loads(line)
                except json.JSONDecodeError:
                    continue
                cat = item.get("category", "")
                if cat == "text_nlu_chat":
                    self.text_done += 1
                elif cat == "vision_chat":
                    self.vision_done += 1
                np_ = item.get("num_pairs")
                self.pairs_counts[str(np_)] = self.pairs_counts.get(str(np_), 0) + 1
                for m in item.get("messages", []):
                    if m.get("role") != "assistant":
                        continue
                    content = str(m.get("content", ""))
                    mch = re.match(r"^(?:<unused[1-6]>)*<unused[1-6]>", content)
                    if mch:
                        for t in re.findall(r"<unused[1-6]>", mch.group(0)):
                            self.prefix_counts[t] = self.prefix_counts.get(t, 0) + 1

    def record(self, item: Dict[str, Any]) -> None:
        cat = item.get("category", "")
        if cat == "text_nlu_chat":
            self.text_done += 1
        elif cat == "vision_chat":
            self.vision_done += 1
        np_ = item.get("num_pairs")
        self.pairs_counts[str(np_)] = self.pairs_counts.get(str(np_), 0) + 1
        for m in item.get("messages", []):
            if m.get("role") != "assistant":
                continue
            content = str(m.get("content", ""))
            mch = re.match(r"^(?:<unused[1-6]>)*<unused[1-6]>", content)
            if mch:
                for t in re.findall(r"<unused[1-6]>", mch.group(0)):
                    self.prefix_counts[t] = self.prefix_counts.get(t, 0) + 1

    async def next_category(self) -> Optional[str]:
        async with self._lock:
            text_left = self.text_target - self.text_done
            vision_left = self.vision_target - self.vision_done
            if text_left <= 0 and vision_left <= 0:
                return None
            if text_left <= 0:
                return "vision"
            if vision_left <= 0:
                return "text"
            tr = self.text_done / self.text_target if self.text_target else 1.0
            vr = self.vision_done / self.vision_target if self.vision_target else 1.0
            return "text" if tr <= vr else "vision"

    def prefix_hint(self) -> str:
        total = sum(self.prefix_counts.values())
        if total == 0:
            return ("Belum ada percakapan tersimpan — pilih prefix sesuai task isi pesan, "
                    "dan VARIASIKAN (jangan semua <unused4>).")
        ordered = sorted(self.prefix_counts.items(), key=lambda kv: kv[1])
        least = [k for k, v in ordered if v == ordered[0][1]]
        most = ordered[-1][1]
        if most > 0 and ordered[0][1] >= most * 0.5:
            return "Distribusi prefix sudah cukup seimbang — lanjutkan memilih sesuai task."
        names = ", ".join(f"{k} {_PREFIX_LABEL[k]}" for k in least[:3])
        return (f"Prefix yang masih JARANG dipakai: {names}. Tulis percakapan yang memang butuh task itu "
                f"supaya distribusi <unused1..6> seimbang di seluruh dataset — tetap sesuaikan dengan isi pesan.")

    def pairs_hint(self) -> str:
        total = sum(self.pairs_counts.values())
        if total == 0:
            return "Belum ada percakapan — VARIASIKAN jumlah pasang: 3, 4, atau 5 (6/8/10 pesan)."
        least_val = min(self.pairs_counts.values())
        most_val = max(self.pairs_counts.values())
        if most_val > 0 and least_val >= most_val * 0.5:
            return "Distribusi jumlah pasang (3/4/5) sudah seimbang — lanjutkan sesuai kebutuhan konten."
        least = [k for k, v in self.pairs_counts.items() if v == least_val]
        names = " dan ".join(f"{k} pasang ({int(k) * 2} pesan)" for k in least)
        return (f"Jumlah pasang yang masih JARANG: {names}. Tulis percakapan dengan jumlah pasang itu "
                f"supaya variasi 3/4/5 (6/8/10 pesan) seimbang.")

    def suggest_num_pairs(self) -> int:
        # Pilih jumlah pasang yang paling jarang (3/4/5) — deterministik balancing.
        least = min(self.pairs_counts, key=lambda k: self.pairs_counts[k])
        return int(least)


# ─── Rate Limiter (RPM & TPM) ──────────────────────────────────────────────────

class RateLimiter:
    """Token bucket async: RPM (request/menit) & TPM (tokens/menit), refill kontinu."""

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


# ─── Prompt builder (PERSIS MCPB v2.1.18) ─────────────────────────────────────

def build_prompt_for_conversation(meta: Dict[str, Any], raw_context: str, source_detail: str,
                                  num_pairs: int, prefix_hint: str = "", pairs_hint: str = "",
                                  is_vision: bool = False, disclaimer: str = "",
                                  min_len: int = 60) -> str:
    total_msgs = num_pairs * 2
    vision_note = (" (Catatan: Ini adalah percakapan visual. Turn user pertama HARUS dimulai dengan token "
                   "📷 yang menandai adanya gambar. Seluruh dialog HARUS NYAMBUNG 100% dengan objek/teks asli "
                   "yang ada di dalam gambar).") if is_vision else ""
    etika = f"\n\nCATATAN ETIKA (khusus KTP):\n{disclaimer}" if disclaimer else ""
    return f"""Buatkan 1 percakapan multi-turn sintetis Bahasa Indonesia antara User dan Gemma (Assistant) sebanyak TEPAT {num_pairs} pasang dialog (total TEPAT {total_msgs} pesan selang-seling, turn pertama selalu USER).

Konteks / Baris Dataset Asli dari HuggingFace:
- Sumber Metadata Detail: {source_detail}
- Kategori Task: {meta['task_type']}
- Instruksi Tuning Khusus Sumber: {meta['tuning_instruction']}
- Isi Teks Mentah Baris Asli:
\"\"\"
{raw_context}
\"\"\"
{etika}

CARA KERJA (kerjakan URUTAN ini SEBELUM menulis output):
1. BACA & PAHAMI teks mentah asli di atas. Identifikasi 2-3 FAKTA KUNCI yang menjadi dasar percakapan.
2. SUSUN ALUR: turn user [1] membuka topik dari salah satu fakta kunci; turn user berikutnya adalah follow-up yang NATURAL menyambung jawaban sebelumnya (klarifikasi, minta contoh, gali aspek lain); turn terakhir mengeksplorasi aspek lain dari konteks yang sama atau penutup ringan. JANGAN menanyakan hal yang sama berulang dengan kata-kata berbeda.
3. REASONING PREFIX (untuk SETIAP turn assistant): (a) baca pertanyaan user, (b) tentukan TASK dominan jawaban yang akan ditulis, (c) pilih 1-3 token yang TEPAT. Pemetaan: meringkas teks → <unused1> SUMMARIZE; menerjemahkan → <unused2> TRANSLATE; mengekstrak entitas/identitas → <unused3> NER; menjawab pertanyaan → <unused4> QA; memparafrase → <unused5> PARAPHRASE; basa-basi/obrolan umum → <unused6> GENERAL_CHAT. Task berbeda antar turn → prefix WAJIB berbeda; task sama boleh sama.
4. PERIKSA ULANG sebelum menyerahkan: setiap turn assistant BENAR-BENAR menjawab pertanyaan user (bukan jawaban lepas), semua klaim konsisten dengan fakta kunci, dan TIDAK ADA kalimat/frasa yang diulang persis di turn lain.

ATURAN WAJIB:
1. JUMLAH PESAN: TEPAT {total_msgs} pesan ({num_pairs} pasang user-assistant, selang-seling). Validator MENOLAK bila bukan 6/8/10 pesan.
2. KEDALAMAN JAWABAN: setiap turn assistant MINIMAL {min_len} karakter dan berisi ELABORASI yang benar-benar menjawab pertanyaan: berikan alasan, langkah, contoh, atau analogi. JANGAN sekadar mengulang kunci jawaban/teks mentah, JANGAN jawaban satu kalimat, dan JANGAN mengulang kalimat yang sama persis (atau hampir persis) di turn lain.
3. VARIASI PEMBUKA: jangan memulai setiap jawaban assistant dengan kata/frasa yang sama berulang (mis. 'Benar,', 'Tentu,', 'Ya,', 'Berdasarkan konteks,') — variasikan struktur kalimat pembuka.
4. `prefixes` (role assistant): 1-3 token UNIK (tanpa spasi/duplikat) hasil REASONING langkah 3. Khusus role user: prefixes wajib [].
5. DILARANG menulis token <unusedX> atau kata nama task (SUMMARIZE/NER/dll) di dalam konten.
6. GROUNDING: seluruh turn konsisten dengan 2-3 FAKTA KUNCI dari teks mentah asli — JANGAN menambah fakta di luar konteks. Bila konteks tidak memuat informasi yang ditanyakan, jawab dengan menyebut keterbatasan konteks secara wajar (jangan mengarang).
7. GAYA TURN USER (PENTING): turn user ditulis seperti MANUSIA biasa yang bertanya — variasi tipe pertanyaan (open-ended, klarifikasi, minta contoh, pengecekan ulang), variasi panjang kalimat, bahasa sehari-hari natural, jangan memulai semua turn dengan pola sama ("Halo"/"Tolong"/"Bisakah kamu"), jangan menulis dari sudut pandang AI di turn user, dan pertanyaan follow-up nyambung natural dari jawaban sebelumnya.
8. DISTRIBUSI PREFIX (lintas percakapan): {prefix_hint}
9. VARIASI JUMLAH PASANG (lintas percakapan): {pairs_hint}
10. Percakapan 100% nyambung dengan isi teks mentah asli.{vision_note}"""


# ─── Format akhir (sanitasi PERSIS MCPB) ───────────────────────────────────────

def _format_final(validated: ConversationOutput, is_vision: bool) -> List[Dict[str, str]]:
    formatted = [{"role": "system", "content": SYSTEM_PROMPT}]
    for idx, msg in enumerate(validated.conversations):
        content_clean = msg.content.replace("\\n", "\n").strip()
        for leak in ["SUMMARIZE", "TRANSLATE", "NER", "QA", "PARAPHRASE", "GENERAL_CHAT"]:
            content_clean = re.sub(rf"^\s*{leak}\s*", "", content_clean, flags=re.IGNORECASE)
        # Strip token <unusedX> di SELURUH konten (token hanya sah di field prefixes)
        content_clean = re.sub(r"<unused[1-6]>", "", content_clean)
        # Runtuhkan spasi ganda, tapi PERTAHANKAN newline (format JSON/code block tidak dirusak)
        content_clean = re.sub(r"[ \t]{2,}", " ", content_clean)
        content_clean = re.sub(r"\n{3,}", "\n\n", content_clean).strip()
        if msg.role == "user":
            if is_vision and idx == 0 and not content_clean.startswith("📷"):
                content_clean = "📷\n" + content_clean
            formatted.append({"role": "user", "content": content_clean})
        else:
            tokens = re.findall(r"<unused[1-6]>", "".join(msg.prefixes))
            seen: List[str] = []
            for t in tokens:
                if t not in seen:
                    seen.append(t)
            prefix_str = "".join(seen) if seen else "<unused4>"
            formatted.append({"role": "assistant", "content": f"{prefix_str} {content_clean}".strip()})
    return formatted


def _download_vision_image(image_ref: str, fname_base: str) -> str:
    """Unduh gambar vision dari URL src → data/multimodal/images/ (fallback: tetap URL)."""
    try:
        VISION_IMAGE_DIR.mkdir(parents=True, exist_ok=True)
        path = VISION_IMAGE_DIR / f"{fname_base}.jpg"
        if not path.exists() and image_ref.startswith(("http://", "https://")):
            req = urllib.request.Request(image_ref, headers={"User-Agent": "generate-conv-extra-20260802"})
            with urllib.request.urlopen(req, timeout=60) as resp:
                PILImage.open(io.BytesIO(resp.read())).convert("RGB").save(path, "JPEG", quality=90)
        if path.exists():
            return str(path)
    except Exception:
        pass
    return image_ref


def _render_draft(convs: List[TurnMessage]) -> str:
    lines = []
    for i, m in enumerate(convs, 1):
        if m.role == "user":
            lines.append(f"[{i}] USER: {m.content}")
        else:
            pfx = "".join(m.prefixes) if m.prefixes else "<unused4>"
            lines.append(f"[{i}] ASSISTANT ({pfx}): {m.content}")
    return "\n".join(lines)


def _count_edited_turns(a: List[TurnMessage], b: List[TurnMessage]) -> int:
    return sum(1 for ma, mb in zip(a, b) if ma.role != mb.role or ma.content != mb.content or ma.prefixes != mb.prefixes)


def _repetition_issues(convs: List[TurnMessage]) -> List[int]:
    """Indeks turn assistant yang mengulang kalimat (>=20 char, dinormalisasi) persis di turn lain."""
    seen: Dict[str, int] = {}
    issues: List[int] = []
    for i, m in enumerate(convs):
        if m.role != "assistant":
            continue
        for sent in re.split(r"(?<=[.!?])\s+", m.content):
            norm = re.sub(r"\s+", " ", sent).strip().lower()
            if len(norm) >= 20:
                if norm in seen:
                    issues.append(i)
                else:
                    seen[norm] = i
    return sorted(set(issues))


def build_review_prompt(meta: Dict[str, Any], raw_context: str, source_detail: str,
                        draft_convs: List[TurnMessage], num_pairs: int,
                        is_vision: bool = False, min_len: int = 60) -> str:
    total_msgs = num_pairs * 2
    draft_text = _render_draft(draft_convs)
    return f'''Kamu adalah REVIEWER ketat untuk percakapan sintetis Bahasa Indonesia (data training model Gemma).
Analisa DRAFT percakapan turn demi turn terhadap konteks sumber asli, lalu keluarkan laporan terstruktur.

Konteks Sumber Asli:
- Sumber: {source_detail}
- Kategori Task: {meta['task_type']}
- Instruksi Tuning: {meta['tuning_instruction']}
- Isi Teks Mentah Asli:
"""
{raw_context}
"""

DRAFT PERCAKAPAN (wajib TEPAT {total_msgs} pesan = {num_pairs} pasang):
"""
{draft_text}
"""

CHECKLIST (periksa SATU PER SATU, bandingkan DENGAN KONTEKS di atas):
1. GROUNDING KETAT: setiap klaim/fakta di draft harus ADA di isi teks mentah asli — bandingkan satu per satu. Klaim di luar konteks = error (halusinasi). Pengecualian: menyebut keterbatasan konteks dengan wajar itu BOLEH.
2. KEDALAMAN & REPETISI: setiap turn assistant harus BENAR-BENAR menjawab pertanyaan user dengan elaborasi (alasan/langkah/contoh/analogi), BUKAN sekadar mengulang kunci jawaban atau teks mentah. Deteksi kalimat/frasa yang diulang persis atau hampir persis antar turn — termasuk pembuka jawaban yang sama berulang ('Benar,', 'Tentu,', 'Berdasarkan konteks,') → laporkan sebagai error.
3. Prefix turn assistant TEPAT & sesuai isi pesan: <unused1> SUMMARIZE, <unused2> TRANSLATE, <unused3> NER, <unused4> QA, <unused5> PARAPHRASE, <unused6> GENERAL_CHAT. 1-3 token UNIK; bila task antar turn berbeda, prefix WAJIB berbeda (jangan semua <unused4>).
4. Panjang: setiap turn assistant minimal {min_len} karakter.
5. GAYA TURN USER: seperti manusia biasa — variasi tipe pertanyaan & panjang kalimat, bukan robot/AI, bukan sudut pandang asisten, tidak ada pola pembuka yang sama berulang.
6. TIDAK ada kebocoran kata nama task (SUMMARIZE/TRANSLATE/NER/QA/PARAPHRASE/GENERAL_CHAT) atau token <unusedX> di dalam konten.
7. Struktur: turn selang-seling user/assistant, turn pertama user{', user pertama diawali 📷' if is_vision else ''}, jumlah pesan TEPAT {total_msgs}.
8. Bahasa Indonesia natural dan sesuai Instruksi Tuning.

Aturan laporan:
- Untuk setiap masalah: beri turn_index (0-based; -1 untuk masalah global), severity (error/warning), problem, suggestion.
- approved=true HANYA bila tidak ada masalah severity "error".
- Jangan membuat masalah yang tidak ada — hanya laporkan yang benar-benar melanggar checklist.'''


# ─── Generate satu percakapan (single-stage, pydantic-ai retries) ─────────────

async def generate_single_conversation(
    writer: Agent[Any, ConversationOutput],
    reviewer: Optional[Agent[Any, ReviewReport]],
    state: GenState,
    category_mode: str,
    num_pairs: int,
    limiter: Optional[RateLimiter] = None,
    max_manual_retry: int = 2,
    max_review_rounds: int = 2,
    quick: bool = False,
) -> Optional[Dict[str, Any]]:
    """Writer → Reviewer (sub-agent) → Revisi (loop).

    - pydantic-ai `retries=3` otomatis mengulang bila output gagal skema Pydantic.
    - Panjang minimal turn assistant (40 vision / 60 text) dicek setelah run;
      bila kurang, error di-append & diulang (maks max_manual_retry).
    - Reviewer menilai draft dengan checklist; bila ada error, writer merevisi
      dengan laporan reviewer (maks max_review_rounds).
    """
    specs = [s for s in SOURCES if s["category"] == category_mode and (not quick or s["label"] == "ringan")]
    if not specs:
        print(f"⚠️ Tidak ada sumber untuk kategori {category_mode}" + (" (pool ringan)" if quick else ""), file=sys.stderr)
        return None
    spec = _pick_source(specs)
    sampled = await asyncio.to_thread(_sample_from_spec, spec, None)
    is_vision = category_mode == "vision"
    min_len = 40 if is_vision else 60
    prefix_hint = state.prefix_hint()
    pairs_hint = state.pairs_hint()
    base_prompt = build_prompt_for_conversation(
        spec, sampled["raw_context"], sampled["source"], num_pairs,
        prefix_hint=prefix_hint, pairs_hint=pairs_hint,
        is_vision=is_vision, disclaimer=spec.get("disclaimer", ""), min_len=min_len)

    async def run_agent(agent: Agent[Any, Any], p_text: str):
        if limiter is not None:
            est = max(1, (len(p_text) + 12000) // 4)
            await limiter.acquire(est)
        return await agent.run(p_text)

    # ── Tahap 1: DRAFT (dengan retry manual utk panjang minimal) ──
    draft: Optional[ConversationOutput] = None
    prompt = base_prompt
    for _ in range(max_manual_retry + 1):
        result = await run_agent(writer, prompt)
        draft = result.output
        short = [i for i, m in enumerate(draft.conversations)
                 if m.role == "assistant" and len(m.content.strip()) < min_len]
        reps = _repetition_issues(draft.conversations)
        if not short and not reps:
            break
        notes = []
        if short:
            notes.append(f"turn assistant {short} terlalu pendek (minimal {min_len} karakter)")
        if reps:
            notes.append(f"turn assistant {reps} mengulang kalimat yang sama persis dengan turn lain")
        prompt = prompt + "\n\nVALIDASI GAGAL: " + "; ".join(notes) + ". Perbaiki masalah tersebut."
    if draft is None:
        return None

    # ── Tahap 2: REVIEW + REVISI (sub-agent) ──
    reviewed = reviewer is not None
    edited_turns = 0
    total_issues = 0
    final_convs: List[TurnMessage] = draft.conversations
    if reviewed:
        review_prompt = build_review_prompt(spec, sampled["raw_context"], sampled["source"],
                                            final_convs, num_pairs, is_vision=is_vision, min_len=min_len)
        for _round in range(max_review_rounds + 1):
            rep = (await run_agent(reviewer, review_prompt)).output
            total_issues += len(rep.issues)
            if rep.approved:
                break
            # Revisi: writer memperbaiki dengan laporan reviewer
            fix_notes = "\n".join(
                f"- turn {i.turn_index} [{i.severity}]: {i.problem} → saran: {i.suggestion}"
                for i in rep.issues[:8])
            fix_prompt = (f"{base_prompt}\n\nREVIEWER MENEMUKAN MASALAH (perbaiki SEMUA yang severity 'error'):\n"
                          f"{fix_notes}\n\nKembalikan VERSI FINAL yang sudah diperbaiki (tetap TEPAT {num_pairs} pasang = {num_pairs * 2} pesan).")
            new_result = await run_agent(writer, fix_prompt)
            new_convs = new_result.output.conversations
            edited_turns += _count_edited_turns(final_convs, new_convs)
            final_convs = new_convs
            review_prompt = build_review_prompt(spec, sampled["raw_context"], sampled["source"],
                                                final_convs, num_pairs, is_vision=is_vision, min_len=min_len)

    final_out = ConversationOutput(conversations=final_convs)
    formatted = _format_final(final_out, is_vision)
    result = {
        "source": sampled["source"],
        "category": "vision_chat" if is_vision else "text_nlu_chat",
        "num_turns": len(formatted),
        "num_pairs": len(final_convs) // 2,
        "reviewed": reviewed,
        "edited_turns": edited_turns,
        "review_issues": total_issues,
        "messages": formatted,
    }
    if is_vision and sampled.get("image_ref"):
        ident = re.sub(r"[^\w-]+", "_", sampled["source"])[-50:]
        result["images"] = [_download_vision_image(sampled["image_ref"], f"{spec['key']}_{ident}")]
    return result


# ─── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
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

    parser = argparse.ArgumentParser(
        description="Generate Synthetic Conversational Dataset EXTRA v2 — StepFun API via pydantic-ai, fitur PERSIS MCPB (random-access, kuota 2:1, distribusi prefix/pasang, validasi strict)")
    parser.add_argument("--target", type=int, default=20, help="Jumlah total percakapan (default: 20)")
    parser.add_argument("--limit", type=int, default=0, help="Alias --target untuk uji sampel kecil")
    parser.add_argument("--mode", type=str, choices=["text", "vision", "mixed"], default="mixed",
                        help="Mode data (dipakai bila --text-target/--vision-target tidak diberikan)")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT, help="Path file output JSONL")
    parser.add_argument("--base-url", type=str, default=default_base_url, help="OpenAI-compatible API Base URL")
    parser.add_argument("--model", type=str, default=default_model_name, help="Model name (misal: step-3.7-flash)")
    parser.add_argument("--concurrency", type=int, default=5, help="Jumlah concurrent API requests")
    parser.add_argument("--rpm", type=int, default=10, help="Rate limit: request per menit")
    parser.add_argument("--tpm", type=int, default=5_000_000, help="Rate limit: token per menit")
    parser.add_argument("--max-tokens", type=int, default=0, help="Batas token output per respons (0 = default model)")
    parser.add_argument("--quick", action="store_true", help="Hanya pool ringan (IndoMMLU/IndoCareer/IndoCulture/KTP-VLM)")
    parser.add_argument("--text-target", type=int, default=0, help="Kuota percakapan TEXT (0 = ikut --mode)")
    parser.add_argument("--vision-target", type=int, default=0, help="Kuota percakapan VISION (0 = ikut --mode)")
    parser.add_argument("--exclude-file", nargs="*", type=Path, default=[], help="File JSONL tambahan untuk exclude row/id")
    parser.add_argument("--overwrite", action="store_true", help="Mulai dari nol (timpa output); default append")
    parser.add_argument("--no-review", action="store_true",
                        help="Nonaktifkan sub-agent REVIEWER (hemat ~2x request API; default: review AKTIF)")
    args = parser.parse_args()

    global _EXCLUDED_ROWS, _EXCLUDED_IDS
    use_quota = args.text_target > 0 or args.vision_target > 0
    if use_quota:
        total_count = args.text_target + args.vision_target
    else:
        total_count = args.limit if args.limit > 0 else args.target

    print("\n=== Synthetic Dataset Generator EXTRA v2 (pydantic-ai / StepFun) — fitur MCPB ===")
    print(f"Mode      : {args.mode.upper()}")
    print(f"Target    : {total_count}" + (f"  (TEXT {args.text_target} / VISION {args.vision_target})" if use_quota else ""))
    print(f"Model     : {args.model} @ {args.base_url}")
    print(f"Output    : {args.output} ({'OVERWRITE' if args.overwrite else 'APPEND'})")
    print(f"Concurrency: {args.concurrency} | Rate: {args.rpm} req/mnt | {args.tpm:,} tok/mnt"
          + (f" | max_tokens={args.max_tokens}" if args.max_tokens > 0 else ""))
    print(f"Pool      : {'ringan (quick)' if args.quick else 'semua (15 sumber)'}")
    print(f"Review    : {'OFF (--no-review)' if args.no_review else 'ON (sub-agent reviewer, revisi otomatis)'}\n")

    args.output.parent.mkdir(parents=True, exist_ok=True)

    # Exclude: output file (resume) + --exclude-file (kecuali --overwrite)
    exclude_paths = list(args.exclude_file)
    if args.output.exists() and not args.overwrite:
        exclude_paths.append(args.output)
    if exclude_paths:
        _EXCLUDED_ROWS, _EXCLUDED_IDS = load_excluded_sources(exclude_paths)
        n_excluded = sum(len(v) for v in _EXCLUDED_ROWS.values()) + sum(len(v) for v in _EXCLUDED_IDS.values())
        if n_excluded:
            print(f"🚫 {n_excluded} row/id dikecualikan dari {len(exclude_paths)} file JSONL sebelumnya.")

    # ID global lanjut (append)
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

    provider = OpenAIProvider(base_url=args.base_url, api_key=api_key)
    pydantic_model = OpenAIChatModel(args.model, provider=provider)
    writer_settings = ModelSettings(max_tokens=args.max_tokens, temperature=0.6) if args.max_tokens > 0 else ModelSettings(temperature=0.6)
    writer = Agent(
        model=pydantic_model,
        output_type=ConversationOutput,
        retries=3,
        name="conv-writer",
        model_settings=writer_settings,
        instructions=(
            "Kamu adalah generator percakapan sintetis terstruktur yang mematuhi skema Pydantic. "
            "Sebelum menulis setiap turn assistant, lakukan REASONING task untuk memilih prefix <unused1..6> "
            "yang TEPAT (SUMMARIZE/TRANSLATE/NER/QA/PARAPHRASE/GENERAL_CHAT) dan variasikan bila task berbeda. "
            "Tulis turn USER seperti manusia biasa yang bertanya (variasi gaya & tipe pertanyaan, bukan robot/AI, "
            "bukan sudut pandang asisten) dan turn ASSISTANT sesuai persona Gemma dengan elaborasi "
            "(alasan/langkah/contoh/analogi), tanpa mengulang kalimat antar turn. "
            "Kamu HANYA mengembalikan token prefix murni seperti ['<unused1>'] tanpa pernah menyertakan "
            "teks kata 'SUMMARIZE' atau 'NER'."
        ),
    )
    reviewer: Optional[Agent[Any, ReviewReport]] = None
    if not args.no_review:
        reviewer = Agent(
            model=OpenAIChatModel(args.model, provider=provider),
            output_type=ReviewReport,
            retries=3,
            name="conv-reviewer",
            model_settings=ModelSettings(temperature=0.1),
            instructions=(
                "Kamu adalah REVIEWER ketat dan objektif untuk percakapan sintetis Bahasa Indonesia. "
                "Periksa setiap turn terhadap konteks sumber asli dan checklist yang diberikan — utamakan: "
                "grounding (klaim di luar konteks = error), kedalaman jawaban (bukan sekadar mengulang kunci "
                "jawaban), dan repetisi (kalimat/frasa yang diulang persis antar turn). "
                "Hanya laporkan masalah yang BENAR-BENAR ada (jangan mengarang). "
                "approved=true HANYA bila tidak ada masalah severity 'error'."
            ),
        )

    state = GenState(args.output, args.text_target if use_quota else (total_count if args.mode in ("text", "mixed") else 0),
                     args.vision_target if use_quota else (total_count if args.mode == "vision" else 0))
    limiter = RateLimiter(rpm=args.rpm, tpm=args.tpm)
    semaphore = asyncio.Semaphore(args.concurrency)
    write_lock = asyncio.Lock()
    generated_count = 0
    start_time = time.time()
    write_mode = "w" if args.overwrite else "a"

    async def worker(conv_id: int) -> None:
        nonlocal generated_count
        async with semaphore:
            if use_quota:
                category_mode = await state.next_category()
                if category_mode is None:
                    return
            elif args.mode == "mixed":
                category_mode = random.choice(["text", "vision"])
            else:
                category_mode = args.mode
            num_pairs = state.suggest_num_pairs()
            res = await generate_single_conversation(writer, reviewer, state, category_mode, num_pairs, limiter=limiter, quick=args.quick)
            if res is None:
                return
            res["id"] = existing_max_id + conv_id
            async with write_lock:
                with open(args.output, "a", encoding="utf-8") as out_file:
                    out_file.write(json.dumps(res, ensure_ascii=False) + "\n")
            state.record(res)
            generated_count += 1
            elapsed = time.time() - start_time
            print(f"[{generated_count}/{total_count}] ID {res['id']} ({res['category']}) | "
                  f"pairs {res['num_pairs']} | review={res.get('reviewed')} edited={res.get('edited_turns')} issues={res.get('review_issues', 0)} | "
                  f"{res['source']} | {elapsed:.0f}s", flush=True)

    try:
        tasks = [worker(i + 1) for i in range(total_count)]

        async def _run_all():
            await asyncio.gather(*tasks)
        asyncio.run(_run_all())
    finally:
        tt, vt = state.text_done, state.vision_done
        print(f"\n📊 Selesai: {tt} text + {vt} vision (rasio target 2:1)")
        print(f"   Distribusi prefix: {dict(sorted(state.prefix_counts.items()))}")
        print(f"   Distribusi pasang: {state.pairs_counts}")


_EXCLUDED_ROWS: Dict[str, set] = {}
_EXCLUDED_IDS: Dict[str, set] = {}


if __name__ == "__main__":
    main()
