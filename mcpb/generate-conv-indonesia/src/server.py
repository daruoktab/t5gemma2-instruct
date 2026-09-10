"""
MCP Bundle Server — Generator Percakapan Indonesia (Claude Desktop yang mengerjakan)
===================================================================================
Desain: server TIDAK memanggil model/API eksternal. Claude (Claude Desktop) yang
MENULIS percakapan; server menyediakan data konteks asli + validasi + penyimpanan.

Dibangun dengan MCP SDK low-level (`mcp.server.lowlevel.Server`, mcp>=2.0).

  Tools:
    - list_sources(category)        -> katalog sumber text/vision (INSTANT, statis)
    - sample_row(category, source)  -> 1 baris asli via RANDOM-ACCESS datasets-server
                                        (TANPA unduh dataset penuh; row yang sudah
                                        dipakai otomatis diskip/exclude). Vision:
                                        MENGEMBALIKAN GAMBAR langsung (ImageContent)
                                        — tanpa perlu read_image terpisah. category
                                        'auto' → pilih jenis sesuai kuota 2:1.
    - read_image(image_ref)         -> MULTIMODAL: ImageContent (base64 PNG)
                                        (fallback bila ingin lihat ulang / path lokal)
    - save_conversation(...)        -> validasi skema (Pydantic) + normalisasi prefix
                                        + format akhir + append JSONL
    - get_output_stats(path)        -> statistik file output JSONL
    - get_progress()                -> kuota target (default 2000 text / 1000 vision),
                                        jumlah sudah/belum, kategori berikutnya (2:1)
  Prompt:
    - generate_conversation         -> panduan langkah demi langkah untuk Claude

Akses data — "randomize dulu, stream row-nya saja":
  - Dataset yang viewable di datasets-server (14/16 sumber) dipanggil via endpoint
    `/rows?offset=<acak>&length=1` — 1 HTTP call, 1 row, file dataset TIDAK pernah
    diunduh penuh (termasuk Cendol 12,8 juta row).
  - CVQA-Indonesia & SEACrowd SEA-VL tetap MULTIMODAL: peta offset subset Indonesia
    dihitung via pyarrow columnar-read (baca kolom kecil Subset/native_lang + id
    dari semua shard parquet, tanpa mentransfer kolom gambar), lalu row + gambar
    diambil via `/rows` di offset terpilih.
  - IndoMMLU (loader Python custom, tidak viewable) -> CSV asli di-cache lokal sekali,
    lalu random index.

Jalankan:
    uv run --directory <bundle> src/server.py              # stdio (default, oleh Claude Desktop)
    python src/server.py --transport streamable-http --port 8765
"""

import argparse
import asyncio
import base64
import io
import json
import os
import random
import re
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any, Literal, cast
from urllib.parse import urlencode

from PIL import Image as PILImage

# Windows: pipa stdout/stderr default cp1252 tidak bisa encode emoji → crash UnicodeEncodeError.
# Paksa UTF-8 dengan fallback 'replace' agar log server tidak pernah gagal.
for _stream in (sys.stdout, sys.stderr):
    try:
        cast(io.TextIOWrapper, _stream).reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

# Pastikan direktori src masuk dalam sys.path saat dipanggil via uv run
_SRC_DIR = Path(__file__).resolve().parent
if str(_SRC_DIR) not in sys.path:
    sys.path.insert(0, str(_SRC_DIR))

from generate_conv_extra_20260731 import (
    SOURCE_ID_RE,
    SOURCE_ROW_RE,
    SYSTEM_PROMPT,
    ConversationOutput,
    TurnMessage,
    _build_cendol_text,
    _build_cvqa_id_text,
    _build_indocareer_text,
    _build_indoculture_text,
    _build_indommlu_csv_text,
    _build_indorad_text,
    _build_ktp_text,
    _build_lfqa_text,
    _build_qqpr_text,
    _build_sea_vl_text,
    _build_simple_summaries_text,
    _build_wikipedia_text,
    _ktp_image_url,
    load_excluded_sources,
)
from mcp import types
from mcp.server.lowlevel import Server
from mcp.server.stdio import stdio_server

# ─── Catatan tipe statis ───────────────────────────────────────────────────────
# Resolver checker (ty) memakai mcp 1.29 dari env unsloth-env, sedangkan bundle ini
# BERJALAN di mcp 2.0.0 (lihat uv.lock: mcp 2.0.0 + mcp-types 2.0.0). API low-level
# mcp 2.0 memakai nama parameter snake_case (input_schema, mime_type) dan
# Server(...) dengan handler kwargs; mcp 1.29 belum mengenalnya → semua error statis
# di bawah ini FALSE POSITIVE. Penanda suppress di akhir baris pemanggilan hanya
# menenangkan checker; runtime mcp 2.0.0 menerima semua call ini (sudah teruji).

# ─── State & Konfigurasi ──────────────────────────────────────────────────────
_CONFIG: dict[str, Any] = {}
_EXCLUDED_ROWS: dict[str, set] = {}   # source_name -> {row_idx}       (akses index)
_EXCLUDED_IDS: dict[str, set] = {}    # source_name -> {id_value}      (akses id)
_COUNT_CACHE: dict[tuple, int] = {}   # (dataset, config, split) -> num_rows
_CSV_CACHE: dict[str, Any] = {}       # dataset -> Dataset (IndoMMLU lokal)
_OFFSETS_CACHE: dict[str, list[tuple[int, str]]] = {}  # key -> [(offset_global, id)] subset

DEFAULT_OUTPUT_NAME = "generated_conv_agent.jsonl"
DS_API = "https://datasets-server.huggingface.co"


class _DataSourceError(RuntimeError):
    """Error dari datasets-server (bukan error logika sampling)."""


# ─── Registry Sumber (katalog statis + spesifikasi akses random-access) ────────
# access:
#   "rows"    -> /rows?offset=<acak>  (random access seluruh split)
#   "offsets" -> subset via offset map pyarrow (baca kolom kecil saja) + /rows fetch (vision tetap)
#   "csv"     -> CSV asli di-cache lokal sekali, lalu random index (IndoMMLU)
# id_key: bila diisi, deduplikasi via nilai kolom id (bukan Row #).
# offsets_cols/offsets_keep: hanya utk access "offsets" (kolom label + id).
SOURCES: list[dict[str, Any]] = [
    # ── TEXT ─────────────────────────────────────────────────────────────
    {
        "key": "IndoMMLU", "category": "text", "label": "ringan", "access": "csv",
        "dataset": "indolem/IndoMMLU", "config": "default", "split": "train",
        "csv_url": "https://huggingface.co/datasets/indolem/IndoMMLU/resolve/main/IndoMMLU.csv",
        "builder": _build_indommlu_csv_text, "id_key": None, "approx_rows": 14981,
        "source_name": "IndoMMLU (HuggingFace: indolem/IndoMMLU, CSV asli)",
        "task_type": "Educational Concept Explanation & Science QA",
        "tuning_instruction": "Fokus pada gaya penjelasan akademis/edukatif yang mudah dipahami, penguraian konsep sains/teknologi secara terstruktur, analogi, dan penjelasan bertahap.",
    },
    {
        "key": "IndoCareer", "category": "text", "label": "ringan", "access": "rows",
        "dataset": "indolem/IndoCareer", "config": "all", "split": "test",
        "builder": _build_indocareer_text, "id_key": None,
        "source_name": "IndoCareer (HuggingFace: indolem/IndoCareer, all/test)",
        "task_type": "Professional QA & Career Counseling",
        "tuning_instruction": "Fokus pada penjelasan konteks profesional, persiapan ujian sertifikasi, konseling karir, dan pengetahuan bidang hukum/keuangan/medis.",
    },
    {
        "key": "IndoCulture", "category": "text", "label": "ringan", "access": "rows",
        "dataset": "indolem/IndoCulture", "config": "default", "split": "test",
        "builder": _build_indoculture_text, "id_key": None,
        "source_name": "IndoCulture (HuggingFace: indolem/IndoCulture, test)",
        "task_type": "Cultural QA & Local Knowledge",
        "tuning_instruction": "Fokus pada pengetahuan kebudayaan Indonesia, adat istiadat daerah, kuliner, seni, dan Kearifan lokal.",
    },
    {
        "key": "FineWeb-Edu-25K", "category": "text", "label": "berat", "access": "rows",
        "dataset": "irfanfadhullah/FineWeb-Edu-25K", "config": "default", "split": "train",
        "text_key": "text_indonesian", "id_key": None,
        "source_name": "FineWeb-Edu-25K (HuggingFace: irfanfadhullah/FineWeb-Edu-25K)",
        "task_type": "Educational Concept Explanation & Science QA",
        "tuning_instruction": "Fokus pada gaya penjelasan akademis/edukatif yang mudah dipahami, penguraian konsep sains/teknologi secara terstruktur, analogi, dan penjelasan bertahap.",
    },
    {
        "key": "OpenWebText-10k", "category": "text", "label": "berat", "access": "rows",
        "dataset": "irfanfadhullah/OpenWebText-Indonesia-10k", "config": "default", "split": "train",
        "text_key": "translated_text", "id_key": None,
        "source_name": "OpenWebText-Indonesia-10k (HuggingFace: irfanfadhullah/OpenWebText-Indonesia-10k)",
        "task_type": "News Summarization & Public Policy Discussion",
        "tuning_instruction": "Fokus pada perangkuman berita, diskusi isu sosial/ekonomi, analisis kebijakan publik, serta evaluasi dampak bagi masyarakat.",
    },
    {
        "key": "Indonesian-Simple-Summaries", "category": "text", "label": "berat", "access": "rows",
        "dataset": "irfanfadhullah/indonesian-simple-summaries", "config": "default", "split": "train",
        "builder": _build_simple_summaries_text, "id_key": None,
        "source_name": "Indonesian Simple Summaries (HuggingFace: irfanfadhullah/indonesian-simple-summaries)",
        "task_type": "News Summarization & Text Simplification",
        "tuning_instruction": "Fokus pada perangkuman berita, penyederhanaan teks, dan pembelajaran konseptual yang mudah dipahami.",
    },
    {
        "key": "QQPR-Triplets-ID", "category": "text", "label": "berat", "access": "rows",
        "dataset": "robinsyihab/QQPR-triplets-ID", "config": "default", "split": "train",
        "builder": _build_qqpr_text, "id_key": None,
        "source_name": "QQPR-Triplets-ID (HuggingFace: robinsyihab/QQPR-triplets-ID)",
        "task_type": "Paraphrase Generation & Text Similarity",
        "tuning_instruction": "Fokus pada generasi paraphrasing, identifikasi kemiripan teks, dan penjelasan perbedaan makna antar kalimat.",
    },
    {
        "key": "Indonesian-Wikipedia", "category": "text", "label": "berat", "access": "rows",
        "dataset": "indonesian-nlp/wikipedia-id", "config": "default", "split": "train",
        "builder": _build_wikipedia_text, "id_key": None,
        "source_name": "Indonesian Wikipedia (HuggingFace: indonesian-nlp/wikipedia-id)",
        "task_type": "General Knowledge & Encyclopedia QA",
        "tuning_instruction": "Fokus pada penjelasan pengetahuan umum, informasi ensiklopedia, fakta sejarah, sains, dan geografi Indonesia.",
    },
    {
        "key": "Cendol-Collection-v2", "category": "text", "label": "berat", "access": "rows",
        "dataset": "indonlp/cendol_collection_v2", "config": "default", "split": "train",
        "builder": _build_cendol_text, "id_key": "prompt_id",
        "source_name": "Cendol Collection v2 (HuggingFace: indonlp/cendol_collection_v2)",
        "task_type": "General Instruction Following & Knowledge QA",
        "tuning_instruction": "Fokus pada instruction following, pengetahuan umum, dan percakapan sehari-hari dalam bahasa Indonesia.",
    },
    {
        "key": "LFQA-ID", "category": "text", "label": "berat", "access": "rows",
        "dataset": "indonesian-nlp/lfqa_id", "config": "default", "split": "train",
        "builder": _build_lfqa_text, "id_key": "q_id",
        "source_name": "LFQA-ID (HuggingFace: indonesian-nlp/lfqa_id)",
        "task_type": "Long-Form QA & Explanatory Dialogue",
        "tuning_instruction": "Fokus pada jawaban panjang yang mendidik, penjelasan konsep dengan analogi, dan percakapan tanya-jawab mendalam ala forum.",
    },
    # ── VISION (MULTIMODAL) ───────────────────────────────────────────────
    {
        "key": "CVQA-Indonesia", "category": "vision", "label": "berat", "access": "offsets",
        "dataset": "afaji/cvqa", "config": "default", "split": "test",
        "offsets_cols": ["Subset", "ID"],
        "offsets_keep": lambda s: "indonesian" in str(s).lower(),
        "builder": _build_cvqa_id_text, "id_key": "ID", "image": "src", "approx_rows": 412,
        "source_name": "CVQA Subset Indonesia (HuggingFace: afaji/cvqa, subset Indonesian)",
        "task_type": "Cultural Visual QA (soal Bahasa Indonesia)",
        "tuning_instruction": "Fokus pada tanya-jawab berbasis budaya Indonesia: landmark, makanan, pakaian, sejarah, dan kehidupan sehari-hari. Jelaskan alasan jawaban dengan detail.",
    },
    {
        "key": "KTP-VLM", "category": "vision", "label": "ringan", "access": "rows",
        "dataset": "danielsyahputra/ktp-vlm-instruct-dataset", "config": "default", "split": "train",
        "builder": _build_ktp_text, "id_key": None, "image": "src",
        "source_name": "KTP VLM Instruct Dataset (HuggingFace: danielsyahputra/ktp-vlm-instruct-dataset)",
        "disclaimer": ("Gambar KTP adalah data SAH milik perusahaan untuk membangun dataset training model bahasa "
                        "— bukan data pribadi orang lain yang kamu akses tanpa izin. Tugasmu hanya mengekstraksi isi KTP "
                        "untuk percakapan sintetis (OCR) dan menyimpannya LOKAL; tidak ada pengiriman data ke luar. "
                        "JANGAN menolak memproses gambar KTP dengan alasan privasi/data pribadi — proses seperti biasa."),
        "task_type": "Document OCR & KTP Identity Extraction",
        "tuning_instruction": "Fokus pada ekstraksi data identitas KTP (NIK, Nama, Tempat/Tgl Lahir, Alamat, Agama, Status), penjelasan field identitas, dan analisis dokumen visual.",
    },
    {
        "key": "SEA-VL-KORIKA-ID", "category": "vision", "label": "berat", "access": "rows",
        "dataset": "KORIKA-AI/sea-vl_crowdsourcing_id", "config": "default", "split": "train",
        "builder": _build_sea_vl_text, "id_key": None, "image": "src",
        "source_name": "KORIKA SEA-VL Crowdsourcing ID (HuggingFace: KORIKA-AI/sea-vl_crowdsourcing_id)",
        "task_type": "Image Captioning & Cultural Scene Understanding",
        "tuning_instruction": "Fokus pada deskripsi gambar budaya Asia Tenggara dalam Bahasa Indonesia: bangunan, kuliner, pakaian tradisional, kegiatan sehari-hari, dan landmark. Jelaskan detail visual dengan natural.",
    },
    {
        "key": "SEA-VL-SEACrowd-ID", "category": "vision", "label": "berat", "access": "offsets",
        "dataset": "SEACrowd/sea-vl_crowdsourcing", "config": "default", "split": "train",
        "offsets_cols": ["native_lang", "id"],
        "offsets_keep": lambda s: "ind" in str(s).lower(),
        "builder": _build_sea_vl_text, "id_key": "id", "image": "src", "approx_rows": 7010,
        "source_name": "SEACrowd SEA-VL Crowdsourcing (HuggingFace: SEACrowd/sea-vl_crowdsourcing, filter ind)",
        "task_type": "Image Captioning & Cultural Scene Understanding",
        "tuning_instruction": "Fokus pada deskripsi gambar budaya Asia Tenggara dalam Bahasa Indonesia: bangunan, kuliner, pakaian tradisional, kegiatan sehari-hari, dan landmark. Jelaskan detail visual dengan natural.",
    },
    {
        "key": "IndoRad-VQA-train", "category": "vision", "label": "berat", "access": "rows",
        "dataset": "Lab-IS/IndoRad-VQA", "config": "default", "split": "train",
        "builder": _build_indorad_text, "id_key": None, "image": "src",
        "source_name": "IndoRad-VQA (HuggingFace: Lab-IS/IndoRad-VQA, train)",
        "task_type": "Medical VQA & Radiology Image Analysis",
        "tuning_instruction": "Fokus pada analisis citra radiologi (X-ray, CT, MRI): identifikasi kelainan, jawaban ya/tidak terstruktur, dan penjelasan medis sederhana dalam Bahasa Indonesia.",
    },
]


def _output_path() -> Path:
    return Path(_CONFIG.get("output_dir", str(Path.home() / "generate-conv-indonesia"))) / DEFAULT_OUTPUT_NAME


# ─── Kuota & progress (target 2000 text / 1000 vision — 2:1) ──────────────────

def _targets() -> tuple[int, int]:
    return (int(_CONFIG.get("text_target", 2000) or 0), int(_CONFIG.get("vision_target", 1000) or 0))


def _count_done() -> tuple[int, int]:
    """Hitung percakapan text/vision yang sudah tersimpan di file output."""
    text_done = vision_done = 0
    p = _output_path()
    if p.exists():
        with open(p, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    cat = json.loads(line).get("category", "")
                except json.JSONDecodeError:
                    continue
                if cat == "text_nlu_chat":
                    text_done += 1
                elif cat == "vision_chat":
                    vision_done += 1
    return text_done, vision_done


def _prefix_stats() -> dict[str, int]:
    """Hitung distribusi pemakaian token prefix <unused1..6> dari percakapan tersimpan.

    Dipakai untuk mengingatkan model agar SEMUA token terpakai seimbang (bukan cuma <unused4>).
    """
    counts = {f"<unused{i}>": 0 for i in range(1, 7)}
    p = _output_path()
    if p.exists():
        with open(p, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    item = json.loads(line)
                except json.JSONDecodeError:
                    continue
                for m in item.get("messages", []):
                    if m.get("role") != "assistant":
                        continue
                    content = str(m.get("content", ""))
                    mch = re.match(r"^(?:<unused[1-6]>)*<unused[1-6]>", content)
                    if mch:
                        for t in re.findall(r"<unused[1-6]>", mch.group(0)):
                            counts[t] = counts.get(t, 0) + 1
    return counts


_PREFIX_LABEL = {"<unused1>": "SUMMARIZE", "<unused2>": "TRANSLATE", "<unused3>": "NER",
                 "<unused4>": "QA", "<unused5>": "PARAPHRASE", "<unused6>": "GENERAL_CHAT"}


def _pairs_stats() -> dict[str, int]:
    """Distribusi jumlah pasang (3/4/5 = 6/8/10 pesan) dari percakapan tersimpan."""
    counts = {"3": 0, "4": 0, "5": 0}
    p = _output_path()
    if p.exists():
        with open(p, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    np_ = json.loads(line).get("num_pairs")
                except json.JSONDecodeError:
                    continue
                counts[str(np_)] = counts.get(str(np_), 0) + 1
    return counts


def _pairs_hint(stats: dict[str, int]) -> str:
    total = sum(stats.values())
    if total == 0:
        return "Belum ada percakapan tersimpan — VARIASIKAN jumlah pasang: 3, 4, atau 5 (6/8/10 pesan), jangan selalu 4."
    least_val = min(stats.values())
    most_val = max(stats.values())
    if most_val > 0 and least_val >= most_val * 0.5:
        return "Distribusi jumlah pasang (3/4/5 = 6/8/10 pesan) sudah seimbang — lanjutkan sesuai kebutuhan konten."
    least = [k for k, v in stats.items() if v == least_val]
    names = " dan ".join(f"{k} pasang ({int(k) * 2} pesan)" for k in least)
    return (f"Jumlah pasang yang masih JARANG: {names}. Tulis percakapan dengan jumlah pasang itu "
            f"supaya variasi 3/4/5 (6/8/10 pesan) seimbang di seluruh dataset.")


def _prefix_hint(stats: dict[str, int]) -> str:
    total = sum(stats.values())
    if total == 0:
        return "Belum ada percakapan tersimpan — pilih prefix sesuai task isi pesan, dan VARIASIKAN (jangan semua <unused4>)."
    ordered = sorted(stats.items(), key=lambda kv: kv[1])
    least = [k for k, v in ordered if v == ordered[0][1]]
    most = ordered[-1][1]
    if most > 0 and ordered[0][1] >= most * 0.5:
        return "Distribusi prefix sudah cukup seimbang — lanjutkan memilih sesuai task."
    names = ", ".join(f"{k} {_PREFIX_LABEL[k]}" for k in least[:3])
    return (f"Prefix yang masih JARANG dipakai: {names}. Tulis percakapan yang memang butuh task "
            f"itu (mis. parafrase/chat umum) supaya distribusi <unused1..6> seimbang di seluruh dataset — "
            f"tetap sesuaikan dengan isi pesan, jangan memaksakan token yang tidak cocok.")


def _next_category() -> str | None:
    """Kategori berikutnya agar rasio 2:1 terjaga (yang paling tertinggal rasionya)."""
    text_target, vision_target = _targets()
    text_done, vision_done = _count_done()
    text_left = text_target - text_done
    vision_left = vision_target - vision_done
    if text_left <= 0 and vision_left <= 0:
        return None
    if text_left <= 0:
        return "vision"
    if vision_left <= 0:
        return "text"
    tr = text_done / text_target
    vr = vision_done / vision_target
    return "text" if tr <= vr else "vision"


def _progress_impl() -> str:
    text_target, vision_target = _targets()
    text_done, vision_done = _count_done()
    next_cat = _next_category()
    next_src: str | None = None
    if next_cat:
        cands = [s for s in SOURCES if s["category"] == next_cat]
        if cands:
            next_src = _pick_source(cands)["key"]
    pstats = _prefix_stats()
    nstats = _pairs_stats()
    return json.dumps({
        "text": {"target": text_target, "done": text_done, "remaining": max(0, text_target - text_done)},
        "vision": {"target": vision_target, "done": vision_done, "remaining": max(0, vision_target - vision_done)},
        "done": text_done >= text_target and vision_done >= vision_target,
        "next_category": next_cat,
        "next_source_key": next_src,
        "prefix_stats": pstats,
        "prefix_hint": _prefix_hint(pstats),
        "pairs_stats": nstats,
        "pairs_hint": _pairs_hint(nstats),
        "note": "sample_row() tanpa category otomatis memilih jenis yang paling tertinggal rasionya (2:1).",
    }, ensure_ascii=False, indent=2)


# ─── Exclusions (row/id yang sudah dipakai di JSONL) ──────────────────────────

def _refresh_exclusions() -> int:
    """Baca ulang file output → set row/id yang sudah dipakai (dipanggil saat start)."""
    global _EXCLUDED_ROWS, _EXCLUDED_IDS
    rows, ids = load_excluded_sources([_output_path()])
    _EXCLUDED_ROWS, _EXCLUDED_IDS = rows, ids
    return len(rows) + len(ids)


# ─── Datasets-server (random access, tanpa unduh penuh) ───────────────────────

def _ds_retry(path_query: str, attempts: int = 10, base_delay: float = 3.0) -> dict[str, Any]:
    """GET endpoint datasets-server dengan retry backoff untuk error transien.

    Error "the dataset index is loading" & HTTP 429/5xx di-retry;
    error permanen (mis. 422/404) langsung dilempar sebagai _DataSourceError.
    """
    headers = {"User-Agent": "generate-conv-indonesia-mcpb"}
    token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
    if not token:
        try:
            for p in [Path(__file__).resolve().parents[3] / ".env", Path(__file__).resolve().parents[2] / ".env"]:
                if p.exists():
                    for line in p.read_text(encoding="utf-8").splitlines():
                        if line.startswith("HF_TOKEN="):
                            token = line.split("=", 1)[1].strip().strip('"').strip("'")
                            break
                if token:
                    break
        except Exception:
            pass
    if not token:
        try:
            token_file = Path.home() / ".cache" / "huggingface" / "token"
            if token_file.exists():
                token = token_file.read_text().strip()
        except Exception:
            pass
    if token:
        headers["Authorization"] = f"Bearer {token}"
    url = f"{DS_API}{path_query}"
    last: Exception | None = None
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
                break  # error permanen — jangan buang waktu retry
        except urllib.error.HTTPError as e:
            last = _DataSourceError(f"HTTP {e.code} {e.reason}: {path_query}")
            if e.code != 429 and e.code < 500:
                break
        except (urllib.error.URLError, TimeoutError, OSError) as e:
            last = e
        if i < attempts - 1:
            time.sleep(base_delay * (1.5 ** i) + random.uniform(1.0, 3.0))
    raise last if isinstance(last, Exception) else _DataSourceError(f"datasets-server gagal: {path_query}")



def _count_rows(spec: dict[str, Any]) -> int:
    """Jumlah row split (cache per sumber) via /size."""
    key = (spec["dataset"], spec["config"], spec["split"])
    if key in _COUNT_CACHE:
        return _COUNT_CACHE[key]
    q = urlencode({"dataset": spec["dataset"], "config": spec["config"]})
    payload = _ds_retry(f"/size?{q}")
    n = next((s.get("num_rows") for s in payload.get("splits", []) if s.get("split") == spec["split"]), None)
    if not n:
        # Fallback: num_rows_total dari /rows offset 0
        q = urlencode({"dataset": spec["dataset"], "config": spec["config"],
                       "split": spec["split"], "offset": 0, "length": 1})
        n = _ds_retry(f"/rows?{q}").get("num_rows_total")
    if not n:
        raise _DataSourceError(f"/size tidak mengenali split {spec['split']} utk {spec['key']}")
    _COUNT_CACHE[key] = n
    return n


def _source_weight(spec: dict[str, Any]) -> int:
    """Bobot sampling proporsional jumlah row. 'rows' → hitungan riil (cache);
    'csv'/'offsets' → approx_rows (subset), biar Cendol (12,8jt) tidak menenggelamkan sumber kecil."""
    if spec["access"] == "rows":
        try:
            return max(1, _count_rows(spec))
        except Exception:
            pass
    return max(1, int(spec.get("approx_rows", 1000) or 1000))


def _pick_source(specs: list[dict[str, Any]]) -> dict[str, Any]:
    """Pilih source berbobot jumlah row (proporsional)."""
    weights = [_source_weight(s) for s in specs]
    return random.choices(specs, weights=weights, k=1)[0]


def _fetch_row_at(spec: dict[str, Any], offset: int) -> dict[str, Any]:
    """Ambil 1 row di offset acak via /rows (random access seluruh split)."""
    q = urlencode({"dataset": spec["dataset"], "config": spec["config"], "split": spec["split"],
                   "offset": offset, "length": 1})
    payload = _ds_retry(f"/rows?{q}")
    rows = payload.get("rows") or []
    if not rows:
        raise _DataSourceError(f"datasets-server mengembalikan row kosong di offset {offset} ({spec['key']})")
    return rows[0]["row"]


def _pick_offset(rng: Any, num_rows: int, excluded: set) -> int:
    """Offset acak yang belum dipakai (rejection sampling; scan hanya bila dataset nyaris habis)."""
    for _ in range(40):
        off = rng.randint(0, num_rows - 1)
        if off not in excluded:
            return off
    allowed = [i for i in range(num_rows) if i not in excluded]
    if not allowed:
        raise ValueError("Semua row sumber ini sudah dipakai (excluded).")
    return rng.choice(allowed)


def _csv_dataset(spec: dict[str, Any]) -> Any:
    """IndoMMLU: unduh CSV asli SEKALI ke cache lokal, lalu random index (tanpa unduh ulang)."""
    key = spec["dataset"]
    if key in _CSV_CACHE:
        return _CSV_CACHE[key]
    cache_dir = Path.home() / ".cache" / "generate-conv-indonesia"
    cache_dir.mkdir(parents=True, exist_ok=True)
    path = cache_dir / "IndoMMLU.csv"
    if not path.exists():
        print(f"⬇️ Mengunduh CSV asli {spec['dataset']} → {path}", file=sys.stderr)
        req = urllib.request.Request(spec["csv_url"], headers={"User-Agent": "generate-conv-indonesia-mcpb"})
        with urllib.request.urlopen(req, timeout=120) as resp, open(path, "wb") as f:
            f.write(resp.read())
    import datasets  # lazy: import berat hanya saat benar-benar dibutuhkan
    ds = datasets.load_dataset("csv", data_files=str(path), split="train")
    _CSV_CACHE[key] = ds
    return ds


# ─── Image helpers (kolom image dari datasets-server) ─────────────────────────

def _image_src_from_row(row: dict[str, Any]) -> str | None:
    """Ambil URL `src` dari kolom image (dict tunggal atau List[dict])."""
    img = row.get("image")
    items = img if isinstance(img, list) else [img]
    for item in items:
        if isinstance(item, dict):
            src = item.get("src")
            if isinstance(src, str) and src:
                return src
    return None


def _image_dims_from_row(row: dict[str, Any]) -> dict[str, int] | None:
    img = row.get("image")
    items = img if isinstance(img, list) else [img]
    for item in items:
        if isinstance(item, dict) and isinstance(item.get("width"), int) and isinstance(item.get("height"), int):
            return {"width": item["width"], "height": item["height"]}
    return None


def _image_info(spec: dict[str, Any], row: dict[str, Any], off: int | None) -> tuple[str | None, dict[str, int] | None]:
    if spec.get("image") != "src":
        return None, None
    src = _image_src_from_row(row)
    if not src and spec["key"] == "KTP-VLM":
        # Fallback KTP: URL resolve HF (pola ktp_row_{idx}.jpg, stabil tanpa expiry)
        src = _ktp_image_url(row, off if off is not None else 0)
    return src, _image_dims_from_row(row)


# ─── Sampling 1 baris acak ────────────────────────────────────────────────────

def _build_context(spec: dict[str, Any], row: dict[str, Any]) -> str:
    if spec.get("builder"):
        raw = spec["builder"](row)
    elif spec.get("text_key"):
        raw = str(row.get(spec["text_key"], "")) or str(row)
    else:
        raw = str(row)
    return str(raw)[:1500]


def _build_sample_result(spec: dict[str, Any], row: dict[str, Any],
                         off: int | None, idv: str | None, note: str) -> dict[str, Any]:
    raw = _build_context(spec, row)
    image_ref, image_size = _image_info(spec, row, off)
    if idv:
        source_detail = f"{spec['source_name']}, id={idv}"
    else:
        source_detail = f"{spec['source_name']}, Row #{off}"
    result = {
        "source": source_detail,
        "source_key": spec["key"],
        "category": spec["category"],
        "task_type": spec["task_type"],
        "is_vision": spec["category"] == "vision",
        "image_available": bool(image_ref),
        "image_ref": image_ref,
        "image_size": image_size,
        "raw_context": raw,
        "note": note,
    }
    if spec.get("disclaimer"):
        result["etika"] = spec["disclaimer"]
    return result


def _sample_from_spec(spec: dict[str, Any], seed: int | None) -> dict[str, Any]:
    if spec["access"] == "offsets":
        return _sample_offsets(spec, seed)
    rng = random.Random(seed) if seed is not None else random
    name = spec["source_name"]
    ex_rows = _EXCLUDED_ROWS.get(name, set())
    ex_ids = _EXCLUDED_IDS.get(name, set())
    idv: str | None = None
    off: int | None = None
    row: dict[str, Any] | None = None
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
                continue  # re-roll: id sudah dipakai
        break
    else:
        raise RuntimeError(f"Tidak ada row tersisa untuk {name} (semua sudah terpakai).")
    if spec["access"] == "csv":
        note = "IndoMMLU via CSV asli lokal (cache sekali), random index — tanpa unduh berulang."
    else:
        note = "Random-access datasets-server (/rows) — 1 HTTP call, tanpa unduh dataset penuh."
    return _build_sample_result(spec, row, off, idv, note)


# ─── Implementasi tools (pure functions) ──────────────────────────────────────
# ─── Offset map untuk subset (CVQA/SEACrowd): pyarrow columnar-read ──────────

def _offsets_for(spec: dict[str, Any]) -> list[tuple[int, str]]:
    """Peta [(offset_global, id), ...] baris subset Indonesia — TANPA unduh penuh.

    Baca hanya kolom label + id dari semua shard parquet (columnar read via pyarrow
    remote), lalu hitung offset global = jumlah row shard sebelumnya + index dalam
    shard. Row aslinya (termasuk gambar) tetap diambil via /rows di offset tsb.
    Dibungkus timeout 180s (parquet remote bisa lambat/hang) — kalau gagal,
    pemanggil jatuh ke fallback /rows rejection sampling.
    """
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


def _build_offset_map(spec: dict[str, Any]) -> list[tuple[int, str]]:
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
    result: list[tuple[int, str]] = []
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


def _sample_subset_reject(spec: dict[str, Any], seed: int | None, fallback_err: str = "") -> dict[str, Any]:
    """Fallback: /rows + rejection sampling baris subset (tetap MULTIMODAL via src gambar).

    Jeda 0,35s antar request supaya tidak kena rate-limit datasets-server (429).
    """
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


def _sample_offsets(spec: dict[str, Any], seed: int | None) -> dict[str, Any]:
    """Sample acak dari subset Indonesia — vision tetap (gambar via /rows src).
    Fallback ke rejection sampling bila offset map pyarrow gagal/hang."""
    rng = random.Random(seed) if seed is not None else random
    try:
        entries = _offsets_for(spec)
        ex_ids = _EXCLUDED_IDS.get(spec["source_name"], set())
        candidates = [(off, idv) for off, idv in entries if idv not in ex_ids]
        if not candidates:
            raise _DataSourceError(f"Semua baris {spec['source_name']} sudah terpakai (excluded).")
        off, idv = rng.choice(candidates)
        row = _fetch_row_at(spec, off)
        return _build_sample_result(
            spec, row, off, idv,
            note="Subset Indonesia via offset map pyarrow (columnar) + /rows — random-access, vision dengan gambar.")
    except _DataSourceError as e:
        return _sample_subset_reject(spec, seed, fallback_err=str(e))


# ─── Implementasi tools (pure functions) ──────────────────────────────────────

def _list_sources_impl(category: str) -> str:
    """Katalog statis — INSTANT, tanpa load dataset."""
    lines = [
        "Katalog 15 sumber Indonesia — RANDOM-ACCESS (datasets-server), tanpa unduh dataset penuh.",
        "Semua sumber INSTAN (termasuk Cendol 12,8jt row); sample_row default include_heavy=true.",
        "CVQA-Indonesia kini VISION (dengan gambar). IndoRad-VQA (split train saja). Row yang sudah dipakai otomatis diskip.",
        "",
    ]
    for s in SOURCES:
        if category in ("all", s["category"]):
            acc = {"rows": "rows", "offsets": "subset", "csv": "csv"}[s["access"]]
            lines.append(f"{s['category'].upper()} | {s['key']} | {s['label']} | {acc} | {s['task_type']} | {s['dataset']}")
    return "\n".join(lines) if len(lines) > 4 else "(tidak ada sumber untuk kategori ini)"


async def _sample_row_impl(category: str, source_key: str, seed: int | None, include_heavy: bool) -> dict[str, Any]:
    if category not in ("text", "vision"):
        # 'auto': pilih jenis sesuai kuota 2:1 (rasio paling tertinggal)
        category = _next_category() or random.choice(["text", "vision"])
    specs = [s for s in SOURCES if category in ("all", s["category"])]
    if not include_heavy:
        specs = [s for s in specs if s["label"] == "ringan"]
    if not specs:
        raise ValueError(f"Tidak ada sumber untuk kategori '{category}' (include_heavy={include_heavy}). Gunakan list_sources().")
    if source_key:
        spec = next((s for s in specs if s["key"] == source_key), None)
        if spec is None:
            raise ValueError(f"Sumber '{source_key}' tidak ditemukan di kategori '{category}'. Gunakan list_sources().")
    else:
        spec = _pick_source(specs)
    info = await asyncio.to_thread(_sample_from_spec, spec, seed)
    info["auto_category"] = category
    pstats = _prefix_stats()
    info["prefix_stats"] = pstats
    info["prefix_hint"] = _prefix_hint(pstats)
    nstats = _pairs_stats()
    info["pairs_stats"] = nstats
    info["pairs_hint"] = _pairs_hint(nstats)
    return info


# Batas hasil tool Claude Desktop = 1MB total. Base64 ≈ 4/3 × ukuran asli,
# jadi target raw bytes ~700KB → base64 ~933KB (+JSON kecil) masih aman < 1MB.
_IMAGE_MAX_RAW = 700 * 1024


def _image_to_limited_jpeg(pil: PILImage.Image, max_raw: int = _IMAGE_MAX_RAW) -> tuple[bytes, str]:
    """Kompres on-the-fly: downscale + JPEG adaptif sampai muat di bawah limit tool.

    Mulai dari 1024px/q85; bila masih melebihi max_raw, turunkan kualitas JPEG,
    lalu kecilkan dimensi secara bertahap (floor 160px).
    """
    pil = pil.convert("RGB")
    dim = 1024
    quality = 85
    while True:
        img = pil
        if max(pil.size) > dim:
            img = pil.copy()
            img.thumbnail((dim, dim), PILImage.Resampling.LANCZOS)
        buf = io.BytesIO()
        img.save(buf, format="JPEG", quality=quality)
        data = buf.getvalue()
        if len(data) <= max_raw or dim <= 160:
            return data, "image/jpeg"
        if quality > 45:
            quality -= 20
        else:
            dim = max(160, dim // 2)
            quality = 80


def _refresh_datasets_server_image_url(ref: str, source: str = "") -> str | None:
    """Jika URL gambar Hugging Face datasets-server kadaluarsa (HTTP 403), minta signed URL segar."""
    m_row = SOURCE_ROW_RE.match(source) if source else None
    m_id = SOURCE_ID_RE.match(source) if source else None
    ds_name: str | None = None
    row_idx: int | None = None

    if m_row:
        src_name = m_row.group("name")
        row_idx = int(m_row.group("idx"))
        for s in SOURCES:
            if s["source_name"] == src_name or s["key"] in src_name:
                ds_name = s["dataset"]
                break
    elif m_id:
        src_name = m_id.group("name")
        for s in SOURCES:
            if s["source_name"] == src_name or s["key"] in src_name:
                ds_name = s["dataset"]
                break

    # Fallback: ekstrak dataset name dan row index langsung dari pola URL cached-assets
    # Pola: cached-assets/<org>/<repo>/--/<hash>/--/<config>/<split>/<row_idx>/image/...
    if not ds_name or row_idx is None:
        m_url = re.search(r"cached-assets/([^/]+/[^/]+)/--/[^/]+/--/[^/]+/([^/]+)/(\d+)/image", ref)
        if m_url:
            ds_name = m_url.group(1)
            row_idx = int(m_url.group(3))

    if ds_name and row_idx is not None:
        try:
            url_query = f"/rows?dataset={urllib.parse.quote(ds_name, safe='')}&config=default&split=train&offset={row_idx}&length=1"
            req = urllib.request.Request(f"{DS_API}{url_query}", headers={"User-Agent": "generate-conv-indonesia-mcpb"})
            with urllib.request.urlopen(req, timeout=20) as resp:
                data = json.loads(resp.read().decode("utf-8"))
                rows = data.get("rows", [])
                if rows:
                    row_data = rows[0].get("row", {})
                    fresh_src = _image_src_from_row(row_data)
                    if fresh_src:
                        return fresh_src
        except Exception:
            pass
    return None


def _load_pil_from_ref(ref: str, source: str = "") -> PILImage.Image | None:
    if ref.startswith(("http://", "https://")):
        req = urllib.request.Request(ref, headers={"User-Agent": "generate-conv-indonesia-mcpb"})
        try:
            with urllib.request.urlopen(req, timeout=45) as resp:
                return PILImage.open(io.BytesIO(resp.read())).convert("RGB")
        except urllib.error.HTTPError as e:
            if e.code == 403:
                fresh_url = _refresh_datasets_server_image_url(ref, source)
                if fresh_url:
                    req_fresh = urllib.request.Request(fresh_url, headers={"User-Agent": "generate-conv-indonesia-mcpb"})
                    with urllib.request.urlopen(req_fresh, timeout=45) as resp_fresh:
                        return PILImage.open(io.BytesIO(resp_fresh.read())).convert("RGB")
            raise
    p = Path(ref)
    if p.exists():
        return PILImage.open(p).convert("RGB")
    return None


def _read_image_base64(image_ref: str, source: str = "") -> tuple[str, str]:
    """Baca gambar → (base64 JPEG terkompres, mime_type) — muat < 1MB hasil tool."""
    pil = _load_pil_from_ref(image_ref, source)
    if pil is None:
        raise ValueError(f"Gambar tidak dapat dibaca: {image_ref}")
    data, mime = _image_to_limited_jpeg(pil)
    return base64.b64encode(data).decode("ascii"), mime


def _save_conversation_impl(source: str, category: str, conversation_json: str,
                            image_ref: str, output_path: str, num_pairs: str = "") -> dict[str, Any]:
    try:
        data = json.loads(conversation_json)
        if not isinstance(data, list) or not data:
            raise ValueError("conversation_json harus array turn (tidak kosong)")
        turns = []
        for t in data:
            role_str = str(t.get("role", ""))
            content_str = str(t.get("content", ""))
            pref_list = t.get("prefixes", [])
            if not pref_list and role_str == "assistant":
                pref_list = re.findall(r"<unused[1-6]>", content_str)
            turns.append(
                TurnMessage(
                    role=cast(Literal["user", "assistant"], role_str),
                    prefixes=pref_list,
                    content=content_str,
                )
            )
        validated = ConversationOutput(conversations=turns)  # validasi + normalisasi prefix
        actual_pairs = len(validated.conversations) // 2
        if num_pairs:
            try:
                want = int(num_pairs)
            except (TypeError, ValueError):
                raise ValueError(f"num_pairs harus angka 3-5, bukan '{num_pairs}'.")
            if want < 3 or want > 5:
                raise ValueError(f"num_pairs harus 3, 4, atau 5 (bukan {want}).")
            if actual_pairs != want:
                raise ValueError(
                    f"num_pairs yang kamu deklarasikan ({want} pasang = {want * 2} pesan) "
                    f"tidak cocok dengan isi percakapan ({actual_pairs} pasang = {actual_pairs * 2} pesan). "
                    f"Samakan jumlah turn-nya."
                )
        # Panjang minimal turn assistant — threshold lebih longgar utk vision (jawaban ya/tidak dsb.)
        min_len = 40 if category == "vision_chat" else 60
        for i, m in enumerate(validated.conversations):
            if m.role == "assistant" and len(m.content.strip()) < min_len:
                raise ValueError(
                    f"Turn assistant [{i}] terlalu pendek ({len(m.content.strip())} karakter). "
                    f"Elaborasi minimal {min_len} karakter per turn assistant (vision 40 / text 60)."
                )
    except Exception as e:
        raise ValueError(
            f"Validasi percakapan gagal: {e}. "
            "Perbaiki: jumlah pesan TEPAT 6/8/10 (3/4/5 pasang, selang-seling user/assistant), "
            "turn assistant elaborasi minimal 40 karakter (vision) / 60 (text), "
            "prefixes berisi token valid <unused1..6> (1-3 unik tanpa spasi/duplikat), "
            "dan minimal 1 turn assistant memakai 2-3 prefix (multi-task). Lalu coba lagi."
        ) from e

    formatted = [{"role": "system", "content": SYSTEM_PROMPT}]
    is_vision = category == "vision_chat"
    for idx, msg in enumerate(validated.conversations):
        content_clean = msg.content.replace("\\n", "\n").strip()
        for leak in ["SUMMARIZE", "TRANSLATE", "NER", "QA", "PARAPHRASE", "GENERAL_CHAT"]:
            content_clean = re.sub(rf"^\s*{leak}\s*", "", content_clean, flags=re.IGNORECASE)
        # Buang token <unusedX> di SELURUH konten (bukan cuma awal) — token hanya sah di field `prefixes`,
        # formatter di bawah menambahkannya lagi dari prefixes. Cegah token dobel di tengah kalimat.
        content_clean = re.sub(r"<unused[1-6]>", "", content_clean)
        # Runtuhkan spasi ganda tapi PERTAHANKAN newline (format JSON/code block tidak dirusak)
        content_clean = re.sub(r"[ \t]{2,}", " ", content_clean)
        content_clean = re.sub(r"\n{3,}", "\n\n", content_clean).strip()
        if msg.role == "user":
            if is_vision and idx == 0 and not content_clean.startswith("📷"):
                content_clean = "📷\n" + content_clean
            formatted.append({"role": "user", "content": content_clean})
        else:
            # Prefix sudah dijamin 1-3 token unik oleh validator; TIDAK ADA fallback.
            prefix_str = "".join(dict.fromkeys(msg.prefixes))
            if not prefix_str:
                raise ValueError("Prefix assistant kosong — seharusnya sudah ditolak validator.")
            formatted.append({"role": "assistant", "content": f"{prefix_str} {content_clean}".strip()})

    path = Path(output_path) if output_path else _output_path()
    path.parent.mkdir(parents=True, exist_ok=True)

    existing_max = 0
    if path.exists():
        try:
            with open(path, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        iid = json.loads(line).get("id")
                        if isinstance(iid, int) and iid > existing_max:
                            existing_max = iid
                    except json.JSONDecodeError:
                        continue
        except Exception:
            pass

    record = {
        "id": existing_max + 1,
        "source": source,
        "category": category,
        "num_turns": len(formatted),
        "num_pairs": len(validated.conversations) // 2,
        "reviewed": False,
        "edited_turns": 0,
        "messages": formatted,
    }
    if is_vision and image_ref:
        record["images"] = [image_ref]

    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")

    # Update exclusion set secara inkremental (baris ini tidak boleh di-sample lagi)
    m = SOURCE_ROW_RE.match(source)
    if m:
        _EXCLUDED_ROWS.setdefault(m.group("name"), set()).add(int(m.group("idx")))
    else:
        m = SOURCE_ID_RE.match(source)
        if m:
            _EXCLUDED_IDS.setdefault(m.group("name"), set()).add(m.group("id"))

    return {"ok": True, "id": record["id"], "path": str(path),
            "num_pairs": record["num_pairs"], "num_turns": record["num_turns"]}


def _normalize_prefix_text(content: str, prefixes: list[str] | None = None) -> str:
    """Normalisasi prefix pada konten asisten: tanpa spasi antar prefix, tanpa repetisi."""
    if not isinstance(content, str):
        return content

    content_clean = content.replace("\\n", "\n").strip()
    for leak in ["SUMMARIZE", "TRANSLATE", "NER", "QA", "PARAPHRASE", "GENERAL_CHAT"]:
        content_clean = re.sub(rf"^\s*{leak}\s*", "", content_clean, flags=re.IGNORECASE)

    if prefixes is not None:
        seen = set()
        deduped = []
        for p in prefixes:
            p_clean = re.sub(r"\s+", "", str(p))
            if p_clean in {"<unused1>", "<unused2>", "<unused3>", "<unused4>", "<unused5>", "<unused6>"} and p_clean not in seen:
                seen.add(p_clean)
                deduped.append(p_clean)
        body = re.sub(r"<unused[1-6]>", "", content_clean)
        body = re.sub(r"[ \t]{2,}", " ", body)
        body = re.sub(r"\n{3,}", "\n\n", body).strip()
        prefix_str = "".join(deduped)
        return f"{prefix_str} {body}".strip() if prefix_str else body

    match = re.match(r"^(\s*(?:<unused[1-6]>\s*)+)(.*)$", content_clean, flags=re.DOTALL)
    if not match:
        return content_clean

    prefix_block = match.group(1)
    rest_body = match.group(2)

    tokens = re.findall(r"<unused[1-6]>", prefix_block)
    seen = set()
    deduped = []
    for t in tokens:
        if t not in seen:
            seen.add(t)
            deduped.append(t)

    prefix_str = "".join(deduped)
    body = re.sub(r"<unused[1-6]>", "", rest_body)
    body = re.sub(r"[ \t]{2,}", " ", body)
    body = re.sub(r"\n{3,}", "\n\n", body).strip()
    return f"{prefix_str} {body}".strip()


def _review_conversation_impl(
    conversation_id: Any,
    action: str = "read",
    updates: Any | None = None,
    output_path: str = ""
) -> dict[str, Any]:
    path = Path(output_path) if output_path else _output_path()
    if not path.exists():
        raise FileNotFoundError(f"File dataset tidak ditemukan: {path}")

    try:
        conv_id = int(conversation_id)
    except (TypeError, ValueError):
        raise ValueError(f"conversation_id harus berupa angka integer valid, bukan '{conversation_id}'")

    if action not in ("read", "edit"):
        raise ValueError(f"action harus 'read' atau 'edit', bukan '{action}'")

    lines: list[str] = []
    target_idx: int | None = None
    target_record: dict[str, Any] | None = None

    with open(path, "r", encoding="utf-8") as f:
        for idx, line in enumerate(f):
            lines.append(line)
            if target_record is None:
                stripped = line.strip()
                if stripped:
                    try:
                        item = json.loads(stripped)
                        if item.get("id") == conv_id:
                            target_idx = idx
                            target_record = item
                    except json.JSONDecodeError:
                        continue

    if target_record is None or target_idx is None:
        raise ValueError(f"Percakapan dengan id {conv_id} tidak ditemukan di {path.name}")

    if action == "read":
        return {
            "ok": True,
            "action": "read",
            "id": conv_id,
            "category": target_record.get("category"),
            "source": target_record.get("source"),
            "num_turns": target_record.get("num_turns"),
            "reviewed": target_record.get("reviewed", False),
            "edited_turns": target_record.get("edited_turns", 0),
            "messages": target_record.get("messages", []),
            "images": target_record.get("images", []),
        }

    if not updates:
        raise ValueError("Parameter 'updates' wajib diisi jika action='edit'")

    if isinstance(updates, str):
        try:
            updates_list = json.loads(updates)
        except json.JSONDecodeError as e:
            raise ValueError(f"Format JSON 'updates' tidak valid: {e}")
    else:
        updates_list = updates

    if not isinstance(updates_list, list) or not updates_list:
        raise ValueError("'updates' harus berupa array objek modifikasi turn [{turn_index, content}]")

    messages = target_record.get("messages", [])
    total_msgs = len(messages)
    modified_count = 0
    edit_logs: list[dict[str, Any]] = []

    for up in updates_list:
        if not isinstance(up, dict):
            continue
        turn_idx_raw = up.get("turn_index")
        if turn_idx_raw is None:
            raise ValueError(f"Setiap update wajib menentukan 'turn_index': {up}")
        try:
            t_idx = int(turn_idx_raw)
        except (ValueError, TypeError):
            raise ValueError(f"'turn_index' harus integer: {turn_idx_raw}")

        if t_idx < 0 or t_idx >= total_msgs:
            raise ValueError(f"'turn_index' {t_idx} di luar rentang pesan (0 .. {total_msgs - 1})")

        msg = messages[t_idx]
        curr_role = msg.get("role", "")
        new_content = up.get("content")
        new_prefixes = up.get("prefixes")

        if new_content is None:
            raise ValueError(f"Update turn_index {t_idx} wajib menyertakan 'content'")

        new_content_str = str(new_content).strip()

        if curr_role == "assistant":
            cleaned_content = _normalize_prefix_text(new_content_str, new_prefixes)
            min_len = 40 if target_record.get("category") == "vision_chat" else 60
            if len(cleaned_content) < min_len:
                raise ValueError(
                    f"Turn assistant [{t_idx}] terlalu pendek ({len(cleaned_content)} kar). "
                    f"Minimal elaborasi {min_len} karakter."
                )
            msg["content"] = cleaned_content
        elif curr_role == "user":
            user_clean = re.sub(r"<unused[1-6]>", "", new_content_str).strip()
            if t_idx == 1 and target_record.get("category") == "vision_chat" and not user_clean.startswith("📷"):
                user_clean = "📷\n" + user_clean
            msg["content"] = user_clean
        elif curr_role == "system":
            msg["content"] = new_content_str

        modified_count += 1
        edit_logs.append({
            "turn_index": t_idx,
            "role": curr_role,
            "snippet": msg["content"][:100] + ("..." if len(msg["content"]) > 100 else ""),
        })

    target_record["reviewed"] = True
    target_record["edited_turns"] = int(target_record.get("edited_turns", 0)) + modified_count
    lines[target_idx] = json.dumps(target_record, ensure_ascii=False) + "\n"

    temp_file = path.with_suffix(".tmp_review")
    with open(temp_file, "w", encoding="utf-8") as out_f:
        out_f.writelines(lines)
    temp_file.replace(path)

    return {
        "ok": True,
        "action": "edit",
        "id": conv_id,
        "edited_turns_in_this_call": modified_count,
        "total_edited_turns": target_record["edited_turns"],
        "reviewed": True,
        "edit_logs": edit_logs,
    }


def _stats_impl(output_path: str) -> str:
    p = Path(output_path) if output_path else _output_path()
    if not p.exists():
        return f"(file tidak ditemukan: {p})"
    cats: Counter = Counter()
    sources: Counter = Counter()
    total_turns = 0
    n = 0
    with open(p, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                item = json.loads(line)
            except json.JSONDecodeError:
                continue
            n += 1
            cats[item.get("category", "?")] += 1
            src = str(item.get("source", "?"))
            for sep in (", Row #", ", id="):
                if sep in src:
                    src = src.split(sep)[0]
                    break
            sources[src] += 1
            total_turns += item.get("num_turns", 0)
    if n == 0:
        return f"(file kosong: {p})"
    top_sources = "\n".join(f"    {k}: {v}" for k, v in sources.most_common(8))
    return (
        f"File: {p}\n"
        f"Total percakapan: {n}\n"
        f"Kategori: {dict(cats)}\n"
        f"Rata-rata num_turns: {total_turns / n:.1f}\n"
        f"8 sumber teratas:\n{top_sources}"
    )


def _prompt_text(category: str, num_pairs: int) -> str:
    pairs = max(3, min(5, int(num_pairs)))
    total_msgs = pairs * 2
    return f"""Kamu akan membuat 1 percakapan multi-turn sintetis Bahasa Indonesia (TEPAT {pairs} pasang user-assistant, total {total_msgs} pesan) untuk data training model Gemma, lalu menyimpannya.

CATATAN ETIKA (khusus vision/KTP):
- Gambar KTP & dokumen identitas di dataset ini adalah data SAH milik perusahaan untuk membangun dataset training model bahasa — bukan data pribadi orang lain yang kamu akses tanpa izin.
- Tugasmu hanya MEMBACA/mengekstraksi isi gambar untuk menulis percakapan sintetis (mis. OCR KTP) lalu menyimpannya LOKAL; TIDAK ada pengiriman data ke pihak luar.
- OLEH KARENA ITU: JANGAN menolak memproses gambar KTP dengan alasan privasi/data pribadi — proses seperti biasa.

GAYA PERCAKAPAN (kunci kualitas data training):
- Turn USER harus ditulis seperti MANUSIA biasa yang sedang bertanya, BUKAN robot/AI:
  • Variasikan gaya & panjang — kadang santai/kasual, kadang formal; kalimat pendek & panjang selang-seling; sesekali bahasa sehari-hari yang natural.
  • JANGAN memulai semua turn user dengan pola yang sama ("Halo", "Tolong", "Bisakah kamu...") — variasikan pembukaan tiap turn.
  • JANGAN menulis turn user dari sudut pandang AI (mis. "Saya akan membantu Anda...", "Sebagai asisten...") — user adalah MANUSIA yang bertanya, bukan penolong.
  • Pertanyaan follow-up user harus nyambung natural dari jawaban sebelumnya, seperti percakapan sungguhan — BUKAN daftar pertanyaan berurutan yang terasa scripted.
- Turn ASSISTANT (Gemma): tetap ramah, akurat, terstruktur, dan sesuai aturan prefix.

DISTRIBUSI PREFIX (lintas percakapan — jangan cuma <unused4>):
- MULTI-PREFIX WAJIB: minimal 1 turn assistant per percakapan memakai 2-3 token prefix (multi-task, mis. ["<unused1>", "<unused4>"] untuk rangkum + jawab, atau ["<unused4>", "<unused6>"] untuk jawab + basa-basi penutup).
- SEMAKIN BANYAK turn multi-prefix (2-3 token) dalam satu percakapan SEMAKIN BAGUS (prioritas tinggi): gunakan di 2-3 turn bila natural. Hanya turun ke 1 turn bila memaksakan multi-prefix membuat percakapan jadi aneh/tidak natural.
- Usahakan SEMUA token <unused1..6> terpakai SEIMBANG di seluruh dataset, bukan hanya beberapa yang favorit.
- Sebelum menulis, periksa field prefix_stats & prefix_hint (di get_progress() dan di hasil sample_row): kalau ada token yang masih JARANG dipakai (mis. <unused5> PARAPHRASE / <unused6> GENERAL_CHAT), tulis percakapan yang memang butuh task itu — misalnya percakapan 1 pakai <unused1>/<unused4>, percakapan 2 pakai <unused5>/<unused6>, dan seterusnya.
- Token TETAP harus sesuai task isi pesan — jangan memaksakan token yang tidak cocok.

VARIASI JUMLAH PASANG (3/4/5 = 6/8/10 pesan, lintas percakapan):
- Seimbangkan pemakaian 3, 4, dan 5 pasang — jangan selalu 4 pasang.
- Sebelum menulis, periksa field pairs_stats & pairs_hint (di get_progress() dan di hasil sample_row): kalau ada jumlah pasang yang masih JARANG (mis. 5 pasang = 10 pesan), tulis percakapan dengan jumlah pasang itu.

LANGKAH:
0. (Wajib sekali jalan) Panggil get_progress() untuk melihat kuota: target 2000 text + 1000 vision (2:1), jumlah sudah/belum, dan kategori berikutnya.
1. (Opsional) Panggil list_sources(category="{category}") untuk melihat katalog sumber.
2. Panggil sample_row() TANPA category → otomatis memilih jenis (text/vision) yang paling tertinggal rasionya (2:1), INSTAN (random-access datasets-server, tanpa unduh dataset penuh). Row yang sudah dipakai otomatis diskip (exclude).
3. Bila hasilnya vision (ada image_ref non-null): GAMBARNYA SUDAH DISERTAKAN langsung di hasil tool (ImageContent) — langsung lihat, TANPA perlu read_image terpisah.
4. Tulis percakapan yang 100% nyambung dengan konteks asli, dengan aturan:
   - GROUNDING: SEBELUM menulis, identifikasi 2-3 FAKTA KUNCI dari konteks asli yang akan jadi dasar percakapan; seluruh turn harus konsisten dengan fakta itu (JANGAN menambah fakta di luar konteks).
   - JUMLAH PESAN WAJIB TEPAT {total_msgs} pesan ({pairs} pasang user-assistant, selang-seling). {total_msgs} pesan = {pairs} pasang; save_conversation MENOLAK bila bukan 6/8/10 pesan.
   - Setiap turn assistant WAJIB elaborasi MINIMAL 40 karakter (vision) / 60 karakter (text) — jangan jawaban satu kalimat pendek.
   - User pertama pada percakapan vision WAJIB diawali token 📷.
   - Setiap turn assistant WAJIB punya "prefixes" berisi 1-3 token UNIK (maksimal 3, tanpa spasi/duplikat) sesuai task pesan:
     "<unused1>" SUMMARIZE, "<unused2>" TRANSLATE, "<unused3>" NER, "<unused4>" QA, "<unused5>" PARAPHRASE, "<unused6>" GENERAL_CHAT.
     Multi-task boleh (mis. ["<unused1>", "<unused2>"] untuk rangkum + terjemah) — maksimal 3 token per pesan.
   - WAJIB minimal 1 turn assistant memakai 2-3 prefix (multi-task); SEMAKIN BANYAK turn multi-prefix semakin bagus (save_conversation MENOLAK kalau tidak ada satupun turn multi-prefix).
   - DILARANG menulis token <unusedX> di dalam konten; DILARANG menulis kata nama task (SUMMARIZE/NER/dll) di konten.
   - Bahasa Indonesia natural.
5. Panggil save_conversation dengan:
   - source: "<field `source` PERSIS dari output sample_row>" (penting untuk deduplikasi/exclude baris)
   - category: "vision_chat" bila ada image_ref, selain itu "text_nlu_chat"
   - image_ref: "<image_ref dari sample_row>" (bila vision)
   - num_pairs: {pairs} (WAJIB dikirim agar jumlah turn dipaksa cocok)
   - conversation_json: array turn JSON, contoh:
     [{{"role": "user", "content": "...", "prefixes": []}}, {{"role": "assistant", "content": "...", "prefixes": ["<unused4>"]}}]
6. Setelah tiap save: panggil get_progress() lagi, lanjutkan sample berikutnya sampai kuota target tercapai (kerjakan dalam batch ~10 text + 5 vision per sesi kerja).
7. Bila save_conversation mengembalikan error (isError), perbaiki percakapan dan coba lagi.
8. Laporkan ringkas: id, path file, jumlah pasang dialog, dan sisa kuota text/vision."""


# ─── MCP Handlers (low-level, mcp>=2.0) ───────────────────────────────────────

def _tool_defs() -> list[types.Tool]:
    return [
        types.Tool(name="list_sources", description="Katalog INSTANT 15 sumber Indonesia (text/vision) — random-access datasets-server, tanpa load dataset, selalu cepat. category: all|text|vision.",  # type: ignore
                   input_schema={"type": "object", "properties": {"category": {"type": "string", "enum": ["all", "text", "vision"], "default": "all"}}, "additionalProperties": False}),
        types.Tool(name="sample_row", description="Sample 1 baris dataset asli secara acak via random-access datasets-server — INSTAN untuk semua sumber (termasuk Cendol 12,8jt row), tanpa unduh dataset penuh. Row yang sudah dipakai otomatis diskip (exclude). Untuk VISION hasilnya LANGSUNG menyertakan gambarnya (ImageContent) — tidak perlu read_image terpisah. category='auto' (default) → pilih jenis sesuai kuota 2:1 (yang paling tertinggal). include_heavy=false → batasi ke pool ringan (IndoMMLU/IndoCareer/IndoCulture/KTP-VLM). source_key opsional dari list_sources.",  # type: ignore
                   input_schema={"type": "object", "properties": {"category": {"type": "string", "enum": ["auto", "text", "vision"], "default": "auto"}, "source_key": {"type": "string"}, "seed": {"type": "integer"}, "include_heavy": {"type": "boolean", "default": True}}, "additionalProperties": False}),
        types.Tool(name="get_progress", description="Kuota & progress: target default 2000 text / 1000 vision (2:1), jumlah sudah/belum tersimpan (dihitung dari file output), kategori berikutnya, dan source yang disarankan.",  # type: ignore
                   input_schema={"type": "object", "properties": {}, "additionalProperties": False}),
        types.Tool(name="read_image", description="MULTIMODAL: baca gambar dari path lokal / URL (termasuk URL datasets-server), kembalikan ImageContent (base64 PNG, downscale 1024px) — Claude bisa melihat gambarnya.",  # type: ignore
                   input_schema={"type": "object", "properties": {"image_ref": {"type": "string"}}, "required": ["image_ref"], "additionalProperties": False}),
        types.Tool(name="save_conversation", description="Validasi & simpan percakapan yang DITULIS CLAUDE ke file JSONL. WAJIB TEPAT 6/8/10 pesan (3-5 pasang) & turn assistant minimal 60 karakter — kalau kurang, isError. num_pairs (3-5) opsional tapi disarankan agar jumlah turn dipaksa cocok. conversation_json: array turn [{role, content, prefixes}]. source: field `source` PERSIS dari output sample_row (dipakai untuk deduplikasi/exclude).",  # type: ignore
                   input_schema={"type": "object", "properties": {"source": {"type": "string"}, "category": {"type": "string"}, "conversation_json": {"type": "string"}, "image_ref": {"type": "string"}, "output_path": {"type": "string"}, "num_pairs": {"type": "integer"}},
                                "required": ["source", "category", "conversation_json"], "additionalProperties": False}),
        types.Tool(name="get_output_stats", description="Statistik file output JSONL (jumlah percakapan, kategori, sumber teratas, turn rata-rata).",  # type: ignore
                   input_schema={"type": "object", "properties": {"output_path": {"type": "string"}}, "additionalProperties": False}),
        types.Tool(name="review_conversation", description="Inspeksi (action='read') atau edit manual turn percakapan (action='edit') pada file output JSONL. Mendukung verifikasi isi, perbaikan role user/assistant, normalisasi multi-prefix bebas spasi/repetisi, dan pengeditan multi-turn sekaligus via looping updates=[{turn_index, content, prefixes}].",  # type: ignore
                   input_schema={"type": "object", "properties": {"conversation_id": {"type": "integer", "description": "ID percakapan yang ingin dibaca atau diedit"}, "action": {"type": "string", "enum": ["read", "edit"], "default": "read", "description": "'read' untuk melihat isi pesan, 'edit' untuk memperbarui pesan"}, "updates": {"type": "array", "description": "Daftar pembaruan turn jika action='edit': [{turn_index, content, prefixes}]", "items": {"type": "object", "properties": {"turn_index": {"type": "integer"}, "content": {"type": "string"}, "prefixes": {"type": "array", "items": {"type": "string"}}}}}, "output_path": {"type": "string", "description": "Path kustom file JSONL (opsional)"}}, "required": ["conversation_id"], "additionalProperties": False}),
    ]


async def handle_list_tools(ctx, params: types.PaginatedRequestParams | None) -> types.ListToolsResult:
    return types.ListToolsResult(tools=_tool_defs())


async def handle_call_tool(ctx, params: types.CallToolRequestParams) -> types.CallToolResult:
    name = params.name
    args = params.arguments or {}
    try:
        if name == "list_sources":
            return types.CallToolResult(content=[types.TextContent(type="text", text=_list_sources_impl(str(args.get("category", "all"))))])
        if name == "get_progress":
            return types.CallToolResult(content=[types.TextContent(type="text", text=_progress_impl())])
        if name == "sample_row":
            info = await _sample_row_impl(
                str(args.get("category", "auto")), str(args.get("source_key", "")),
                args.get("seed"), bool(args.get("include_heavy", True)),
            )
            # TextContent (JSON konteks) + ImageContent (gambar vision) dalam SATU hasil
            parts: list[Any] = [types.TextContent(type="text", text=json.dumps(info, ensure_ascii=False, indent=2))]
            if info.get("image_ref") and info.get("image_available"):
                try:
                    b64, mime = _read_image_base64(str(info["image_ref"]))
                    parts.append(types.ImageContent(type="image", data=b64, mime_type=mime))  # type: ignore
                except Exception as e:
                    parts.append(types.TextContent(type="text", text=f"⚠️ (gambar gagal dimuat: {e})"))
            return types.CallToolResult(content=parts)
        if name == "read_image":
            b64, mime = _read_image_base64(str(args.get("image_ref", "")))
            return types.CallToolResult(content=[types.ImageContent(type="image", data=b64, mime_type=mime)])  # type: ignore
        if name == "save_conversation":
            result = _save_conversation_impl(
                str(args.get("source", "")),
                str(args.get("category", "")),
                str(args.get("conversation_json", "")),
                str(args.get("image_ref", "")),
                str(args.get("output_path", "")),
                str(args.get("num_pairs", "")),
            )
            return types.CallToolResult(content=[types.TextContent(type="text", text=json.dumps(result, ensure_ascii=False, indent=2))])
        if name == "get_output_stats":
            return types.CallToolResult(content=[types.TextContent(type="text", text=_stats_impl(str(args.get("output_path", ""))))])
        if name == "review_conversation":
            res_review = _review_conversation_impl(
                conversation_id=args.get("conversation_id"),
                action=str(args.get("action", "read")),
                updates=args.get("updates"),
                output_path=str(args.get("output_path", "")),
            )
            parts_review: list[Any] = [types.TextContent(type="text", text=json.dumps(res_review, ensure_ascii=False, indent=2))]
            # Jika mode read dan ada gambar terkait, sertakan ImageContent agar agent langsung bisa melihat gambarnya
            if res_review.get("action") == "read" and res_review.get("images"):
                source_val = str(res_review.get("source", ""))
                for img_ref in res_review["images"]:
                    try:
                        b64_img, mime_img = _read_image_base64(str(img_ref), source=source_val)
                        parts_review.append(types.ImageContent(type="image", data=b64_img, mime_type=mime_img))
                    except Exception as e:
                        parts_review.append(types.TextContent(type="text", text=f"⚠️ (gambar {img_ref} gagal dimuat: {e})"))
            return types.CallToolResult(content=parts_review)
        return types.CallToolResult(is_error=True, content=[types.TextContent(type="text", text=f"Unknown tool: {name}")])
    except Exception as e:
        # Tool execution error (isError=true) — Claude bisa self-correct (spesifikasi MCP)
        return types.CallToolResult(is_error=True, content=[types.TextContent(type="text", text=str(e))])


async def handle_list_prompts(ctx, params: types.PaginatedRequestParams | None) -> types.ListPromptsResult:
    return types.ListPromptsResult(prompts=[
        types.Prompt(
            name="generate_conversation",
            description="Buat 1 percakapan sintetis Bahasa Indonesia dari dataset asli, lalu simpan via save_conversation",
            arguments=[
                types.PromptArgument(name="category", description="text | vision", required=False),
                types.PromptArgument(name="num_pairs", description="Jumlah pasang dialog (3-5)", required=False),
            ],
        )
    ])


async def handle_get_prompt(ctx, params: types.GetPromptRequestParams) -> types.GetPromptResult:
    if params.name != "generate_conversation":
        raise ValueError(f"Unknown prompt: {params.name}")
    category = str((params.arguments or {}).get("category", "text"))
    num_pairs = int((params.arguments or {}).get("num_pairs", 4))
    text = _prompt_text(category, num_pairs)
    return types.GetPromptResult(
        description="Buat 1 percakapan sintetis Bahasa Indonesia dari dataset asli, lalu simpan",
        messages=[types.PromptMessage(role="user", content=types.TextContent(type="text", text=text))],  # type: ignore[arg-type]
    )


server = Server(
    "generate-conv-indonesia",
    version="2.1.18",
    title="Generator Percakapan Indonesia (Text + Vision)",  # type: ignore
    description="Buat percakapan sintetis Bahasa Indonesia untuk training Gemma — CLAUDE yang menulis, server menyediakan data asli (random-access) + validasi + penyimpanan.",  # type: ignore
    on_list_tools=handle_list_tools,  # type: ignore
    on_call_tool=handle_call_tool,  # type: ignore
    on_list_prompts=handle_list_prompts,  # type: ignore
    on_get_prompt=handle_get_prompt,  # type: ignore
)


async def run_stdio() -> None:
    # Muat exclusion set dari file output (jika sudah ada) — tanpa preload dataset.
    excluded = await asyncio.to_thread(_refresh_exclusions)
    if excluded:
        print(f"🚫 EXCLUDE: {excluded} row/id yang sudah dipakai dimuat dari {_output_path()}", file=sys.stderr)
    tt, vt = _targets()
    print(f"🎯 KUOTA: {tt} text + {vt} vision (2:1)", file=sys.stderr)
    async with stdio_server() as (read_stream, write_stream):
        await server.run(read_stream, write_stream, server.create_initialization_options())


def _env_int(name: str, default: int) -> int:
    v = os.environ.get(name, "").strip()
    try:
        return int(v) if v else default
    except ValueError:
        return default


def main() -> None:
    ap = argparse.ArgumentParser(description="MCP Bundle — Generator Percakapan Indonesia (Claude Desktop yang mengerjakan, low-level MCP SDK)")
    ap.add_argument("--transport", choices=["stdio", "streamable-http"], default="stdio", help="Transport MCP (default: stdio)")
    ap.add_argument("--host", type=str, default="127.0.0.1", help="Bind host untuk streamable-http")
    ap.add_argument("--port", type=int, default=8765, help="Port untuk streamable-http")
    ap.add_argument("--output-dir", type=str, default="", help="Folder penyimpanan JSONL (default: env OUTPUT_DIR atau ~/generate-conv-indonesia)")
    ap.add_argument("--text-target", type=int, default=None, help="Target percakapan text (default: env TEXT_TARGET atau 2000)")
    ap.add_argument("--vision-target", type=int, default=None, help="Target percakapan vision (default: env VISION_TARGET atau 1000)")
    args = ap.parse_args()

    _CONFIG.update({
        "output_dir": args.output_dir or os.environ.get("OUTPUT_DIR") or str(Path.home() / "generate-conv-indonesia"),
        "text_target": args.text_target if args.text_target is not None else _env_int("TEXT_TARGET", 2000),
        "vision_target": args.vision_target if args.vision_target is not None else _env_int("VISION_TARGET", 1000),
    })

    print(f"🧩 MCP 'generate-conv-indonesia' v2.1.18 | transport={args.transport} | output_dir={_CONFIG['output_dir']} "
          f"| kuota {_CONFIG['text_target']} text / {_CONFIG['vision_target']} vision", file=sys.stderr)
    if args.transport == "streamable-http":
        import uvicorn
        uvicorn.run(server.streamable_http_app(), host=args.host, port=args.port)  # type: ignore
    else:
        asyncio.run(run_stdio())


if __name__ == "__main__":
    main()
