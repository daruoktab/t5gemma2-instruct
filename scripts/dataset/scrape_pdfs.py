"""
scrape_pdfs.py (Agent Version)
================================
Refactored menggunakan pydantic-ai Agent pattern.
Agent memiliki tools lengkap untuk mengontrol seluruh pipeline scraping:
  - check_existing_docs()      → lihat PDF apa yang sudah ada + topiknya
  - search_pdfs(query)         → cari URL PDF di Yahoo
  - download_pdf(url)          → unduh PDF ke lokal
  - read_pdf_text(path)        → baca isi teks PDF (untuk evaluasi)
  - accept_pdf(path, topic, reason) → konversi ke gambar & simpan metadata
  - reject_pdf(path, reason)   → hapus PDF & lapor alasan
  - get_progress()             → cek progress (berapa yang sudah diterima)

Agent bertugas:
  1. Cek dokumen yang sudah ada, identifikasi topik yang sudah tercover
  2. Tentukan topik/query baru yang belum ada
  3. Cari, unduh, baca, dan evaluasi setiap PDF secara mandiri
  4. Putuskan menerima (≤10 halaman, konten kasual) atau menolak
  5. Lanjutkan sampai target terpenuhi
"""

import asyncio
import os
import re
import sys
import json
import time
import urllib.request
import urllib.parse
import ssl
import shutil
from pathlib import Path
from dataclasses import dataclass, field
from typing import cast

import httpx
from pydantic import BaseModel, Field
from pydantic_ai import Agent, RunContext

# ─── Load .env ────────────────────────────────────────────────────────────────
SCRIPT_DIR = Path(__file__).resolve().parent
ROOT_DIR = SCRIPT_DIR.parent.parent

try:
    from dotenv import load_dotenv
    load_dotenv(ROOT_DIR / ".env")
except ImportError:
    pass

# ─── Path Konfigurasi ─────────────────────────────────────────────────────────
PDF_DIR = ROOT_DIR / "data" / "multimodal" / "raw_pdfs"
IMAGE_DIR = ROOT_DIR / "data" / "multimodal" / "images"
DOC_METADATA_FILE = IMAGE_DIR / "doc_metadata.json"

# ─── SSL Context ──────────────────────────────────────────────────────────────
try:
    SSL_CONTEXT = ssl._create_unverified_context()
except AttributeError:
    SSL_CONTEXT = None

# ─── Konfigurasi API ──────────────────────────────────────────────────────────
API_KEY = os.environ.get("OPENAGENTIC_API_KEY", "")
API_BASE_URL = os.environ.get("OPENAGENTIC_API_URL") or "https://openagentic.id/api/v1"
API_MODEL = os.environ.get("OPENAGENTIC_MODEL") or os.environ.get("MODEL_NAME") or "claude-sonnet-4.6"

# ─── State ────────────────────────────────────────────────────────────────────
@dataclass
class ScraperState:
    target_docs: int
    pdf_idx: int                        # Index PDF berikutnya
    accepted_count: int                 # Berapa yang sudah diterima
    tried_urls: set = field(default_factory=set)     # URL yang sudah dicoba (hindari duplikat)
    pending_pdf_path: str = ""          # Path PDF yang sedang di-review agent

# ─── Output Schema ────────────────────────────────────────────────────────────
class ScraperResult(BaseModel):
    is_done: bool = Field(description="Apakah target dokumen sudah tercapai atau semua URL sudah habis?")
    summary: str = Field(description="Ringkasan akhir: berapa yang diterima, ditolak, dan kenapa selesai.")

# ─── Rate Limited HTTP Client ─────────────────────────────────────────────────
class RateLimitedAsyncClient(httpx.AsyncClient):
    def __init__(self, *args, rate_limit_delay: float = 2.0, **kwargs):
        super().__init__(*args, **kwargs)
        self.rate_limit_delay = rate_limit_delay
        self.last_request_time = 0.0
        self.request_lock = asyncio.Lock()

    async def send(self, request: httpx.Request, *args, **kwargs):
        async with self.request_lock:
            now = time.time()
            elapsed = now - self.last_request_time
            if elapsed < self.rate_limit_delay:
                await asyncio.sleep(self.rate_limit_delay - elapsed)
            self.last_request_time = time.time()

        for attempt in range(5):
            try:
                request.read()
                response = await super().send(request, *args, **kwargs)
                if response.status_code == 429:
                    wait_time = (attempt + 1) * 30
                    print(f"  [!] HTTP 429. Menunggu {wait_time}s...")
                    await asyncio.sleep(wait_time)
                    continue
                if response.status_code in [500, 502, 503, 504]:
                    await asyncio.sleep((attempt + 1) * 5)
                    continue
                return response
            except Exception as e:
                if attempt == 4:
                    raise e
                await asyncio.sleep((attempt + 1) * 5)

def create_model_instance(base_url: str, api_key: str, model_name: str):
    from pydantic_ai.models.openai import OpenAIChatModel
    from pydantic_ai.providers.openai import OpenAIProvider
    client = RateLimitedAsyncClient(rate_limit_delay=2.0, timeout=60.0)
    provider = OpenAIProvider(base_url=base_url, api_key=api_key, http_client=client)
    return OpenAIChatModel(model_name, provider=provider)

# ─── Agent ────────────────────────────────────────────────────────────────────
agent = Agent(
    f"openai-chat:{API_MODEL}",
    deps_type=ScraperState,
    output_type=ScraperResult,
    retries=3,
    system_prompt=(
        "Kamu adalah agen pencari dan kurator dokumen PDF Indonesia untuk dataset multimodal visual AI.\n\n"
        "MISIMU:\n"
        "Temukan dan kumpulkan dokumen PDF kasual/sehari-hari Indonesia yang cocok untuk percakapan santai.\n"
        "Target: dokumen tentang resep, brosur wisata, info warga, menu kuliner, panduan hobi, buletin komunitas, dsb.\n\n"
        "KRITERIA DOKUMEN YANG DITERIMA:\n"
        "✅ Konten kasual/sehari-hari (resep, brosur, info warga, panduan umum, katalog ringan)\n"
        "✅ Bahasa Indonesia\n"
        "✅ Maksimal 10 halaman\n"
        "✅ Bisa divisualkan dan dijadikan bahan percakapan\n\n"
        "KRITERIA DOKUMEN YANG DITOLAK:\n"
        "❌ Paper akademik / skripsi / tesis\n"
        "❌ Dokumen hukum formal (undang-undang, peraturan)\n"
        "❌ Laporan keuangan atau database teknis\n"
        "❌ Lebih dari 10 halaman\n"
        "❌ Tidak ada teks (hanya scan/gambar)\n"
        "❌ Topik yang sudah banyak tercover\n\n"
        "ALUR KERJA:\n"
        "1. Panggil `check_existing_docs()` untuk lihat apa yang sudah ada dan topik apa yang belum tercover\n"
        "2. Tentukan query pencarian yang relevan dengan topik yang belum ada\n"
        "3. Panggil `search_pdfs(query)` untuk cari URL PDF\n"
        "4. Untuk setiap URL: `download_pdf(url)`, lalu `read_pdf_text(path)` untuk baca isinya\n"
        "5. Evaluasi apakah konten sesuai kriteria dan halamannya ≤ 10\n"
        "6. Panggil `accept_pdf(path, topic, reason)` atau `reject_pdf(path, reason)`\n"
        "7. Panggil `get_progress()` secara berkala untuk cek status\n"
        "8. Ulangi sampai target tercapai atau tidak ada URL tersisa\n\n"
        "PENTING: Jangan menerima dokumen tentang topik yang sudah banyak ada. Prioritaskan keberagaman topik!"
    )
)

# ─── Tools ────────────────────────────────────────────────────────────────────

@agent.tool
def check_existing_docs(ctx: RunContext[ScraperState]) -> str:
    """Lihat daftar dokumen yang sudah ada, distribusi topik, dan rekomendasi topik yang masih kurang."""
    if not DOC_METADATA_FILE.exists():
        return "Belum ada dokumen yang berhasil di-scrape. Mulai dari nol!"

    try:
        with DOC_METADATA_FILE.open("r", encoding="utf-8") as f:
            doc_metadata = json.load(f)
    except Exception as e:
        return f"Gagal membaca metadata: {e}"

    if not doc_metadata:
        return "Metadata ada tapi kosong. Belum ada dokumen."

    total = len(doc_metadata)
    remaining = ctx.deps.target_docs - total

    lines = [
        f"=== STATUS DOKUMEN ===",
        f"Total tersedia : {total}/{ctx.deps.target_docs}",
        f"Masih butuh    : {remaining} dokumen lagi\n",
        "=== DAFTAR LENGKAP (dengan topik & alasan) ==="
    ]

    # Tampilkan setiap dokumen — gunakan agent_reason sebagai sumber topik utama
    # karena format lama tidak punya field "topic"
    for doc_key, meta in sorted(doc_metadata.items()):
        topic = meta.get("topic", "")
        reason = meta.get("agent_reason", "")
        pages = len(meta.get("pages", []))
        source_url = meta.get("source_url", "")

        # Gunakan 80 karakter pertama dari agent_reason sebagai deskripsi topik
        topic_display = topic if topic else reason[:100]
        lines.append(f"  [{doc_key}] {pages} hal | {topic_display}...")
        if source_url and source_url.startswith("http"):
            lines.append(f"    URL: {source_url[:80]}")

    # Analisis distribusi topik berdasarkan kata kunci dari agent_reason
    keywords = {
        "resep/kuliner": ["resep", "masakan", "kuliner", "makanan", "kue", "menu", "takjil"],
        "brosur sekolah/bimbel": ["sekolah", "bimbel", "les", "ppdb", "siswa", "smp", "sma", "smk", "tk", "penerimaan"],
        "wisata": ["wisata", "brosur wisata", "pariwisata", "destinasi", "liburan", "peta wisata"],
        "berkebun/tanaman": ["berkebun", "tanaman", "tanaman hias", "hortikultura", "kebun"],
        "kesehatan": ["kesehatan", "tips kesehatan", "sehat", "keluarga sehat"],
        "keagamaan/komunitas": ["kajian", "masjid", "buletin", "warga", "komunitas", "syukuran", "pernikahan"],
        "panduan/tips": ["panduan", "tips", "cara", "petunjuk"],
        "pemerintahan": ["rt", "rw", "karang taruna", "siskamling", "pemerintah"],
        "hobi/lifestyle": ["hobi", "peliharaan", "kreatif", "fotografi", "seni"],
        "lainnya": [],
    }

    topic_counts = {k: [] for k in keywords}
    for doc_key, meta in doc_metadata.items():
        reason_lower = (meta.get("agent_reason", "") + " " + meta.get("topic", "")).lower()
        matched = False
        for topic_cat, kws in keywords.items():
            if topic_cat == "lainnya":
                continue
            if any(kw in reason_lower for kw in kws):
                topic_counts[topic_cat].append(doc_key)
                matched = True
                break
        if not matched:
            topic_counts["lainnya"].append(doc_key)

    lines.append("\n=== DISTRIBUSI TOPIK (untuk menghindari duplikasi) ===")
    for topic_cat, docs in topic_counts.items():
        if docs:
            bar = "█" * len(docs)
            lines.append(f"  {topic_cat:<30} {bar} ({len(docs)})")

    lines.append("\n=== REKOMENDASI TOPIK YANG MASIH KURANG ===")
    missing = [cat for cat, docs in topic_counts.items() if len(docs) == 0 and cat != "lainnya"]
    low = [cat for cat, docs in topic_counts.items() if 0 < len(docs) <= 2 and cat != "lainnya"]
    overloaded = [cat for cat, docs in topic_counts.items() if len(docs) >= 5 and cat != "lainnya"]

    if missing:
        lines.append(f"🔴 BELUM ADA sama sekali: {', '.join(missing)}")
    if low:
        lines.append(f"🟡 Masih sedikit (1-2): {', '.join(low)}")
    if overloaded:
        lines.append(f"🟢 Sudah banyak (≥5), hindari: {', '.join(overloaded)}")

    lines.append(
        f"\nSARANAN PENCARIAN BERIKUTNYA: Fokus pada topik yang 🔴 atau 🟡 di atas. "
        f"Hindari duplikasi topik 🟢!"
    )

    return "\n".join(lines)


@agent.tool
def search_pdfs(ctx: RunContext[ScraperState], query: str) -> str:
    """Cari URL PDF di Yahoo Search berdasarkan query yang diberikan. Kembalikan daftar URL yang belum dicoba."""
    print(f"  [SEARCH] Query: '{query}'")
    encoded_query = urllib.parse.quote_plus(query)
    url = f"https://search.yahoo.com/search?p={encoded_query}"

    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
    }

    req = urllib.request.Request(url, headers=headers)
    pdf_urls = []

    try:
        with urllib.request.urlopen(req, timeout=15, context=SSL_CONTEXT) as r:
            html = r.read().decode("utf-8", errors="ignore")

        # Cari redirect Yahoo /RU=URL/
        matches = re.findall(r'/RU=([^/&"]+)', html)
        for m in matches:
            decoded = urllib.parse.unquote(m)
            if ".pdf" in decoded.lower() and decoded not in ctx.deps.tried_urls:
                pdf_urls.append(decoded)

        # Cari tautan langsung
        direct = re.findall(r'href="(https?://[^"]+\.pdf)"', html)
        for m in direct:
            if m not in ctx.deps.tried_urls and m not in pdf_urls:
                pdf_urls.append(m)

    except Exception as e:
        return f"GAGAL mencari di Yahoo: {e}"

    time.sleep(2)

    if not pdf_urls:
        return f"Tidak menemukan URL PDF baru untuk query: '{query}'"

    # Tambahkan ke tried_urls agar tidak dicoba ulang
    ctx.deps.tried_urls.update(pdf_urls)

    result = f"Menemukan {len(pdf_urls)} URL PDF baru:\n"
    for i, u in enumerate(pdf_urls[:15]):  # Tampilkan max 15
        result += f"  [{i+1}] {u}\n"
    if len(pdf_urls) > 15:
        result += f"  ... dan {len(pdf_urls) - 15} lainnya\n"

    return result


@agent.tool
def download_pdf(ctx: RunContext[ScraperState], url: str) -> str:
    """Unduh PDF dari URL ke folder lokal. Kembalikan path file jika berhasil."""
    PDF_DIR.mkdir(parents=True, exist_ok=True)
    dest = PDF_DIR / f"scraped_{ctx.deps.pdf_idx}.pdf"

    print(f"  [DOWNLOAD] {url[:80]}...")

    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
    }
    req = urllib.request.Request(url, headers=headers)

    try:
        with urllib.request.urlopen(req, timeout=15, context=SSL_CONTEXT) as r, open(dest, "wb") as f:
            f.write(r.read())

        size_kb = dest.stat().st_size // 1024
        if dest.stat().st_size < 5000:
            dest.unlink()
            return f"GAGAL: File terlalu kecil ({dest.stat().st_size} bytes) — bukan PDF valid."

        print(f"  [OK] Berhasil mengunduh ({size_kb} KB) → {dest.name}")
        ctx.deps.pending_pdf_path = str(dest)
        return f"BERHASIL: PDF tersimpan di '{dest}' ({size_kb} KB). Gunakan `read_pdf_text` untuk membaca isinya."

    except Exception as e:
        if dest.exists():
            dest.unlink()
        return f"GAGAL mengunduh: {e}"


@agent.tool
def read_pdf_text(ctx: RunContext[ScraperState], pdf_path: str, max_chars: int = 2000) -> str:
    """Baca teks dari PDF untuk evaluasi konten. Juga mengembalikan jumlah halaman dokumen."""
    try:
        import fitz  # PyMuPDF
    except ImportError:
        return "ERROR: PyMuPDF (fitz) belum terinstal. Jalankan: uv pip install pymupdf"

    path = Path(pdf_path)
    if not path.exists():
        return f"GAGAL: File tidak ditemukan di '{pdf_path}'"

    try:
        doc = fitz.open(str(path))
        num_pages = len(doc)

        # Baca teks dari beberapa halaman pertama
        text_snippet = ""
        for i in range(min(num_pages, 3)):
            text_snippet += doc[i].get_text()

        text_snippet = text_snippet[:max_chars].strip()

        if not text_snippet:
            return (
                f"DOKUMEN TIDAK MEMILIKI TEKS (hanya gambar/scan). Jumlah halaman: {num_pages}.\n"
                "SARAN: Tolak dokumen ini karena tidak dapat dievaluasi."
            )

        result = (
            f"=== INFO DOKUMEN ===\n"
            f"Jumlah halaman: {num_pages}\n"
            f"{'⚠️ MELEBIHI BATAS 10 HALAMAN!' if num_pages > 10 else '✅ Jumlah halaman OK'}\n\n"
            f"=== CUPLIKAN TEKS (dari {min(num_pages, 3)} halaman pertama) ===\n"
            f"{text_snippet}"
        )
        return result

    except Exception as e:
        return f"GAGAL membaca PDF: {e}"


@agent.tool
def accept_pdf(ctx: RunContext[ScraperState], pdf_path: str, topic: str, reason: str) -> str:
    """Terima PDF: konversi ke gambar, simpan metadata, dan hapus file PDF asli."""
    try:
        import fitz
    except ImportError:
        return "ERROR: PyMuPDF (fitz) belum terinstal."

    path = Path(pdf_path)
    if not path.exists():
        return f"GAGAL: File tidak ditemukan di '{pdf_path}'"

    try:
        doc = fitz.open(str(path))
        num_pages = len(doc)

        if num_pages > 10:
            return f"GAGAL: Dokumen memiliki {num_pages} halaman — melebihi batas 10. Gunakan `reject_pdf` saja."

        IMAGE_DIR.mkdir(parents=True, exist_ok=True)

        saved_images = []
        for page_num in range(num_pages):
            page = doc.load_page(page_num)
            pix = page.get_pixmap(dpi=150)
            output_name = f"doc_scraped_{ctx.deps.pdf_idx}_page_{page_num + 1}.png"
            output_path = IMAGE_DIR / output_name
            pix.save(str(output_path))
            saved_images.append(output_name)

        # Simpan metadata
        doc_metadata = {}
        if DOC_METADATA_FILE.exists():
            try:
                with DOC_METADATA_FILE.open("r", encoding="utf-8") as f:
                    doc_metadata = json.load(f)
            except:
                pass

        doc_key = f"doc_scraped_{ctx.deps.pdf_idx}"
        doc_metadata[doc_key] = {
            "pdf_idx": ctx.deps.pdf_idx,
            "source_url": path.name,
            "original_filename": path.name,
            "topic": topic,
            "pages": saved_images,
            "agent_reason": reason
        }

        DOC_METADATA_FILE.parent.mkdir(parents=True, exist_ok=True)
        with DOC_METADATA_FILE.open("w", encoding="utf-8") as f:
            json.dump(doc_metadata, f, ensure_ascii=False, indent=2)

        # Hapus PDF asli
        try:
            path.unlink()
        except:
            pass

        ctx.deps.pdf_idx += 1
        ctx.deps.accepted_count += 1

        print(f"  ✓ DITERIMA: '{topic}' ({num_pages} halaman) → {saved_images}")
        return (
            f"BERHASIL DITERIMA!\n"
            f"Topik: {topic}\n"
            f"Halaman dikonversi: {num_pages}\n"
            f"Total diterima sekarang: {ctx.deps.accepted_count}/{ctx.deps.target_docs}\n"
            f"{'🎉 TARGET TERCAPAI!' if ctx.deps.accepted_count >= ctx.deps.target_docs else f'Butuh {ctx.deps.target_docs - ctx.deps.accepted_count} lagi.'}"
        )

    except Exception as e:
        return f"GAGAL menerima PDF: {e}"


@agent.tool
def reject_pdf(ctx: RunContext[ScraperState], pdf_path: str, reason: str) -> str:
    """Tolak PDF: hapus file dan catat alasan penolakan."""
    path = Path(pdf_path)
    deleted = False
    if path.exists():
        try:
            path.unlink()
            deleted = True
        except Exception as e:
            return f"PDF ditolak tapi gagal dihapus: {e}\nAlasan penolakan: {reason}"

    print(f"  ✗ DITOLAK: {reason[:80]}")
    return f"PDF ditolak dan {'dihapus' if deleted else 'tidak ditemukan'}.\nAlasan: {reason}"


@agent.tool
def get_progress(ctx: RunContext[ScraperState]) -> str:
    """Cek progress scraping saat ini: berapa yang sudah diterima, target, dan file PDF pending."""
    accepted = ctx.deps.accepted_count
    target = ctx.deps.target_docs
    remaining = target - accepted
    tried = len(ctx.deps.tried_urls)
    pending = ctx.deps.pending_pdf_path

    status = (
        f"=== STATUS SCRAPING ===\n"
        f"Diterima  : {accepted}/{target}\n"
        f"Sisa      : {remaining}\n"
        f"URL dicoba: {tried}\n"
        f"PDF pending: {pending if pending else 'tidak ada'}\n"
    )

    if accepted >= target:
        status += "\n🎉 TARGET SUDAH TERCAPAI! Kembalikan hasil akhir."
    else:
        status += f"\nLanjutkan mencari {remaining} dokumen lagi."

    return status


@agent.system_prompt
def add_scraper_context(ctx: RunContext[ScraperState]) -> str:
    return (
        f"STATUS SAAT INI:\n"
        f"- Target dokumen: {ctx.deps.target_docs}\n"
        f"- Sudah diterima: {ctx.deps.accepted_count}\n"
        f"- PDF index berikutnya: {ctx.deps.pdf_idx}\n"
        f"- URL yang sudah dicoba: {len(ctx.deps.tried_urls)}\n\n"
        f"Mulailah dengan memanggil `check_existing_docs()` untuk melihat apa yang sudah ada, "
        f"lalu tentukan strategi pencarian yang tepat untuk melengkapi keberagaman topik."
    )


# ─── Main ─────────────────────────────────────────────────────────────────────
async def main():
    import argparse

    if not API_KEY:
        print("[ERROR] OPENAGENTIC_API_KEY tidak ditemukan di .env atau variabel lingkungan.")
        sys.exit(1)

    parser = argparse.ArgumentParser(description="Agentic PDF Scraper untuk Dataset Multimodal Indonesia")
    parser.add_argument("--docs", type=int, default=200, help="Jumlah total dokumen PDF yang diinginkan")
    parser.add_argument("--model", type=str, default=API_MODEL, help="Nama model LLM")
    args = parser.parse_args()

    PDF_DIR.mkdir(parents=True, exist_ok=True)
    IMAGE_DIR.mkdir(parents=True, exist_ok=True)

    print(f"=== Agentic PDF Scraper ===")
    print(f"[INFO] Endpoint: {API_BASE_URL}")
    print(f"[INFO] Model   : {args.model}")
    print(f"[INFO] Target  : {args.docs} dokumen")

    # Load existing metadata untuk resume
    doc_metadata = {}
    if DOC_METADATA_FILE.exists():
        try:
            with DOC_METADATA_FILE.open("r", encoding="utf-8") as f:
                doc_metadata = json.load(f)
            print(f"[INFO] Resume: {len(doc_metadata)} dokumen sudah ada")
        except:
            pass

    # Tentukan index PDF dan jumlah yang sudah diterima
    existing_indices = [meta["pdf_idx"] for meta in doc_metadata.values() if "pdf_idx" in meta]
    pdf_idx = max(existing_indices) + 1 if existing_indices else 1
    accepted_count = len(doc_metadata)

    if accepted_count >= args.docs:
        print(f"[INFO] Target sudah tercapai ({accepted_count}/{args.docs}). Tidak ada yang perlu dilakukan.")
        return

    state = ScraperState(
        target_docs=args.docs,
        pdf_idx=pdf_idx,
        accepted_count=accepted_count,
    )

    worker_model = create_model_instance(API_BASE_URL, API_KEY, args.model)

    print(f"\n[INFO] Memulai agent scraping dari {accepted_count}/{args.docs}...")
    print(f"[INFO] PDF index mulai dari: {pdf_idx}\n")

    max_retries = 3
    for attempt in range(max_retries):
        try:
            with agent.override(model=worker_model):
                result = await agent.run(
                    "Mulai scraping! Pertama, cek dokumen yang sudah ada, lalu cari dan kumpulkan dokumen PDF baru yang belum tercover.",
                    deps=state
                )

            final = cast(ScraperResult, result.output)
            print(f"\n=== AGEN SELESAI ===")
            print(f"Status: {'✓ Selesai' if final.is_done else '⚠ Belum selesai'}")
            print(f"Ringkasan: {final.summary}")
            print(f"Total diterima: {state.accepted_count}/{args.docs}")
            break

        except Exception as e:
            err = str(e)
            if "429" in err or "rate limit" in err.lower():
                wait = (attempt + 1) * 60
                print(f"[!] Rate Limit. Menunggu {wait} detik (Attempt {attempt+1}/{max_retries})...")
                await asyncio.sleep(wait)
            else:
                print(f"[!] Error Attempt {attempt+1}/{max_retries}: {e}")
                await asyncio.sleep(10)

    print(f"\n=== SELESAI ===")
    print(f"Total dokumen berhasil dikonversi: {state.accepted_count}")


if __name__ == "__main__":
    asyncio.run(main())
