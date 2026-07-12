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
import hashlib
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
from pydantic_ai import Agent, RunContext, UsageLimits

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
IMAGE_DIR = ROOT_DIR / "data" / "multimodal" / "images" / "documents"
DOC_METADATA_FILE = ROOT_DIR / "data" / "multimodal" / "metadata" / "doc_metadata.json"
TRIED_URLS_FILE = ROOT_DIR / "data" / "multimodal" / "metadata" / "tried_urls.json"

# ─── Utility Functions ────────────────────────────────────────────────────────
def _is_valid_pdf_url(url: str) -> bool:
    """Memeriksa apakah URL kemungkinan besar adalah file PDF langsung."""
    url_lower = url.lower()
    
    # Daftar domain sampah / bukan direct PDF download
    blacklist_domains = [
        "scribd.com",
        "academia.edu",
        "pinterest.com",
        "canva.com",
        "play.google.com",
        "youtube.com",
        "facebook.com",
        "instagram.com",
        "twitter.com",
        "github.com",
        "slideshare.net",
        "pdfcoffee.com",
    ]
    
    if any(domain in url_lower for domain in blacklist_domains):
        return False
        
    # Memastikan file berakhiran .pdf (sebelum query param)
    parsed = urllib.parse.urlparse(url)
    path = parsed.path.lower()
    if path.endswith(".pdf"):
        return True
        
    # Fallback: jika ada `.pdf` di dalam URL tapi diikuti parameter (misal .pdf?x=123)
    if re.search(r'\.pdf(?:\?|$)', url_lower):
        return True
        
    return False


def cleanup_temp_files():
    """Hapus file temp PDF lama yang tertinggal dari sesi sebelumnya."""
    if PDF_DIR.exists():
        for f in PDF_DIR.glob("temp_*.pdf"):
            try:
                f.unlink()
            except Exception:
                pass


# ─── SSL Context ──────────────────────────────────────────────────────────────
try:
    SSL_CONTEXT = ssl._create_unverified_context()
except AttributeError:
    SSL_CONTEXT = None

# ─── Konfigurasi API ──────────────────────────────────────────────────────────
def _load_api_keys() -> list[str]:
    keys = []
    # Coba dari OPENROUTER_API_KEY_1 sampai 9
    for i in range(1, 10):
        k = os.environ.get(f"OPENROUTER_API_KEY_{i}")
        if k:
            keys.append(k)
    # Jika tidak ada, coba OPENROUTER_API_KEY tunggal
    if not keys:
        single = os.environ.get("OPENROUTER_API_KEY")
        if single:
            keys.append(single)
    # Jika tidak ada juga, coba OPENAGENTIC_API_KEY
    if not keys:
        single = os.environ.get("OPENAGENTIC_API_KEY")
        if single:
            keys.append(single)
    return keys

API_KEYS: list[str] = _load_api_keys()
API_KEY: str = API_KEYS[0] if API_KEYS else ""

_url = os.environ.get("OPENAGENTIC_API_URL")
if not _url and API_KEY:
    # If falling back to OpenRouter, direct the endpoint to OpenRouter
    _url = "https://openrouter.ai/api/v1"
if not _url:
    _url = "https://openagentic.id/api/v1"
API_BASE_URL: str = _url

default_model = "claude-sonnet-4.6"
if "openrouter.ai" in API_BASE_URL:
    default_model = os.environ.get("OPENROUTER_MODEL") or "google/gemma-4-31b-it:free"

API_MODEL: str = os.environ.get("OPENAGENTIC_MODEL") or os.environ.get("MODEL_NAME") or default_model

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
    def __init__(self, *args, api_keys: list[str] | None = None, rate_limit_delay: float = 2.0, **kwargs):
        super().__init__(*args, **kwargs)
        self.rate_limit_delay = rate_limit_delay
        self.last_request_time = 0.0
        self.request_lock = asyncio.Lock()
        self.api_keys = api_keys or []
        self.current_key_idx = 0

    async def send(self, request: httpx.Request, *args, **kwargs):
        async with self.request_lock:
            now = time.time()
            elapsed = now - self.last_request_time
            if elapsed < self.rate_limit_delay:
                await asyncio.sleep(self.rate_limit_delay - elapsed)
            self.last_request_time = time.time()

        rotations_tried = 0
        for attempt in range(5):
            # Rotasi API Key di header Authorization jika ada beberapa kunci
            if self.api_keys:
                key = self.api_keys[self.current_key_idx]
                request.headers["Authorization"] = f"Bearer {key}"

            try:
                request.read()
                response = await super().send(request, *args, **kwargs)
                if response.status_code == 429:
                    if self.api_keys and len(self.api_keys) > 1:
                        rotations_tried += 1
                        if rotations_tried < len(self.api_keys):
                            old_idx = self.current_key_idx
                            self.current_key_idx = (self.current_key_idx + 1) % len(self.api_keys)
                            print(f"  [!] HTTP 429 Rate Limit. Merotasi API Key dari indeks {old_idx} ke {self.current_key_idx}...")
                            # Tunggu sebentar saja (2s) lalu langsung coba request baru dengan key baru!
                            await asyncio.sleep(2.0)
                            continue
                        else:
                            # Jika semua key dalam pool sudah dicoba dan tetap 429, reset counter dan lakukan sleep panjang
                            rotations_tried = 0
                            wait_time = (attempt + 1) * 30
                            print(f"  [!] Semua API Key ({len(self.api_keys)} kunci) terkena Rate Limit. Menunggu {wait_time}s untuk cooldown...")
                            await asyncio.sleep(wait_time)
                            continue

                    wait_time = (attempt + 1) * 30
                    print(f"  [!] HTTP 429. Menunggu {wait_time}s...")
                    await asyncio.sleep(wait_time)
                    continue
                if response.status_code in [500, 502, 503, 504]:
                    await asyncio.sleep((attempt + 1) * 5)
                    continue
                return response
            except Exception as e:
                # Jika errornya adalah rate limit, rotasi juga
                err_msg = str(e)
                if "429" in err_msg or "rate limit" in err_msg.lower():
                    if self.api_keys and len(self.api_keys) > 1:
                        rotations_tried += 1
                        if rotations_tried < len(self.api_keys):
                            old_idx = self.current_key_idx
                            self.current_key_idx = (self.current_key_idx + 1) % len(self.api_keys)
                            print(f"  [!] Exception Rate Limit. Merotasi API Key dari indeks {old_idx} ke {self.current_key_idx}...")
                            await asyncio.sleep(2.0)
                            continue
                        else:
                            rotations_tried = 0
                            wait_time = (attempt + 1) * 30
                            print(f"  [!] Semua API Key ({len(self.api_keys)} kunci) terkena Exception Rate Limit. Menunggu {wait_time}s...")
                            await asyncio.sleep(wait_time)
                            continue
                if attempt == 4:
                    raise e
                await asyncio.sleep((attempt + 1) * 5)

def create_model_instance(base_url: str, api_keys: list[str], model_name: str):
    from pydantic_ai.models.openai import OpenAIChatModel
    from pydantic_ai.providers.openai import OpenAIProvider
    default_key = api_keys[0] if api_keys else ""
    client = RateLimitedAsyncClient(api_keys=api_keys, rate_limit_delay=2.0, timeout=60.0)
    provider = OpenAIProvider(base_url=base_url, api_key=default_key, http_client=client)
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


def search_duckduckgo_api(query: str, tried_urls: set) -> list[str]:
    """Cari URL PDF di DuckDuckGo menggunakan library ddgs (API internal, lebih tahan anti-bot)."""
    from ddgs import DDGS
    import random
    
    urls = []
    # Pastikan query sudah ada filetype:pdf
    search_query = query if "filetype:pdf" in query.lower() else query + " filetype:pdf"
    
    try:
        with DDGS() as ddgs:
            results = ddgs.text(search_query, max_results=25)
        
        for r in results:
            link = r.get("href", "")
            if _is_valid_pdf_url(link) and link not in tried_urls and link not in urls:
                urls.append(link)
        
        if urls:
            print(f"  [DDG API] Ditemukan {len(urls)} URL PDF baru")
        
        # Jeda acak agar tidak terlalu agresif
        time.sleep(random.randint(2, 5))
        
    except Exception as e:
        err_msg = str(e).lower()
        if "ratelimit" in err_msg or "429" in err_msg:
            print(f"  [DDG API] Rate limit, menunggu 30s...")
            time.sleep(30)
        else:
            print(f"  [DDG API Error] {e}")
    
    return urls


# ─── Brave Search API helper ───
BRAVE_API_KEY = os.getenv("BRAVE_API_KEY", "")

def _search_brave(query: str, tried_urls: set) -> list[str]:
    """Cari URL PDF via Brave Search API (sync, menggunakan httpx langsung)."""
    if not BRAVE_API_KEY:
        print("  [Brave] API key tidak ditemukan, skip Brave Search")
        return []
    
    import random
    
    urls = []
    search_query = query if "pdf" in query.lower() else query + " pdf"
    
    try:
        resp = httpx.get(
            "https://api.search.brave.com/res/v1/web/search",
            headers={
                "Accept": "application/json",
                "Accept-Encoding": "gzip",
                "X-Subscription-Token": BRAVE_API_KEY,
            },
            params={"q": search_query, "count": 20},
            timeout=15,
        )
        resp.raise_for_status()
        data = resp.json()
        
        web_results = data.get("web", {}).get("results", [])
        for r in web_results:
            link = r.get("url", "")
            if _is_valid_pdf_url(link) and link not in tried_urls and link not in urls:
                urls.append(link)
        
        if urls:
            print(f"  [Brave] Ditemukan {len(urls)} URL PDF baru")
        
        time.sleep(random.randint(1, 3))
        
    except Exception as e:
        err_msg = str(e).lower()
        if "429" in err_msg or "rate" in err_msg:
            print(f"  [Brave] Rate limit, menunggu 30s...")
            time.sleep(30)
        else:
            print(f"  [Brave Error] {e}")
    
    return urls


# ─── Tavily Search API helper ───
TAVILY_API_KEY = os.getenv("TAVILY_API_KEY", "")

def _search_tavily(query: str, tried_urls: set) -> list[str]:
    """Cari URL PDF via Tavily Search API (sync)."""
    if not TAVILY_API_KEY:
        print("  [Tavily] API key tidak ditemukan, skip Tavily Search")
        return []
    
    from tavily import TavilyClient
    import random
    
    urls = []
    search_query = query if "pdf" in query.lower() else query + " pdf"
    
    try:
        client = TavilyClient(api_key=TAVILY_API_KEY)
        results = client.search(
            query=search_query,
            max_results=10,
            include_domains=[],
            search_depth="basic",
        )
        
        for r in results.get("results", []):
            link = r.get("url", "")
            if _is_valid_pdf_url(link) and link not in tried_urls and link not in urls:
                urls.append(link)
        
        if urls:
            print(f"  [Tavily] Ditemukan {len(urls)} URL PDF baru")
        
        time.sleep(random.randint(1, 2))
        
    except Exception as e:
        err_msg = str(e).lower()
        if "429" in err_msg or "rate" in err_msg or "limit" in err_msg:
            print(f"  [Tavily] Rate limit, menunggu 30s...")
            time.sleep(30)
        else:
            print(f"  [Tavily Error] {e}")
    
    return urls


def _download_pdf_helper(url: str, dest: Path) -> bool:
    """Helper to download a PDF file with size limits."""
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
    }
    req = urllib.request.Request(url, headers=headers)
    MAX_FILE_SIZE = 5 * 1024 * 1024  # 5 MB
    try:
        with urllib.request.urlopen(req, timeout=12, context=SSL_CONTEXT) as r:
            cl = r.headers.get("Content-Length")
            if cl:
                try:
                    if int(cl) > MAX_FILE_SIZE:
                        return False
                except ValueError:
                    pass

            downloaded = 0
            with open(dest, "wb") as f:
                while True:
                    chunk = r.read(16384)
                    if not chunk:
                        break
                    downloaded += len(chunk)
                    if downloaded > MAX_FILE_SIZE:
                        f.close()
                        if dest.exists():
                            dest.unlink()
                        return False
                    f.write(chunk)
                    
        if dest.exists() and dest.stat().st_size >= 5000:
            # Validasi otomatis menggunakan PyMuPDF (fitz)
            try:
                import fitz
                doc = fitz.open(str(dest))
                num_pages = len(doc)
                
                # Rule 1: Maksimal 10 halaman
                if num_pages > 10:
                    print(f"  [AUTO-SKIP] {dest.name} diabaikan: Jumlah halaman {num_pages} > 10")
                    doc.close()
                    dest.unlink()
                    return False
                
                doc.close()
                return True
            except Exception as e:
                print(f"  [AUTO-SKIP] {dest.name} diabaikan: Gagal validasi PDF ({e})")
                if dest.exists():
                    dest.unlink()
                return False
                
        if dest.exists():
            dest.unlink()
    except Exception:
        if dest.exists():
            dest.unlink()
    return False


@agent.tool
def search_pdfs(ctx: RunContext[ScraperState], query: str) -> str:
    """Cari URL PDF di Brave, Tavily, dan DuckDuckGo berdasarkan query, lalu secara otomatis mengunduhnya secara lokal."""
    print(f"  [SEARCH] Query: '{query}'")
    
    import random
    pdf_urls = []

    # ─── 1. Brave Search API ───
    try:
        brave_urls = _search_brave(query, ctx.deps.tried_urls)
        pdf_urls.extend(brave_urls)
    except Exception as e:
        print(f"  [Brave Error] {e}")

    # ─── 2. DuckDuckGo Search (via ddgs library) ───
    try:
        ddg_urls = search_duckduckgo_api(query, ctx.deps.tried_urls | set(pdf_urls))
        for u in ddg_urls:
            if u not in pdf_urls:
                pdf_urls.append(u)
    except Exception as e:
        print(f"  [DDG Integration Error] {e}")

    # ─── 3. Tavily Search API ───
    try:
        tavily_urls = _search_tavily(query, ctx.deps.tried_urls | set(pdf_urls))
        for u in tavily_urls:
            if u not in pdf_urls:
                pdf_urls.append(u)
    except Exception as e:
        print(f"  [Tavily Integration Error] {e}")

    time.sleep(2)

    if not pdf_urls:
        return f"Tidak menemukan URL PDF baru untuk query: '{query}'"

    # Tambahkan ke tried_urls
    ctx.deps.tried_urls.update(pdf_urls)

    # Simpan tried_urls secara real-time ke file
    try:
        TRIED_URLS_FILE.parent.mkdir(parents=True, exist_ok=True)
        with TRIED_URLS_FILE.open("w", encoding="utf-8") as f:
            json.dump(list(ctx.deps.tried_urls), f, ensure_ascii=False, indent=2)
            f.flush()
            try:
                os.fsync(f.fileno())
            except OSError:
                pass
    except Exception as e:
        print(f"  [WARN] Gagal menyimpan tried_urls: {e}")

    print(f"  [SEARCH_RESULT] Menemukan {len(pdf_urls)} URL PDF baru. Mulai mengunduh batch otomatis (maks 5 file)...")
    
    # Buat folder temp jika belum ada
    PDF_DIR.mkdir(parents=True, exist_ok=True)
    
    downloaded_files = []
    for u in pdf_urls:
        if len(downloaded_files) >= 5:
            break
            
        h = hashlib.md5(u.encode('utf-8')).hexdigest()
        temp_dest = PDF_DIR / f"temp_{h}.pdf"
        
        print(f"  [DOWNLOAD] Mengunduh {u[:75]}...")
        if _download_pdf_helper(u, temp_dest):
            size_kb = temp_dest.stat().st_size // 1024
            print(f"  [OK] Berhasil mengunduh ({size_kb} KB) -> {temp_dest.name}")
            downloaded_files.append((u, temp_dest))
        else:
            print(f"  [FAILED] Gagal/melebihi batas ukuran.")

    if not downloaded_files:
        return f"Menemukan {len(pdf_urls)} URL baru, tetapi semuanya gagal diunduh atau melebihi batas 5MB."

    result = f"Berhasil menemukan dan mengunduh {len(downloaded_files)} PDF baru ke penyimpanan lokal:\n"
    for i, (u, p) in enumerate(downloaded_files):
        result += f"  [{i+1}] Path: {p.as_posix()} (URL: {u})\n"
    result += "\nSilakan gunakan tool `read_pdf_text` untuk mengevaluasi isi/teks dari path lokal di atas secara satu per satu, kemudian gunakan `accept_pdf` atau `reject_pdf` berdasarkan isi teks tersebut."
    
    return result


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
                "INFO: Ini adalah dokumen gambar/scan (misal: brosur paket wisata atau menu makanan bertipe gambar). "
                "Untuk dokumen visual seperti brosur/menu restoran Indonesia, ini adalah format yang valid dan sangat berguna untuk dataset multimodal. "
                "Jika nama file (atau asal URL) meyakinkan bahwa ini adalah brosur/menu restoran Indonesia yang valid, silakan gunakan `accept_pdf` untuk menerimanya."
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
            f.flush()
            try:
                os.fsync(f.fileno())
            except OSError:
                pass

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
    """Cek progress scraping saat ini: berapa yang sudah diterima dan target."""
    accepted = ctx.deps.accepted_count
    target = ctx.deps.target_docs
    remaining = target - accepted
    tried = len(ctx.deps.tried_urls)

    status = (
        f"=== STATUS SCRAPING ===\n"
        f"Diterima  : {accepted}/{target}\n"
        f"Sisa      : {remaining}\n"
        f"URL dicoba: {tried}\n"
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
        f"ALUR KERJA:\n"
        f"1. Mulailah dengan memanggil `check_existing_docs()` untuk melihat topik yang kurang.\n"
        f"2. Panggil `search_pdfs(query)` untuk mencari PDF baru. Tool ini otomatis mencari sekaligus MENGUNDUH dokumen PDF baru ke penyimpanan lokal.\n"
        f"3. DOKUMEN YANG DIUNDUH SUDAH PASTI: memiliki <= 10 halaman dan di bawah 5MB. Validasi batas halaman ini dilakukan otomatis secara internal oleh python.\n"
        f"4. Dari path file lokal yang dikembalikan oleh `search_pdfs`, gunakan `read_pdf_text(pdf_path)` untuk membaca isinya. Tugas utama Anda sebagai LLM hanyalah mengevaluasi apakah topik/konten dokumen tersebut layak, kasual, relevan dengan target, berbahasa Indonesia, dan menambah keberagaman. Catatan: Untuk brosur paket wisata atau menu makanan restoran, jika dokumen tidak memiliki teks terbaca (hanya scan/gambar) tetapi nama file atau URL meyakinkan bahwa dokumen tersebut valid, Anda diperbolehkan untuk MENERIMANYA.\n"
        f"5. Terima dokumen yang layak dengan `accept_pdf(pdf_path, topic, reason)` atau tolak dengan `reject_pdf(pdf_path, reason)`."
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
    
    # Hapus file temp sisa sesi sebelumnya agar tidak menumpuk
    cleanup_temp_files()

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

    # Load tried_urls untuk resume
    tried_urls = set()
    if TRIED_URLS_FILE.exists():
        try:
            with TRIED_URLS_FILE.open("r", encoding="utf-8") as f:
                tried_urls = set(json.load(f))
            print(f"[INFO] Resume: {len(tried_urls)} URL sudah pernah dicoba sebelumnya")
        except:
            pass

    state = ScraperState(
        target_docs=args.docs,
        pdf_idx=pdf_idx,
        accepted_count=accepted_count,
        tried_urls=tried_urls,
    )

    worker_model = create_model_instance(API_BASE_URL, API_KEYS, args.model)

    print(f"\n[INFO] Memulai agent scraping dari {accepted_count}/{args.docs}...")
    print(f"[INFO] PDF index mulai dari: {pdf_idx}\n")

    consecutive_no_progress = 0
    while state.accepted_count < args.docs:
        last_accepted = state.accepted_count
        print(f"\n=== ITERASI AGENT SCRAPING ({state.accepted_count}/{args.docs}) ===")
        try:
            with agent.override(model=worker_model):
                result = await agent.run(
                    "Cari dan kumpulkan dokumen PDF baru yang belum tercover untuk menambah keberagaman topik. "
                    "Gunakan tool check_existing_docs() terlebih dahulu untuk melihat topik apa saja yang kurang.",
                    deps=state,
                    usage_limits=UsageLimits(request_limit=20)
                )
            
            final = cast(ScraperResult, result.output)
            print(f"  [ITERASI SELESAI] Summary: {final.summary}")
            if final.is_done and state.accepted_count >= args.docs:
                print("  [INFO] Agen melaporkan scraping selesai.")
                break

            # Cek kemajuan
            if state.accepted_count > last_accepted:
                consecutive_no_progress = 0
            else:
                consecutive_no_progress += 1
                if consecutive_no_progress >= 5:
                    print("\n[INFO] 5 iterasi berturut-turut tidak mendapatkan dokumen baru. Menghentikan scraping untuk mencegah loop tanpa akhir.")
                    break

            await asyncio.sleep(2)

        except Exception as e:
            err = str(e)
            if "429" in err or "rate limit" in err.lower():
                print(f"  [!] Rate Limit terdeteksi. Menunggu 60 detik sebelum iterasi berikutnya...")
                await asyncio.sleep(60)
            elif "request_limit" in err.lower() or "limit of 20" in err:
                print(f"  [INFO] Iterasi mencapai batas request_limit (20). Melanjutkan ke iterasi berikutnya dengan context bersih...")
                # Jika mencapai request_limit, itu wajar. Kita reset consecutive_no_progress karena agen sedang aktif bekerja
                consecutive_no_progress = 0
            else:
                print(f"  [!] Error pada iterasi: {e}. Menunggu 15 detik sebelum mencoba lagi...")
                await asyncio.sleep(15)

    print(f"\n=== SELESAI ===")
    print(f"Total dokumen berhasil dikonversi: {state.accepted_count}/{args.docs}")


if __name__ == "__main__":
    asyncio.run(main())
