import asyncio
import io
import os
import sys
import json
import time
import argparse
import random
import re
from pathlib import Path
from dataclasses import dataclass, field
from typing import Any, cast
from enum import Enum

import httpx
from PIL import Image
from pydantic import BaseModel, Field
from pydantic_ai import Agent, RunContext, BinaryContent, ModelRetry
from pydantic_ai.models.openrouter import OpenRouterModel
from pydantic_ai.providers.openrouter import OpenRouterProvider

# ─── Load .env ───────────────────────────────────────────────────────────────
SCRIPT_DIR = Path(__file__).resolve().parent
ROOT_DIR = SCRIPT_DIR.parent.parent
IMAGE_DIR = ROOT_DIR / "data" / "multimodal" / "images"
RANDOM_METADATA_FILE = ROOT_DIR / "data" / "multimodal" / "metadata" / "random_metadata.json"
DOC_METADATA_FILE = ROOT_DIR / "data" / "multimodal" / "metadata" / "doc_metadata.json"
OUTPUT_FILE = ROOT_DIR / "data" / "multimodal" / "train_vision.jsonl"

try:
    from dotenv import load_dotenv
    load_dotenv(ROOT_DIR / ".env")
except ImportError:
    pass

SYSTEM_PROMPT = (
    "Kamu adalah Gemma, asisten AI cerdas berbahasa Indonesia yang dirancang untuk membantu pengguna dalam berbagai tugas pemrosesan bahasa (NLP), pemahaman visual, maupun percakapan sehari-hari. "
    "Berikan respons yang akurat, terstruktur, ramah, dan natural."
)

# ─── Enum TaskPrefix ─────────────────────────────────────────────────────────
class TaskType(str, Enum):
    SUMMARIZE = "<unused1>"
    TRANSLATE = "<unused2>"
    NER = "<unused3>"
    QA = "<unused4>"
    PARAPHRASE = "<unused5>"
    GENERAL_CHAT = "<unused6>"

# ─── Konfigurasi API (OpenRouter) ──────────────────────────────────────────────
RATE_LIMIT_DELAY = float(os.environ.get("RATE_LIMIT_DELAY", "2.0"))  # detik
API_MODEL = os.environ.get("OPENROUTER_MODEL") or "google/gemma-4-31b-it:free"

def _load_api_keys() -> list[str]:
    keys = []
    for i in range(1, 10):
        k = os.environ.get(f"OPENROUTER_API_KEY_{i}")
        if k:
            keys.append(k)
    if not keys:
        single = os.environ.get("OPENROUTER_API_KEY", "")
        if single:
            keys.append(single)
    return keys

API_KEYS = _load_api_keys()
API_KEY = API_KEYS[0] if API_KEYS else ""

# Set environment variable agar instansiasi Agent pydantic-ai di tingkat modul tidak error
if API_KEY:
    os.environ["OPENROUTER_API_KEY"] = API_KEY

# Opsi jumlah turn yang valid (6, 8, atau 10 — masing-masing 3+3, 4+4, 5+5 pasang)
VALID_TURN_COUNTS = [6, 8, 10]

# ─── Image Size Limit (OpenRouter max ~30MB, pakai 20MB sebagai safety margin) ──
MAX_TOTAL_IMAGE_MB = float(os.environ.get("MAX_TOTAL_IMAGE_MB", "20"))
MAX_TOTAL_IMAGE_BYTES = int(MAX_TOTAL_IMAGE_MB * 1024 * 1024)


def prepare_images(paths: list[Path], max_total_bytes: int = MAX_TOTAL_IMAGE_BYTES) -> list[bytes]:
    """Load & kompres gambar jika total ukuran melebihi batas OpenRouter (~30MB).

    Strategi kompresi bertingkat:
        1. Jika total <= max_total_bytes, kembalikan raw bytes apa adanya.
        2. Jika > limit, resize gambar secara proporsional dengan cascade max_dim:
           2048 → 1600 → 1200 → 1024 → 800
        3. Jika masih > limit setelah resize ke 800px, lakukan kuantisasi warna
           (128 colors palette) sebagai fallback terakhir.

    Returns:
        list[bytes]: Data gambar (original atau terkompresi) per halaman.
    """
    raw_data_list = [p.read_bytes() for p in paths]
    total_raw = sum(len(d) for d in raw_data_list)

    if total_raw <= max_total_bytes:
        return raw_data_list

    print(f"      [compress] Total {total_raw / 1024 / 1024:.1f}MB > "
          f"{max_total_bytes / 1024 / 1024:.0f}MB limit, compressing...")

    # Cascade resize: coba max_dim dari besar ke kecil
    for max_dim in (2048, 1600, 1200, 1024, 800):
        compressed: list[bytes] = []
        for raw in raw_data_list:
            img = Image.open(io.BytesIO(raw))
            w, h = img.size
            if max(w, h) > max_dim:
                ratio = max_dim / max(w, h)
                new_size = (int(w * ratio), int(h * ratio))
                img = img.resize(new_size, Image.Resampling.LANCZOS)
            buf = io.BytesIO()
            img.save(buf, format="PNG", optimize=True)
            compressed.append(buf.getvalue())

        total_compressed = sum(len(d) for d in compressed)
        if total_compressed <= max_total_bytes:
            print(f"      [compress] OK: {total_compressed / 1024 / 1024:.1f}MB "
                  f"(max_dim={max_dim})")
            return compressed
        else:
            print(f"      [compress] Still {total_compressed / 1024 / 1024:.1f}MB "
                  f"at max_dim={max_dim}, trying smaller...")

    # Fallback terakhir: resize ke 800px + kuantisasi warna 128 palette
    compressed = []
    for raw in raw_data_list:
        img = Image.open(io.BytesIO(raw))
        w, h = img.size
        if max(w, h) > 800:
            ratio = 800 / max(w, h)
            new_size = (int(w * ratio), int(h * ratio))
            img = img.resize(new_size, Image.Resampling.LANCZOS)
    # Kuantisasi: kurangi jumlah warna menjadi 128
        if img.mode in ("RGBA", "LA", "P"):
            img = img.convert("RGBA")
        else:
            img = img.convert("RGB")
        img = img.quantize(colors=128)
        buf = io.BytesIO()
        img.save(buf, format="PNG", optimize=True)
        compressed.append(buf.getvalue())

    total_compressed = sum(len(d) for d in compressed)
    print(f"      [compress] Final (quantized): {total_compressed / 1024 / 1024:.1f}MB")
    return compressed

def validate_conversation(turns: list[dict]) -> tuple[bool, list[str]]:
    """Validasi HARD (di level kode, bukan sekadar saran ke LLM) sebelum sebuah
    percakapan diizinkan untuk ditulis ke output file. Ini menutup celah di mana
    LLM generator melaporkan is_valid=True padahal masih ada prefix yang salah
    tempat (bocor ke user turn) atau assistant turn yang kehilangan prefix
    setelah proses edit_turn_content."""
    issues: list[str] = []
    for i, t in enumerate(turns):
        prefixes = re.findall(r"<unused\d+>", t["content"])
        if t["role"] == "user" and prefixes:
            issues.append(f"Turn [{i}] (user) mengandung task prefix yang tidak seharusnya ada: {prefixes}")
        if t["role"] == "assistant":
            if not prefixes:
                issues.append(f"Turn [{i}] (assistant) tidak memiliki task prefix sama sekali")
            elif len(prefixes) > 3:
                issues.append(f"Turn [{i}] (assistant) memiliki {len(prefixes)} prefix (melebihi batas 3)")
    return (len(issues) == 0), issues


# Stopword list Bahasa Indonesia (minimal) khusus untuk heuristik overlap kata di bawah.
# TIDAK dipakai untuk apapun selain menyaring kata umum yang akan mendominasi skor Jaccard
# kalau tidak dibuang (mis. "yang", "di", "itu"). Ini heuristik murni berbasis kata, BUKAN
# model tambahan apapun.
_ID_STOPWORDS = {
    "yang", "dan", "di", "ke", "dari", "untuk", "pada", "dengan", "ini", "itu",
    "adalah", "atau", "juga", "akan", "sudah", "belum", "tidak", "bisa", "ada",
    "saya", "aku", "kamu", "anda", "kita", "kami", "dia", "mereka", "nya",
    "apa", "apakah", "bagaimana", "kenapa", "mengapa", "kapan", "dimana", "siapa",
    "boleh", "mau", "ingin", "coba", "tolong", "gimana", "kalau", "jika", "seperti",
    "banget", "nih", "ya", "sih", "deh", "kok", "dong", "aja", "saja", "lagi",
    "lebih", "sangat", "sekali", "hal", "cara", "salah", "satu", "dua", "beberapa",
}


def _tokenize_id(text: str) -> set[str]:
    """Tokenisasi kasar: lowercase, buang <unusedX>/📷/tanda baca, buang stopword & kata <=2 huruf."""
    text = re.sub(r"<unused\d+>", " ", text)
    text = text.replace("📷", " ")
    words = re.findall(r"[a-z]+", text.lower())
    return {w for w in words if len(w) > 2 and w not in _ID_STOPWORDS}


def _jaccard(a: set[str], b: set[str]) -> float:
    if not a or not b:
        return 0.0
    return len(a & b) / len(a | b)


def detect_coherence_drift(turns: list[dict]) -> list[str]:
    """Heuristik (bukan LLM judge) untuk mendeteksi pola 'assistant menjawab pertanyaan
    turn SEBELUMNYA, bukan turn saat ini' — pola yang ditemukan di ID 300029/300050 dkk.
    Membandingkan overlap kata (Jaccard, minus stopword) antara jawaban assistant dengan
    (a) pertanyaan user pada turn yang sama, vs (b) pertanyaan user 2 turn sebelumnya.
    Kalau overlap ke (b) jauh lebih tinggi dari overlap ke (a), kemungkinan besar itu drift.

    Ini heuristik murah dan bisa false-positive pada obrolan santai yang memang tidak
    banyak berbagi kata kunci dengan pertanyaannya (mis. GENERAL_CHAT) — makanya threshold
    dibuat konservatif (score_prev harus signifikan DAN jauh lebih tinggi dari score_current)
    supaya tidak menghabiskan budget retry untuk kasus yang sebenarnya sah-sah saja."""
    issues: list[str] = []
    for i, t in enumerate(turns):
        if t["role"] != "assistant" or i < 3:
            continue
        current_user_idx = i - 1
        prev_user_idx = i - 3
        if turns[current_user_idx]["role"] != "user" or turns[prev_user_idx]["role"] != "user":
            continue

        assistant_tokens = _tokenize_id(t["content"])
        current_tokens = _tokenize_id(turns[current_user_idx]["content"])
        prev_tokens = _tokenize_id(turns[prev_user_idx]["content"])

        score_current = _jaccard(assistant_tokens, current_tokens)
        score_prev = _jaccard(assistant_tokens, prev_tokens)

        if score_prev >= 0.12 and score_prev > score_current * 1.8:
            issues.append(
                f"Turn [{i}] (assistant) kata kuncinya lebih mirip pertanyaan di turn [{prev_user_idx}] "
                f"(overlap={score_prev:.2f}) dibanding pertanyaan di turn [{current_user_idx}] yang seharusnya "
                f"dijawab (overlap={score_current:.2f}) — kemungkinan jawaban 'ketinggalan' 1 giliran."
            )
    return issues


# ─── State Management ────────────────────────────────────────────────────────
@dataclass
class ConvState:
    item_key: str              # E.g., 'doc_scraped_1' atau 'random_1.png'
    category: str              # 'document' atau 'random'
    image_paths: list[Path]    # List of image paths (1 for random, multiple for doc)
    caption_id: str
    caption_en: str
    culture_loc: str
    target_turns: int          # 6, 8, atau 10
    turns: list[dict[str, str]] = field(default_factory=list)

# ─── Pydantic Output Schema ─────────────────────────────────────────────────
class ConversationResult(BaseModel):
    is_valid: bool = Field(description="Apakah percakapan ini sudah memenuhi semua kriteria?")
    rationale: str = Field(description="Alasan mengapa percakapan ini dianggap valid dan selesai.")

# ─── Rate Limited HTTP Client dengan API Key Rotation & Worker Logging (Robust) ──
class RateLimitedAsyncClient(httpx.AsyncClient):
    def __init__(self, *args, api_keys: list[str] | None = None, worker_id: int = 1, rate_limit_delay: float = 2.0, **kwargs):
        super().__init__(*args, **kwargs)
        self.rate_limit_delay = rate_limit_delay
        self.last_request_time = 0.0
        self.request_lock = asyncio.Lock()
        self.api_keys = api_keys or []
        self.current_key_idx = 0
        self.worker_id = worker_id

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
                            print(f"  [!] [Worker {self.worker_id}] HTTP 429 Rate Limit. Merotasi API Key dari indeks {old_idx} ke {self.current_key_idx}...")
                            # Tunggu sebentar saja (2s) lalu langsung coba request baru dengan key baru!
                            await asyncio.sleep(2.0)
                            continue
                        else:
                            # Jika semua key dalam pool sudah dicoba dan tetap 429, reset counter dan lakukan sleep panjang
                            rotations_tried = 0
                            wait_time = (attempt + 1) * 30
                            print(f"  [!] [Worker {self.worker_id}] Semua API Key ({len(self.api_keys)} kunci) terkena Rate Limit. Menunggu {wait_time}s untuk cooldown...")
                            await asyncio.sleep(wait_time)
                            continue

                    wait_time = (attempt + 1) * 30
                    print(f"  [!] [Worker {self.worker_id}] HTTP 429 Rate Limit. Menunggu {wait_time}s untuk retry...")
                    await asyncio.sleep(wait_time)
                    continue
                if response.status_code in [500, 502, 503, 504]:
                    wait_time = (attempt + 1) * 10
                    print(f"  [!] [Worker {self.worker_id}] HTTP {response.status_code} Error. Menunggu {wait_time}s untuk retry...")
                    await asyncio.sleep(wait_time)
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
                            print(f"  [!] [Worker {self.worker_id}] Exception Rate Limit. Merotasi API Key dari indeks {old_idx} ke {self.current_key_idx}...")
                            await asyncio.sleep(2.0)
                            continue
                        else:
                            rotations_tried = 0
                            wait_time = (attempt + 1) * 30
                            print(f"  [!] [Worker {self.worker_id}] Semua API Key ({len(self.api_keys)} kunci) terkena Exception Rate Limit. Menunggu {wait_time}s...")
                            await asyncio.sleep(wait_time)
                            continue
                if attempt == 4:
                    raise e
                await asyncio.sleep((attempt + 1) * 5)

def create_model_instance(api_keys: list[str], model_name: str, worker_id: int) -> OpenRouterModel:
    """Buat model instance memakai OpenRouterModel + OpenRouterProvider resmi (bukan lagi
    OpenAIChatModel + OpenAIProvider generik). Model yang dipakai TETAP SAMA (tidak diganti),
    ini murni penggantian jalur wiring ke class yang memang didesain untuk OpenRouter —
    termasuk penanganan finish_reason non-standar OpenRouter yang tidak selalu ditangani
    dengan baik oleh parser Chat Completions generik."""
    default_key = api_keys[0] if api_keys else ""
    client = RateLimitedAsyncClient(api_keys=api_keys, worker_id=worker_id, rate_limit_delay=2.0, timeout=120.0)
    provider = OpenRouterProvider(api_key=default_key, http_client=client)
    return OpenRouterModel(model_name, provider=provider)

# ─── Agent Definition ────────────────────────────────────────────────────────
model_string = f"openrouter:{API_MODEL}"

agent = Agent(
    model_string,
    deps_type=ConvState,
    output_type=ConversationResult,
    # retries=8 (bukan 5): sekarang tool-tool (append_turn_pair, edit_turn_content) dan
    # output_validator benar-benar raise ModelRetry, jadi budget ini AKTIF dipakai untuk
    # setiap kali model perlu diminta memperbaiki output-nya. 5 dirasa terlalu ketat karena
    # 1 percakapan bisa butuh beberapa siklus append+koreksi.
    retries=8,
    instructions=(
        "Kamu adalah spesialis pembuat dataset percakapan multi-turn Bahasa Indonesia berbasis gambar/visual.\n"
        "Tugasmu adalah membangun percakapan secara bertahap menggunakan tool `append_turn_pair`.\n\n"
        "ATURAN SUPER KETAT:\n"
        "1. IDENTITAS BOT: Asisten AI di dalam percakapan bernama 'Gemma'. Selalu jawab dan posisikan AI sebagai Gemma.\n"
        "2. Percakapan HARUS dalam Bahasa Indonesia (semi-formal/casual, santai dan ramah).\n"
        "3. HARUS VISUALLY GROUNDED (SANGAT PENTING):\n"
        "   - Pertanyaan yang diajukan oleh user HARUS berupa pertanyaan spesifik yang HANYA BISA DIJAWAB dengan melihat gambar/dokumen yang disediakan (misalnya menanyakan harga menu tertentu yang tertera di gambar, isi teks di halaman tertentu, membandingkan elemen visual antar halaman, dsb.).\n"
        "   - JANGAN membuat pertanyaan umum/teoretis yang bisa dijawab lewat internet tanpa melihat gambar.\n"
        "   - Respons asisten harus merujuk langsung pada detail visual atau teks spesifik yang ada pada gambar/halaman dokumen tersebut sebagai bukti (grounding).\n"
        "4. Di setiap turn assistant, kamu WAJIB menentukan array `task_prefixes` dari TaskType Enum yang relevan (MAKSIMAL 3 token per turn):\n"
        "   - SUMMARIZE: meringkas informasi dari gambar/halaman dokumen\n"
        "   - TRANSLATE: menerjemahkan istilah/nama\n"
        "   - NER: menyebutkan/mengekstrak entitas (nama tempat, makanan, budaya, tanggal, dsb.)\n"
        "   - QA: menjawab pertanyaan spesifik tentang gambar/konteks/halaman tertentu\n"
        "   - PARAPHRASE: menulis ulang/memformat penjelasan\n"
        "   - GENERAL_CHAT: obrolan santai, brainstorm, penjelasan teori\n"
        "   Jika memilih lebih dari 1 token, respons asisten HARUS mencakup gabungan semua tugas token tersebut.\n"
        "5. PENTING: `human_user_message` HARUS berisi ucapan PENGGUNA MANUSIA. `ai_assistant_message` HARUS berisi respons dari BOT AI. JANGAN PERNAH TERTUKAR!\n"
        "6. JAWAB INSTAN & ATURAN PERTANYAAN LANJUTAN:\n"
        "   - Asisten harus LANGSUNG MENJAWAB semua pertanyaan user PADA TURN YANG SAMA secara lengkap dan akurat. JANGAN menjawab pertanyaan dari turn sebelumnya.\n"
        "   - JANGAN MENUNDA jawaban ke turn berikutnya atau malah bertanya balik 'Apakah kamu tertarik mengetahui X?' ketika user baru saja menanyakan X.\n"
        "   - Di akhir setiap turn asisten, WAJIB ditutup dengan pertanyaan baru yang memperluas topik pembicaraan secara logis, bukan menanyakan kesediaan user untuk mendengar jawaban yang sudah mereka tanyakan.\n"
        "7. CONTOH JAWABAN YANG BENAR (FEW-SHOT):\n"
        "   User: 📷\\nBerapa harga Es Teh Manis dan Kopi Susu di menu ini?\n"
        "   Gemma: <unused4> Di menu halaman 1, Es Teh Manis dihargai Rp 10.000 dan Kopi Susu dihargai Rp 15.000. Apakah kamu ingin tahu opsi makanan ringan seperti gorengan yang cocok menemani minumanmu?\n"
        "   User: Boleh, apa saja pilihannya dan berapa harganya?\n"
        "   Gemma: <unused4><unused3> Tersedia Pisang Goreng seharga Rp 12.000 dan Kentang Goreng seharga Rp 14.000. Apakah kamu ingin langsung memesan menu-menu tersebut?\n"
        "8. TAHAP EVALUASI & SELF-REVIEW (WAJIB):\n"
        "   Setelah target jumlah turn tercapai, JANGAN langsung mengembalikan hasil.\n"
        "   Lakukan evaluasi ulang terhadap seluruh percakapan menggunakan tool `get_conversation_status`:\n"
        "   - Pastikan semua turn asisten memiliki prefix token yang tepat dan maksimal 3 token per turn.\n"
        "   - Pastikan turn assistant terakhir mengandung pertanyaan (tanda tanya '?').\n"
        "   - Pastikan setiap turn asisten benar-benar menjawab pertanyaan user PADA TURN YANG SAMA, bukan pertanyaan di turn sebelumnya.\n"
        "   - Gunakan `edit_turn_content` untuk memperbaiki isi atau token jika ada yang salah.\n"
        "   - Setelah semuanya bersih dan berkualitas tinggi, barulah kembalikan hasil akhir via output model."
    )
)

@agent.tool
async def append_turn_pair(
    ctx: RunContext[ConvState], 
    human_user_message: str, 
    task_prefixes: list[TaskType], 
    ai_assistant_message: str
) -> str:
    """Tambahkan 1 pasang pesan (user lalu assistant) ke akhir percakapan dengan Task Prefixes yang sesuai."""
    await asyncio.sleep(0.5)
    
    state = ctx.deps
    
    if not task_prefixes:
        raise ModelRetry("Kamu harus memilih setidaknya 1 task_prefix dari Enum TaskType!")

    if len(task_prefixes) > 3:
        raise ModelRetry(f"Kamu memilih {len(task_prefixes)} prefix. Maksimal 3 prefix per turn!")

    if len(human_user_message.strip()) < 10:
        raise ModelRetry("Pesan user terlalu pendek! Buat pertanyaan yang lebih berbobot.")

    if len(ai_assistant_message.strip()) < 30:
        raise ModelRetry("Pesan assistant terlalu pendek! Buat jawaban yang lebih informatif dan panjang.")

    if re.search(r"<unused\d+>", human_user_message):
        raise ModelRetry(
            "`human_user_message` TIDAK BOLEH mengandung task prefix token <unusedX>! "
            "Prefix HANYA milik `ai_assistant_message`. Tulis ulang pesan user tanpa token apapun di depannya."
        )

    if re.search(r"<unused\d+>", ai_assistant_message):
        raise ModelRetry(
            "Jangan menulis token <unusedX> secara manual di dalam `ai_assistant_message`. "
            "Prefix akan ditambahkan otomatis oleh sistem berdasarkan `task_prefixes` yang kamu pilih."
        )

    prefix_str = "".join([t.value for t in task_prefixes])
    
    # Prepend prefix token ke pesan assistant
    final_assistant_message = f"{prefix_str} {ai_assistant_message}"
    
    state.turns.append({"role": "user", "content": human_user_message})
    state.turns.append({"role": "assistant", "content": final_assistant_message})
    
    total_turns = len(state.turns)
    sisa = state.target_turns - total_turns
    
    status = f"BERHASIL: 2 pesan ditambahkan dengan prefix {prefix_str}. Total sekarang: {total_turns} pesan.\n"
    if sisa > 0:
        status += f"MASIH KURANG {sisa} pesan lagi. Terus buat turn baru yang mengalir natural!"
        if sisa == 2:
            status += "\nINGAT: Turn berikutnya adalah TURN TERAKHIR. Jawaban assistant HARUS diakhiri dengan pertanyaan terkait gambar/percakapan!"
    else:
        status += f"TARGET TERCAPAI ({state.target_turns} pesan). Panggil `get_conversation_status` untuk review, lalu kembalikan hasil akhir."
        
    return status

@agent.tool
def edit_turn_content(ctx: RunContext[ConvState], turn_index: int, new_content: str) -> str:
    """Edit isi pesan pada indeks tertentu jika dirasa kurang pas (0-indexed). Jika mengedit pesan asisten, PASTIKAN menulis token <unusedX> di awalnya secara manual!"""
    state = ctx.deps
    if turn_index < 0 or turn_index >= len(state.turns):
        raise ModelRetry(f"Index {turn_index} di luar batas (0 sampai {len(state.turns) - 1}).")

    old_role = state.turns[turn_index]["role"]
    prefixes = re.findall(r"<unused\d+>", new_content)

    if old_role == "user" and prefixes:
        raise ModelRetry(
            f"Turn [{turn_index}] adalah pesan USER, TIDAK BOLEH mengandung task prefix <unusedX>! "
            "Hapus token tersebut dari new_content."
        )

    if old_role == "assistant" and not prefixes:
        raise ModelRetry(
            f"Turn [{turn_index}] adalah pesan ASSISTANT, WAJIB diawali minimal 1 task prefix <unusedX>! "
            "Tambahkan token yang sesuai (mis. <unused6> untuk general chat) di awal new_content."
        )

    if old_role == "assistant" and len(prefixes) > 3:
        raise ModelRetry(f"Turn [{turn_index}] memiliki {len(prefixes)} prefix, maksimal 3!")

    state.turns[turn_index]["content"] = new_content
    
    return f"BERHASIL: Pesan index {turn_index} ({old_role}) diperbarui."

@agent.tool
def get_conversation_status(ctx: RunContext[ConvState]) -> str:
    """Melihat ringkasan seluruh percakapan yang sudah dibuat sampai saat ini, termasuk validasi prefix dan pertanyaan akhir."""
    state = ctx.deps
    if not state.turns:
        return "Percakapan masih kosong."
        
    lines = []
    
    for i, t in enumerate(state.turns):
        preview = t['content'][:80].replace('\n', ' ')
        lines.append(f"[{i}] {t['role'].upper()}: {preview}...")

    # Reuse fungsi validasi yang sama persis dengan yang dipakai output_validator,
    # supaya model bisa mendeteksi & memperbaiki masalah SEBELUM kena ModelRetry.
    _, prefix_issues = validate_conversation(state.turns)
    drift_issues = detect_coherence_drift(state.turns)
    issues = [f"⚠️ {i}" for i in prefix_issues] + [f"⚠️ {i}" for i in drift_issues]

    # Cek turn terakhir
    if state.turns and state.turns[-1]['role'] == 'assistant':
        last_content = state.turns[-1]['content']
        if '?' not in last_content:
            issues.append("⚠️ Turn TERAKHIR (assistant) TIDAK mengandung pertanyaan (tanda tanya '?')! Harus diakhiri dengan pertanyaan!")
    
    summary = "\n".join(lines)
    summary += f"\n\nTotal: {len(state.turns)}/{state.target_turns} pesan."
    
    if issues:
        summary += "\n\n🔴 MASALAH YANG HARUS DIPERBAIKI:\n" + "\n".join(issues)
    else:
        summary += "\n\n✅ Semua validasi lolos. Percakapan siap dikembalikan."
    
    return summary


async def validate_final_output(ctx: RunContext[ConvState], data: ConversationResult) -> ConversationResult:
    """Hard gate resmi pydantic-ai: dijalankan setiap kali model mencoba mengembalikan
    ConversationResult sebagai hasil akhir. Kalau ada masalah, raise ModelRetry supaya
    pydantic-ai secara otomatis meminta model memperbaiki (dihitung terhadap budget
    `retries=8` di Agent), bukan cuma menerima klaim `is_valid` dari model begitu saja."""
    state = ctx.deps

    if len(state.turns) < state.target_turns:
        raise ModelRetry(
            f"Percakapan baru {len(state.turns)}/{state.target_turns} pesan. Panggil `append_turn_pair` "
            f"lagi sampai target tercapai sebelum mengembalikan hasil akhir."
        )

    is_clean, prefix_issues = validate_conversation(state.turns)
    if not is_clean:
        raise ModelRetry(
            "Masih ada masalah task-prefix yang harus diperbaiki via `edit_turn_content` sebelum hasil "
            "bisa diterima:\n" + "\n".join(prefix_issues)
        )

    drift_issues = detect_coherence_drift(state.turns)
    if drift_issues:
        raise ModelRetry(
            "Terdeteksi kemungkinan jawaban assistant tidak menjawab pertanyaan pada turn yang sama "
            "(kata kunci jawaban lebih mirip pertanyaan SEBELUMNYA). Cek dan perbaiki via "
            "`edit_turn_content` supaya tiap assistant turn benar-benar menjawab pertanyaan user "
            "di turn yang sama:\n" + "\n".join(drift_issues)
        )

    if not data.is_valid:
        raise ModelRetry(
            f"Kamu menandai percakapan ini is_valid=False dengan alasan: '{data.rationale}'. "
            f"Perbaiki dulu turn yang bermasalah via `edit_turn_content`/`append_turn_pair`, "
            f"baru kembalikan is_valid=True."
        )

    return data

# Cast to bypass Pyright overload matching issue with covariant OutputDataT
agent.output_validator(cast(Any, validate_final_output))

@agent.instructions
def add_image_context(ctx: RunContext[ConvState]) -> str:
    """Tambahkan konteks gambar/dokumen dan metadata ke system prompt."""
    state = ctx.deps
    
    if state.category == "document":
        context = (
            f"KONTEKS DOKUMEN YANG HARUS DIBAHAS:\n"
            f"- Nama Dokumen Asal: {state.item_key}.pdf\n"
            f"- Jumlah Halaman: {len(state.image_paths)}\n"
            f"Setiap gambar yang dikirim secara berurutan mewakili Halaman 1, Halaman 2, dst.\n"
        )
        if state.caption_id:
            context += f"- Deskripsi/Topik Dokumen: {state.caption_id}\n"
        if state.culture_loc:
            context += f"- Alasan Pemilihan Dokumen: {state.culture_loc}\n"
    else:
        context = (
            f"KONTEKS GAMBAR YANG HARUS DIBAHAS:\n"
            f"- Nama File: {state.item_key}\n"
        )
        if state.caption_id:
            context += f"- Deskripsi Gambar (Bahasa Indonesia): {state.caption_id}\n"
        if state.caption_en:
            context += f"- Deskripsi Gambar (Bahasa Inggris): {state.caption_en}\n"
        if state.culture_loc:
            context += f"- Lokasi/Konteks Budaya: {state.culture_loc}\n"
    
    context += (
        f"\nGunakan informasi metadata di atas sebagai kebenaran mutlak (ground truth) untuk membahas gambar/dokumen.\n"
        f"Gambar/halaman dokumen akan dikirimkan secara berurutan bersama prompt ini.\n\n"
        f"ATURAN JUMLAH TURN:\n"
        f"1. Percakapan HARUS mencapai TEPAT {state.target_turns} pesan total ({state.target_turns // 2} user + {state.target_turns // 2} assistant).\n"
        f"   Gunakan `append_turn_pair` berulang kali sampai tool merespons 'TARGET TERCAPAI'.\n"
        f"2. Turn pertama user harus menanyakan sesuatu tentang gambar/dokumen secara natural.\n"
        f"3. Turn TERAKHIR assistant WAJIB diakhiri dengan pertanyaan terkait gambar/percakapan.\n"
        f"4. ALUR VERIFIKASI MANDIRI:\n"
        f"   Setelah target tercapai, panggil `get_conversation_status` untuk review.\n"
        f"   Gunakan `edit_turn_content` untuk memperbaiki jika ada masalah.\n"
        f"   Baru kembalikan hasil akhir!"
    )
    
    return context

# ============================================================================
# MAIN
# ============================================================================
async def main():
    if not API_KEYS:
        print("[ERROR] OPENAGENTIC_API_KEY / OPENROUTER_API_KEY tidak ditemukan di .env atau variabel lingkungan.")
        sys.exit(1)

    parser = argparse.ArgumentParser(description="Generate conversational multimodal Indonesian dataset sequentially")
    parser.add_argument("--model", type=str, default=API_MODEL, help="Nama model LLM")
    parser.add_argument("--output", type=str, default=str(OUTPUT_FILE), help="File output JSONL")
    parser.add_argument("--limit", type=int, default=0, help="Batasi jumlah item (0 = semua)")
    parser.add_argument("--workers", type=int, default=1, help="Jumlah worker paralel (default: 1)")
    args = parser.parse_args()

    num_workers = args.workers
    print(f"[INFO] Provider          : OpenRouter")
    print(f"[INFO] Menggunakan Model  : {args.model}")
    print(f"[INFO] Rate Limit Delay  : {RATE_LIMIT_DELAY}s per request")
    print(f"[INFO] Jumlah API Key     : {len(API_KEYS)}")
    print(f"[INFO] Jumlah Workers     : {num_workers}")

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # 1. Kelompokkan Dokumen PDF
    doc_groups = {}
    doc_dir = IMAGE_DIR / "documents"
    if doc_dir.exists():
        for img_path in doc_dir.glob("doc_scraped_*_page_*.png"):
            m = re.match(r"doc_scraped_(\d+)_page_(\d+)\.png", img_path.name)
            if m:
                pdf_idx = int(m.group(1))
                page_num = int(m.group(2))
                doc_key = f"doc_scraped_{pdf_idx}"
                if doc_key not in doc_groups:
                    doc_groups[doc_key] = []
                doc_groups[doc_key].append((page_num, img_path))

    # Urutkan halaman per PDF
    sorted_doc_groups = {}
    for doc_key, pages in doc_groups.items():
        pages.sort(key=lambda x: x[0])
        sorted_doc_groups[doc_key] = [x[1] for x in pages]

    # 2. Ambil Gambar Random
    random_images = []
    general_dir = IMAGE_DIR / "general"
    if general_dir.exists():
        random_images = sorted(
            list(general_dir.glob("random_*.png")),
            key=lambda x: int(m.group(1)) if (m := re.search(r"random_(\d+)", x.name)) else 0
        )

    # 3. Load Metadata
    random_metadata = {}
    if RANDOM_METADATA_FILE.exists():
        try:
            with RANDOM_METADATA_FILE.open("r", encoding="utf-8") as f:
                random_metadata = json.load(f)
        except Exception as e:
            print(f"[WARN] Gagal membaca random_metadata: {e}")

    doc_metadata = {}
    if DOC_METADATA_FILE.exists():
        try:
            with DOC_METADATA_FILE.open("r", encoding="utf-8") as f:
                doc_metadata = json.load(f)
        except Exception as e:
            print(f"[WARN] Gagal membaca doc_metadata: {e}")

    # 4. Bangun antrean sekuensial penuh (Dokumen lalu Gambar Random)
    items_to_process = []
    
    # Tambahkan Dokumen PDF terurut
    for doc_key, pages in sorted(sorted_doc_groups.items(), key=lambda x: int(x[0].split("_")[-1])):
        items_to_process.append((doc_key, pages, "document"))
        
    # Tambahkan Gambar Random terurut
    for img_path in random_images:
        items_to_process.append((img_path.name, [img_path], "random"))

    print(f"\n=== Analisis Sumber Data ===")
    print(f"Total Dokumen PDF (doc_scraped_*): {len(sorted_doc_groups)}")
    print(f"Total Gambar Random (random_*.png): {len(random_images)}")
    print(f"Total Gabungan Entri             : {len(items_to_process)}")

    if args.limit > 0:
        items_to_process = items_to_process[:args.limit]
        print(f"[INFO] Limit aktif: hanya memproses {args.limit} item")

    # 5. Resume check dari file output
    processed_keys = set()
    total_existing = 0
    if output_path.exists():
        with output_path.open("r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    try:
                        record = json.loads(line)
                        img_paths = record.get("images", [])
                        if img_paths:
                            first_img_name = Path(img_paths[0]).name
                            if "doc_scraped" in first_img_name:
                                doc_key = first_img_name.split("_page_")[0]
                                processed_keys.add(doc_key)
                            else:
                                processed_keys.add(first_img_name)
                            total_existing += 1
                    except:
                        pass
        print(f"[INFO] Resume: {total_existing} percakapan sudah selesai di {output_path.name}")

    remaining_items = [item for item in items_to_process if item[0] not in processed_keys]
    print(f"Sisa yang perlu diproses: {len(remaining_items)} dari {len(items_to_process)}")

    if not remaining_items:
        print("[INFO] Semua percakapan sudah selesai!")
        return

    # 6. Distribusi turn count (6/8/10) untuk sisa item secara deterministik/seimbang
    turn_dist_map = {}
    for i, item in enumerate(remaining_items):
        turn_dist_map[item[0]] = VALID_TURN_COUNTS[i % len(VALID_TURN_COUNTS)]
    
    # Acak variasi distribusinya agar merata
    keys_list = list(turn_dist_map.keys())
    vals_list = list(turn_dist_map.values())
    random.seed(42)
    random.shuffle(vals_list)
    turn_dist_map = dict(zip(keys_list, vals_list))

    print(f"\n=== Distribusi Turn Count Sisa ===")
    for tc in VALID_TURN_COUNTS:
        print(f"  {tc} turns: {list(turn_dist_map.values()).count(tc)} item")

    # ─── Shared state untuk workers ──────────────────────────────────────────
    start_time = time.time()
    success_count = 0

    image_index = 0
    image_lock = asyncio.Lock()
    success_lock = asyncio.Lock()
    write_lock = asyncio.Lock()

    target_total = len(items_to_process)

    # ─── Worker function ──────────────────────────────────────────────────────
    async def worker(worker_id: int, api_key: str):
        nonlocal image_index, success_count

        worker_model = create_model_instance(API_KEYS, args.model, worker_id)

        while True:
            # Ambil item berikutnya secara atomic
            async with image_lock:
                if image_index >= len(remaining_items):
                    break
                item_key, paths, category = remaining_items[image_index]
                idx = image_index
                image_index += 1

            async with success_lock:
                current_id = total_existing + idx + 1

            target_turns = turn_dist_map[item_key]
            print(f"\n[Worker {worker_id} | {current_id}/{target_total}] {item_key} ({category}, {target_turns} turns)...")

            # Ambil metadata berdasarkan kategori
            caption_id = ""
            caption_en = ""
            culture_loc = ""
            if category == "random":
                meta = random_metadata.get(item_key, {})
                caption_id = meta.get("caption_native_lang") or meta.get("caption_id", "")
                caption_en = meta.get("caption") or meta.get("caption_en", "")
                culture_loc = meta.get("culture_relevant_loc") or meta.get("location", "")
            elif category == "document":
                meta = doc_metadata.get(item_key, {})
                caption_id = meta.get("topic", "")
                caption_en = ""
                culture_loc = meta.get("agent_reason", "")

            state = ConvState(
                item_key=item_key,
                category=category,
                image_paths=paths,
                caption_id=caption_id,
                caption_en=caption_en,
                culture_loc=culture_loc,
                target_turns=target_turns,
            )

            max_api_retries = 3
            success = False
            current_image_limit = MAX_TOTAL_IMAGE_BYTES

            for attempt in range(max_api_retries):
                try:
                    await asyncio.sleep(RATE_LIMIT_DELAY)

                    # Masukkan instruksi awal dan list binary data halaman (dengan kompresi otomatis)
                    image_bytes_list = prepare_images(paths, current_image_limit)
                    contents = [
                        f"Silakan mulai membangun percakapan tentang {category} berikut. Ingat gunakan metadata untuk keakuratan dan Enum TaskType untuk prefix."
                    ]
                    for img_bytes in image_bytes_list:
                        contents.append(BinaryContent(data=img_bytes, media_type="image/png"))

                    result = await agent.run(
                        contents,
                        deps=state,
                        model=worker_model
                    )

                    final_output = cast(ConversationResult, result.output)

                    # Di titik ini, `validate_final_output` (output_validator) SUDAH menjamin:
                    # target_turns tercapai, tidak ada masalah prefix, tidak ada indikasi
                    # coherence drift, dan is_valid=True — karena kalau salah satu gagal,
                    # pydantic-ai akan raise ModelRetry (di-loop otomatis) atau akhirnya
                    # UnexpectedModelBehavior (ditangkap oleh except Exception di bawah).
                    # validate_conversation() di sini murni safety net murah (defense in depth),
                    # BUKAN gate utama lagi.
                    is_clean, validation_issues = validate_conversation(state.turns)

                    if final_output.is_valid and len(state.turns) >= target_turns and is_clean:
                        # Prepend 📷 ke turn pertama user sebanyak jumlah gambar
                        if state.turns and state.turns[0]["role"] == "user":
                            camera_tokens = "📷" * len(paths)
                            state.turns[0]["content"] = f"{camera_tokens}\n{state.turns[0]['content']}"

                        final_messages = [{"role": "system", "content": SYSTEM_PROMPT}] + state.turns

                        prefix_usage: dict[str, int] = {}
                        for t in state.turns:
                            if t["role"] == "assistant":
                                for p_tok in re.findall(r"<unused\d+>", t["content"]):
                                    prefix_usage[p_tok] = prefix_usage.get(p_tok, 0) + 1

                        relative_paths = [p.relative_to(ROOT_DIR).as_posix() for p in paths]

                        async with write_lock:
                            async with success_lock:
                                success_count += 1
                                rec_id = 300000 + total_existing + success_count

                            record = {
                                "id": rec_id,
                                "images": relative_paths,
                                "num_turns": len(final_messages),
                                "prefix_usage": prefix_usage,
                                "rationale": final_output.rationale,
                                "messages": final_messages
                            }
                            # Tulis dengan aman
                            with output_path.open("a", encoding="utf-8") as f:
                                f.write(json.dumps(record, ensure_ascii=False) + "\n")
                                f.flush()
                                try:
                                    os.fsync(f.fileno())
                                except OSError:
                                    pass

                        success = True
                        print(f"  ✓ [Worker {worker_id}] BERHASIL: {len(state.turns)} turns | prefix: {prefix_usage}")
                    else:
                        if not is_clean:
                            print(f"  ✗ [Worker {worker_id}] GAGAL VALIDASI PREFIX: {validation_issues}")
                        else:
                            print(f"  ✗ [Worker {worker_id}] GAGAL LOGIKA: turns={len(state.turns)} | {final_output.rationale}")

                    break

                except (asyncio.CancelledError, KeyboardInterrupt):
                    print(f"  [!] [Worker {worker_id}] Proses dibatalkan oleh pengguna/sistem.")
                    raise
                except Exception as e:
                    err_msg = str(e)
                    if "429" in err_msg or "rate limit" in err_msg.lower():
                        wait_time = (attempt + 1) * 30
                        print(f"  [!] [Worker {worker_id}] Rate Limit. Menunggu {wait_time}s (Attempt {attempt+1}/{max_api_retries})...")
                        await asyncio.sleep(wait_time)
                    elif "413" in err_msg or "cannot exceed 30MB" in err_msg.lower():
                        # 413 = gambar terlalu besar. Kurangi limit kompresi dan retry.
                        old_limit_mb = current_image_limit / (1024 * 1024)
                        current_image_limit = max(int(current_image_limit * 0.5), 5 * 1024 * 1024)  # min 5MB
                        new_limit_mb = current_image_limit / (1024 * 1024)
                        print(f"  [!] [Worker {worker_id}] 413 Payload Too Large. "
                              f"Turunkan limit kompresi {old_limit_mb:.0f}MB → {new_limit_mb:.0f}MB "
                              f"(Attempt {attempt+1}/{max_api_retries})...")
                        await asyncio.sleep(3)
                    else:
                        print(f"  [!] [Worker {worker_id}] Error Attempt {attempt+1}/{max_api_retries}: {e}")
                        await asyncio.sleep(5)

                    state.turns = []

            if not success:
                print(f"  [!] [Worker {worker_id}] {item_key} gagal setelah {max_api_retries} percobaan. Dilewati.")

    # ─── Jalankan workers secara paralel ─────────────────────────────────────
    worker_tasks = []
    for w_id in range(num_workers):
        api_key = API_KEYS[w_id % len(API_KEYS)]
        worker_tasks.append(worker(w_id + 1, api_key))

    await asyncio.gather(*worker_tasks)

    elapsed = time.time() - start_time
    print(f"\n=== PROSES GENERASI SELESAI ===")
    print(f"Berhasil membuat {success_count} percakapan baru dalam {elapsed/60:.1f} menit.")

if __name__ == "__main__":
    asyncio.run(main())
