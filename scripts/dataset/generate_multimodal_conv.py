import asyncio
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
from pydantic import BaseModel, Field
from pydantic_ai import Agent, RunContext, BinaryContent

# ─── Load .env ───────────────────────────────────────────────────────────────
SCRIPT_DIR = Path(__file__).resolve().parent
ROOT_DIR = SCRIPT_DIR.parent.parent
IMAGE_DIR = ROOT_DIR / "data" / "multimodal" / "images" / "general"
METADATA_FILE = ROOT_DIR / "data" / "multimodal" / "metadata" / "random_metadata.json"
OUTPUT_FILE = ROOT_DIR / "data" / "multimodal" / "train_vision_conv.jsonl"

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
# Rate limit delay: OpenRouter tidak se-strict Google AI Studio, tapi jeda 2s aman
RATE_LIMIT_DELAY = float(os.environ.get("RATE_LIMIT_DELAY", "2.0"))  # detik
API_MODEL = os.environ.get("OPENROUTER_MODEL") or "google/gemma-4-31b-it:free"

# Support multiple API keys: OPENROUTER_API_KEY_1, OPENROUTER_API_KEY_2, dst.
# Fallback ke OPENROUTER_API_KEY
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

# Opsi jumlah turn yang valid (6, 8, atau 10 — masing-masing 3+3, 4+4, 5+5 pasang)
VALID_TURN_COUNTS = [6, 8, 10]

# ─── State Management ────────────────────────────────────────────────────────
@dataclass
class ConvState:
    image_name: str
    image_path: Path
    caption_id: str
    caption_en: str
    culture_loc: str
    target_turns: int  # 6, 8, atau 10
    turns: list[dict[str, str]] = field(default_factory=list)

# ─── Pydantic Output Schema ─────────────────────────────────────────────────
class ConversationResult(BaseModel):
    is_valid: bool = Field(description="Apakah percakapan ini sudah memenuhi semua kriteria?")
    rationale: str = Field(description="Alasan mengapa percakapan ini dianggap valid dan selesai.")

# ─── Rate Limited HTTP Client ───────────────────────────────────────────────
class RateLimitedAsyncClient(httpx.AsyncClient):
    def __init__(self, *args, rate_limit_delay: float = 3.0, **kwargs):
        super().__init__(*args, **kwargs)
        self.rate_limit_delay = rate_limit_delay
        self.last_request_time = 0.0
        self.request_lock = asyncio.Lock()

    async def send(self, request: httpx.Request, *args, **kwargs):
        async with self.request_lock:
            now = time.time()
            elapsed = now - self.last_request_time
            if elapsed < self.rate_limit_delay:
                wait_needed = self.rate_limit_delay - elapsed
                await asyncio.sleep(wait_needed)
            self.last_request_time = time.time()

        max_http_retries = 5
        for attempt in range(max_http_retries):
            try:
                request.read()
                response = await super().send(request, *args, **kwargs)
                
                if response.status_code == 429:
                    wait_time = (attempt + 1) * 30
                    print(f"  [!] HTTP 429 Rate Limit. Menunggu {wait_time} detik (Attempt {attempt+1}/{max_http_retries})...")
                    await asyncio.sleep(wait_time)
                    continue
                
                if response.status_code in [500, 502, 503, 504]:
                    wait_time = (attempt + 1) * 5
                    print(f"  [!] HTTP {response.status_code}. Menunggu {wait_time} detik...")
                    await asyncio.sleep(wait_time)
                    continue
                    
                return response
            except Exception as e:
                if attempt == max_http_retries - 1:
                    raise e
                wait_time = (attempt + 1) * 5
                print(f"  [!] HTTP Request error: {e}. Menunggu {wait_time} detik...")
                await asyncio.sleep(wait_time)

def create_model_instance(api_key: str, model_name: str):
    """Buat model instance OpenRouter."""
    from pydantic_ai.models.openrouter import OpenRouterModel
    from pydantic_ai.providers.openrouter import OpenRouterProvider

    provider = OpenRouterProvider(api_key=api_key)
    return OpenRouterModel(model_name, provider=provider)

# ─── Agent Definition ────────────────────────────────────────────────────────
# Placeholder model string — akan di-override saat runtime
model_string = f"openrouter:{API_MODEL}"

agent = Agent(
    model_string,
    deps_type=ConvState,
    output_type=ConversationResult,
    retries=5,
    system_prompt=(
        "Kamu adalah spesialis pembuat dataset percakapan multi-turn Bahasa Indonesia berbasis gambar/visual.\n"
        "Tugasmu adalah membangun percakapan secara bertahap menggunakan tool `append_turn_pair`.\n\n"
        "ATURAN SUPER KETAT:\n"
        "1. IDENTITAS BOT: Asisten AI di dalam percakapan bernama 'Gemma'. Selalu jawab dan posisikan AI sebagai Gemma.\n"
        "2. Percakapan HARUS dalam Bahasa Indonesia (semi-formal/casual, santai dan ramah).\n"
        "3. Percakapan harus natural dan mendalam tentang gambar yang diberikan. User bertanya tentang detail gambar, meminta penjelasan, kesimpulan, atau konteks budaya, dan Assistant menjawab dengan lengkap dan informatif.\n"
        "4. Di setiap turn assistant, kamu WAJIB menentukan array `task_prefixes` dari TaskType Enum yang relevan (MAKSIMAL 3 token per turn):\n"
        "   - SUMMARIZE: meringkas informasi dari gambar\n"
        "   - TRANSLATE: menerjemahkan istilah/nama\n"
        "   - NER: menyebutkan/mengekstrak entitas (nama tempat, makanan, budaya, tanggal, dsb.)\n"
        "   - QA: menjawab pertanyaan spesifik tentang gambar/konteks\n"
        "   - PARAPHRASE: menulis ulang/memformat penjelasan\n"
        "   - GENERAL_CHAT: obrolan santai, brainstorm, penjelasan teori\n"
        "   Jika memilih lebih dari 1 token, respons asisten HARUS mencakup gabungan semua tugas token tersebut.\n"
        "5. PENTING: `human_user_message` HARUS berisi ucapan PENGGUNA MANUSIA. `ai_assistant_message` HARUS berisi respons dari BOT AI. JANGAN PERNAH TERTUKAR!\n"
        "6. TURN TERAKHIR (assistant paling akhir) WAJIB diakhiri dengan pertanyaan lanjutan yang terkait dengan percakapan dan/atau gambar (langsung maupun tidak langsung).\n"
        "   Contoh pertanyaan akhir: 'Kamu pernah coba masakan ini?', 'Kalau ke daerah ini, mau coba yang mana dulu?', 'Menurut kamu, apa yang paling menarik dari gambar ini?'\n"
        "7. TAHAP EVALUASI & SELF-REVIEW (WAJIB):\n"
        "   Setelah target jumlah turn tercapai, JANGAN langsung mengembalikan hasil.\n"
        "   Lakukan evaluasi ulang terhadap seluruh percakapan menggunakan tool `get_conversation_status`:\n"
        "   - Pastikan semua turn asisten memiliki prefix token yang tepat dan maksimal 3 token per turn.\n"
        "   - Pastikan turn assistant terakhir mengandung pertanyaan (tanda tanya '?').\n"
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
    """Tambahkan 1 pasang pesan (user lalu assistant) ke akhir percakapan tentang gambar dengan Task Prefixes yang sesuai."""
    await asyncio.sleep(0.5)
    
    state = ctx.deps
    
    if not task_prefixes:
        return "GAGAL: Kamu harus memilih setidaknya 1 task_prefix dari Enum TaskType!"
        
    if len(task_prefixes) > 3:
        return f"GAGAL: Kamu memilih {len(task_prefixes)} prefix. Maksimal 3 prefix per turn!"
    
    # Validasi panjang minimal
    if len(human_user_message.strip()) < 10:
        return "GAGAL: Pesan user terlalu pendek! Buat pertanyaan yang lebih berbobot."
    
    if len(ai_assistant_message.strip()) < 30:
        return "GAGAL: Pesan assistant terlalu pendek! Buat jawaban yang lebih informatif dan panjang."
        
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
        return f"GAGAL: Index {turn_index} di luar batas (0 sampai {len(state.turns) - 1})."
        
    old_role = state.turns[turn_index]["role"]
    state.turns[turn_index]["content"] = new_content
    
    return f"BERHASIL: Pesan index {turn_index} ({old_role}) diperbarui."

@agent.tool
def get_conversation_status(ctx: RunContext[ConvState]) -> str:
    """Melihat ringkasan seluruh percakapan yang sudah dibuat sampai saat ini, termasuk validasi prefix dan pertanyaan akhir."""
    state = ctx.deps
    if not state.turns:
        return "Percakapan masih kosong."
        
    lines = []
    issues = []
    
    for i, t in enumerate(state.turns):
        preview = t['content'][:80].replace('\n', ' ')
        lines.append(f"[{i}] {t['role'].upper()}: {preview}...")
        
        # Cek prefix pada turn assistant
        if t['role'] == 'assistant':
            prefixes = re.findall(r"<unused\d+>", t['content'])
            if not prefixes:
                issues.append(f"⚠️ Turn [{i}] (assistant) TIDAK MEMILIKI prefix token <unusedX>!")
            elif len(prefixes) > 3:
                issues.append(f"⚠️ Turn [{i}] (assistant) memiliki {len(prefixes)} prefix (melebihi batas 3)!")
    
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

@agent.system_prompt
def add_image_context(ctx: RunContext[ConvState]) -> str:
    """Tambahkan konteks gambar dan metadata ke system prompt."""
    state = ctx.deps
    
    context = (
        f"KONTEKS GAMBAR YANG HARUS DIBAHAS:\n"
        f"- Nama File: {state.image_name}\n"
    )
    
    if state.caption_id:
        context += f"- Deskripsi Gambar (Bahasa Indonesia): {state.caption_id}\n"
    if state.caption_en:
        context += f"- Deskripsi Gambar (Bahasa Inggris): {state.caption_en}\n"
    if state.culture_loc:
        context += f"- Lokasi/Konteks Budaya: {state.culture_loc}\n"
    
    context += (
        f"\nGunakan informasi metadata di atas sebagai kebenaran mutlak (ground truth) untuk membahas gambar.\n"
        f"Gambar akan dikirimkan bersama prompt ini.\n\n"
        f"ATURAN JUMLAH TURN:\n"
        f"1. Percakapan HARUS mencapai TEPAT {state.target_turns} pesan total ({state.target_turns // 2} user + {state.target_turns // 2} assistant).\n"
        f"   Gunakan `append_turn_pair` berulang kali sampai tool merespons 'TARGET TERCAPAI'.\n"
        f"2. Turn pertama user harus menanyakan sesuatu tentang gambar secara natural.\n"
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
        print("[ERROR] OPENAGENTIC_API_KEY tidak ditemukan di .env atau variabel lingkungan.")
        sys.exit(1)

    parser = argparse.ArgumentParser(description="Generate conversational multimodal Indonesian dataset with agent pattern")
    parser.add_argument("--model", type=str, default=API_MODEL, help="Nama model LLM")
    parser.add_argument("--output", type=str, default=str(OUTPUT_FILE), help="File output JSONL")
    parser.add_argument("--limit", type=int, default=0, help="Batasi jumlah gambar (0 = semua)")
    parser.add_argument("--workers", type=int, default=1, help="Jumlah worker paralel (default: 1 untuk Google TPM limit)")
    args = parser.parse_args()

    num_workers = args.workers
    print(f"[INFO] Provider          : OpenRouter")
    print(f"[INFO] Menggunakan Model  : {args.model}")
    print(f"[INFO] Rate Limit Delay  : {RATE_LIMIT_DELAY}s per request")
    print(f"[INFO] Jumlah API Key     : {len(API_KEYS)}")
    print(f"[INFO] Jumlah Workers     : {num_workers}")

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Baca daftar gambar random
    if not IMAGE_DIR.exists():
        print(f"[ERROR] Folder gambar {IMAGE_DIR} tidak ditemukan.")
        sys.exit(1)

    random_images = sorted(
        list(IMAGE_DIR.glob("random_*.png")),
        key=lambda x: int(m.group(1)) if (m := re.search(r"random_(\d+)", x.name)) else 0
    )

    # Load metadata
    image_metadata = {}
    if METADATA_FILE.exists():
        try:
            with METADATA_FILE.open("r", encoding="utf-8") as f:
                image_metadata = json.load(f)
            print(f"[INFO] Berhasil memuat metadata untuk {len(image_metadata)} gambar")
        except Exception as e:
            print(f"[WARN] Gagal membaca metadata: {e}")

    print(f"=== Analisis Gambar ===")
    print(f"Gambar Random (random_*.png): {len(random_images)}")

    # Buat antrean item
    items_to_process = list(random_images)
    random.seed(42)
    random.shuffle(items_to_process)

    if args.limit > 0:
        items_to_process = items_to_process[:args.limit]
        print(f"[INFO] Limit aktif: hanya memproses {args.limit} gambar")

    # Resume check
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
                            processed_keys.add(Path(img_paths[0]).name)
                            total_existing += 1
                    except:
                        pass
        print(f"[INFO] Resume: {total_existing} percakapan sudah selesai")

    remaining_items = [item for item in items_to_process if item.name not in processed_keys]
    print(f"Sisa yang perlu diproses: {len(remaining_items)} dari {len(items_to_process)}")

    if not remaining_items:
        print("[INFO] Semua percakapan sudah selesai!")
        return

    # Distribusi turn count (seimbang 6/8/10)
    turn_dist_map: dict[str, int] = {}
    for i, item in enumerate(remaining_items):
        turn_dist_map[item.name] = VALID_TURN_COUNTS[i % len(VALID_TURN_COUNTS)]
    # Acak distribusi
    keys_list = list(turn_dist_map.keys())
    vals_list = list(turn_dist_map.values())
    random.shuffle(vals_list)
    turn_dist_map = dict(zip(keys_list, vals_list))

    print(f"\n=== Distribusi Turn Count ===")
    for tc in VALID_TURN_COUNTS:
        print(f"  {tc} turns: {list(turn_dist_map.values()).count(tc)} gambar")

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

        worker_model = create_model_instance(api_key, args.model)

        while True:
            # Ambil item berikutnya secara atomic
            async with image_lock:
                if image_index >= len(remaining_items):
                    break
                img_path = remaining_items[image_index]
                idx = image_index
                image_index += 1

            async with success_lock:
                current_id = total_existing + idx + 1

            target_turns = turn_dist_map[img_path.name]
            print(f"\n[Worker {worker_id} | {current_id}/{target_total}] {img_path.name} ({target_turns} turns)...")

            # Ambil metadata gambar
            meta = image_metadata.get(img_path.name, {})
            caption_id = meta.get("caption_native_lang") or meta.get("caption_id", "")
            caption_en = meta.get("caption") or meta.get("caption_en", "")
            culture_loc = meta.get("culture_relevant_loc", "")

            state = ConvState(
                image_name=img_path.name,
                image_path=img_path,
                caption_id=caption_id,
                caption_en=caption_en,
                culture_loc=culture_loc,
                target_turns=target_turns,
            )

            max_api_retries = 3
            success = False

            for attempt in range(max_api_retries):
                try:
                    await asyncio.sleep(RATE_LIMIT_DELAY)

                    image_data = img_path.read_bytes()

                    with agent.override(model=worker_model):
                        result = await agent.run(
                            [
                                "Silakan mulai membangun percakapan tentang gambar berikut. Ingat gunakan metadata untuk keakuratan dan Enum TaskType untuk prefix.",
                                BinaryContent(data=image_data, media_type="image/png"),
                            ],
                            deps=state
                        )

                    final_output = cast(ConversationResult, result.output)

                    if final_output.is_valid and len(state.turns) >= target_turns:
                        # Prepend 📷 ke turn pertama user
                        if state.turns and state.turns[0]["role"] == "user":
                            state.turns[0]["content"] = f"📷\n{state.turns[0]['content']}"

                        final_messages = [{"role": "system", "content": SYSTEM_PROMPT}] + state.turns

                        prefix_usage: dict[str, int] = {}
                        for t in state.turns:
                            if t["role"] == "assistant":
                                for p in re.findall(r"<unused\d+>", t["content"]):
                                    prefix_usage[p] = prefix_usage.get(p, 0) + 1

                        relative_path = img_path.relative_to(ROOT_DIR).as_posix()

                        async with write_lock:
                            async with success_lock:
                                success_count += 1
                                rec_id = 300000 + total_existing + success_count

                            record = {
                                "id": rec_id,
                                "images": [relative_path],
                                "num_turns": len(final_messages),
                                "prefix_usage": prefix_usage,
                                "rationale": final_output.rationale,
                                "messages": final_messages
                            }
                            # Tulis dengan aman: flush + fsync untuk menjamin data fisik tertulis ke disk
                            with output_path.open("a", encoding="utf-8") as f:
                                f.write(json.dumps(record, ensure_ascii=False) + "\n")
                                f.flush()
                                try:
                                    os.fsync(f.fileno())
                                except OSError:
                                    pass  # fsync mungkin tidak didukung di beberapa environment virtual/network mount

                        success = True
                        print(f"  ✓ [Worker {worker_id}] BERHASIL: {len(state.turns)} turns | prefix: {prefix_usage}")
                    else:
                        print(f"  ✗ [Worker {worker_id}] GAGAL LOGIKA: turns={len(state.turns)} | {final_output.rationale}")

                    break

                except (asyncio.CancelledError, KeyboardInterrupt):
                    # Jika dicancel atau Ctrl+C, hentikan secara anggun tanpa merusak data
                    print(f"  [!] [Worker {worker_id}] Proses dibatalkan oleh pengguna/sistem.")
                    raise
                except Exception as e:
                    err_msg = str(e)
                    if "429" in err_msg or "rate limit" in err_msg.lower():
                        wait_time = (attempt + 1) * 30
                        print(f"  [!] [Worker {worker_id}] Rate Limit. Menunggu {wait_time}s (Attempt {attempt+1}/{max_api_retries})...")
                        await asyncio.sleep(wait_time)
                    else:
                        print(f"  [!] [Worker {worker_id}] Error Attempt {attempt+1}/{max_api_retries}: {e}")
                        await asyncio.sleep(5)

                    state.turns = []

            if not success:
                print(f"  [!] [Worker {worker_id}] {img_path.name} gagal setelah {max_api_retries} percobaan. Dilewati.")

    # ─── Jalankan workers secara paralel ─────────────────────────────────────
    worker_tasks = []
    for w_id in range(num_workers):
        api_key = API_KEYS[w_id % len(API_KEYS)]
        worker_tasks.append(worker(w_id + 1, api_key))

    await asyncio.gather(*worker_tasks)

    elapsed = time.time() - start_time
    print(f"\n=== PROSES GENERASI SELESAI ===")
    print(f"Berhasil membuat {success_count} percakapan baru dalam {elapsed/60:.1f} menit.")

    # Distribusi final
    if output_path.exists():
        turn_counts: dict[int | str, int] = {6: 0, 8: 0, 10: 0, "other": 0}
        with output_path.open("r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    try:
                        record = json.loads(line)
                        n = record.get("num_turns", 0) - 1
                        if n in turn_counts:
                            turn_counts[n] += 1
                        else:
                            turn_counts["other"] += 1
                    except:
                        pass
        print(f"\n=== Distribusi Turn Count Final ===")
        for tc, count in sorted((k, v) for k, v in turn_counts.items() if isinstance(k, int)):
            print(f"  {tc} turns: {count} percakapan")
        if turn_counts["other"] > 0:
            print(f"  other: {turn_counts['other']} percakapan")

if __name__ == "__main__":
    asyncio.run(main())
