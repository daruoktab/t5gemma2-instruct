import asyncio
import json
import os
import sys
import random
import argparse
import re
import time
import httpx
from pathlib import Path
from dataclasses import dataclass, field
from typing import Any, cast
from enum import Enum

from pydantic import BaseModel, Field
from pydantic_ai import Agent, RunContext
from transformers import AutoTokenizer

# ─── Load .env ───────────────────────────────────────────────────────────────
SCRIPT_DIR = Path(__file__).resolve().parent
ROOT_DIR = SCRIPT_DIR.parent.parent
DATA_DIR = ROOT_DIR / "data"

try:
    from dotenv import load_dotenv
    load_dotenv(ROOT_DIR / ".env")
except ImportError:
    pass

# ─── Enum TaskPrefix ─────────────────────────────────────────────────────────
class TaskType(str, Enum):
    SUMMARIZE = "<unused1>"
    TRANSLATE = "<unused2>"
    NER = "<unused3>"
    QA = "<unused4>"
    PARAPHRASE = "<unused5>"
    GENERAL_CHAT = "<unused6>"

# ─── Konfigurasi API ─────────────────────────────────────────────────────────
API_BASE_URL = os.environ.get("API_BASE_URL", "https://api.openmodel.ai/v1")
API_MODEL    = os.environ.get("API_MODEL", "deepseek-chat")

# Ambil semua API key yang dikonfigurasi
API_KEYS = []
idx = 1
while True:
    k = os.environ.get(f"OPENMODEL_API_KEY_{idx}") or os.environ.get(f"API_KEY_{idx}")
    if k:
        API_KEYS.append(k.strip())
        idx += 1
    else:
        break

if not API_KEYS:
    raw_key = os.environ.get("OPENMODEL_API_KEY") or os.environ.get("API_KEY")
    if raw_key:
        if "," in raw_key:
            API_KEYS = [k.strip() for k in raw_key.split(",") if k.strip()]
        else:
            API_KEYS = [raw_key]

# ─── Paths ───────────────────────────────────────────────────────────────────
TOPICS_FILE         = DATA_DIR / "generated_topics_new_2500.json"
OUTPUT_FILE         = DATA_DIR / "generated_prefix_tasks_agentic.jsonl"

SYSTEM_PROMPT = (
    "Kamu adalah Gemma, asisten AI cerdas berbahasa Indonesia yang dirancang untuk membantu pengguna dalam berbagai tugas pemrosesan bahasa (NLP) maupun percakapan sehari-hari. "
    "Kamu ahli dalam merangkum teks, menerjemahkan, mengekstraksi informasi (NER), menjawab pertanyaan (QA), serta memparafrase kalimat. "
    "Selain tugas-tugas teknis tersebut, kamu juga sangat mumpuni dalam menjawab pertanyaan umum, berdiskusi, atau sekadar mengobrol santai secara general. "
    "Selalu berikan respons yang akurat, terstruktur, namun tetap mempertahankan gaya bahasa yang ramah dan natural."
)

# ─── State Management ────────────────────────────────────────────────────────
@dataclass
class ConvState:
    topik: str
    summary: str
    task_hint: str
    tokenizer: Any
    turns: list[dict[str, str]] = field(default_factory=list)
    min_turns: int = 10  # 5 pasang
    max_turns: int = 14
    min_tokens_user: int = 15
    min_tokens_assistant: int = 50

    def get_token_count(self, text: str) -> int:
        if not self.tokenizer:
            return len(text.split())
        return len(self.tokenizer.encode(text))

class ConversationResult(BaseModel):
    is_valid: bool = Field(description="Apakah percakapan ini sudah memenuhi kriteria panjang dan kualitas?")
    rationale: str = Field(description="Alasan mengapa percakapan ini dianggap valid dan selesai.")

# ─── Agent Definition ────────────────────────────────────────────────────────
provider = API_MODEL.split(":")[0] if ":" in API_MODEL else "openai-chat"
model_name_only = API_MODEL.split(":")[-1]
if "deepseek" in model_name_only.lower():
    provider = "anthropic"
model_string = f"{provider}:{model_name_only}"

# Fallback global env configuration
fallback_key = API_KEYS[0] if API_KEYS else ""

if provider in ["openai", "openai-chat"]:
    if API_BASE_URL:
        os.environ["OPENAI_BASE_URL"] = API_BASE_URL
    if fallback_key:
        os.environ["OPENAI_API_KEY"] = fallback_key
elif provider == "anthropic":
    if API_BASE_URL:
        base = API_BASE_URL
        if base.endswith("/v1"):
            base = base[:-3]
        elif base.endswith("/v1/"):
            base = base[:-4]
        os.environ["ANTHROPIC_BASE_URL"] = base
    if fallback_key:
        os.environ["ANTHROPIC_API_KEY"] = fallback_key

class RateLimitedAsyncClient(httpx.AsyncClient):
    def __init__(self, *args, rate_limit_delay: float = 6.0, worker_id: int = 0, **kwargs):
        super().__init__(*args, **kwargs)
        self.rate_limit_delay = rate_limit_delay
        self.worker_id = worker_id
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
                
                # Check for 429
                if response.status_code == 429:
                    wait_time = (attempt + 1) * 30
                    print(f"  [!] [Worker {self.worker_id}] HTTP 429 Rate Limit. Menunggu {wait_time} detik sebelum mencoba lagi (Attempt {attempt+1}/{max_http_retries})...")
                    await asyncio.sleep(wait_time)
                    continue
                
                # Check for other server errors
                if response.status_code in [500, 502, 503, 504]:
                    wait_time = (attempt + 1) * 5
                    print(f"  [!] [Worker {self.worker_id}] HTTP {response.status_code}. Menunggu {wait_time} detik sebelum mencoba lagi...")
                    await asyncio.sleep(wait_time)
                    continue
                    
                return response
            except Exception as e:
                if attempt == max_http_retries - 1:
                    raise e
                wait_time = (attempt + 1) * 5
                print(f"  [!] [Worker {self.worker_id}] HTTP Request error: {e}. Menunggu {wait_time} detik sebelum mencoba lagi...")
                await asyncio.sleep(wait_time)

def create_model_instance(model_string: str, base_url: str | None, api_key: str, worker_id: int):
    provider_name = model_string.split(":")[0] if ":" in model_string else "openai-chat"
    model_name_only = model_string.split(":")[-1]
    if "deepseek" in model_name_only.lower():
        provider_name = "anthropic"
        
    client = RateLimitedAsyncClient(rate_limit_delay=6.0, worker_id=worker_id, timeout=60.0)
        
    if provider_name in ["openai", "openai-chat"]:
        from pydantic_ai.models.openai import OpenAIChatModel
        from pydantic_ai.providers.openai import OpenAIProvider
        
        custom_provider = OpenAIProvider(
            base_url=base_url,
            api_key=api_key,
            http_client=client
        )
        return OpenAIChatModel(model_name_only, provider=custom_provider)
    elif provider_name == "anthropic":
        from pydantic_ai.models.anthropic import AnthropicModel
        from pydantic_ai.providers.anthropic import AnthropicProvider
        
        base = base_url
        if base:
            if base.endswith("/v1"):
                base = base[:-3]
            elif base.endswith("/v1/"):
                base = base[:-4]
                
        custom_provider = AnthropicProvider(
            base_url=base,
            api_key=api_key,
            http_client=client
        )
        return AnthropicModel(model_name_only, provider=custom_provider)
    else:
        from pydantic_ai import models
        return models.infer_model(model_string)

agent = Agent(
    model_string,
    deps_type=ConvState,
    output_type=ConversationResult,
    retries=5,
    system_prompt=(
        "Kamu adalah spesialis pembuat dataset percakapan multi-turn Bahasa Indonesia untuk tugas NLP campuran.\n"
        "Tugasmu adalah membangun percakapan secara bertahap menggunakan tool `append_turn_pair`.\n\n"
        "ATURAN SUPER KETAT:\n"
        "1. IDENTITAS BOT: Asisten AI di dalam percakapan bernama 'Gemma'. Selalu jawab dan posisikan AI sebagai Gemma.\n"
        "2. Di setiap turn assistant, kamu WAJIB menentukan array `task_prefixes` dari TaskType Enum yang relevan dengan pertanyaan user (MAKSIMAL 3 pilihan token per turn).\n"
        "   - PENTING: Jika kamu memilih lebih dari 1 token (misal 2 atau 3), respons asisten yang kamu buat HARUS secara nyata memenuhi dan mencakup gabungan tugas dari seluruh token yang kamu pilih tersebut.\n"
        "   - Contoh: Jika memilih [SUMMARIZE, TRANSLATE], respons asisten harus berupa rangkuman DAN diterjemahkan.\n"
        "   - Jika hanya ngobrol biasa, pilih [GENERAL_CHAT].\n"
        "3. PENTING: `human_user_message` HARUS berisi ucapan PENGGUNA MANUSIA. `ai_assistant_message` HARUS berisi respons dari BOT AI. JANGAN PERNAH TERTUKAR PERANNYA!\n"
        "4. Asisten menjawab seperti biasa (kamu cukup isi teksnya di `ai_assistant_message`, prefix token akan disisipkan otomatis oleh tool).\n"
        "5. TAHAP EVALUASI & SELF-REVIEW (WAJIB):\n"
        "   Setelah target jumlah turn tercapai (misal 30/40 turn), JANGAN langsung mengembalikan hasil.\n"
        "   Lakukan evaluasi ulang terhadap seluruh percakapan yang telah dibuat menggunakan tool `get_conversation_status`:\n"
        "   - Periksa apakah semua turn asisten memiliki prefix token `<unusedX>` yang tepat, relevan, dan maksimal 3 token per turn.\n"
        "   - DILARANG menumpuk lebih dari 3 token. Jika ada turn asisten yang memiliki lebih dari 3 token, ini adalah KESALAHAN logika. Segera edit turn tersebut menggunakan `edit_turn_content` untuk memangkasnya menjadi 1 atau maksimal 2-3 token yang paling relevan.\n"
        "   - Kamu bisa mengoreksi isi pesan maupun pilihan token prefix asisten dengan menggunakan tool `edit_turn_content` (ingat untuk menuliskan token `<unusedX>` di awal secara manual saat mengedit pesan asisten).\n"
        "   - Setelah semuanya dipastikan bersih, konsisten, dan berkualitas tinggi, barulah kembalikan hasil akhir via output model."
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
    # Rate limit handled globally by RateLimitedAsyncClient
    await asyncio.sleep(0.5)
    
    state = ctx.deps
    
    if not task_prefixes:
        return "GAGAL: Kamu harus memilih setidaknya 1 task_prefix dari Enum TaskType!"
        
    if len(task_prefixes) > 3:
        return f"GAGAL: Kamu memilih {len(task_prefixes)} prefix. Jumlah maksimal prefix yang diperbolehkan per turn adalah 3! Silakan pilih kembali dengan maksimal 3 prefix."
        
    prefix_str = "".join([t.value for t in task_prefixes])
    
    # Prepend the unused tokens
    final_assistant_message = f"{prefix_str}{ai_assistant_message}"
    
    u_tok = state.get_token_count(human_user_message)
    a_tok = state.get_token_count(final_assistant_message)
    
    if u_tok < state.min_tokens_user or a_tok < state.min_tokens_assistant:
        return f"GAGAL: Pesan terlalu pendek. User tok: {u_tok} (Min: {state.min_tokens_user}), Assistant tok: {a_tok} (Min: {state.min_tokens_assistant}). Buat argumen/jawaban yang lebih berbobot dan panjang!"
    
    state.turns.append({"role": "user", "content": human_user_message})
    state.turns.append({"role": "assistant", "content": final_assistant_message})
    
    total_turns = len(state.turns)
    sisa = state.min_turns - total_turns
    
    status = f"BERHASIL: 2 pesan ditambahkan dengan prefix {prefix_str}. Total sekarang: {total_turns} pesan.\n"
    if sisa > 0:
        status += f"MASIH KURANG {sisa} pesan lagi. Terus buat turn baru yang mengalir natural!"
    else:
        status += f"TARGET TERCAPAI. Kamu sudah mencapai batas minimal {state.min_turns} pesan. Boleh panggil final result."
        
    return status

@agent.tool
def edit_turn_content(ctx: RunContext[ConvState], turn_index: int, new_content: str) -> str:
    """Edit isi pesan pada indeks tertentu jika dirasa kurang pas (0-indexed). Perhatian: Jika mengedit pesan asisten, PASTIKAN menulis token <unusedX> di awalnya secara manual!"""
    state = ctx.deps
    if turn_index < 0 or turn_index >= len(state.turns):
        return f"GAGAL: Index {turn_index} di luar batas."
        
    old_role = state.turns[turn_index]["role"]
    state.turns[turn_index]["content"] = new_content
    
    return f"BERHASIL: Pesan index {turn_index} ({old_role}) diperbarui."

@agent.tool
def get_conversation_status(ctx: RunContext[ConvState]) -> str:
    """Melihat ringkasan seluruh percakapan yang sudah dibuat sampai saat ini."""
    state = ctx.deps
    if not state.turns:
        return "Percakapan masih kosong."
        
    lines = []
    for i, t in enumerate(state.turns):
        lines.append(f"[{i}] {t['role'].upper()}: {t['content'][:60]}...")
    
    summary = "\n".join(lines)
    summary += f"\n\nTotal: {len(state.turns)} pesan."
    return summary

@agent.system_prompt
def add_topic_context(ctx: RunContext[ConvState]) -> str:
    return (
        f"Konteks Topik Percakapan yang harus kamu buat:\n"
        f"- Topik: {ctx.deps.topik}\n"
        f"- Ringkasan: {ctx.deps.summary}\n"
        f"- Task Hint: {ctx.deps.task_hint}\n\n"
        f"ATURAN SUPER KETAT:\n"
        f"1. Percakapan INI HARUS mencapai TEPAT {ctx.deps.min_turns} pesan total (termasuk user dan assistant).\n"
        f"   Gunakan `append_turn_pair` berulang kali sampai tool tersebut merespons 'TARGET TERCAPAI'.\n"
        f"2. ALUR VERIFIKASI MANDIRI:\n"
        f"   Setelah target tercapai, panggil `get_conversation_status` untuk me-review kembali seluruh percakapan.\n"
        f"   Pastikan pilihan token prefix `<unusedX>` logis dan MAKSIMAL 3 token per turn (jangan sampai bertumpuk lebih dari 3!).\n"
        f"   Gunakan `edit_turn_content` untuk memperbaiki isi atau token jika ada yang salah sebelum mengembalikan hasil akhir!"
    )

async def main():
    if not API_KEYS:
        print("[ERROR] Set OPENMODEL_API_KEY atau OPENMODEL_API_KEY_1, OPENMODEL_API_KEY_2, dst.")
        sys.exit(1)

    parser = argparse.ArgumentParser()
    parser.add_argument("--target", type=int, default=50, help="Jumlah percakapan yang ingin dibuat (harus kelipatan 50)")
    parser.add_argument("--workers", type=int, default=len(API_KEYS), help="Jumlah worker paralel (default: jumlah API key)")
    args = parser.parse_args()

    # Pastikan target kelipatan 50
    if args.target % 50 != 0:
        print("[WARNING] Target bukan kelipatan 50. Membulatkan target ke atas ke kelipatan 50 terdekat.")
        args.target = ((args.target + 49) // 50) * 50
        print(f"[INFO] Target disesuaikan menjadi: {args.target}")

    num_workers = args.workers
    print(f"[INFO] Menggunakan {num_workers} worker paralel dengan {len(API_KEYS)} API key.")

    tokenizer = AutoTokenizer.from_pretrained("google/t5gemma-2-270m-270m", trust_remote_code=True)

    with TOPICS_FILE.open("r", encoding="utf-8") as f:
        all_topics = json.load(f)
        
    # Lacak topik yang sudah di-generate sebelumnya dari file JSONL
    used_topics = set()
    total_existing = 0
    if OUTPUT_FILE.exists():
        with OUTPUT_FILE.open("r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    try:
                        used_topics.add(json.loads(line).get("topik", ""))
                        total_existing += 1
                    except:
                        pass
    
    print(f"[INFO] Total topik di file: {len(all_topics)} | Sudah di-generate di JSONL: {total_existing}")
    
    # Saring hanya topik yang belum dipakai (unused topics)
    unused_topics = [t for t in all_topics if t.get("topik", "") not in used_topics]
    print(f"[INFO] Sisa topik yang belum dibuat percakapan (Unused): {len(unused_topics)}")

    if len(unused_topics) == 0:
        print("[INFO] Semua topik sudah selesai di-generate percakapannya.")
        return

    produced_in_this_run = 0
    batch_buffer = []
    
    # Acak daftar unused_topics agar bervariasi
    random.shuffle(unused_topics)
    
    # State control for workers
    topic_index = 0
    topic_lock = asyncio.Lock()
    
    successful_count = 0
    success_lock = asyncio.Lock()
    
    write_lock = asyncio.Lock()

    async def worker(worker_id: int, api_key: str):
        nonlocal produced_in_this_run, topic_index, successful_count
        
        worker_model = create_model_instance(model_string, API_BASE_URL, api_key, worker_id)
        
        while True:
            async with success_lock:
                if successful_count >= args.target:
                    break
            
            async with topic_lock:
                if topic_index >= len(unused_topics):
                    break
                entry = unused_topics[topic_index]
                topic_index += 1
                
            topik = entry.get("topik", "")
            
            async with success_lock:
                current_id = total_existing + successful_count
                
            print(f"\n[Worker {worker_id} | Progress: {current_id + 1}/{total_existing + args.target}] Generating percakapan untuk topik: '{topik}'")
            
            target_turns = random.choice([30, 32, 34, 36, 38, 40])
            
            state = ConvState(
                topik=topik,
                summary=entry.get("summary", ""),
                task_hint=entry.get("task", ""),
                tokenizer=tokenizer,
                min_turns=target_turns,
                max_turns=target_turns,
                min_tokens_user=15,
                min_tokens_assistant=50
            )
            
            max_api_retries = 3
            success = False
            
            for attempt in range(max_api_retries):
                try:
                    await asyncio.sleep(2)
                    
                    with agent.override(model=worker_model):
                        result = await agent.run(
                            "Silakan mulai membangun percakapan dari awal. Ingat untuk menggunakan Enum TaskType dengan benar.", 
                            deps=state
                        )
                    
                    final_output = cast(ConversationResult, result.output)
                    if final_output.is_valid and len(state.turns) >= state.min_turns:
                        final_conv = [{"role": "system", "content": SYSTEM_PROMPT}] + state.turns
                        total_tok = sum(state.get_token_count(t["content"]) for t in state.turns)
                        
                        user_tokens = [state.get_token_count(t["content"]) for t in state.turns if t["role"] == "user"]
                        asst_tokens = [state.get_token_count(t["content"]) for t in state.turns if t["role"] == "assistant"]
                        
                        prefix_usage = {}
                        for t in state.turns:
                            if t["role"] == "assistant":
                                prefixes = re.findall(r"<unused\d+>", t["content"])
                                for p in prefixes:
                                    prefix_usage[p] = prefix_usage.get(p, 0) + 1
                        
                        stats = {
                            "avg_turn_tokens": round(total_tok / len(state.turns) if state.turns else 0, 1),
                            "min_user_tokens": min(user_tokens) if user_tokens else 0,
                            "max_user_tokens": max(user_tokens) if user_tokens else 0,
                            "avg_user_tokens": round(sum(user_tokens) / len(user_tokens) if user_tokens else 0, 1),
                            "min_asst_tokens": min(asst_tokens) if asst_tokens else 0,
                            "max_asst_tokens": max(asst_tokens) if asst_tokens else 0,
                            "avg_asst_tokens": round(sum(asst_tokens) / len(asst_tokens) if asst_tokens else 0, 1),
                            "prefix_usage": prefix_usage
                        }
                        
                        async with write_lock:
                            actual_id = 90000 + total_existing + successful_count
                            entry_out = {
                                "id": actual_id,
                                "topic_id": entry.get("id", None),
                                "topik": topik,
                                "topik_summary": entry.get("summary", ""),
                                "task_hint": entry.get("task", ""),
                                "num_turns": len(final_conv),
                                "tokens": total_tok,
                                "stats": stats,
                                "rationale": final_output.rationale,
                                "conversations": final_conv
                            }
                            
                            batch_buffer.append(entry_out)
                            successful_count += 1
                            
                            print(f"  ✓ [Worker {worker_id}] BERHASIL GENERATE: {len(state.turns)} turns, {total_tok} tokens. (Di buffer: {len(batch_buffer)}/50)")
                            
                            if len(batch_buffer) == 50:
                                OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
                                with OUTPUT_FILE.open("a", encoding="utf-8") as f:
                                    for item in batch_buffer:
                                        f.write(json.dumps(item, ensure_ascii=False) + "\n")
                                produced_in_this_run += 50
                                batch_buffer.clear()
                                print(f"\n==================================================")
                                print(f"✓ BERHASIL MENULIS BATCH 50 PERCAKAPAN KE FILE!")
                                print(f"Total yang ditulis pada sesi ini: {produced_in_this_run}")
                                print(f"Total data di file output saat ini: {total_existing + produced_in_this_run}")
                                print(f"==================================================\n")
                        
                        success = True
                    else:
                        print(f"  ✗ [Worker {worker_id}] GAGAL LOGIKA AGEN: (Turns: {len(state.turns)}). Rationale: {final_output.rationale}")
                    
                    break
                    
                except Exception as e:
                    err_msg = str(e)
                    if "429" in err_msg or "rate limit" in err_msg.lower():
                        wait_time = (attempt + 1) * 30
                        print(f"  [!] [Worker {worker_id}] Terkena Rate Limit (429) pada Percobaan {attempt+1}/{max_api_retries}. Menunggu {wait_time} detik...")
                        await asyncio.sleep(wait_time)
                    else:
                        print(f"  [!] [Worker {worker_id}] ERROR API/Sistem pada Percobaan {attempt+1}/{max_api_retries}: {e}")
                        await asyncio.sleep(5)
            
            if not success:
                print(f"  [!] [Worker {worker_id}] Topik '{topik}' gagal dibuat setelah {max_api_retries} percobaan. Dilewati (skipping)...")

    # Run workers
    worker_tasks = []
    for w_id in range(num_workers):
        api_key = API_KEYS[w_id % len(API_KEYS)]
        worker_tasks.append(worker(w_id + 1, api_key))
        
    await asyncio.gather(*worker_tasks)

    # Jika script selesai tapi buffer belum genap 50
    if len(batch_buffer) > 0:
        print(f"\n[INFO] Sesi selesai. Terdapat {len(batch_buffer)} data di buffer yang BELUM ditulis ke JSONL.")
        print(f"[INFO] Karena aturan kelipatan 50, data sisa ini dibuang agar file output tetap konsisten kelipatan 50.")
        print(f"[INFO] Anda dapat menjalankan ulang script untuk meng-generate ulang dari topik-topik tersebut.")

if __name__ == "__main__":
    asyncio.run(main())
