import asyncio
import json
import os
import sys
import random
import time
import httpx
from pathlib import Path
from dataclasses import dataclass, field
from typing import Any, cast

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
INPUT_FILE = DATA_DIR / "chat_train.jsonl"
OUTPUT_FILE = DATA_DIR / "orpo_train.jsonl"

TARGET_SAMPLES = int(os.environ.get("ORPO_TARGET_SAMPLES", "1000"))

# ─── Kategori Flaw (Cacat) ────────────────────────────────────────────────────
FLAWS = {
    "hallucination": "Berikan fakta yang salah, mengarang informasi medis/ilmiah/sejarah yang keliru namun terdengar meyakinkan.",
    "vague_and_short": "Berikan jawaban yang sangat singkat, basa-basi kosong, atau terkesan malas menjawab dan tidak solutif.",
    "off_topic": "Alihkan pembicaraan ke topik lain yang tidak ditanyakan oleh user, atau salah fokus ke detail yang tidak penting.",
    "ignore_instruction": "Abaikan instruksi spesifik dari user (misalnya jika user minta format tabel, berikan paragraf biasa; jika minta bahasa Inggris, berikan bahasa Indonesia).",
    "repetitive": "Ulangi kata-kata user atau ulangi poin yang sama berkali-kali tanpa memberikan nilai tambah.",
    "rude_or_condescending": "Gunakan nada bicara yang menggurui, merendahkan pemahaman user, atau terkesan tidak sabar (tapi tetap tanpa kata kasar).",
    "bad_list_formatting": "Gunakan format atau simbol yang aneh/tidak lazim saat membuat daftar/list (contohnya: '(1)', '[a]', '*)', '>', '+', atau campuran simbol acak). SENGAJA hindari format markdown standar ('- ' atau '1. '). Jika respon 'chosen' memuat daftar instruksi/barang, salin daftar tersebut namun rusak total format list-nya dengan simbol-simbol aneh ini."
}

# ─── State Management ────────────────────────────────────────────────────────
@dataclass
class OrpoState:
    inp: str
    chosen: str
    flaw_name: str
    flaw_desc: str

class OrpoGenerationResult(BaseModel):
    rejected_response: str = Field(description="Respon asisten yang sengaja dibuat lebih buruk (rejected) sesuai dengan instruksi flaw yang diberikan. Jangan tambahkan penjelasan di dalam respon ini.")
    rationale: str = Field(description="Alasan singkat (1-2 kalimat) mengapa respon rejected ini lebih buruk dari aslinya dan mencerminkan flaw tersebut.")

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
        # Selalu nonaktifkan verify SSL untuk openmodel jika ada masalah SSL
        kwargs["verify"] = False
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
    deps_type=OrpoState,
    output_type=OrpoGenerationResult,
    retries=3,
    system_prompt=(
        "Kamu adalah pakar AI yang bertugas membuat dataset Preference Tuning (ORPO/DPO).\\n"
        "Tugasmu adalah menghasilkan respon 'rejected' (respon yang buruk) berdasarkan riwayat percakapan dan respon 'chosen' (respon yang baik).\\n"
        "Saya akan memberikan:\\n"
        "1. Riwayat percakapan (INPUT)\\n"
        "2. Respon Asisten yang sangat baik (CHOSEN)\\n"
        "3. Tipe cacat/flaw yang harus kamu aplikasikan pada responmu.\\n\\n"
        "Hasilkan respon 'rejected' yang secara nyata menderita cacat tersebut. Buatlah tampak natural seolah-olah dihasilkan oleh model LLM yang kurang cerdas. "
        "Jangan menyisipkan metateks seperti 'Berikut adalah jawaban buruknya:' atau sejenisnya. Langsung berikan responnya seolah-olah kamu adalah asistennya."
    )
)

async def main():
    if not INPUT_FILE.exists():
        print(f"[ERROR] Input file {INPUT_FILE} not found. Pastikan merge_and_split_datasets.py sudah dijalankan.")
        return

    print(f"[INFO] Membaca SFT dataset dari {INPUT_FILE}...")
    sft_rows = []
    with INPUT_FILE.open("r", encoding="utf-8") as f:
        for idx, line in enumerate(f):
            if line.strip():
                try:
                    obj = json.loads(line)
                    obj["id"] = f"train_turn_{idx}"
                    sft_rows.append(obj)
                except Exception:
                    pass

    print(f"[INFO] Total {len(sft_rows)} turn SFT ditemukan.")
    
    # 2. Resume mechanism
    processed_ids = set()
    if OUTPUT_FILE.exists():
        with OUTPUT_FILE.open("r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    try:
                        obj = json.loads(line)
                        processed_ids.add(obj["id"])
                    except:
                        pass
        print(f"[INFO] Resume mode: mendeteksi {len(processed_ids)} data ORPO yang sudah diproses.")

    # 3. Filter data yang belum diproses
    pending_rows = [r for r in sft_rows if r["id"] not in processed_ids]
    
    # 4. Sampling
    random.seed(42)
    needed = max(0, TARGET_SAMPLES - len(processed_ids))
    if needed > 0 and needed < len(pending_rows):
        pending_rows = random.sample(pending_rows, needed)
    
    print(f"[INFO] Akan memproses {len(pending_rows)} turn untuk di-generate respon rejected-nya.")
    if not pending_rows:
        print("[INFO] Tidak ada data yang perlu diproses. Selesai.")
        return

    # Saring hanya topik yang belum dipakai (unused) => dalam hal ini pending rows
    unused_topics = pending_rows

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
    flaw_keys = list(FLAWS.keys())
    
    num_workers = len(API_KEYS)
    print(f"[INFO] Menjalankan {num_workers} parallel workers dengan arsitektur generate_prefix_tasks...")

    async def worker(worker_id: int, api_key: str):
        nonlocal produced_in_this_run, topic_index, successful_count
        
        worker_model = create_model_instance(model_string, API_BASE_URL, api_key, worker_id)
        
        while True:
            async with success_lock:
                if successful_count >= len(pending_rows):
                    break
            
            async with topic_lock:
                if topic_index >= len(unused_topics):
                    break
                entry = unused_topics[topic_index]
                topic_index += 1
                
            turn_id = entry["id"]
            inp = entry["input"]
            chosen = entry["target"]
            flaw_name = random.choice(flaw_keys)
            flaw_desc = FLAWS[flaw_name]
            
            async with success_lock:
                current_id = len(processed_ids) + successful_count
                
            print(f"\\n[Worker {worker_id} | Progress: {current_id + 1}/{len(processed_ids) + len(pending_rows)}] Generating rejected response for ID: {turn_id}")
            
            state = OrpoState(
                inp=inp,
                chosen=chosen,
                flaw_name=flaw_name,
                flaw_desc=flaw_desc
            )
            
            prompt = f"RIWAYAT PERCAKAPAN:\\n{inp}\\n\\nRESPON ASISTEN YANG BAIK (CHOSEN):\\n{chosen}\\n\\nINSTRUKSI UNTUK RESPON REJECTED:\\nBuatlah respon asisten alternatif yang menderita cacat: [{flaw_name}]\\nDeskripsi cacat: {flaw_desc}"
            
            max_api_retries = 3
            success = False
            
            for attempt in range(max_api_retries):
                try:
                    await asyncio.sleep(1)
                    
                    with agent.override(model=worker_model):
                        result = await agent.run(
                            prompt, 
                            deps=state
                        )
                    
                    generated = getattr(result, "data", None) or getattr(result, "output", None)
                    if not generated:
                        raise ValueError("Pydantic AI run returned no data.")
                        
                    final_output = {
                        "id": turn_id,
                        "prompt": inp,
                        "chosen": chosen,
                        "rejected": generated.rejected_response,
                        "flaw": flaw_name,
                        "rationale": generated.rationale
                    }
                    
                    async with write_lock:
                        batch_buffer.append(final_output)
                        successful_count += 1
                        
                        print(f"  ✓ [Worker {worker_id}] BERHASIL GENERATE: ID {turn_id} (Flaw: {flaw_name}). (Di buffer: {len(batch_buffer)}/50)")
                        
                        if len(batch_buffer) == 50:
                            OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
                            with OUTPUT_FILE.open("a", encoding="utf-8") as f:
                                for item in batch_buffer:
                                    f.write(json.dumps(item, ensure_ascii=False) + "\\n")
                            produced_in_this_run += 50
                            batch_buffer.clear()
                            print(f"\\n==================================================")
                            print(f"✓ BERHASIL MENULIS BATCH 50 ORPO DATA KE FILE!")
                            print(f"Total yang ditulis pada sesi ini: {produced_in_this_run}")
                            print(f"==================================================\\n")
                    
                    success = True
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
                print(f"  [!] [Worker {worker_id}] ID '{turn_id}' gagal dibuat setelah {max_api_retries} percobaan. Dilewati (skipping)...")

    # Run workers
    worker_tasks = []
    for w_id in range(num_workers):
        api_key = API_KEYS[w_id % len(API_KEYS)]
        worker_tasks.append(worker(w_id + 1, api_key))
        
    await asyncio.gather(*worker_tasks)

    # Jika script selesai tapi buffer belum genap 50
    if len(batch_buffer) > 0:
        OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
        with OUTPUT_FILE.open("a", encoding="utf-8") as f:
            for item in batch_buffer:
                f.write(json.dumps(item, ensure_ascii=False) + "\\n")
        print(f"\\n[INFO] Menulis sisa {len(batch_buffer)} data di buffer ke JSONL.")

    print("[INFO] Semua generasi ORPO dataset telah selesai!")

if __name__ == "__main__":
    asyncio.run(main())
