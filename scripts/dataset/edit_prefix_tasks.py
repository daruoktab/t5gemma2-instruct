import asyncio
import json
import os
import re
import sys
from pathlib import Path
from dataclasses import dataclass, field
from typing import Any, cast

from pydantic import BaseModel, Field
from pydantic_ai import Agent, RunContext

# ─── Load .env ───────────────────────────────────────────────────────────────
SCRIPT_DIR = Path(__file__).resolve().parent
ROOT_DIR = SCRIPT_DIR.parent.parent
DATA_DIR = ROOT_DIR / "data"

try:
    from dotenv import load_dotenv
    load_dotenv(ROOT_DIR / ".env")
except ImportError:
    pass

# ─── Configuration ───────────────────────────────────────────────────────────
API_BASE_URL = os.environ.get("API_BASE_URL", "https://api.openmodel.ai/v1")
API_MODEL    = os.environ.get("API_MODEL", "deepseek-chat")

# Extract OpenModel API keys
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

if not API_KEYS:
    print("[ERROR] No OpenModel API keys found in environment or .env file.")
    sys.exit(1)

print(f"[INFO] Loaded {len(API_KEYS)} API keys for concurrent processing.")

# Paths
INPUT_FILE = DATA_DIR / "t5-gemma-2-chat-instruct-dataset.jsonl"
OUTPUT_FILE = DATA_DIR / "t5-gemma-2-chat-instruct-dataset-edited.jsonl"
TOKENIZER_DIR = DATA_DIR / "tokenizernya-t5gemma2"

# ─── Pydantic Schemas ────────────────────────────────────────────────────────
class TurnEdit(BaseModel):
    turn_index: int = Field(description="Indeks turn asisten di dalam array 'conversations' (0-indexed).")
    task_prefixes: list[str] = Field(description="Daftar token prefix. Pilih dari: '<unused1>' (summarize), '<unused2>' (translate), '<unused3>' (ner), '<unused4>' (qa), '<unused5>' (paraphrase), '<unused6>' (general_chat). Maksimal 3 prefix.")
    edited_content: str | None = Field(default=None, description="Konten asisten yang baru jika ingin diedit/diperbaiki karena salah/kurang pas. Kosongkan jika teks asli sudah baik.")

class ConversationEditResult(BaseModel):
    is_valid: bool = Field(description="Apakah seluruh turn asisten telah sukses diklasifikasikan/diedit?")
    rationale: str = Field(description="Penjelasan singkat mengapa seluruh pilihan prefix di turn asisten tersebut tepat.")

# ─── State Management ────────────────────────────────────────────────────────
@dataclass
class EditState:
    conversation_id: int
    topik: str
    original_turns: list[dict[str, str]]
    edited_turns: list[dict[str, str]]
    has_edited: set[int] = field(default_factory=set)

# ─── Rate Limited HTTP Client ───────────────────────────────────────────────
import httpx

class CustomHTTPClient(httpx.AsyncClient):
    def __init__(self, rate_limit_delay: float = 6.0, worker_id: int = 1, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.rate_limit_delay = rate_limit_delay
        self.worker_id = worker_id

    async def send(self, request, *args, **kwargs):
        await asyncio.sleep(self.rate_limit_delay)
        max_http_retries = 3
        for attempt in range(max_http_retries):
            try:
                response = await super().send(request, *args, **kwargs)
                if response.status_code == 429:
                    wait_time = (attempt + 1) * 30
                    print(f"  [!] [Worker {self.worker_id}] Rate Limit (429). Menunggu {wait_time} detik...")
                    await asyncio.sleep(wait_time)
                    continue
                if response.status_code in [500, 502, 503, 504]:
                    wait_time = (attempt + 1) * 5
                    print(f"  [!] [Worker {self.worker_id}] HTTP {response.status_code}. Menunggu {wait_time} detik...")
                    await asyncio.sleep(wait_time)
                    continue
                return response
            except Exception as e:
                if attempt == max_http_retries - 1:
                    raise e
                wait_time = (attempt + 1) * 5
                print(f"  [!] [Worker {self.worker_id}] Request error: {e}. Menunggu {wait_time} detik...")
                await asyncio.sleep(wait_time)

def create_model_instance(model_string: str, base_url: str | None, api_key: str, worker_id: int):
    provider_name = model_string.split(":")[0] if ":" in model_string else "openai-chat"
    model_name_only = model_string.split(":")[-1]
    if "deepseek" in model_name_only.lower():
        provider_name = "anthropic"

    client = CustomHTTPClient(rate_limit_delay=4.0, worker_id=worker_id, timeout=60.0)

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

# ─── Agent Setup ─────────────────────────────────────────────────────────────
provider = API_MODEL.split(":")[0] if ":" in API_MODEL else "openai-chat"
model_name_only = API_MODEL.split(":")[-1]
if "deepseek" in model_name_only.lower():
    provider = "anthropic"
model_string = f"{provider}:{model_name_only}"

agent = Agent(
    model_string,
    deps_type=EditState,
    output_type=ConversationEditResult,
    retries=5,
    system_prompt=(
        "Kamu adalah spesialis kurator dan penyunting dataset percakapan Bahasa Indonesia.\n"
        "Tugasmu adalah menganalisis percakapan multi-turn yang diberikan dan menyisipkan token prefix (<unusedX>) pada setiap turn assistant.\n\n"
        "Tugas Utama:\n"
        "1. Analisis seluruh pesan di dalam percakapan secara runtut.\n"
        "2. Identifikasi turn milik 'assistant' (asisten AI). Tentukan token prefix apa saja yang relevan (maksimal 3):\n"
        "   - <unused1>: SUMMARIZE (meringkas teks)\n"
        "   - <unused2>: TRANSLATE (menerjemahkan bahasa)\n"
        "   - <unused3>: NER (ekstraksi entitas seperti nama, tanggal, list barang/niche, key-value, dsb.)\n"
        "   - <unused4>: QA (menjawab pertanyaan berdasarkan dokumen atau FAQ terstruktur)\n"
        "   - <unused5>: PARAPHRASE (memparafrase kalimat, menulis ulang/rewrite, memformat teks)\n"
        "   - <unused6>: GENERAL_CHAT (obrolan santai, curhat, brainstorm ide, penjelasan teori, dsb.)\n"
        "3. Kamu WAJIB memanggil tool `apply_edits` untuk menyimpan prefix hasil klasifikasi di seluruh turn asisten di percakapan ini secara sekaligus.\n"
        "4. Jika ada turn asisten yang kontennya salah, kurang natural, atau tidak menjawab instruksi user dengan baik, kamu diperbolehkan untuk merevisinya dengan mengisi parameter `edited_content`. Jika teks aslinya sudah baik, biarkan `edited_content` bernilai null.\n"
        "5. Setelah memanggil tool `apply_edits` dan mendapat konfirmasi sukses, kembalikan hasil edit akhir via output model."
    )
)

@agent.tool
async def apply_edits(
    ctx: RunContext[EditState],
    edits: list[TurnEdit]
) -> str:
    """Terapkan semua prefix token dan revisi konten asisten di dalam percakapan ini secara sekaligus."""
    state = ctx.deps
    
    # Validasi dan terapkan
    for edit in edits:
        idx = edit.turn_index
        if idx < 0 or idx >= len(state.original_turns):
            return f"GAGAL: Indeks turn {idx} berada di luar jangkauan percakapan!"
        
        turn = state.original_turns[idx]
        if turn.get("role") != "assistant":
            return f"GAGAL: Turn indeks {idx} adalah {turn.get('role')}, bukan assistant!"
            
        if not edit.task_prefixes:
            return f"GAGAL: Kamu harus memilih minimal 1 prefix token untuk turn {idx}!"
            
        if len(edit.task_prefixes) > 3:
            return f"GAGAL: Maksimal 3 prefix token per turn! Turn {idx} memiliki {len(edit.task_prefixes)} prefix."

        # Bersihkan konten dari prefix lama jika ada, lalu susun konten baru
        content = edit.edited_content if edit.edited_content else turn.get("content", "")
        content_clean = re.sub(r"^<unused\d+>", "", content.strip()).strip()
        prefix_str = "".join(edit.task_prefixes)
        
        state.edited_turns[idx] = {
            "role": "assistant",
            "content": f"{prefix_str} {content_clean}"
        }
        state.has_edited.add(idx)

    # Pastikan semua turn assistant sudah disentuh
    missing = []
    for i, t in enumerate(state.original_turns):
        if t.get("role") == "assistant" and i not in state.has_edited:
            missing.append(i)
            
    if missing:
        return f"GAGAL: Kamu melewatkan turn assistant dengan indeks: {missing}. Kamu harus menyunting SEMUA turn assistant!"
        
    return "SUKSES: Semua suntingan turn asisten berhasil diterapkan!"

# ─── Worker Setup ─────────────────────────────────────────────────────────────
async def main():
    if not INPUT_FILE.exists():
        print(f"[ERROR] Input file {INPUT_FILE} not found.")
        return

    # Check local tokenizer to verify
    if not TOKENIZER_DIR.exists():
        print(f"[ERROR] Local tokenizer folder {TOKENIZER_DIR} not found.")
        return

    # Resume mechanism
    processed_ids = set()
    total_existing = 0
    if OUTPUT_FILE.exists():
        with OUTPUT_FILE.open("r", encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                try:
                    obj = json.loads(line)
                    processed_ids.add(obj["id"])
                    total_existing += 1
                except:
                    pass
        print(f"[INFO] Resume mode: detected {total_existing} already processed conversations.")

    # Read original conversations
    all_conversations = []
    with INPUT_FILE.open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            try:
                obj = json.loads(line)
                if obj["id"] not in processed_ids:
                    all_conversations.append(obj)
            except Exception as e:
                print(f"[WARN] Failed to parse input line: {e}")

    total_to_process = len(all_conversations)
    print(f"[INFO] Remaining conversations to process: {total_to_process}")

    if total_to_process == 0:
        print("[INFO] All conversations have already been processed.")
        return

    # Queues & locks
    conversation_queue = asyncio.Queue()
    for item in all_conversations:
        await conversation_queue.put(item)

    write_lock = asyncio.Lock()
    progress_counter = total_existing
    progress_lock = asyncio.Lock()

    async def worker(worker_id: int, api_key: str):
        nonlocal progress_counter
        worker_model = create_model_instance(model_string, API_BASE_URL, api_key, worker_id)
        
        while not conversation_queue.empty():
            try:
                item = await conversation_queue.get()
            except asyncio.QueueEmpty:
                break
                
            conv_id = item["id"]
            topik = item.get("topik", "")
            original_conv = item.get("conversations", [])
            
            # Prepare state
            # Copy all turns as original, and edited turns initialized with the same
            edited_conv = [dict(t) for t in original_conv]
            state = EditState(
                conversation_id=conv_id,
                topik=topik,
                original_turns=original_conv,
                edited_turns=edited_conv
            )

            # Build prompt
            prompt_lines = [
                f"Percakapan ID: {conv_id}",
                f"Topik: {topik}",
                "Berikut adalah daftar turn dalam percakapan (gunakan indeks untuk merujuk saat edit):"
            ]
            for i, turn in enumerate(original_conv):
                prompt_lines.append(f"Indeks {i} - [{turn['role']}]: {turn['content']}")
            
            prompt_lines.append(
                "\nSilakan analisis seluruh percakapan di atas. Panggil tool `apply_edits` untuk menetapkan prefix token "
                "(<unused1> sampai <unused6>) di awal pesan assistant di setiap turn assistant. "
                "Jangan lupa untuk mengoreksi teks di parameter `edited_content` jika asisten memberikan jawaban yang salah/kurang pas."
            )
            prompt = "\n".join(prompt_lines)

            success = False
            max_api_retries = 3
            
            for attempt in range(max_api_retries):
                try:
                    async with progress_lock:
                        current_num = progress_counter + 1
                    print(f"[Worker {worker_id} | Progress: {current_num}/2500] Editing conversation ID: {conv_id} ('{topik}')")
                    
                    with agent.override(model=worker_model):
                        result = await agent.run(prompt, deps=state)
                        
                    final_res = cast(ConversationEditResult, result.output)
                    if final_res.is_valid:
                        # Write to file
                        async with write_lock:
                            OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
                            entry_out = {
                                "id": conv_id,
                                "topik": topik,
                                "num_turns": len(state.edited_turns),
                                "tokens": 0,  # Will be calculated during tokenization
                                "rationale": final_res.rationale,
                                "conversations": state.edited_turns
                            }
                            with OUTPUT_FILE.open("a", encoding="utf-8") as out_f:
                                out_f.write(json.dumps(entry_out, ensure_ascii=False) + "\n")
                                
                        async with progress_lock:
                            progress_counter += 1
                        print(f"  ✓ [Worker {worker_id}] SUCCESS for ID {conv_id}. Rationale: {final_res.rationale}")
                        success = True
                    else:
                        print(f"  ✗ [Worker {worker_id}] INVALID logis for ID {conv_id}: {final_res.rationale}")
                    break
                except Exception as e:
                    print(f"  [!] [Worker {worker_id}] Attempt {attempt+1}/{max_api_retries} failed for ID {conv_id}: {e}")
                    await asyncio.sleep(5)
            
            if not success:
                # Put back in queue to retry later or mark as failed
                print(f"  [!] [Worker {worker_id}] ID {conv_id} failed after {max_api_retries} attempts.")
                # We do not block the pipeline, we just skip it or log it
            
            conversation_queue.task_done()

    # Launch workers in parallel
    num_workers = min(len(API_KEYS), 6) # Up to 6 workers or keys count
    print(f"[INFO] Starting {num_workers} parallel workers...")
    worker_tasks = []
    for w_id in range(num_workers):
        api_key = API_KEYS[w_id % len(API_KEYS)]
        worker_tasks.append(worker(w_id + 1, api_key))
        
    await asyncio.gather(*worker_tasks)
    print("\n🎉 EDITING PROCESS COMPLETED!")

if __name__ == "__main__":
    asyncio.run(main())
