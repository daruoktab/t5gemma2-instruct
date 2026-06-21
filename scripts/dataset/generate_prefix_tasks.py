import asyncio
import json
import os
import sys
import random
import argparse
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
API_KEY      = os.environ.get("OPENMODEL_API_KEY") or os.environ.get("API_KEY")
API_MODEL    = os.environ.get("API_MODEL", "deepseek-chat")

# ─── Paths ───────────────────────────────────────────────────────────────────
TOPICS_MANUAL_FILE  = DATA_DIR / "generated_topics_manual.json"
OUTPUT_FILE         = DATA_DIR / "generated_prefix_tasks_agentic.jsonl"

SYSTEM_PROMPT = (
    "Kamu adalah asisten AI yang helpful, santai, dan ramah. "
    "Gunakan Bahasa Indonesia sebagai bahasa utama."
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
    min_tokens_per_turn: int = 10

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

if provider in ["openai", "openai-chat"]:
    if API_BASE_URL:
        os.environ["OPENAI_BASE_URL"] = API_BASE_URL
    if API_KEY:
        os.environ["OPENAI_API_KEY"] = API_KEY
elif provider == "anthropic":
    if API_BASE_URL:
        base = API_BASE_URL
        if base.endswith("/v1"):
            base = base[:-3]
        elif base.endswith("/v1/"):
            base = base[:-4]
        os.environ["ANTHROPIC_BASE_URL"] = base
    if API_KEY:
        os.environ["ANTHROPIC_API_KEY"] = API_KEY

agent = Agent(
    model_string,
    deps_type=ConvState,
    output_type=ConversationResult,
    system_prompt=(
        "Kamu adalah spesialis pembuat dataset percakapan multi-turn Bahasa Indonesia untuk tugas NLP campuran.\n"
        "Tugasmu adalah membangun percakapan secara bertahap menggunakan tool `append_turn_pair`.\n\n"
        "ATURAN SUPER KETAT:\n"
        "1. Percakapan HARUS mencapai target {ctx.deps.min_turns} hingga {ctx.deps.max_turns} pesan total (termasuk user dan assistant).\n"
        "2. Di setiap turn assistant, kamu WAJIB menentukan array `task_prefixes` dari TaskType Enum yang relevan dengan pertanyaan user.\n"
        "   - Contoh: Jika user minta ringkas & terjemahkan, pilih [SUMMARIZE, TRANSLATE].\n"
        "   - Jika hanya ngobrol biasa, pilih [GENERAL_CHAT].\n"
        "3. User (pengguna manusia) TIDAK BOLEH tahu atau mengucapkan prefix. Pengguna ngobrol dengan bahasa natural biasa.\n"
        "4. Asisten menjawab seperti biasa (kamu cukup isi teksnya di `assistant_message`, prefix token akan disisipkan otomatis oleh tool).\n"
    )
)

@agent.tool
def append_turn_pair(
    ctx: RunContext[ConvState], 
    user_message: str, 
    task_prefixes: list[TaskType], 
    assistant_message: str
) -> str:
    """Tambahkan 1 pasang pesan (user lalu assistant) ke akhir percakapan dengan Task Prefixes yang sesuai."""
    state = ctx.deps
    
    if not task_prefixes:
        return "GAGAL: Kamu harus memilih setidaknya 1 task_prefix dari Enum TaskType!"
        
    prefix_str = "".join([t.value for t in task_prefixes])
    
    # Prepend the unused tokens
    final_assistant_message = f"{prefix_str}{assistant_message}"
    
    u_tok = state.get_token_count(user_message)
    a_tok = state.get_token_count(final_assistant_message)
    
    if u_tok < 5 or a_tok < state.min_tokens_per_turn:
        return f"GAGAL: Pesan terlalu pendek. User tok: {u_tok}, Assistant tok: {a_tok} (Min: {state.min_tokens_per_turn})."
    
    state.turns.append({"role": "user", "content": user_message})
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
        f"Gunakan `append_turn_pair` berulang kali. Ingat, setiap respons asisten butuh task_prefixes yang tepat!"
    )

async def main():
    if not API_KEY:
        print("[ERROR] Set OPENMODEL_API_KEY")
        sys.exit(1)

    parser = argparse.ArgumentParser()
    parser.add_argument("--target", type=int, default=1)
    args = parser.parse_args()

    tokenizer = AutoTokenizer.from_pretrained("google/t5gemma-2-270m-270m", trust_remote_code=True)

    with TOPICS_MANUAL_FILE.open("r", encoding="utf-8") as f:
        all_topics = json.load(f)
    
    to_generate = random.sample(all_topics, min(args.target, len(all_topics)))
    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    
    produced = 0
    for i, entry in enumerate(to_generate):
        topik = entry.get("topik", "")
        print(f"\n[{i+1}/{len(to_generate)}] Generating percakapan agentic untuk topik: '{topik}'")
        
        state = ConvState(
            topik=topik,
            summary=entry.get("summary", ""),
            task_hint=entry.get("task", ""),
            tokenizer=tokenizer,
            min_turns=10, # Kita buat 5 pasang saja (10 turns) untuk testing
            max_turns=14,
            min_tokens_per_turn=10
        )
        
        try:
            result = await agent.run("Silakan mulai membangun percakapan dari awal. Ingat untuk menggunakan Enum TaskType dengan benar.", deps=state)
            
            final_output = cast(ConversationResult, result.output)
            if final_output.is_valid and len(state.turns) >= state.min_turns:
                final_conv = [{"role": "system", "content": SYSTEM_PROMPT}] + state.turns
                total_tok = sum(state.get_token_count(t["content"]) for t in state.turns)
                
                entry_out = {
                    "id": 90000 + produced,
                    "topik": topik,
                    "num_turns": len(final_conv),
                    "tokens": total_tok,
                    "conversations": final_conv,
                    "rationale": final_output.rationale
                }
                
                with OUTPUT_FILE.open("w", encoding="utf-8") as f:
                    f.write(json.dumps(entry_out, ensure_ascii=False) + "\n")
                    
                print(f"  ✓ BERHASIL: {len(state.turns)} turns, {total_tok} tokens.")
                produced += 1
            else:
                print(f"  ✗ GAGAL: (Turns: {len(state.turns)}). Rationale: {final_output.rationale}")
                
        except Exception as e:
            print(f"  ✗ ERROR: {e}")

if __name__ == "__main__":
    asyncio.run(main())
