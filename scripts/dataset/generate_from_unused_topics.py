"""
[Phase 1.1] Generate Percakapan Baru (Agentic)
===================================================================
Script ini menggunakan Pydantic AI untuk melakukan generasi percakapan
multi-turn secara iteratif (Trial and Error).
"""

import asyncio
import json
import os
import sys
import random
import argparse
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
API_KEY      = os.environ.get("OPENMODEL_API_KEY") or os.environ.get("API_KEY")
API_MODEL    = os.environ.get("API_MODEL", "deepseek-chat")

# ─── Paths ───────────────────────────────────────────────────────────────────
TOPICS_MANUAL_FILE  = DATA_DIR / "generated_topics_manual.json"
OUTPUT_FILE         = DATA_DIR / "generated_from_unused_topics_agentic.jsonl"
PROGRESS_LOG_FILE   = DATA_DIR / "generated_from_unused_topics_progress.jsonl"

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
    min_turns: int = 30
    max_turns: int = 40
    min_tokens_per_turn: int = 15

    def get_token_count(self, text: str) -> int:
        if not self.tokenizer:
            return len(text.split()) # Fallback
        return len(self.tokenizer.encode(text))

class ConversationResult(BaseModel):
    is_valid: bool = Field(description="Apakah percakapan ini sudah memenuhi kriteria panjang dan kualitas?")
    rationale: str = Field(description="Alasan mengapa percakapan ini dianggap valid dan selesai.")

# ─── Agent Definition ────────────────────────────────────────────────────────

# Setup Model
provider = API_MODEL.split(":")[0] if ":" in API_MODEL else "openai-chat"
model_name_only = API_MODEL.split(":")[-1]
if "deepseek" in model_name_only.lower():
    provider = "anthropic"
model_string = f"{provider}:{model_name_only}"

print(f"[DEBUG] Using provider: {provider}, model_string: {model_string}, base_url: {API_BASE_URL}")


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
        "Kamu adalah spesialis pembuat dataset percakapan multi-turn Bahasa Indonesia.\n"
        "Tugasmu adalah membangun percakapan antara 'user' dan 'assistant' sedikit demi sedikit "
        "menggunakan alat (tools) yang disediakan.\n\n"
        "ATURAN SUPER KETAT:\n"
        "1. Percakapan total HARUS mencapai target {ctx.deps.min_turns} hingga {ctx.deps.max_turns} pesan (termasuk user dan assistant).\n"
        "   (Misal: 15 pesan user + 15 pesan assistant = 30 turns total).\n"
        "2. Kamu TIDAK BOLEH memanggil result akhir (selesai) sebelum jumlah turns mencapai minimal {ctx.deps.min_turns}.\n"
        "3. Setiap turn sebaiknya punya bobot isi (minimal {ctx.deps.min_tokens_per_turn} tokens).\n"
        "4. Gunakan tool `append_turn_pair` untuk menambahkan 1 pasang (user & assistant) secara bertahap.\n"
        "5. Cek kualitasnya secara iteratif. Jika ada pesan yang salah, gunakan `edit_turn_content`.\n"
        "6. Bahasa Indonesia santai/natural. Sisipkan task kompleks di tengah-tengah percakapan (summarize, QA, rewrite).\n"
    )
)

@agent.tool
def append_turn_pair(ctx: RunContext[ConvState], user_message: str, assistant_message: str) -> str:
    """Tambahkan 1 pasang pesan (user lalu assistant) ke akhir percakapan."""
    state = ctx.deps
    
    # Validasi panjang
    u_tok = state.get_token_count(user_message)
    a_tok = state.get_token_count(assistant_message)
    
    if u_tok < 5 or a_tok < state.min_tokens_per_turn:
        return f"GAGAL: Pesan terlalu pendek. User tok: {u_tok}, Assistant tok: {a_tok} (Min asst: {state.min_tokens_per_turn}). Buat lebih detail!"
    
    state.turns.append({"role": "user", "content": user_message})
    state.turns.append({"role": "assistant", "content": assistant_message})
    
    total_turns = len(state.turns)
    sisa = state.min_turns - total_turns
    
    status = f"BERHASIL: 2 pesan ditambahkan. Total sekarang: {total_turns} pesan.\n"
    if sisa > 0:
        status += f"Kamu MASIH KURANG {sisa} pesan lagi untuk mencapai target minimal {state.min_turns}. Terus tambahkan percakapan!"
    else:
        status += f"TARGET TERCAPAI. Kamu sudah mencapai batas minimal {state.min_turns} pesan. Jika kualitas sudah bagus, kamu boleh menyelesaikan percakapan."
        
    return status

@agent.tool
def edit_turn_content(ctx: RunContext[ConvState], turn_index: int, new_content: str) -> str:
    """Edit isi pesan pada indeks tertentu jika dirasa kurang pas (0-indexed)."""
    state = ctx.deps
    if turn_index < 0 or turn_index >= len(state.turns):
        return f"GAGAL: Index {turn_index} di luar batas (0 sampai {len(state.turns)-1})."
        
    old_role = state.turns[turn_index]["role"]
    state.turns[turn_index]["content"] = new_content
    tok = state.get_token_count(new_content)
    
    return f"BERHASIL: Pesan index {turn_index} ({old_role}) diperbarui. Token count: {tok}."

@agent.tool
def get_conversation_status(ctx: RunContext[ConvState]) -> str:
    """Melihat ringkasan seluruh percakapan yang sudah dibuat sampai saat ini."""
    state = ctx.deps
    if not state.turns:
        return "Percakapan masih kosong."
        
    lines = []
    for i, t in enumerate(state.turns):
        lines.append(f"[{i}] {t['role'].upper()} ({state.get_token_count(t['content'])} tok): {t['content'][:60]}...")
    
    summary = "\n".join(lines)
    summary += f"\n\nTotal: {len(state.turns)} pesan. Target: {state.min_turns}-{state.max_turns} pesan."
    return summary

# ─── System Prompt Dinamis ───────────────────────────────────────────────────
@agent.system_prompt
def add_topic_context(ctx: RunContext[ConvState]) -> str:
    return (
        f"Konteks Topik Percakapan yang harus kamu buat:\n"
        f"- Topik: {ctx.deps.topik}\n"
        f"- Ringkasan: {ctx.deps.summary}\n"
        f"- Task Hint: {ctx.deps.task_hint}\n\n"
        f"Sekarang mulai gunakan tool `append_turn_pair` berkali-kali untuk membangun percakapan!"
    )

# ─── Main Execution ──────────────────────────────────────────────────────────

async def main():
    if not API_KEY:
        print("[ERROR] Set OPENMODEL_API_KEY di environment atau .env")
        sys.exit(1)

    parser = argparse.ArgumentParser()
    parser.add_argument("--target", type=int, default=5, help="Jumlah percakapan yang ingin dibuat")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    # Load tokenizer
    print("[INFO] Loading tokenizer google/t5gemma-2-270m-270m...")
    tokenizer = AutoTokenizer.from_pretrained("google/t5gemma-2-270m-270m", trust_remote_code=True)

    # Load topics
    if not TOPICS_MANUAL_FILE.exists():
        print(f"[ERROR] {TOPICS_MANUAL_FILE} tidak ditemukan.")
        sys.exit(1)

    with TOPICS_MANUAL_FILE.open("r", encoding="utf-8") as f:
        all_topics = json.load(f)

    print(f"[INFO] Total topik tersedia: {len(all_topics)}")
    
    # Ambil sample acak
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
            min_turns=30,  # 15 pasang
            max_turns=40,
            min_tokens_per_turn=20
        )
        
        try:
            # Jalankan agent loop
            result = await agent.run("Silakan mulai membangun percakapan dari awal.", deps=state)
            
            final_output = cast(ConversationResult, result.output)
            if final_output.is_valid and len(state.turns) >= state.min_turns:
                # Format ke JSONL sesuai standar model SFT
                final_conv = [{"role": "system", "content": SYSTEM_PROMPT}] + state.turns
                
                # Hitung total token
                total_tok = sum(state.get_token_count(t["content"]) for t in state.turns)
                
                entry_out = {
                    "id": 20000 + produced,
                    "topik": topik,
                    "num_turns": len(final_conv),
                    "tokens": total_tok,
                    "conversations": final_conv,
                    "rationale": final_output.rationale
                }
                
                with OUTPUT_FILE.open("a", encoding="utf-8") as f:
                    f.write(json.dumps(entry_out, ensure_ascii=False) + "\n")
                    
                print(f"  ✓ BERHASIL: {len(state.turns)} turns, {total_tok} tokens.")
                produced += 1
            else:
                print(f"  ✗ GAGAL: Agent menyelesaikan loop tapi syarat minimal tidak tercapai (Turns: {len(state.turns)}).")
                
        except Exception as e:
            print(f"  ✗ ERROR: {e}")

    print(f"\n[SELESAI] {produced} percakapan agentic berhasil disimpan ke {OUTPUT_FILE}")

if __name__ == "__main__":
    asyncio.run(main())
