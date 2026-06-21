import os
import json
import time
import argparse
from datetime import datetime
from pydantic import BaseModel, Field
from typing import List
from google import genai
from google.genai import types

# ============================================================================
# KONFIGURASI
# ============================================================================
MODEL_NAME = "gemini-3.1-flash-lite-preview"
OUTPUT_FILE = "t5-gemma-2-chat-instruct-dataset.jsonl"
SLEEP_SECONDS = 1 
MAX_RETRIES = 3
TEMPERATURE = 0.8
MIN_TURNS = 20
MAX_TURNS = 30

SYSTEM_PROMPT = (
    "Kamu adalah asisten AI yang helpful, santai, dan ramah. "
    "Gunakan Bahasa Indonesia sebagai bahasa utama. "
    "Switch ke English hanya kalau user memang minta atau konteksnya English. "
    "Boleh casual dan natural — pakai 'aku/kamu' atau 'saya/Anda' sesuai situasi. "
    "Kalau ada task seperti translate, summarize, paraphrase, atau rewrite "
    "muncul dalam obrolan, langsung bantu dengan natural tanpa basa-basi berlebihan. "
    "Jangan terlalu formal kecuali situasinya memang mengharuskan."
)

# ============================================================================
# SCHEMA PYDANTIC
# ============================================================================
class Turn(BaseModel):
    role: str = Field(..., description="Role: 'user' atau 'assistant'")
    content: str = Field(..., description="Isi percakapan")

class Conversation(BaseModel):
    conversations: List[Turn] = Field(..., description="List of turns in the conversation")

# ============================================================================
# LOGIKA GENERASI & VALIDASI
# ============================================================================
def validate_conversation_logic(convs):
    """Cek apakah role selang-seling."""
    for i in range(len(convs) - 1):
        if convs[i].role == convs[i+1].role:
            return False, i + 1
    return True, -1

def generate_conversation(client, id_num, topik, summary, task_type):
    current_convs = []
    
    base_prompt = f"""Generate sebuah percakapan chatbot antara User dan Assistant dalam Bahasa Indonesia.
Topik: {topik}
Konteks/Summary: {summary}
Tipe Task: {task_type}

Aturan Ketat:
1. Role HARUS bergantian: user, assistant, user, assistant, dst.
2. Minimal {MIN_TURNS} turns, Maksimal {MAX_TURNS} turns.
3. Gaya bahasa casual, santai, dan ramah (pake 'aku/kamu' atau 'saya/Anda' sesuai konteks).
4. Assistant harus memberikan jawaban yang detail, kreatif, dan tidak template.
5. Dilarang keras menyalin 'Summary' atau 'Topik' bulat-bulat ke dalam percakapan. Gunakan sebagai ide dasar saja.
6. Percakapan harus terasa mengalir secara natural.
7. Jika task adalah 'summarization' atau 'rewriting', pastikan teks input di turn 'user' cukup panjang dan realistis.
8. Gunakan istilah-istilah yang umum digunakan orang Indonesia saat ini.
9. Pastikan tidak ada pengulangan kata atau kalimat yang sama persis.
10. JANGAN ADA DUA TURN BERURUTAN DENGAN ROLE YANG SAMA.
"""

    for attempt in range(MAX_RETRIES):
        now = datetime.now().strftime("%H:%M:%S")
        if not current_convs:
            prompt = base_prompt
            print(f"    [{now}] Memanggil API {MODEL_NAME} (Attempt {attempt+1})...")
        else:
            valid_part_json = json.dumps([{"role": c.role, "content": c.content} for c in current_convs], indent=2)
            last_role = current_convs[-1].role
            next_role = "assistant" if last_role == "user" else "user"
            
            prompt = f"""{base_prompt}

KESALAHAN SEBELUMNYA: Role tidak bergantian dengan benar.
BAGIAN YANG SUDAH BENAR (Turn 1 sampai {len(current_convs)}):
{valid_part_json}

TUGAS KAMU:
Lanjutkan percakapan di atas mulai dari Turn {len(current_convs) + 1} dengan role '{next_role}'.
Pastikan TOTAL turns (termasuk bagian yang sudah benar di atas) berada dalam rentang {MIN_TURNS} sampai {MAX_TURNS} turns.
Berikan FULL percakapan dari awal (termasuk bagian yang sudah benar) dalam format JSON.
"""
            print(f"    [{now}] Meminta REFINE (Turn {len(current_convs)} -> {len(current_convs)+1})...")

        try:
            start_time = time.time()
            response = client.models.generate_content(
                model=MODEL_NAME,
                contents=prompt,
                config=types.GenerateContentConfig(
                    response_mime_type="application/json",
                    response_schema=Conversation,
                    max_output_tokens=8192,
                    temperature=TEMPERATURE
                )
            )
            elapsed = time.time() - start_time
            
            if not response.text:
                raise ValueError("API mengembalikan response kosong.")
            
            print(f"    [{datetime.now().strftime('%H:%M:%S')}] Response diterima dalam {elapsed:.1f}s. Validasi...")
            raw_data = json.loads(response.text)
            new_convs = [Turn(**t) for t in raw_data["conversations"]]
            
            is_valid, error_idx = validate_conversation_logic(new_convs)
            if is_valid:
                # Injeksi System Prompt
                if new_convs[0].role != "system":
                    new_convs.insert(0, Turn(role="system", content=SYSTEM_PROMPT))
                return new_convs, response.usage_metadata.candidates_token_count
            else:
                print(f"    [WARNING] Role double di turn {error_idx+1}. Retrying via Refine...")
                current_convs = new_convs[:error_idx]
                continue

        except Exception as e:
            print(f"    [ERROR] Attempt {attempt+1} gagal: {str(e)}")
            time.sleep(2)
    
    return None, 0

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--start", type=int, default=None)
    parser.add_argument("--end", type=int, default=None)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()

    client = genai.Client()
    
    # Header Fancy
    print("="*60)
    print("Dataset Generator — T5Gemma2 Chatbot Distillation")
    print("Mode: Pydantic Structured Output + Auto-Refine")
    print("="*60)

    # Load Topics
    with open("generated_topics_manual.json", "r", encoding="utf-8") as f:
        topics = json.load(f)
    print(f"[INFO] Loaded {len(topics)} topik dari generated_topics_manual.json")

    # Load existing IDs
    existing_ids = set()
    if os.path.exists(OUTPUT_FILE):
        with open(OUTPUT_FILE, "r", encoding="utf-8") as f:
            for line in f:
                try:
                    data = json.loads(line)
                    existing_ids.add(data["id"])
                except: continue

    start_id = args.start
    if args.resume:
        start_id = max(existing_ids) + 1 if existing_ids else 1
        print(f"[RESUME] Melanjutkan dari ID {start_id}")

    end_id = args.end if args.end else len(topics)
    
    print(f"[INFO] Model: {MODEL_NAME}")
    print(f"[INFO] Output: {OUTPUT_FILE}")
    print(f"\n[SCHEMA] Output JSON per entry:")
    print("  id, topik, num_turns, tokens, conversations: List[{role, content}]")
    print(f"\n[SYSTEM PROMPT]")
    print(f"  \"{SYSTEM_PROMPT[:100]}...\"")
    print("="*60)

    success_count = 0
    fail_count = 0
    total_tokens_all = 0

    for item in topics:
        id_num = item["id"]
        if id_num < start_id: continue
        if id_num > end_id: break
        if id_num in existing_ids: continue

        print(f"\n[{success_count+fail_count+1}/{(end_id-start_id)+1}] ID {id_num}: {item['topik']}")
        
        convs, tokens = generate_conversation(client, id_num, item["topik"], item["summary"], item["task"])
        
        if convs:
            entry = {
                "id": id_num,
                "topik": item["topik"],
                "num_turns": len(convs),
                "tokens": tokens,
                "conversations": [{"role": c.role, "content": c.content} for c in convs]
            }
            with open(OUTPUT_FILE, "a", encoding="utf-8") as f:
                f.write(json.dumps(entry, ensure_ascii=False) + "\n")
            
            success_count += 1
            total_tokens_all += tokens
            print(f"  ✓ Berhasil — {len(convs)} turns, ~{tokens} tokens")
        else:
            fail_count += 1
            print(f"  ✗ Gagal.")
        
        time.sleep(SLEEP_SECONDS)

    print("\n" + "="*60)
    print("SELESAI!")
    print(f"  Berhasil: {success_count}")
    print(f"  Gagal:    {fail_count}")
    print(f"  Est. Total Tokens: {total_tokens_all}")
    print("="*60)

if __name__ == "__main__":
    main()