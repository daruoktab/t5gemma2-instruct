"""
Generate Real DPO Preferences via DeepSeek API (Random Sampling)
==================================================================
Menghasilkan 100 data preferensi DPO berkualitas tinggi menggunakan DeepSeek API:
- 70 data percakapan multi-turn yang diambil secara acak dari full chat threads.
- 30 data QA diambil secara acak dari IndoQA.
- Respon 'rejected' dirusak secara realistis oleh DeepSeek.
"""

import os
import sys
import json
import time
import random
import re
from pathlib import Path

# Load dotenv if exists
SCRIPT_DIR = Path(__file__).resolve().parent
try:
    from dotenv import load_dotenv
    load_dotenv(SCRIPT_DIR / "../../.env")
except ImportError:
    pass

try:
    from openai import OpenAI
except ImportError:
    print("Install: pip install openai", file=sys.stderr)
    raise SystemExit(1)

CHAT_MERGED_FILE = "data/t5-gemma-2-chat-instruct-merged.jsonl"
INDOQA_TRAIN_FILE = "data/indoqa_train.jsonl"
OUTPUT_FILE = "data/preferences_dpo_light.jsonl"

DEEPSEEK_BASE_URL = os.environ.get("DEEPSEEK_BASE_URL", "https://api.deepseek.com")
API_KEY = os.environ.get("DEEPSEEK_API_KEY")
MODEL_CHAT = os.environ.get("DEEPSEEK_MODEL", "deepseek-chat")

SYSTEM_PROMPT = (
    "Kamu adalah asisten AI yang helpful, santai, dan ramah. "
    "Gunakan Bahasa Indonesia sebagai bahasa utama."
)

def extract_multi_turn_history(conversations: list[dict]) -> tuple[str, str]:
    """
    Mengekstrak riwayat percakapan penuh (multi-turn) hingga turn user terakhir
    dan respon asisten terakhir sebagai target asli.
    Format keluaran input: system:\n...\nuser:\n...\nassistant:\n...\nuser: (terakhir)
    """
    # Cari indeks asisten terakhir
    last_assistant_idx = -1
    for idx, turn in enumerate(conversations):
        if turn.get("role") == "assistant":
            last_assistant_idx = idx
            
    if last_assistant_idx == -1:
        raise ValueError("Tidak ditemukan turn assistant dalam percakapan.")
        
    # Respon asisten terakhir
    target = conversations[last_assistant_idx]["content"].strip()
    
    # Kumpulkan history hingga turn user terakhir (last_assistant_idx - 1)
    history_turns = conversations[:last_assistant_idx]
    
    formatted_lines = []
    for turn in history_turns:
        role = turn.get("role")
        content = turn.get("content", "").strip()
        if role == "system":
            formatted_lines.append(f"system: {content}")
        elif role == "user":
            formatted_lines.append(f"user: {content}")
        elif role == "assistant":
            formatted_lines.append(f"assistant: {content}")
            
    inp = "\n".join(formatted_lines)
    return inp, target

def build_api_prompt(inp: str, target: str, flaw: str) -> str:
    flaw_descriptions = {
        "echo_user": "rejected harus mengulang atau hampir menyalin kata-kata pertanyaan user di turn terakhir secara repetitif tanpa memberikan jawaban substantif.",
        "vague": "rejected berisi basa-basi kosong Bahasa Indonesia santai (seperti 'Wah seru sekali pertanyaannya!') tanpa isi yang menjawab substansi masalah.",
        "hallucination": "rejected menyatakan klaim atau fakta ilmiah/sejarah konkret yang terdengar sangat meyakinkan tapi sebenarnya salah atau tidak diverifikasi (misleading).",
        "incomplete": "rejected memotong kalimat secara tiba-tiba di tengah jalan (hanya menyisakan tanda '...').",
        "off_topic": "rejected menjawab topik lain yang tidak relevan dengan pertanyaan user."
    }
    
    return f"""Kamu adalah penilai dataset DPO (Direct Preference Optimization).
Tugasmu adalah menghasilkan data preferensi ("chosen" vs "rejected") dari data percakapan asli berikut:

PROMPT INPUT (percakapan asli dengan riwayat lengkap):
{inp}

TARGET ASLI (jawaban asli dari dataset):
{target}

TUGAS:
1. chosen: Gunakan TARGET ASLI secara utuh, atau sedikit sempurnakan agar lebih ramah, alami, dan informatif dalam Bahasa Indonesia santai (sesuai sistem asisten santai).
2. rejected: Hasilkan respon alternatif yang LEBIH BURUK (rejected) dengan cacat tipe: "{flaw}".
   Deskripsi cacat: {flaw_descriptions[flaw]}
3. flaw_type: Harus persis string "{flaw}".
4. rationale: Alasan singkat satu kalimat dalam Bahasa Indonesia mengapa respon rejected tersebut buruk.

Keluarkan HANYA objek JSON valid berikut tanpa markdown code fence:
{{
  "input": "...",  // Salin PROMPT INPUT asli apa adanya
  "chosen": "...", // Respon chosen yang berkualitas
  "rejected": "...", // Respon rejected yang dirusak
  "flaw_type": "{flaw}",
  "rationale": "..." // Alasan
}}"""

def parse_json_loose(text: str) -> dict:
    raw = (text or "").strip()
    if raw.startswith("```"):
        raw = re.sub(r"^```(?:json)?\s*", "", raw)
        raw = re.sub(r"\s*```\s*$", "", raw).strip()
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        start = raw.find("{")
        end = raw.rfind("}")
        if start < 0 or end <= start:
            raise ValueError(f"bukan JSON: {raw[:100]}")
        return json.loads(raw[start:end+1])

def main():
    print("Memulai generator dataset DPO preferensi (Random Sampling) menggunakan DeepSeek API...")
    
    if not API_KEY:
        print("[ERROR] DEEPSEEK_API_KEY tidak ditemukan di environment. Silakan set di terminal atau file .env.")
        sys.exit(1)
        
    client = OpenAI(api_key=API_KEY, base_url=DEEPSEEK_BASE_URL)
    rng = random.Random(42)
    
    # 1. Load full chat threads
    chat_rows = []
    if os.path.exists(CHAT_MERGED_FILE):
        with open(CHAT_MERGED_FILE, "r", encoding="utf-8") as f:
            chat_rows = [json.loads(line) for line in f if line.strip()]
            
    # 2. Load IndoQA
    qa_rows = []
    if os.path.exists(INDOQA_TRAIN_FILE):
        with open(INDOQA_TRAIN_FILE, "r", encoding="utf-8") as f:
            qa_rows = [json.loads(line) for line in f if line.strip()]
            
    print(f"Loaded: t5-gemma-2-chat-merged={len(chat_rows)} threads, indoqa_train={len(qa_rows)} samples")
    
    # Random sampling 70 dari chat merged dan 30 dari IndoQA
    selected_chat = rng.sample(chat_rows, 70) if len(chat_rows) >= 70 else chat_rows
    selected_qa = rng.sample(qa_rows, 30) if len(qa_rows) >= 30 else qa_rows
    
    print("Randomly selected 70 chat threads and 30 IndoQA samples.")
    
    # Proses unrolling & formatting
    combined = []
    
    # Proses chat threads (full conversation context)
    for row in selected_chat:
        try:
            inp, target = extract_multi_turn_history(row["conversations"])
            combined.append({"input": inp, "target": target, "type": "chat"})
        except Exception as e:
            print(f"  [warn] skip thread id {row.get('id')}: {e}")
            
    # Proses IndoQA (single-turn QA)
    for row in selected_qa:
        combined.append({
            "input": row.get("input", ""),
            "target": row.get("target", ""),
            "type": "qa"
        })
        
    rng.shuffle(combined)
    print(f"Total data DPO jangkar siap diproses: {len(combined)}")
    
    flaw_types = ["echo_user", "vague", "hallucination", "incomplete", "off_topic"]
    os.makedirs(os.path.dirname(OUTPUT_FILE), exist_ok=True)
    
    produced = 0
    with open(OUTPUT_FILE, "w", encoding="utf-8") as out:
        for i, row in enumerate(combined):
            inp = row["input"]
            target = row["target"]
            if not inp or not target:
                continue
                
            flaw = rng.choice(flaw_types)
            prompt = build_api_prompt(inp, target, flaw)
            
            print(f"[{i+1}/{len(combined)}] Menghubungi DeepSeek ({row['type']}) untuk flaw={flaw}...")
            
            retries = 3
            success = False
            while retries > 0 and not success:
                try:
                    completion = client.chat.completions.create(
                        model=MODEL_CHAT,
                        messages=[{"role": "user", "content": prompt}],
                        temperature=0.7,
                        max_tokens=2048,
                        response_format={"type": "json_object"}
                    )
                    text = completion.choices[0].message.content
                    data = parse_json_loose(text)
                    
                    # Tulis ke file output secara real-time
                    out.write(json.dumps(data, ensure_ascii=False) + "\n")
                    produced += 1
                    print(f"  ✓ Sukses disimpan. Chosen length: {len(data['chosen'])}, Rejected length: {len(data['rejected'])}")
                    success = True
                except Exception as e:
                    retries -= 1
                    print(f"  ✗ Gagal: {e}. Retries remaining: {retries}")
                    time.sleep(2)
            
            # Rate limit friendly sleep
            time.sleep(1.0)
            
    print(f"\n✅ Selesai! Berhasil menghasilkan {produced} data DPO preferensi di {OUTPUT_FILE}")

if __name__ == "__main__":
    main()
