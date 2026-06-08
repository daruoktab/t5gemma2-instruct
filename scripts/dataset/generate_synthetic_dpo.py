"""
Generate Synthetic DPO Dataset from SFT Data
=============================================
Menghasilkan dataset preferensi DPO sintetik secara deterministik dari 
gabungan data `chat_train.jsonl` dan `indoqa_train.jsonl` tanpa API berbayar.
Cacat pada 'rejected' dibuat menggunakan aturan cerdas (repetitif, basa-basi kosong, atau terputus).
"""

import json
import os
import random

CHAT_TRAIN_FILE = "data/chat_train.jsonl"
INDOQA_TRAIN_FILE = "data/indoqa_train.jsonl"
OUTPUT_FILE = "data/preferences_dpo_light.jsonl"

def make_rejected_response(chosen: str, flaw_type: str, rng: random.Random) -> str:
    chosen = chosen.strip()
    if flaw_type == "echo_user":
        # Pengulangan repetitif kata/kalimat chosen
        words = chosen.split()
        if len(words) > 5:
            # Ulangi 3 kata pertama berkali-kali
            phrase = " ".join(words[:4])
            return f"{phrase}. {phrase}. {chosen}"
        else:
            return f"{chosen} {chosen} {chosen}"
            
    elif flaw_type == "vague":
        # Kalimat basa-basi kosong Bahasa Indonesia
        vague_templates = [
            "Wah, pertanyaan kamu sangat menarik sekali ya! Mengenai hal itu, sebenarnya ada banyak sekali sudut pandang yang bisa dibahas, tapi aku sendiri kurang begitu yakin dengan detail pastinya. Mungkin kamu bisa coba mencari informasi lebih lanjut di internet atau buku ya.",
            "Hmm, aku mengerti apa yang kamu tanyakan. Tapi sayangnya aku tidak memiliki jawaban yang cukup memuaskan saat ini. Semoga lain kali aku bisa membantu ya! Tetap semangat!",
            "Itu adalah sesuatu yang bagus untuk didiskusikan! Namun, aku sarankan untuk tidak terlalu memikirkan hal itu secara berlebihan. Terima kasih sudah bertanya ya, asisten AI ramah siap membantu kapan saja."
        ]
        return rng.choice(vague_templates)
        
    elif flaw_type == "incomplete":
        # Potong jawaban secara tiba-tiba di tengah kata
        words = chosen.split()
        if len(words) > 8:
            cut_idx = rng.randint(3, len(words) // 2 + 1)
            return " ".join(words[:cut_idx]) + "..."
        else:
            return chosen[:len(chosen)//2] + "..."
    else:
        return chosen

def main():
    print("Memulai pembuatan dataset DPO sintetik lokal...")
    rng = random.Random(42)
    
    # Target total: 1000 data preferensi (700 chat, 300 IndoQA)
    n_chat_target = 700
    n_qa_target = 300
    
    # Muat SFT data
    chat_rows = []
    if os.path.exists(CHAT_TRAIN_FILE):
        with open(CHAT_TRAIN_FILE, "r", encoding="utf-8") as f:
            chat_rows = [json.loads(line) for line in f if line.strip()]
    
    qa_rows = []
    if os.path.exists(INDOQA_TRAIN_FILE):
        with open(INDOQA_TRAIN_FILE, "r", encoding="utf-8") as f:
            qa_rows = [json.loads(line) for line in f if line.strip()]
            
    print(f"Loaded: chat_train={len(chat_rows)} baris, indoqa_train={len(qa_rows)} baris")
    
    # Ambil sample secara deterministik
    selected_chat = chat_rows[:n_chat_target] if len(chat_rows) >= n_chat_target else chat_rows
    selected_qa = qa_rows[:n_qa_target] if len(qa_rows) >= n_qa_target else qa_rows
    
    combined_rows = selected_chat + selected_qa
    rng.shuffle(combined_rows)
    
    print(f"Total data SFT gabungan terpilih: {len(combined_rows)} baris")
    
    dpo_records = []
    flaw_types = ["echo_user", "vague", "incomplete"]
    
    for row in combined_rows:
        inp = row.get("input", "")
        chosen = row.get("target", "")
        if not inp or not chosen:
            continue
            
        flaw = rng.choice(flaw_types)
        rejected = make_rejected_response(chosen, flaw, rng)
        
        # Validasi sederhana
        if chosen.strip() == rejected.strip() or len(rejected.strip()) < 2:
            # Fallback jika rusak terlalu parah
            rejected = "Maaf, aku tidak tahu jawaban dari pertanyaan itu sama sekali."
            flaw = "vague"
            
        rec = {
            "input": inp,
            "chosen": chosen,
            "rejected": rejected,
            "flaw_type": flaw
        }
        dpo_records.append(rec)
        
    # Tulis ke file output
    os.makedirs(os.path.dirname(OUTPUT_FILE), exist_ok=True)
    with open(OUTPUT_FILE, "w", encoding="utf-8") as out:
        for rec in dpo_records:
            out.write(json.dumps(rec, ensure_ascii=False) + "\n")
            
    print(f"✅ Sukses menghasilkan {len(dpo_records)} data preferensi DPO sintetik di {OUTPUT_FILE}")

if __name__ == "__main__":
    main()
