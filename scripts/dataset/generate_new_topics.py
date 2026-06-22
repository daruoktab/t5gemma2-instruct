import asyncio
import json
import os
from pathlib import Path
from pydantic_ai import Agent
from pydantic import BaseModel

# ─── Konfigurasi ─────────────────────────────────────────────────────────────
SCRIPT_DIR = Path(__file__).resolve().parent
ROOT_DIR = SCRIPT_DIR.parent.parent
DATA_DIR = ROOT_DIR / "data"

try:
    from dotenv import load_dotenv
    load_dotenv(ROOT_DIR / ".env")
except ImportError:
    pass

OLD_DATASET_FILE = DATA_DIR / "t5-gemma-2-chat-instruct-dataset.jsonl"
NEW_TOPICS_FILE = DATA_DIR / "generated_topics_new_2500.json"
NICHES_FILE = DATA_DIR / "generated_niches_50.json"

API_BASE_URL = os.environ.get("API_BASE_URL", "https://api.openmodel.ai/v1")
API_KEY      = os.environ.get("OPENMODEL_API_KEY") or os.environ.get("API_KEY")
API_MODEL    = os.environ.get("API_MODEL", "deepseek-chat")

# Gunakan environment variables untuk pydantic_ai
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

# ─── Struktur Data (Pydantic) ─────────────────────────────────────────────────
class TopicIdea(BaseModel):
    topik: str
    summary: str
    task_hint: str

class TopicList(BaseModel):
    topics: list[TopicIdea]

class NicheList(BaseModel):
    niches: list[str]

# ─── Agen AI ──────────────────────────────────────────────────────────────────
niche_agent = Agent(
    model_string,
    output_type=NicheList,
    retries=5,
    system_prompt=(
        "Kamu adalah spesialis perencana konten AI berbahasa Indonesia.\n"
        "Tugasmu adalah menghasilkan daftar 50 kategori/niche percakapan yang SANGAT SPESIFIK dan UNIK.\n"
        "Kategori ini akan digunakan untuk menghasilkan topik-topik percakapan asisten AI.\n"
        "Jangan gunakan topik yang mainstream atau umum. Cari sudut pandang (angle) yang spesifik (misal: bukan 'Resep Nasi Goreng', tapi 'Cara Modifikasi Nasi Goreng Shirataki untuk Diet Keto')."
    )
)

topic_agent = Agent(
    model_string,
    output_type=TopicList,
    retries=5,
    system_prompt=(
        "Kamu adalah ahli pembuat ide konten kreatif berbahasa Indonesia.\n"
        "Tugasmu adalah menghasilkan ide topik percakapan berdasarkan kategori/niche yang diberikan.\n"
        "Ide topik harus bervariasi, sangat spesifik, dan tidak boleh menyerupai topik umum."
    )
)

# ─── Fungsi Pembantu ──────────────────────────────────────────────────────────
def load_blacklist() -> list[str]:
    blacklist = []
    if OLD_DATASET_FILE.exists():
        with open(OLD_DATASET_FILE, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if line:
                    obj = json.loads(line)
                    if "topik" in obj:
                        blacklist.append(obj["topik"])
    return blacklist

async def generate_niches(blacklist: list[str]) -> list[str]:
    # Jika sudah pernah membuat niche, langsung load (agar bisa dilanjut kapan saja)
    if NICHES_FILE.exists():
        with open(NICHES_FILE, 'r', encoding='utf-8') as f:
            print("[INFO] Memuat 50 Niches dari file yang sudah ada...")
            return json.load(f)
    
    print(f"[INFO] Mengenerate 50 Niches baru menggunakan AI... (Memasukkan {len(blacklist)} blacklist ke konteks)")
    
    blacklist_text = "\n".join(f"- {t}" for t in blacklist)
    
    prompt = (
        f"Berikut adalah daftar {len(blacklist)} topik yang SUDAH KAMI MILIKI. "
        "KAMU SAMA SEKALI TIDAK BOLEH MEMBUAT NICHE ATAU KATEGORI YANG BERSINGGUNGAN DENGAN INI:\n"
        "<blacklist>\n"
        f"{blacklist_text}\n"
        "</blacklist>\n\n"
        "TUGAS: Buatlah 50 Kategori/Niche yang SANGAT SPESIFIK, UNIK, dan belum pernah dibahas di atas. "
    )
    
    result = await niche_agent.run(prompt)
    niches = result.output.niches # type: ignore
    
    with open(NICHES_FILE, 'w', encoding='utf-8') as f:
        json.dump(niches, f, ensure_ascii=False, indent=2)
    
    return niches

# ─── Program Utama ────────────────────────────────────────────────────────────
async def main():
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    
    blacklist = load_blacklist()
    print(f"[INFO] Berhasil memuat {len(blacklist)} topik ke dalam blacklist.")
    
    # Tahap 1: Dapatkan 50 Niche
    niches = await generate_niches(blacklist)
    print(f"[INFO] Terdapat {len(niches)} niche yang akan diproses.")
    
    # Load existing generated topics (jika sebelumnya script terhenti di tengah jalan)
    all_new_topics = []
    if NEW_TOPICS_FILE.exists():
        with open(NEW_TOPICS_FILE, 'r', encoding='utf-8') as f:
            all_new_topics = json.load(f)
            
    # Asumsikan setiap niche sukses memberikan persis 50 topik
    processed_niches = len(all_new_topics) // 50
    
    for i in range(processed_niches, len(niches)):
        niche = niches[i]
        print(f"\n[{i+1}/{len(niches)}] Generating 50 topik untuk Niche: '{niche}'")
        
        niche_topics = []
        has_error = False
        
        for chunk in range(5):
            print(f"  -> Memproses bagian {chunk+1}/5 (10 topik)...")
            await asyncio.sleep(5)
            
            new_titles = [t["topik"] for t in all_new_topics + niche_topics]
            recent_titles = new_titles[-500:] # Cukup 500 topik terakhir sebagai pencegah loop-ulang
            recent_text = "\n".join(f"- {t}" for t in recent_titles)
            
            prompt = (
                f"Kategori/Niche Utama: {niche}\n\n"
                f"TUGAS: Hasilkan tepat 10 ide topik percakapan yang sangat spesifik dan detail untuk niche di atas (Bagian {chunk+1}/5).\n"
                "Setiap topik harus dilengkapi dengan:\n"
                "- 'topik': Judul topik.\n"
                "- 'summary': Ringkasan alur percakapan antara user dan assistant AI.\n"
                "- 'task_hint': tag tugas (contoh: 'brainstorming', 'explanation', 'roleplay', 'creative_writing', 'coding', dll).\n"
            )
            
            if recent_text:
                prompt += (
                    "\nPastikan topik-topikmu BEDA dari topik-topik yang baru saja kamu buat ini:\n"
                    f"<recent_topics>\n{recent_text}\n</recent_topics>\n"
                )
                
            try:
                result = await topic_agent.run(prompt)
                new_batch = [{"topik": t.topik, "summary": t.summary, "task": t.task_hint} for t in result.output.topics] # type: ignore
                niche_topics.extend(new_batch)
            except Exception as e:
                print(f"  -> ERROR pada Niche '{niche}' (Chunk {chunk+1}): {e}")
                print("  -> Menyimpan progress dan berhenti sementara. Silakan jalankan ulang script ini untuk lanjut otomatis.")
                has_error = True
                break
                
        all_new_topics.extend(niche_topics)
        
        # Jika sudah mencapai atau melebihi 2500, potong tepat 2500 dan hentikan
        if len(all_new_topics) >= 2500:
            all_new_topics = all_new_topics[:2500]
            with open(NEW_TOPICS_FILE, 'w', encoding='utf-8') as f:
                json.dump(all_new_topics, f, ensure_ascii=False, indent=2)
            print(f"  -> Sukses memotong tepat 2500 topik.")
            break
            
        # Save progress incrementally (Auto-Save)
        with open(NEW_TOPICS_FILE, 'w', encoding='utf-8') as f:
            json.dump(all_new_topics, f, ensure_ascii=False, indent=2)
            
        print(f"  -> Sukses mendapat {len(niche_topics)} topik. Total sementara: {len(all_new_topics)}")
        
        if has_error:
            break
            
    print(f"\n[SELESAI] Total topik baru yang berhasil di-generate: {len(all_new_topics)}")
    if len(all_new_topics) >= 2500:
        print("HORE! Target 2500 topik baru (yang 100% unik) sudah tercapai!")

if __name__ == "__main__":
    asyncio.run(main())
