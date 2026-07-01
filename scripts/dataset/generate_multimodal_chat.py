import os
import sys
import json
import time
import argparse
import random
import re
import base64
import requests
from pathlib import Path
from PIL import Image
from pydantic import BaseModel, Field
from dotenv import load_dotenv

# Muat variabel dari .env
load_dotenv()

# Path Konfigurasi
SCRIPT_DIR = Path(__file__).resolve().parent
ROOT_DIR = SCRIPT_DIR.parent.parent
IMAGE_DIR = ROOT_DIR / "data" / "multimodal" / "images"
METADATA_FILE = IMAGE_DIR / "random_metadata.json"
DOC_METADATA_FILE = IMAGE_DIR / "doc_metadata.json"
OUTPUT_FILE = ROOT_DIR / "data" / "multimodal" / "train_vision.jsonl"

SYSTEM_PROMPT = (
    "Kamu adalah Gemma, asisten AI cerdas berbahasa Indonesia yang dirancang untuk membantu pengguna dalam berbagai tugas pemrosesan bahasa (NLP), pemahaman visual, maupun percakapan sehari-hari. "
    "Berikan respons yang akurat, terstruktur, ramah, dan natural."
)

# ============================================================================
# CLIENT OPENAGENTIC KUSTOM
# ============================================================================
class OpenAgenticClient:
    def __init__(self, api_key: str, base_url: str):
        self.api_key = api_key
        self.base_url = base_url
        
    def generate_chat(self, model: str, messages: list, temperature: float = 0.7) -> str:
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
        payload = {
            "model": model,
            "messages": messages,
            "temperature": temperature
        }
        url = f"{self.base_url.rstrip('/')}/chat/completions"
        
        # Timeout diset cukup panjang (120s) karena memproses banyak gambar
        response = requests.post(url, headers=headers, json=payload, timeout=120)
        
        if response.status_code != 200:
            raise Exception(f"HTTP Error {response.status_code}: {response.text}")
            
        # Bersihkan respon dari bug "data: [DONE]" dari OpenAgentic gateway
        text = response.text.strip()
        if text.endswith("data: [DONE]"):
            text = text[:-12].strip()
            
        data = json.loads(text)
        return data["choices"][0]["message"]["content"]

# ============================================================================
# SCHEMA PYDANTIC
# ============================================================================
class Turn(BaseModel):
    role: str = Field(..., description="Role: 'user' atau 'assistant'")
    content: str = Field(..., description="Isi pesan percakapan")

class Conversation(BaseModel):
    conversations: list[Turn] = Field(..., description="Daftar giliran percakapan")

def validate_alternating_roles(turns: list[Turn]) -> bool:
    """Memastikan role bergantian user, assistant, user, assistant dst."""
    if not turns:
        return False
    if turns[0].role != "user":
        return False
    for i in range(len(turns) - 1):
        if turns[i].role == turns[i+1].role:
            return False
    return True

def encode_image_to_base64(image_path: Path) -> str:
    """Mengonversi file gambar ke string base64."""
    with open(image_path, "rb") as img_file:
        return base64.b64encode(img_file.read()).decode("utf-8")

def extract_json_block(text: str) -> str:
    """Mengekstrak blok JSON secara kokoh dari respon teks markdown."""
    match = re.search(r"```json\s*(.*?)\s*```", text, re.DOTALL)
    if match:
        return match.group(1).strip()
    match = re.search(r"```\s*(.*?)\s*```", text, re.DOTALL)
    if match:
        return match.group(1).strip()
    start_idx = text.find("{")
    end_idx = text.rfind("}")
    if start_idx != -1 and end_idx != -1:
        return text[start_idx:end_idx+1].strip()
    return text.strip()

def generate_chat_for_images(client: OpenAgenticClient, image_paths: list[Path], model_name: str, image_metadata: dict) -> list[dict] | None:
    # 1. Konversi gambar ke format base64 untuk OpenAI API
    content = []
    
    is_multipage_doc = len(image_paths) > 1
    metadata_context = ""
    
    if not is_multipage_doc:
        # Single image (KORIKA)
        image_path = image_paths[0]
        if image_path.name in image_metadata:
            meta = image_metadata[image_path.name]
            cap_id = meta.get("caption_id")
            cap_en = meta.get("caption_en")
            loc = meta.get("location")
            metadata_context = (
                f"\nINFORMASI UTAMA TENTANG GAMBAR INI (Fakta Grounding):\n"
                f"- Deskripsi Gambar (Bahasa Indonesia): {cap_id}\n"
                f"- Deskripsi Gambar (Bahasa Inggris): {cap_en}\n"
                f"- Lokasi/Konteks Budaya: {loc}\n"
                f"Gunakan informasi di atas sebagai kebenaran mutlak untuk membahas gambar di dalam percakapan."
            )
        
        prompt = f"""Generate a realistic, high-quality, multi-turn conversation between a human user and an AI assistant named Gemma.
The conversation is about the provided image.
{metadata_context}

Aturan Penting:
1. Percakapan HARUS dalam Bahasa Indonesia (semi-formal/casual, santai dan ramah).
2. Jumlah giliran percakapan HARUS tepat 10 giliran (5 pesan user, 5 pesan assistant).
3. Urutan role HARUS bergantian: user, assistant, user, assistant, dst. (dimulai dengan user).
4. Buatlah percakapan yang natural dan mendalam tentang isi gambar tersebut. User bertanya tentang detail, meminta penjelasan, meminta kesimpulan, atau menanyakan hal-hal terkait kontekstual gambar, dan Assistant menjawab dengan lengkap dan informatif.
5. Format respon HARUS berupa JSON sesuai skema berikut di dalam blok markdown ```json ... ```:
{{
  "conversations": [
    {{"role": "user", "content": "..."}},
    {{"role": "assistant", "content": "..."}},
    ...
  ]
}}"""

    else:
        # Multi-page document
        doc_name = image_paths[0].name.split("_page_")[0]
        metadata_context = (
            f"\nINFORMASI UTAMA TENTANG DOKUMEN INI:\n"
            f"- Nama Dokumen Asal: {doc_name}.pdf\n"
            f"- Jumlah Halaman: {len(image_paths)}\n"
            f"Setiap gambar yang dikirim secara berurutan mewakili Halaman 1, Halaman 2, dst."
        )
        
        prompt = f"""Generate a realistic, high-quality, multi-turn conversation between a human user and an AI assistant named Gemma.
The conversation is about the provided multi-page document (all pages are provided as sequential images).
{metadata_context}

Aturan Penting:
1. Percakapan HARUS dalam Bahasa Indonesia (semi-formal/casual, santai dan ramah).
2. Jumlah giliran percakapan HARUS tepat 10 giliran (5 pesan user, 5 pesan assistant).
3. Urutan role HARUS bergantian: user, assistant, user, assistant, dst. (dimulai dengan user).
4. Buatlah percakapan yang mendalam mengenai isi dokumen ini. User bisa menanyakan informasi spesifik dari halaman tertentu (misalnya halaman 1 atau halaman terakhir), meminta ringkasan seluruh dokumen, perbandingan antar halaman, atau instruksi lain yang relevan secara kontekstual, dan Assistant menjawab secara akurat berdasarkan isi halaman-halaman dokumen.
5. Pembahasan di dalam percakapan harus menjelaskan isi dokumen secara terstruktur dan informatif dalam Bahasa Indonesia.
6. Format respon HARUS berupa JSON sesuai skema berikut di dalam blok markdown ```json ... ```:
{{
  "conversations": [
    {{"role": "user", "content": "..."}},
    {{"role": "assistant", "content": "..."}},
    ...
  ]
}}"""

    # Bangun array content multimodal untuk OpenAI
    content.append({"type": "text", "text": prompt})
    for path in image_paths:
        try:
            base64_str = encode_image_to_base64(path)
            content.append({
                "type": "image_url",
                "image_url": {
                    "url": f"data:image/png;base64,{base64_str}"
                }
            })
        except Exception as e:
            print(f"  [ERROR] Gagal membaca gambar {path.name}: {e}")
            return None

    max_retries = 3
    for attempt in range(max_retries):
        try:
            start_time = time.time()
            raw_text = client.generate_chat(
                model=model_name,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": content}
                ],
                temperature=0.7
            )
            elapsed = time.time() - start_time
            
            # Ekstrak JSON
            json_str = extract_json_block(raw_text)
            raw_data = json.loads(json_str)
            
            # Validasi struktur menggunakan Pydantic
            validated_conv = Conversation(**raw_data)
            turns = validated_conv.conversations
            
            # Validasi panjang turn (harus 10 turn) dan bergantian
            if len(turns) != 10:
                print(f"  [!] Turn length mismatch: {len(turns)} turns. Retrying...")
                continue
                
            if not validate_alternating_roles(turns):
                print(f"  [!] Alternating roles validation failed. Retrying...")
                continue
                
            # Prepend token image 📷 sebanyak jumlah gambar ke turn pertama user
            image_tokens = "📷" * len(image_paths)
            turns[0].content = f"{image_tokens}\n{turns[0].content}"
            
            # Format pesan final termasuk system prompt
            messages = [{"role": "system", "content": SYSTEM_PROMPT}]
            for t in turns:
                messages.append({"role": t.role, "content": t.content})
                
            print(f"  [OK] Berhasil generate dalam {elapsed:.1f} detik.")
            return messages
            
        except Exception as e:
            err_msg = str(e)
            if "429" in err_msg or "rate limit" in err_msg.lower():
                wait_time = (attempt + 1) * 30
                print(f"  [!] Terkena Rate Limit (429). Menunggu {wait_time} detik (Attempt {attempt+1}/{max_retries})...")
                time.sleep(wait_time)
            else:
                print(f"  [!] Error pada Attempt {attempt+1}/{max_retries}: {e}")
                time.sleep(2)
                
    return None

def main():
    default_model = os.environ.get("OPENAGENTIC_MODEL") or os.environ.get("MODEL_NAME") or "claude-sonnet-4.6"
    
    parser = argparse.ArgumentParser(description="Generate structured multimodal Indonesian chat dataset using OpenAI compatible API")
    parser.add_argument("--model", type=str, default=default_model, help="Nama model LLM yang digunakan")
    parser.add_argument("--output", type=str, default=str(OUTPUT_FILE), help="File output JSONL")
    args = parser.parse_args()
    
    # 1. Setup OpenAI compatible client kustom
    api_key = os.environ.get("OPENAGENTIC_API_KEY")
    base_url = os.environ.get("OPENAGENTIC_API_URL") or "https://openagentic.id/api/v1"
    
    if not api_key:
        print("[ERROR] OPENAGENTIC_API_KEY tidak ditemukan di .env atau variabel lingkungan.")
        sys.exit(1)
        
    print(f"[INFO] Menggunakan OpenAgentic API Endpoint: {base_url}")
    print(f"[INFO] Menggunakan Model: {args.model}")
    
    client = OpenAgenticClient(
        api_key=api_key,
        base_url=base_url
    )
    
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # 2. Baca daftar gambar dan metadata
    if not IMAGE_DIR.exists():
        print(f"[ERROR] Folder gambar {IMAGE_DIR} tidak ditemukan. Silakan jalankan scraper dan downloader terlebih dahulu.")
        sys.exit(1)
        
    random_images = sorted(list(IMAGE_DIR.glob("random_*.png")), key=lambda x: x.name)
    
    # Kelompokkan berkas gambar dokumen berdasarkan PDF asal
    doc_groups = {}
    for img_path in IMAGE_DIR.glob("doc_scraped_*_page_*.png"):
        m = re.match(r"doc_scraped_(\d+)_page_(\d+)\.png", img_path.name)
        if m:
            pdf_idx = int(m.group(1))
            page_num = int(m.group(2))
            doc_key = f"doc_scraped_{pdf_idx}"
            if doc_key not in doc_groups:
                doc_groups[doc_key] = []
            doc_groups[doc_key].append((page_num, img_path))
            
    # Urutkan halaman per PDF
    sorted_doc_groups = {}
    for doc_key, pages in doc_groups.items():
        pages.sort(key=lambda x: x[0])
        sorted_doc_groups[doc_key] = [x[1] for x in pages]
        
    # Load metadata KORIKA jika ada
    image_metadata = {}
    if METADATA_FILE.exists():
        try:
            with METADATA_FILE.open("r", encoding="utf-8") as f:
                image_metadata = json.load(f)
            print(f"[INFO] Berhasil memuat metadata gambar dari {METADATA_FILE.name}")
        except Exception as e:
            print(f"[WARN] Gagal membaca metadata file: {e}")
            
    print(f"=== Analisis Gambar di Direktori ===")
    print(f"Gambar Acak/Kebudayaan (random_*.png): {len(random_images)}")
    print(f"Kelompok Dokumen PDF (doc_scraped_*): {len(sorted_doc_groups)}")
    total_doc_pages = sum(len(pages) for pages in sorted_doc_groups.values())
    print(f"Total Halaman Dokumen: {total_doc_pages}")
    
    # 3. Buat antrean item yang akan diproses
    items_to_process = []
    
    # Tambahkan kelompok dokumen
    for doc_key, pages in sorted(sorted_doc_groups.items(), key=lambda x: int(x[0].split("_")[-1])):
        items_to_process.append((doc_key, pages, "document"))
        
    # Tambahkan gambar acak
    for img_path in random_images:
        items_to_process.append((img_path.name, [img_path], "random"))
        
    # Acak urutan item agar bervariasi antara dokumen dan gambar acak
    random.seed(42)
    random.shuffle(items_to_process)
    
    # 4. Lacak item yang sudah diproses sebelumnya (Resume check)
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
                            first_img_name = Path(img_paths[0]).name
                            if "doc_scraped" in first_img_name:
                                doc_key = first_img_name.split("_page_")[0]
                                processed_keys.add(doc_key)
                            else:
                                processed_keys.add(first_img_name)
                            total_existing += 1
                    except:
                        pass
        print(f"[INFO] Resume: Menemukan {total_existing} percakapan yang sudah selesai di {output_path.name}")
        
    # Saring item yang belum diproses
    remaining_items = [item for item in items_to_process if item[0] not in processed_keys]
    print(f"Sisa item yang perlu diproses: {len(remaining_items)} dari {len(items_to_process)}")
    
    if not remaining_items:
        print("[INFO] Semua percakapan sudah selesai dibuat!")
        return
        
    # 5. Loop generasi
    start_time = time.time()
    success_count = 0
    target_total = len(items_to_process)
    
    with output_path.open("a", encoding="utf-8") as f:
        for idx, (item_key, paths, category) in enumerate(remaining_items):
            current_id = total_existing + idx + 1
            print(f"\n[{current_id}/{target_total}] Memproses {item_key} ({category}) dengan {len(paths)} halaman...")
            
            relative_paths = [f"data/multimodal/images/{p.name}" for p in paths]
            
            messages = generate_chat_for_images(client, paths, args.model, image_metadata)
            if messages:
                record = {
                    "id": 200000 + current_id,
                    "images": relative_paths,
                    "messages": messages
                }
                f.write(json.dumps(record, ensure_ascii=False) + "\n")
                f.flush()
                success_count += 1
            else:
                print(f"  [!] Gagal memproses {item_key}. Dilewati.")
                
            time.sleep(1.5)
            
    elapsed = time.time() - start_time
    print(f"\n=== PROSES GENERASI SELESAI ===")
    print(f"Berhasil membuat {success_count} percakapan baru dalam {elapsed/60:.1f} menit.")

if __name__ == "__main__":
    main()
