import os
import sys
import json
import fitz  # PyMuPDF

def prepare_pdf_multimodal_dataset(pdf_dir, image_output_dir, jsonl_output_path, dpi=150):
    """
    Mengonversi semua PDF di pdf_dir menjadi gambar halaman, lalu membuat berkas JSONL
    yang terformat untuk training Vision-Language Model (VLM) menggunakan Unsloth.
    """
    print("=== Memulai Persiapan Dataset Multimodal ===")
    print(f"Direktori PDF: {pdf_dir}")
    print(f"Output Gambar: {image_output_dir}")
    print(f"Output JSONL: {jsonl_output_path}")
    
    os.makedirs(image_output_dir, exist_ok=True)
    os.makedirs(os.path.dirname(jsonl_output_path), exist_ok=True)
    
    if not os.path.exists(pdf_dir):
        print(f"ERROR: Direktori {pdf_dir} tidak ditemukan.")
        sys.exit(1)
        
    pdf_files = [f for f in os.listdir(pdf_dir) if f.endswith(".pdf")]
    if not pdf_files:
        print("Tidak ada file PDF yang ditemukan.")
        return
        
    dataset_records = []
    
    for pdf_file in pdf_files:
        pdf_path = os.path.join(pdf_dir, pdf_file)
        pdf_name = os.path.splitext(pdf_file)[0]
        print(f"\nMemproses: {pdf_file}")
        
        try:
            doc = fitz.open(pdf_path)
            num_pages = len(doc)
            print(f"-> Total halaman: {num_pages}")
            
            for page_num in range(num_pages):
                page = doc.load_page(page_num)
                pix = page.get_pixmap(dpi=dpi)
                
                # Simpan gambar per halaman
                image_name = f"{pdf_name}_page_{page_num + 1}.png"
                image_path = os.path.join(image_output_dir, image_name)
                pix.save(image_path)
                
                # Buat contoh instruksi visual (QA)
                # Catatan: T5Gemma-2 / Gemma 3 menggunakan token '📷' sebagai awal gambar
                prompt = f"📷\nTolong jelaskan konten dari dokumen halaman {page_num + 1} ini."
                target = f"Ini adalah halaman {page_num + 1} dari dokumen '{pdf_file}'. Halaman ini berisi informasi ilmiah/teknis mengenai riset."
                
                record = {
                    "messages": [
                        {"role": "user", "content": prompt},
                        {"role": "assistant", "content": target}
                    ],
                    "images": [image_path] # Unsloth membaca path gambar lokal/absolut
                }
                dataset_records.append(record)
                
            print(f"-> Selesai mengonversi {pdf_file} ({num_pages} halaman)")
        except Exception as e:
            print(f"-> GAGAL memproses {pdf_file}: {e}")
            
    # Tulis hasil ke JSONL
    try:
        with open(jsonl_output_path, "w", encoding="utf-8") as f:
            for record in dataset_records:
                f.write(json.dumps(record, ensure_ascii=False) + "\n")
        print(f"\n=== SUKSES! Berhasil menulis {len(dataset_records)} record ke {jsonl_output_path} ===")
    except Exception as e:
        print(f"ERROR saat menulis file JSONL: {e}")

if __name__ == "__main__":
    # Path default
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    pdf_dir = os.path.join(base_dir, "docs", "paper")
    image_output_dir = os.path.join(base_dir, "data", "multimodal", "images")
    jsonl_output_path = os.path.join(base_dir, "data", "multimodal", "train_vision.jsonl")
    
    prepare_pdf_multimodal_dataset(pdf_dir, image_output_dir, jsonl_output_path, dpi=150)
