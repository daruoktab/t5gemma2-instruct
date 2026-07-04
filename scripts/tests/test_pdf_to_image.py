import os
import sys

def convert_pdf_to_images(pdf_path, output_dir, dpi=150):
    print(f"Mengonversi PDF: {pdf_path}")
    print(f"Output folder: {output_dir}")
    os.makedirs(output_dir, exist_ok=True)
    
    try:
        import fitz  # PyMuPDF
        print("Menggunakan PyMuPDF (fitz)...")
        doc = fitz.open(pdf_path)
        print(f"Total halaman: {len(doc)}")
        
        saved_files = []
        for page_num in range(len(doc)):
            page = doc.load_page(page_num)
            pix = page.get_pixmap(dpi=dpi)
            
            output_filename = f"page_{page_num + 1}.png"
            output_path = os.path.join(output_dir, output_filename)
            pix.save(output_path)
            
            print(f"  [OK] Halaman {page_num + 1} disimpan ke {output_path}")
            saved_files.append(output_path)
            
        print("Konversi selesai sukses dengan PyMuPDF!")
        return saved_files
        
    except ImportError:
        print("PyMuPDF (fitz) tidak terdeteksi. Mencoba alternatif pdf2image...")
        try:
            from pdf2image import convert_from_path
            print("Menggunakan pdf2image...")
            # pdf2image membutuhkan poppler terinstal di sistem
            pages = convert_from_path(pdf_path, dpi=dpi)
            print(f"Total halaman: {len(pages)}")
            
            saved_files = []
            for i, page in enumerate(pages):
                output_filename = f"page_{i + 1}.png"
                output_path = os.path.join(output_dir, output_filename)
                page.save(output_path, "PNG")
                print(f"  [OK] Halaman {i + 1} disimpan ke {output_path}")
                saved_files.append(output_path)
                
            print("Konversi selesai sukses dengan pdf2image!")
            return saved_files
        except ImportError:
            print("ERROR: Kedua library fitz (PyMuPDF) dan pdf2image tidak terinstal.")
            print("Silakan jalankan:")
            print("  uv pip install pymupdf")
            sys.exit(1)
        except Exception as e:
            print(f"ERROR saat menggunakan pdf2image: {e}")
            print("Catatan: pdf2image membutuhkan 'poppler' diinstal di path sistem Anda.")
            sys.exit(1)
    except Exception as e:
        print(f"ERROR saat menggunakan PyMuPDF: {e}")
        sys.exit(1)

if __name__ == "__main__":
    # Mencari PDF sampel di docs/paper/
    paper_dir = os.path.join(os.path.dirname(__file__), "..", "docs", "paper")
    pdf_files = [f for f in os.listdir(paper_dir) if f.endswith(".pdf")] if os.path.exists(paper_dir) else []
    
    if not pdf_files:
        # Cari file pdf apa saja di docs/
        docs_dir = os.path.join(os.path.dirname(__file__), "..", "docs")
        pdf_files = [f for f in os.listdir(docs_dir) if f.endswith(".pdf")] if os.path.exists(docs_dir) else []
        if pdf_files:
            pdf_path = os.path.join(docs_dir, pdf_files[0])
        else:
            print("Tidak ditemukan file PDF untuk diuji di docs/ atau docs/paper/")
            sys.exit(1)
    else:
        # Gunakan salah satu paper, misalnya paper T5Gemma-2
        t5gemma_paper = [f for f in pdf_files if "T5Gemma_2" in f]
        if t5gemma_paper:
            pdf_path = os.path.join(paper_dir, t5gemma_paper[0])
        else:
            pdf_path = os.path.join(paper_dir, pdf_files[0])
            
    output_dir = os.path.join(os.path.dirname(__file__), "output_images")
    convert_pdf_to_images(pdf_path, output_dir, dpi=150)
