import os
import json
import fitz

BASE = os.path.dirname(os.path.abspath(__file__))
PDF_DIR = os.path.join(BASE, "pdfs")

papers_info = {}

for f in os.listdir(PDF_DIR):
    if f.endswith(".pdf"):
        tag = f.split("_")[0]
        pdf_path = os.path.join(PDF_DIR, f)
        doc = fitz.open(pdf_path)
        
        full_text = []
        for page_num in range(len(doc)):
            page = doc[page_num]
            full_text.append(f"--- PAGE {page_num+1} ---\n" + page.get_text())
            
        doc.close()
        
        papers_info[tag] = {
            "filename": f,
            "total_pages": len(full_text),
            "full_text": "\n".join(full_text)
        }

# Save per paper text file for easy reading
out_text_dir = os.path.join(BASE, "extracted_texts")
os.makedirs(out_text_dir, exist_ok=True)

for tag, data in papers_info.items():
    tfile = os.path.join(out_text_dir, f"{tag}_text.txt")
    with open(tfile, "w", encoding="utf-8") as out:
        out.write(str(data["full_text"]))
    print(f"Extracted {tag}: {data['total_pages']} pages -> {tfile}")

print("\nDone extracting text from all PDFs.")
