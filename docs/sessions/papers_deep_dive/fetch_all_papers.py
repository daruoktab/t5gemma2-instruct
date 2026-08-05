"""Download priority arXiv PDFs and render pages to PNG for deep-dive analysis.

Paper Selection:
  1. 2512.14856: T5Gemma 2 (DeepMind Base Paper) -> T5GEMMA2
  2. 2210.11399: UL2R (Transcend Scaling Laws) -> UL2R
  3. 2604.26553: TLPO (Token-Level Policy Optimization) -> TLPO
  4. 2605.05806: INTRA (Intrinsic Retrieval) -> INTRA
  5. 2605.03780: Task Vector Geometry & Inference -> TV_GEOM
  6. 2512.22511: Decomposing Task Vectors -> TV_DECOMP
  7. 2605.07815: OrScale (Trust-Ratio Scaling) -> ORSCALE
  8. 2607.21016: CultureTalk-ID (Indonesian Commonsense) -> CULTURE_ID
  9. 2606.24841: MTO (Matching Tasks to Objectives) -> MTO
 10. 2603.17512: XBridge (Cross-Model Enc-Dec Bridges) -> XBRIDGE
 11. 2607.25583: LoRA Rank / Modules / Quant Trade-offs -> LORA_TRADEOFFS
"""
import os
import sys
import time
import urllib.request
import fitz  # PyMuPDF

BASE = os.path.dirname(os.path.abspath(__file__))
PDF_DIR = os.path.join(BASE, "pdfs")
PAGE_DIR = os.path.join(BASE, "pages")
os.makedirs(PDF_DIR, exist_ok=True)
os.makedirs(PAGE_DIR, exist_ok=True)

PAPERS = {
    "2512.14856": "T5GEMMA2",
    "2210.11399": "UL2R",
    "2604.26553": "TLPO",
    "2605.05806": "INTRA",
    "2605.03780": "TV_GEOM",
    "2512.22511": "TV_DECOMP",
    "2605.07815": "ORSCALE",
    "2607.21016": "CULTURE_ID",
    "2606.24841": "MTO",
    "2603.17512": "XBRIDGE",
    "2607.25583": "LORA_TRADEOFFS",
}

HEADERS = {"User-Agent": "unsloth-porto-research/2.0 (contact: local)"}


def download_pdf(arxiv_id: str, tag: str) -> str:
    out = os.path.join(PDF_DIR, f"{tag}_{arxiv_id}.pdf")
    if os.path.exists(out) and os.path.getsize(out) > 50_000:
        print(f"[skip] {out} exists ({os.path.getsize(out)} bytes)")
        return out
    url = f"https://arxiv.org/pdf/{arxiv_id}"
    print(f"[downloading] {url} -> {out}")
    req = urllib.request.Request(url, headers=HEADERS)
    with urllib.request.urlopen(req, timeout=120) as r, open(out, "wb") as f:
        while True:
            chunk = r.read(1 << 16)
            if not chunk:
                break
            f.write(chunk)
    print(f"[done] {out} ({os.path.getsize(out)} bytes)")
    return out


def render_and_extract(pdf_path: str, tag: str, dpi: int = 130) -> dict:
    outdir = os.path.join(PAGE_DIR, tag)
    os.makedirs(outdir, exist_ok=True)
    doc = fitz.open(pdf_path)
    paths = []
    text_by_page = []
    zoom = dpi / 72.0
    mat = fitz.Matrix(zoom, zoom)
    
    for i in range(len(doc)):
        page = doc[i]
        out_img = os.path.join(outdir, f"p{i+1:02d}.png")
        if not os.path.exists(out_img):
            pix = page.get_pixmap(matrix=mat, alpha=False)
            pix.save(out_img)
        paths.append(out_img)
        text_by_page.append(page.get_text())
        
    doc.close()
    print(f"[rendered] {tag}: {len(paths)} pages -> {outdir}")
    return {"tag": tag, "pages": len(paths), "text": text_by_page, "img_paths": paths}


def main():
    extracted = {}
    for arxiv_id, tag in PAPERS.items():
        try:
            pdf = download_pdf(arxiv_id, tag)
            time.sleep(2)
            res = render_and_extract(pdf, tag)
            extracted[tag] = res
        except Exception as e:
            print(f"[ERR] {tag} ({arxiv_id}): {e}", file=sys.stderr)
            
    summary_path = os.path.join(BASE, "downloaded_papers_summary.json")
    import json
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump({k: {"tag": v["tag"], "pages": v["pages"], "pdf": f"{k}.pdf"} for k, v in extracted.items()}, f, indent=2)
    print(f"\nSaved summary of downloaded papers to {summary_path}")

if __name__ == "__main__":
    main()
