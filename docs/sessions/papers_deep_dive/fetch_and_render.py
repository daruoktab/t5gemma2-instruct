"""Download 3 priority arXiv PDFs and render pages to PNG.

Papers:
  - TLPO  : 2604.26553
  - UL2R  : 2210.11399
  - INTRA : 2605.05806
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
    "2604.26553": "TLPO",
    "2210.11399": "UL2R",
    "2605.05806": "INTRA",
}

HEADERS = {"User-Agent": "unsloth-porto-research/1.0 (contact: local)"}


def download_pdf(arxiv_id: str, tag: str) -> str:
    out = os.path.join(PDF_DIR, f"{tag}_{arxiv_id}.pdf")
    if os.path.exists(out) and os.path.getsize(out) > 50_000:
        print(f"[skip] {out} already exists ({os.path.getsize(out)} bytes)")
        return out
    url = f"https://arxiv.org/pdf/{arxiv_id}"
    print(f"[dl] {url} -> {out}")
    req = urllib.request.Request(url, headers=HEADERS)
    with urllib.request.urlopen(req, timeout=120) as r, open(out, "wb") as f:
        while True:
            chunk = r.read(1 << 16)
            if not chunk:
                break
            f.write(chunk)
    print(f"[ok] {out} ({os.path.getsize(out)} bytes)")
    return out


def render_pdf(pdf_path: str, tag: str, dpi: int = 130) -> list[str]:
    outdir = os.path.join(PAGE_DIR, tag)
    os.makedirs(outdir, exist_ok=True)
    doc = fitz.open(pdf_path)
    paths = []
    zoom = dpi / 72.0
    mat = fitz.Matrix(zoom, zoom)
    for i, page in enumerate(doc):
        out = os.path.join(outdir, f"p{i+1:02d}.png")
        if not os.path.exists(out):
            pix = page.get_pixmap(matrix=mat, alpha=False)
            pix.save(out)
        paths.append(out)
    print(f"[render] {tag}: {len(paths)} pages -> {outdir}")
    doc.close()
    return paths


def main():
    for arxiv_id, tag in PAPERS.items():
        try:
            pdf = download_pdf(arxiv_id, tag)
            time.sleep(3)  # be polite to arxiv
            render_pdf(pdf, tag)
        except Exception as e:
            print(f"[ERR] {tag} ({arxiv_id}): {e}", file=sys.stderr)


if __name__ == "__main__":
    main()
