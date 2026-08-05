"""Download all 26 verified arXiv PDFs and render pages to PNG.
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

# 26 Verified arXiv Papers
PAPERS_26 = {
    # Category A: Internal Training (10)
    "2604.26553": "TLPO",
    "2607.16169": "MUON_RL",
    "2605.07815": "ORSCALE",
    "2512.22511": "TV_DECOMP",
    "2605.03780": "TV_GEOM",
    "2607.16821": "PAIRWISE_FRAGILE",
    "2606.10929": "LOCAL_LINEAR",
    "2601.12639": "FT_OBJECTIVES_SAFETY",
    "2606.09850": "ALIGNMENT_MECHANISTIC",
    "2607.21016": "CULTURE_ID",

    # Category B: Strategi & Komposisi (8)
    "2512.14856": "T5GEMMA2",
    "2606.24841": "MTO",
    "2603.17512": "XBRIDGE",
    "2606.30336": "FLEXTAB",
    "2604.01760": "PM_ROPE_TTS",
    "2604.11687": "STYLE_TRANSFER",
    "2607.06613": "DOMAIN_ADAPTATION",
    "2512.10561": "CAUSAL_ENCODERS",

    # Category C: Ide Baru Terapan (8)
    "2605.05806": "INTRA",
    "2210.11399": "UL2R",
    "2606.20911": "LATENT_MEMORY",
    "2308.07269": "EASYEDIT",
    "2607.25583": "LORA_TRADEOFFS",
    "2603.04759": "STACKED_CONTEXT",
    "2605.26558": "CASSANDRA_SPECULATIVE",
    "2607.21356": "EMERGENT_MISALIGNMENT",
}

HEADERS = {"User-Agent": "unsloth-porto-research/3.0 (contact: local)"}


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


def render_pdf(pdf_path: str, tag: str, dpi: int = 130) -> list:
    outdir = os.path.join(PAGE_DIR, tag)
    os.makedirs(outdir, exist_ok=True)
    doc = fitz.open(pdf_path)
    paths = []
    zoom = dpi / 72.0
    mat = fitz.Matrix(zoom, zoom)
    for i in range(len(doc)):
        page = doc[i]
        out_img = os.path.join(outdir, f"p{i+1:02d}.png")
        if not os.path.exists(out_img):
            pix = page.get_pixmap(matrix=mat, alpha=False)
            pix.save(out_img)
        paths.append(out_img)
    doc.close()
    print(f"[rendered] {tag}: {len(paths)} pages -> {outdir}")
    return paths


def main():
    for arxiv_id, tag in PAPERS_26.items():
        try:
            pdf = download_pdf(arxiv_id, tag)
            time.sleep(1.5)
            render_pdf(pdf, tag)
        except Exception as e:
            print(f"[ERR] {tag} ({arxiv_id}): {e}", file=sys.stderr)


if __name__ == "__main__":
    main()
