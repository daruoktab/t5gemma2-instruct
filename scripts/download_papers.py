"""Download paper-paper riset yang relevan ke folder docs/paper.
Jalankan sekali dari folder instruct:
  python scripts/download_papers.py
"""
import urllib.request
import os
import time

OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "..", "docs", "paper")
os.makedirs(OUTPUT_DIR, exist_ok=True)

PAPERS = [
    # Task Arithmetic & Model Merging
    ("2212.04089", "Task_Arithmetic_Ilharco2023"),
    ("2502.20186", "Layer_Aware_Task_Arithmetic_2025"),
    ("2505.12021", "Cross_Model_Transfer_Task_Vectors_2025"),

    # Encoder-Decoder Adaptation
    ("2504.06225", "Encoder_Decoder_Gemma_2025"),
    ("2503.02656", "Gemma_Encoder_2025"),
    ("1907.12461", "Warm_Starting_Enc_Dec_Rothe2019"),
    ("2501.16273", "Return_of_the_Encoder_Microsoft_2025"),

    # T5 / Instruction Tuning
    ("1910.10683", "T5_Raffel2020"),
    ("2210.11416", "FLAN_Instruction_Scaling_Chung2022"),
]

def download(arxiv_id: str, name: str):
    filename = f"{arxiv_id}_{name}.pdf"
    dest = os.path.join(OUTPUT_DIR, filename)

    if os.path.exists(dest) and os.path.getsize(dest) > 10_000:
        print(f"  [SKIP] {filename} sudah ada ({os.path.getsize(dest)//1024} KB)")
        return

    url = f"https://arxiv.org/pdf/{arxiv_id}"
    print(f"  Downloading {arxiv_id} -> {filename} ...")
    try:
        headers = {"User-Agent": "Mozilla/5.0 (research download)"}
        req = urllib.request.Request(url, headers=headers)
        with urllib.request.urlopen(req, timeout=60) as r, open(dest, "wb") as f:
            f.write(r.read())
        size_kb = os.path.getsize(dest) // 1024
        print(f"  -> OK ({size_kb} KB)")
    except Exception as e:
        print(f"  -> GAGAL: {e}")
    time.sleep(1.5)  # jangan spam arxiv

if __name__ == "__main__":
    print(f"Output dir: {os.path.abspath(OUTPUT_DIR)}")
    print(f"Downloading {len(PAPERS)} papers...\n")
    for arxiv_id, name in PAPERS:
        download(arxiv_id, name)
    print("\nSelesai.")
