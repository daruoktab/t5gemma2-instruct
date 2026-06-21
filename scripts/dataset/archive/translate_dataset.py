"""
Translate dataset EN → ID menggunakan translategemma-12b GGUF.
================================================================
Ambil source EN dari dataset yang sudah ada, translate input+output ke ID,
simpan sebagai dataset tambahan untuk di-merge ke v2_dataset.

Features:
- Resume support: skip baris yang sudah ditranslate (cache di JSON)
- Progress tracking dengan tqdm
- Batch save setiap N samples
- Dry-run mode untuk preview

Target sources yang ditranslate:
  - oasst1          (~8.9K EN)
  - aya_translated_dolly  (~9.8K EN)
  - aya_translated_flan_cot (~9.6K EN)
  Total estimasi: ~28K samples → menjadi dataset ID tambahan

Usage:
    conda activate ai
    python instruct/scripts/translate_dataset.py               # Full run
    python instruct/scripts/translate_dataset.py --dry-run     # Preview saja
    python instruct/scripts/translate_dataset.py --sources oasst1  # Satu source saja
    python instruct/scripts/translate_dataset.py --limit 100   # Test 100 pertama
"""
from __future__ import annotations

import argparse
import json
import os
import random
import sys
import time
from typing import Any, cast, Iterable

# ─── Config ────────────────────────────────────────────────────────────────────
GGUF_PATH    = "D:/Program Files/lmStudio models/mradermacher/translategemma-12b-it-GGUF/translategemma-12b-it.Q4_K_M.gguf"
N_GPU_LAYERS = 99       # Offload semua ke GPU jika muat, kurangi jika OOM
N_CTX        = 1024
N_BATCH      = 256

SOURCES_TO_TRANSLATE = ["oasst1", "aya_translated_dolly", "aya_translated_flan_cot"]
SAVE_EVERY           = 50    # Simpan progress setiap N samples

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR    = os.path.normpath(os.path.join(_SCRIPT_DIR, "..", ".."))
DATA_DIR    = os.path.join(ROOT_DIR, "data")
CACHE_PATH  = os.path.join(DATA_DIR, "translated_en2id_cache.json")   # Resume cache
OUTPUT_PATH = os.path.join(DATA_DIR, "translated_en2id.json")         # Output final


# ─── Arg parsing ───────────────────────────────────────────────────────────────
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--dry-run",  action="store_true", help="Preview tanpa translate")
    p.add_argument("--limit",    type=int, default=None, help="Max samples per source")
    p.add_argument("--sources",  nargs="+", default=None,
                   help=f"Sources (default: {SOURCES_TO_TRANSLATE})")
    p.add_argument("--gpu-layers", type=int, default=N_GPU_LAYERS,
                   help="Jumlah GPU layers untuk GGUF (0=CPU)")
    return p.parse_args()


# ─── Load GGUF model ───────────────────────────────────────────────────────────
def load_model(n_gpu_layers: int) -> Any:
    try:
        from llama_cpp import Llama
    except ImportError:
        print("ERROR: llama-cpp-python tidak ditemukan. Install: pip install llama-cpp-python")
        sys.exit(1)
    print(f"Loading model ({n_gpu_layers} GPU layers)...")
    llm = Llama(
        model_path=GGUF_PATH,
        n_ctx=N_CTX,
        n_batch=N_BATCH,
        n_gpu_layers=n_gpu_layers,
        verbose=False,
    )
    print("✅ Model loaded\n")
    return llm


# ─── Translate dengan cache ────────────────────────────────────────────────────
def translate_to_id(llm: Any, text: str, cache: dict[str, str]) -> str:
    """Translate teks EN ke ID. Gunakan cache jika sudah ada."""
    text = text.strip()
    if not text:
        return ""

    # Cek cache
    if text in cache:
        return cache[text]

    # Adaptive max_tokens: estimasi token input (~4 char/token EN), output ID ~30% lebih panjang
    # + buffer 32 token. Cap 1024 untuk teks sangat panjang.
    estimated_input_tokens = len(text) // 4
    max_tokens = max(32, min(1024, int(estimated_input_tokens * 1.3) + 32))

    # Format translategemma: lang codes di chat_template_kwargs, content plain string
    try:
        response = llm.create_chat_completion(
            messages=[
                {"role": "user", "content": text}
            ],
            chat_template_kwargs={
                "source_lang_code": "en",
                "target_lang_code": "id",
            },
            max_tokens=max_tokens,
            temperature=0.1,
            top_p=0.95,
        )
        result = str(response["choices"][0]["message"]["content"]).strip()
        cache[text] = result
        return result
    except Exception:
        pass

    # Fallback: raw Gemma completion — sudah verified bekerja
    prompt = (
        "<start_of_turn>user\n"
        "Translate the following text from English to Indonesian. "
        "Output only the translation, nothing else.\n\n"
        f"{text}<end_of_turn>\n"
        "<start_of_turn>model\n"
    )
    response2 = llm(
        prompt,
        max_tokens=max_tokens,
        temperature=0.1,
        stop=["<end_of_turn>", "\n\n\n"],
    )
    result = str(response2["choices"][0]["text"]).strip()
    cache[text] = result
    return result


# ─── Main ──────────────────────────────────────────────────────────────────────
def main() -> None:
    args = parse_args()
    sources = args.sources or SOURCES_TO_TRANSLATE

    # Load source dataset
    try:
        from datasets import load_from_disk, Dataset
    except ImportError:
        print("ERROR: datasets library tidak ditemukan.")
        sys.exit(1)

    # Coba v2 dulu, fallback ke v1
    ds_path = os.path.join(DATA_DIR, "v2_hf_dataset")
    if not os.path.exists(ds_path):
        ds_path = os.path.join(DATA_DIR, "final_hf_dataset")
    if not os.path.exists(ds_path):
        print(f"ERROR: Dataset tidak ditemukan di {ds_path}")
        sys.exit(1)

    print(f"Loading dataset dari: {ds_path}")
    _ds = load_from_disk(ds_path)
    assert isinstance(_ds, Dataset)

    # Filter hanya source yang EN dan ada di daftar
    en_samples: list[dict[str, str]] = [
        {"inputs": str(r["inputs"]), "targets": str(r["targets"]),
         "source": str(r["source"]), "language": str(r["language"])}
        for r in cast("Iterable[dict[str, Any]]", _ds)
        if r["source"] in sources and r.get("language") == "en"
    ]

    print(f"Ditemukan {len(en_samples)} samples EN dari sources: {sources}")

    # Per-source stats
    from collections import Counter
    src_cnt = Counter(s["source"] for s in en_samples)
    for src, cnt in src_cnt.most_common():
        print(f"  {src}: {cnt}")

    if args.limit:
        # Ambil --limit pertama per source (deterministik)
        limited: list[dict[str, str]] = []
        per_src: dict[str, list[dict[str, str]]] = {}
        for s in en_samples:
            per_src.setdefault(s["source"], []).append(s)
        for src, rows in per_src.items():
            limited.extend(rows[:args.limit])
        en_samples = limited
        print(f"  → Dibatasi ke {len(en_samples)} samples (--limit {args.limit})\n")

    if args.dry_run:
        print("\n[DRY RUN] Preview 5 samples pertama:")
        for i, s in enumerate(en_samples[:5], 1):
            print(f"\n  [{i}] {s['source']}")
            print(f"  IN : {s['inputs'][:120]}")
            print(f"  OUT: {s['targets'][:120]}")
        print(f"\n  Total yang akan ditranslate: {len(en_samples)} samples")
        print("  Estimasi waktu: {} jam (@ 2 detik/sample)".format(
            round(len(en_samples) * 2 / 3600, 1)
        ))
        return

    # Load GGUF model
    llm = load_model(args.gpu_layers)

    # Load cache (resume)
    cache: dict[str, str] = {}
    if os.path.exists(CACHE_PATH):
        with open(CACHE_PATH, "r", encoding="utf-8") as f:
            cache = json.load(f)
        print(f"Resume: {len(cache)} entri cache ditemukan")

    # Load hasil sebelumnya
    results: list[dict[str, str]] = []
    if os.path.exists(OUTPUT_PATH):
        with open(OUTPUT_PATH, "r", encoding="utf-8") as f:
            results = json.load(f)
        print(f"Resume: {len(results)} samples sudah ditranslate sebelumnya")

    # Set already-done berdasarkan original input
    done_inputs: set[str] = {r["original_input"] for r in results}

    # Filter yang belum dikerjakan
    todo = [s for s in en_samples if s["inputs"] not in done_inputs]
    print(f"\n{len(todo)} samples belum ditranslate, mulai...\n")

    # Cek tqdm
    tqdm_class = None
    try:
        from tqdm import tqdm as _tqdm
        tqdm_class = _tqdm
        use_tqdm = True
    except ImportError:
        use_tqdm = False

    iterator = tqdm_class(todo, desc="Translating", unit="sample") if (use_tqdm and tqdm_class is not None) else todo

    errors = 0
    t_start = time.time()

    for i, sample in enumerate(iterator, 1):
        try:
            trans_input  = translate_to_id(llm, sample["inputs"],  cache)
            trans_output = translate_to_id(llm, sample["targets"], cache)
        except Exception as e:
            errors += 1
            if not use_tqdm:
                print(f"  ERROR [{i}]: {e}")
            continue

        results.append({
            "inputs":   trans_input,
            "targets":  trans_output,
            "source":   sample["source"] + "_id",   # tag dengan _id
            "language": "id",
            # Simpan original untuk referensi
            "original_input":  sample["inputs"],
            "original_output": sample["targets"],
        })

        # Save progress
        if i % SAVE_EVERY == 0:
            with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
                json.dump(results, f, ensure_ascii=False)
            with open(CACHE_PATH, "w", encoding="utf-8") as f:
                json.dump(cache, f, ensure_ascii=False)
            elapsed = time.time() - t_start
            rate = i / elapsed if elapsed > 0 else 1
            eta_h = (len(todo) - i) / rate / 3600
            if not use_tqdm:
                print(f"  [{i}/{len(todo)}] saved. ETA: {eta_h:.1f}h")

    # Final save
    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False)
    with open(CACHE_PATH, "w", encoding="utf-8") as f:
        json.dump(cache, f, ensure_ascii=False)

    size_mb = os.path.getsize(OUTPUT_PATH) / 1024 / 1024
    print(f"\n✅ Selesai!")
    print(f"  Total hasil : {len(results)} samples")
    print(f"  Errors      : {errors}")
    print(f"  Output      : {OUTPUT_PATH} ({size_mb:.1f} MB)")
    print(f"\nSelanjutnya: jalankan build_dataset.py untuk merge ke v2_dataset")
    print("  Dataset ini akan otomatis ter-include jika ada di data/translated_en2id.json")


if __name__ == "__main__":
    main()
