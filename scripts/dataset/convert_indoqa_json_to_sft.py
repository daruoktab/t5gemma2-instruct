"""
Konversi IndoQA (JSON list) ke format SFT (JSONL input/target).
Digunakan untuk df_train.json -> indoqa_train.jsonl.

Jalankan dari folder instruct:
    python scripts/dataset/convert_indoqa_json_to_sft.py --input data/df_train.json --output data/indoqa_train.jsonl
"""

import argparse
import json
import sys
from pathlib import Path

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--input", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()

    if not args.input.is_file():
        print(f"[ERR] tidak ada: {args.input}", file=sys.stderr)
        sys.exit(1)

    with args.input.open("r", encoding="utf-8") as f:
        data = json.load(f)

    if not isinstance(data, list):
        print("[ERR] input harus berupa list of objects", file=sys.stderr)
        sys.exit(1)

    system_prompt = "Kamu adalah asisten AI yang ahli dalam menganalisis dokumen."
    
    n = 0
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as f:
        for item in data:
            context = item.get("context", "")
            question = item.get("question", "")
            answer = item.get("answer", "")
            
            if not context or not question:
                continue
            
            # Jika answer null (UNANSWERABLE), kita bisa tangani atau skip.
            # Berdasarkan indoqa_train.jsonl sebelumnya, sepertinya kita pakai answer apa adanya.
            if answer is None:
                answer = "Maaf, saya tidak dapat menemukan jawaban untuk pertanyaan tersebut dalam teks yang diberikan."
                
            inp = (
                f"system: {system_prompt}\n"
                f"user: Jawablah pertanyaan berikut berdasarkan konteks yang tersedia.\n\n"
                f"Konteks: {context}\n\n"
                f"Pertanyaan: {question}"
            )
            
            out_obj = {"input": inp, "target": answer}
            f.write(json.dumps(out_obj, ensure_ascii=False) + "\n")
            n += 1
            
    print(f"[OK] {len(data)} items -> {n} rows -> {args.output}")

if __name__ == "__main__":
    main()
