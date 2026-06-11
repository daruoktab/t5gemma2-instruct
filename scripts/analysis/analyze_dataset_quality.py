import os
import json
import re
from collections import Counter

# Paths to dataset files
DATA_DIR = "data"
FILES = {
    "Chat Train": os.path.join(DATA_DIR, "chat_train.jsonl"),
    "Chat Val": os.path.join(DATA_DIR, "chat_val.jsonl"),
    "IndoQA Train": os.path.join(DATA_DIR, "indoqa_train.jsonl"),
    "IndoQA Val": os.path.join(DATA_DIR, "indoqa_val.jsonl")
}

INDONESIAN_QUESTION_WORDS = [
    r"\bapa\b", r"\bsiapa\b", r"\bbagaimana\b", r"\bdi\s*mana\b", 
    r"\bkapan\b", r"\bmengapa\b", r"\bkenapa\b", r"\bberapa\b",
    r"\bjelaskan\b", r"\bsebutkan\b", r"\btuliskan\b", r"\bapakah\b",
    r"\bsiapakah\b", r"\bbagaimanakah\b", r"\bberapakah\b"
]

def load_and_analyze_file(name, path):
    if not os.path.exists(path):
        return {
            "exists": False,
            "error": f"File {path} not found"
        }
    
    file_size_mb = os.path.getsize(path) / (1024 * 1024)
    total_rows = 0
    malformed_rows = 0
    empty_input = 0
    empty_target = 0
    
    inputs = []
    targets = []
    
    input_char_lens = []
    target_char_lens = []
    input_word_lens = []
    target_word_lens = []
    
    # Track duplicates within this file
    raw_pairs = []
    unique_inputs = set()
    dup_inputs_count = 0
    
    question_word_counter = Counter()
    
    with open(path, "r", encoding="utf-8") as f:
        for idx, line in enumerate(f, 1):
            if not line.strip():
                continue
            total_rows += 1
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                malformed_rows += 1
                continue
                
            inp = obj.get("input", "")
            tgt = obj.get("target", "")
            
            # Clean up inputs/targets for counting
            if not inp or not str(inp).strip():
                empty_input += 1
            if not tgt or not str(tgt).strip():
                empty_target += 1
                
            inp_str = str(inp).strip()
            tgt_str = str(tgt).strip()
            
            inputs.append(inp_str)
            targets.append(tgt_str)
            raw_pairs.append((inp_str, tgt_str))
            
            if inp_str in unique_inputs:
                dup_inputs_count += 1
            else:
                unique_inputs.add(inp_str)
            
            # Length stats
            input_char_lens.append(len(inp_str))
            target_char_lens.append(len(tgt_str))
            
            inp_words = inp_str.split()
            tgt_words = tgt_str.split()
            input_word_lens.append(len(inp_words))
            target_word_lens.append(len(tgt_words))
            
            # Analyze query intent (first 100 chars of input to search for question words)
            inp_lower = inp_str[:200].lower()
            for pattern in INDONESIAN_QUESTION_WORDS:
                if re.search(pattern, inp_lower):
                    word_clean = pattern.replace(r"\b", "").replace(r"\s*", " ")
                    question_word_counter[word_clean] += 1
                    
    # Calculate stats
    def get_stats(lens):
        if not lens:
            return {"min": 0, "max": 0, "mean": 0, "median": 0}
        lens_sorted = sorted(lens)
        n = len(lens_sorted)
        median = lens_sorted[n // 2] if n % 2 != 0 else (lens_sorted[n // 2 - 1] + lens_sorted[n // 2]) / 2
        return {
            "min": min(lens),
            "max": max(lens),
            "mean": sum(lens) / len(lens),
            "median": median
        }
        
    unique_pairs = set(raw_pairs)
    total_duplicates = len(raw_pairs) - len(unique_pairs)
    
    return {
        "exists": True,
        "file_size_mb": file_size_mb,
        "total_rows": total_rows,
        "malformed_rows": malformed_rows,
        "empty_input": empty_input,
        "empty_target": empty_target,
        "input_char": get_stats(input_char_lens),
        "target_char": get_stats(target_char_lens),
        "input_word": get_stats(input_word_lens),
        "target_word": get_stats(target_word_lens),
        "duplicate_rows": total_duplicates,
        "duplicate_inputs": dup_inputs_count,
        "question_words": question_word_counter,
        "inputs_set": unique_inputs,
        "pairs_set": unique_pairs,
        "raw_pairs": raw_pairs
    }

def analyze_data_leakage(train_res, val_res):
    if not train_res.get("exists") or not val_res.get("exists"):
        return None
        
    train_inputs = train_res["inputs_set"]
    val_inputs = val_res["inputs_set"]
    
    leak_inputs = val_inputs.intersection(train_inputs)
    
    # Exact row leakage (both prompt and target are identical)
    train_pairs = train_res["pairs_set"]
    val_pairs = val_res["pairs_set"]
    leak_pairs = val_pairs.intersection(train_pairs)
    
    return {
        "input_leak_count": len(leak_inputs),
        "input_leak_percentage": (len(leak_inputs) / len(val_inputs) * 100) if val_inputs else 0,
        "exact_row_leak_count": len(leak_pairs),
        "exact_row_leak_percentage": (len(leak_pairs) / len(val_pairs) * 100) if val_pairs else 0,
        "sample_leaks": list(leak_inputs)[:3]  # show up to 3 leaked prompt samples
    }

def main():
    results = {}
    print("=" * 60)
    print("     STARTING COMPREHENSIVE DATASET QUALITY ANALYSIS")
    print("=" * 60)
    
    for name, path in FILES.items():
        print(f"Analyzing {name} from: {path}...")
        results[name] = load_and_analyze_file(name, path)
        
    report = []
    report.append("# Laporan Analisis Kualitas Dataset (SFT & Validation)\n")
    
    # 1. Basic Stats Table
    report.append("## 1. Statistik Dasar Dataset")
    report.append("| Dataset | File Size (MB) | Total Rows | Malformed Rows | Duplicate Rows (Exact) | Duplicate Inputs (Different Targets) | Empty Input/Target |")
    report.append("|---|---|---|---|---|---|---|")
    
    for name in FILES.keys():
        res = results[name]
        if not res["exists"]:
            report.append(f"| {name} | *File tidak ditemukan* | - | - | - | - | - |")
            continue
        report.append(
            f"| {name} | {res['file_size_mb']:.2f} MB | {res['total_rows']:,} | {res['malformed_rows']} | {res['duplicate_rows']:,} | {res['duplicate_inputs']:,} | {res['empty_input']}/{res['empty_target']} |"
        )
    report.append("")
    
    # 2. Length Statistics (Word level is cleaner)
    report.append("## 2. Analisis Panjang Teks (Jumlah Kata)")
    report.append("| Dataset | Input (Min/Mean/Median/Max) | Target (Min/Mean/Median/Max) | Rasio Target-to-Input |")
    report.append("|---|---|---|---|")
    
    for name in FILES.keys():
        res = results[name]
        if not res["exists"]:
            continue
        in_w = res["input_word"]
        tg_w = res["target_word"]
        ratio = tg_w["mean"] / in_w["mean"] if in_w["mean"] > 0 else 0
        report.append(
            f"| {name} | {in_w['min']} / {in_w['mean']:.1f} / {in_w['median']:.0f} / {in_w['max']} | {tg_w['min']} / {tg_w['mean']:.1f} / {tg_w['median']:.0f} / {tg_w['max']} | {ratio:.2f}x |"
        )
    report.append("")
    
    # 3. Data Leakage Analysis (Train vs Val)
    report.append("## 3. Analisis Kebocoran Data (Data Leakage)")
    report.append("Data leakage terjadi jika prompt evaluasi (validation) juga terdapat pada training set. Ini membuat evaluasi kurang valid karena model hanya 'menghafal'.\n")
    
    # Chat Leakage
    chat_leak = analyze_data_leakage(results["Chat Train"], results["Chat Val"])
    if chat_leak:
        report.append("### Chat Dataset Leakage:")
        report.append(f"- **Prompt Leakage:** {chat_leak['input_leak_count']} dari {len(results['Chat Val']['inputs_set'])} prompt validasi ada di training set ({chat_leak['input_leak_percentage']:.2f}%)")
        report.append(f"- **Exact Row Leakage:** {chat_leak['exact_row_leak_count']} baris ada secara identik di train dan val ({chat_leak['exact_row_leak_percentage']:.2f}%)")
        if chat_leak['sample_leaks']:
            report.append("- **Contoh Prompt yang Bocor:**")
            for leak in chat_leak['sample_leaks']:
                # Truncate sample
                leak_trunc = leak[:120] + "..." if len(leak) > 120 else leak
                # Replace newlines with space to make it look clean
                leak_trunc = leak_trunc.replace("\n", " ")
                report.append(f"  * *\"{leak_trunc}\"*")
        report.append("")
        
    # IndoQA Leakage
    qa_leak = analyze_data_leakage(results["IndoQA Train"], results["IndoQA Val"])
    if qa_leak:
        report.append("### IndoQA Dataset Leakage:")
        report.append(f"- **Prompt Leakage:** {qa_leak['input_leak_count']} dari {len(results['IndoQA Val']['inputs_set'])} prompt validasi ada di training set ({qa_leak['input_leak_percentage']:.2f}%)")
        report.append(f"- **Exact Row Leakage:** {qa_leak['exact_row_leak_count']} baris ada secara identik di train dan val ({qa_leak['exact_row_leak_percentage']:.2f}%)")
        if qa_leak['sample_leaks']:
            report.append("- **Contoh Prompt yang Bocor:**")
            for leak in qa_leak['sample_leaks']:
                leak_trunc = leak[:120] + "..." if len(leak) > 120 else leak
                leak_trunc = leak_trunc.replace("\n", " ")
                report.append(f"  * *\"{leak_trunc}\"*")
        report.append("")

    # 4. Intent & Question Types (Indonesia)
    report.append("## 4. Distribusi Kata Tanya (Intent Analysis)")
    report.append("Menunjukkan variasi dan sebaran tipe instruksi/pertanyaan dalam Bahasa Indonesia:\n")
    
    for name in FILES.keys():
        res = results[name]
        if not res["exists"] or not res["question_words"]:
            continue
        report.append(f"### Tipe Pertanyaan di {name}:")
        top_words = res["question_words"].most_common(8)
        word_strs = [f"**{w}**: {c} ({c/res['total_rows']*100:.1f}%)" for w, c in top_words]
        report.append(", ".join(word_strs))
        report.append("")
        
    # 5. Conclusions & Quality Assessment
    report.append("## 5. Kesimpulan Kualitas Dataset & Temuan Kunci")
    
    # Build findings
    findings = []
    
    # Duplicate findings
    for name in FILES.keys():
        res = results[name]
        if not res["exists"]:
            continue
        if res["duplicate_rows"] > 0:
            findings.append(f"- ⚠️ **{name}** memiliki **{res['duplicate_rows']:,} baris duplikat persis**. Duplikat ini sebaiknya dibuang (*deduplicated*) karena dapat menyebabkan model overfitting pada pola yang sama.")
        if res["duplicate_inputs"] > res["duplicate_rows"]:
            diff_tgt = res["duplicate_inputs"] - res["duplicate_rows"]
            findings.append(f"- ℹ️ **{name}** memiliki **{diff_tgt:,} prompt yang sama tetapi dengan target jawaban berbeda**. Ini wajar untuk skenario percakapan terbuka, tetapi perlu dipantau jika berupa pertanyaan faktual.")
            
    # Leakage findings
    if chat_leak and chat_leak["input_leak_count"] > 0:
        findings.append(f"- ⚠️ **Data Leakage pada Chat Dataset**: Ada **{chat_leak['input_leak_percentage']:.2f}%** data bocor dari training ke validation set. Ini akan membuat evaluasi metrik (loss, ROUGE, BLEU) terlihat 'terlalu bagus' secara artifisial.")
    if qa_leak and qa_leak["input_leak_count"] > 0:
        findings.append(f"- ⚠️ **Data Leakage pada IndoQA Dataset**: Ada **{qa_leak['input_leak_percentage']:.2f}%** data bocor. Mengingat IndoQA adalah QA berbasis dokumen, kebocoran dokumen/konteks yang persis sama membuat validasi kehilangan esensi pengujian generalisasi.")
        
    # Length findings
    for name in ["Chat Train", "IndoQA Train"]:
        res = results[name]
        if not res["exists"]:
            continue
        max_w = res["input_word"]["max"]
        if max_w > 2000:
            findings.append(f"- ℹ️ **{name}** memiliki prompt sangat panjang (maksimal **{max_w} kata**). Pastikan batas `MAX_SOURCE_LENGTH` tidak memotong informasi penting.")
            
    if not findings:
        findings.append("- ✅ **Kualitas dataset sangat bersih!** Tidak ditemukan duplikasi berarti, kebocoran data, atau masalah integritas data.")
        
    report.extend(findings)
    report.append("")
    
    # Write report to markdown file in docs
    report_text = "\n".join(report)
    output_report_path = os.path.join("docs", "dataset_quality_report.md")
    os.makedirs("docs", exist_ok=True)
    with open(output_report_path, "w", encoding="utf-8") as f:
        f.write(report_text)
        
    print(f"\n✅ Analisis selesai! Laporan lengkap disimpan ke: {output_report_path}")
    print("\n--- RINGKASAN TEMUAN UTAMA ---")
    for line in findings:
        print(line)
    print("------------------------------")

if __name__ == "__main__":
    main()
