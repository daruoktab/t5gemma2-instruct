"""
[Phase 1.3] Fix & Validasi Dataset — Bersihkan Empty Targets & Anomali
=======================================================================
Script utilitas untuk:
  1. Hapus/perbaiki baris dengan target kosong atau sangat pendek
  2. Hapus exact duplicate rows
  3. Buat laporan kualitas singkat
  4. Rebuild file yang sudah bersih

Tidak memerlukan API key — pure data cleaning.

Contoh:
  conda activate unsloth
  python scripts/dataset/fix_dataset_quality.py --all          # proses semua dataset
  python scripts/dataset/fix_dataset_quality.py --file data/indoqa_train.jsonl
  python scripts/dataset/fix_dataset_quality.py --dry-run      # hanya laporan
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

# ─── Paths ───────────────────────────────────────────────────────────────────
SCRIPT_DIR = Path(__file__).resolve().parent
ROOT_DIR   = SCRIPT_DIR.parent.parent
DATA_DIR   = ROOT_DIR / "data"

ALL_DATASETS = [
    DATA_DIR / "chat_train.jsonl",
    DATA_DIR / "chat_val.jsonl",
    DATA_DIR / "indoqa_train.jsonl",
    DATA_DIR / "indoqa_val.jsonl",
    DATA_DIR / "indoqa_train_augmented.jsonl",
    DATA_DIR / "indoqa_val_augmented.jsonl",
]

# ─── Helpers ─────────────────────────────────────────────────────────────────

def load_jsonl(path: Path) -> list[dict]:
    rows = []
    if not path.exists():
        return rows
    with path.open("r", encoding="utf-8") as f:
        for i, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as e:
                print(f"  [WARN] Baris {i} malformed JSON: {e}")
    return rows


def save_jsonl(path: Path, rows: list[dict], overwrite: bool = False) -> None:
    if path.exists() and not overwrite:
        backup = path.with_suffix(".jsonl.bak")
        path.rename(backup)
        print(f"  [INFO] Backup tersimpan: {backup}")
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def count_words(text: str) -> int:
    return len((text or "").split())


def analyze_dataset(path: Path, rows: list[dict]) -> dict:
    """Analisis kualitas dataset."""
    total = len(rows)
    empty_targets   = [i for i, r in enumerate(rows) if not r.get("target", "").strip()]
    empty_inputs    = [i for i, r in enumerate(rows) if not r.get("input", "").strip()]
    very_short_tgt  = [i for i, r in enumerate(rows) if 0 < count_words(r.get("target", "")) < 3]
    very_short_inp  = [i for i, r in enumerate(rows) if 0 < count_words(r.get("input", "")) < 10]

    # Dedup by exact (input + target)
    seen: set[str] = set()
    exact_dups: list[int] = []
    for i, r in enumerate(rows):
        key = (r.get("input", "") + "|||" + r.get("target", ""))[:300]
        if key in seen:
            exact_dups.append(i)
        else:
            seen.add(key)

    # Dedup by input only (same input, different target)
    input_counts: Counter[str] = Counter(r.get("input", "")[:200] for r in rows)
    dup_input_count = sum(1 for c in input_counts.values() if c > 1)

    target_words = [count_words(r.get("target", "")) for r in rows]
    input_words  = [count_words(r.get("input", "")) for r in rows]

    return {
        "path": str(path),
        "total": total,
        "size_mb": path.stat().st_size / 1024 / 1024 if path.exists() else 0,
        "empty_targets": len(empty_targets),
        "empty_inputs": len(empty_inputs),
        "very_short_targets": len(very_short_tgt),
        "very_short_inputs": len(very_short_inp),
        "exact_duplicates": len(exact_dups),
        "duplicate_inputs": dup_input_count,
        "avg_target_words": sum(target_words) / max(total, 1),
        "avg_input_words":  sum(input_words)  / max(total, 1),
        "min_target_words": min(target_words) if target_words else 0,
        "max_target_words": max(target_words) if target_words else 0,
        "_empty_target_indices":   empty_targets,
        "_empty_input_indices":    empty_inputs,
        "_exact_dup_indices":      exact_dups,
    }


def print_report(info: dict) -> None:
    print(f"\n{'─' * 60}")
    print(f"  📄 {Path(info['path']).name}  ({info['size_mb']:.2f} MB, {info['total']:,} rows)")
    print(f"{'─' * 60}")
    print(f"  Empty targets        : {info['empty_targets']}")
    print(f"  Empty inputs         : {info['empty_inputs']}")
    print(f"  Very short targets   : {info['very_short_targets']}  (<3 kata)")
    print(f"  Exact duplicate rows : {info['exact_duplicates']}")
    print(f"  Duplicate inputs     : {info['duplicate_inputs']}  (input sama, target beda)")
    print(f"  Avg input  words     : {info['avg_input_words']:.1f}")
    print(f"  Avg target words     : {info['avg_target_words']:.1f}  (min={info['min_target_words']}, max={info['max_target_words']})")


def clean_dataset(rows: list[dict], info: dict) -> tuple[list[dict], int]:
    """Bersihkan dataset: hapus empty target + exact duplicates."""
    to_remove = set(info["_empty_target_indices"]) | set(info["_exact_dup_indices"])

    cleaned = []
    removed_count = 0
    for i, row in enumerate(rows):
        if i in to_remove:
            removed_count += 1
            continue
        # Strip whitespace dari target & input
        row = dict(row)
        row["input"]  = (row.get("input", "") or "").strip()
        row["target"] = (row.get("target", "") or "").strip()
        cleaned.append(row)

    return cleaned, removed_count


# ─── Main ────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description="Fix & validasi kualitas dataset")
    parser.add_argument("--all", action="store_true",
                        help="Proses semua dataset standar")
    parser.add_argument("--file", type=Path, default=None,
                        help="Path ke file JSONL tertentu")
    parser.add_argument("--dry-run", action="store_true",
                        help="Hanya laporan, tidak menulis file")
    parser.add_argument("--no-backup", action="store_true",
                        help="Timpa langsung tanpa backup")
    args = parser.parse_args()

    if not args.all and args.file is None:
        parser.error("Harus pilih --all atau --file <path>")

    targets: list[Path] = []
    if args.all:
        targets = [p for p in ALL_DATASETS if p.exists()]
    else:
        targets = [args.file]  # type: ignore[list-item]

    if not targets:
        print("[WARN] Tidak ada file yang ditemukan.")
        return

    print(f"\n{'='*60}")
    print(f"  Dataset Quality Fix & Validation")
    print(f"  Mode: {'DRY-RUN' if args.dry_run else 'WRITE'}")
    print(f"{'='*60}")

    total_removed = 0
    for path in targets:
        if not path.exists():
            print(f"[SKIP] Tidak ada: {path}")
            continue

        rows = load_jsonl(path)
        info = analyze_dataset(path, rows)
        print_report(info)

        if args.dry_run:
            continue

        if info["empty_targets"] == 0 and info["exact_duplicates"] == 0:
            print("  ✓ Bersih, tidak ada yang perlu dihapus.")
            continue

        cleaned, n_removed = clean_dataset(rows, info)
        total_removed += n_removed

        print(f"\n  ➜ Menghapus {n_removed} baris (empty/duplikat)...")
        save_jsonl(path, cleaned, overwrite=args.no_backup)
        print(f"  ✓ Tersimpan: {path.name} ({len(cleaned):,} baris)")

    if not args.dry_run:
        print(f"\n{'='*60}")
        print(f"  Total baris dihapus: {total_removed}")
        print(f"{'='*60}")


if __name__ == "__main__":
    main()
