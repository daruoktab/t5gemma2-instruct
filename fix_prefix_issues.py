"""
Audit & fix task-prefix issues di data multimodal T5Gemma2 instruct project.

Masalah 1: <unusedX> bocor ke pesan USER (harusnya cuma di ASSISTANT).
Masalah 2: pesan ASSISTANT tanpa <unusedX> sama sekali.

Cara pakai (jalankan dari root project D:\Codings\unsloth-porto\t5-gemma-2\instruct):
    python fix_prefix_issues.py --check
        -> hanya laporan, tidak menulis apapun.

    python fix_prefix_issues.py --fix
        -> menulis train_vision.fixed.jsonl (masalah 1 di-strip otomatis).
           Masalah 2 (7 kasus) TIDAK di-auto-fix karena butuh keputusan manual
           soal task type yang benar -> akan dicetak di akhir supaya kamu
           bisa perbaiki manual.

File asli TIDAK PERNAH ditimpa langsung, selalu tulis ke file *.fixed.jsonl
supaya kamu bisa diff & review dulu sebelum replace.
"""
import json
import re
import argparse
from pathlib import Path

PREFIX_RE = re.compile(r"<unused\d+>")

ROOT = Path(__file__).resolve().parent  # jalankan dari root project t5-gemma-2/instruct
TRAIN_VISION = ROOT / "data" / "multimodal" / "train_vision.jsonl"
ORPO_MULTIMODAL = ROOT / "data" / "preference" / "orpo_multimodal.jsonl"


def strip_leading_prefix(content: str) -> str:
    """Hapus <unusedX> berturut-turut di awal string (+ spasi setelahnya)."""
    while True:
        m = re.match(r"^\s*<unused\d+>\s*", content)
        if not m:
            break
        content = content[m.end():]
    return content


def audit_and_fix_train_vision(do_fix: bool):
    if not TRAIN_VISION.exists():
        print(f"[SKIP] {TRAIN_VISION} tidak ditemukan.")
        return

    total = 0
    user_prefix_issues = []
    assistant_missing = []
    fixed_records = []

    with TRAIN_VISION.open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            rec = json.loads(line)
            total += 1
            rec_id = rec.get("id")

            for i, m in enumerate(rec["messages"]):
                role, content = m.get("role"), m.get("content", "")
                prefixes = PREFIX_RE.findall(content)
                if role == "user" and prefixes:
                    user_prefix_issues.append((rec_id, i, content[:80]))
                    if do_fix:
                        m["content"] = strip_leading_prefix(content)
                if role == "assistant" and not prefixes:
                    assistant_missing.append((rec_id, i, content[:80]))

            fixed_records.append(rec)

    print(f"\n=== train_vision.jsonl ({total} records) ===")
    print(f"Masalah 1 (user bawa prefix): {len(user_prefix_issues)} kasus")
    print(f"Masalah 2 (assistant tanpa prefix): {len(assistant_missing)} kasus -> BUTUH REVIEW MANUAL:")
    for rec_id, i, preview in assistant_missing:
        print(f"    id={rec_id} turn[{i}]: {preview}")

    if do_fix:
        out_path = TRAIN_VISION.with_suffix(".fixed.jsonl")
        with out_path.open("w", encoding="utf-8") as f:
            for rec in fixed_records:
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
        print(f"[OK] Ditulis: {out_path} (masalah 1 sudah di-strip, masalah 2 masih perlu tangan manusia)")


def audit_orpo_multimodal():
    if not ORPO_MULTIMODAL.exists():
        print(f"[SKIP] {ORPO_MULTIMODAL} tidak ditemukan.")
        return

    total = 0
    prompt_user_prefix_issues = []
    chosen_missing = []
    rejected_missing = []

    with ORPO_MULTIMODAL.open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            rec = json.loads(line)
            total += 1
            rec_id = rec.get("id")
            prompt, chosen, rejected = rec.get("prompt", ""), rec.get("chosen", ""), rec.get("rejected", "")

            # parse role dari prompt (format "role: content" per baris awal segmen)
            current_role, buf, turns = None, [], []
            for l in prompt.split("\n"):
                m = re.match(r"^(system|user|assistant):\s?(.*)$", l)
                if m:
                    if current_role is not None:
                        turns.append((current_role, "\n".join(buf)))
                    current_role, buf = m.group(1), [m.group(2)]
                else:
                    buf.append(l)
            if current_role is not None:
                turns.append((current_role, "\n".join(buf)))

            for role, content in turns:
                if role == "user" and PREFIX_RE.findall(content):
                    prompt_user_prefix_issues.append((rec_id, content[:80]))

            if not PREFIX_RE.findall(chosen):
                chosen_missing.append((rec_id, chosen[:80]))
            if not PREFIX_RE.findall(rejected):
                rejected_missing.append((rec_id, rejected[:80]))

    print(f"\n=== orpo_multimodal.jsonl ({total} records) ===")
    print(f"Prompt user-turn bawa prefix: {len(prompt_user_prefix_issues)} kasus")
    for e in prompt_user_prefix_issues:
        print(f"    {e}")
    print(f"'chosen' tanpa prefix: {len(chosen_missing)} kasus (kemungkinan warisan dari train_vision.jsonl yang bermasalah -> cek sumbernya)")
    for e in chosen_missing:
        print(f"    {e}")
    print(f"'rejected' tanpa prefix: {len(rejected_missing)} kasus")
    for e in rejected_missing:
        print(f"    {e}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--fix", action="store_true", help="Tulis train_vision.fixed.jsonl dengan masalah 1 di-strip")
    parser.add_argument("--check", action="store_true", help="Hanya audit, tidak menulis file")
    args = parser.parse_args()

    do_fix = args.fix and not args.check
    audit_and_fix_train_vision(do_fix=do_fix)
    audit_orpo_multimodal()
