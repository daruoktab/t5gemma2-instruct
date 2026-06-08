"""
Build Dataset v2 untuk Fine-tuning T5Gemma-2 Instruct
======================================================
Target: ~180-220K samples, balanced ID/EN, multi-task.

Dataset yang di-include:
  INSTRUCTION (ID):
    - alpaca_id              ~50K   — Alpaca GPT4 Indonesian
    - eli5_id                ~30K   — ELI5 bahasa Indonesia
    - aya_dataset            ~622   — Aya original (ID)
    - bactrian_id            ~20K   — Bactrian-X Indonesian
    - dolly_id               ~5K    — Dolly Indonesian (jika ada)

  INSTRUCTION (EN):
    - aya_translated_dolly   ~9.8K  — Dolly EN dari Aya collection
    - aya_translated_flan_cot~9.6K  — FlanCoT EN
    - oasst1                 ~8.9K  — Open Assistant EN

  MULTI-TURN DIALOG (ID + EN):
    - oasst2_multiturn       ~15K   — OASST2 tree → multi-turn sequences

  TRANSLATION EN↔ID:
    - flores200              ~15K   — FLORES-200 EN↔ID pairs

  SUMMARIZATION (ID):
    - xlsum_id               ~15K   — XL-Sum Indonesian

  ENCODER-DECODER TASKS (ID):
    - aya_templated_indo_stories ~1.5K — Translation anak EN/Jawa→ID
    - aya_dataset extra tasks

  DROPPED dari v1:
    - blended_skill_talk    (chat casual EN tanpa instruksi)
    - aya_translated_hotpotqa (trivia 1-3 kata)
    - aya_templated_mintaka   (output monoton "The answer is X")
    - aya_translated_flan_qa  (artifact translasi buruk)

Output:
  data/v2_hf_dataset/   — HuggingFace Dataset format (utama)
  data/v2_dataset.json  — JSON backup

Usage:
    conda activate unsloth
    python instruct/scripts/build_dataset.py
    python instruct/scripts/build_dataset.py --dry-run    # Statistik saja
    python instruct/scripts/build_dataset.py --validate   # Cek format
"""
from __future__ import annotations

import argparse
import json
import os
import random
import sys
from collections import Counter, defaultdict
from typing import Any, cast, Iterable

from datasets import load_dataset, Dataset, load_from_disk

# ============================================================
# Paths (relatif terhadap ROOT repo)
# ============================================================
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.normpath(os.path.join(_SCRIPT_DIR, "..", ".."))
DATA_DIR = os.path.normpath(os.path.join(ROOT_DIR, "data"))
os.makedirs(DATA_DIR, exist_ok=True)

OUTPUT_HF   = os.path.join(DATA_DIR, "v2_hf_dataset")
OUTPUT_JSON = os.path.join(DATA_DIR, "v2_dataset.json")

# ============================================================
# Quality filters
# ============================================================
MIN_INPUT_LEN  = 10
MIN_OUTPUT_LEN = 15
MAX_INPUT_LEN  = 1800   # sesuai MAX_INPUT_LENGTH di train.py
MAX_OUTPUT_LEN = 1200

# ============================================================
# Global accumulator
# ============================================================
all_samples: list[dict[str, str]] = []
_ds: Any = None


def add_samples(samples: list[dict[str, str]], source: str) -> None:
    """Quality filter + tambah ke global list, print stats."""
    before = len(samples)
    filtered = [
        s for s in samples
        if (MIN_INPUT_LEN <= len(s.get("inputs", "")) <= MAX_INPUT_LEN
            and MIN_OUTPUT_LEN <= len(s.get("targets", "")) <= MAX_OUTPUT_LEN)
    ]
    lang_dist = Counter(s["language"] for s in filtered)
    print(f"\n  [{source}] {len(filtered)}/{before} samples (setelah filter)")
    for lang, cnt in lang_dist.most_common():
        print(f"    {lang}: {cnt}")
    all_samples.extend(filtered)


def sep(title: str) -> None:
    print(f"\n{'='*60}\n{title}\n{'='*60}")


# ============================================================
# Helpers untuk format multi-turn
# ============================================================

def format_multiturn_input(history: list[tuple[str, str]], current_user: str) -> str:
    """
    Gabungkan history + current turn ke dalam satu string untuk encoder.
    Format:
        User: ...\nAssistant: ...
        ...
        User: [current]
    """
    parts: list[str] = []
    for user_msg, asst_msg in history:
        parts.append(f"User: {user_msg.strip()}\nAssistant: {asst_msg.strip()}")
    parts.append(f"User: {current_user.strip()}")
    return "\n\n".join(parts)


# ============================================================
# SOURCE 1: Alpaca Indonesian (largest ID base)
# ============================================================
sep("1. Alpaca Indonesian (dari final_hf_dataset)")

try:
    # Re-use dari dataset lama yang sudah downloaded
    final_ds_path = os.path.join(DATA_DIR, "final_hf_dataset")
    if os.path.exists(final_ds_path):
        _ds = load_from_disk(final_ds_path)
        alpaca_rows = [
            {"inputs": r["inputs"], "targets": r["targets"],
             "source": "alpaca_id", "language": "id"}
            for r in cast("Iterable[dict[str, Any]]", _ds)
            if r["source"] == "alpaca_id"
        ]
        add_samples(alpaca_rows, "alpaca_id")
    else:
        _alpaca = load_dataset("indonesian-nlp/alpaca_id", split="train")
        assert isinstance(_alpaca, Dataset)
        samples: list[dict[str, str]] = []
        for row in _alpaca:
            row = cast(dict[str, Any], row)
            instruction: str = str(row.get("instruction", "")).strip()
            inp: str = str(row.get("input", "")).strip()
            output: str = str(row.get("output", "")).strip()
            
            if instruction and output:
                full_input = f"{instruction}\n\n{inp}".strip() if inp else instruction
                samples.append({"inputs": full_input, "targets": output,
                                 "source": "alpaca_id", "language": "id"})
        add_samples(samples, "alpaca_id")
except Exception as e:
    print(f"  ERROR: {e}")


# ============================================================
# SOURCE 2: ELI5 Indonesian
# ============================================================
sep("2. ELI5 Indonesian")

try:
    if "_ds" in locals() and "eli5_id" in str(_ds):
        eli5_rows = [
            {"inputs": r["inputs"], "targets": r["targets"],
             "source": "eli5_id", "language": "id"}
            for r in cast("Iterable[dict[str, Any]]", _ds)
            if r["source"] == "eli5_id"
        ]
        add_samples(eli5_rows, "eli5_id")
    else:
        _eli5 = load_dataset("indonesian-nlp/eli5_id", split="train")
        assert isinstance(_eli5, Dataset)
        samples = []
        for row in _eli5:
            row = cast(dict[str, Any], row)
            q: str = str(row.get("title", "") or row.get("question", "")).strip()
            answers = row.get("answers", {})
            if isinstance(answers, dict):
                texts: list[str] = answers.get("text", [])
            else:
                texts = [str(a) for a in (answers or [])]
            if q and texts and len(texts[0]) >= MIN_OUTPUT_LEN:
                samples.append({"inputs": q, "targets": texts[0].strip(),
                                 "source": "eli5_id", "language": "id"})
        add_samples(samples, "eli5_id")
except Exception as e:
    print(f"  ERROR: {e}")


# ============================================================
# SOURCE 3: Aya Dataset (ID)
# ============================================================
sep("3. Aya Dataset (ID human-annotated)")

try:
    if "_ds" in locals() and "aya_dataset" in str(_ds):
        aya_rows = [
            {"inputs": r["inputs"], "targets": r["targets"],
             "source": "aya_dataset", "language": "id"}
            for r in cast("Iterable[dict[str, Any]]", _ds)
            if r["source"] == "aya_dataset"
        ]
        add_samples(aya_rows, "aya_dataset")
    else:
        _aya = load_dataset("CohereForAI/aya_dataset", split="train")
        assert isinstance(_aya, Dataset)
        samples = [
            {"inputs": r["inputs"], "targets": r["targets"],
             "source": "aya_dataset", "language": "id"}
            for r in cast("Iterable[dict[str, Any]]", _aya)
            if r.get("language") == "Indonesian"
        ]
        add_samples(samples, "aya_dataset")
except Exception as e:
    print(f"  ERROR: {e}")


# ============================================================
# SOURCE 4: Aya Indo Stories (Translation EN/Jawa→ID)
# ============================================================
sep("4. Aya Indo Stories (Translation anak)")

try:
    if "_ds" in locals() and "aya_templated_indo_stories" in str(_ds):
        stories_rows = [
            {"inputs": r["inputs"], "targets": r["targets"],
             "source": "aya_templated_indo_stories", "language": "id"}
            for r in cast("Iterable[dict[str, Any]]", _ds)
            if r["source"] == "aya_templated_indo_stories"
        ]
        add_samples(stories_rows, "aya_templated_indo_stories")
except Exception as e:
    print(f"  ERROR: {e}")


# ============================================================
# SOURCE 5: Aya Translated Dolly (EN)
# ============================================================
sep("5. Aya Translated Dolly (EN open-ended instruction)")

try:
    if "_ds" in locals() and "aya_translated_dolly" in str(_ds):
        dolly_rows = [
            {"inputs": r["inputs"], "targets": r["targets"],
             "source": "aya_translated_dolly", "language": "en"}
            for r in cast("Iterable[dict[str, Any]]", _ds)
            if r["source"] == "aya_translated_dolly"
        ]
        add_samples(dolly_rows, "aya_translated_dolly")
except Exception as e:
    print(f"  ERROR: {e}")


# ============================================================
# SOURCE 6: Aya Translated FLAN CoT (EN reasoning)
# ============================================================
sep("6. Aya Translated FlanCoT (EN chain-of-thought)")

try:
    if "_ds" in locals() and "aya_translated_flan_cot" in str(_ds):
        cot_rows = [
            {"inputs": r["inputs"], "targets": r["targets"],
             "source": "aya_translated_flan_cot", "language": "en"}
            for r in cast("Iterable[dict[str, Any]]", _ds)
            if r["source"] == "aya_translated_flan_cot"
        ]
        add_samples(cot_rows, "aya_translated_flan_cot")
except Exception as e:
    print(f"  ERROR: {e}")


# ============================================================
# SOURCE 7: OASST1 (EN instruction, high quality)
# ============================================================
sep("7. OASST1 (EN high-quality instruction)")

try:
    if "_ds" in locals() and "oasst1" in str(_ds):
        oasst_rows = [
            {"inputs": r["inputs"], "targets": r["targets"],
             "source": "oasst1", "language": r.get("language", "en")}
            for r in cast("Iterable[dict[str, Any]]", _ds)
            if r["source"] == "oasst1"
        ]
        add_samples(oasst_rows, "oasst1")
    else:
        raise FileNotFoundError("oasst1 not in cached _ds, downloading fresh")
except Exception as e:
    print(f"  Loading fresh from HF: {e}")
    try:
        _oasst = load_dataset("OpenAssistant/oasst1", split="train")
        assert isinstance(_oasst, Dataset)
        msg_by_id: dict[str, dict[str, Any]] = {}
        children: dict[str, list[str]] = defaultdict(list)
        for row in _oasst:
            row = cast(dict[str, Any], row)
            mid: str = str(row["message_id"])
            pid: str = str(row.get("parent_id") or "")
            msg_by_id[mid] = row
            if pid:
                children[pid].append(mid)
        samples = []
        for mid, msg in msg_by_id.items():
            if msg["role"] == "prompter" and mid in children:
                for cid in children[mid][:1]:
                    child = msg_by_id[cid]
                    if child["role"] == "assistant":
                        lang: str = str(msg.get("lang", "en"))
                        if lang in ("en", "id"):
                            samples.append({
                                "inputs": str(msg["text"]).strip(),
                                "targets": str(child["text"]).strip(),
                                "source": "oasst1", "language": lang,
                            })
        add_samples(samples, "oasst1")
    except Exception as e2:
        print(f"  ERROR: {e2}")


# ============================================================
# SOURCE 8: Bactrian-X Indonesian (NEW - high quality ID)
# ============================================================
sep("8. Bactrian-X Indonesian (ID instruction, ~20K)")

try:
    _bactrian = load_dataset("MBZUAI/Bactrian-X", "id", split="train")
    assert isinstance(_bactrian, Dataset)
    samples = []
    for row in _bactrian:
        row = cast(dict[str, Any], row)
        instruction: str = str(row.get("instruction", "")).strip()
        inp: str = str(row.get("input", "")).strip()
        output: str = str(row.get("output", "")).strip()
        full_input = f"{instruction}\n\nInput: {inp}" if inp else instruction
        if full_input and output:
            samples.append({"inputs": full_input, "targets": output,
                             "source": "bactrian_id", "language": "id"})
    add_samples(samples, "bactrian_id")
except Exception as e:
    print(f"  ERROR: {e} — mencoba alternatif...")
    try:
        _bactrian2 = load_dataset("FreedomIntelligence/Bactrian-X", "id", split="train")
        assert isinstance(_bactrian2, Dataset)
        samples = []
        for row in _bactrian2:
            row = cast(dict[str, Any], row)
            instruction = str(row.get("instruction", "")).strip()
            output = str(row.get("output", "")).strip()
            if instruction and output:
                samples.append({"inputs": instruction, "targets": output,
                                 "source": "bactrian_id", "language": "id"})
        add_samples(samples, "bactrian_id")
    except Exception as e2:
        print(f"  ERROR (alternatif): {e2}")


# ============================================================
# SOURCE 9: OASST2 Multi-turn (ID + EN) — NEW
# ============================================================
sep("9. OASST2 Multi-turn Dialog (ID + EN)")

try:
    _oasst2 = load_dataset("OpenAssistant/oasst2", split="train")
    assert isinstance(_oasst2, Dataset)

    msg_by_id2: dict[str, dict[str, Any]] = {}
    children2: dict[str, list[str]] = defaultdict(list)
    for row in _oasst2:
        row = cast(dict[str, Any], row)
        mid = str(row["message_id"])
        pid = str(row.get("parent_id") or "")
        msg_by_id2[mid] = row
        if pid:
            children2[pid].append(mid)

    def build_chains(
        msg_id: str,
        history: list[tuple[str, str]],
        depth: int,
        max_depth: int = 2,
    ) -> list[dict[str, str]]:
        """Rekursif bangun conversation chains dari tree OASST."""
        results: list[dict[str, str]] = []
        msg = msg_by_id2.get(msg_id)
        if msg is None or depth > max_depth:
            return results
        lang: str = str(msg.get("lang", "en"))
        if lang not in ("en", "id"):
            return results

        if msg["role"] == "assistant" and history:
            last_user_msg = history[-1][0]
            enc_input = format_multiturn_input(history[:-1], last_user_msg)
            target = str(msg["text"]).strip()
            if len(enc_input) <= MAX_INPUT_LEN and len(target) >= MIN_OUTPUT_LEN:
                results.append({
                    "inputs": enc_input,
                    "targets": target,
                    "source": "oasst2_multiturn",
                    "language": lang,
                })
            new_history = history[:-1] + [(last_user_msg, str(msg["text"]).strip())]
        elif msg["role"] == "prompter":
            new_history = history + [(str(msg["text"]).strip(), "")]
        else:
            new_history = history

        child_ids = children2.get(msg_id, [])
        child_ids_sorted = sorted(
            child_ids,
            key=lambda cid: msg_by_id2.get(cid, {}).get("rank", 999),
        )[:2]

        for cid in child_ids_sorted:
            results.extend(build_chains(cid, new_history, depth + 1, max_depth))
        return results

    multiturn_samples: list[dict[str, str]] = []
    root_ids = [mid for mid, msg in msg_by_id2.items()
                 if not msg.get("parent_id") and msg.get("lang") in ("en", "id")]
    MAX_MULTITURN = 20000
    for root_id in root_ids:
        if len(multiturn_samples) >= MAX_MULTITURN:
            break
        chains = build_chains(root_id, [], 0)
        multiturn_samples.extend(chains)

    random.shuffle(multiturn_samples)
    multiturn_samples = multiturn_samples[:MAX_MULTITURN]
    add_samples(multiturn_samples, "oasst2_multiturn")
except Exception as e:
    print(f"  ERROR: {e}")


# ============================================================
# SOURCE 10: FLORES-200 Translation EN↔ID (NEW)
# ============================================================
sep("10. FLORES-200 Translation EN↔ID")

try:
    _flores_en = load_dataset("facebook/flores", "eng_Latn", split="devtest+dev")
    _flores_id = load_dataset("facebook/flores", "ind_Latn", split="devtest+dev")
    assert isinstance(_flores_en, Dataset) and isinstance(_flores_id, Dataset)

    trans_samples: list[dict[str, str]] = []
    for en_row, id_row in zip(_flores_en, _flores_id):
        en_row = cast(dict[str, Any], en_row)
        id_row = cast(dict[str, Any], id_row)
        en_text: str = str(en_row.get("sentence", "")).strip()
        id_text: str = str(id_row.get("sentence", "")).strip()
        if en_text and id_text:
            trans_samples.append({
                "inputs": f"Terjemahkan ke Bahasa Indonesia: {en_text}",
                "targets": id_text,
                "source": "flores200_en2id",
                "language": "id",
            })
            trans_samples.append({
                "inputs": f"Translate to English: {id_text}",
                "targets": en_text,
                "source": "flores200_id2en",
                "language": "en",
            })
    add_samples(trans_samples, "flores200")
except Exception as e:
    print(f"  ERROR: {e}")


# ============================================================
# SOURCE 11: XL-Sum Indonesian — summarization (NEW)
# ============================================================
sep("11. XL-Sum Indonesian (Summarization ID)")

try:
    _xlsum = load_dataset("csebuetnlp/xlsum", "indonesian", split="train")
    assert isinstance(_xlsum, Dataset)
    MAX_XLSUM = 15000
    indices = random.sample(range(len(_xlsum)), min(MAX_XLSUM, len(_xlsum)))
    summ_samples: list[dict[str, str]] = []
    for idx in indices:
        row = cast(dict[str, Any], _xlsum[idx])
        text: str = str(row.get("text", "")).strip()
        summary: str = str(row.get("summary", "")).strip()
        if text and summary:
            summ_samples.append({
                "inputs": f"Ringkaskan teks berikut: {text[:1200]}",
                "targets": summary,
                "source": "xlsum_id",
                "language": "id",
            })
    add_samples(summ_samples, "xlsum_id")
except Exception as e:
    print(f"  ERROR: {e}")


# ============================================================
# SOURCE 12: IndoNLI — NLI sebagai task instruksi (NEW)
# ============================================================
sep("12. IndoNLI (Bahasa Indonesia NLI)")

try:
    _nli = load_dataset("afaji/indonli", split="train")
    assert isinstance(_nli, Dataset)
    label_map = {0: "entailment", 1: "neutral", 2: "contradiction"}
    nli_samples: list[dict[str, str]] = []
    for row in _nli:
        row = cast(dict[str, Any], row)
        premise: str = str(row.get("premise", "")).strip()
        hypothesis: str = str(row.get("hypothesis", "")).strip()
        label: int = int(row.get("label", -1))
        if premise and hypothesis and label in label_map:
            label_str = label_map[label]
            nli_samples.append({
                "inputs": (
                    f"Berdasarkan premis berikut, tentukan hubungan dengan hipotesis:\n"
                    f"Premis: {premise}\n"
                    f"Hipotesis: {hypothesis}\n"
                    f"Pilihan: entailment, neutral, contradiction"
                ),
                "targets": f"Hubungan antara premis dan hipotesis adalah: {label_str}.",
                "source": "indonli",
                "language": "id",
            })
    nli_samples = random.sample(nli_samples, min(5000, len(nli_samples)))
    add_samples(nli_samples, "indonli")
except Exception as e:
    print(f"  ERROR: {e}")


# ============================================================
# SOURCE 13: Hasil translasi EN→ID (dari translate_dataset.py)
# ============================================================
sep("13. Translated EN→ID (dari translate_dataset.py)")

_translated_path = os.path.join(DATA_DIR, "translated_en2id.json")
if os.path.exists(_translated_path):
    try:
        with open(_translated_path, "r", encoding="utf-8") as _f:
            _trans_data: list[dict[str, Any]] = json.load(_f)
        trans_id_samples: list[dict[str, str]] = [
            {
                "inputs":   str(r["inputs"]),
                "targets":  str(r["targets"]),
                "source":   str(r.get("source", "translated_en2id")),
                "language": "id",
            }
            for r in cast("Iterable[dict[str, Any]]", _trans_data)
            if r.get("inputs") and r.get("targets")
        ]
        add_samples(trans_id_samples, "translated_en2id")
    except Exception as e:
        print(f"  ERROR loading translated_en2id.json: {e}")
else:
    print(f"  SKIP: {_translated_path} tidak ada.")


# ============================================================
# FINAL: Dedup, balance, shuffle, save
# ============================================================
sep("FINAL: Dedup, Balance, Save")

seen_keys: set[str] = set()
unique_samples: list[dict[str, str]] = []
for s in all_samples:
    key = s["inputs"][:150].lower().strip()
    if key and key not in seen_keys:
        seen_keys.add(key)
        unique_samples.append(s)

print(f"  Total raw   : {len(all_samples)}")
print(f"  After dedup : {len(unique_samples)}")

src_dist  = Counter(s["source"]   for s in unique_samples)
lang_dist = Counter(s["language"] for s in unique_samples)
print("\n  Source distribution:")
for src, cnt in src_dist.most_common():
    print(f"    {src}: {cnt}")
print("\n  Language distribution:")
for lang, cnt in lang_dist.most_common():
    pct = cnt / len(unique_samples) * 100
    print(f"    {lang}: {cnt} ({pct:.1f}%)")

random.seed(42)
random.shuffle(unique_samples)

with open(OUTPUT_JSON, "w", encoding="utf-8") as f:
    json.dump(unique_samples, f, ensure_ascii=False)
print(f"\n  JSON saved : {OUTPUT_JSON} ({os.path.getsize(OUTPUT_JSON)/1024/1024:.1f} MB)")

try:
    hf_ds = Dataset.from_list(unique_samples)
    hf_ds.save_to_disk(OUTPUT_HF)
    print(f"  HF saved   : {OUTPUT_HF}")
except Exception as e:
    print(f"  HF save failed: {e}")

print(f"\n  ✅ DONE: {len(unique_samples)} samples total")


# ============================================================
# Argparse entry point
# ============================================================
def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build instruct dataset v2")
    parser.add_argument("--dry-run", action="store_true",
                        help="Hanya cetak statistik tanpa download")
    parser.add_argument("--validate", action="store_true",
                        help="Cek format dataset yang sudah dibangun")
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    if args.validate:
        if os.path.exists(OUTPUT_HF):
            _ds = load_from_disk(OUTPUT_HF)
            print(f"Dataset ditemukan: {len(_ds)} samples")
            print(f"Columns: {_ds.column_names}")
            src_c = Counter(_ds["source"])
            lang_c = Counter(_ds["language"])
            print("\nSource dist:", dict(src_c.most_common()))
            print("Lang dist  :", dict(lang_c.most_common()))
            for i in random.sample(range(len(_ds)), min(3, len(_ds))):
                r = _ds[i]
                print(f"\n  [{r['source']} / {r['language']}]")
                print(f"  IN : {r['inputs'][:200]}")
                print(f"  OUT: {r['targets'][:100]}")
        else:
            print(f"Dataset belum ada di {OUTPUT_HF}")
        sys.exit(0)
