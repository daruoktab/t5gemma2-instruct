"""
Analisis statistik perbandingan dataset nested chat instruct:
  - t5-gemma-2-chat-instruct-dataset.jsonl (base)
  - t5-gemma-2-chat-instruct-dataset-extra8970.jsonl (extra)

Metrik: jumlah percakapan, pesan per percakapan, pasangan user–assistant,
token (tokenizer model), bandingkan field `tokens` / `num_turns` di JSON jika ada.

Contoh:
  python scripts/analyze_instruct_datasets.py
  python scripts/analyze_instruct_datasets.py --model google/t5gemma-2-4b-4b
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
from dataclasses import dataclass, field
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parents[1]


@dataclass
class ConvStats:
    n_messages: int = 0
    n_pairs: int = 0  # user+assistant rounds (setelah system)
    total_tokens: int = 0
    tokens_per_msg: list[int] = field(default_factory=list)
    json_num_turns: int | None = None
    json_tokens: int | None = None
    role_ok: bool = True


def count_tokens(tok, text: str) -> int:
    if not (text or "").strip():
        return 0
    return len(tok.encode(text, add_special_tokens=False))


def analyze_file(path: Path, tok) -> tuple[list[ConvStats], dict]:
    rows: list[ConvStats] = []
    bad_structure = 0
    with path.open(encoding="utf-8") as f:
        for line_no, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError as e:
                print(f"[WARN] {path.name}:{line_no} JSON: {e}", file=sys.stderr)
                continue
            conv = obj.get("conversations")
            if not isinstance(conv, list) or not conv:
                bad_structure += 1
                continue
            st = ConvStats()
            st.json_num_turns = obj.get("num_turns")
            st.json_tokens = obj.get("tokens")
            # Validasi struktur: [system], user, assistant, ...
            role_ok = True
            if conv[0].get("role") != "system":
                role_ok = False
            for i, m in enumerate(conv[1:], start=1):
                want = "user" if i % 2 == 1 else "assistant"
                if m.get("role") != want:
                    role_ok = False
                    break
            st.role_ok = role_ok
            if not role_ok:
                bad_structure += 1

            st.n_messages = len(conv)
            # Pasangan giliran: setelah system, (user, assistant) *
            st.n_pairs = max(0, (st.n_messages - 1) // 2)

            for m in conv:
                c = (m.get("content") or "")
                nt = count_tokens(tok, c)
                st.tokens_per_msg.append(nt)
                st.total_tokens += nt
            rows.append(st)

    return rows, {"bad_or_nonstandard": bad_structure, "lines_ok": len(rows)}


def summarize(name: str, rows: list[ConvStats], meta: dict) -> None:
    n = len(rows)
    if n == 0:
        print(f"\n=== {name} ===\n(tidak ada baris valid)\n")
        return

    msgs = [r.n_messages for r in rows]
    pairs = [r.n_pairs for r in rows]
    tot_tok = [r.total_tokens for r in rows]
    # rata token per pesan dalam satu conv (rata dari rata per conv)
    avg_msg_per_conv = [statistics.mean(r.tokens_per_msg) if r.tokens_per_msg else 0 for r in rows]

    def pct(xs: list, p: float) -> float:
        xs = sorted(xs)
        if not xs:
            return 0.0
        k = (len(xs) - 1) * p / 100.0
        f = int(k)
        c = min(f + 1, len(xs) - 1)
        return xs[f] + (k - f) * (xs[c] - xs[f]) if f != c else xs[f]

    print(f"\n{'=' * 72}")
    print(f"  {name}")
    print(f"{'=' * 72}")
    print(f"  Path: {meta.get('path')}")
    print(f"  Percakapan (baris valid):     {n}")
    if meta.get("bad_or_nonstandard"):
        print(f"  Baris struktur tidak standar: {meta['bad_or_nonstandard']}")

    print("\n  --- Jumlah pesan per percakapan (termasuk system) ---")
    print(f"    min / max:     {min(msgs)} / {max(msgs)}")
    print(f"    mean / median: {statistics.mean(msgs):.2f} / {statistics.median(msgs):.1f}")
    print(f"    stdev:         {statistics.stdev(msgs) if len(msgs) > 1 else 0:.2f}")
    print(f"    p95 / p99:     {pct(msgs, 95):.1f} / {pct(msgs, 99):.1f}")

    print("\n  --- Pasangan user→assistant (giliran dialog, tanpa system) ---")
    print(f"    min / max:     {min(pairs)} / {max(pairs)}")
    print(f"    mean / median: {statistics.mean(pairs):.2f} / {statistics.median(pairs):.1f}")
    print(f"    stdev:         {statistics.stdev(pairs) if len(pairs) > 1 else 0:.2f}")

    print("\n  --- Token per percakapan (jumlah semua isi pesan, tokenizer) ---")
    print(f"    min / max:     {min(tot_tok)} / {max(tot_tok)}")
    print(f"    mean / median: {statistics.mean(tot_tok):.1f} / {statistics.median(tot_tok):.1f}")
    print(f"    stdev:         {statistics.stdev(tot_tok) if len(tot_tok) > 1 else 0:.1f}")
    print(f"    p95 / p99:     {pct([float(x) for x in tot_tok], 95):.1f} / {pct([float(x) for x in tot_tok], 99):.1f}")

    print("\n  --- Rata-rata token per pesan (dalam satu conv, ratakan lalu ratakan lagi) ---")
    print(f"    mean of avg:   {statistics.mean(avg_msg_per_conv):.2f}")
    print(f"    median of avg: {statistics.median(avg_msg_per_conv):.2f}")

    # Token per "turn" = user+assistant pair: bagi total_tokens dengan n_pairs
    t_per_pair = []
    for r in rows:
        if r.n_pairs > 0:
            t_per_pair.append(r.total_tokens / r.n_pairs)
    if t_per_pair:
        print("\n  --- Token per pasangan user+assistant (total_tokens / n_pairs) ---")
        print(f"    mean / median: {statistics.mean(t_per_pair):.1f} / {statistics.median(t_per_pair):.1f}")
        print(f"    min / max:     {min(t_per_pair):.1f} / {max(t_per_pair):.1f}")

    # Bandingkan field JSON
    has_nt = [r for r in rows if r.json_num_turns is not None]
    has_jt = [r for r in rows if r.json_tokens is not None and r.json_tokens > 0]
    if has_nt:
        mism_nt = sum(1 for r in has_nt if r.json_num_turns != r.n_messages)
        print("\n  --- Field `num_turns` di JSON vs len(conversations) ---")
        print(f"    baris punya num_turns: {len(has_nt)}")
        if mism_nt:
            print(f"    mismatch num_turns != n_messages: {mism_nt}")
        else:
            print("    (num_turns == jumlah pesan termasuk system)")
    if has_jt:
        print("\n  --- Field `tokens` di JSON (hanya >0) vs tokenizer sum ---")
        diffs = []
        for r in has_jt:
            if r.json_tokens:
                diffs.append(r.total_tokens - r.json_tokens)
        print(f"    baris dengan tokens>0: {len(has_jt)}")
        if diffs:
            print(
                f"    selisih (tokenizer - json): mean={statistics.mean(diffs):.1f}, "
                f"median={statistics.median(diffs):.1f}"
            )


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument(
        "--model",
        default="google/t5gemma-2-270m-270m",
        help="Tokenizer Hugging Face (vocab T5-Gemma-2 sama keluarga)",
    )
    p.add_argument(
        "--base",
        type=Path,
        default=SCRIPT_DIR / "t5-gemma-2-chat-instruct-dataset.jsonl",
    )
    p.add_argument(
        "--extra",
        type=Path,
        default=SCRIPT_DIR / "t5-gemma-2-chat-instruct-dataset-extra8970.jsonl",
    )
    args = p.parse_args()

    try:
        from transformers import AutoTokenizer
    except ImportError:
        print("pip install transformers", file=sys.stderr)
        sys.exit(1)

    print(f"Loading tokenizer: {args.model}")
    tokenizer = AutoTokenizer.from_pretrained(args.model)

    for label, path in [("BASE", args.base), ("EXTRA", args.extra)]:
        if not path.exists():
            print(f"\n[SKIP] tidak ada: {path}")
            continue
        rows, meta = analyze_file(path, tokenizer)
        meta["path"] = str(path.resolve())
        summarize(label, rows, meta)

    print(f"\n{'=' * 72}\nSelesai.\n")


if __name__ == "__main__":
    main()
