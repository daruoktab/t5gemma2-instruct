"""Uji offline pipeline EXTRA v2 (TANPA API): sampling random-access, skema, kuota, prompt."""
import json
import os
import sys
import tempfile

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "dataset"))
import generate_conv_api as G  # type: ignore  # noqa: E402  (module dimuat via sys.path runtime)


def ok(label, cond):
    print(f"{'✅' if cond else '❌'} {label}")
    return cond


def main():
    all_ok = True

    # 1) Sampling random-access (rows source, seed deterministik)
    spec = next(s for s in G.SOURCES if s["key"] == "IndoCareer")
    s = G._sample_from_spec(spec, seed=7)
    all_ok &= ok(f"sample IndoCareer (seed=7) -> {s['source']} | ctx {len(s['raw_context'])} char", s["source"].endswith("Row #5305"))
    print("    raw_context:", s["raw_context"][:120].replace("\n", " | "))

    # 2) Skema strict: 1 pasang ditolak, 3 pasang lolos, user ber-prefix ditolak
    def conv(n_pairs, user_pfx=False):
        t = []
        for i in range(n_pairs):
            t.append({"role": "user", "content": f"Pertanyaan {i} yang cukup panjang dan jelas.", "prefixes": ["<unused4>"] if (user_pfx and i == 0) else []})
            t.append({"role": "assistant", "content": f"Jawaban {i}: penjelasan panjang dan mendetail dengan analogi dan kesimpulan.", "prefixes": ["<unused4>", "<unused6>"] if i == 0 else ["<unused4>"]})
        return t

    try:
        G.ConversationOutput(conversations=conv(1))
        all_ok &= ok("1 pasang DITOLAK", False)
    except ValueError:
        all_ok &= ok("1 pasang DITOLAK", True)
    G.ConversationOutput(conversations=conv(3))
    all_ok &= ok("3 pasang LOLOS", True)
    try:
        G.ConversationOutput(conversations=conv(3, user_pfx=True))
        all_ok &= ok("user ber-prefix DITOLAK", False)
    except ValueError:
        all_ok &= ok("user ber-prefix DITOLAK", True)

    # 3) Kuota & distribusi (GenState)
    tmp = tempfile.mkdtemp(prefix="pipeline_offline_")
    out = os.path.join(tmp, "out.jsonl")
    state = G.GenState(G.Path(out), 2000, 1000)
    cat = G.asyncio.get_event_loop().run_until_complete(state.next_category())
    all_ok &= ok(f"next_category awal -> {cat} (harus text)", cat == "text")
    all_ok &= ok(f"suggest_num_pairs -> {state.suggest_num_pairs()} (harus 3)", state.suggest_num_pairs() == 3)
    all_ok &= ok("prefix_hint awal ada", "Belum ada" in state.prefix_hint())

    # record 1 percakapan text 3 pasang → next harus vision
    rec = {"category": "text_nlu_chat", "num_pairs": 3, "messages": [
        {"role": "system", "content": "x"},
        {"role": "user", "content": "Q1"},
        {"role": "assistant", "content": "<unused4> A1 yang cukup panjang dan mendetail sesuai ketentuan."},
        {"role": "user", "content": "Q2"},
        {"role": "assistant", "content": "<unused4> A2 yang cukup panjang dan mendetail sesuai ketentuan."},
        {"role": "user", "content": "Q3"},
        {"role": "assistant", "content": "<unused4> A3 yang cukup panjang dan mendetail sesuai ketentuan."},
    ]}
    state.record(rec)
    cat2 = G.asyncio.get_event_loop().run_until_complete(state.next_category())
    all_ok &= ok(f"next_category setelah 1 text -> {cat2} (harus vision)", cat2 == "vision")
    all_ok &= ok(f"prefix_counts -> {state.prefix_counts}", state.prefix_counts["<unused4>"] == 3)
    all_ok &= ok(f"pairs_counts -> {state.pairs_counts}", state.pairs_counts["3"] == 1)
    all_ok &= ok("pairs_hint menyebut jarang", "JARANG" in state.pairs_hint())

    # 4) Prompt builder memuat semua aturan
    p = G.build_prompt_for_conversation(spec, s["raw_context"], s["source"], num_pairs=4,
                                        prefix_hint="hint", pairs_hint="pairs", is_vision=False)
    for kw in ["GROUNDING", "GAYA TURN USER", "DISTRIBUSI PREFIX", "VARIASI JUMLAH PASANG", "8 pesan"]:
        all_ok &= ok(f"prompt memuat '{kw}'", kw in p)

    # 5) Format akhir: strip token bocor di tengah
    good = G.ConversationOutput(conversations=[
        {"role": "user", "content": "Pertanyaan yang cukup panjang dan jelas."},
        {"role": "assistant", "content": "<unused4> <unused4> Jawaban panjang dengan <unused2> token bocor di tengah.", "prefixes": ["<unused4>"]},
        {"role": "user", "content": "Pertanyaan lanjutan yang cukup panjang dan jelas."},
        {"role": "assistant", "content": "Jawaban lanjutan yang panjang dan mendetail sesuai ketentuan.", "prefixes": ["<unused4>", "<unused1>"]},
        {"role": "user", "content": "Pertanyaan penutup yang cukup panjang dan jelas."},
        {"role": "assistant", "content": "Jawaban penutup yang panjang dan mendetail sesuai ketentuan.", "prefixes": ["<unused4>"]},
    ])
    formatted = G._format_final(good, is_vision=False)
    a1 = [m["content"] for m in formatted if m["role"] == "assistant"][0]
    all_ok &= ok(f"strip token global: {a1[:70]!r}", a1.startswith("<unused4> ") and "<unused2>" not in a1)

    # 6) Sub-agent review: ReviewReport & build_review_prompt
    rep = G.ReviewReport(approved=False, summary="ada masalah", issues=[
        G.ReviewIssue(turn_index=1, severity="error", problem="prefix tidak sesuai", suggestion="ganti <unused4> jadi <unused2>")])
    all_ok &= ok("ReviewReport valid", rep.approved is False and len(rep.issues) == 1)
    draft = G.ConversationOutput(conversations=conv(3))
    rp = G.build_review_prompt(spec, s["raw_context"], s["source"], draft.conversations, 3, is_vision=False, min_len=60)
    for kw in ["CHECKLIST", "DRAFT PERCAKAPAN", "Prefix turn", "GROUNDING KETAT", "KEDALAMAN & REPETISI", "approved"]:
        all_ok &= ok(f"review prompt memuat '{kw}'", kw in rp)
    draft_txt = G._render_draft(draft.conversations)
    all_ok &= ok("_render_draft ada USER & ASSISTANT", "USER" in draft_txt and "ASSISTANT" in draft_txt)

    print("\n" + ("✅ SEMUA UJI OFFLINE LULUS" if all_ok else "❌ ADA YANG GAGAL"))
    sys.exit(0 if all_ok else 1)


if __name__ == "__main__":
    main()
