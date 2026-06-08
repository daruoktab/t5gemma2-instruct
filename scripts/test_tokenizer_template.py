"""
Test Tokenizer & Chat Template Behavior
=======================================
Bandingkan:
  1. Gemma 3 270M PT  (base, decoder-only)
  2. Gemma 3 270M IT  (instruct, decoder-only)
  3. T5Gemma2 270M    (base, encoder-decoder)

Fokus:
  - Token IDs untuk special tokens yang kita pakai
  - Chat template format masing-masing
  - Bagaimana model menghasilkan output (tanpa SFT)
  - Apakah tokenizer IT vs T5Gemma2 menghasilkan ID berbeda untuk teks yang sama
"""

import torch
import os
from typing import cast, Any
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    AutoModelForSeq2SeqLM,
    PreTrainedTokenizerFast,
    GenerationMixin,
)
from transformers.models.auto.tokenization_auto import AutoTokenizer as _AT

# ─────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────
G3_PT  = "google/gemma-3-270m"
G3_IT  = "google/gemma-3-270m-it"
T5G    = "google/t5gemma-2-270m-270m"

TASK_VECTOR_MODEL = "models/t5gemma2-270m-task-vector"  # hasil cangkok kita

SPECIAL_TOKENS_TO_CHECK = [
    "<start_of_turn>",
    "<end_of_turn>",
    "<eos>",
    "<bos>",
    "<pad>",
    "<unk>",
    "<image_soft_token>",
    "<start_of_image>",
    "<end_of_image>",
]

TEST_TEXT = "Halo, siapa kamu?"
SYSTEM_PROMPT = (
    "Kamu adalah asisten AI yang helpful, santai, dan ramah. "
    "Gunakan Bahasa Indonesia sebagai bahasa utama."
)


def section(title: str) -> None:
    print(f"\n{'='*70}")
    print(f"  {title}")
    print("=" * 70)


def subsection(title: str) -> None:
    print(f"\n  {'─'*60}")
    print(f"  {title}")
    print(f"  {'─'*60}")


def load_tok(model_id: str, **kwargs: object) -> PreTrainedTokenizerFast:
    """Load tokenizer dan pastikan tipenya PreTrainedTokenizerFast."""
    tok = AutoTokenizer.from_pretrained(model_id, **kwargs)
    assert isinstance(tok, PreTrainedTokenizerFast), (
        f"Tokenizer dari {model_id} bukan PreTrainedTokenizerFast: {type(tok)}"
    )
    return tok


# ─────────────────────────────────────────────
# BAGIAN 1: ANALISIS TOKENIZER
# ─────────────────────────────────────────────
def analyze_tokenizers() -> tuple[PreTrainedTokenizerFast, PreTrainedTokenizerFast]:
    section("BAGIAN 1: ANALISIS TOKENIZER")

    print("\nLoading tokenizers...")
    tok_pt  = load_tok(G3_PT)
    tok_it  = load_tok(G3_IT)
    tok_t5g = load_tok(T5G, trust_remote_code=True)

    tokenizers: dict[str, PreTrainedTokenizerFast] = {
        "Gemma3-PT ": tok_pt,
        "Gemma3-IT ": tok_it,
        "T5Gemma2  ": tok_t5g,
    }

    subsection("A. Vocab Size & Basic Info")
    for name, tok in tokenizers.items():
        print(
            f"  {name}: vocab_size={tok.vocab_size}, "
            f"bos={tok.bos_token!r}(id={tok.bos_token_id}), "
            f"eos={tok.eos_token!r}(id={tok.eos_token_id}), "
            f"pad={tok.pad_token!r}(id={tok.pad_token_id})"
        )

    subsection("B. Special Token IDs yang Kita Pakai")
    header = f"  {'Token':<30}"
    for name in tokenizers:
        header += f" | {name:>12}"
    print(header)
    print("  " + "─" * 75)

    for token in SPECIAL_TOKENS_TO_CHECK:
        row = f"  {token:<30}"
        for name, tok in tokenizers.items():
            tid = tok.convert_tokens_to_ids(token)
            unk_id = tok.unk_token_id if tok.unk_token_id is not None else -1
            flag = " ⚠️" if tid == unk_id else ""
            row += f" | {str(tid) + flag:>12}"
        print(row)

    subsection("C. Test Encoding Teks Biasa (harus identik antar tokenizer)")
    test_text = "Apa ibu kota Indonesia?"
    for name, tok in tokenizers.items():
        ids = tok.encode(test_text, add_special_tokens=False)
        print(f"  {name}: {ids[:20]}{'...' if len(ids) > 20 else ''}")

    subsection("D. Test Encoding Prompt Format Gemma Chat")
    prompt_gemma = (
        f"<start_of_turn>user\n{SYSTEM_PROMPT}\n\n{TEST_TEXT}"
        f"<end_of_turn>\n<start_of_turn>model\n"
    )
    print(f"\n  Prompt:\n{prompt_gemma!r}\n")

    for name, tok in tokenizers.items():
        ids = tok.encode(prompt_gemma, add_special_tokens=False)
        tokens = tok.convert_ids_to_tokens(ids)
        print(f"  {name} ({len(ids)} tokens):")
        preview = list(zip(tokens[:15], ids[:15]))
        print(f"    First 15: {preview}")
        if len(ids) > 15:
            last5 = list(zip(tokens[-5:], ids[-5:]))
            print(f"    Last   5: {last5}")

    subsection("E. Apakah tok_it vs tok_t5g menghasilkan ID BERBEDA?")
    ids_it  = tok_it.encode(prompt_gemma, add_special_tokens=False)
    ids_t5g = tok_t5g.encode(prompt_gemma, add_special_tokens=False)

    if ids_it == ids_t5g:
        print("  ✅ IDENTIK: kedua tokenizer menghasilkan ID yang sama persis")
    else:
        print(f"  ⚠️ BERBEDA! IT={len(ids_it)} tokens, T5G={len(ids_t5g)} tokens")
        max_len = max(len(ids_it), len(ids_t5g))
        diffs = []
        for i in range(max_len):
            id_it  = ids_it[i]  if i < len(ids_it)  else None
            id_t5g = ids_t5g[i] if i < len(ids_t5g) else None
            if id_it != id_t5g:
                tok_it_str  = tok_it.convert_ids_to_tokens([id_it])[0]   if id_it  is not None else "N/A"
                tok_t5g_str = tok_t5g.convert_ids_to_tokens([id_t5g])[0] if id_t5g is not None else "N/A"
                diffs.append((i, id_it, tok_it_str, id_t5g, tok_t5g_str))
        print(f"  Jumlah perbedaan: {len(diffs)}")
        for pos, iit, tit, it5g, tt5g in diffs[:20]:
            print(f"    Pos {pos}: IT={iit}({tit!r})  T5G={it5g}({tt5g!r})")

    subsection("F. Chat Template yang Terdaftar")
    for name, tok in tokenizers.items():
        tmpl: str | dict | None = tok.chat_template  # type: ignore[assignment]
        has_template = tmpl is not None
        print(f"  {name}: has_chat_template={has_template}")
        if has_template and isinstance(tmpl, str):
            preview = tmpl.replace("\n", "\\n")[:200]
            print(f"    Template[:200]: {preview}...")

    subsection("G. Test apply_chat_template")
    messages = [{"role": "user", "content": f"{SYSTEM_PROMPT}\n\n{TEST_TEXT}"}]

    print("  Gemma3-IT apply_chat_template:")
    try:
        formatted = tok_it.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        print(f"  {formatted!r}")
    except Exception as exc:
        print(f"  Error: {exc}")

    print("\n  T5Gemma2 apply_chat_template:")
    try:
        formatted_t5g = tok_t5g.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        print(f"  {formatted_t5g!r}")
    except Exception as exc:
        print(f"  Error: {exc}")

    subsection("H. Unused Token Range — Yang Harus Di-Suppress")
    print("  Blok 1 (unused0–unused98): ID 6–104")
    for name, tok in tokenizers.items():
        sample = tok.convert_ids_to_tokens([6, 50, 104])
        print(f"  {name}: {list(zip([6, 50, 104], sample))}")

    print("\n  Blok 2 (unused100–unused6241): ID 256002–262143")
    for name, tok in tokenizers.items():
        sample = tok.convert_ids_to_tokens([256002, 259000, 262143])
        print(f"  {name}: {list(zip([256002, 259000, 262143], sample))}")

    print("\n  Vision tokens: 255999, 256000, 256001")
    for name, tok in tokenizers.items():
        sample = tok.convert_ids_to_tokens([255999, 256000, 256001])
        print(f"  {name}: {list(zip([255999, 256000, 256001], sample))}")

    return tok_it, tok_t5g


# ─────────────────────────────────────────────
# BAGIAN 2: TEST INFERENCE BEHAVIOR
# ─────────────────────────────────────────────
def test_inference_behavior(
    tok_it: PreTrainedTokenizerFast,
    tok_t5g: PreTrainedTokenizerFast,
) -> None:
    section("BAGIAN 2: TEST INFERENCE BEHAVIOR")

    prompt_gemma = (
        f"<start_of_turn>user\n"
        f"{SYSTEM_PROMPT}\n\n"
        f"{TEST_TEXT}<end_of_turn>\n"
        f"<start_of_turn>model\n"
    )
    end_of_turn_id = tok_it.convert_tokens_to_ids("<end_of_turn>")
    eos_id = tok_it.eos_token_id or 1
    stop_ids = list({end_of_turn_id, eos_id})

    print(f"\n  Prompt: {prompt_gemma!r}")
    print(f"  <end_of_turn> ID = {end_of_turn_id}, eos ID = {eos_id}")

    # ── A. Gemma 3 IT ──────────────────────────────────────────────────────
    subsection("A. Gemma 3 IT — Inference (Decoder-Only)")
    print(f"  Loading {G3_IT}...")
    try:
        model_it = AutoModelForCausalLM.from_pretrained(
            G3_IT, dtype=torch.bfloat16, device_map="auto"
        )
        model_it.eval()

        inputs_it = tok_it(prompt_gemma, return_tensors="pt", add_special_tokens=True)
        inputs_it = inputs_it.to(model_it.device)  # type: ignore[assignment]
        n_in = int(inputs_it["input_ids"].shape[1])
        print(f"  Input tokens: {n_in}")

        with torch.no_grad():
            out_it = getattr(model_it, "generate")(
                **inputs_it,
                max_new_tokens=150,
                do_sample=False,
                eos_token_id=stop_ids,
                repetition_penalty=1.0,
            )

        new_toks = out_it[0][n_in:]
        raw   = tok_it.decode(new_toks, skip_special_tokens=False)
        clean = tok_it.decode(new_toks, skip_special_tokens=True)
        print(f"\n  Raw  : {raw!r}")
        print(f"  Clean: {clean!r}")

        bad = [t for t in new_toks.tolist() if (6 <= t <= 104) or (256002 <= t <= 262143)]
        if bad:
            print(f"  ⚠️ Unusual IDs in output: {bad[:10]} → {tok_it.convert_ids_to_tokens(bad[:10])}")
        else:
            print("  ✅ Tidak ada unusual token")

        del model_it
        torch.cuda.empty_cache()

    except Exception as exc:
        print(f"  Error: {exc}")

    # ── B. Gemma 3 PT ──────────────────────────────────────────────────────
    subsection("B. Gemma 3 PT — Inference (Decoder-Only, base)")
    print(f"  Loading {G3_PT}...")
    try:
        model_pt = AutoModelForCausalLM.from_pretrained(
            G3_PT, dtype=torch.bfloat16, device_map="auto"
        )
        model_pt.eval()

        inputs_pt = tok_it(prompt_gemma, return_tensors="pt", add_special_tokens=True)
        inputs_pt = inputs_pt.to(model_pt.device)  # type: ignore[assignment]
        n_in = int(inputs_pt["input_ids"].shape[1])
        print(f"  Input tokens: {n_in}")

        with torch.no_grad():
            out_pt = getattr(model_pt, "generate")(
                **inputs_pt,
                max_new_tokens=100,
                do_sample=False,
                eos_token_id=stop_ids,
            )

        new_toks = out_pt[0][n_in:]
        raw   = tok_it.decode(new_toks, skip_special_tokens=False)
        clean = tok_it.decode(new_toks, skip_special_tokens=True)
        print(f"\n  Raw  : {raw!r}")
        print(f"  Clean: {clean!r}")

        bad = [t for t in new_toks.tolist() if (6 <= t <= 104) or (256002 <= t <= 262143)]
        if bad:
            print(f"  ⚠️ Unusual IDs: {bad[:10]} → {tok_it.convert_ids_to_tokens(bad[:10])}")
        else:
            print("  ✅ Tidak ada unusual token")

        del model_pt
        torch.cuda.empty_cache()

    except Exception as exc:
        print(f"  Error: {exc}")

    # ── C. T5Gemma2 Task-Vector Model ──────────────────────────────────────
    subsection("C. T5Gemma2 Task-Vector Model — Inference (Encoder-Decoder)")
    if not os.path.exists(TASK_VECTOR_MODEL):
        print(f"  ⚠️ Model tidak ditemukan di '{TASK_VECTOR_MODEL}', skip.")
        return

    print(f"  Loading {TASK_VECTOR_MODEL}...")
    try:
        tok_m = load_tok(TASK_VECTOR_MODEL)
        model_t5 = AutoModelForSeq2SeqLM.from_pretrained(
            TASK_VECTOR_MODEL,
            trust_remote_code=True,
            dtype=torch.bfloat16,
            device_map="auto",
        )
        model_t5.eval()

        eot_t5 = tok_m.convert_tokens_to_ids("<end_of_turn>")
        eos_t5 = tok_m.eos_token_id or 1
        stop_t5 = list({eot_t5, eos_t5})
        print(f"  stop_ids = {stop_t5}")

        def run_t5(prompt: str, label: str) -> None:
            enc = tok_m(
                prompt, return_tensors="pt", truncation=True, max_length=2048
            ).to(model_t5.device)  # type: ignore[assignment]
            n_in = int(enc["input_ids"].shape[1])
            print(f"\n  [{label}] Input tokens: {n_in}")
            with torch.no_grad():
                out = model_t5.generate(
                    **enc, max_new_tokens=150, do_sample=False, eos_token_id=stop_t5
                )
            raw   = tok_m.decode(out[0], skip_special_tokens=False)
            clean = tok_m.decode(out[0], skip_special_tokens=True)
            print(f"  Raw  : {raw!r}")
            print(f"  Clean: {clean!r}")
            bad = [t for t in out[0].tolist() if (6 <= t <= 104) or (256002 <= t <= 262143)]
            print(f"  {'⚠️  Unusual IDs: ' + str(bad[:10]) if bad else '✅ Tidak ada unusual token'}")

        # Test 1: dengan trailing <start_of_turn>model
        run_t5(prompt_gemma, "Dengan <start_of_turn>model cue")

        # Test 2: tanpa trailing cue
        prompt_no_cue = (
            f"<start_of_turn>user\n{SYSTEM_PROMPT}\n\n{TEST_TEXT}<end_of_turn>\n"
        )
        run_t5(prompt_no_cue, "Tanpa <start_of_turn>model cue")

        del model_t5
        torch.cuda.empty_cache()

    except Exception as exc:
        import traceback
        print(f"  Error: {exc}")
        traceback.print_exc()


# ─────────────────────────────────────────────
# BAGIAN 3: KESIMPULAN
# ─────────────────────────────────────────────
def print_conclusions() -> None:
    section("BAGIAN 3: CHECKLIST PERTANYAAN KRITIS")
    questions = [
        "Apakah tok_it dan tok_t5g menghasilkan ID yang sama untuk teks biasa?",
        "Apakah <start_of_turn>/<end_of_turn> punya ID yang sama di kedua tokenizer?",
        "Apakah Gemma 3 IT menghasilkan <end_of_turn> saat selesai menjawab?",
        "Apakah base PT model output repetisi/garbage dengan Gemma chat format?",
        "Apakah T5Gemma2 task vector menghasilkan spam sebelum SFT?",
        "Apakah trailing <start_of_turn>model berpengaruh di encoder-decoder?",
    ]
    for i, q in enumerate(questions, 1):
        print(f"\n  [{i}] {q}")
        print(f"       → Lihat output di atas untuk jawaban")


if __name__ == "__main__":
    print("\n🔍 T5Gemma2 Cangkok — Tokenizer & Template Analysis")
    print("=" * 70)

    tok_it, tok_t5g = analyze_tokenizers()
    test_inference_behavior(tok_it, tok_t5g)
    print_conclusions()

    print("\n\n✅ Analysis selesai!")
