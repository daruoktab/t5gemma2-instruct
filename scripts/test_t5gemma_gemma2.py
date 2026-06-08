"""
Test T5Gemma Models (Gemma 2 based)
====================================
Bandingkan 4 varian:
  1. t5gemma-2b-2b-ul2          (PT, UL2 objective)
  2. t5gemma-2b-2b-ul2-it       (IT, UL2 objective)
  3. t5gemma-2b-2b-prefixlm     (PT, PrefixLM objective)
  4. t5gemma-2b-2b-prefixlm-it  (IT, PrefixLM objective)

Semua adalah encoder-decoder (seq2seq) berbasis Gemma 2.
Kita test:
  - Tokenizer info & special tokens
  - Chat template (ada/tidak)
  - Inference dengan berbagai format prompt
  - Kualitas output bahasa Indonesia
"""

import torch
import os
from transformers import (
    AutoTokenizer,
    AutoModelForSeq2SeqLM,
    PreTrainedTokenizerFast,
)

# ─────────────────────────────────────────────
# MODELS
# ─────────────────────────────────────────────
MODELS = {
    "ul2-pt"        : "google/t5gemma-2b-2b-ul2",
    "ul2-it"        : "google/t5gemma-2b-2b-ul2-it",
    "prefixlm-pt"   : "google/t5gemma-2b-2b-prefixlm",
    "prefixlm-it"   : "google/t5gemma-2b-2b-prefixlm-it",
}

# ─────────────────────────────────────────────
# TEST QUERIES
# ─────────────────────────────────────────────
SYSTEM_PROMPT = (
    "Kamu adalah asisten AI yang helpful dan ramah. "
    "Gunakan Bahasa Indonesia."
)
QUERIES = [
    "Siapa presiden Indonesia pertama?",
    "Jelaskan fotosintesis dengan bahasa sederhana.",
    "Apa ibukota Jepang?",
]

SPECIAL_TOKENS = [
    "<start_of_turn>", "<end_of_turn>",
    "<bos>", "<eos>", "<pad>", "<unk>",
    "<extra_id_0>", "<extra_id_1>",  # T5-style sentinel tokens
]


def sep(char: str = "=", n: int = 70) -> None:
    print(char * n)


def header(title: str) -> None:
    print(f"\n{'='*70}")
    print(f"  {title}")
    print("=" * 70)


def subheader(title: str) -> None:
    print(f"\n  {'─'*60}")
    print(f"  {title}")
    print(f"  {'─'*60}")


# ─────────────────────────────────────────────
# STEP 1: TOKENIZER ANALYSIS (tanpa load model)
# ─────────────────────────────────────────────
def analyze_all_tokenizers() -> None:
    header("STEP 1: TOKENIZER ANALYSIS (semua model)")

    tokenizers: dict[str, PreTrainedTokenizerFast] = {}

    for name, model_id in MODELS.items():
        print(f"\n  Loading tokenizer: {name} ({model_id})...")
        try:
            tok = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
            if not isinstance(tok, PreTrainedTokenizerFast):
                print(f"  ⚠️ Bukan PreTrainedTokenizerFast: {type(tok)}")
            else:
                tokenizers[name] = tok
                print(f"  ✅ Loaded. vocab_size={tok.vocab_size}")
        except Exception as e:
            print(f"  ❌ Error: {e}")

    if not tokenizers:
        print("Semua tokenizer gagal dimuat!")
        return

    subheader("A. Vocab Size & Special Tokens")
    print(f"  {'Name':<16} | {'vocab':>8} | {'bos':>6} | {'eos':>6} | {'pad':>6}")
    print("  " + "─" * 52)
    for name, tok in tokenizers.items():
        print(
            f"  {name:<16} | {tok.vocab_size:>8} | "
            f"{tok.bos_token_id!r:>6} | {tok.eos_token_id!r:>6} | "
            f"{tok.pad_token_id!r:>6}"
        )

    subheader("B. Special Token IDs Kritis")
    first_tok = next(iter(tokenizers.values()))
    ref_name  = next(iter(tokenizers.keys()))
    all_names = list(tokenizers.keys())

    print(f"  {'Token':<25}", end="")
    for n in all_names:
        print(f" | {n:>16}", end="")
    print()
    print("  " + "─" * (27 + 19 * len(all_names)))

    for token in SPECIAL_TOKENS:
        print(f"  {token:<25}", end="")
        for n, tok in tokenizers.items():
            tid = tok.convert_tokens_to_ids(token)
            unk = tok.unk_token_id if tok.unk_token_id is not None else -99
            flag = "⚠️" if tid == unk else ""
            print(f" | {str(tid)+flag:>16}", end="")
        print()

    subheader("C. Chat Template")
    for name, tok in tokenizers.items():
        tmpl = getattr(tok, "chat_template", None)
        has  = tmpl is not None
        print(f"  {name:<16}: has_chat_template={has}", end="")
        if has and isinstance(tmpl, str):
            preview = tmpl.replace("\n", "\\n")[:120]
            print(f"\n    {preview}...")
        else:
            print()

    subheader("D. Test apply_chat_template")
    msgs = [{"role": "user", "content": f"{SYSTEM_PROMPT}\n\n{QUERIES[0]}"}]
    for name, tok in tokenizers.items():
        try:
            result = tok.apply_chat_template(
                msgs, tokenize=False, add_generation_prompt=True
            )
            preview = repr(result)[:200]
            print(f"  {name}: {preview}")
        except Exception as e:
            print(f"  {name}: ❌ {e}")

    subheader("E. Encoding Identik? (ref: ul2-it)")
    if "ul2-it" in tokenizers:
        ref_tok = tokenizers["ul2-it"]
        test_prompt = f"<start_of_turn>user\n{SYSTEM_PROMPT}\n\n{QUERIES[0]}<end_of_turn>\n<start_of_turn>model\n"
        ref_ids = ref_tok.encode(test_prompt, add_special_tokens=False)
        print(f"  Ref (ul2-it): {ref_ids[:15]}...")
        for name, tok in tokenizers.items():
            if name == "ul2-it":
                continue
            ids = tok.encode(test_prompt, add_special_tokens=False)
            same = ids == ref_ids
            print(f"  {name:<16}: {'IDENTIK ✅' if same else 'BEDA ⚠️  → ' + str(ids[:15])}")

    subheader("F. Extra ID / Sentinel Tokens (T5 style)")
    for name, tok in tokenizers.items():
        sentinels = []
        for i in range(5):
            tid = tok.convert_tokens_to_ids(f"<extra_id_{i}>")
            unk = tok.unk_token_id if tok.unk_token_id is not None else -99
            if tid != unk:
                sentinels.append((f"<extra_id_{i}>", tid))
        print(f"  {name:<16}: sentinels={sentinels}")


# ─────────────────────────────────────────────
# STEP 2: INFERENCE TEST
# ─────────────────────────────────────────────
def test_model(
    model_id: str,
    name: str,
    is_it: bool,
) -> None:
    """Test inference untuk satu model."""
    print(f"\n  Loading {name} ({model_id})...")
    try:
        tok = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
        assert isinstance(tok, PreTrainedTokenizerFast)

        model = AutoModelForSeq2SeqLM.from_pretrained(
            model_id,
            trust_remote_code=True,
            dtype=torch.bfloat16,
            device_map="auto",
        )
        model.eval()

        # eos_token_id bisa int | list[int] | None — extract ke int dulu
        # convert_tokens_to_ids juga bisa list[int] jika input list — kita pastikan string
        _eot_raw = tok.convert_tokens_to_ids("<end_of_turn>")
        eot_id: int = _eot_raw[0] if isinstance(_eot_raw, list) else _eot_raw
        _raw_eos = tok.eos_token_id
        if isinstance(_raw_eos, list):
            eos_id: int = _raw_eos[0] if _raw_eos else 1
        elif _raw_eos is None:
            eos_id = 1
        else:
            eos_id = _raw_eos
        unk_id: int = tok.unk_token_id if tok.unk_token_id is not None else -1
        stop_ids: list[int] = [eos_id]
        if eot_id != unk_id and eot_id not in stop_ids:
            stop_ids.append(eot_id)

        print(f"  stop_ids={stop_ids}")
        print(f"  Config: encoder_layers={getattr(model.config, 'num_encoder_layers', '?')}, "
              f"decoder_layers={getattr(model.config, 'num_decoder_layers', '?')}")

        query = QUERIES[0]  # "Siapa presiden Indonesia pertama?"

        # ── Test Format 1: Gemma Chat Format ────────────────────────────
        print(f"\n  [Format 1] Gemma chat format:")
        prompt_gemma = (
            f"<start_of_turn>user\n{SYSTEM_PROMPT}\n\n{query}"
            f"<end_of_turn>\n<start_of_turn>model\n"
        )
        _run_inference(model, tok, prompt_gemma, stop_ids, label="Gemma chat")

        # ── Test Format 2: Plain Q&A ─────────────────────────────────────
        print(f"\n  [Format 2] Plain Q&A:")
        prompt_plain = f"{SYSTEM_PROMPT}\n\nPertanyaan: {query}\nJawaban:"
        _run_inference(model, tok, prompt_plain, stop_ids, label="Plain Q&A")

        # ── Test Format 3: apply_chat_template (jika ada) ───────────────
        try:
            msgs = [{"role": "user", "content": f"{SYSTEM_PROMPT}\n\n{query}"}]
            prompt_tmpl = tok.apply_chat_template(
                msgs, tokenize=False, add_generation_prompt=True
            )
            assert isinstance(prompt_tmpl, str)
            print(f"\n  [Format 3] apply_chat_template:")
            _run_inference(model, tok, prompt_tmpl, stop_ids, label="chat_template")
        except Exception as e:
            print(f"\n  [Format 3] apply_chat_template: ❌ {e}")

        # ── Test Format 4: Hanya pertanyaan (minimalist) ─────────────────
        print(f"\n  [Format 4] Minimalist (hanya question):")
        _run_inference(model, tok, query, stop_ids, label="Minimalist")

        del model
        torch.cuda.empty_cache()

    except Exception as e:
        import traceback
        print(f"  ❌ Error loading/running {name}: {e}")
        traceback.print_exc()


def _run_inference(
    model: object,
    tok: PreTrainedTokenizerFast,
    prompt: str,
    stop_ids: list[int],
    label: str,
    max_new_tokens: int = 150,
) -> None:
    """Helper: encode → generate → decode → print."""
    enc = tok(
        prompt,
        return_tensors="pt",
        truncation=True,
        max_length=2048,
        add_special_tokens=False,  # Jangan prepend BOS di encoder
    )
    enc = {k: v.to(getattr(model, "device", "cpu")) for k, v in enc.items()}  # type: ignore[arg-type]
    n_in = int(enc["input_ids"].shape[1])

    with torch.no_grad():
        out = getattr(model, "generate")(
            **enc,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            eos_token_id=stop_ids,
            repetition_penalty=1.1,  # Sedikit penalti untuk cegah repetisi
        )

    raw   = tok.decode(out[0], skip_special_tokens=False)
    clean = tok.decode(out[0], skip_special_tokens=True)

    # Deteksi masalah
    # decode() bisa return str | list[str] tergantung input — narrow ke str
    clean_str: str = clean if isinstance(clean, str) else " ".join(clean)
    words = clean_str.split()
    is_rep = len(set(words)) < max(1, len(words) * 0.4) and len(words) > 10
    flag = " ⚠️ REPETITIVE" if is_rep else ""

    print(f"    [{label}] in={n_in} tokens, out={len(out[0])} tokens")
    print(f"    Raw  : {repr(raw)[:250]}")
    print(f"    Clean: {repr(clean_str)[:200]}{flag}")


# ─────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────
def main() -> None:
    print("\n🔍 T5Gemma Models Test (Gemma 2 based)")
    sep()
    print("Models yang akan ditest:")
    for name, mid in MODELS.items():
        print(f"  {name:<16} → {mid}")

    # Step 1: Tokenizer analysis (semua, ringan)
    analyze_all_tokenizers()

    # Step 2: Inference test satu per satu (berat, load/unload per model)
    header("STEP 2: INFERENCE TEST (satu per satu)")
    print("  ⚠️ Setiap model ~2B params, akan load/unload bergantian untuk hemat VRAM")

    for name, model_id in MODELS.items():
        subheader(f"Model: {name.upper()}")
        is_it = name.endswith("-it")
        test_model(model_id, name, is_it)

    header("SELESAI")
    print("  Lihat output di atas untuk perbandingan lengkap.")
    print("  Key questions:")
    print("  1. Model IT mana yang paling baik untuk Bahasa Indonesia?")
    print("  2. UL2 vs PrefixLM — mana yang lebih coherent?")
    print("  3. Format prompt mana yang paling efektif?")
    print("  4. Ada repetisi? Token aneh?")


if __name__ == "__main__":
    main()
