"""
Evaluasi Komparatif: Base Model vs Fine-tuned Model (Instruction Tuning)
========================================================================
Membandingkan output base model (google/t5gemma-2-270m-270m)
vs model yang sudah di-finetune (checkpoint-11000) pada prompt yang sama.

Metrik:
- Response length (lebih panjang = lebih informatif)
- Non-empty rate
- Side-by-side comparison untuk qualitative evaluation

Usage (dari root repo):
    conda activate unsloth
    python instruct/evaluate_compare.py
    python instruct/evaluate_compare.py --checkpoint checkpoint-9000
    python instruct/evaluate_compare.py --no-base   # hanya finetuned
"""

from __future__ import annotations

import argparse
import gc
import json
import os
import time
from typing import Optional

import torch
from peft import PeftModel
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, PreTrainedModel, PreTrainedTokenizerBase
from typing import cast

# ============================================================
# Config
# ============================================================
BASE_MODEL = "google/t5gemma-2-270m-270m"

# Resolve semua path relatif terhadap lokasi script ini
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
ADAPTER_DIR = os.path.normpath(os.path.join(_SCRIPT_DIR, "..", "t5gemma2-chat-v1"))

# Prompt pairs: (kategori, prompt)
EVAL_PROMPTS: list[tuple[str, str]] = [
    # Summarization
    (
        "Summarize (EN)",
        "Summarize: Machine learning is a branch of artificial intelligence that enables "
        "systems to learn from data and improve their accuracy over time without being "
        "explicitly programmed. It uses algorithms to find patterns in large datasets.",
    ),
    (
        "Ringkasan (ID)",
        "Buatkan ringkasan: Indonesia adalah negara kepulauan dengan lebih dari 17.000 "
        "pulau dan populasi 270 juta jiwa. Memiliki keragaman budaya dan kekayaan alam "
        "yang luar biasa, mulai dari hutan hujan tropis hingga terumbu karang.",
    ),
    # Translation
    (
        "Translate EN→ID",
        "Translate to Indonesian: Education is the most powerful weapon which you can "
        "use to change the world.",
    ),
    (
        "Translate ID→EN",
        "Terjemahkan ke bahasa Inggris: Teknologi kecerdasan buatan semakin berkembang "
        "pesat dan mengubah berbagai aspek kehidupan manusia.",
    ),
    # Instruction (EN)
    (
        "Instruction (EN)",
        "Explain what photosynthesis is in simple terms.",
    ),
    (
        "Instruction (EN) 2",
        "List 3 benefits of regular exercise.",
    ),
    # Instruction (ID)
    (
        "Instruksi (ID)",
        "Jelaskan apa itu kecerdasan buatan dengan bahasa yang mudah dipahami.",
    ),
    (
        "Instruksi (ID) 2",
        "Sebutkan 3 cara menjaga kesehatan di era digital.",
    ),
    # QA
    (
        "QA (EN)",
        "What is the capital city of Indonesia and what is it known for?",
    ),
    (
        "QA (ID)",
        "Siapa presiden pertama Indonesia dan apa yang beliau perjuangkan?",
    ),
    # Creative
    (
        "Creative (EN)",
        "Write a short paragraph about the beauty of nature.",
    ),
    (
        "Creative (ID)",
        "Tulis paragraf singkat tentang pentingnya membaca buku.",
    ),
]


# ============================================================
# Helper
# ============================================================

def free_gpu() -> None:
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def load_base_model() -> tuple[PreTrainedModel, PreTrainedTokenizerBase]:
    """Load base model tanpa adapter."""
    print("\n📦 Loading BASE model (no fine-tuning)...")
    _tok = AutoTokenizer.from_pretrained(BASE_MODEL)
    assert isinstance(_tok, PreTrainedTokenizerBase), "AutoTokenizer harus return PreTrainedTokenizerBase"
    tokenizer: PreTrainedTokenizerBase = _tok
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForSeq2SeqLM.from_pretrained(
        BASE_MODEL,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        attn_implementation="eager",
    )
    if getattr(model.config, "decoder_start_token_id", None) is None:
        model.config.decoder_start_token_id = tokenizer.bos_token_id

    model.eval()
    print(f"   ✅ Base model loaded ({sum(p.numel() for p in model.parameters()) / 1e6:.1f}M params)")
    return model, tokenizer


def load_finetuned_model(checkpoint: str) -> tuple[PreTrainedModel, PreTrainedTokenizerBase]:
    """Load base model + LoRA adapter dari checkpoint."""
    adapter_path = os.path.join(ADAPTER_DIR, checkpoint)
    print(f"\n🎯 Loading FINE-TUNED model (adapter: {checkpoint})...")

    _tok2 = AutoTokenizer.from_pretrained(adapter_path)
    assert isinstance(_tok2, PreTrainedTokenizerBase), "AutoTokenizer harus return PreTrainedTokenizerBase"
    tokenizer: PreTrainedTokenizerBase = _tok2
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    base = AutoModelForSeq2SeqLM.from_pretrained(
        BASE_MODEL,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        attn_implementation="eager",
    )
    if getattr(base.config, "decoder_start_token_id", None) is None:
        base.config.decoder_start_token_id = tokenizer.bos_token_id

    # Cast PeftModel to PreTrainedModel because it behaves like one for our purposes
    model = cast(PreTrainedModel, PeftModel.from_pretrained(base, adapter_path))
    model.eval()
    print(f"   ✅ Fine-tuned model loaded (LoRA adapter: {adapter_path})")
    return model, tokenizer


def generate(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizerBase,
    prompt: str,
    max_new_tokens: int = 200,
    use_beam: bool = False,
) -> tuple[str, float]:
    """Generate response dan kembalikan (text, latency_ms)."""
    inputs = tokenizer(
        prompt,
        return_tensors="pt",
        max_length=512,
        truncation=True,
    ).to(model.device)

    gen_kwargs: dict = {"max_new_tokens": max_new_tokens, "repetition_penalty": 1.2}
    if use_beam:
        gen_kwargs["num_beams"] = 4
        gen_kwargs["early_stopping"] = True
        gen_kwargs["do_sample"] = False
    else:
        gen_kwargs["do_sample"] = True
        gen_kwargs["temperature"] = 0.7
        gen_kwargs["top_p"] = 0.9
        gen_kwargs["top_k"] = 50

    t0 = time.perf_counter()
    with torch.no_grad():
        # Gunakan getattr agar checker tidak confuse dengan Tensor.__call__
        generate_fn = getattr(model, "generate")
        raw_out = generate_fn(**inputs, **gen_kwargs)
    latency_ms = (time.perf_counter() - t0) * 1000

    # model.generate bisa return Tensor shape [batch, seq] atau list[Tensor]
    if isinstance(raw_out, torch.Tensor):
        first_seq: torch.Tensor = raw_out[0]
    else:
        first_seq = raw_out[0] if isinstance(raw_out[0], torch.Tensor) else torch.tensor(raw_out[0])
    # cast eksplisit karena tokenizer.decode bisa return str | list[str]
    text: str = str(tokenizer.decode(first_seq, skip_special_tokens=True))
    return text, latency_ms


# ============================================================
# Main Evaluation
# ============================================================

def run_evaluation(
    checkpoint: str,
    run_base: bool = True,
    run_finetuned: bool = True,
    output_file: str = "eval_comparison.json",
) -> None:
    results: list[dict] = []

    # ── 1. Run Base Model ──────────────────────────────────────
    base_outputs: dict[str, str] = {}
    base_latencies: dict[str, float] = {}

    if run_base:
        base_model, base_tok = load_base_model()

        print("\n" + "=" * 70)
        print("EVALUATING: BASE MODEL")
        print("=" * 70)

        for cat, prompt in EVAL_PROMPTS:
            use_beam = any(k in cat for k in ("Translate", "Summarize", "Ringkasan"))
            resp, ms = generate(base_model, base_tok, prompt, use_beam=use_beam)
            base_outputs[cat] = resp
            base_latencies[cat] = ms
            short_p = prompt[:80] + "..." if len(prompt) > 80 else prompt
            short_r = resp[:100] + "..." if len(resp) > 100 else resp
            print(f"  [{cat}]")
            print(f"    INPUT:  {short_p}")
            print(f"    OUTPUT: {short_r}  ({ms:.0f}ms)")

        # Bersihkan setelah selesai
        del base_model
        free_gpu()

    # ── 2. Run Fine-tuned Model ────────────────────────────────
    ft_outputs: dict[str, str] = {}
    ft_latencies: dict[str, float] = {}

    if run_finetuned:
        ft_model, ft_tok = load_finetuned_model(checkpoint)

        print("\n" + "=" * 70)
        print(f"EVALUATING: FINE-TUNED MODEL ({checkpoint})")
        print("=" * 70)

        for cat, prompt in EVAL_PROMPTS:
            use_beam = any(k in cat for k in ("Translate", "Summarize", "Ringkasan"))
            resp, ms = generate(ft_model, ft_tok, prompt, use_beam=use_beam)
            ft_outputs[cat] = resp
            ft_latencies[cat] = ms
            short_p = prompt[:80] + "..." if len(prompt) > 80 else prompt
            short_r = resp[:100] + "..." if len(resp) > 100 else resp
            print(f"  [{cat}]")
            print(f"    INPUT:  {short_p}")
            print(f"    OUTPUT: {short_r}  ({ms:.0f}ms)")

        del ft_model
        free_gpu()

    # ── 3. Side-by-side Comparison ────────────────────────────
    print("\n\n" + "=" * 70)
    print("SIDE-BY-SIDE COMPARISON")
    print("=" * 70)

    stats_base = {"total_len": 0, "empty": 0}
    stats_ft = {"total_len": 0, "empty": 0}

    for cat, prompt in EVAL_PROMPTS:
        base_resp = base_outputs.get(cat, "N/A")
        ft_resp = ft_outputs.get(cat, "N/A")

        if run_base:
            stats_base["total_len"] += len(base_resp)
            if len(base_resp.strip()) < 5:
                stats_base["empty"] += 1

        if run_finetuned:
            stats_ft["total_len"] += len(ft_resp)
            if len(ft_resp.strip()) < 5:
                stats_ft["empty"] += 1

        print(f"\n{'─' * 70}")
        print(f"📌 [{cat}]")
        print(f"   Prompt: {prompt[:100]}{'...' if len(prompt) > 100 else ''}")
        if run_base:
            print(f"\n   🔵 BASE : {base_resp}")
        if run_finetuned:
            print(f"   🟠 TUNED: {ft_resp}")

        results.append({
            "category": cat,
            "prompt": prompt,
            "base_response": base_resp if run_base else None,
            "finetuned_response": ft_resp if run_finetuned else None,
            "base_latency_ms": base_latencies.get(cat),
            "finetuned_latency_ms": ft_latencies.get(cat),
        })

    # ── 4. Summary Stats ──────────────────────────────────────
    n = len(EVAL_PROMPTS)
    avg_len_b, avg_lat_b, empty_b = 0.0, 0.0, 0
    avg_len_f, avg_lat_f, empty_f = 0.0, 0.0, 0

    print("\n\n" + "=" * 70)
    print("SUMMARY STATISTICS")
    print("=" * 70)

    if run_base:
        avg_len_b = stats_base["total_len"] / n
        avg_lat_b = sum(base_latencies.values()) / len(base_latencies) if base_latencies else 0
        empty_b = stats_base["empty"]
        print(f"\n🔵 BASE MODEL:")
        print(f"   Avg response length : {avg_len_b:.0f} chars")
        print(f"   Empty responses     : {empty_b}/{n}")
        print(f"   Avg latency         : {avg_lat_b:.0f} ms")

    if run_finetuned:
        avg_len_f = stats_ft["total_len"] / n
        avg_lat_f = sum(ft_latencies.values()) / len(ft_latencies) if ft_latencies else 0
        empty_f = stats_ft["empty"]
        print(f"\n🟠 FINE-TUNED ({checkpoint}):")
        print(f"   Avg response length : {avg_len_f:.0f} chars")
        print(f"   Empty responses     : {empty_f}/{n}")
        print(f"   Avg latency         : {avg_lat_f:.0f} ms")

    if run_base and run_finetuned:
        delta_len = avg_len_f - avg_len_b
        print(f"\n📊 DELTA (finetuned - base):")
        print(f"   Response length: {delta_len:+.0f} chars ({delta_len/avg_len_b*100:+.1f}%)")
        print(f"   Fewer empty    : {empty_b - empty_f:+d} (positive = improvement)")

    # ── 5. Save ───────────────────────────────────────────────
    save_path = os.path.join(os.path.dirname(__file__), output_file)
    with open(save_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "checkpoint": checkpoint,
                "base_model": BASE_MODEL,
                "num_prompts": n,
                "results": results,
                "stats": {
                    "base": {
                        "avg_response_len": round(stats_base["total_len"] / n, 1) if run_base else None,
                        "empty_count": stats_base["empty"] if run_base else None,
                        "avg_latency_ms": round(avg_lat_b, 1) if run_base else None,
                    },
                    "finetuned": {
                        "avg_response_len": round(stats_ft["total_len"] / n, 1) if run_finetuned else None,
                        "empty_count": stats_ft["empty"] if run_finetuned else None,
                        "avg_latency_ms": round(avg_lat_f, 1) if run_finetuned else None,
                    },
                },
            },
            f,
            ensure_ascii=False,
            indent=2,
        )
    print(f"\n✅ Results saved to: {save_path}")


# ============================================================
# Entry Point
# ============================================================

def _latest_checkpoint() -> str:
    """Ambil checkpoint terbaru dari ADAPTER_DIR."""
    candidates = [
        d for d in os.listdir(ADAPTER_DIR)
        if d.startswith("checkpoint-") and os.path.isdir(os.path.join(ADAPTER_DIR, d))
    ]
    if not candidates:
        raise FileNotFoundError(f"Tidak ada checkpoint di {ADAPTER_DIR}")
    return sorted(candidates, key=lambda x: int(x.split("-")[1]))[-1]


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluasi komparatif Base vs Fine-tuned model")
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=None,
        help="Nama checkpoint (e.g. 'checkpoint-11000'). Default: pakai yang terbaru.",
    )
    parser.add_argument(
        "--no-base",
        action="store_true",
        help="Skip evaluasi base model",
    )
    parser.add_argument(
        "--no-finetuned",
        action="store_true",
        help="Skip evaluasi fine-tuned model",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="eval_comparison.json",
        help="Output file name (default: eval_comparison.json)",
    )
    args = parser.parse_args()

    # Auto-detect checkpoint
    checkpoint = args.checkpoint or _latest_checkpoint()
    print(f"Using checkpoint: {checkpoint}")

    run_evaluation(
        checkpoint=checkpoint,
        run_base=not args.no_base,
        run_finetuned=not args.no_finetuned,
        output_file=args.output,
    )
