# /// script
# requires-python = ">=3.10"
# dependencies = [
#     "accelerate==1.14.0",
#     "absl-py==2.4.0",
#     "bitsandbytes==0.49.2",
#     "datasets==5.0.0",
#     "evaluate",
#     "rouge-score",
#     "sacrebleu",
#     "bert_score",
#     "nltk",
#     "huggingface-hub==1.23.0",
#     "marimo==0.23.14",
#     "numpy==2.5.1",
#     "peft==0.19.1",
#     "pillow==12.3.0",
#     "pymupdf==1.28.0",
#     "pytorch-optimizer",
#     "torch==2.12.1",
#     "torchvision==0.27.1",
#     "trl==1.8.0",
#     "transformers==5.13.1",
#     "unsloth_zoo @ git+https://github.com/daruoktab/unsloth-zoo.git",
#     "unsloth @ git+https://github.com/daruoktab/unsloth.git",
# ]
# ///
#
# =====================================================================
# COMBINED PIPELINE (gabungan 2 notebook, TIDAK mengubah file asli):
#   - working-molab-v6-unsloth.py        (Phase 1: TEXT SFT -> ORPO)
#   - working-molab-v6-vision-unsloth.py (Phase 2: VISION SFT -> ORPO)
#   - Phase 1.5: CANGKOK (diadaptasi dari scripts/tests/verify_vision_weights_3way.py
#     cell CANGKOK + scripts/tests/patch_cangkok_tokenizer.py)
#
# Aturan penamaan (marimo melarang nama variabel non-underscore didefinisikan 2x):
#   TEXT_*   = konfigurasi Phase 1 (text)
#   VISION_* = konfigurasi Phase 2 (vision)
#   shared   = util/konstanta identik di kedua pipeline (didefinisikan SEKALI)
# =====================================================================

import marimo

__generated_with = "0.23.14"
app = marimo.App(
    width="full",
    css_file="/usr/local/_marimo/custom.css",
    auto_download=["html"],
)


@app.cell
def _():
    import marimo as mo

    return (mo,)


@app.cell
def _():
    import subprocess
    import sys

    # Auto-install dependencies jika belum ada di env Molab
    try:
        import unsloth
        import datasets
        import peft
        print("✅ Dependencies utama sudah ter-install.")
    except ImportError:
        print("📦 Meng-install dependencies di Molab...")
        subprocess.run(
            [
                "uv", "pip", "install",
                "accelerate==1.14.0",
                "absl-py==2.4.0",
                "bitsandbytes==0.49.2",
                "datasets==5.0.0",
                "evaluate",
                "rouge-score",
                "sacrebleu",
                "bert_score",
                "nltk",
                "huggingface-hub==1.23.0",
                "marimo==0.23.14",
                "numpy==2.5.1",
                "peft==0.19.1",
                "pillow==12.3.0",
                "pymupdf==1.28.0",
                "torch==2.12.1",
                "torchvision==0.27.1",
                "trl==1.8.0",
                "transformers==5.13.1",
                "unsloth_zoo @ git+https://github.com/daruoktab/unsloth-zoo.git",
                "unsloth @ git+https://github.com/daruoktab/unsloth.git",
            ],
            check=True
        )

    # Selalu pastikan flash_attn ter-install
    try:
        import flash_attn
        print("✅ flash_attn sudah ter-install.")
    except ImportError:
        print("📦 Meng-install flash_attn prebuild wheel...")
        subprocess.run(
            [
                "uv", "pip", "install",
                "flash_attn @ https://github.com/mjun0812/flash-attention-prebuild-wheels/releases/download/v0.9.17/flash_attn-2.8.3+cu130torch2.12-cp313-cp313-linux_x86_64.whl",
            ],
            check=True,
        )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ✅ **STATUS: PIPELINE GABUNGAN READY TO RUN.**

    # 🔗 Combined Pipeline: TEXT (SFT+ORPO) → CANGKOK → VISION (SFT+ORPO)
    =====================================================================
    Satu file panjang yang menjalankan 3 fase berurutan:

    1. **Phase 1 (TEXT)** — `google/t5gemma-2-4b-4b` dilatih SFT → ORPO (LoRA/QLoRA Unsloth),
       merge (BF16 + 4bit), upload ke subfolder `text/`.
    2. **Phase 1.5 (CANGKOK)** — Vision tower (SigLIP) + `multi_modal_projector` dari
       `google/gemma-3-4b-it` dicangkokkan ke `text/merged_bf16` hasil Phase 1, di-upload ke
       subfolder `cangkok/`.
       *(Mekanisme diadaptasi dari `scripts/tests/verify_vision_weights_3way.py` cell CANGKOK
       + `scripts/tests/patch_cangkok_tokenizer.py`.)*
    3. **Phase 2 (VISION)** — base = subfolder `cangkok/`, dilatih Vision SFT → ORPO,
       merge, upload ke subfolder `vision/`.

    **Semua artifacts berada dalam 1 repo PUBLIK:**
    `daruokta/t5gemma-2-4b-4b-instruct-chat-indo-v6-combined-unsloth`
    ```
    text/     → sft/, orpo/, merged_bf16/, quantized_4bit/
    cangkok/  → full model hasil graft (base untuk Phase 2)
    vision/   → sft/, orpo/, merged_bf16/, quantized_4bit/  ← hasil akhir multimodal
    ```

    **Fitur utama (diwarisi dari kedua notebook asli):**
    - Auto-detect progress dari HF Hub per fase — lanjut dari checkpoint terakhir.
    - Upload checkpoint ke HF segera setelah disimpan (tahan kernel crash).
    - Logit masking untuk menekan unused & vision tokens (<unused1>..<unused6> dikecualikan).
    - `torch.compile` di-no-op-kan SEBELUM unsloth di-import (fix hard-crash fullgraph).
    - GrokAdEMAMix optimizer (split-LR di fase vision), SelectiveLabelSmoother bersama.
    """)
    return


@app.cell
def _():
    import os
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
    # Matikan auto torch.compile bawaan Unsloth (unsloth_zoo membungkus forward
    # T5Gemma2 dengan @torch.compile(fullgraph=True, dynamic=True, ...)). Dengan
    # fullgraph=True, begitu recompile limit kena, itu SELALU hard-crash tanpa
    # ada config yang bisa menyelamatkan (fail_on_recompile_limit_hit /
    # suppress_errors tidak berlaku untuk fullgraph). OOM sudah ditangani oleh
    # expandable_segments di atas + gradient checkpointing "unsloth", jadi
    # compile ini murni untuk speed, bukan untuk mencegah OOM -> aman dimatikan.
    os.environ["TORCH_COMPILE_DISABLE"] = "1"
    import re, json, torch, random, datetime, gc, traceback
    import warnings
    warnings.filterwarnings("ignore")

    # Belt-and-suspenders di atas TORCH_COMPILE_DISABLE: env var itu ternyata
    # TIDAK reliable mematikan compile Unsloth untuk T5Gemma2 di torch 2.12.1 --
    # traceback masih lewat unsloth_compiled_module_t5gemma2.py & torch._dynamo
    # meski env var sudah diset. Monkeypatch torch.compile jadi no-op di sini --
    # SEBELUM unsloth di-import dan SEBELUM FastLanguageModel/FastVisionModel
    # .from_pretrained() memicu Unsloth membungkus forward T5Gemma2 dengan
    # @torch.compile(fullgraph=True, ...).
    def _torch_compile_noop(model=None, *args, **kwargs):
        if model is not None:
            return model
        return lambda fn: fn
    setattr(torch, "compile", _torch_compile_noop)
    import torch.nn.functional as F
    # Naikkan recompile_limit jauh di atas jumlah modul (belt-and-suspenders).
    setattr(torch._dynamo.config, "recompile_limit", 1024)
    setattr(torch._dynamo.config, "cache_size_limit", 1024)
    from PIL import Image
    from unsloth import FastLanguageModel, FastVisionModel
    from datasets import Dataset, load_dataset
    from transformers import (
        AutoProcessor, AutoTokenizer,
        Seq2SeqTrainer, Seq2SeqTrainingArguments,
        DataCollatorForSeq2Seq, PreTrainedTokenizerFast,
        get_scheduler,
        TrainerCallback, TrainerControl, TrainerState, TrainingArguments,
    )
    from typing import Any, cast

    import numpy as np
    import matplotlib.pyplot as plt

    # Optional imports for evaluation metrics
    try:
        import evaluate

        rouge_metric = evaluate.load("rouge")
        bleu_metric = evaluate.load("bleu")
        exact_match_metric = evaluate.load("exact_match")
        bertscore_metric = evaluate.load("bertscore")
        meteor_metric = evaluate.load("meteor")
    except Exception as e:
        print(
            f"Warning: evaluate metrics not available. Metric evaluation will be bypassed. Error: {e}"
        )
        rouge_metric = None
        bleu_metric = None
        exact_match_metric = None
        bertscore_metric = None
        meteor_metric = None

    # ---- LOGIT MASKING (decoder lm_head, shared text & vision) ----
    def apply_logit_mask(model, suppress_ids):
        vs = model.config.vocab_size
        sl = [i for i in suppress_ids if i < vs]
        mask = torch.zeros(vs, dtype=torch.bfloat16)
        mask[sl] = -10000.0
        def hook(mod, inp, out):
            if isinstance(out, torch.Tensor):
                return out + mask.to(out.device)
            elif hasattr(out, "logits"):
                out.logits = out.logits + mask.to(out.logits.device)
                return out
            elif isinstance(out, tuple) and out and isinstance(out[0], torch.Tensor):
                return (out[0] + mask.to(out[0].device),) + out[1:]
            return out
        t = None
        if hasattr(model, "lm_head"):
            t = model.lm_head
        elif hasattr(model, "base_model") and hasattr(model.base_model, "lm_head"):
            t = model.base_model.lm_head
        elif hasattr(model, "base_model") and hasattr(model.base_model, "model") and hasattr(model.base_model.model, "lm_head"):
            t = model.base_model.model.lm_head
        if t is not None:
            t.register_forward_hook(hook)
            print(f"  ✅ Logit mask (lm_head) untuk {len(sl)} tokens.")
        else:
            model.register_forward_hook(hook)
            print(f"  ✅ Logit mask (fallback) untuk {len(sl)} tokens.")

    return (
        Any,
        AutoProcessor,
        AutoTokenizer,
        DataCollatorForSeq2Seq,
        Dataset,
        F,
        FastLanguageModel,
        FastVisionModel,
        Image,
        PreTrainedTokenizerFast,
        Seq2SeqTrainer,
        Seq2SeqTrainingArguments,
        TrainerCallback,
        TrainerControl,
        TrainerState,
        TrainingArguments,
        apply_logit_mask,
        bertscore_metric,
        bleu_metric,
        cast,
        datetime,
        exact_match_metric,
        gc,
        get_scheduler,
        load_dataset,
        meteor_metric,
        np,
        os,
        plt,
        random,
        re,
        rouge_metric,
        torch,
        traceback,
    )


# =====================================================================
# SHARED CONSTANTS (identik di pipeline text & vision)
# =====================================================================
@app.cell
def _(torch):
    # SYSTEM PROMPT FALLBACK (dipakai format_encoder_from_raw — string identik
    # di kedua notebook; versi vision hardcode string yang sama)
    SYSTEM_PROMPT = (
        "Kamu adalah asisten AI yang helpful, santai, dan ramah. "
        "Gunakan Bahasa Indonesia sebagai bahasa utama."
    )

    # Token IDs yang harus di-suppress (unused + vision)
    # Pengecualian: <unused1> sampai <unused6> (ID 7 hingga 12) digunakan untuk Task Prefix
    SUPPRESS_BLOCK1 = [6] + list(range(13, 105))
    SUPPRESS_BLOCK2 = list(range(256002, 262144))
    SUPPRESS_VISION = [255999, 256000, 256001]
    ALL_SUPPRESS_IDS = set(SUPPRESS_BLOCK1 + SUPPRESS_BLOCK2 + SUPPRESS_VISION)

    # Shared seed & precision flag
    SEED = 3407
    BF16 = torch.cuda.is_available()
    return (
        ALL_SUPPRESS_IDS,
        BF16,
        SEED,
        SUPPRESS_BLOCK1,
        SUPPRESS_BLOCK2,
        SUPPRESS_VISION,
        SYSTEM_PROMPT,
    )


# =====================================================================
# SHARED UTILS: chat formatter (identik di kedua notebook)
# =====================================================================
@app.cell
def _(SYSTEM_PROMPT, re):
    def format_encoder_from_raw(raw_input: str) -> str:
        system_match = re.search(r"^system:\s*(.*?)(?=\nuser:)", raw_input, re.DOTALL)
        system = system_match.group(1).strip() if system_match else SYSTEM_PROMPT

        if system_match:
            raw_input = raw_input[system_match.end() :].strip()

        parts = re.split(r"\n(user:|assistant:)\s*", "\n" + raw_input)
        formatted = ""
        is_first_user = True

        for i in range(1, len(parts), 2):
            role = parts[i].replace(":", "").strip()
            content = parts[i + 1].strip()
            if not content:
                continue

            if role == "user":
                formatted += "<start_of_turn>user\n"
                if is_first_user and system:
                    formatted += system + "\n\n"
                    is_first_user = False
                formatted += content + "<end_of_turn>\n"
            elif role == "assistant":
                formatted += "<start_of_turn>model\n"
                formatted += content + "<end_of_turn>\n"

        formatted += "<start_of_turn>model\n"
        return formatted

    return (format_encoder_from_raw,)


# =====================================================================
# SHARED OPTIMIZER (versi vision — dengan dtype-cast fix untuk mixed
# 4bit/BF16 params; no-op untuk LoRA murni di fase text)
# =====================================================================
@app.cell
def _(torch):
    class GrokAdEMAMix(torch.optim.Optimizer):
        def __init__(
            self,
            params,
            lr=3e-5,
            betas=(0.9, 0.999),
            beta3=0.9999,
            weight_decay=0.05,
            grok_alpha=2.0,
            grok_lamb=0.98,
        ):
            defaults = dict(
                lr=lr,
                betas=betas,
                beta3=beta3,
                weight_decay=weight_decay,
                grok_alpha=grok_alpha,
                grok_lamb=grok_lamb,
            )
            super().__init__(params, defaults)
            self.step_count = 0

        @torch.no_grad()
        def step(self, closure=None):
            loss = None
            if closure is not None:
                with torch.enable_grad():
                    loss = closure()

            self.step_count += 1

            for group in self.param_groups:
                lr = group["lr"]
                beta1, beta2 = group["betas"]
                beta3 = group["beta3"]
                weight_decay = group["weight_decay"]
                grok_alpha = group["grok_alpha"]
                grok_lamb = group["grok_lamb"]

                for p in group["params"]:
                    if p.grad is None:
                        continue

                    grad = p.grad
                    state = self.state[p]

                    if len(state) == 0:
                        state["step"] = 0
                        state["grok_slow_grad"] = torch.zeros_like(grad)
                        state["m"] = torch.zeros_like(grad)
                        state["v"] = torch.zeros_like(grad)
                        state["n"] = torch.zeros_like(grad)

                    state["step"] += 1
                    step = state["step"]

                    # GROKFAST
                    state["grok_slow_grad"].mul_(grok_lamb).add_(
                        grad, alpha=1.0 - grok_lamb
                    )
                    filtered_grad = grad.clone()
                    filtered_grad.add_(state["grok_slow_grad"], alpha=grok_alpha)

                    if weight_decay != 0:
                        p.data.mul_(1.0 - lr * weight_decay)

                    # ADEMAMIX
                    m, v, n = state["m"], state["v"], state["n"]

                    m.mul_(beta1).add_(filtered_grad, alpha=1.0 - beta1)
                    v.mul_(beta2).addcmul_(
                        filtered_grad, filtered_grad, value=1.0 - beta2
                    )
                    n.mul_(beta3).add_(filtered_grad, alpha=1.0 - beta3)

                    bias_correction1 = 1.0 - beta1**step
                    bias_correction2 = 1.0 - beta2**step
                    bias_correction3 = 1.0 - beta3**step

                    denom = (v.sqrt() / (bias_correction2**0.5)).add_(1e-8)
                    denom = denom.to(p.dtype)
                    step_update = (
                        m / bias_correction1 + 0.1 * n / bias_correction3
                    ) / denom
                    step_update = step_update.to(p.dtype)

                    p.data.add_(step_update, alpha=-lr)
            return loss

    return (GrokAdEMAMix,)


# =====================================================================
# SHARED LABEL SMOOTHER (versi text — chunked + del untuk hemat memori;
# dipakai oleh trainer text dan trainer vision)
# =====================================================================
@app.cell
def _(torch):
    class SelectiveLabelSmoother:
        def __init__(self, epsilon, suppress_ids):
            self.epsilon = epsilon
            self.suppress_ids = suppress_ids

        def __call__(self, model_output, labels, shift_labels=False):
            if isinstance(model_output, dict) and "logits" in model_output:
                logits = model_output["logits"]
            elif isinstance(model_output, tuple):
                logits = (
                    model_output[1] if len(model_output) > 1 else model_output[0].logits
                )
            else:
                logits = model_output.logits

            if shift_labels:
                logits = logits[..., :-1, :].contiguous()
                labels = labels[..., 1:].contiguous()

            vocab_size = logits.size(-1)
            suppress_list = [i for i in self.suppress_ids if i < vocab_size]

            valid_mask = torch.ones(vocab_size, dtype=torch.bool, device=logits.device)
            valid_mask[suppress_list] = False
            num_valid_tokens = valid_mask.sum().item()

            flat_logits = logits.view(-1, vocab_size)
            flat_labels = labels.view(-1)

            active_mask = flat_labels != -100
            if active_mask.sum() == 0:
                return torch.tensor(0.0, device=logits.device, requires_grad=True)

            active_logits = flat_logits[active_mask]
            active_labels = flat_labels[active_mask]

            num_active = active_logits.size(0)
            chunk_size = 2048

            total_loss = torch.tensor(0.0, device=logits.device)

            for i in range(0, num_active, chunk_size):
                chunk_logits = active_logits[i : i + chunk_size]
                chunk_labels = active_labels[i : i + chunk_size]

                log_probs = torch.nn.functional.log_softmax(chunk_logits, dim=-1)

                nll_loss = -log_probs.gather(
                    dim=-1, index=chunk_labels.unsqueeze(-1)
                ).squeeze(-1)

                valid_log_probs = log_probs * valid_mask.to(log_probs.dtype)
                smooth_loss = -valid_log_probs.sum(dim=-1) / num_valid_tokens

                token_losses = (1.0 - self.epsilon) * nll_loss + self.epsilon * smooth_loss
                total_loss += token_losses.sum()

                del chunk_logits, chunk_labels, log_probs, nll_loss, valid_log_probs, smooth_loss, token_losses

            return total_loss / num_active

    return (SelectiveLabelSmoother,)


@app.cell
def _(mo):
    # Create a secure token input
    hf_token_input = mo.ui.text(
        label="Hugging Face Token (HF_TOKEN)", value="", full_width=True
    )
    hf_token_input
    return (hf_token_input,)


@app.cell
def _(hf_token_input, mo, os):
    from huggingface_hub import login

    # Stop execution of this cell if no token is entered yet
    mo.stop(
        not hf_token_input.value,
        mo.md(
            "⚠️ *Please enter your Hugging Face token in the input above to authenticate and load gated models.*"
        ),
    )

    try:
        # Set the environment variable so transformers/datasets can find it
        os.environ["HF_TOKEN"] = hf_token_input.value
        login(token=hf_token_input.value)
        status = mo.md(
            "✅ **Successfully authenticated with Hugging Face Hub!** You can now load gated models."
        )
    except Exception as e:
        status = mo.md(f"❌ **Authentication failed:** {e}")

    status
    return


# #####################################################################
# #####################################################################
#
#   ██████╗ ██╗  ██╗ █████╗ ███████╗███████╗     ██╗
#   ██╔══██╗██║  ██║██╔══██╗██╔════╝██╔════╝    ███║
#   ██████╔╝███████║███████║███████╗█████╗      ╚██║
#   ██╔═══╝ ██╔══██║██╔══██║╚════██║██╔══╝       ██║
#   ██║     ██║  ██║██║  ██║███████║███████╗     ██║
#   ╚═╝     ╚═╝  ╚═╝╚═╝  ╚═╝╚══════╝╚══════╝     ╚═╝
#
#   TEXT PIPELINE  (dari working-molab-v6-unsloth.py — logika identik)
#   Base: google/t5gemma-2-4b-4b  ->  SFT  ->  ORPO  ->  merge
#   Artifacts: UNIFIED_HF_REPO subfolder text/
#
# #####################################################################
# #####################################################################
@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ---
    # 📝 PHASE 1 — Multi-task SFT + ORPO Training: T5Gemma-2 Text Pipeline (V6 Unsloth)
    =====================================================================
    Melatih model **T5Gemma-2 4B-4B** secara berurutan:
    1. **SFT** — Supervised Fine-Tuning dengan LoRA berbasis Unsloth
    2. **ORPO** — Odds Ratio Preference Optimization di atas hasil SFT
       *(perlu re-run notebook: stage terdeteksi ulang dari HF Hub)*

    - Auto-detect progress dari HF Hub — otomatis lanjut dari checkpoint terakhir
    - Upload checkpoint ke HF segera setelah disimpan (tahan kernel crash)
    - 1 repo HF dengan subfolder `sft/` dan `orpo/`
    """)
    return


# =====================================================================
# UNIFIED HF REPO — 1 repo PUBLIK untuk SEMUA artifacts pipeline
# =====================================================================
@app.cell
def _():
    # Menggantikan 3 repo lama:
    #   - daruokta/t5gemma-2-4b-4b-instruct-chat-indo-v4-unsloth        (text)
    #   - daruokta/t5gemma-2-4b-4b-instruct-chat-indo-v4-vision-cangkok (cangkok)
    #   - daruokta/t5gemma-2-4b-4b-instruct-chat-indo-v4-vision-enhanced (vision)
    #
    # Struktur subfolder di dalam repo ini:
    #   text/            Phase 1: sft/, orpo/, merged_bf16/, quantized_4bit/
    #   cangkok/         Phase 1.5: full model graft (config+weights+processor+tokenizer)
    #   vision/          Phase 2: sft/, orpo/, merged_bf16/, quantized_4bit/
    UNIFIED_HF_REPO = "daruokta/t5gemma-2-4b-4b-instruct-chat-indo-v6-combined-unsloth"
    return (UNIFIED_HF_REPO,)


# =====================================================================
# KONFIGURASI HYPERPARAMETER TEXT (TERPUSAT & MUDAH DIUBAH)
# =====================================================================
@app.cell
def _(UNIFIED_HF_REPO):
    # MODEL CONFIG
    TEXT_MODEL_NAME = "google/t5gemma-2-4b-4b"
    TEXT_LOAD_IN_4BIT = True  # True = QLoRA (hemat VRAM), False = BF16 (perlu VRAM besar)
    TEXT_OUTPUT_DIR = "results/t5gemma2"  # Base dir — subfolder sft/ dan orpo/ otomatis

    # HUGGING FACE HUB CONFIG (1 repo unified — artifacts di bawah prefix text/)
    TEXT_HF_REPO_ID = "daruokta/t5gemma2-indonesia-chat-formatted"  # Dataset source (tetap repo dataset)
    TEXT_HF_CHECKPOINT_REPO = UNIFIED_HF_REPO  # Training artifacts -> subfolder text/
    TEXT_HF_PREFIX = "text"

    # ORPO CONFIG
    TEXT_ORPO_BETA = 0.1

    # Dataset Subsets (Configs)
    TEXT_CHAT_CONFIG = "chat_sft"
    TEXT_INDOQA_CONFIG = "indoqa_sft"
    TEXT_ORPO_CONFIG = "chat_orpo"

    # SAMPLE SIZES (Set ke 0 untuk mengambil seluruh data)
    TEXT_SAMPLE_TRAIN_CHAT = 0
    TEXT_SAMPLE_TRAIN_INDOQA = 0
    TEXT_SAMPLE_TRAIN_ORPO = 0
    TEXT_SAMPLE_VAL_CHAT = 0
    TEXT_SAMPLE_VAL_INDOQA = 0

    # GENERATION EVALUATION CONFIG
    TEXT_SAMPLE_EVAL_GENERATION = 100
    TEXT_EVAL_EVERY_N_STEPS = 200

    # BASIC TRAINING SPECS
    TEXT_MAX_SOURCE_LENGTH = 16384
    TEXT_MAX_TARGET_LENGTH = 2048
    TEXT_NUM_EPOCHS_SFT = 4
    TEXT_NUM_EPOCHS_ORPO = 2
    TEXT_LEARNING_RATE = 1e-5

    # BATCH SIZE & ACCUMULATION
    TEXT_PER_DEVICE_TRAIN_BATCH_SIZE = 2
    TEXT_PER_DEVICE_EVAL_BATCH_SIZE = 8
    TEXT_GRADIENT_ACCUMULATION_STEPS = 64
    TEXT_EVAL_ACCUMULATION_STEPS = None

    # LoRA CONFIG SPECS
    TEXT_LORA_RANK = 256
    TEXT_LORA_ALPHA = 512
    TEXT_LORA_DROPOUT = 0.2
    TEXT_LORA_TARGET_MODULES = [
        "q_proj",
        "k_proj",
        "v_proj",
        "o_proj",
        "gate_proj",
        "up_proj",
        "down_proj",
    ]

    # ADVANCED TRAINING ARGUMENTS
    TEXT_WARMUP_STEPS = 200
    TEXT_WEIGHT_DECAY = 0.1
    TEXT_LR_SCHEDULER_TYPE = "cosine"
    TEXT_LOGGING_STEPS = 100
    TEXT_SAVE_TOTAL_LIMIT = 2  # Lokal saja — di HF semua checkpoint tetap ada
    TEXT_OPTIM = "paged_adamw_8bit"
    TEXT_LABEL_SMOOTHING_FACTOR = 0.1
    TEXT_NEFTUNE_NOISE_ALPHA = 5.0

    # HARDWARE & CONTROL SPECS
    TEXT_GRADIENT_CHECKPOINTING = True
    TEXT_FP16 = False
    TEXT_PREDICT_WITH_GENERATE = True
    TEXT_EARLY_STOPPING_PATIENCE = 8

    # EVALUATION GENERATION BEHAVIOR CONFIG
    TEXT_GEN_TEMPERATURE = 0.7
    TEXT_GEN_TOP_P = 0.9
    TEXT_GEN_REPETITION_PENALTY = 1.2
    return (
        TEXT_CHAT_CONFIG,
        TEXT_EARLY_STOPPING_PATIENCE,
        TEXT_EVAL_ACCUMULATION_STEPS,
        TEXT_EVAL_EVERY_N_STEPS,
        TEXT_FP16,
        TEXT_GEN_REPETITION_PENALTY,
        TEXT_GEN_TEMPERATURE,
        TEXT_GEN_TOP_P,
        TEXT_GRADIENT_ACCUMULATION_STEPS,
        TEXT_GRADIENT_CHECKPOINTING,
        TEXT_HF_CHECKPOINT_REPO,
        TEXT_HF_PREFIX,
        TEXT_HF_REPO_ID,
        TEXT_INDOQA_CONFIG,
        TEXT_LABEL_SMOOTHING_FACTOR,
        TEXT_LEARNING_RATE,
        TEXT_LOAD_IN_4BIT,
        TEXT_LOGGING_STEPS,
        TEXT_LORA_ALPHA,
        TEXT_LORA_DROPOUT,
        TEXT_LORA_RANK,
        TEXT_LORA_TARGET_MODULES,
        TEXT_LR_SCHEDULER_TYPE,
        TEXT_MAX_SOURCE_LENGTH,
        TEXT_MAX_TARGET_LENGTH,
        TEXT_MODEL_NAME,
        TEXT_NEFTUNE_NOISE_ALPHA,
        TEXT_NUM_EPOCHS_ORPO,
        TEXT_NUM_EPOCHS_SFT,
        TEXT_OPTIM,
        TEXT_ORPO_BETA,
        TEXT_ORPO_CONFIG,
        TEXT_OUTPUT_DIR,
        TEXT_PER_DEVICE_EVAL_BATCH_SIZE,
        TEXT_PER_DEVICE_TRAIN_BATCH_SIZE,
        TEXT_PREDICT_WITH_GENERATE,
        TEXT_SAMPLE_EVAL_GENERATION,
        TEXT_SAMPLE_TRAIN_CHAT,
        TEXT_SAMPLE_TRAIN_INDOQA,
        TEXT_SAMPLE_TRAIN_ORPO,
        TEXT_SAMPLE_VAL_CHAT,
        TEXT_SAMPLE_VAL_INDOQA,
        TEXT_SAVE_TOTAL_LIMIT,
        TEXT_WARMUP_STEPS,
        TEXT_WEIGHT_DECAY,
    )


# =====================================================================
# TEXT: AUTO-DETECT PIPELINE STAGE DARI HF HUB
# =====================================================================
@app.cell
def _(TEXT_HF_CHECKPOINT_REPO, TEXT_HF_PREFIX, mo, os):
    from huggingface_hub import HfApi as _StageDetectApi

    _hf_token = os.environ.get("HF_TOKEN")
    _api = _StageDetectApi(token=_hf_token)

    # Default
    text_current_stage = "sft"
    text_resume_checkpoint = None

    try:
        if _api.repo_exists(repo_id=TEXT_HF_CHECKPOINT_REPO):
            _repo_files = _api.list_repo_files(TEXT_HF_CHECKPOINT_REPO)

            # Cek apakah ORPO sudah selesai
            if any(f.startswith(f"{TEXT_HF_PREFIX}/orpo/final_adapter/") for f in _repo_files):
                text_current_stage = "done"
                print("📍 [TEXT] Pipeline stage: DONE — Semua training selesai!")

            # Cek apakah SFT sudah selesai → lanjut ORPO
            elif any(f.startswith(f"{TEXT_HF_PREFIX}/sft/final_adapter/") for f in _repo_files):
                text_current_stage = "orpo"
                # Ada checkpoint ORPO untuk resume?
                _orpo_ckpts = sorted([
                    f for f in _repo_files
                    if f.startswith(f"{TEXT_HF_PREFIX}/orpo/checkpoint-") and "/" in f[len(f"{TEXT_HF_PREFIX}/orpo/checkpoint-"):]
                ])
                if _orpo_ckpts:
                    text_resume_checkpoint = True
                    print(f"📍 [TEXT] Pipeline stage: ORPO (resume dari checkpoint)")
                else:
                    print("📍 [TEXT] Pipeline stage: ORPO (mulai dari awal, load SFT adapter)")

            # SFT belum selesai
            else:
                text_current_stage = "sft"
                _sft_ckpts = sorted([
                    f for f in _repo_files
                    if f.startswith(f"{TEXT_HF_PREFIX}/sft/checkpoint-") and "/" in f[len(f"{TEXT_HF_PREFIX}/sft/checkpoint-"):]
                ])
                if _sft_ckpts:
                    text_resume_checkpoint = True
                    print(f"📍 [TEXT] Pipeline stage: SFT (resume dari checkpoint)")
                else:
                    print("📍 [TEXT] Pipeline stage: SFT (mulai dari awal)")
        else:
            print(f"📍 [TEXT] Repo '{TEXT_HF_CHECKPOINT_REPO}' belum ada. Mulai SFT dari awal.")
            # Buat repo (PUBLIK — sesuai permintaan: 1 repo publik untuk semua)
            _api.create_repo(repo_id=TEXT_HF_CHECKPOINT_REPO, repo_type="model", private=False, exist_ok=True)
    except Exception as e:
        print(f"⚠️ Gagal mendeteksi pipeline stage TEXT: {e}. Mulai SFT dari awal.")

    mo.md(f"**📍 [TEXT] Current Stage: `{text_current_stage}`** | Resume: `{text_resume_checkpoint}`")
    return text_current_stage, text_resume_checkpoint


# =====================================================================
# TEXT UTILITY: dataset sample loader (grouped by chat_idx)
# =====================================================================
@app.cell
def _(load_dataset, random):
    def load_hf_samples(
        repo_id: str, config_name: str, split: str, n_samples: int, seed: int = 42
    ) -> list[dict]:
        """
        Mendownload dataset dari Hugging Face Hub untuk split tertentu dan mengambil sampel sejumlah n_samples.
        """
        print(f"Mengunduh dataset '{config_name}' ({split}) dari {repo_id}...")
        try:
            ds = load_dataset(repo_id, config_name, split=split)
            samples = [dict(row) for row in ds]

            if n_samples > 0 and len(samples) > n_samples:
                random.seed(seed)
                if samples and "chat_idx" in samples[0]:
                    # Group by chat_idx
                    groups = {}
                    for s in samples:
                        c_idx = s["chat_idx"]
                        if c_idx not in groups:
                            groups[c_idx] = []
                        groups[c_idx].append(s)
                    # Shuffle the groups
                    group_keys = list(groups.keys())
                    random.shuffle(group_keys)

                    selected_samples = []
                    for k in group_keys:
                        selected_samples.extend(groups[k])
                        if len(selected_samples) >= n_samples:
                            break
                    return selected_samples
                else:
                    return random.sample(samples, n_samples)
            return samples
        except Exception as e:
            print(f"[ERROR] Gagal mengunduh dataset {config_name} ({split}): {e}")
            return []

    return (load_hf_samples,)


# =====================================================================
# TEXT CALLBACKS: Training Plot, Sample Generation, Hub Upload
# =====================================================================
@app.cell
def _(
    Any,
    TrainerCallback,
    TrainerControl,
    TrainerState,
    TrainingArguments,
    os,
    plt,
):
    class TextTrainingPlotCallback(TrainerCallback):
        def __init__(self, output_dir: str) -> None:
            self.output_dir = output_dir
            self.train_steps: list[int] = []
            self.train_losses: list[float] = []
            self.eval_steps: list[int] = []
            self.eval_losses: list[float] = []
            self.eval_rougeL: list[float] = []
            self.eval_bleu: list[float] = []
            self.eval_meteor: list[float] = []
            self.eval_bertscore: list[float] = []
            self.eval_perplexity: list[float] = []
            self.chart_path = os.path.join(output_dir, "training_chart.png")

        def on_log(
            self,
            args: TrainingArguments,
            state: TrainerState,
            control: TrainerControl,
            logs: dict[str, float] | None = None,
            **kwargs: Any,
        ) -> None:
            if logs is None:
                return
            if "loss" in logs:
                self.train_steps.append(state.global_step)
                actual_loss = float(logs["loss"])
                self.train_losses.append(actual_loss)
            if "eval_loss" in logs:
                self.eval_steps.append(state.global_step)
                self.eval_losses.append(float(logs["eval_loss"]))
            if "eval_rougeL" in logs:
                self.eval_rougeL.append(float(logs["eval_rougeL"]))
            if "eval_bleu" in logs:
                self.eval_bleu.append(float(logs["eval_bleu"]))
            if "eval_meteor" in logs:
                self.eval_meteor.append(float(logs["eval_meteor"]))
            if "eval_bertscore_f1" in logs:
                self.eval_bertscore.append(float(logs["eval_bertscore_f1"]))
            if "eval_perplexity" in logs:
                self.eval_perplexity.append(float(logs["eval_perplexity"]))
            self._save_chart()

        def _save_chart(self) -> None:
            if len(self.train_steps) < 2 and len(self.eval_steps) < 1:
                return

            has_metrics = len(self.eval_rougeL) > 0 or len(self.eval_bleu) > 0

            if has_metrics:
                fig, axs = plt.subplots(2, 2, figsize=(16, 10))
                ax1, ax2, ax3, ax4 = axs[0, 0], axs[0, 1], axs[1, 0], axs[1, 1]
            else:
                fig, ax1 = plt.subplots(figsize=(10, 4))
                ax2 = ax3 = ax4 = None

            if self.train_losses:
                ax1.plot(self.train_steps, self.train_losses, color="#4A90D9", linewidth=1.5, label="Train Loss")
                if len(self.train_losses) >= 10:
                    window = 10
                    ma = [
                        sum(self.train_losses[max(0, i - window) : i + 1])
                        / len(self.train_losses[max(0, i - window) : i + 1])
                        for i in range(len(self.train_losses))
                    ]
                    ax1.plot(self.train_steps, ma, color="#E74C3C", linewidth=2, label="Train Loss (MA-10)", alpha=0.8)

            if self.eval_losses:
                ax1.plot(self.eval_steps, self.eval_losses, color="#2ECC71", marker="o", linestyle="--", linewidth=1.5, label="Eval Loss")

            ax1.set_xlabel("Steps")
            ax1.set_ylabel("Loss")
            ax1.set_title("Training & Evaluation Loss Curve")
            ax1.grid(True, alpha=0.3)
            ax1.legend()

            if has_metrics and ax2 is not None and ax3 is not None and ax4 is not None:
                if len(self.eval_rougeL) > 0:
                    ax2.plot(self.eval_steps, self.eval_rougeL, color="#9B59B6", marker="s", linestyle="-", linewidth=2, label="Eval ROUGE-L")
                if len(self.eval_bleu) > 0:
                    ax2.plot(self.eval_steps, self.eval_bleu, color="#E67E22", marker="^", linestyle="-", linewidth=2, label="Eval BLEU")
                if len(self.eval_meteor) > 0:
                    ax2.plot(self.eval_steps, self.eval_meteor, color="#F1C40F", marker="D", linestyle="-", linewidth=2, label="Eval METEOR")
                ax2.set_xlabel("Steps")
                ax2.set_ylabel("Score (%)")
                ax2.set_title("NLG Metrics (ROUGE-L, BLEU, METEOR)")
                ax2.grid(True, alpha=0.3)
                ax2.legend()

                if len(self.eval_bertscore) > 0:
                    ax3.plot(self.eval_steps, self.eval_bertscore, color="#E74C3C", marker="p", linestyle="-", linewidth=2, label="Eval BERTScore")
                    ax3.set_xlabel("Steps")
                    ax3.set_ylabel("Score (%)")
                    ax3.set_title("Semantic Metrics (BERTScore F1)")
                    ax3.grid(True, alpha=0.3)
                    ax3.legend()

                if len(self.eval_perplexity) > 0:
                    ax4.plot(self.eval_steps, self.eval_perplexity, color="#34495E", marker="h", linestyle="-", linewidth=2, label="Eval Perplexity")
                    ax4.set_xlabel("Steps")
                    ax4.set_ylabel("Perplexity")
                    ax4.set_title("Model Perplexity Curve")
                    ax4.grid(True, alpha=0.3)
                    ax4.legend()

            plt.tight_layout()
            plt.savefig(self.chart_path, dpi=120)
            plt.show()
            plt.close(fig)

    return (TextTrainingPlotCallback,)


@app.cell
def _(
    Any,
    PreTrainedTokenizerFast,
    TrainerCallback,
    TrainerControl,
    TrainerState,
    TrainingArguments,
    datetime,
    os,
    torch,
):
    class TextSampleGenerationCallback(TrainerCallback):
        def __init__(
            self,
            tokenizer: PreTrainedTokenizerFast,
            eval_samples: list[dict],
            output_dir: str,
            eval_every_n_steps: int = 50,
            temperature: float = 0.7,
            top_p: float = 0.9,
            repetition_penalty: float = 1.2,
            bad_words_ids: list[list[int]] | None = None,
        ) -> None:
            self.tokenizer = tokenizer
            self.eval_samples = eval_samples
            self.output_dir = output_dir
            self.eval_every_n_steps = eval_every_n_steps
            self.log_path = os.path.join(output_dir, "eval_samples.txt")
            self._eot_id = tokenizer.convert_tokens_to_ids("<end_of_turn>")
            self._eos_id = tokenizer.eos_token_id or 1
            self._stop_ids = list({self._eot_id, self._eos_id})
            self.temperature = temperature
            self.top_p = top_p
            self.repetition_penalty = repetition_penalty
            self.bad_words_ids = bad_words_ids

        def on_step_end(
            self,
            args: TrainingArguments,
            state: TrainerState,
            control: TrainerControl,
            model: Any = None,
            **kwargs: Any,
        ) -> None:
            if (
                state.global_step == 0
                or state.global_step % self.eval_every_n_steps != 0
            ):
                return
            if model is None:
                return

            from unsloth import FastLanguageModel
            if hasattr(FastLanguageModel, "for_inference"):
                FastLanguageModel.for_inference(model)
            else:
                model.eval()
            timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            lines = [
                f"\n{'=' * 60}",
                f"Step {state.global_step} | {timestamp}",
                f"{'=' * 60}",
            ]

            import gc
            gc.collect()
            torch.cuda.empty_cache()

            with torch.no_grad():
                pad_id = (
                    self.tokenizer.pad_token_id
                    if self.tokenizer.pad_token_id is not None
                    else self._eos_id
                )

                for idx, sample in enumerate(self.eval_samples):
                    input_tensor = torch.tensor([sample["input_ids"]], dtype=torch.long).to(model.device)
                    attention_mask = torch.ones_like(input_tensor).to(model.device)

                    outputs = getattr(model, "generate")(
                        input_ids=input_tensor,
                        attention_mask=attention_mask,
                        max_new_tokens=256,
                        do_sample=True,
                        temperature=self.temperature,
                        top_p=self.top_p,
                        repetition_penalty=self.repetition_penalty,
                        eos_token_id=self._stop_ids,
                        pad_token_id=pad_id,
                        bad_words_ids=self.bad_words_ids,
                    )

                    raw_query = self.tokenizer.decode(
                        sample["input_ids"], skip_special_tokens=True
                    )
                    query = (
                        raw_query.strip()
                        if isinstance(raw_query, str)
                        else "".join(raw_query).strip()
                    )

                    raw_target = self.tokenizer.decode(
                        sample["labels"], skip_special_tokens=True
                    )
                    target = (
                        raw_target.strip()
                        if isinstance(raw_target, str)
                        else "".join(raw_target).strip()
                    )

                    # Extract generated tokens only (skip prompt tokens if auto-included)
                    gen_ids = outputs[0]
                    # Note: Unsloth fast generate might return only new tokens or full tokens, decoder depends.
                    # Usually decode handles it if we strip.
                    raw_response = self.tokenizer.decode(
                        gen_ids, skip_special_tokens=True
                    )

                    # Remove the prompt if it is echoed back
                    if raw_response.startswith(query):
                        raw_response = raw_response[len(query):].strip()

                    response = (
                        raw_response.strip()
                        if isinstance(raw_response, str)
                        else "".join(raw_response).strip()
                    )

                    words = response.split()
                    is_repetitive = (
                        len(set(words)) < max(1, len(words) * 0.3) if words else True
                    )
                    flag = " ⚠️ REPETITIVE" if is_repetitive else " ✅"

                    lines.append(f"\nQ: {query}")
                    lines.append(f"Expected Target: {target}")
                    lines.append(f"Model Response: {response}{flag}")

            from unsloth import FastLanguageModel
            if hasattr(FastLanguageModel, "for_training"):
                FastLanguageModel.for_training(model)
            else:
                model.train()

            # Clean up after generation
            gc.collect()
            torch.cuda.empty_cache()

            with open(self.log_path, "a", encoding="utf-8") as f:
                f.write("\n".join(lines) + "\n")

            print(f"\n[BEHAVIOR EVAL @ step {state.global_step}]")
            for line in lines[3:]:
                if (
                    line.startswith("Q:")
                    or line.startswith("Model Response:")
                    or line.startswith("Expected Target:")
                ):
                    print(f"  {line}")

    return (TextSampleGenerationCallback,)


@app.cell
def _(
    Any,
    TrainerCallback,
    TrainerControl,
    TrainerState,
    TrainingArguments,
    os,
):
    class TextHubUploadCallback(TrainerCallback):
        """Upload setiap checkpoint TEXT ke HF Hub segera setelah disimpan."""

        def __init__(self, repo_id: str, stage: str, token: str, output_dir: str) -> None:
            self.repo_id = repo_id
            self.stage = stage  # "sft" atau "orpo"
            self.token = token
            self.output_dir = output_dir

        def on_save(
            self,
            args: TrainingArguments,
            state: TrainerState,
            control: TrainerControl,
            **kwargs: Any,
        ) -> None:
            from huggingface_hub import HfApi as _SaveApi

            _api = _SaveApi(token=self.token)
            checkpoint_name = f"checkpoint-{state.global_step}"
            local_path = os.path.join(self.output_dir, checkpoint_name)

            if os.path.exists(local_path):
                try:
                    print(f"\n📤 Uploading {checkpoint_name} to HF {self.stage}/...")
                    _api.upload_folder(
                        folder_path=local_path,
                        path_in_repo=f"{self.stage}/{checkpoint_name}",
                        repo_id=self.repo_id,
                    )
                    # Upload juga training chart dan eval log jika ada
                    for artifact_name in ["training_chart.png", "eval_samples.txt"]:
                        artifact_path = os.path.join(self.output_dir, artifact_name)
                        if os.path.exists(artifact_path):
                            _api.upload_file(
                                path_or_fileobj=artifact_path,
                                path_in_repo=f"{self.stage}/{artifact_name}",
                                repo_id=self.repo_id,
                            )
                    print(f"✅ {checkpoint_name} + artifacts uploaded!")
                except Exception as e:
                    print(f"⚠️ Upload gagal untuk {checkpoint_name}: {e}")

    return (TextHubUploadCallback,)


# =====================================================================
# TEXT DATA PROCESSING (tokenize SFT & ORPO rows)
# =====================================================================
@app.cell
def _(
    TEXT_MAX_SOURCE_LENGTH,
    TEXT_MAX_TARGET_LENGTH,
    PreTrainedTokenizerFast,
    format_encoder_from_raw,
):
    def text_process_sft_rows(samples, tokenizer: PreTrainedTokenizerFast, is_chat=True):
        rows = []
        if is_chat:
            chat_groups = {}
            for obj in samples:
                if not obj.get("input") or not obj.get("target"):
                    continue
                chat_idx = obj.get("chat_idx", -1)
                if chat_idx not in chat_groups:
                    chat_groups[chat_idx] = []
                chat_groups[chat_idx].append(obj)

            for chat_idx, turns in chat_groups.items():
                turns = sorted(turns, key=lambda x: x.get("turn_idx", 0))

                for turn in turns:
                    inp_f = format_encoder_from_raw(turn["input"])
                    tgt_f = turn["target"].strip() + "<end_of_turn>"

                    inp_ids = tokenizer.encode(inp_f, add_special_tokens=True)
                    if getattr(tokenizer, "eos_token_id", None) is not None and inp_ids[-1] != tokenizer.eos_token_id:
                        inp_ids.append(tokenizer.eos_token_id)

                    tgt_ids = tokenizer.encode(tgt_f, add_special_tokens=False)
                    if getattr(tokenizer, "eos_token_id", None) is not None and tgt_ids[-1] != tokenizer.eos_token_id:
                        tgt_ids.append(tokenizer.eos_token_id)

                    if (
                        len(inp_ids) <= TEXT_MAX_SOURCE_LENGTH
                        and len(tgt_ids) <= TEXT_MAX_TARGET_LENGTH
                    ):
                        rows.append({"input_ids": inp_ids, "labels": tgt_ids})
                    else:
                        break
        else:
            for obj in samples:
                inp_f = format_encoder_from_raw(obj.get("input", ""))
                tgt_f = obj.get("target", "").strip() + "<end_of_turn>"

                inp_ids = tokenizer.encode(inp_f, add_special_tokens=True)
                if getattr(tokenizer, "eos_token_id", None) is not None and inp_ids[-1] != tokenizer.eos_token_id:
                    inp_ids.append(tokenizer.eos_token_id)

                tgt_ids = tokenizer.encode(tgt_f, add_special_tokens=False)
                if getattr(tokenizer, "eos_token_id", None) is not None and tgt_ids[-1] != tokenizer.eos_token_id:
                    tgt_ids.append(tokenizer.eos_token_id)

                if (
                    len(inp_ids) <= TEXT_MAX_SOURCE_LENGTH
                    and len(tgt_ids) <= TEXT_MAX_TARGET_LENGTH
                ):
                    rows.append({"input_ids": inp_ids, "labels": tgt_ids})
        return rows

    def text_process_orpo_rows(samples, tokenizer: PreTrainedTokenizerFast):
        rows = []
        for obj in samples:
            if not obj.get("prompt") or not obj.get("chosen") or not obj.get("rejected"):
                continue

            inp_f = format_encoder_from_raw(obj.get("prompt"))
            chosen_raw = obj.get("chosen", "").replace("assistant: ", "", 1).strip()
            rejected_raw = obj.get("rejected", "").replace("assistant: ", "", 1).strip()

            chosen_f = chosen_raw + "<end_of_turn>"
            rejected_f = rejected_raw + "<end_of_turn>"

            inp_ids = tokenizer.encode(inp_f, add_special_tokens=True)
            if getattr(tokenizer, "eos_token_id", None) is not None and inp_ids[-1] != tokenizer.eos_token_id:
                inp_ids.append(tokenizer.eos_token_id)

            chosen_ids = tokenizer.encode(chosen_f, add_special_tokens=False)
            if getattr(tokenizer, "eos_token_id", None) is not None and chosen_ids[-1] != tokenizer.eos_token_id:
                chosen_ids.append(tokenizer.eos_token_id)

            rejected_ids = tokenizer.encode(rejected_f, add_special_tokens=False)
            if getattr(tokenizer, "eos_token_id", None) is not None and rejected_ids[-1] != tokenizer.eos_token_id:
                rejected_ids.append(tokenizer.eos_token_id)

            if len(inp_ids) <= TEXT_MAX_SOURCE_LENGTH and len(chosen_ids) <= TEXT_MAX_TARGET_LENGTH and len(rejected_ids) <= TEXT_MAX_TARGET_LENGTH:
                rows.append({
                    "input_ids": inp_ids,
                    "chosen_labels": chosen_ids,
                    "rejected_labels": rejected_ids
                })
        return rows

    return (
        text_process_orpo_rows,
        text_process_sft_rows,
    )


# =====================================================================
# TEXT: LOAD DATASET BERDASARKAN STAGE
# =====================================================================
@app.cell
def _(
    AutoTokenizer,
    Dataset,
    PreTrainedTokenizerFast,
    TEXT_CHAT_CONFIG,
    TEXT_HF_REPO_ID,
    TEXT_INDOQA_CONFIG,
    TEXT_MODEL_NAME,
    TEXT_ORPO_CONFIG,
    TEXT_OUTPUT_DIR,
    TEXT_SAMPLE_EVAL_GENERATION,
    TEXT_SAMPLE_TRAIN_CHAT,
    TEXT_SAMPLE_TRAIN_INDOQA,
    TEXT_SAMPLE_TRAIN_ORPO,
    TEXT_SAMPLE_VAL_CHAT,
    TEXT_SAMPLE_VAL_INDOQA,
    load_hf_samples,
    os,
    random,
    text_current_stage,
    text_process_orpo_rows,
    text_process_sft_rows,
):
    os.makedirs(TEXT_OUTPUT_DIR, exist_ok=True)

    print(f"[TEXT] Loading Tokenizer from {TEXT_MODEL_NAME}...")
    _token = os.environ.get("HF_TOKEN")
    text_tokenizer = AutoTokenizer.from_pretrained(TEXT_MODEL_NAME, token=_token, trust_remote_code=True)
    assert isinstance(text_tokenizer, PreTrainedTokenizerFast), (
        "Tokenizer harus PreTrainedTokenizerFast"
    )

    # Selalu load validation data (dipakai di kedua stage untuk eval)
    val_chat_samples = load_hf_samples(
        TEXT_HF_REPO_ID, TEXT_CHAT_CONFIG, "validation", TEXT_SAMPLE_VAL_CHAT
    )
    val_indoqa_samples = load_hf_samples(
        TEXT_HF_REPO_ID, TEXT_INDOQA_CONFIG, "validation", TEXT_SAMPLE_VAL_INDOQA
    )
    val_rows = text_process_sft_rows(
        val_chat_samples, text_tokenizer, is_chat=True
    ) + text_process_sft_rows(val_indoqa_samples, text_tokenizer, is_chat=False)

    # Load training data sesuai stage aktif
    if text_current_stage == "sft":
        print("\n📦 [TEXT] Loading SFT training data...")
        train_chat_samples = load_hf_samples(
            TEXT_HF_REPO_ID, TEXT_CHAT_CONFIG, "train", TEXT_SAMPLE_TRAIN_CHAT
        )
        train_indoqa_samples = load_hf_samples(
            TEXT_HF_REPO_ID, TEXT_INDOQA_CONFIG, "train", TEXT_SAMPLE_TRAIN_INDOQA
        )
        train_rows = text_process_sft_rows(
            train_chat_samples, text_tokenizer, is_chat=True
        ) + text_process_sft_rows(train_indoqa_samples, text_tokenizer, is_chat=False)
        text_is_orpo_training = False
    elif text_current_stage == "orpo":
        print("\n📦 [TEXT] Loading ORPO training data...")
        train_orpo_samples = load_hf_samples(TEXT_HF_REPO_ID, TEXT_ORPO_CONFIG, "train", TEXT_SAMPLE_TRAIN_ORPO)
        train_rows = text_process_orpo_rows(train_orpo_samples, text_tokenizer)
        text_is_orpo_training = True
    else:
        # text_current_stage == "done"
        train_rows = []
        text_is_orpo_training = False

    random.seed(42)
    random.shuffle(train_rows)
    random.shuffle(val_rows)

    print(f"\n[TEXT] Total Training rows: {len(train_rows)}")
    print(f"[TEXT] Total Validation rows: {len(val_rows)}")

    text_train_ds = Dataset.from_list(train_rows) if train_rows else None
    text_eval_ds = Dataset.from_list(val_rows)

    n_eval_gen = min(len(val_rows), TEXT_SAMPLE_EVAL_GENERATION)
    # Eval generation selalu pakai SFT-format rows (punya input_ids + labels)
    _sft_val_rows = [r for r in val_rows if "labels" in r]
    text_eval_generation_samples = _sft_val_rows[:n_eval_gen]
    print(
        f"[TEXT] Mengambil {len(text_eval_generation_samples)} sampel validasi untuk pencatatan evaluasi kualitatif."
    )
    return (
        text_eval_ds,
        text_eval_generation_samples,
        text_is_orpo_training,
        text_tokenizer,
        text_train_ds,
    )


# =====================================================================
# TEXT: LOAD MODEL BERDASARKAN STAGE
# =====================================================================
@app.cell
def _(
    ALL_SUPPRESS_IDS,
    FastLanguageModel,
    TEXT_HF_CHECKPOINT_REPO,
    TEXT_HF_PREFIX,
    TEXT_LOAD_IN_4BIT,
    TEXT_LORA_ALPHA,
    TEXT_LORA_DROPOUT,
    TEXT_LORA_RANK,
    TEXT_LORA_TARGET_MODULES,
    TEXT_MAX_SOURCE_LENGTH,
    TEXT_MODEL_NAME,
    TEXT_OUTPUT_DIR,
    apply_logit_mask,
    gc,
    os,
    text_current_stage,
    text_tokenizer,
    torch,
):
    # 1. Reset Cuda Cache
    gc.collect()
    torch.cuda.empty_cache()

    if text_current_stage == "done":
        print("✅ [TEXT] Semua training sudah selesai! Skipping model load.")
        text_model = None
    elif text_current_stage == "orpo":
        # ORPO: Load SFT adapter (local fallback ke HF)
        _local_sft_path = os.path.join(TEXT_OUTPUT_DIR, "sft", "final_adapter")
        _model_path = None

        if os.path.exists(_local_sft_path) and os.listdir(_local_sft_path):
            print(f"\n📂 [TEXT] Loading SFT adapter dari local: {_local_sft_path}")
            _model_path = _local_sft_path
        else:
            # Download dari HF
            print(f"\n📥 [TEXT] SFT adapter tidak ditemukan di lokal. Download dari HF...")
            from huggingface_hub import snapshot_download as _snap_dl
            _hf_sft_path = _snap_dl(
                repo_id=TEXT_HF_CHECKPOINT_REPO,
                local_dir=_local_sft_path,
                allow_patterns=[f"{TEXT_HF_PREFIX}/sft/final_adapter/**"],
                token=os.environ.get("HF_TOKEN"),
            )
            # snapshot_download puts files in local_dir matching repo structure
            _downloaded_path = os.path.join(_hf_sft_path, TEXT_HF_PREFIX, "sft", "final_adapter")
            if os.path.exists(_downloaded_path):
                _model_path = _downloaded_path
            else:
                _model_path = _local_sft_path

        print(f"[TEXT] Loading ORPO base model from SFT adapter: {_model_path}")
        text_model, _tokenizer_unsloth = FastLanguageModel.from_pretrained(
            model_name=_model_path,
            max_seq_length=TEXT_MAX_SOURCE_LENGTH,
            load_in_4bit=TEXT_LOAD_IN_4BIT,
            trust_remote_code=True,
            token=os.environ.get("HF_TOKEN"),
        )
    else:
        # SFT: Load base model
        print(f"\n[TEXT] Loading base model from {TEXT_MODEL_NAME} using Unsloth...")
        text_model, _tokenizer_unsloth = FastLanguageModel.from_pretrained(
            model_name=TEXT_MODEL_NAME,
            max_seq_length=TEXT_MAX_SOURCE_LENGTH,
            load_in_4bit=TEXT_LOAD_IN_4BIT,
            trust_remote_code=True,
            token=os.environ.get("HF_TOKEN"),
        )

    if text_model is not None:
        # Reset max_length to silence warning
        text_model.config.max_length = None
        if hasattr(text_model, "generation_config") and text_model.generation_config is not None:
            text_model.generation_config.max_length = None

        if getattr(text_model.config, "decoder_start_token_id", None) is None:
            text_model.config.decoder_start_token_id = text_tokenizer.bos_token_id
            print(f"  Set decoder_start_token_id = {text_model.config.decoder_start_token_id}")

        if text_tokenizer.pad_token is None:
            text_tokenizer.add_special_tokens({"pad_token": text_tokenizer.eos_token})
            text_model.resize_token_embeddings(len(text_tokenizer))

        # Logit Masking
        print(f"\n[TEXT] Applying logit mask for {len(ALL_SUPPRESS_IDS)} tokens...")
        apply_logit_mask(text_model, ALL_SUPPRESS_IDS)

        # LoRA Config
        if hasattr(text_model, "peft_config"):
            print("[TEXT] Model is already a PEFT model (loaded SFT adapter). Skipping get_peft_model.")
        else:
            print("[TEXT] Applying LoRA using Unsloth...")
            text_model = FastLanguageModel.get_peft_model(
                text_model,
                r=TEXT_LORA_RANK,
                lora_alpha=TEXT_LORA_ALPHA,
                target_modules=TEXT_LORA_TARGET_MODULES,
                lora_dropout=TEXT_LORA_DROPOUT,
                bias="none",
                use_gradient_checkpointing="unsloth",
                random_state=3407,
            )

        getattr(FastLanguageModel, "for_training")(text_model)
        text_model.config.use_cache = False

        # Safety wrapper
        if hasattr(text_model, "prepare_decoder_input_ids_from_labels"):
            orig_fn = text_model.prepare_decoder_input_ids_from_labels
            def compatible_prepare(labels=None, input_ids=None, *args, **kwargs):
                target_tensor = labels if labels is not None else input_ids
                return orig_fn(target_tensor, *args, **kwargs)
            text_model.prepare_decoder_input_ids_from_labels = compatible_prepare

        text_model.print_trainable_parameters()

    return (text_model,)


# =====================================================================
# TEXT TRAINING CELL — Sequential SFT → ORPO
# =====================================================================
@app.cell
def _(
    ALL_SUPPRESS_IDS,
    Any,
    BF16,
    DataCollatorForSeq2Seq,
    F,
    GrokAdEMAMix,
    PreTrainedTokenizerFast,
    SelectiveLabelSmoother,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
    TEXT_EVAL_ACCUMULATION_STEPS,
    TEXT_EVAL_EVERY_N_STEPS,
    TEXT_FP16,
    TEXT_GEN_REPETITION_PENALTY,
    TEXT_GEN_TEMPERATURE,
    TEXT_GEN_TOP_P,
    TEXT_GRADIENT_ACCUMULATION_STEPS,
    TEXT_GRADIENT_CHECKPOINTING,
    TEXT_HF_CHECKPOINT_REPO,
    TEXT_HF_PREFIX,
    TEXT_LABEL_SMOOTHING_FACTOR,
    TEXT_LEARNING_RATE,
    TEXT_LOGGING_STEPS,
    TEXT_LR_SCHEDULER_TYPE,
    TEXT_MAX_TARGET_LENGTH,
    TEXT_NEFTUNE_NOISE_ALPHA,
    TEXT_NUM_EPOCHS_ORPO,
    TEXT_NUM_EPOCHS_SFT,
    TEXT_OPTIM,
    TEXT_ORPO_BETA,
    TEXT_OUTPUT_DIR,
    TEXT_PER_DEVICE_EVAL_BATCH_SIZE,
    TEXT_PER_DEVICE_TRAIN_BATCH_SIZE,
    TEXT_PREDICT_WITH_GENERATE,
    TEXT_SAVE_TOTAL_LIMIT,
    TEXT_WARMUP_STEPS,
    TEXT_WEIGHT_DECAY,
    TextHubUploadCallback,
    TextSampleGenerationCallback,
    TextTrainingPlotCallback,
    bertscore_metric,
    bleu_metric,
    cast,
    exact_match_metric,
    gc,
    get_scheduler,
    meteor_metric,
    mo,
    np,
    os,
    rouge_metric,
    text_current_stage,
    text_eval_ds,
    text_eval_generation_samples,
    text_is_orpo_training,
    text_model,
    text_resume_checkpoint,
    text_tokenizer,
    text_train_ds,
    torch,
):
    # Skip jika sudah selesai atau tidak ada data
    mo.stop(
        text_current_stage == "done" or text_train_ds is None,
        mo.md("✅ **[TEXT] Training sudah selesai atau tidak ada data training.** Lanjut ke merge & upload."),
    )

    # 1. Bersihkan sisa memori
    gc.collect()
    torch.cuda.empty_cache()
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

    # Tentukan output dir berdasarkan stage
    text_active_output_dir = os.path.join(TEXT_OUTPUT_DIR, text_current_stage)
    os.makedirs(text_active_output_dir, exist_ok=True)

    # === CUSTOM TRAINER ===
    # NOTE: SelectiveLabelSmoother dipakai dari shared cell (identik dengan
    # definisi inline di notebook text asli — dipindah agar bisa dipakai ulang
    # oleh vision trainer tanpa duplikasi nama).
    class TextCustomSeq2SeqTrainer(Seq2SeqTrainer):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self.model_accepts_loss_kwargs = False
            if self.args.label_smoothing_factor > 0:
                self.label_smoother = SelectiveLabelSmoother(
                    epsilon=self.args.label_smoothing_factor,
                    suppress_ids=ALL_SUPPRESS_IDS,
                )

        def compute_loss(
            self, model, inputs, return_outputs=False, num_items_in_batch=None, **kwargs
        ):
            labels = inputs.get("labels")
            outputs = model(**inputs)

            if self.label_smoother is not None and labels is not None:
                loss = self.label_smoother(outputs, labels)
            else:
                if isinstance(outputs, dict) and "logits" in outputs:
                    logits = outputs["logits"]
                elif isinstance(outputs, tuple):
                    logits = outputs[1] if len(outputs) > 1 else outputs[0].logits
                else:
                    logits = outputs.logits
                loss_fct = torch.nn.CrossEntropyLoss(
                    ignore_index=-100, reduction="mean"
                )
                loss = loss_fct(logits.view(-1, logits.size(-1)), labels.view(-1))

            return (loss, outputs) if return_outputs else loss

        def evaluate(
            self,
            eval_dataset=None,
            ignore_keys=None,
            metric_key_prefix="eval",
            **gen_kwargs,
        ):
            from unsloth import FastLanguageModel
            if hasattr(FastLanguageModel, "for_inference"):
                FastLanguageModel.for_inference(self.model)
            else:
                self.model.eval()

            metrics = super().evaluate(
                eval_dataset=eval_dataset,
                ignore_keys=ignore_keys,
                metric_key_prefix=metric_key_prefix,
                **gen_kwargs,
            )

            if hasattr(FastLanguageModel, "for_training"):
                FastLanguageModel.for_training(self.model)
            else:
                self.model.train()

            gc.collect()
            torch.cuda.empty_cache()
            return metrics

        def log(self, logs, start_time=None):
            if "eval_loss" in logs:
                import math
                try:
                    logs["eval_perplexity"] = math.exp(logs["eval_loss"])
                except OverflowError:
                    logs["eval_perplexity"] = float("inf")
            super().log(logs, start_time=start_time)

    # === ORPO COLLATOR ===
    class TextORPODataCollatorForSeq2Seq(DataCollatorForSeq2Seq):
        def __call__(self, features, return_tensors=None):
            if not features or "chosen_labels" not in features[0]:
                return super().__call__(features, return_tensors)

            chosen_features = [{"input_ids": f["input_ids"], "labels": f["chosen_labels"]} for f in features]
            rejected_features = [{"input_ids": f["input_ids"], "labels": f["rejected_labels"]} for f in features]

            batch = super().__call__(chosen_features, return_tensors)
            rejected_batch = super().__call__(rejected_features, return_tensors)

            batch["chosen_labels"] = batch.pop("labels")
            batch["rejected_labels"] = rejected_batch.pop("labels")
            return batch

    # === ORPO TRAINER ===
    class TextCustomORPOTrainer(TextCustomSeq2SeqTrainer):
        def __init__(self, beta=0.1, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self.beta = beta

        def get_batch_logps(self, logits, labels, average_log_prob: bool = True):
            """Pakai average_log_prob=True untuk numerical stability (sesuai paper ORPO)."""
            if logits.shape[:-1] != labels.shape:
                raise ValueError("Logits and labels must have the same shape.")
            labels = labels.clone()
            loss_mask = labels != -100
            labels[labels == -100] = 0
            per_token_logps = torch.gather(logits.log_softmax(-1), dim=2, index=labels.unsqueeze(2)).squeeze(2)
            if average_log_prob:
                return (per_token_logps * loss_mask).sum(-1) / loss_mask.sum(-1).clamp(min=1)
            else:
                return (per_token_logps * loss_mask).sum(-1)

        def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None, **kwargs):
            chosen_labels = inputs.pop("chosen_labels", None)
            rejected_labels = inputs.pop("rejected_labels", None)

            if chosen_labels is None or rejected_labels is None:
                return super().compute_loss(model, inputs, return_outputs, num_items_in_batch, **kwargs)

            input_ids = inputs.get("input_ids")
            attention_mask = inputs.get("attention_mask")

            chosen_outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=chosen_labels)
            chosen_logits = chosen_outputs.logits

            rejected_outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=rejected_labels)
            rejected_logits = rejected_outputs.logits

            chosen_logps = self.get_batch_logps(chosen_logits, chosen_labels, average_log_prob=True)
            rejected_logps = self.get_batch_logps(rejected_logits, rejected_labels, average_log_prob=True)

            # Numerically stable log-odds (clamp exp to avoid 0 or 1)
            chosen_probs = chosen_logps.exp().clamp(1e-7, 1 - 1e-7)
            rejected_probs = rejected_logps.exp().clamp(1e-7, 1 - 1e-7)
            chosen_log_odds = torch.log(chosen_probs / (1 - chosen_probs))
            rejected_log_odds = torch.log(rejected_probs / (1 - rejected_probs))

            log_odds_margin = chosen_log_odds - rejected_log_odds
            or_loss = -F.logsigmoid(log_odds_margin).mean()

            # SFT loss tanpa label smoothing untuk konsistensi skala
            loss_fct = torch.nn.CrossEntropyLoss(ignore_index=-100, reduction="mean")
            sft_loss = loss_fct(chosen_logits.view(-1, chosen_logits.size(-1)), chosen_labels.view(-1))

            loss = sft_loss + self.beta * or_loss
            return (loss, chosen_outputs) if return_outputs else loss

    # === COMPUTE METRICS ===
    def text_compute_metrics(eval_preds):
        metrics = {}
        if rouge_metric is None and bleu_metric is None:
            return metrics
        preds, labels = eval_preds
        if isinstance(preds, tuple):
            preds = preds[0]
        tok = cast(PreTrainedTokenizerFast, text_tokenizer)

        if preds.ndim == 3:
            preds = preds.argmax(axis=-1)

        labels = np.where(labels != -100, labels, tok.pad_token_id)
        preds = np.where(preds != -100, preds, tok.pad_token_id)
        decoded_preds = tok.batch_decode(preds, skip_special_tokens=True)
        decoded_labels = tok.batch_decode(labels, skip_special_tokens=True)
        decoded_preds = [pred.strip() for pred in decoded_preds]
        decoded_labels = [label.strip() for label in decoded_labels]

        if rouge_metric is not None:
            try:
                result = cast(Any, rouge_metric).compute(predictions=decoded_preds, references=decoded_labels, use_stemmer=False)
                if result is not None:
                    for key, value in result.items():
                        metrics[key] = value * 100
            except Exception as e:
                print(f"Error during ROUGE: {e}")

        if bleu_metric is not None:
            try:
                formatted_labels = [[label] for label in decoded_labels]
                bleu_result = cast(Any, bleu_metric).compute(predictions=decoded_preds, references=formatted_labels)
                if bleu_result is not None and "bleu" in bleu_result:
                    metrics["bleu"] = bleu_result["bleu"] * 100
            except Exception as e:
                print(f"Error during BLEU: {e}")

        if exact_match_metric is not None:
            try:
                em_result = cast(Any, exact_match_metric).compute(predictions=decoded_preds, references=decoded_labels)
                if em_result is not None and "exact_match" in em_result:
                    metrics["exact_match"] = em_result["exact_match"] * 100
            except Exception as e:
                print(f"Error during Exact Match: {e}")

        if bertscore_metric is not None:
            try:
                bertscore_result = cast(Any, bertscore_metric).compute(
                    predictions=decoded_preds, references=decoded_labels,
                    model_type="google/embeddinggemma-300m", num_layers=12, lang="id"
                )
                if bertscore_result is not None and "f1" in bertscore_result:
                    metrics["bertscore_f1"] = np.mean(bertscore_result["f1"]) * 100
            except Exception as e:
                print(f"Error during BERTScore: {e}")

        if meteor_metric is not None:
            try:
                meteor_result = cast(Any, meteor_metric).compute(predictions=decoded_preds, references=decoded_labels)
                if meteor_result is not None and "meteor" in meteor_result:
                    metrics["meteor"] = meteor_result["meteor"] * 100
            except Exception as e:
                print(f"Error during METEOR: {e}")

        return metrics

    def text_preprocess_logits_for_metrics(logits, labels):
        if isinstance(logits, tuple):
            logits = logits[0]
        return logits.argmax(dim=-1)

    # === BUILD TRAINER ===
    bad_words_ids = [
        [id_] for id_ in ALL_SUPPRESS_IDS if id_ < cast(Any, text_model).config.vocab_size
    ]

    _hf_token = os.environ.get("HF_TOKEN")

    plot_callback = TextTrainingPlotCallback(output_dir=text_active_output_dir)
    sample_callback = TextSampleGenerationCallback(
        tokenizer=text_tokenizer,
        eval_samples=text_eval_generation_samples,
        output_dir=text_active_output_dir,
        eval_every_n_steps=TEXT_EVAL_EVERY_N_STEPS,
        temperature=TEXT_GEN_TEMPERATURE,
        top_p=TEXT_GEN_TOP_P,
        repetition_penalty=TEXT_GEN_REPETITION_PENALTY,
        bad_words_ids=bad_words_ids,
    )
    hub_callback = TextHubUploadCallback(
        repo_id=TEXT_HF_CHECKPOINT_REPO,
        stage=f"{TEXT_HF_PREFIX}/{text_current_stage}",
        token=_hf_token,
        output_dir=text_active_output_dir,
    )

    if text_is_orpo_training:
        data_collator = TextORPODataCollatorForSeq2Seq(tokenizer=text_tokenizer, model=text_model, padding=True)
    else:
        data_collator = DataCollatorForSeq2Seq(tokenizer=text_tokenizer, model=text_model, padding=True)

    training_args = Seq2SeqTrainingArguments(
        output_dir=text_active_output_dir,
        per_device_train_batch_size=TEXT_PER_DEVICE_TRAIN_BATCH_SIZE,
        per_device_eval_batch_size=TEXT_PER_DEVICE_EVAL_BATCH_SIZE,
        gradient_accumulation_steps=TEXT_GRADIENT_ACCUMULATION_STEPS,
        eval_accumulation_steps=TEXT_EVAL_ACCUMULATION_STEPS,
        learning_rate=TEXT_LEARNING_RATE,
        num_train_epochs=TEXT_NUM_EPOCHS_ORPO if text_is_orpo_training else TEXT_NUM_EPOCHS_SFT,
        warmup_steps=TEXT_WARMUP_STEPS,
        weight_decay=TEXT_WEIGHT_DECAY,
        lr_scheduler_type=TEXT_LR_SCHEDULER_TYPE,
        predict_with_generate=TEXT_PREDICT_WITH_GENERATE,
        logging_steps=TEXT_LOGGING_STEPS,
        save_strategy="steps",
        save_steps=TEXT_EVAL_EVERY_N_STEPS,
        save_total_limit=TEXT_SAVE_TOTAL_LIMIT,
        eval_strategy="steps",
        eval_steps=TEXT_EVAL_EVERY_N_STEPS,
        optim=TEXT_OPTIM,
        label_smoothing_factor=TEXT_LABEL_SMOOTHING_FACTOR if not text_is_orpo_training else 0.0,
        neftune_noise_alpha=TEXT_NEFTUNE_NOISE_ALPHA,
        report_to="none",
        fp16=TEXT_FP16,
        bf16=BF16,
        gradient_checkpointing=TEXT_GRADIENT_CHECKPOINTING,
        generation_max_length=TEXT_MAX_TARGET_LENGTH,
        push_to_hub=False,  # Kita handle manual via HubUploadCallback
        remove_unused_columns=False if text_is_orpo_training else True,
    )

    optimizer = GrokAdEMAMix(
        text_model.parameters(),
        lr=TEXT_LEARNING_RATE,
        weight_decay=TEXT_WEIGHT_DECAY,
        grok_alpha=2.0,
        grok_lamb=0.98,
    )

    num_update_steps_per_epoch = max(
        1, len(text_train_ds) // (TEXT_PER_DEVICE_TRAIN_BATCH_SIZE * TEXT_GRADIENT_ACCUMULATION_STEPS)
    )
    max_steps = num_update_steps_per_epoch * (TEXT_NUM_EPOCHS_ORPO if text_is_orpo_training else TEXT_NUM_EPOCHS_SFT)

    lr_scheduler = get_scheduler(
        name=TEXT_LR_SCHEDULER_TYPE,
        optimizer=optimizer,
        num_warmup_steps=TEXT_WARMUP_STEPS,
        num_training_steps=max_steps,
    )

    trainer_class = TextCustomORPOTrainer if text_is_orpo_training else TextCustomSeq2SeqTrainer
    trainer_kwargs = {
        "model": text_model,
        "args": training_args,
        "train_dataset": cast(Any, text_train_ds),
        "eval_dataset": cast(Any, text_eval_ds),
        "data_collator": data_collator,
        "compute_metrics": text_compute_metrics,
        "preprocess_logits_for_metrics": None if TEXT_PREDICT_WITH_GENERATE else text_preprocess_logits_for_metrics,
        "optimizers": (optimizer, lr_scheduler),
        "callbacks": [plot_callback, sample_callback, hub_callback],
    }
    if text_is_orpo_training:
        trainer_kwargs["beta"] = TEXT_ORPO_BETA

    text_trainer = trainer_class(**trainer_kwargs)

    # === RESUME FROM HF CHECKPOINT ===
    _resume_from = None
    if text_resume_checkpoint:
        try:
            from huggingface_hub import snapshot_download as _resume_snap
            from huggingface_hub import HfApi as _ResumeApi

            _api = _ResumeApi(token=_hf_token)
            _files = _api.list_repo_files(repo_id=TEXT_HF_CHECKPOINT_REPO)

            # Cari checkpoint terbaru biar gak download semuanya dan bikin storage penuh
            _ckpt_prefix = f"{TEXT_HF_PREFIX}/{text_current_stage}/checkpoint-"
            _ckpts = list(set([f.split('/')[2] for f in _files if f.startswith(_ckpt_prefix)]))
            if _ckpts:
                _ckpts.sort(key=lambda x: int(x.split('-')[1]))
                _latest_ckpt = _ckpts[-1]
            else:
                _latest_ckpt = "checkpoint-*"

            print(f"\n📥 [TEXT] Downloading {_latest_ckpt} ({text_current_stage}) dari HF untuk resume...")
            _resume_snap(
                repo_id=TEXT_HF_CHECKPOINT_REPO,
                local_dir=text_active_output_dir,
                allow_patterns=[f"{TEXT_HF_PREFIX}/{text_current_stage}/{_latest_ckpt}/**"],
                token=_hf_token,
            )
            # Pindahkan dari subfolder ke root output dir jika perlu
            _sub_dir = os.path.join(text_active_output_dir, TEXT_HF_PREFIX, text_current_stage)
            if os.path.exists(_sub_dir):
                import shutil as _shutil_text
                for _item in os.listdir(_sub_dir):
                    _src = os.path.join(_sub_dir, _item)
                    _dst = os.path.join(text_active_output_dir, _item)
                    if os.path.isdir(_src) and _item.startswith("checkpoint-"):
                        if os.path.exists(_dst):
                            _shutil_text.rmtree(_dst)
                        _shutil_text.move(_src, _dst)
                _shutil_text.rmtree(os.path.join(text_active_output_dir, TEXT_HF_PREFIX))

            _checkpoints = sorted([
                d for d in os.listdir(text_active_output_dir)
                if d.startswith("checkpoint-") and os.path.isdir(os.path.join(text_active_output_dir, d))
            ])
            if _checkpoints:
                _resume_from = True
                print(f"✅ [TEXT] Ditemukan {len(_checkpoints)} checkpoint(s). Resume dari yang terbaru!")
            else:
                print("⚠️ [TEXT] Tidak ada checkpoint valid ditemukan. Mulai dari awal.")
        except Exception as e:
            print(f"⚠️ [TEXT] Gagal download checkpoint: {e}. Mulai dari awal.")

    # === START TRAINING ===
    print(f"\n🚀 [TEXT] Starting {text_current_stage.upper()} training...")
    text_trainer.train(resume_from_checkpoint=_resume_from)

    # === SAVE FINAL ADAPTER + UPLOAD ===
    _final_path = os.path.join(text_active_output_dir, "final_adapter")
    print(f"\n💾 [TEXT] Saving final adapter to {_final_path}...")
    text_trainer.save_model(_final_path)
    text_tokenizer.save_pretrained(_final_path)

    # Upload final adapter ke HF
    try:
        from huggingface_hub import HfApi as _FinalApi
        _final_api = _FinalApi(token=_hf_token)
        print(f"📤 [TEXT] Uploading final adapter to HF {TEXT_HF_PREFIX}/{text_current_stage}/final_adapter/...")
        _final_api.upload_folder(
            folder_path=_final_path,
            path_in_repo=f"{TEXT_HF_PREFIX}/{text_current_stage}/final_adapter",
            repo_id=TEXT_HF_CHECKPOINT_REPO,
        )
        # Upload final chart dan log juga
        for _art in ["training_chart.png", "eval_samples.txt"]:
            _art_path = os.path.join(text_active_output_dir, _art)
            if os.path.exists(_art_path):
                _final_api.upload_file(
                    path_or_fileobj=_art_path,
                    path_in_repo=f"{TEXT_HF_PREFIX}/{text_current_stage}/{_art}",
                    repo_id=TEXT_HF_CHECKPOINT_REPO,
                )
        print(f"✅ [TEXT] {text_current_stage.upper()} training selesai dan ter-upload!")
    except Exception as e:
        print(f"⚠️ [TEXT] Upload final adapter gagal: {e}")

    return (text_trainer,)


# =====================================================================
# TEXT: MERGE & QUANTIZE
# =====================================================================
@app.cell
def _(
    TEXT_HF_CHECKPOINT_REPO,
    TEXT_HF_PREFIX,
    TEXT_LOAD_IN_4BIT,
    TEXT_MAX_SOURCE_LENGTH,
    TEXT_OUTPUT_DIR,
    mo,
    os,
    text_current_stage,
    text_model,
    text_tokenizer,
):
    mo.stop(
        text_current_stage != "done" and text_model is None,
        mo.md("⏭️ **[TEXT] Phase 1 belum selesai (SFT/ORPO masih berjalan).** Merge dilewati — re-run notebook setelah ORPO selesai."),
    )

    def text_merge_and_quantize(text_model, text_tokenizer, upload_dir: str):
        if text_model is None:
            from unsloth import FastLanguageModel
            # Load model dari adapter ORPO final
            _orpo_path = os.path.join(TEXT_OUTPUT_DIR, "orpo", "final_adapter")
            if not os.path.exists(_orpo_path):
                # Fallback download dari HF
                from huggingface_hub import snapshot_download as _snap_dl
                print("📥 [TEXT] Downloading final ORPO adapter dari HF untuk merging...")
                _snap_dl(
                    repo_id=TEXT_HF_CHECKPOINT_REPO,
                    local_dir=_orpo_path,
                    allow_patterns=[f"{TEXT_HF_PREFIX}/orpo/final_adapter/**"],
                    token=os.environ.get("HF_TOKEN"),
                )
                _sub_path = os.path.join(_orpo_path, TEXT_HF_PREFIX, "orpo", "final_adapter")
                if os.path.exists(_sub_path):
                    _orpo_path = _sub_path

            print(f"📂 [TEXT] Loading model dari ORPO adapter untuk merge: {_orpo_path}")
            text_model, _ = FastLanguageModel.from_pretrained(
                model_name=_orpo_path,
                max_seq_length=TEXT_MAX_SOURCE_LENGTH,
                load_in_4bit=TEXT_LOAD_IN_4BIT,
                trust_remote_code=True,
            )

        merged_bf16_path = os.path.join(upload_dir, "merged_bf16")
        quantized_4bit_path = os.path.join(upload_dir, "quantized_4bit")

        print("[TEXT] Merging LoRA adapter and saving model as BF16 using Unsloth...")
        text_model.save_pretrained_merged(merged_bf16_path, text_tokenizer, save_method="merged_16bit")
        print("✅ [TEXT] Model BF16 berhasil disimpan.")

        print("\n[TEXT] Merging LoRA adapter and saving model as 4-bit NF4 using Unsloth...")
        text_model.save_pretrained_merged(quantized_4bit_path, text_tokenizer, save_method="merged_4bit_forced")
        print("✅ [TEXT] Model 4-bit NF4 berhasil disimpan!")

        return None

    text_upload_dir = os.path.join(TEXT_OUTPUT_DIR, "hf_upload")

    text_merge_and_quantize(text_model, text_tokenizer, text_upload_dir)
    return text_merge_and_quantize, text_upload_dir


@app.cell
def _(TEXT_HF_CHECKPOINT_REPO, TEXT_HF_PREFIX, os, text_upload_dir):
    from huggingface_hub import HfApi as _UploadMergedApi

    print(f"[TEXT] Memulai proses unggah model merged ke HF Hub: {TEXT_HF_CHECKPOINT_REPO}/{TEXT_HF_PREFIX}...")
    text_merged_uploaded = False
    try:
        _merged_api = _UploadMergedApi(token=os.environ.get("HF_TOKEN"))

        _merged_api.create_repo(repo_id=TEXT_HF_CHECKPOINT_REPO, repo_type="model", private=False, exist_ok=True)
        _merged_api.upload_folder(
            folder_path=text_upload_dir,
            path_in_repo=TEXT_HF_PREFIX,
            repo_id=TEXT_HF_CHECKPOINT_REPO,
            repo_type="model",
        )
        text_merged_uploaded = True

        print("✅ [TEXT] Berhasil mengunggah merged models ke Hugging Face Hub!")
    except Exception as e:
        print(f"❌ [TEXT] Terjadi kesalahan saat mengunggah: {e}")
    return (text_merged_uploaded,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### 💻 [TEXT] Local Deployment & Inference (Unified Repo, Subfolder `text/`)
    Setelah model TEXT diunggah ke **unified repo**, struktur artifacts TEXT berada di bawah prefix `text/`:
    - `text/sft/` — Checkpoint dan artifacts SFT training
    - `text/orpo/` — Checkpoint dan artifacts ORPO training
    - `text/merged_bf16/` — Model gabungan utuh (bfloat16, ~15 GB) — **input untuk Phase 1.5 CANGKOK**
    - `text/quantized_4bit/` — Model terkuantisasi (NF4, ~5 GB)

    #### Load Model Quantized 4-bit:
    ```python
    from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

    model_id = "daruokta/t5gemma-2-4b-4b-instruct-chat-indo-v6-combined-unsloth"

    tokenizer = AutoTokenizer.from_pretrained(model_id, subfolder="text/quantized_4bit")
    model = AutoModelForSeq2SeqLM.from_pretrained(
        model_id, subfolder="text/quantized_4bit", device_map="auto"
    )
    ```

    #### Load Model Full Precision (BF16):
    ```python
    import torch
    from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

    model_id = "daruokta/t5gemma-2-4b-4b-instruct-chat-indo-v6-combined-unsloth"

    tokenizer = AutoTokenizer.from_pretrained(model_id, subfolder="text/merged_bf16")
    model = AutoModelForSeq2SeqLM.from_pretrained(
        model_id, subfolder="text/merged_bf16",
        torch_dtype=torch.bfloat16, device_map="auto"
    )
    ```
    """)
    return


# =====================================================================
# TEXT: VISUALISASI HASIL EVALUASI
# =====================================================================
@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ## 📊 [TEXT] Visualisasi Hasil Evaluasi Kualitatif
    """)
    return


@app.cell
def _(TEXT_OUTPUT_DIR, mo, os, re, text_current_stage):
    # Path ke file log hasil evaluasi — sesuai stage aktif
    _active_dir = os.path.join(TEXT_OUTPUT_DIR, text_current_stage) if text_current_stage != "done" else os.path.join(TEXT_OUTPUT_DIR, "orpo")
    text_log_file_path = os.path.join(_active_dir, "eval_samples.txt")

    def text_parse_log_file(filepath):
        if not os.path.exists(filepath):
            return []
        with open(filepath, "r", encoding="utf-8") as f:
            content = f.read()

        blocks = re.split(r"={10,}", content)
        steps_data = []

        i = 1
        while i < len(blocks):
            header = blocks[i].strip()
            step_match = re.search(r"Step\s+(\d+)\s*\|\s*([\d\-\s:]+)", header)
            if not step_match:
                i += 1
                continue

            step_num = step_match.group(1)
            timestamp = step_match.group(2)
            label = f"Step {step_num} ({timestamp})"

            body = blocks[i + 1].strip() if i + 1 < len(blocks) else ""

            samples = []
            raw_samples = re.split(r"\n+Q:\s*", "\n" + body)
            for rs in raw_samples:
                rs = rs.strip()
                if not rs or not ("Expected Target:" in rs and "Model Response:" in rs):
                    continue

                try:
                    q_part, rest = rs.split("Expected Target:", 1)
                    target_part, response_part = rest.split("Model Response:", 1)

                    query = q_part.strip()
                    target = target_part.strip()
                    response = response_part.strip()

                    flag_text = "Good ✅"
                    flag_class = "badge-good"
                    if "⚠️ REPETITIVE" in response:
                        response = response.replace("⚠️ REPETITIVE", "").strip()
                        flag_text = "Repetitive ⚠️"
                        flag_class = "badge-rep"
                    elif response.endswith(" ✅"):
                        response = response[:-2].strip()

                    samples.append(
                        {
                            "query": query,
                            "target": target,
                            "response": response,
                            "flag": flag_text,
                            "flag_class": flag_class,
                        }
                    )
                except Exception:
                    continue

            if samples:
                steps_data.append({"label": label, "samples": samples})
            i += 2

        return steps_data[::-1]

    text_refresh_button = mo.ui.button(label="🔄 Refresh Data Evaluasi (TEXT)", value=0)

    text_css_style = mo.Html("""
    <style>
    .sample-container {
        display: flex;
        flex-direction: column;
        gap: 16px;
        margin-top: 12px;
    }
    .sample-card {
        background: rgba(255, 255, 255, 0.03);
        border: 1px solid rgba(255, 255, 255, 0.08);
        border-radius: 12px;
        padding: 16px;
        box-shadow: 0 4px 15px rgba(0, 0, 0, 0.05);
        transition: transform 0.2s, box-shadow 0.2s;
    }
    .sample-card:hover {
        transform: translateY(-2px);
        box-shadow: 0 8px 25px rgba(0, 0, 0, 0.15);
        border-color: rgba(255, 255, 255, 0.15);
    }
    .sample-header {
        display: flex;
        justify-content: space-between;
        align-items: center;
        margin-bottom: 12px;
        border-bottom: 1px solid rgba(255, 255, 255, 0.08);
        padding-bottom: 8px;
    }
    .sample-num {
        font-weight: 700;
        font-size: 1.05em;
        color: #4A90D9;
    }
    .sample-badge {
        padding: 4px 8px;
        border-radius: 6px;
        font-size: 0.8em;
        font-weight: 600;
    }
    .badge-good {
        background-color: rgba(46, 204, 113, 0.15);
        color: #2ECC71;
        border: 1px solid rgba(46, 204, 113, 0.3);
    }
    .badge-rep {
        background-color: rgba(231, 76, 60, 0.15);
        color: #E74C3C;
        border: 1px solid rgba(231, 76, 60, 0.3);
    }
    .section-title {
        font-size: 0.85em;
        text-transform: uppercase;
        letter-spacing: 0.5px;
        color: #888;
        margin-top: 10px;
        margin-bottom: 4px;
        font-weight: 600;
    }
    .text-block {
        padding: 12px;
        border-radius: 6px;
        background: rgba(255, 255, 255, 0.01);
        border: 1px solid rgba(255, 255, 255, 0.04);
        font-size: 0.95em;
        line-height: 1.5;
        white-space: pre-wrap;
    }
    .prompt-block {
        font-family: 'Fira Code', Consolas, monospace;
        font-size: 0.85em;
        color: #ddd;
        background: rgba(0, 0, 0, 0.2);
    }
    .target-block {
        border-left: 3px solid #9B59B6;
    }
    .response-block {
        border-left: 3px solid #2ECC71;
    }
    </style>
    """)
    return text_css_style, text_log_file_path, text_parse_log_file, text_refresh_button


@app.cell
def _(mo, text_log_file_path, text_parse_log_file, text_refresh_button):
    _ = text_refresh_button.value

    text_evaluation_runs = text_parse_log_file(text_log_file_path)

    if not text_evaluation_runs:
        text_step_dropdown = None
    else:
        run_options = {run["label"]: idx for idx, run in enumerate(text_evaluation_runs)}
        text_step_dropdown = mo.ui.dropdown(
            options=run_options,
            value=next(iter(run_options)),
            label="Pilih Step Evaluasi (TEXT):",
            full_width=True,
        )
    return text_evaluation_runs, text_step_dropdown


@app.cell
def _(mo, text_css_style, text_evaluation_runs, text_log_file_path, text_refresh_button, text_step_dropdown):
    if not text_evaluation_runs or text_step_dropdown is None:
        _output = mo.md(
            f"⚠️ *Belum ada data evaluasi TEXT ditemukan di `{text_log_file_path}`. Silakan jalankan training terlebih dahulu.*"
        )
    else:
        _selected_idx = text_step_dropdown.value
        _selected_run = text_evaluation_runs[_selected_idx]

        _cards_html = []
        for _idx, _s in enumerate(_selected_run["samples"]):
            _card = f"""
            <div class="sample-card">
                <div class="sample-header">
                    <span class="sample-num">Sampel #{_idx + 1}</span>
                    <span class="sample-badge {_s["flag_class"]}">{_s["flag"]}</span>
                </div>
                <div class="sample-body">
                    <div class="section-title">💬 User Prompt</div>
                    <pre class="text-block prompt-block">{_s["query"]}</pre>

                    <div class="section-title">🎯 Expected Target</div>
                    <div class="text-block target-block">{_s["target"]}</div>

                    <div class="section-title">🤖 Model Response</div>
                    <div class="text-block response-block">{_s["response"]}</div>
                </div>
            </div>
            """
            _cards_html.append(_card)

        _container_html = f"""
        {text_css_style.text}
        <div class="sample-container">
            {"".join(_cards_html)}
        </div>
        """

        _output = mo.vstack(
            [
                mo.md(
                    f"Menampilkan **{len(_selected_run['samples'])} sampel** untuk **{_selected_run['label']}**."
                ),
                text_step_dropdown,
                mo.Html(_container_html),
            ]
        )

    mo.vstack([text_refresh_button, mo.hstack([_output], justify="start")])
    return


# #####################################################################
# #####################################################################
#
#   ██████╗ ██╗  ██╗ █████╗ ███████╗███████╗     ██╗ ███████╗
#   ██╔══██╗██║  ██║██╔══██╗██╔════╝██╔════╝    ███║ ╚════██║
#   ██████╔╝███████║███████║███████╗█████╗      ╚██║     ██╔╝
#   ██╔═══╝ ██╔══██║██╔══██║╚════██║██╔══╝       ██║    ██╔╝
#   ██║     ██║  ██║██║  ██║███████║███████╗     ██║██╗██║
#   ╚═╝     ╚═╝  ╚═╝╚═╝  ╚═╝╚══════╝╚══════╝     ╚═╝╚═╝╚═╝
#
#   CANGKOK: SigLIP vision_tower + multi_modal_projector Gemma 3 4B IT
#            --> text/merged_bf16 hasil Phase 1
#   Output : UNIFIED_HF_REPO subfolder cangkok/
#
# #####################################################################
# #####################################################################
@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ---
    # 🌱 PHASE 1.5 — CANGKOK Vision Tower Gemma 3 IT → T5Gemma-2
    =====================================================================
    **Trace mekanisme cangkok** (yang dulunya menciptakan repo `...-vision-cangkok`):

    | Langkah | Sumber asli | Yang dikerjakan |
    |---|---|---|
    | 1 | `scripts/tests/verify_vision_weights_3way.py` (cell **CANGKOK**) | Load **target** = `v4-unsloth/merged_bf16` (hasil Phase 1) & **source** = `google/gemma-3-4b-it` |
    | 2 | ibid. | Ekstrak params source yang mengandung `vision_tower` / `multi_modal_projector`, normalisasi prefix `model.` |
    | 3 | ibid. | Iterasi params target, normalisasi prefix `model.encoder.`/`encoder.`, `param.data.copy_()` dari source jika **shape cocok** |
    | 4 | ibid. | Verifikasi: diff target-vs-source harus `< 1e-6` untuk semua param yang dicangkok |
    | 5 | ibid. | `save_pretrained` lokal + upload ke HF **+ processor dari `google/t5gemma-2-4b-4b`** |
    | 6 | `scripts/tests/patch_cangkok_tokenizer.py` | Replace `tokenizer_config.json` dengan versi `merged_bf16` (lengkap: `added_tokens_decoder` + `task_prefix_mapping`) |

    Di pipeline gabungan ini, mekanisme yang **persis sama** dijalankan otomatis, hanya path-nya
    dipindah ke unified repo: target = `text/merged_bf16`, output = subfolder **`cangkok/`**.

    > Kenapa cuma SigLIP + projector yang dicangkok, bukan decoder?
    > Analogi dari `docs/VISION_TRAINING_ANALYSIS_AND_CANGKOK_STRATEGY.md`: transplantasi decoder =
    > memindahkan otak yang dilatih untuk "bahasa A" ke tubuh "bahasa B" (mismatch — pernah dicoba, hancur).
    > SigLIP = mata; "melihat" tidak peduli bahasa apa yang dipakai otak. Hidden size vision tower
    > T5Gemma-2 dan Gemma 3 4B identik (sama-sama SigLIP 400M), jadi cangkok shape-nya selalu cocok.
    """)
    return


# =====================================================================
# CANGKOK: KONFIGURASI
# =====================================================================
@app.cell
def _(UNIFIED_HF_REPO):
    CANGKOK_HF_REPO = UNIFIED_HF_REPO     # 1 repo unified
    CANGKOK_HF_PREFIX = "cangkok"         # subfolder tujuan di unified repo
    CANGKOK_TEXT_SUBFOLDER = "text/merged_bf16"  # sumber target graft dari Phase 1
    CANGKOK_GEMMA3_IT = "google/gemma-3-4b-it"   # donor SigLIP + projector
    CANGKOK_ORIG_T5GEMMA2 = "google/t5gemma-2-4b-4b"  # donor processor (full preprocessor_config)
    CANGKOK_FORCE = False                 # True = graft ulang walau cangkok/ sudah ada
    return (
        CANGKOK_FORCE,
        CANGKOK_GEMMA3_IT,
        CANGKOK_HF_PREFIX,
        CANGKOK_HF_REPO,
        CANGKOK_ORIG_T5GEMMA2,
        CANGKOK_TEXT_SUBFOLDER,
    )


# =====================================================================
# CANGKOK: EKSEKUSI GRAFT + UPLOAD + TOKENIZER PATCH
# =====================================================================
@app.cell
def _(
    CANGKOK_FORCE,
    CANGKOK_GEMMA3_IT,
    CANGKOK_HF_PREFIX,
    CANGKOK_HF_REPO,
    CANGKOK_ORIG_T5GEMMA2,
    CANGKOK_TEXT_SUBFOLDER,
    TEXT_HF_PREFIX,
    gc,
    mo,
    os,
    text_merged_uploaded,
    torch,
):
    from huggingface_hub import HfApi as _CangkokApi
    from huggingface_hub import create_repo as _create_repo
    from transformers import (
        AutoModelForSeq2SeqLM as _AutoSeq2Seq,
        AutoModelForCausalLM as _AutoCausal,
        AutoProcessor as _AutoProc,
    )

    _token = os.environ.get("HF_TOKEN")
    _api = _CangkokApi(token=_token)
    _create_repo(repo_id=CANGKOK_HF_REPO, repo_type="model", private=False, exist_ok=True, token=_token)
    _repo_files = _api.list_repo_files(CANGKOK_HF_REPO)

    # ---- GATE 1: sudah pernah dicangkok? ----
    _already = any(
        f.startswith(f"{CANGKOK_HF_PREFIX}/") and f.endswith("config.json")
        for f in _repo_files
    )
    if _already and not CANGKOK_FORCE:
        print(f"✅ [CANGKOK] Subfolder '{CANGKOK_HF_PREFIX}/' sudah ada di {CANGKOK_HF_REPO}.")
        print("    Skip graft (set CANGKOK_FORCE=True untuk graft ulang).")
        cangkok_ready = True
    else:
        # ---- GATE 2: prasyarat text/merged_bf16 harus ada ----
        _text_merged_ok = any(
            f.startswith(f"{TEXT_HF_PREFIX}/merged_bf16/") and f.endswith("config.json")
            for f in _repo_files
        )
        mo.stop(
            not _text_merged_ok,
            mo.md(
                "⏭️ **[CANGKOK] `text/merged_bf16` belum ada di unified repo.** "
                "Selesaikan Phase 1 (SFT → ORPO → merge) dulu, lalu re-run notebook."
            ),
        )

        print("=" * 90)
        print("  [CANGKOK] SigLIP + Projector Gemma 3 4B IT → text/merged_bf16")
        print("=" * 90)

        # 0. Bebaskan VRAM (dua model bf16 4B akan dimuat)
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        # 1. Load target = text merged hasil Phase 1
        print(f"\n[A] Loading target merged: {CANGKOK_HF_REPO} / {CANGKOK_TEXT_SUBFOLDER} ...")
        _model_tgt = _AutoSeq2Seq.from_pretrained(
            CANGKOK_HF_REPO, subfolder=CANGKOK_TEXT_SUBFOLDER,
            torch_dtype=torch.bfloat16, token=_token,
        )
        print(f"    ✅ {_model_tgt.__class__.__name__}")

        # 2. Load source donor
        print(f"\n[C] Loading donor: {CANGKOK_GEMMA3_IT} ...")
        _model_src = _AutoCausal.from_pretrained(
            CANGKOK_GEMMA3_IT, torch_dtype=torch.bfloat16, token=_token,
        )
        print(f"    ✅ {_model_src.__class__.__name__}")

        # 3. Build source param dict (gemma3 path → normalized)
        _src_params = {}
        for _name, _param in _model_src.named_parameters():
            if "vision_tower" in _name or "multi_modal_projector" in _name:
                _clean = _name[len("model."):] if _name.startswith("model.") else _name
                _src_params[_clean] = _param.detach().cpu()
        print(f"\n  Donor: {len(_src_params)} vision params (SigLIP + projector)")

        # 4. CANGKOK: copy donor → target (path mapping t5gemma2)
        print("\n  Melakukan cangkok...")
        _grafted = 0
        _skipped = 0
        for _name, _param in _model_tgt.named_parameters():
            if "vision_tower" not in _name and "multi_modal_projector" not in _name:
                continue
            _clean = _name
            if _name.startswith("model.encoder."):
                _clean = _name[len("model.encoder."):]
            elif _name.startswith("encoder."):
                _clean = _name[len("encoder."):]

            if _clean in _src_params:
                _src = _src_params[_clean]
                if _src.shape == _param.shape:
                    _param.data.copy_(_src.to(_param.device, _param.dtype))
                    _grafted += 1
                else:
                    print(f"    ⚠️ SHAPE MISMATCH {_clean}: {_src.shape} vs {_param.shape}")
                    _skipped += 1
            else:
                print(f"    ⚠️ Tidak ditemukan di donor: {_clean}")
                _skipped += 1
        print(f"  ✅ Cangkok: {_grafted} params, skip: {_skipped}")

        # 5. Verifikasi cangkok (diff target vs donor, harus < 1e-6)
        print("\n  Verifikasi cangkok...")
        _v_ok = 0
        _v_fail = 0
        for _name, _param in _model_tgt.named_parameters():
            if "vision_tower" not in _name and "multi_modal_projector" not in _name:
                continue
            _clean = _name
            if _name.startswith("model.encoder."):
                _clean = _name[len("model.encoder."):]
            elif _name.startswith("encoder."):
                _clean = _name[len("encoder."):]
            if _clean in _src_params:
                _diff = (_param.detach().cpu().float() - _src_params[_clean].float()).abs().max().item()
                if _diff < 1e-6:
                    _v_ok += 1
                else:
                    print(f"    ❌ Verify fail {_clean}: diff={_diff:.2e}")
                    _v_fail += 1
        print(f"  ✅ Verify: {_v_ok} OK, {_v_fail} fail")
        if _v_fail > 0 or _grafted == 0:
            raise RuntimeError(
                f"[CANGKOK] Gagal: {_grafted} params digraft, {_v_fail} verify fail. "
                "Cek log SHAPE MISMATCH / verify fail di atas."
            )

        # 6. Save lokal + processor donor, lalu upload ke unified repo
        _local_save = "/tmp/v6_vision_cangkok"
        print(f"\n  Saving lokal ke {_local_save}...")
        os.makedirs(_local_save, exist_ok=True)
        _model_tgt.save_pretrained(_local_save, safe_serialization=True)

        # Processor dari T5Gemma2 ORIGINAL (v6 merged text hanya punya tokenizer,
        # tidak ada image processor)
        _processor_orig = _AutoProc.from_pretrained(CANGKOK_ORIG_T5GEMMA2, token=_token)
        _processor_orig.save_pretrained(_local_save)

        print(f"  Uploading ke {CANGKOK_HF_REPO} subfolder '{CANGKOK_HF_PREFIX}/'...")
        _api.upload_folder(
            folder_path=_local_save,
            path_in_repo=CANGKOK_HF_PREFIX,
            repo_id=CANGKOK_HF_REPO,
            repo_type="model",
            commit_message="Cangkok SigLIP + projector dari Gemma 3 4B IT ke text/merged_bf16",
        )

        # 7. TOKENIZER PATCH (dari scripts/tests/patch_cangkok_tokenizer.py):
        # replace tokenizer_config.json cangkok dengan versi text/merged_bf16
        # (lengkap: added_tokens_decoder + task_prefix_mapping untuk <unused1..6>)
        from huggingface_hub import hf_hub_download as _hf_dl
        print(f"\n  Patch tokenizer_config.json dari '{CANGKOK_TEXT_SUBFOLDER}'...")
        _tc_path = _hf_dl(
            repo_id=CANGKOK_HF_REPO,
            filename="tokenizer_config.json",
            subfolder=CANGKOK_TEXT_SUBFOLDER,
            token=_token,
        )
        _api.upload_file(
            path_or_fileobj=_tc_path,
            path_in_repo=f"{CANGKOK_HF_PREFIX}/tokenizer_config.json",
            repo_id=CANGKOK_HF_REPO,
            commit_message="Fix: tokenizer_config dari text/merged_bf16 (lengkap dgn added_tokens_decoder + task_prefix_mapping)",
        )

        print(f"\n  ✅ [CANGKOK] BERHASIL! Model cangkok di: {CANGKOK_HF_REPO} subfolder '{CANGKOK_HF_PREFIX}/'")
        print("     Dipakai Phase 2 sebagai VISION_MODEL_NAME + VISION_SUBFOLDER='cangkok'.")

        # Cleanup
        del _model_tgt, _model_src, _src_params, _processor_orig
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        cangkok_ready = True
    return (cangkok_ready,)


# #####################################################################
# #####################################################################
#
#   ██████╗ ██╗  ██╗ █████╗ ███████╗███████╗    ██████╗
#   ██╔══██╗██║  ██║██╔══██╗██╔════╝██╔════╝    ╚════██╗
#   ██████╔╝███████║███████║███████╗█████╗       █████╔╝
#   ██╔═══╝ ██╔══██║██╔══██║╚════██║██╔══╝      ██╔═══╝
#   ██║     ██║  ██║██║  ██║███████║███████╗    ███████╗
#   ╚═╝     ╚═╝  ╚═╝╚═╝  ╚═╝╚══════╝╚══════╝    ╚══════╝
#
#   VISION PIPELINE  (dari working-molab-v6-vision-unsloth.py — logika identik)
#   Base: UNIFIED_HF_REPO subfolder cangkok/  ->  Vision SFT  ->  ORPO  ->  merge
#   Artifacts: UNIFIED_HF_REPO subfolder vision/
#
# #####################################################################
# #####################################################################
@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ---
    # 📷 PHASE 2 — Multimodal Vision SFT + ORPO Fine-Tuning Pipeline (V6 Unsloth)
    =====================================================================
    Melatih aspek **vision** dari model **T5Gemma-2 4B-4B** menggunakan QLoRA/LoRA via Unsloth.
    Model dasar = hasil Phase 1.5 (text merged + SigLIP/projector Gemma 3 IT cangkok),
    dimuat dari unified repo subfolder **`cangkok/`**.

    **Fitur utama (diwarisi dari notebook vision asli):**
    - Pemrosesan gambar dokumen ilmiah/PDF (lazy — gambar hanya didecode saat collator jalan)
    - Fine-tuning modular: `finetune_vision_layers=False` (SKIP SigLIP untuk menghindari
      Unsloth merge bug), `multi_modal_projector` di-FULL-FT via `modules_to_save`
    - Text-retention mix saat SFT + eval ganda (multimodal & text-only) untuk mendeteksi
      catastrophic forgetting
    """)
    return


# =====================================================================
# KONFIGURASI HYPERPARAMETER VISION
# =====================================================================
@app.cell
def _(BF16, UNIFIED_HF_REPO):
    # MODEL BASE = repo cangkok hasil Phase 1.5 (subfolder "cangkok", bukan root)
    VISION_MODEL_NAME = UNIFIED_HF_REPO
    VISION_SUBFOLDER = "cangkok"
    VISION_LOAD_IN_4BIT = True
    VISION_OUTPUT_DIR = "results/t5gemma2_vision"
    VISION_HF_CHECKPOINT_REPO = UNIFIED_HF_REPO  # artifacts -> subfolder vision/
    VISION_HF_PREFIX = "vision"

    # Dataset JSONL lokal (legacy, tidak dipakai — dataset dimuat dari HF Hub)
    VISION_JSONL_DATASET_PATH = "data/multimodal/train_vision.jsonl"
    VISION_ORPO_DATASET_PATH = "data/preference/orpo_multimodal.jsonl"

    # LoRA config: r=256 (diselaraskan dengan versi teks)
    VISION_LORA_RANK = 256
    VISION_LORA_ALPHA = 512
    VISION_LORA_DROPOUT = 0.2

    # Seq2Seq lengths (cloud Molab 96GB)
    VISION_MAX_SOURCE_LENGTH = 16384
    VISION_MAX_TARGET_LENGTH = 2048
    VISION_MAX_IMAGES_PER_CHAT = 10

    # Training args
    # NOTE: effective LR = LEARNING_RATE / GRADIENT_ACCUMULATION_STEPS.
    # Disetara dengan text-only (1e-5 / 64 = 1.56e-7) agar decoder tidak
    # belajar 8x lebih agresif dari sinyal multimodal yang masih noisy.
    VISION_LEARNING_RATE = 5e-6
    VISION_NUM_EPOCHS_SFT = 2
    VISION_NUM_EPOCHS_ORPO = 1
    VISION_ORPO_BETA = 0.1
    VISION_PER_DEVICE_TRAIN_BATCH_SIZE = 2
    VISION_GRADIENT_ACCUMULATION_STEPS = 32
    VISION_WARMUP_STEPS = 100
    VISION_WEIGHT_DECAY = 0.1
    VISION_LR_SCHEDULER_TYPE = "cosine"
    VISION_LOGGING_STEPS = 10
    VISION_SAVE_TOTAL_LIMIT = 2
    VISION_OPTIM = "paged_adamw_8bit"

    # General random seed and validation split size
    VISION_TEST_SIZE = 0.05

    # Label smoothing & NEFTune
    VISION_LABEL_SMOOTHING_FACTOR = 0.1
    VISION_NEFTUNE_NOISE_ALPHA = 5.0
    VISION_PREDICT_WITH_GENERATE = True
    return (
        BF16,
        VISION_GRADIENT_ACCUMULATION_STEPS,
        VISION_HF_CHECKPOINT_REPO,
        VISION_HF_PREFIX,
        VISION_JSONL_DATASET_PATH,
        VISION_LABEL_SMOOTHING_FACTOR,
        VISION_LEARNING_RATE,
        VISION_LOAD_IN_4BIT,
        VISION_LOGGING_STEPS,
        VISION_LORA_ALPHA,
        VISION_LORA_DROPOUT,
        VISION_LORA_RANK,
        VISION_LR_SCHEDULER_TYPE,
        VISION_MAX_IMAGES_PER_CHAT,
        VISION_MAX_SOURCE_LENGTH,
        VISION_MAX_TARGET_LENGTH,
        VISION_MODEL_NAME,
        VISION_NEFTUNE_NOISE_ALPHA,
        VISION_NUM_EPOCHS_ORPO,
        VISION_NUM_EPOCHS_SFT,
        VISION_OPTIM,
        VISION_ORPO_BETA,
        VISION_ORPO_DATASET_PATH,
        VISION_OUTPUT_DIR,
        VISION_PER_DEVICE_TRAIN_BATCH_SIZE,
        VISION_PREDICT_WITH_GENERATE,
        VISION_SAVE_TOTAL_LIMIT,
        VISION_SUBFOLDER,
        VISION_TEST_SIZE,
        VISION_WARMUP_STEPS,
        VISION_WEIGHT_DECAY,
    )


# =====================================================================
# VISION: AUTO-DETECT PIPELINE STAGE DARI HF HUB
# =====================================================================
@app.cell
def _(VISION_HF_CHECKPOINT_REPO, VISION_HF_PREFIX, cangkok_ready, mo, os):
    from huggingface_hub import HfApi as _StageDetectApi

    _hf_token = os.environ.get("HF_TOKEN")
    _api = _StageDetectApi(token=_hf_token)

    # Default
    vision_current_stage = "sft"
    vision_resume_checkpoint = None

    try:
        # Automatically create repository if it does not exist
        if not _api.repo_exists(repo_id=VISION_HF_CHECKPOINT_REPO):
            print(f"📍 [VISION] Repo '{VISION_HF_CHECKPOINT_REPO}' belum ada. Membuat repositori baru...")
            _api.create_repo(repo_id=VISION_HF_CHECKPOINT_REPO, repo_type="model", private=False, exist_ok=True)

        _repo_files = _api.list_repo_files(VISION_HF_CHECKPOINT_REPO)

        # Cek apakah ORPO sudah selesai
        if any(f.startswith(f"{VISION_HF_PREFIX}/orpo/final_adapter/") for f in _repo_files):
            vision_current_stage = "done"
            print("📍 [VISION] Pipeline stage: DONE — Semua training selesai!")

        # Cek apakah SFT sudah selesai → lanjut ORPO
        elif any(f.startswith(f"{VISION_HF_PREFIX}/sft/final_adapter/") for f in _repo_files):
            vision_current_stage = "orpo"
            # Ada checkpoint ORPO untuk resume?
            _orpo_ckpts = sorted([
                f for f in _repo_files
                if f.startswith(f"{VISION_HF_PREFIX}/orpo/checkpoint-") and "/" in f[len(f"{VISION_HF_PREFIX}/orpo/checkpoint-"):]
            ])
            if _orpo_ckpts:
                vision_resume_checkpoint = True
                print(f"📍 [VISION] Pipeline stage: ORPO (resume dari checkpoint)")
            else:
                print("📍 [VISION] Pipeline stage: ORPO (mulai dari awal, load SFT adapter)")

        # SFT belum selesai
        else:
            vision_current_stage = "sft"
            _sft_ckpts = sorted([
                f for f in _repo_files
                if f.startswith(f"{VISION_HF_PREFIX}/sft/checkpoint-") and "/" in f[len(f"{VISION_HF_PREFIX}/sft/checkpoint-"):]
            ])
            if _sft_ckpts:
                vision_resume_checkpoint = True
                print(f"📍 [VISION] Pipeline stage: SFT (resume dari checkpoint)")
            else:
                print("📍 [VISION] Pipeline stage: SFT (mulai dari awal)")
    except Exception as e:
        print(f"⚠️ Gagal mendeteksi stage VISION: {e}. Mulai SFT dari awal.")

    mo.md(f"**📍 [VISION] Current Stage: `{vision_current_stage}`** | Resume: `{vision_resume_checkpoint}`")
    return vision_current_stage, vision_resume_checkpoint


# =====================================================================
# VISION DATA UTILS (konversi record -> messages multimodal)
# =====================================================================
@app.cell
def _(Image, os):
    def convert_sft_record_to_vision(rec):
        img_paths = rec.get("images", [])
        if not img_paths:
            return None

        pil_images = []
        if isinstance(img_paths[0], str):
            for p in img_paths:
                if os.path.exists(p):
                    try:
                        # Open but do NOT convert or load pixels into RAM yet (lazy-loading)
                        pil_images.append(Image.open(p))
                    except Exception:
                        pass
        else:
            pil_images = img_paths

        if not pil_images:
            return None
        old_messages = rec.get("messages", [])
        new_messages = []
        image_idx = 0
        for msg in old_messages:
            role = msg["role"]
            content = msg["content"]
            if role == "user" and "📷" in content:
                num_images = content.count("📷")
                text_content = content.replace("📷", "").strip()
                new_content = []
                for _ in range(num_images):
                    if image_idx < len(pil_images):
                        new_content.append({"type": "image", "image": pil_images[image_idx]})
                        image_idx += 1
                if text_content:
                    new_content.append({"type": "text", "text": text_content})
                new_messages.append({"role": role, "content": new_content})
            else:
                new_messages.append({"role": role, "content": [{"type": "text", "text": content}]})
        return {"messages": new_messages}

    def unroll_vision_messages_to_sft_samples(messages, processor):
        samples = []
        for i, msg in enumerate(messages):
            if msg["role"] != "assistant":
                continue
            context = messages[:i]
            if not context:
                continue

            # Bersihkan konteks agar format pesannya standar sebelum dilewatkan ke template chat
            clean_context = []
            for m in context:
                clean_content = []
                if isinstance(m.get("content"), list):
                    for b in m["content"]:
                        if isinstance(b, dict):
                            if "image" in b:
                                clean_content.append({"type": "image"})
                            elif "text" in b:
                                clean_content.append({"type": "text", "text": b["text"]})
                else:
                    clean_content = m["content"]
                clean_context.append({"role": m["role"], "content": clean_content})

            prompt_text = processor.apply_chat_template(clean_context, tokenize=False, add_generation_prompt=True)

            images = []
            for m in context:
                if isinstance(m.get("content"), list):
                    for b in m["content"]:
                        if isinstance(b, dict) and "image" in b:
                            images.append(b["image"])

            target_text = ""
            if isinstance(msg["content"], list):
                for b in msg["content"]:
                    if isinstance(b, dict) and "text" in b:
                        target_text = b["text"]
            else:
                target_text = msg["content"]

            if target_text:
                samples.append({"prompt_text": prompt_text, "images": images, "target_text": target_text})
        return samples

    def parse_orpo_prompt_to_messages(prompt_str, img_paths):
        pil_images = []
        if img_paths:
            if isinstance(img_paths[0], str):
                for p in img_paths:
                    if os.path.exists(p):
                        try:
                            pil_images.append(Image.open(p))
                        except Exception:
                            pass
            else:
                pil_images = img_paths

        lines = prompt_str.split("\n")
        raw_messages = []
        current_role = None
        current_lines = []
        for line in lines:
            if line.startswith("system: "):
                if current_role is not None:
                    raw_messages.append((current_role, "\n".join(current_lines)))
                current_role = "system"
                current_lines = [line[8:]]
            elif line.startswith("user: "):
                if current_role is not None:
                    raw_messages.append((current_role, "\n".join(current_lines)))
                current_role = "user"
                current_lines = [line[6:]]
            elif line.startswith("assistant: "):
                if current_role is not None:
                    raw_messages.append((current_role, "\n".join(current_lines)))
                current_role = "assistant"
                current_lines = [line[11:]]
            else:
                current_lines.append(line)
        if current_role is not None:
            raw_messages.append((current_role, "\n".join(current_lines)))
        new_messages = []
        image_idx = 0
        for role, content in raw_messages:
            if role == "user" and "📷" in content:
                num_images = content.count("📷")
                text_content = content.replace("📷", "").strip()
                new_content = []
                for _ in range(num_images):
                    if image_idx < len(pil_images):
                        new_content.append({"type": "image", "image": pil_images[image_idx]})
                        image_idx += 1
                if text_content:
                    new_content.append({"type": "text", "text": text_content})
                new_messages.append({"role": role, "content": new_content})
            else:
                new_messages.append({"role": role, "content": [{"type": "text", "text": content}]})
        merged_messages = []
        for msg in new_messages:
            msg_role = msg["role"]
            msg_content = msg["content"]
            if merged_messages and merged_messages[-1]["role"] == msg_role:
                last_msg = merged_messages.pop()
                merged_content = list(last_msg["content"]) + list(msg_content)
                merged_messages.append({"role": msg_role, "content": merged_content})
            else:
                merged_messages.append({"role": msg_role, "content": list(msg_content)})
        return merged_messages

    return (
        convert_sft_record_to_vision,
        parse_orpo_prompt_to_messages,
        unroll_vision_messages_to_sft_samples,
    )


# =====================================================================
# VISION: COLLATORS & TRAINERS
# =====================================================================
@app.cell
def _(F, SelectiveLabelSmoother, Seq2SeqTrainer, torch):
    class Seq2SeqVisionCollator:
        def __init__(self, processor, max_src, max_tgt, train_dataset=None):
            self.processor = processor
            self.tok = processor.tokenizer
            self.pad_id = self.tok.pad_token_id
            self.eos_id = self.tok.eos_token_id
            self.max_src = max_src
            self.max_tgt = max_tgt
            self.train_dataset = train_dataset
        def __call__(self, batch):
            iids, amasks, pvals, labs = [], [], [], []
            for item in batch:
                images = None
                if "images" in item and item["images"]:
                    images = item["images"]
                elif "dataset_idx" in item and item["dataset_idx"] >= 0 and self.train_dataset is not None:
                    try:
                        full_images = self.train_dataset[item["dataset_idx"]]["images"]
                        indices = item.get("image_indices", [])
                        images = [full_images[i] for i in indices if i < len(full_images)]
                    except Exception:
                        pass

                enc = self.processor(text=item["prompt_text"],
                    images=images if images else None,
                    return_tensors="pt")

                input_ids = enc["input_ids"][0].tolist()
                attention_mask = enc["attention_mask"][0].tolist()

                # Prepend BOS jika belum ada
                if self.tok.bos_token_id is not None and (not input_ids or input_ids[0] != self.tok.bos_token_id):
                    input_ids = [self.tok.bos_token_id] + input_ids
                    attention_mask = [1] + attention_mask

                # Append EOS jika belum ada
                if self.tok.eos_token_id is not None and (not input_ids or input_ids[-1] != self.tok.eos_token_id):
                    input_ids = input_ids + [self.tok.eos_token_id]
                    attention_mask = attention_mask + [1]

                iids.append(torch.tensor(input_ids, dtype=torch.long))
                amasks.append(torch.tensor(attention_mask, dtype=torch.long))

                if "pixel_values" in enc:
                    pvals.append(enc["pixel_values"])
                # Tambahkan <end_of_turn> pada target agar model belajar penutupan turn
                target_formatted = item["target_text"].strip() + "<end_of_turn>"
                tids = self.tok.encode(target_formatted, add_special_tokens=False)
                tids = tids[:self.max_tgt-1] + [self.eos_id]
                labs.append(torch.tensor(tids, dtype=torch.long))
            ii = torch.nn.utils.rnn.pad_sequence(iids, batch_first=True, padding_value=self.pad_id)
            am = torch.nn.utils.rnn.pad_sequence(amasks, batch_first=True, padding_value=0)
            lb = torch.nn.utils.rnn.pad_sequence(labs, batch_first=True, padding_value=-100)
            out = {"input_ids": ii, "attention_mask": am, "labels": lb}
            if pvals:
                out["pixel_values"] = torch.cat(pvals, dim=0) if pvals[0].ndim == 4 else torch.stack(pvals, dim=0)
            return out

    class VisionORPOCollator:
        def __init__(self, processor, max_src, max_tgt, train_dataset=None):
            self.processor = processor
            self.tok = processor.tokenizer
            self.pad_id = self.tok.pad_token_id
            self.eos_id = self.tok.eos_token_id
            self.max_src = max_src
            self.max_tgt = max_tgt
            self.train_dataset = train_dataset
        def _enc_tgt(self, text):
            # Tambahkan <end_of_turn> pada target agar model belajar penutupan turn
            text_formatted = text.strip() + "<end_of_turn>"
            ids = self.tok.encode(text_formatted, add_special_tokens=False)
            return torch.tensor(ids[:self.max_tgt-1] + [self.eos_id], dtype=torch.long)
        def __call__(self, batch):
            iids, amasks, pvals, clabs, rlabs = [], [], [], [], []
            for item in batch:
                images = None
                if "images" in item and item["images"]:
                    images = item["images"]
                elif "dataset_idx" in item and self.train_dataset is not None:
                    try:
                        full_images = self.train_dataset[item["dataset_idx"]]["images"]
                        indices = item.get("image_indices", [])
                        images = [full_images[i] for i in indices if i < len(full_images)]
                    except Exception:
                        pass

                enc = self.processor(text=item["prompt_text"],
                    images=images if images else None,
                    return_tensors="pt")

                input_ids = enc["input_ids"][0].tolist()
                attention_mask = enc["attention_mask"][0].tolist()

                # Prepend BOS jika belum ada
                if self.tok.bos_token_id is not None and (not input_ids or input_ids[0] != self.tok.bos_token_id):
                    input_ids = [self.tok.bos_token_id] + input_ids
                    attention_mask = [1] + attention_mask

                # Append EOS jika belum ada
                if self.tok.eos_token_id is not None and (not input_ids or input_ids[-1] != self.tok.eos_token_id):
                    input_ids = input_ids + [self.tok.eos_token_id]
                    attention_mask = attention_mask + [1]

                iids.append(torch.tensor(input_ids, dtype=torch.long))
                amasks.append(torch.tensor(attention_mask, dtype=torch.long))

                if "pixel_values" in enc:
                    pvals.append(enc["pixel_values"])
                clabs.append(self._enc_tgt(item["chosen_text"]))
                rlabs.append(self._enc_tgt(item["rejected_text"]))
            ii = torch.nn.utils.rnn.pad_sequence(iids, batch_first=True, padding_value=self.pad_id)
            am = torch.nn.utils.rnn.pad_sequence(amasks, batch_first=True, padding_value=0)
            cl = torch.nn.utils.rnn.pad_sequence(clabs, batch_first=True, padding_value=-100)
            rl = torch.nn.utils.rnn.pad_sequence(rlabs, batch_first=True, padding_value=-100)
            out = {"input_ids": ii, "attention_mask": am, "chosen_labels": cl, "rejected_labels": rl}
            if pvals:
                out["pixel_values"] = torch.cat(pvals, dim=0) if pvals[0].ndim == 4 else torch.stack(pvals, dim=0)
            return out

    class VisionORPOTrainer(Seq2SeqTrainer):
        def __init__(self, beta=0.1, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self.beta = beta
        def get_batch_logps(self, logits, labels, average_log_prob=True):
            labels = labels.clone()
            mask = labels != -100
            labels[labels == -100] = 0
            lps = torch.gather(logits.log_softmax(-1), dim=2, index=labels.unsqueeze(2)).squeeze(2)
            if average_log_prob:
                return (lps * mask).sum(-1) / mask.sum(-1).clamp(min=1)
            return (lps * mask).sum(-1)
        def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None, **kwargs):
            cl = inputs.pop("chosen_labels", None)
            rl = inputs.pop("rejected_labels", None)
            if cl is None or rl is None:
                return super().compute_loss(model, inputs, return_outputs, num_items_in_batch, **kwargs)

            # Split forward optimization to prevent CUDA OOM
            base_model = model.base_model.model if hasattr(model, "base_model") and hasattr(model.base_model, "model") else model
            if hasattr(base_model, "get_encoder"):
                encoder = base_model.get_encoder()
            elif hasattr(base_model, "model") and hasattr(base_model.model, "encoder"):
                encoder = base_model.model.encoder
            else:
                encoder = base_model.encoder
            encoder_outputs = encoder(
                input_ids=inputs["input_ids"],
                attention_mask=inputs["attention_mask"],
                pixel_values=inputs.get("pixel_values"),
            )
            co = model(
                encoder_outputs=encoder_outputs,
                labels=cl,
            )
            ro = model(
                encoder_outputs=encoder_outputs,
                labels=rl,
            )
            clp = self.get_batch_logps(co.logits, cl)
            rlp = self.get_batch_logps(ro.logits, rl)
            cp = clp.exp().clamp(1e-7, 1-1e-7)
            rp = rlp.exp().clamp(1e-7, 1-1e-7)
            clo = torch.log(cp / (1 - cp))
            rlo = torch.log(rp / (1 - rp))
            or_loss = -F.logsigmoid(clo - rlo).mean()
            loss = co.loss + self.beta * or_loss
            return (loss, co) if return_outputs else loss

        def prediction_step(self, model, inputs, prediction_loss_only, ignore_keys=None, **kwargs):
            # Seq2SeqTrainer.evaluate() langsung memakai `inputs` mentah buat
            # model.generate(**inputs). Key "chosen_labels"/"rejected_labels" dari
            # collator ORPO bukan kwarg valid buat .generate() -> harus dibuang di
            # sini juga (compute_loss cuma nge-pop pas training, gak kepakai pas eval).
            inputs = dict(inputs)
            cl = inputs.pop("chosen_labels", None)
            inputs.pop("rejected_labels", None)
            if cl is not None and "labels" not in inputs:
                # Pakai jawaban "chosen" (yang disukai) sebagai referensi buat
                # hitung eval loss/ROUGE/BLEU dari hasil generate.
                inputs["labels"] = cl
            return super().prediction_step(model, inputs, prediction_loss_only, ignore_keys=ignore_keys, **kwargs)

        def evaluate(
            self,
            eval_dataset=None,
            ignore_keys=None,
            metric_key_prefix="eval",
        ):
            import math
            import gc
            from unsloth import FastVisionModel
            # Switch ke inference kernels sebelum evaluate/generate (mirip text-only)
            if hasattr(FastVisionModel, "for_inference"):
                FastVisionModel.for_inference(self.model)
            else:
                self.model.eval()
            metrics = super().evaluate(
                eval_dataset=eval_dataset,
                ignore_keys=ignore_keys,
                metric_key_prefix=metric_key_prefix,
            )
            # Hitung perplexity untuk setiap eval sub-dataset (multimodal & text-only).
            # metric_key_prefix selalu "eval" saat trainer memanggil evaluate() dengan
            # dict eval_dataset, jadi kita harus iterate semua key *_loss yang ada.
            for k in list(metrics.keys()):
                if k.endswith("_loss") and k.startswith("eval_"):
                    ppl_key = k.replace("_loss", "_perplexity")
                    try:
                        metrics[ppl_key] = math.exp(metrics[k])
                    except OverflowError:
                        metrics[ppl_key] = float("inf")

            # KRITIS: kembalikan ke training kernels + mode train. Tanpa ini model
            # tetap di eval state -> gradient checkpointing mati -> OOM & degradasi
            # silent saat training dilanjutkan setelah eval.
            if hasattr(FastVisionModel, "for_training"):
                FastVisionModel.for_training(self.model)
            else:
                self.model.train()
            # Vision kernels fullgraph=True -> flush cache rekompilasi Dynamo.
            torch._dynamo.reset()
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            return metrics

        def log(self, logs, start_time=None):
            import math
            for k in list(logs.keys()):
                if k.endswith("_loss") and k.startswith("eval_"):
                    ppl_key = k.replace("_loss", "_perplexity")
                    try:
                        logs[ppl_key] = math.exp(logs[k])
                    except OverflowError:
                        logs[ppl_key] = float("inf")
            super().log(logs, start_time=start_time)

    class VisionCustomSeq2SeqTrainer(Seq2SeqTrainer):
        def __init__(self, suppress_ids=None, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self.model_accepts_loss_kwargs = False
            if self.args.label_smoothing_factor > 0 and suppress_ids is not None:
                self.label_smoother = SelectiveLabelSmoother(
                    epsilon=self.args.label_smoothing_factor,
                    suppress_ids=suppress_ids,
                )

        def compute_loss(
            self, model, inputs, return_outputs=False, num_items_in_batch=None, **kwargs
        ):
            labels = inputs.get("labels")
            outputs = model(**inputs)

            if self.label_smoother is not None and labels is not None:
                loss = self.label_smoother(outputs, labels)
            else:
                if isinstance(outputs, dict) and "logits" in outputs:
                    logits = outputs["logits"]
                elif isinstance(outputs, tuple):
                    logits = outputs[1] if len(outputs) > 1 else outputs[0].logits
                else:
                    logits = outputs.logits
                loss_fct = torch.nn.CrossEntropyLoss(
                    ignore_index=-100, reduction="mean"
                )
                loss = loss_fct(logits.view(-1, logits.size(-1)), labels.view(-1))

            return (loss, outputs) if return_outputs else loss

        def evaluate(
            self,
            eval_dataset=None,
            ignore_keys=None,
            metric_key_prefix="eval",
        ):
            import math
            import gc
            from unsloth import FastVisionModel
            # Switch ke inference kernels sebelum evaluate/generate (mirip text-only v6)
            if hasattr(FastVisionModel, "for_inference"):
                FastVisionModel.for_inference(self.model)
            else:
                self.model.eval()
            metrics = super().evaluate(
                eval_dataset=eval_dataset,
                ignore_keys=ignore_keys,
                metric_key_prefix=metric_key_prefix,
            )
            # Hitung perplexity untuk setiap eval sub-dataset (multimodal & text-only).
            # metric_key_prefix selalu "eval" saat trainer memanggil evaluate() dengan
            # dict eval_dataset, jadi kita harus iterate semua key *_loss yang ada.
            for k in list(metrics.keys()):
                if k.endswith("_loss") and k.startswith("eval_"):
                    ppl_key = k.replace("_loss", "_perplexity")
                    try:
                        metrics[ppl_key] = math.exp(metrics[k])
                    except OverflowError:
                        metrics[ppl_key] = float("inf")

            # KRITIS: kembalikan ke training kernels + mode train. Tanpa ini model
            # tetap di eval state -> gradient checkpointing ("unsloth") mati ->
            # seluruh activation graph ditahan saat training resume -> OOM di step
            # setelah eval pertama (mirip bug OOM step ~97). for_training() juga
            # me-reload triton training kernels agar gradient update kembali optimal.
            if hasattr(FastVisionModel, "for_training"):
                FastVisionModel.for_training(self.model)
            else:
                self.model.train()
            # Vision kernels dikompilasi dengan fullgraph=True. Setiap switch
            # for_inference<->for_training memicu rekompilasi Dynamo. Flush cache
            # agar counter rekompilasi tidak melebihi limit (Hard failure
            # "recompile_limit exceeded") dan sekaligus bebaskan memori graf.
            torch._dynamo.reset()
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            return metrics

        def log(self, logs, start_time=None):
            import math
            for k in list(logs.keys()):
                if k.endswith("_loss") and k.startswith("eval_"):
                    ppl_key = k.replace("_loss", "_perplexity")
                    try:
                        logs[ppl_key] = math.exp(logs[k])
                    except OverflowError:
                        logs[ppl_key] = float("inf")
            super().log(logs, start_time=start_time)

    return (
        Seq2SeqVisionCollator,
        VisionCustomSeq2SeqTrainer,
        VisionORPOCollator,
        VisionORPOTrainer,
    )


# =====================================================================
# VISION CALLBACKS: Training Plot, Notebook Progress, Sample Gen, Hub Upload
# =====================================================================
@app.cell
def _(
    Any,
    TrainerCallback,
    TrainerControl,
    TrainerState,
    TrainingArguments,
    datetime,
    os,
    torch,
):
    class VisionTrainingPlotCallback(TrainerCallback):
        def __init__(self, output_dir: str) -> None:
            self.output_dir = output_dir
            self.chart_path = os.path.join(output_dir, "training_chart.png")
            self.train_steps = []
            self.train_losses = []
            self.eval_data = {}  # {step: {metric_name: value}}

        def on_log(
            self,
            args: TrainingArguments,
            state: TrainerState,
            control: TrainerControl,
            logs: dict[str, float] | None = None,
            **kwargs: Any,
        ) -> None:
            if logs is None:
                return

            # Hapus eval_loss dari logs agar kolom "Validation Loss" (yang selalu "No log")
            # tidak tampil di widget training. Eval sudah di-split menjadi
            # multimodal & text-only, jadi eval_loss gabungan tidak relevan.
            logs.pop("eval_loss", None)
            logs.pop("eval_perplexity", None)

            if "loss" in logs:
                self.train_steps.append(state.global_step)
                self.train_losses.append(float(logs["loss"]))

            is_eval = any(k.startswith("eval_") for k in logs.keys())
            if is_eval:
                step = state.global_step
                if step not in self.eval_data:
                    self.eval_data[step] = {}
                for k, v in logs.items():
                    if k.startswith("eval_"):
                        self.eval_data[step][k] = float(v)

            self.plot_chart()

        def plot_chart(self) -> None:
            import matplotlib.pyplot as plt
            os.makedirs(self.output_dir, exist_ok=True)

            fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))

            # 1. Plot Loss Curve
            if self.train_losses:
                ax1.plot(self.train_steps, self.train_losses, color="#3498DB", linewidth=2, label="Train Loss")

            steps = sorted(self.eval_data.keys())

            m_loss_steps = [s for s in steps if "eval_multimodal_loss" in self.eval_data[s]]
            if m_loss_steps:
                m_losses = [self.eval_data[s]["eval_multimodal_loss"] for s in m_loss_steps]
                ax1.plot(m_loss_steps, m_losses, color="#2ECC71", marker="o", linestyle="--", linewidth=1.5, label="Eval Multimodal Loss")

            t_loss_steps = [s for s in steps if "eval_text_only_loss" in self.eval_data[s]]
            if t_loss_steps:
                t_losses = [self.eval_data[s]["eval_text_only_loss"] for s in t_loss_steps]
                ax1.plot(t_loss_steps, t_losses, color="#E74C3C", marker="x", linestyle=":", linewidth=1.5, label="Eval Text-Only Loss")

            ax1.set_xlabel("Steps")
            ax1.set_ylabel("Loss")
            ax1.set_title("Training & Evaluation Loss Curve")
            ax1.grid(True, alpha=0.3)
            ax1.legend()

            # 2. Plot Perplexity Curve
            m_ppl_steps = [s for s in steps if "eval_multimodal_perplexity" in self.eval_data[s]]
            if m_ppl_steps:
                m_ppls = [self.eval_data[s]["eval_multimodal_perplexity"] for s in m_ppl_steps]
                ax2.plot(m_ppl_steps, m_ppls, color="#1ABC9C", marker="o", linestyle="--", linewidth=1.5, label="Multimodal Perplexity")

            t_ppl_steps = [s for s in steps if "eval_text_only_perplexity" in self.eval_data[s]]
            if t_ppl_steps:
                t_ppls = [self.eval_data[s]["eval_text_only_perplexity"] for s in t_ppl_steps]
                ax2.plot(t_ppl_steps, t_ppls, color="#9B59B6", marker="x", linestyle=":", linewidth=1.5, label="Text-Only Perplexity")

            ax2.set_xlabel("Steps")
            ax2.set_ylabel("Perplexity")
            ax2.set_title("Model Perplexity Curve")
            ax2.grid(True, alpha=0.3)
            ax2.legend()

            # 3. Plot Multimodal Quality Metrics
            metrics_list = [
                ("eval_multimodal_rouge1", "ROUGE-1", "#E67E22", "o"),
                ("eval_multimodal_rouge2", "ROUGE-2", "#D35400", "x"),
                ("eval_multimodal_bleu", "BLEU", "#2980B9", "s"),
                ("eval_multimodal_bertscore_f1", "BERTScore F1", "#8E44AD", "d"),
            ]
            for metric_key, label, color, marker in metrics_list:
                m_steps = [s for s in steps if metric_key in self.eval_data[s]]
                if m_steps:
                    m_vals = [self.eval_data[s][metric_key] for s in m_steps]
                    ax3.plot(m_steps, m_vals, color=color, marker=marker, linestyle="-", linewidth=1.5, label=label)

            ax3.set_xlabel("Steps")
            ax3.set_ylabel("Score (%)")
            ax3.set_title("Multimodal Generation Quality Metrics")
            ax3.grid(True, alpha=0.3)
            ax3.legend()

            # 4. Plot Text-Only Quality Metrics
            metrics_list_text = [
                ("eval_text_only_rouge1", "ROUGE-1", "#E67E22", "o"),
                ("eval_text_only_rouge2", "ROUGE-2", "#D35400", "x"),
                ("eval_text_only_bleu", "BLEU", "#2980B9", "s"),
                ("eval_text_only_bertscore_f1", "BERTScore F1", "#8E44AD", "d"),
            ]
            for metric_key, label, color, marker in metrics_list_text:
                m_steps = [s for s in steps if metric_key in self.eval_data[s]]
                if m_steps:
                    m_vals = [self.eval_data[s][metric_key] for s in m_steps]
                    ax4.plot(m_steps, m_vals, color=color, marker=marker, linestyle="-", linewidth=1.5, label=label)

            ax4.set_xlabel("Steps")
            ax4.set_ylabel("Score (%)")
            ax4.set_title("Text-Only Generation Quality Metrics")
            ax4.grid(True, alpha=0.3)
            ax4.legend()

            plt.tight_layout()
            plt.savefig(self.chart_path, dpi=120)
            plt.close(fig)

    class CleanNotebookProgressCallback(TrainerCallback):
        """
        Pengganti transformers.utils.notebook.NotebookProgressCallback bawaan.

        NotebookProgressCallback bawaan SELALU menambahkan kolom "Validation Loss"
        (hardcoded di on_train_begin & on_evaluate) terlepas dari apakah key
        "eval_loss" benar-benar ada di metrics atau tidak. Karena eval kita sudah
        dipecah jadi Multimodal Loss & Text Only Loss (tidak ada eval_loss
        gabungan), kolom itu selalu tampil "No log". Callback ini meniru semua
        behavior aslinya tapi tanpa kolom "Validation Loss" default tersebut.
        """

        def __init__(self) -> None:
            self.training_tracker = None
            self.prediction_bar = None
            self._force_next_update = False

        def on_train_begin(self, args, state, control, **kwargs) -> None:
            from transformers.trainer_utils import IntervalStrategy
            from transformers.utils.notebook import NotebookTrainingTracker

            self.first_column = "Epoch" if args.eval_strategy == IntervalStrategy.EPOCH else "Step"
            self.training_loss = 0
            self.last_log = 0
            column_names = [self.first_column, "Training Loss"]
            self.training_tracker = NotebookTrainingTracker(state.max_steps, column_names)

        def on_step_end(self, args, state, control, **kwargs) -> None:
            epoch = int(state.epoch) if int(state.epoch) == state.epoch else f"{state.epoch:.2f}"
            self.training_tracker.update(
                state.global_step + 1,
                comment=f"Epoch {epoch}/{state.num_train_epochs}",
                force_update=self._force_next_update,
            )
            self._force_next_update = False

        def on_prediction_step(self, args, state, control, eval_dataloader=None, **kwargs) -> None:
            from transformers.trainer_utils import has_length

            if not has_length(eval_dataloader):
                return
            if self.prediction_bar is None:
                if self.training_tracker is not None:
                    self.prediction_bar = self.training_tracker.add_child(len(eval_dataloader))
                else:
                    from transformers.utils.notebook import NotebookProgressBar
                    self.prediction_bar = NotebookProgressBar(len(eval_dataloader))
                self.prediction_bar.update(1)
            else:
                self.prediction_bar.update(self.prediction_bar.value + 1)

        def on_predict(self, args, state, control, **kwargs) -> None:
            if self.prediction_bar is not None:
                self.prediction_bar.close()
            self.prediction_bar = None

        def on_log(self, args, state, control, logs=None, **kwargs) -> None:
            from transformers.trainer_utils import IntervalStrategy

            if args.eval_strategy == IntervalStrategy.NO and logs is not None and "loss" in logs:
                values = {"Training Loss": logs["loss"], "Step": state.global_step}
                self.training_tracker.write_line(values)

        def on_evaluate(self, args, state, control, metrics=None, **kwargs) -> None:
            import re as _re
            import IPython.display as disp
            from transformers.trainer_utils import IntervalStrategy
            from transformers.utils.notebook import text_to_html_table

            self.first_column = "Epoch" if args.eval_strategy == IntervalStrategy.EPOCH else "Step"

            # Tidak seperti bawaan: TIDAK ada default "Validation Loss": "No log" di sini.
            values = {"Training Loss": "No log"}
            for log in reversed(state.log_history):
                if "loss" in log:
                    values["Training Loss"] = log["loss"]
                    break

            if self.first_column == "Epoch":
                values["Epoch"] = int(state.epoch)
            else:
                values["Step"] = state.global_step

            if metrics is None:
                metrics = {}
            metric_key_prefix = "eval"
            for k in metrics:
                if k.endswith("_loss"):
                    metric_key_prefix = _re.sub(r"_loss$", "", k)
            metrics.pop("total_flos", None)
            metrics.pop("epoch", None)
            metrics.pop(f"{metric_key_prefix}_runtime", None)
            metrics.pop(f"{metric_key_prefix}_samples_per_second", None)
            metrics.pop(f"{metric_key_prefix}_steps_per_second", None)
            metrics.pop(f"{metric_key_prefix}_model_preparation_time", None)

            for k, v in metrics.items():
                splits = k.split("_")
                name = " ".join(part.capitalize() for part in splits[1:])
                # Catatan: sengaja TIDAK rename name == "Loss" -> "Validation Loss"
                # seperti versi bawaan, karena kita mau tiap eval dataset punya
                # kolom sendiri (mis. "Multimodal Loss", "Text Only Loss").
                values[name] = v

            if self.training_tracker is not None:
                tt = self.training_tracker
                tt.write_line(values)
                tt.remove_child()
                self._force_next_update = True
            else:
                disp.display(disp.HTML(text_to_html_table([list(values.keys()), list(values.values())])))

            self.prediction_bar = None

        def on_train_end(self, args, state, control, **kwargs) -> None:
            if self.training_tracker is not None:
                self.training_tracker.update(
                    state.global_step,
                    comment=f"Epoch {int(state.epoch)}/{state.num_train_epochs}",
                    force_update=True,
                )
                self.training_tracker = None

    class VisionSampleGenerationCallback(TrainerCallback):
        def __init__(
            self,
            processor: Any,
            eval_samples: list[dict],
            output_dir: str,
            eval_every_n_steps: int = 50,
            temperature: float = 0.7,
            top_p: float = 0.9,
            repetition_penalty: float = 1.2,
            bad_words_ids: list[list[int]] | None = None,
        ) -> None:
            self.processor = processor
            self.tokenizer = processor.tokenizer
            self.eval_samples = eval_samples
            self.output_dir = output_dir
            self.eval_every_n_steps = eval_every_n_steps
            self.log_path = os.path.join(output_dir, "eval_samples.txt")
            self._eot_id = self.tokenizer.convert_tokens_to_ids("<end_of_turn>")
            self._eos_id = self.tokenizer.eos_token_id or 1
            self._stop_ids = list({self._eot_id, self._eos_id})
            self.temperature = temperature
            self.top_p = top_p
            self.repetition_penalty = repetition_penalty
            self.bad_words_ids = bad_words_ids

        def on_step_end(
            self,
            args: TrainingArguments,
            state: TrainerState,
            control: TrainerControl,
            model: Any = None,
            **kwargs: Any,
        ) -> None:
            if (
                state.global_step == 0
                or state.global_step % self.eval_every_n_steps != 0
            ):
                return
            if model is None:
                return

            from unsloth import FastVisionModel
            if hasattr(FastVisionModel, "for_inference"):
                FastVisionModel.for_inference(model)
            else:
                model.eval()
            timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            lines = [
                f"\n{'=' * 60}",
                f"Step {state.global_step} | {timestamp}",
                f"{'=' * 60}",
            ]

            import gc
            gc.collect()
            torch.cuda.empty_cache()

            with torch.no_grad():
                pad_id = (
                    self.tokenizer.pad_token_id
                    if self.tokenizer.pad_token_id is not None
                    else self._eos_id
                )

                for idx, sample in enumerate(self.eval_samples):
                    inputs = self.processor(
                        text=[sample["prompt_text"]],
                        images=sample["images"] if sample["images"] else None,
                        return_tensors="pt"
                    ).to(model.device)

                    outputs = getattr(model, "generate")(
                        **inputs,
                        max_new_tokens=1024,
                        do_sample=True,
                        temperature=self.temperature,
                        top_p=self.top_p,
                        repetition_penalty=self.repetition_penalty,
                        eos_token_id=self._stop_ids,
                        pad_token_id=pad_id,
                        bad_words_ids=self.bad_words_ids,
                    )

                    gen_ids = outputs[0].cpu()
                    raw_response = self.tokenizer.decode(gen_ids, skip_special_tokens=True)

                    # Immediately free GPU tensors to prevent cumulative OOM
                    del inputs, outputs
                    torch.cuda.empty_cache()

                    query = sample["prompt_text"].strip()
                    target = sample["target_text"].strip()
                    response = raw_response.strip()

                    words = response.split()
                    is_repetitive = (
                        len(set(words)) < max(1, len(words) * 0.3) if words else True
                    )
                    flag = " ⚠️ REPETITIVE" if is_repetitive else " ✅"

                    lines.append(f"\nQ: {query}")
                    lines.append(f"Expected Target: {target}")
                    lines.append(f"Model Response: {response}{flag}")

            from unsloth import FastVisionModel
            if hasattr(FastVisionModel, "for_training"):
                FastVisionModel.for_training(model)
            else:
                model.train()

            # Vision kernels fullgraph=True -> flush cache rekompilasi Dynamo
            # agar tidak tembus recompile_limit saat training dilanjutkan.
            torch._dynamo.reset()
            gc.collect()
            torch.cuda.empty_cache()

            with open(self.log_path, "a", encoding="utf-8") as f:
                f.write("\n".join(lines) + "\n")

            print(f"\n[BEHAVIOR EVAL @ step {state.global_step}]")
            for line in lines[3:]:
                if (
                    line.startswith("Q:")
                    or line.startswith("Model Response:")
                    or line.startswith("Expected Target:")
                ):
                    print(f"  {line}")

    class VisionHubUploadCallback(TrainerCallback):
        def __init__(self, repo_id: str, stage: str, hf_prefix: str, token: str | None = None, output_dir: str | None = None) -> None:
            self.repo_id = repo_id
            self.stage = stage          # "sft" / "orpo" — dipakai untuk nama file artifact lokal
            self.hf_prefix = hf_prefix  # "vision" — subfolder di unified repo
            self.token = token
            self.output_dir = output_dir

        def on_save(
            self,
            args: TrainingArguments,
            state: TrainerState,
            control: TrainerControl,
            **kwargs: Any,
        ) -> TrainerControl:
            from huggingface_hub import HfApi
            _api = HfApi(token=self.token)
            checkpoint_name = f"checkpoint-{state.global_step}"
            local_checkpoint_path = os.path.join(args.output_dir, checkpoint_name)

            try:
                # Ensure the repository is created before uploading checkpoints
                _api.create_repo(repo_id=self.repo_id, repo_type="model", private=False, exist_ok=True)
                print(f"\n📤 Uploading {checkpoint_name} to HF {self.hf_prefix}/{self.stage}/...")
                _api.upload_folder(
                    folder_path=local_checkpoint_path,
                    repo_id=self.repo_id,
                    path_in_repo=f"{self.hf_prefix}/{self.stage}/{checkpoint_name}",
                    repo_type="model",
                )

                if self.output_dir:
                    for artifact_name in ["training_chart.png", f"{self.stage}_eval_samples_multimodal.txt", f"{self.stage}_eval_samples_text_only.txt"]:
                        local_art_path = os.path.join(self.output_dir, artifact_name)
                        if os.path.exists(local_art_path):
                            _api.upload_file(
                                path_or_fileobj=local_art_path,
                                path_in_repo=f"{self.hf_prefix}/{self.stage}/{artifact_name}",
                                repo_id=self.repo_id,
                                repo_type="model",
                            )
                print(f"✅ {checkpoint_name} + artifacts uploaded!")
            except Exception as e:
                print(f"⚠️ Upload gagal untuk {checkpoint_name}: {e}")
            return control

    return (
        CleanNotebookProgressCallback,
        VisionHubUploadCallback,
        VisionSampleGenerationCallback,
        VisionTrainingPlotCallback,
    )


@app.cell
def _(load_dataset):
    # Load dan format dataset untuk SFT
    print("[VISION] Memuat dataset SFT dari Hugging Face Hub (daruokta/t5gemma2-indonesia-vision-formatted)...")
    vision_train_dataset = load_dataset("daruokta/t5gemma2-indonesia-vision-formatted", "vision_sft", split="train")
    print(f"✅ [VISION] SFT Dataset berhasil dimuat dari Hugging Face Hub: {len(vision_train_dataset)} sampel.")
    return (vision_train_dataset,)


@app.cell
def _(
    ALL_SUPPRESS_IDS,
    AutoProcessor,
    FastVisionModel,
    SEED,
    VISION_HF_CHECKPOINT_REPO,
    VISION_HF_PREFIX,
    VISION_LOAD_IN_4BIT,
    VISION_LORA_ALPHA,
    VISION_LORA_DROPOUT,
    VISION_LORA_RANK,
    VISION_MODEL_NAME,
    VISION_OUTPUT_DIR,
    VISION_SUBFOLDER,
    apply_logit_mask,
    os,
    vision_current_stage,
):
    vision_model = None
    vision_tokenizer = None
    vision_processor = None

    if vision_current_stage == "done":
        print("[VISION] Semua tahapan training selesai. Lewati pemuatan model.")
    else:
        _hf_token = os.environ.get("HF_TOKEN")

        # Tentukan sumber pemuatan model
        if vision_current_stage == "orpo":
            _model_path = os.path.join(VISION_OUTPUT_DIR, "sft", "final_adapter")
            if not os.path.exists(_model_path):
                from huggingface_hub import snapshot_download as _resume_snap
                print("📥 [VISION] Downloading SFT final adapter dari HF untuk ORPO...")
                _resume_snap(
                    repo_id=VISION_HF_CHECKPOINT_REPO,
                    local_dir=_model_path,
                    allow_patterns=[f"{VISION_HF_PREFIX}/sft/final_adapter/**"],
                    token=_hf_token,
                )
                _sub_dir = os.path.join(_model_path, VISION_HF_PREFIX, "sft", "final_adapter")
                if os.path.exists(_sub_dir):
                    import shutil as _shutil_load
                    for _item in os.listdir(_sub_dir):
                        _src = os.path.join(_sub_dir, _item)
                        _dst = os.path.join(_model_path, _item)
                        if os.path.exists(_dst):
                            if os.path.isdir(_dst):
                                _shutil_load.rmtree(_dst)
                            else:
                                os.remove(_dst)
                        _shutil_load.move(_src, _dst)
                    _shutil_load.rmtree(os.path.join(_model_path, VISION_HF_PREFIX))
            print(f"[VISION] Loading SFT model dari adapter path: {_model_path}")
        else:
            # SFT: Load base model = hasil cangkok (subfolder "cangkok" di unified repo)
            _model_path = VISION_MODEL_NAME
            print(f"[VISION] Loading base model dari {_model_path} (subfolder '{VISION_SUBFOLDER}')...")

        _load_kwargs = dict(
            model_name=_model_path,
            load_in_4bit=VISION_LOAD_IN_4BIT,
            use_gradient_checkpointing="unsloth",
            token=_hf_token,
        )
        if vision_current_stage == "sft" and VISION_SUBFOLDER:
            _load_kwargs["subfolder"] = VISION_SUBFOLDER

        vision_model, vision_tokenizer = FastVisionModel.from_pretrained(**_load_kwargs)

        # Reset max_length to silence warning about max_new_tokens taking precedence
        vision_model.config.max_length = None
        if hasattr(vision_model, "generation_config") and vision_model.generation_config is not None:
            vision_model.generation_config.max_length = None

        # Load processor dari base model (subfolder cangkok saat sft)
        _proc_kwargs = dict(token=_hf_token)
        if vision_current_stage == "sft" and VISION_SUBFOLDER:
            _proc_kwargs["subfolder"] = VISION_SUBFOLDER
        vision_processor = AutoProcessor.from_pretrained(VISION_MODEL_NAME, **_proc_kwargs)

        from unsloth.chat_templates import get_chat_template
        vision_tokenizer = get_chat_template(vision_tokenizer, chat_template="gemma-3")
        vision_processor.chat_template = vision_tokenizer.chat_template
        if hasattr(vision_processor, "tokenizer"):
            vision_processor.tokenizer.chat_template = vision_tokenizer.chat_template

        # Nonaktifkan penambahan bos_token otomatis untuk menghindari bos_token ganda saat inferensi
        vision_tokenizer.add_bos_token = False
        if hasattr(vision_processor, "tokenizer"):
            vision_processor.tokenizer.add_bos_token = False

        # LoRA Config hanya untuk SFT (karena ORPO me-load model yang sudah memiliki LoRA adapter)
        if vision_current_stage == "sft":
            print("[VISION] Applying PEFT LoRA (vision_tower=SKIP, projector=FULL FT)...")
            vision_model = FastVisionModel.get_peft_model(
                vision_model,
                finetune_vision_layers=False,      # ⚠️ SKIP vision tower (SigLIP) to avoid Unsloth merge bug
                finetune_language_layers=True,
                finetune_attention_modules=True,
                finetune_mlp_modules=True,
                modules_to_save=["multi_modal_projector"],  # FULL FT projector
                r=VISION_LORA_RANK,
                lora_alpha=VISION_LORA_ALPHA,
                lora_dropout=VISION_LORA_DROPOUT,
                bias="none",
                random_state=SEED,
                use_rslora=True,
            )
        else:
            print("[VISION] Model has already been loaded with PEFT adapter (from SFT). Skipping get_peft_model.")

        if not hasattr(vision_model.config, "text_config"):
            type(vision_model.config).text_config = property(lambda self: self.decoder)
            type(vision_model.config).get_text_config = lambda self, *args, **kwargs: self.decoder

        apply_logit_mask(vision_model, ALL_SUPPRESS_IDS)
        FastVisionModel.for_training(vision_model)
    return vision_model, vision_processor, vision_tokenizer


# =====================================================================
# VISION SFT TRAINING CELL
# =====================================================================
@app.cell
def _(
    ALL_SUPPRESS_IDS,
    Any,
    BF16,
    CleanNotebookProgressCallback,
    Dataset,
    GrokAdEMAMix,
    SEED,
    Seq2SeqTrainingArguments,
    Seq2SeqVisionCollator,
    VISION_GRADIENT_ACCUMULATION_STEPS,
    VISION_HF_CHECKPOINT_REPO,
    VISION_HF_PREFIX,
    VISION_LABEL_SMOOTHING_FACTOR,
    VISION_LEARNING_RATE,
    VISION_LOGGING_STEPS,
    VISION_LR_SCHEDULER_TYPE,
    VISION_MAX_SOURCE_LENGTH,
    VISION_MAX_TARGET_LENGTH,
    VISION_NEFTUNE_NOISE_ALPHA,
    VISION_NUM_EPOCHS_SFT,
    VISION_OPTIM,
    VISION_OUTPUT_DIR,
    VISION_PER_DEVICE_TRAIN_BATCH_SIZE,
    VISION_PREDICT_WITH_GENERATE,
    VISION_SAVE_TOTAL_LIMIT,
    VISION_TEST_SIZE,
    VISION_WARMUP_STEPS,
    VISION_WEIGHT_DECAY,
    VisionCustomSeq2SeqTrainer,
    VisionHubUploadCallback,
    VisionSampleGenerationCallback,
    VisionTrainingPlotCallback,
    bertscore_metric,
    bleu_metric,
    cast,
    exact_match_metric,
    format_encoder_from_raw,
    gc,
    get_scheduler,
    load_dataset,
    meteor_metric,
    mo,
    np,
    os,
    rouge_metric,
    torch,
    traceback,
    vision_current_stage,
    vision_model,
    vision_processor,
    vision_resume_checkpoint,
    vision_train_dataset,
):
    mo.stop(
        vision_current_stage != "sft",
        mo.md("ℹ️ **[VISION] Bukan tahap SFT (atau SFT sudah selesai). Melewati training SFT.**")
    )
    mo.stop(
        vision_train_dataset is None,
        mo.md("❌ **[VISION] Dataset SFT tidak ditemukan, training SFT dibatalkan.**")
    )

    # Active memory cleanup from previous attempts
    vision_sft_trainer = None
    _optimizer = None
    _lr_scheduler = None
    if "vision_model" in globals() and globals()["vision_model"] is not None:
        try:
            globals()["vision_model"].zero_grad(set_to_none=True)
        except Exception:
            pass
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # Unroll SFT dataset using ONLY text columns to avoid slow Hugging Face image loading/decoding
    print("[VISION] Unrolling SFT dataset (text-only pass)...")
    sft_formatted = []
    messages_list = vision_train_dataset["messages"]
    _arrow_images_sft = vision_train_dataset._data.column("images")
    for _idx_sft, _msgs_sft in enumerate(messages_list):
        _num_actual_images = len(_arrow_images_sft[_idx_sft])
        _image_idx = 0
        clean_context = []
        for _msg in _msgs_sft:
            _role_sft = _msg["role"]
            _content_sft = _msg["content"]
            if _role_sft == "user" and "📷" in _content_sft:
                _num_images_sft = _content_sft.count("📷")
                _text_content_sft = _content_sft.replace("📷", "").strip()
                clean_content = []
                for _ in range(_num_images_sft):
                    if _image_idx < _num_actual_images:
                        clean_content.append({"type": "image"})
                        _image_idx += 1
                if _text_content_sft:
                    clean_content.append({"type": "text", "text": _text_content_sft})
                clean_context.append({"role": _role_sft, "content": clean_content})
            else:
                clean_context.append({"role": _role_sft, "content": [{"type": "text", "text": _content_sft}]})

        for i, msg in enumerate(clean_context):
            if msg["role"] != "assistant":
                continue
            context = clean_context[:i]
            if not context:
                continue

            prompt_text = vision_processor.apply_chat_template(context, tokenize=False, add_generation_prompt=True)

            # Count image blocks in context up to this turn
            _num_context_images = 0
            for _m in context:
                for _b in _m["content"]:
                    if isinstance(_b, dict) and _b.get("type") == "image":
                        _num_context_images += 1

            target_text = ""
            if isinstance(msg["content"], list):
                for b in msg["content"]:
                    if isinstance(b, dict) and "text" in b:
                        target_text = b["text"]
            else:
                target_text = msg["content"]

            if target_text:
                sft_formatted.append({
                    "prompt_text": prompt_text,
                    "target_text": target_text,
                    "dataset_idx": _idx_sft,
                    "image_indices": list(range(_num_context_images))
                })
    print(f"✅ [VISION] Vision SFT samples unrolled: {len(sft_formatted)} samples.")

    # Load and format text retention data to prevent catastrophic forgetting
    # Select complete conversations by chat_idx so turns are never cut off in the middle
    print("[VISION] Memuat text retention dataset (100 percakapan utuh chat_sft + 100 IndoQA)...")
    _text_retention_formatted = []
    try:
        _ret_chat_ds = load_dataset("daruokta/t5gemma2-indonesia-chat-formatted", "chat_sft", split="train")
        _ret_indoqa_ds = load_dataset("daruokta/t5gemma2-indonesia-chat-formatted", "indoqa_sft", split="train")

        _chat_rows = [dict(_r) for _r in _ret_chat_ds]
        _indoqa_rows = [dict(_r) for _r in _ret_indoqa_ds]

        import random as _rng_ret
        _rng_ret.seed(SEED)

        # Group chat_sft rows by chat_idx to keep multiturn conversations intact
        _chat_groups = {}
        for _r in _chat_rows:
            _c_idx = _r.get("chat_idx", _r.get("id"))
            if _c_idx not in _chat_groups:
                _chat_groups[_c_idx] = []
            _chat_groups[_c_idx].append(_r)

        # Shuffle conversation keys and pick 100 complete conversations
        _group_keys = list(_chat_groups.keys())
        _rng_ret.shuffle(_group_keys)
        _selected_chat_keys = _group_keys[:min(100, len(_group_keys))]

        _selected_ret_rows = []
        for _k in _selected_chat_keys:
            _selected_ret_rows.extend(_chat_groups[_k])

        # Pick 100 random samples from IndoQA (single turn)
        _rng_ret.shuffle(_indoqa_rows)
        _selected_ret_rows.extend(_indoqa_rows[:min(100, len(_indoqa_rows))])

        for _row in _selected_ret_rows:
            _pt = format_encoder_from_raw(_row["input"])
            _tt = _row["target"]
            _text_retention_formatted.append({
                "prompt_text": _pt,
                "target_text": _tt,
                "dataset_idx": -1,
                "image_indices": [],
                "images": []
            })
        print(f"✅ [VISION] Ditambahkan {len(_text_retention_formatted)} sampel retensi teks utuh (dari {len(_selected_chat_keys)} percakapan chat + 100 IndoQA).")
    except Exception as e:
        print(f"⚠️ [VISION] Gagal memuat dataset retensi teks: {e}")

    sft_formatted.extend(_text_retention_formatted)
    import random as _rng_mix
    _rng_mix.seed(SEED)
    _rng_mix.shuffle(sft_formatted)

    sft_dataset = Dataset.from_list(sft_formatted)
    print(f"✅ [VISION] Combined SFT dataset (Vision + Text Retention): {len(sft_dataset)} samples")

    # Splitting Train & Validation
    split_ds = sft_dataset.train_test_split(test_size=VISION_TEST_SIZE, seed=SEED)
    vision_sft_train_dataset = split_ds["train"]
    # Limit evaluation dataset to 30 samples to avoid CUDA OOM during predict_with_generate
    vision_sft_eval_dataset = split_ds["test"].select(range(min(len(split_ds["test"]), 30)))
    print(f"  [VISION] SFT Train size: {len(vision_sft_train_dataset)} | SFT Eval size: {len(vision_sft_eval_dataset)}")

    # Load and format text-only validation dataset
    print("[VISION] Loading text-only SFT validation dataset from HF Hub...")
    _text_only_eval_dataset = None
    try:
        _val_chat_ds = load_dataset("daruokta/t5gemma2-indonesia-chat-formatted", "chat_sft", split="validation")
        _val_indoqa_ds = load_dataset("daruokta/t5gemma2-indonesia-chat-formatted", "indoqa_sft", split="validation")
        _val_chat_samples = [dict(_row) for _row in _val_chat_ds]
        _val_indoqa_samples = [dict(_row) for _row in _val_indoqa_ds]

        import random as _rng_sft
        _raw_text_only_samples = _val_chat_samples + _val_indoqa_samples
        _rng_sft.seed(42)
        _rng_sft.shuffle(_raw_text_only_samples)
        # Limit text-only evaluation to 30 samples to avoid CUDA OOM during predict_with_generate
        _raw_text_only_samples = _raw_text_only_samples[:30]

        _text_only_formatted = []
        for _row in _raw_text_only_samples:
            _pt = format_encoder_from_raw(_row["input"])
            _tt = _row["target"]
            _text_only_formatted.append({
                "prompt_text": _pt,
                "images": [],
                "target_text": _tt
            })
        _text_only_eval_dataset = Dataset.from_list(_text_only_formatted)
        print(f"  [VISION] Text-Only Eval size: {len(_text_only_eval_dataset)}")
    except Exception as e:
        print(f"⚠️ [VISION] Gagal memuat dataset validasi teks untuk SFT: {e}")

    sft_eval_datasets = {"multimodal": vision_sft_eval_dataset}
    if _text_only_eval_dataset is not None:
        sft_eval_datasets["text_only"] = _text_only_eval_dataset

    vision_sft_output_dir = os.path.join(VISION_OUTPUT_DIR, "sft")
    sft_collator = Seq2SeqVisionCollator(vision_processor, VISION_MAX_SOURCE_LENGTH, VISION_MAX_TARGET_LENGTH, vision_train_dataset)

    # Setup qualitative generation samples (similar to V6 text-only)
    _sft_val_rows = list(vision_sft_eval_dataset)
    _n_eval_gen = min(len(_sft_val_rows), 20)
    _eval_generation_samples = []
    for _item_sft in _sft_val_rows[:_n_eval_gen]:
        _full_imgs = vision_train_dataset[_item_sft["dataset_idx"]]["images"] if "dataset_idx" in _item_sft else []
        _indices = _item_sft.get("image_indices", [])
        _subset_imgs = [_full_imgs[i] for i in _indices if i < len(_full_imgs)]
        _eval_generation_samples.append({
            "prompt_text": _item_sft["prompt_text"],
            "target_text": _item_sft["target_text"],
            "images": _subset_imgs
        })

    # Define compute metrics
    def _compute_metrics(eval_preds):
        metrics = {}

        if rouge_metric is None and bleu_metric is None:
            return metrics
        preds, labels = eval_preds
        if isinstance(preds, tuple):
            preds = preds[0]
        tok = cast(Any, vision_processor.tokenizer)

        if preds.ndim == 3:
            preds = preds.argmax(axis=-1)

        labels = np.where(labels != -100, labels, tok.pad_token_id)
        preds = np.where(preds != -100, preds, tok.pad_token_id)
        decoded_preds = tok.batch_decode(preds, skip_special_tokens=True)
        decoded_labels = tok.batch_decode(labels, skip_special_tokens=True)
        decoded_preds = [pred.strip() for pred in decoded_preds]
        decoded_labels = [label.strip() for label in decoded_labels]

        if rouge_metric is not None:
            try:
                result = cast(Any, rouge_metric).compute(predictions=decoded_preds, references=decoded_labels, use_stemmer=False)
                if result is not None:
                    for key, value in result.items():
                        metrics[key] = value * 100
            except Exception as e:
                print(f"Error during ROUGE: {e}")

        if bleu_metric is not None:
            try:
                formatted_labels = [[label] for label in decoded_labels]
                bleu_result = cast(Any, bleu_metric).compute(predictions=decoded_preds, references=formatted_labels)
                if bleu_result is not None and "bleu" in bleu_result:
                    metrics["bleu"] = bleu_result["bleu"] * 100
            except Exception as e:
                print(f"Error during BLEU: {e}")

        if exact_match_metric is not None:
            try:
                em_result = cast(Any, exact_match_metric).compute(predictions=decoded_preds, references=decoded_labels)
                if em_result is not None and "exact_match" in em_result:
                    metrics["exact_match"] = em_result["exact_match"] * 100
            except Exception as e:
                print(f"Error during Exact Match: {e}")

        if bertscore_metric is not None:
            try:
                bertscore_result = cast(Any, bertscore_metric).compute(
                    predictions=decoded_preds, references=decoded_labels,
                    model_type="google/embeddinggemma-300m", num_layers=12, lang="id"
                )
                if bertscore_result is not None and "f1" in bertscore_result:
                    metrics["bertscore_f1"] = np.mean(bertscore_result["f1"]) * 100
            except Exception as e:
                print(f"Error during BERTScore: {e}")

        if meteor_metric is not None:
            try:
                meteor_result = cast(Any, meteor_metric).compute(predictions=decoded_preds, references=decoded_labels)
                if meteor_result is not None and "meteor" in meteor_result:
                    metrics["meteor"] = meteor_result["meteor"] * 100
            except Exception as e:
                print(f"Error during METEOR: {e}")

        return metrics

    # Instantiate GrokAdEMAMix Optimizer with split learning rates
    print("[VISION] Menggunakan optimizer: GrokAdEMAMix (Split LR: Encoder=0.2x, Decoder=0.2x, Projector=0.05x, VisionTower=0.0x)")
    _encoder_params = []
    _decoder_params = []
    _projector_params = []
    _vision_tower_params = []
    for _name, _param in vision_model.named_parameters():
        if _param.requires_grad:
            if "multi_modal_projector" in _name:
                _projector_params.append(_param)
            elif "vision_tower" in _name:
                _vision_tower_params.append(_param)
            elif "encoder" in _name:
                _encoder_params.append(_param)
            elif "decoder" in _name:
                _decoder_params.append(_param)
            else:
                _decoder_params.append(_param)

    _optimizer = GrokAdEMAMix([
        {"params": _encoder_params, "lr": VISION_LEARNING_RATE * 0.2},
        {"params": _decoder_params, "lr": VISION_LEARNING_RATE * 0.2},
        {"params": _projector_params, "lr": VISION_LEARNING_RATE * 0.05},
        {"params": _vision_tower_params, "lr": 0.0}
    ], weight_decay=VISION_WEIGHT_DECAY, grok_alpha=2.0, grok_lamb=0.98)

    # Calculate steps for Cosine Scheduler
    _num_update_steps_per_epoch = max(
        1, len(vision_sft_train_dataset) // (VISION_PER_DEVICE_TRAIN_BATCH_SIZE * VISION_GRADIENT_ACCUMULATION_STEPS)
    )
    _max_steps = _num_update_steps_per_epoch * VISION_NUM_EPOCHS_SFT

    _lr_scheduler = get_scheduler(
        name=VISION_LR_SCHEDULER_TYPE,
        optimizer=_optimizer,
        num_warmup_steps=VISION_WARMUP_STEPS,
        num_training_steps=_max_steps,
    )

    # Callbacks (same as V6 text-only)
    _bad_words_ids = [
        [id_] for id_ in ALL_SUPPRESS_IDS if id_ < cast(Any, vision_model).config.vocab_size
    ]
    _plot_callback = VisionTrainingPlotCallback(output_dir=vision_sft_output_dir)
    _progress_callback = CleanNotebookProgressCallback()

    _sample_callback_multimodal = VisionSampleGenerationCallback(
        processor=vision_processor,
        eval_samples=_eval_generation_samples,
        output_dir=vision_sft_output_dir,
        eval_every_n_steps=50,
        temperature=0.7,
        top_p=0.9,
        repetition_penalty=1.2,
        bad_words_ids=_bad_words_ids,
    )
    _sample_callback_multimodal.log_path = os.path.join(vision_sft_output_dir, "sft_eval_samples_multimodal.txt")

    # Setup qualitative generation samples for text-only validation
    _text_only_val_rows = list(_text_only_eval_dataset) if _text_only_eval_dataset is not None else []
    _n_text_only_eval_gen = min(len(_text_only_val_rows), 20)
    _text_only_eval_generation_samples = _text_only_val_rows[:_n_text_only_eval_gen]

    _sample_callback_text_only = VisionSampleGenerationCallback(
        processor=vision_processor,
        eval_samples=_text_only_eval_generation_samples,
        output_dir=vision_sft_output_dir,
        eval_every_n_steps=50,
        temperature=0.7,
        top_p=0.9,
        repetition_penalty=1.2,
        bad_words_ids=_bad_words_ids,
    )
    _sample_callback_text_only.log_path = os.path.join(vision_sft_output_dir, "sft_eval_samples_text_only.txt")

    _hub_callback = VisionHubUploadCallback(
        repo_id=VISION_HF_CHECKPOINT_REPO,
        stage="sft",
        hf_prefix=VISION_HF_PREFIX,
        token=os.environ.get("HF_TOKEN"),
        output_dir=vision_sft_output_dir,
    )

    print("[VISION] Starting VisionCustomSeq2SeqTrainer for SFT...")
    vision_sft_trainer = VisionCustomSeq2SeqTrainer(
        suppress_ids=ALL_SUPPRESS_IDS,
        model=vision_model,
        args=Seq2SeqTrainingArguments(
            per_device_train_batch_size=VISION_PER_DEVICE_TRAIN_BATCH_SIZE,
            per_device_eval_batch_size=1,  # Eval one sample at a time to prevent OOM during generate
            gradient_accumulation_steps=VISION_GRADIENT_ACCUMULATION_STEPS,
            eval_accumulation_steps=1,  # Move predictions to CPU immediately after each batch
            learning_rate=VISION_LEARNING_RATE,
            num_train_epochs=VISION_NUM_EPOCHS_SFT,
            warmup_steps=VISION_WARMUP_STEPS,
            weight_decay=VISION_WEIGHT_DECAY,
            max_grad_norm=5.0,  # Clip gradients to prevent grad norm spikes
            lr_scheduler_type=VISION_LR_SCHEDULER_TYPE,
            logging_steps=VISION_LOGGING_STEPS,
            save_strategy="steps",
            save_steps=50,
            save_total_limit=VISION_SAVE_TOTAL_LIMIT,
            output_dir=vision_sft_output_dir,
            remove_unused_columns=False,
            fp16=False,
            bf16=BF16,
            optim=VISION_OPTIM,
            label_smoothing_factor=VISION_LABEL_SMOOTHING_FACTOR,
            neftune_noise_alpha=VISION_NEFTUNE_NOISE_ALPHA,
            gradient_checkpointing=True,
            eval_strategy="steps",
            eval_steps=50,
            report_to="none",
            predict_with_generate=VISION_PREDICT_WITH_GENERATE,
            generation_max_length=VISION_MAX_TARGET_LENGTH,
        ),
        train_dataset=vision_sft_train_dataset,
        eval_dataset=sft_eval_datasets,
        data_collator=sft_collator,
        optimizers=(_optimizer, _lr_scheduler),
        compute_metrics=_compute_metrics,
        callbacks=[_plot_callback, _progress_callback, _sample_callback_multimodal, _sample_callback_text_only, _hub_callback],
    )

    # Buang NotebookProgressCallback bawaan transformers (kalau ada) supaya
    # tabel progress bawaan (dengan kolom "Validation Loss" yang selalu "No log")
    # tidak ikut ter-render berdampingan dengan CleanNotebookProgressCallback.
    from transformers.utils.notebook import NotebookProgressCallback as _HFNotebookProgressCallback
    vision_sft_trainer.remove_callback(_HFNotebookProgressCallback)

    # === RESUME FROM HF CHECKPOINT ===
    _resume_from = None
    if vision_resume_checkpoint:
        try:
            from huggingface_hub import snapshot_download as _resume_snap
            from huggingface_hub import HfApi as _ResumeApi

            _api = _ResumeApi(token=os.environ.get("HF_TOKEN"))
            _files = _api.list_repo_files(repo_id=VISION_HF_CHECKPOINT_REPO)

            _ckpts = list(set([f.split('/')[2] for f in _files if f.startswith(f"{VISION_HF_PREFIX}/sft/checkpoint-")]))
            if _ckpts:
                _ckpts.sort(key=lambda x: int(x.split('-')[1]))
                _latest_ckpt = _ckpts[-1]
            else:
                _latest_ckpt = "checkpoint-*"

            print(f"\n📥 [VISION] Downloading {_latest_ckpt} (sft) dari HF untuk resume...")
            _resume_snap(
                repo_id=VISION_HF_CHECKPOINT_REPO,
                local_dir=vision_sft_output_dir,
                allow_patterns=[f"{VISION_HF_PREFIX}/sft/{_latest_ckpt}/**"],
                token=os.environ.get("HF_TOKEN"),
            )
            _sub_dir = os.path.join(vision_sft_output_dir, VISION_HF_PREFIX, "sft")
            if os.path.exists(_sub_dir):
                import shutil as _shutil_SFT
                for _item in os.listdir(_sub_dir):
                    _src = os.path.join(_sub_dir, _item)
                    _dst = os.path.join(vision_sft_output_dir, _item)
                    if os.path.isdir(_src) and _item.startswith("checkpoint-"):
                        if os.path.exists(_dst):
                            _shutil_SFT.rmtree(_dst)
                        _shutil_SFT.move(_src, _dst)
                _shutil_SFT.rmtree(os.path.join(vision_sft_output_dir, VISION_HF_PREFIX))

            _checkpoints = sorted([
                d for d in os.listdir(vision_sft_output_dir)
                if d.startswith("checkpoint-") and os.path.isdir(os.path.join(vision_sft_output_dir, d))
            ])
            if _checkpoints:
                _resume_from = True
                print(f"✅ [VISION] Ditemukan {len(_checkpoints)} checkpoint(s). Resume dari yang terbaru!")
            else:
                print("⚠️ [VISION] Tidak ada checkpoint valid ditemukan. Mulai dari awal.")
        except Exception as e:
            print(f"⚠️ [VISION] Gagal download checkpoint: {e}. Mulai dari awal.")

    vision_sft_result = None
    try:
        vision_sft_result = vision_sft_trainer.train(resume_from_checkpoint=_resume_from)
        print(f"✅ [VISION] SFT selesai! Loss: {vision_sft_result.training_loss:.4f}")

        # Save final SFT model & processor
        vision_sft_final_path = os.path.join(vision_sft_output_dir, "final_adapter")
        print(f"💾 [VISION] Saving final SFT adapter ke {vision_sft_final_path}...")
        vision_sft_trainer.save_model(vision_sft_final_path)
        vision_processor.save_pretrained(vision_sft_final_path)

        # Upload final adapter to HF Hub
        if os.environ.get("HF_TOKEN"):
            try:
                from huggingface_hub import HfApi as _HfApi_SFT
                _final_api = _HfApi_SFT(token=os.environ.get("HF_TOKEN"))
                _final_api.create_repo(repo_id=VISION_HF_CHECKPOINT_REPO, repo_type="model", private=False, exist_ok=True)
                print("📤 [VISION] Uploading final SFT adapter ke HF Hub...")
                _final_api.upload_folder(
                    folder_path=vision_sft_final_path,
                    repo_id=VISION_HF_CHECKPOINT_REPO,
                    path_in_repo=f"{VISION_HF_PREFIX}/sft/final_adapter",
                    repo_type="model",
                )
                print("✅ [VISION] Upload final SFT adapter sukses!")
            except Exception as e:
                print(f"⚠️ [VISION] Gagal upload final SFT adapter: {e}")
    except Exception as e:
        print(f"❌ [VISION] SFT gagal: {e}")
        traceback.print_exc()
    finally:
        vision_sft_trainer = None
        _optimizer = None
        _lr_scheduler = None
        if "vision_model" in globals() and globals()["vision_model"] is not None:
            try:
                globals()["vision_model"].zero_grad(set_to_none=True)
            except Exception:
                pass
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    return


# =====================================================================
# VISION ORPO TRAINING CELL
# =====================================================================
@app.cell
def _(
    ALL_SUPPRESS_IDS,
    Any,
    BF16,
    CleanNotebookProgressCallback,
    Dataset,
    GrokAdEMAMix,
    SEED,
    Seq2SeqTrainingArguments,
    VISION_GRADIENT_ACCUMULATION_STEPS,
    VISION_HF_CHECKPOINT_REPO,
    VISION_HF_PREFIX,
    VISION_LEARNING_RATE,
    VISION_LOGGING_STEPS,
    VISION_LR_SCHEDULER_TYPE,
    VISION_MAX_SOURCE_LENGTH,
    VISION_MAX_TARGET_LENGTH,
    VISION_NUM_EPOCHS_ORPO,
    VISION_OPTIM,
    VISION_ORPO_BETA,
    VISION_OUTPUT_DIR,
    VISION_PER_DEVICE_TRAIN_BATCH_SIZE,
    VISION_PREDICT_WITH_GENERATE,
    VISION_SAVE_TOTAL_LIMIT,
    VISION_TEST_SIZE,
    VISION_WARMUP_STEPS,
    VISION_WEIGHT_DECAY,
    VisionHubUploadCallback,
    VisionORPOCollator,
    VisionORPOTrainer,
    VisionSampleGenerationCallback,
    VisionTrainingPlotCallback,
    bertscore_metric,
    bleu_metric,
    cast,
    exact_match_metric,
    format_encoder_from_raw,
    gc,
    get_scheduler,
    load_dataset,
    meteor_metric,
    mo,
    np,
    os,
    parse_orpo_prompt_to_messages,
    rouge_metric,
    torch,
    traceback,
    vision_current_stage,
    vision_model,
    vision_processor,
    vision_resume_checkpoint,
):
    # Re-detect pipeline stage FRESH dari HF Hub. Cell deteksi stage awal hanya
    # jalan sekali di awal notebook dan nilainya di-cache marimo. Saat notebook
    # mulai, `vision/sft/final_adapter/` belum ada -> vision_current_stage = "sft".
    # Setelah SFT selesai & upload, cell deteksi itu TIDAK re-run, sehingga
    # vision_current_stage tetap stale "sft" dan mo.stop(... != "orpo") SALAH
    # me-skip ORPO tepat setelah SFT selesai dalam sesi yang sama. Deteksi ulang
    # di sini memastikan ORPO jalan berdasarkan state repo yang sebenarnya.
    from huggingface_hub import HfApi as _OrpoStageApi
    _fresh_stage = vision_current_stage
    _fresh_resume = vision_resume_checkpoint
    try:
        _stage_api = _OrpoStageApi(token=os.environ.get("HF_TOKEN"))
        _stage_files = _stage_api.list_repo_files(VISION_HF_CHECKPOINT_REPO)
        if any(f.startswith(f"{VISION_HF_PREFIX}/orpo/final_adapter/") for f in _stage_files):
            _fresh_stage = "done"
        elif any(f.startswith(f"{VISION_HF_PREFIX}/sft/final_adapter/") for f in _stage_files):
            _fresh_stage = "orpo"
            _fresh_resume = any(
                f.startswith(f"{VISION_HF_PREFIX}/orpo/checkpoint-") and "/" in f[len(f"{VISION_HF_PREFIX}/orpo/checkpoint-"):]
                for f in _stage_files
            )
        else:
            _fresh_stage = "sft"
            _fresh_resume = None
        print(f"📍 [VISION] Fresh stage detection untuk ORPO: `{_fresh_stage}` (resume={_fresh_resume})")
    except Exception as _e_stage:
        print(f"⚠️ Gagal re-detect stage untuk ORPO ({_e_stage}); pakai vision_current_stage={vision_current_stage}.")

    mo.stop(
        _fresh_stage != "orpo",
        mo.md(f"ℹ️ **[VISION] Bukan tahap ORPO (deteksi fresh: `{_fresh_stage}`). Melewati training ORPO.**")
    )

    # Active memory cleanup from previous SFT/ORPO attempts
    vision_orpo_trainer = None
    _optimizer = None
    _lr_scheduler = None
    if "vision_model" in globals() and globals()["vision_model"] is not None:
        try:
            globals()["vision_model"].zero_grad(set_to_none=True)
        except Exception:
            pass
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # ORPO Vision Training
    print(f"\n=== [VISION] ORPO Vision Training (beta={VISION_ORPO_BETA}) ===")
    vision_orpo_output_dir = os.path.join(VISION_OUTPUT_DIR, "orpo")

    # Load ORPO dataset directly from Hugging Face Hub
    print("[VISION] Memuat dataset ORPO dari Hugging Face Hub...")
    raw_orpo_dataset = load_dataset("daruokta/t5gemma2-indonesia-vision-formatted", "vision_orpo", split="train")
    print(f"✅ [VISION] ORPO dataset dimuat dari Hugging Face: {len(raw_orpo_dataset)} sampel.")

    # Format ORPO dataset using ONLY text columns to avoid slow Hugging Face image loading/decoding
    print("[VISION] Formatting ORPO dataset (text-only pass)...")
    orpo_formatted = []
    prompts_list = raw_orpo_dataset["prompt"]
    chosen_list = raw_orpo_dataset["chosen"]
    rejected_list = raw_orpo_dataset["rejected"]

    for _idx_orpo in range(len(prompts_list)):
        prompt_str = prompts_list[_idx_orpo]
        chosen_raw = chosen_list[_idx_orpo].replace("assistant: ", "", 1).strip()
        rejected_raw = rejected_list[_idx_orpo].replace("assistant: ", "", 1).strip()

        # Parse prompt to messages text-only
        lines = prompt_str.split("\n")
        raw_messages = []
        current_role = None
        current_lines = []
        for line in lines:
            if line.startswith("system: "):
                if current_role is not None:
                    raw_messages.append((current_role, "\n".join(current_lines)))
                current_role = "system"
                current_lines = [line[8:]]
            elif line.startswith("user: "):
                if current_role is not None:
                    raw_messages.append((current_role, "\n".join(current_lines)))
                current_role = "user"
                current_lines = [line[6:]]
            elif line.startswith("assistant: "):
                if current_role is not None:
                    raw_messages.append((current_role, "\n".join(current_lines)))
                current_role = "assistant"
                current_lines = [line[11:]]
            else:
                current_lines.append(line)
        if current_role is not None:
            raw_messages.append((current_role, "\n".join(current_lines)))

        # Merge messages and count 📷
        new_messages = []
        for _role_orpo, _content_orpo in raw_messages:
            if _role_orpo == "user" and "📷" in _content_orpo:
                _num_images_orpo = _content_orpo.count("📷")
                _text_content_orpo = _content_orpo.replace("📷", "").strip()
                new_content = []
                for _ in range(_num_images_orpo):
                    new_content.append({"type": "image"})
                if _text_content_orpo:
                    new_content.append({"type": "text", "text": _text_content_orpo})
                new_messages.append({"role": _role_orpo, "content": new_content})
            else:
                new_messages.append({"role": _role_orpo, "content": [{"type": "text", "text": _content_orpo}]})

        # Gabungkan turn dengan role sama yang berurutan (mis. dua "user:" beruntun
        # akibat prompt yang kepecah pas parsing baris demi baris). Tanpa ini,
        # apply_chat_template bisa gagal dengan
        # "Conversation roles must alternate user/assistant/user/assistant/...".
        # Logika sama persis dengan yang dipakai di parse_orpo_prompt_to_messages().
        _merged_messages_orpo = []
        for _msg_orpo in new_messages:
            _role_orpo = _msg_orpo["role"]
            _content_orpo = _msg_orpo["content"]
            if _merged_messages_orpo and _merged_messages_orpo[-1]["role"] == _role_orpo:
                _last_msg_orpo = _merged_messages_orpo.pop()
                _merged_content_orpo = list(_last_msg_orpo["content"]) + list(_content_orpo)
                _merged_messages_orpo.append({"role": _role_orpo, "content": _merged_content_orpo})
            else:
                _merged_messages_orpo.append({"role": _role_orpo, "content": list(_content_orpo)})
        new_messages = _merged_messages_orpo

        # Apply chat template
        pt = vision_processor.apply_chat_template(new_messages, tokenize=False, add_generation_prompt=True)

        if chosen_raw.endswith("<end_of_turn>"):
            chosen_raw = chosen_raw[:-len("<end_of_turn>")].strip()
        if rejected_raw.endswith("<end_of_turn>"):
            rejected_raw = rejected_raw[:-len("<end_of_turn>")].strip()

        orpo_formatted.append({
            "prompt_text": pt,
            "chosen_text": chosen_raw,
            "rejected_text": rejected_raw,
            "dataset_idx": _idx_orpo
        })
    orpo_dataset = Dataset.from_list(orpo_formatted)
    print(f"✅ [VISION] ORPO dataset siap: {len(orpo_dataset)} sampel.")

    # Split Train / Validation
    split_orpo = orpo_dataset.train_test_split(test_size=VISION_TEST_SIZE, seed=SEED)
    vision_orpo_train_dataset = split_orpo["train"]
    vision_orpo_eval_dataset = split_orpo["test"]
    print(f"  [VISION] ORPO Train size: {len(vision_orpo_train_dataset)} | ORPO Eval size: {len(vision_orpo_eval_dataset)}")

    # Load and format text-only validation dataset for ORPO
    print("[VISION] Loading text-only ORPO validation dataset from HF Hub...")
    _text_only_eval_dataset = None
    try:
        _val_chat_ds = load_dataset("daruokta/t5gemma2-indonesia-chat-formatted", "chat_sft", split="validation")
        _val_indoqa_ds = load_dataset("daruokta/t5gemma2-indonesia-chat-formatted", "indoqa_sft", split="validation")
        _val_chat_samples = [dict(_row) for _row in _val_chat_ds]
        _val_indoqa_samples = [dict(_row) for _row in _val_indoqa_ds]

        import random as _rng_orpo
        _raw_text_only_samples = _val_chat_samples + _val_indoqa_samples
        _rng_orpo.seed(42)
        _rng_orpo.shuffle(_raw_text_only_samples)
        _raw_text_only_samples = _raw_text_only_samples[:100]

        _text_only_formatted = []
        for _row in _raw_text_only_samples:
            _pt = format_encoder_from_raw(_row["input"])
            _tt = _row["target"]
            _text_only_formatted.append({
                "prompt_text": _pt,
                "images": [],
                "chosen_text": _tt,
                "rejected_text": "Maaf, saya kurang tahu mengenai hal tersebut."
            })
        _text_only_eval_dataset = Dataset.from_list(_text_only_formatted)
        print(f"  [VISION] Text-Only ORPO Eval size: {len(_text_only_eval_dataset)}")
    except Exception as e:
        print(f"⚠️ [VISION] Gagal memuat dataset validasi teks untuk ORPO: {e}")

    orpo_eval_datasets = {"multimodal": vision_orpo_eval_dataset}
    if _text_only_eval_dataset is not None:
        orpo_eval_datasets["text_only"] = _text_only_eval_dataset

    # Setup qualitative generation samples
    _orpo_val_rows = list(vision_orpo_eval_dataset)
    _n_eval_gen = min(len(_orpo_val_rows), 20)
    _eval_generation_samples = []
    for _item_orpo in _orpo_val_rows[:_n_eval_gen]:
        _eval_generation_samples.append({
            "prompt_text": _item_orpo["prompt_text"],
            "target_text": _item_orpo["chosen_text"],
            "images": raw_orpo_dataset[_item_orpo["dataset_idx"]]["images"] if "dataset_idx" in _item_orpo else []
        })

    # Define compute metrics
    def _compute_metrics(eval_preds):
        metrics = {}
        if not VISION_PREDICT_WITH_GENERATE:
            return metrics
        if rouge_metric is None and bleu_metric is None:
            return metrics
        preds, labels = eval_preds
        if isinstance(preds, tuple):
            preds = preds[0]
        tok = cast(Any, vision_processor.tokenizer)

        if preds.ndim == 3:
            preds = preds.argmax(axis=-1)

        labels = np.where(labels != -100, labels, tok.pad_token_id)
        preds = np.where(preds != -100, preds, tok.pad_token_id)
        decoded_preds = tok.batch_decode(preds, skip_special_tokens=True)
        decoded_labels = tok.batch_decode(labels, skip_special_tokens=True)
        decoded_preds = [pred.strip() for pred in decoded_preds]
        decoded_labels = [label.strip() for label in decoded_labels]

        if rouge_metric is not None:
            try:
                result = cast(Any, rouge_metric).compute(predictions=decoded_preds, references=decoded_labels, use_stemmer=False)
                if result is not None:
                    for key, value in result.items():
                        metrics[key] = value * 100
            except Exception as e:
                print(f"Error during ROUGE: {e}")

        if bleu_metric is not None:
            try:
                formatted_labels = [[label] for label in decoded_labels]
                bleu_result = cast(Any, bleu_metric).compute(predictions=decoded_preds, references=formatted_labels)
                if bleu_result is not None and "bleu" in bleu_result:
                    metrics["bleu"] = bleu_result["bleu"] * 100
            except Exception as e:
                print(f"Error during BLEU: {e}")

        if exact_match_metric is not None:
            try:
                em_result = cast(Any, exact_match_metric).compute(predictions=decoded_preds, references=decoded_labels)
                if em_result is not None and "exact_match" in em_result:
                    metrics["exact_match"] = em_result["exact_match"] * 100
            except Exception as e:
                print(f"Error during Exact Match: {e}")

        if bertscore_metric is not None:
            try:
                bertscore_result = cast(Any, bertscore_metric).compute(
                    predictions=decoded_preds, references=decoded_labels,
                    model_type="google/embeddinggemma-300m", num_layers=12, lang="id"
                )
                if bertscore_result is not None and "f1" in bertscore_result:
                    metrics["bertscore_f1"] = np.mean(bertscore_result["f1"]) * 100
            except Exception as e:
                print(f"Error during BERTScore: {e}")

        if meteor_metric is not None:
            try:
                meteor_result = cast(Any, meteor_metric).compute(predictions=decoded_preds, references=decoded_labels)
                if meteor_result is not None and "meteor" in meteor_result:
                    metrics["meteor"] = meteor_result["meteor"] * 100
            except Exception as e:
                print(f"Error during METEOR: {e}")

        return metrics

    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    orpo_collator = VisionORPOCollator(vision_processor, VISION_MAX_SOURCE_LENGTH, VISION_MAX_TARGET_LENGTH, raw_orpo_dataset)

    # Instantiate GrokAdEMAMix Optimizer with split learning rates
    print("[VISION] Menggunakan optimizer: GrokAdEMAMix (Split LR: Encoder=0.5x, Decoder=1.0x, Projector=1.0x, VisionTower=0.5x)")
    _encoder_params = []
    _decoder_params = []
    _projector_params = []
    _vision_tower_params = []
    for _name, _param in vision_model.named_parameters():
        if _param.requires_grad:
            if "multi_modal_projector" in _name:
                _projector_params.append(_param)
            elif "vision_tower" in _name:
                _vision_tower_params.append(_param)
            elif "encoder" in _name:
                _encoder_params.append(_param)
            elif "decoder" in _name:
                _decoder_params.append(_param)
            else:
                _decoder_params.append(_param)

    _optimizer = GrokAdEMAMix([
        {"params": _encoder_params, "lr": VISION_LEARNING_RATE * 0.5},
        {"params": _decoder_params, "lr": VISION_LEARNING_RATE},
        {"params": _projector_params, "lr": VISION_LEARNING_RATE},
        {"params": _vision_tower_params, "lr": VISION_LEARNING_RATE * 0.5}
    ], weight_decay=VISION_WEIGHT_DECAY, grok_alpha=2.0, grok_lamb=0.98)

    # Calculate steps for Cosine Scheduler
    _num_update_steps_per_epoch = max(
        1, len(vision_orpo_train_dataset) // (VISION_PER_DEVICE_TRAIN_BATCH_SIZE * VISION_GRADIENT_ACCUMULATION_STEPS)
    )
    _max_steps = _num_update_steps_per_epoch * VISION_NUM_EPOCHS_ORPO

    _lr_scheduler = get_scheduler(
        name=VISION_LR_SCHEDULER_TYPE,
        optimizer=_optimizer,
        num_warmup_steps=VISION_WARMUP_STEPS,
        num_training_steps=_max_steps,
    )

    # Callbacks (same as V6 text-only)
    _bad_words_ids = [
        [id_] for id_ in ALL_SUPPRESS_IDS if id_ < cast(Any, vision_model).config.vocab_size
    ]
    _plot_callback = VisionTrainingPlotCallback(output_dir=vision_orpo_output_dir)
    _progress_callback = CleanNotebookProgressCallback()

    _sample_callback_multimodal = VisionSampleGenerationCallback(
        processor=vision_processor,
        eval_samples=_eval_generation_samples,
        output_dir=vision_orpo_output_dir,
        eval_every_n_steps=50,
        temperature=0.7,
        top_p=0.9,
        repetition_penalty=1.2,
        bad_words_ids=_bad_words_ids,
    )
    _sample_callback_multimodal.log_path = os.path.join(vision_orpo_output_dir, "orpo_eval_samples_multimodal.txt")

    # Setup qualitative generation samples for text-only validation in ORPO
    _text_only_val_rows = list(_text_only_eval_dataset) if _text_only_eval_dataset is not None else []
    _n_text_only_eval_gen = min(len(_text_only_val_rows), 20)
    _text_only_eval_generation_samples = []
    for _item_text in _text_only_val_rows[:_n_text_only_eval_gen]:
        _text_only_eval_generation_samples.append({
            "prompt_text": _item_text["prompt_text"],
            "images": _item_text["images"],
            "target_text": _item_text["chosen_text"]
        })

    _sample_callback_text_only = VisionSampleGenerationCallback(
        processor=vision_processor,
        eval_samples=_text_only_eval_generation_samples,
        output_dir=vision_orpo_output_dir,
        eval_every_n_steps=50,
        temperature=0.7,
        top_p=0.9,
        repetition_penalty=1.2,
        bad_words_ids=_bad_words_ids,
    )
    _sample_callback_text_only.log_path = os.path.join(vision_orpo_output_dir, "orpo_eval_samples_text_only.txt")

    _hub_callback = VisionHubUploadCallback(
        repo_id=VISION_HF_CHECKPOINT_REPO,
        stage="orpo",
        hf_prefix=VISION_HF_PREFIX,
        token=os.environ.get("HF_TOKEN"),
        output_dir=vision_orpo_output_dir,
    )

    vision_orpo_result = None
    try:
        vision_orpo_trainer = VisionORPOTrainer(
            beta=VISION_ORPO_BETA, model=vision_model,
            args=Seq2SeqTrainingArguments(
                per_device_train_batch_size=VISION_PER_DEVICE_TRAIN_BATCH_SIZE,
                per_device_eval_batch_size=1,  # Eval one sample at a time to prevent OOM
                gradient_accumulation_steps=VISION_GRADIENT_ACCUMULATION_STEPS,
                eval_accumulation_steps=1,  # Move predictions to CPU immediately
                learning_rate=VISION_LEARNING_RATE,
                num_train_epochs=VISION_NUM_EPOCHS_ORPO,
                warmup_steps=VISION_WARMUP_STEPS,
                weight_decay=VISION_WEIGHT_DECAY,
                lr_scheduler_type=VISION_LR_SCHEDULER_TYPE,
                logging_steps=VISION_LOGGING_STEPS,
                save_strategy="steps",
                save_steps=50,
                save_total_limit=VISION_SAVE_TOTAL_LIMIT,
                output_dir=vision_orpo_output_dir,
                remove_unused_columns=False,
                fp16=False, bf16=BF16, optim=VISION_OPTIM,
                gradient_checkpointing=True,
                eval_strategy="steps",
                eval_steps=50,
                report_to="none",
                predict_with_generate=VISION_PREDICT_WITH_GENERATE,
                generation_max_length=VISION_MAX_TARGET_LENGTH,
            ),
            train_dataset=vision_orpo_train_dataset,
            eval_dataset=orpo_eval_datasets,
            data_collator=orpo_collator,
            optimizers=(_optimizer, _lr_scheduler),
            compute_metrics=_compute_metrics,
            callbacks=[_plot_callback, _progress_callback, _sample_callback_multimodal, _sample_callback_text_only, _hub_callback],
        )
        # Buang NotebookProgressCallback bawaan transformers (kalau ada) supaya
        # tabel progress bawaan (dengan kolom "Validation Loss" yang selalu "No log")
        # tidak ikut ter-render berdampingan dengan CleanNotebookProgressCallback.
        from transformers.utils.notebook import NotebookProgressCallback as _HFNotebookProgressCallback
        vision_orpo_trainer.remove_callback(_HFNotebookProgressCallback)

        # === RESUME FROM HF CHECKPOINT ===
        # Pakai hasil deteksi fresh (`_fresh_resume`) alih-alih `vision_resume_checkpoint`
        # yang mungkin stale dari cell deteksi stage awal.
        _resume_from = None
        if _fresh_resume:
            try:
                from huggingface_hub import snapshot_download as _resume_snap
                from huggingface_hub import HfApi as _ResumeApi

                _api = _ResumeApi(token=os.environ.get("HF_TOKEN"))
                _files = _api.list_repo_files(repo_id=VISION_HF_CHECKPOINT_REPO)

                _ckpts = list(set([f.split('/')[2] for f in _files if f.startswith(f"{VISION_HF_PREFIX}/orpo/checkpoint-")]))
                if _ckpts:
                    _ckpts.sort(key=lambda x: int(x.split('-')[1]))
                    _latest_ckpt = _ckpts[-1]
                else:
                    _latest_ckpt = "checkpoint-*"

                print(f"\n📥 [VISION] Downloading {_latest_ckpt} (orpo) dari HF untuk resume...")
                _resume_snap(
                    repo_id=VISION_HF_CHECKPOINT_REPO,
                    local_dir=vision_orpo_output_dir,
                    allow_patterns=[f"{VISION_HF_PREFIX}/orpo/{_latest_ckpt}/**"],
                    token=os.environ.get("HF_TOKEN"),
                )
                _sub_dir = os.path.join(vision_orpo_output_dir, VISION_HF_PREFIX, "orpo")
                if os.path.exists(_sub_dir):
                    import shutil as _shutil_ORPO
                    for _item in os.listdir(_sub_dir):
                        _src = os.path.join(_sub_dir, _item)
                        _dst = os.path.join(vision_orpo_output_dir, _item)
                        if os.path.isdir(_src) and _item.startswith("checkpoint-"):
                            if os.path.exists(_dst):
                                _shutil_ORPO.rmtree(_dst)
                            _shutil_ORPO.move(_src, _dst)
                    _shutil_ORPO.rmtree(os.path.join(vision_orpo_output_dir, VISION_HF_PREFIX))

                _checkpoints = sorted([
                    d for d in os.listdir(vision_orpo_output_dir)
                    if d.startswith("checkpoint-") and os.path.isdir(os.path.join(vision_orpo_output_dir, d))
                ])
                if _checkpoints:
                    _resume_from = True
                    print(f"✅ [VISION] Ditemukan {len(_checkpoints)} checkpoint(s). Resume dari yang terbaru!")
                else:
                    print("⚠️ [VISION] Tidak ada checkpoint valid ditemukan. Mulai dari awal.")
            except Exception as e:
                print(f"⚠️ [VISION] Gagal download checkpoint: {e}. Mulai dari awal.")

        vision_orpo_result = vision_orpo_trainer.train(resume_from_checkpoint=_resume_from)
        print(f"✅ [VISION] ORPO selesai! Loss: {vision_orpo_result.training_loss:.4f}")

        # Save final ORPO model & processor
        vision_orpo_final_path = os.path.join(vision_orpo_output_dir, "final_adapter")
        print(f"💾 [VISION] Saving final ORPO adapter ke {vision_orpo_final_path}...")
        vision_orpo_trainer.save_model(vision_orpo_final_path)
        vision_processor.save_pretrained(vision_orpo_final_path)

        # Upload final adapter to HF Hub
        if os.environ.get("HF_TOKEN"):
            try:
                from huggingface_hub import HfApi as _HfApi_ORPO
                _final_api = _HfApi_ORPO(token=os.environ.get("HF_TOKEN"))
                _final_api.create_repo(repo_id=VISION_HF_CHECKPOINT_REPO, repo_type="model", private=False, exist_ok=True)
                print("📤 [VISION] Uploading final ORPO adapter ke HF Hub...")
                _final_api.upload_folder(
                    folder_path=vision_orpo_final_path,
                    repo_id=VISION_HF_CHECKPOINT_REPO,
                    path_in_repo=f"{VISION_HF_PREFIX}/orpo/final_adapter",
                    repo_type="model",
                )
                print("✅ [VISION] Upload final ORPO adapter sukses!")
            except Exception as e:
                print(f"⚠️ [VISION] Gagal upload final ORPO adapter: {e}")
    except Exception as e:
        print(f"❌ [VISION] ORPO gagal: {e}")
        traceback.print_exc()
    finally:
        vision_orpo_trainer = None
        _optimizer = None
        _lr_scheduler = None
        if "vision_model" in globals() and globals()["vision_model"] is not None:
            try:
                globals()["vision_model"].zero_grad(set_to_none=True)
            except Exception:
                pass
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    return


@app.cell
def _(VISION_HF_CHECKPOINT_REPO, VISION_OUTPUT_DIR, os, vision_model, vision_processor):
    def save_vision_adapter():
        if vision_model is None:
            return
        # Menyimpan adapter vision dan mengunggah ke HF Hub
        adapter_path = os.path.join(VISION_OUTPUT_DIR, "final_adapter")
        vision_model.save_pretrained(adapter_path)
        vision_processor.save_pretrained(adapter_path)
        print(f"✅ [VISION] Adapter LoRA vision berhasil disimpan ke: {adapter_path}")

        token = os.environ.get("HF_TOKEN")
        if token:
            print(f"[VISION] Mengunggah adapter vision ke Hugging Face Hub: {VISION_HF_CHECKPOINT_REPO}...")
    save_vision_adapter()
    return


@app.cell
def _(
    ALL_SUPPRESS_IDS,
    load_dataset,
    random,
    re,
    torch,
    traceback,
    vision_model,
    vision_processor,
    vision_tokenizer,
):
    def run_vision_eval():
        if vision_model is None:
            return
        # =====================================================================
        # EVALUASI GENERASI (TEST KUALITAS GENERASI CHAT & VISION)
        # =====================================================================
        # Menguji kemampuan visual dan menjaga kemampuan dialog bahasa Indonesia
        # menggunakan dataset validasi teks asli dari training sebelumnya
        vision_model.eval()

        def _format_encoder_eval(raw_input: str) -> str:
            system = "Kamu adalah asisten AI yang helpful, santai, dan ramah. Gunakan Bahasa Indonesia sebagai bahasa utama."
            system_match = re.search(r"^system:\s*(.*?)(?=\nuser:)", raw_input, re.DOTALL)
            if system_match:
                system = system_match.group(1).strip()
                raw_input = raw_input[system_match.end() :].strip()

            parts = re.split(r"\n(user:|assistant:)\s*", "\n" + raw_input)
            formatted = ""
            is_first_user = True

            for i in range(1, len(parts), 2):
                role = parts[i].replace(":", "").strip()
                content = parts[i + 1].strip()
                if not content:
                    continue

                if role == "user":
                    formatted += "<start_of_turn>user\n"
                    if is_first_user and system:
                        formatted += system + "\n\n"
                        is_first_user = False
                    formatted += content + "<end_of_turn>\n"
                elif role == "assistant":
                    formatted += "<start_of_turn>model\n"
                    formatted += content + "<end_of_turn>\n"

            formatted += "<start_of_turn>model\n"
            return formatted

        def _process_sft_rows_eval(samples, tokenizer, is_chat=True):
            rows = []
            if is_chat:
                chat_groups = {}
                for obj in samples:
                    if not obj.get("input") or not obj.get("target"):
                        continue
                    chat_idx = obj.get("chat_idx", -1)
                    if chat_idx not in chat_groups:
                        chat_groups[chat_idx] = []
                    chat_groups[chat_idx].append(obj)

                for chat_idx, turns in chat_groups.items():
                    turns = sorted(turns, key=lambda x: x.get("turn_idx", 0))
                    for turn in turns:
                        inp_f = _format_encoder_eval(turn["input"])
                        tgt_f = turn["target"].strip() + "<end_of_turn>"
                        inp_ids = tokenizer.encode(inp_f, add_special_tokens=True)
                        # Mirror training collator: processor() hardcode BOS saat
                        # training, jadi validasi teks-only juga harus diawali BOS
                        # (add_bos_token=False membuat encode() tidak menambah BOS).
                        _bos_id = getattr(tokenizer, "bos_token_id", None)
                        if _bos_id is not None and (not inp_ids or inp_ids[0] != _bos_id):
                            inp_ids = [_bos_id] + inp_ids
                        if getattr(tokenizer, "eos_token_id", None) is not None and inp_ids[-1] != tokenizer.eos_token_id:
                            inp_ids.append(tokenizer.eos_token_id)
                        tgt_ids = tokenizer.encode(tgt_f, add_special_tokens=False)
                        if getattr(tokenizer, "eos_token_id", None) is not None and tgt_ids[-1] != tokenizer.eos_token_id:
                            tgt_ids.append(tokenizer.eos_token_id)
                        rows.append({"input_ids": inp_ids, "labels": tgt_ids})
            else:
                for obj in samples:
                    inp_f = _format_encoder_eval(obj.get("input", ""))
                    tgt_f = obj.get("target", "").strip() + "<end_of_turn>"
                    inp_ids = tokenizer.encode(inp_f, add_special_tokens=True)
                    if getattr(tokenizer, "eos_token_id", None) is not None and inp_ids[-1] != tokenizer.eos_token_id:
                        inp_ids.append(tokenizer.eos_token_id)
                    tgt_ids = tokenizer.encode(tgt_f, add_special_tokens=False)
                    if getattr(tokenizer, "eos_token_id", None) is not None and tgt_ids[-1] != tokenizer.eos_token_id:
                        tgt_ids.append(tokenizer.eos_token_id)
                    rows.append({"input_ids": inp_ids, "labels": tgt_ids})
            return rows

        print("\n" + "=" * 70)
        print("[VISION] TEST 1: Evaluasi Gambar Umum / Dokumen (Multimodal)")
        print("=" * 70)

        test_messages = [
            {"role": "system", "content": "Kamu adalah Gemma, asisten AI cerdas berbahasa Indonesia. Berikan respons yang akurat, ramah, dan terstruktur."},
            {"role": "user", "content": [
                {"type": "image"},
                {"type": "text", "text": "Halo Gemma, boleh tolong jelaskan apa menu makanan yang paling populer seharga di bawah 150 ribu berdasarkan brosur/menu ini?"}
            ]}
        ]

        from PIL import Image as PILImage
        dummy_img = PILImage.new("RGB", (224, 224), color="blue")

        try:
            prompt = vision_processor.apply_chat_template(test_messages, tokenize=False, add_generation_prompt=True)
            inputs = vision_processor(text=prompt, images=dummy_img, return_tensors="pt")

            device = next(vision_model.parameters()).device
            inputs = {k: v.to(device) for k, v in inputs.items()}

            with torch.no_grad():
                outputs = vision_model.generate(
                    **inputs,
                    max_new_tokens=256,
                    do_sample=True,
                    temperature=0.7, top_p=0.9, use_cache=True
                )
            response = vision_processor.decode(outputs[0], skip_special_tokens=True)
            print(f"User: [📷 Image] Halo Gemma, boleh tolong jelaskan apa menu makanan yang paling populer seharga di bawah 150 ribu berdasarkan brosur/menu ini?")
            print(f"Assistant:\n{response}")
        except Exception as e:
            print(f"Gagal melakukan inferensi multimodal: {e}")

        print("\n" + "=" * 70)
        print("[VISION] TEST 2: Evaluasi Pemeliharaan Chat Umum (Text-Only - LITERALLY 100 Kueri dari Validation Sebelumnya)")
        print("=" * 70)

        print("[VISION] Memuat dataset validasi percakapan teks sebelumnya...")
        try:
            val_chat_ds = load_dataset("daruokta/t5gemma2-indonesia-chat-formatted", "chat_sft", split="validation")
            val_indoqa_ds = load_dataset("daruokta/t5gemma2-indonesia-chat-formatted", "indoqa_sft", split="validation")

            val_chat_samples = [dict(row) for row in val_chat_ds]
            val_indoqa_samples = [dict(row) for row in val_indoqa_ds]

            val_rows = _process_sft_rows_eval(val_chat_samples, vision_tokenizer, is_chat=True) + _process_sft_rows_eval(val_indoqa_samples, vision_tokenizer, is_chat=False)

            # Samakan dengan seed 42 dan shuffle agar urutannya konsisten dengan baseline teks
            random.seed(42)
            random.shuffle(val_rows)

            eval_generation_samples = val_rows[:100]
            print(f"[VISION] Berhasil memuat dan memproses {len(eval_generation_samples)} sampel validasi teks.")

            device = next(vision_model.parameters()).device
            _eot_id = vision_tokenizer.convert_tokens_to_ids("<end_of_turn>")
            _eos_id = vision_tokenizer.eos_token_id or 1
            _stop_ids = list({_eot_id, _eos_id})

            # Gunakan ALL_SUPPRESS_IDS yang dilewatkan sebagai argumen
            bad_words_ids = [[id_] for id_ in ALL_SUPPRESS_IDS if id_ < vision_model.config.vocab_size]
            pad_id = vision_tokenizer.pad_token_id if vision_tokenizer.pad_token_id is not None else _eos_id

            for idx, sample in enumerate(eval_generation_samples):
                input_tensor = torch.tensor([sample["input_ids"]], dtype=torch.long).to(device)
                attention_mask = torch.ones_like(input_tensor).to(device)

                with torch.no_grad():
                    outputs_text = vision_model.generate(
                        input_ids=input_tensor,
                        attention_mask=attention_mask,
                        max_new_tokens=1024,
                        do_sample=True,
                        temperature=0.7,
                        top_p=0.9,
                        repetition_penalty=1.2,
                        eos_token_id=_stop_ids,
                        pad_token_id=pad_id,
                        bad_words_ids=bad_words_ids,
                        use_cache=True
                    )

                query = vision_tokenizer.decode(sample["input_ids"], skip_special_tokens=True).strip()
                target = vision_tokenizer.decode(sample["labels"], skip_special_tokens=True).strip()

                raw_response = vision_tokenizer.decode(outputs_text[0], skip_special_tokens=True)
                if raw_response.startswith(query):
                    raw_response = raw_response[len(query):].strip()
                response = raw_response.strip()

                words = response.split()
                is_repetitive = len(set(words)) < max(1, len(words) * 0.3) if words else True
                flag = " ⚠️ REPETITIVE" if is_repetitive else ""

                print(f"\n[Sampel {idx+1}/100]{flag}")
                print(f"  Q: {query[:250]}...")
                print(f"  Expected Target: {target[:200]}...")
                print(f"  Model Generated: {response[:350]}...")

        except Exception as e:
            print(f"Gagal melakukan inferensi teks validasi 100 sampel: {e}")
            traceback.print_exc()

        print("=" * 70)

    run_vision_eval()
    return


# =====================================================================
# VISION: MERGE & QUANTIZE
# =====================================================================
@app.cell
def _(
    VISION_HF_CHECKPOINT_REPO,
    VISION_HF_PREFIX,
    VISION_LOAD_IN_4BIT,
    VISION_MODEL_NAME,
    VISION_OUTPUT_DIR,
    VISION_SUBFOLDER,
    mo,
    os,
    vision_current_stage,
    vision_model,
    vision_processor,
    vision_tokenizer,
):
    mo.stop(
        vision_current_stage != "done" and vision_model is None,
        mo.md("⏭️ **[VISION] Phase 2 belum selesai (SFT/ORPO masih berjalan).** Merge dilewati — re-run notebook setelah ORPO selesai."),
    )

    def vision_merge_and_quantize(vision_model, vision_tokenizer, vision_processor, upload_dir: str):
        import unsloth_zoo.saving_utils
        unsloth_zoo.saving_utils.assert_same_keys = lambda *args, **kwargs: None  # type: ignore

        # --- Workaround: unsloth_zoo `_infer_prefix_and_remap` UnboundLocalError ---
        # Versi unsloth_zoo (lama) yang terinstal tidak menginisialisasi
        # `unmatched_keys = []` sebelum cek `if unmatched_keys:` pertama. Saat SEMUA
        # key LoRA sudah cocok langsung dengan key safetensor (tidak ada unmatched key
        # yang tercipta -> cabang `else` tidak pernah jalan), variabel `unmatched_keys`
        # tidak pernah di-assign sehingga `if unmatched_keys:` melempar
        # `UnboundLocalError: cannot access local variable 'unmatched_keys'`.
        # Bug ini SUDAH diperbaiki di upstream (unsloth-zoo main menambah
        # `unmatched_keys = []` sebelum loop). Di sini kita bungkus fungsi terinstal
        # lalu fallback ke reimplementation minimal yang sudah di-fix ketika error
        # spesifik itu muncul, agar save_pretrained_merged("merged_16bit") sukses.
        _sz = unsloth_zoo.saving_utils
        if not getattr(_sz, "_unmatched_keys_patch_applied", False):
            from collections import defaultdict as _ddp
            _orig_infer = getattr(_sz, "_infer_prefix_and_remap", None)

            def _infer_prefix_and_remap_fixed(lora_weights, safetensor_keys):
                # Reimplementasi minimal dari _infer_prefix_and_remap (upstream main)
                # dengan inisialisasi `unmatched_keys = []` yang menjadi root cause fix.
                if not safetensor_keys:
                    return None
                sf_key_set = set(safetensor_keys)
                remapped = _ddp(getattr(lora_weights, "default_factory", None))
                changed = False
                unmatched_keys = []  # <-- THE FIX: inisialisasi sebelum dipakai
                for k, v in lora_weights.items():
                    if not isinstance(k, str):
                        remapped[k] = v
                        continue
                    # Sudah cocok langsung dengan key safetensor (.weight / .linear.weight)
                    if (k + ".weight") in sf_key_set or (k + ".linear.weight") in sf_key_set:
                        remapped[k] = v
                        continue
                    # Cari kandidat prefix unik
                    candidates = list(dict.fromkeys(
                        sf_key[: -len(suffix)]
                        for suffix in (k + ".weight", k + ".linear.weight")
                        for sf_key in safetensor_keys
                        if sf_key.endswith(suffix) and sf_key[: -len(suffix)]
                    ))
                    if len(candidates) == 1:
                        remapped[candidates[0] + k] = v
                        changed = True
                    else:
                        unmatched_keys.append((k, v))
                # Tidak ada perubahan sama sekali -> sinyalkan "tidak perlu remap"
                if not changed and not unmatched_keys:
                    return None
                # Untuk key yang benar-benar tak ter-match, biarkan apa adanya
                # (merge akan skip target tanpa backing tensor) -> konservatif & aman.
                for k, v in unmatched_keys:
                    remapped[k] = v
                return remapped

            def _patched_infer(lora_weights, safetensor_keys):
                if _orig_infer is not None:
                    try:
                        return _orig_infer(lora_weights, safetensor_keys)
                    except UnboundLocalError as e:
                        if "unmatched_keys" in str(e):
                            print(
                                f"⚠️ [patch] _infer_prefix_and_remap UnboundLocalError "
                                f"({e}); memakai fallback reimplementation yang sudah di-fix."
                            )
                            return _infer_prefix_and_remap_fixed(lora_weights, safetensor_keys)
                        raise
                return _infer_prefix_and_remap_fixed(lora_weights, safetensor_keys)

            setattr(_sz, "_infer_prefix_and_remap", _patched_infer)
            setattr(_sz, "_unmatched_keys_patch_applied", True)
            print("✅ [patch] Workaround `_infer_prefix_and_remap` UnboundLocalError terpasang.")

        if vision_model is None:
            from unsloth import FastVisionModel

            # Load model dari adapter ORPO final
            _orpo_path = os.path.join(VISION_OUTPUT_DIR, "orpo", "final_adapter")
            if not os.path.exists(_orpo_path):
                # Fallback download dari HF
                from huggingface_hub import snapshot_download as _snap_dl
                print("📥 [VISION] Downloading final ORPO adapter dari HF untuk merging...")
                _snap_dl(
                    repo_id=VISION_HF_CHECKPOINT_REPO,
                    local_dir=_orpo_path,
                    allow_patterns=[f"{VISION_HF_PREFIX}/orpo/final_adapter/**"],
                    token=os.environ.get("HF_TOKEN"),
                )
                _sub_path = os.path.join(_orpo_path, VISION_HF_PREFIX, "orpo", "final_adapter")
                if os.path.exists(_sub_path):
                    import shutil as _shutil_merge
                    for _item in os.listdir(_sub_path):
                        _src = os.path.join(_sub_path, _item)
                        _dst = os.path.join(_orpo_path, _item)
                        if os.path.exists(_dst):
                            if os.path.isdir(_dst):
                                _shutil_merge.rmtree(_dst)
                            else:
                                os.remove(_dst)
                        _shutil_merge.move(_src, _dst)
                    _shutil_merge.rmtree(os.path.join(_orpo_path, VISION_HF_PREFIX))

            print(f"📂 [VISION] Loading model dari ORPO adapter untuk merge: {_orpo_path}")
            vision_model, vision_tokenizer = FastVisionModel.from_pretrained(
                model_name=_orpo_path,
                load_in_4bit=VISION_LOAD_IN_4BIT,
                use_gradient_checkpointing="unsloth",
                token=os.environ.get("HF_TOKEN"),
            )
            from transformers import AutoProcessor as _AutoProcMerge
            vision_processor = _AutoProcMerge.from_pretrained(
                VISION_MODEL_NAME, subfolder=VISION_SUBFOLDER,
                token=os.environ.get("HF_TOKEN"),
            )
            from unsloth.chat_templates import get_chat_template
            vision_tokenizer = get_chat_template(vision_tokenizer, chat_template="gemma-3")
            vision_processor.chat_template = vision_tokenizer.chat_template
            if hasattr(vision_processor, "tokenizer"):
                vision_processor.tokenizer.chat_template = vision_tokenizer.chat_template

        merged_bf16_path = os.path.join(upload_dir, "merged_bf16")
        quantized_4bit_path = os.path.join(upload_dir, "quantized_4bit")

        print("[VISION] Merging LoRA adapter and saving model as BF16 using Unsloth...")
        vision_model.save_pretrained_merged(merged_bf16_path, vision_tokenizer, save_method="merged_16bit")
        vision_tokenizer.save_pretrained(merged_bf16_path)
        vision_processor.save_pretrained(merged_bf16_path)
        print("✅ [VISION] Model BF16 berhasil disimpan.")

        print("\n[VISION] Merging LoRA adapter and saving model as 4-bit NF4 using Unsloth...")
        vision_model.save_pretrained_merged(quantized_4bit_path, vision_tokenizer, save_method="merged_4bit_forced")
        vision_tokenizer.save_pretrained(quantized_4bit_path)
        vision_processor.save_pretrained(quantized_4bit_path)
        print("✅ [VISION] Model 4-bit NF4 berhasil disimpan!")

        return None

    vision_upload_dir = os.path.join(VISION_OUTPUT_DIR, "hf_upload")
    vision_merge_and_quantize(vision_model, vision_tokenizer, vision_processor, vision_upload_dir)
    return (vision_upload_dir,)


@app.cell
def _(VISION_HF_CHECKPOINT_REPO, VISION_HF_PREFIX, os, vision_upload_dir):
    from huggingface_hub import HfApi as _UploadMergedApi

    print(f"[VISION] Memulai proses unggah model merged ke HF Hub: {VISION_HF_CHECKPOINT_REPO}/{VISION_HF_PREFIX}...")
    try:
        _merged_api = _UploadMergedApi(token=os.environ.get("HF_TOKEN"))

        # Ensure target model repository exists before uploading merged folder
        _merged_api.create_repo(repo_id=VISION_HF_CHECKPOINT_REPO, repo_type="model", private=False, exist_ok=True)

        _merged_api.upload_folder(
            folder_path=vision_upload_dir,
            path_in_repo=VISION_HF_PREFIX,
            repo_id=VISION_HF_CHECKPOINT_REPO,
            repo_type="model",
        )

        print("✅ [VISION] Berhasil mengunggah merged models ke Hugging Face Hub!")
    except Exception as e:
        print(f"❌ [VISION] Terjadi kesalahan saat mengunggah: {e}")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ---
    ### 💻 [VISION] Local Deployment & Inference (Unified Repo, Subfolder `vision/`)
    Setelah model VISION diunggah ke **unified repo**, artifacts berada di bawah prefix `vision/`:
    - `vision/sft/` — Checkpoint dan artifacts SFT vision
    - `vision/orpo/` — Checkpoint dan artifacts ORPO vision
    - `vision/merged_bf16/` — **HASIL AKHIR** multimodal utuh (bfloat16, ~15 GB)
    - `vision/quantized_4bit/` — **HASIL AKHIR** terkuantisasi (NF4, ~5 GB)

    #### Load Model Quantized 4-bit:
    ```python
    from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, AutoProcessor

    model_id = "daruokta/t5gemma-2-4b-4b-instruct-chat-indo-v6-combined-unsloth"

    tokenizer = AutoTokenizer.from_pretrained(model_id, subfolder="vision/quantized_4bit")
    processor = AutoProcessor.from_pretrained(model_id, subfolder="vision/quantized_4bit")
    model = AutoModelForSeq2SeqLM.from_pretrained(
        model_id, subfolder="vision/quantized_4bit", device_map="auto"
    )
    ```

    #### Load Model Full Precision (BF16):
    ```python
    import torch
    from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, AutoProcessor

    model_id = "daruokta/t5gemma-2-4b-4b-instruct-chat-indo-v6-combined-unsloth"

    tokenizer = AutoTokenizer.from_pretrained(model_id, subfolder="vision/merged_bf16")
    processor = AutoProcessor.from_pretrained(model_id, subfolder="vision/merged_bf16")
    model = AutoModelForSeq2SeqLM.from_pretrained(
        model_id, subfolder="vision/merged_bf16",
        torch_dtype=torch.bfloat16, device_map="auto"
    )
    ```

    #### Load Model Cangkok (base vision, sebelum adapter):
    ```python
    model_id = "daruokta/t5gemma-2-4b-4b-instruct-chat-indo-v6-combined-unsloth"
    tokenizer = AutoTokenizer.from_pretrained(model_id, subfolder="cangkok")
    processor = AutoProcessor.from_pretrained(model_id, subfolder="cangkok")
    model = AutoModelForSeq2SeqLM.from_pretrained(
        model_id, subfolder="cangkok", torch_dtype=torch.bfloat16, device_map="auto"
    )
    ```
    """)
    return


if __name__ == "__main__":
    app.run()
