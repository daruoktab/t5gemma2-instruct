# /// script
# requires-python = ">=3.10"
# dependencies = [
#     "accelerate==1.14.0",
#     "absl-py==2.4.0",
#     "bitsandbytes==0.50.0",
#     "datasets==5.0.0",
#     "evaluate",
#     "rouge-score",
#     "sacrebleu",
#     "bert_score",
#     "nltk",
#     "hf-transfer",
#     "huggingface-hub==1.25.1",
#     "marimo==0.23.15",
#     "numpy==2.5.1",
#     "peft==0.19.1",
#     "pillow==12.3.0",
#     "pymupdf==1.28.0",
#     "pytorch-optimizer",
#     "torch==2.12.1",
#     "torchvision==0.27.1",
#     "trl==1.9.2",
#     "transformers==5.14.1",
#     "unsloth_zoo @ git+https://github.com/daruoktab/unsloth-zoo.git",
#     "unsloth @ git+https://github.com/daruoktab/unsloth.git",
# ]
# ///
#
# =====================================================================
# T5Gemma-2 JOINT MULTIMODAL PIPELINE (v7 — 1-Stage Joint Co-Training)
# =====================================================================
#   Phase 0.5 : 3-Way Task Vector Steering (decoder T5Gemma <- Δ(Gemma3-IT - Gemma3-Base))
#   Phase 1.5 : Vision Grafting (SigLIP + multi_modal_projector <- Gemma 3 4B IT)
#   Phase 1   : JOINT SFT  (teks chat/indoqa + vision dicampur dalam 1 loop)
#   Phase 2   : JOINT ORPO (teks orpo + vision orpo dicampur dalam 1 loop, ε=0)
#   Final     : 1x Merge (BF16 + 4bit) -> unified repo subfolder final/
#
# Semua artifacts dalam 1 repo HF PUBLIK (nama repo di CONTROL CENTER, cell ke-2):
#   steered/  -> checkpoint hasil Phase 0.5
#   cangkok/  -> checkpoint hasil Phase 1.5 (base model untuk training)
#   joint/    -> sft/, orpo/ (checkpoints + final_adapter + logs)
#   final/    -> merged_bf16/, quantized_4bit/  (HASIL AKHIR)
# =====================================================================

import marimo

__generated_with = "0.23.9"
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
    # =====================================================================
    # 1A. REPO & MODEL SOURCES
    # =====================================================================
    UNIFIED_HF_REPO = "daruokta/t5gemma-2-4b-4b-instruct-chat-indo-v7-joint-unsloth"
    DATASET_TEXT_REPO = "daruokta/t5gemma2-indonesia-chat-formatted"
    DATASET_VISION_REPO = "daruokta/t5gemma2-indonesia-vision-formatted"

    BASE_T5_MODEL = "google/t5gemma-2-4b-4b"
    GEMMA_BASE_MODEL = "google/gemma-3-4b-pt"
    GEMMA_IT_MODEL = "google/gemma-3-4b-it"

    # Subfolder layout di dalam UNIFIED_HF_REPO
    STEERED_SUBFOLDER = "steered"
    CANGKOK_SUBFOLDER = "cangkok"
    JOINT_PREFIX = "joint"                     # joint/sft, joint/orpo
    FINAL_PREFIX = "final"                     # final/merged_bf16, final/quantized_4bit

    OUTPUT_DIR = "results/t5gemma2_joint"      # working dir lokal (checkpoints, logs, merge)

    # =====================================================================
    # 1B. PHASE FLAGS & GATES
    # =====================================================================
    ENABLE_STEERING = True        # Phase 0.5 — task vector steering decoder
    STEERING_FORCE = False        # True = steer ulang walau steered/ sudah ada di repo
    CANGKOK_FORCE = False         # True = graft ulang walau cangkok/ sudah ada di repo
    RUN_SFT = True                # False = skip Phase 2.1 (langsung cek ORPO/merge)
    RUN_ORPO = True               # False = stop setelah SFT

    # α per kelompok modul — Layer-Wise Ramp-Up (SOTA optimal):
    STEERING_ALPHA_FFN_EARLY = 0.05     # Layer awal (< 25% depth) — subtle
    STEERING_ALPHA_FFN_MID   = 0.25     # Layer tengah (25%-80% depth) — peak IT reasoning & knowledge
    STEERING_ALPHA_FFN_LATE  = 0.08     # Layer akhir (> 80% depth) — menjaga kalibrasi output
    STEERING_ALPHA_NORM_EARLY = 0.02
    STEERING_ALPHA_NORM_MID   = 0.08
    STEERING_ALPHA_NORM_LATE  = 0.03

    STEERING_ALPHA_QO = 0.0       # q_proj & o_proj — WAJIB 0.0 untuk keamanan Merged Attention [X;H]
    STEERING_ALPHA_KV = 0.0       # k_proj & v_proj — WAJIB 0.0 (joint projection [X;H])
    STEERING_ALPHA_QKNORM = 0.0   # q_norm & k_norm — terikat kalibrasi joint softmax
    STEERING_SMOKE_TEST = True    # generate 3 prompt singkat untuk sanity check hasil steering

    # =====================================================================
    # 1D. DATA & MIXING (Joint Co-Training)
    # =====================================================================
    TEXT_CHAT_CONFIG = "chat_sft"
    TEXT_INDOQA_CONFIG = "indoqa_sft"
    TEXT_ORPO_CONFIG = "chat_orpo"
    VISION_SFT_CONFIG = "vision_sft"
    VISION_ORPO_CONFIG = "vision_orpo"

    SAMPLE_TRAIN_CHAT = 0         # 0 = ambil seluruh data
    SAMPLE_TRAIN_INDOQA = 0
    SAMPLE_TRAIN_TEXT_ORPO = 0
    SAMPLE_TRAIN_VISION_SFT = 0
    SAMPLE_TRAIN_VISION_ORPO = 0

    VISION_TEST_SIZE = 0.05       # hold-out PERCAKAPAN vision utuh untuk eval-mm (95/5 di level conv — train/eval tidak kepotong)
    MAX_EVAL_TEXT_SAMPLES = 200   # cap eval teks per-step (deterministik, group-aware per chat_idx)
    MAX_EVAL_GEN_SAMPLES = 20     # cap sample kualitatif per eval-kind

    # =====================================================================
    # 1E. SFT HYPERPARAMS (Phase 1 - Joint)
    # =====================================================================
    LOAD_IN_4BIT = True
    MAX_SOURCE_LENGTH = 16384
    MAX_TARGET_LENGTH = 2048

    LORA_RANK = 256
    LORA_ALPHA = 512
    LORA_DROPOUT = 0.2
    LORA_USE_RSLORA = True

    SFT_LEARNING_RATE = 5e-6
    SFT_NUM_EPOCHS = 2
    SFT_PER_DEVICE_TRAIN_BATCH_SIZE = 4
    SFT_PER_DEVICE_EVAL_BATCH_SIZE = 16   # eval = no_grad/inference-only → aman dinaikkan (VRAM eval jauh di bawah train)
    SFT_GRADIENT_ACCUMULATION_STEPS = 16
    SFT_WARMUP_STEPS = 100
    SFT_WEIGHT_DECAY = 0.1
    SFT_LR_SCHEDULER_TYPE = "cosine"
    SFT_LOGGING_STEPS = 10
    SFT_SAVE_EVAL_STEPS = 100       # ±13 titik eval untuk ~1.3k steps (setiap 100 cukup granular)
    SFT_SAVE_TOTAL_LIMIT = 2
    SFT_LABEL_SMOOTHING_FACTOR = 0.1
    SFT_NEFTUNE_NOISE_ALPHA = 5.0
    SFT_MAX_GRAD_NORM = 5.0
    SFT_PREDICT_WITH_GENERATE = True

    # Split-LR multiplier per param group (relatif terhadap SFT_LEARNING_RATE)
    SFT_LR_MULT_ENCODER = 0.2
    SFT_LR_MULT_DECODER = 0.2
    SFT_LR_MULT_PROJECTOR = 0.05
    SFT_LR_MULT_VISION_TOWER = 0.0   # vision tower frozen (finetune_vision_layers=False)

    # =====================================================================
    # 1F. ORPO HYPERPARAMS (Phase 2 - Joint)
    # =====================================================================
    ORPO_BETA = 0.1
    ORPO_LEARNING_RATE = 5e-6
    ORPO_NUM_EPOCHS = 1
    ORPO_PER_DEVICE_TRAIN_BATCH_SIZE = 4
    ORPO_PER_DEVICE_EVAL_BATCH_SIZE = 8
    ORPO_GRADIENT_ACCUMULATION_STEPS = 16
    ORPO_WARMUP_STEPS = 2           # total step ORPO hanya ~18 — warmup 100 = LR tidak pernah peak
    ORPO_WEIGHT_DECAY = 0.1
    ORPO_LR_SCHEDULER_TYPE = "cosine"
    ORPO_LOGGING_STEPS = 10
    ORPO_SAVE_EVAL_STEPS = 6        # ~3 titik eval dalam ~18 steps total
    ORPO_SAVE_TOTAL_LIMIT = 2
    ORPO_LABEL_SMOOTHING_FACTOR = 0.0   # WAJIB 0.0 — smoothing merusak odds-ratio ORPO
    ORPO_MAX_GRAD_NORM = 5.0
    ORPO_PREDICT_WITH_GENERATE = True

    ORPO_LR_MULT_ENCODER = 0.5
    ORPO_LR_MULT_DECODER = 1.0
    ORPO_LR_MULT_PROJECTOR = 1.0
    ORPO_LR_MULT_VISION_TOWER = 0.5

    # =====================================================================
    # 1G. OPTIMIZER
    # =====================================================================
    # "grokmuonadema" = GrokFast filter + Muon (param 2D, Newton-Schulz) + AdEMAMix (param 1D)
    # "grokademamix"  = GrokFast + AdEMAMix murni (optimizer v6 yang sudah terbukti)
    # "paged_adamw_8bit" = bawaan HF/Unsloth (fallback paling hemat VRAM)
    OPTIMIZER_TYPE = "grokmuonadema"
    # GrokFast
    GROK_ALPHA = 2.0
    GROK_LAMB = 0.98
    # AdEMAMix
    ADEMA_BETA1 = 0.9
    ADEMA_BETA2 = 0.999
    ADEMA_BETA3 = 0.9999
    # Muon
    MUON_MOMENTUM = 0.95
    MUON_NS_STEPS = 5
    MUON_NESTEROV = True
    MUON_MAX_GRAD_NORM = 1.0          # MuonClip threshold
    # Update Muon di-ortonormalisasi (magnitudo ≈ lr × ~1, tidak diskalakan
    # statistik gradien seperti Adam) -> butuh LR lebih besar dari Adam-family.
    # Skala ini mengalikan LR khusus param cabang Muon (LoRA A/B 2D dll).
    # ORPO pakai skala lebih kecil — preference update harus lembut & stabil.
    SFT_MUON_LR_SCALE = 20.0          # mis. decoder SFT: 5e-6 × 0.2 × 20 ≈ 2e-5
    ORPO_MUON_LR_SCALE = 5.0          # mis. decoder ORPO: 5e-6 × 1.0 × 5 ≈ 2.5e-5
    # Routing komponen multi_modal_projector (2D bobotnya):
    #   "muon"  = ikut aturan 2D -> cabang Muon
    #   "adema" = paksa ke cabang AdEMAMix (konservatif untuk bobot pretrained graft)
    PROJECTOR_BRANCH = "muon"

    # =====================================================================
    # 1H. GENERATION EVAL & MISC
    # =====================================================================
    GEN_TEMPERATURE = 0.7
    GEN_TOP_P = 0.9
    GEN_REPETITION_PENALTY = 1.05   # 1.2 merusak tabel/list (token repetitif dihukum); logit mask sudah menahan garbage
    SEED = 3407
    return (
        ADEMA_BETA1,
        ADEMA_BETA2,
        ADEMA_BETA3,
        BASE_T5_MODEL,
        CANGKOK_FORCE,
        CANGKOK_SUBFOLDER,
        DATASET_TEXT_REPO,
        DATASET_VISION_REPO,
        ENABLE_STEERING,
        FINAL_PREFIX,
        GEMMA_BASE_MODEL,
        GEMMA_IT_MODEL,
        GEN_REPETITION_PENALTY,
        GEN_TEMPERATURE,
        GEN_TOP_P,
        GROK_ALPHA,
        GROK_LAMB,
        JOINT_PREFIX,
        LOAD_IN_4BIT,
        LORA_ALPHA,
        LORA_DROPOUT,
        LORA_RANK,
        LORA_USE_RSLORA,
        MAX_EVAL_GEN_SAMPLES,
        MAX_EVAL_TEXT_SAMPLES,
        MAX_SOURCE_LENGTH,
        MAX_TARGET_LENGTH,
        MUON_MAX_GRAD_NORM,
        MUON_MOMENTUM,
        MUON_NESTEROV,
        MUON_NS_STEPS,
        OPTIMIZER_TYPE,
        ORPO_BETA,
        ORPO_GRADIENT_ACCUMULATION_STEPS,
        ORPO_LABEL_SMOOTHING_FACTOR,
        ORPO_LEARNING_RATE,
        ORPO_LOGGING_STEPS,
        ORPO_LR_MULT_DECODER,
        ORPO_LR_MULT_ENCODER,
        ORPO_LR_MULT_PROJECTOR,
        ORPO_LR_MULT_VISION_TOWER,
        ORPO_LR_SCHEDULER_TYPE,
        ORPO_MAX_GRAD_NORM,
        ORPO_MUON_LR_SCALE,
        ORPO_NUM_EPOCHS,
        ORPO_PER_DEVICE_EVAL_BATCH_SIZE,
        ORPO_PER_DEVICE_TRAIN_BATCH_SIZE,
        ORPO_PREDICT_WITH_GENERATE,
        ORPO_SAVE_EVAL_STEPS,
        ORPO_SAVE_TOTAL_LIMIT,
        ORPO_WARMUP_STEPS,
        ORPO_WEIGHT_DECAY,
        OUTPUT_DIR,
        PROJECTOR_BRANCH,
        RUN_ORPO,
        RUN_SFT,
        SAMPLE_TRAIN_CHAT,
        SAMPLE_TRAIN_INDOQA,
        SAMPLE_TRAIN_TEXT_ORPO,
        SAMPLE_TRAIN_VISION_ORPO,
        SAMPLE_TRAIN_VISION_SFT,
        SEED,
        SFT_GRADIENT_ACCUMULATION_STEPS,
        SFT_LABEL_SMOOTHING_FACTOR,
        SFT_LEARNING_RATE,
        SFT_LOGGING_STEPS,
        SFT_LR_MULT_DECODER,
        SFT_LR_MULT_ENCODER,
        SFT_LR_MULT_PROJECTOR,
        SFT_LR_MULT_VISION_TOWER,
        SFT_LR_SCHEDULER_TYPE,
        SFT_MAX_GRAD_NORM,
        SFT_MUON_LR_SCALE,
        SFT_NEFTUNE_NOISE_ALPHA,
        SFT_NUM_EPOCHS,
        SFT_PER_DEVICE_EVAL_BATCH_SIZE,
        SFT_PER_DEVICE_TRAIN_BATCH_SIZE,
        SFT_PREDICT_WITH_GENERATE,
        SFT_SAVE_EVAL_STEPS,
        SFT_SAVE_TOTAL_LIMIT,
        SFT_WARMUP_STEPS,
        SFT_WEIGHT_DECAY,
        STEERED_SUBFOLDER,
        STEERING_ALPHA_FFN_EARLY,
        STEERING_ALPHA_FFN_LATE,
        STEERING_ALPHA_FFN_MID,
        STEERING_ALPHA_KV,
        STEERING_ALPHA_NORM_EARLY,
        STEERING_ALPHA_NORM_LATE,
        STEERING_ALPHA_NORM_MID,
        STEERING_ALPHA_QKNORM,
        STEERING_ALPHA_QO,
        STEERING_FORCE,
        STEERING_SMOKE_TEST,
        TEXT_CHAT_CONFIG,
        TEXT_INDOQA_CONFIG,
        TEXT_ORPO_CONFIG,
        UNIFIED_HF_REPO,
        VISION_ORPO_CONFIG,
        VISION_SFT_CONFIG,
        VISION_TEST_SIZE,
    )


@app.cell
def _(mo):
    hf_token_widget = mo.ui.text(
        label="🔑 Hugging Face Token (HF_TOKEN):",
        placeholder="hf_...",
        kind="password",
        full_width=True,
    )
    hf_token_widget
    return (hf_token_widget,)


@app.cell
def _(hf_token_widget, mo, os):
    from huggingface_hub import login

    _val = hf_token_widget.value.strip() if hf_token_widget.value else ""
    if not _val:
        _val = os.environ.get("HF_TOKEN", "").strip()

    # FAIL-HARD: token invalid/kosong harus ketahuan DI SINI (bukan saat upload checkpoint).
    if not _val:
        raise RuntimeError(
            "HF_TOKEN kosong — pipeline memerlukan token (WRITE) untuk upload checkpoint ke HF "
            "dan mengunduh gated models. Isi widget di atas lalu jalankan ulang."
        )

    os.environ["HF_TOKEN"] = _val
    try:
        login(token=_val)
    except Exception as _e_login:
        raise RuntimeError(f"HF_TOKEN tidak valid (login gagal): {_e_login}") from _e_login

    # Verifikasi nyata ke server: whoami() melempar error kalau token invalid/expired.
    from huggingface_hub import HfApi as _WhoAmIApi
    try:
        _who = _WhoAmIApi(token=_val).whoami()
    except Exception as _e_whoami:
        raise RuntimeError(f"HF_TOKEN tidak bisa diautentikasi server (whoami gagal): {_e_whoami}") from _e_whoami

    _role = "?"
    if isinstance(_who, dict):
        _role = (_who.get("auth", {}).get("accessToken", {}) or {}).get("role") or _who.get("type") or "?"
    _name = _who.get("name", "?") if isinstance(_who, dict) else str(_who)
    if _role == "read":
        raise RuntimeError(
            f"HF_TOKEN valid tapi READ-ONLY (akun: {_name}) — upload checkpoint butuh token WRITE. "
            "Buat token bertipe WRITE di https://huggingface.co/settings/tokens."
        )

    print(f"✅ HF auth OK: {_name} (role={_role})")
    auth_status = mo.md(
        f"✅ **HF Token valid** (`{_name}`, role=`{_role}`). "
        "Siap akses gated models (`google/gemma-3-4b-pt`, `google/gemma-3-4b-it`, `google/t5gemma-2-4b-4b`) + upload checkpoints."
    )

    auth_status
    return (auth_status,)


@app.cell
def _():
    import subprocess
    import sys

    # Auto-install dependencies utama jika belum ada di env Molab
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
                "bitsandbytes==0.50.0",
                "datasets==5.0.0",
                "evaluate",
                "rouge-score",
                "sacrebleu",
                "bert_score",
                "nltk",
                "hf-transfer",
                "huggingface-hub==1.25.1",
                "marimo==0.23.15",
                "numpy==2.5.1",
                "peft==0.19.1",
                "pillow==12.3.0",
                "pymupdf==1.28.0",
                "pytorch-optimizer",
                "trl==1.9.2",
                "transformers==5.14.1",
                "unsloth_zoo @ git+https://github.com/daruoktab/unsloth-zoo.git",
                "unsloth @ git+https://github.com/daruoktab/unsloth.git",
            ],
            check=True
        )

    # Force update PyTorch, torchvision ke CUDA 13.2 (cu132)
    print("📦 Force update PyTorch, torchvision (cu132)...")
    subprocess.run(
        [
            "uv", "pip", "install",
            "torch", "torchvision",
            "--index-url", "https://download.pytorch.org/whl/cu132",
            "-U", "--force-reinstall",
        ],
        check=True
    )

    # Force install/update flash_attn prebuild wheel (cu132 torch2.13 cp313)
    print("📦 Meng-install/update flash_attn prebuild wheel (v0.9.47 cu132 torch2.13)...")
    subprocess.run(
        [
            "uv", "pip", "install", "-U",
            "flash_attn @ https://github.com/mjun0812/flash-attention-prebuild-wheels/releases/download/v0.9.47/flash_attn-2.8.3+cu132torch2.13-cp313-cp313-linux_x86_64.whl",
        ],
        check=True,
    )

    # Print verifikasi versi Environment & Library
    import torch as _torch
    import flash_attn as _flash_attn

    print("=" * 60)
    print("📌 VERIFIKASI ENVIRONMENT & LIBRARY VERSIONS:")
    print(f"   • Python version    : {sys.version.split()[0]}")
    print(f"   • PyTorch version   : {_torch.__version__} (CUDA build: {_torch.version.cuda})")
    print(f"   • CUDA Available    : {_torch.cuda.is_available()}")
    print(f"   • flash_attn version: {getattr(_flash_attn, '__version__', 'Installed')}")
    print("=" * 60)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ✅ **STATUS: JOINT PIPELINE READY.**

    # 🔗 T5Gemma-2 v7 — 1-Stage Joint Multimodal Co-Training

    ---

    ### 🗺️ Alur Pipeline Multimodal (Sequential DAG)

    | Tahap | Nama Phase | Deskripsi & Komponen | Output Checkpoint |
    |---|---|---|---|
    | **0.5** | **Task Vector Steering** | Menyuntikkan `Δ = Gemma3-IT − Gemma3-Base` ke Decoder T5Gemma (Layer-Wise Ramp-Up) | `steered/` |
    | **1.5** | **Vision Grafting** | Mencangkokkan `SigLIP 400M` + `multi_modal_projector` dari Gemma 3 4B IT | `cangkok/` (Base Model) |
    | **1.0** | **JOINT SFT** | Joint co-training: Teks (`chat` & `indoqa`) + Vision dalam 1 loop | `joint/sft/` |
    | **2.0** | **JOINT ORPO** | Joint preference tuning: Teks ORPO + Vision ORPO dalam 1 loop | `joint/orpo/` |
    | **Final** | **Unified Merge** | 1x Merge BF16 + 4-bit Quantization ke subfolder final | `final/merged_bf16`, `final/quantized_4bit` |

    > 💡 **Mengapa 1-Stage Joint (vs 2-Stage v6)?**
    > Mencegah *catastrophic text forgetting* (kemampuan Bahasa Indonesia tidak hancur saat vision training), kalibrasi *Merged Attention* teks ↔ gambar dipelajari bersamaan, dan total compute 2x lebih hemat (1x SFT + 1x ORPO saja).

    > ⚙️ **Semua konfigurasi tweakable ada di cell CONTROL CENTER di bawah.**
    """)
    return


@app.cell
def _():
    import os
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
    # Matikan auto torch.compile bawaan Unsloth (unsloth_zoo membungkus forward
    # T5Gemma2 dengan @torch.compile(fullgraph=True, dynamic=True, ...)). Dengan
    # fullgraph=True, begitu recompile limit kena, itu SELALU hard-crash tanpa
    # ada config yang bisa menyelamatkan. OOM sudah ditangani oleh
    # expandable_segments + gradient checkpointing "unsloth".
    os.environ["TORCH_COMPILE_DISABLE"] = "1"
    import re, json, torch, random, datetime, gc, traceback
    import warnings
    warnings.filterwarnings("ignore")

    # Belt-and-suspenders di atas TORCH_COMPILE_DISABLE: monkeypatch torch.compile
    # jadi no-op SEBELUM unsloth di-import dan SEBELUM FastVisionModel.from_pretrained()
    # memicu Unsloth membungkus forward T5Gemma2 dengan @torch.compile(fullgraph=True, ...).
    def _torch_compile_noop(model=None, *args, **kwargs):
        if model is not None:
            return model
        return lambda fn: fn
    setattr(torch, "compile", _torch_compile_noop)
    import torch.nn.functional as F
    setattr(torch._dynamo.config, "recompile_limit", 1024)
    setattr(torch._dynamo.config, "cache_size_limit", 1024)
    from PIL import Image
    from unsloth import FastVisionModel
    from datasets import Dataset, load_dataset
    from transformers import (
        AutoProcessor, AutoTokenizer,
        Seq2SeqTrainer, Seq2SeqTrainingArguments,
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

    # Probe satu kali backbone BERTScore (google/embeddinggemma-300m = GATED repo):
    # tanpa akses/token valid → BERTScore dinonaktifkan dengan 1 pesan jelas,
    # alih-alih melempar stack-trace 401 di SETIAP titik eval.
    if bertscore_metric is not None:
        try:
            cast(Any, bertscore_metric).compute(
                predictions=["tes"], references=["tes"],
                model_type="google/embeddinggemma-300m", num_layers=12, lang="id",
            )
        except Exception as _e_bs:
            print(f"ℹ️ BERTScore dinonaktifkan (probe gagal): {_e_bs}")
            bertscore_metric = None

    # ---- LOGIT MASKING (decoder lm_head) ----
    def apply_logit_mask(model, suppress_ids):
        _cfg = getattr(model, "config", model)
        vs = getattr(_cfg, "vocab_size", getattr(getattr(_cfg, "text_config", None), "vocab_size", getattr(getattr(_cfg, "decoder", None), "vocab_size", 262144)))
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
            if getattr(t, "_logit_mask_hook_registered", False):
                print("  ℹ️ Logit mask sudah terpasang — skip (hindari double hook).")
                return
            t.register_forward_hook(hook)
            t._logit_mask_hook_registered = True
            print(f"  ✅ Logit mask (lm_head) untuk {len(sl)} tokens.")
        else:
            if getattr(model, "_logit_mask_hook_registered", False):
                print("  ℹ️ Logit mask (fallback) sudah terpasang — skip.")
                return
            model.register_forward_hook(hook)
            model._logit_mask_hook_registered = True
            print(f"  ✅ Logit mask (fallback) untuk {len(sl)} tokens.")

    return (
        Any,
        AutoProcessor,
        Dataset,
        F,
        FastVisionModel,
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
        random,
        re,
        rouge_metric,
        torch,
        traceback,
    )


@app.cell
def _(torch):
    # Token IDs yang harus di-suppress (unused + vision)
    # Pengecualian: <unused1> sampai <unused6> (ID 7 hingga 12) digunakan untuk Task Prefix
    SUPPRESS_BLOCK1 = [6] + list(range(13, 105))
    SUPPRESS_BLOCK2 = list(range(256002, 262144))
    SUPPRESS_VISION = [255999, 256000, 256001]   # boi, eoi, image_soft_token
    ALL_SUPPRESS_IDS = set(SUPPRESS_BLOCK1 + SUPPRESS_BLOCK2 + SUPPRESS_VISION)

    # SYSTEM PROMPT FALLBACK (identik dengan pipeline v6)
    SYSTEM_PROMPT = (
        "Kamu adalah asisten AI yang helpful, santai, dan ramah. "
        "Gunakan Bahasa Indonesia sebagai bahasa utama."
    )

    BF16 = torch.cuda.is_available()
    return ALL_SUPPRESS_IDS, BF16, SYSTEM_PROMPT


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
def _(torch):
    import math as _math

    # ---------- GrokAdEMAMix: optimizer v6 (GrokFast + AdEMAMix semua parameter) ----------
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

    # ---------- Muon primitive: 5-step Quintic Newton-Schulz ----------
    def zeropower_via_newtonschulz5(G: "torch.Tensor", steps: int = 5, eps: float = 1e-7) -> "torch.Tensor":
        """
        Ortogonalisasi matriks momentum G (2D) memakai quintic Newton-Schulz
        iteration (Keller Jordan / Muon).
        """
        assert G.ndim == 2, f"Muon zeropower memerlukan tensor 2D, dapat {G.ndim}D"

        a = 3.4445
        b = -4.7750
        c = 2.0315

        X = G.to(torch.float32)
        norm = X.norm() + eps
        X = X / norm

        if X.size(0) < X.size(1):
            X = X.T

        for _ in range(steps):
            A = X @ X.T
            B = b * A + c * (A @ A)
            X = a * X + B @ X

        if G.size(0) < G.size(1):
            X = X.T

        scale = max(1.0, (G.size(0) / G.size(1)) ** 0.5)
        return (X * scale).to(G.dtype)

    # ---------- GrokMuonAdEMA: GrokFast filter -> Muon (2D) / AdEMAMix (1D) ----------
    class GrokMuonAdEMA(torch.optim.Optimizer):
        """
        - GrokFast: menyaring gradien (slow EMA amplified) sebelum optimizer step.
        - Cabang 2D (linear/LoRA A/B dst): Muon update (momentum + Newton-Schulz).
        - Cabang 1D (RMSNorm/bias/embed): AdEMAMix dual-EMA.
        - MuonClip: clip norm gradien hasil filter di atas threshold.
        """
        def __init__(
            self,
            params,
            lr=2e-4,
            betas=(0.9, 0.999),
            beta3=0.9999,
            weight_decay=0.01,
            grok_alpha=2.0,
            grok_lamb=0.98,
            momentum=0.95,
            nesterov=True,
            ns_steps=5,
            max_grad_norm=1.0,
        ):
            defaults = dict(
                lr=lr,
                betas=betas,
                beta3=beta3,
                weight_decay=weight_decay,
                grok_alpha=grok_alpha,
                grok_lamb=grok_lamb,
                momentum=momentum,
                nesterov=nesterov,
                ns_steps=ns_steps,
                max_grad_norm=max_grad_norm,
            )
            super().__init__(params, defaults)

        @torch.no_grad()
        def step(self, closure=None):
            loss = None
            if closure is not None:
                with torch.enable_grad():
                    loss = closure()

            for group in self.param_groups:
                lr = group["lr"]
                beta1, beta2 = group["betas"]
                beta3 = group["beta3"]
                weight_decay = group["weight_decay"]
                grok_alpha = group["grok_alpha"]
                grok_lamb = group["grok_lamb"]
                momentum = group["momentum"]
                nesterov = group["nesterov"]
                ns_steps = group["ns_steps"]
                max_grad_norm = group["max_grad_norm"]

                for p in group["params"]:
                    if p.grad is None:
                        continue

                    grad = p.grad
                    state = self.state[p]

                    # Routing cabang: None = auto by ndim; "muon"/"adema" = paksa
                    _force_branch = group.get("force_branch", None)
                    _use_muon = (p.ndim == 2) if _force_branch is None else (_force_branch == "muon")

                    if len(state) == 0:
                        state["step"] = 0
                        state["grok_slow_grad"] = torch.zeros_like(grad)
                        state["m"] = torch.zeros_like(grad)
                        state["v"] = torch.zeros_like(grad)
                        state["n"] = torch.zeros_like(grad)
                        state["muon_buf"] = torch.zeros_like(grad) if _use_muon else None

                    state["step"] += 1
                    step = state["step"]

                    # 1) GROKFAST FILTERING
                    state["grok_slow_grad"].mul_(grok_lamb).add_(
                        grad, alpha=1.0 - grok_lamb
                    )
                    filtered_grad = grad.clone()
                    filtered_grad.add_(state["grok_slow_grad"], alpha=grok_alpha)

                    # MuonClip pada gradien hasil filter
                    if max_grad_norm > 0:
                        f_norm = filtered_grad.norm()
                        if f_norm > max_grad_norm:
                            filtered_grad.mul_(max_grad_norm / (f_norm + 1e-6))

                    # Weight decay (decoupled, seperti AdamW)
                    if weight_decay != 0:
                        p.data.mul_(1.0 - lr * weight_decay)

                    # 2) CABANG 2D: MUON UPDATE (atau paksa via force_branch)
                    if _use_muon:
                        buf = state["muon_buf"]
                        buf.mul_(momentum).add_(filtered_grad)
                        g_update = (
                            filtered_grad.add(buf, alpha=momentum) if nesterov else buf
                        )
                        g_ortho = zeropower_via_newtonschulz5(g_update, steps=ns_steps)
                        p.data.add_(g_ortho.to(p.dtype), alpha=-lr)

                    # 3) CABANG 1D: ADEMAMIX UPDATE
                    else:
                        m, v, n = state["m"], state["v"], state["n"]
                        m.mul_(beta1).add_(filtered_grad, alpha=1.0 - beta1)
                        v.mul_(beta2).addcmul_(
                            filtered_grad, filtered_grad, value=1.0 - beta2
                        )
                        n.mul_(beta3).add_(filtered_grad, alpha=1.0 - beta3)

                        bc1 = 1.0 - beta1**step
                        bc2 = 1.0 - beta2**step
                        bc3 = 1.0 - beta3**step

                        denom = (v.sqrt() / (bc2**0.5)).add_(1e-8).to(p.dtype)
                        step_update = ((m / bc1 + 0.1 * n / bc3) / denom).to(p.dtype)
                        p.data.add_(step_update, alpha=-lr)

            return loss

    return GrokAdEMAMix, GrokMuonAdEMA


@app.cell
def _(GrokAdEMAMix, GrokMuonAdEMA):
    def create_optimizer(
        model,
        base_lr: float,
        weight_decay: float,
        lr_mults: dict,
        opt_type: str,
        grok_alpha: float,
        grok_lamb: float,
        adema_betas: tuple,
        adema_beta3: float,
        muon_momentum: float,
        muon_ns_steps: int,
        muon_nesterov: bool,
        muon_max_grad_norm: float,
        muon_lr_scale: float = 1.0,
        projector_branch: str = "muon",
    ):
        """
        Return optimizer custom, atau None (kalau opt_type="paged_adamw_8bit"
        → biarkan HF Trainer yang bangun optimizer bawaan dari args.optim).

        Partisi per komponen × ndim:
          - param 2D (LoRA A/B, linear) → LR × muon_lr_scale, cabang Muon
            (KECUALI projector saat projector_branch="adema" → LR normal)
          - param 1D (norms/bias)       → LR normal, cabang AdEMAMix
        """
        if opt_type == "paged_adamw_8bit":
            return None

        # Skala Muon HANYA masuk akal untuk update ter-ortonormalisasi (GrokMuonAdEMA);
        # untuk GrokAdEMAMix semua update gaya Adam -> tanpa skala.
        _scale = muon_lr_scale if opt_type == "grokmuonadema" else 1.0

        comp_params = {"encoder": [], "decoder": [], "projector": [], "vision_tower": []}
        for _name, _param in model.named_parameters():
            if not _param.requires_grad:
                continue
            if "multi_modal_projector" in _name:
                comp_params["projector"].append(_param)
            elif "vision_tower" in _name:
                comp_params["vision_tower"].append(_param)
            elif "encoder" in _name:
                comp_params["encoder"].append(_param)
            else:
                comp_params["decoder"].append(_param)

        param_groups = []
        for _comp in ["encoder", "decoder", "projector", "vision_tower"]:
            _plist = comp_params[_comp]
            if not _plist:
                continue
            _base = base_lr * lr_mults[_comp]
            _p2d = [p for p in _plist if p.ndim == 2]
            _p1d = [p for p in _plist if p.ndim != 2]

            if _comp == "projector" and projector_branch == "adema" and opt_type == "grokmuonadema":
                # Projector dipaksa ke cabang AdEMAMix dgn LR normal (konservatif)
                if _p2d:
                    param_groups.append({"params": _p2d, "lr": _base, "force_branch": "adema"})
            else:
                if _p2d:
                    param_groups.append({"params": _p2d, "lr": _base * _scale})
            if _p1d:
                param_groups.append({"params": _p1d, "lr": _base})

        _ginfo = [
            (len(g["params"]), format(g["lr"], ".2e"), g.get("force_branch", "auto"))
            for g in param_groups
        ]
        print(f"  Param groups (n, lr, branch): {_ginfo}")

        if opt_type == "grokmuonadema":
            return GrokMuonAdEMA(
                param_groups,
                weight_decay=weight_decay,
                grok_alpha=grok_alpha,
                grok_lamb=grok_lamb,
                betas=adema_betas,
                beta3=adema_beta3,
                momentum=muon_momentum,
                nesterov=muon_nesterov,
                ns_steps=muon_ns_steps,
                max_grad_norm=muon_max_grad_norm,
            )
        elif opt_type == "grokademamix":
            return GrokAdEMAMix(
                param_groups,
                weight_decay=weight_decay,
                grok_alpha=grok_alpha,
                grok_lamb=grok_lamb,
                betas=adema_betas,
                beta3=adema_beta3,
            )
        else:
            raise ValueError(f"OPTIMIZER_TYPE tidak dikenal: {opt_type}")

    return (create_optimizer,)


@app.cell
def _(format_encoder_from_raw, load_dataset, random):
    def load_hf_samples(
        repo_id: str, config_name: str, split: str, n_samples: int, seed: int = 42
    ) -> list[dict]:
        """
        Download dataset dari HF Hub; kalau n_samples > 0, sampling per-group chat_idx
        (percakapan multi-turn tidak pernah terpotong di tengah).
        """
        print(f"Mengunduh dataset '{config_name}' ({split}) dari {repo_id}...")
        try:
            ds = load_dataset(repo_id, config_name, split=split)
            samples = [dict(row) for row in ds]

            if n_samples > 0 and len(samples) > n_samples:
                random.seed(seed)
                if samples and "chat_idx" in samples[0]:
                    groups = {}
                    for s in samples:
                        c_idx = s["chat_idx"]
                        if c_idx not in groups:
                            groups[c_idx] = []
                        groups[c_idx].append(s)
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

    def text_sft_to_joint(samples, is_chat: bool):
        """
        Baris teks (chat_sft / indoqa_sft) -> format joint vision-collator.
        target_text = RAW (collator yang menambahkan <end_of_turn> + EOS).
        """
        rows = []
        if is_chat:
            chat_groups = {}
            for obj in samples:
                if not obj.get("input") or not obj.get("target"):
                    continue
                chat_idx = obj.get("chat_idx", -1)
                chat_groups.setdefault(chat_idx, []).append(obj)

            for chat_idx, turns in chat_groups.items():
                turns = sorted(turns, key=lambda x: x.get("turn_idx", 0))
                for turn in turns:
                    rows.append({
                        "prompt_text": format_encoder_from_raw(turn["input"]),
                        "target_text": turn["target"].strip(),
                        "dataset_idx": -1,
                        "image_indices": [],
                        "images": [],
                        "_modality": "text",
                    })
        else:
            for obj in samples:
                if not obj.get("input") or not obj.get("target"):
                    continue
                rows.append({
                    "prompt_text": format_encoder_from_raw(obj["input"]),
                    "target_text": obj["target"].strip(),
                    "dataset_idx": -1,
                    "image_indices": [],
                    "images": [],
                    "_modality": "text",
                })
        return rows

    def text_orpo_to_joint(samples):
        """Baris chat_orpo -> format joint VisionORPOCollator."""
        rows = []
        for obj in samples:
            if not obj.get("prompt") or not obj.get("chosen") or not obj.get("rejected"):
                continue
            chosen_raw = obj["chosen"].replace("assistant: ", "", 1).strip()
            rejected_raw = obj["rejected"].replace("assistant: ", "", 1).strip()
            if chosen_raw.endswith("<end_of_turn>"):
                chosen_raw = chosen_raw[:-len("<end_of_turn>")].strip()
            if rejected_raw.endswith("<end_of_turn>"):
                rejected_raw = rejected_raw[:-len("<end_of_turn>")].strip()
            rows.append({
                "prompt_text": format_encoder_from_raw(obj["prompt"]),
                "chosen_text": chosen_raw,
                "rejected_text": rejected_raw,
                "dataset_idx": -1,
                "image_indices": [],
                "images": [],
                "_modality": "text",
            })
        return rows

    return load_hf_samples, text_orpo_to_joint, text_sft_to_joint


@app.cell
def _(BASE_T5_MODEL, os):
    # =====================================================================
    # UPLOAD INTEGRITY & PROVENANCE HELPERS
    # Prinsip: artefak di HF harus CLEAN — upload baru dianggap selesai SETELAH
    # seluruh file lokal terverifikasi ada di remote, lalu ditandai marker
    # `upload_complete.json` (commit TERAKHIR). Upload yang gagal di tengah
    # (jaringan putus / server crash) → prefix remote dibersihkan sehingga
    # tidak pernah ada artefak parsial. Cache lokal hanya sah jika
    # provenance-nya cocok dengan marker remote (HF = source of truth).
    # =====================================================================
    import hashlib as _hashlib
    import io as _io
    import json as _json
    from datetime import datetime as _dt
    from datetime import timezone as _tz

    UPLOAD_MARKER = "upload_complete.json"

    import re as _re_mod

    _REPO_ID_RE = _re_mod.compile(r"^[A-Za-z0-9][\w.\-]*/[A-Za-z0-9][\w.\-]*$")

    def _normalize_readme_frontmatter(content: str, base_model: str) -> str:
        """Parse front-matter YAML README, lalu emit ulang HANYA field whitelist
        dengan tipe yang sah menurut validator HF (/api/validate-yaml):
        base_model -> repo id valid, tags/language/datasets -> list[str],
        library_name/license/pipeline_tag -> str. Field lain dibuang.
        (JANGAN regex: tag `base_model:adapter:...` pernah rusak jadi dict YAML.)"""
        import yaml as _yaml

        _body = content
        _meta = {}
        if content.startswith("---"):
            _parts = content.split("---", 2)
            if len(_parts) >= 3:
                _fm_raw, _body = _parts[1], _parts[2]
                try:
                    _loaded = _yaml.safe_load(_fm_raw)
                    if isinstance(_loaded, dict):
                        _meta = _loaded
                except Exception:
                    _meta = {}

        _clean = {}
        _bm = _meta.get("base_model")
        _clean["base_model"] = _bm if (isinstance(_bm, str) and _REPO_ID_RE.match(_bm)) else base_model
        for _k in ("library_name", "license", "pipeline_tag"):
            _v = _meta.get(_k)
            if isinstance(_v, str) and _v.strip():
                _clean[_k] = _v
        for _k in ("tags", "language", "datasets"):
            _v = _meta.get(_k)
            if isinstance(_v, list):
                _items = [x for x in _v if isinstance(x, str) and x.strip()]
                if _items:
                    _clean[_k] = _items
            elif isinstance(_v, str) and _v.strip():
                _clean[_k] = [_v]

        return f"---\n{_yaml.safe_dump(_clean, sort_keys=False, allow_unicode=True)}---{_body}"

    def sanitize_hf_generated_metadata(folder_path: str, base_model: str) -> int:
        """README.md & adapter_config.json hasil Trainer/Peft save sering merekam
        PATH LOKAL (mis. /tmp/... atau results/...) sebagai base_model — validasi
        metadata HF menolak nilai non repo-id sehingga upload_folder GAGAL TOTAL.
        adapter_config: replace nilai JSON invalid. README: normalisasi YAML."""
        _fixes = 0

        _p_adp = os.path.join(folder_path, "adapter_config.json")
        if os.path.exists(_p_adp):
            with open(_p_adp, "r", encoding="utf-8") as _f:
                _c = _f.read()

            def _rep_adapter(m):
                if _REPO_ID_RE.match(m.group(1)):
                    return m.group(0)
                return f'"base_model_name_or_path": "{base_model}"'

            _n = _re_mod.sub(r'"base_model_name_or_path":\s*"([^"]*)"', _rep_adapter, _c)
            if _n != _c:
                with open(_p_adp, "w", encoding="utf-8") as _f:
                    _f.write(_n)
                _fixes += 1

        _p_rd = os.path.join(folder_path, "README.md")
        if os.path.exists(_p_rd):
            with open(_p_rd, "r", encoding="utf-8") as _f:
                _c = _f.read()
            _n = _normalize_readme_frontmatter(_c, base_model)
            if _n != _c:
                with open(_p_rd, "w", encoding="utf-8") as _f:
                    _f.write(_n)
                _fixes += 1

        if _fixes:
            print(f"  🧽 Metadata disanitasi ({_fixes} file): base_model -> {base_model}")
        return _fixes

    def _marker_sha(marker: dict) -> str:
        return _hashlib.sha256(_json.dumps(marker, sort_keys=True).encode()).hexdigest()

    def remote_marker(repo_id: str, prefix: str, token=None):
        """Ambil marker upload_complete.json dari remote (None jika belum ada)."""
        from huggingface_hub import hf_hub_download
        try:
            _p = hf_hub_download(
                repo_id=repo_id,
                filename=f"{prefix.rstrip('/')}/{UPLOAD_MARKER}",
                repo_type="model",
                token=token,
            )
            with open(_p, "r", encoding="utf-8") as _f:
                return _json.load(_f)
        except Exception:
            return None

    def write_local_provenance(local_dir: str, marker) -> None:
        """Catat dari artefak remote mana cache lokal ini berasal."""
        os.makedirs(local_dir, exist_ok=True)
        prov = {"marker_sha": _marker_sha(marker)} if marker else {"marker_sha": "legacy"}
        with open(os.path.join(local_dir, "_hf_provenance.json"), "w", encoding="utf-8") as _f:
            _json.dump(prov, _f, indent=2)

    def local_provenance_valid(local_dir: str, marker) -> bool:
        """Cache lokal sah HANYA jika provenance cocok dengan marker remote saat ini.
        Artefak baru (ber-marker) selalu menang atas cache legacy."""
        _pf = os.path.join(local_dir, "_hf_provenance.json")
        if not os.path.exists(_pf):
            return False
        try:
            with open(_pf, "r", encoding="utf-8") as _f:
                prov = _json.load(_f)
        except Exception:
            return False
        if marker is None:
            return prov.get("marker_sha") == "legacy"
        return prov.get("marker_sha") == _marker_sha(marker)

    def delete_remote_prefix(api, repo_id: str, prefix: str) -> int:
        """Hapus SEMUA file di bawah prefix remote (cleanup upload parsial)."""
        prefix = prefix.rstrip("/") + "/"
        victims = [f for f in api.list_repo_files(repo_id) if f.startswith(prefix)]
        if not victims:
            return 0
        try:
            api.delete_files(repo_id=repo_id, file_paths=victims, repo_type="model")
        except Exception:
            for _v in victims:
                try:
                    api.delete_file(path_in_repo=_v, repo_id=repo_id, repo_type="model")
                except Exception as _e_df:
                    print(f"  ⚠️ Gagal menghapus {_v}: {_e_df}")
        print(f"  🧹 Remote dibersihkan: {prefix} ({len(victims)} file)")
        return len(victims)

    def upload_folder_atomic(api, repo_id: str, folder_path: str, path_in_repo: str, commit_message=None, sanitize_base_model: str = BASE_T5_MODEL) -> dict:
        """Upload folder + verifikasi SEMUA file sampai + marker sebagai commit TERAKHIR.
        Metadata README/adapter_config disanitasi dulu dari path lokal (penyebab
        kegagalan validasi HF). Gagal/parsial di tengah → prefix remote dibersihkan,
        lalu raise."""
        path_in_repo = path_in_repo.rstrip("/")
        sanitize_hf_generated_metadata(folder_path, sanitize_base_model)
        local_files = []
        total_bytes = 0
        for _root, _dirs, _fnames in os.walk(folder_path):
            for _fn in _fnames:
                _full = os.path.join(_root, _fn)
                _rel = os.path.relpath(_full, folder_path).replace("\\", "/")
                local_files.append(_rel)
                total_bytes += os.path.getsize(_full)

        try:
            api.upload_folder(
                folder_path=folder_path,
                path_in_repo=path_in_repo,
                repo_id=repo_id,
                repo_type="model",
                commit_message=commit_message,
            )
            _remote = set(api.list_repo_files(repo_id))
            _missing = [f for f in local_files if f"{path_in_repo}/{f}" not in _remote]
            if _missing:
                raise RuntimeError(f"{len(_missing)}/{len(local_files)} file TIDAK ter-upload: {_missing[:5]}")
        except Exception as _e_up:
            print(f"  ⚠️ Upload gagal/parsial ({_e_up}) — membersihkan '{path_in_repo}' di remote...")
            try:
                delete_remote_prefix(api, repo_id, path_in_repo)
            except Exception as _e_clean:
                print(f"  ⚠️ Cleanup remote gagal: {_e_clean}")
            raise

        _marker = {
            "version": 1,
            "path_in_repo": path_in_repo,
            "n_files": len(local_files),
            "bytes": total_bytes,
            "timestamp": _dt.now(_tz.utc).isoformat(),
        }
        api.upload_file(
            path_or_fileobj=_io.BytesIO(_json.dumps(_marker, indent=2).encode()),
            path_in_repo=f"{path_in_repo}/{UPLOAD_MARKER}",
            repo_id=repo_id,
            repo_type="model",
            commit_message=f"integrity marker: {path_in_repo}",
        )
        print(f"  ✅ Upload terverifikasi lengkap + marker: {path_in_repo} ({len(local_files)} files, {total_bytes/1e6:.1f} MB)")
        return _marker

    def _has_any(files_set, prefix: str, predicate) -> bool:
        return any(f.startswith(prefix) and predicate(f) for f in files_set)

    def prefix_is_complete(files, prefix: str, kind: str = "model") -> bool:
        """Artefak valid = marker ADA, atau legacy-complete (critical files lengkap —
        untuk artefak hasil upload era sebelum marker)."""
        prefix = prefix.rstrip("/") + "/"
        files_set = set(files)
        if f"{prefix}{UPLOAD_MARKER}" in files_set:
            return True
        if kind == "model":
            return _has_any(files_set, prefix, lambda f: f.endswith("config.json")) and _has_any(files_set, prefix, lambda f: f.endswith(".safetensors"))
        if kind == "adapter":
            return _has_any(files_set, prefix, lambda f: f.endswith("adapter_config.json")) and _has_any(files_set, prefix, lambda f: "adapter_model." in f)
        if kind == "checkpoint":
            return (
                _has_any(files_set, prefix, lambda f: f.endswith("adapter_config.json"))
                and _has_any(files_set, prefix, lambda f: "adapter_model." in f)
                and _has_any(files_set, prefix, lambda f: f.endswith("trainer_state.json"))
                and _has_any(files_set, prefix, lambda f: f.rsplit("/", 1)[-1].startswith("optimizer"))
            )
        return False

    def _checkpoint_dirs(files, stage_prefix: str):
        stage_prefix = stage_prefix.rstrip("/")
        return sorted({
            f.split("/")[2]
            for f in files
            if f.startswith(f"{stage_prefix}/checkpoint-") and len(f.split("/")) >= 4
        })

    def complete_checkpoint_dirs(files, stage_prefix: str):
        """Checkpoint-* yang LENGKAP (punya marker / legacy-complete)."""
        stage_prefix = stage_prefix.rstrip("/")
        return [d for d in _checkpoint_dirs(files, stage_prefix) if prefix_is_complete(files, f"{stage_prefix}/{d}", "checkpoint")]

    def incomplete_checkpoint_dirs(files, stage_prefix: str):
        stage_prefix = stage_prefix.rstrip("/")
        return [d for d in _checkpoint_dirs(files, stage_prefix) if not prefix_is_complete(files, f"{stage_prefix}/{d}", "checkpoint")]

    def prefix_has_files(files, prefix: str) -> bool:
        prefix = prefix.rstrip("/") + "/"
        return any(f.startswith(prefix) for f in files)

    return (
        complete_checkpoint_dirs,
        delete_remote_prefix,
        incomplete_checkpoint_dirs,
        local_provenance_valid,
        prefix_has_files,
        prefix_is_complete,
        remote_marker,
        upload_folder_atomic,
        write_local_provenance,
    )


@app.cell
def _(
    CANGKOK_FORCE,
    CANGKOK_SUBFOLDER,
    ENABLE_STEERING,
    FINAL_PREFIX,
    JOINT_PREFIX,
    STEERED_SUBFOLDER,
    STEERING_FORCE,
    UNIFIED_HF_REPO,
    auth_status,
    complete_checkpoint_dirs,
    delete_remote_prefix,
    incomplete_checkpoint_dirs,
    mo,
    os,
    prefix_has_files,
    prefix_is_complete,
):
    from huggingface_hub import HfApi as _StageApi

    _token = os.environ.get("HF_TOKEN")
    _api = _StageApi(token=_token)
    _api.create_repo(repo_id=UNIFIED_HF_REPO, repo_type="model", private=False, exist_ok=True)
    _files = _api.list_repo_files(UNIFIED_HF_REPO)

    # ---- CLEANUP: hapus artefak PARSIAL (upload terputus / server crash) ----
    # Aturan: artefak valid = punya marker upload_complete.json, atau legacy-complete.
    # Selain itu (folder ada isinya tapi critical files hilang) → kotor → hapus.
    _cleanup_targets = []
    for _p, _kind in (
        (STEERED_SUBFOLDER, "model"),
        (CANGKOK_SUBFOLDER, "model"),
        (f"{JOINT_PREFIX}/sft/final_adapter", "adapter"),
        (f"{JOINT_PREFIX}/orpo/final_adapter", "adapter"),
        (f"{FINAL_PREFIX}/merged_bf16", "model"),
        (f"{FINAL_PREFIX}/quantized_4bit", "model"),
    ):
        if prefix_has_files(_files, _p) and not prefix_is_complete(_files, _p, _kind):
            _cleanup_targets.append((_p, f"artefak parsial ({_kind})"))
    for _stage_p in (f"{JOINT_PREFIX}/sft", f"{JOINT_PREFIX}/orpo"):
        for _d in incomplete_checkpoint_dirs(_files, _stage_p):
            _cleanup_targets.append((f"{_stage_p}/{_d}", "checkpoint parsial"))
    if _cleanup_targets:
        print(f"🧹 Membersihkan {len(_cleanup_targets)} artefak upload parsial di HF...")
        for _p, _why in _cleanup_targets:
            try:
                print(f"   • {_p} — {_why}")
                delete_remote_prefix(_api, UNIFIED_HF_REPO, _p)
            except Exception as _e_cl:
                print(f"  ⚠️ Gagal membersihkan {_p}: {_e_cl}")
        _files = _api.list_repo_files(UNIFIED_HF_REPO)  # refresh pasca-cleanup

    steered_exists = prefix_is_complete(_files, STEERED_SUBFOLDER, "model")
    cangkok_exists = prefix_is_complete(_files, CANGKOK_SUBFOLDER, "model")
    sft_done = prefix_is_complete(_files, f"{JOINT_PREFIX}/sft/final_adapter", "adapter")
    orpo_done = prefix_is_complete(_files, f"{JOINT_PREFIX}/orpo/final_adapter", "adapter")
    final_done = prefix_is_complete(_files, f"{FINAL_PREFIX}/merged_bf16", "model")

    sft_resume = len(complete_checkpoint_dirs(_files, f"{JOINT_PREFIX}/sft")) > 0
    orpo_resume = len(complete_checkpoint_dirs(_files, f"{JOINT_PREFIX}/orpo")) > 0

    if final_done:
        pipeline_stage = "done"
    elif orpo_done:
        pipeline_stage = "merge"
    elif sft_done:
        pipeline_stage = "orpo"
    elif cangkok_exists and not CANGKOK_FORCE:
        pipeline_stage = "sft"
    elif (steered_exists and not STEERING_FORCE) or not ENABLE_STEERING:
        pipeline_stage = "cangkok"
    else:
        pipeline_stage = "steering"

    _labels = {
        "steering": "Phase 0.5 (Task Vector Steering)",
        "cangkok": "Phase 1.5 (Vision Grafting)",
        "sft": "Phase 1 (JOINT SFT)",
        "orpo": "Phase 2 (JOINT ORPO)",
        "merge": "Final Merge",
        "done": "✅ SEMUA SELESAI",
    }
    print("=" * 70)
    print(f"📊 PIPELINE STATE REPORT — {UNIFIED_HF_REPO}")
    print(f"   • steered/ exists : {steered_exists}")
    print(f"   • cangkok/ exists : {cangkok_exists}")
    print(f"   • SFT final_adapter: {sft_done} (resume={sft_resume})")
    print(f"   • ORPO final_adapter: {orpo_done} (resume={orpo_resume})")
    print(f"   • Final Merged BF16: {final_done}")
    print(f"   👉 Current Active Stage: {_labels[pipeline_stage]}")
    print("=" * 70)

    mo.md(
        f"**📍 Pipeline Status:** `{pipeline_stage}` ({_labels[pipeline_stage]}) | "
        f"Cangkok: `{cangkok_exists}` | SFT done: `{sft_done}` | ORPO done: `{orpo_done}`"
    )
    return (
        cangkok_exists,
        final_done,
        orpo_done,
        orpo_resume,
        pipeline_stage,
        sft_done,
        sft_resume,
        steered_exists,
    )


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ---
    # 🧭 Phase 0.5 — 3-Way Task Vector Steering
    Menyuntikkan *vektor kemahiran instruksi* `Δ = W_Gemma3-IT − W_Gemma3-Base`
    ke **DECODER** T5Gemma-2 (encoder & vision tower TIDAK disentuh — mereka
    menjaga pre-training UL2 & SigLIP).

    **Merged-attention-aware & Layer-Wise Ramp-Up α** (dapat di-tweak di CONTROL CENTER):

    | Modul | Strategy / α | Alasan |
    |---|---|---|
    | FFN (`gate/up/down_proj`) | **Layer Ramp**: Early=0.05, Mid=0.25, Late=0.08 | Token-wise murni — peak IT reasoning & knowledge di layer tengah (25%-80% depth) |
    | RMSNorm layer | **Layer Ramp**: Early=0.02, Mid=0.08, Late=0.03 | 1D scale — menjaga kalibrasi layer-wise |
    | `q_proj`, `o_proj` | α_QO (0.0) | Multi-head attention decoder |
    | **`k_proj`, `v_proj`** | α_KV (0.0) | Proyeksi joint **[X;H]** — Gemma-IT tak pernah melihat H; WAJIB SKIP (0.0) |
    | `q_norm`, `k_norm` | α_QKNORM (0.0) | Terikat kalibrasi joint softmax |
    | embed / lm_head | — | vocab beda (262144 vs 262208) → auto-skip shape-guard |

    > ⚠️ Berbeda dengan skrip riset awal (yang key-mapping-nya keliru sehingga attention
    > tak pernah tersuntik), implementasi ini memakai **tabel mapping eksplisit**
    """)
    return


@app.cell
def _(
    BASE_T5_MODEL,
    ENABLE_STEERING,
    GEMMA_BASE_MODEL,
    GEMMA_IT_MODEL,
    STEERED_SUBFOLDER,
    STEERING_ALPHA_FFN_EARLY,
    STEERING_ALPHA_FFN_LATE,
    STEERING_ALPHA_FFN_MID,
    STEERING_ALPHA_KV,
    STEERING_ALPHA_NORM_EARLY,
    STEERING_ALPHA_NORM_LATE,
    STEERING_ALPHA_NORM_MID,
    STEERING_ALPHA_QKNORM,
    STEERING_ALPHA_QO,
    STEERING_FORCE,
    STEERING_SMOKE_TEST,
    UNIFIED_HF_REPO,
    format_encoder_from_raw,
    gc,
    os,
    steered_exists,
    torch,
    upload_folder_atomic,
    write_local_provenance,
):
    _token = os.environ.get("HF_TOKEN")

    _should_run = (
        ENABLE_STEERING
        and (STEERING_FORCE or not steered_exists)
    )
    if _should_run:

        print("=" * 90)
        print("  [STEER] 3-Way Task Vector Delta Steering (decoder T5Gemma-2)")
        print("=" * 90)
        print(f"  α_FFN (Early/Mid/Late)  = {STEERING_ALPHA_FFN_EARLY} / {STEERING_ALPHA_FFN_MID} / {STEERING_ALPHA_FFN_LATE}")
        print(f"  α_NORM (Early/Mid/Late) = {STEERING_ALPHA_NORM_EARLY} / {STEERING_ALPHA_NORM_MID} / {STEERING_ALPHA_NORM_LATE}")
        print(f"  α_QO={STEERING_ALPHA_QO} | α_KV={STEERING_ALPHA_KV} | α_QKNORM={STEERING_ALPHA_QKNORM}")

        # ---- 1. Load 3 model di CPU (aman memori; one-time operation) ----
        from transformers import AutoModelForSeq2SeqLM as _SteerSeq2Seq
        from transformers import AutoModelForCausalLM as _SteerCausal

        _load_ok = False
        if not _token:
            print("❌ [STEER] Error: HF_TOKEN belum diset di Cell 'Hugging Face Token'.")
            print("ℹ️ Model 'google/gemma-3-4b-pt' & 'google/gemma-3-4b-it' adalah GATED MODELS di Hugging Face.")
            print("👉 Pastikan Anda telah menyetujui lisensi pada link HF berikut:")
            print("   1. https://huggingface.co/google/gemma-3-4b-pt")
            print("   2. https://huggingface.co/google/gemma-3-4b-it")
            print("   3. https://huggingface.co/google/t5gemma-2-4b-4b")
            print("👉 Masukkan HF Token Anda di input widget pada Cell 'Hugging Face Token' lalu jalankan ulang cell ini.")
            steered_ready = False
        else:
            gc.collect()
            try:
                print(f"\n[1/3] Loading Base T5Gemma-2: {BASE_T5_MODEL} (CPU)...")
                _t5 = _SteerSeq2Seq.from_pretrained(
                    BASE_T5_MODEL, torch_dtype=torch.bfloat16, token=_token, trust_remote_code=True
                )
                print(f"[2/3] Loading Gemma 3 Base: {GEMMA_BASE_MODEL} (CPU)...")
                _g_base = _SteerCausal.from_pretrained(
                    GEMMA_BASE_MODEL, torch_dtype=torch.bfloat16, token=_token, trust_remote_code=True
                )
                print(f"[3/3] Loading Gemma 3 IT: {GEMMA_IT_MODEL} (CPU)...")
                _g_it = _SteerCausal.from_pretrained(
                    GEMMA_IT_MODEL, torch_dtype=torch.bfloat16, token=_token, trust_remote_code=True
                )
                _load_ok = True
            except Exception as _e_load:
                print(f"\n❌ [STEER] Gagal memuat model gated HF: {_e_load}")
                print("ℹ️ Model 'google/gemma-3-4b-pt' & 'google/gemma-3-4b-it' memerlukan akses lisensi HuggingFace!")
                print("👉 Buka link berikut di browser dan klik 'Access repository / Accept license':")
                print("   • https://huggingface.co/google/gemma-3-4b-pt")
                print("   • https://huggingface.co/google/gemma-3-4b-it")
                print("   • https://huggingface.co/google/t5gemma-2-4b-4b")
                print("👉 Kemudian pastikan HF_TOKEN di widget diisi token HuggingFace yang valid.")
                steered_ready = False

            if _load_ok:
                _t5_sd = _t5.state_dict()
                _gb_sd = _g_base.state_dict()
                _gi_sd = _g_it.state_dict()

                _t5_layers = getattr(_t5.config.decoder, "num_hidden_layers", 34)
                _g_cfg = getattr(_g_it.config, "text_config", _g_it.config)
                _g_layers = getattr(_g_cfg, "num_hidden_layers", 34)
                _L = min(_t5_layers, _g_layers)
                print(f"\n  Decoder layers: T5Gemma={_t5_layers}, Gemma3={_g_layers} → steer {_L} layers pertama")

                # ---- 2. Steering dengan mapping eksplisit + dynamic suffix matcher ----
                _counts = {}
                _mismatch = []

                def _find_key(sd, suffix):
                    for k in sd.keys():
                        if k.endswith(suffix):
                            return k
                    return None

                def _steer(g_suf, t_suf, alpha, cat):
                    if alpha == 0:
                        return
                    g_key = _find_key(_gi_sd, g_suf)
                    t_key = _find_key(_t5_sd, t_suf)
                    if g_key and g_key in _gb_sd and t_key:
                        if _t5_sd[t_key].shape == _gi_sd[g_key].shape == _gb_sd[g_key].shape:
                            _t5_sd[t_key] += alpha * (_gi_sd[g_key] - _gb_sd[g_key])
                            _counts[cat] = _counts.get(cat, 0) + 1
                        else:
                            _mismatch.append(
                                f"{t_key}: t5{tuple(_t5_sd[t_key].shape)} vs gemma{tuple(_gi_sd[g_key].shape)}"
                            )
                    else:
                        _mismatch.append(f"missing key: {g_suf} / {t_suf}")

                for _l in range(_L):
                    # Layer-Wise Ramp-Up Alpha Calculation
                    _depth_ratio = _l / float(_L)
                    if _depth_ratio < 0.25:
                        _curr_alpha_ffn = STEERING_ALPHA_FFN_EARLY
                        _curr_alpha_norm = STEERING_ALPHA_NORM_EARLY
                    elif _depth_ratio < 0.80:
                        _curr_alpha_ffn = STEERING_ALPHA_FFN_MID
                        _curr_alpha_norm = STEERING_ALPHA_NORM_MID
                    else:
                        _curr_alpha_ffn = STEERING_ALPHA_FFN_LATE
                        _curr_alpha_norm = STEERING_ALPHA_NORM_LATE

                    # FFN — aman penuh
                    for _proj in ("gate_proj", "up_proj", "down_proj"):
                        _steer(f"layers.{_l}.mlp.{_proj}.weight",
                               f"decoder.layers.{_l}.mlp.{_proj}.weight",
                               _curr_alpha_ffn, "ffn")
                    # Attention projections
                    for _proj, _a in (
                        ("q_proj", STEERING_ALPHA_QO),
                        ("o_proj", STEERING_ALPHA_QO),
                        ("k_proj", STEERING_ALPHA_KV),
                        ("v_proj", STEERING_ALPHA_KV),
                    ):
                        _steer(f"layers.{_l}.self_attn.{_proj}.weight",
                               f"decoder.layers.{_l}.self_attn.{_proj}.weight",
                               _a, f"attn.{_proj}")
                    # q_norm / k_norm
                    for _proj in ("q_norm", "k_norm"):
                        _steer(f"layers.{_l}.self_attn.{_proj}.weight",
                               f"decoder.layers.{_l}.self_attn.{_proj}.weight",
                               STEERING_ALPHA_QKNORM, f"attn.{_proj}")
                    # RMSNorms (Gemma input_layernorm→T5 pre_self_attn, post_attention→post_self_attn)
                    for _g_suf, _t_suf in (
                        ("input_layernorm", "pre_self_attn_layernorm"),
                        ("post_attention_layernorm", "post_self_attn_layernorm"),
                        ("pre_feedforward_layernorm", "pre_feedforward_layernorm"),
                        ("post_feedforward_layernorm", "post_feedforward_layernorm"),
                    ):
                        _steer(f"layers.{_l}.{_g_suf}.weight",
                               f"decoder.layers.{_l}.{_t_suf}.weight",
                               _curr_alpha_norm, "layernorm")

                # Final decoder norm — gunakan NORM_LATE untuk layer paling akhir
                _steer("norm.weight", "decoder.norm.weight", STEERING_ALPHA_NORM_LATE, "final_norm")

                _total = sum(_counts.values())
                print(f"\n  ✅ Steered {_total} tensors: {_counts}")
                if _mismatch:
                    print(f"  ⚠️ {len(_mismatch)} keys di-skip (missing/shape mismatch). Contoh:")
                    for _m in _mismatch[:10]:
                        print(f"     - {_m}")
                if _total == 0:
                    raise RuntimeError("[STEER] Tidak ada satupun tensor yang tersuntik — cek key mapping / shape!")

                # Bebaskan 2 model donor sebelum test & save
                del _g_base, _g_it, _gb_sd, _gi_sd
                gc.collect()

                # ---- 3. Smoke test (generate singkat sebelum upload) ----
                if STEERING_SMOKE_TEST:
                    print("\n  [SMOKE TEST] Generate 3 prompt singkat (eyeball garbage check)...")
                    from transformers import AutoTokenizer as _SteerTok
                    _smoke_tok = _SteerTok.from_pretrained(BASE_T5_MODEL, token=_token, trust_remote_code=True)
                    assert _smoke_tok is not None, f"Gagal memuat tokenizer dari {BASE_T5_MODEL} untuk smoke test"
                    _t5.to("cuda" if torch.cuda.is_available() else "cpu")
                    _t5.eval()
                    _smoke_prompts = [
                        "user: Halo! Perkenalkan dirimu secara singkat.",
                        "user: Apa ibu kota Indonesia?",
                        "user: Tolong ringkas: Fotosintesis adalah proses tumbuhan mengubah cahaya matahari menjadi energi.",
                    ]
                    with torch.no_grad():
                        for _p in _smoke_prompts:
                            _fmt = format_encoder_from_raw(_p)
                            _ids = _smoke_tok.encode(_fmt, add_special_tokens=True, return_tensors="pt").to(_t5.device)
                            _out = _t5.generate(
                                input_ids=_ids, max_new_tokens=48, do_sample=False,
                                pad_token_id=_smoke_tok.pad_token_id,
                            )
                            _resp = _smoke_tok.decode(_out[0], skip_special_tokens=True)
                            print(f"\n  Q: {_p}\n  A: {_resp}")
                    _t5.to("cpu")
                    gc.collect()
                    torch.cuda.empty_cache() if torch.cuda.is_available() else None

                # ---- 4. Save + tokenizer + patch + upload ----
                _local = "/tmp/t5gemma2_steered"
                print(f"\n  Saving steered checkpoint ke {_local} ...")
                os.makedirs(_local, exist_ok=True)
                _t5.save_pretrained(_local, safe_serialization=True)
                from transformers import AutoTokenizer as _SteerTok2
                _steer_tok = _SteerTok2.from_pretrained(BASE_T5_MODEL, token=_token, trust_remote_code=True)
                assert _steer_tok is not None, f"Gagal memuat tokenizer dari {BASE_T5_MODEL}"
                _steer_tok.save_pretrained(_local)

                # Patch tokenizer_config: tambahkan task_prefix_mapping (inline — setara
                # dengan isi tokenizer_config_patched.json di repo v6)
                import json as _json
                _tc_path = os.path.join(_local, "tokenizer_config.json")
                with open(_tc_path, "r", encoding="utf-8") as _f:
                    _tc = _json.load(_f)
                _tc.setdefault("task_prefix_mapping", {
                    "<unused1>": "summarize",
                    "<unused2>": "translate",
                    "<unused3>": "ner",
                    "<unused4>": "qa",
                    "<unused5>": "paraphrase",
                    "<unused6>": "general_chat",
                })
                with open(_tc_path, "w", encoding="utf-8") as _f:
                    _json.dump(_tc, _f, indent=2, ensure_ascii=False)
                print("  ✅ tokenizer_config dipatch dengan task_prefix_mapping")

                print(f"\n  Uploading ke {UNIFIED_HF_REPO} subfolder '{STEERED_SUBFOLDER}/' (verified-atomic)...")
                from huggingface_hub import HfApi as _SteerApi
                _api = _SteerApi(token=_token)
                _marker = upload_folder_atomic(
                    _api,
                    UNIFIED_HF_REPO,
                    _local,
                    STEERED_SUBFOLDER,
                    commit_message=(
                        f"Phase 0.5 Task Vector Steering: ffn_mid={STEERING_ALPHA_FFN_MID}, "
                        f"norm_mid={STEERING_ALPHA_NORM_MID} (Gemma3-IT − Gemma3-Base)"
                    ),
                )
                write_local_provenance(_local, _marker)

                del _t5, _t5_sd, _steer_tok
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

                print(f"\n  ✅ [STEER] BERHASIL! Checkpoint steered di {UNIFIED_HF_REPO}/{STEERED_SUBFOLDER}")
                steered_ready = True
    else:
        print("⏭️ [STEER] Dilewati — ENABLE_STEERING=False / steered sudah ada (tanpa FORCE) / stage sudah lewat.")
        steered_ready = True
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ---
    # 🌱 Phase 1.5 — Cangkok Vision Tower
    Mencangkokkan `vision_tower` (SigLIP 400M) + `multi_modal_projector` dari
    **Gemma 3 4B IT** ke checkpoint hasil Phase 0.5 (atau base T5Gemma jika steering OFF).

    - **Aman**: SigLIP = encoder visual murni, tidak tersentuh *Merged Attention* decoder.
      Shape dimensi SigLIP Gemma 3 4B ≡ T5Gemma-2 (sama-sama SigLIP 400M).
    - Output di-upload ke subfolder `cangkok/` + tokenizer dipatch `task_prefix_mapping`.
    - **TIDAK mencangkok decoder Gemma-IT mentah-mentah** (terbukti merusak output — lihat
      `docs/Reverse Engineering T5Gemma Merge Attention.md`).
    """)
    return


@app.cell
def _(
    BASE_T5_MODEL,
    CANGKOK_FORCE,
    CANGKOK_SUBFOLDER,
    ENABLE_STEERING,
    GEMMA_IT_MODEL,
    STEERED_SUBFOLDER,
    UNIFIED_HF_REPO,
    cangkok_exists,
    gc,
    local_provenance_valid,
    os,
    pipeline_stage,
    remote_marker,
    torch,
    upload_folder_atomic,
    write_local_provenance,
):
    from huggingface_hub import HfApi as _GraftApi

    _token = os.environ.get("HF_TOKEN")
    cangkok_ready = False

    _should_run = (CANGKOK_FORCE or not cangkok_exists)
    if _should_run:

        print("=" * 90)
        print("  [CANGKOK] Grafting SigLIP + Projector Gemma 3 4B IT")
        print("=" * 90)

        # ---- 0. Bebaskan VRAM ----
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        # ---- 1. Tentukan & load TARGET ----
        from transformers import AutoModelForSeq2SeqLM as _GraftSeq2Seq
        from transformers import AutoModelForCausalLM as _GraftCausal
        from transformers import AutoProcessor as _GraftProc

        _steered_local = "/tmp/t5gemma2_steered"
        # Staleness guard: cache lokal steered hanya sah jika cocok dengan marker remote.
        _marker_steered = remote_marker(UNIFIED_HF_REPO, STEERED_SUBFOLDER, _token) if _token else None
        if os.path.isdir(_steered_local) and not local_provenance_valid(_steered_local, _marker_steered):
            import shutil as _sh_wipe_st
            _sh_wipe_st.rmtree(_steered_local, ignore_errors=True)
            print(f"  🧹 Cache lokal steered usang (marker remote beda) — dihapus: {_steered_local}")
        if ENABLE_STEERING:
            if os.path.exists(_steered_local) and os.path.exists(os.path.join(_steered_local, "config.json")):
                _tgt_id = _steered_local
                _tgt_kw = {}
                print(f"\n[A] Target: {_steered_local} (lokal hasil Phase 0.5 — provenance valid)")
            else:
                _tgt_id = UNIFIED_HF_REPO
                _tgt_kw = dict(subfolder=STEERED_SUBFOLDER)
                print(f"\n[A] Target: {UNIFIED_HF_REPO} / steered (HF subfolder Phase 0.5)")
        else:
            _tgt_id = BASE_T5_MODEL
            _tgt_kw = {}
            print(f"\n[A] Target: {BASE_T5_MODEL} (steering OFF)")

        print("    Loading target (CPU, bf16)...")
        _graft_ok = False
        try:
            _model_tgt = _GraftSeq2Seq.from_pretrained(
                _tgt_id, torch_dtype=torch.bfloat16, token=_token, trust_remote_code=True, **_tgt_kw
            )
            print(f"    ✅ {_model_tgt.__class__.__name__}")

            # ---- 2. Load DONOR ----
            print(f"\n[C] Loading donor: {GEMMA_IT_MODEL} ...")
            _model_src = _GraftCausal.from_pretrained(
                GEMMA_IT_MODEL, torch_dtype=torch.bfloat16, token=_token, trust_remote_code=True
            )
            print(f"    ✅ {_model_src.__class__.__name__}")
            _graft_ok = True
        except Exception as _e_graft_load:
            print(f"\n❌ [CANGKOK] Gagal memuat model untuk cangkok: {_e_graft_load}")
            cangkok_ready = False

        if _graft_ok:
            # ---- 3. Ekstrak vision params donor (normalisasi prefix model.) ----
            _src_params = {}
            for _name, _param in _model_src.named_parameters():
                if "vision_tower" in _name or "multi_modal_projector" in _name:
                    _clean = _name[len("model."):] if _name.startswith("model.") else _name
                    _src_params[_clean] = _param.detach().cpu()
            print(f"\n  Donor: {len(_src_params)} vision params (SigLIP + projector)")

            # ---- 4. CANGKOK: copy donor → target ----
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

            # ---- 5. Verifikasi (diff target vs donor harus < 1e-6) ----
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
                    f"[CANGKOK] Gagal: {_grafted} params digraft, {_v_fail} verify fail."
                )

            # ---- 6. Save + processor donor-kompatibel + tokenizer patch + upload ----
            _local_save = "/tmp/v7_vision_cangkok"
            os.makedirs(_local_save, exist_ok=True)
            print(f"\n  Saving lokal ke {_local_save}...")
            _model_tgt.save_pretrained(_local_save, safe_serialization=True)

            # Processor dari T5Gemma2 ORIGINAL (punya full preprocessor_config.json)
            _processor_orig = _GraftProc.from_pretrained(BASE_T5_MODEL, token=_token)
            _processor_orig.save_pretrained(_local_save)

            # Patch tokenizer_config: task_prefix_mapping (inline, sama seperti Phase 0.5)
            import json as _json
            _tc_path = os.path.join(_local_save, "tokenizer_config.json")
            with open(_tc_path, "r", encoding="utf-8") as _f:
                _tc = _json.load(_f)
            _tc.setdefault("task_prefix_mapping", {
                "<unused1>": "summarize",
                "<unused2>": "translate",
                "<unused3>": "ner",
                "<unused4>": "qa",
                "<unused5>": "paraphrase",
                "<unused6>": "general_chat",
            })
            with open(_tc_path, "w", encoding="utf-8") as _f:
                _json.dump(_tc, _f, indent=2, ensure_ascii=False)

            print(f"  Uploading ke {UNIFIED_HF_REPO} subfolder '{CANGKOK_SUBFOLDER}/' (verified-atomic)...")
            from huggingface_hub import HfApi as _GraftApi
            _api = _GraftApi(token=_token)
            _marker = upload_folder_atomic(
                _api,
                UNIFIED_HF_REPO,
                _local_save,
                CANGKOK_SUBFOLDER,
                commit_message="Phase 1.5 Vision Grafting: SigLIP + projector dari Gemma 3 4B IT",
            )
            write_local_provenance(_local_save, _marker)

            del _model_tgt, _model_src, _src_params, _processor_orig
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            print(f"\n  ✅ [CANGKOK] BERHASIL! Base model training di: {UNIFIED_HF_REPO}/{CANGKOK_SUBFOLDER}")
            cangkok_ready = True
    else:
        _is_past = pipeline_stage in ("sft", "orpo", "merge", "done")
        cangkok_ready = cangkok_exists or _is_past
        if not cangkok_ready:
            raise RuntimeError("❌ [CANGKOK] Gagal: Base model `cangkok/` belum tersedia di repo HF dan proses grafting dilewati!")
        else:
            print("⏭️ [CANGKOK] Dilewati — cangkok sudah ada (tanpa FORCE) / stage sudah lewat.")
    if not cangkok_ready:
        raise RuntimeError("❌ [CANGKOK] Gagal: Phase 1.5 Vision Grafting gagal dilakukan!")
    return (cangkok_ready,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ---
    # 🎓 Phase 1 — JOINT SFT (Single-Stage Co-Training)
    Dataset **vision** (`vision_sft`) dan **teks** (`chat_sft` + `indoqa_sft`) dicampur 100% penuh
    dalam satu training loop. **5% percakapan vision di-hold-out UTUH untuk eval**
    (train/eval tidak pernah berbagi percakapan); eval teks dari split `validation` HF.

    - Text rows → format `{prompt_text, target_text, images: []}` (collator yang sama,
      `pixel_values=None` aman bercampur dengan batch multimodal).
    - Gambar vision tetap **lazy-load** (hanya didecode saat collator membaca batch).
    - `multi_modal_projector` di-FULL-FT (`modules_to_save`), SigLIP frozen
      (`finetune_vision_layers=False` — menghindari Unsloth merge bug).
    """)
    return


@app.cell
def _(
    DATASET_VISION_REPO,
    SAMPLE_TRAIN_VISION_SFT,
    SEED,
    VISION_SFT_CONFIG,
    load_dataset,
):
    print(f"[DATA] Memuat vision SFT dari {DATASET_VISION_REPO} ({VISION_SFT_CONFIG})...")
    vision_train_dataset = load_dataset(DATASET_VISION_REPO, VISION_SFT_CONFIG, split="train")

    if SAMPLE_TRAIN_VISION_SFT > 0 and len(vision_train_dataset) > SAMPLE_TRAIN_VISION_SFT:
        vision_train_dataset = vision_train_dataset.shuffle(seed=SEED).select(range(SAMPLE_TRAIN_VISION_SFT))
        print(f"  (disampel menjadi {len(vision_train_dataset)})")
    print(f"✅ [DATA] Vision SFT: {len(vision_train_dataset)} sampel.")
    return (vision_train_dataset,)


@app.cell
def _(
    ALL_SUPPRESS_IDS,
    AutoProcessor,
    CANGKOK_SUBFOLDER,
    FastVisionModel,
    JOINT_PREFIX,
    LOAD_IN_4BIT,
    LORA_ALPHA,
    LORA_DROPOUT,
    LORA_RANK,
    LORA_USE_RSLORA,
    OUTPUT_DIR,
    SEED,
    UNIFIED_HF_REPO,
    apply_logit_mask,
    cangkok_ready,
    gc,
    local_provenance_valid,
    orpo_done,
    os,
    pipeline_stage,
    remote_marker,
    sft_done,
    torch,
    write_local_provenance,
):
    _token = os.environ.get("HF_TOKEN")

    model = None
    processor = None
    tokenizer = None

    if orpo_done or pipeline_stage in ("done", "merge"):
        print(f"[MODEL] Training sudah selesai (orpo_done={orpo_done}); model tidak dimuat untuk training loop.")
    elif not cangkok_ready:
        print("⏭️ [MODEL] Base model `cangkok/` belum siap. Skip loading model.")
    else:
        if sft_done and not orpo_done:
            _model_path = os.path.join(OUTPUT_DIR, JOINT_PREFIX, "sft", "final_adapter")
            # Staleness guard: adapter lokal hanya sah jika cocok marker remote HF.
            _marker_sft_adp = remote_marker(UNIFIED_HF_REPO, f"{JOINT_PREFIX}/sft/final_adapter", _token) if _token else None
            if os.path.isdir(_model_path) and not local_provenance_valid(_model_path, _marker_sft_adp):
                import shutil as _sh_wipe_adp
                _sh_wipe_adp.rmtree(_model_path, ignore_errors=True)
                print(f"  🧹 Cache lokal final_adapter usang (marker remote beda) — dihapus: {_model_path}")
            if not os.path.exists(os.path.join(_model_path, "adapter_config.json")):
                from huggingface_hub import snapshot_download as _model_snap
                print(f"📥 [MODEL] Downloading joint/sft/final_adapter dari HF untuk ORPO...")
                _model_snap(
                    repo_id=UNIFIED_HF_REPO,
                    local_dir=_model_path,
                    allow_patterns=[f"{JOINT_PREFIX}/sft/final_adapter/**"],
                    token=_token,
                )
                _sub_dir = os.path.join(_model_path, JOINT_PREFIX, "sft", "final_adapter")
                if os.path.exists(_sub_dir):
                    import shutil as _sh_load
                    for _item in os.listdir(_sub_dir):
                        _src = os.path.join(_sub_dir, _item)
                        _dst = os.path.join(_model_path, _item)
                        if os.path.exists(_dst):
                            if os.path.isdir(_dst):
                                _sh_load.rmtree(_dst)
                            else:
                                os.remove(_dst)
                        _sh_load.move(_src, _dst)
                    _sh_load.rmtree(os.path.join(_model_path, JOINT_PREFIX))
                write_local_provenance(_model_path, _marker_sft_adp)
            print(f"[MODEL] ORPO stage — load SFT adapter dari: {_model_path}")
            _load_kwargs = dict(
                model_name=_model_path,
                load_in_4bit=LOAD_IN_4BIT,
                use_gradient_checkpointing="unsloth",
                token=_token,
            )
        else:
            # SFT stage — base = cangkok/ hasil Phase 1.5
            # Staleness guard: cache lokal hanya sah jika cocok marker remote HF.
            _cangkok_local = "/tmp/v7_vision_cangkok"
            _marker_cangkok = remote_marker(UNIFIED_HF_REPO, CANGKOK_SUBFOLDER, _token) if _token else None
            for _cand_dir in (_cangkok_local, os.path.join(OUTPUT_DIR, CANGKOK_SUBFOLDER)):
                if os.path.isdir(_cand_dir) and not local_provenance_valid(_cand_dir, _marker_cangkok):
                    import shutil as _sh_wipe_ck
                    _sh_wipe_ck.rmtree(_cand_dir, ignore_errors=True)
                    print(f"  🧹 Cache lokal cangkok usang (marker remote beda) — dihapus: {_cand_dir}")
            if os.path.exists(_cangkok_local) and os.path.exists(os.path.join(_cangkok_local, "config.json")):
                _model_path = _cangkok_local
                print(f"[MODEL] SFT stage — load base dari lokal: {_model_path} (provenance valid)")
            else:
                _model_path = os.path.join(OUTPUT_DIR, CANGKOK_SUBFOLDER)
                if not os.path.exists(os.path.join(_model_path, "config.json")):
                    from huggingface_hub import snapshot_download as _cangkok_snap
                    print(f"📥 [MODEL] Downloading subfolder '{CANGKOK_SUBFOLDER}' dari HF...")
                    _cangkok_snap(
                        repo_id=UNIFIED_HF_REPO,
                        local_dir=_model_path,
                        allow_patterns=[f"{CANGKOK_SUBFOLDER}/**"],
                        token=_token,
                    )
                    _sub_dir = os.path.join(_model_path, CANGKOK_SUBFOLDER)
                    if os.path.exists(_sub_dir):
                        import shutil as _sh_cangkok
                        for _item in os.listdir(_sub_dir):
                            _src = os.path.join(_sub_dir, _item)
                            _dst = os.path.join(_model_path, _item)
                            if os.path.exists(_dst):
                                if os.path.isdir(_dst):
                                    _sh_cangkok.rmtree(_dst)
                                else:
                                    os.remove(_dst)
                            _sh_cangkok.move(_src, _dst)
                        _sh_cangkok.rmtree(_sub_dir)
                    write_local_provenance(_model_path, _marker_cangkok)
                print(f"[MODEL] SFT stage — load base dari local path: {_model_path}")

            _load_kwargs = dict(
                model_name=_model_path,
                load_in_4bit=LOAD_IN_4BIT,
                use_gradient_checkpointing="unsloth",
                token=_token,
            )

        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        model, tokenizer = FastVisionModel.from_pretrained(**_load_kwargs)

        # Reset max_length to silence warning about max_new_tokens taking precedence
        model.config.max_length = None
        if hasattr(model, "generation_config") and model.generation_config is not None:
            model.generation_config.max_length = None

        processor = AutoProcessor.from_pretrained(_model_path, token=_token)

        from unsloth.chat_templates import get_chat_template
        tokenizer = get_chat_template(tokenizer, chat_template="gemma-3")
        processor.chat_template = tokenizer.chat_template
        if hasattr(processor, "tokenizer"):
            processor.tokenizer.chat_template = tokenizer.chat_template

        # Nonaktifkan penambahan bos_token otomatis untuk menghindari bos ganda saat inferensi
        tokenizer.add_bos_token = False
        if hasattr(processor, "tokenizer"):
            processor.tokenizer.add_bos_token = False

        # LoRA hanya saat SFT (ORPO me-load model yang sudah memiliki adapter)
        if not sft_done:
            print("[MODEL] Applying PEFT LoRA (vision_tower=SKIP, projector=FULL FT)...")
            model = FastVisionModel.get_peft_model(
                model,
                finetune_vision_layers=False,      # ⚠️ SKIP vision tower (SigLIP) to avoid Unsloth merge bug
                finetune_language_layers=True,
                finetune_attention_modules=True,
                finetune_mlp_modules=True,
                modules_to_save=["multi_modal_projector"],  # FULL FT projector
                r=LORA_RANK,
                lora_alpha=LORA_ALPHA,
                lora_dropout=LORA_DROPOUT,
                bias="none",
                random_state=SEED,
                use_rslora=LORA_USE_RSLORA,
            )
        else:
            print("[MODEL] Model sudah berisi adapter SFT (ORPO stage). Skip get_peft_model.")

        if not hasattr(model.config, "text_config"):
            type(model.config).text_config = property(lambda self: self.decoder)
            type(model.config).get_text_config = lambda self, *args, **kwargs: self.decoder

        apply_logit_mask(model, ALL_SUPPRESS_IDS)
        FastVisionModel.for_training(model)
    return model, processor, tokenizer


@app.cell
def _(
    DATASET_TEXT_REPO,
    Dataset,
    MAX_EVAL_TEXT_SAMPLES,
    SAMPLE_TRAIN_CHAT,
    SAMPLE_TRAIN_INDOQA,
    SEED,
    TEXT_CHAT_CONFIG,
    TEXT_INDOQA_CONFIG,
    VISION_TEST_SIZE,
    load_hf_samples,
    mo,
    processor,
    random,
    text_sft_to_joint,
    vision_train_dataset,
):
    mo.stop(
        processor is None,
        mo.md("⏭️ **[JOINT-SFT] Model tidak dimuat (stage done/merge) — data prep dilewati.**"),
    )
    print("[JOINT-SFT] ===== Membangun dataset joint (vision + teks) =====")

    # ---- 1. Unroll VISION SFT (gambar lazy via dataset_idx + image_indices) ----
    print("[JOINT-SFT] Unrolling vision SFT (text-only pass)...")
    vision_rows = []
    messages_list = vision_train_dataset["messages"]
    _arrow_images = vision_train_dataset._data.column("images")
    for _idx, _msgs in enumerate(messages_list):
        _num_actual_images = len(_arrow_images[_idx])
        _image_idx = 0
        clean_context = []
        for _msg in _msgs:
            _role = _msg["role"]
            _content = _msg["content"]
            if _role == "user" and "📷" in _content:
                _n_imgs = _content.count("📷")
                _text_content = _content.replace("📷", "").strip()
                clean_content = []
                for _ in range(_n_imgs):
                    if _image_idx < _num_actual_images:
                        clean_content.append({"type": "image"})
                        _image_idx += 1
                if _text_content:
                    clean_content.append({"type": "text", "text": _text_content})
                clean_context.append({"role": _role, "content": clean_content})
            else:
                clean_context.append({"role": _role, "content": [{"type": "text", "text": _content}]})

        for i, msg in enumerate(clean_context):
            if msg["role"] != "assistant":
                continue
            context = clean_context[:i]
            if not context:
                continue

            prompt_text = processor.apply_chat_template(context, tokenize=False, add_generation_prompt=True)

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
                vision_rows.append({
                    "prompt_text": prompt_text,
                    "target_text": target_text,
                    "dataset_idx": _idx,
                    "image_indices": list(range(_num_context_images)),
                    "_modality": "vision",
                })
    print(f"  ✅ Vision rows (unrolled): {len(vision_rows)}")

    # ---- 2. TEKS rows (chat_sft + indoqa_sft -> joint format) ----
    print("[JOINT-SFT] Memuat teks train (chat_sft + indoqa_sft)...")
    _chat_samples = load_hf_samples(DATASET_TEXT_REPO, TEXT_CHAT_CONFIG, "train", SAMPLE_TRAIN_CHAT, seed=SEED)
    _indoqa_samples = load_hf_samples(DATASET_TEXT_REPO, TEXT_INDOQA_CONFIG, "train", SAMPLE_TRAIN_INDOQA, seed=SEED)
    text_rows = text_sft_to_joint(_chat_samples, is_chat=True) + text_sft_to_joint(_indoqa_samples, is_chat=False)
    print(f"  ✅ Text rows total: {len(text_rows)} (chat={len(_chat_samples)}, indoqa={len(_indoqa_samples)})")

    # ---- 3. HOLD-OUT EVAL VISION: 5% PERCAKAPAN UTUH (train/eval tidak kepotong) ----
    # Split di level PERCAKAPAN (dataset_idx) SEBELUM mixing — semua turn dari
    # percakapan yang sama masuk satu sisi saja (train XOR eval). Meniru niat v6
    # (eval valid) sekaligus menambal leak v7 (eval-mm dulu diambil dari pool train).
    _conv_ids = sorted({r["dataset_idx"] for r in vision_rows})
    random.seed(SEED)
    random.shuffle(_conv_ids)
    _n_eval_conv = max(5, int(len(_conv_ids) * VISION_TEST_SIZE))
    _eval_conv_set = set(_conv_ids[:_n_eval_conv])
    _vision_eval_rows = [r for r in vision_rows if r["dataset_idx"] in _eval_conv_set]
    vision_train_rows = [r for r in vision_rows if r["dataset_idx"] not in _eval_conv_set]
    print(f"  ✅ Hold-out eval vision: {_n_eval_conv}/{len(_conv_ids)} percakapan "
          f"({len(_vision_eval_rows)} turn-rows eval / {len(vision_train_rows)} turn-rows train)")

    # ---- 4. JOINT MIXING: 100% vision train (pasca hold-out) + 100% teks train ----
    _actual_ratio = len(text_rows) / max(1, len(text_rows) + len(vision_train_rows))
    print(f"  📊 Joint Mixing (100% Data): vision={len(vision_train_rows)} | teks={len(text_rows)} "
          f"| total={len(vision_train_rows) + len(text_rows)} (rasio teks aktual={_actual_ratio:.2f})")

    joint_rows = vision_train_rows + text_rows
    random.seed(SEED)
    random.shuffle(joint_rows)

    # ---- 5. EVAL SETS ----
    joint_eval_multimodal = Dataset.from_list(_vision_eval_rows) if _vision_eval_rows else None

    # Eval text-only: split validation HF sudah conversation-level sejak di sumbernya;
    # di-cap via sampling group-aware per chat_idx (percakapan tidak kepotong).
    _eval_text_rows = []
    try:
        _per_cfg = max(1, MAX_EVAL_TEXT_SAMPLES // 2)
        _val_chat = load_hf_samples(DATASET_TEXT_REPO, TEXT_CHAT_CONFIG, "validation", _per_cfg, seed=SEED)
        _val_indoqa = load_hf_samples(DATASET_TEXT_REPO, TEXT_INDOQA_CONFIG, "validation", _per_cfg, seed=SEED)
        _eval_text_rows = text_sft_to_joint(_val_chat, is_chat=True) + text_sft_to_joint(_val_indoqa, is_chat=False)
        print(f"  ✅ Text-only eval (validation HF, cap {MAX_EVAL_TEXT_SAMPLES}): {len(_eval_text_rows)} rows")
    except Exception as e:
        print(f"  ⚠️ Gagal memuat eval text-only: {e}")
    joint_eval_text_only = Dataset.from_list(_eval_text_rows) if _eval_text_rows else None

    joint_sft_train_dataset = Dataset.from_list(joint_rows)

    joint_sft_eval_datasets = {}
    if joint_eval_multimodal is not None:
        joint_sft_eval_datasets["multimodal"] = joint_eval_multimodal
    if joint_eval_text_only is not None:
        joint_sft_eval_datasets["text_only"] = joint_eval_text_only

    print(f"\n  ✅ JOINT SFT train: {len(joint_sft_train_dataset)} | "
          f"eval sets: {list(joint_sft_eval_datasets.keys())}")
    return (
        joint_eval_multimodal,
        joint_eval_text_only,
        joint_sft_eval_datasets,
        joint_sft_train_dataset,
    )


@app.cell
def _(torch):
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
                elif "dataset_idx" in item and self.train_dataset is not None and item["dataset_idx"] >= 0:
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

                if self.tok.bos_token_id is not None and (not input_ids or input_ids[0] != self.tok.bos_token_id):
                    input_ids = [self.tok.bos_token_id] + input_ids
                    attention_mask = [1] + attention_mask

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

    return Seq2SeqVisionCollator, VisionORPOCollator


@app.cell
def _(F, SelectiveLabelSmoother, Seq2SeqTrainer, torch):
    class JointSFTTrainer(Seq2SeqTrainer):
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
            if hasattr(FastVisionModel, "for_inference"):
                FastVisionModel.for_inference(self.model)
            else:
                self.model.eval()
            metrics = super().evaluate(
                eval_dataset=eval_dataset,
                ignore_keys=ignore_keys,
                metric_key_prefix=metric_key_prefix,
            )
            for k in list(metrics.keys()):
                if k.endswith("_loss") and k.startswith("eval_"):
                    ppl_key = k.replace("_loss", "_perplexity")
                    try:
                        metrics[ppl_key] = math.exp(metrics[k])
                    except OverflowError:
                        metrics[ppl_key] = float("inf")

            # KRITIS: kembalikan ke training kernels + mode train setelah eval.
            if hasattr(FastVisionModel, "for_training"):
                FastVisionModel.for_training(self.model)
            else:
                self.model.train()
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

    class JointORPOTrainer(Seq2SeqTrainer):
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
            # Buang chosen/rejected labels sebelum .generate() saat eval,
            # dan pakai "chosen" sebagai referensi untuk hitung metrics.
            inputs = dict(inputs)
            cl = inputs.pop("chosen_labels", None)
            inputs.pop("rejected_labels", None)
            if cl is not None and "labels" not in inputs:
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
            if hasattr(FastVisionModel, "for_inference"):
                FastVisionModel.for_inference(self.model)
            else:
                self.model.eval()
            metrics = super().evaluate(
                eval_dataset=eval_dataset,
                ignore_keys=ignore_keys,
                metric_key_prefix=metric_key_prefix,
            )
            for k in list(metrics.keys()):
                if k.endswith("_loss") and k.startswith("eval_"):
                    ppl_key = k.replace("_loss", "_perplexity")
                    try:
                        metrics[ppl_key] = math.exp(metrics[k])
                    except OverflowError:
                        metrics[ppl_key] = float("inf")

            if hasattr(FastVisionModel, "for_training"):
                FastVisionModel.for_training(self.model)
            else:
                self.model.train()
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

    return JointORPOTrainer, JointSFTTrainer


@app.cell
def _(
    Any,
    bertscore_metric,
    bleu_metric,
    cast,
    exact_match_metric,
    meteor_metric,
    np,
    rouge_metric,
):
    def make_compute_metrics(processor):
        def _compute_metrics(eval_preds):
            metrics = {}
            if rouge_metric is None and bleu_metric is None:
                return metrics
            preds, labels = eval_preds
            if isinstance(preds, tuple):
                preds = preds[0]
            tok = cast(Any, processor.tokenizer)

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
        return _compute_metrics

    return (make_compute_metrics,)


@app.cell
def _(
    Any,
    TrainerCallback,
    TrainerControl,
    TrainerState,
    TrainingArguments,
    datetime,
    delete_remote_prefix,
    os,
    torch,
    upload_folder_atomic,
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

            # Hapus eval_loss dari logs agar kolom "Validation Loss" bawaan
            # (yang selalu "No log") tidak tampil; eval sudah di-split per-modality.
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

                # Render chart hanya saat eval (bukan tiap logging step) — hemat CPU.
                self.plot_chart()

        def plot_chart(self) -> None:
            import matplotlib.pyplot as plt
            os.makedirs(self.output_dir, exist_ok=True)

            fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))

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
        Pengganti NotebookProgressCallback bawaan yang SELALU menambahkan kolom
        "Validation Loss" hardcoded walau key eval_loss tidak ada (eval kita
        sudah di-split multimodal & text_only). Meniru semua behavior aslinya
        tanpa kolom default tersebut dan menggabungkan metrik Multimodal & Text-Only
        dalam 1 baris tabel secara berdampingan.
        """

        def __init__(self) -> None:
            self.training_tracker = None
            self.prediction_bar = None
            self._force_next_update = False
            self._buffered_step = None
            self._buffered_values = {}

        def on_train_begin(self, args, state, control, **kwargs) -> None:
            from transformers.trainer_utils import IntervalStrategy
            from transformers.utils.notebook import NotebookTrainingTracker

            self.first_column = "Epoch" if args.eval_strategy == IntervalStrategy.EPOCH else "Step"
            self.training_loss = 0
            self.last_log = 0
            self._buffered_step = None
            self._buffered_values = {}
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

            # Deteksi prefix (eval_multimodal atau eval_text_only)
            is_text_only = any("text_only" in k for k in metrics.keys())
            is_multimodal = any("multimodal" in k for k in metrics.keys())

            metric_key_prefix = "eval"
            for k in metrics:
                if k.endswith("_loss"):
                    metric_key_prefix = _re.sub(r"_loss$", "", k)

            metrics_copy = dict(metrics)
            metrics_copy.pop("total_flos", None)
            metrics_copy.pop("epoch", None)
            metrics_copy.pop(f"{metric_key_prefix}_runtime", None)
            metrics_copy.pop(f"{metric_key_prefix}_samples_per_second", None)
            metrics_copy.pop(f"{metric_key_prefix}_steps_per_second", None)
            metrics_copy.pop(f"{metric_key_prefix}_model_preparation_time", None)

            for k, v in metrics_copy.items():
                splits = k.split("_")
                name = " ".join(part.capitalize() for part in splits[1:])
                values[name] = v

            current_step = state.global_step

            # Buffer dan gabungkan metrik multimodal & text_only di step yang sama
            if self._buffered_step != current_step:
                self._buffered_step = current_step
                self._buffered_values = values
            else:
                self._buffered_values.update(values)

            # Jika ini adalah eval terakhir untuk step ini (text_only atau tidak ada eval lagi), cetak ke tabel
            if is_text_only or (not is_multimodal):
                if self.training_tracker is not None:
                    self.training_tracker.write_line(self._buffered_values)
                    self.training_tracker.remove_child()
                    self._force_next_update = True
                else:
                    disp.display(disp.HTML(text_to_html_table([list(self._buffered_values.keys()), list(self._buffered_values.values())])))
                self._buffered_values = {}

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
            log_filename: str = "eval_samples.txt",
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
            self.log_path = os.path.join(output_dir, log_filename)
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

                    # Immediately free GPU tensor refs (cache cukup di-clear sekali per event)
                    del inputs, outputs

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

    class JointHubUploadCallback(TrainerCallback):
        def __init__(self, repo_id: str, stage: str, hf_prefix: str, token: str | None = None, output_dir: str | None = None, base_model: str = "google/t5gemma-2-4b-4b") -> None:
            self.repo_id = repo_id
            self.stage = stage          # "sft" / "orpo" — nama file artifact lokal
            self.hf_prefix = hf_prefix  # "joint" — subfolder di unified repo
            self.token = token
            self.output_dir = output_dir
            self.base_model = base_model

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
                # Sanitasi metadata base_model (path lokal) ditangani terpusat oleh upload_folder_atomic
                _api.create_repo(repo_id=self.repo_id, repo_type="model", private=False, exist_ok=True)
                print(f"\n📤 Uploading {checkpoint_name} to HF {self.hf_prefix}/{self.stage}/ (verified-atomic)...")
                upload_folder_atomic(
                    _api,
                    self.repo_id,
                    local_checkpoint_path,
                    f"{self.hf_prefix}/{self.stage}/{checkpoint_name}",
                    sanitize_base_model=self.base_model,
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

                # Prune remote: sisakan 2 checkpoint terbaru (remote tidak menumpuk tanpa batas)
                try:
                    _files_now = _api.list_repo_files(self.repo_id)
                    _ckpt_dirs = sorted(
                        {f.split("/")[2] for f in _files_now
                         if f.startswith(f"{self.hf_prefix}/{self.stage}/checkpoint-") and len(f.split("/")) >= 4},
                        key=lambda d: int(d.rsplit("-", 1)[1]),
                    )
                    for _old_ckpt in _ckpt_dirs[:-2]:
                        delete_remote_prefix(_api, self.repo_id, f"{self.hf_prefix}/{self.stage}/{_old_ckpt}")
                except Exception as _e_prune:
                    print(f"⚠️ Prune remote checkpoint gagal (non-fatal): {_e_prune}")
            except Exception as e:
                # FAIL-HARD (permintaan eksplisit): upload checkpoint GAGAL = training STOP.
                # Jangan lanjutkan training tanpa resume protection di HF.
                print(f"❌ Upload GAGAL untuk {checkpoint_name}: {e}")
                raise RuntimeError(
                    f"[HUB-UPLOAD] {checkpoint_name} gagal di-upload ke HF — training dihentikan (fail-hard)."
                ) from e
            return control

    return (
        CleanNotebookProgressCallback,
        JointHubUploadCallback,
        VisionSampleGenerationCallback,
        VisionTrainingPlotCallback,
    )


@app.cell
def _(
    ADEMA_BETA1,
    ADEMA_BETA2,
    ADEMA_BETA3,
    ALL_SUPPRESS_IDS,
    BASE_T5_MODEL,
    BF16,
    CleanNotebookProgressCallback,
    GEN_REPETITION_PENALTY,
    GEN_TEMPERATURE,
    GEN_TOP_P,
    GROK_ALPHA,
    GROK_LAMB,
    JOINT_PREFIX,
    JointHubUploadCallback,
    JointSFTTrainer,
    MAX_EVAL_GEN_SAMPLES,
    MAX_SOURCE_LENGTH,
    MAX_TARGET_LENGTH,
    MUON_MAX_GRAD_NORM,
    MUON_MOMENTUM,
    MUON_NESTEROV,
    MUON_NS_STEPS,
    OPTIMIZER_TYPE,
    OUTPUT_DIR,
    PROJECTOR_BRANCH,
    RUN_SFT,
    SFT_GRADIENT_ACCUMULATION_STEPS,
    SFT_LABEL_SMOOTHING_FACTOR,
    SFT_LEARNING_RATE,
    SFT_LOGGING_STEPS,
    SFT_LR_MULT_DECODER,
    SFT_LR_MULT_ENCODER,
    SFT_LR_MULT_PROJECTOR,
    SFT_LR_MULT_VISION_TOWER,
    SFT_LR_SCHEDULER_TYPE,
    SFT_MAX_GRAD_NORM,
    SFT_MUON_LR_SCALE,
    SFT_NEFTUNE_NOISE_ALPHA,
    SFT_NUM_EPOCHS,
    SFT_PER_DEVICE_EVAL_BATCH_SIZE,
    SFT_PER_DEVICE_TRAIN_BATCH_SIZE,
    SFT_PREDICT_WITH_GENERATE,
    SFT_SAVE_EVAL_STEPS,
    SFT_SAVE_TOTAL_LIMIT,
    SFT_WARMUP_STEPS,
    SFT_WEIGHT_DECAY,
    Seq2SeqTrainingArguments,
    Seq2SeqVisionCollator,
    UNIFIED_HF_REPO,
    VisionSampleGenerationCallback,
    VisionTrainingPlotCallback,
    create_optimizer,
    gc,
    get_scheduler,
    joint_eval_multimodal,
    joint_eval_text_only,
    joint_sft_eval_datasets,
    joint_sft_train_dataset,
    make_compute_metrics,
    model,
    os,
    processor,
    sft_done,
    sft_resume,
    torch,
    upload_folder_atomic,
    vision_train_dataset,
):
    _should_run = RUN_SFT and (not sft_done) and (model is not None)
    if not _should_run:
        print(
            f"⏭️ [JOINT-SFT] Dilewati — RUN_SFT={RUN_SFT}, sft_done={sft_done}, model={'OK' if model is not None else 'None'}."
        )
    if _should_run:
        # Cleanup sisa memori
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        joint_sft_output_dir = os.path.join(OUTPUT_DIR, JOINT_PREFIX, "sft")
        os.makedirs(joint_sft_output_dir, exist_ok=True)
        print(f"[JOINT-SFT] Output dir: {joint_sft_output_dir}")
        print(f"[JOINT-SFT] Train: {len(joint_sft_train_dataset)} | Eval sets: {list(joint_sft_eval_datasets.keys())}")

        # ---- Eval generation samples (multimodal + text-only) ----
        _mm_rows = list(joint_eval_multimodal) if joint_eval_multimodal is not None else []
        _mm_gen_samples = []
        for _item in _mm_rows[:MAX_EVAL_GEN_SAMPLES]:
            _full_imgs = vision_train_dataset[_item["dataset_idx"]]["images"] if _item.get("dataset_idx", -1) >= 0 else []
            _indices = _item.get("image_indices", [])
            _subset = [_full_imgs[i] for i in _indices if i < len(_full_imgs)]
            _mm_gen_samples.append({
                "prompt_text": _item["prompt_text"],
                "target_text": _item["target_text"],
                "images": _subset,
            })
        _to_rows = list(joint_eval_text_only) if joint_eval_text_only is not None else []
        _to_gen_samples = [
            {"prompt_text": r["prompt_text"], "target_text": r["target_text"], "images": []}
            for r in _to_rows[:MAX_EVAL_GEN_SAMPLES]
        ]

        # ---- Optimizer + scheduler ----
        _optimizer = create_optimizer(
            model,
            base_lr=SFT_LEARNING_RATE,
            weight_decay=SFT_WEIGHT_DECAY,
            lr_mults={
                "encoder": SFT_LR_MULT_ENCODER,
                "decoder": SFT_LR_MULT_DECODER,
                "projector": SFT_LR_MULT_PROJECTOR,
                "vision_tower": SFT_LR_MULT_VISION_TOWER,
            },
            opt_type=OPTIMIZER_TYPE,
            grok_alpha=GROK_ALPHA,
            grok_lamb=GROK_LAMB,
            adema_betas=(ADEMA_BETA1, ADEMA_BETA2),
            adema_beta3=ADEMA_BETA3,
            muon_momentum=MUON_MOMENTUM,
            muon_ns_steps=MUON_NS_STEPS,
            muon_nesterov=MUON_NESTEROV,
            muon_max_grad_norm=MUON_MAX_GRAD_NORM,
            muon_lr_scale=SFT_MUON_LR_SCALE,
            projector_branch=PROJECTOR_BRANCH,
        )

        _num_update_steps = max(
            1, len(joint_sft_train_dataset) // (SFT_PER_DEVICE_TRAIN_BATCH_SIZE * SFT_GRADIENT_ACCUMULATION_STEPS)
        )
        _max_steps = _num_update_steps * SFT_NUM_EPOCHS

        if _optimizer is not None:
            _lr_scheduler = get_scheduler(
                name=SFT_LR_SCHEDULER_TYPE,
                optimizer=_optimizer,
                num_warmup_steps=SFT_WARMUP_STEPS,
                num_training_steps=_max_steps,
            )
            _optimizers = (_optimizer, _lr_scheduler)
            _optim_str = "adamw_torch"  # diabaikan — optimizer custom dipasok eksplisit
            print(f"[JOINT-SFT] Optimizer: {type(_optimizer).__name__} | max_steps={_max_steps}")
        else:
            _optimizers = ()
            _optim_str = "paged_adamw_8bit"
            print("[JOINT-SFT] Optimizer: paged_adamw_8bit (dibangun Trainer)")

        # ---- Callbacks ----
        _cfg_sft = getattr(model, "config", None)
        _v_size_sft = getattr(_cfg_sft, "vocab_size", getattr(getattr(_cfg_sft, "text_config", None), "vocab_size", getattr(getattr(_cfg_sft, "decoder", None), "vocab_size", 262144)))
        _bad_words_ids = [
            [id_] for id_ in ALL_SUPPRESS_IDS if id_ < _v_size_sft
        ]
        _plot_cb = VisionTrainingPlotCallback(output_dir=joint_sft_output_dir)
        _progress_cb = CleanNotebookProgressCallback()
        _smp_mm = VisionSampleGenerationCallback(
            processor=processor,
            eval_samples=_mm_gen_samples,
            output_dir=joint_sft_output_dir,
            log_filename="sft_eval_samples_multimodal.txt",
            eval_every_n_steps=SFT_SAVE_EVAL_STEPS,
            temperature=GEN_TEMPERATURE, top_p=GEN_TOP_P, repetition_penalty=GEN_REPETITION_PENALTY,
            bad_words_ids=_bad_words_ids,
        )
        _smp_to = VisionSampleGenerationCallback(
            processor=processor,
            eval_samples=_to_gen_samples,
            output_dir=joint_sft_output_dir,
            log_filename="sft_eval_samples_text_only.txt",
            eval_every_n_steps=SFT_SAVE_EVAL_STEPS,
            temperature=GEN_TEMPERATURE, top_p=GEN_TOP_P, repetition_penalty=GEN_REPETITION_PENALTY,
            bad_words_ids=_bad_words_ids,
        )
        _hub_cb = JointHubUploadCallback(
            repo_id=UNIFIED_HF_REPO,
            stage="sft",
            hf_prefix=JOINT_PREFIX,
            token=os.environ.get("HF_TOKEN"),
            output_dir=joint_sft_output_dir,
            base_model=BASE_T5_MODEL,
        )

        sft_collator = Seq2SeqVisionCollator(processor, MAX_SOURCE_LENGTH, MAX_TARGET_LENGTH, vision_train_dataset)

        joint_sft_trainer = JointSFTTrainer(
            suppress_ids=ALL_SUPPRESS_IDS,
            model=model,
            args=Seq2SeqTrainingArguments(
                output_dir=joint_sft_output_dir,
                per_device_train_batch_size=SFT_PER_DEVICE_TRAIN_BATCH_SIZE,
                per_device_eval_batch_size=SFT_PER_DEVICE_EVAL_BATCH_SIZE,
                gradient_accumulation_steps=SFT_GRADIENT_ACCUMULATION_STEPS,
                eval_accumulation_steps=1,
                learning_rate=SFT_LEARNING_RATE,
                num_train_epochs=SFT_NUM_EPOCHS,
                warmup_steps=SFT_WARMUP_STEPS,
                weight_decay=SFT_WEIGHT_DECAY,
                max_grad_norm=SFT_MAX_GRAD_NORM,
                lr_scheduler_type=SFT_LR_SCHEDULER_TYPE,
                logging_steps=SFT_LOGGING_STEPS,
                save_strategy="steps",
                save_steps=SFT_SAVE_EVAL_STEPS,
                save_total_limit=SFT_SAVE_TOTAL_LIMIT,
                remove_unused_columns=False,
                fp16=False,
                bf16=BF16,
                optim=_optim_str,
                label_smoothing_factor=SFT_LABEL_SMOOTHING_FACTOR,
                neftune_noise_alpha=SFT_NEFTUNE_NOISE_ALPHA,
                gradient_checkpointing=True,
                eval_strategy="steps",
                eval_steps=SFT_SAVE_EVAL_STEPS,
                report_to="none",
                predict_with_generate=SFT_PREDICT_WITH_GENERATE,
                generation_max_length=MAX_TARGET_LENGTH,
            ),
            train_dataset=joint_sft_train_dataset,
            eval_dataset=joint_sft_eval_datasets,
            data_collator=sft_collator,
            optimizers=_optimizers,
            compute_metrics=make_compute_metrics(processor),
            callbacks=[_plot_cb, _progress_cb, _smp_mm, _smp_to, _hub_cb],
        )
        from transformers.utils.notebook import NotebookProgressCallback as _HFNPC
        joint_sft_trainer.remove_callback(_HFNPC)

        # ---- Resume dari HF checkpoint ----
        _resume_from = None
        if sft_resume:
            try:
                from huggingface_hub import snapshot_download as _resume_snap
                from huggingface_hub import HfApi as _ResumeApi

                _api = _ResumeApi(token=os.environ.get("HF_TOKEN"))
                _files = _api.list_repo_files(repo_id=UNIFIED_HF_REPO)

                _ckpt_prefix = f"{JOINT_PREFIX}/sft/checkpoint-"
                _ckpts = list(set([f.split('/')[2] for f in _files if f.startswith(_ckpt_prefix)]))
                if _ckpts:
                    _ckpts.sort(key=lambda x: int(x.split('-')[1]))
                    _latest_ckpt = _ckpts[-1]
                else:
                    _latest_ckpt = "checkpoint-*"

                print(f"\n📥 [JOINT-SFT] Downloading {_latest_ckpt} untuk resume...")
                _resume_snap(
                    repo_id=UNIFIED_HF_REPO,
                    local_dir=joint_sft_output_dir,
                    allow_patterns=[f"{JOINT_PREFIX}/sft/{_latest_ckpt}/**"],
                    token=os.environ.get("HF_TOKEN"),
                )
                _sub_dir = os.path.join(joint_sft_output_dir, JOINT_PREFIX, "sft")
                if os.path.exists(_sub_dir):
                    import shutil as _shutil_r
                    for _item in os.listdir(_sub_dir):
                        _src = os.path.join(_sub_dir, _item)
                        _dst = os.path.join(joint_sft_output_dir, _item)
                        if os.path.isdir(_src) and _item.startswith("checkpoint-"):
                            if os.path.exists(_dst):
                                _shutil_r.rmtree(_dst)
                            _shutil_r.move(_src, _dst)
                    _shutil_r.rmtree(os.path.join(joint_sft_output_dir, JOINT_PREFIX))

                _checkpoints = sorted([
                    d for d in os.listdir(joint_sft_output_dir)
                    if d.startswith("checkpoint-") and os.path.isdir(os.path.join(joint_sft_output_dir, d))
                    and os.path.exists(os.path.join(joint_sft_output_dir, d, "adapter_config.json"))
                ])
                if _checkpoints:
                    _resume_from = True
                    print(f"✅ [JOINT-SFT] {len(_checkpoints)} checkpoint(s) ditemukan — resume!")
                else:
                    print("⚠️ [JOINT-SFT] Tidak ada checkpoint valid — mulai dari awal.")
            except Exception as e:
                print(f"❌ [JOINT-SFT] Gagal download checkpoint: {e}.")
                raise RuntimeError(
                    "❌ [JOINT-SFT] Resume ditandai tersedia di HF tapi gagal di-download — "
                    "pipeline dihentikan (fail-hard; dagi dari awal tanpa resume berbahaya)."
                ) from e

        # ---- Train ----
        print("\n🚀 [JOINT-SFT] Starting JOINT training (vision + teks)...")
        joint_sft_trainer.train(resume_from_checkpoint=_resume_from)

        # ---- Save & upload final adapter ----
        _final_path = os.path.join(joint_sft_output_dir, "final_adapter")
        print(f"\n💾 [JOINT-SFT] Saving final adapter ke {_final_path}...")
        joint_sft_trainer.save_model(_final_path)
        processor.save_pretrained(_final_path)
        # Save chat_template.jinja for easy deployment
        _chat_template = getattr(processor, 'chat_template', None) or getattr(getattr(processor, 'tokenizer', None), 'chat_template', None)
        if _chat_template:
            _jinja_path = os.path.join(_final_path, "chat_template.jinja")
            with open(_jinja_path, "w", encoding="utf-8") as _f:
                _f.write(_chat_template)
            print(f"   💾 Chat template saved to {_jinja_path}")
        try:
            from huggingface_hub import HfApi as _FinalApi
            _final_api = _FinalApi(token=os.environ.get("HF_TOKEN"))
            _final_api.create_repo(repo_id=UNIFIED_HF_REPO, repo_type="model", private=False, exist_ok=True)
            upload_folder_atomic(
                _final_api,
                UNIFIED_HF_REPO,
                _final_path,
                f"{JOINT_PREFIX}/sft/final_adapter",
            )
            print("✅ [JOINT-SFT] Final adapter ter-upload ke joint/sft/final_adapter!")
        except Exception as e:
            print(f"❌ [JOINT-SFT] Upload final adapter GAGAL: {e}")
            raise RuntimeError("❌ [JOINT-SFT] Upload final adapter gagal — pipeline dihentikan (fail-hard).") from e
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ---
    # 🎯 Phase 2 — JOINT ORPO (Preference Optimization)
    Dataset **vision_orpo** + **chat_orpo** dicampur 100% penuh (semua untuk train).
    **Eval memakai himpunan eval SFT yang sama** (holdout vision conv + subset validation teks)
    sehingga delta kualitas ORPO vs SFT terbaca langsung. Loss = `CE(chosen)` + `β · OR-loss`.
    **Label smoothing WAJIB 0** saat ORPO (smoothing merusak kurva odds-ratio).
    Forward di-split encoder→decoder (hemat ~40% VRAM, mencegah OOM dual forward chosen/rejected).
    """)
    return


@app.cell
def _(
    DATASET_TEXT_REPO,
    DATASET_VISION_REPO,
    Dataset,
    SAMPLE_TRAIN_TEXT_ORPO,
    SAMPLE_TRAIN_VISION_ORPO,
    SEED,
    TEXT_ORPO_CONFIG,
    VISION_ORPO_CONFIG,
    joint_eval_multimodal,
    joint_eval_text_only,
    load_dataset,
    load_hf_samples,
    mo,
    processor,
    random,
    text_orpo_to_joint,
    torch,
    vision_train_dataset,
):
    mo.stop(
        processor is None,
        mo.md("⏭️ **[JOINT-ORPO] Model tidak dimuat — data prep dilewati.**"),
    )
    mo.stop(
        torch is None,  # dependency-edge guard (tidak pernah True; hanya ordering)
        mo.md("unreachable"),
    )

    # ---- 1. Load & format VISION ORPO (gambar lazy via dataset_idx) ----
    print(f"[JOINT-ORPO] Memuat vision ORPO dari {DATASET_VISION_REPO}...")
    raw_orpo_dataset = load_dataset(DATASET_VISION_REPO, VISION_ORPO_CONFIG, split="train")
    if SAMPLE_TRAIN_VISION_ORPO > 0 and len(raw_orpo_dataset) > SAMPLE_TRAIN_VISION_ORPO:
        raw_orpo_dataset = raw_orpo_dataset.shuffle(seed=SEED).select(range(SAMPLE_TRAIN_VISION_ORPO))
    print(f"  ✅ Vision ORPO: {len(raw_orpo_dataset)} sampel.")

    vision_orpo_rows = []
    prompts_list = raw_orpo_dataset["prompt"]
    chosen_list = raw_orpo_dataset["chosen"]
    rejected_list = raw_orpo_dataset["rejected"]

    for _idx_orpo in range(len(prompts_list)):
        prompt_str = prompts_list[_idx_orpo]
        chosen_raw = chosen_list[_idx_orpo].replace("assistant: ", "", 1).strip()
        rejected_raw = rejected_list[_idx_orpo].replace("assistant: ", "", 1).strip()

        # Parse prompt -> messages (hitung penanda 📷 tanpa load gambar)
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
        for _role_o, _content_o in raw_messages:
            if _role_o == "user" and "📷" in _content_o:
                _num_images_o = _content_o.count("📷")
                _text_content_o = _content_o.replace("📷", "").strip()
                new_content = []
                for _ in range(_num_images_o):
                    new_content.append({"type": "image"})
                if _text_content_o:
                    new_content.append({"type": "text", "text": _text_content_o})
                new_messages.append({"role": _role_o, "content": new_content})
            else:
                new_messages.append({"role": _role_o, "content": [{"type": "text", "text": _content_o}]})

        # Gabungkan turn dengan role sama yang berurutan (apply_chat_template
        # menolak "Conversation roles must alternate user/assistant/...")
        _merged_messages_o = []
        for _msg_o in new_messages:
            _role_o = _msg_o["role"]
            _content_o = _msg_o["content"]
            if _merged_messages_o and _merged_messages_o[-1]["role"] == _role_o:
                _last_msg_o = _merged_messages_o.pop()
                _merged_content_o = list(_last_msg_o["content"]) + list(_content_o)
                _merged_messages_o.append({"role": _role_o, "content": _merged_content_o})
            else:
                _merged_messages_o.append({"role": _role_o, "content": list(_content_o)})
        new_messages = _merged_messages_o

        pt = processor.apply_chat_template(new_messages, tokenize=False, add_generation_prompt=True)

        if chosen_raw.endswith("<end_of_turn>"):
            chosen_raw = chosen_raw[:-len("<end_of_turn>")].strip()
        if rejected_raw.endswith("<end_of_turn>"):
            rejected_raw = rejected_raw[:-len("<end_of_turn>")].strip()

        vision_orpo_rows.append({
            "prompt_text": pt,
            "chosen_text": chosen_raw,
            "rejected_text": rejected_raw,
            "dataset_idx": _idx_orpo,
            "image_indices": list(range(_num_images_o)),
            "_modality": "vision",
        })
    print(f"  ✅ Vision ORPO rows: {len(vision_orpo_rows)}")

    # ---- 2. TEKS ORPO (chat_orpo -> joint format) ----
    print("[JOINT-ORPO] Memuat teks ORPO (chat_orpo)...")
    _text_orpo_samples = load_hf_samples(DATASET_TEXT_REPO, TEXT_ORPO_CONFIG, "train", SAMPLE_TRAIN_TEXT_ORPO, seed=SEED)
    text_orpo_rows = text_orpo_to_joint(_text_orpo_samples)
    print(f"  ✅ Text ORPO rows: {len(text_orpo_rows)}")

    # ---- 3. JOINT MIXING: Gunakan 100% data vision ORPO + 100% data teks ORPO ----
    _actual_ratio_o = len(text_orpo_rows) / max(1, len(text_orpo_rows) + len(vision_orpo_rows))
    print(f"  📊 Joint ORPO Mix (100% Data): vision={len(vision_orpo_rows)} | teks={len(text_orpo_rows)} "
          f"| total={len(vision_orpo_rows) + len(text_orpo_rows)} (rasio teks aktual={_actual_ratio_o:.2f})")

    joint_orpo_rows = vision_orpo_rows + text_orpo_rows
    random.seed(SEED)
    random.shuffle(joint_orpo_rows)

    # ---- 4. EVAL ORPO = himpunan eval SFT yang sama (apple-to-apple SFT vs ORPO) ----
    # ORPO berfungsi "memperbaiki" policy; kualitasnya diukur pada eval sets SFT
    # (holdout vision conversation-level + subset validation teks) agar delta
    # metrik ORPO vs SFT terbaca langsung. vision_orpo + chat_orpo = 100% train.
    _orpo_eval_mm_rows = []
    if joint_eval_multimodal is not None:
        for _r_mm in joint_eval_multimodal:
            _full_o = vision_train_dataset[_r_mm["dataset_idx"]]["images"] if _r_mm.get("dataset_idx", -1) >= 0 else []
            _imgs_o = [_full_o[_i] for _i in _r_mm.get("image_indices", []) if _i < len(_full_o)]
            _orpo_eval_mm_rows.append({
                "prompt_text": _r_mm["prompt_text"],
                "chosen_text": _r_mm["target_text"],
                "rejected_text": _r_mm["target_text"],  # mirror — hanya dipakai path label CE saat eval
                "images": _imgs_o,                      # eager-load: eval-mm memakai sumber SFT (bukan raw_orpo_dataset)
                "dataset_idx": -1,
                "_modality": "vision",
            })
    orpo_eval_mm = Dataset.from_list(_orpo_eval_mm_rows) if _orpo_eval_mm_rows else None

    _orpo_eval_text_rows = []
    if joint_eval_text_only is not None:
        for _r_to in joint_eval_text_only:
            _orpo_eval_text_rows.append({
                "prompt_text": _r_to["prompt_text"],
                "chosen_text": _r_to["target_text"],
                "rejected_text": _r_to["target_text"],
                "images": [],
                "dataset_idx": -1,
                "_modality": "text",
            })
    orpo_eval_text = Dataset.from_list(_orpo_eval_text_rows) if _orpo_eval_text_rows else None

    joint_orpo_train_dataset = Dataset.from_list(joint_orpo_rows)
    joint_orpo_eval_datasets = {}
    if orpo_eval_mm is not None:
        joint_orpo_eval_datasets["multimodal"] = orpo_eval_mm
    if orpo_eval_text is not None:
        joint_orpo_eval_datasets["text_only"] = orpo_eval_text

    print(f"\n  ✅ JOINT ORPO train: {len(joint_orpo_train_dataset)} | eval (set SFT): {list(joint_orpo_eval_datasets.keys())}")
    return (
        joint_orpo_eval_datasets,
        joint_orpo_train_dataset,
        orpo_eval_mm,
        orpo_eval_text,
        raw_orpo_dataset,
    )


@app.cell
def _(
    ADEMA_BETA1,
    ADEMA_BETA2,
    ADEMA_BETA3,
    ALL_SUPPRESS_IDS,
    BF16,
    CleanNotebookProgressCallback,
    GEN_REPETITION_PENALTY,
    GEN_TEMPERATURE,
    GEN_TOP_P,
    GROK_ALPHA,
    GROK_LAMB,
    JOINT_PREFIX,
    JointHubUploadCallback,
    JointORPOTrainer,
    MAX_EVAL_GEN_SAMPLES,
    MAX_SOURCE_LENGTH,
    MAX_TARGET_LENGTH,
    MUON_MAX_GRAD_NORM,
    MUON_MOMENTUM,
    MUON_NESTEROV,
    MUON_NS_STEPS,
    OPTIMIZER_TYPE,
    ORPO_BETA,
    ORPO_GRADIENT_ACCUMULATION_STEPS,
    ORPO_LABEL_SMOOTHING_FACTOR,
    ORPO_LEARNING_RATE,
    ORPO_LOGGING_STEPS,
    ORPO_LR_MULT_DECODER,
    ORPO_LR_MULT_ENCODER,
    ORPO_LR_MULT_PROJECTOR,
    ORPO_LR_MULT_VISION_TOWER,
    ORPO_LR_SCHEDULER_TYPE,
    ORPO_MAX_GRAD_NORM,
    ORPO_MUON_LR_SCALE,
    ORPO_NUM_EPOCHS,
    ORPO_PER_DEVICE_EVAL_BATCH_SIZE,
    ORPO_PER_DEVICE_TRAIN_BATCH_SIZE,
    ORPO_PREDICT_WITH_GENERATE,
    ORPO_SAVE_EVAL_STEPS,
    ORPO_SAVE_TOTAL_LIMIT,
    ORPO_WARMUP_STEPS,
    ORPO_WEIGHT_DECAY,
    OUTPUT_DIR,
    PROJECTOR_BRANCH,
    RUN_ORPO,
    Seq2SeqTrainingArguments,
    UNIFIED_HF_REPO,
    VisionORPOCollator,
    VisionSampleGenerationCallback,
    VisionTrainingPlotCallback,
    create_optimizer,
    gc,
    get_scheduler,
    joint_orpo_eval_datasets,
    joint_orpo_train_dataset,
    make_compute_metrics,
    model,
    orpo_done,
    orpo_eval_mm,
    orpo_eval_text,
    orpo_resume,
    os,
    processor,
    raw_orpo_dataset,
    torch,
    upload_folder_atomic,
):
    _should_run = RUN_ORPO and (not orpo_done) and (model is not None)
    if not _should_run:
        print(
            f"⏭️ [JOINT-ORPO] Dilewati — RUN_ORPO={RUN_ORPO}, orpo_done={orpo_done}, model={'OK' if model is not None else 'None'}."
        )
    if _should_run:
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        joint_orpo_output_dir = os.path.join(OUTPUT_DIR, JOINT_PREFIX, "orpo")
        os.makedirs(joint_orpo_output_dir, exist_ok=True)
        print(f"[JOINT-ORPO] Output dir: {joint_orpo_output_dir}")
        print(f"[JOINT-ORPO] Train: {len(joint_orpo_train_dataset)} | Eval sets: {list(joint_orpo_eval_datasets.keys())} | beta={ORPO_BETA}")

        # ---- Eval generation samples (gambar sudah eager-embedded dari set eval SFT) ----
        _mm_o = list(orpo_eval_mm) if orpo_eval_mm is not None else []
        _mm_gen_o = []
        for _item_o in _mm_o[:MAX_EVAL_GEN_SAMPLES]:
            _mm_gen_o.append({
                "prompt_text": _item_o["prompt_text"],
                "target_text": _item_o["chosen_text"],
                "images": _item_o.get("images", []),
            })
        _to_o = list(orpo_eval_text) if orpo_eval_text is not None else []
        _to_gen_o = [
            {"prompt_text": r["prompt_text"], "target_text": r["chosen_text"], "images": []}
            for r in _to_o[:MAX_EVAL_GEN_SAMPLES]
        ]

        # ---- Optimizer + scheduler (ORPO lr mults) ----
        _optimizer_o = create_optimizer(
            model,
            base_lr=ORPO_LEARNING_RATE,
            weight_decay=ORPO_WEIGHT_DECAY,
            lr_mults={
                "encoder": ORPO_LR_MULT_ENCODER,
                "decoder": ORPO_LR_MULT_DECODER,
                "projector": ORPO_LR_MULT_PROJECTOR,
                "vision_tower": ORPO_LR_MULT_VISION_TOWER,
            },
            opt_type=OPTIMIZER_TYPE,
            grok_alpha=GROK_ALPHA,
            grok_lamb=GROK_LAMB,
            adema_betas=(ADEMA_BETA1, ADEMA_BETA2),
            adema_beta3=ADEMA_BETA3,
            muon_momentum=MUON_MOMENTUM,
            muon_ns_steps=MUON_NS_STEPS,
            muon_nesterov=MUON_NESTEROV,
            muon_max_grad_norm=MUON_MAX_GRAD_NORM,
            muon_lr_scale=ORPO_MUON_LR_SCALE,
            projector_branch=PROJECTOR_BRANCH,
        )

        _num_update_o = max(
            1, len(joint_orpo_train_dataset) // (ORPO_PER_DEVICE_TRAIN_BATCH_SIZE * ORPO_GRADIENT_ACCUMULATION_STEPS)
        )
        _max_steps_o = _num_update_o * ORPO_NUM_EPOCHS

        if _optimizer_o is not None:
            _lr_scheduler_o = get_scheduler(
                name=ORPO_LR_SCHEDULER_TYPE,
                optimizer=_optimizer_o,
                num_warmup_steps=ORPO_WARMUP_STEPS,
                num_training_steps=_max_steps_o,
            )
            _optimizers_o = (_optimizer_o, _lr_scheduler_o)
            _optim_str_o = "adamw_torch"
            print(f"[JOINT-ORPO] Optimizer: {type(_optimizer_o).__name__} | max_steps={_max_steps_o}")
        else:
            _optimizers_o = ()
            _optim_str_o = "paged_adamw_8bit"
            print("[JOINT-ORPO] Optimizer: paged_adamw_8bit (dibangun Trainer)")

        _cfg_orpo = getattr(model, "config", None)
        _v_size_o = getattr(_cfg_orpo, "vocab_size", getattr(getattr(_cfg_orpo, "text_config", None), "vocab_size", getattr(getattr(_cfg_orpo, "decoder", None), "vocab_size", 262144)))
        _bad_words_o = [
            [id_] for id_ in ALL_SUPPRESS_IDS if id_ < _v_size_o
        ]
        _plot_o = VisionTrainingPlotCallback(output_dir=joint_orpo_output_dir)
        _progress_o = CleanNotebookProgressCallback()
        _smp_mm_o = VisionSampleGenerationCallback(
            processor=processor,
            eval_samples=_mm_gen_o,
            output_dir=joint_orpo_output_dir,
            log_filename="orpo_eval_samples_multimodal.txt",
            eval_every_n_steps=ORPO_SAVE_EVAL_STEPS,
            temperature=GEN_TEMPERATURE, top_p=GEN_TOP_P, repetition_penalty=GEN_REPETITION_PENALTY,
            bad_words_ids=_bad_words_o,
        )
        _smp_to_o = VisionSampleGenerationCallback(
            processor=processor,
            eval_samples=_to_gen_o,
            output_dir=joint_orpo_output_dir,
            log_filename="orpo_eval_samples_text_only.txt",
            eval_every_n_steps=ORPO_SAVE_EVAL_STEPS,
            temperature=GEN_TEMPERATURE, top_p=GEN_TOP_P, repetition_penalty=GEN_REPETITION_PENALTY,
            bad_words_ids=_bad_words_o,
        )
        _hub_o = JointHubUploadCallback(
            repo_id=UNIFIED_HF_REPO,
            stage="orpo",
            hf_prefix=JOINT_PREFIX,
            token=os.environ.get("HF_TOKEN"),
            output_dir=joint_orpo_output_dir,
        )

        orpo_collator = VisionORPOCollator(processor, MAX_SOURCE_LENGTH, MAX_TARGET_LENGTH, raw_orpo_dataset)

        joint_orpo_trainer = JointORPOTrainer(
            beta=ORPO_BETA,
            model=model,
            args=Seq2SeqTrainingArguments(
                output_dir=joint_orpo_output_dir,
                per_device_train_batch_size=ORPO_PER_DEVICE_TRAIN_BATCH_SIZE,
                per_device_eval_batch_size=ORPO_PER_DEVICE_EVAL_BATCH_SIZE,
                gradient_accumulation_steps=ORPO_GRADIENT_ACCUMULATION_STEPS,
                eval_accumulation_steps=1,
                learning_rate=ORPO_LEARNING_RATE,
                num_train_epochs=ORPO_NUM_EPOCHS,
                warmup_steps=ORPO_WARMUP_STEPS,
                weight_decay=ORPO_WEIGHT_DECAY,
                max_grad_norm=ORPO_MAX_GRAD_NORM,
                lr_scheduler_type=ORPO_LR_SCHEDULER_TYPE,
                logging_steps=ORPO_LOGGING_STEPS,
                save_strategy="steps",
                save_steps=ORPO_SAVE_EVAL_STEPS,
                save_total_limit=ORPO_SAVE_TOTAL_LIMIT,
                remove_unused_columns=False,
                fp16=False,
                bf16=BF16,
                optim=_optim_str_o,
                label_smoothing_factor=ORPO_LABEL_SMOOTHING_FACTOR,  # 0.0 — WAJIB
                gradient_checkpointing=True,
                eval_strategy="steps",
                eval_steps=ORPO_SAVE_EVAL_STEPS,
                report_to="none",
                predict_with_generate=ORPO_PREDICT_WITH_GENERATE,
                generation_max_length=MAX_TARGET_LENGTH,
            ),
            train_dataset=joint_orpo_train_dataset,
            eval_dataset=joint_orpo_eval_datasets,
            data_collator=orpo_collator,
            optimizers=_optimizers_o,
            compute_metrics=make_compute_metrics(processor),
            callbacks=[_plot_o, _progress_o, _smp_mm_o, _smp_to_o, _hub_o],
        )
        from transformers.utils.notebook import NotebookProgressCallback as _HFNPC2
        joint_orpo_trainer.remove_callback(_HFNPC2)

        # ---- Resume dari HF checkpoint ----
        _resume_from_o = None
        if orpo_resume:
            try:
                from huggingface_hub import snapshot_download as _resume_snap_o
                from huggingface_hub import HfApi as _ResumeApiO

                _api_o = _ResumeApiO(token=os.environ.get("HF_TOKEN"))
                _files_o = _api_o.list_repo_files(repo_id=UNIFIED_HF_REPO)

                _ckpt_prefix_o = f"{JOINT_PREFIX}/orpo/checkpoint-"
                _ckpts_o = list(set([f.split('/')[2] for f in _files_o if f.startswith(_ckpt_prefix_o)]))
                if _ckpts_o:
                    _ckpts_o.sort(key=lambda x: int(x.split('-')[1]))
                    _latest_ckpt_o = _ckpts_o[-1]
                else:
                    _latest_ckpt_o = "checkpoint-*"

                print(f"\n📥 [JOINT-ORPO] Downloading {_latest_ckpt_o} untuk resume...")
                _resume_snap_o(
                    repo_id=UNIFIED_HF_REPO,
                    local_dir=joint_orpo_output_dir,
                    allow_patterns=[f"{JOINT_PREFIX}/orpo/{_latest_ckpt_o}/**"],
                    token=os.environ.get("HF_TOKEN"),
                )
                _sub_dir_o = os.path.join(joint_orpo_output_dir, JOINT_PREFIX, "orpo")
                if os.path.exists(_sub_dir_o):
                    import shutil as _shutil_ro
                    for _item_o2 in os.listdir(_sub_dir_o):
                        _src_o = os.path.join(_sub_dir_o, _item_o2)
                        _dst_o = os.path.join(joint_orpo_output_dir, _item_o2)
                        if os.path.isdir(_src_o) and _item_o2.startswith("checkpoint-"):
                            if os.path.exists(_dst_o):
                                _shutil_ro.rmtree(_dst_o)
                            _shutil_ro.move(_src_o, _dst_o)
                    _shutil_ro.rmtree(os.path.join(joint_orpo_output_dir, JOINT_PREFIX))

                _checkpoints_o = sorted([
                    d for d in os.listdir(joint_orpo_output_dir)
                    if d.startswith("checkpoint-") and os.path.isdir(os.path.join(joint_orpo_output_dir, d))
                    and os.path.exists(os.path.join(joint_orpo_output_dir, d, "adapter_config.json"))
                ])
                if _checkpoints_o:
                    _resume_from_o = True
                    print(f"✅ [JOINT-ORPO] {len(_checkpoints_o)} checkpoint(s) ditemukan — resume!")
                else:
                    print("⚠️ [JOINT-ORPO] Tidak ada checkpoint valid — mulai dari awal.")
            except Exception as e:
                print(f"❌ [JOINT-ORPO] Gagal download checkpoint: {e}.")
                raise RuntimeError(
                    "❌ [JOINT-ORPO] Resume ditandai tersedia di HF tapi gagal di-download — "
                    "pipeline dihentikan (fail-hard; dagi dari awal tanpa resume berbahaya)."
                ) from e

        # ---- Train ----
        print("\n🚀 [JOINT-ORPO] Starting JOINT ORPO training...")
        joint_orpo_trainer.train(resume_from_checkpoint=_resume_from_o)

        # ---- Save & upload final adapter ----
        _final_path_o = os.path.join(joint_orpo_output_dir, "final_adapter")
        print(f"\n💾 [JOINT-ORPO] Saving final adapter ke {_final_path_o}...")
        joint_orpo_trainer.save_model(_final_path_o)
        processor.save_pretrained(_final_path_o)
        # Save chat_template.jinja for easy deployment (konsisten dengan final_adapter SFT)
        _chat_template_o = getattr(processor, 'chat_template', None) or getattr(getattr(processor, 'tokenizer', None), 'chat_template', None)
        if _chat_template_o:
            _jinja_path_o = os.path.join(_final_path_o, "chat_template.jinja")
            with open(_jinja_path_o, "w", encoding="utf-8") as _f:
                _f.write(_chat_template_o)
            print(f"   💾 Chat template saved to {_jinja_path_o}")
        try:
            from huggingface_hub import HfApi as _FinalApiO
            _final_api_o = _FinalApiO(token=os.environ.get("HF_TOKEN"))
            _final_api_o.create_repo(repo_id=UNIFIED_HF_REPO, repo_type="model", private=False, exist_ok=True)
            upload_folder_atomic(
                _final_api_o,
                UNIFIED_HF_REPO,
                _final_path_o,
                f"{JOINT_PREFIX}/orpo/final_adapter",
            )
            print("✅ [JOINT-ORPO] Final adapter ter-upload ke joint/orpo/final_adapter!")
        except Exception as e:
            print(f"❌ [JOINT-ORPO] Upload final adapter GAGAL: {e}")
            raise RuntimeError("❌ [JOINT-ORPO] Upload final adapter gagal — pipeline dihentikan (fail-hard).") from e
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ---
    # 📦 Final Merge & Quantize
    LoRA hasil ORPO di-merge **sekali** (menghindari degradasi double-merge v6)
    menjadi `final/merged_bf16` + `final/quantized_4bit` di unified repo.
    """)
    return


@app.cell
def _(
    CANGKOK_SUBFOLDER,
    FINAL_PREFIX,
    JOINT_PREFIX,
    LOAD_IN_4BIT,
    OUTPUT_DIR,
    UNIFIED_HF_REPO,
    final_done,
    model,
    os,
    processor,
    tokenizer,
):
    from huggingface_hub import HfApi as _MergeApi

    _token = os.environ.get("HF_TOKEN")
    _api = _MergeApi(token=_token)
    _files = _api.list_repo_files(UNIFIED_HF_REPO)

    _orpo_adapter_exists = any(
        f.startswith(f"{JOINT_PREFIX}/orpo/final_adapter/") for f in _files
    )
    _sft_adapter_exists = any(
        f.startswith(f"{JOINT_PREFIX}/sft/final_adapter/") for f in _files
    )
    # Fallback: kalau adapter ORPO belum ada (mis. RUN_ORPO=False), merge dari SFT.
    _adapter_stage = "orpo" if _orpo_adapter_exists else ("sft" if _sft_adapter_exists else None)

    _should_merge = (model is not None or _adapter_stage is not None) and not final_done
    final_upload_dir = os.path.join(OUTPUT_DIR, "hf_upload")

    if not _should_merge:
        print(
            "⏭️ [MERGE] Dilewati — "
            + (
                f"`{FINAL_PREFIX}/merged_bf16` sudah ada di repo."
                if final_done
                else "adapter ORPO/SFT belum ada & model tidak dimuat (training belum selesai)."
            )
        )
    else:
        import unsloth_zoo.saving_utils
        unsloth_zoo.saving_utils.assert_same_keys = lambda *args, **kwargs: None  # type: ignore

        # --- Workaround: unsloth_zoo `_infer_prefix_and_remap` UnboundLocalError ---
        # Versi unsloth_zoo yang terinstal tidak menginisialisasi `unmatched_keys = []`
        # sebelum cek `if unmatched_keys:` pertama. Saat SEMUA key LoRA langsung cocok,
        # variabel itu tidak pernah ter-assign -> UnboundLocalError. Diperbaiki di
        # upstream; di sini dipatch via wrapper + fallback reimplementation.
        _sz = unsloth_zoo.saving_utils
        if not getattr(_sz, "_unmatched_keys_patch_applied", False):
            from collections import defaultdict as _ddp
            _orig_infer = getattr(_sz, "_infer_prefix_and_remap", None)

            def _infer_prefix_and_remap_fixed(lora_weights, safetensor_keys):
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
                    if (k + ".weight") in sf_key_set or (k + ".linear.weight") in sf_key_set:
                        remapped[k] = v
                        continue
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
                if not changed and not unmatched_keys:
                    return None
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
                                f"⚠️ [patch] _infer_prefix_and_remap UnboundLocalError ({e}); "
                                "memakai fallback reimplementation."
                            )
                            return _infer_prefix_and_remap_fixed(lora_weights, safetensor_keys)
                        raise
                return _infer_prefix_and_remap_fixed(lora_weights, safetensor_keys)

            setattr(_sz, "_infer_prefix_and_remap", _patched_infer)
            setattr(_sz, "_unmatched_keys_patch_applied", True)
            print("✅ [patch] Workaround `_infer_prefix_and_remap` terpasang.")

        _model_to_merge = model
        _merge_tokenizer = tokenizer
        _merge_processor = processor

        if _model_to_merge is None:
            from unsloth import FastVisionModel as _FVMerge

            if _adapter_stage is None:
                raise RuntimeError("[MERGE] Tidak ada adapter (orpo/sft) untuk di-merge!")

            _orpo_path = os.path.join(OUTPUT_DIR, JOINT_PREFIX, _adapter_stage, "final_adapter")
            if not os.path.exists(_orpo_path):
                from huggingface_hub import snapshot_download as _merge_snap
                print(f"📥 [MERGE] Downloading {_adapter_stage}/final_adapter dari HF untuk merging...")
                _merge_snap(
                    repo_id=UNIFIED_HF_REPO,
                    local_dir=_orpo_path,
                    allow_patterns=[f"{JOINT_PREFIX}/{_adapter_stage}/final_adapter/**"],
                    token=_token,
                )
                _sub_path = os.path.join(_orpo_path, JOINT_PREFIX, _adapter_stage, "final_adapter")
                if os.path.exists(_sub_path):
                    import shutil as _shutil_m
                    for _mi in os.listdir(_sub_path):
                        _src_m = os.path.join(_sub_path, _mi)
                        _dst_m = os.path.join(_orpo_path, _mi)
                        if os.path.exists(_dst_m):
                            if os.path.isdir(_dst_m):
                                _shutil_m.rmtree(_dst_m)
                            else:
                                os.remove(_dst_m)
                        _shutil_m.move(_src_m, _dst_m)
                    _shutil_m.rmtree(os.path.join(_orpo_path, JOINT_PREFIX))

            print(f"📂 [MERGE] Loading model dari adapter {_adapter_stage}: {_orpo_path}")
            _model_to_merge, _merge_tokenizer = _FVMerge.from_pretrained(
                model_name=_orpo_path,
                load_in_4bit=LOAD_IN_4BIT,
                use_gradient_checkpointing="unsloth",
                token=_token,
            )
            from transformers import AutoProcessor as _MergeProc
            _merge_processor = _MergeProc.from_pretrained(
                UNIFIED_HF_REPO, subfolder=CANGKOK_SUBFOLDER, token=_token
            )
            from unsloth.chat_templates import get_chat_template as _gct
            _merge_tokenizer = _gct(_merge_tokenizer, chat_template="gemma-3")
            _merge_processor.chat_template = _merge_tokenizer.chat_template
            if hasattr(_merge_processor, "tokenizer"):
                _merge_processor.tokenizer.chat_template = _merge_tokenizer.chat_template

        merged_bf16_path = os.path.join(final_upload_dir, "merged_bf16")
        quantized_4bit_path = os.path.join(final_upload_dir, "quantized_4bit")

        print("[MERGE] Merging LoRA adapter → BF16 (merged_16bit)...")
        _model_to_merge.save_pretrained_merged(merged_bf16_path, _merge_tokenizer, save_method="merged_16bit")
        _merge_tokenizer.save_pretrained(merged_bf16_path)
        _merge_processor.save_pretrained(merged_bf16_path)
        print("✅ [MERGE] Model BF16 tersimpan.")

        print("\n[MERGE] Merging LoRA adapter → 4-bit NF4 (merged_4bit_forced)...")
        _model_to_merge.save_pretrained_merged(quantized_4bit_path, _merge_tokenizer, save_method="merged_4bit_forced")
        _merge_tokenizer.save_pretrained(quantized_4bit_path)
        _merge_processor.save_pretrained(quantized_4bit_path)
        print("✅ [MERGE] Model 4-bit NF4 tersimpan!")
    return (final_upload_dir,)


@app.cell
def _(FINAL_PREFIX, UNIFIED_HF_REPO, final_upload_dir, os, upload_folder_atomic):
    from huggingface_hub import HfApi as _UpFinalApi

    _has_merged = os.path.exists(os.path.join(final_upload_dir, "merged_bf16", "config.json"))
    if not _has_merged:
        print("⏭️ [UPLOAD] Tidak ada hasil merge lokal — upload final dilewati.")
    else:
        print(f"[UPLOAD] Mengunggah hasil merge ke {UNIFIED_HF_REPO}/{FINAL_PREFIX} (verified-atomic per subfolder)...")
        try:
            _final_up_api = _UpFinalApi(token=os.environ.get("HF_TOKEN"))
            _final_up_api.create_repo(repo_id=UNIFIED_HF_REPO, repo_type="model", private=False, exist_ok=True)

            for _sub in ("merged_bf16", "quantized_4bit"):
                _p = os.path.join(final_upload_dir, _sub)
                if os.path.exists(os.path.join(_p, "config.json")):
                    upload_folder_atomic(_final_up_api, UNIFIED_HF_REPO, _p, f"{FINAL_PREFIX}/{_sub}")
                else:
                    print(f"  ⚠️ [UPLOAD] Subfolder {_sub} tidak lengkap (config.json hilang) — dilewati.")

            print("✅ [UPLOAD] final/merged_bf16 & final/quantized_4bit ter-upload!")
        except Exception as e:
            print(f"❌ [UPLOAD] Terjadi kesalahan saat mengunggah: {e}")
            raise RuntimeError(f"❌ [UPLOAD] Gagal mengunggah hasil merge: {e} — pipeline dihentikan (fail-hard).") from e
    return


@app.cell(hide_code=True)
def _(UNIFIED_HF_REPO, mo):
    mo.md(f"""
    ---
    ### 💻 Deployment & Inference (Unified Repo)
    Repo: **`{UNIFIED_HF_REPO}`** (PUBLIC)

    ```
    steered/            → checkpoint Phase 0.5 (Task Vector Steering)
    cangkok/            → checkpoint Phase 1.5 (SigLIP+projector graft, base training)
    joint/sft/          → checkpoints + final_adapter SFT joint
    joint/orpo/         → checkpoints + final_adapter ORPO joint
    final/merged_bf16/  → 🏁 model akhir bfloat16 (~15 GB)
    final/quantized_4bit/ → model akhir NF4 (~5 GB)
    ```

    #### Load Model Final 4-bit:
    ```python
    from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, AutoProcessor

    model_id = "{UNIFIED_HF_REPO}"

    tokenizer = AutoTokenizer.from_pretrained(model_id, subfolder="final/quantized_4bit")
    processor = AutoProcessor.from_pretrained(model_id, subfolder="final/quantized_4bit")
    model = AutoModelForSeq2SeqLM.from_pretrained(
        model_id, subfolder="final/quantized_4bit", device_map="auto"
    )
    ```

    #### Load Model Final BF16:
    ```python
    import torch
    from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, AutoProcessor

    model_id = "{UNIFIED_HF_REPO}"

    tokenizer = AutoTokenizer.from_pretrained(model_id, subfolder="final/merged_bf16")
    processor = AutoProcessor.from_pretrained(model_id, subfolder="final/merged_bf16")
    model = AutoModelForSeq2SeqLM.from_pretrained(
        model_id, subfolder="final/merged_bf16",
        torch_dtype=torch.bfloat16, device_map="auto"
    )
    ```
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ---
    ## 🧪 Evaluasi Pasca-Training (berjalan hanya jika model masih di memori)
    """)
    return


@app.cell
def _(
    ALL_SUPPRESS_IDS,
    DATASET_TEXT_REPO,
    GEN_REPETITION_PENALTY,
    GEN_TEMPERATURE,
    GEN_TOP_P,
    SEED,
    TEXT_CHAT_CONFIG,
    TEXT_INDOQA_CONFIG,
    format_encoder_from_raw,
    load_dataset,
    model,
    processor,
    random,
    torch,
    traceback,
):
    if model is not None:
        print("\n" + "=" * 70)
        print("[EVAL] TEST 1: Inferensi multimodal (dummy image)")
        print("=" * 70)

        test_messages = [
            {"role": "system", "content": "Kamu adalah Gemma, asisten AI cerdas berbahasa Indonesia. Berikan respons yang akurat, ramah, dan terstruktur."},
            {"role": "user", "content": [
                {"type": "image"},
                {"type": "text", "text": "Halo Gemma, boleh tolong jelaskan isi gambar ini secara singkat?"}
            ]}
        ]

        from PIL import Image as PILImageEval
        dummy_img = PILImageEval.new("RGB", (224, 224), color="blue")

        try:
            prompt = processor.apply_chat_template(test_messages, tokenize=False, add_generation_prompt=True)
            inputs = processor(text=prompt, images=dummy_img, return_tensors="pt")

            device = next(model.parameters()).device
            inputs = {k: v.to(device) for k, v in inputs.items()}

            with torch.no_grad():
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=256,
                    do_sample=True,
                    temperature=GEN_TEMPERATURE, top_p=GEN_TOP_P, use_cache=True
                )
            response = processor.tokenizer.decode(outputs[0], skip_special_tokens=True)
            print(f"User: [📷 Image] Halo Gemma, boleh tolong jelaskan isi gambar ini secara singkat?")
            print(f"Assistant:\n{response}")
        except Exception as e:
            print(f"Gagal inferensi multimodal: {e}")

        print("\n" + "=" * 70)
        print("[EVAL] TEST 2: Pemeliharaan chat teks (20 kueri pertama dari validation)")
        print("=" * 70)

        try:
            val_chat_ds = load_dataset(DATASET_TEXT_REPO, TEXT_CHAT_CONFIG, split="validation")
            val_indoqa_ds = load_dataset(DATASET_TEXT_REPO, TEXT_INDOQA_CONFIG, split="validation")

            _val_rows = []
            for _row in [dict(r) for r in val_chat_ds] + [dict(r) for r in val_indoqa_ds]:
                _val_rows.append({
                    "prompt_text": format_encoder_from_raw(_row.get("input", "")),
                    "target_text": _row.get("target", "").strip(),
                })
            random.seed(SEED)
            random.shuffle(_val_rows)
            eval_samples = _val_rows[:20]
            print(f"[EVAL] {len(eval_samples)} sampel validasi teks dimuat.")

            device = next(model.parameters()).device
            _eot_id = processor.tokenizer.convert_tokens_to_ids("<end_of_turn>")
            _eos_id = processor.tokenizer.eos_token_id or 1
            _stop_ids = list({_eot_id, _eos_id})
            bad_words_ids = [[id_] for id_ in ALL_SUPPRESS_IDS if id_ < model.config.vocab_size]
            pad_id = processor.tokenizer.pad_token_id or _eos_id

            for idx, sample in enumerate(eval_samples):
                inputs = processor(text=sample["prompt_text"], return_tensors="pt").to(device)
                with torch.no_grad():
                    outputs_text = model.generate(
                        **inputs,
                        max_new_tokens=256,
                        do_sample=True,
                        temperature=GEN_TEMPERATURE,
                        top_p=GEN_TOP_P,
                        repetition_penalty=GEN_REPETITION_PENALTY,
                        eos_token_id=_stop_ids,
                        pad_token_id=pad_id,
                        bad_words_ids=bad_words_ids,
                        use_cache=True
                    )
                raw_response = processor.tokenizer.decode(outputs_text[0], skip_special_tokens=True)
                query = sample["prompt_text"].strip()
                if raw_response.startswith(query):
                    raw_response = raw_response[len(query):].strip()
                response = raw_response.strip()

                print(f"\n[Sampel {idx+1}/{len(eval_samples)}]")
                print(f"  Q: {query[:200]}...")
                print(f"  Target: {sample['target_text'][:150]}...")
                print(f"  Model: {response[:300]}...")

        except Exception as e:
            print(f"Gagal inferensi teks: {e}")
            traceback.print_exc()

        print("=" * 70)
    else:
        print("⏭️ [EVAL] Model tidak di memori (training belum jalan di sesi ini) — eval dilewati.")
    return


if __name__ == "__main__":
    app.run()
