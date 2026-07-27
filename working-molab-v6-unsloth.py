# /// script
# requires-python = ">=3.10"
# dependencies = [
#     "accelerate",
#     "absl-py",
#     "bitsandbytes",
#     "datasets",
#     "evaluate",
#     "huggingface-hub",
#     "numpy",
#     "peft==0.19.1",
#     "pytorch-optimizer",
#     "rouge-score",
#     "sacrebleu",
#     "bert_score",
#     "nltk",
#     "torch",
#     "trl",
#     "transformers==5.12.1",
#     "unsloth_zoo @ git+https://github.com/daruoktab/unsloth-zoo.git",
#     "unsloth @ git+https://github.com/daruoktab/unsloth.git",
# ]
# ///

import marimo

__generated_with = "0.23.11"
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
    ✅ **STATUS: PIPELINE READY TO RUN.**
    Dataset untuk Task Prefix menggunakan unused tokens telah selesai di-generate dan siap digunakan untuk pelatihan V6.

    # Multi-task SFT + ORPO Training: T5Gemma-2 Cloud Pipeline (Version 6 - Unsloth)
    =====================================================================
    Notebook ini melatih model **T5Gemma-2 4B-4B** secara berurutan:
    1. **Phase 1: SFT** — Supervised Fine-Tuning dengan LoRA berbasis Unsloth
    2. **Phase 2: ORPO** — Odds Ratio Preference Optimization di atas hasil SFT

    **Fitur utama:**
    - Auto-detect progress dari HF Hub — otomatis lanjut dari checkpoint terakhir
    - Upload checkpoint ke HF segera setelah disimpan (tahan kernel crash)
    - 1 repo HF dengan subfolder `sft/` dan `orpo/`
    - Logit masking untuk menekan unused & vision tokens
    - Akselerasi QLoRA (4-bit) dan LoRA (16-bit) via Unsloth
    """)
    return


@app.cell
def _():
    import os
    import re
    import torch
    import random
    import datetime
    import gc
    import matplotlib.pyplot as plt
    try:
        import torch._inductor.config
    except ImportError:
        pass
    from unsloth import FastLanguageModel
    from datasets import Dataset, load_dataset
    from transformers import (
        AutoTokenizer,
        Seq2SeqTrainer,
        Seq2SeqTrainingArguments,
        DataCollatorForSeq2Seq,
        PreTrainedTokenizerFast,
        TrainerCallback,
        TrainerControl,
        TrainerState,
        TrainingArguments,
        get_scheduler,
    )
    from typing import cast, Any

    # Gunakan inline backend untuk matplotlib di Jupyter Notebook
    import numpy as np

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
    return (
        Any,
        AutoTokenizer,
        DataCollatorForSeq2Seq,
        Dataset,
        PreTrainedTokenizerFast,
        Seq2SeqTrainer,
        Seq2SeqTrainingArguments,
        TrainerCallback,
        TrainerControl,
        TrainerState,
        TrainingArguments,
        FastLanguageModel,
        bleu_metric,
        cast,
        datetime,
        gc,
        get_scheduler,
        load_dataset,
        np,
        os,
        plt,
        random,
        re,
        rouge_metric,
        exact_match_metric,
        bertscore_metric,
        meteor_metric,
        torch,
    )


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


# =====================================================================
# KONFIGURASI HYPERPARAMETER (TERPUSAT & MUDAH DIUBAH)
# =====================================================================
@app.cell
def _(torch):
    # MODEL CONFIG
    MODEL_NAME = "google/t5gemma-2-4b-4b"
    LOAD_IN_4BIT = True  # True = QLoRA (hemat VRAM), False = BF16 (perlu VRAM besar)
    OUTPUT_DIR = "results/t5gemma2"  # Base dir — subfolder sft/ dan orpo/ otomatis

    # HUGGING FACE HUB CONFIG (1 repo untuk semua)
    HF_REPO_ID = "daruokta/t5gemma2-indonesia-chat-formatted"  # Dataset source
    HF_CHECKPOINT_REPO = "daruokta/t5gemma-2-4b-4b-instruct-chat-indo-v4-unsloth"  # Training artifacts

    # ORPO CONFIG
    ORPO_BETA = 0.1

    # Dataset Subsets (Configs)
    CHAT_CONFIG = "chat_sft"
    INDOQA_CONFIG = "indoqa_sft"
    ORPO_CONFIG = "chat_orpo"

    # SAMPLE SIZES (Set ke 0 untuk mengambil seluruh data)
    SAMPLE_TRAIN_CHAT = 0
    SAMPLE_TRAIN_INDOQA = 0
    SAMPLE_TRAIN_ORPO = 0
    SAMPLE_VAL_CHAT = 0
    SAMPLE_VAL_INDOQA = 0

    # GENERATION EVALUATION CONFIG
    SAMPLE_EVAL_GENERATION = 100
    EVAL_EVERY_N_STEPS = 200

    # SYSTEM PROMPT FALLBACK
    SYSTEM_PROMPT = (
        "Kamu adalah asisten AI yang helpful, santai, dan ramah. "
        "Gunakan Bahasa Indonesia sebagai bahasa utama."
    )

    # BASIC TRAINING SPECS
    MAX_SOURCE_LENGTH = 16384
    MAX_TARGET_LENGTH = 2048
    NUM_EPOCHS_SFT = 4
    NUM_EPOCHS_ORPO = 2
    LEARNING_RATE = 1e-5

    # BATCH SIZE & ACCUMULATION
    PER_DEVICE_TRAIN_BATCH_SIZE = 2
    PER_DEVICE_EVAL_BATCH_SIZE = 8
    GRADIENT_ACCUMULATION_STEPS = 64
    EVAL_ACCUMULATION_STEPS = None

    # LoRA CONFIG SPECS
    LORA_RANK = 256
    LORA_ALPHA = 512
    LORA_DROPOUT = 0.2
    LORA_TARGET_MODULES = [
        "q_proj",
        "k_proj",
        "v_proj",
        "o_proj",
        "gate_proj",
        "up_proj",
        "down_proj",
    ]

    # ADVANCED TRAINING ARGUMENTS
    WARMUP_STEPS = 200
    WEIGHT_DECAY = 0.1
    LR_SCHEDULER_TYPE = "cosine"
    LOGGING_STEPS = 100
    SAVE_TOTAL_LIMIT = 2  # Lokal saja — di HF semua checkpoint tetap ada
    OPTIM = "paged_adamw_8bit"
    LABEL_SMOOTHING_FACTOR = 0.1
    NEFTUNE_NOISE_ALPHA = 5.0

    # HARDWARE & CONTROL SPECS
    GRADIENT_CHECKPOINTING = True
    FP16 = False
    BF16 = torch.cuda.is_available()
    PREDICT_WITH_GENERATE = True
    EARLY_STOPPING_PATIENCE = 8

    # EVALUATION GENERATION BEHAVIOR CONFIG
    GEN_TEMPERATURE = 0.7
    GEN_TOP_P = 0.9
    GEN_REPETITION_PENALTY = 1.2

    # Token IDs yang harus di-suppress (unused + vision)
    # Pengecualian: <unused1> sampai <unused6> (ID 7 hingga 12) digunakan untuk Task Prefix
    SUPPRESS_BLOCK1 = [6] + list(range(13, 105))
    SUPPRESS_BLOCK2 = list(range(256002, 262144))
    SUPPRESS_VISION = [255999, 256000, 256001]
    ALL_SUPPRESS_IDS = set(SUPPRESS_BLOCK1 + SUPPRESS_BLOCK2 + SUPPRESS_VISION)
    return (
        ALL_SUPPRESS_IDS,
        BF16,
        CHAT_CONFIG,
        EARLY_STOPPING_PATIENCE,
        EVAL_ACCUMULATION_STEPS,
        EVAL_EVERY_N_STEPS,
        FP16,
        GEN_REPETITION_PENALTY,
        GEN_TEMPERATURE,
        GEN_TOP_P,
        GRADIENT_ACCUMULATION_STEPS,
        GRADIENT_CHECKPOINTING,
        HF_CHECKPOINT_REPO,
        HF_REPO_ID,
        INDOQA_CONFIG,
        LABEL_SMOOTHING_FACTOR,
        LEARNING_RATE,
        LOAD_IN_4BIT,
        LOGGING_STEPS,
        LORA_ALPHA,
        LORA_DROPOUT,
        LORA_RANK,
        LORA_TARGET_MODULES,
        LR_SCHEDULER_TYPE,
        MAX_SOURCE_LENGTH,
        MAX_TARGET_LENGTH,
        MODEL_NAME,
        NEFTUNE_NOISE_ALPHA,
        NUM_EPOCHS_ORPO,
        NUM_EPOCHS_SFT,
        OPTIM,
        ORPO_BETA,
        ORPO_CONFIG,
        OUTPUT_DIR,
        PER_DEVICE_EVAL_BATCH_SIZE,
        PER_DEVICE_TRAIN_BATCH_SIZE,
        PREDICT_WITH_GENERATE,
        SAMPLE_EVAL_GENERATION,
        SAMPLE_TRAIN_CHAT,
        SAMPLE_TRAIN_INDOQA,
        SAMPLE_TRAIN_ORPO,
        SAMPLE_VAL_CHAT,
        SAMPLE_VAL_INDOQA,
        SAVE_TOTAL_LIMIT,
        SYSTEM_PROMPT,
        WARMUP_STEPS,
        WEIGHT_DECAY,
    )


# =====================================================================
# AUTO-DETECT PIPELINE STAGE DARI HF HUB
# =====================================================================
@app.cell
def _(HF_CHECKPOINT_REPO, mo, os):
    from huggingface_hub import HfApi as _StageDetectApi

    _hf_token = os.environ.get("HF_TOKEN")
    _api = _StageDetectApi(token=_hf_token)

    # Default
    current_stage = "sft"
    resume_checkpoint = None

    try:
        if _api.repo_exists(repo_id=HF_CHECKPOINT_REPO):
            _repo_files = _api.list_repo_files(HF_CHECKPOINT_REPO)

            # Cek apakah ORPO sudah selesai
            if any(f.startswith("orpo/final_adapter/") for f in _repo_files):
                current_stage = "done"
                print("📍 Pipeline stage: DONE — Semua training selesai!")

            # Cek apakah SFT sudah selesai → lanjut ORPO
            elif any(f.startswith("sft/final_adapter/") for f in _repo_files):
                current_stage = "orpo"
                # Ada checkpoint ORPO untuk resume?
                _orpo_ckpts = sorted([
                    f for f in _repo_files
                    if f.startswith("orpo/checkpoint-") and "/" in f[len("orpo/checkpoint-"):]
                ])
                if _orpo_ckpts:
                    resume_checkpoint = True
                    print(f"📍 Pipeline stage: ORPO (resume dari checkpoint)")
                else:
                    print("📍 Pipeline stage: ORPO (mulai dari awal, load SFT adapter)")

            # SFT belum selesai
            else:
                current_stage = "sft"
                _sft_ckpts = sorted([
                    f for f in _repo_files
                    if f.startswith("sft/checkpoint-") and "/" in f[len("sft/checkpoint-"):]
                ])
                if _sft_ckpts:
                    resume_checkpoint = True
                    print(f"📍 Pipeline stage: SFT (resume dari checkpoint)")
                else:
                    print("📍 Pipeline stage: SFT (mulai dari awal)")
        else:
            print(f"📍 Repo '{HF_CHECKPOINT_REPO}' belum ada. Mulai SFT dari awal.")
            # Buat repo
            _api.create_repo(repo_id=HF_CHECKPOINT_REPO, repo_type="model", private=True, exist_ok=True)
    except Exception as e:
        print(f"⚠️ Gagal mendeteksi pipeline stage: {e}. Mulai SFT dari awal.")

    mo.md(f"**📍 Current Stage: `{current_stage}`** | Resume: `{resume_checkpoint}`")
    return current_stage, resume_checkpoint


# =====================================================================
# UTILITY FUNCTIONS (format, load, logit mask)
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


@app.cell
def _(Any, torch):
    # ==========================================
    # NON-DESTRUCTIVE LOGIT MASKING
    # ==========================================
    def apply_logit_mask(model: Any, suppress_ids: set[int]) -> None:
        vocab_size = model.config.vocab_size
        suppress_list = [i for i in suppress_ids if i < vocab_size]

        mask = torch.zeros(vocab_size, dtype=torch.bfloat16)
        mask[suppress_list] = -10000.0

        def forward_hook(module, inputs, outputs):
            if isinstance(outputs, torch.Tensor):
                return outputs + mask.to(outputs.device)
            elif hasattr(outputs, "logits"):
                outputs.logits = outputs.logits + mask.to(outputs.logits.device)
                return outputs
            elif (
                isinstance(outputs, tuple)
                and len(outputs) > 0
                and isinstance(outputs[0], torch.Tensor)
            ):
                logits = outputs[0]
                outputs = (logits + mask.to(logits.device),) + outputs[1:]
                return outputs
            return outputs

        target_module = None
        if hasattr(model, "lm_head"):
            target_module = model.lm_head
        elif hasattr(model, "base_model") and hasattr(model.base_model, "lm_head"):
            target_module = model.base_model.lm_head
        elif (
            hasattr(model, "base_model")
            and hasattr(model.base_model, "model")
            and hasattr(model.base_model.model, "lm_head")
        ):
            target_module = model.base_model.model.lm_head

        if target_module is not None:
            target_module.register_forward_hook(forward_hook)
            print(
                f"  ✅ Logit masking registered pada final linear layer (lm_head) untuk {len(suppress_list)} suppressed tokens."
            )
        else:
            model.register_forward_hook(forward_hook)
            print(
                f"  ✅ Logit masking registered pada top-level model (fallback) untuk {len(suppress_list)} suppressed tokens."
            )

    return (apply_logit_mask,)


# =====================================================================
# CALLBACKS: Training Plot, Sample Generation, Hub Upload
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
    class TrainingPlotCallback(TrainerCallback):
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

    return (TrainingPlotCallback,)


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
    class SampleGenerationCallback(TrainerCallback):
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

    return (SampleGenerationCallback,)


@app.cell
def _(
    Any,
    TrainerCallback,
    TrainerControl,
    TrainerState,
    TrainingArguments,
    os,
):
    class HubUploadCallback(TrainerCallback):
        """Upload setiap checkpoint ke HF Hub segera setelah disimpan."""

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

    return (HubUploadCallback,)


# =====================================================================
# DATA PROCESSING
# =====================================================================
@app.cell
def _(
    MAX_SOURCE_LENGTH,
    MAX_TARGET_LENGTH,
    PreTrainedTokenizerFast,
    format_encoder_from_raw,
):
    def process_sft_rows(samples, tokenizer: PreTrainedTokenizerFast, is_chat=True):
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
                        len(inp_ids) <= MAX_SOURCE_LENGTH
                        and len(tgt_ids) <= MAX_TARGET_LENGTH
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
                    len(inp_ids) <= MAX_SOURCE_LENGTH
                    and len(tgt_ids) <= MAX_TARGET_LENGTH
                ):
                    rows.append({"input_ids": inp_ids, "labels": tgt_ids})
        return rows

    def process_orpo_rows(samples, tokenizer: PreTrainedTokenizerFast):
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

            if len(inp_ids) <= MAX_SOURCE_LENGTH and len(chosen_ids) <= MAX_TARGET_LENGTH and len(rejected_ids) <= MAX_TARGET_LENGTH:
                rows.append({
                    "input_ids": inp_ids,
                    "chosen_labels": chosen_ids,
                    "rejected_labels": rejected_ids
                })
        return rows

    return (
        process_orpo_rows,
        process_sft_rows,
    )


# =====================================================================
# LOAD DATASET BERDASARKAN STAGE
# =====================================================================
@app.cell
def _(
    AutoTokenizer,
    CHAT_CONFIG,
    Dataset,
    HF_REPO_ID,
    INDOQA_CONFIG,
    MODEL_NAME,
    ORPO_CONFIG,
    OUTPUT_DIR,
    PreTrainedTokenizerFast,
    SAMPLE_EVAL_GENERATION,
    SAMPLE_TRAIN_CHAT,
    SAMPLE_TRAIN_INDOQA,
    SAMPLE_TRAIN_ORPO,
    SAMPLE_VAL_CHAT,
    SAMPLE_VAL_INDOQA,
    current_stage,
    load_hf_samples,
    os,
    process_orpo_rows,
    process_sft_rows,
    random,
):
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print(f"Loading Tokenizer from {MODEL_NAME}...")
    _token = os.environ.get("HF_TOKEN")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, token=_token, trust_remote_code=True)
    assert isinstance(tokenizer, PreTrainedTokenizerFast), (
        "Tokenizer harus PreTrainedTokenizerFast"
    )

    # Selalu load validation data (dipakai di kedua stage untuk eval)
    val_chat_samples = load_hf_samples(
        HF_REPO_ID, CHAT_CONFIG, "validation", SAMPLE_VAL_CHAT
    )
    val_indoqa_samples = load_hf_samples(
        HF_REPO_ID, INDOQA_CONFIG, "validation", SAMPLE_VAL_INDOQA
    )
    val_rows = process_sft_rows(
        val_chat_samples, tokenizer, is_chat=True
    ) + process_sft_rows(val_indoqa_samples, tokenizer, is_chat=False)

    # Load training data sesuai stage aktif
    if current_stage == "sft":
        print("\n📦 Loading SFT training data...")
        train_chat_samples = load_hf_samples(
            HF_REPO_ID, CHAT_CONFIG, "train", SAMPLE_TRAIN_CHAT
        )
        train_indoqa_samples = load_hf_samples(
            HF_REPO_ID, INDOQA_CONFIG, "train", SAMPLE_TRAIN_INDOQA
        )
        train_rows = process_sft_rows(
            train_chat_samples, tokenizer, is_chat=True
        ) + process_sft_rows(train_indoqa_samples, tokenizer, is_chat=False)
        is_orpo_training = False
    elif current_stage == "orpo":
        print("\n📦 Loading ORPO training data...")
        train_orpo_samples = load_hf_samples(HF_REPO_ID, ORPO_CONFIG, "train", SAMPLE_TRAIN_ORPO)
        train_rows = process_orpo_rows(train_orpo_samples, tokenizer)
        is_orpo_training = True
    else:
        # current_stage == "done"
        train_rows = []
        is_orpo_training = False

    random.seed(42)
    random.shuffle(train_rows)
    random.shuffle(val_rows)

    print(f"\nTotal Training rows: {len(train_rows)}")
    print(f"Total Validation rows: {len(val_rows)}")

    train_ds = Dataset.from_list(train_rows) if train_rows else None
    eval_ds = Dataset.from_list(val_rows)

    n_eval_gen = min(len(val_rows), SAMPLE_EVAL_GENERATION)
    # Eval generation selalu pakai SFT-format rows (punya input_ids + labels)
    _sft_val_rows = [r for r in val_rows if "labels" in r]
    eval_generation_samples = _sft_val_rows[:n_eval_gen]
    print(
        f"Mengambil {len(eval_generation_samples)} sampel validasi untuk pencatatan evaluasi kualitatif."
    )
    return eval_ds, eval_generation_samples, is_orpo_training, tokenizer, train_ds


# =====================================================================
# LOAD MODEL BERDASARKAN STAGE
# =====================================================================
@app.cell
def _(
    ALL_SUPPRESS_IDS,
    FastLanguageModel,
    HF_CHECKPOINT_REPO,
    LORA_ALPHA,
    LORA_DROPOUT,
    LORA_RANK,
    LORA_TARGET_MODULES,
    MODEL_NAME,
    MAX_SOURCE_LENGTH,
    LOAD_IN_4BIT,
    OUTPUT_DIR,
    apply_logit_mask,
    current_stage,
    tokenizer,
    torch,
    gc,
    os,
):
    # 1. Reset Cuda Cache
    gc.collect()
    torch.cuda.empty_cache()

    if current_stage == "done":
        print("✅ Semua training sudah selesai! Skipping model load.")
        model = None
    elif current_stage == "orpo":
        # ORPO: Load SFT adapter (local fallback ke HF)
        _local_sft_path = os.path.join(OUTPUT_DIR, "sft", "final_adapter")
        _model_path = None

        if os.path.exists(_local_sft_path) and os.listdir(_local_sft_path):
            print(f"\n📂 Loading SFT adapter dari local: {_local_sft_path}")
            _model_path = _local_sft_path
        else:
            # Download dari HF
            print(f"\n📥 SFT adapter tidak ditemukan di lokal. Download dari HF...")
            from huggingface_hub import snapshot_download as _snap_dl
            _hf_sft_path = _snap_dl(
                repo_id=HF_CHECKPOINT_REPO,
                local_dir=_local_sft_path,
                allow_patterns=["sft/final_adapter/**"],
                token=os.environ.get("HF_TOKEN"),
            )
            # snapshot_download puts files in local_dir matching repo structure
            _downloaded_path = os.path.join(_hf_sft_path, "sft", "final_adapter")
            if os.path.exists(_downloaded_path):
                _model_path = _downloaded_path
            else:
                _model_path = _local_sft_path

        print(f"Loading ORPO base model from SFT adapter: {_model_path}")
        model, _tokenizer_unsloth = FastLanguageModel.from_pretrained(
            model_name=_model_path,
            max_seq_length=MAX_SOURCE_LENGTH,
            load_in_4bit=LOAD_IN_4BIT,
            trust_remote_code=True,
            token=os.environ.get("HF_TOKEN"),
        )
    else:
        # SFT: Load base model
        print(f"\nLoading base model from {MODEL_NAME} using Unsloth...")
        model, _tokenizer_unsloth = FastLanguageModel.from_pretrained(
            model_name=MODEL_NAME,
            max_seq_length=MAX_SOURCE_LENGTH,
            load_in_4bit=LOAD_IN_4BIT,
            trust_remote_code=True,
            token=os.environ.get("HF_TOKEN"),
        )

    if model is not None:
        # Reset max_length to silence warning
        model.config.max_length = None
        if hasattr(model, "generation_config") and model.generation_config is not None:
            model.generation_config.max_length = None

        if getattr(model.config, "decoder_start_token_id", None) is None:
            model.config.decoder_start_token_id = tokenizer.bos_token_id
            print(f"  Set decoder_start_token_id = {model.config.decoder_start_token_id}")

        if tokenizer.pad_token is None:
            tokenizer.add_special_tokens({"pad_token": tokenizer.eos_token})
            model.resize_token_embeddings(len(tokenizer))

        # Logit Masking
        print(f"\nApplying logit mask for {len(ALL_SUPPRESS_IDS)} tokens...")
        apply_logit_mask(model, ALL_SUPPRESS_IDS)

        # LoRA Config
        if hasattr(model, "peft_config"):
            print("Model is already a PEFT model (loaded SFT adapter). Skipping get_peft_model.")
        else:
            print("Applying LoRA using Unsloth...")
            model = FastLanguageModel.get_peft_model(
                model,
                r=LORA_RANK,
                lora_alpha=LORA_ALPHA,
                target_modules=LORA_TARGET_MODULES,
                lora_dropout=LORA_DROPOUT,
                bias="none",
                use_gradient_checkpointing="unsloth",
                random_state=3407,
            )

        getattr(FastLanguageModel, "for_training")(model)
        model.config.use_cache = False

        # Safety wrapper
        if hasattr(model, "prepare_decoder_input_ids_from_labels"):
            orig_fn = model.prepare_decoder_input_ids_from_labels
            def compatible_prepare(labels=None, input_ids=None, *args, **kwargs):
                target_tensor = labels if labels is not None else input_ids
                return orig_fn(target_tensor, *args, **kwargs)
            model.prepare_decoder_input_ids_from_labels = compatible_prepare

        model.print_trainable_parameters()

    return (model,)


# =====================================================================
# CUSTOM OPTIMIZER
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
                    step_update = (
                        m / bias_correction1 + 0.1 * n / bias_correction3
                    ) / denom

                    p.data.add_(step_update, alpha=-lr)
            return loss

    return (GrokAdEMAMix,)


# =====================================================================
# TRAINING CELL — Sequential SFT → ORPO
# =====================================================================
@app.cell
def _(
    ALL_SUPPRESS_IDS,
    Any,
    BF16,
    DataCollatorForSeq2Seq,
    EARLY_STOPPING_PATIENCE,
    EVAL_ACCUMULATION_STEPS,
    EVAL_EVERY_N_STEPS,
    FP16,
    GEN_REPETITION_PENALTY,
    GEN_TEMPERATURE,
    GEN_TOP_P,
    GRADIENT_ACCUMULATION_STEPS,
    GRADIENT_CHECKPOINTING,
    GrokAdEMAMix,
    HF_CHECKPOINT_REPO,
    HubUploadCallback,
    LABEL_SMOOTHING_FACTOR,
    LEARNING_RATE,
    LOGGING_STEPS,
    LR_SCHEDULER_TYPE,
    MAX_TARGET_LENGTH,
    NEFTUNE_NOISE_ALPHA,
    NUM_EPOCHS_ORPO,
    NUM_EPOCHS_SFT,
    OPTIM,
    ORPO_BETA,
    OUTPUT_DIR,
    PER_DEVICE_EVAL_BATCH_SIZE,
    PER_DEVICE_TRAIN_BATCH_SIZE,
    PREDICT_WITH_GENERATE,
    PreTrainedTokenizerFast,
    SAVE_TOTAL_LIMIT,
    SampleGenerationCallback,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
    TrainingPlotCallback,
    WARMUP_STEPS,
    WEIGHT_DECAY,
    bleu_metric,
    cast,
    current_stage,
    eval_ds,
    eval_generation_samples,
    get_scheduler,
    is_orpo_training,
    model,
    mo,
    np,
    os,
    resume_checkpoint,
    rouge_metric,
    exact_match_metric,
    bertscore_metric,
    meteor_metric,
    tokenizer,
    torch,
    train_ds,
    gc,
):
    # Skip jika sudah selesai atau tidak ada data
    mo.stop(
        current_stage == "done" or train_ds is None,
        mo.md("✅ **Training sudah selesai atau tidak ada data training.** Lanjut ke merge & upload."),
    )

    # 1. Bersihkan sisa memori
    gc.collect()
    torch.cuda.empty_cache()
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

    # Tentukan output dir berdasarkan stage
    active_output_dir = os.path.join(OUTPUT_DIR, current_stage)
    os.makedirs(active_output_dir, exist_ok=True)

    # === CUSTOM LABEL SMOOTHER ===
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

    # === CUSTOM TRAINER ===
    class CustomSeq2SeqTrainer(Seq2SeqTrainer):
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
    class ORPODataCollatorForSeq2Seq(DataCollatorForSeq2Seq):
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

    import torch.nn.functional as F

    # === ORPO TRAINER ===
    class CustomORPOTrainer(CustomSeq2SeqTrainer):
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
    def compute_metrics(eval_preds):
        metrics = {}
        if rouge_metric is None and bleu_metric is None:
            return metrics
        preds, labels = eval_preds
        if isinstance(preds, tuple):
            preds = preds[0]
        tok = cast(PreTrainedTokenizerFast, tokenizer)

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

    def preprocess_logits_for_metrics(logits, labels):
        if isinstance(logits, tuple):
            logits = logits[0]
        return logits.argmax(dim=-1)

    # === BUILD TRAINER ===
    bad_words_ids = [
        [id_] for id_ in ALL_SUPPRESS_IDS if id_ < cast(Any, model).config.vocab_size
    ]

    _hf_token = os.environ.get("HF_TOKEN")

    plot_callback = TrainingPlotCallback(output_dir=active_output_dir)
    sample_callback = SampleGenerationCallback(
        tokenizer=tokenizer,
        eval_samples=eval_generation_samples,
        output_dir=active_output_dir,
        eval_every_n_steps=EVAL_EVERY_N_STEPS,
        temperature=GEN_TEMPERATURE,
        top_p=GEN_TOP_P,
        repetition_penalty=GEN_REPETITION_PENALTY,
        bad_words_ids=bad_words_ids,
    )
    hub_callback = HubUploadCallback(
        repo_id=HF_CHECKPOINT_REPO,
        stage=current_stage,
        token=_hf_token,
        output_dir=active_output_dir,
    )

    if is_orpo_training:
        data_collator = ORPODataCollatorForSeq2Seq(tokenizer=tokenizer, model=model, padding=True)
    else:
        data_collator = DataCollatorForSeq2Seq(tokenizer=tokenizer, model=model, padding=True)

    training_args = Seq2SeqTrainingArguments(
        output_dir=active_output_dir,
        per_device_train_batch_size=PER_DEVICE_TRAIN_BATCH_SIZE,
        per_device_eval_batch_size=PER_DEVICE_EVAL_BATCH_SIZE,
        gradient_accumulation_steps=GRADIENT_ACCUMULATION_STEPS,
        eval_accumulation_steps=EVAL_ACCUMULATION_STEPS,
        learning_rate=LEARNING_RATE,
        num_train_epochs=NUM_EPOCHS_ORPO if is_orpo_training else NUM_EPOCHS_SFT,
        warmup_steps=WARMUP_STEPS,
        weight_decay=WEIGHT_DECAY,
        lr_scheduler_type=LR_SCHEDULER_TYPE,
        predict_with_generate=PREDICT_WITH_GENERATE,
        logging_steps=LOGGING_STEPS,
        save_strategy="steps",
        save_steps=EVAL_EVERY_N_STEPS,
        save_total_limit=SAVE_TOTAL_LIMIT,
        eval_strategy="steps",
        eval_steps=EVAL_EVERY_N_STEPS,
        optim=OPTIM,
        label_smoothing_factor=LABEL_SMOOTHING_FACTOR if not is_orpo_training else 0.0,
        neftune_noise_alpha=NEFTUNE_NOISE_ALPHA,
        report_to="none",
        fp16=FP16,
        bf16=BF16,
        gradient_checkpointing=GRADIENT_CHECKPOINTING,
        generation_max_length=MAX_TARGET_LENGTH,
        push_to_hub=False,  # Kita handle manual via HubUploadCallback
        remove_unused_columns=False if is_orpo_training else True,
    )

    optimizer = GrokAdEMAMix(
        model.parameters(),
        lr=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY,
        grok_alpha=2.0,
        grok_lamb=0.98,
    )

    num_update_steps_per_epoch = max(
        1, len(train_ds) // (PER_DEVICE_TRAIN_BATCH_SIZE * GRADIENT_ACCUMULATION_STEPS)
    )
    max_steps = num_update_steps_per_epoch * (NUM_EPOCHS_ORPO if is_orpo_training else NUM_EPOCHS_SFT)

    lr_scheduler = get_scheduler(
        name=LR_SCHEDULER_TYPE,
        optimizer=optimizer,
        num_warmup_steps=WARMUP_STEPS,
        num_training_steps=max_steps,
    )

    trainer_class = CustomORPOTrainer if is_orpo_training else CustomSeq2SeqTrainer
    trainer_kwargs = {
        "model": model,
        "args": training_args,
        "train_dataset": cast(Any, train_ds),
        "eval_dataset": cast(Any, eval_ds),
        "data_collator": data_collator,
        "compute_metrics": compute_metrics,
        "preprocess_logits_for_metrics": None if PREDICT_WITH_GENERATE else preprocess_logits_for_metrics,
        "optimizers": (optimizer, lr_scheduler),
        "callbacks": [plot_callback, sample_callback, hub_callback],
    }
    if is_orpo_training:
        trainer_kwargs["beta"] = ORPO_BETA

    trainer = trainer_class(**trainer_kwargs)

    # === RESUME FROM HF CHECKPOINT ===
    _resume_from = None
    if resume_checkpoint:
        try:
            from huggingface_hub import snapshot_download as _resume_snap
            from huggingface_hub import HfApi as _ResumeApi
            
            _api = _ResumeApi(token=_hf_token)
            _files = _api.list_repo_files(repo_id=HF_CHECKPOINT_REPO)
            
            # Cari checkpoint terbaru biar gak download semuanya dan bikin storage penuh
            _ckpts = list(set([f.split('/')[1] for f in _files if f.startswith(f"{current_stage}/checkpoint-")]))
            if _ckpts:
                _ckpts.sort(key=lambda x: int(x.split('-')[1]))
                _latest_ckpt = _ckpts[-1]
            else:
                _latest_ckpt = "checkpoint-*"
                
            print(f"\n📥 Downloading {_latest_ckpt} ({current_stage}) dari HF untuk resume...")
            _resume_snap(
                repo_id=HF_CHECKPOINT_REPO,
                local_dir=active_output_dir,
                allow_patterns=[f"{current_stage}/{_latest_ckpt}/**"],
                token=_hf_token,
            )
            # Pindahkan dari subfolder ke root output dir jika perlu
            _sub_dir = os.path.join(active_output_dir, current_stage)
            if os.path.exists(_sub_dir):
                import shutil
                for _item in os.listdir(_sub_dir):
                    _src = os.path.join(_sub_dir, _item)
                    _dst = os.path.join(active_output_dir, _item)
                    if os.path.isdir(_src) and _item.startswith("checkpoint-"):
                        if os.path.exists(_dst):
                            shutil.rmtree(_dst)
                        shutil.move(_src, _dst)

            _checkpoints = sorted([
                d for d in os.listdir(active_output_dir)
                if d.startswith("checkpoint-") and os.path.isdir(os.path.join(active_output_dir, d))
            ])
            if _checkpoints:
                _resume_from = True
                print(f"✅ Ditemukan {len(_checkpoints)} checkpoint(s). Resume dari yang terbaru!")
            else:
                print("⚠️ Tidak ada checkpoint valid ditemukan. Mulai dari awal.")
        except Exception as e:
            print(f"⚠️ Gagal download checkpoint: {e}. Mulai dari awal.")

    # === START TRAINING ===
    print(f"\n🚀 Starting {current_stage.upper()} training...")
    trainer.train(resume_from_checkpoint=_resume_from)

    # === SAVE FINAL ADAPTER + UPLOAD ===
    _final_path = os.path.join(active_output_dir, "final_adapter")
    print(f"\n💾 Saving final adapter to {_final_path}...")
    trainer.save_model(_final_path)
    tokenizer.save_pretrained(_final_path)

    # Upload final adapter ke HF
    try:
        from huggingface_hub import HfApi as _FinalApi
        _final_api = _FinalApi(token=_hf_token)
        print(f"📤 Uploading final adapter to HF {current_stage}/final_adapter/...")
        _final_api.upload_folder(
            folder_path=_final_path,
            path_in_repo=f"{current_stage}/final_adapter",
            repo_id=HF_CHECKPOINT_REPO,
        )
        # Upload final chart dan log juga
        for _art in ["training_chart.png", "eval_samples.txt"]:
            _art_path = os.path.join(active_output_dir, _art)
            if os.path.exists(_art_path):
                _final_api.upload_file(
                    path_or_fileobj=_art_path,
                    path_in_repo=f"{current_stage}/{_art}",
                    repo_id=HF_CHECKPOINT_REPO,
                )
        print(f"✅ {current_stage.upper()} training selesai dan ter-upload!")
    except Exception as e:
        print(f"⚠️ Upload final adapter gagal: {e}")

    return (trainer,)


# =====================================================================
# MERGE & QUANTIZE
# =====================================================================
@app.cell
def _(
    HF_CHECKPOINT_REPO,
    LOAD_IN_4BIT,
    MAX_SOURCE_LENGTH,
    OUTPUT_DIR,
    os,
    tokenizer,
    model,
):
    def merge_and_quantize(model, tokenizer, upload_dir: str):
        if model is None:
            from unsloth import FastLanguageModel
            # Load model dari adapter ORPO final
            _orpo_path = os.path.join(OUTPUT_DIR, "orpo", "final_adapter")
            if not os.path.exists(_orpo_path):
                # Fallback download dari HF
                from huggingface_hub import snapshot_download as _snap_dl
                print("📥 Downloading final ORPO adapter dari HF untuk merging...")
                _snap_dl(
                    repo_id=HF_CHECKPOINT_REPO,
                    local_dir=_orpo_path,
                    allow_patterns=["orpo/final_adapter/**"],
                    token=os.environ.get("HF_TOKEN"),
                )
                _sub_path = os.path.join(_orpo_path, "orpo", "final_adapter")
                if os.path.exists(_sub_path):
                    _orpo_path = _sub_path
            
            print(f"📂 Loading model dari ORPO adapter untuk merge: {_orpo_path}")
            model, _ = FastLanguageModel.from_pretrained(
                model_name=_orpo_path,
                max_seq_length=MAX_SOURCE_LENGTH,
                load_in_4bit=LOAD_IN_4BIT,
                trust_remote_code=True,
            )

        merged_bf16_path = os.path.join(upload_dir, "merged_bf16")
        quantized_4bit_path = os.path.join(upload_dir, "quantized_4bit")

        print("Merging LoRA adapter and saving model as BF16 using Unsloth...")
        model.save_pretrained_merged(merged_bf16_path, tokenizer, save_method="merged_16bit")
        print("✅ Model BF16 berhasil disimpan.")

        print("\nMerging LoRA adapter and saving model as 4-bit NF4 using Unsloth...")
        model.save_pretrained_merged(quantized_4bit_path, tokenizer, save_method="merged_4bit_forced")
        print("✅ Model 4-bit NF4 berhasil disimpan!")

        return None

    upload_dir = os.path.join(OUTPUT_DIR, "hf_upload")

    merge_and_quantize(model, tokenizer, upload_dir)
    return merge_and_quantize, upload_dir


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### 💻 Local Deployment & Inference (Direct Load from Hugging Face Hub Subfolders)
    Setelah model diunggah ke Hugging Face Hub, repositori Anda akan memiliki struktur:
    - `sft/` — Checkpoint dan artifacts SFT training
    - `orpo/` — Checkpoint dan artifacts ORPO training
    - `merged_bf16/` — Model gabungan utuh (bfloat16, ~15 GB)
    - `quantized_4bit/` — Model terkuantisasi (NF4, ~5 GB)

    #### Load Model Quantized 4-bit:
    ```python
    from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

    model_id = "daruokta/t5gemma-2-4b-4b-instruct-chat-indo-v4-unsloth"

    tokenizer = AutoTokenizer.from_pretrained(model_id, subfolder="quantized_4bit")
    model = AutoModelForSeq2SeqLM.from_pretrained(
        model_id, subfolder="quantized_4bit", device_map="auto"
    )
    ```

    #### Load Model Full Precision (BF16):
    ```python
    import torch
    from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

    model_id = "daruokta/t5gemma-2-4b-4b-instruct-chat-indo-v4-unsloth"

    tokenizer = AutoTokenizer.from_pretrained(model_id, subfolder="merged_bf16")
    model = AutoModelForSeq2SeqLM.from_pretrained(
        model_id, subfolder="merged_bf16",
        torch_dtype=torch.bfloat16, device_map="auto"
    )
    ```
    """)
    return


@app.cell
def _(HF_CHECKPOINT_REPO, upload_dir, os):
    from huggingface_hub import HfApi as _UploadMergedApi

    print(f"Memulai proses unggah model merged ke HF Hub: {HF_CHECKPOINT_REPO}...")
    try:
        _merged_api = _UploadMergedApi(token=os.environ.get("HF_TOKEN"))

        _merged_api.upload_folder(
            folder_path=upload_dir,
            repo_id=HF_CHECKPOINT_REPO,
            repo_type="model",
        )

        print("✅ Berhasil mengunggah merged models ke Hugging Face Hub!")
    except Exception as e:
        print(f"❌ Terjadi kesalahan saat mengunggah: {e}")
    return


# =====================================================================
# VISUALISASI HASIL EVALUASI
# =====================================================================
@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ## 📊 Visualisasi Hasil Evaluasi Kualitatif
    """)
    return


@app.cell
def _(OUTPUT_DIR, current_stage, mo, os, re):
    # Path ke file log hasil evaluasi — sesuai stage aktif
    _active_dir = os.path.join(OUTPUT_DIR, current_stage) if current_stage != "done" else os.path.join(OUTPUT_DIR, "orpo")
    log_file_path = os.path.join(_active_dir, "eval_samples.txt")

    def parse_log_file(filepath):
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

    refresh_button = mo.ui.button(label="🔄 Refresh Data Evaluasi", value=0)

    css_style = mo.Html("""
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
    return css_style, log_file_path, parse_log_file, refresh_button


@app.cell
def _(log_file_path, mo, parse_log_file, refresh_button):
    _ = refresh_button.value

    evaluation_runs = parse_log_file(log_file_path)

    if not evaluation_runs:
        step_dropdown = None
    else:
        run_options = {run["label"]: idx for idx, run in enumerate(evaluation_runs)}
        step_dropdown = mo.ui.dropdown(
            options=run_options,
            value=next(iter(run_options)),
            label="Pilih Step Evaluasi:",
            full_width=True,
        )
    return evaluation_runs, step_dropdown


@app.cell
def _(css_style, evaluation_runs, log_file_path, mo, refresh_button, step_dropdown):
    if not evaluation_runs or step_dropdown is None:
        _output = mo.md(
            f"⚠️ *Belum ada data evaluasi ditemukan di `{log_file_path}`. Silakan jalankan training terlebih dahulu.*"
        )
    else:
        _selected_idx = step_dropdown.value
        _selected_run = evaluation_runs[_selected_idx]

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
        {css_style.text}
        <div class="sample-container">
            {"".join(_cards_html)}
        </div>
        """

        _output = mo.vstack(
            [
                mo.md(
                    f"Menampilkan **{len(_selected_run['samples'])} sampel** untuk **{_selected_run['label']}**."
                ),
                step_dropdown,
                mo.Html(_container_html),
            ]
        )

    mo.vstack([refresh_button, mo.hstack([_output], justify="start")])
    return


if __name__ == "__main__":
    app.run()
