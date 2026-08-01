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
#     "torch",
#     "transformers==5.12.1",
# ]
# ///

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


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Multi-task SFT Training: T5Gemma-2 Cloud Pipeline (Version 5)
    =====================================================================
    Notebook ini melatih model **T5Gemma-2 4B-4B** (atau varian lain) menggunakan **Supervised Fine-Tuning (SFT)** dengan LoRA.
    - Dataset dimuat langsung dari Hugging Face Hub (split `train` dan `validation`).
    - Menggunakan logit masking untuk menekan unused & vision tokens secara dinamis.
    - Mendukung pelatihan QLoRA (4-bit) untuk penghematan memori GPU.
    - Menyediakan alur penggabungan (*merge*) adapter LoRA ke base model secara langsung di presisi tinggi (`bfloat16`).
    - Menyediakan petunjuk kuantisasi 4-bit NF4 untuk deployment/inferensi hemat memori di GPU lokal (6 GB VRAM).
    """)
    return


@app.cell
def _():
    import os
    import re
    import torch
    import random
    import datetime
    import matplotlib.pyplot as plt
    from datasets import Dataset, load_dataset
    from transformers import (
        AutoTokenizer,
        AutoModelForSeq2SeqLM,
        Seq2SeqTrainer,
        Seq2SeqTrainingArguments,
        DataCollatorForSeq2Seq,
        PreTrainedTokenizerFast,
        TrainerCallback,
        TrainerControl,
        TrainerState,
        TrainingArguments,
        EarlyStoppingCallback,
        get_scheduler,
    )
    from peft import LoraConfig, get_peft_model, TaskType
    from typing import cast, Any

    # Gunakan inline backend untuk matplotlib di Jupyter Notebook
    # '%matplotlib inline' command supported automatically in marimo
    import numpy as np

    # Optional imports for ROUGE & BLEU evaluation
    try:
        import evaluate

        rouge_metric = evaluate.load("rouge")
        bleu_metric = evaluate.load("bleu")
    except Exception as e:
        print(
            f"Warning: evaluate, rouge_score or bleu not available. Metric evaluation will be bypassed. Error: {e}"
        )
        rouge_metric = None
        bleu_metric = None
    return (
        Any,
        AutoModelForSeq2SeqLM,
        AutoTokenizer,
        DataCollatorForSeq2Seq,
        Dataset,
        EarlyStoppingCallback,
        LoraConfig,
        PreTrainedTokenizerFast,
        Seq2SeqTrainer,
        Seq2SeqTrainingArguments,
        TaskType,
        TrainerCallback,
        TrainerControl,
        TrainerState,
        TrainingArguments,
        bleu_metric,
        cast,
        datetime,
        get_peft_model,
        get_scheduler,
        load_dataset,
        np,
        os,
        plt,
        random,
        re,
        rouge_metric,
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
        # Removed write_permission to support newer huggingface_hub versions
        login(token=hf_token_input.value)
        status = mo.md(
            "✅ **Successfully authenticated with Hugging Face Hub!** You can now load gated models."
        )
    except Exception as e:
        status = mo.md(f"❌ **Authentication failed:** {e}")

    status
    return


@app.cell
def _():
    # Install library yang diperlukan (uncomment jika dijalankan di Google Colab atau environment baru)
    # !pip install -q transformers datasets peft accelerate matplotlib ipywidgets -U
    return


@app.cell
def _(torch):
    # =====================================================================
    # 1. KONFIGURASI HYPERPARAMETER (TERPUSAT & MUDAH DIUBAH)
    # =====================================================================

    # MODEL CONFIG
    MODEL_NAME = "google/t5gemma-2-4b-4b"  # "google/t5gemma-2-1b-1b" atau "google/t5gemma-2-270m-270m"
    LOAD_IN_4BIT = True  # Set True untuk menggunakan QLoRA (GPU RAM kecil), False untuk BF16 murni (GPU RAM besar)
    OUTPUT_DIR = "results/t5gemma2-clean-sft"

    # DATASET CONFIG (Hugging Face)
    HF_REPO_ID = "daruokta/t5gemma2-indonesia-chat-formatted"
    CHAT_CONFIG = "chat_sft"
    INDOQA_CONFIG = "indoqa_sft"

    # SAMPLE SIZES (Set ke 0 untuk mengambil seluruh data)
    SAMPLE_TRAIN_CHAT = 0  # Jumlah sampel chat untuk training
    SAMPLE_TRAIN_INDOQA = 0  # Jumlah sampel QA untuk training
    SAMPLE_VAL_CHAT = (
        0  # Jumlah sampel chat untuk validation/eval_loss (0 = semua data)
    )
    SAMPLE_VAL_INDOQA = (
        0  # Jumlah sampel QA untuk validation/eval_loss (0 = semua data)
    )

    # GENERATION EVALUATION CONFIG
    SAMPLE_EVAL_GENERATION = (
        10  # Jumlah sampel dari validation set untuk pengujian teks generasi di log
    )
    EVAL_EVERY_N_STEPS = 200  # Jalankan evaluasi loss & generasi setiap N steps

    # SYSTEM PROMPT FALLBACK
    SYSTEM_PROMPT = (
        "Kamu adalah asisten AI yang helpful, santai, dan ramah. "
        "Gunakan Bahasa Indonesia sebagai bahasa utama."
    )

    # BASIC TRAINING SPECS
    MAX_SOURCE_LENGTH = 2048
    MAX_TARGET_LENGTH = 512
    NUM_EPOCHS = 5
    LEARNING_RATE = 1e-5

    # BATCH SIZE & ACCUMULATION
    PER_DEVICE_TRAIN_BATCH_SIZE = 4
    PER_DEVICE_EVAL_BATCH_SIZE = 32
    GRADIENT_ACCUMULATION_STEPS = 32
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
    WARMUP_STEPS = 100
    WEIGHT_DECAY = 0.1
    LR_SCHEDULER_TYPE = "cosine"
    LOGGING_STEPS = 50
    SAVE_TOTAL_LIMIT = 2
    OPTIM = "paged_adamw_8bit"
    LABEL_SMOOTHING_FACTOR = 0.1
    NEFTUNE_NOISE_ALPHA = 5.0

    # HARDWARE & CONTROL SPECS
    GRADIENT_CHECKPOINTING = True
    FP16 = False  # Set True jika ingin menggunakan FP16 biasa
    BF16 = torch.cuda.is_available()  # Gunakan BF16 jika GPU mendukung
    PREDICT_WITH_GENERATE = True
    EARLY_STOPPING_PATIENCE = 8

    # EVALUATION GENERATION BEHAVIOR CONFIG
    GEN_TEMPERATURE = 0.7
    GEN_TOP_P = 0.9
    GEN_REPETITION_PENALTY = 1.2

    # Token IDs yang harus di-suppress (unused + vision)
    SUPPRESS_BLOCK1 = list(range(6, 105))  # <unused0>–<unused98>
    SUPPRESS_BLOCK2 = list(range(256002, 262144))  # <unused100>–<unused6241>
    SUPPRESS_VISION = [255999, 256000, 256001]  # <end_of_image>, <image_soft_token>
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
        NUM_EPOCHS,
        OPTIM,
        OUTPUT_DIR,
        PER_DEVICE_EVAL_BATCH_SIZE,
        PER_DEVICE_TRAIN_BATCH_SIZE,
        PREDICT_WITH_GENERATE,
        SAMPLE_EVAL_GENERATION,
        SAMPLE_TRAIN_CHAT,
        SAMPLE_TRAIN_INDOQA,
        SAMPLE_VAL_CHAT,
        SAMPLE_VAL_INDOQA,
        SAVE_TOTAL_LIMIT,
        SYSTEM_PROMPT,
        WARMUP_STEPS,
        WEIGHT_DECAY,
    )


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
            samples = []
            for row in ds:
                item = {"input": row["input"], "target": row["target"]}
                if "chat_idx" in row:
                    item["chat_idx"] = row["chat_idx"]
                if "turn_idx" in row:
                    item["turn_idx"] = row["turn_idx"]
                samples.append(item)

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
        """
        Menerapkan logit masking secara dinamis lewat PyTorch forward hook.
        Mendaftarkannya pada lapisan proyeksi terakhir (lm_head) agar tidak mengganggu
        mekanisme gradient checkpointing (autograd recomputation).
        """
        vocab_size = model.config.vocab_size
        suppress_list = [i for i in suppress_ids if i < vocab_size]

        # Gunakan nilai negatif besar yang kompatibel dengan bfloat16 (-10000.0)
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

        # Cari lapisan proyeksi akhir (lm_head) untuk dipasangi hook
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
                # Tampilkan angka persis sesuai dengan yang dicetak di log (tanpa pembagian)
                actual_loss = float(logs["loss"])
                self.train_losses.append(actual_loss)
            if "eval_loss" in logs:
                self.eval_steps.append(state.global_step)
                self.eval_losses.append(float(logs["eval_loss"]))
            if "eval_rougeL" in logs:
                self.eval_rougeL.append(float(logs["eval_rougeL"]))
            self._save_chart()

        def _save_chart(self) -> None:
            if len(self.train_steps) < 2 and len(self.eval_steps) < 1:
                return

            has_rouge = len(self.eval_rougeL) > 0

            if has_rouge:
                fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
            else:
                fig, ax1 = plt.subplots(figsize=(10, 4))
                ax2 = None

            # Plot Loss on ax1
            if self.train_losses:
                ax1.plot(
                    self.train_steps,
                    self.train_losses,
                    color="#4A90D9",
                    linewidth=1.5,
                    label="Train Loss",
                )
                if len(self.train_losses) >= 10:
                    window = 10
                    ma = [
                        sum(self.train_losses[max(0, i - window) : i + 1])
                        / len(self.train_losses[max(0, i - window) : i + 1])
                        for i in range(len(self.train_losses))
                    ]
                    ax1.plot(
                        self.train_steps,
                        ma,
                        color="#E74C3C",
                        linewidth=2,
                        label="Train Loss (MA-10)",
                        alpha=0.8,
                    )

            if self.eval_losses:
                ax1.plot(
                    self.eval_steps,
                    self.eval_losses,
                    color="#2ECC71",
                    marker="o",
                    linestyle="--",
                    linewidth=1.5,
                    label="Eval Loss",
                )

            ax1.set_xlabel("Steps")
            ax1.set_ylabel("Loss")
            ax1.set_title("Training & Evaluation Loss Curve")
            ax1.grid(True, alpha=0.3)
            ax1.legend()

            # Plot ROUGE-L on ax2
            if has_rouge and ax2 is not None:
                ax2.plot(
                    self.eval_steps,
                    self.eval_rougeL,
                    color="#9B59B6",
                    marker="s",
                    linestyle="-",
                    linewidth=2,
                    label="Eval ROUGE-L",
                )
                ax2.set_xlabel("Steps")
                ax2.set_ylabel("ROUGE-L Score (%)")
                ax2.set_title("Evaluation ROUGE-L Score Curve")
                ax2.grid(True, alpha=0.3)
                ax2.legend()

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

            model.eval()
            timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            lines = [
                f"\n{'=' * 60}",
                f"Step {state.global_step} | {timestamp}",
                f"{'=' * 60}",
            ]

            # LAKUKAN BATCH GENERATION DENGAN CASTING LONG SECARA KETAT
            with torch.no_grad():
                # Memastikan input_ids bertipe torch.long sejak awal
                input_ids_list = [
                    torch.tensor(s["input_ids"], dtype=torch.long)
                    for s in self.eval_samples
                ]

                pad_id = (
                    self.tokenizer.pad_token_id
                    if self.tokenizer.pad_token_id is not None
                    else self._eos_id
                )

                max_len = max(len(x) for x in input_ids_list)
                padded_inputs = []
                attention_masks = []
                for x in input_ids_list:
                    pad_len = max_len - len(x)
                    # Gunakan dtype=torch.long untuk padding & attention mask
                    padded_inputs.append(
                        torch.cat(
                            [torch.tensor([pad_id] * pad_len, dtype=torch.long), x]
                        )
                    )
                    attention_masks.append(
                        torch.cat(
                            [
                                torch.zeros(pad_len, dtype=torch.long),
                                torch.ones(len(x), dtype=torch.long),
                            ]
                        )
                    )

                # Paksa tipe data Long (.long() atau dtype=torch.long) saat ditaruh di device GPU
                batch_inputs = torch.stack(padded_inputs).to(
                    device=model.device, dtype=torch.long
                )
                batch_masks = torch.stack(attention_masks).to(
                    device=model.device, dtype=torch.long
                )

                # Generate sekaligus dalam 1 batch forward pass!
                outputs = getattr(model, "generate")(
                    input_ids=batch_inputs,
                    attention_mask=batch_masks,
                    max_new_tokens=256,
                    do_sample=True,
                    temperature=self.temperature,
                    top_p=self.top_p,
                    repetition_penalty=self.repetition_penalty,
                    no_repeat_ngram_size=3,
                    eos_token_id=self._stop_ids,
                    pad_token_id=pad_id,
                    bad_words_ids=self.bad_words_ids,
                )

                # Decode hasil batch
                for idx, sample in enumerate(self.eval_samples):
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

                    raw_response = self.tokenizer.decode(
                        outputs[idx], skip_special_tokens=True
                    )
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

            model.train()
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
                # Sort turns by turn_idx to keep the sequence order
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
                        break  # subsequent turns will definitely be longer, so we can stop processing this chat
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

    return (process_sft_rows,)


@app.cell
def _(
    AutoTokenizer,
    CHAT_CONFIG,
    Dataset,
    HF_REPO_ID,
    INDOQA_CONFIG,
    MODEL_NAME,
    OUTPUT_DIR,
    PreTrainedTokenizerFast,
    SAMPLE_EVAL_GENERATION,
    SAMPLE_TRAIN_CHAT,
    SAMPLE_TRAIN_INDOQA,
    SAMPLE_VAL_CHAT,
    SAMPLE_VAL_INDOQA,
    load_hf_samples,
    os,
    process_sft_rows,
    random,
):
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print(f"Loading Tokenizer from {MODEL_NAME}...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
    assert isinstance(tokenizer, PreTrainedTokenizerFast), (
        "Tokenizer harus PreTrainedTokenizerFast"
    )

    # Load datasets dari Hugging Face Hub (Split Train & Validation terpisah)
    print("\nLoading training & validation datasets...")
    train_chat_samples = load_hf_samples(
        HF_REPO_ID, CHAT_CONFIG, "train", SAMPLE_TRAIN_CHAT
    )
    train_indoqa_samples = load_hf_samples(
        HF_REPO_ID, INDOQA_CONFIG, "train", SAMPLE_TRAIN_INDOQA
    )

    val_chat_samples = load_hf_samples(
        HF_REPO_ID, CHAT_CONFIG, "validation", SAMPLE_VAL_CHAT
    )
    val_indoqa_samples = load_hf_samples(
        HF_REPO_ID, INDOQA_CONFIG, "validation", SAMPLE_VAL_INDOQA
    )

    print("Processing training rows...")
    train_rows = process_sft_rows(
        train_chat_samples, tokenizer, is_chat=True
    ) + process_sft_rows(train_indoqa_samples, tokenizer, is_chat=False)
    print("Processing validation rows...")
    val_rows = process_sft_rows(
        val_chat_samples, tokenizer, is_chat=True
    ) + process_sft_rows(val_indoqa_samples, tokenizer, is_chat=False)

    random.seed(42)
    random.shuffle(train_rows)
    random.shuffle(val_rows)

    print(f"Total SFT Training rows: {len(train_rows)}")
    print(f"Total SFT Validation rows: {len(val_rows)}")

    train_ds = Dataset.from_list(train_rows)
    eval_ds = Dataset.from_list(val_rows)

    # Ambil sampel dari validation set untuk evaluasi teks generasi berkala
    n_eval_gen = min(len(val_rows), SAMPLE_EVAL_GENERATION)
    eval_generation_samples = val_rows[:n_eval_gen]
    print(
        f"Mengambil {n_eval_gen} sampel validasi untuk pencatatan evaluasi kualitatif."
    )
    return eval_ds, eval_generation_samples, tokenizer, train_ds


@app.cell
def _(
    ALL_SUPPRESS_IDS,
    AutoModelForSeq2SeqLM,
    LORA_ALPHA,
    LORA_DROPOUT,
    LORA_RANK,
    LORA_TARGET_MODULES,
    LoraConfig,
    MODEL_NAME,
    LOAD_IN_4BIT,
    TaskType,
    apply_logit_mask,
    get_peft_model,
    tokenizer,
    torch,
):
    from transformers import BitsAndBytesConfig
    from peft import prepare_model_for_kbit_training

    # Load Model
    # Triggering model load with new logit masking
    print(f"\nLoading Model from {MODEL_NAME}...")
    
    if LOAD_IN_4BIT:
        print("  Using 4-bit Quantization (QLoRA) config...")
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
            llm_int8_skip_modules=['model.encoder.vision_tower', 'lm_head', 'embed_tokens']
        )
        model = AutoModelForSeq2SeqLM.from_pretrained(
            MODEL_NAME,
            trust_remote_code=True,
            quantization_config=bnb_config,
            device_map="auto"
        )
        # Prepare model for 4-bit training
        model = prepare_model_for_kbit_training(model)
    else:
        print("  Using BF16 precision...")
        model = AutoModelForSeq2SeqLM.from_pretrained(
            MODEL_NAME,
            trust_remote_code=True,
            dtype=torch.bfloat16,
            device_map="auto"
        )

    # Reset max_length to silence warning about max_new_tokens taking precedence
    model.config.max_length = None
    if hasattr(model, "generation_config") and model.generation_config is not None:
        model.generation_config.max_length = None

    if getattr(model.config, "decoder_start_token_id", None) is None:
        model.config.decoder_start_token_id = tokenizer.bos_token_id
        print(f"  Set decoder_start_token_id = {model.config.decoder_start_token_id}")

    # Set pad token dan resize jika diperlukan
    if tokenizer.pad_token is None:
        tokenizer.add_special_tokens({"pad_token": tokenizer.eos_token})
        model.resize_token_embeddings(len(tokenizer))

    # Logit Masking (Dinamis, non-destructive!)
    print(f"\nApplying logit mask for {len(ALL_SUPPRESS_IDS)} tokens...")
    apply_logit_mask(model, ALL_SUPPRESS_IDS)

    # LoRA Config
    lora_config = LoraConfig(
        task_type=TaskType.SEQ_2_SEQ_LM,
        r=LORA_RANK,
        lora_alpha=LORA_ALPHA,
        target_modules=LORA_TARGET_MODULES,
        lora_dropout=LORA_DROPOUT,
    )
    model = get_peft_model(model, lora_config)
    setattr(model.config, "use_cache", False)
    model.print_trainable_parameters()
    return (model,)


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

                    # ─── GROKFAST ───
                    state["grok_slow_grad"].mul_(grok_lamb).add_(
                        grad, alpha=1.0 - grok_lamb
                    )
                    filtered_grad = grad.clone()
                    filtered_grad.add_(state["grok_slow_grad"], alpha=grok_alpha)

                    if weight_decay != 0:
                        p.data.mul_(1.0 - lr * weight_decay)

                    # ─── ADEMAMIX ───
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


@app.cell
def _(
    ALL_SUPPRESS_IDS,
    Any,
    BF16,
    DataCollatorForSeq2Seq,
    EARLY_STOPPING_PATIENCE,
    EVAL_ACCUMULATION_STEPS,
    EVAL_EVERY_N_STEPS,
    EarlyStoppingCallback,
    FP16,
    GEN_REPETITION_PENALTY,
    GEN_TEMPERATURE,
    GEN_TOP_P,
    GRADIENT_ACCUMULATION_STEPS,
    GRADIENT_CHECKPOINTING,
    GrokAdEMAMix,
    LABEL_SMOOTHING_FACTOR,
    LEARNING_RATE,
    LOGGING_STEPS,
    LR_SCHEDULER_TYPE,
    MAX_TARGET_LENGTH,
    NEFTUNE_NOISE_ALPHA,
    NUM_EPOCHS,
    OPTIM,
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
    eval_ds,
    eval_generation_samples,
    get_scheduler,
    model,
    np,
    os,
    rouge_metric,
    tokenizer,
    torch,
    train_ds,
):
    # Callbacks - Updated to trigger run
    import gc

    # 1. Bersihkan sisa memori sebelum memulai training baru
    gc.collect()
    torch.cuda.empty_cache()

    # 2. Atur konfigurasi anti-fragmentasi memori PyTorch
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

    # === CUSTOM LABEL SMOOTHER UNTUK MENGHINDARI OOM DAN ESTIMASI LOSS YANG KONSISTEN ===
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

            # Buat mask penanda token valid (bukan suppressed)
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

            # Hitung log_softmax secara efisien hanya pada token aktif
            log_probs = torch.nn.functional.log_softmax(active_logits, dim=-1)

            # NLL Loss
            nll_loss = -log_probs.gather(
                dim=-1, index=active_labels.unsqueeze(-1)
            ).squeeze(-1)

            # Label smoothing hanya pada token valid
            valid_log_probs = log_probs * valid_mask.to(log_probs.dtype)
            smooth_loss = -valid_log_probs.sum(dim=-1) / num_valid_tokens

            # Kombinasikan loss per token
            token_losses = (1.0 - self.epsilon) * nll_loss + self.epsilon * smooth_loss
            return token_losses.mean()

    # === CUSTOM TRAINER UNTUK MENGUBAH CARA HITUNG LOSS MENJADI RERATA (MEAN) PER TOKEN ===
    class CustomSeq2SeqTrainer(Seq2SeqTrainer):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self.model_accepts_loss_kwargs = False  # <-- Force disable automatic loss kwargs to fix HF's accumulation scaling logic
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

    def compute_metrics(eval_preds):
        metrics = {}
        if rouge_metric is None and bleu_metric is None:
            return metrics
        preds, labels = eval_preds
        if isinstance(preds, tuple):
            preds = preds[0]
        tok = cast(PreTrainedTokenizerFast, tokenizer)

        # Handle logits jika predict_with_generate=False secara tidak sengaja diaktifkan
        if preds.ndim == 3:
            preds = preds.argmax(axis=-1)

        labels = np.where(labels != -100, labels, tok.pad_token_id)
        preds = np.where(preds != -100, preds, tok.pad_token_id)
        decoded_preds = tok.batch_decode(preds, skip_special_tokens=True)
        decoded_labels = tok.batch_decode(labels, skip_special_tokens=True)
        decoded_preds = [pred.strip() for pred in decoded_preds]
        decoded_labels = [label.strip() for label in decoded_labels]

        # ROUGE
        if rouge_metric is not None:
            try:
                result = cast(Any, rouge_metric).compute(
                    predictions=decoded_preds,
                    references=decoded_labels,
                    use_stemmer=False,
                )
                if result is not None:
                    for key, value in result.items():
                        metrics[key] = value * 100
            except Exception as e:
                print(f"Error during ROUGE metric calculation: {e}")

        # BLEU
        if bleu_metric is not None:
            try:
                formatted_labels = [[label] for label in decoded_labels]
                bleu_result = cast(Any, bleu_metric).compute(
                    predictions=decoded_preds, references=formatted_labels
                )
                if bleu_result is not None and "bleu" in bleu_result:
                    metrics["bleu"] = bleu_result["bleu"] * 100
            except Exception as e:
                print(f"Error during BLEU metric calculation: {e}")

        return metrics

    def preprocess_logits_for_metrics(logits, labels):
        """
        Mengambil argmax langsung di GPU sebelum dikumpulkan ke RAM.
        Mencegah penumpukan logits mentah sebesar Terabytes.
        """
        if isinstance(logits, tuple):
            logits = logits[0]
        return logits.argmax(dim=-1)

    bad_words_ids = [
        [id_] for id_ in ALL_SUPPRESS_IDS if id_ < cast(Any, model).config.vocab_size
    ]

    plot_callback = TrainingPlotCallback(output_dir=OUTPUT_DIR)
    sample_callback = SampleGenerationCallback(
        tokenizer=tokenizer,
        eval_samples=eval_generation_samples,
        output_dir=OUTPUT_DIR,
        eval_every_n_steps=EVAL_EVERY_N_STEPS,
        temperature=GEN_TEMPERATURE,
        top_p=GEN_TOP_P,
        repetition_penalty=GEN_REPETITION_PENALTY,
        bad_words_ids=bad_words_ids,
    )

    data_collator = DataCollatorForSeq2Seq(
        tokenizer=tokenizer, model=model, padding=True
    )

    training_args = Seq2SeqTrainingArguments(
        output_dir=OUTPUT_DIR,
        per_device_train_batch_size=PER_DEVICE_TRAIN_BATCH_SIZE,
        per_device_eval_batch_size=PER_DEVICE_EVAL_BATCH_SIZE,
        gradient_accumulation_steps=GRADIENT_ACCUMULATION_STEPS,
        eval_accumulation_steps=EVAL_ACCUMULATION_STEPS,
        learning_rate=LEARNING_RATE,
        num_train_epochs=NUM_EPOCHS,
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
        # load_best_model_at_end=True,
        # metric_for_best_model="eval_loss",
        optim=OPTIM,
        label_smoothing_factor=LABEL_SMOOTHING_FACTOR,
        neftune_noise_alpha=NEFTUNE_NOISE_ALPHA,
        report_to="none",
        fp16=FP16,
        bf16=BF16,
        gradient_checkpointing=GRADIENT_CHECKPOINTING,
        generation_max_length=MAX_TARGET_LENGTH,
    )

    # Inisialisasi GrokAdEMAMix custom optimizer
    optimizer = GrokAdEMAMix(
        model.parameters(),
        lr=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY,
        grok_alpha=2.0,
        grok_lamb=0.98,
    )

    # Hitung total training steps secara manual untuk scheduler
    num_update_steps_per_epoch = max(
        1, len(train_ds) // (PER_DEVICE_TRAIN_BATCH_SIZE * GRADIENT_ACCUMULATION_STEPS)
    )
    max_steps = num_update_steps_per_epoch * NUM_EPOCHS

    lr_scheduler = get_scheduler(
        name=LR_SCHEDULER_TYPE,
        optimizer=optimizer,
        num_warmup_steps=WARMUP_STEPS,
        num_training_steps=max_steps,
    )

    # Gunakan Custom Trainer di sini
    trainer = CustomSeq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=cast(Any, train_ds),
        eval_dataset=cast(Any, eval_ds),
        data_collator=data_collator,
        compute_metrics=compute_metrics,
        preprocess_logits_for_metrics=None
        if PREDICT_WITH_GENERATE
        else preprocess_logits_for_metrics,
        optimizers=(optimizer, lr_scheduler),
        callbacks=[
            plot_callback,
            sample_callback,
            # EarlyStoppingCallback(early_stopping_patience=EARLY_STOPPING_PATIENCE),
        ],
    )

    print("\nStarting Clean SFT on Cloud/Notebook...")
    trainer.train()
    return (trainer,)


@app.cell
def _(OUTPUT_DIR, os, tokenizer, trainer):
    # Save

    final_path = os.path.join(OUTPUT_DIR, "final_adapter")
    print(f"\nSaving final SFT adapter to {final_path}...")
    trainer.save_model(final_path)
    tokenizer.save_pretrained(final_path)
    print("✅ Clean SFT Selesai!")
    return


@app.cell
def _(MODEL_NAME, OUTPUT_DIR, os, tokenizer):
    # Merging LoRA Adapter to Base Model (BF16) & Quantizing to 4-bit NF4
    def merge_and_quantize(model_name: str, adapter_path: str, upload_dir: str):
        from transformers import AutoModelForSeq2SeqLM, BitsAndBytesConfig
        from peft import PeftModel
        import torch
        import gc
        
        merged_bf16_path = os.path.join(upload_dir, "merged_bf16")
        quantized_4bit_path = os.path.join(upload_dir, "quantized_4bit")

        # 1. Pemuatan base model dalam bfloat16 (full precision)
        print("Pemuatan base model dalam presisi tinggi (BF16) untuk penggabungan...")
        base_model = AutoModelForSeq2SeqLM.from_pretrained(
            model_name,
            trust_remote_code=True,
            torch_dtype=torch.bfloat16,
            device_map="auto"
        )
        
        # 2. Pemuatan adapter LoRA
        print(f"Pemuatan adapter LoRA dari {adapter_path}...")
        peft_model = PeftModel.from_pretrained(base_model, adapter_path)
        
        # 3. Merging bobot
        print("Memulai proses penggabungan bobot (Merging weights)...")
        merged_model = peft_model.merge_and_unload()
        
        # 4. Simpan model gabungan utuh (BF16) ke hf_upload/merged_bf16
        print(f"Menyimpan model hasil penggabungan (BF16) ke {merged_bf16_path}...")
        merged_model.save_pretrained(merged_bf16_path)
        tokenizer.save_pretrained(merged_bf16_path)
        print("✅ Model BF16 berhasil disimpan.")
        
        # Bersihkan memori agar tidak OOM saat melakukan kuantisasi berikutnya
        del peft_model
        del base_model
        del merged_model
        gc.collect()
        torch.cuda.empty_cache()
        
        # 5. Kuantisasi model hasil merging ke 4-bit (NF4)
        print("\nMemuat ulang model hasil penggabungan dalam mode kuantisasi 4-bit (NF4)...")
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
            llm_int8_skip_modules=['model.encoder.vision_tower', 'lm_head', 'embed_tokens']
        )
        quantized_model = AutoModelForSeq2SeqLM.from_pretrained(
            merged_bf16_path,
            quantization_config=bnb_config,
            device_map="auto"
        )
        
        # 6. Simpan model yang sudah terkuantisasi ke hf_upload/quantized_4bit
        print(f"Menyimpan model terkuantisasi (4-bit) ke {quantized_4bit_path}...")
        quantized_model.save_pretrained(quantized_4bit_path, safe_serialization=True)
        tokenizer.save_pretrained(quantized_4bit_path)
        print("✅ Proses kuantisasi dan penyimpanan model berhasil selesai!")
        
        return quantized_model

    # Folder penampung kedua model untuk diunggah ke Hugging Face
    upload_dir = os.path.join(OUTPUT_DIR, "hf_upload")
    
    # Helper ini bisa dijalankan secara manual dengan mencopot tanda komentar di bawah ini:
    # merge_and_quantize(MODEL_NAME, os.path.join(OUTPUT_DIR, "final_adapter"), upload_dir)
    return merge_and_quantize, upload_dir


@app.cell
def _(mo):
    mo.md(
        r"""
        ### 💻 Local Deployment & Inference (Direct Load from Hugging Face Hub Subfolders)
        Setelah model diunggah ke Hugging Face Hub, repositori Anda akan memiliki dua subfolder:
        1.  `merged_bf16` (Model gabungan utuh dengan presisi tinggi bfloat16, ~15 GB)
        2.  `quantized_4bit` (Model terkuantisasi hemat memori, ~5 GB)
        
        Anda dapat memuat salah satunya secara langsung menggunakan parameter `subfolder` tanpa harus mengunduh weight yang tidak diperlukan:
        
        #### Pilihan A: Load Mode Kuantisasi 4-bit NF4 (Direkomendasikan untuk GPU 6 VRAM):
        ```python
        from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
        
        model_id = "daruokta/t5gemma-2-4b-4b-instruct-chat-indo-v2-exp"
        
        # Pemuatan langsung model terkuantisasi dari subfolder
        tokenizer = AutoTokenizer.from_pretrained(model_id, subfolder="quantized_4bit")
        model = AutoModelForSeq2SeqLM.from_pretrained(
            model_id,
            subfolder="quantized_4bit",
            device_map="auto"
        )
        ```
        
        #### Pilihan B: Load Mode Full Precision (BF16) (Untuk GPU VRAM besar / Cloud):
        ```python
        import torch
        from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
        
        model_id = "daruokta/t5gemma-2-4b-4b-instruct-chat-indo-v2-exp"
        
        tokenizer = AutoTokenizer.from_pretrained(model_id, subfolder="merged_bf16")
        model = AutoModelForSeq2SeqLM.from_pretrained(
            model_id,
            subfolder="merged_bf16",
            torch_dtype=torch.bfloat16,
            device_map="auto"
        )
        ```
        """
    )
    return


@app.cell
def _(upload_dir):
    from huggingface_hub import HfApi

    # GANTI dengan nama repository tujuan Anda di Hugging Face Hub
    REPO_NAME = "daruokta/t5gemma-2-4b-4b-instruct-chat-indo-v2-exp"

    print(f"Memulai proses unggah folder model {upload_dir} ke Hugging Face Hub: {REPO_NAME}...")
    try:
        api = HfApi()
        
        # Buat repo jika belum ada
        api.create_repo(repo_id=REPO_NAME, repo_type="model", private=True, exist_ok=True)
        
        # Mengunggah seluruh isi folder (akan mengunggah subfolder merged_bf16 & quantized_4bit)
        api.upload_folder(
            folder_path=upload_dir,
            repo_id=REPO_NAME,
            repo_type="model",
        )

        print("✅ Berhasil mengunggah kedua model ke Hugging Face Hub!")
    except Exception as e:
        print(f"❌ Terjadi kesalahan saat mengunggah: {e}")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ## 📊 Visualisasi Hasil Evaluasi Kualitatif
    """)
    return


@app.cell
def _(OUTPUT_DIR, mo, os, re):
    # Path ke file log hasil evaluasi
    log_file_path = os.path.join(OUTPUT_DIR, "eval_samples.txt")

    def parse_log_file(filepath):
        if not os.path.exists(filepath):
            return []
        with open(filepath, "r", encoding="utf-8") as f:
            content = f.read()

        # Split berdasarkan separator baris '='
        blocks = re.split(r"={10,}", content)
        steps_data = []

        # blocks[0] bisa jadi kosong atau header awal
        # Pola log: [..., "Step X | YYYY-MM-DD HH:MM:SS", "Q: ... Expected Target: ... Model Response: ...", ...]
        i = 1
        while i < len(blocks):
            header = blocks[i].strip()
            # Cari informasi Step dan Waktu
            step_match = re.search(r"Step\s+(\d+)\s*\|\s*([\d\-\s:]+)", header)
            if not step_match:
                i += 1
                continue

            step_num = step_match.group(1)
            timestamp = step_match.group(2)
            label = f"Step {step_num} ({timestamp})"

            body = blocks[i + 1].strip() if i + 1 < len(blocks) else ""

            # Parsing sampel individual
            samples = []
            # Split berdasarkan "\nQ: "
            raw_samples = re.split(r"\n+Q:\s*", "\n" + body)
            for rs in raw_samples:
                rs = rs.strip()
                if not rs or not ("Expected Target:" in rs and "Model Response:" in rs):
                    continue

                # Parse bagian target dan response
                try:
                    q_part, rest = rs.split("Expected Target:", 1)
                    target_part, response_part = rest.split("Model Response:", 1)

                    query = q_part.strip()
                    target = target_part.strip()
                    response = response_part.strip()

                    # Bersihkan flag repetitive/good dari respons
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

        # Urutkan berdasarkan waktu/step (terbaru paling atas)
        return steps_data[::-1]

    # Tombol refresh manual
    refresh_button = mo.ui.button(label="🔄 Refresh Data Evaluasi", value=0)

    # Definisikan stylesheet CSS khusus
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
    # React to manual refresh clicks
    _ = refresh_button.value

    # Ambil data evaluasi terbaru
    evaluation_runs = parse_log_file(log_file_path)

    if not evaluation_runs:
        step_dropdown = None
    else:
        # Pilihan dropdown untuk memilih Step/Waktu Run
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
        # Ambil sampel berdasarkan pilihan dropdown
        _selected_idx = step_dropdown.value
        _selected_run = evaluation_runs[_selected_idx]

        # Buat visualisasi kartu untuk sampel
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

        # Gabungkan semua kartu ke dalam container
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
