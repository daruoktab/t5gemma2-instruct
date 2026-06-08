"""
SFT Training: T5Gemma-2 270M Cangkok
=====================================
Training Seq2Seq dengan LoRA setelah Task Vector Transplant.

Temuan dari tokenizer analysis:
  - Tokenizer T5Gemma2 dan Gemma 3 IT IDENTIK untuk teks biasa
  - ID 256001 = <image_soft_token> di T5G (bukan <unused99> seperti di IT)
  - T5G tidak punya chat_template → format prompt manual
  - BOS (ID=2) auto-prepend jika add_special_tokens=True di encoder → pastikan False

Suppress list (3 blok):
  - Blok 1: ID 6-104   (<unused0> - <unused98>)
  - Blok 2: ID 256002-262143 (<unused100> - <unused6241>)
  - Vision: ID 255999, 256000, 256001 (<start_of_image>, <end_of_image>, <image_soft_token>)

Format prompt (encoder input):
  <start_of_turn>user
  {system}

  {question}<end_of_turn>
  <start_of_turn>model

  Catatan: TIDAK ada BOS di depan (add_special_tokens=False saat tokenize encoder)
  T5Gemma2 encoder adalah bidirectional → BOS tidak diperlukan

Format target (decoder label):
  {response}<end_of_turn>
"""

import os
import re
import json
import math
import torch
import random
import datetime
import matplotlib

matplotlib.use("Agg")  # Non-interactive backend untuk server
import matplotlib.pyplot as plt
from datasets import Dataset
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
)
from peft import LoraConfig, get_peft_model, TaskType
from typing import cast, Any

# ==========================================
# KONFIGURASI HYPERPARAMETER
# ==========================================
MODEL_NAME = "models/t5gemma2-270m-task-vector"
OUTPUT_DIR = "results/t5gemma2-270m-sft"

CHAT_TRAIN_FILE = "data/chat_train.jsonl"
INDOQA_TRAIN_FILE = "data/indoqa_train.jsonl"

MAX_SOURCE_LENGTH = 2048
MAX_TARGET_LENGTH = 1024

SAMPLE_CHAT = 0  # 0 = semua data
SAMPLE_INDOQA = 0  # 0 = semua data

# Token IDs yang harus di-suppress (text-only training)
# Sumber: analisis tokenizer T5Gemma2
SUPPRESS_BLOCK1 = list(range(6, 105))  # <unused0>–<unused98>
SUPPRESS_BLOCK2 = list(range(256002, 262144))  # <unused100>–<unused6241>
SUPPRESS_VISION = [
    255999,
    256000,
    256001,
]  # <start_of_image>, <end_of_image>, <image_soft_token>
ALL_SUPPRESS_IDS = set(SUPPRESS_BLOCK1 + SUPPRESS_BLOCK2 + SUPPRESS_VISION)

SYSTEM_PROMPT = (
    "Kamu adalah asisten AI yang helpful, santai, dan ramah. "
    "Gunakan Bahasa Indonesia sebagai bahasa utama."
)

# Sample queries untuk eval callback
EVAL_QUERIES = [
    "Siapa presiden Indonesia pertama?",
    "Jelaskan apa itu fotosintesis dengan bahasa sederhana.",
    "Apa perbedaan antara simile dan metafora?",
]


# ==========================================
# FORMAT PROMPT
# ==========================================
def format_encoder_input(system: str, conversation_history: str, question: str) -> str:
    """
    Format encoder input sesuai Gemma chat format.

    Untuk T5Gemma2 encoder-decoder:
    - Encoder menerima SELURUH konteks (history + pertanyaan terakhir)
    - Trailing <start_of_turn>model memberi sinyal ke decoder bahwa giliran model
    - TIDAK perlu BOS di depan (encoder bidirectional, BOS tidak bermakna)

    Catatan: Format ini identik yang digunakan Gemma 3 IT untuk generation.
    Karena tokenizer identik, embedding akan memahami format ini dengan benar.
    """
    # Bangun header dengan system prompt di turn pertama
    header = f"<start_of_turn>user\n{system}\n\n"
    body = f"{conversation_history}{question}<end_of_turn>\n<start_of_turn>model\n"
    return header + body


def format_encoder_from_raw(raw_input: str) -> str:
    """
    Parse format 'system: ...\nuser: ...\nassistant: ...' ke Gemma chat format.
    Digunakan untuk dataset chat multi-turn.
    """
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

    # Trailing model cue untuk decoder
    formatted += "<start_of_turn>model\n"
    return formatted


# ==========================================
# DATASET UTILS
# ==========================================
def load_jsonl_samples(path: str, n_samples: int, seed: int = 42) -> list[dict]:
    if not os.path.exists(path):
        print(f"[WARN] File {path} tidak ditemukan, skip.")
        return []
    with open(path, "r", encoding="utf-8") as f:
        lines = [json.loads(line) for line in f if line.strip()]
    if n_samples > 0 and len(lines) > n_samples:
        random.seed(seed)
        return random.sample(lines, n_samples)
    return lines


# ==========================================
# UNUSED TOKEN SUPPRESSION
# ==========================================
def suppress_unused_tokens(model: Any, suppress_ids: set[int]) -> None:
    """
    Layer 1: Re-init embedding unused tokens ke mean + noise
    Layer 2: Register gradient hook untuk freeze gradients unused tokens

    Suppress 3 blok:
    - Blok 1: ID 6-104 (<unused0>–<unused98>) — tokens yang tidak pernah dipakai
    - Blok 2: ID 256002-262143 (<unused100>–<unused6241>) — tokens tidak pernah dipakai
    - Vision: 255999-256001 — <start_of_image>, <end_of_image>, <image_soft_token>
      (text-only training, vision tokens harus di-suppress)

    Catatan: Suppress vision tokens HANYA karena training ini text-only.
    Jika suatu saat multi-modal, vision tokens JANGAN di-suppress.
    """
    suppress_list = sorted(suppress_ids)

    with torch.no_grad():
        embed_weight = model.get_input_embeddings().weight
        vocab_size = embed_weight.shape[0]

        # Filter hanya ID yang valid (dalam range vocab)
        valid_suppress = [i for i in suppress_list if i < vocab_size]
        invalid = [i for i in suppress_list if i >= vocab_size]
        if invalid:
            print(
                f"  [INFO] {len(invalid)} suppress IDs out of vocab range ({vocab_size}), skipped."
            )

        # Re-init ke mean of valid embeddings + noise kecil
        valid_ids = [i for i in range(vocab_size) if i not in suppress_ids]
        valid_embeds = embed_weight[valid_ids]
        mean_val = valid_embeds.mean(dim=0)

        hidden_size = embed_weight.shape[1]
        noise = (
            torch.randn((len(valid_suppress), hidden_size), device=embed_weight.device)
            * 0.001
        )  # Noise sangat kecil untuk stabilitas
        new_embeds = mean_val.unsqueeze(0) + noise

        suppress_tensor_valid = torch.tensor(valid_suppress, dtype=torch.long)
        embed_weight[suppress_tensor_valid] = new_embeds.to(embed_weight.dtype)
        print(f"  Re-initialized {len(valid_suppress)} suppressed tokens.")

    # Gradient hook: zero-out gradients untuk suppress tokens setiap backward pass
    def freeze_suppressed_hook(grad: torch.Tensor) -> torch.Tensor:
        grad = grad.clone()
        grad[suppress_tensor_valid] = 0.0
        return grad

    if model.get_input_embeddings().weight.requires_grad:
        model.get_input_embeddings().weight.register_hook(freeze_suppressed_hook)
        print(
            f"  Gradient hook registered untuk {len(valid_suppress)} suppressed tokens."
        )
    else:
        print(
            "  Gradient hook skipped karena input embeddings dibekukan (requires_grad=False)."
        )


# ==========================================
# CALLBACKS
# ==========================================
class TrainingPlotCallback(TrainerCallback):
    """
    Realtime training loss visualization — save PNG setiap N steps.
    """

    def __init__(self, output_dir: str) -> None:
        self.output_dir = output_dir
        self.steps: list[int] = []
        self.losses: list[float] = []
        self.chart_path = os.path.join(output_dir, "training_chart.png")

    def on_log(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        logs: dict[str, float] | None = None,
        **kwargs: Any,
    ) -> None:
        if logs is None or "loss" not in logs:
            return
        self.steps.append(state.global_step)
        self.losses.append(float(logs["loss"]))
        self._save_chart()

    def _save_chart(self) -> None:
        if len(self.steps) < 2:
            return
        fig, ax = plt.subplots(figsize=(10, 4))
        ax.plot(self.steps, self.losses, color="#4A90D9", linewidth=1.5)
        ax.set_xlabel("Steps")
        ax.set_ylabel("Loss")
        ax.set_title("Training Loss Curve — T5Gemma2-270M Cangkok SFT")
        ax.grid(True, alpha=0.3)
        # Moving average
        if len(self.losses) >= 20:
            window = 20
            ma = [
                sum(self.losses[max(0, i - window) : i + 1])
                / len(self.losses[max(0, i - window) : i + 1])
                for i in range(len(self.losses))
            ]
            ax.plot(
                self.steps, ma, color="#E74C3C", linewidth=2, label="MA-20", alpha=0.8
            )
            ax.legend()
        plt.tight_layout()
        plt.savefig(self.chart_path, dpi=120)
        plt.close(fig)


class SampleGenerationCallback(TrainerCallback):
    """
    Generate sample output setiap 0.1 epoch untuk monitoring kualitas output.
    Deteksi dini: spam, repetisi, atau output yang tidak bermakna.
    """

    def __init__(
        self,
        tokenizer: PreTrainedTokenizerFast,
        queries: list[str],
        output_dir: str,
        eval_every_n_steps: int = 100,
    ) -> None:
        self.tokenizer = tokenizer
        self.queries = queries
        self.output_dir = output_dir
        self.eval_every_n_steps = eval_every_n_steps
        self.log_path = os.path.join(output_dir, "eval_samples.txt")
        self._eot_id = tokenizer.convert_tokens_to_ids("<end_of_turn>")
        self._eos_id = tokenizer.eos_token_id or 1
        self._stop_ids = list({self._eot_id, self._eos_id})

    def on_step_end(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        model: Any = None,
        **kwargs: Any,
    ) -> None:
        if state.global_step == 0 or state.global_step % self.eval_every_n_steps != 0:
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

        with torch.no_grad():
            for q in self.queries:
                prompt = (
                    f"<start_of_turn>user\n{SYSTEM_PROMPT}\n\n{q}<end_of_turn>\n"
                    f"<start_of_turn>model\n"
                )
                enc = self.tokenizer(
                    prompt,
                    return_tensors="pt",
                    add_special_tokens=False,  # Jangan prepend BOS di encoder
                    truncation=True,
                    max_length=512,
                )
                enc = {k: v.to(model.device) for k, v in enc.items()}

                out = getattr(model, "generate")(
                    **enc,
                    max_new_tokens=128,
                    do_sample=False,
                    eos_token_id=self._stop_ids,
                    bad_words_ids=[[i] for i in sorted(ALL_SUPPRESS_IDS)[:50]],
                )
                _resp_raw = self.tokenizer.decode(out[0], skip_special_tokens=True)
                response: str = (
                    _resp_raw if isinstance(_resp_raw, str) else " ".join(_resp_raw)
                )

                # Deteksi spam
                words = response.split()
                is_repetitive = (
                    len(set(words)) < max(1, len(words) * 0.3) if words else True
                )
                flag = " ⚠️ REPETITIVE" if is_repetitive else " ✅"

                lines.append(f"\nQ: {q}")
                lines.append(f"A: {response[:300]}{flag}")

        model.train()
        with open(self.log_path, "a", encoding="utf-8") as f:
            f.write("\n".join(lines) + "\n")

        # Print ke console
        print(f"\n[EVAL @ step {state.global_step}]")
        for line in lines[3:]:  # Skip header
            if line.startswith("Q:") or line.startswith("A:"):
                print(f"  {line}")




# ==========================================
# MAIN TRAINING
# ==========================================
def main() -> None:
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print(f"Loading Tokenizer from {MODEL_NAME}...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    assert isinstance(tokenizer, PreTrainedTokenizerFast), (
        "Tokenizer harus PreTrainedTokenizerFast"
    )

    # Validasi token IDs kritis
    eot_id = tokenizer.convert_tokens_to_ids("<end_of_turn>")
    sot_id = tokenizer.convert_tokens_to_ids("<start_of_turn>")
    assert eot_id == 106, f"<end_of_turn> ID salah: {eot_id} (expected 106)"
    assert sot_id == 105, f"<start_of_turn> ID salah: {sot_id} (expected 105)"
    print(f"  ✅ Token IDs validated: <start_of_turn>={sot_id}, <end_of_turn>={eot_id}")

    # ── Load & Format Dataset ─────────────────────────────────────────────
    print(
        f"\nLoading datasets (SAMPLE_CHAT={SAMPLE_CHAT}, SAMPLE_INDOQA={SAMPLE_INDOQA})..."
    )
    chat_samples = load_jsonl_samples(CHAT_TRAIN_FILE, SAMPLE_CHAT)
    indoqa_samples = load_jsonl_samples(INDOQA_TRAIN_FILE, SAMPLE_INDOQA)
    print(f"Raw: Chat={len(chat_samples)}, IndoQA={len(indoqa_samples)}")

    print("Formatting & Thread-Level Filtering...")
    final_rows: list[dict] = []

    # 1. Chat (Multi-turn) — thread-level filtering
    chat_conversations: list[list[dict]] = []
    current_conv: list[dict] = []
    for obj in chat_samples:
        if not obj.get("input") or not obj.get("target"):
            continue
        if "assistant:" not in obj["input"]:
            if current_conv:
                chat_conversations.append(current_conv)
            current_conv = []
        current_conv.append(obj)
    if current_conv:
        chat_conversations.append(current_conv)

    n_chat_kept = 0
    for conv in chat_conversations:
        conv_rows: list[dict] = []
        is_valid = True
        for turn in conv:
            inp_f = format_encoder_from_raw(turn["input"])
            tgt_f = turn["target"].strip() + "<end_of_turn>"

            # Tokenize dengan add_special_tokens=False — encoder tidak butuh BOS
            inp_ids = tokenizer.encode(inp_f, add_special_tokens=False)
            tgt_ids = tokenizer.encode(tgt_f, add_special_tokens=False)

            if len(inp_ids) > MAX_SOURCE_LENGTH or len(tgt_ids) > MAX_TARGET_LENGTH:
                is_valid = False
                break
            conv_rows.append({"input_ids": inp_ids, "labels": tgt_ids})

        if is_valid:
            final_rows.extend(conv_rows)
            n_chat_kept += 1

    print(f"Chat: Kept {n_chat_kept}/{len(chat_conversations)} conversations")

    # 2. IndoQA (Single-turn)
    n_qa_kept = 0
    for obj in indoqa_samples:
        inp_f = format_encoder_from_raw(obj.get("input", ""))
        tgt_f = obj.get("target", "").strip() + "<end_of_turn>"

        inp_ids = tokenizer.encode(inp_f, add_special_tokens=False)
        tgt_ids = tokenizer.encode(tgt_f, add_special_tokens=False)

        if len(inp_ids) <= MAX_SOURCE_LENGTH and len(tgt_ids) <= MAX_TARGET_LENGTH:
            final_rows.append({"input_ids": inp_ids, "labels": tgt_ids})
            n_qa_kept += 1

    print(f"IndoQA: Kept {n_qa_kept}/{len(indoqa_samples)} samples")
    print(f"Total training rows: {len(final_rows)}")

    if not final_rows:
        raise ValueError("Tidak ada training rows! Cek path data dan format.")

    # Split 95/5 untuk eval
    random.seed(42)
    random.shuffle(final_rows)
    split_idx = max(1, int(len(final_rows) * 0.95))
    train_rows = final_rows[:split_idx]
    eval_rows = final_rows[split_idx:]
    print(f"Train={len(train_rows)}, Eval={len(eval_rows)}")

    train_ds = Dataset.from_list(train_rows)
    eval_ds = Dataset.from_list(eval_rows)

    # ── Load Model ────────────────────────────────────────────────────────
    print(f"\nLoading Model from {MODEL_NAME}...")
    model = AutoModelForSeq2SeqLM.from_pretrained(
        MODEL_NAME,
        trust_remote_code=True,
        dtype=torch.bfloat16,
        device_map="auto",
    )
    
    if getattr(model.config, "decoder_start_token_id", None) is None:
        model.config.decoder_start_token_id = tokenizer.bos_token_id
        print(f"  Set decoder_start_token_id = {model.config.decoder_start_token_id}")

    # ── Unused Token Suppression ──────────────────────────────────────────
    print(f"\nSuppressing {len(ALL_SUPPRESS_IDS)} tokens (3 blok)...")
    print(f"  Blok 1: ID 6-104       ({len(SUPPRESS_BLOCK1)} tokens)")
    print(f"  Blok 2: ID 256002-262143 ({len(SUPPRESS_BLOCK2)} tokens)")
    print(f"  Vision: ID 255999-256001  ({len(SUPPRESS_VISION)} tokens)")
    suppress_unused_tokens(model, ALL_SUPPRESS_IDS)

    # ── LoRA Config ───────────────────────────────────────────────────────
    lora_config = LoraConfig(
        task_type=TaskType.SEQ_2_SEQ_LM,
        r=32,
        lora_alpha=64,
        # k_proj dan v_proj: tetap masuk LoRA karena LoRA hanya TAMBAH adapter
        # (tidak replace base weight). Skip k/v hanya berlaku di task vector injection.
        target_modules=[
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
        ],
        lora_dropout=0.05,
    )
    model = get_peft_model(model, lora_config)
    setattr(
        model.config, "use_cache", False
    )  # PEFT config tidak selalu punya use_cache attr typed
    model.print_trainable_parameters()

    # Re-register hook setelah PEFT wrapping (PEFT bisa replace module)
    suppress_unused_tokens(model, ALL_SUPPRESS_IDS)

    # ── Hitung eval_every_n_steps (per 0.1 epoch) ─────────────────────────
    steps_per_epoch = math.ceil(len(train_rows) / (2 * 4))  # batch_size * grad_accum
    eval_every = max(50, steps_per_epoch // 10)
    print(f"\nSteps per epoch ~{steps_per_epoch}, eval sample every {eval_every} steps")

    # ── Callbacks ─────────────────────────────────────────────────────────
    plot_callback = TrainingPlotCallback(output_dir=OUTPUT_DIR)
    sample_callback = SampleGenerationCallback(
        tokenizer=tokenizer,
        queries=EVAL_QUERIES,
        output_dir=OUTPUT_DIR,
        eval_every_n_steps=eval_every,
    )

    # ── Training Args ─────────────────────────────────────────────────────
    data_collator = DataCollatorForSeq2Seq(
        tokenizer=tokenizer, model=model, padding=True
    )

    training_args = Seq2SeqTrainingArguments(
        output_dir=OUTPUT_DIR,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=8,
        learning_rate=5e-5,
        num_train_epochs=1,
        warmup_steps=200,
        weight_decay=0.01,
        lr_scheduler_type="cosine",
        predict_with_generate=True,
        logging_steps=10,
        save_strategy="steps",
        save_steps=500,
        save_total_limit=3,
        eval_strategy="steps",
        eval_steps=500,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        optim="adamw_torch",
        report_to="none",
        bf16=torch.cuda.is_available(),
        gradient_checkpointing=True,
        generation_max_length=MAX_TARGET_LENGTH,
    )

    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=cast(Any, train_ds),
        eval_dataset=cast(Any, eval_ds),
        data_collator=data_collator,
        callbacks=[plot_callback, sample_callback],
    )

    # ── Resume from Checkpoint ────────────────────────────────────────────
    resume_from_checkpoint = None
    if os.path.exists(OUTPUT_DIR):
        checkpoints = [d for d in os.listdir(OUTPUT_DIR) if d.startswith("checkpoint-")]
        if checkpoints:
            resume_from_checkpoint = True
            print(f"Ditemukan checkpoint, melanjutkan training dari {OUTPUT_DIR}")

    print("\nStarting SFT...")
    trainer.train(resume_from_checkpoint=resume_from_checkpoint)

    # ── Save Final ────────────────────────────────────────────────────────
    final_path = os.path.join(OUTPUT_DIR, "final_adapter")
    print(f"\nSaving final adapter to {final_path}...")
    trainer.save_model(final_path)
    tokenizer.save_pretrained(final_path)
    print(f"✅ Done! Final adapter saved to {final_path}")
    print(f"   Training chart: {os.path.join(OUTPUT_DIR, 'training_chart.png')}")
    print(f"   Eval samples:   {os.path.join(OUTPUT_DIR, 'eval_samples.txt')}")


if __name__ == "__main__":
    main()
