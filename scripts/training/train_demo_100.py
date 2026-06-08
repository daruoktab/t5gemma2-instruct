import os
import re
import json
import torch
import random
import datetime
import matplotlib

matplotlib.use("Agg")  # Non-interactive backend for server
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
    EarlyStoppingCallback,
)
from peft import LoraConfig, get_peft_model, TaskType
from typing import cast, Any

# ==========================================
# KONFIGURASI HYPERPARAMETER
# ==========================================
MODEL_NAME = "google/t5gemma-2-270m-270m"
OUTPUT_DIR = "results/t5gemma2-270m-demo-100"

CHAT_TRAIN_FILE = "data/chat_train_demo.jsonl"
CHAT_VAL_FILE   = "data/chat_val_demo.jsonl"

MAX_SOURCE_LENGTH = 4096
MAX_TARGET_LENGTH = 1024

# Token IDs yang harus di-suppress (unused + vision)
SUPPRESS_BLOCK1 = list(range(6, 105))  # <unused0>–<unused98>
SUPPRESS_BLOCK2 = list(range(256002, 262144))  # <unused100>–<unused6241>
SUPPRESS_VISION = [255999, 256000, 256001]  # <end_of_image>, <image_soft_token>
ALL_SUPPRESS_IDS = set(SUPPRESS_BLOCK1 + SUPPRESS_BLOCK2 + SUPPRESS_VISION)

SYSTEM_PROMPT = (
    "Kamu adalah asisten AI yang helpful, santai, dan ramah. "
    "Gunakan Bahasa Indonesia sebagai bahasa utama."
)

EVAL_QUERIES = [
    "Siapa presiden Indonesia pertama?",
    "Jelaskan apa itu fotosintesis dengan bahasa sederhana.",
    "Terjemahkan ke bahasa Inggris: 'Matahari terbit di timur.'",
]

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

def load_jsonl_samples(path: str) -> list[dict]:
    if not os.path.exists(path):
        print(f"[WARN] File {path} tidak ditemukan, skip.")
        return []
    with open(path, "r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]

def apply_logit_mask(model: Any, suppress_ids: set[int]) -> None:
    vocab_size = model.config.vocab_size
    suppress_list = [i for i in suppress_ids if i < vocab_size]

    mask = torch.zeros(vocab_size, dtype=torch.bfloat16)
    mask[suppress_list] = -10000.0

    def forward_hook(module, inputs, outputs):
        if hasattr(outputs, "logits"):
            outputs.logits = outputs.logits + mask.to(outputs.logits.device)
        elif isinstance(outputs, tuple):
            logits = outputs[0]
            outputs = (logits + mask.to(logits.device),) + outputs[1:]
        return outputs

    model.register_forward_hook(forward_hook)
    print(f"  ✅ Logit masking registered untuk {len(suppress_list)} suppressed tokens.")

class TrainingPlotCallback(TrainerCallback):
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
        ax.set_title("Training Loss Curve — T5Gemma2-270M Demo 100 Chats")
        ax.grid(True, alpha=0.3)
        if len(self.losses) >= 10:
            window = 10
            ma = [
                sum(self.losses[max(0, i - window) : i + 1])
                / len(self.losses[max(0, i - window) : i + 1])
                for i in range(len(self.losses))
            ]
            ax.plot(
                self.steps, ma, color="#E74C3C", linewidth=2, label="MA-10", alpha=0.8
            )
            ax.legend()
        plt.tight_layout()
        plt.savefig(self.chart_path, dpi=120)
        plt.close(fig)

class SampleGenerationCallback(TrainerCallback):
    def __init__(
        self,
        tokenizer: PreTrainedTokenizerFast,
        queries: list[str],
        output_dir: str,
        eval_every_n_steps: int = 25,
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
                    add_special_tokens=False,
                    truncation=True,
                    max_length=512,
                )
                enc = {k: v.to(model.device) for k, v in enc.items()}

                out = getattr(model, "generate")(
                    **enc,
                    max_new_tokens=128,
                    do_sample=False,
                    eos_token_id=self._stop_ids,
                )
                _resp_raw = self.tokenizer.decode(out[0], skip_special_tokens=True)
                response: str = (
                    _resp_raw if isinstance(_resp_raw, str) else " ".join(_resp_raw)
                )

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

        print(f"\n[BEHAVIOR EVAL @ step {state.global_step}]")
        for line in lines[3:]:
            if line.startswith("Q:") or line.startswith("A:"):
                print(f"  {line}")

def process_sft_samples(samples, tokenizer):
    # Group turns back to conversation to validate thread completeness
    conversations = []
    current_conv = []
    for obj in samples:
        if not obj.get("input") or not obj.get("target"):
            continue
        if "assistant:" not in obj["input"]:
            if current_conv:
                conversations.append(current_conv)
            current_conv = []
        current_conv.append(obj)
    if current_conv:
        conversations.append(current_conv)

    final_rows = []
    for conv in conversations:
        conv_rows = []
        is_valid = True
        for turn in conv:
            inp_f = format_encoder_from_raw(turn["input"])
            tgt_f = turn["target"].strip() + "<end_of_turn>"

            inp_ids = tokenizer.encode(inp_f, add_special_tokens=False)
            tgt_ids = tokenizer.encode(tgt_f, add_special_tokens=False)

            if len(inp_ids) > MAX_SOURCE_LENGTH or len(tgt_ids) > MAX_TARGET_LENGTH:
                is_valid = False
                break
            conv_rows.append({"input_ids": inp_ids, "labels": tgt_ids})

        if is_valid:
            final_rows.extend(conv_rows)
            
    return final_rows

def main() -> None:
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print(f"Loading Tokenizer from {MODEL_NAME}...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
    assert isinstance(tokenizer, PreTrainedTokenizerFast), (
        "Tokenizer harus PreTrainedTokenizerFast"
    )

    print(f"\nLoading demo datasets...")
    train_samples = load_jsonl_samples(CHAT_TRAIN_FILE)
    val_samples   = load_jsonl_samples(CHAT_VAL_FILE)

    print("Processing train samples...")
    train_rows = process_sft_samples(train_samples, tokenizer)
    print("Processing validation samples...")
    eval_rows  = process_sft_samples(val_samples, tokenizer)

    print(f"Total training SFT rows: {len(train_rows)}")
    print(f"Total validation SFT rows: {len(eval_rows)}")

    train_ds = Dataset.from_list(train_rows)
    eval_ds  = Dataset.from_list(eval_rows)

    # Load Model
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

    if tokenizer.pad_token is None:
        tokenizer.add_special_tokens({"pad_token": tokenizer.eos_token})
        model.resize_token_embeddings(len(tokenizer))

    print(f"\nApplying logit mask for {len(ALL_SUPPRESS_IDS)} tokens...")
    apply_logit_mask(model, ALL_SUPPRESS_IDS)

    # LoRA Config
    lora_config = LoraConfig(
        task_type=TaskType.SEQ_2_SEQ_LM,
        r=32,
        lora_alpha=64,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        lora_dropout=0.05,
    )
    model = get_peft_model(model, lora_config)
    setattr(model.config, "use_cache", False)
    model.print_trainable_parameters()

    apply_logit_mask(model, ALL_SUPPRESS_IDS)

    # Callbacks
    plot_callback = TrainingPlotCallback(output_dir=OUTPUT_DIR)
    sample_callback = SampleGenerationCallback(
        tokenizer=tokenizer,
        queries=EVAL_QUERIES,
        output_dir=OUTPUT_DIR,
        eval_every_n_steps=25,
    )

    data_collator = DataCollatorForSeq2Seq(
        tokenizer=tokenizer, model=model, padding=True
    )

    training_args = Seq2SeqTrainingArguments(
        output_dir=OUTPUT_DIR,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=8,
        learning_rate=5e-5,
        num_train_epochs=3,  # Train for 3 epochs (quick and stable)
        warmup_steps=30,
        weight_decay=0.01,
        lr_scheduler_type="cosine",
        predict_with_generate=True,
        logging_steps=10,
        save_strategy="steps",
        save_steps=25,
        save_total_limit=2,
        eval_strategy="steps",
        eval_steps=25,
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
        callbacks=[
            plot_callback,
            sample_callback,
            EarlyStoppingCallback(early_stopping_patience=10),
        ],
    )

    print("\nStarting SFT Training on 100 Chats...")
    trainer.train()

    # Save
    final_path = os.path.join(OUTPUT_DIR, "final_adapter")
    print(f"\nSaving final SFT adapter to {final_path}...")
    trainer.save_model(final_path)
    tokenizer.save_pretrained(final_path)
    print("✅ SFT Training Selesai!")

if __name__ == "__main__":
    main()
