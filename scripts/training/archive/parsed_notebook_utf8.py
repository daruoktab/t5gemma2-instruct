--------------------------------------------------
# Install library yang diperlukan (uncomment jika dijalankan di Google Colab atau environment baru)
# !pip install -q transformers datasets peft accelerate matplotlib ipywidgets -U
--------------------------------------------------
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
)
from peft import LoraConfig, get_peft_model, TaskType
from typing import cast, Any

# Gunakan inline backend untuk matplotlib di Jupyter Notebook
%matplotlib inline
--------------------------------------------------
# =====================================================================
# 1. KONFIGURASI HYPERPARAMETER (MUDAH DIUBAH)
# =====================================================================

# MODEL CONFIG
# Ganti ke "google/t5gemma-2-4b-4b" jika ingin melatih model 4B di Cloud/GPU besar
# Ganti ke "google/t5gemma-2-270m-270m" jika ingin melatih model versi ringan (270M)
MODEL_NAME = "google/t5gemma-2-270m-270m"
OUTPUT_DIR = "results/t5gemma2-clean-sft"

# DATASET CONFIG (Hugging Face)
HF_REPO_ID = "daruokta/t5gemma2-indonesia-chat-formatted"
CHAT_CONFIG = "chat_sft"
INDOQA_CONFIG = "indoqa_sft"

# SAMPLE SIZES (Set ke 0 untuk mengambil seluruh data)
SAMPLE_TRAIN_CHAT = 700      # Jumlah sampel chat untuk training
SAMPLE_TRAIN_INDOQA = 300    # Jumlah sampel QA untuk training
SAMPLE_VAL_CHAT = 100        # Jumlah sampel chat untuk validation/eval_loss
SAMPLE_VAL_INDOQA = 50       # Jumlah sampel QA untuk validation/eval_loss

# GENERATION EVALUATION CONFIG
SAMPLE_EVAL_GENERATION = 10  # Jumlah sampel dari validation set untuk pengujian teks generasi di log
EVAL_EVERY_N_STEPS = 25      # Jalankan evaluasi loss & generasi setiap N steps

# SYSTEM PROMPT FALLBACK
SYSTEM_PROMPT = (
    "Kamu adalah asisten AI yang helpful, santai, dan ramah. "
    "Gunakan Bahasa Indonesia sebagai bahasa utama."
)

# TRAINING SPECS
MAX_SOURCE_LENGTH = 1024
MAX_TARGET_LENGTH = 512
NUM_EPOCHS = 3
LEARNING_RATE = 5e-5
BATCH_SIZE = 1
GRAD_ACCUMULATION = 8
LORA_RANK = 32
LORA_ALPHA = 64

# Token IDs yang harus di-suppress (unused + vision)
SUPPRESS_BLOCK1 = list(range(6, 105))         # <unused0>–<unused98>
SUPPRESS_BLOCK2 = list(range(256002, 262144))  # <unused100>–<unused6241>
SUPPRESS_VISION = [255999, 256000, 256001]     # <end_of_image>, <image_soft_token>
ALL_SUPPRESS_IDS = set(SUPPRESS_BLOCK1 + SUPPRESS_BLOCK2 + SUPPRESS_VISION)
--------------------------------------------------
def format_encoder_from_raw(raw_input: str) -> str:
    system_match = re.search(r"^system:\\s*(.*?)(?=\\nuser:)", raw_input, re.DOTALL)
    system = system_match.group(1).strip() if system_match else SYSTEM_PROMPT
    
    if system_match:
        raw_input = raw_input[system_match.end():].strip()
        
    parts = re.split(r"\\n(user:|assistant:)\\s*", "\\n" + raw_input)
    formatted = ""
    is_first_user = True
    
    for i in range(1, len(parts), 2):
        role = parts[i].replace(":", "").strip()
        content = parts[i+1].strip()
        if not content:
            continue
            
        if role == "user":
            formatted += "<start_of_turn>user\\n"
            if is_first_user and system:
                formatted += system + "\\n\\n"
                is_first_user = False
            formatted += content + "<end_of_turn>\\n"
        elif role == "assistant":
            formatted += "<start_of_turn>model\\n"
            formatted += content + "<end_of_turn>\\n"
            
    formatted += "<start_of_turn>model\\n"
    return formatted
--------------------------------------------------
def load_hf_samples(repo_id: str, config_name: str, split: str, n_samples: int, seed: int = 42) -> list[dict]:
    """
    Mendownload dataset dari Hugging Face Hub untuk split tertentu dan mengambil sampel sejumlah n_samples.
    """
    print(f"Mengunduh dataset '{config_name}' ({split}) dari {repo_id}...")
    try:
        ds = load_dataset(repo_id, config_name, split=split)
        samples = [{"input": row["input"], "target": row["target"]} for row in ds]
        
        if n_samples > 0 and len(samples) > n_samples:
            random.seed(seed)
            return random.sample(samples, n_samples)
        return samples
    except Exception as e:
        print(f"[ERROR] Gagal mengunduh dataset {config_name} ({split}): {e}")
        return []
--------------------------------------------------
# ==========================================
# NON-DESTRUCTIVE LOGIT MASKING
# ==========================================
def apply_logit_mask(model: Any, suppress_ids: set[int]) -> None:
    """
    Menerapkan logit masking secara dinamis lewat PyTorch forward hook.
    Cara ini tidak memodifikasi bobot model (non-destructive) dan aman
    untuk training maupun generation.
    """
    vocab_size = model.config.vocab_size
    suppress_list = [i for i in suppress_ids if i < vocab_size]
    
    # Gunakan nilai negatif besar yang kompatibel dengan bfloat16 (-10000.0)
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
--------------------------------------------------
class TrainingPlotCallback(TrainerCallback):
    def __init__(self, output_dir: str) -> None:
        self.output_dir = output_dir
        self.train_steps: list[int] = []
        self.train_losses: list[float] = []
        self.eval_steps: list[int] = []
        self.eval_losses: list[float] = []
        self.chart_path = os.path.join(output_dir, "training_chart.png")
        
    def on_log(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        logs: dict[str, float] | None = None,
        **kwargs: Any
    ) -> None:
        if logs is None:
            return
        if "loss" in logs:
            self.train_steps.append(state.global_step)
            self.train_losses.append(float(logs["loss"]))
        if "eval_loss" in logs:
            self.eval_steps.append(state.global_step)
            self.eval_losses.append(float(logs["eval_loss"]))
        self._save_chart()
        
    def _save_chart(self) -> None:
        if len(self.train_steps) < 2 and len(self.eval_steps) < 1:
            return
        fig, ax = plt.subplots(figsize=(10, 4))
        
        # Plot training loss
        if self.train_losses:
            ax.plot(self.train_steps, self.train_losses, color="#4A90D9", linewidth=1.5, label="Train Loss")
            if len(self.train_losses) >= 10:
                window = 10
                ma = [sum(self.train_losses[max(0, i-window):i+1]) / len(self.train_losses[max(0, i-window):i+1]) for i in range(len(self.train_losses))]
                ax.plot(self.train_steps, ma, color="#E74C3C", linewidth=2, label="Train Loss (MA-10)", alpha=0.8)
                
        # Plot eval loss
        if self.eval_losses:
            ax.plot(self.eval_steps, self.eval_losses, color="#2ECC71", marker="o", linestyle="--", linewidth=1.5, label="Eval Loss")
            
        ax.set_xlabel("Steps")
        ax.set_ylabel("Loss")
        ax.set_title("Training & Evaluation Loss Curve")
        ax.grid(True, alpha=0.3)
        ax.legend()
        plt.tight_layout()
        plt.savefig(self.chart_path, dpi=120)
        plt.show()
        plt.close(fig)
--------------------------------------------------
class SampleGenerationCallback(TrainerCallback):
    def __init__(
        self,
        tokenizer: PreTrainedTokenizerFast,
        eval_samples: list[dict],
        output_dir: str,
        eval_every_n_steps: int = 50
    ) -> None:
        self.tokenizer = tokenizer
        self.eval_samples = eval_samples
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
        **kwargs: Any
    ) -> None:
        if state.global_step == 0 or state.global_step % self.eval_every_n_steps != 0:
            return
        if model is None:
            return
            
        model.eval()
        timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        lines = [
            f"\\n{'='*60}",
            f"Step {state.global_step} | {timestamp}",
            f"{'='*60}"
        ]
        
        with torch.no_grad():
            for sample in self.eval_samples:
                input_ids = torch.tensor([sample["input_ids"]]).to(model.device)
                
                out = getattr(model, "generate")(
                    input_ids=input_ids,
                    max_new_tokens=128,
                    do_sample=False,
                    eos_token_id=self._stop_ids
                )
                
                query = self.tokenizer.decode(sample["input_ids"], skip_special_tokens=True).strip()
                target = self.tokenizer.decode(sample["labels"], skip_special_tokens=True).strip()
                response = self.tokenizer.decode(out[0], skip_special_tokens=True).strip()
                
                words = response.split()
                is_repetitive = len(set(words)) < max(1, len(words) * 0.3) if words else True
                flag = " ⚠️ REPETITIVE" if is_repetitive else " ✅"
                
                lines.append(f"\\nQ: {query[:200]}...")
                lines.append(f"Expected Target: {target[:200]}...")
                lines.append(f"Model Response: {response[:300]}{flag}")
                
        model.train()
        with open(self.log_path, "a", encoding="utf-8") as f:
            f.write("\\n".join(lines) + "\\n")
            
        print(f"\\n[BEHAVIOR EVAL @ step {state.global_step}]")
        for line in lines[3:]:
            if line.startswith("Q:") or line.startswith("Model Response:") or line.startswith("Expected Target:"):
                print(f"  {line}")
--------------------------------------------------
os.makedirs(OUTPUT_DIR, exist_ok=True)

print(f"Loading Tokenizer from {MODEL_NAME}...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
assert isinstance(tokenizer, PreTrainedTokenizerFast), "Tokenizer harus PreTrainedTokenizerFast"

# Load datasets dari Hugging Face Hub (Split Train & Validation terpisah)
print(f"\\nLoading training & validation datasets...")
train_chat_samples = load_hf_samples(HF_REPO_ID, CHAT_CONFIG, "train", SAMPLE_TRAIN_CHAT)
train_indoqa_samples = load_hf_samples(HF_REPO_ID, INDOQA_CONFIG, "train", SAMPLE_TRAIN_INDOQA)

val_chat_samples = load_hf_samples(HF_REPO_ID, CHAT_CONFIG, "validation", SAMPLE_VAL_CHAT)
val_indoqa_samples = load_hf_samples(HF_REPO_ID, INDOQA_CONFIG, "validation", SAMPLE_VAL_INDOQA)

# Helper function untuk memproses sampel percakapan ke baris SFT
def process_sft_rows(samples, is_chat=True):
    rows = []
    if is_chat:
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
                rows.extend(conv_rows)
    else: 
        for obj in samples:
            inp_f = format_encoder_from_raw(obj.get("input", ""))
            tgt_f = obj.get("target", "").strip() + "<end_of_turn>"
            
            inp_ids = tokenizer.encode(inp_f, add_special_tokens=False)
            tgt_ids = tokenizer.encode(tgt_f, add_special_tokens=False)
            
            if len(inp_ids) <= MAX_SOURCE_LENGTH and len(tgt_ids) <= MAX_TARGET_LENGTH:
                rows.append({"input_ids": inp_ids, "labels": tgt_ids})
    return rows

print("Processing training rows...")
train_rows = process_sft_rows(train_chat_samples, is_chat=True) + process_sft_rows(train_indoqa_samples, is_chat=False)
print("Processing validation rows...")
val_rows = process_sft_rows(val_chat_samples, is_chat=True) + process_sft_rows(val_indoqa_samples, is_chat=False)

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
print(f"Mengambil {n_eval_gen} sampel validasi untuk pencatatan evaluasi kualitatif.")
--------------------------------------------------
# Load Model
print(f"\\nLoading Model from {MODEL_NAME}...")
model = AutoModelForSeq2SeqLM.from_pretrained(
    MODEL_NAME,
    trust_remote_code=True,
    dtype=torch.bfloat16,
    device_map="auto"
)

if getattr(model.config, "decoder_start_token_id", None) is None:
    model.config.decoder_start_token_id = tokenizer.bos_token_id
    print(f"  Set decoder_start_token_id = {model.config.decoder_start_token_id}")

# Set pad token dan resize jika diperlukan
if tokenizer.pad_token is None:
    tokenizer.add_special_tokens({"pad_token": tokenizer.eos_token})
    model.resize_token_embeddings(len(tokenizer))

# Logit Masking (Dinamis, non-destructive!)
print(f"\\nApplying logit mask for {len(ALL_SUPPRESS_IDS)} tokens...")
apply_logit_mask(model, ALL_SUPPRESS_IDS)

# LoRA Config
lora_config = LoraConfig(
    task_type=TaskType.SEQ_2_SEQ_LM,
    r=LORA_RANK,
    lora_alpha=LORA_ALPHA,
    target_modules=[
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj"
    ],
    lora_dropout=0.05,
)
model = get_peft_model(model, lora_config)
setattr(model.config, "use_cache", False)
model.print_trainable_parameters()

# Re-apply logit mask setelah dibungkus PEFT
apply_logit_mask(model, ALL_SUPPRESS_IDS)
--------------------------------------------------
# Callbacks
plot_callback = TrainingPlotCallback(output_dir=OUTPUT_DIR)
sample_callback = SampleGenerationCallback(
    tokenizer=tokenizer,
    eval_samples=eval_generation_samples,
    output_dir=OUTPUT_DIR,
    eval_every_n_steps=EVAL_EVERY_N_STEPS
)

data_collator = DataCollatorForSeq2Seq(tokenizer=tokenizer, model=model, padding=True)

training_args = Seq2SeqTrainingArguments(
    output_dir=OUTPUT_DIR,
    per_device_train_batch_size=BATCH_SIZE,
    gradient_accumulation_steps=GRAD_ACCUMULATION,
    learning_rate=LEARNING_RATE,
    num_train_epochs=NUM_EPOCHS,
    warmup_steps=50,
    weight_decay=0.01,
    lr_scheduler_type="cosine",
    predict_with_generate=True,
    logging_steps=10,
    save_strategy="steps",
    save_steps=EVAL_EVERY_N_STEPS,
    save_total_limit=2,
    eval_strategy="steps",
    eval_steps=EVAL_EVERY_N_STEPS,
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

print("\\nStarting Clean SFT on Cloud/Notebook...")
trainer.train()
--------------------------------------------------
# Save
final_path = os.path.join(OUTPUT_DIR, "final_adapter")
print(f"\\nSaving final SFT adapter to {final_path}...")
trainer.save_model(final_path)
tokenizer.save_pretrained(final_path)
print("✅ Clean SFT Selesai!")
