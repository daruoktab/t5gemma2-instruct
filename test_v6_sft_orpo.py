import os
import re
import torch
import random
import datetime
import gc
import numpy as np
import matplotlib.pyplot as plt
from typing import cast, Any
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
import evaluate

try:
    rouge_metric = evaluate.load("rouge")
    bleu_metric = evaluate.load("bleu")
    exact_match_metric = evaluate.load("exact_match")
    bertscore_metric = evaluate.load("bertscore")
    meteor_metric = evaluate.load("meteor")
except Exception as e:
    print(f"Warning: evaluate metrics not available. Error: {e}")
    rouge_metric = None
    bleu_metric = None
    exact_match_metric = None
    bertscore_metric = None
    meteor_metric = None

# =====================================================================
# 1. KONFIGURASI HYPERPARAMETER (LOKAL TEST)
# =====================================================================
MODEL_NAME = "google/t5gemma-2-270m-270m"
LOAD_IN_4BIT = False
OUTPUT_DIR = "results/t5gemma2-mini-test"

HF_REPO_ID = "daruokta/t5gemma2-indonesia-chat-formatted"
CHAT_CONFIG = "chat_sft"
INDOQA_CONFIG = "indoqa_sft"
ORPO_CONFIG = "chat_orpo"

TRAIN_MODE = "orpo" # can be "sft" or "orpo"

SAMPLE_TRAIN_CHAT = 10
SAMPLE_TRAIN_INDOQA = 10
SAMPLE_TRAIN_ORPO = 10

SAMPLE_VAL_CHAT = 10
SAMPLE_VAL_INDOQA = 10
SAMPLE_EVAL_GENERATION = 2

EVAL_EVERY_N_STEPS = 5

SYSTEM_PROMPT = (
    "Kamu adalah asisten AI yang helpful, santai, dan ramah. "
    "Gunakan Bahasa Indonesia sebagai bahasa utama."
)

MAX_SOURCE_LENGTH = 4096
MAX_TARGET_LENGTH = 1024
NUM_EPOCHS = 3
LEARNING_RATE = 1e-4

PER_DEVICE_TRAIN_BATCH_SIZE = 1
PER_DEVICE_EVAL_BATCH_SIZE = 1
GRADIENT_ACCUMULATION_STEPS = 1
EVAL_ACCUMULATION_STEPS = None

LORA_RANK = 8
LORA_ALPHA = 16
LORA_DROPOUT = 0.0
LORA_TARGET_MODULES = [
    "q_proj",
    "k_proj",
    "v_proj",
    "o_proj",
    "gate_proj",
    "up_proj",
    "down_proj",
]

WARMUP_STEPS = 10
WEIGHT_DECAY = 0.1
LR_SCHEDULER_TYPE = "cosine"
LOGGING_STEPS = 1
SAVE_TOTAL_LIMIT = 2
OPTIM = "paged_adamw_8bit"
LABEL_SMOOTHING_FACTOR = 0.1
NEFTUNE_NOISE_ALPHA = 5.0

GRADIENT_CHECKPOINTING = True
FP16 = False
BF16 = torch.cuda.is_available()
PREDICT_WITH_GENERATE = True
EARLY_STOPPING_PATIENCE = 8

GEN_TEMPERATURE = 0.7
GEN_TOP_P = 0.9
GEN_REPETITION_PENALTY = 1.2

SUPPRESS_BLOCK1 = [6] + list(range(13, 105))
SUPPRESS_BLOCK2 = list(range(256002, 262144))
SUPPRESS_VISION = [255999, 256000, 256001]
ALL_SUPPRESS_IDS = set(SUPPRESS_BLOCK1 + SUPPRESS_BLOCK2 + SUPPRESS_VISION)


# =====================================================================
# UTILITIES
# =====================================================================
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

def load_hf_samples(repo_id: str, config_name: str, split: str, n_samples: int, seed: int = 42) -> list[dict]:
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

                if len(inp_ids) <= MAX_SOURCE_LENGTH and len(tgt_ids) <= MAX_TARGET_LENGTH:
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

            if len(inp_ids) <= MAX_SOURCE_LENGTH and len(tgt_ids) <= MAX_TARGET_LENGTH:
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
        elif isinstance(outputs, tuple) and len(outputs) > 0 and isinstance(outputs[0], torch.Tensor):
            logits = outputs[0]
            outputs = (logits + mask.to(logits.device),) + outputs[1:]
            return outputs
        return outputs

    target_module = None
    if hasattr(model, "lm_head"):
        target_module = model.lm_head
    elif hasattr(model, "base_model") and hasattr(model.base_model, "lm_head"):
        target_module = model.base_model.lm_head
    elif hasattr(model, "base_model") and hasattr(model.base_model, "model") and hasattr(model.base_model.model, "lm_head"):
        target_module = model.base_model.model.lm_head

    if target_module is not None:
        target_module.register_forward_hook(forward_hook)
    else:
        model.register_forward_hook(forward_hook)

# =====================================================================
# CALLBACKS
# =====================================================================
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
            self.train_losses.append(float(logs["loss"]))
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
                    sum(self.train_losses[max(0, i - window) : i + 1]) / len(self.train_losses[max(0, i - window) : i + 1])
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
            
            if len(self.eval_bertscore) > 0 and ax3 is not None:
                ax3.plot(self.eval_steps, self.eval_bertscore, color="#E74C3C", marker="p", linestyle="-", linewidth=2, label="Eval BERTScore")
                ax3.set_xlabel("Steps")
                ax3.set_ylabel("Score (%)")
                ax3.set_title("Semantic Metrics (BERTScore F1)")
                ax3.grid(True, alpha=0.3)
                ax3.legend()
            
            if len(self.eval_perplexity) > 0 and ax4 is not None:
                ax4.plot(self.eval_steps, self.eval_perplexity, color="#34495E", marker="h", linestyle="-", linewidth=2, label="Eval Perplexity")
                ax4.set_xlabel("Steps")
                ax4.set_ylabel("Perplexity")
                ax4.set_title("Model Perplexity Curve")
                ax4.grid(True, alpha=0.3)
                ax4.legend()

        plt.tight_layout()
        plt.savefig(self.chart_path, dpi=120)
        plt.close(fig)

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
        if state.global_step == 0 or state.global_step % self.eval_every_n_steps != 0:
            return
        if model is None:
            return

        if hasattr(FastLanguageModel, "for_inference"):
            FastLanguageModel.for_inference(model)
        else:
            model.eval()

        timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        lines = [f"\n{'=' * 60}", f"Step {state.global_step} | {timestamp}", f"{'=' * 60}"]

        with torch.no_grad():
            input_ids_list = [torch.tensor(s["input_ids"], dtype=torch.long) for s in self.eval_samples]
            pad_id = self.tokenizer.pad_token_id if self.tokenizer.pad_token_id is not None else self._eos_id
            max_len = max(len(x) for x in input_ids_list)
            
            padded_inputs = []
            attention_masks = []
            for x in input_ids_list:
                 pad_len = max_len - len(x)
                 padded_inputs.append(torch.cat([torch.tensor([pad_id] * pad_len, dtype=torch.long), x]))
                 attention_masks.append(torch.cat([torch.zeros(pad_len, dtype=torch.long), torch.ones(len(x), dtype=torch.long)]))

            batch_inputs = torch.stack(padded_inputs).to(device=model.device, dtype=torch.long)
            batch_masks = torch.stack(attention_masks).to(device=model.device, dtype=torch.long)

            outputs = getattr(model, "generate")(
                input_ids=batch_inputs,
                attention_mask=batch_masks,
                max_new_tokens=128,
                do_sample=True,
                temperature=self.temperature,
                top_p=self.top_p,
                repetition_penalty=self.repetition_penalty,
                eos_token_id=self._stop_ids,
                pad_token_id=pad_id,
                bad_words_ids=self.bad_words_ids,
            )

            for idx, sample in enumerate(self.eval_samples):
                raw_query = self.tokenizer.decode(sample["input_ids"], skip_special_tokens=True)
                raw_target = self.tokenizer.decode(sample["labels"], skip_special_tokens=True)
                raw_response = self.tokenizer.decode(outputs[idx], skip_special_tokens=True)
                
                query = raw_query.strip() if isinstance(raw_query, str) else "".join(raw_query).strip()
                target = raw_target.strip() if isinstance(raw_target, str) else "".join(raw_target).strip()
                response = raw_response.strip() if isinstance(raw_response, str) else "".join(raw_response).strip()

                words = response.split()
                is_repetitive = len(set(words)) < max(1, len(words) * 0.3) if words else True
                flag = " ⚠️ REPETITIVE" if is_repetitive else " ✅"

                lines.append(f"\nQ: {query}")
                lines.append(f"Expected Target: {target}")
                lines.append(f"Model Response: {response}{flag}")

        if hasattr(FastLanguageModel, "for_training"):
            FastLanguageModel.for_training(model)
        else:
            model.train()

        with open(self.log_path, "a", encoding="utf-8") as f:
            f.write("\n".join(lines) + "\n")

        print(f"\n[BEHAVIOR EVAL @ step {state.global_step}]")
        for line in lines[3:]:
            if line.startswith("Q:") or line.startswith("Model Response:") or line.startswith("Expected Target:"):
                print(f"  {line}")


class SelectiveLabelSmoother:
    def __init__(self, epsilon, suppress_ids):
        self.epsilon = epsilon
        self.suppress_ids = suppress_ids

    def __call__(self, model_output, labels, shift_labels=False):
        if isinstance(model_output, dict) and "logits" in model_output:
            logits = model_output["logits"]
        elif isinstance(model_output, tuple):
            logits = model_output[1] if len(model_output) > 1 else model_output[0].logits
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
        chunk_size = 2048  # Adjustable for VRAM vs Speed tradeoff
        
        total_loss = torch.tensor(0.0, device=logits.device)
        
        for i in range(0, num_active, chunk_size):
            chunk_logits = active_logits[i : i + chunk_size]
            chunk_labels = active_labels[i : i + chunk_size]
            
            log_probs = torch.nn.functional.log_softmax(chunk_logits, dim=-1)
            
            nll_loss = -log_probs.gather(dim=-1, index=chunk_labels.unsqueeze(-1)).squeeze(-1)
            valid_log_probs = log_probs * valid_mask.to(log_probs.dtype)
            smooth_loss = -valid_log_probs.sum(dim=-1) / num_valid_tokens
            
            token_losses = (1.0 - self.epsilon) * nll_loss + self.epsilon * smooth_loss
            total_loss += token_losses.sum()
            
            # Clean up chunk variables to free VRAM immediately
            del chunk_logits, chunk_labels, log_probs, nll_loss, valid_log_probs, smooth_loss, token_losses
            
        return total_loss / num_active

class CustomSeq2SeqTrainer(Seq2SeqTrainer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.model_accepts_loss_kwargs = False
        if self.args.label_smoothing_factor > 0:
            self.label_smoother = SelectiveLabelSmoother(
                epsilon=self.args.label_smoothing_factor,
                suppress_ids=ALL_SUPPRESS_IDS,
            )

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None, **kwargs):
        labels = inputs.get("labels")
        outputs = model(**inputs)

        smoother = getattr(self, "label_smoother", None)
        if smoother is not None and labels is not None:
            loss = smoother(outputs, labels)
        else:
            if isinstance(outputs, dict) and "logits" in outputs:
                logits = outputs["logits"]
            elif isinstance(outputs, tuple):
                logits = outputs[1] if len(outputs) > 1 else outputs[0].logits
            else:
                logits = outputs.logits
            loss_fct = torch.nn.CrossEntropyLoss(ignore_index=-100, reduction="mean")
            loss = loss_fct(logits.view(-1, logits.size(-1)), labels.view(-1))
        
        return (loss, outputs) if return_outputs else loss

    def evaluate(
        self,
        eval_dataset=None,
        ignore_keys=None,
        metric_key_prefix="eval",
        **gen_kwargs,
    ):
        if self.model is not None:
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

        if self.model is not None:
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

class CustomORPOTrainer(CustomSeq2SeqTrainer):
    def __init__(self, beta=0.1, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.beta = beta

    def get_batch_logps(self, logits, labels, average_log_prob: bool = False):
        if logits.shape[:-1] != labels.shape:
            raise ValueError("Logits and labels must have the same shape.")
        labels = labels.clone()
        loss_mask = labels != -100
        labels[labels == -100] = 0
        per_token_logps = torch.gather(logits.log_softmax(-1), dim=2, index=labels.unsqueeze(2)).squeeze(2)
        if average_log_prob:
            return (per_token_logps * loss_mask).sum(-1) / loss_mask.sum(-1)
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

        chosen_logps = self.get_batch_logps(chosen_logits, chosen_labels)
        rejected_logps = self.get_batch_logps(rejected_logits, rejected_labels)

        chosen_log_odds = chosen_logps - torch.log1p(-torch.exp(chosen_logps) + 1e-6)
        rejected_log_odds = rejected_logps - torch.log1p(-torch.exp(rejected_logps) + 1e-6)

        log_odds_margin = chosen_log_odds - rejected_log_odds
        or_loss = -F.logsigmoid(log_odds_margin).mean()

        sft_loss = super().compute_loss(model, {"input_ids": input_ids, "attention_mask": attention_mask, "labels": chosen_labels}, return_outputs=False)

        loss = sft_loss + self.beta * or_loss
        return (loss, chosen_outputs) if return_outputs else loss

class GrokAdEMAMix(torch.optim.Optimizer):
    def __init__(self, params, lr=3e-5, betas=(0.9, 0.999), beta3=0.9999, weight_decay=0.05, grok_alpha=2.0, grok_lamb=0.98):
        defaults = dict(lr=lr, betas=betas, beta3=beta3, weight_decay=weight_decay, grok_alpha=grok_alpha, grok_lamb=grok_lamb)
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

                state["grok_slow_grad"].mul_(grok_lamb).add_(grad, alpha=1.0 - grok_lamb)
                filtered_grad = grad.clone()
                filtered_grad.add_(state["grok_slow_grad"], alpha=grok_alpha)

                if weight_decay != 0:
                    p.data.mul_(1.0 - lr * weight_decay)

                m, v, n = state["m"], state["v"], state["n"]
                m.mul_(beta1).add_(filtered_grad, alpha=1.0 - beta1)
                v.mul_(beta2).addcmul_(filtered_grad, filtered_grad, value=1.0 - beta2)
                n.mul_(beta3).add_(filtered_grad, alpha=1.0 - beta3)

                bias_correction1 = 1.0 - beta1**step
                bias_correction2 = 1.0 - beta2**step
                bias_correction3 = 1.0 - beta3**step

                denom = (v.sqrt() / (bias_correction2**0.5)).add_(1e-8)
                step_update = (m / bias_correction1 + 0.1 * n / bias_correction3) / denom
                p.data.add_(step_update, alpha=-lr)
        return loss

def compute_metrics(eval_preds, tokenizer):
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
            rouge = cast(Any, rouge_metric)
            result = rouge.compute(predictions=decoded_preds, references=decoded_labels, use_stemmer=False)
            if result is not None:
                for key, value in result.items():
                    metrics[key] = value * 100
        except Exception as e:
            print(f"Error during ROUGE metric calculation: {e}")

    if bleu_metric is not None:
        try:
            bleu = cast(Any, bleu_metric)
            formatted_labels = [[label] for label in decoded_labels]
            bleu_result = bleu.compute(predictions=decoded_preds, references=formatted_labels)
            if bleu_result is not None and "bleu" in bleu_result:
                metrics["bleu"] = bleu_result["bleu"] * 100
        except Exception as e:
            print(f"Error during BLEU metric calculation: {e}")

    # Exact Match
    if exact_match_metric is not None:
        try:
            em_result = cast(Any, exact_match_metric).compute(
                predictions=decoded_preds, references=decoded_labels
            )
            if em_result is not None and "exact_match" in em_result:
                metrics["exact_match"] = em_result["exact_match"] * 100
        except Exception as e:
            print(f"Error during Exact Match calculation: {e}")

    # BERTScore
    if bertscore_metric is not None:
        try:
            # Using google/embeddinggemma-300m and forcing num_layers=12 to avoid KeyError
            bertscore_result = cast(Any, bertscore_metric).compute(
                predictions=decoded_preds,
                references=decoded_labels,
                model_type="google/embeddinggemma-300m",
                num_layers=12,
                lang="id"
            )
            if bertscore_result is not None and "f1" in bertscore_result:
                metrics["bertscore_f1"] = np.mean(bertscore_result["f1"]) * 100
        except Exception as e:
            print(f"Error during BERTScore calculation: {e}")

    # METEOR
    if meteor_metric is not None:
        try:
            meteor_result = cast(Any, meteor_metric).compute(
                predictions=decoded_preds, references=decoded_labels
            )
            if meteor_result is not None and "meteor" in meteor_result:
                metrics["meteor"] = meteor_result["meteor"] * 100
        except Exception as e:
            print(f"Error during METEOR metric calculation: {e}")

    return metrics

def preprocess_logits_for_metrics(logits, labels):
    if isinstance(logits, tuple):
        logits = logits[0]
    return logits.argmax(dim=-1)

def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    print(f"Loading Tokenizer from {MODEL_NAME}...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
    assert isinstance(tokenizer, PreTrainedTokenizerFast)

    train_chat_samples = load_hf_samples(HF_REPO_ID, CHAT_CONFIG, "train", SAMPLE_TRAIN_CHAT)
    train_indoqa_samples = load_hf_samples(HF_REPO_ID, INDOQA_CONFIG, "train", SAMPLE_TRAIN_INDOQA)
    val_chat_samples = load_hf_samples(HF_REPO_ID, CHAT_CONFIG, "validation", SAMPLE_VAL_CHAT)
    val_indoqa_samples = load_hf_samples(HF_REPO_ID, INDOQA_CONFIG, "validation", SAMPLE_VAL_INDOQA)

    if TRAIN_MODE == "orpo":
        print("Processing ORPO training rows...")
        train_orpo_samples = load_hf_samples(HF_REPO_ID, ORPO_CONFIG, "train", SAMPLE_TRAIN_ORPO)
        train_rows = process_orpo_rows(train_orpo_samples, tokenizer)
        val_rows = process_sft_rows(val_chat_samples, tokenizer, is_chat=True) + process_sft_rows(val_indoqa_samples, tokenizer, is_chat=False)
    else:
        print("Processing SFT training rows...")
        train_rows = process_sft_rows(train_chat_samples, tokenizer, is_chat=True) + process_sft_rows(train_indoqa_samples, tokenizer, is_chat=False)
        print("Processing SFT validation rows...")
        val_rows = process_sft_rows(val_chat_samples, tokenizer, is_chat=True) + process_sft_rows(val_indoqa_samples, tokenizer, is_chat=False)

    random.seed(42)
    random.shuffle(train_rows)
    random.shuffle(val_rows)

    print(f"Total {TRAIN_MODE.upper()} Training rows: {len(train_rows)}")
    print(f"Total SFT Validation rows: {len(val_rows)}")

    train_ds = Dataset.from_list(train_rows)
    eval_ds = Dataset.from_list(val_rows)

    n_eval_gen = min(len(val_rows), SAMPLE_EVAL_GENERATION)
    eval_generation_samples = val_rows[:n_eval_gen]

    # === LOAD MODEL ===
    print(f"\nLoading Model from {MODEL_NAME} using Unsloth...")
    model, tokenizer_unsloth = FastLanguageModel.from_pretrained(
        model_name=MODEL_NAME,
        max_seq_length=MAX_SOURCE_LENGTH,
        load_in_4bit=LOAD_IN_4BIT,
        trust_remote_code=True,
    )

    model.config.max_length = None
    if hasattr(model, "generation_config") and model.generation_config is not None:
        model.generation_config.max_length = None

    if getattr(model.config, "decoder_start_token_id", None) is None:
        model.config.decoder_start_token_id = tokenizer.bos_token_id
        
    if tokenizer.pad_token is None:
        tokenizer.add_special_tokens({"pad_token": tokenizer.eos_token})
        model.resize_token_embeddings(len(tokenizer))

    print(f"\nApplying logit mask for {len(ALL_SUPPRESS_IDS)} tokens...")
    apply_logit_mask(model, ALL_SUPPRESS_IDS)

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

    if hasattr(model, "prepare_decoder_input_ids_from_labels"):
        orig_fn = model.prepare_decoder_input_ids_from_labels
        def compatible_prepare(labels=None, input_ids=None, *args, **kwargs):
            target_tensor = labels if labels is not None else input_ids
            return orig_fn(target_tensor, *args, **kwargs)
        model.prepare_decoder_input_ids_from_labels = compatible_prepare

    bad_words_ids = [[id_] for id_ in ALL_SUPPRESS_IDS if id_ < model.config.vocab_size]

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

    if TRAIN_MODE == "orpo":
        data_collator = ORPODataCollatorForSeq2Seq(tokenizer=tokenizer, model=model, padding=True)
    else:
        data_collator = DataCollatorForSeq2Seq(tokenizer=tokenizer, model=model, padding=True)

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
        optim=OPTIM,
        label_smoothing_factor=LABEL_SMOOTHING_FACTOR,
        neftune_noise_alpha=NEFTUNE_NOISE_ALPHA,
        report_to="none",
        fp16=FP16,
        bf16=BF16,
        gradient_checkpointing=GRADIENT_CHECKPOINTING,
        generation_max_length=MAX_TARGET_LENGTH,
        remove_unused_columns=False if TRAIN_MODE == "orpo" else True,
        push_to_hub=False,
    )

    optimizer = GrokAdEMAMix(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY, grok_alpha=2.0, grok_lamb=0.98)
    num_update_steps_per_epoch = max(1, len(train_ds) // (PER_DEVICE_TRAIN_BATCH_SIZE * GRADIENT_ACCUMULATION_STEPS))
    max_steps = num_update_steps_per_epoch * NUM_EPOCHS
    lr_scheduler = get_scheduler(name=LR_SCHEDULER_TYPE, optimizer=optimizer, num_warmup_steps=WARMUP_STEPS, num_training_steps=max_steps)

    trainer_class = CustomORPOTrainer if TRAIN_MODE == "orpo" else CustomSeq2SeqTrainer
    
    trainer_kwargs = {
        "model": model,
        "args": training_args,
        "train_dataset": train_ds,
        "eval_dataset": eval_ds,
        "data_collator": data_collator,
        "compute_metrics": lambda eval_preds: compute_metrics(eval_preds, tokenizer),
        "preprocess_logits_for_metrics": None if PREDICT_WITH_GENERATE else preprocess_logits_for_metrics,
        "optimizers": (optimizer, lr_scheduler),
        "callbacks": [plot_callback, sample_callback],
    }
    
    if TRAIN_MODE == "orpo":
        trainer_kwargs["beta"] = 0.1

    trainer = trainer_class(**trainer_kwargs)

    # ==============================================================================
    # MEKANISME AUTO-RESUME CHECKPOINT LOKAL (TEST)
    # ==============================================================================
    resume_from_checkpoint = None
    if os.path.exists(OUTPUT_DIR):
        checkpoints = [d for d in os.listdir(OUTPUT_DIR) if d.startswith("checkpoint-")]
        if checkpoints:
            resume_from_checkpoint = True
            print(f"\n✅ Ditemukan {len(checkpoints)} checkpoint lokal. Melanjutkan pelatihan!")
        else:
            print("\n⚠️ Belum ada checkpoint. Memulai dari awal.")

    print("\nStarting Clean SFT on Local Test (with Unsloth)...")
    trainer.train(resume_from_checkpoint=resume_from_checkpoint)
    print("SFT Training DONE.")

if __name__ == '__main__':
    import multiprocessing
    multiprocessing.freeze_support()
    main()