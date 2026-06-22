import os
import re
import torch
import random
import datetime
import gc
from datasets import Dataset, load_dataset
from transformers import (
    AutoTokenizer,
    PreTrainedTokenizerFast,
    Seq2SeqTrainingArguments,
    DataCollatorForSeq2Seq,
)
from unsloth import FastLanguageModel

# === KONFIGURASI HYPERPARAMETER (MINI TEST) ===
MODEL_NAME = "google/t5gemma-2-270m-270m"
LOAD_IN_4BIT = True
OUTPUT_DIR = "results/t5gemma2-mini-test"

HF_REPO_ID = "daruokta/t5gemma2-indonesia-chat-formatted"
CHAT_CONFIG = "chat_sft"
SAMPLE_TRAIN_CHAT = 1  # SFT pakai 1 data aja

MAX_SOURCE_LENGTH = 512
MAX_TARGET_LENGTH = 512
NUM_EPOCHS = 1
LEARNING_RATE = 1e-4

PER_DEVICE_TRAIN_BATCH_SIZE = 1
GRADIENT_ACCUMULATION_STEPS = 1
LORA_RANK = 8
LORA_ALPHA = 16
LORA_TARGET_MODULES = ["q_proj", "v_proj"]
LORA_DROPOUT = 0.0

# SUPPRESS IDS
SUPPRESS_BLOCK1 = [6] + list(range(13, 105))
SUPPRESS_VISION = [255999, 256000, 256001]
ALL_SUPPRESS_IDS = set(SUPPRESS_BLOCK1 + SUPPRESS_VISION)

SYSTEM_PROMPT = "Kamu adalah asisten AI yang helpful, santai, dan ramah."

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

def load_hf_samples(repo_id: str, config_name: str, split: str, n_samples: int) -> list[dict]:
    print(f"Mengunduh dataset '{config_name}' ({split})...")
    ds = load_dataset(repo_id, config_name, split=split)
    samples = []
    for row in ds:
        item = {"input": row["input"], "target": row["target"]}
        if "chat_idx" in row:
            item["chat_idx"] = row["chat_idx"]
        if "turn_idx" in row:
            item["turn_idx"] = row["turn_idx"]
        samples.append(item)
        if len(samples) >= n_samples:
            break
    return samples

def process_sft_rows(samples, tokenizer: PreTrainedTokenizerFast, is_chat=True):
    rows = []
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


def main():
    print(f"Loading Tokenizer from {MODEL_NAME}...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)

    train_chat_samples = load_hf_samples(HF_REPO_ID, CHAT_CONFIG, "train", SAMPLE_TRAIN_CHAT)
    train_rows = process_sft_rows(train_chat_samples, tokenizer, is_chat=True)
    train_ds = Dataset.from_list(train_rows)

    print(f"Total SFT Training rows: {len(train_ds)}")

    # === LOAD MODEL ===
    print(f"\nLoading Model {MODEL_NAME} (SDPA otomatis)...")
    model, tokenizer_unsloth = FastLanguageModel.from_pretrained(
        model_name = MODEL_NAME,
        max_seq_length = MAX_SOURCE_LENGTH,
        load_in_4bit = LOAD_IN_4BIT,
        trust_remote_code = True,
        # attn_implementation="sdpa" # Dihapus karena Unsloth/HF di PyTorch 2 otomatis pakai SDPA
    )

    model.config.max_length = None
    if hasattr(model, "generation_config") and model.generation_config is not None:
        model.generation_config.max_length = None

    if getattr(model.config, "decoder_start_token_id", None) is None:
        model.config.decoder_start_token_id = tokenizer.bos_token_id
        print(f"Set decoder_start_token_id = {model.config.decoder_start_token_id}")

    if tokenizer.pad_token is None:
        tokenizer.add_special_tokens({"pad_token": tokenizer.eos_token})
        model.resize_token_embeddings(len(tokenizer))

    print("Applying LoRA...")
    model = FastLanguageModel.get_peft_model(
        model,
        r = LORA_RANK,
        lora_alpha = LORA_ALPHA,
        target_modules = LORA_TARGET_MODULES,
        lora_dropout = LORA_DROPOUT,
        bias = "none",
        use_gradient_checkpointing = False,
        random_state = 3407,
    )

    getattr(FastLanguageModel, "for_training")(model)
    model.config.use_cache = False

    if hasattr(model, "prepare_decoder_input_ids_from_labels"):
        orig_fn = model.prepare_decoder_input_ids_from_labels
        def compatible_prepare(labels=None, input_ids=None, *args, **kwargs):
            target_tensor = labels if labels is not None else input_ids
            return orig_fn(target_tensor, *args, **kwargs)
        model.prepare_decoder_input_ids_from_labels = compatible_prepare

    # === TRAINING SFT (Mini Test) ===
    print("\n--- TEST: SFTTrainer (via Seq2SeqTrainer) ---")
    from transformers import Seq2SeqTrainer
    data_collator = DataCollatorForSeq2Seq(tokenizer=tokenizer, model=model, padding=True)

    sft_args = Seq2SeqTrainingArguments(
        output_dir=OUTPUT_DIR + "/sft",
        per_device_train_batch_size=PER_DEVICE_TRAIN_BATCH_SIZE,
        gradient_accumulation_steps=GRADIENT_ACCUMULATION_STEPS,
        max_steps=2,
        learning_rate=LEARNING_RATE,
        report_to="none",
        logging_steps=1,
        remove_unused_columns=False,
    )

    sft_trainer = Seq2SeqTrainer(
        model=model,
        args=sft_args,
        train_dataset=train_ds,
        data_collator=data_collator,
    )
    sft_trainer.train()
    print("SFT Training DONE.")

    # === TRAINING ORPO (Dummy Data) ===
    print("\n--- TEST: ORPOTrainer (Dummy Data) ---")
    from trl.experimental.orpo import ORPOConfig, ORPOTrainer

    # Dummy ORPO Data: requires "prompt", "chosen", "rejected"
    orpo_dummy_data = Dataset.from_list([
        {
            "prompt": "<start_of_turn>user\nBerikan aku ide resep sarapan.<end_of_turn>\n<start_of_turn>model\n",
            "chosen": "Tentu, bagaimana kalau telur dadar dengan bayam dan keju? Sangat sehat dan cepat dibuat.<end_of_turn>",
            "rejected": "Makan saja batu.<end_of_turn>"
        },
        {
            "prompt": "<start_of_turn>user\nApa ibu kota Prancis?<end_of_turn>\n<start_of_turn>model\n",
            "chosen": "Ibu kota Prancis adalah Paris.<end_of_turn>",
            "rejected": "Saya tidak tahu.<end_of_turn>"
        }
    ])

    orpo_args = ORPOConfig(
        output_dir=OUTPUT_DIR + "/orpo",
        per_device_train_batch_size=PER_DEVICE_TRAIN_BATCH_SIZE,
        gradient_accumulation_steps=GRADIENT_ACCUMULATION_STEPS,
        max_steps=2,
        learning_rate=LEARNING_RATE,
        report_to="none",
        logging_steps=1,
        max_length=MAX_SOURCE_LENGTH + MAX_TARGET_LENGTH,
        max_completion_length=MAX_TARGET_LENGTH,
        remove_unused_columns=False,
    )

    orpo_trainer = ORPOTrainer(
        model=model,
        args=orpo_args,
        train_dataset=orpo_dummy_data,
        processing_class=tokenizer,
    )
    orpo_trainer.train()
    print("ORPO Training DONE.")

    print("\nSemua Test Selesai!")

if __name__ == '__main__':
    # Fix for multiprocessing in Windows when calling map with multiple procs
    import multiprocessing
    multiprocessing.freeze_support()
    main()
