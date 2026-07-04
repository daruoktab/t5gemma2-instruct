# /// script
# requires-python = ">=3.10"
# dependencies = [
#     "accelerate",
#     "absl-py",
#     "bitsandbytes",
#     "datasets",
#     "huggingface-hub",
#     "numpy",
#     "peft==0.19.1",
#     "pillow",
#     "pymupdf",
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
    # 📷 Multimodal Vision SFT Fine-Tuning Pipeline (Version 6 - Unsloth)
    =====================================================================
    Notebook ini melatih aspek **vision** dari model **T5Gemma-2 4B-4B** menggunakan QLoRA/LoRA via Unsloth.
    Model dasar yang digunakan adalah model hasil SFT + ORPO teks (`t5gemma-2-4b-4b-instruct-chat-indo-v4-unsloth`).
    
    **Fitur utama:**
    - Pemrosesan gambar dokumen ilmiah/PDF yang diconvert ke format PIL Image.
    - Fine-tuning vision encoder dan attention/language adapter secara modular.
    - Integrasi penuh dengan `FastVisionModel` dan `UnslothVisionDataCollator`.
    """)
    return


@app.cell
def _():
    import os
    import re
    import json
    import torch
    setattr(torch._dynamo.config, "recompile_limit", 128)
    import random
    import datetime
    import gc
    from PIL import Image
    from unsloth import FastVisionModel
    from unsloth.trainer import UnslothVisionDataCollator
    from datasets import Dataset
    from trl import SFTTrainer, SFTConfig
    return (
        os, re, json, torch, random, datetime, gc, Image,
        FastVisionModel, UnslothVisionDataCollator, Dataset,
        SFTTrainer, SFTConfig
    )


@app.cell
def _(mo, os):
    # Hugging Face Token Login Interactive
    hf_token_input = mo.ui.text(
        label="Masukkan Hugging Face Write Token Anda:",
        placeholder="hf_...",
        kind="password"
    )

    status = mo.md("🔑 Silakan masukkan token Hugging Face Anda untuk mengunggah model/adapter.")
    
    if hf_token_input.value:
        try:
            from huggingface_hub import login
            os.environ["HF_TOKEN"] = hf_token_input.value
            login(token=hf_token_input.value)
            status = mo.md("✅ **Berhasil terautentikasi dengan Hugging Face Hub!**")
        except Exception as e:
            status = mo.md(f"❌ **Gagal login:** {e}")

    mo.vstack([hf_token_input, status])
    return hf_token_input, status


@app.cell
def _(torch):
    # =====================================================================
    # KONFIGURASI HYPERPARAMETER
    # =====================================================================
    MODEL_NAME = "daruokta/t5gemma-2-4b-4b-instruct-chat-indo-v4-unsloth"
    SUBFOLDER = "merged_bf16" # load base model + text adapters yang sudah di-merge
    LOAD_IN_4BIT = True
    OUTPUT_DIR = "results/t5gemma2_vision"
    HF_CHECKPOINT_REPO = "daruokta/t5gemma-2-4b-4b-instruct-chat-indo-v4-vision"
    
    # Dataset JSONL lokal
    JSONL_DATASET_PATH = "data/multimodal/train_vision.jsonl"
    
    # LoRA config
    LORA_RANK = 64
    LORA_ALPHA = 128
    LORA_DROPOUT = 0.0
    
    # Training args
    LEARNING_RATE = 2e-5
    NUM_EPOCHS = 3
    PER_DEVICE_TRAIN_BATCH_SIZE = 2
    GRADIENT_ACCUMULATION_STEPS = 16
    WARMUP_STEPS = 50
    WEIGHT_DECAY = 0.01
    LR_SCHEDULER_TYPE = "cosine"
    LOGGING_STEPS = 10
    SAVE_TOTAL_LIMIT = 2
    OPTIM = "paged_adamw_8bit"
    
    BF16 = torch.cuda.is_available()
    return (
        MODEL_NAME, SUBFOLDER, LOAD_IN_4BIT, OUTPUT_DIR, HF_CHECKPOINT_REPO,
        JSONL_DATASET_PATH, LORA_RANK, LORA_ALPHA, LORA_DROPOUT,
        LEARNING_RATE, NUM_EPOCHS, PER_DEVICE_TRAIN_BATCH_SIZE,
        GRADIENT_ACCUMULATION_STEPS, WARMUP_STEPS, WEIGHT_DECAY,
        LR_SCHEDULER_TYPE, LOGGING_STEPS, SAVE_TOTAL_LIMIT, OPTIM, BF16
    )


@app.cell
def _(JSONL_DATASET_PATH, Dataset, Image, os, json):
    # Load dan format dataset untuk input vision model
    if not os.path.exists(JSONL_DATASET_PATH):
        print(f"⚠️ Dataset tidak ditemukan di {JSONL_DATASET_PATH}. Silakan buat terlebih dahulu.")
        train_dataset = None
    else:
        records = []
        with open(JSONL_DATASET_PATH, "r", encoding="utf-8") as f:
            for line in f:
                records.append(json.loads(line.strip()))
                
        formatted = []
        for rec in records:
            img_paths = rec.get("images", [])
            pil_images = []
            for path in img_paths:
                if os.path.exists(path):
                    pil_images.append(Image.open(path).convert("RGB"))
                else:
                    # Alternatif relative path
                    base_dir = os.path.dirname(os.path.abspath(JSONL_DATASET_PATH))
                    alt_path = os.path.join(base_dir, "images", os.path.basename(path))
                    if os.path.exists(alt_path):
                        pil_images.append(Image.open(alt_path).convert("RGB"))
                        
            if not pil_images:
                continue
                
            old_messages = rec.get("messages", [])
            new_messages = []
            image_idx = 0
            for msg in old_messages:
                role = msg["role"]
                content = msg["content"]
                if role == "user" and "📷" in content and image_idx < len(pil_images):
                    text_content = content.replace("📷", "").strip()
                    new_content = [
                        {"type": "image", "image": pil_images[image_idx]},
                        {"type": "text", "text": text_content}
                    ]
                    image_idx += 1
                else:
                    new_content = [
                        {"type": "text", "text": content}
                    ]
                new_messages.append({"role": role, "content": new_content})
            formatted.append({"messages": new_messages})
        
        train_dataset = Dataset.from_list(formatted)
        print(f"✅ Dataset vision berhasil dimuat dengan {len(train_dataset)} sampel.")
    return (train_dataset,)


@app.cell
def _(
    MODEL_NAME, SUBFOLDER, LOAD_IN_4BIT, FastVisionModel, os
):
    # Memuat VLM base model + processor
    print(f"Loading base model + LoRA vision adapter dari {MODEL_NAME} (subfolder: {SUBFOLDER})...")
    token = os.environ.get("HF_TOKEN")
    model, tokenizer = FastVisionModel.from_pretrained(
        model_name=MODEL_NAME,
        subfolder=SUBFOLDER,
        load_in_4bit=LOAD_IN_4BIT,
        use_gradient_checkpointing="unsloth",
        token=token,
    )
    from transformers import AutoProcessor
    processor = AutoProcessor.from_pretrained(MODEL_NAME, token=token)
    from unsloth.chat_templates import get_chat_template
    tokenizer = get_chat_template(tokenizer, chat_template="gemma-3")
    processor.chat_template = tokenizer.chat_template
    if hasattr(processor, "tokenizer"):
        processor.tokenizer.chat_template = tokenizer.chat_template
        
    # Nonaktifkan penambahan bos_token otomatis untuk menghindari bos_token ganda saat inferensi
    tokenizer.add_bos_token = False
    if hasattr(processor, "tokenizer"):
        processor.tokenizer.add_bos_token = False
        
    return model, tokenizer, processor


@app.cell
def _(model, FastVisionModel, LORA_RANK, LORA_ALPHA, LORA_DROPOUT):
    # Setup adapter PEFT/LoRA pada vision tower dan language backbone
    print("Applying PEFT LoRA adapters for vision and language layers...")
    model = FastVisionModel.get_peft_model(
        model,
        finetune_vision_layers=True,
        finetune_language_layers=True,
        finetune_attention_modules=True,
        finetune_mlp_modules=True,
        r=LORA_RANK,
        lora_alpha=LORA_ALPHA,
        lora_dropout=LORA_DROPOUT,
        bias="none",
        random_state=3407,
    )
    if not hasattr(model.config, "text_config"):
        type(model.config).text_config = property(lambda self: self.decoder)
        type(model.config).get_text_config = lambda self, *args, **kwargs: self.decoder
    FastVisionModel.for_training(model)
    return model


@app.cell
def _(
    model, processor, train_dataset, UnslothVisionDataCollator,
    SFTTrainer, SFTConfig, OUTPUT_DIR, NUM_EPOCHS,
    PER_DEVICE_TRAIN_BATCH_SIZE, GRADIENT_ACCUMULATION_STEPS,
    LEARNING_RATE, WARMUP_STEPS, WEIGHT_DECAY, LR_SCHEDULER_TYPE,
    LOGGING_STEPS, SAVE_TOTAL_LIMIT, OPTIM, BF16, mo
):
    if train_dataset is None:
        return mo.md("❌ **Dataset tidak ditemukan, training dibatalkan.**")
        
    print("Starting SFT Trainer for Multimodal Vision model...")
    trainer = SFTTrainer(
        model=model,
        processing_class=processor,
        train_dataset=train_dataset,
        data_collator=UnslothVisionDataCollator(model, processor),
        args=SFTConfig(
            per_device_train_batch_size=PER_DEVICE_TRAIN_BATCH_SIZE,
            gradient_accumulation_steps=GRADIENT_ACCUMULATION_STEPS,
            learning_rate=LEARNING_RATE,
            num_train_epochs=NUM_EPOCHS,
            warmup_steps=WARMUP_STEPS,
            weight_decay=WEIGHT_DECAY,
            lr_scheduler_type=LR_SCHEDULER_TYPE,
            logging_steps=LOGGING_STEPS,
            save_total_limit=SAVE_TOTAL_LIMIT,
            output_dir=OUTPUT_DIR,
            remove_unused_columns=False,
            dataset_text_field="",
            dataset_kwargs={"skip_prepare_dataset": True},
            loss_type="nll",  # Disable chunked CE loss for Seq2Seq compatibility
            fp16=False,
            bf16=BF16,
            optim=OPTIM,
            save_strategy="epoch",
        ),
    )
    
    train_result = trainer.train()
    return trainer, train_result


@app.cell
def _(model, processor, OUTPUT_DIR, HF_CHECKPOINT_REPO, os):
    # Menyimpan adapter vision dan mengunggah ke HF Hub
    adapter_path = os.path.join(OUTPUT_DIR, "final_adapter")
    model.save_pretrained(adapter_path)
    processor.save_pretrained(adapter_path)
    print(f"✅ Adapter LoRA vision berhasil disimpan ke: {adapter_path}")
    
    token = os.environ.get("HF_TOKEN")
    if token:
        print(f"Mengunggah adapter vision ke Hugging Face Hub: {HF_CHECKPOINT_REPO}...")
        try:
            model.push_to_hub(HF_CHECKPOINT_REPO, subfolder="vision_adapter", token=token)
            processor.push_to_hub(HF_CHECKPOINT_REPO, subfolder="vision_adapter", token=token)
            print("✅ Berhasil mengunggah adapter vision!")
        except Exception as e:
            print(f"❌ Gagal mengunggah ke Hugging Face Hub: {e}")
    return


if __name__ == "__main__":
    app.run()
