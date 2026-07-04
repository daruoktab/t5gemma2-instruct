import os
os.environ["UNSLOTH_COMPILE_DISABLE"] = "1"
import torch
setattr(torch._dynamo.config, "recompile_limit", 128)
import fitz  # PyMuPDF
from PIL import Image
from datasets import Dataset
from unsloth import FastVisionModel
from unsloth.trainer import UnslothVisionDataCollator
from trl import SFTTrainer, SFTConfig

# Configuration
MODEL_NAME = "google/t5gemma-2-270m-270m"
OUTPUT_DIR = "results/test_vision_output"
LOAD_IN_4BIT = True

print("=== 1. Mempersiapkan Dataset Uji Coba (Vision) ===")
os.makedirs("data/multimodal/images", exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Membangun dataset menggunakan gambar-gambar random dari random_metadata.json
import json
metadata_path = "data/multimodal/metadata/random_metadata.json"
images_dir = "data/multimodal/images/general"

if not os.path.exists(metadata_path):
    raise FileNotFoundError(f"Metadata file not found: {metadata_path}")

with open(metadata_path, "r", encoding="utf-8") as f:
    metadata = json.load(f)

samples = []
all_files = sorted(os.listdir(images_dir))
random_files = [f for f in all_files if f.startswith("random_") and f.endswith(".png")]

loaded_meta = []
for filename in random_files:
    img_path = os.path.join(images_dir, filename)
    if not os.path.exists(img_path):
        continue
        
    meta = metadata.get(filename)
    if not meta:
        continue
        
    caption_id = meta.get("caption_native_lang")
    caption_en = meta.get("caption")
    desc = caption_id or caption_en
    if not desc:
        continue
        
    try:
        pil_img = Image.open(img_path).convert("RGB")
        loaded_meta.append((pil_img, desc))
    except Exception as e:
        continue

# 1. Bangun single-image samples (8 sampel)
for i in range(min(8, len(loaded_meta))):
    pil_img, desc = loaded_meta[i]
    record = {
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": pil_img},
                    {"type": "text", "text": "Tolong deskripsikan gambar ini."}
                ]
            },
            {
                "role": "assistant",
                "content": [
                    {"type": "text", "text": f"Gambar ini memperlihatkan {desc}."}
                ]
            }
        ]
    }
    samples.append(record)

# 2. Bangun multi-image sample (1 sampel menggabungkan 2 gambar budaya)
if len(loaded_meta) >= 10:
    img1, desc1 = loaded_meta[8]
    img2, desc2 = loaded_meta[9]
    multi_record = {
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": img1},
                    {"type": "image", "image": img2},
                    {"type": "text", "text": "Bandingkan gambar pertama dan gambar kedua ini secara singkat."}
                ]
            },
            {
                "role": "assistant",
                "content": [
                    {"type": "text", "text": f"Gambar pertama memperlihatkan {desc1}, sedangkan gambar kedua memperlihatkan {desc2}."}
                ]
            }
        ]
    }
    samples.append(multi_record)

dataset = Dataset.from_list(samples)
print(f"Dataset berhasil dibangun dengan {len(dataset)} sampel gambar.")

print("\n=== 2. Memuat VLM dari Unsloth ===")
# Set token HF jika ada di environment variable
token = os.environ.get("HF_TOKEN")
model, tokenizer = FastVisionModel.from_pretrained(
    model_name=MODEL_NAME,
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

print("\n=== 2.5 Mengukur Panjang Konteks (Context Length) ===")
try:
    first_msg = samples[0]["messages"]
    text_prompt = processor.apply_chat_template(first_msg, tokenize=False)
    model_inputs = processor(
        text=text_prompt, 
        images=[first_msg[0]["content"][0]["image"]],  # type: ignore
        return_tensors="pt"
    )
    input_len = model_inputs["input_ids"].shape[1]
    print(f"✅ Contoh teks prompt (1 image):\n{text_prompt}")
    print(f"✅ Panjang input (termasuk token visual gambar): {input_len} token")
    
    if len(samples) > 8:
        multi_msg = samples[8]["messages"]
        multi_prompt = processor.apply_chat_template(multi_msg, tokenize=False)
        multi_inputs = processor(
            text=multi_prompt,
            images=[multi_msg[0]["content"][0]["image"], multi_msg[0]["content"][1]["image"]],  # type: ignore
            return_tensors="pt"
        )
        multi_len = multi_inputs["input_ids"].shape[1]
        print(f"\n✅ Contoh teks prompt (2 images):\n{multi_prompt}")
        print(f"✅ Panjang input (termasuk 2 token visual gambar): {multi_len} token")
        if multi_len <= 1024:
            print("   -> Sangat aman untuk GPU VRAM laptop 6GB Anda!")
        else:
            print("   -> Cukup besar, pastikan batch_size tetap kecil.")
except Exception as e:
    print(f"⚠️ Gagal mengukur panjang token: {e}")

print("\n=== 3. Menerapkan LoRA Adapter (Vision + Language) ===")
model = FastVisionModel.get_peft_model(
    model,
    finetune_vision_layers=True,
    finetune_language_layers=True,
    finetune_attention_modules=True,
    finetune_mlp_modules=True,
    r=8,
    lora_alpha=16,
    lora_dropout=0.0,
    bias="none",
    random_state=3407,
)

if not hasattr(model.config, "text_config"):
    type(model.config).text_config = property(lambda self: self.decoder)
    type(model.config).get_text_config = lambda self, *args, **kwargs: self.decoder

FastVisionModel.for_training(model)

print("\n=== 4. Inisialisasi SFTTrainer (Test 5 Steps) ===")
trainer = SFTTrainer(
    model=model,
    processing_class=processor,
    train_dataset=dataset,
    data_collator=UnslothVisionDataCollator(model, processor),
    args=SFTConfig(
        per_device_train_batch_size=1,
        gradient_accumulation_steps=1,
        learning_rate=2e-4,
        max_steps=5,  # Ditingkatkan menjadi 5 langkah
        logging_steps=1,
        output_dir=OUTPUT_DIR,
        remove_unused_columns=False, # Mandatory for Unsloth Vision collator
        dataset_text_field="",  # Required for Unsloth Vision models
        dataset_kwargs={"skip_prepare_dataset": True},
        loss_type="nll",  # Disable chunked CE loss for Seq2Seq compatibility
        fp16=False,
        bf16=torch.cuda.is_available(),
    ),
)

print("\n=== 5. Memulai Training Uji Coba ===")
trainer.train()
print("\n✅ TEST SUKSES! VLM training pipeline berjalan dengan baik secara lokal!")

print("\n" + "="*70)
print("=== 6. HASIL GENERASI MODEL (EVALUASI SETELAH TRAINING) ===")
print("="*70)

# Siapkan model untuk inferensi (restore KV cache)
FastVisionModel.for_inference(model)

# Nonaktifkan autograd untuk inferensi cepat
with torch.no_grad():
    model.eval()
    
    # 1. Single-Image Inference
    print("\n[ 📸 TEST GENERASI: 1 GAMBAR ]")
    print("-" * 60)
    single_msg = samples[0]["messages"]
    # Hanya ambil turn milik user
    user_turn = [single_msg[0]]
    prompt_single = processor.apply_chat_template(user_turn, add_generation_prompt=True, tokenize=False)
    
    inputs_single = processor(
        text=prompt_single,
        images=[single_msg[0]["content"][0]["image"]],  # type: ignore
        return_tensors="pt"
    ).to(model.device)
    
    outputs_single = model.generate(
        **inputs_single,
        max_new_tokens=64,
        use_cache=True,
    )
    response_single = processor.decode(outputs_single[0], skip_special_tokens=True)
    
    print(f"User   : {single_msg[0]['content'][1]['text']}")  # type: ignore
    print(f"Target : {single_msg[1]['content'][0]['text']}")  # type: ignore
    print(f"Model  : {response_single.strip()}")
    print("-" * 60)
    
    # 2. Multi-Image Inference
    if len(samples) > 8:
        print("\n[ 📸 TEST GENERASI: 2 GAMBAR (MULTI-IMAGE) ]")
        print("-" * 60)
        multi_msg = samples[8]["messages"]
        user_turn_multi = [multi_msg[0]]
        prompt_multi = processor.apply_chat_template(user_turn_multi, add_generation_prompt=True, tokenize=False)
        
        inputs_multi = processor(
            text=prompt_multi,
            images=[multi_msg[0]["content"][0]["image"], multi_msg[0]["content"][1]["image"]],  # type: ignore
            return_tensors="pt"
        ).to(model.device)
        
        outputs_multi = model.generate(
            **inputs_multi,
            max_new_tokens=64,
            use_cache=True,
        )
        response_multi = processor.decode(outputs_multi[0], skip_special_tokens=True)
        
        print(f"User   : {multi_msg[0]['content'][2]['text']}")  # type: ignore
        print(f"Target : {multi_msg[1]['content'][0]['text']}")  # type: ignore
        print(f"Model  : {response_multi.strip()}")
        print("-" * 60)

print("\n" + "="*70)
