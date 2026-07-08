import os
os.environ["UNSLOTH_COMPILE_DISABLE"] = "1"
import torch
from PIL import Image
from transformers import AutoProcessor
from unsloth import FastVisionModel
import json

MODEL_NAME = "google/t5gemma-2-270m-270m"

print("Loading model and processor...")
model, tokenizer = FastVisionModel.from_pretrained(
    model_name=MODEL_NAME,
    load_in_4bit=True,
)
processor = AutoProcessor.from_pretrained(MODEL_NAME)

from unsloth.chat_templates import get_chat_template
tokenizer = get_chat_template(tokenizer, chat_template="gemma-3")
processor.chat_template = tokenizer.chat_template
if hasattr(processor, "tokenizer"):
    processor.tokenizer.chat_template = tokenizer.chat_template

tokenizer.add_bos_token = False
if hasattr(processor, "tokenizer"):
    processor.tokenizer.add_bos_token = False

# Load satu sample dari data
with open("data/multimodal/train_vision.jsonl", "r", encoding="utf-8") as f:
    records = [json.loads(line.strip()) for line in f]

# Ambil sample dengan banyak images
sample = None
for r in records:
    if len(r.get("images", [])) >= 6:
        sample = r
        break

if not sample:
    sample = records[0]

print(f"\nSample ID: {sample.get('id')}")
print(f"Jumlah images: {len(sample.get('images', []))}")

# Load images
pil_images = []
for path in sample.get("images", [])[:8]:  # Max 8 images
    if os.path.exists(path):
        pil_images.append(Image.open(path).convert("RGB"))

print(f"Berhasil load {len(pil_images)} images")

# Convert ke vision format
old_messages = sample.get("messages", [])
new_messages = []
image_idx = 0

for msg in old_messages:
    role = msg["role"]
    content = msg["content"]
    
    if role == "user" and "📷" in content:
        num_images = content.count("📷")
        text_content = content.replace("📷", "").strip()
        new_content = []
        for _ in range(num_images):
            if image_idx < len(pil_images):
                new_content.append({"type": "image", "image": pil_images[image_idx]})
                image_idx += 1
        if text_content:
            new_content.append({"type": "text", "text": text_content})
        new_messages.append({"role": role, "content": new_content})
    else:
        new_messages.append({"role": role, "content": [{"type": "text", "text": content}]})

print(f"\nTotal images di messages: {image_idx}")
print(f"Total messages: {len(new_messages)}")

# Cek image count per message
for i, msg in enumerate(new_messages):
    if isinstance(msg.get("content"), list):
        img_count = sum(1 for b in msg["content"] if isinstance(b, dict) and b.get("type") == "image")
        print(f"  Message {i} ({msg['role']}): {img_count} images")

# Tokenize
print("\nTokenizing...")
text_prompt = processor.apply_chat_template(new_messages, tokenize=False)
print(f"Text prompt length: {len(text_prompt)} chars")

model_inputs = processor(
    text=text_prompt,
    images=pil_images,
    return_tensors="pt"
)

input_len = model_inputs["input_ids"].shape[1]
print(f"Input IDs shape: {model_inputs['input_ids'].shape}")
print(f"Total tokens: {input_len}")

if "pixel_values" in model_inputs:
    print(f"Pixel values shape: {model_inputs['pixel_values'].shape}")
    expected_image_tokens = model_inputs["pixel_values"].shape[0] * 256
    print(f"Expected image tokens: {expected_image_tokens}")
    print(f"Mismatch: {expected_image_tokens - (input_len if input_len < expected_image_tokens else 'N/A')}")

print("\nDone!")
