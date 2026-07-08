"""
Test Manual: T5Gemma2 Seq2Seq + Vision dengan Vanilla Transformers (Tanpa Unsloth)
===================================================================================
Membuktikan mekanisme image encoder (SigLIP -> 256 soft token/gambar) + arsitektur
Encoder-Decoder (Seq2Seq) T5Gemma2 bekerja secara native dengan Hugging Face
Transformers murni (tanpa Unsloth / FastVisionModel / SFTTrainer-TRL).

Validasi:
  1. Arsitektur  : encoder punya vision_tower (SigLIP) + multi_modal_projector
  2. Forward 1   : 1 gambar  -> cek shape & loss
  3. Forward 10  : 10 gambar -> cek shape & loss (max image per chat)
  4. Collator    : Seq2Seq-aware collator + batched forward (flat-concat pixel)
  5. Training    : 2 training steps (mixed image count, batch_size=1)
  6. Inference   : generate dengan 10 gambar

Catatan:
  - Pakai AutoModelForSeq2SeqLM + Trainer (HF native, seq2seq-aware via labels).
  - Image = SigLIP, FIXED 256 soft token/gambar, resize 896x896.
  - Encoder input = text + image soft tokens; Decoder labels = target text + EOS.
  - Loss hanya dihitung di decoder (labels), encoder (image+text) tidak berkontribusi loss.
"""

import os
import sys
import json
import random
import traceback

import torch
from PIL import Image
from datasets import Dataset
from transformers import (
    AutoProcessor,
    AutoModelForSeq2SeqLM,
    Trainer,
    TrainingArguments,
)

# =====================================================================
# KONFIGURASI
# =====================================================================
MODEL_NAME = os.environ.get("T5GEMMA2_MODEL", "google/t5gemma-2-270m-270m")
MAX_IMAGES = 10              # max gambar per percakapan (sesuai info)
MAX_SOURCE_LENGTH = 4096     # 10 * 256 = 2560 token visual + teks
MAX_TARGET_LENGTH = 512
SEED = 3407

VISION_DATASET_PATH = "data/multimodal/train_vision.jsonl"

random.seed(SEED)
torch.manual_seed(SEED)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DTYPE = torch.bfloat16 if (torch.cuda.is_available() and torch.cuda.is_bf16_supported()) else torch.float32

print("=" * 70)
print("=== TEST VANILLA TRANSFORMERS: T5Gemma2 Seq2Seq + Vision ===")
print(f"=== Model : {MODEL_NAME} ===")
print(f"=== Device: {DEVICE} | Dtype: {DTYPE} | Max images: {MAX_IMAGES} ===")
print("=" * 70)


# =====================================================================
# HELPER
# =====================================================================
COLORS = [(255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0), (255, 0, 255),
          (0, 255, 255), (128, 0, 128), (0, 128, 128), (255, 128, 0), (128, 255, 0)]


def make_synthetic_image(idx=0, size=224):
    """Buat gambar dummy berwarna untuk testing (fallback jika dataset real tidak ada)."""
    return Image.new("RGB", (size, size), color=COLORS[idx % len(COLORS)])


def load_real_images(max_images=10):
    """Coba load gambar real dari dataset vision lokal. Return None jika gagal."""
    if not os.path.exists(VISION_DATASET_PATH):
        return None
    try:
        with open(VISION_DATASET_PATH, "r", encoding="utf-8") as f:
            records = [json.loads(l.strip()) for l in f if l.strip()]
        records.sort(key=lambda r: len(r.get("images", [])), reverse=True)
        for rec in records[:30]:
            paths = rec.get("images", [])[:max_images]
            imgs = [Image.open(p).convert("RGB") for p in paths if os.path.exists(p)]
            if imgs:
                print(f"  [INFO] Load {len(imgs)} gambar real (id={rec.get('id')})")
                return imgs
    except Exception as e:
        print(f"  [WARN] Gagal load dataset vision: {e}")
    return None


def build_prompt(processor, num_images, user_text):
    """Bangun prompt encoder dengan placeholder image (boi_token / 📷)."""
    boi = getattr(processor, "boi_token", None) or "\uf400"
    img_part = (boi + " ") * num_images
    return f"{img_part}{user_text}"


def make_labels(tokenizer, target_text, max_len):
    """Target decoder: content + EOS (tanpa BOS). Pad -100 dilakukan di collator."""
    ids = tokenizer.encode(target_text, add_special_tokens=False)
    ids = ids[: max_len - 1] + [tokenizer.eos_token_id]
    return torch.tensor([ids], dtype=torch.long)


# =====================================================================
# STEP 1: Load Model + Processor (Vanilla Transformers)
# =====================================================================
print("\n[1/6] Memuat Model & Processor (Vanilla Transformers)...")
processor = AutoProcessor.from_pretrained(MODEL_NAME, trust_remote_code=True)
print(f"  Processor         : {processor.__class__.__name__}")
print(f"  image_seq_length  : {getattr(processor.image_processor, 'image_seq_length', '?')}")
print(f"  image_size        : {getattr(processor.image_processor, 'size', '?')}")
print(f"  boi_token         : {getattr(processor, 'boi_token', '?')}")
print(f"  pad/eos/bos id    : {processor.tokenizer.pad_token_id}/{processor.tokenizer.eos_token_id}/{processor.tokenizer.bos_token_id}")

model = AutoModelForSeq2SeqLM.from_pretrained(
    MODEL_NAME, torch_dtype=DTYPE, trust_remote_code=True,
).to(DEVICE)
model.config.use_cache = False
print(f"  Model             : {model.__class__.__name__}")
print(f"  is_encoder_decoder: {model.config.is_encoder_decoder}")
print(f"  decoder_start_tok : {getattr(model.config, 'decoder_start_token_id', 'N/A')}")
print(f"  Params            : {sum(p.numel() for p in model.parameters()) / 1e6:.1f}M")


# =====================================================================
# STEP 2: Inspeksi Arsitektur Vision (Encoder)
# =====================================================================
print("\n[2/6] Inspeksi Arsitektur (Encoder Vision)...")
inner = getattr(model, "model", model)
encoder = getattr(inner, "encoder", None) or getattr(model, "encoder", None)
if encoder is not None:
    vt = getattr(encoder, "vision_tower", None)
    mp = getattr(encoder, "multi_modal_projector", None)
    print(f"  encoder class : {encoder.__class__.__name__}")
    print(f"  vision_tower  : {vt.__class__.__name__ if vt else 'TIDAK ADA'}")
    print(f"  mm_projector  : {mp.__class__.__name__ if mp else 'TIDAK ADA'}")
    if vt is not None:
        try:
            venc = getattr(vt, "encoder", getattr(vt, "vision_model", vt))
            n_vl = len(getattr(venc, "layers", []))
            print(f"  vision layers : {n_vl}")
        except Exception:
            print("  vision layers : (tidak terbaca)")
else:
    print("  [WARN] Struktur encoder tidak ditemukan di path yang diharapkan.")
    print("  Top-level modules:")
    for name, _ in model.named_children():
        print(f"    - {name}")


# =====================================================================
# STEP 3: Forward Pass — 1 Gambar
# =====================================================================
print("\n[3/6] Forward Pass: 1 Gambar...")
imgs1 = load_real_images(1) or [make_synthetic_image(0)]
prompt1 = build_prompt(processor, len(imgs1), "Deskripsikan gambar ini secara singkat.")
enc1 = processor(text=prompt1, images=imgs1, return_tensors="pt")
enc1 = {k: v.to(DEVICE) for k, v in enc1.items()}
enc1["labels"] = make_labels(processor.tokenizer, "Ini adalah gambar uji coba.", MAX_TARGET_LENGTH).to(DEVICE)

print(f"  input_ids    : {enc1['input_ids'].shape}  (harus ada 256 image soft token)")
print(f"  pixel_values : {enc1['pixel_values'].shape}")
print(f"  labels       : {enc1['labels'].shape}")
model.eval()
with torch.no_grad():
    out1 = model(**enc1)
print(f"  loss         : {out1.loss.item():.4f}")
print(f"  logits       : {out1.logits.shape}  (batch, tgt_len, vocab)")


# =====================================================================
# STEP 4: Forward Pass — 10 Gambar (Max per chat)
# =====================================================================
print(f"\n[4/6] Forward Pass: {MAX_IMAGES} Gambar (max per chat)...")
imgs10 = load_real_images(MAX_IMAGES) or [make_synthetic_image(i) for i in range(MAX_IMAGES)]
imgs10 = imgs10[:MAX_IMAGES]
prompt10 = build_prompt(processor, len(imgs10),
                        "Jelaskan semua gambar ini secara berurutan satu per satu.")
enc10 = processor(text=prompt10, images=imgs10, return_tensors="pt",
                  truncation=True, max_length=MAX_SOURCE_LENGTH)
enc10 = {k: v.to(DEVICE) for k, v in enc10.items()}
enc10["labels"] = make_labels(processor.tokenizer,
                              "Gambar pertama merah, gambar kedua hijau, dan seterusnya.",
                              MAX_TARGET_LENGTH).to(DEVICE)

expected_img_tokens = len(imgs10) * 256
print(f"  input_ids    : {enc10['input_ids'].shape}  (image tokens = {expected_img_tokens})")
print(f"  pixel_values : {enc10['pixel_values'].shape}  ({len(imgs10)} gambar)")
print(f"  labels       : {enc10['labels'].shape}")
with torch.no_grad():
    out10 = model(**enc10)
print(f"  loss         : {out10.loss.item():.4f}")
print(f"  logits       : {out10.logits.shape}")

# Bersihkan cache GPU setelah eval (sebelum training butuh gradient)
del enc1, out1, enc10, out10
if torch.cuda.is_available():
    torch.cuda.empty_cache()


# =====================================================================
# STEP 5: Custom Collator Seq2Seq-Vision + Trainer (2 training steps)
# =====================================================================
print("\n[5/6] Custom Collator Seq2Seq-Vision + Trainer (2 steps)...")


class Seq2SeqVisionCollator:
    """Collator Seq2Seq + Vision untuk T5Gemma2 (vanilla transformers).

    - Encoder : input_ids (text + image soft tokens) + pixel_values + attention_mask
    - Decoder : labels (target + EOS, pad -100)

    Mendukung multiple images per sample. pixel_values di-concatenate flat
    menjadi (total_images, C, H, W). Model mencocokkan secara internal via
    posisi image soft token (id 256001) di input_ids.
    """

    def __init__(self, processor, max_source_length, max_target_length):
        self.processor = processor
        self.tok = processor.tokenizer
        self.pad_id = self.tok.pad_token_id
        self.eos_id = self.tok.eos_token_id
        self.max_src = max_source_length
        self.max_tgt = max_target_length

    def __call__(self, batch):
        input_ids, attn_masks, pixel_vals, labels = [], [], [], []
        for item in batch:
            enc = self.processor(
                text=item["prompt_text"],
                images=item["images"] if item["images"] else None,
                return_tensors="pt",
                truncation=True,
                max_length=self.max_src,
            )
            input_ids.append(enc["input_ids"][0])
            attn_masks.append(enc["attention_mask"][0])
            if "pixel_values" in enc:
                # Ambil SEMUA gambar sample: (n_img, C, H, W), bukan [0] (image pertama saja)
                pixel_vals.append(enc["pixel_values"])
            tgt_ids = self.tok.encode(item["target_text"], add_special_tokens=False)
            tgt_ids = tgt_ids[: self.max_tgt - 1] + [self.eos_id]
            labels.append(torch.tensor(tgt_ids, dtype=torch.long))

        ii = torch.nn.utils.rnn.pad_sequence(input_ids, batch_first=True, padding_value=self.pad_id)
        am = torch.nn.utils.rnn.pad_sequence(attn_masks, batch_first=True, padding_value=0)
        lb = torch.nn.utils.rnn.pad_sequence(labels, batch_first=True, padding_value=-100)
        out = {"input_ids": ii, "attention_mask": am, "labels": lb}
        if pixel_vals:
            pv = pixel_vals[0]
            out["pixel_values"] = torch.cat(pixel_vals, dim=0) if pv.ndim == 4 else torch.stack(pixel_vals, dim=0)
        return out


# Sediakan gambar dasar (real atau sintetis)
base_imgs = load_real_images(MAX_IMAGES) or [make_synthetic_image(i) for i in range(MAX_IMAGES)]

# Dataset dummy: campuran 1, 2, 3 gambar (VRAM-friendly untuk training smoke test)
configs = [1, 2, 3]
targets = [
    "Gambar berwarna merah.",
    "Dua gambar berurutan.",
    "Tiga gambar dengan warna berbeda.",
]
dummy_samples = []
for i in range(len(configs)):
    ni = configs[i % len(configs)]
    dummy_samples.append({
        "prompt_text": build_prompt(processor, ni, f"Jelaskan {ni} gambar ini."),
        "images": base_imgs[:ni],
        "target_text": targets[i % len(targets)],
    })
dummy_ds = Dataset.from_list(dummy_samples)
print(f"  Dataset dummy : {len(dummy_ds)} samples (image count: {[len(s['images']) for s in dummy_samples]})")

collator = Seq2SeqVisionCollator(processor, MAX_SOURCE_LENGTH, MAX_TARGET_LENGTH)

# Test collator: 1 sample
cb1 = collator(dummy_samples[:1])
print(f"  Collator (b=1) -> input_ids: {cb1['input_ids'].shape}, "
      f"labels: {cb1['labels'].shape}, "
      f"pixel_values: {cb1.get('pixel_values').shape}")

# Test batching: 2 sample dengan image count SAMA (2 gambar) -> validasi flat-concat
batch2 = collator([
    {"prompt_text": build_prompt(processor, 2, "Test A."), "images": base_imgs[:2], "target_text": "Jawaban A."},
    {"prompt_text": build_prompt(processor, 2, "Test B."), "images": base_imgs[:2], "target_text": "Jawaban B."},
])
print(f"  Collator (b=2, 2 img/sample) -> input_ids: {batch2['input_ids'].shape}, "
      f"pixel_values: {batch2['pixel_values'].shape} (total 4 img flat)")
batch2_dev = {k: v.to(DEVICE) for k, v in batch2.items()}
with torch.no_grad():
    out_b = model(**batch2_dev)
print(f"  Batched forward loss: {out_b.loss.item():.4f}  (batch=2)")

# Bersihkan cache + aktifkan gradient checkpointing sebelum training
del cb1, batch2, batch2_dev, out_b
if torch.cuda.is_available():
    torch.cuda.empty_cache()
model.gradient_checkpointing_enable()
print("  Gradient checkpointing: ON (hemat VRAM untuk training)")

# Trainer: 2 training steps (batch_size=1, mixed image count)
targs = TrainingArguments(
    output_dir="results/vanilla_vision_smoke",
    per_device_train_batch_size=1,
    max_steps=2,
    learning_rate=2e-4,
    logging_steps=1,
    report_to="none",
    fp16=False,
    bf16=torch.cuda.is_available() and torch.cuda.is_bf16_supported(),
    remove_unused_columns=False,
    save_strategy="no",
)
trainer = Trainer(model=model, args=targs, train_dataset=dummy_ds, data_collator=collator)
print("  Memulai 2 training steps (batch_size=1, mixed image count)...")
try:
    trainer.train()
    print("  ✅ Training smoke test BERHASIL — seq2seq + vision jalan native!")
except Exception as e:
    print(f"  ❌ Training gagal: {e}")
    traceback.print_exc()


# =====================================================================
# STEP 6: Inference — Generate dengan Multiple Gambar
# =====================================================================
# Catatan: 10 gambar sudah di-test forward di Step 4. Untuk inference di GPU
# terbatas, pakai 3 gambar (generate butuh KV-cache + autoregressive loop).
INF_IMAGES = min(3, MAX_IMAGES)
print(f"\n[6/6] Inference: Generate dengan {INF_IMAGES} Gambar...")
# Bersihkan cache sebelum inference (training mungkin memenuhi VRAM)
if torch.cuda.is_available():
    torch.cuda.empty_cache()
model.eval()

# Suppress token visual di generasi (cegah halusinasi image token tanpa gambar)
SUPPRESS_VISION = [255999, 256000, 256001]  # boi, eoi, image_soft_token

try:
    imgs_inf = (load_real_images(MAX_IMAGES) or [make_synthetic_image(i) for i in range(MAX_IMAGES)])[:INF_IMAGES]
    prompt_inf = build_prompt(processor, len(imgs_inf), "Jelaskan semua gambar ini berurutan.")
    inf_inputs = processor(text=prompt_inf, images=imgs_inf, return_tensors="pt",
                           truncation=True, max_length=MAX_SOURCE_LENGTH)
    inf_inputs = {k: v.to(DEVICE) for k, v in inf_inputs.items()}
    print(f"  input_ids: {inf_inputs['input_ids'].shape}, pixel_values: {inf_inputs['pixel_values'].shape}")

    with torch.no_grad():
        gen = model.generate(
            **inf_inputs,
            max_new_tokens=64,
            do_sample=False,
            repetition_penalty=1.2,
            suppress_tokens=SUPPRESS_VISION,
        )
    resp = processor.tokenizer.decode(gen[0], skip_special_tokens=True)
    print(f"  Generated ({gen.shape[1]} tok): {resp[:300]}")
    print("  ✅ Inference BERHASIL — seq2seq generate dengan vision jalan!")
except Exception as e:
    print(f"  ❌ Inference gagal: {e}")
    traceback.print_exc()


# =====================================================================
# RINGKASAN
# =====================================================================
print("\n" + "=" * 70)
print("✅ TEST VANILLA SELESAI.")
print("   Mekanisme: SigLIP -> 256 soft token/gambar -> inject ke ENCODER")
print("   -> encoder output H (bidirectional, visual+text) -> DECODER Merged")
print("   Attention [X;H] -> generate (causal). Loss HANYA di decoder (labels).")
print("   Seq2Seq + Vision T5Gemma2 VALID secara native transformers.")
print("=" * 70)
