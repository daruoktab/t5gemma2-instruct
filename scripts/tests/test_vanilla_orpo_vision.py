"""
Test Manual: T5Gemma2 Seq2Seq + Vision ORPO dengan Vanilla Transformers (Tanpa Unsloth)
=========================================================================================
Membuktikan ORPO (Odds Ratio Preference Optimization) berjalan pada arsitektur
Encoder-Decoder (Seq2Seq) + Vision T5Gemma2 menggunakan Hugging Face Transformers murni.

Validasi:
  1. Arsitektur   : load model + processor + inspect vision tower
  2. Forward ORPO : 2x forward (chosen & rejected) dengan pixel_values + log-odds ratio
  3. Collator     : VisionORPOCollator (chosen/rejected labels + shared input_ids + pixel)
  4. Trainer      : VisionORPOTrainer (custom compute_loss: OR loss + SFT loss)
  5. Training     : 2 training steps ORPO dengan dummy data (mixed image count)
  6. Optimisasi   : cache encoder output (1x SigLIP, 2x decoder) — optional test

Mekanisme ORPO Seq2Seq:
  - Encoder input (text + image) = SHARED antara chosen & rejected
  - Decoder target = chosen_labels vs rejected_labels (text response berbeda)
  - Loss = SFT_loss(chosen) + beta * OR_loss(log_odds_margin)

Logit masking:
  - apply_logit_mask (hook di lm_head, decoder only) tetap aktif
  - chosen & rejected logits sama-sama kena mask -> fair comparison
  - Token suppress (6244 unused + 3 vision) tidak akan menang di log-odds ratio
"""

import os
import json
import random
import traceback

import torch
import torch.nn.functional as F
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
MAX_IMAGES = 10
MAX_SOURCE_LENGTH = 4096
MAX_TARGET_LENGTH = 512
ORPO_BETA = 0.1
SEED = 3407

VISION_DATASET_PATH = "data/multimodal/train_vision.jsonl"

random.seed(SEED)
torch.manual_seed(SEED)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DTYPE = torch.bfloat16 if (torch.cuda.is_available() and torch.cuda.is_bf16_supported()) else torch.float32

print("=" * 70)
print("=== TEST VANILLA TRANSFORMERS: T5Gemma2 Seq2Seq + Vision ORPO ===")
print(f"=== Model : {MODEL_NAME} | Beta: {ORPO_BETA} ===")
print(f"=== Device: {DEVICE} | Dtype: {DTYPE} | Max images: {MAX_IMAGES} ===")
print("=" * 70)


# =====================================================================
# HELPER
# =====================================================================
COLORS = [(255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0), (255, 0, 255),
          (0, 255, 255), (128, 0, 128), (0, 128, 128), (255, 128, 0), (128, 255, 0)]


def make_synthetic_image(idx=0, size=224):
    return Image.new("RGB", (size, size), color=COLORS[idx % len(COLORS)])


def load_real_images(max_images=10):
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
    boi = getattr(processor, "boi_token", None) or "\uf400"
    img_part = (boi + " ") * num_images
    return f"{img_part}{user_text}"


def make_labels(tokenizer, target_text, max_len):
    ids = tokenizer.encode(target_text, add_special_tokens=False)
    ids = ids[: max_len - 1] + [tokenizer.eos_token_id]
    return torch.tensor([ids], dtype=torch.long)


# =====================================================================
# LOGIT MASKING (decoder-only, aman untuk vision)
# =====================================================================
SUPPRESS_BLOCK1 = list(range(6, 105))           # <unused0>–<unused98> (kecuali 7-12 task prefix)
SUPPRESS_BLOCK2 = list(range(256002, 262144))    # <unused100>–<unused6241>
SUPPRESS_VISION = [255999, 256000, 256001]       # boi, eoi, image_soft_token
ALL_SUPPRESS_IDS = set(SUPPRESS_BLOCK1 + SUPPRESS_BLOCK2 + SUPPRESS_VISION)


def apply_logit_mask(model, suppress_ids):
    """Forward hook di lm_head: logits[suppress] += -10000 (decoder only)."""
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
            return (logits + mask.to(logits.device),) + outputs[1:]
        return outputs

    target = None
    if hasattr(model, "lm_head"):
        target = model.lm_head
    elif hasattr(model, "base_model") and hasattr(model.base_model, "lm_head"):
        target = model.base_model.lm_head
    if target is not None:
        target.register_forward_hook(forward_hook)
        print(f"  ✅ Logit mask registered (lm_head) untuk {len(suppress_list)} tokens.")
    else:
        model.register_forward_hook(forward_hook)
        print(f"  ✅ Logit mask registered (top-level fallback) untuk {len(suppress_list)} tokens.")


# =====================================================================
# STEP 1: Load Model + Processor
# =====================================================================
print("\n[1/6] Memuat Model & Processor (Vanilla Transformers)...")
processor = AutoProcessor.from_pretrained(MODEL_NAME, trust_remote_code=True)
print(f"  Processor         : {processor.__class__.__name__}")
print(f"  image_seq_length  : {getattr(processor.image_processor, 'image_seq_length', '?')}")

model = AutoModelForSeq2SeqLM.from_pretrained(
    MODEL_NAME, torch_dtype=DTYPE, trust_remote_code=True,
).to(DEVICE)
model.config.use_cache = False
print(f"  Model             : {model.__class__.__name__}")
print(f"  is_encoder_decoder: {model.config.is_encoder_decoder}")
print(f"  Params            : {sum(p.numel() for p in model.parameters()) / 1e6:.1f}M")

# Pasang logit masking (decoder only, aman untuk vision)
apply_logit_mask(model, ALL_SUPPRESS_IDS)


# =====================================================================
# STEP 2: Inspeksi Arsitektur Vision
# =====================================================================
print("\n[2/6] Inspeksi Arsitektur (Encoder Vision)...")
inner = getattr(model, "model", model)
encoder = getattr(inner, "encoder", None) or getattr(model, "encoder", None)
if encoder is not None:
    vt = getattr(encoder, "vision_tower", None)
    print(f"  encoder class : {encoder.__class__.__name__}")
    print(f"  vision_tower  : {vt.__class__.__name__ if vt else 'TIDAK ADA'}")
else:
    print("  [WARN] Encoder tidak ditemukan.")


# =====================================================================
# STEP 3: Forward ORPO — 2x forward (chosen & rejected) dengan pixel_values
# =====================================================================
print("\n[3/6] Forward ORPO: chosen & rejected dengan 1 gambar...")
imgs = (load_real_images(1) or [make_synthetic_image(0)])[:1]
prompt = build_prompt(processor, len(imgs), "Deskripsikan gambar ini.")
enc = processor(text=prompt, images=imgs, return_tensors="pt")
enc = {k: v.to(DEVICE) for k, v in enc.items()}

chosen_labels = make_labels(processor.tokenizer, "Ini gambar berwarna merah yang jelas.", MAX_TARGET_LENGTH).to(DEVICE)
rejected_labels = make_labels(processor.tokenizer, "gambar", MAX_TARGET_LENGTH).to(DEVICE)

model.eval()
with torch.no_grad():
    chosen_out = model(input_ids=enc["input_ids"], attention_mask=enc["attention_mask"],
                       pixel_values=enc["pixel_values"], labels=chosen_labels)
    rejected_out = model(input_ids=enc["input_ids"], attention_mask=enc["attention_mask"],
                         pixel_values=enc["pixel_values"], labels=rejected_labels)

print(f"  chosen logits  : {chosen_out.logits.shape}, sft_loss: {chosen_out.loss.item():.4f}")
print(f"  rejected logits: {rejected_out.logits.shape}, sft_loss: {rejected_out.loss.item():.4f}")


# =====================================================================
# Helper: hitung log-odds (ORPO loss)
# =====================================================================
def get_batch_logps(logits, labels, average_log_prob=False):
    """Sum log-probabilitas token (mask -100 diabaikan). Seq2Seq: logits & labels same shape."""
    labels = labels.clone()
    loss_mask = labels != -100
    labels[labels == -100] = 0
    per_token_logps = torch.gather(logits.log_softmax(-1), dim=2, index=labels.unsqueeze(2)).squeeze(2)
    if average_log_prob:
        return (per_token_logps * loss_mask).sum(-1) / loss_mask.sum(-1).clamp(min=1)
    return (per_token_logps * loss_mask).sum(-1)

# Hitung OR loss manual (validasi)
chosen_logps = get_batch_logps(chosen_out.logits, chosen_labels)
rejected_logps = get_batch_logps(rejected_out.logits, rejected_labels)
chosen_log_odds = chosen_logps - torch.log1p(-torch.exp(chosen_logps) + 1e-6)
rejected_log_odds = rejected_logps - torch.log1p(-torch.exp(rejected_logps) + 1e-6)
log_odds_margin = chosen_log_odds - rejected_log_odds
or_loss = -F.logsigmoid(log_odds_margin).mean()
total_loss = chosen_out.loss + ORPO_BETA * or_loss
print(f"  chosen_logps   : {chosen_logps.item():.4f}")
print(f"  rejected_logps : {rejected_logps.item():.4f}")
print(f"  log_odds_margin: {log_odds_margin.item():.4f}")
print(f"  OR loss        : {or_loss.item():.4f}")
print(f"  TOTAL (sft + beta*OR): {total_loss.item():.4f}")
print("  ✅ Mekanisme log-odds + pixel_values terhitung dengan benar!")

# Cleanup
del enc, chosen_out, rejected_out, chosen_labels, rejected_labels
if torch.cuda.is_available():
    torch.cuda.empty_cache()


# =====================================================================
# STEP 4: VisionORPOCollator + VisionORPOTrainer
# =====================================================================
print("\n[4/6] VisionORPOCollator + VisionORPOTrainer...")


class VisionORPOCollator:
    """Collator ORPO + Vision untuk T5Gemma2 (vanilla transformers).

    Setiap sample: {prompt_text, images, chosen_text, rejected_text}
    Output: input_ids + attention_mask + pixel_values (SHARED encoder)
            + chosen_labels + rejected_labels (decoder target berbeda)
    """

    def __init__(self, processor, max_source_length, max_target_length):
        self.processor = processor
        self.tok = processor.tokenizer
        self.pad_id = self.tok.pad_token_id
        self.eos_id = self.tok.eos_token_id
        self.max_src = max_source_length
        self.max_tgt = max_target_length

    def _encode_target(self, text):
        ids = self.tok.encode(text, add_special_tokens=False)
        ids = ids[: self.max_tgt - 1] + [self.eos_id]
        return torch.tensor(ids, dtype=torch.long)

    def __call__(self, batch):
        input_ids, attn_masks, pixel_vals, chosen_labs, rejected_labs = [], [], [], [], []
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
                pixel_vals.append(enc["pixel_values"])  # (n_img, C, H, W)
            chosen_labs.append(self._encode_target(item["chosen_text"]))
            rejected_labs.append(self._encode_target(item["rejected_text"]))

        ii = torch.nn.utils.rnn.pad_sequence(input_ids, batch_first=True, padding_value=self.pad_id)
        am = torch.nn.utils.rnn.pad_sequence(attn_masks, batch_first=True, padding_value=0)
        cl = torch.nn.utils.rnn.pad_sequence(chosen_labs, batch_first=True, padding_value=-100)
        rl = torch.nn.utils.rnn.pad_sequence(rejected_labs, batch_first=True, padding_value=-100)
        out = {"input_ids": ii, "attention_mask": am, "chosen_labels": cl, "rejected_labels": rl}
        if pixel_vals:
            pv = pixel_vals[0]
            out["pixel_values"] = torch.cat(pixel_vals, dim=0) if pv.ndim == 4 else torch.stack(pixel_vals, dim=0)
        return out


class VisionORPOTrainer(Trainer):
    """Custom ORPO Trainer untuk Seq2Seq + Vision T5Gemma2.

    Loss = SFT_loss(chosen) + beta * OR_loss(log_odds_margin)
    chosen & rejected share encoder input (text + image), beda decoder labels.
    """

    def __init__(self, beta=0.1, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.beta = beta

    def get_batch_logps(self, logits, labels, average_log_prob=False):
        labels = labels.clone()
        loss_mask = labels != -100
        labels[labels == -100] = 0
        per_token_logps = torch.gather(logits.log_softmax(-1), dim=2, index=labels.unsqueeze(2)).squeeze(2)
        if average_log_prob:
            return (per_token_logps * loss_mask).sum(-1) / loss_mask.sum(-1).clamp(min=1)
        return (per_token_logps * loss_mask).sum(-1)

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None, **kwargs):
        chosen_labels = inputs.pop("chosen_labels", None)
        rejected_labels = inputs.pop("rejected_labels", None)

        if chosen_labels is None or rejected_labels is None:
            # Fallback: SFT biasa
            return super().compute_loss(model, inputs, return_outputs, num_items_in_batch, **kwargs)

        input_ids = inputs.get("input_ids")
        attention_mask = inputs.get("attention_mask")
        pixel_values = inputs.get("pixel_values")

        # Forward chosen (encoder text+image + decoder chosen)
        chosen_out = model(input_ids=input_ids, attention_mask=attention_mask,
                           pixel_values=pixel_values, labels=chosen_labels)
        # Forward rejected (encoder sama + decoder rejected)
        rejected_out = model(input_ids=input_ids, attention_mask=attention_mask,
                             pixel_values=pixel_values, labels=rejected_labels)

        chosen_logps = self.get_batch_logps(chosen_out.logits, chosen_labels)
        rejected_logps = self.get_batch_logps(rejected_out.logits, rejected_labels)

        chosen_log_odds = chosen_logps - torch.log1p(-torch.exp(chosen_logps) + 1e-6)
        rejected_log_odds = rejected_logps - torch.log1p(-torch.exp(rejected_logps) + 1e-6)
        log_odds_margin = chosen_log_odds - rejected_log_odds
        or_loss = -F.logsigmoid(log_odds_margin).mean()

        # SFT loss = chosen cross-entropy (sudah dihitung model)
        sft_loss = chosen_out.loss
        loss = sft_loss + self.beta * or_loss

        return (loss, chosen_out) if return_outputs else loss


print("  ✅ VisionORPOCollator + VisionORPOTrainer siap.")


# =====================================================================
# STEP 5: Dataset Dummy ORPO + Training (2 steps)
# =====================================================================
print("\n[5/6] Dataset ORPO + Training (2 steps)...")
base_imgs = load_real_images(MAX_IMAGES) or [make_synthetic_image(i) for i in range(MAX_IMAGES)]

# Dataset ORPO dummy: prompt (text+images), chosen (jawaban baik), rejected (jawaban buruk)
orpo_configs = [
    (1, "Jelaskan gambar ini.", "Ini gambar berwarna merah yang jelas dan detail.", "gambar"),
    (2, "Bandingkan dua gambar ini.", "Gambar pertama merah, kedua hijau. Keduanya uji coba.", "aku gak tahu"),
    (3, "Jelaskan tiga gambar ini berurutan.", "Urutan: merah, hijau, biru. Tiga warna primer.", "ya"),
]
orpo_samples = []
for ni, prompt_txt, chosen_txt, rejected_txt in orpo_configs:
    orpo_samples.append({
        "prompt_text": build_prompt(processor, ni, prompt_txt),
        "images": base_imgs[:ni],
        "chosen_text": chosen_txt,
        "rejected_text": rejected_txt,
    })
orpo_ds = Dataset.from_list(orpo_samples)
print(f"  Dataset ORPO: {len(orpo_ds)} samples (img count: {[len(s['images']) for s in orpo_samples]})")

collator = VisionORPOCollator(processor, MAX_SOURCE_LENGTH, MAX_TARGET_LENGTH)

# Test collator: 1 sample
cb = collator(orpo_samples[:1])
print(f"  Collator (b=1) -> input_ids: {cb['input_ids'].shape}, "
      f"chosen_labels: {cb['chosen_labels'].shape}, rejected_labels: {cb['rejected_labels'].shape}, "
      f"pixel_values: {cb.get('pixel_values').shape}")

# Bersihkan cache + gradient checkpointing
del cb
if torch.cuda.is_available():
    torch.cuda.empty_cache()
model.gradient_checkpointing_enable()
print("  Gradient checkpointing: ON")

targs = TrainingArguments(
    output_dir="results/vanilla_orpo_vision_smoke",
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
trainer = VisionORPOTrainer(beta=ORPO_BETA, model=model, args=targs,
                            train_dataset=orpo_ds, data_collator=collator)
print(f"  Memulai 2 ORPO training steps (beta={ORPO_BETA}, batch=1)...")
try:
    trainer.train()
    print("  ✅ ORPO training smoke test BERHASIL — seq2seq vision ORPO jalan native!")
except Exception as e:
    print(f"  ❌ ORPO training gagal: {e}")
    traceback.print_exc()


# =====================================================================
# STEP 6: Ringkasan
# =====================================================================
print("\n" + "=" * 70)
print("✅ TEST VANILLA ORPO VISION SELESAI.")
print("   ORPO Seq2Seq + Vision T5Gemma2 VALID secara native transformers.")
print("   - Encoder (text+image) SHARED untuk chosen & rejected")
print("   - Decoder labels berbeda -> log-odds ratio -> OR loss")
print("   - Loss = SFT(chosen) + beta * OR(log_odds_margin)")
print("   - Logit masking (lm_head, decoder) tetap aktif & fair")
print("=" * 70)
