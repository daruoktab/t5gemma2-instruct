import os
os.environ["UNSLOTH_COMPILE_DISABLE"] = "1"
import torch
setattr(torch._dynamo.config, "recompile_limit", 128)
import torch.nn.functional as F
import json, random, traceback
from PIL import Image
from datasets import Dataset
from unsloth import FastVisionModel
from transformers import AutoProcessor, AutoTokenizer, Seq2SeqTrainer, Seq2SeqTrainingArguments

MODEL_NAME = "google/t5gemma-2-270m-270m"
OUTPUT_DIR = "results/test_vision_output"
LOAD_IN_4BIT = True
SFT_DATASET_PATH = "data/multimodal/train_vision.jsonl"
ORPO_DATASET_PATH = "data/preference/orpo_multimodal.jsonl"
SFT_DOC_COUNT = 20
SFT_IMG_COUNT = 80
ORPO_DOC_COUNT = 2
ORPO_IMG_COUNT = 8
MAX_IMAGES_PER_CONVERSATION = 10
MAX_SOURCE_LENGTH = 4096
MAX_TARGET_LENGTH = 512
ORPO_BETA = 0.1
SEED = 3407
random.seed(SEED)

SUPPRESS_BLOCK1 = [6] + list(range(13, 105))
SUPPRESS_BLOCK2 = list(range(256002, 262144))
SUPPRESS_VISION = [255999, 256000, 256001]
ALL_SUPPRESS_IDS = set(SUPPRESS_BLOCK1 + SUPPRESS_BLOCK2 + SUPPRESS_VISION)

print("=" * 70)
print("=== TEST V6 VISION - Seq2Seq + Logit Masking (270M) ===")
print("=" * 70)
os.makedirs(OUTPUT_DIR, exist_ok=True)


def apply_logit_mask(model, suppress_ids):
    """Hook di lm_head (decoder output). Encoder tidak kena."""
    vs = model.config.vocab_size
    sl = [i for i in suppress_ids if i < vs]
    mask = torch.zeros(vs, dtype=torch.bfloat16)
    mask[sl] = -10000.0

    def hook(mod, inp, out):
        if isinstance(out, torch.Tensor):
            return out + mask.to(out.device)
        elif hasattr(out, "logits"):
            out.logits = out.logits + mask.to(out.logits.device)
            return out
        elif isinstance(out, tuple) and out and isinstance(out[0], torch.Tensor):
            return (out[0] + mask.to(out[0].device),) + out[1:]
        return out

    t = None
    if hasattr(model, "lm_head"):
        t = model.lm_head
    elif hasattr(model, "base_model") and hasattr(model.base_model, "lm_head"):
        t = model.base_model.lm_head
    elif hasattr(model, "base_model") and hasattr(model.base_model, "model") and hasattr(model.base_model.model, "lm_head"):
        t = model.base_model.model.lm_head
    if t is not None:
        t.register_forward_hook(hook)
        print(f"  ✅ Logit mask (lm_head) untuk {len(sl)} tokens.")
    else:
        model.register_forward_hook(hook)
        print(f"  ✅ Logit mask (fallback) untuk {len(sl)} tokens.")


def convert_sft_record_to_vision(rec):
    """Konversi 1 percakapan train_vision.jsonl → messages format."""
    img_paths = rec.get("images", [])
    pil_images = [Image.open(p).convert("RGB") for p in img_paths if os.path.exists(p)]
    if not pil_images:
        return None
    old_messages = rec.get("messages", [])
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
    return {"messages": new_messages}


def unroll_vision_messages_to_sft_samples(messages, processor):
    """Unroll multi-turn conversation ke individual SFT samples untuk Seq2Seq."""
    samples = []
    for i, msg in enumerate(messages):
        if msg["role"] != "assistant":
            continue
        context = messages[:i]
        if not context:
            continue
        prompt_text = processor.apply_chat_template(context, tokenize=False, add_generation_prompt=True)
        images = []
        for m in context:
            if isinstance(m.get("content"), list):
                for b in m["content"]:
                    if isinstance(b, dict) and b.get("type") == "image":
                        images.append(b["image"])
        target_text = ""
        if isinstance(msg["content"], list):
            for b in msg["content"]:
                if isinstance(b, dict) and b.get("type") == "text":
                    target_text = b["text"]
        else:
            target_text = msg["content"]
        if target_text:
            samples.append({"prompt_text": prompt_text, "images": images, "target_text": target_text})
    return samples


def parse_orpo_prompt_to_messages(prompt_str, img_paths):
    """Parse ORPO prompt string → list of messages dengan image blocks."""
    pil_images = [Image.open(p).convert("RGB") for p in img_paths if os.path.exists(p)]
    lines = prompt_str.split("\n")
    raw_messages = []
    current_role = None
    current_lines = []
    for line in lines:
        if line.startswith("system: "):
            if current_role is not None:
                raw_messages.append((current_role, "\n".join(current_lines)))
            current_role = "system"
            current_lines = [line[8:]]
        elif line.startswith("user: "):
            if current_role is not None:
                raw_messages.append((current_role, "\n".join(current_lines)))
            current_role = "user"
            current_lines = [line[6:]]
        elif line.startswith("assistant: "):
            if current_role is not None:
                raw_messages.append((current_role, "\n".join(current_lines)))
            current_role = "assistant"
            current_lines = [line[11:]]
        else:
            current_lines.append(line)
    if current_role is not None:
        raw_messages.append((current_role, "\n".join(current_lines)))
    new_messages = []
    image_idx = 0
    for role, content in raw_messages:
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
    return new_messages


class Seq2SeqVisionCollator:
    """Collator Seq2Seq + Vision untuk T5Gemma2 SFT."""
    def __init__(self, processor, max_src, max_tgt):
        self.processor = processor
        self.tok = processor.tokenizer
        self.pad_id = self.tok.pad_token_id
        self.eos_id = self.tok.eos_token_id
        self.max_src = max_src
        self.max_tgt = max_tgt

    def __call__(self, batch):
        iids, amasks, pvals, labs = [], [], [], []
        for item in batch:
            enc = self.processor(text=item["prompt_text"],
                images=item["images"] if item["images"] else None,
                return_tensors="pt")
            iids.append(enc["input_ids"][0])
            amasks.append(enc["attention_mask"][0])
            if "pixel_values" in enc:
                pvals.append(enc["pixel_values"])
            tids = self.tok.encode(item["target_text"], add_special_tokens=False)
            tids = tids[:self.max_tgt-1] + [self.eos_id]
            labs.append(torch.tensor(tids, dtype=torch.long))
        ii = torch.nn.utils.rnn.pad_sequence(iids, batch_first=True, padding_value=self.pad_id)
        am = torch.nn.utils.rnn.pad_sequence(amasks, batch_first=True, padding_value=0)
        lb = torch.nn.utils.rnn.pad_sequence(labs, batch_first=True, padding_value=-100)
        out = {"input_ids": ii, "attention_mask": am, "labels": lb}
        if pvals:
            out["pixel_values"] = torch.cat(pvals, dim=0) if pvals[0].ndim == 4 else torch.stack(pvals, dim=0)
        return out


class VisionORPOCollator:
    """Collator ORPO + Vision. Encoder SHARED, decoder labels berbeda."""
    def __init__(self, processor, max_src, max_tgt):
        self.processor = processor
        self.tok = processor.tokenizer
        self.pad_id = self.tok.pad_token_id
        self.eos_id = self.tok.eos_token_id
        self.max_src = max_src
        self.max_tgt = max_tgt

    def _enc_tgt(self, text):
        ids = self.tok.encode(text, add_special_tokens=False)
        return torch.tensor(ids[:self.max_tgt-1] + [self.eos_id], dtype=torch.long)

    def __call__(self, batch):
        iids, amasks, pvals, clabs, rlabs = [], [], [], [], []
        for item in batch:
            enc = self.processor(text=item["prompt_text"],
                images=item["images"] if item["images"] else None,
                return_tensors="pt")
            iids.append(enc["input_ids"][0])
            amasks.append(enc["attention_mask"][0])
            if "pixel_values" in enc:
                pvals.append(enc["pixel_values"])
            clabs.append(self._enc_tgt(item["chosen_text"]))
            rlabs.append(self._enc_tgt(item["rejected_text"]))
        ii = torch.nn.utils.rnn.pad_sequence(iids, batch_first=True, padding_value=self.pad_id)
        am = torch.nn.utils.rnn.pad_sequence(amasks, batch_first=True, padding_value=0)
        cl = torch.nn.utils.rnn.pad_sequence(clabs, batch_first=True, padding_value=-100)
        rl = torch.nn.utils.rnn.pad_sequence(rlabs, batch_first=True, padding_value=-100)
        out = {"input_ids": ii, "attention_mask": am, "chosen_labels": cl, "rejected_labels": rl}
        if pvals:
            out["pixel_values"] = torch.cat(pvals, dim=0) if pvals[0].ndim == 4 else torch.stack(pvals, dim=0)
        return out


class VisionORPOTrainer(Seq2SeqTrainer):
    """Custom ORPO: Loss = SFT(chosen) + beta * OR(log_odds_margin)."""
    def __init__(self, beta=0.1, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.beta = beta

    def get_batch_logps(self, logits, labels, average_log_prob=True):
        labels = labels.clone()
        mask = labels != -100
        labels[labels == -100] = 0
        lps = torch.gather(logits.log_softmax(-1), dim=2, index=labels.unsqueeze(2)).squeeze(2)
        if average_log_prob:
            return (lps * mask).sum(-1) / mask.sum(-1).clamp(min=1)
        return (lps * mask).sum(-1)

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None, **kwargs):
        cl = inputs.pop("chosen_labels", None)
        rl = inputs.pop("rejected_labels", None)
        if cl is None or rl is None:
            return super().compute_loss(model, inputs, return_outputs, num_items_in_batch, **kwargs)
        iids = inputs.get("input_ids")
        amask = inputs.get("attention_mask")
        pvals = inputs.get("pixel_values")
        
        # Split forward: Run encoder once to save VRAM
        encoder = model.get_encoder()
        enc_out = encoder(input_ids=iids, attention_mask=amask, pixel_values=pvals)
        
        co = model(encoder_outputs=enc_out, attention_mask=amask, labels=cl)
        ro = model(encoder_outputs=enc_out, attention_mask=amask, labels=rl)
        
        clp = self.get_batch_logps(co.logits, cl)
        rlp = self.get_batch_logps(ro.logits, rl)
        cp = clp.exp().clamp(1e-7, 1-1e-7)
        rp = rlp.exp().clamp(1e-7, 1-1e-7)
        clo = torch.log(cp / (1 - cp))
        rlo = torch.log(rp / (1 - rp))
        or_loss = -F.logsigmoid(clo - rlo).mean()
        loss = co.loss + self.beta * or_loss
        return (loss, co) if return_outputs else loss


# STEP 1: Load processor
print("\n=== 1. Memuat Processor ===")
token = os.environ.get("HF_TOKEN")
processor = AutoProcessor.from_pretrained(MODEL_NAME, token=token)
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, token=token)
from unsloth.chat_templates import get_chat_template
tokenizer = get_chat_template(tokenizer, chat_template="gemma-3")
processor.chat_template = tokenizer.chat_template
if hasattr(processor, "tokenizer"):
    processor.tokenizer.chat_template = tokenizer.chat_template
tokenizer.add_bos_token = False
if hasattr(processor, "tokenizer"):
    processor.tokenizer.add_bos_token = False
print("  ✅ Processor + chat template siap.")

# STEP 2: Load & prepare SFT dataset (unroll multi-turn)
print("\n=== 2. Mempersiapkan Dataset SFT (8+ gambar saja) ===")
with open(SFT_DATASET_PATH, "r", encoding="utf-8") as f:
    sft_records = [json.loads(l.strip()) for l in f]
sft_samples = [r for r in sft_records if len(r.get("images", [])) >= 8]
random.shuffle(sft_samples)
sft_formatted = []
for rec in sft_samples:
    conv = convert_sft_record_to_vision(rec)
    if conv:
        unrolled = unroll_vision_messages_to_sft_samples(conv["messages"], processor)
        for s in unrolled:
            enc = processor(text=s["prompt_text"], images=s["images"] if s["images"] else None, return_tensors="pt")
            token_len = enc["input_ids"].shape[1]
            if token_len <= MAX_SOURCE_LENGTH:
                sft_formatted.append(s)
            else:
                print(f"  [Skip SFT] ID {rec['id']} too long: {token_len} tokens")
sft_dataset = Dataset.from_list(sft_formatted)
print(f"  ✅ SFT dataset: {len(sft_dataset)} samples (dari {len(sft_samples)} percakapan)")

# STEP 3: Load & prepare ORPO dataset
print(f"\n=== 3. Mempersiapkan Dataset ORPO (8+ gambar saja) ===")
with open(ORPO_DATASET_PATH, "r", encoding="utf-8") as f:
    orpo_records = [json.loads(l.strip()) for l in f]
orpo_samples = [r for r in orpo_records if len(r.get("images", [])) >= 8]
random.shuffle(orpo_samples)
orpo_formatted = []
for rec in orpo_samples:
    msgs = parse_orpo_prompt_to_messages(rec.get("prompt", ""), rec.get("images", []))
    if msgs:
        pt = processor.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
        imgs = [b["image"] for m in msgs if isinstance(m.get("content"), list)
                for b in m["content"] if isinstance(b, dict) and b.get("type") == "image"]
        enc = processor(text=pt, images=imgs if imgs else None, return_tensors="pt")
        token_len = enc["input_ids"].shape[1]
        if token_len <= MAX_SOURCE_LENGTH:
            orpo_formatted.append({"prompt_text": pt, "images": imgs,
                "chosen_text": rec.get("chosen", ""), "rejected_text": rec.get("rejected", "")})
        else:
            print(f"  [Skip ORPO] ID {rec['id']} too long: {token_len} tokens")
orpo_dataset = Dataset.from_list(orpo_formatted)
print(f"  ✅ ORPO dataset: {len(orpo_dataset)} turns")

# STEP 4: Load Model + LoRA + Logit Masking
print("\n=== 4. Memuat Model + LoRA + Logit Masking ===")
model, tokenizer = FastVisionModel.from_pretrained(
    model_name=MODEL_NAME, load_in_4bit=LOAD_IN_4BIT,
    max_seq_length=MAX_SOURCE_LENGTH,
    use_gradient_checkpointing="unsloth", token=token)
model = FastVisionModel.get_peft_model(model,
    finetune_vision_layers=False, finetune_language_layers=True,
    finetune_attention_modules=True, finetune_mlp_modules=True,
    r=8, lora_alpha=16, lora_dropout=0.0, bias="none", random_state=SEED)
if not hasattr(model.config, "text_config"):
    type(model.config).text_config = property(lambda self: self.decoder)
    type(model.config).get_text_config = lambda self, *a, **kw: self.decoder
apply_logit_mask(model, ALL_SUPPRESS_IDS)
FastVisionModel.for_training(model)
print("  ✅ Model + LoRA + logit mask siap.")

# STEP 5: SFT Training (Seq2SeqTrainer + custom collator)
print(f"\n=== 5. SFT Training (Seq2SeqTrainer, max_src={MAX_SOURCE_LENGTH}) ===")
sft_collator = Seq2SeqVisionCollator(processor, MAX_SOURCE_LENGTH, MAX_TARGET_LENGTH)
sft_trainer = Seq2SeqTrainer(model=model,
    args=Seq2SeqTrainingArguments(
        per_device_train_batch_size=1, gradient_accumulation_steps=1,
        learning_rate=2e-4, max_steps=5, logging_steps=1,
        output_dir=os.path.join(OUTPUT_DIR, "sft"),
        remove_unused_columns=False, fp16=False,
        bf16=torch.cuda.is_available(), gradient_checkpointing=True,
        save_strategy="no", report_to="none", predict_with_generate=False),
    train_dataset=sft_dataset, data_collator=sft_collator)
print("  Memulai SFT training...")
try:
    sft_result = sft_trainer.train()
    print(f"  ✅ SFT selesai! Loss: {sft_result.training_loss:.4f}")
except Exception as e:
    print(f"  ❌ SFT gagal: {e}")
    traceback.print_exc()
finally:
    # Delete trainer and free VRAM
    del sft_trainer
    import gc
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

# STEP 6: ORPO Training (VisionORPOTrainer + custom collator)
print(f"\n=== 6. ORPO Training (VisionORPOTrainer, beta={ORPO_BETA}) ===")
if torch.cuda.is_available():
    torch.cuda.empty_cache()
orpo_collator = VisionORPOCollator(processor, MAX_SOURCE_LENGTH, MAX_TARGET_LENGTH)
try:
    orpo_trainer = VisionORPOTrainer(beta=ORPO_BETA, model=model,
        args=Seq2SeqTrainingArguments(
            per_device_train_batch_size=1, gradient_accumulation_steps=1,
            learning_rate=1e-5, max_steps=3, logging_steps=1,
            output_dir=os.path.join(OUTPUT_DIR, "orpo"),
            remove_unused_columns=False, fp16=False,
            bf16=torch.cuda.is_available(), gradient_checkpointing=True,
            save_strategy="no", report_to="none", predict_with_generate=False),
        train_dataset=orpo_dataset, data_collator=orpo_collator)
    print("  Memulai ORPO training...")
    orpo_result = orpo_trainer.train()
    print(f"  ✅ ORPO selesai! Loss: {orpo_result.training_loss:.4f}")
except Exception as e:
    print(f"  ❌ ORPO gagal: {e}")
    traceback.print_exc()
    print("  ⚠️ Lanjut ke inferensi...")
finally:
    # Delete trainer and free VRAM
    if 'orpo_trainer' in locals():
        del orpo_trainer
    import gc
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

# STEP 7: Evaluasi Inferensi
print("\n" + "=" * 70)
print("=== 7. EVALUASI GENERASI ===")
print("=" * 70)
FastVisionModel.for_inference(model)
SUPPRESS_GEN = sorted([i for i in ALL_SUPPRESS_IDS if i < model.config.vocab_size])

def format_single_rec(rec, processor):
    conv = convert_sft_record_to_vision(rec)
    if conv:
        unrolled = unroll_vision_messages_to_sft_samples(conv["messages"], processor)
        if unrolled:
            return unrolled[0]
    return None

with torch.no_grad():
    model.eval()
    
    # Cari sampel dengan tepat 1 gambar dari dataset asli agar inferensi ringan
    rec1 = next((r for r in sft_records if len(r.get("images", [])) == 1), None)
    test_sample = format_single_rec(rec1, processor) if rec1 else None
    
    if test_sample:
        print("\n[ 📸 TEST: 1 GAMBAR ]")
        print("-" * 60)
        inputs = processor(text=test_sample["prompt_text"], images=test_sample["images"],
                           return_tensors="pt").to(model.device)
        outputs = model.generate(**inputs, max_new_tokens=64, use_cache=True,
                                 suppress_tokens=SUPPRESS_GEN, repetition_penalty=1.2)
        response = processor.decode(outputs[0], skip_special_tokens=True)
        print(f"Target : {test_sample['target_text'][:80]}...")
        print(f"Model  : {response.strip()[:200]}")
        print("-" * 60)
        
    # Cari sampel dengan tepat 2 gambar dari dataset asli agar inferensi ringan
    rec2 = next((r for r in sft_records if len(r.get("images", [])) == 2), None)
    multi_test = format_single_rec(rec2, processor) if rec2 else None
    
    if multi_test:
        print("\n[ 📸 TEST: MULTI-IMAGE ]")
        print("-" * 60)
        inputs = processor(text=multi_test["prompt_text"], images=multi_test["images"],
                           return_tensors="pt").to(model.device)
        outputs = model.generate(**inputs, max_new_tokens=64, use_cache=True,
                                 suppress_tokens=SUPPRESS_GEN, repetition_penalty=1.2)
        response = processor.decode(outputs[0], skip_special_tokens=True)
        print(f"Images : {len(multi_test['images'])} gambar")
        print(f"Model  : {response.strip()[:200]}")
        print("-" * 60)
print("\n" + "=" * 70)
print("✅ TEST SELESAI! Pipeline SFT + ORPO vision diuji lokal.")
print("   Seq2Seq + Logit Masking + Custom Collator + VisionORPOTrainer")
print("=" * 70)