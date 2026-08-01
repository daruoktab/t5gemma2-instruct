# Diagnostic Report: Vision Training Mechanism

> Dijalankan dengan `vision_diagnostic.py` — mock model CPU-only, 7 tests, 33 checks

---

## Ringkasan Eksekutif

```
Total checks : 33
Passed       : 19
Failed       : 4   ← bugs confirmed
Warnings     : 10
```

**4 bug dikonfirmasi oleh diagnostic.** Berikut detailnya dari paling kritis:

---

## BUG #3 — KRITIS: Model Stuck in Eval Mode Setelah Setiap Evaluate()

**Status: ✅ CONFIRMED**

**Bukti dari diagnostic:**
```
State SEBELUM evaluate (vision): model.training=True
State SETELAH evaluate (vision): model.training=False   ← STUCK!
for_inference called: False
for_training called: False
```

**Perbandingan dengan text-only yang benar:**
```
State SEBELUM evaluate (text-only): model.training=True
State SETELAH evaluate (text-only): model.training=True ← KEMBALI NORMAL
for_inference called: True
for_training called: True
```

**Lokasi bug:** `CustomSeq2SeqTrainer.evaluate()` di vision code (sekitar line 863-898)

**Vision code:**
```python
def evaluate(self, ...):
    gc.collect()
    self.model.eval()           # ← hanya ini
    metrics = super().evaluate(...)
    torch._dynamo.reset()       # ← tidak ada model.train() / for_training() !!
    gc.collect()
    return metrics
```

**Text-only code yang benar:**
```python
def evaluate(self, ...):
    FastLanguageModel.for_inference(self.model)  # ← switch ke inference kernels
    metrics = super().evaluate(...)
    FastLanguageModel.for_training(self.model)   # ← switch KEMBALI ke training kernels
    self.model.train()                           # ← set training mode
    gc.collect()
    return metrics
```

**Dampak:**
- Setelah tiap evaluation step, model tetap dalam `model.training=False`
- Di Unsloth: tanpa `for_training()` → custom triton attention kernels (training mode) tidak diaktifkan kembali
- Training dilanjutkan dengan model dalam eval state → gradient checkpointing tidak aktif → memory leak + gradient tidak optimal
- **Ini silent bug**: tidak crash, loss tetap turun, tapi kualitas output tidak membaik karena training kernels tidak aktif

**Fix:**
```python
def evaluate(self, eval_dataset=None, ignore_keys=None, metric_key_prefix="eval"):
    from unsloth import FastVisionModel
    import gc, torch
    if hasattr(FastVisionModel, "for_inference"):
        FastVisionModel.for_inference(self.model)   # ← TAMBAH INI
    
    gc.collect()
    self.model.eval()
    metrics = super().evaluate(
        eval_dataset=eval_dataset,
        ignore_keys=ignore_keys,
        metric_key_prefix=metric_key_prefix
    )
    torch._dynamo.reset()
    
    if hasattr(FastVisionModel, "for_training"):
        FastVisionModel.for_training(self.model)    # ← TAMBAH INI
    self.model.train()                              # ← TAMBAH INI
    
    gc.collect()
    return metrics
```

---

## BUG #4 — KRITIS: BOS Mismatch antara Training dan Validation

**Status: ✅ CONFIRMED**

**Bukti dari diagnostic:**
```
processor.__call__() output    : [2, 500, 501, 1]   ← BOS ada (ID=2)
processor.tokenizer.encode()   : [500, 501, 1]       ← TIDAK ada BOS!

Training has BOS at start  : True
Validation has BOS at start: False

Training seq len : 5
Validation seq len: 4          ← 1 token lebih pendek
```

**Mekanisme bug:**
1. Kode vision (line 1432): `tokenizer.add_bos_token = False`
2. Kode vision (line 1434): `processor.tokenizer.add_bos_token = False`
3. **Asumsi**: setting ini membuat processor juga tidak menambah BOS
4. **Kenyataan**: `Gemma3Processor.__call__()` menambah BOS secara **hardcoded** di internal-nya, bukan via `tokenizer.add_bos_token`. Setting `add_bos_token=False` hanya mempengaruhi `tokenizer.encode()`.

**Akibatnya:**
- Training data (masuk via `Seq2SeqVisionCollator` → `processor.__call__()`): punya BOS
- Validation data (masuk via `run_eval()` → `tokenizer.encode(add_special_tokens=True)`): tidak punya BOS
- Model belajar mengharapkan BOS di awal input, tapi saat validation ditest tanpa BOS → kualitas output lebih buruk dari seharusnya
- **Validation metrics misleading**: skor lebih rendah bukan karena model jelek, tapi karena input validation formatnya berbeda

**Fix:**
```python
# Di run_eval() dan process_sft_rows(), paksa ada BOS:
inp_ids = tokenizer.encode(inp_f, add_special_tokens=True)
if inp_ids and inp_ids[0] != tokenizer.bos_token_id:
    inp_ids = [tokenizer.bos_token_id] + inp_ids   # ← ensure BOS konsisten

# ATAU: gunakan processor untuk encode validation juga (lebih konsisten):
enc = processor(text=inp_f, images=None, return_tensors="pt")
inp_ids = enc["input_ids"][0].tolist()
```

---

## BUG #5 — PENTING: Effective Learning Rate 8x Lebih Agresif dari Text-Only

**Status: ✅ CONFIRMED**

**Bukti dari diagnostic:**
```
Vision:    LR=2e-5, grad_accum=16 → effective LR = 1.25e-06
Text-only: LR=1e-5, grad_accum=64 → effective LR = 1.56e-07
Ratio: 8.0x  ← vision training 8x lebih agresif!
```

**Juga: RSLoRA scaling 16x lebih besar dari standard LoRA:**
```
Standard LoRA scaling (alpha/r)    : 2.0
RSLoRA scaling (alpha/sqrt(r))     : 32.0   ← 16x lebih besar!
```

**Dampak:**
- Decoder menerima update yang jauh lebih besar per step dari sinyal multimodal
- Jika visual projector belum alignment sempurna → decoder belajar dari noise dengan LR besar
- Catastrophic forgetting lebih cepat dari text-only capability
- Dengan `use_rslora=True` + `r=256, alpha=512`: scaling 32x menyebabkan update magnitude sangat besar

**Fix:**
```python
# Turunkan LR dan naikkan grad_accum agar effective LR mendekati text-only:
LEARNING_RATE = 5e-6            # dari 2e-5 → turun 4x
GRADIENT_ACCUMULATION_STEPS = 32  # dari 16 → naik 2x
# Effective LR = 5e-6/32 = 1.56e-7 ← sama dengan text-only!

# Atau hapus RSLoRA:
use_rslora = False  # dari True → gunakan standard LoRA scaling
```

---

## BUG #2 — MINOR: ORPO Path pixel_values tidak masuk ke model.forward()

**Status: ⚠️ PARTIAL (tidak crash, tapi ada nuance)**

**Bukti dari diagnostic:**
```
Encoder call count: 1
Encoder received pixel_values: True          ← encoder MENERIMA pixel_values

model.forward() call #1: encoder_outputs=True, pv=False   ← model forward TANPA pv
model.forward() call #2: encoder_outputs=True, pv=False   ← model forward TANPA pv
```

**Penjelasan:**
- Encoder dipanggil langsung dengan `pixel_values=...` → OK, visual info masuk ke encoder
- `encoder_outputs` (hasil encoder) kemudian di-share untuk chosen dan rejected
- `model.forward()` dipanggil dengan `encoder_outputs` (pre-computed), bukan `pixel_values`

**Ini sebenarnya benar** secara logika — encoder hanya perlu dipanggil sekali, hasilnya di-share.

**TAPI ada risiko di PEFT-wrapped model:**
- Jika `get_encoder()` via PEFT path tidak mengembalikan encoder asli
- LoRA wrapper mungkin tidak forward `pixel_values` ke base encoder
- Di mock (tanpa PEFT): works fine. Di real model dengan LoRA: perlu verifikasi

---

## BUG #1 — CLEARED: BOS/EOS Double Addition

**Status: ✅ NOT A BUG (diagnostic confirmed)**

**Bukti dari diagnostic:**
```
Before collator check: [2, 500, 501, 502, 503, 1]
After collator check : [2, 500, 501, 502, 503, 1]   ← identik, tidak ada duplikasi!
BOS added by collator: False
EOS added by collator: False
```

Penjelasan: Kondisi check di collator (`input_ids[0] != bos_token_id`) sudah benar — karena processor sudah menambah BOS, kondisi check menghasilkan `False` sehingga BOS tidak ditambah lagi. Collator BOS/EOS logic bekerja dengan benar.

---

## Visualisasi Alur Bug

```
TRAINING LOOP (vision):
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Batch (via Seq2SeqVisionCollator):
  [BOS, img_tokens, text, EOS]  ← Training input DENGAN BOS ✅
  
  ↓ forward pass (model.train() = True)
  ↓ compute loss
  ↓ backward
  ↓ optimizer.step()
  
  ┌─────────────────────────────────────┐
  │ evaluate() dipanggil                │
  │   model.eval()          ← OK       │
  │   super().evaluate()               │
  │   torch._dynamo.reset()            │
  │   # MISSING: model.train()         │  ← BUG #3
  │   # MISSING: for_training()        │
  └─────────────────────────────────────┘
  
  ↓ Training continues... tapi model.training = FALSE!
  ↓ Unsloth training kernels TIDAK aktif
  ↓ Gradient tidak optimal → silent degradation
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

VALIDATION (via run_eval()):
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  tokenizer.encode(inp_f, add_special_tokens=True)
  → add_bos_token=False → [text, EOS]  TANPA BOS!  ← BUG #4
  
  vs training: [BOS, text, EOS]
  
  Model belajar dengan BOS, ditest tanpa BOS
  → Metrics misleading + performance drop dalam inference
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
```

---

## Fix Priority Matrix

| Bug | Severity | Fix Effort | Expected Impact |
|-----|----------|-----------|-----------------|
| #3: Model stuck eval | 🔴 KRITIS | 10 menit (3 baris) | Gradient update benar → peningkatan terbesar |
| #4: BOS mismatch | 🔴 KRITIS | 15 menit | Validation metrics akurat, inference konsisten |
| #5: LR 8x agresif | 🟠 PENTING | 2 menit (ubah konstanta) | Stabilitas training, less forgetting |
| #2: ORPO pixel_values | 🟡 MINOR | 30 menit (verifikasi) | Mungkin sudah OK di real PEFT model |

---

> Diagnostic script: [vision_diagnostic.py](file:///d:/Codings/unsloth-porto/t5-gemma-2/instruct/vision_diagnostic.py)
> Kode vision: [working-molab-v6-vision-unsloth.py](file:///d:/Codings/unsloth-porto/t5-gemma-2/instruct/working-molab-v6-vision-unsloth.py)
