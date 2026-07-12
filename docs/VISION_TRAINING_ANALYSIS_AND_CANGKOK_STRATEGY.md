# [MASTER] Analisis Vision T5Gemma2, Cangkok SigLIP & Strategi Training Vision

**Last Updated:** 8 Juli 2026
**Status:** Cangkok selesai, vision training siap dijalankan
**Models:** `google/t5gemma-2-4b-4b` | `google/gemma-3-4b-it` | `daruokta/t5gemma-2-4b-4b-instruct-chat-indo-v4-vision-cangkok`

---

## Daftar Isi

1. [Arsitektur T5Gemma2 vs Gemma 3](#1-arsitektur-t5gemma2-vs-gemma-3)
2. [Mekanisme Image Encoder (SigLIP)](#2-mekanisme-image-encoder-siglip)
3. [Soft Token vs Hard Token](#3-soft-token-vs-hard-token)
4. [Perbandingan Config: T5Gemma2 vs Gemma 3 IT](#4-perbandingan-config-t5gemma2-vs-gemma-3-it)
5. [Logit Masking & Suppression](#5-logit-masking--suppression)
6. [Isu Seq2Seq di Unsloth & TRL](#6-isu-seq2seq-di-unsloth--trl)
7. [Validasi Vanilla Seq2Seq + Vision](#7-validasi-vanilla-seq2seq--vision)
8. [Validasi Vanilla ORPO Vision](#8-validasi-vanilla-orpo-vision)
9. [Verifikasi Nama Module (Empiris)](#9-verifikasi-nama-module-empiris)
10. ["Bocor" LoRA ke Vision Tower](#10-bocor-lora-ke-vision-tower)
11. [Verifikasi 3-Arah Bobot Vision](#11-verifikasi-3-arah-bobot-vision)
12. [Cangkok SigLIP + Projector](#12-cangkok-siglip--projector)
13. [Decoder Transplant vs SigLIP Transplant](#13-decoder-transplant-vs-siglip-transplant)
14. [Fix Tokenizer Config Repo Cangkok](#14-fix-tokenizer-config-repo-cangkok)
15. [Chat Template: Original Tidak Punya](#15-chat-template-original-tidak-punya)
16. [EOS Ganda & image_token_index](#16-eos-ganda--image_token_index)
17. [Strategi Vision Training Final](#17-strategi-vision-training-final)
18. [Daftar File yang Dibuat](#18-daftar-file-yang-dibuat)

---

## 1. Arsitektur T5Gemma2 vs Gemma 3

### Perbandingan Fundamental

| Aspek | Gemma 3 (4B) | T5Gemma2 (4B-4B) |
|---|---|---|
| **Arsitektur** | Multimodal Decoder-only | **Multimodal Encoder-Decoder** |
| **Model type** | `gemma3` | `t5gemma2` |
| **Total parameter** | ~4.28B (3.88B text + 0.4B vision) | **~7.51B** (3.88B enc + 3.88B dec + 0.4B vis) |
| **Encoder layers** | — | 34 (text) + 27 (vision SigLIP) |
| **Decoder layers** | 34 | 34 |
| **Hidden size** | 2560 | 2560 |
| **Attention** | Standard Self-Attention | **Merged Attention** (Self + Cross) |
| **Vision Tower** | SigLIP (Hidden 1152) | SigLIP (Hidden 1152) — IDENTIK |
| **Tied embeddings** | ✅ Yes (Embed ↔ Head) | ✅ Yes (Enc ↔ Dec ↔ Head) |
| **Vocab Size** | 262,208 (extra 64 padding) | 262,144 (exact) |
| **is_encoder_decoder** | False | True |
| **Sliding window** | 1024 (pattern 5+1) | 1024 (pattern 5+1) — SAMA |
| **RoPE scaling** | factor=8.0, linear | factor=8.0, linear — SAMA |
| **Max position** | 131072 (128k) | 131072 (128k) |

### Merged Attention: Mekanisme Inti Decoder T5Gemma2

T5Gemma2 tidak memiliki modul `cross_attention` terpisah. Sebagai gantinya, ia menggunakan **Merged Attention**:
- **Query (Q):** Dibentuk dari decoder hidden states (**X**).
- **Key (K) & Value (V):** Dibentuk dengan mengkonkatenasi decoder input (**X**) dan encoder outputs (**H**): `[X; H]`.
- **Masking:** Bidirectional untuk encoder tokens (H), causal untuk decoder tokens (X).

$ K, V = [X; H] $

**Implikasi:** Modul `self_attn` decoder memproses self + cross dalam satu langkah. Bobot Q/K/V kompatibel dengan decoder-only, tapi perilaku attention berbeda karena K/V juga "melihat" encoder output.

### Catatan Variasi Ukuran

| Model | Text Hidden | Vision Hidden (SigLIP) | Multimodal |
|---|---|---|---|
| Gemma 3 270M / 1B | 640 / 1152 | — | **Text Only** (tidak ada vision) |
| Gemma 3 4B | 2560 | 1152 | Multimodal |
| T5Gemma2 270M | 640 | 1152 | Multimodal |
| T5Gemma2 1B | 1152 | 1152 | Multimodal |
| T5Gemma2 4B | 2560 | 1152 | Multimodal |

> **Penting:** Gemma 3 270M/1B **tidak punya vision tower**. T5Gemma2 semua ukuran punya vision (meminjam SigLIP 400M). Ini membuat testing lokal 270m untuk cangkok Gemma 3 tidak bisa (Gemma 3 270m tidak ada SigLIP).

---

## 2. Mekanisme Image Encoder (SigLIP)

### Spesifikasi SigLIP

| Komponen | Spesifikasi |
|---|---|
| Vision Tower | SigLIP 400M, 27 layers, hidden 1152, 16 heads |
| Patch size | 14×14 (Conv2d 3→1152, 14×14) |
| Image input size | **896×896 (FIXED)** |
| Token per gambar | **256 (FIXED)** |
| Lokasi vision | Di **Encoder** (`model.encoder.vision_tower`) |
| Projector | `T5Gemma2MultiModalProjector` (1152→2560) |

### Cara Kerja Image Pipeline

```
Gambar (896×896)
   ↓ SigLIP: Conv2d(3→1152, 14×14) → 64×64 = 4096 patch
   ↓ (spatial merge 4×4: 16 patch → 1 token)
256 feature vectors (1152-dim)        ← VECTOR KONTINU, bukan token diskrit
   ↓ multi_modal_projector (1152 → 2560)
256 vectors (2560-dim)                ← ukuran sama dgn text embedding
   ↓ INSERT ke posisi placeholder <image_soft_token> (ID 256001)
Sequence encoder: [text_emb, vis_emb×256, text_emb, ...]
   ↓ Transformer encoder proses bidirectional
Encoder output H
   ↓ Decoder akses H via Merged Attention [X; H]
Decoder generate response (causal)
```

### Token Image

| Token | ID | Fungsi |
|---|---|---|
| `📷` / `<start_of_image>` (boi) | 255999 | Penanda awal gambar |
| `<end_of_image>` (eoi) | 256000 | Penanda akhir gambar |
| `<image_soft_token>` | 256001 | Placeholder (di-replace vision features) |

### Processor: Gemma3Processor

HF **tidak mendefinisikan processor khusus** untuk T5Gemma2. `T5Gemma2Config` diarahkan ke `Gemma3Processor`. Perilaku pemrosesan gambar & token placeholder sepenuhnya mengikuti Gemma 3.

---

## 3. Soft Token vs Hard Token

### Perbedaan Fundamental

| Aspek | Text Token (Hard) | Image "Token" (Soft) |
|---|---|---|
| **Asal vector** | ID diskrit → lookup embedding matrix | Pixel → SigLIP → projector → vector kontinu |
| **Sifat** | Diskrit — tiap ID punya 1 baris embedding tetap | Kontinu — vector berubah tergantung isi gambar |
| **Sumber parameter** | `embed_tokens` (262144×2560) | `vision_tower` + `multi_modal_projector` |
| **Bentuk akhir** | 2560-dim | 2560-dim (sama!) |
| **Diproses transformer** | ✅ Ya, biasa | ✅ Ya, biasa (setelah di-inject) |

### Kenapa Disebut "Soft Token"

"Soft" = bukan hard/diskrit. Token teks adalah **hard selection** (1 ID → 1 baris tetap). Image "token" adalah **soft injection** — vector kontinu hasil komputasi berbeda untuk setiap gambar. Dua gambar berbeda menghasilkan 256 vector berbeda, padahal keduanya "mengisi" placeholder yang sama (ID 256001).

### Mekanisme Injection

`<image_soft_token>` (ID 256001) adalah **PLACEHOLDER** — punya baris di embedding matrix, tapi baris itu **TIDAK dipakai**. Saat forward pass:
1. Processor: teks + `📷` → token IDs (termasuk 256× placeholder 256001)
2. SigLIP proses gambar → projector → 256 vector
3. Model **inject** 256 vector ke posisi placeholder, **menimpa** embedding-nya
4. Transformer encoder proses sequence yang sudah "diisi"

### Implikasi di T5Gemma2

- Image token masuk ke **ENCODER input** → encoder proses bidirectional → output `H`
- Decoder akses `H` via Merged Attention `[X; H]` — **tidak pernah melihat pixel/vision token langsung**
- Decoder output = teks response, bukan image token → suppress image token di decoder = benar

---

## 4. Perbandingan Config: T5Gemma2 vs Gemma 3 IT

### image_token_index — BEDA

| | Gemma 3 IT | T5Gemma2 |
|---|---|---|
| `image_token_index` | 262144 (slot padding ekstra) | 256001 (`<image_soft_token>`) |
| Vocab size | 262208 (64 padding) | 262144 (exact) |

**TIDAK perlu ikut Gemma 3.** 262144 di T5Gemma2 = out of bounds (vocab 262144, index 0-262143). SigLIP bobot tidak peduli image_token_index — urusan text encoder.

### EOS — BEDA

| | Gemma 3 IT | T5Gemma2 |
|---|---|---|
| `eos_token_id` | [1, 106] (ganda) | 1 (tunggal) |
| 106 = `<end_of_turn>` | Dipakai sebagai stop generation | Tidak dipakai sebagai EOS |

**TIDAK perlu ikut Gemma 3.** T5Gemma2 dilatih UL2 dengan `<eos>` (1) sebagai stop. Kalau paksa tambah 106 sebagai EOS → model bingung. Chat template tetap pakai `<end_of_turn>` sebagai formatting token (bukan EOS).

### Sliding Window — SAMA

Keduanya pakai sliding_window=1024, pattern 5 sliding + 1 full attention, RoPE scaling factor=8.0.

### Chat Template — BEDA Sumber

| | Gemma 3 IT | T5Gemma2 |
|---|---|---|
| File `chat_template.json` | ✅ Ada di repo | ❌ Tidak ada |
| Embedded di `tokenizer_config.json` | ❌ | ❌ |
| Sumber template | File repo | Unsloth `get_chat_template("gemma-3")` runtime |

---

## 5. Logit Masking & Suppression

### Mekanisme

```python
SUPPRESS_BLOCK1 = [6] + list(range(13, 105))   # unused0, unused7-98 (kecuali 7-12)
SUPPRESS_BLOCK2 = list(range(256002, 262144))   # unused100-6241
SUPPRESS_VISION = [255999, 256000, 256001]       # boi, eoi, image_soft_token
ALL_SUPPRESS_IDS = set(BLOCK1 + BLOCK2 + VISION) # 6238 tokens total
```

Hook dipasang di `lm_head` (decoder output): `logits[suppress] += -10000.0`

### Kenapa Aman untuk Vision

| Komponen | Lewat `lm_head`? | Kena mask? |
|---|---|---|
| **Encoder** (text + image) | ❌ Tidak (output = hidden states) | ❌ Tidak |
| **Decoder** (generate text) | ✅ Ya (output = logits) | ✅ Ya |

- Hook hanya trigger di decoder → encoder tetap proses image token via `embed_tokens`
- 3 vision token (boi/eoi/image_soft_token) hanya di **encoder input** → suppress di decoder = benar
- Decoder output = teks response → tidak boleh generate image token
- chosen & rejected logits sama-sama kena mask → fair comparison di ORPO

### Konsistensi dengan Text-Only v6

Logit masking identik (6238 tokens) dengan text-only v6. Model vision training **melanjutkan** perilaku suppress yang sudah diajarkan text training. **Tidak perlu retrain text-only.**

### Task Prefix (TIDAK di-suppress)

`<unused1>`–`<unused6>` (ID 7-12) = Task Prefix, **dikecualikan** dari suppression:
```python
SUPPRESS_BLOCK1 = [6] + list(range(13, 105))  # ID 7-12 di-skip
```
Task prefix mapping (dari tokenizer_config.json v6):
```json
"task_prefix_mapping": {
    "<unused1>": "summarize",
    "<unused2>": "translate",
    "<unused3>": "ner",
    "<unused4>": "qa",
    "<unused5>": "paraphrase",
    "<unused6>": "general_chat"
}
```

### SelectiveLabelSmoother

Label smoothing yang **aware suppress** — smoothing hanya distribusikan ke token valid (262144 - 6238 = ~155900 token). Tanpa ini, smoothing "bocor" probability ke token suppress.

---

## 6. Isu Seq2Seq di Unsloth & TRL

### Root Cause

| Komponen | Dirancang untuk | Masalah untuk T5Gemma2 |
|---|---|---|
| `SFTTrainer` (TRL) | Causal LM (decoder-only) | Asumsi label masking autoregressive, bukan seq2seq |
| `DPOTrainer` (TRL) | Causal LM | Sama |
| `FastVisionModel` + `UnslothVisionDataCollator` | Gemma 3 (decoder-only) | Menggabungkan image+text jadi single sequence |
| `Unsloth` main branch | Belum support seq2seq | PR #4226 belum di-merge |

### Text-Only v6 (Berhasil) vs Vision v6 Awal (Bermasalah)

**Text-only v6** (berhasil):
```
FastLanguageModel + Seq2SeqTrainer (HF native) + DataCollatorForSeq2Seq
+ patch Unsloth bypass batch sampler saat is_encoder_decoder=True
```

**Vision v6 awal** (bermasalah):
```
FastVisionModel + SFTTrainer (TRL) + UnslothVisionDataCollator
→ dirancang untuk Gemma 3 decoder-only, bukan T5Gemma2 seq2seq
```

Bukti kompromi di kode awal (`test_v6_vision_unsloth.py:309-311`):
```python
if not hasattr(model.config, "text_config"):
    type(model.config).text_config = property(lambda self: self.decoder)  # SHIM
```
Ini menipu `FastVisionModel` agar "melihat" `text_config` (atribut Gemma 3) padahal T5Gemma2 punya `encoder`/`decoder` terpisah.

### Solusi: Vanilla Seq2Seq Approach

Ganti seluruh stack TRL/UnslothVision dengan HF native seq2seq:
- `SFTTrainer (TRL)` → `Seq2SeqTrainer (HF)`
- `UnslothVisionDataCollator` → `Seq2SeqVisionCollator` (custom)
- `DPOTrainer (TRL)` → `VisionORPOTrainer` (custom, extends Seq2SeqTrainer)

### CustomORPOTrainer (Text-Only v6 Sudah Seq2Seq)

Temuan penting: ORPO text-only v6 **sudah custom seq2seq** (`CustomORPOTrainer` extends `Seq2SeqTrainer`), bukan pakai `DPOTrainer` TRL. Jadi vision ORPO = extension minimal, bukan rewrite.

---

## 7. Validasi Vanilla Seq2Seq + Vision

### Test: `scripts/tests/test_vanilla_seq2seq_vision.py`

Test murni HF Transformers (tanpa Unsloth) di RTX 3060 6GB, model `google/t5gemma-2-270m-270m`.

### Hasil Validasi

| Step | Test | Hasil |
|---|---|---|
| 1 | Load model + processor | `T5Gemma2ForConditionalGeneration`, 786M, `Gemma3Processor` |
| 2 | Inspeksi arsitektur | `T5Gemma2Encoder` + `SiglipVisionModel` + `T5Gemma2MultiModalProjector` |
| 3 | Forward 1 gambar | loss 4.3169, input_ids [1, 269] (256 image tokens), pixel_values [1, 3, 896, 896] |
| 4 | Forward 10 gambar | loss 2.4590, input_ids [1, 2623] (2560 image tokens), pixel_values [10, 3, 896, 896] |
| 5 | Collator batch=2 | pixel_values [4, 3, 896, 896] (flat-concat), loss 4.0075 |
| 5 | Training step 1 | `loss=4.283, grad_norm=588` — forward→loss→backward→optimizer ✅ |

### Bug Fix Penting: Collator pixel_values

Bug awal: `enc["pixel_values"][0]` (ambil image pertama saja) → error:
```
ValueError: Image features and image tokens do not match: tokens: 1024, features 512
```
Fix: `enc["pixel_values"]` (ambil semua gambar per sample) → pixel_values [4, ...] benar.

**Model handle batched multi-image secara internal** — dia hitung image token per sample lalu cocokkan dengan flat pixel_values berurutan.

### Seq2SeqVisionCollator (Custom)

```python
class Seq2SeqVisionCollator:
    # Encoder: input_ids (text + image soft tokens) + pixel_values + attention_mask
    # Decoder: labels (target + EOS, pad -100)
    # pixel_values: torch.cat (flat-concat semua gambar di batch)
```

---

## 8. Validasi Vanilla ORPO Vision

### Test: `scripts/tests/test_vanilla_orpo_vision.py`

### Hasil Validasi

| Step | Test | Hasil |
|---|---|---|
| 1 | Load + logit mask | Logit mask 6244 tokens di lm_head ✅ |
| 3 | Forward ORPO (chosen & rejected + pixel) | chosen loss 4.83, rejected 6.84 (rejected > chosen ✅) |
| 3 | Log-odds ratio | chosen_logps -38.75, rejected -13.69, margin -25.0, OR loss 25.0, total 7.33 |
| 5 | ORPO training step 1 | `loss=7.166, grad_norm=764` — 2x forward + log-odds + 2x backward ✅ |

### VisionORPOTrainer (Custom)

```python
class VisionORPOTrainer(Seq2SeqTrainer):
    # Loss = SFT_loss(chosen) + beta * OR_loss(log_odds_margin)
    # chosen & rejected share encoder input (text + image), beda decoder labels
    # average_log_prob=True untuk mitigasi URSLA shortcut
```

### URSLA Shortcut Risk

`log_odds_margin: -25.0` (negatif) menunjukkan rejected ("gambar" pendek) punya odds lebih tinggi — karena rejected sangat pendek (2 token) sehingga log-prob per token tinggi. Ini **URSLA shortcut** — model belajar "truncate respons buruk lebih awal". `average_log_prob=True` di `get_batch_logps` untuk normalisasi panjang.

---

## 9. Verifikasi Nama Module (Empiris)

### Script: `scripts/tests/verify_modules.py`

Verifikasi empiris via `model.named_modules()` di `google/t5gemma-2-270m-270m`.

### Encoder Text (34 layers) — `model.encoder.text_model.layers.{X}`
```
self_attn.q_proj   (1024, 640)   ← 270m; 4B: (2048, 2560)
self_attn.k_proj   (256, 640)    ← GQA, 4 KV heads
self_attn.v_proj   (256, 640)
self_attn.o_proj   (640, 1024)
mlp.gate_proj       (2048, 640)
mlp.up_proj         (2048, 640)
mlp.down_proj       (640, 2048)
```
**Total: 34 × 7 = 238 module linear**

### Vision Tower — SigLIP (27 layers) — `model.encoder.vision_tower.encoder.layers.{X}`
```
self_attn.q_proj    (1152, 1152)   ← ⚠️ NAMA SAMA dgn text!
self_attn.k_proj    (1152, 1152)   ← ⚠️ NAMA SAMA
self_attn.v_proj    (1152, 1152)   ← ⚠️ NAMA SAMA
self_attn.out_proj  (1152, 1152)   ← ⚠️ BEDA! "out_proj" bukan "o_proj"
mlp.fc1             (4304, 1152)   ← ⚠️ BEDA! "fc1" bukan "gate_proj"
mlp.fc2             (1152, 4304)   ← ⚠️ BEDA! "fc2" bukan "up_proj"/"down_proj"
```
Plus: `vision_tower.embeddings` (Conv2d 3→1152, 14×14 + Embedding 4096, 1152)
**Total: 27 × 6 = 162 module linear**

### Multi-Modal Projector — `model.encoder.multi_modal_projector`
```
T5Gemma2MultiModalProjector (0.738M params)
├── mm_soft_emb_norm    T5Gemma2RMSNorm  (1152,)
└── avg_pool            AvgPool2d
```
**Nama path persis: `model.encoder.multi_modal_projector`** (bukan `mm_projector` atau `vision_projector`).

### Decoder (34 layers) — `model.decoder.layers.{X}`
```
self_attn.q_proj   (1024, 640)   ← Merged Attention (self+cross digabung)
self_attn.k_proj   (256, 640)
self_attn.v_proj   (256, 640)
self_attn.o_proj   (640, 1024)
mlp.gate_proj       (2048, 640)
mlp.up_proj         (2048, 640)
mlp.down_proj       (640, 2048)
```
**Total: 34 × 7 = 238 module linear**

### lm_head
```
lm_head                T5Gemma2LMHead (container)
lm_head.out_proj       Linear (262144, 640)  ← hook logit mask di sini
```

---

## 10. "Bocor" LoRA ke Vision Tower

### Mekanisme Bocor

Config text-only v6:
```python
LORA_TARGET_MODULES = ["q_proj", "k_proj", "v_proj", "o_proj",
                       "gate_proj", "up_proj", "down_proj"]
```

PEFT match by suffix → q/k/v_proj match SigLIP vision_tower layers juga:

| Target Module | Encoder Text | Decoder | Vision Tower | Projector |
|---|---|---|---|---|
| `q_proj` | ✅ | ✅ | ✅ **BOCOR!** | ❌ |
| `k_proj` | ✅ | ✅ | ✅ **BOCOR!** | ❌ |
| `v_proj` | ✅ | ✅ | ✅ **BOCOR!** | ❌ |
| `o_proj` | ✅ | ✅ | ❌ (SigLIP pakai `out_proj`) | ❌ |
| `gate_proj` | ✅ | ✅ | ❌ (SigLIP pakai `fc1`) | ❌ |
| `up_proj` | ✅ | ✅ | ❌ (SigLIP pakai `fc2`) | ❌ |
| `down_proj` | ✅ | ✅ | ❌ | ❌ |

### Bocor ≠ Rusak

"Bocor" = LoRA adapter kosong terpasang di vision_tower, bukan bobot vision_tower rusak.

Saat text-only training:
- Tidak ada `pixel_values` → vision_tower TIDAK di-forward
- Tidak ada forward → gradient ke vision_tower LoRA = 0
- LoRA init standar: B=0 → delta = A@B = 0
- Gradient = 0 → B tetap 0 → delta tetap 0

Setelah merge: `merged_weight = base + delta(0) = base`

**Bocor = no-op.** Diverifikasi empiris di §11.

---

## 11. Verifikasi 3-Arah Bobot Vision

### Script: `scripts/tests/verify_vision_weights_3way.py` (marimo, dijalankan di Molab)

Compare 3 sumber:
- **[A]** `daruokta/t5gemma-2-4b-4b-instruct-chat-indo-v4-unsloth/merged_bf16` (v6 text hasil)
- **[B]** `google/t5gemma-2-4b-4b` (original T5Gemma2)
- **[C]** `google/gemma-3-4b-it` (Gemma 3 IT, kandidat cangkok)

### Hasil

| Comparison | max_diff | Verdict |
|---|---|---|
| **A vs B** (v6 merged vs original) | **0.00e+00** | ✅ IDENTIK — bocor tidak ada efek |
| **B vs C** (original vs Gemma 3 IT) | **1.74e-01** | BERBEDA signifikan — cangkok worth it |

### Analisis

1. **v6 merged vs original T5Gemma2 = IDENTIK (0.00e+00)**
   - Training text v6 TIDAK mengubah vision_tower
   - "Bocor" LoRA delta = 0 persis (konfirmasi §10)
   - Tidak perlu menambat/reset vision_tower

2. **original T5Gemma2 vs Gemma 3 IT = BERBEDA (1.74e-01)**
   - SigLIP Gemma 3 IT sudah instruct-tuned (lebih paham instruksi visual)
   - SigLIP original T5Gemma2 = pre-training UL2 (belum instruct-tuned vision)
   - Cangkok Gemma 3 IT WORTH IT

---

## 12. Cangkok SigLIP + Projector

### Script: `scripts/tests/verify_vision_weights_3way.py` (cell cangkok, dijalankan di Molab)

### Proses

1. Load v6 merged_bf16 (target, `T5Gemma2ForConditionalGeneration`)
2. Load Gemma 3 4B IT (source, `Gemma3ForConditionalGeneration`)
3. Extract 439 vision params dari source (path mapping: `model.vision_tower` → `vision_tower`)
4. Copy source → target (path mapping: `model.encoder.vision_tower` → `vision_tower`)
5. Verify: re-compare target vs source (harus ≈ 0 setelah cangkok)
6. Save lokal + upload ke HF repo publik

### Hasil

```
Source (Gemma 3 IT): 439 vision params
✅ Cangkok: 439 params, skip: 0
✅ Verify: 439 OK, 0 fail
✅ BERHASIL! Model cangkok tersimpan di: daruokta/t5gemma-2-4b-4b-instruct-chat-indo-v4-vision-cangkok
```

### Repo Cangkok

`daruokta/t5gemma-2-4b-4b-instruct-chat-indo-v4-vision-cangkok` (PUBLIC)

| Komponen | Sumber |
|---|---|
| Encoder text (34 layers) | v6 merged_bf16 (hasil SFT+ORPO text Indonesia) |
| Decoder (34 layers, merged attn) | v6 merged_bf16 |
| Embeddings + lm_head (tied) | v6 merged_bf16 |
| **Vision tower (SigLIP 27 layers)** | **Gemma 3 4B IT** (instruct-ready) |
| **Multi-modal projector** | **Gemma 3 4B IT** |
| Processor (preprocessor + tokenizer) | original T5Gemma2 + v6 tokenizer_config |

### Path Mapping Cangkok

```
Source (Gemma 3):           Target (T5Gemma2):
model.vision_tower.*    →   model.encoder.vision_tower.*
model.multi_modal_projector.*  →  model.encoder.multi_modal_projector.*
```

---

## 13. Decoder Transplant vs SigLIP Transplant

### Kenapa Decoder Transplant Hancur

Pernah dilakukan sebelumnya: cangkok decoder Gemma 3 IT → T5Gemma2 decoder → **hancur**.

Penyebab: **Mismatch arsitektur** — Merged Attention.

```
Gemma 3 IT decoder:  Standard Self-Attention
  Q, K, V semua dari decoder hidden states
  K/V hanya "melihat" token decoder sendiri

T5Gemma2 decoder:    MERGED Attention (self + cross digabung)
  Q dari decoder, K/V = [decoder_input ; encoder_output] di-concat
  K/V "melihat" decoder DAN encoder
```

Saat cangkok bobot Gemma 3 IT decoder → T5Gemma2 decoder:
- Bobot K/V Gemma 3 IT dilatih untuk attend hanya ke token decoder
- Tapi di T5Gemma2, K/V dipaksa juga attend ke encoder output (belum pernah dilihat)
- Plus encoder output = representasi UL2 bidirectional, beda distribusi
- **Hasil: mismatch arsitektur + distribusi → hancur**

### Kenapa SigLIP Transplant Aman

SigLIP adalah modul **standalone** — input = pixel, output = vision features (1152-dim). Tidak ada interaksi langsung dengan text encoder/decoder. Tidak ada merged attention, tidak ada cross-attention internal. Input/output identik di kedua model.

**Analogi:** Decoder transplant = mencangkok otak yang dilatih untuk bahasa A ke tubuh yang pakai bahasa B (mismatch). SigLIP transplant = mencangkok mata yang lebih tajam ke tubuh yang sama (mata kerjanya cuma "melihat", tidak peduli bahasa apa yang dipakai otak).

### Projector — Perlu Hati-hati tapi Aman

Projector dimensi sama (1152→2560), tapi dilatih untuk "berbicara" dengan decoder Gemma 3 IT. T5Gemma2 decoder (merged attention, UL2) "mendengar" berbeda.

**Tapi aman karena:** Projector akan di-FULL fine-tune di vision SFT. Cangkok = starting point "paham vision instruksi", vision SFT re-align = adaptasi output ke T5Gemma2 decoder.

---

## 14. Fix Tokenizer Config Repo Cangkok

### Masalah 1: task_prefix_mapping Hilang

Cangkok load processor dari `google/t5gemma-2-4b-4b` (original) yang tokenizer_config minim (830 bytes, tidak punya `task_prefix_mapping`).

**Fix:** Patch `scripts/tests/patch_cangkok_tokenizer.py` — tambah `task_prefix_mapping` manual.

### Masalah 2: added_tokens_decoder Hilang

Setelah patch task_prefix, ternyata tokenizer_config masih minim (1037 bytes) vs v6 merged (1,206,875 bytes). v6 merged punya `added_tokens_decoder` — dict lengkap mapping ID→token untuk ratusan ribu token.

**Fix:** Replace tokenizer_config repo cangkok dengan tokenizer_config dari v6 merged (sudah lengkap: `added_tokens_decoder` + `task_prefix_mapping`).

### Hasil Verifikasi

| Aspek | Repo Cangkok (after fix) | v6 merged_bf16 |
|---|---|---|
| Ukuran | 1,206,875 bytes | 1,206,875 bytes |
| `added_tokens_decoder` | ✅ Ada | ✅ Ada |
| `task_prefix_mapping` | ✅ Ada | ✅ Ada |
| Semua field lain | ✅ Identik | ✅ Identik |

**100% identik.** Repo cangkok sekarang self-contained.

### Script: `scripts/tests/patch_cangkok_tokenizer.py`

```python
# Download tokenizer_config.json dari v6 merged (lengkap)
# Upload ke repo cangkok (replace)
```

---

## 15. Chat Template: Original Tidak Punya

### Verifikasi

Dari fetch `google/t5gemma-2-4b-4b` repo:
- ❌ TIDAK ada file `chat_template.jinja`
- ❌ TIDAK ada file `chat_template.json`
- ❌ TIDAK ada field `chat_template` embedded di `tokenizer_config.json` (hanya 27 lines)

File di repo original T5Gemma2:
```
.gitattributes, README.md, config.json, generation_config.json,
model-*.safetensors, model.safetensors.index.json,
processor_config.json, special_tokens_map.json,
tokenizer.json, tokenizer_config.json
```

### Sumber Chat Template

Chat template gemma-3 di-apply **runtime** via Unsloth:
```python
from unsloth.chat_templates import get_chat_template
tokenizer = get_chat_template(tokenizer, chat_template="gemma-3")
processor.chat_template = tokenizer.chat_template
```

Unsloth punya library template bawaan, `"gemma-3"` adalah preset. Saat dipanggil, Unsloth inject Jinja template ke `tokenizer.chat_template` (in-memory attribute, bukan file).

### Template Content (gemma-3)

```jinja
{{ bos_token }}
{%- for message in loop_messages -%}
    {{ '<start_of_turn>' + role + '\n' }}
    {%- for item in message['content'] -%}
        {%- if item['type'] == 'image' -%}
            {{ '<start_of_image>' }}
        {%- elif item['type'] == 'text' -%}
            {{ item['text'] | trim }}
        {%- endif -%}
    {%- endfor -%}
    {{ '<end_of_turn>\n' }}
{%- endfor -%}
{%- if add_generation_prompt -%}
    {{'<start_of_turn>model\n'}}
{%- endif -%}
```

### Gemma 3 4B IT: Punya chat_template.json

Dari download `data/tokenizers/gemma3-4b-it/`: ada `chat_template.json` (1.6KB) dengan template embedded. Tapi ini tidak dipakai di T5Gemma2 — T5Gemma2 pakai Unsloth runtime injection.

### Implikasi

Repo cangkok tidak punya chat_template file, dan itu **tidak masalah** — training script tetap inject via Unsloth `get_chat_template("gemma-3")`.

---

## 16. EOS Ganda & image_token_index

### EOS: TIDAK Perlu Ikut Gemma 3

| | Gemma 3 IT | T5Gemma2 (repo cangkok) |
|---|---|---|
| `eos_token_id` | [1, 106] (ganda) | 1 (tunggal) |
| 106 = `<end_of_turn>` | Dipakai sebagai stop generation | Tidak dipakai sebagai EOS |

**TIDAK perlu ikut Gemma 3.** T5Gemma2 dilatih UL2 dengan `<eos>` (1) sebagai stop. Kalau paksa tambah 106 → model bingung. Chat template tetap pakai `<end_of_turn>` sebagai **formatting token** (bukan EOS).

Repo cangkok config: `"eos_token_id": 1` ✅

### image_token_index: TIDAK Perlu Ikut Gemma 3

| | Gemma 3 IT | T5Gemma2 |
|---|---|---|
| `image_token_index` | 262144 (slot padding ekstra) | 256001 (`<image_soft_token>`) |
| Vocab size | 262208 (64 padding) | 262144 (exact) |

**TIDAK perlu ikut.** 262144 di T5Gemma2 = out of bounds. SigLIP bobot tidak peduli image_token_index.

Repo cangkok config: `"image_token_index": 256001` ✅

### Sliding Window: SAMA

T5Gemma2 juga pakai sliding_window=1024, pattern 5 sliding + 1 full attention. Sama seperti Gemma 3.

---

## 17. Strategi Vision Training Final

### Config (`working-molab-v6-vision-unsloth.py`)

```python
MODEL_NAME = "daruokta/t5gemma-2-4b-4b-instruct-chat-indo-v4-vision-cangkok"
SUBFOLDER = ""                    # repo cangkok langsung root
LOAD_IN_4BIT = True              # QLoRA
LORA_RANK = 128; LORA_ALPHA = 256; LORA_DROPOUT = 0.1
NUM_EPOCHS_SFT = 2; NUM_EPOCHS_ORPO = 1
ORPO_BETA = 0.1
MAX_SOURCE_LENGTH = 16384; MAX_TARGET_LENGTH = 2048
MAX_IMAGES_PER_CHAT = 10
```

### LoRA Strategy

```python
finetune_vision_layers=False,      # SKIP vision tower (sudah cangkok Gemma 3 IT)
modules_to_save=["multi_modal_projector"],  # FULL FT projector (re-align ke decoder)
apply_logit_mask(model, ALL_SUPPRESS_IDS)   # 6238 tokens
```

### Alur Training

```
daruokta/...v4-vision-cangkok (v6 text + SigLIP/projector Gemma 3 IT)
   ↓ QLoRA 4-bit + LoRA r=128 (skip vision_tower, full FT projector)
   ↓ + Logit masking 6238 tokens
[VISION SFT 2 epochs] → [VISION ORPO 1 epoch] → SAVE → UPLOAD HF
```

### Komponen Training

| Phase | Trainer | Collator |
|---|---|---|
| SFT | `Seq2SeqTrainer` (HF native) | `Seq2SeqVisionCollator` (custom) |
| ORPO | `VisionORPOTrainer` (custom, extends Seq2SeqTrainer) | `VisionORPOCollator` (custom) |

---

## 18. Daftar File yang Dibuat

### Test & Validation Scripts

| File | Fungsi | Status |
|---|---|---|
| `scripts/tests/test_vanilla_seq2seq_vision.py` | Test SFT vanilla seq2seq+vision (270m) | ✅ Validated |
| `scripts/tests/test_vanilla_orpo_vision.py` | Test ORPO vanilla seq2seq+vision (270m) | ✅ Validated |
| `scripts/tests/verify_modules.py` | Dump nama module T5Gemma2 | ✅ Validated |
| `scripts/tests/verify_vision_weights_3way.py` | Verifikasi 3-arah + cangkok (marimo) | ✅ Validated |
| `scripts/tests/patch_cangkok_tokenizer.py` | Fix tokenizer_config repo cangkok | ✅ Validated |
| `test_v6_vision_unsloth.py` | Test vision training Unsloth (270m) | ✅ Validated |

### Production Scripts

| File | Fungsi | Status |
|---|---|---|
| `working-molab-v6-vision-unsloth.py` | Production vision training (Molab 96GB) | ✅ Ready |

### HF Repos

| Repo | Status |
|---|---|
| `daruokta/...v4-unsloth` | Text-only v6 (existing) |
| `daruokta/...v4-vision-cangkok` | **Cangkok (PUBLIC)** — v6 text + SigLIP/projector Gemma 3 IT |
| `daruokta/...v4-vision` | Vision training target (future) |

---

## Ringkasan Eksekutif

1. **Mekanisme image T5Gemma2**: SigLIP → 256 soft token/gambar → inject ke ENCODER → decoder akses via Merged Attention. Fixed 256 token, 896×896.

2. **Logit masking aman untuk vision**: Hook di lm_head (decoder only), 6238 tokens. Encoder tidak kena. Konsisten dengan text-only v6.

3. **"Bocor" LoRA = no-op**: Text-only training tidak forward SigLIP → delta=0 → merged=original. Diverifikasi: max_diff=0.00e+00.

4. **Cangkok Gemma 3 IT worth it**: SigLIP IT berbeda 0.174 dari original. Cangkok 439/439 verified. Aman karena SigLIP standalone.

5. **EOS & image_token_index**: TETAP T5Gemma2 style (eos=1, image_token=256001). TIDAK ikut Gemma 3.

6. **Chat template**: Original T5Gemma2 tidak punya. Applied runtime via Unsloth.

7. **Vision training**: QLoRA r=128, skip vision_tower, full FT projector, logit masking, Seq2SeqTrainer + custom collator, SFT 2ep + ORPO 1ep.

8. **Tokenizer config**: Repo cangkok 100% identik v6 merged (added_tokens_decoder + task_prefix_mapping).