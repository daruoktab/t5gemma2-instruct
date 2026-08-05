# 🏗️ Pipeline Training T5Gemma-2 v7 — Ringkasan Teknis

**Tanggal:** 2026-08-05 · **Sumber:** `notebooks/working-molab-v7-combined-unsloth.py`
(4636 baris, marimo/unsloth) — dibaca langsung dari disk, bukan dari ingatan.
**Model:** `google/t5gemma-2-4b-4b` (encoder-decoder, UL2-adapted dari Gemma 3,
merged attention, tied word embeddings)

---

## 0. Arsitektur & Fase Pipeline

```
Phase 0.5  Task-vector steering   (Gemma3-IT − Gemma3-Base → decoder FFN, layer-wise α)
Phase 1.5  Vision grafting        (SigLIP + projector dari Gemma 3 4B IT)
Phase 1    Joint SFT              (teks chat/indoqa + vision — 1 loop)
Phase 2    Joint ORPO             (teks chat_orpo + vision vision_orpo)
Final      Merge BF16 + 4bit      (deploy chatbot multimodal Indonesia)
```

Subfolder model: `steered/` (hasil Phase 0.5), `cangkok/` (hasil grafting).
Flag `STEERING_FORCE` / `CANGKOK_FORCE` untuk memaksa ulang walau sudah ada di repo.

---

## 1. Phase 0.5 — Task-Vector Steering

| Parameter | Nilai | Catatan |
|-----------|-------|---------|
| `ENABLE_STEERING` | `True` | Steering decoder, task vector = Gemma3-IT − Gemma3-Base |
| `STEERING_ALPHA_FFN_EARLY` | `0.05` | Layer awal (< 25% depth) — subtle |
| `STEERING_ALPHA_FFN_MID` | `0.25` | Layer tengah (25–80%) — **peak IT reasoning & knowledge** |
| `STEERING_ALPHA_FFN_LATE` | `0.08` | Layer akhir (> 80%) — jaga kalibrasi output |
| `STEERING_ALPHA_NORM_EARLY / MID / LATE` | `0.02 / 0.08 / 0.03` | Layer-norm ikut di-steer |
| `STEERING_ALPHA_QO / KV / QKNORM` | `0.0` | ⚠️ **WAJIB 0.0** — keamanan Merged Attention [X;H] |
| `STEERING_SMOKE_TEST` | `True` | Generate 3 prompt singkat untuk sanity check |

> Steering hanya menyentuh FFN + norm (bukan q/k/v/o) karena merged attention
> memproyeksikan [X;H] bersama — mengubah proyeksi joint berisiko merusak arsitektur.

---

## 2. Phase 1.5 — Vision Grafting

- **Vision tower:** SigLIP (dari Gemma 3 4B IT), dicangkok ke T5Gemma-2.
- **Projector:** 2-layer MLP (+ dropout 0.1) — diambil dari Gemma 3 4B IT.
- **Branch optimizer projector:** `PROJECTOR_BRANCH = "muon"` — bisa `"adema"` untuk
  konservatif (bobot pretrained graft).
- Token gambar interleaved `<image>` dalam sequence.

---

## 3. Phase 1 — Joint SFT (Teks + Vision)

| Parameter | Nilai |
|-----------|-------|
| `SFT_LEARNING_RATE` | `5e-6` |
| `SFT_NUM_EPOCHS` | `2` |
| `SFT_PER_DEVICE_TRAIN_BATCH_SIZE` | `4` |
| `SFT_PER_DEVICE_EVAL_BATCH_SIZE` | `16` (eval = no_grad → aman naik) |
| `SFT_GRADIENT_ACCUMULATION_STEPS` | `16` (effective batch = 64) |
| `SFT_WARMUP_STEPS` | `100` |
| `SFT_LR_SCHEDULER_TYPE` | `cosine` |
| `SFT_MAX_GRAD_NORM` | `5.0` |
| `SFT_NEFTUNE_NOISE_ALPHA` | `5.0` |

**Split-LR per komponen** (multiplier relatif ke `SFT_LEARNING_RATE`):

| Komponen | Multiplier | Efektif | Catatan |
|----------|-----------|---------|---------|
| encoder | `0.2` | 1e-6 | |
| decoder | `0.2` | 1e-6 | × Muon scale 20 → ~2e-5 |
| projector | `0.05` | 2.5e-7 | |
| vision_tower | `0.0` | — | **FROZEN** (`finetune_vision_layers=False`) |

`SFT_MUON_LR_SCALE = 20.0` — mis. decoder SFT: `5e-6 × 0.2 × 20 ≈ 2e-5`.

---

## 4. Phase 2 — Joint ORPO (Teks + Vision)

| Parameter | Nilai | Catatan |
|-----------|-------|---------|
| `ORPO_BETA` | `0.1` | |
| `ORPO_LEARNING_RATE` | `5e-6` | |
| `ORPO_NUM_EPOCHS` | `1` | Total step ORPO hanya ~18 |
| `ORPO_PER_DEVICE_TRAIN_BATCH_SIZE` | `4` | |
| `ORPO_PER_DEVICE_EVAL_BATCH_SIZE` | `8` | |
| `ORPO_GRADIENT_ACCUMULATION_STEPS` | `16` | |
| `ORPO_WARMUP_STEPS` | `2` | warmup 100 = LR tidak pernah peak |
| `ORPO_WEIGHT_DECAY` | `0.1` | |
| `ORPO_LR_SCHEDULER_TYPE` | `cosine` | |
| `ORPO_SAVE_EVAL_STEPS` | `6` | ~3 titik eval dalam ~18 steps |
| `ORPO_SAVE_TOTAL_LIMIT` | `2` | |
| `ORPO_LABEL_SMOOTHING_FACTOR` | `0.0` | ⚠️ **WAJIB 0.0** — smoothing merusak odds-ratio ORPO |
| `ORPO_MAX_GRAD_NORM` | `5.0` | |
| `ORPO_PREDICT_WITH_GENERATE` | `True` | |

**Split-LR per komponen:**

| Komponen | Multiplier | Efektif |
|----------|-----------|---------|
| encoder | `0.5` | 2.5e-6 |
| decoder | `1.0` | 5e-6 × Muon scale 5 → **~2.5e-5** |
| projector | `1.0` | 5e-6 |
| vision_tower | `0.5` | 2.5e-6 |

`ORPO_MUON_LR_SCALE = 5.0` — decoder ORPO ≈ `2.5e-5` (di sweet spot paper "When Does Muon
Help Agentic RL" — lihat review-01 #3).

Config: `TEXT_ORPO_CONFIG = "chat_orpo"`, `VISION_ORPO_CONFIG = "vision_orpo"`.

---

## 5. Optimizer Custom — GrokMuonAdEMA

| Parameter | Nilai | Catatan |
|-----------|-------|---------|
| `MUON_MOMENTUM` | `0.95` | |
| `MUON_NS_STEPS` | `5` | Newton-Schulz zeropower |
| `MUON_NESTEROV` | `True` | |
| `MUON_MAX_GRAD_NORM` | `1.0` | MuonClip threshold |

- **Cabang Muon** → parameter 2D (matriks): LoRA A/B (rank 256), proyeksi, dll. —
  orthonormalisasi via Newton-Schulz, update via `zeropower`.
- **Cabang AdEMAMix** → parameter 1D (bias, norm, embedding).
- MuonClip memotong gradien hasil filter sebelum cabang.
- Weight decay **decoupled** (per-param `mul_(1 - lr*wd)`).
- Split LR per komponen (encoder/decoder/projector/vision_tower) via param groups.

---

## 6. Konfigurasi Umum & Teknik Tambahan

| Item | Nilai |
|------|-------|
| `MAX_SOURCE_LENGTH` | `16384` |
| `MAX_TARGET_LENGTH` | `2048` |
| `LORA_RANK` | `256` |
| `LORA_ALPHA` | `512` |
| `LORA_DROPOUT` | `0.2` |
| Teknik | Logit masking · selective label smoothing · NEFTune (alpha 5.0) · task prefix mapping |

**Task prefix mapping** (tokenizer): `summarize`, `translate`, `ner`, `qa`,
`paraphrase`, `general_chat` (6 prefix, dari analisis awal — cek ulang di notebook
untuk daftar final).

**Final:** merge adapter BF16 + kuantisasi 4bit → deploy chatbot multimodal Indonesia.

---

## 7. Perbedaan Config "v6-era" (dipakai di obrolan) vs v7 (di disk)

> Versi notebook yang dilampirkan di obrolan (v6-era) berbeda dari v7 di disk
> (update terakhir 3 Agu 2026). Perbedaan penting yang terdeteksi:

| Parameter | v6-era (obrolan) | v7 (disk) |
|-----------|------------------|-----------|
| `LORA_DROPOUT` | 0.1 | **0.2** |
| `MAX_TARGET_LENGTH` | 512 | **2048** |
| `ORPO_BETA` | 0.4 | **0.1** |
| `SFT_LEARNING_RATE` | 1.5e-4 | **5e-6** (dengan split-LR multiplier) |
| Split-LR SFT | 3.0/2.0/1.0/0.5 | **0.2/0.2/0.05/0.0** (vision tower FROZEN) |
| `NEFTUNE_NOISE_ALPHA` | 10.0 | **5.0** |
| `SFT_NUM_EPOCHS` | — | **2** |
| Batch SFT | 3 (accum 8) | **4 (accum 16, effective 64)** |

> ⚠️ Analisis & rekomendasi paper di docs ini berlaku untuk konsep pipeline secara umum;
> angka yang dikutip (mis. sweet spot Muon 2.5e-5) sudah mengacu ke **v7**.

---

## 8. Catatan Evaluasi (dari obrolan)

- `SampleGenerationCallback` (teks) & `VisionSampleGenerationCallback` (vision) —
  eval set **sama** dipakai di Phase 1 & Phase 2 → perbandingan SFT-vs-ORPO apple-to-apple.
- Eval set (dari konteks obrolan): I-ViD, OpenI, RSICD (vision) + SQuAD, common_gen,
  HateSpeech, mixed NLU (teks) — cek notebook untuk daftar final.
- Usulan tambahan (dari riset 2026-08-05): **CultureTalk-ID** [2607.21016] sebagai
  eval pasca-training (budaya Indonesia, 11 bahasa).
