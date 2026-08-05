# [RESEARCH] T5Gemma V7 Combined Training Methodology & Blueprint

**Last Updated:** 5 Agustus 2026  
**Implementation Script:** [working-molab-v7-combined-unsloth.py](file:///d:/Codings/unsloth-porto/t5-gemma-2/instruct/notebooks/working-molab-v7-combined-unsloth.py)  
**Model Base:** `google/t5gemma-2-4b-4b` / `daruokta/t5gemma-2-4b-4b-instruct-chat-indo-v4-vision-cangkok`

---

## 1. Latar Belakang & Motivasi V7 Combined Training

Pada eksperimen sebelumnya (V6 & V6-Vision):
- Training **Text-only** dan **Vision-only** dilakukan secara terpisah (decoupled).
- Kerugian decoupled training: Ketika model menerima input campur (teks tanpa gambar diikuti pertanyaan multimodal), terjadi penurunan performa atau kebingungan pemrosesan soft token visual.

V7 dirancang untuk menyatukan **Combined Multi-task Training (Text + Vision + Reasoning)** dalam satu loop pelatihan terpadu (*unified training pipeline*).

---

## 2. Arsitektur Pipeline Combined V7

```
                            Combined Dataset Batch
                                      │
           ┌──────────────────────────┴──────────────────────────┐
           ▼                                                     ▼
Text-Only Samples (Instruction/Code)             Vision-Multimodal Samples (Image+Text)
           │                                                     │
           ▼                                                     ▼
Token Processing (No Vision Tower active)        SigLIP Vision Tower (896x896 -> 256 soft tokens)
           │                                                     │
           └──────────────────────────┬──────────────────────────┘
                                      ▼
                      T5Gemma 2 Encoder-Decoder Model
                                      │
                        Merged Attention (Self + Cross)
                                      │
                     Shared LoRA Targets & Cross-Entropy Loss
```

---

## 3. Komponen Utama Script V7

1. **Integrated Multi-Modal Collator:** Menangani sampel teks murni dan sampel ber-gambar secara fleksibel tanpa crash dimensi.
2. **Unified LoRA Target Modules:**
   ```python
   target_modules = [
       "q_proj", "k_proj", "v_proj", "o_proj",
       "gate_proj", "up_proj", "down_proj"
   ]
   ```
3. **Gradient Accumulation & Precision:**
   - Presisi: `bfloat16`
   - Gradient Checkpointing: `unsloth` / `true`
   - Loss Normalization: Penyesuaian `gradient_accumulation_steps` untuk mencegah lonjakan loss pada batch heterogen.

---

## 4. Hasil & Temuan Evaluasi V7

- Performa pemahaman instruksi bahasa Indonesia tetap stabil bersamaan dengan peningkatan akurasi visual QA.
- Membantu menjaga stabilitas bobot *Merged Attention* decoder agar tidak *overfit* ke salah satu modalitas saja.
