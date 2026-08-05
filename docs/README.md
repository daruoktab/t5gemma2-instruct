# T5-Gemma-2 Documentation & Research Knowledge Base

Selamat datang di direktori dokumentasi resmi proyek **T5-Gemma-2 Instruct & Multimodal Fine-Tuning**. Folder ini berisi dokumen arsitektur, panduan teknis, hasil riset diagnostik, dan spesifikasi proyek yang terorganisir secara sistematis.

---

## 🗺️ Struktur & Peta Dokumen

```
docs/
├── 📁 01_architecture/              ← Arsitektur Model, Tokenizer, & Merged Attention
│   ├── ARCHITECTURE_MASTER.md
│   └── REVERSE_ENGINEERING_MERGE_ATTENTION.md
├── 📁 02_guides/                    ← Panduan Praktis Finetuning & Post-Training
│   ├── UNSLOTH_FINETUNING_GUIDE.md
│   ├── POST_TRAINING_ALGORITHMS.md
│   └── DATASET_PREPARATION_AND_AGENT.md
├── 📁 03_research_and_diagnostics/  ← Riset Multimodal, V7 Pipeline, & Debugging Loss
│   ├── COMBINED_TRAINING_METHODOLOGY_V7.md
│   ├── VISION_TRAINING_AND_CANGKOK_STRATEGY.md
│   ├── LOGIT_MASKING_AND_LOSS_DIVERGENCE_ANALYSIS.md
│   └── QUANTIZATION_AND_METRICS_FINDINGS.md
├── 📁 04_specifications/            ← Referensi API, Format Paper, & Assets
│   ├── OPENMODEL_API_REFERENCE.md
│   ├── PAPER_CREATION_GUIDE.md
│   ├── dataset_spec.html
│   └── project-infographic.png
├── 📁 paper/                        ← ArXiv Reference PDFs (T5, Gemma 3, PEFT, Task Vector, dll.)
└── 📁 sessions/                     ← Log Sesi & Review Paper Harian
```

---

## 📑 Ringkasan Isi Dokumentasi

### 1. Arsitektur & Fondasi (`01_architecture/`)
- **[ARCHITECTURE_MASTER.md](file:///d:/Codings/unsloth-porto/t5-gemma-2/instruct/docs/01_architecture/ARCHITECTURE_MASTER.md)**: Referensi utama perbandingan Gemma 3 (Decoder-only) vs T5Gemma2 (Encoder-Decoder), struktur layer, token ID unused, dan alokasi `<image_soft_token>` (ID 256001).
- **[REVERSE_ENGINEERING_MERGE_ATTENTION.md](file:///d:/Codings/unsloth-porto/t5-gemma-2/instruct/docs/01_architecture/REVERSE_ENGINEERING_MERGE_ATTENTION.md)**: Bedah aljabar Merged Attention ($Q, K, V = [X; H]$), analisis kegagalan pencangkokan bobot Gemma-IT mentah, dan teori *Task Vector / Model Arithmetic* ($\Delta_{\text{instruct}} = W_{\text{IT}} - W_{\text{Base}}$).

### 2. Panduan & Best Practices (`02_guides/`)
- **[UNSLOTH_FINETUNING_GUIDE.md](file:///d:/Codings/unsloth-porto/t5-gemma-2/instruct/docs/02_guides/UNSLOTH_FINETUNING_GUIDE.md)**: Langkah instalasi branch Seq2Seq Unsloth (`dh/recover-3153-seq2seq`), patch `_utils.py`, `peft` import fix, dan script pengujian finetuning.
- **[POST_TRAINING_ALGORITHMS.md](file:///d:/Codings/unsloth-porto/t5-gemma-2/instruct/docs/02_guides/POST_TRAINING_ALGORITHMS.md)**: Penjelasan algoritma SFT, DPO, ORPO, GRPO, serta rekomendasi penerapan ORPO pada model vision tanpa *reference model*.
- **[DATASET_PREPARATION_AND_AGENT.md](file:///d:/Codings/unsloth-porto/t5-gemma-2/instruct/docs/02_guides/DATASET_PREPARATION_AND_AGENT.md)**: Spesifikasi lengkap 3 repositori Hugging Face (`daruokta`), skema data (SFT, Multiturn, ORPO, Vision, Embedding), match file lokal, dan operasional Dataset Agent.

### 3. Riset & Diagnosis Training (`03_research_and_diagnostics/`)
- **[COMBINED_TRAINING_METHODOLOGY_V7.md](file:///d:/Codings/unsloth-porto/t5-gemma-2/instruct/docs/03_research_and_diagnostics/COMBINED_TRAINING_METHODOLOGY_V7.md)**: Metodologi pelatihan gabungan (Text + Vision + Code) pada script [working-molab-v7-combined-unsloth.py](file:///d:/Codings/unsloth-porto/t5-gemma-2/instruct/notebooks/working-molab-v7-combined-unsloth.py).
- **[VISION_TRAINING_AND_CANGKOK_STRATEGY.md](file:///d:/Codings/unsloth-porto/t5-gemma-2/instruct/docs/03_research_and_diagnostics/VISION_TRAINING_AND_CANGKOK_STRATEGY.md)**: Analisis Vision Encoder (SigLIP 400M, 256 soft tokens per gambar), transplantasi projector dari Gemma 3 IT, serta fix EOS & image token index.
- **[LOGIT_MASKING_AND_LOSS_DIVERGENCE_ANALYSIS.md](file:///d:/Codings/unsloth-porto/t5-gemma-2/instruct/docs/03_research_and_diagnostics/LOGIT_MASKING_AND_LOSS_DIVERGENCE_ANALYSIS.md)**: Patofisiologi mendalam "loss terbang", manipulasi logit mask, investigasi gradient accumulation, dan pencegahan error `NaN`/`inf`.
- **[QUANTIZATION_AND_METRICS_FINDINGS.md](file:///d:/Codings/unsloth-porto/t5-gemma-2/instruct/docs/03_research_and_diagnostics/QUANTIZATION_AND_METRICS_FINDINGS.md)**: Evaluasi kuantisasi 4-bit NF4 / 8-bit dan pemahaman metrik pelatihan (PPL, Grad Norm, Loss).

### 4. Spesifikasi & Assets (`04_specifications/`)
- **[OPENMODEL_API_REFERENCE.md](file:///d:/Codings/unsloth-porto/t5-gemma-2/instruct/docs/04_specifications/OPENMODEL_API_REFERENCE.md)**: Penanganan error 404 Route Not Found, konfigurasi SDK Anthropic, dan routing DeepSeek.
- **[PAPER_CREATION_GUIDE.md](file:///d:/Codings/unsloth-porto/t5-gemma-2/instruct/docs/04_specifications/PAPER_CREATION_GUIDE.md)**: Pedoman penulisan paper ilmiah format JACoW menggunakan Typst & CeTZ.

---

## 🔗 Hubungan dengan Notebooks Repositori

- **Active Pipeline:** [working-molab-v7-combined-unsloth.py](file:///d:/Codings/unsloth-porto/t5-gemma-2/instruct/notebooks/working-molab-v7-combined-unsloth.py) (Mengimplementasikan metodologi V7 Combined Multi-task & Unsloth Seq2Seq Patches).
- **Legacy Notebooks:** [notebooks/legacy/](file:///d:/Codings/unsloth-porto/t5-gemma-2/instruct/notebooks/legacy) (V3–V6, V6-Vision unsloth).
