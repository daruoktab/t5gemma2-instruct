# 📑 Deep Dive Research Papers Index & Visual Archive (V8 Roadmap)

Halaman ini berisi indeks lengkap 26 paper riset prioritas (beserta paper fondasi pendukung di `docs/paper/`) yang telah di-download, di-render ke gambar PNG (halaman per halaman), dan dianalisis secara visual dan matematis untuk mendukung pengembangan model **T5-Gemma-2 Instruct & Multimodal V8** berbasis `notebooks/working-molab-v7-combined-unsloth.py`.

---

## 🗺️ Peta Berkas & Hasil Rendered Pages

```
docs/
├── paper/                                  ← Paper Fondasi & Eksternal (Di Luar 26 Paper Sesi Core)
│   ├── 1907.12461_Warm_Starting_Enc_Dec_Rothe2019.pdf
│   ├── 1910.10683_T5_Raffel2020.pdf
│   ├── 2210.11416_FLAN_Instruction_Scaling_Chung2022.pdf
│   ├── 2212.04089_Task_Arithmetic_Ilharco2023.pdf
│   ├── 2501.16273_Return_of_the_Encoder_Microsoft_2025.pdf
│   ├── 2502.01968v2_Token_Cleaning_Pang2025.pdf
│   ├── 2502.17823v2_Enhance_Unlearning_Ren2025.pdf
│   ├── 2502.20186_Layer_Aware_Task_Arithmetic_2025.pdf
│   ├── 2503.02656_Gemma_Encoder_2025.pdf
│   ├── 2503.19786_Gemma_3_Technical_Report_DeepMind_2025.pdf
│   ├── 2504.06225_Encoder_Decoder_Gemma_2025.pdf
│   ├── 2505.12021_Cross_Model_Transfer_Task_Vectors_2025.pdf
│   ├── 2511.16147v3_TS_PEFT_Ma2025.pdf
│   └── 2602.01227v2_Token_Priority_Shen2026.pdf
└── sessions/papers_deep_dive/
    ├── 📁 pdfs/                            ← Berkas PDF Asli 26 Paper Core arXiv
    ├── 📁 pages/                           ← Render Halaman PDF ke Gambar PNG (26 Sub-folder)
    ├── 📄 INDEX.md                        ← Indeks Utama
    ├── 📄 2026-08-06_v8_deepdive_roadmap.md ← Pemetaan Roadmap Eksekusi V8
    ├── 📄 2026-08-06_v8_deepdive_01_tlpo_alignment.md
    ├── 📄 2026-08-06_v8_deepdive_02_orscale_optimizer.md
    ├── 📄 2026-08-06_v8_deepdive_03_task_vectors_and_subspace_editing.md
    ├── 📄 2026-08-06_v8_deepdive_04_architecture_retrieval_multimodal.md
    ├── 📄 2026-08-06_v8_deepdive_05_objectives_data_domain_adaptation.md
    ├── 📄 2026-08-06_v8_deepdive_06_memory_context_speculative_decoding.md
    ├── 📄 2026-08-06_v8_deepdive_07_foundational_and_external_papers.md
    └── 📄 2026-08-16_v8_deepdive_08_cross_papers_ideas_extraction.md
```

---

## 📚 Daftar Dokumen Deep Dive Teknis V8 (01 s.d. 08)

### 1. [Deep-Dive 01 — Token-Level Policy Optimization & Alignment Mechanisms](file:///d:/Codings/unsloth-porto/t5-gemma-2/instruct/docs/sessions/papers_deep_dive/2026-08-06_v8_deepdive_01_tlpo_alignment.md)
- **TLPO** (2604.26553): Token-Level Policy Optimization (Samsung SDS) untuk mencegah *language confusion*.
- **FT_OBJECTIVES_SAFETY** (2601.12639): Pengaruh objective fine-tuning pada batas keamanan model.
- **ALIGNMENT_MECHANISTIC** (2606.09850): Analisis mekanistik posisi aliniasi pada layer 12–24.
- **CULTURE_ID** (2607.21016): Benchmark CultureTalk-ID 11 bahasa daerah Indonesia.

### 2. [Deep-Dive 02 — OrScale Optimizer & Non-Linear Training Dynamics](file:///d:/Codings/unsloth-porto/t5-gemma-2/instruct/docs/sessions/papers_deep_dive/2026-08-06_v8_deepdive_02_orscale_optimizer.md)
- **ORSCALE** (2605.07815): OrScale-LM layer-wise trust ratio scaling untuk Muon.
- **MUON_RL** (2607.16169): Karakteristik optimasi Muon pada RL post-training.
- **PAIRWISE_FRAGILE** (2607.16821): Pairwise gradient fragility pada multi-task SFT.
- **LOCAL_LINEAR** (2606.10929): Struktur linier lokal pada residual stream.

### 3. [Deep-Dive 03 — Task Vector Geometry, DeVec & Subspace Editing](file:///d:/Codings/unsloth-porto/t5-gemma-2/instruct/docs/sessions/papers_deep_dive/2026-08-06_v8_deepdive_03_task_vectors_and_subspace_editing.md)
- **TV_GEOM** (2605.03780): Dual Modes of Task Inference (Mode M1 Bayesian Retrieval vs M2 OOD Learning).
- **TV_DECOMP** (2512.22511): DeVec - SVD Decomposing Task Vectors into Shared & Unique components.
- **EASYEDIT** (2308.07269): ROME / MEMIT Knowledge Editing Framework.
- **EMERGENT_MISALIGNMENT** (2607.21356): Emergent Misalignment Subspace Removal via activation projection.
- **CAUSAL_ENCODERS** (2512.10561): Causal Encoders for Representation Learning.

### 4. [Deep-Dive 04 — Architecture, Internal Retrieval & Cross-Model Multimodal](file:///d:/Codings/unsloth-porto/t5-gemma-2/instruct/docs/sessions/papers_deep_dive/2026-08-06_v8_deepdive_04_architecture_retrieval_multimodal.md)
- **INTRA** (2605.05806): INTrinsic Retrieval via Attention (Reverse-QWK) menghemat 30x KV cache.
- **XBRIDGE** (2603.17512): Enc-LLM-Dec Cross-Model Bridge dengan Optimal Transport (OT) Alignment.
- **T5GEMMA2** (2512.14856): Base Seq2Seq 4B-4B Architecture & SigLIP Vision Encoder Integration.
- **UL2R** (2210.11399): Mixture-of-Denoisers Pre-training.
- **PM_ROPE_TTS** (2604.01760): Pattern-Masked RoPE untuk Audio/Speech extension.

### 5. [Deep-Dive 05 — Objectives, Data Formatting & Domain Adaptation](file:///d:/Codings/unsloth-porto/t5-gemma-2/instruct/docs/sessions/papers_deep_dive/2026-08-06_v8_deepdive_05_objectives_data_domain_adaptation.md)
- **MTO** (2606.24841): Matching Tasks to Objectives (Mask-Filling vs Map-Phrasal).
- **FLEXTAB** (2606.30336): Target-Agnostic Row Embedding Enc-Dec untuk Tabular ICL.
- **STYLE_TRANSFER** (2604.11687): Enc-Dec vs Dec-Only AI-to-Human Style Transfer.
- **DOMAIN_ADAPTATION** (2607.06613): Continual Pre-Training on Structured SE Texts.
- **LORA_TRADEOFFS** (2607.25583): Parameter Trade-offs & Early Rank Saturation ($r=16$).

### 6. [Deep-Dive 06 — Latent Memory, Context Window Extension & Speculative Decoding](file:///d:/Codings/unsloth-porto/t5-gemma-2/instruct/docs/sessions/papers_deep_dive/2026-08-06_v8_deepdive_06_memory_context_speculative_decoding.md)
- **LATENT_MEMORY** (2606.20911): Latent Personal Memory (LPM) Dynamic Soft Prompts.
- **STACKED_CONTEXT** (2603.04759): SHAREDLLM Multi-Scale Self-Injection for 128K+ Context Extension.
- **CASSANDRA_SPECULATIVE** (2605.26558): Cassandra Training-Free Self-Speculative Decoding.

### 7. [Deep-Dive 07 — Foundational & External Papers in `docs/paper/`](file:///d:/Codings/unsloth-porto/t5-gemma-2/instruct/docs/sessions/papers_deep_dive/2026-08-06_v8_deepdive_07_foundational_and_external_papers.md)
- **2501.16273** (Return of the Encoder / Microsoft 2025): 47% lower latency & 4.7x throughput Enc-Dec vs Dec-only.
- **2502.01968** (Token Cleaning / Pang 2025): Logit Masking for unused tokens (V7/V8).
- **2503.19786** (Gemma 3 Technical Report / DeepMind 2025): SigLIP Vision Grafting.
- **1910.10683** (T5 Raffel 2020) & **1907.12461** (Rothe 2019).
- **2210.11416** (FLAN Chung 2022).
- **2212.04089** (Task Arithmetic 2023) & **2502.20186** (Layer-Aware Task Arithmetic).
- **2503.02656** (Gemma Encoder) & **2504.06225** (Encoder-Decoder Gemma).
- **2505.12021** (Cross-Model Transfer) & **2511.16147** (TS-PEFT) & **2602.01227** (Token Priority).

### 8. [Deep-Dive 08 — Cross-Papers Ideas Extraction (CSV & Foundation)](file:///d:/Codings/unsloth-porto/t5-gemma-2/instruct/docs/sessions/papers_deep_dive/2026-08-16_v8_deepdive_08_cross_papers_ideas_extraction.md)
- **Cross-Architecture Distillation** (Microsoft `2501.16273` & Google `2504.06225`): Teacher Decoder $\to$ Student Seq2Seq KD.
- **GRUN Unlearning Framework** (`2502.17823`): Gated Representation Tuning & Soft-Gate Guard.
- **Layer-Aware Task Arithmetic** (`2502.20186`): Layer-wise rank & logarithmic scaling.
- **Cross-Model Task Vector Transfer** (`2505.12021`): Orthogonal alignment matrix ($U^T \Delta W U$).
- **Gemma Encoder Downstream Tuning** (`2503.02656`): Bidirectional masking & dropout 0.1.

