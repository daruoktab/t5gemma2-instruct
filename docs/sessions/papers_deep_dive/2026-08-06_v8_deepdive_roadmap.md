# 🗺️ Roadmap V8 & Deep Dive 26 Paper arXiv + Foundational Papers

**Tanggal Sesi:** 6 Agustus 2026  
**Referensi Utama:** [`docs/sessions/2026-08-05_verifikasi-26-paper-arxiv.md`](file:///d:/Codings/unsloth-porto/t5-gemma-2/instruct/docs/sessions/2026-08-05_verifikasi-26-paper-arxiv.md)  
**Tujuan:** Membedah 26 paper arXiv pilihan + paper fondasi pendukung di `docs/paper/`, memisahkan paper yang **sudah di-apply di V7**, serta membuat **rencana eksekusi teknis konkret pada kode `notebooks/working-molab-v7-combined-unsloth.py` menuju Pipeline V8** untuk paper yang **belum di-apply**.

---

## 🟢 1. Paper yang Sudah Di-apply di Pipeline V7 (Ringkasan Brief)

| # | arXiv ID | Judul Paper & Penggunaan di V7 | Kode Terkait di Repositori |
|---|---|---|---|
| 1 | [2512.14856](https://arxiv.org/abs/2512.14856) | **T5Gemma 2 (Base Model):** Arsitektur dasar Seq2Seq 4B-4B, *Tied Embeddings* (262.144 vocab), dan *Merged Attention* ($K,V = [X; H]$). | `BASE_MODEL = "google/t5gemma-2-4b-4b"` |
| 2 | [2210.11399](https://arxiv.org/abs/2210.11399) | **UL2R:** Adaptasi pre-training denoising mixture (PrefixLM + Span Corruption) yang melekat pada base model T5Gemma 2. | Mode Prefix & Denoising Base |
| 3 | [2607.25583](https://arxiv.org/abs/2607.25583) | **LoRA Module Target & Saturation:** Konfigurasi target modul LoRA ke seluruh proyeksi linear (`q,k,v,o,gate,up,down`) dengan rank 128/16. | `FastLanguageModel.get_peft_model` |
| 4 | [2502.01968](https://arxiv.org/abs/2502.01968) | **Logit Masking / Token Cleaning:** Menekan *unused tokens* (`<unused0>` s.d. `<unused98>` & `<unused100>` s.d. `<unused6241>`) kecuali 6 task prefix. | `ALL_SUPPRESS_IDS` & Forward Hook di `inference.py` |
| 5 | [2503.19786](https://arxiv.org/abs/2503.19786) | **Cangkok Vision Encoder (SigLIP 400M):** Ekstraksi 256 soft tokens gambar disuntikkan ke Encoder pada posisi token `<image_soft_token>` (`256001`). | `test_vanilla_seq2seq_vision.py` |
| 6 | [2606.24841](https://arxiv.org/abs/2606.24841) | **Task Prefix Mapping (Parsial):** Pemetaan awalan tugas `<unused1>` s.d. `<unused6>` (Summarize, Translate, NER, QA, Paraphrase, Chat). | `generate_prefix_tasks.py` |

---

## 🚀 2. Indeks Dokumen Deep Dive Teknis V8 (01 s.d. 07)

Seluruh paper telah dianalisis secara visual dari PDF asli (`view_file` PDF) dan didokumentasikan ke dalam 7 berkas deep-dive:

1. **[Deep-Dive 01 — Token-Level Policy Optimization & Alignment Mechanisms](file:///d:/Codings/unsloth-porto/t5-gemma-2/instruct/docs/sessions/papers_deep_dive/2026-08-06_v8_deepdive_01_tlpo_alignment.md)**
   - TLPO (2604.26553), FT_OBJECTIVES_SAFETY (2601.12639), ALIGNMENT_MECHANISTIC (2606.09850), CULTURE_ID (2607.21016).
2. **[Deep-Dive 02 — OrScale Optimizer & Non-Linear Training Dynamics](file:///d:/Codings/unsloth-porto/t5-gemma-2/instruct/docs/sessions/papers_deep_dive/2026-08-06_v8_deepdive_02_orscale_optimizer.md)**
   - ORSCALE (2605.07815), MUON_RL (2607.16169), PAIRWISE_FRAGILE (2607.16821), LOCAL_LINEAR (2606.10929).
3. **[Deep-Dive 03 — Task Vector Geometry, DeVec & Subspace Editing](file:///d:/Codings/unsloth-porto/t5-gemma-2/instruct/docs/sessions/papers_deep_dive/2026-08-06_v8_deepdive_03_task_vectors_and_subspace_editing.md)**
   - TV_GEOM (2605.03780), TV_DECOMP (2512.22511), EASYEDIT (2308.07269), EMERGENT_MISALIGNMENT (2607.21356), CAUSAL_ENCODERS (2512.10561).
4. **[Deep-Dive 04 — Architecture, Internal Retrieval & Cross-Model Multimodal](file:///d:/Codings/unsloth-porto/t5-gemma-2/instruct/docs/sessions/papers_deep_dive/2026-08-06_v8_deepdive_04_architecture_retrieval_multimodal.md)**
   - INTRA (2605.05806), XBRIDGE (2603.17512), T5GEMMA2 (2512.14856), UL2R (2210.11399), PM_ROPE_TTS (2604.01760).
5. **[Deep-Dive 05 — Objectives, Data Formatting & Domain Adaptation](file:///d:/Codings/unsloth-porto/t5-gemma-2/instruct/docs/sessions/papers_deep_dive/2026-08-06_v8_deepdive_05_objectives_data_domain_adaptation.md)**
   - MTO (2606.24841), FLEXTAB (2606.30336), STYLE_TRANSFER (2604.11687), DOMAIN_ADAPTATION (2607.06613), LORA_TRADEOFFS (2607.25583).
6. **[Deep-Dive 06 — Latent Memory, Context Window Extension & Speculative Decoding](file:///d:/Codings/unsloth-porto/t5-gemma-2/instruct/docs/sessions/papers_deep_dive/2026-08-06_v8_deepdive_06_memory_context_speculative_decoding.md)**
   - LATENT_MEMORY (2606.20911), STACKED_CONTEXT (2603.04759), CASSANDRA_SPECULATIVE (2605.26558).
7. **[Deep-Dive 07 — Foundational & External Papers in `docs/paper/`](file:///d:/Codings/unsloth-porto/t5-gemma-2/instruct/docs/sessions/papers_deep_dive/2026-08-06_v8_deepdive_07_foundational_and_external_papers.md)**
   - 2501.16273 (Return of the Encoder), 2502.01968 (Token Cleaning), 2503.19786 (Gemma 3 Tech Report), 1910.10683 (T5), 2210.11416 (FLAN), 2212.04089 & 2502.20186 (Task Arithmetic & Layer Aware), 2503.02656 & 2504.06225 (Gemma Encoder/Enc-Dec), 2505.12021 & 2511.16147 & 2602.01227.

---

## 🎯 Ringkasan Peta Eksekusi Menuju Pipeline V8

1. **Loss & Alignment:** Penambahan **TLPO** (Token-Level Policy Optimization) dan evaluasi **CultureTalk-ID**.
2. **Optimizer:** Implementasi **OrScale-LM** (layer-wise trust ratio scaling).
3. **Weight Management:** **DeVec** (SVD Task Vector Decomposition) untuk mencegah forgetting & unlearning.
4. **Retrieval Internal:** Modul **INTRA Reverse-QWK** pada Cross-Attention T5Gemma 2 untuk menghemat 30x KV cache.
5. **Memory & Context:** Latent Personal Memory (**LPM**) dynamic soft prompts & **SHAREDLLM** self-injection context tree.
6. **Inference Acceleration:** **Cassandra** self-speculative decoding tanpa pelatihan ulang.
