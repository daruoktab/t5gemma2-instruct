# 📚 Katalog Paper & Pemetaan Status Implementasi Versi (V7, V8, V9, V10)
# T5Gemma-2 Indonesian Instruct Multimodal Research & Engineering

**Tanggal Dokumen:** 16 Agustus 2026  
**Repositori Proyek:** `daruoktab/t5gemma2-instruct`  
**Lokasi Direktori PDF:**
1. `docs/sessions/papers_deep_dive/pdfs/` (26 Berkas PDF Deep-Dive Utama arXiv)
2. `docs/paper/` (21 Berkas PDF Fondasi, Arsitektur, & Riset Tambahan dari CSV)

---

## 📌 1. Ringkasan Status Versi

| Kode Status | Keterangan Versi | Total Paper Terpetakan |
|---|---|:---:|
| `V7_APPLIED` | Fitur sudah diimplementasikan dan aktif di skrip pipeline `working-molab-v7-combined-unsloth.py` | 6 |
| `V8_APPLIED` | Fitur utama yang diimplementasikan pada skrip pipeline `working-molab-v8-combined-unsloth.py` | 5 |
| `V9_PLANNED` | Fitur lanjutan yang direncanakan untuk iterasi V9 (Distilasi Guru-Murid, Guardrail Unlearning, Cross-Model OT) | 4 |
| `V10_PLANNED`| Arsitektur SOTA efisiensi tinggi masa depan (30x KV cache compression, Speculative decoding, Long-context 128k+) | 4 |
| `REFERENCE`  | Landasan teori, benchmark empiris, evaluasi budaya, dan referensi domain/arsitektur pendukung | 28 |

---

## 🗂️ 2. Katalog Terpadu PDF & Pemetaan Status Versi

### A. Deep-Dive arXiv Papers (`docs/sessions/papers_deep_dive/pdfs/`)

| No | Nama Berkas PDF | Judul / Topik Utama | arXiv ID | Status Versi | Implementasi & Catatan Teknis |
|:---:|---|---|:---:|:---:|---|
| 1 | `T5GEMMA2_2512.14856.pdf` | T5Gemma 2 Technical Report | [2512.14856](https://arxiv.org/abs/2512.14856) | `V7_APPLIED` | Arsitektur dasar Seq2Seq 4B-4B, Tied Embeddings (262.144 vocab), Merged Attention $[X; H]$. |
| 2 | `UL2R_2210.11399.pdf` | UL2R Pre-training Mixtures | [2210.11399](https://arxiv.org/abs/2210.11399) | `V7_APPLIED` | Denoising span corruption + PrefixLM bawaan model dasar T5Gemma-2. |
| 3 | `LORA_TRADEOFFS_2607.25583.pdf` | LoRA Module Target & Trade-offs | [2607.25583](https://arxiv.org/abs/2607.25583) | `V7_APPLIED` | Penargetan LoRA linear projections (`q,k,v,o,gate,up,down`) rank 128/256. |
| 4 | `ORSCALE_2605.07815.pdf` | OrScale-LM Optimizer | [2605.07815](https://arxiv.org/abs/2605.07815) | `V8_APPLIED` | Pengganti Muon: Layer-wise Trust Ratio Scaling pada matriks 2D LoRA/Linear. |
| 5 | `TV_DECOMP_2512.22511.pdf` | DeVec SVD Task Vector Decomposition | [2512.22511](https://arxiv.org/abs/2512.22511) | `V8_APPLIED` | SVD Subspace Filtering ($\tau=0.85$) sebelum steering FFN & LayerNorm decoder. |
| 6 | `TLPO_2604.26553.pdf` | Token-Level Policy Optimization (TLPO) | [2604.26553](https://arxiv.org/abs/2604.26553) | `V8_APPLIED` | Token-level cross-entropy penalty pada token non-Indonesian/confusion di ORPO. |
| 7 | `MTO_2606.24841.pdf` | Matching Tasks to Objectives (MTO) | [2606.24841](https://arxiv.org/abs/2606.24841) | `V8_APPLIED` | Formatting prompt & injection decoder task prefix (`<unused1>` s.d. `<unused6>`). |
| 8 | `CULTURE_ID_2607.21016.pdf` | CultureTalk-ID Benchmark | [2607.21016](https://arxiv.org/abs/2607.21016) | `V8_APPLIED` | Metrik evaluasi keselarasan budaya & nuansa bahasa Indonesia. |
| 9 | `XBRIDGE_2603.17512.pdf` | X-Bridge Cross-Modal Optimal Transport | [2603.17512](https://arxiv.org/abs/2603.17512) | `V9_PLANNED` | Sinkhorn optimal transport bridge untuk alignment multi-modal tanpa fine-tune penuh. |
| 10 | `EASYEDIT_2308.07269.pdf` | EasyEdit Knowledge Editing Framework | [2308.07269](https://arxiv.org/abs/2308.07269) | `V9_PLANNED` | Direct model editing & localized parameter updates tanpa retraining. |
| 11 | `INTRA_2605.05806.pdf` | INTRA Reverse-QWK Internal Retrieval | [2605.05806](https://arxiv.org/abs/2605.05806) | `V10_PLANNED` | Kompresi 30x KV cache cross-attention pada dokumen panjang. |
| 12 | `CASSANDRA_SPECULATIVE_2605.26558.pdf` | Cassandra Training-Free Speculative Decoding | [2605.26558](https://arxiv.org/abs/2605.26558) | `V10_PLANNED` | 2.5x throughput generation acceleration tanpa model draft tambahan. |
| 13 | `STACKED_CONTEXT_2603.04759.pdf` | SHAREDLLM Stacked Context Injection | [2603.04759](https://arxiv.org/abs/2603.04759) | `V10_PLANNED` | Ekstensi panjang konteks 128k+ via self-injection hierarchical context tree. |
| 14 | `LATENT_MEMORY_2606.20911.pdf` | Latent Personal Memory (LPM) | [2606.20911](https://arxiv.org/abs/2606.20911) | `V10_PLANNED` | Dynamic soft prompts & persistent multi-session user memory. |
| 15 | `ALIGNMENT_MECHANISTIC_2606.09850.pdf` | Mechanistic Analysis of Alignment | [2606.09850](https://arxiv.org/abs/2606.09850) | `REFERENCE` | Analisis layer representasi keselamatan dan alignment subruang laten. |
| 16 | `CAUSAL_ENCODERS_2512.10561.pdf` | Causal Reasoning Favors Encoders | [2512.10561](https://arxiv.org/abs/2512.10561) | `REFERENCE` | Bukti empiris keunggulan encoder-decoder pada multi-hop reasoning. |
| 17 | `DOMAIN_ADAPTATION_2607.06613.pdf` | Domain Adaptation Dynamics | [2607.06613](https://arxiv.org/abs/2607.06613) | `REFERENCE` | Analisis dinamika pergeseran representasi saat adaptasi domain khusus. |
| 18 | `EMERGENT_MISALIGNMENT_2607.21356.pdf` | Emergent Misalignment Analysis | [2607.21356](https://arxiv.org/abs/2607.21356) | `REFERENCE` | Pencegahan fenomena misalign tersembunyi pada model yang di-steer. |
| 19 | `FLEXTAB_2606.30336.pdf` | FlexTab Table Reasoning | [2606.30336](https://arxiv.org/abs/2606.30336) | `REFERENCE` | Struktur pemahaman data tabular multimodal. |
| 20 | `FT_OBJECTIVES_SAFETY_2601.12639.pdf` | Fine-Tuning Objectives & Safety | [2601.12639](https://arxiv.org/abs/2601.12639) | `REFERENCE` | Studi perbandingan stabilitas safety guardrails pada SFT vs DPO/ORPO. |
| 21 | `LOCAL_LINEAR_2606.10929.pdf` | Local Linearity in LLM Training | [2606.10929](https://arxiv.org/abs/2606.10929) | `REFERENCE` | Dinamika non-linear loss landscape model skala besar. |
| 22 | `MUON_RL_2607.16169.pdf` | Muon in Reinforcement Learning | [2607.16169](https://arxiv.org/abs/2607.16169) | `REFERENCE` | Studi efektivitas ortogonalisasi Newton-Schulz pada policy updates. |
| 23 | `PAIRWISE_FRAGILE_2607.16821.pdf` | Pairwise Preference Fragility | [2607.16821](https://arxiv.org/abs/2607.16821) | `REFERENCE` | Justifikasi stabilisasi numerik odds-ratio pada ORPO. |
| 24 | `PM_ROPE_TTS_2604.01760.pdf` | PM-RoPE Positional Encoding | [2604.01760](https://arxiv.org/abs/2604.01760) | `REFERENCE` | RoPE multimodal interleaved positioning. |
| 25 | `STYLE_TRANSFER_2604.11687.pdf` | Style Transfer & Tone Control | [2604.11687](https://arxiv.org/abs/2604.11687) | `REFERENCE` | Modulasi tone kesantunan berbahasa Indonesia. |
| 26 | `TV_GEOM_2605.03780.pdf` | Task Vector Geometry in LLMs | [2605.03780](https://arxiv.org/abs/2605.03780) | `REFERENCE` | Karakteristik topologi sudut kosinus antar task vector. |

---

### B. Foundational, Architecture, & Additional CSV Papers (`docs/paper/`)

| No | Nama Berkas PDF | Judul / Topik Utama | arXiv / Sumber | Status Versi | Implementasi & Catatan Teknis |
|:---:|---|---|:---:|:---:|---|
| 27 | `2503.19786_Gemma_3_Technical_Report_DeepMind_2025.pdf` | Gemma 3 Technical Report | [2503.19786](https://arxiv.org/abs/2503.19786) | `V7_APPLIED` | Transplantasi Vision Encoder SigLIP 400M & Multi-Modal Projector ke T5Gemma-2. |
| 28 | `2502.01968v2_Token_Cleaning_Pang2025.pdf` | Token Cleaning / Logit Masking | [2502.01968](https://arxiv.org/abs/2502.01968) | `V7_APPLIED` | Forward hook suppression logit token kosong (`<unused0>` s.d. `<unused6241>`). |
| 29 | `2212.04089_Task_Arithmetic_Ilharco2023.pdf` | Task Arithmetic & Vector Steering | [2212.04089](https://arxiv.org/abs/2212.04089) | `V7_APPLIED` | Phase 0.5 Parameter Steering $\Delta = W_{\text{IT}} - W_{\text{Base}}$ pada FFN & Norm. |
| 30 | `2502.20186_Layer_Aware_Task_Arithmetic_2025.pdf` | Layer-Aware Task Arithmetic (LATA) | [2502.20186](https://arxiv.org/abs/2502.20186) | `V8_APPLIED` | Skema Ramp-Up Layer-wise Alpha ($\alpha_{\text{early}}=0.05, \alpha_{\text{mid}}=0.25, \alpha_{\text{late}}=0.08$). |
| 31 | `2501.16273_Return_of_the_Encoder_Microsoft_2025.pdf` | Return of the Encoder (SLMs) | [2501.16273](https://arxiv.org/abs/2501.16273) | `V9_PLANNED` | Distilasi Guru Decoder-only (Gemma 3 IT) $\to$ Murid Encoder-Decoder (T5Gemma-2). |
| 32 | `2504.06225_Encoder_Decoder_Gemma_2025.pdf` | Encoder-Decoder Gemma Architecture | [2504.06225](https://arxiv.org/abs/2504.06225) | `V9_PLANNED` | Topologi cross-attention asimetris 2:1 Encoder-to-Decoder. |
| 33 | `2502.17823v2_Enhance_Unlearning_Ren2025.pdf` | GRUN Unlearning Framework | [2502.17823](https://arxiv.org/abs/2502.17823) | `V9_PLANNED` | Gated Representation Soft-ReFT untuk guardrail unlearning tanpa retraining. |
| 34 | `2505.12021_Cross_Model_Transfer_Task_Vectors_2025.pdf` | Cross-Model Transfer of Task Vectors | [2505.12021](https://arxiv.org/abs/2505.12021) | `V9_PLANNED` | Orthogonal subspace alignment $U^T \Delta W U$ transfer antar arsitektur berbeda. |
| 35 | `1907.12461_Warm_Starting_Enc_Dec_Rothe2019.pdf` | Warm-Starting Encoder-Decoder Models | [1907.12461](https://arxiv.org/abs/1907.12461) | `REFERENCE` | Fondasi warm-starting bobot BERT/RoBERTa/GPT ke arsitektur Seq2Seq. |
| 36 | `1910.10683_T5_Raffel2020.pdf` | Exploring the Limits of Transfer (T5) | [1910.10683](https://arxiv.org/abs/1910.10683) | `REFERENCE` | Fondasi arsitektur Text-to-Text Transfer Transformer (T5). |
| 37 | `2210.11416_FLAN_Instruction_Scaling_Chung2022.pdf` | Scaling Instruction-Finetuned Language Models | [2210.11416](https://arxiv.org/abs/2210.11416) | `REFERENCE` | Fondasi skala dataset FLAN dan task prefixing. |
| 38 | `2503.02656_Gemma_Encoder_2025.pdf` | Gemma as an Encoder | [2503.02656](https://arxiv.org/abs/2503.02656) | `REFERENCE` | Karakteristik representasi bidireksional pada encoder Gemma. |
| 39 | `2511.16147v3_TS_PEFT_Ma2025.pdf` | TS-PEFT Two-Stage Adapter Tuning | [2511.16147](https://arxiv.org/abs/2511.16147) | `REFERENCE` | Desain adapter dua tahap untuk representasi multi-task. |
| 40 | `2602.01227v2_Token_Priority_Shen2026.pdf` | Token Priority Aware Optimization | [2602.01227](https://arxiv.org/abs/2602.01227) | `REFERENCE` | Teori pengalokasian prioritas gradien pada token khusus/task prefix. |
| 41 | `2510.26622_Encoder_decoder_or_decoder_only__revisit.pdf` | RedLLM: Revisiting Encoder-Decoder LLM | [2510.26622](https://arxiv.org/abs/2510.26622) | `REFERENCE` | Komparasi empiris kapasitas memori & throughput Enc-Dec vs Decoder-only. |
| 42 | `2506.00807_Enhancing_llm_reasoning_for_time_series.pdf` | Enhancing LLM Reasoning & Fused Decision | [2506.00807](https://arxiv.org/abs/2506.00807) | `REFERENCE` | Pola tailored thinking & multi-granularity patching pada T5 encoder-decoder. |
| 43 | `2601.03450_Soft_Contextualized_Encoder_For_User_Def.pdf` | Soft Contextualized Encoder for Text Classification | [2601.03450](https://arxiv.org/abs/2601.03450) | `REFERENCE` | Efisiensi forward pass representasi encoder untuk zero-shot adaptasi. |
| 44 | `2603.11991_BTZSC__A_Benchmark_for_Zero_Shot_Text_Cl.pdf` | BTZSC Zero-Shot Benchmark Across Encoders/LLMs | [2603.11991](https://arxiv.org/abs/2603.11991) | `REFERENCE` | Benchmark evaluasi cross-encoder vs decoder-only instruction-tuned LLMs. |
| 45 | `2603.25005_Error_Understanding_in_Program_Code_With.pdf` | Error Understanding in Code With Hybrid LLM-DL | [2603.25005](https://arxiv.org/abs/2603.25005) | `REFERENCE` | Integrasi representasi encoder-decoder untuk multi-label classification. |
| 46 | `nonarxiv_Understanding_Gut_Brain_Interplay_in_Sci.pdf` | Hybrid Seq2Seq & LLM Reasoning in Scientific Literature | [CEUR-WS 4038](https://ceur-ws.org/Vol-4038/paper_27.pdf) | `REFERENCE` | Metodologi integrasi generative sequence-to-sequence dan token classification. |
| 47 | `2506.21812_Towards_transparent_ai__A_survey_on_expl.pdf` | Towards Transparent AI: Survey on Explainable LLMs | [2506.21812](https://arxiv.org/abs/2506.21812) | `REFERENCE` | Taksonomi XAI untuk komponen representasi encoder-decoder. |

---

## 🎯 3. Ringkasan Hubungan Versi & Rekomendasi Pipeline

1. **Pipeline V7 Baseline (`notebooks/working-molab-v7-combined-unsloth.py`)**:
   - Berhasil mengintegrasikan: `T5Gemma 2 TR` (Base), `SigLIP 400M Grafting`, `Token Cleaning` (Logit Masking), `Task Arithmetic` (FFN/Norm Steering), `UL2R Denoising`, dan `LoRA Targeting`.
2. **Pipeline V8 (`notebooks/working-molab-v8-combined-unsloth.py`)**:
   - Menambahkan: **DeVec SVD Subspace Filtering** ($\tau=0.85$), **OrScaleLM Optimizer** (Layer-wise Trust Ratio Scaling), **TLPO Regularization** (Indonesian Language Confusion Penalty), **MTO Data Formatter**, dan **Layer-Aware Ramp-Up** Steering.
3. **Pipeline V9 Roadmap**:
   - Fokus pada: **Knowledge Distillation Cross-Architecture** (`Return of Encoder` & `Enc-Dec Gemma`), **GRUN Unlearning Framework** (Safety guardrail), dan **Cross-Model Subspace Transfer**.
4. **Pipeline V10 SOTA Horizon**:
   - Fokus pada: **INTRA Reverse-QWK** (30x KV cache reduction), **Cassandra Speculative Decoding** (2.5x inference speedup), dan **Stacked Context** (128K context window).
