# Encoder-Decoder LLM Papers 2025–2026 (Kurasi)

**Tanggal:** 2026-08-22 · **Metode:** sweep 22 query arXiv API (metadata: judul+abstrak) → 368 unik → filter 2025–2026 + buang speech/medicine → 181 → **kurasi manual ke encoder-decoder LLM**. Full harvest mentah: `2026-08-22_encdec-llm-search-harvest.md`.

> ⚠️ Catatan jujur: arXiv API (dan markxiv) hanya mencari **metadata** (judul+abstrak), bukan isi penuh. Daftar ini hasil kurasi manual dari judul+abstrak; verifikasi isi penuh perlu `convert_paper` per paper.

---

## Tier A — Inti (arsitektur / pretraining / scaling / efisiensi / steering)

| ID | Tanggal | Judul | Relevansi ke proyek |
|---|---|---|---|
| **2512.14856** | 2025-12 | **T5Gemma 2: Seeing, Reading, and Understanding Longer** | ⭐⭐⭐ **Paper base-model kamu!** Multimodal + long-context T5Gemma; tied embeddings + **merged attention [X;H]** (persis yang dirujuk kode v8). UL2 adaptation dari Gemma 3. Release 270M/1B/4B-4B. |
| **2504.06225** | 2025-04 | **Encoder-Decoder Gemma** | ⭐⭐⭐ Adaptasi decoder-only → encoder-decoder (Gemma 2B/9B), +7% post-instruction-tuning. Fondasi metodologi kamu. |
| **2510.26622** | 2025-10 | **Encoder-Decoder or Decoder-Only? Revisiting RedLLM** | ⭐⭐⭐ Studi scaling E-D vs decoder-only 150M→8B. |
| **2501.16273** | 2025-01 | **Return of the Encoder (SLMs)** | ⭐⭐ Efisiensi E-D di edge (47% lower FLT, 4.7× throughput) + framework KD. |
| **2512.03803** | 2025-12 | **Enhancing Instruction-Following in Seq2Seq: DoLA for T5** | ⭐⭐ **Activation steering** di decoder T5 (Mem Trap 52→99.7%) — sangat nyambung ke Phase 0.5 DeVec steering kamu. |
| **2603.16413** | 2026-03 | **Trained Persistent Memory for Frozen Encoder-Decoder LLMs (6 methods)** | ⭐⭐ Persistent latent memory di frozen Flan-T5 — relevan ke arah "memory/adaptasi" encoder-decoder. |
| **2606.24841** | 2026-06 | **Matching Tasks to Objectives (MTO)** | ⭐⭐ Sudah dibahas (dipakai di v8). |
| **2510.22852** | 2025-10 | **Encoder-Decoder Diffusion LMs (E2D2)** | ⭐ Discrete diffusion E-D, inferensi lebih cepat. |
| **2605.15976** | 2026-05 | **Reference-Free RL FT for MT: GRPO on NLLB seq2seq** | ⭐ RL (GRPO) untuk E-D seq2seq — analog Phase 2 (ORPO) kamu. |
| **2603.17512** | 2026-03 | **Language on Demand (XBridge)** | ⭐ Komposisi encoder-LLM-decoder untuk multibahasa. |
| **2604.05551** | 2026-04 | **FastDiSS: Diffusion LM pada seq2seq generation** | Diffusion LM seq2seq. |
| **2601.17602** | 2026-01 | **Understanding Transformer E-D Representations via Bernoulli Dropout** | Interpretabilitas representasi E-D. |
| **2502.05233** | 2025-02 | **Efficient Knowledge Feeding: Integrated Encoder-Decoder** | Retrieval+generation terintegrasi di satu E-D. |
| **2502.12304** | 2025-02 | **Warmup Generations** | Inisialisasi unsupervised untuk seq2seq SFT. |
| **2604.11687** | 2026-04 | **E-D vs Decoder-Only Transformers for Style Transfer** | Head-to-head E-D vs decoder-only. |
| **2606.30336** | 2026-06 | **FlexTab: Encoder-Decoder for In-Context Tabular** | E-D untuk ICL data tabular. |

## Tier B — Terapan (MT / multibahasa / seq2seq task)

**Machine Translation & multilingual:**
- `2603.16309` Omnilingual MT (1,600 bahasa) · `2603.22186` Doc-Level MT via two-stage LLM adaptation · `2511.08145` Still Not There (LLM vs seq2seq poetry→prose) · `2507.21568` Multi-Hypothesis Distillation NMT · `2501.13927` CRPO (preference opt. MT) · `2603.27938` Top-down string-to-dependency NMT · `2510.08870` QE Reranking doc-level · `2510.27337` TransAlign (MT encoder = word aligner) · `2506.13044` Just Go Parallel · `2602.03551` Typological features MT · `2603.02258` NLLB-200 multilingual geometry · `2603.29345` Open MT Esperanto · `2505.11421` Bahnaric-Vietnamese · `2512.13552` PrahokBART (Khmer) · `2505.17102` BanglaByT5.

**Seq2seq task spesifik:**
- `2603.23949` Argument Mining text-to-text · `2510.16604` Constituency parsing seq2seq · `2508.08514` DeCAL tokenwise compression (E-D denoising) · `2602.12146` Seq2Seq2Seq lossless compression · `2503.05935` DETQUS summarization · `2501.10328` BoK dialogue loss · `2508.18780` GEC + RL · `2508.10366` / `2508.10369` / `2508.07866` Cross-lingual ABSA seq2seq · `2606.15883` Koshur diacritizer · `2506.12843` Chatbot text seq2seq · `2502.07490` Mask-Enhanced AR prediction.

---

## Dibuang (lolos filter tanggal tapi bukan LLM / bukan topik)
Sign language (Text2Sign, POSESTITCH-SLT, SignBart, multitask SLT), stock-prediction GAN, MIMO beamforming, wind health monitoring, quantum RNN, PDE neural operator, agricultural ecosystem, remote-sensing captioning, vision-only/image-editing, text-to-image, watermarking, GUI agents, time-series, dsb.

## Rekomendasi bacaan prioritas untuk riset kamu
1. **2512.14856** (T5Gemma 2) — wajib, ini kertas model kamu.
2. **2512.03803** (DoLA-T5 steering) — paling nyambung ke Phase 0.5 steering v8.
3. **2504.06225 + 2510.26622** — fondasi "kenapa E-D" (referensi di dokumen).
4. **2603.16413** (persistent memory) — arah eksperimen baru yang menarik.
