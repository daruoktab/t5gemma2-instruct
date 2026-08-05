# ✅ Verifikasi 26 Paper di arXiv

**Tanggal verifikasi:** 2026-08-05 · **Metode:** API `https://export.arxiv.org/api/query`
(HTTP 200, respon Atom di-parse) — `id_list=` untuk 17 paper, pencarian query untuk sisanya.

**Legenda status:**
- 🟢 **Langsung** = ID di-fetch via `id_list` → abstract utuh diterima
- 🔵 **Search** = muncul di hasil pencarian query (title + abstract terlihat)

---

## A. Paper Level Internal Training (10)

| # | ID | Judul (ringkas) | Tanggal arXiv | Status |
|---|----|-----------------|---------------|--------|
| 1 | [2604.26553](https://arxiv.org/abs/2604.26553) | TLPO: Token-Level Policy Optimization (language confusion) | Apr 2026 | 🟢 Langsung |
| 2 | [2607.16169](https://arxiv.org/abs/2607.16169) | When Does Muon Help Agentic Reinforcement Learning? | Jul 2026 (v4) | 🟢 Langsung |
| 3 | [2605.07815](https://arxiv.org/abs/2605.07815) | OrScale: Layer-Wise Trust-Ratio Scaling | Mei 2026 | 🟢 Langsung |
| 4 | [2512.22511](https://arxiv.org/abs/2512.22511) | Decomposing Task Vectors for Refined Model Editing | Des 2025 | 🟢 Langsung |
| 5 | [2605.03780](https://arxiv.org/abs/2605.03780) | Task Vector Geometry — Dual Modes of Task Inference | Mei 2026 | 🟢 Langsung |
| 6 | [2607.16821](https://arxiv.org/abs/2607.16821) | First-Order Predictable but Pairwise Fragile | Jul 2026 | 🟢 Langsung |
| 7 | [2606.10929](https://arxiv.org/abs/2606.10929) | Recoverable but Not Stationary (local linear structures) | Jun 2026 | 🟢 Langsung |
| 8 | [2601.12639](https://arxiv.org/abs/2601.12639) | Objective Matters: Fine-Tuning Objectives & Safety | Jan 2026 | 🟢 Langsung |
| 9 | [2606.09850](https://arxiv.org/abs/2606.09850) | Mechanistic Analysis of Alignment Algorithms | Jun 2026 | 🟢 Langsung |
| 10 | [2607.21016](https://arxiv.org/abs/2607.21016) | CultureTalk-ID: Cultural Commonsense Indonesia | Jul 2026 | 🟢 Langsung |

## B. Paper Level Strategi & Komposisi (8)

| # | ID | Judul (ringkas) | Tanggal arXiv | Status |
|---|----|-----------------|---------------|--------|
| 11 | [2512.14856](https://arxiv.org/abs/2512.14856) | T5Gemma 2: Seeing, Reading, and Understanding Longer (base) | Des 2025 | 🔵 Search |
| 12 | [2606.24841](https://arxiv.org/abs/2606.24841) | Matching Tasks to Objectives (MTO) — enc-dec PLMs | Jun 2026 | 🟢 Langsung |
| 13 | [2603.17512](https://arxiv.org/abs/2603.17512) | Composing LLMs with Enc-Dec Translation Models (XBridge) | Mar 2026 | 🟢 Langsung |
| 14 | [2606.30336](https://arxiv.org/abs/2606.30336) | FlexTab — Enc-Dec for Tabular In-Context Learning | Jun 2026 | 🟢 Langsung |
| 15 | [2604.01760](https://arxiv.org/abs/2604.01760) | T5Gemma-TTS Technical Report (PM-RoPE) | Apr 2026 | 🟢 Langsung |
| 16 | [2604.11687](https://arxiv.org/abs/2604.11687) | Enc-Dec vs Dec-Only for AI-to-Human Style Transfer | Apr 2026 | 🟢 Langsung |
| 17 | [2607.06613](https://arxiv.org/abs/2607.06613) | Pre-Training on SE Texts (domain adaptation CPT vs PTS) | Jul 2026 | 🟢 Langsung |
| 18 | [2512.10561](https://arxiv.org/abs/2512.10561) | Causal Reasoning Favors Encoders | Des 2025 | 🟢 Langsung |

## C. Paper Ide Baru Terapan (8)

| # | ID | Judul (ringkas) | Tanggal arXiv | Status |
|---|----|-----------------|---------------|--------|
| 19 | [2605.05806](https://arxiv.org/abs/2605.05806) | INTRA: Retrieval from Within (intrinsic retrieval) | Mei 2026 | 🔵 Search |
| 20 | [2210.11399](https://arxiv.org/abs/2210.11399) | UL2R: Transcending Scaling Laws with 0.1% Extra Compute | Okt 2022 | 🔵 Search |
| 21 | [2606.20911](https://arxiv.org/abs/2606.20911) | Latent Personal Memory — dynamic soft prompts | Jun 2026 | 🔵 Search |
| 22 | [2308.07269](https://arxiv.org/abs/2308.07269) | EasyEdit: Knowledge Editing Framework | Agu 2023 | 🔵 Search |
| 23 | [2607.25583](https://arxiv.org/abs/2607.25583) | LoRA Rank / Modules / Quantization trade-offs | Jul 2026 | 🔵 Search |
| 24 | [2603.04759](https://arxiv.org/abs/2603.04759) | Stacked from One — context window extension | Mar 2026 | 🔵 Search |
| 25 | [2605.26558](https://arxiv.org/abs/2605.26558) | Cassandra: Self-Speculative Decoding at Edge | Mei 2026 | 🔵 Search |
| 26 | [2607.21356](https://arxiv.org/abs/2607.21356) | Emergent Misalignment — persona subspace | Jul 2026 | 🔵 Search |

---

## Catatan Metodologi

1. **17 paper status 🟢** diverifikasi dengan fetch `id_list` — semua mengembalikan entry
   lengkap (ID dengan suffix versi, mis. `2607.16169v4`).
2. **9 paper status 🔵** muncul di hasil pencarian query (dengan `sortBy=submittedDate&
   sortOrder=descending`) — title + abstract terlihat langsung di respon.
3. **Koreksi yang terekam:** FlexTab [2606.30336] awalnya dideskripsikan salah (multi-task
   chatbot) — judul asli mengonfirmasi fokus **tabular ICL** (lihat review-02 #3).
4. Kecuali T5Gemma 2 base [2512.14856], semua paper ini **tidak** tersimpan sebagai PDF
   di `docs/paper/` — file PDF lama di folder itu adalah koleksi sebelum sesi ini.
