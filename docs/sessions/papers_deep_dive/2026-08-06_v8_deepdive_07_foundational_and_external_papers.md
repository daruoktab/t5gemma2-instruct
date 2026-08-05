# 🔬 Deep-Dive 07 — Foundational & External Papers in `docs/paper/` (V8 & Foundational Archive)

**Tanggal Analysis:** 5 Agustus 2026  
**Analisis Visual PDF & Berkas Dokumentasi:**
- `2512.10561` (Causal Encoders): [pdfs/CAUSAL_ENCODERS_2512.10561.pdf](file:///d:/Codings/unsloth-porto/t5-gemma-2/instruct/docs/sessions/papers_deep_dive/pdfs/CAUSAL_ENCODERS_2512.10561.pdf)
- `2501.16273` (Return of the Encoder): [docs/paper/2501.16273_Return_of_the_Encoder_Microsoft_2025.pdf](file:///d:/Codings/unsloth-porto/t5-gemma-2/instruct/docs/paper/2501.16273_Return_of_the_Encoder_Microsoft_2025.pdf)
- `2502.01968` (Token Cleaning): [docs/paper/2502.01968v2_Token_Cleaning_Pang2025.pdf](file:///d:/Codings/unsloth-porto/t5-gemma-2/instruct/docs/paper/2502.01968v2_Token_Cleaning_Pang2025.pdf)
- `2503.19786` (Gemma 3 Technical Report): [docs/paper/2503.19786_Gemma_3_Technical_Report_DeepMind_2025.pdf](file:///d:/Codings/unsloth-porto/t5-gemma-2/instruct/docs/paper/2503.19786_Gemma_3_Technical_Report_DeepMind_2025.pdf)
- `1910.10683` (T5 - Raffel 2020) & `1907.12461` (Warm-Starting Enc-Dec - Rothe 2019)
- `2210.11416` (FLAN Instruction Scaling - Chung 2022)
- `2212.04089` (Task Arithmetic - Ilharco 2023) & `2502.20186` (Layer-Aware Task Arithmetic 2025)
- `2503.02656` (Gemma Encoder) & `2504.06225` (Encoder-Decoder Gemma)
- `2505.12021` (Cross-Model Transfer Task Vectors) & `2511.16147` (TS-PEFT) & `2602.01227` (Token Priority)

---

## 1. Causal Reasoning Favors Encoders: On the Limits of Decoder-Only Models (Microsoft & IIT, Des 2025)
**arXiv:** [2512.10561](https://arxiv.org/abs/2512.10561)

### A. Masalah & Temuan Utama
In-Context Learning (ICL) pada model Decoder-only bersifat rapuh terhadap pergeseran distribusi (*distributional shifts*) dan rentan terjebak pada korelasi leksikal semu. Sebaliknya, model berbasis Enkoder dan Encoder-Decoder memiliki kemampuan proyeksi konteks penuh ke subruang laten yang membuat penalaran kausal berantai (*multi-hop conjunctive reasoning*) jauh lebih stabil.

### B. Implikasi untuk T5-Gemma-2
Mengukuhkan pilihan arsitektur Seq2Seq (Encoder-Decoder) T5Gemma 2 sebagai fondasi yang jauh lebih tangguh untuk tugas penalaran instruksi terstruktur daripada model Decoder-only murni.

---

## 2. Return of the Encoder: Maximizing Parameter Efficiency for SLMs (Microsoft, Jan 2025)
**arXiv:** [2501.16273](https://arxiv.org/abs/2501.16273)

### A. Keunggulan Efisiensi Arsitektur Enc-Dec pada Small Language Models (SLMs)
- Model Encoder-Decoder $\le 1$B parameter menghasilkan **47% lebih rendah first-token latency** dan **4.7x throughput lebih tinggi** dibandingkan model Decoder-only pada perangkat edge (GPU/CPU/NPU).
- Hal ini disebabkan oleh pemrosesan input satu kali oleh Enkoder (*one-time input processing*) tanpa pembengkakan KV cache pada token input.
- **Asymmetric Parameter Allocation:** Alokasi layer $2/3$ Enkoder dan $1/3$ Dekoder konsisten mengalahkan alokasi seimbang $1/2 - 1/2$.

---

## 3. Token Cleaning / Logit Masking (Pang et al., Feb 2025)
**arXiv:** [2502.01968](https://arxiv.org/abs/2502.01968)

### A. Aplikasi di V7 & Ekstensi di V8
- Menekan *unused tokens* (`<unused0>` s.d. `<unused98>` & `<unused100>` s.d. `<unused6241>`) pada keluaran logit dekoder agar sampel generasi tidak pernah melompat ke token cadangan yang tidak terdefinisi.
- Di V8, logit masking dipadukan dengan mask task prefix spesifik (`<unused1>` s.d. `<unused6>`).

---

## 4. Gemma 3 Technical Report (Google DeepMind, Mar 2025)
**arXiv:** [2503.19786](https://arxiv.org/abs/2503.19786)

### A. Integrasi Vision Encoder SigLIP 400M
- Ekstraksi 256 soft tokens visual dari gambar resolusi tinggi disuntikkan ke Enkoder T5Gemma pada posisi `<image_soft_token>` (`256001`).

---

## 5. Task Arithmetic & Layer-Aware Task Vectors (Ilharco 2023, 2025)
**arXiv:** [2212.04089](https://arxiv.org/abs/2212.04089) & [2502.20186](https://arxiv.org/abs/2502.20186)

### A. Landasan Parameter Steerability
- Mengubah perilaku model melalui operasi aritmatika linier pada delta weight ($\Delta W = W_{\text{finetuned}} - W_{\text{base}}$).
- V8 mengadopsi prinsip ini yang dipercanggih oleh DeVec (SVD decomposition) dan OrScale.

---

## 6. TS-PEFT, Token Priority & Cross-Model Transfer (2025–2026)
**arXiv:** [2511.16147](https://arxiv.org/abs/2511.16147), [2602.01227](https://arxiv.org/abs/2602.01227), [2505.12021](https://arxiv.org/abs/2505.12021)

### A. Rangkuman Teknis
- **TS-PEFT:** Parameter-efficient fine-tuning berskala dua tahap untuk adapter LoRA.
- **Token Priority:** Pengalokasian prioritas bobot gradien berdasarkan tingkat kepentingan token masukan.
- **Cross-Model Transfer:** Transfer task vector antar arsitektur berbasis subruang kemiripan kosinus.
