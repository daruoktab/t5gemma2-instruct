# 📄 Review 02 — Paper Level Strategi & Komposisi

**Tanggal review:** 2026-08-05 · **Sumber:** arXiv API (diverifikasi eksis ✅)
**Relevansi:** Pipeline T5Gemma-2 4B chatbot Indonesia (v7) — ide level
strategi/komposisi/kegunaan yang bisa ditempel di flow training.

---

## 0. Paper Base — T5Gemma 2: Seeing, Reading, and Understanding Longer
**arXiv:** [2512.14856](https://arxiv.org/abs/2512.14856) · Des 2025 · Google DeepMind

**Isi:** Generasi berikutnya keluarga T5Gemma (encoder-decoder ringan open-weight).
Diadaptasi dari pruned decoder-only (Gemma 3) ke encoder-decoder **via UL2** — resep yang
sama dengan T5Gemma generasi 1. Ukuran: 270M, 1B, 4B. Inovasi kunci: **tied word
embeddings** (embedding dibagi encoder & decoder) dan **merged attention** (decoder
self-attention & cross-attention digabung dalam satu module). Kuat di multilingual,
multimodal, long-context. Post-training (fine-tuning) signifikan lebih baik dari
counterpart Gemma 3 decoder-only.

**Relevansi:** Model base project ini. Paham merged attention = kunci kenapa steering
hanya boleh menyentuh FFN (α_QO=α_KV=0.0 di v7 — proyeksi joint [X;H] tidak boleh
diganggu). Tied embeddings berarti encoder & decoder berbagi ruang representasi → efek
steering/grafting di decoder ikut terasa di encoder.

---

## 1. MTO — Matching Tasks to Objectives
**arXiv:** [2606.24841](https://arxiv.org/abs/2606.24841) · Jun 2026

**Isi:** Strategi fine-tuning & prompt-tuning untuk encoder-decoder PLM: **cocokkan
objective pre-training dengan objective fine-tuning** (bukan asal fine-tune). Analisis
awal menyebut klaim gain besar di few-shot (120%+) — angka detail belum diverifikasi dari
abstract, perlu baca full paper.

**Cara tempel:** Pipeline sudah punya `task_prefix_mapping` (summarize/translate/ner/qa/
paraphrase/general_chat). Gunakan kerangka MTO untuk mengaudit: apakah format prefix
& template tiap task selaras dengan objective UL2 yang sudah dikenal model? Task yang
dekat dengan denoising/span-corruption (mis. qa) mungkin lebih baik diformat ulang.
Bisa jadi penentu apakah `general_chat` perlu prefix khusus.

---

## 2. XBridge — Composing LLMs with Encoder-Decoder Translation Models
**arXiv:** [2603.17512](https://arxiv.org/abs/2603.17512) · Mar 2026
*(Judul asli: "Language on Demand, Knowledge at Core: Composing LLMs with Encoder-Decoder
Translation Models for Extensible Multilinguality")*

**Isi:** Komposisi LLM (knowledge di inti) + model terjemahan encoder-decoder (bahasa di
permintaan) untuk multilinguality yang extensible — lewat cross-model mapping layers +
optimal transport alignment. Menambah bahasa baru tanpa retrain LLM inti.

**Cara tempel:** T5Gemma-2 sudah multilingual; tapi kalau nanti perlu bahasa lokal
tambahan (Jawa, Sunda, dll. — ingat CultureTalk-ID mencakup 11 bahasa), pendekatan
XBridge bisa menambah bahasa dengan plug-in model terjemahan tanpa fine-tune ulang penuh.

---

## 3. FlexTab — Flexible Encoder-Decoder for Tabular ICL
**arXiv:** [2606.30336](https://arxiv.org/abs/2606.30336) · Jun 2026
*(Judul asli: "A Flexible Encoder-Decoder Architecture for In-Context Learning Across
Diverse **Tabular** Tasks")*

> ⚠️ **Koreksi (2026-08-05):** Deskripsi di analisis awal ("shared encoder + task-specific
> decoder heads untuk chatbot multi-purpose") **tidak akurat** — paper ini untuk **tabular
> ICL** (in-context learning pada tugas tabel), bukan chatbot multi-task. Tidak relevan
> langsung untuk pipeline; dicatat agar tidak salah pakai.

---

## 4. T5Gemma-TTS Technical Report
**arXiv:** [2604.01760](https://arxiv.org/abs/2604.01760) · Apr 2026

**Isi:** Bangun sistem TTS di atas backbone T5Gemma dengan **enhanced cross-attention** —
PM-RoPE (position/instruction signal) di lapisan cross-attention. Motivasinya: arsitektur
decoder-only memperlakukan input teks sebagai prefix yang bersaing dengan audio yang
tumbuh untuk kapasitas posisi → text conditioning melemah; encoder-decoder (T5Gemma)
menyelesaikan ini.

**Cara tempel:** Teknik PM-RoPE (menyuntikkan sinyal progress/instruction ke
cross-attention) bisa diadaptasi untuk **response coherence**: inject sinyal tahap
percakapan (turn ke-n, instruksi sistem) ke lapisan cross-attention decoder → koherensi
multi-turn lebih baik. Ini enhancement arsitektur, bukan sekadar prompt.

---

## 5. Enc-Dec vs Dec-Only untuk AI-to-Human Text Style Transfer
**arXiv:** [2604.11687](https://arxiv.org/abs/2604.11687) · Apr 2026
*(Judul asli: "Please Make it Sound like Human: Encoder-Decoder vs. Decoder-Only
Transformers for AI-to-Human Text Style Transfer")*

**Isi:** BART-large (encoder-decoder) mencapai kualitas lebih tinggi dengan **17× lebih
sedikit parameter** dari Mistral-7B untuk style transfer teks.

**Relevansi:** Validasi keputusan arsitektur: untuk tugas generasi terstruktur/restyling,
encoder-decoder kecil bisa mengalahkan decoder-only besar — mendukung strategi
efisiensi biaya pipeline.

---

## 6. Pre-Training on Software Engineering Texts — Domain Adaptation
**arXiv:** [2607.06613](https://arxiv.org/abs/2607.06613) · Jul 2026

**Isi:** Studi perbandingan **continual pre-training (CPT)** vs **pre-training from
scratch (PTS)** untuk adaptasi domain pada encoder vs decoder LMs, dengan ukuran
general-language understanding sebagai kontrol.

**Cara tempel:** Kalau chatbot diarahkan ke domain spesifik (customer service, kesehatan,
legal), paper ini memberi guidance strategi adaptasi domain untuk encoder-decoder —
termasuk apakah encoder perlu ikut di-CPT atau cukup decoder. Kombinasikan dengan ide
UL2R (review-03 #2): CPT dengan objective UL2 (denoising) di korpus domain Indonesia.

---

## 7. Causal Reasoning Favors Encoders
**arXiv:** [2512.10561](https://arxiv.org/abs/2512.10561) · Des 2025
*(Judul asli: "Causal Reasoning Favors Encoders: On The Limits of Decoder-Only Models")*

**Isi:** Encoder-decoder lebih robust untuk causal reasoning; fine-tuned enc-dec
generalisasi lebih baik terhadap distributional shifts dibanding decoder-only.

**Relevansi:** Chatbot sering butuh reasoning (follow-up question, context tracking) →
validasi mengapa arsitektur T5Gemma-2 (enc-dec) pilihan yang masuk akal untuk use case ini.
Bonus: dengan merged attention, encoder bisa "membaca ulang" seluruh konteks multi-turn
di tiap langkah decoding — keunggulan alami untuk chatbot.

---

## Ringkasan

| # | Paper | Tipe | Relevansi pipeline |
|---|-------|------|--------------------|
| 0 | T5Gemma 2 [2512.14856] | Base | Memahami merged attention & tied embeddings (dasar keputusan steering) |
| 1 | MTO [2606.24841] | Strategi | Audit task prefix & template vs objective UL2 |
| 2 | XBridge [2603.17512] | Komposisi | Ekstensi bahasa lokal tanpa retrain (opsional) |
| 3 | FlexTab [2606.30336] | — | ❌ Tidak relevan (tabular ICL) — dikoreksi |
| 4 | T5Gemma-TTS [2604.01760] | Arsitektur | PM-RoPE → koherensi multi-turn (enhancement) |
| 5 | Style transfer [2604.11687] | Validasi | Efisiensi enc-dec vs dec-only |
| 6 | SE domain [2607.06613] | Strategi | Guidance CPT untuk domain adaptation |
| 7 | Causal encoders [2512.10561] | Validasi | Pembenaran arsitektur enc-dec untuk chatbot |
