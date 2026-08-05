# 📄 Review 03 — Paper Ide Baru yang Bisa Ditempel di Flow Training

**Tanggal review:** 2026-08-05 · **Sumber:** Ditemukan via pencarian arXiv (sort by
submittedDate desc) · ID paper muncul di hasil pencarian → **eksis di arXiv**
**Relevansi:** Pipeline T5Gemma-2 4B chatbot Indonesia (v7) — ide level
strategi/komposisi/kegunaan di luar teknik internal training.

---

## 1. 🔥 INTRA — Retrieval from Within (Intrinsic Retrieval)
**arXiv:** [2605.05806](https://arxiv.org/abs/2605.05806) · Mei 2026

**Esensi (dari abstract):** RAG biasanya memperlakukan retrieval dan generation sebagai
sistem terpisah. INTRA bertanya: bisakah **attention-based encoder-decoder retrieve
langsung dari representasi internalnya sendiri**? (INTrinsic Retrieval via Attention).

**Cara tempel:** T5Gemma-2 adalah encoder-decoder — arsitektur yang persis ditarget
paper ini. Setelah fine-tuning, chatbot bisa diberi kemampuan **knowledge-access tanpa
infra retriever eksternal** (tanpa BM25/embedding model): cukup masukkan dokumen
(FAQ, regulasi, SOP) ke konteks encoder, biarkan attention internal yang menemukan
jawabannya. Ini upgrade kegunaan murni untuk chatbot Indonesia.

**Status:** Eksperimen — perlu baca full paper untuk detail training/objective.

---

## 2. 🔥 UL2R — Transcending Scaling Laws with 0.1% Extra Compute
**arXiv:** [2210.11399](https://arxiv.org/abs/2210.11399) · Okt 2022

**Esensi (dari abstract):** UL2R: metode yang **meningkatkan model LLM yang sudah ada dan
kurva scaling-nya dengan tambahan compute sangat kecil (~0.1%)** — continued pre-training
dengan objective denoising UL2.

**Kenapa relevan:** T5Gemma-2 dibangun via **adaptasi UL2** — objective denoising UL2
adalah bahasa yang sudah dikenal model. UL2R = lanjutkan pre-training singkat dengan
objective yang identik, super murah (0.1% compute). Ini konsisten dengan filosofi MTO
(review-02 #1): cocokkan objective.

**Cara tempel — "Phase 0.75":** sisipkan **UL2R Indonesian Denoising CPT** antara
Phase 0.5 (steering) dan Phase 1 (SFT):
- Korpus: teks umum Bahasa Indonesia (bukan cuma chat).
- Objective: span-corruption UL2 (sama dengan pre-training T5Gemma-2).
- Budget: ~0.1% dari compute SFT.
- Efek yang diharapkan: fondasi bahasa Indonesia lebih kuat *sebelum* dibentuk jadi
  chatbot → SFT lebih bersih, capaian lebih tinggi.

**Prioritas tinggi** — ini inovasi training terbesar berikutnya yang paling "nempel"
di flow karena konsisten dengan UL2/MTO.

---

## 3. 🧠 Latent Personal Memory (LPM) — Memory sebagai Dynamic Soft Prompts
**arXiv:** [2606.20911](https://arxiv.org/abs/2606.20911) · Jun 2026

**Esensi (dari abstract):** Personalisasi LLM butuh encoding pola perilaku user
jangka panjang yang efisien & skalabel, kompatibel dengan **frozen base model**. LPM:
representasikan personal memory sebagai **dynamic soft prompts**.

**Cara tempel:** Pasca-training — tambah lapisan personalisasi per-user di atas model
final (yang tetap frozen): nama user, preferensi, riwayat → dynamic soft prompts.
Chatbot Indonesia bisa ingat preferensi tiap user tanpa retraining. Murni "kegunaan"
produk. Effort sedang.

---

## 4. 🔧 EasyEdit — Knowledge Editing Framework
**arXiv:** [2308.07269](https://arxiv.org/abs/2308.07269) · Agu 2023 (framework matang)

**Esensi:** Framework knowledge editing untuk LLM (metode ROME/MEMIT-style dkk.) —
memperbaiki fakta salah/outdated **langsung di weights, tanpa retrain**.

**Cara tempel:** Setelah fine-tuning, jika ada fakta salah di model (mis. informasi
kadaluarsa), edit langsung via EasyEdit — mendukung model keluarga T5. Pelengkap
sempurna: SFT/ORPO untuk *behavior*, EasyEdit untuk *knowledge*. Effort rendah,
utility tinggi.

---

## 5. 📏 LoRA Rank, Target Modules & Quantization Trade-offs
**arXiv:** [2607.25583](https://arxiv.org/abs/2607.25583) · Jul 2026

**Esensi (dari abstract):** Studi **terkontrol** LoRA rank × target modules × quantization
untuk adaptasi di budget komputasi ketat — pada model 60M (text-to-SQL) agar design space
bisa dieksplorasi murah, dengan insight untuk model miliaran parameter.

**Cara tempel — validasi `LORA_RANK=256 / LORA_ALPHA=512`:** rank 256 itu agresif untuk
model 4B. Pakai metodologi paper untuk A/B: rank 64/128/256 dengan alpha ratio sama,
di satu run terpisah, ukur delta eval (bukan cuma loss). Kalau 64/128 menyamai 256,
hemat VRAM & mempercepat training. Effort rendah, memperkuat klaim pipeline.

---

## 6. ⚡ Stacked from One — Multi-Scale Self-Injection Context Extension
**arXiv:** [2603.04759](https://arxiv.org/abs/2603.04759) · Mar 2026

**Esensi (dari abstract):** Context window LLM terbatas; continual pre-training panjang
untuk ekstensi konteks mahal. Solusi: **multi-scale self-injection** — injeksi
representasi multi-skala dari lapisan yang sama untuk memperpanjang konteks, jauh
lebih murah dari CPT.

**Cara tempel:** Extend `MAX_SOURCE_LENGTH=16384` (v7) ke 32K+ tanpa CPT mahal —
berguna untuk percakapan multi-turn panjang + dokumen. Effort sedang, opsional.

---

## 7. 🚀 Cassandra — Self-Speculative Decoding di Edge
**arXiv:** [2605.26558](https://arxiv.org/abs/2605.26558) · Mei 2026

**Esensi (dari abstract):** Speculative decoding lossless untuk akselerasi LLM; pendekatan
berbasis draf model terpisah & approximation menurunkan akurasi. Cassandra:
**self-speculative decoding** — lossless, tanpa draft model terpisah, untuk reasoning LLM
di edge (beban decode-stage).

**Cara tempel:** Sisi deployment — model 4B (BF16 ~15GB / 4bit ~5GB) bisa diakselerasi
saat serve tanpa kehilangan kualitas & tanpa model draf tambahan. Utility langsung untuk
produksi. Effort sedang (baca detail teknis untuk enc-dec).

---

## 8. 🛡️ Emergent Misalignment — Persona Subspace
**arXiv:** [2607.21356](https://arxiv.org/abs/2607.21356) · Jul 2026

**Esensi (dari abstract):** Fine-tuning model aligned pada aliran data sempit (nasihat
buruk) bisa membuat model **broadly misaligned** pada pertanyaan yang tidak terkait —
emergent misalignment. Penyebabnya: training sempit **merekrut pre-existing persona
subspace**.

**Kenapa penting untuk pipeline ini:** Nyambung **langsung** ke Phase 0.5 — pipeline
*sengaja* menyuntikkan persona IT (Gemma3-IT − Gemma3-Base) via task vector. Paper ini
menunjukkan persona adalah subspace yang bisa direkrut → steering bekerja lewat mekanisme
nyata, tapi juga berarti **wajib monitoring safety**: bandingkan output model final vs
base di prompt safety-sensitive. Satu baris eval tambahan di
`VisionSampleGenerationCallback`.

---

## Ringkasan

| # | Paper | Tipe | Effort | Impact |
|---|-------|------|--------|--------|
| 1 | INTRA [2605.05806] | Kemampuan baru | Eksperimen | 🔴 Tinggi (beda dari RAG biasa) |
| 2 | UL2R [2210.11399] | Training strategy | Sedang | 🔴 Tinggi (fondasi bahasa) |
| 3 | LPM [2606.20911] | Pasca-training | Sedang | 🟡 Utility (personalisasi) |
| 4 | EasyEdit [2308.07269] | Pasca-training | Rendah | 🟡 Utility (edit knowledge) |
| 5 | LoRA rank [2607.25583] | Validasi | Rendah | 🟡 Validasi rank=256 |
| 6 | Stacked from One [2603.04759] | Arsitektur | Sedang | 🟢 Opsional (konteks 32K) |
| 7 | Cassandra [2605.26558] | Deployment | Sedang | 🟢 Opsional (kecepatan serve) |
| 8 | Misalignment [2607.21356] | Safety | Sangat rendah | 🟡 Penting (monitoring) |
