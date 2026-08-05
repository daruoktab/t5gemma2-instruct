# 📝 Log Sesi 01 — Analisis Pipeline & Paper Review

**Tanggal sesi:** Rabu, 2026-08-05
**Durasi dicatat:** Satu sesi obrolan (Hermes Agent desktop)
**Model aktif:** deepseek-v4-flash (provider: deepseek)
**Topik:** Riset paper untuk pipeline fine-tuning T5Gemma-2 4B chatbot multimodal Bahasa Indonesia
**Status:** ✅ Semua paper diverifikasi via arXiv API (`export.arxiv.org`) tanggal 2026-08-05

---

## 0. Konteks Project (Input User)

User sedang membangun **chatbot multimodal Bahasa Indonesia** dengan fine-tuning
`google/t5gemma-2-4b-4b` (encoder-decoder, adaptasi UL2 dari Gemma 3; merged attention;
tied word embeddings). Pipeline berlapis 5 fase:

```
Phase 0.5  Task-vector steering   (Gemma3-IT − Gemma3-Base → decoder FFN, layer-wise α)
Phase 1.5  Vision grafting        (SigLIP + projector, disambung dari Gemma 3 4B IT)
Phase 1    Joint SFT              (teks chat/indoqa + vision, 1 loop, split-LR per komponen)
Phase 2    Joint ORPO             (teks + vision)
Final      Merge BF16 + 4bit      (deploy chatbot Indonesia multimodal)
```

Detail hyperparameter lengkap (versi v7 di disk) ada di
[`2026-08-05_pipeline-training-v7-ringkasan.md`](2026-08-05_pipeline-training-v7-ringkasan.md).
Versi config yang dipakai di obrolan (dari notebook versi lebih lama) tercatat sebagai
**"config v6-era"** di file tersebut — ada beberapa perbedaan penting dengan v7.

---

## 1. Pertukaran #1 — Analisis Awal: Paper Level Strategi

**User:** Memperkenalkan pipeline (steering, grafting, SFT+ORPO, optimizer custom
GrokMuonAdEMA, LoRA rank 256) dan meminta rekomendasi paper yang applicable.

**Analisis awal (level strategi):**

| Paper | Ide |
|-------|-----|
| T5Gemma 2 base [2512.14856] | Arsitektur base: tied embedding, merged attention; dirancang agar fine-tuning lebih efektif |
| MTO [2606.24841] | Cocokkan objective pre-training ↔ objective fine-tuning (matching tasks to objectives); klaim gain besar di few-shot |
| XBridge [2603.17512] | Komposisi LLM + model terjemahan encoder-decoder untuk multilinguality extensible (mapping layers + optimal transport) |
| FlexTab [2606.30336] | ⚠️ *Dikoreksi di pertukaran #4* — ternyata untuk tabular ICL, bukan multi-task chatbot |
| T5Gemma-TTS [2604.01760] | Backbone T5Gemma + cross-attention enhancement (PM-RoPE) |
| Enc-Dec vs Dec-Only style transfer [2604.11687] | BART-large mengalahkan Mistral-7B dengan 17× lebih sedikit parameter |
| Domain adaptation SE texts [2607.06613] | CPT vs from-scratch untuk adaptasi domain, encoder vs decoder |
| Causal reasoning favors encoders [2512.10561] | Enc-dec lebih robust untuk reasoning & distributional shift |

Usulan strategi fine-tuning 4 fase (diagnose → MTO objective matching → domain adaptation →
instruction tuning + enhancement) — dicatat sebagai **kerangka strategi awal**, lalu
disempurnakan oleh pertukaran berikutnya.

---

## 2. Pertukaran #2 — "lanjutkan" → Deep-Dive Paper Level Internal Training

**User:** "lanjutkan"

**Deep-dive yang dilakukan:** Verifikasi 10 paper via arXiv API `id_list` (semua ✅ eksis),
lalu analisis per paper + sketsa patch untuk pipeline:

| Paper | Fokus | Aksi |
|-------|-------|------|
| OrScale [2605.07815] | Layer-wise trust-ratio untuk Muon; 3 failure mode Muon-LAMB hybrid | Patch `GrokMuonAdEMA`: ganti MuonClip global dengan trust-ratio per layer; coba coupled WD |
| TLPO [2604.26553] | Token-level policy optimization untuk language confusion; kritik sequence-level ORPO/DPO/GRPO | Patch `get_batch_logps`: pembobotan token di posisi chosen/rejected divergen |
| Muon in RL [2607.16169] | Muon vs AdamW di post-training; keunggulan = effective step lebih besar, bukan sihir | Kalibrasi `ORPO_MUON_LR_SCALE`; ukur update RMS; `2.5e-5` decoder ORPO ada di sweet spot paper |
| Decomposing Task Vectors [2512.22511] | Dekomposisi task vector: komponen shared vs task-unique | Steering: steer hanya komponen task-unique |
| Task Vector Geometry [2605.03780] | Task directions hidup di middle layers; dua mode inferensi | ✅ Validasi α_FFN_MID=0.25 di 25–80% depth |
| First-Order Predictable [2607.16821] | Validity window ~1e-2; komposisi berpasangan rapuh | ⚠️ α=0.25 kemungkinan di luar window; monitor probe-loss per fase |
| Recoverable but Not Stationary [2606.10929] | Basis task vector drift dalam ~100 step | Ide: re-steer ringan setelah SFT |
| Objective Matters [2601.12639] | SFT & preference tuning → persona drift & kerentanan di budget besar | Tambah KL-regularization; monitor drift persona |
| Mechanistic Alignment [2606.09850] | ORPO & DPO **menurunkan** separability; SimPO/PPO preservasi; KTO/GRPO meningkat | Pertimbangkan SimPO/KTO untuk iterasi berikutnya |
| CultureTalk-ID [2607.21016] | Benchmark dialog budaya Indonesia: 4.496 dialog, 11 bahasa, 13 topik | Tambah sebagai eval pasca-training |

Sketsa patch kode ada di `2026-08-05_review-01-paper-internal-training.md`.
Prioritas lengkap ada di `2026-08-05_rekomendasi-prioritas-dan-roadmap.md`.

---

## 3. Pertukaran #3 — Klarifikasi User

**User:** *"maksud ku diluar teknikal training ku sekarang... paper2 kayak begini
[menunjuk analisis level strategi], yang mungkin ide-ide lain yang bisa ku
terapin/tempelin/tambahkan di flow training ku sekarang untuk mendapatkan tambahan
inovasi atau kegunaan dari fine-tuning modelnya gitu"*

→ User **tidak** menginginkan lebih banyak detail optimizer/loss internal (sudah cukup),
melainkan **ide level strategi/komposisi/kegunaan** yang bisa ditempel di flow:
arsitektur add-on, strategi training tambahan, utilits pasca-training.

---

## 4. Pertukaran #4 — Verifikasi Paper Lama + Paper Strategi Baru

**Langkah:**
1. **Verifikasi 7 paper dari analisis awal** via arXiv API `id_list` → **semua 7 ✅ eksis**
   (MTO, XBridge, FlexTab, T5Gemma-TTS, style transfer, SE domain, causal reasoning).
2. **12 query pencarian** arXiv (sort by submittedDate desc) → menemukan paper baru
   yang relevan di level strategi.

**Koreksi penting:** FlexTab [2606.30336] judul aslinya *"A Flexible Encoder-Decoder
Architecture for In-Context Learning Across Diverse **Tabular** Tasks"* — deskripsi
awal ("shared encoder + task-specific decoder heads untuk chatbot multi-purpose")
**tidak akurat**. FlexTab fokus ke tabular ICL, bukan chatbot multi-task.

**Paper baru yang ditemukan:**

| Paper | Ide | Cara tempel di flow |
|-------|-----|---------------------|
| INTRA [2605.05806] | Encoder-decoder retrieve dari representasi internal sendiri (tanpa retriever eksternal) | Kemampuan knowledge-access untuk chatbot tanpa infra RAG |
| UL2R [2210.11399] | Continued pre-training UL2 dengan 0.1% extra compute | **Phase 0.75**: denoising CPT Bahasa Indonesia antara steering & SFT |
| Latent Personal Memory [2606.20911] | Personal memory sebagai dynamic soft prompts (frozen-base compatible) | Lapisan personalisasi per-user pasca-training |
| EasyEdit [2308.07269] | Knowledge editing (ROME/MEMIT-style) tanpa retrain | Perbaiki fakta salah/outdated pasca-fine-tuning |
| LoRA rank study [2607.25583] | Studi terkontrol rank × target modules × quantization | Validasi `LORA_RANK=256` via A/B 64/128/256 |
| Stacked from One [2603.04759] | Multi-scale self-injection untuk ekstensi konteks (murah dari CPT) | Extend `MAX_SOURCE_LENGTH` 16K → 32K |
| Cassandra [2605.26558] | Self-speculative decoding lossless di edge | Akselerasi inference deployment 2–3× |
| Emergent Misalignment [2607.21356] | Fine-tuning sempit merekrut persona subspace → misalignment luas | ⚠️ Monitoring safety pasca-steering (nyambung ke Phase 0.5) |

---

## 5. Pertukaran #5 — Permintaan Dokumentasi

**User:** *"tulis semua obrolan, paper2 di bagian docs, jangan lupa kasih tanggal,
karena obrolan kita isinya banyak, pisahkan jadi beberapa file tergantung isinya"*

→ Sesi ini dihasilkan: 8 file dokumentasi (lihat [`2026-08-05_INDEX.md`](2026-08-05_INDEX.md))
di `D:\Codings\unsloth-porto\t5-gemma-2\instruct\docs\sessions\` (folder sesi).

---

## Lampiran: Semua Paper yang Dibahas

- **Internal training (10):** 2604.26553, 2607.16169, 2605.07815, 2512.22511,
  2605.03780, 2607.16821, 2606.10929, 2601.12639, 2606.09850, 2607.21016
- **Strategi/komposisi (8):** 2512.14856, 2606.24841, 2603.17512, 2606.30336,
  2604.01760, 2604.11687, 2607.06613, 2512.10561
- **Ide baru terapan (8):** 2605.05806, 2210.11399, 2606.20911, 2308.07269,
  2607.25583, 2603.04759, 2605.26558, 2607.21356

Detail lengkap: [`2026-08-05_verifikasi-26-paper-arxiv.md`](2026-08-05_verifikasi-26-paper-arxiv.md)
