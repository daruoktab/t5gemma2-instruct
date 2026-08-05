# 🎯 Rekomendasi, Prioritas & Roadmap

**Tanggal:** 2026-08-05 · **Kontekst:** Pipeline T5Gemma-2 4B chatbot Indonesia (v7)
**Sumber:** Review-01 (internal training), Review-02 (strategi), Review-03 (ide baru)

---

## 1. Prioritas — Level Internal Training (dari Review-01)

| # | Aksi | Paper | Effort | Risiko |
|---|------|-------|--------|--------|
| 1 | **Patch TLPO-weighted** ke `get_batch_logps` (flag eksperimen) | [2604.26553] | Rendah | Perlu validasi eval |
| 2 | **Ukur update RMS** Muon vs AdamW (`g_ortho.norm()` per step) | [2607.16169] | Sangat rendah | — |
| 3 | **OrScale trust-ratio** sebagai flag eksperimen (ganti MuonClip) | [2605.07815] | Sedang | ⚠️ Tinggi (ubah optimizer) |
| 4 | **Cek window first-order**: log probe-loss per fase | [2607.16821] | Rendah | — |
| 5 | **Re-steer ringan pasca-SFT** (sebelum ORPO) | [2606.10929] | Sedang | Medium |
| 6 | **SimPO/KTO** sebagai alternatif Phase 2 di iterasi berikutnya | [2606.09850] | Sedang | Medium |
| 7 | **KL-reg / persona-drift monitoring** di Phase 2 | [2601.12639] | Rendah | — |
| 8 | **Eval CultureTalk-ID** pasca-training (11 bahasa Indonesia) | [2607.21016] | Rendah | — |

## 2. Prioritas — Level Strategi/Komposisi (dari Review-02 & 03)

| # | Aksi | Paper | Effort | Impact |
|---|------|-------|--------|--------|
| 1 | **UL2R "Phase 0.75"**: denoising CPT Bahasa Indonesia antara steering & SFT | [2210.11399] | Sedang | 🔴 Tinggi — fondasi bahasa |
| 2 | **INTRA**: knowledge-access tanpa retriever eksternal | [2605.05806] | Eksperimen | 🔴 Tinggi — beda dari RAG |
| 3 | **LoRA rank A/B** (64/128 vs 256) untuk validasi | [2607.25583] | Rendah | 🟡 Validasi |
| 4 | **EasyEdit**: edit fakta pasca-training tanpa retrain | [2308.07269] | Rendah | 🟡 Utility |
| 5 | **LPM**: personalisasi per-user (soft prompts, frozen base) | [2606.20911] | Sedang | 🟡 Utility |
| 6 | **Stacked from One**: extend konteks 16K → 32K tanpa CPT | [2603.04759] | Sedang | 🟢 Opsional |
| 7 | **Cassandra**: self-speculative decoding saat deploy | [2605.26558] | Sedang | 🟢 Opsional |
| 8 | **Misalignment check**: eval safety pasca-steering | [2607.21356] | Sangat rendah | 🟡 Penting |
| 9 | **Audit MTO**: task prefix & template vs objective UL2 | [2606.24841] | Rendah | 🟡 |
| 10 | **PM-RoPE** untuk koherensi multi-turn (enhancement arsitektur) | [2604.01760] | Tinggi | 🟢 |

---

## 3. Roadmap Eksekusi yang Disarankan

### 🚀 Gelombang 1 — Murah & Informatif (sebelum run produksi berikutnya)
1. **Misalignment check** (gratis): bandingkan output model final vs base di prompt
   safety-sensitive → tambah 1 baris eval di `VisionSampleGenerationCallback`.
2. **Ukur update RMS** Muon vs AdamW → logging `g_ortho.norm()` per step (validasi
   `ORPO_MUON_LR_SCALE=5.0` vs temuan paper [2607.16169]).
3. **LoRA rank A/B** (64/128/256) di run kecil → buktikan rank=256 benar-benar menang.
4. **TLPO-weighted patch** sebagai flag eksperimen di `get_batch_logps`.

### 🧪 Gelombang 2 — Eksperimen Terisolasi (satu variabel per run)
5. **UL2R Phase 0.75** — denoising CPT Indonesia (budget ~0.1% compute) → lanjut
   SFT+ORPO normal, bandingkan dengan baseline tanpa UL2R.
6. **Re-steer pasca-SFT** dengan α kecil → sinkronkan arah IT dengan state baru.
7. **OrScale trust-ratio** (A/B terisolasi, flag `ORSCALE_MODE`).

### 🏗️ Gelombang 3 — Iterasi v8 & Deployment
8. Ganti/bandingkan Phase 2: **SimPO** (preserve geometry) atau **KTO** vs ORPO.
9. **KL-regularization** kecil di Phase 2 + monitoring persona drift.
10. Utility pasca-training: **EasyEdit** untuk knowledge, **LPM** untuk personalisasi.
11. Deployment: **Cassandra** (self-speculative decoding), **Stacked from One** (konteks
    lebih panjang), eval **CultureTalk-ID**.

---

## 4. Prinsip yang Dipegang (dari riset sesi ini)

1. **Muon di post-training = effective step size, bukan sihir algoritma**
   ([2607.16169]) → kalibrasi LR scale, bukan ganti optimizer.
2. **ORPO/DPO menurunkan separability geometris** ([2606.09850]) → pembuktian tetap lewat
   eval apple-to-apple; alternatif SimPO/KTO tersedia.
3. **Task vector itu rapuh & tidak statis** ([2607.16821], [2606.10929]) → validasi
   probe-loss per fase; pertimbangkan re-steer.
4. **Steering wajib menghormati Merged Attention** (α_QO=α_KV=0.0) — sudah benar di v7.
5. **Objective matching itu penting** (MTO [2606.24841] + UL2R [2210.11399]) → model
   UL2-based sebaiknya disentuh dengan objective denoising UL2 juga (bukan hanya SFT/ORPO).

---

## 5. Langkah Berikutnya (opsi)

- [ ] Baca full paper untuk 3 prioritas: [2604.26553] TLPO, [2210.11399] UL2R, [2605.05806] INTRA
- [ ] Cari & verifikasi repo dataset **CultureTalk-ID** di HuggingFace
- [ ] Draf implementasi UL2R Phase 0.75 (collator span-corruption + titik sisip pipeline)
- [ ] Rancang template A/B rank LoRA (64/128/256) untuk Molab
- [ ] Update doc ini tiap keputusan diambil (beri tanggal baru)
