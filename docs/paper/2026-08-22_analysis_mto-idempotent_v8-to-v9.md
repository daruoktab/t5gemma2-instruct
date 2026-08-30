# Analisis 2 Paper → Rekomendasi T5Gemma-2 (v8 → v9+)

**Tanggal:** 2026-08-22 · **Sumber:** 2606.24841 (MTO) & 2606.22304 (Idempotent E-D Alignment)
**Konteks:** Pipeline `working-molab-v8.py` — T5Gemma-2 4b-4b JOINT MULTIMODAL (DeVec SVD steering + vision grafting SigLIP + joint SFT + joint ORPO/TLPO + OrScale-LM/GrokFast/AdEMAMix). Asumsi: **data prep mudah** (TEXT chat/indoqa, VISION sudah diformat).

---

## A. Paper 2606.24841 — Matching Tasks to Objectives (MTO)

**Satu-klaim inti:** *Menyelaraskan format/template data dengan objective pre-training model* menghasilkan lonjakan besar — **>120% gain vs metode konvensional di few-shot**, dan tetap unggul di full-dataset. Ini berlaku untuk **fine-tuning maupun prompt-tuning**, dan untuk model **encoder-decoder** (fokus paper-nya).

**Framework MTO (4 tahap):**
1. **Klasifikasi task** (expert atau classifier) → kategori: **Mask-Filling** | **Map-Phrasal** | **QA**.
2. **Pilih objective yang cocok** → denoising utk Mask-Filling, LM (language model) utk Map-Phrasal, dsb.
3. **Continued unsupervised pretraining** di korpus task-related dengan objective terpilih (LM / Denoising / **Mixed**).
4. **Fine-tune / prompt-tune** dengan **template yang selaras** dgn objective (MaskedMapping/MaskedPrompting vs Mapping/Prompting).

**Temuan kunci:**
- Template **mask-based** (MaskedMapping, MaskedPrompting) **mengalahkan** mapping/prompting polos untuk task Mask-Filling — karena cocok dgn objective denoising.
- Untuk Map-Phrasal: **LM atau Mixed + Mapping** yang menang; peran objective & template harus **diseleksi barengan** (bukan independen).
- **Mixed objective** (ada obj. classifier per-sentence → pilih denoising vs LM) sering jadi yang terbaik — fleksibel.
- Berhasil juga di **prompt-tuning** (soft prompt): alignment + learnable prefixes.
- Code: `github.com/puraminy/MTO/`

**Status di v8:** Sudah ada `ENABLE_MTO_PREFIX=True` rute `<unused1..6>`. Tapi v8 memakai prefix **statis** per-dataset, belum ada (a) *objective classifier*, (b) continued-pretrain, (c) template yang dipilih per-category. Inilah ruang naik kelas utk v9.

---

## B. Paper 2606.22304 — Encoder-Decoder Manifold Alignment (Idempotency)

**Satu-klaim inti:** encoder-decoder yang tidak "idempotent" (= geometri encoder & decoder mismatch) menyebabkan **drift & instabilitas** di bawah aplikasi berulang. Regularizer **latent-consistency** sederhana menutup gap-nya.

**Mekanisme (bisa dicomot langsung):**
```
L_idem = || E(D(sg(z))) − sg(z) ||²,   z = E(x)
L_total = L_base + λ·L_idem
```
- **Stop-gradient penting!** Tanpa sg, model bisa *collapse ke identity* (f'(z)=1) — trivially min. loss. Dengan sg, degenerate fixed point hilang.
- **Error bound:** output-drift ≤ L·(latent-error) → menekan drift latent = menekan drift output (justifikasi teoretis, bukan sekadar regularizer empiri).
- **Price:** +1 siklus decode-encode per step (lebih mahal memory/timeout).
- **Caveat besar:** validasi hanya di **image generation/editing** skala kecil; **apakah scale ke LLM → open question** (dicatat di Limitations). Jadi perlakukan sebagai **eksperimen**, bukan tambahan yang pasti menang.

---

## C. Insight Paling Menarik & Layak Dicoba (v8 sekarang / v9 ke depan)

Urutan dari yang paling *mudah + potensi tinggi*, ke yang lebih *eksperimental*.

### C1. [v9 — PALING REKOMENDASI] Objektif classifier + template-matching per sample (dari MTO)
Ganti prefix statis `<unused1..6>` dengan **router per-sample** yang mengklasifikasi tiap sampel (text/vision prompt) ke **Mask-Filling / Map-Phrasal / QA**, lalu memilih template & objective yang cocok.
- **Kenapa:** MTO-nya mencatat >120% gain di few-shot dari alignment ini. Di pipeline chat-corpus Indonesia, kita punya campuran: jawaban faktual prompt (cocok mask/QA), jawaban naratif panjang (cocok LM/map-phrasal). Router bisa ambil keduanya.
- **Implementation (data prep mudah):** train classifier kecil di (head, tail) pair seperti Fig.2 MTO — atau enak lagi, **rule-based/keyword heuristic** dulu (cheap, tanpa model tambahan) utk MVP. Kolom target-nya sudah ada (`chat_sft`, `indoqa_sft`).
- **Nilai teoretis:** template & objective harus *diseleksi barengan*, bukan top-down statis.

### C2. [v9] Continued unsupervised pretraining berobjective (MTO tahap 3)
Sebelum SFT, tambah **continued pretraining** di korpus Indonesia dengan objective **Mixed** (bukan hanya denoising/LM tunggal). Data prep mudah (corpus teks sudah ada); tinggal format ke objective.
- **Kenapa:** MTO-nya tunjukkan *adaptation stage* ini yang paling menentukan; "objective alignment" tanpa adaptation-stage tidak menang maksimal.
- **Catatan:** jangan langsung full-scale — ini tambahan tahap, run di sub-sampel korpus dulu (mis. 10–50 juta token) utk lihat sinyal.

### C3. [v9 — EKSPERIMEN] Latent-consistency regularizer (idempotent E-D) di Phase 1 (SFT)
Tambahkan `L_idem = ||E(D(sg(z)))−sg(z)||²` sebagai regularizer di SFT (bukan ORPO), dengan **λ kecil** (0.01–0.05) dan **stop-gradient wajib**.
- **Kenapa di SFT, bukan ORPO:** ORPO pakai odds-ratio → smoothing/extra loss bisa merusak; SFT aman.
- **Nilai potensial:** mengalir ke failure mode nyata seq2seq — **repetisi degenerate** dan **self-consistency drift**. Kalau bekerja, output lebih stabil & identitas terjaga (mirip argumen identity-preservation di paper).
- **Caveat:** extra decode-encode cycle → lebih mahal. Saran: trigger hanya tiap N step / subset batch, awali 1 epoch pendek.
- **Cara memvalidasi:** kumpulkan metrik repetisi & self-consistency (self-BLEU, dist-1/2/3, n-gram repetition ratio) sebelum vs sesudah, bandingkan dgn v8.

### C4. [v9 — IDE TAMBAHAN, murah] Mixed objective per-sample (MTO 4.4)
Kerangka yang sama dgn C1 tapi di level **objective training**: classifier memutuskan per-sentence apakah pakai denoising atau LM. Murah & data-driven; paper menyebut Mixed sering menang.

### C5. [teoretis, buat dokumen riset] "Idempotency = necesary condition for optimal projection"
Argumen Proposisi 1 & 3 bagus dipakai **framing** di `docs` / paper review kamu: "kenapa kita peduli konsistensi manifold encoder-decoder". Tidak perlu diimplementasikan, tapi memperkuat justifikasi riset (mis. mengaitkan dgn DeVec purification yang sudah ada).

---

## D. Ringkas "Peta v8 → v9"

| # | Ide | Tahap | Biaya | Resiko | Source |
|---|-----|-------|-------|--------|--------|
| C1 | Objective classifier + routig template per-sample | SFT (Phase 1) | Rendah (rule-based MVP) | Rendah | MTO |
| C2 | Continued-pretrain berobjective (Mixed) | Pra-SFT | Sedang | Rendah | MTO |
| C3 | Latent-consistency regularizer (idempotency) | SFT (Phase 1) | Sedang-Tinggi (extra E/D cycle) | Sedang (LLM belum terbukti) | Idempotent E-D |
| C4 | Mixed objective per-sentence | adaptasi data | Rendah | Rendah | MTO |
| C5 | Justifikasi teoretis idempotency | docs | Nol | Nol | Idempotent E-D |

**Saran urutan eksekusi:** C1 → C2 → C4 (semua dari MTO, murah & konsisten dengan arah riset yang sudah ada: alignment), lalu **C3** sebagai *experimental arm* terpisah 1 run supaya hasilnya tidak tercampur dengan perubahan lain.

---

## Fitur & data yang sudah ada di v8 (bisa dimanfaatkan)

- `ENABLE_MTO_PREFIX=True` + `<unused1..6>` → fondasi routing sudah ada, tinggal upgrade ke classifier.
- `TEXT_INDOQA_CONFIG="indoqa_sft"` → ada subset QA yang cocok utk kategori Mask-Filling/QA.
- `FLAW_AWARE_LOSS` → sudah ada skema "emphasis" per sample; pola yang sama bisa dipakai utk weighting router/template.
- Vision branch sudah diformat (`DATASET_VISION_REPO`) → C1/C4 berlaku juga ke prompt vision bila template-nya disamakan.
