# Rencana fine-tuning: T5-Gemma-2 instruct / chat (Indonesia utama, English sekunder)

Dokumen ini menggabungkan **visi produk**, **landasan riset singkat**, **anggaran data (~10k unit)**, **kendala compute**, dan **checklist** (270M / 4B, SFT, DPO, distilasi lewat API). Bagian akhir tetap fokus **4B LoRA** (`full_train_4b.py`).

---

## 0. Kendala & prinsip operasi (sesuai setup kamu)

| Aspek | Yang dipakai |
|--------|----------------|
| **Anggaran “guru”** | **API** — generate / perbaiki data, preference, distilasi **lewat teks** (bukan KD logits di GPU). |
| **GPU** | **A100** (satu kartu); latihan **nyicil** (beberapa run pendek, bukan maraton panjang). |
| **Jam train** | **Total jam rendah** — andalkan **kurasi data + epoch sedikit** + eval, bukan ratusan jam SFT kosong. |
| **Model** | **4B** (~**16GB** bf16 penuh); **LoRA lebar** mengurangi risiko vs full FT, tetap waspada VRAM batch × panjang urutan. |
| **Iterasi cepat** | **270M** untuk coba template / pipeline; **4B** untuk run yang diseriuskan. |

**Distilasi via API:** output API disimpan sebagai **`input`/`target`** atau nested conversation → **SFT** (dan opsional **DPO**). Cek **ToS** penyedia API; kurasi fakta & variasi gaya.

---

## 0b. Visi produk (intinya)

| Aspek | Target |
|--------|--------|
| **Bahasa** | Utama **Bahasa Indonesia**; English hanya bila user minta atau konteksnya English (sudah selaras `SYSTEM_PROMPT`). |
| **Cakupan** | Topik **luas** sehari-hari + pengetahuan umum yang **masuk akal**, mengandalkan **kapasitas pengetahuan base** tanpa memaksakan “ahli palsu” di domain berisiko tinggi. |
| **Chat** | **Multi-turn** natural (koherensi, konteks, gaya). |
| **Kekuatan encoder–decoder** | Di dalam percakapan, model juga kuat pada **tugas terstruktur**: ringkas, terjemah, parafrase, jawaban dari **teks yang user tempel** (mirip *in-context retrieval* / **grounded QA** tanpa indeks eksternal), **ekstraksi entitas / poin**, checklist, dsb. |
| **Skala data** | **~10k percakapan nested** (satu baris JSON = satu thread **sebelum** *turn unroll*) sebagai **orde besaran puncak** — bukan 10k baris `input`/`target`. Isi tidak wajib semua multi-turn panjang; **campuran** format lebih sehat (lihat §1). |

---

## 0c. Roadmap jumlah percakapan (nested, pre-unroll)

| Fase | Target | Catatan |
|------|--------|---------|
| **Sekarang** | ~**1.030** (base) + ~**1.470** (extra) = **2.500** thread merged | Satu file: `t5-gemma-2-chat-instruct-merged.jsonl`. |
| **Fase 1** | **~2.500** percakapan nested | Kurasi / tambah sedikit agar **~2.500** utuh, lalu **flatten → SFT** untuk satu run evaluatif. |
| **Fase 2** | **~5.000** thread | Boleh **sedikit ubah komposisi** (lebih multi-turn pendek, lebih grounded QA, dsb.). |
| **Plafon jangka panjang** | **~10k** thread | Setelah ~5k evaluasi, putuskan penuh 10k vs geser anggaran ke **DPO / kualitas / API distil**. |

**Setelah unroll:** baris SFT ≈ **~10–12×** jumlah thread (orde besaran) — yang masuk `Seq2SeqTrainer`, bukan angka thread.

---

## 0d. Mekanisme & referensi (paper / survey)

1. **T5 “text-to-text”** — Raffel et al., JMLR 2020 — https://arxiv.org/abs/1910.10683  
2. **Flan / instruction scaling** — Chung et al., arXiv:2210.11416 — https://arxiv.org/abs/2210.11416 — *Flan Collection*: https://openreview.net/forum?id=ZX4uS605XV  
3. **Multi-turn** — Together.ai (praktis); *TurnWise* (2026): https://arxiv.org/abs/2603.16759; Parrot (ACL 2024): https://aclanthology.org/2024.acl-long.525.pdf  
4. **Grounded di pesan vs RAG** — Lewis et al. RAG NeurIPS 2020; survey RAG mis. arXiv:2503.10677  
5. **Preferensi** — DPO / TRL; cek kompatibilitas **seq2seq** di versi library kamu.

---

## 1. Strategi data (nested pre-unroll; plafon ~10k thread)

**Definisi:** angka **2.5k → 5k → ~10k** = **jumlah percakapan** (`conversations` per baris JSON), **bukan** baris setelah unroll.

| Komponen (indikatif) | Perkiraan bobot | Isi |
|----------------------|-----------------|-----|
| **Multi-turn panjang** (7–15 pasang) | ~35–45% | Dataset nested + ekstra. |
| **Multi-turn pendek** (2–5 pasang) | ~20–30% | Tanya-jawab cepat, klarifikasi, follow-up. |
| **Single-turn / task dalam satu bubble** | ~15–25% | Instruksi jelas + jawaban; persona chat. |
| **Grounded pada teks di pesan** | ~10–20% | IndoQA-style + kutipan fiktif di chat. |

**Skrip:** `generate_dataset_deepseek.py`, `flatten_conversations_jsonl_to_sft.py`, `scripts/analyze_instruct_datasets.py`, `generate_dataset_preferences_deepseek.py`.

### 1a. Komposisi final **~10.000** (unit perencanaan)

| Sumber | Jenis unit | Target (~) | Asal |
|--------|------------|--------------|------|
| **Multi-turn nested** | **thread** | **~6.200** | Base + extra + generate; proporsi internal §1. |
| **IndoQA** | **baris** | **~2.500** | `indoqa_train.jsonl` (~3.3k) → sample stratified. |
| **Grounded + instruksi pendek** (template chat + `SYSTEM_PROMPT`) | **baris** `EXTRA_SFT_JSONL` | **~900–1.200** | Disiapkan / API. |
| **Refusal / batas / minta konteks** | **baris** | **~200–300** | Disiapkan / API. |
| **Opsional EN / variasi** | **baris** | **~0–1.000** | Hanya jika produk butuh. |

**Baris trainer:** `≈ (thread × pasang rata-rata) + IndoQA + EXTRA` → **puluhan ribu** baris SFT (normal).

**DPO (luar 10k di atas):** **~1.500–4.000** pasangan berkualitas; generate + kurasi.

**Fase ~2.5k thread:** subset proporsional atau prioritas thread + IndoQA + EXTRA minimal, lalu scale thread ke **~6.2k**.

---

## 2. Status kode training 4B (`full_train_4b.py`)

- **LoRA lebar**: `q/k/v/o`, `gate/up/down`, `r=32`, `alpha=64`.  
- **bf16** penuh — sesuaikan batch dengan VRAM **~16GB** model + adapter + aktivasi.  
- **Campuran** chat + IndoQA + `EXTRA_SFT_JSONL`; samakan **system prompt** inference dengan data (§5).

---

## 3. Data — urutan kerja

### 3a. Sumber “bersih” (nested + IndoQA)

| Artefak | Penjelasan |
|---------|----------------|
| `t5-gemma-2-chat-instruct-merged.jsonl` | Gabungan **base + extra** nested (`scripts/merge_nested_conversations_jsonl.py`). |
| `chat_train.jsonl` / `chat_val.jsonl` | Hasil **flatten + split thread** (`scripts/rebuild_chat_sft_from_nested.py`, default **40** thread val). |
| `indoqa_train.jsonl` | **2.500** baris (subset deterministik); full train disimpan di `indoqa_train_full.jsonl` otomatis saat pertama kali trim (`scripts/trim_indoqa_train.py`). |

**Catatan:** file pecahan `t5-gemma-2-chat-instruct-dataset*.jsonl` boleh diarsipkan setelah merge diverifikasi; trainer hanya membaca `chat_*.jsonl` + `indoqa_train.jsonl`.

| Langkah | Tindakan | Artefak |
|--------|----------|---------|
| 3b | Flatten nested tunggal (opsional) | `flatten_conversations_jsonl_to_sft.py` |
| 3c | Kurasi + campur §1 | `EXTRA_SFT_JSONL` |
| 3d | Filter echo, target kosong | `scripts/check_dataset_quality.py` (perluas bila perlu) |
| 3e | Satu system string | `SYSTEM_PROMPT`, `test_model_*.py` |

---

## 4. SFT 4B — parameter

- **LR**: `2e-4` agresif; coba **`1e-4`** / **`5e-5`** bila tidak stabil.  
- **Epoch**: **1–2** setelah kurasi.  
- **Eval** + decode multi-turn; **simpan checkpoint** beberapa titik, pilih yang terbaik (bukan otomatis epoch terakhir).

---

## 5. Inference — selaras training

- **EOS**: `eos_token_id` = `<end_of_turn>`.  
- **System** + format Gemma konsisten dengan trainer.

---

## 6. DPO (setelah SFT memadai)

| Langkah | Tindakan |
|--------|----------|
| 6a | `generate_dataset_preferences_deepseek.py` |
| 6b | `train_dpo_4b.py` (TRL + seq2seq) |
| 6c | `beta` sedang, **LR < SFT** |

---

## 7. Opsional ringan (tanpa cluster / tanpa KD logits)

Urutan **ROI tinggi vs compute rendah**:

1. **Distilasi lewat API** — guru menulis/memperbaiki target → SFT (§0).  
2. **DPO / ORPO / KTO** pendek — satu fine-tune preferensi, bukan RLHF penuh.  
3. **Loop eval** — sampel gagal → tambah data / preference **hanya** untuk failure mode.  
4. **Merge checkpoint / LoRA** — stabilisasi ringan (tooling umum, compute kecil).

**Biasanya ditunda:** KD logits teacher per step, PPO/RLHF besar, pretrain lanjutan massif.

---

## 8. Urutan eksekusi singkat

1. **Merge** nested base+extra → `t5-gemma-2-chat-instruct-merged.jsonl` (`scripts/merge_nested_conversations_jsonl.py`).  
2. **Rebuild** `chat_train.jsonl` / `chat_val.jsonl` dari merged (`scripts/rebuild_chat_sft_from_nested.py`).  
3. **Trim** IndoQA → `indoqa_train.jsonl` (**2.500**), backup full → `indoqa_train_full.jsonl` (`scripts/trim_indoqa_train.py`).  
4. Cek total baris & `max_length`; tambah `EXTRA_SFT_JSONL` bila perlu.  
5. **SFT** 270M (smoke) lalu **4B LoRA**; eval.  
6. Naik thread / **preference → DPO** / **API distil** sesuai §7.

**Skrip:** `merge_nested_conversations_jsonl.py`, `rebuild_chat_sft_from_nested.py`, `trim_indoqa_train.py`, `flatten_conversations_jsonl_to_sft.py`, `generate_dataset_preferences_deepseek.py`, env **`EXTRA_SFT_JSONL`** di `full_train_4b.py`.
