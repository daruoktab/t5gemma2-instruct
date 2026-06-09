# [MASTER] Instruct Tuning Pipeline & Weight Transplant Strategy

**Last Updated:** 15 Mei 2026  
**Status:** Ready for Execution (V2.2)

---

## 1. Strategi Pelatihan (Multi-Stage)

Pelatihan dibagi menjadi dua fase utama untuk memaksimalkan efisiensi dan kualitas instruksi:

### A. Tahap 1: Supervised Fine-Tuning (SFT)
- **Model:** `google/t5gemma-2-4b-4b` (Base) dengan suntikan bobot dari Gemma 3 IT.
- **Data:** Campuran `chat_multiturn` (2.5K) dan `indoqa_documents` (4.4K).
- **Curriculum:** IndoQA (Grounding Faktual) → Chat Multi-turn (Sosial/Conversational).

### B. Tahap 2: Alignment (ORPO / SimPO)
- **Metode:** Reference-free alignment untuk menghemat VRAM.
- **Target:** Mengurangi halusinasi dan memperbaiki gaya bahasa agar lebih asisten-sentris.

---

## 2. Strategi "Cangkok" Weight (Transplantasi)

Untuk mendapatkan kemampuan *instruct* secara cepat, kita melakukan transplantasi bobot dari `google/gemma-3-4b-it` ke `google/t5gemma-2-4b-4b`.

### A. Vision Transplant (Aman & Identik)
- **Vision Tower (SigLIP):** Dapat dicangkok 100% karena arsitektur identik.
- **Multi-modal Projector:** Dapat dicangkok karena dimensi output sama (2560).
- **Hasil:** Model memiliki ekstraksi fitur visual yang sudah *instruct-ready*.

### B. Decoder Transplant (Hati-hati: Merged Attention)
Mengingat T5Gemma2 menggunakan **Merged Attention**, transplantasi decoder harus mengikuti hirarki risiko:
1. **Rekomendasi Utama: Task Vector Transfer.**
   - Hitung Delta: `Delta = Gemma3_IT - Gemma3_PT`.
   - Tambahkan Delta ke decoder T5Gemma2 dengan skala (misal: 0.3 - 0.5).
   - *Manfaat:* Mempertahankan adaptasi UL2 (cross-attention) sambil menyuntikkan gaya instruksi.
2. **Alternatif: Partial MLP Transplant.**
   - Hanya pindahkan bobot MLP (`gate_proj`, `up_proj`, `down_proj`) dan LayerNorm.
   - Biarkan modul attention T5Gemma2 tetap aslinya.

---

## 3. Penanganan Teknis Kritis

### A. Tokenizer & Suppression (6.244 Tokens)
T5Gemma2 memiliki 6.241 unused tokens dan 3 vision tokens yang harus di-suppress selama training teks murni:
- **Lapis 1:** Reinisialisasi embedding unused tokens ke `mean + noise`.
- **Lapis 2:** Freeze gradient pada embedding tersebut via hook.
- **Lapis 3:** Gunakan `LogitsProcessor` saat inferensi untuk set `-inf`.

### B. LoRA & DoRA Configuration
- **Rank (r):** 32 | **Alpha:** 64.
- **Target Modules:** Semua linear layers (`q, k, v, o, gate, up, down`).
- **DoRA:** Sangat direkomendasikan untuk stabilitas training lebih baik.

---

## 4. Ringkasan Parameter Efektif
- **TOTAL Parameter:** ~7.51B (Kalkulasi VRAM harus berbasis angka ini).
- **Shared Embeddings:** Pastikan konsistensi tied weights antara encoder, decoder, dan head.

---
