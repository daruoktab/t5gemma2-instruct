# [MASTER] T5-Gemma-2 Architecture & Design Reference

**Last Updated:** 15 Mei 2026  
**Models:** `google/t5gemma-2-4b-4b` | `google/gemma-3-4b-pt/it`

---

## 1. Fundamental Architecture

T5-Gemma-2 adalah keluarga model **Encoder-Decoder** (Seq2Seq) yang dibangun melalui adaptasi model Gemma 3 menggunakan metode UL2. Model ini dirancang untuk memiliki keunggulan pada konteks panjang, kemampuan multibahasa, dan multimodal.

### Perbandingan Kunci: Gemma 3 vs T5Gemma2

| Aspek | Gemma 3 (4B) | T5Gemma2 (4B-4B) |
|---|---|---|
| **Arsitektur** | Multimodal Decoder-only | **Multimodal Encoder-Decoder** |
| **Model type** | `gemma3` | `t5gemma2` |
| **Total parameter** | **~4.28B** (3.88B text + 0.4B vision) | **~7.51B** (3.88B enc + 3.88B dec + 0.4B vis) |
| **Encoder layers** | — | 34 (text) + 27 (vision SigLIP) |
| **Decoder layers** | 34 | 34 |
| **Hidden size** | 2560 | 2560 |
| **Attention** | Standard Self-Attention | **Merged Attention** (Self + Cross) |
| **Vision Tower** | SigLIP (Hidden 1152) | SigLIP (Hidden 1152) |
| **Tied embeddings** | ✅ Yes (Embed ↔ Head) | ✅ Yes (Enc ↔ Dec ↔ Head) |

> [!NOTE]
> **Variasi 1B:** Berdasarkan paper teknis, **Gemma 3 1B adalah Text-Only** (Vision Encoder = 0). Namun, **T5Gemma 2 1B-1B tetap memiliki Vision** dengan meminjam Vision Tower SigLIP 400M.


---

## 2. Architecture Tree

### A. Gemma 3 (Multimodal Decoder-Only)
```
Gemma3ForConditionalGeneration
├── model (Gemma3Model)
│   ├── vision_tower (SiglipVisionModel)              ← Vision encoder
│   │   ├── embeddings: Conv2d(3→1152, 14×14) + Embedding(4096, 1152)
│   │   ├── encoder: 27 × SiglipEncoderLayer
│   ├── multi_modal_projector (Gemma3MultiModalProjector)
│   ├── embed_tokens: (262208, 2560)                  ← 64 extra padding slots
│   ├── layers: 34 × Gemma3DecoderLayer
│   │   ├── self_attn (Gemma3Attention)
│   │   ├── mlp (Gemma3MLP)
│   ├── norm: RMSNorm(2560)
└── lm_head: Linear(2560 → 262208)      ← tied with embed_tokens
```

### B. T5Gemma2 (Multimodal Encoder-Decoder)
```
T5Gemma2ForConditionalGeneration
├── model
│   ├── encoder (T5Gemma2EncoderModel)
│   │   ├── text_model (T5Gemma2TextEncoder)
│   │   │   ├── embed_tokens: (262144, 2560)
│   │   │   ├── layers: 34 × T5Gemma2EncoderLayer
│   │   ├── vision_tower (SiglipVisionModel)          ← IDENTIK dengan G3
│   │   ├── multi_modal_projector (RMSNorm 1152)      ← Adaptasi visual ke text
│   ├── decoder (T5Gemma2Decoder)
│   │   ├── embed_tokens: (262144, 2560)              ← Tied dengan encoder
│   │   ├── layers: 34 × T5Gemma2DecoderLayer
│   │       ├── self_attn (T5Gemma2MergedAttention)   ← MERGED attn!
│   │       ├── mlp (Gemma3-style)
└── lm_head: Linear(2560 → 262144)                    ← Tied dengan encoder/decoder
```

---

## 3. Tokenizer Analysis (Gemma 3 vs T5Gemma2)

Keduanya berbagi DNA tokenizer yang hampir identik dengan **Vocab Size 262.144**.

### Perbedaan Krusial: `<image_soft_token>`
Satu-satunya perbedaan teknis di level ID adalah posisi token vision:
- **Gemma 3:** `id 256.001` = `<unused99>`
- **T5Gemma2:** `id 256.001` = `<image_soft_token>` (mencuri satu slot unused).

### Struktur Unused Tokens (Penting untuk Suppression)
| Blok | ID Range | Jumlah | Keterangan |
|---|---|---|---|
| **Blok 1** | 6 – 104 | 99 | `<unused0>` – `<unused98>`. **Pengecualian:** `<unused1>` hingga `<unused6>` (ID 7-12) *tidak* di-suppress karena digunakan untuk *Task Prefix*! |
| **Blok 2** | 256.002 – 262.143 | 6142 | `<unused100>` – `<unused6241>` |

> [!WARNING]
> **Embedding Mismatch:** Gemma 3 memiliki 262.208 rows (extra 64 padding), sedangkan T5Gemma2 tepat 262.144. Saat transfer bobot embedding, pastikan melakukan *truncation*.

---

## 4. Merged Attention: Mekanisme Inti Decoder

T5Gemma2 tidak memiliki modul `cross_attention` terpisah. Sebagai gantinya, ia menggunakan **Merged Attention**:
1. **Query (Q):** Dibentuk dari decoder hidden states (**X**).
2. **Key (K) & Value (V):** Dibentuk dengan mengkonkatenasi *decoder input* (**X**) dan *encoder outputs* (**H**) secara sekuensial: `[X; H]`.
3. **Masking:** Menggunakan kombinasi bidirectional (untuk encoder tokens/H) dan causal (untuk decoder tokens/X).

**Implikasi:** Modul `self_attn` pada decoder T5Gemma2 memproses hubungan internal output (self) sekaligus hubungan output dengan input (cross) secara simultan dalam satu langkah proyeksi linear. Bobot Q, K, V tetap kompatibel dengan bobot model decoder-only.

---

## 6. Dimensi & Kompatibilitas "Cangkok"

Untuk melakukan transplantasi bobot (cangkok), *hidden size* antara sumber dan target harus identik.

### Tabel Dimensi (Hidden Size)

| Model | Text Hidden Size | Vision Hidden Size (SigLIP) | Status Multimodal |
|---|---|---|---|
| **Gemma 3 270M / 1B** | 640 / 1152 | — | Text Only |
| **Gemma 3 4B** | **2560** | 1152 | Multimodal |
| **T5Gemma2 270M** | 640 | 1152 | Multimodal |
| **T5Gemma2 1B** | 1152 | 1152 | Multimodal |
| **T5Gemma2 4B** | **2560** | 1152 | Multimodal |

### Analisis Kompatibilitas

1.  **Decoder (Teks)**:
    - ✅ Kompatibel jika ukuran model sama (misal: Gemma 3 270M IT -> T5Gemma2 270M).
    - ❌ Tidak kompatibel silang ukuran (misal: Gemma 3 4B -> T5Gemma2 1B) tanpa *reduction*.
2.  **Vision Tower**:
    - ✅ Kompatibel di semua varian karena semuanya menggunakan SigLIP 400M (Hidden 1152).
3.  **Multi-modal Projector (Adapter)**:
    - ✅ **4B-ke-4B**: Kompatibel penuh (1152 → 2560). Kita bisa memindahkan kecerdasan visual Gemma 3 4B ke T5Gemma2 4B.
    - ❌ **4B-ke-1B/270M**: Mismatch dimensi. Adapter Gemma 3 4B berakhir di 2560, sedangkan T5Gemma2 kecil butuh 1152 atau 640.

> [!TIP]
> Untuk pengujian di laptop (versi 270M/1B), fokus utama "cangkok" adalah pada **Decoder** agar model memiliki basis instruksi teks yang kuat, sementara komponen Vision tetap menggunakan bobot asli T5Gemma2 atau hasil *pre-training* UL2.

---

## 7. Normalization & Layer Naming

Karena adanya perbedaan ketersediaan komponen Vision pada versi kecil, strategi inisialisasi yang digunakan adalah:
- **Decoder Weights**: Diambil dari **Gemma 3 1B IT** (terbaik untuk instruksi teks).
- **Vision Weights**: Diambil dari **Gemma 3 4B IT** (terkecil yang memiliki Vision Tower & Projector).
- **Architecture**: Ditanamkan ke dalam **T5Gemma 2 1B-1B**.

| Posisi | Gemma 3 | T5Gemma2 |
|---|---|---|
| Pre-attention | `input_layernorm` | `pre_self_attn_layernorm` |
| Post-attention | `post_attention_layernorm` | `post_self_attn_layernorm` |
| Final Norm | `norm` | `norm` |

---
