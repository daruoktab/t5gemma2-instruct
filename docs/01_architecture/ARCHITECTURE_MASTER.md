# [MASTER] T5-Gemma-2 Architecture & Design Reference

**Last Updated:** 5 Agustus 2026  
**Models:** `google/t5gemma-2-4b-4b` | `google/t5gemma-2-270m-270m` | `google/gemma-3-4b-pt/it`

---

## 1. Fundamental Architecture

T5-Gemma-2 adalah keluarga model **Encoder-Decoder** (Seq2Seq) yang dibangun melalui adaptasi model Gemma 3 menggunakan metode UL2 (Unifying Language Learning Tasks). Model ini dirancang untuk memiliki keunggulan pada konteks panjang, kemampuan multibahasa, dan multimodal.

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
| **Vision Tower** | SigLIP (Hidden 1152) | SigLIP (Hidden 1152) — Identik |
| **Tied embeddings** | ✅ Yes (Embed ↔ Head) | ✅ Yes (Enc ↔ Dec ↔ Head) |
| **Vocab Size** | 262,208 (extra 64 padding) | **262,144** (exact) |
| **is_encoder_decoder** | `False` | `True` |

> [!NOTE]
> **Variasi Ukuran:**
> - Gemma 3 270M & 1B adalah **Text-Only** (Vision Tower = 0).
> - T5Gemma 2 **semua ukuran** (270M, 1B, 4B) memiliki **Vision Tower** SigLIP 400M.

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

## 3. Tokenizer & Chat Format Analysis

Keduanya berbagi DNA tokenizer yang hampir identik dengan **Vocab Size 262.144**.

### Perbedaan Krusial: `<image_soft_token>`
Satu-satunya perbedaan teknis di level ID adalah posisi token vision:
- **Gemma 3:** `id 256001` = `<unused99>`
- **T5Gemma2:** `id 256001` = `<image_soft_token>` (mencuri satu slot unused).

### Struktur Unused Tokens & Pemetaan Task Prefix

| Blok | ID Range | Jumlah | Keterangan |
|---|---|---|---|
| **Blok 1** | 6 – 104 | 99 | `<unused0>` – `<unused98>`. **Pengecualian:** `<unused1>` hingga `<unused6>` (ID 7-12) *tidak* di-suppress karena digunakan untuk *Task Prefix*! |
| **Blok 2** | 256002 – 262143 | 6142 | `<unused100>` – `<unused6241>` |

#### Pemetaan 6 Task Prefix (`<unused1>` – `<unused6>`):
| Token ID | Unused Token | Kategori Task | Deskripsi Fungsi |
|---|---|---|---|
| `7` | `<unused1>` | **SUMMARIZE** | Peringkasan teks & ekstraksi poin utama |
| `8` | `<unused2>` | **TRANSLATE** | Penerjemahan antar bahasa (Indo ↔ Eng) |
| `9` | `<unused3>` | **NER** | Ekstraksi entitas (*Named Entity Recognition*) |
| `10` | `<unused4>` | **QA** | Tanya-Jawab berbasis dokumen (*Grounded QA*) |
| `11` | `<unused5>` | **PARAPHRASE** | Penulisan ulang & penyuntingan gaya bahasa |
| `12` | `<unused6>` | **GENERAL_CHAT** | Percakapan umum & dialog interaktif (*Casual Chat*) |

> [!WARNING]
> **Embedding Mismatch:** Gemma 3 memiliki 262.208 rows (extra 64 padding), sedangkan T5Gemma2 tepat 262.144. Saat transfer bobot embedding, pastikan melakukan *truncation*.

---

## 4. Merged Attention: Mekanisme Inti Decoder

T5Gemma2 tidak memiliki modul `cross_attention` terpisah. Sebagai gantinya, ia menggunakan **Merged Attention**:

1. **Query (Q):** Dibentuk dari decoder hidden states ($X \in \mathbb{R}^{m \times d}$).
2. **Key (K) & Value (V):** Dibentuk dengan mengkonkatenasi decoder input ($X$) dan encoder outputs ($H \in \mathbb{R}^{n \times d}$) secara sekuensial: $[X; H] \in \mathbb{R}^{(m+n) \times d}$.
3. **Masking:** Menggunakan kombinasi bidirectional (untuk encoder tokens / $H$) dan causal (untuk decoder tokens / $X$).

$$\mathbf{Q} = \mathbf{X} \mathbf{W}_q, \quad \mathbf{K} = [\mathbf{X}; \mathbf{H}] \mathbf{W}_k, \quad \mathbf{V} = [\mathbf{X}; \mathbf{H}] \mathbf{W}_v$$

$$\mathbf{A} = \text{SoftMax}\left(\frac{\mathbf{Q} \mathbf{K}^T}{\sqrt{d_h}} \odot \mathbf{M}\right) \mathbf{V}, \quad \mathbf{O} = \mathbf{A} \mathbf{W}_o$$

**Implikasi:** Modul `self_attn` pada decoder T5Gemma2 memproses hubungan internal output (self) sekaligus hubungan output dengan input (cross) secara simultan dalam satu langkah proyeksi linear. Bobot Q, K, V tetap kompatibel dengan bobot model decoder-only secara struktural, namun distribusi perilakunya bergantung pada encoder state $H$.
