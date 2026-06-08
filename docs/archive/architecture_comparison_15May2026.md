# Perbandingan Arsitektur: Gemma 3 (4B) vs T5Gemma2 (4B-4B)

**Tanggal:** 15 Mei 2026  
**Model:** `google/gemma-3-4b-pt` vs `google/t5gemma-2-4b-4b`  
**Script:** [_compare_architectures.py](file:///d:/Codings/unsloth/t5-gemma-2/instruct/data/_compare_architectures.py)

---

## 1. Ringkasan Perbedaan Fundamental

| Aspek | Gemma 3 (4B) | T5Gemma2 (4B-4B) |
|---|---|---|
| **Arsitektur** | Decoder-only | **Encoder-Decoder** |
| **Model type** | `gemma3` | `t5gemma2` |
| **Total parameter** | **3.88B** | **7.51B** |
| **Decoder layers** | 34 | 34 |
| **Encoder layers** | — | 34 (text) + 27 (vision SigLIP) |
| **Hidden size** | 2560 | 2560 |
| **Attention heads** | 8 (KV) / 8 (Q heads inferred) | 8 (KV) / 8 (Q) |
| **Head dim** | 256 | 256 |
| **FFN intermediate** | 10240 | 10240 |
| **Embedding vocab** | **262.208** ⚠️ | **262.144** |
| **Cross-attention** | N/A | **Merged** (no separate module) |
| **Tied embeddings** | ✅ Yes | ✅ Yes |
| **RoPE** | ✅ | ✅ |
| `is_encoder_decoder` | `False` | `True` |

> [!WARNING]
> **Embedding size mismatch!** Gemma 3 memiliki embedding 262.**208** (64 slot lebih dari vocab 262.144), sedangkan T5Gemma2 memiliki embedding tepat 262.**144**. Ini berarti Gemma 3 menyisakan 64 padding slots di embedding matrix (kemungkinan untuk alignment GPU).

---

## 2. Config Comparison (Top-Level)

Hanya menampilkan yang **berbeda** atau unik:

| Key | Gemma 3 | T5Gemma2 | Status |
|---|---|---|---|
| `is_encoder_decoder` | `False` | `True` | ⚠️ Fundamental |
| `image_token_index` | 262.144 | 256.001 | ⚠️ |
| `boi_token_index` | 255.999 | — | G3 only |
| `mm_tokens_per_image` | 256 | — | G3 only |
| `text_config` | {38 keys} | — | G3 only |
| `vision_config` | {23 keys} | — | G3 only |
| `encoder` | — | {23 keys} | T5 only |
| `decoder` | — | {39 keys} | T5 only |
| `vocab_size` (top) | — | 262.144 | T5 only |
| `bos/eos/pad_token_id` | — | 2/1/0 | T5 only |

---

## 3. Architecture Tree

### Gemma 3 (Multimodal Decoder-Only)

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
│   │   │   ├── q_proj: (2048, 2560)
│   │   │   ├── k_proj: (1024, 2560)
│   │   │   ├── v_proj: (1024, 2560)
│   │   │   ├── o_proj: (2560, 2048)
│   │   │   ├── q_norm, k_norm: RMSNorm(256)
│   │   ├── mlp (Gemma3MLP)
│   │   │   ├── gate_proj: (10240, 2560)
│   │   │   ├── up_proj:   (10240, 2560)
│   │   │   ├── down_proj: (2560, 10240)
│   │   ├── input_layernorm, post_attention_layernorm
│   │   ├── pre_feedforward_layernorm, post_feedforward_layernorm
│   ├── norm: RMSNorm(2560)
│   ├── rotary_emb
├── lm_head: Linear(2560 → 262208)      ← tied with embed_tokens
```

### T5Gemma2 (Encoder-Decoder)

```
T5Gemma2ForConditionalGeneration
├── model
│   ├── encoder (T5Gemma2EncoderModel)
│   │   ├── text_model (T5Gemma2TextEncoder)
│   │   │   ├── embed_tokens: (262144, 2560)
│   │   │   ├── layers: 34 × T5Gemma2EncoderLayer
│   │   │   │   ├── self_attn (T5Gemma2SelfAttention)    ← Standard self-attn
│   │   │   │   │   ├── q/k/v/o_proj (same shapes as Gemma3)
│   │   │   │   │   ├── q_norm, k_norm: RMSNorm(256)
│   │   │   │   ├── mlp (same structure as Gemma3)
│   │   │   │   ├── pre/post_self_attn_layernorm
│   │   │   │   ├── pre/post_feedforward_layernorm
│   │   │   ├── norm, rotary_emb
│   │   │
│   │   ├── vision_tower (SiglipVisionModel)              ← Vision encoder
│   │   │   ├── embeddings: Conv2d(3→1152, 14×14) + Embedding(4096, 1152)
│   │   │   ├── encoder: 27 × SiglipEncoderLayer
│   │   │   │   ├── self_attn: q/k/v/out_proj (1152, 1152)
│   │   │   │   ├── mlp: fc1(1152→4304) + fc2(4304→1152)
│   │   │   ├── post_layernorm
│   │   ├── multi_modal_projector
│   │       ├── mm_soft_emb_norm: RMSNorm(1152)
│   │
│   ├── decoder (T5Gemma2Decoder)
│   │   ├── embed_tokens: (262144, 2560)                  ← tied with encoder
│   │   ├── layers: 34 × T5Gemma2DecoderLayer
│   │   │   ├── self_attn (T5Gemma2MergedAttention)       ← MERGED attn!
│   │   │   │   ├── q/k/v/o_proj (same shapes)
│   │   │   │   ├── q_norm, k_norm
│   │   │   ├── mlp (same structure)
│   │   │   ├── pre/post_self_attn_layernorm
│   │   │   ├── pre/post_feedforward_layernorm
│   │   ├── norm, rotary_emb
│
├── lm_head (T5Gemma2LMHead)
│   ├── out_proj: Linear(2560 → 262144)                   ← tied with embed_tokens
```

---

## 4. Parameter Breakdown

| Komponen | Gemma 3 | T5Gemma2 |
|---|---:|---:|
| **Encoder (text)** | — | ~3.88B |
| **Encoder (vision/SigLIP)** | — | ~0.40B |
| **Decoder** | ~3.88B | ~3.88B |
| **Embeddings** | ~671M (shared) | ~671M (shared/tied) |
| **TOTAL** | **3.88B** | **7.51B** |

> [!IMPORTANT]
> **T5Gemma2 decoder = Gemma 3 (text) secara arsitektural.** Layer shapes, hidden dims, head counts — semua identik. Perbedaan utama hanya pada **merged attention** di decoder (self-attn + cross-attn jadi satu module) dan tambahan encoder.

---

## 5. LoRA Target Modules (Projection Layers)

### Gemma 3

| Layer Pattern | Shape |
|---|---|
| `model.layers.X.self_attn.q_proj` | (2048, 2560) |
| `model.layers.X.self_attn.k_proj` | (1024, 2560) |
| `model.layers.X.self_attn.v_proj` | (1024, 2560) |
| `model.layers.X.self_attn.o_proj` | (2560, 2048) |
| `model.layers.X.mlp.gate_proj` | (10240, 2560) |
| `model.layers.X.mlp.up_proj` | (10240, 2560) |
| `model.layers.X.mlp.down_proj` | (2560, 10240) |

### T5Gemma2

Sama persis dengan Gemma 3, tetapi **duplikasi** di encoder dan decoder:

| Layer Pattern | Shape | Note |
|---|---|---|
| `model.encoder.text_model.layers.X.self_attn.{q,k,v,o}_proj` | Same as G3 | Encoder text |
| `model.decoder.layers.X.self_attn.{q,k,v,o}_proj` | Same as G3 | Decoder (merged attn) |
| `model.encoder.text_model.layers.X.mlp.{gate,up,down}_proj` | Same as G3 | Encoder FFN |
| `model.decoder.layers.X.mlp.{gate,up,down}_proj` | Same as G3 | Decoder FFN |
| `model.encoder.vision_tower.encoder.layers.X.self_attn.{q,k,v,out}_proj` | (1152, 1152) | Vision SigLIP |

> [!TIP]
> **Untuk LoRA pure-text fine-tuning T5Gemma2**, target modules yang valid:
> ```python
> target_modules = ["q_proj", "k_proj", "v_proj", "o_proj", 
>                   "gate_proj", "up_proj", "down_proj"]
> ```
> Ini akan match **baik encoder maupun decoder** layers secara otomatis. Vision tower layers (`out_proj` dari SigLIP) juga match — pertimbangkan untuk freeze via `modules_to_save` exclusion jika pure text.

---

## 6. Merged Attention (Temuan Kritis)

```
Gemma 3 decoder:    Gemma3Attention       (standard self-attention)
T5Gemma2 encoder:   T5Gemma2SelfAttention (standard self-attention)  
T5Gemma2 decoder:   T5Gemma2MergedAttention ← MERGED!
```

**T5Gemma2MergedAttention** menggabungkan self-attention dan cross-attention decoder menjadi **satu modul tunggal**:
- K dan V dibentuk dengan **mengkonkatenasi** encoder outputs dan decoder hidden states
- Masking mempertahankan causal visibility untuk decoder dan bidirectional untuk encoder context
- **Tidak ada** module terpisah seperti `EncDecAttention` / `encoder_attn` / `cross_attn`

> [!NOTE]
> Ini dikonfirmasi oleh output: `T5Gemma2: NO separate cross-attention (merged attention confirmed)`

---

## 7. Embedding & Weight Tying

| Aspek | Gemma 3 | T5Gemma2 |
|---|---|---|
| **Encoder embed** | — | `model.encoder.text_model.embed_tokens` (262144, 2560) |
| **Decoder embed** | `model.embed_tokens` (262**208**, 2560) | `model.decoder.embed_tokens` (262144, 2560) |
| **LM Head** | `lm_head` Linear (262208, 2560) | `lm_head.out_proj` Linear (262144, 2560) |
| **Tied?** | embed ↔ lm_head | encoder_embed ↔ decoder_embed ↔ lm_head (3-way tie) |
| **Embedding class** | `Gemma3TextScaledWordEmbedding` | `T5Gemma2TextScaledWordEmbedding` |

> [!WARNING]
> **Gemma 3 embedding = 262.208 rows, T5Gemma2 = 262.144 rows.**  
> Gemma 3 menambahkan 64 extra padding slots (262.208 - 262.144 = 64). Ini mungkin untuk memory alignment di training GPU Google. T5Gemma2 tidak memiliki padding ini — vocab size persis sama dengan tokenizer.

---

## 8. Normalization Layer Naming

Ada perbedaan penamaan layer normalization:

| Posisi | Gemma 3 | T5Gemma2 |
|---|---|---|
| Pre-attention | `input_layernorm` | `pre_self_attn_layernorm` |
| Post-attention | `post_attention_layernorm` | `post_self_attn_layernorm` |
| Pre-FFN | `pre_feedforward_layernorm` | `pre_feedforward_layernorm` ✅ |
| Post-FFN | `post_feedforward_layernorm` | `post_feedforward_layernorm` ✅ |
| Final | `norm` | `norm` ✅ |

---

## 9. Kesimpulan Kunci

1. **T5Gemma2 decoder ≡ Gemma 3 secara arsitektural** — shapes identik, hanya attention yang di-merge
2. **T5Gemma2 ≈ 2× Text Gemma 3 + 1× SigLIP** — Encoder Text (3.88B) + Vision (0.40B) + Decoder (3.88B) = 7.51B. Komponen ini identik dengan yang ada di Gemma 3.
3. **LoRA target modules `[q_proj, k_proj, v_proj, o_proj, gate_proj, up_proj, down_proj]` valid** untuk kedua model — shapes identik
4. **Merged attention** berarti cross-attention "gratis" — tidak ada parameter tambahan vs self-attention saja
5. **Embedding size berbeda** (262.208 vs 262.144) — harus hati-hati jika ingin transfer weights antar model
