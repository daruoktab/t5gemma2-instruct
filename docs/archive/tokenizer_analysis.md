# Analisis Perbandingan Tokenizer: Gemma 3 vs Gemma 4 vs T5Gemma2

**Tanggal:** 12 Mei 2026  
**Sumber data:** File `tokenizer.json` dari masing-masing model di [data/](file:///d:/Codings/unsloth/t5-gemma-2/instruct/data)

---

## 1. Ringkasan Metrik Utama

| Metrik | Gemma 3 | Gemma 4 | T5Gemma2 |
|---|---:|---:|---:|
| **Vocab size** (model.vocab) | 262.144 | 262.144 | 262.144 |
| **Total added_tokens** | 6.415 | **24** | 6.414 |
| **Unused tokens** | 6.242 | **0** | 6.241 |
| **Non-unused added tokens** | 173 | 24 | 173 |
| **Merges count** | 514.906 | 514.906 | 514.906 |
| **Model type** | BPE | BPE | BPE |

> [!IMPORTANT]
> **Gemma 4 secara radikal berbeda** — hanya memiliki 24 added tokens, tanpa unused tokens sama sekali, dan sepenuhnya mengubah paradigma formatting dibanding Gemma 3 / T5Gemma2.

---

## 2. Gemma 4: Revolusi Token Format

Gemma 4 **membuang** seluruh infrastructure lama Gemma 3 dan mengganti semuanya:

### Token Format Baru (Pipe-style `<|...|>`)

| ID | Token | Fungsi |
|---:|---|---|
| 46 | `<\|tool\>` | Pembuka definisi tool |
| 47 | `<tool\|\>` | Penutup definisi tool |
| 48 | `<\|tool_call\>` | Pembuka tool call |
| 49 | `<tool_call\|\>` | Penutup tool call |
| 50 | `<\|tool_response\>` | Pembuka tool response |
| 51 | `<tool_response\|\>` | Penutup tool response |
| 52 | `<\|"\|\>` | Delimiter string literal dalam JSON |
| 98 | `<\|think\|\>` | Chain-of-thought / thinking token |
| 100 | `<\|channel\>` | Pembuka channel |
| 101 | `<channel\|\>` | Penutup channel |
| 105 | `<\|turn\>` | Pembuka turn (**menggantikan** `<start_of_turn>`) |
| 106 | `<turn\|\>` | Penutup turn (**menggantikan** `<end_of_turn>`) |

### Token Multimodal Baru

| ID | Token | Fungsi |
|---:|---|---|
| 255.999 | `<\|image\>` | Pembuka image (menggantikan `<start_of_image>`) |
| 256.000 | `<\|audio\>` | **BARU**: Pembuka audio |
| 258.880 | `<\|image\|\>` | Penutup image |
| 258.881 | `<\|audio\|\>` | Penutup audio |
| 258.882 | `<image\|\>` | Image content marker |
| 258.883 | `<audio\|\>` | Audio content marker |
| 258.884 | `<\|video\|\>` | **BARU**: Video token |

### Yang Dihapus dari Gemma 4

```
DIHAPUS:
  ❌ <start_of_turn> / <end_of_turn>     → diganti <|turn> / <turn|>
  ❌ <start_of_image> / <end_of_image>    → diganti <|image> / <|image|>
  ❌ <image_soft_token>                   → diganti <image|>
  ❌ [multimodal]                         → dihapus
  ❌ 6.242 unused tokens                  → DIHAPUS SEMUA
  ❌ 30 whitespace tokens (▁▁ → ▁×31)    → dihapus dari added_tokens
  ❌ 64 HTML tokens (<table>, <td>, dll)  → dihapus dari added_tokens
  ❌ 47 code tokens (<code_X>)            → dihapus dari added_tokens
```

> [!NOTE]
> **DITAMBAHKAN** di Gemma 4: `<|think|>` (thinking/CoT token), `<|audio>`, `<|video|>`, dan 6 token tool-calling. Ini menunjukkan Gemma 4 dirancang natively untuk agentic + multimodal use cases.

---

## 3. Gemma 3 vs T5Gemma2: Hanya 1 Perbedaan

Kedua tokenizer ini **hampir identik** — BPE vocab identik, merges identik, 173 non-unused added tokens identik.

### Satu-satunya Perbedaan: `<unused99>` → `<image_soft_token>`

```
Gemma 3:
  id 256.001 = <unused99>     (slot unused biasa)

T5Gemma2:
  id 256.001 = <image_soft_token>  (per-patch vision token)
```

### Dampak pada Unused Token Blocks

| | Gemma 3 | T5Gemma2 |
|---|---|---|
| **Blok 1** | id 6–104 (99 token): `<unused0>`–`<unused98>` | id 6–104 (99 token): `<unused0>`–`<unused98>` |
| **Blok 2** | id 256.001–262.143 (**6.143** token): `<unused99>`–`<unused6241>` | id 256.002–262.143 (**6.142** token): `<unused100>`–`<unused6241>` |
| **Total** | **6.242** | **6.241** |

T5Gemma2 "mencuri" satu slot unused (`<unused99>`) dan menggantinya dengan `<image_soft_token>` untuk keperluan vision processing.

---

## 4. BPE Vocab & Merges: Backbone Identik

```
✅ Merges:   Gemma 3 = Gemma 4 = T5Gemma2  (514.906 merges, identik)
✅ BPE Vocab: Gemma 3 ≈ T5Gemma2 (perbedaan hanya <unused99> ↔ <image_soft_token>)
```

### Gemma 3 vs Gemma 4: 19 Token Swap di BPE Vocab

Gemma 4 menghapus 15 token unused tinggi (`<unused6227>`–`<unused6241>`) + 4 token Gemma-style (`<start_of_turn>`, `<end_of_turn>`, `<start_of_image>`, `<end_of_image>`) dari BPE vocab dan menggantinya dengan 19 token baru pipe-style.

```diff
 Hanya di Gemma 3 BPE vocab (19):
-  <end_of_image>, <end_of_turn>, <start_of_image>, <start_of_turn>
-  <unused6227> sampai <unused6241> (15 token)

 Hanya di Gemma 4 BPE vocab (19):
+  <|turn>, <turn|>, <|tool>, <tool|>, <|tool_call>, <tool_call|>
+  <|tool_response>, <tool_response|>, <|"|>, <|think|>
+  <|image>, <|image|>, <image|>, <|audio>, <|audio|>, <audio|>
+  <|channel>, <channel|>, <|video|>
```

---

## 5. Implikasi untuk Training Pipeline T5Gemma2

### ✅ Validasi Strategi Suppress yang Sudah Ada

Analisis ini **mengkonfirmasi** daftar suppress di [training_strategy_t5gemma2_12May2026.md](file:///d:/Codings/unsloth/t5-gemma-2/instruct/docs/training_strategy_t5gemma2_12May2026.md):

```python
SUPPRESS_IDS = list(range(6, 105))           # Blok 1: <unused0>–<unused98> (99 token)
SUPPRESS_IDS += list(range(256002, 262144))  # Blok 2: <unused100>–<unused6241> (6.142 token)
SUPPRESS_IDS += [255999, 256000, 256001]     # Vision: <start_of_image>, <end_of_image>, <image_soft_token>
# Total: 6.244 token
```

### ⚠️ Pertimbangan Migrasi ke Gemma 4 Style (Masa Depan)

Jika di masa depan ingin migrasi ke format Gemma 4:
- Format prompt berubah dari `<start_of_turn>user` / `<end_of_turn>` → `<|turn>user` / `<turn|>`
- Gemma 4 **tidak punya unused tokens** → masalah suppress hilang sepenuhnya
- Namun T5Gemma2 tetap berbasis Gemma 3 tokenizer, jadi format lama tetap yang benar untuk sekarang.

### 🔑 Temuan Kunci

1. **T5Gemma2 dan Gemma 3 berbagi DNA tokenizer yang sama** — perbedaan hanya 1 token
2. **Gemma 4 adalah redesign total** — bukan evolusi dari Gemma 3
3. **BPE merges identik di ketiga model** — berarti subword splitting behavior identik untuk teks biasa
4. **Gemma 4 memperkenalkan thinking token** (`<|think|>`) — relevan jika ingin menambahkan CoT capability
5. **Unused tokens hanya ada di Gemma 3 & T5Gemma2** — masalah ini sudah solved di arsitektur Gemma 4
