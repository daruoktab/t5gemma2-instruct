# Master Dataset Specification & Dataset Agent Documentation

**Last Updated:** 5 Agustus 2026  
**Scope:** Spesifikasi lengkap dataset Hugging Face (`daruokta`), dataset lokal (`data/`), struktur **Flattened vs Unflattened**, format skema, asal-usul data, dan operasional Dataset Agent.

---

## 1. Konsep Arsitektur Data: Flattened vs Unflattened (Multi-Turn)

Dalam ekosistem pelatihan T5-Gemma-2, data percakapan dikelola dalam 2 format struktural utama:

```
                              Raw Multi-Turn Conversation
                                           │
                 ┌─────────────────────────┴─────────────────────────┐
                 ▼                                                   ▼
     Unflattened Format (Full Chat)                     Flattened Format (Turn-by-Turn Pair)
- 1 Row = 1 Percakapan Utuh                        - 1 Row = 1 Pasangan Input/Target (Turn)
- Field: `messages` (Array of role/content)        - Field: `input` (History) & `target` (Response)
- Digunakan untuk: SFT Seq2Seq / Chat Trainer      - Digunakan untuk: Step-wise Prefix Training
```

### Ringkasan Perbandingan Formatan Dataset:

| Nama Dataset / Config | Repositori HF / Path Lokal | Total Baris | Tipe Formatan | Jumlah Percakapan Utuh | Rata-rata Turn per Chat |
|---|---|---|---|---|---|
| `chat_multiturn` | `daruokta/t5gemma2-indonesia-chat-formatted` | **3.000** | **Unflattened** | 3.000 percakapan | ~12.4 turn |
| `chat_sft` | `daruokta/t5gemma2-indonesia-chat-formatted` (`data/sft/chat_train_v2.jsonl`) | **36.015** | **Flattened** | 2.900 percakapan unik | **12.42 turn/chat** |
| `vision_sft` | `daruokta/t5gemma2-indonesia-vision-formatted` (`data/multimodal/train_vision.jsonl`) | **1.000** | **Unflattened** | 1.000 percakapan vision | **3.96 assistant turn/chat** (total 3.964 turn) |
| `chat_orpo` | `daruokta/t5gemma2-indonesia-chat-formatted` (`data/preference/orpo_train.jsonl`) | **1.000** | **Pairwise Preference** | 1.000 pasang | Single/Multi-turn prompt |
| `vision_orpo` | `daruokta/t5gemma2-indonesia-vision-formatted` (`data/preference/orpo_multimodal.jsonl`) | **200** | **Multimodal Preference** | 200 pasang | Single/Multi-turn vision |
| `indoqa_documents` | `daruokta/t5gemma2-indonesia-chat-formatted` | **3.309** | **Unflattened** | 3.309 dokumen Q&A | 1 turn |
| `indoqa_sft` | `daruokta/t5gemma2-indonesia-chat-formatted` | **3.309** | **Flattened** | 3.309 dokumen Q&A | 1 turn |

---

## 2. Ikhtisar Repositori Dataset Hugging Face (`daruokta`)

Proyek T5-Gemma-2 menggunakan 3 repositori utama di Hugging Face Hub:

| Repositori Hugging Face | Terakhir Diperbarui | Total Commits | Jenis Data | Total Ukuran |
|---|---|---|---|---|
| [`daruokta/t5gemma2-indonesia-chat-formatted`](https://huggingface.co/datasets/daruokta/t5gemma2-indonesia-chat-formatted) | 4 Juli 2026 | 43 | Text Chat SFT, Multiturn, ORPO, IndoQA | ~220 MB |
| [`daruokta/t5gemma2-indonesia-vision-formatted`](https://huggingface.co/datasets/daruokta/t5gemma2-indonesia-vision-formatted) | 12 Juli 2026 | 4 | Vision SFT & Multimodal ORPO | ~3.55 GB |
| [`daruokta/t5-gemma-2-multimodal-embedding`](https://huggingface.co/datasets/daruokta/t5-gemma-2-multimodal-embedding) | 26 Juli 2026 | 31 | NLI, Classification, Parallel, Retrieval, STS, Vision STS | ~360 MB |

---

## 3. Spesifikasi Detail Dataset Teks (`chat-formatted`)

### A. Config: `chat_sft` (Flattened Format)
- **Ukuran Data:** 36.015 sampel train (181.7 MB), 1.227 sampel validation (6.2 MB).
- **Match File Lokal:** `data/sft/chat_train_v2.jsonl` (36.015 baris). *(Catatan: versi v1 sebelumnya berisi 27.990 baris).*
- **Detail Formatan:** Merupakan hasil **flattening (perataan)** dari 2.900 percakapan multiturn utuh. Setiap giliran asisten dipecah menjadi 1 baris mandiri.
- **Skema Features:**
  - `input` (string): Prompt sistem + histori percakapan user (contoh: `system: Kamu adalah asisten AI...\nuser: ...`)
  - `target` (string): Respons asisten yang diawali dengan task prefix (contoh: `<unused6> Halo!...`)
  - `chat_idx` (int64): ID percakapan utama (menghubungkan turn-turn dari chat yang sama).
  - `turn_idx` (int64): Indeks putaran percakapan (0, 1, 2, dst).
  - `input_tokens` (int64) & `target_tokens` (int64): Jumlah token masukan & keluaran.

### B. Config: `chat_multiturn` (Unflattened Format)
- **Ukuran Data:** 3.000 sampel train (23.9 MB).
- **Detail Formatan:** Merupakan format **unflattened (percakapan utuh)** di mana 1 baris menampung seluruh riwayat pesan percakapan dari awal hingga akhir.
- **Skema Features:** `messages` (List dictionary `{'role': str, 'content': str}`).

### C. Config: `chat_orpo` (Preference Optimization)
- **Ukuran Data:** 1.000 sampel train (3.48 MB).
- **Match File Lokal:** `data/preference/orpo_train.jsonl` (1.000 baris).
- **Skema Features:** `id`, `prompt`, `chosen`, `rejected`, `flaw`, `rationale`.

### D. Config: `indoqa_sft` & `indoqa_documents`
- **Sumber Data:** `jakartaresearch/indoqa`.
- **Ukuran Data:** 3.309 sampel train, 1.104 sampel validation.

---

## 4. Spesifikasi Detail Dataset Vision & Multimodal (`vision-formatted`)

### A. Config: `vision_sft` (Unflattened Multi-Turn Format)
- **Ukuran Data:** 1.000 sampel train (3.04 GB).
- **Match File Lokal:** `data/multimodal/train_vision.jsonl` (1.000 baris).
- **Detail Formatan:** Merupakan format **unflattened (percakapan utuh)**. Setiap baris menyimpan 1 sesi percakapan multimodal lengkap dengan seluruh rangkaian gambar dan riwayat `messages` (rata-rata 3,96 giliran asisten per percakapan, menghasilkan ekuivalen **3.964 total turn**).
- **Sumber Data Asal:** `SEACrowd/sea-vl_crowdsourcing`, `KORIKA-AI/sea-vl_crowdsourcing_id`, dan scraping dokumen visual.
- **Skema Features:**
  - `id` (string/int).
  - `images` (List Image): Gambar beresolusi tinggi (disimpan sebagai PIL Image di parquet / path lokal di jsonl).
  - `messages` (List dict): Menggunakan token placeholder `📷` pada giliran user untuk menandai posisi gambar, dan respons assistant diawali task prefix (seperti `<unused6>`).

### B. Config: `vision_orpo` (Multimodal Preference Optimization)
- **Ukuran Data:** 200 sampel train (510.4 MB).
- **Match File Lokal:** `data/preference/orpo_multimodal.jsonl` (200 baris).
- **Skema Features:** `id`, `images`, `prompt`, `chosen`, `rejected`, `flaw`, `rationale`.

---

## 5. Operasional Dataset Agent & Script Pembersihan

Dataset Agent bertugas menjalankan validasi otomatis sebelum data di-upload ke Hugging Face Hub:

1. **Prefix Validation:** Memastikan respons assistant selalu diawali token prefix khusus (`<unused1>` s.d. `<unused6>`).
2. **User Turn Cleaning:** Menghapus token `<unusedX>` liar di sisi user.
3. **Format Standardizer:** Memasang token placeholder `📷` pada awal pesan user multimodal.
4. **Repetitive & Quality Filter:** Menyaring baris yang memiliki rasio repetisi tinggi atau terpotong di tengah kalimat.
