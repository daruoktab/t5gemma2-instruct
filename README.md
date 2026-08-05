# 🇮🇩 T5-Gemma-2 Instruct & Multimodal Chat Pipeline (Bahasa Indonesia)

Pipeline instruksi dan percakapan (instruction-tuning, multimodal vision, & chat) berbasis arsitektur **Encoder-Decoder (Seq2Seq)** menggunakan **T5-Gemma-2 (4B & 270M)** yang dioptimalkan secara penuh menggunakan metode **Supervised Fine-Tuning (SFT)**, **Joint Multimodal Training (V7)**, dan **Odds Ratio Preference Optimization (ORPO)** via Unsloth Seq2Seq.

---

## 🚀 Deskripsi Proyek (Project Overview)

Proyek ini dirancang untuk melatih dan menguji model **T5-Gemma-2** agar optimal dalam memahami dan merespons instruksi, percakapan teks, serta pemahaman visual (multimodal) dalam **Bahasa Indonesia** (prioritas utama) dan **Bahasa Inggris** (bilingual).

### 💡 Mengapa Memilih Arsitektur Encoder-Decoder (Seq2Seq)?

Di tengah dominasi mutlak arsitektur *decoder-only* pada standar industri chatbot modern, penggunaan model *encoder-decoder* (Seq2Seq) untuk *instruct & conversational agent* tergolong sangat langka dan menjadi subjek eksplorasi yang menarik. Proyek ini dibangun di atas rasa penasaran ilmiah (*research curiosity*) untuk menguji potensi penuh arsitektur Seq2Seq multimodal modern (**T5-Gemma-2**) pada Bahasa Indonesia:

1. **Pemrosesan Input Kompleks & Asimetris (Hard Tasks)**:
   - **Deep Context Processing**: Encoder memproses seluruh konteks masukan (teks instruksi + SigLIP 256 soft tokens) secara *bidirectional* penuh tanpa batas kausalitas. Hal ini memaksa jaringan untuk mencerna dan memahami struktur input yang kompleks/panjang sebelum Decoder memproduksi keluaran.
   - Sangat unggul untuk tugas-tugas *asymmetric* (input panjang/kompleks $\to$ respons terstruktur) seperti peringkasan (*summarization*), penerjemahan (*translation*), analisis dokumen visual, dan *grounded QA*.

2. **Verifikasi Intent Implisit via Task Prefixing (`<unused1>` – `<unused6>`)**:
   - Tanpa perlu membebani prompt user dengan awalan kaku, Decoder dilatih untuk secara mandiri memunculkan **Task Prefix** di awal generasi menggunakan 6 *Unused Tokens* khusus (ID 7 hingga 12).
   - **Tabel Pemetaan Task Prefix (Unused Tokens 1–6)**:
     | Token ID | Unused Token | Kategori Task | Deskripsi Fungsi |
     |---|---|---|---|
     | `7` | `<unused1>` | **SUMMARIZE** | Peringkasan teks & ekstraksi poin utama |
     | `8` | `<unused2>` | **TRANSLATE** | Penerjemahan antar bahasa (Indo ↔ Eng) |
     | `9` | `<unused3>` | **NER** | Ekstraksi entitas (*Named Entity Recognition*) |
     | `10` | `<unused4>` | **QA** | Tanya-Jawab berbasis dokumen (*Grounded QA*) |
     | `11` | `<unused5>` | **PARAPHRASE** | Penulisan ulang & penyuntingan gaya bahasa |
     | `12` | `<unused6>` | **GENERAL_CHAT** | Percakapan umum & dialog interaktif (*Casual Chat*) |
   - Mekanisme ini berfungsi sebagai **verifikasi intent eksplisit**: Kita dapat mengonfirmasi apakah internal state model benar-benar memahami jenis tugas yang diperintahkan sebelum ia menghasilkan respons utama.

3. **Optimasi Arsitektural Google T5Gemma 2**:
   - **Merged Attention**: Menyatukan Self-Attention dan Cross-Attention di dalam decoder ($K, V = [X; H]$), mengeliminasi modul terpisah dan menghemat beban komputasi.
   - **Tied Embeddings**: Berbagi matriks token embedding yang identik antara Encoder, Decoder, dan LM Head ($262.144$ vocab size) untuk penyelarasan ruang representasi input-output.

---

## 🏗️ Detail Arsitektur: Gemma 3 vs T5Gemma2

T5-Gemma-2 dibangun melalui adaptasi model Gemma 3 menggunakan metode UL2.

| Aspek | Gemma 3 (4B) | T5Gemma2 (4B-4B) |
| :--- | :--- | :--- |
| **Arsitektur** | Multimodal Decoder-only | **Multimodal Encoder-Decoder** |
| **Model Type** | `gemma3` | `t5gemma2` |
| **Total Parameter** | **~4.28B** (3.88B text + 0.4B vision) | **~7.51B** (3.88B enc + 3.88B dec + 0.4B vis) |
| **Attention** | Standard Self-Attention | **Merged Attention** (Self + Cross) |
| **Vision Tower** | SigLIP (Hidden 1152) | SigLIP (Hidden 1152) — Identik |
| **Tied Embeddings** | ✅ Yes (Embed ↔ Head) | ✅ Yes (Enc ↔ Dec ↔ Head) |
| **Vocab Size** | 262,208 (extra 64 padding) | **262,144** (exact) |

### Mekanisme Merged Attention
T5-Gemma-2 tidak memiliki modul `cross_attention` terpisah di decoder. Sebagai gantinya, ia menggunakan **Merged Attention**:
- **Query (Q)**: Dibentuk dari decoder hidden states ($X$).
- **Key (K) & Value (V)**: Dibentuk dengan mengkonkatenasi decoder input ($X$) dan encoder output ($H$) secara sekuensial: $[X; H]$.
- **Masking**: Kombinasi bidirectional (untuk token encoder $H$) dan causal (untuk token decoder $X$).

---

## 🎯 Dataset & Model Checkpoints (HuggingFace Hub)

### 📦 Dataset Repositories (`daruokta`):
1. **[`daruokta/t5gemma2-indonesia-chat-formatted`](https://huggingface.co/datasets/daruokta/t5gemma2-indonesia-chat-formatted)**:
   - `chat_sft`: **36.015** baris SFT flattened (181.7 MB).
   - `chat_multiturn`: **3.000** percakapan utuh unflattened (23.9 MB).
   - `chat_orpo`: **1.000** pasang preferensi teks.
   - `indoqa_sft` & `indoqa_documents`: **3.309** pasang Q&A dokumen.
2. **[`daruokta/t5gemma2-indonesia-vision-formatted`](https://huggingface.co/datasets/daruokta/t5gemma2-indonesia-vision-formatted)**:
   - `vision_sft`: **1.000** percakapan multimodal utuh (3,04 GB, ekuivalen 3.964 assistant turns).
   - `vision_orpo`: **200** pasang preferensi vision (510.4 MB).
3. **[`daruokta/t5-gemma-2-multimodal-embedding`](https://huggingface.co/datasets/daruokta/t5-gemma-2-multimodal-embedding)**:
   - Subset NLI, Parallel Translation (2.2M rows), Retrieval, STS, dan Vision STS.

### 🤖 Model Checkpoints:
- **Joint Multimodal V7 Checkpoint:** [`daruokta/t5gemma-2-4b-4b-instruct-chat-indo-v7-joint-unsloth`](https://huggingface.co/daruokta/t5gemma-2-4b-4b-instruct-chat-indo-v7-joint-unsloth)
- **Vision Cangkok Base Checkpoint:** [`daruokta/t5gemma-2-4b-4b-instruct-chat-indo-v4-vision-cangkok`](https://huggingface.co/daruokta/t5gemma-2-4b-4b-instruct-chat-indo-v4-vision-cangkok)

---

## 🦥 Integrasi Unsloth (Patch Seq2Seq / T5-Gemma-2)

Untuk mempercepat pelatihan model **T5-Gemma-2** dan efisiensi VRAM, proyek ini menggunakan **Unsloth** branch Seq2Seq (`dh/recover-3153-seq2seq`) dengan patch wajib untuk environment modern:

### Patch Kustom:
1. **NameError Fix (`_utils.py`)**: Penambahan import dinamis `auto_docstring` dan `strict`.
2. **PEFT Import Fix (`import_utils.py`)**: Penanganan `is_gptqmodel_available()` pada PEFT 0.19+.
3. **Encoder-Decoder Batch Sampler Fix (`_utils.py`)**: Bypass `_unsloth_get_batch_samples` jika `is_encoder_decoder == True` untuk mencegah `RuntimeError` mismatch dimensi token input vs label.

```bash
pip install --force-reinstall --no-deps "git+https://github.com/unslothai/unsloth.git@dh/recover-3153-seq2seq"
```

---

## 📂 Struktur Direktori Proyek

```directory
instruct/
├── app/                                        # Aplikasi Web Chat Simulator (Flask)
│   ├── server.py                               # Backend Flask
│   └── templates/                              # Antarmuka Web
├── data/                                       # Dataset Lokal (SFT, Multimodal, Preference)
├── docs/                                       # Knowledge Base Utama
│   ├── README.md                               # Master Index Dokumen
│   ├── 01_architecture/                        # Spesifikasi Arsitektur & Merged Attention
│   ├── 02_guides/                              # Panduan Finetuning Unsloth, ORPO, & Dataset
│   ├── 03_research_and_diagnostics/            # Riset V7 Combined, Vision Cangkok, & Logit Masking
│   ├── 04_specifications/                      # API Reference, Paper Guide, & HTML Dashboard
│   ├── paper/                                  # Reference PDFs (ArXiv)
│   └── sessions/                               # Catatan Sesi Riset Harian
├── notebooks/                                  # Notebooks Pelatihan (Marimo / Cloud)
│   ├── working-molab-v7-combined-unsloth.py    # PIPELINE UTAMA ACTIVE (Text + Vision Joint SFT)
│   └── legacy/                                 # Archive Notebooks (V3 s.d. V6-Vision)
├── scratch/                                    # Skrip Utility & Audit Eksperimen
├── inference.py                                # Skrip Inferensi CLI Utama
└── README.md                                   # Dokumentasi Utama Repo Ini
```

---

## 💬 Menjalankan Web Chat Simulator

1. Pastikan environment Python aktif.
2. Jalankan server Flask:
   ```powershell
   python app/server.py
   ```
3. Akses antarmuka di browser: `http://127.0.0.1:5000`

---
*Proyek ini dikembangkan secara aktif untuk eksplorasi arsitektur Seq2Seq Multimodal pada LLM generasi baru (T5-Gemma-2) dalam konteks Bahasa Indonesia.*
