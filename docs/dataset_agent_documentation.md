# Dokumentasi Pembuatan Dataset Berbasis Agen (Dataset Generation Pipeline)

Dokumen ini memetakan seluruh skrip pembuatan, augmentasi, dan kurasi dataset yang ditenagai oleh AI/LLM (Agentic Pipeline) di dalam folder `scripts/dataset/`.

## 📌 Ringkasan Skrip & Fungsinya

Berikut adalah daftar file yang ada di direktori kerja (berdasarkan status Git yang belum di-commit) beserta fungsi utamanya:

### 1. `generate_prefix_tasks.py`
- **Fungsi:** Membuat percakapan *multi-turn* (seperti 2500 data chat) di mana agen (Pydantic AI) **dipaksa** untuk mengeluarkan *Task Prefix* berbasis `<unusedX>` tokens sebelum memberikan jawaban.
- **Mekanisme:** Menggunakan `TaskType(str, Enum)` dan memaksakan *tool calling* pada agen Pydantic AI. Agen secara iteratif memanggil alat `append_turn_pair` dan dipaksa menyisipkan *intent* sebelum menulis teks.
- **Output:** `data/generated_prefix_tasks_agentic.jsonl`

### 2. `generate_from_unused_topics.py`
- **Fungsi:** Versi pendahulu/awal dari generator *multi-turn* agentic. Mirip dengan `generate_prefix_tasks.py` namun menghasilkan percakapan bahasa natural tanpa ada suntikan *Unused Tokens* di setiap giliran (turn).
- **Output:** `data/generated_from_unused_topics_agentic.jsonl`

### 3. `generate_indoqa_v2.py`
- **Fungsi:** Membuat dataset Grounded QA baru (*IndoQA v2*) dengan fokus pada domain-domain kompleks di luar sejarah/geografi (seperti dokumen Hukum, Berita, Bisnis & Keuangan, dan Teknis IT).
- **Mekanisme (Menimpa atau Menambah?):** Skrip ini **MENAMBAH (Append)**. Ia memiliki logika pengecekan atau *resume* bawaan. Jika file `indoqa_v2_dataset.jsonl` sudah ada, ia akan menghitung berapa baris yang sudah berhasil dicetak, lalu melanjutkan *append* data baru di baris terbawah hingga mencapai `--target` yang diinginkan. **Skrip ini tidak menghapus data lama.**
- **Output:** `data/indoqa_v2_dataset.jsonl`

### 4. `augment_indoqa_targets.py`
- **Fungsi:** Meningkatkan kualitas dataset IndoQA yang sudah ada. Skrip ini melacak jawaban-jawaban yang terlalu singkat (di bawah 15 kata), lalu meminta agen LLM untuk memperpanjang dan membuatnya lebih informatif (2-3 kalimat) tanpa menambahkan halusinasi fakta dari konteks aslinya.
- **Output:** `data/indoqa_train_augmented.jsonl`

### 5. `generate_longform_dataset.py`
- **Fungsi:** Secara khusus men-*generate* instruksi dan output berbahasa Indonesia yang sangat panjang (Long-form, 300-800 kata). Sangat berguna agar model akhir tidak malas menulis.
- **Kategori:** Artikel opini, panduan teknis, ulasan/review mendalam, penjelasan konsep, laporan eksekutif.
- **Output:** `data/longform_output_dataset.jsonl`

### 6. `expand_dpo_dataset.py`
- **Fungsi:** Membuat dataset *Direct Preference Optimization* (DPO). Mengambil pertanyaan dari *train set* yang sudah ada, lalu menyuruh agen untuk membuat pasangan jawaban *Chosen* (terbaik) dan *Rejected* (buruk).
- **Mekanisme Cacat (Flaw Types):** Agen secara sengaja diperintahkan membuat respon *rejected* dengan tipe cacat terstruktur: `echo_user` (mengulang pertanyaan), `vague` (basa-basi), `hallucination` (mengarang bebas), `incomplete` (terpotong), `off_topic`, dan `overlong` (bertele-tele).
- **Output:** `data/preferences_dpo_expanded.jsonl`

### 7. `fix_dataset_quality.py`
- **Fungsi:** Alat pembersih (*Data Janitor*). Skrip non-LLM yang memindai semua file JSONL, mencari baris *error*, JSON rusak, mendeteksi jawaban yang benar-benar kosong, atau mendeteksi duplikat absolut. Setelah di-*run*, ia akan menyaring dan menimpa file aslinya dengan versi yang sudah bersih.

---

## ⚙️ Paradigma Prompting Agen

Sebagian besar agen dataset ini menggunakan strategi **System Prompt Dinamis**. 
- Pada tahap SFT dan DPO, prompt dipecah menjadi *Core Prompt* (karakter agen dan aturan absolut format JSON/Tool Calling) dan *Dynamic Context Prompt* (menyisipkan variabel acak seperti domain teks atau topik spesifik).
- Pendekatan ini (terutama di `generate_prefix_tasks.py` dengan Pydantic AI) menjamin *Zero-Shot formatting* berjalan 99% akurat tanpa *parsing error* karena struktur output (JSON/Tool) dikontrol di level API.
