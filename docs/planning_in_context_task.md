# Planning: In-Context Task & Enhancement Dataset

## Latar Belakang & Tujuan
Tujuan dari perencanaan ini adalah untuk mengimplementasikan *implicit task switching* pada model T5/Gemma-2. Model akan bertindak sebagai *router* bagi dirinya sendiri dengan memprediksi **Task Tokens** (menggunakan *unused tokens*) di awal responsnya, berdasarkan pembelajaran dari paper-paper terbaru (T5, Gemma-2, Gemma-3).

## Mekanisme "Latent Routing" (Target-side Task Token)
Alih-alih menyuruh *user* untuk secara eksplisit memberikan token task di awal *prompt* (Encoder), kita menempatkan token task di *Decoder* (respons Asisten). 
Alur kerjanya:
1. User memberikan instruksi natural (misal: "Tolong terjemahkan teks ini...").
2. Model meng-generate token pertama berupa `<unused_X>` yang mewakili mode *Translation*.
3. Model melanjutkan generasi teks terjemahan.
Keuntungannya: Model secara mandiri ("tanpa dibilang udah paham sendiri") mengenali tugas dan mendeklarasikan statusnya sebelum menjawab.

## Rencana Langkah Eksekusi (Untuk Dikerjakan Nanti)

### 1. Definisi Mapping Token Task
Kita akan mengalihfungsikan sebagian token dari daftar yang sebelumnya di-*suppress*:
- `<unused1>`: Mode Translation (Terjemahan)
- `<unused2>`: Mode Summarization (Ringkasan)
- `<unused3>`: Mode Paraphrasing / Rewriting
- `<unused4>`: Mode General Q&A

### 2. Modifikasi Data Pipeline (Enhancement Dataset)
Pada script pembentuk data (misalnya `generate_synthetic_dpo.py` atau fungsi `format_encoder_from_raw`), tambahkan logika untuk:
- Menganalisis kalimat *user* di turn terakhir (menggunakan regex, keyword, atau LLM otomatis).
- Jika instruksi masuk ke kategori *Translate*, sisipkan `<unused1>` tepat di awal teks `chosen` dan `rejected`.
- Contoh: `"<start_of_turn>model\n<unused1> Tentu, berikut terjemahannya..."`

### 3. Penyesuaian Training Script
- Token-token task yang dipilih (`<unused1>` s.d `<unused4>`) **wajib dikeluarkan** dari set `ALL_SUPPRESS_IDS`. Hal ini agar gradien dan *embedding*-nya bisa di-update oleh optimizer (model bisa belajar merepresentasikan task tersebut).
- Melakukan iterasi *fine-tuning* (SFT/DPO) menggunakan dataset baru ini.

---
*Dokumen ini dibuat sebagai kerangka acuan dan referensi untuk implementasi di sesi pekerjaan berikutnya.*
