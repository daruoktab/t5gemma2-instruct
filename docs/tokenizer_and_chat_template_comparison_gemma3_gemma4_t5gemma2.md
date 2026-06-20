# Analisis dan Perbandingan Tokenizer & Chat Template: Gemma 3, Gemma 4, dan T5Gemma-2

Dokumen ini berisi analisis mendalam mengenai kosa kata (vocabulary), token kontrol khusus, dan *chat template* (Jinja) dari keluarga model Gemma 3, Gemma 4, serta kustomisasi template yang kita buat untuk T5Gemma-2. Dokumentasi ini dibuat agar tim pengembang atau AI lain dapat langsung memahami struktur format percakapan dari masing-masing model secara presisi.

---

## 1. Tabel Perbandingan Tokenizer & Token Kontrol

| Parameter | Gemma 3 (PT & IT) | Gemma 4 (PT & IT) | T5Gemma-2 (Model Kita) |
| :--- | :--- | :--- | :--- |
| **Repositori Hub** | `google/gemma-3-4b-pt`<br>`google/gemma-3-4b-it` | `google/gemma-4-31B`<br>`google/gemma-4-31B-it` | `google/t5gemma-2-4b-4b` |
| **Ukuran Kosakata** | 262,144 (Length: 262,145) | 262,144 (Length: 262,144) | 262,144 (Length: 262,145) |
| **Token Batas Turn** | `<start_of_turn>` (ID: 105)<br>`<end_of_turn>` (ID: 106) | `<|turn>` (Start of Turn)<br>`<turn|>` (End of Turn) | `<start_of_turn>` (ID: 105)<br>`<end_of_turn>` (ID: 106) |
| **Token Dasar (IDs)** | `<bos>` = 2, `<eos>` = 1, `<pad>` = 0 | `<bos>` = 2, `<eos>` = 1, `<pad>` = 0 | `<bos>` = 2, `<eos>` = 1, `<pad>` = 0 |
| **Dukungan Audio** | Tidak Ada | **Natif**: `<|audio|>`, `<|audio`, `<audio|>` | Tidak Ada |
| **Dukungan Penalaran**| Tidak Ada | **Natif**: `<|think|>` (CoT / Thinking) | Tidak Ada |
| **Agent / Tool Calling**| Tidak Ada (Text biasa) | **Natif**:<br>`<|tool_call>`, `<tool_call|>`<br>`<|tool>`, `<tool|>`<br>`<|tool_response>`, `<tool_response|>` | Tidak Ada (Text biasa) |
| **Status Chat Template**| Bawaan Hugging Face | Bawaan Hugging Face | **Kustom (Dibuat Manual)** |
| **Panjang Template (IT)**| ~1.5 KB (1,532 karakter) | **~17.4 KB** (17,466 karakter) | ~1.1 KB (1,091 karakter) |

---

## 2. Temuan Utama & Analisis Perubahan

### A. Perubahan Sintaks Penanda Turn (Turn Markers)
Gemma 4 melakukan standarisasi sintaks pembatas turn untuk mempermudah parsing teks percakapan secara terpadu:
* **Gemma 3 / T5Gemma-2**: Menggunakan tag berbasis teks biasa `<start_of_turn>` dan `<end_of_turn>`.
* **Gemma 4**: Menggunakan sintaks simbolis terpadu `<|turn>` untuk memulai turn dan `<turn|>` untuk mengakhiri turn. Pembatas `<|...>` (untuk pembuka) dan `...|>` (untuk penutup) ini menjadi pola standar baru untuk seluruh token kontrol di Gemma 4.

### B. Dukungan Penalaran Berantai (Chain-of-Thought) Natif pada Gemma 4
Hadirnya token **`<|think|>`** di tingkat kosakata/tokenizer menunjukkan bahwa model Gemma 4 secara intrinsik mendukung proses berpikir sebelum mengeluarkan jawaban. Teks proses berpikir diletakkan di antara token berpikir ini agar antarmuka pengguna (UI) dapat menyembunyikan atau menampilkannya dalam bentuk dropdown (seperti perilaku berpikir pada DeepSeek-R1 atau o1).

### C. Protokol Tool Calling & Agentic Native (Gemma 4)
Alih-alih menggunakan teks JSON biasa yang rentan salah format, Gemma 4 menggunakan token khusus untuk memanggil fungsi:
1. **Pemanggilan Fungsi**: Dimulai dengan `<|tool_call>` dan diakhiri `<tool_call|>`.
2. **Konteks Fungsi**: Menggunakan `<|tool>` dan `<tool|>`.
3. **Respon Eksekusi**: Menggunakan `<|tool_call>` dan diakhiri `<tool_response|>`.

### D. Kompleksitas Template Percakapan
* Chat template Gemma 3 IT memiliki panjang **1.5 KB** karena hanya memformat pesan teks, gambar, penggabungan system prompt, dan alternatif role secara bergiliran.
* Chat template Gemma 4 IT memiliki ukuran sangat besar **17.4 KB** karena di dalamnya ditanamkan logika parsing multimodal yang kompleks (termasuk audio), validasi pemanggilan fungsi agen, serta pemisahan output berpikir (*thinking blocks*).

---

## 3. Implementasi Kustom Chat Template untuk T5Gemma-2

### Masalah Awal:
Model `t5gemma-2-4b-4b` yang kita gunakan menggunakan kosa kata (vocabulary) Gemma 3, namun secara bawaan dari Hugging Face tidak memiliki template percakapan (*chat template*). Akibatnya, model tidak kompatibel dengan fungsi `apply_chat_template()` bawaan ekosistem Transformers/Unsloth.

### Solusi Kita:
Kita membuat file kustom [chat_template.jinja](file:///d:/Codings/unsloth/t5-gemma-2/instruct/scratch/chat_template.jinja) untuk disuntikkan ke tokenizer T5Gemma-2.

#### Logika Template:
1. **Penggabungan System Prompt**: Karena basis tokenizer Gemma 3 tidak memiliki tag native khusus untuk system prompt (seperti `<start_of_turn>system`), template kita akan mengambil isi pesan system pertama dan menyatukannya ke pesan user pertama dengan pemisah dua baris baru (`\n\n`):
   ```jinja
   {%- if ns.first_user and ns.found_system -%}
       {{- ns.system_message + '\n\n' -}}
   ```
2. **Kompatibilitas Multi-turn**: Menangani peran `"user"` dan memetakan baik `"model"` maupun `"assistant"` secara aman ke tag `<start_of_turn>model\n`.
3. **Pemicu Generasi**: Mendukung argumen `add_generation_prompt=True` untuk menyisipkan `<start_of_turn>model\n` di ujung prompt guna memicu model menulis balasan.

#### Contoh Output Prompt yang Dihasilkan:
```text
<bos><start_of_turn>user
Kamu adalah asisten AI yang helpful, santai, dan ramah. Gunakan Bahasa Indonesia sebagai bahasa utama.

Hai, bisa jelaskan apa itu black hole?<end_of_turn>
<start_of_turn>model
Tentu! Black hole atau lubang hitam adalah bagian dari ruang waktu yang gravitasi tarikannya begitu kuat sehingga tidak ada apapun, bahkan cahaya, yang bisa lolos darinya.<end_of_turn>
<start_of_turn>user
Wah seram juga ya. Terus gimana kita bisa tahu kalau ada black hole di sana?<end_of_turn>
<start_of_turn>model
```

Hasil pengujian menunjukkan bahwa tokenizer Gemma 3 membaca token `<start_of_turn>` dan `<end_of_turn>` secara sempurna (tanpa terpecah menjadi karakter biasa) dan sukses memicu model melakukan generasi multi-turn secara bersih di memori CUDA.
