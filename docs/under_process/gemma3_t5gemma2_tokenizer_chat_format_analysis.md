# Analisis Tokenizer dan Format Chat Gemma 3 & T5-Gemma-2

Dokumen ini merangkum seluruh temuan dan eksperimen yang telah dilakukan terkait dengan *tokenizer* dari model keluarga Gemma, khususnya `t5gemma-2-270m`, `gemma-3-270m`, dan `gemma-3-4b`. 

## 1. Status Token Percakapan (Chat Tokens)

Berdasarkan inspeksi langsung menggunakan script Python ke dalam file tokenizer model (baik T5-Gemma-2 maupun varian Gemma 3 Base dan Instruct), ditemukan fakta penting:

*   **Bukan Special Token**: Penanda awal dan akhir percakapan seperti `<start_of_turn>` (ID: 105) dan `<end_of_turn>` (ID: 106) **bukanlah** *special tokens* (`is_special=False`).
*   **Peran (Roles) adalah Teks Biasa**: Token yang melambangkan peran seperti `user`, `model`, `assistant`, dan `system` murni merupakan kosakata reguler biasa yang ada di dalam *vocabulary* tokenizer.
*   **Special Token Sejati**: Tokenizer hanya mengakui token-token struktural dasar sebagai *special token* (yaitu `<bos>`, `<eos>`, `<unk>`, `<pad>`, dan token-token terkait gambar seperti `<img>`).

## 2. Perbedaan Versi Base dan Instruct

*   **Versi Base (`gemma-3-270m`)**: Model dasar yang hanya dilatih untuk menebak kata selanjutnya (*next-token prediction*) **tidak memiliki** file `chat_template.jinja` dan tidak mendefinisikan atribut `chat_template` di dalam `tokenizer_config.json`.
*   **Versi Instruct (`gemma-3-270m-it`)**: Karena dilatih khusus untuk berdialog, versi ini memiliki file `chat_template.jinja`.
*   **`special_tokens_map.json`**: Isi dari file kamus *special token* ini **100% identik** antara versi Base dan Instruct (hanya berisi pemetaan ke `<bos>`, `<eos>`, dll), menegaskan bahwa secara arsitektur *tokenizer*, keduanya tidak diubah; perbedaannya hanya ada di *chat template*.

## 3. Analisis Jinja Chat Template Gemma 3

Dari ekstraksi Jinja template resmi milik `gemma-3-4b-it` dan cache lokal `gemma-3-270m-it`, terdapat beberapa aturan baku dalam pembentukan *prompt*:

1.  **Penanganan System Prompt**: Gemma **tidak** memiliki format role `<start_of_turn>system\n`. Jika terdapat pesan `system`, isi teksnya akan diekstrak, diberi tambahan baris baru ganda (`\n\n`), lalu **digabungkan (di-prepend) ke bagian paling awal dari teks pesan `user` yang pertama**.
2.  **Validasi Urutan (Alternating)**: Pesan harus berselang-seling secara ketat. Tidak boleh ada dua pesan `user` berturut-turut atau dua pesan `assistant` berturut-turut.
3.  **Pengubahan Role**: Teks role `assistant` akan secara otomatis diubah menjadi `model` di dalam *template*.
4.  **Dukungan Gambar (Multimodal)**: Template sudah mengantisipasi input berbasis `iterable` untuk mengenali dan memformat elemen tipe `image` dengan tag `<img>`.

---

## Planning untuk v5

Apakah kita bisa menerapkan Jinja template untuk model bertipe `T5` (seperti `T5-Gemma`) dengan baik?

**Jawabannya: SANGAT BISA.**

Meskipun arsitektur `T5-Gemma` merupakan Encoder-Decoder (berbeda dengan arsitektur Decoder-only pada standar Gemma/Llama), proses pemformatan *chat* terjadi **sebelum** teks masuk ke dalam tahapan Encoder. 

Berikut poin-poin perencanaannya:

1.  **Penggunaan `apply_chat_template`**: Di *pipeline* v5 (`working-molab-v5.py`), daripada kita merakit *string* percakapan secara manual (`"<start_of_turn>user\n" + ...`), kita bisa sepenuhnya menggunakan fungsi bawaan Hugging Face `tokenizer.apply_chat_template(messages, tokenize=False)`. 
2.  **Kecocokan Sempurna**: Karena `T5-Gemma-2` juga mewarisi *vocabulary* Gemma, output dari `apply_chat_template` (yang menghasilkan teks *raw* dengan kurung siku seperti `<start_of_turn>`) akan di-tokenisasi dengan sempurna oleh tokenizer T5. Teks `<start_of_turn>` akan otomatis diubah menjadi token ID `105`, tanpa error, karena tokenizer memang menganggapnya teks biasa.
3.  **Memasukkan Template ke Tokenizer**: Karena saat ini *tokenizer* T5-Gemma mungkin belum disisipkan string Jinja template secara *default*, kita bisa memuat file `chat_template.jinja` asli dari `gemma-3-270m-it` yang ada di *cache* lokal, lalu menetapkannya ke dalam objek tokenizer script kita:
    ```python
    with open("path/to/chat_template.jinja", "r") as f:
        tokenizer.chat_template = f.read()
    ```
4.  **Keuntungan di v5**: Dengan menerapkan ini, kode dataset preparation akan menjadi jauh lebih bersih, tahan terhadap *bug* urutan (karena dijaga oleh Jinja template), dan secara otomatis kompatibel dengan fitur *multimodal* (gambar) bawaan Gemma 3 jika kelak dibutuhkan. Ini akan memastikan model dilatih dengan format SFT (Supervised Fine-Tuning) *best practice* versi Google.
