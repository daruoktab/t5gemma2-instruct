# Panduan Kuantisasi T5Gemma-2 & Penanganan AssertionError (Tied Weights Mismatch)

Dokumen ini mendokumentasikan temuan teknis mengenai kuantisasi model **T5Gemma-2** ke 4-bit (NF4), permasalahan `AssertionError` saat memuat model terkuantisasi, penyebab utama (*root cause*), dan solusi konfigurasinya.

---

## 1. Latar Belakang & Kebutuhan Kuantisasi
Model **T5Gemma-2 4B-4B** (Version 5) memiliki ukuran parameter sekitar ~15 GB dalam format `bfloat16`. 
* **Tujuan:** Agar model dapat dijalankan pada GPU lokal dengan VRAM terbatas (misalnya laptop dengan GPU 6 GB VRAM), model perlu dikompresi menggunakan kuantisasi 4-bit NF4.
* **Kebutuhan Multimodal:** Untuk mempertahankan performa pemahaman gambar, komponen **Vision Tower** (`model.encoder.vision_tower`) **tidak boleh dikuantisasi** (tetap dalam presisi tinggi `bfloat16`).

---

## 2. Masalah yang Ditemukan (The AssertionError)
Ketika mencoba mempertahankan presisi `vision_tower` dengan menyetel parameter kuantisasi sebagai berikut:
```python
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_use_double_quant=True,
    bnb_4bit_compute_dtype=torch.bfloat16,
    llm_int8_skip_modules=['model.encoder.vision_tower']  # Menghindari kuantisasi vision tower
)
```
Model dapat dikuantisasi dan disimpan ke disk. Namun, saat model dimuat kembali untuk proses inferensi:
```python
local_model = AutoModelForSeq2SeqLM.from_pretrained(load_source, device_map="auto")
```
Pemuatan model mengalami kegagalan fatal (*crash*) dengan error berikut pada modul output head:
```tb
AssertionError: assert module.weight.shape[1] == 1
# Terjadi pada lm_head.out_proj
```

---

## 3. Analisis Penyebab Utama (Root Cause)
1. **Mekanisme Override pada `BitsAndBytesConfig`:**
   Saat kita menyertakan daftar kustom pada parameter `llm_int8_skip_modules`, pustaka `bitsandbytes` akan **menimpa secara total** daftar modul yang dilewati secara default.
2. **Tied Weights (Bobot Terikat):**
   Model T5Gemma-2 menggunakan arsitektur *tied weights*, di mana matriks bobot input embedding (`model.decoder.embed_tokens` & `model.encoder.text_model.embed_tokens`) dan output language modeling head (`lm_head` / `lm_head.out_proj`) saling berbagi pointer bobot yang sama untuk menghemat memori dan meningkatkan konvergensi.
3. **Konflik Presisi:**
   Karena daftar skip default ditimpa oleh `['model.encoder.vision_tower']`, modul `lm_head` dan `embed_tokens` dipaksa untuk masuk ke proses kuantisasi 4-bit. Namun, karena keterikatan bobot (*tied weights*) dan sifat sebagian modul yang tidak terkuantisasi dengan benar, terjadi ketidakcocokan tipe data (sebagian bfloat16, sebagian INT4 terkuantisasi) dan ketidakcocokan dimensi weight tensor saat dimuat kembali ke memori, memicu `AssertionError` pada pustaka PyTorch/Transformers.

---

## 4. Solusi Konfigurasi
Untuk mencegah error tersebut, kita harus **secara eksplisit** mendaftarkan seluruh modul *tied weights* (embedding dan LM head) ke dalam `llm_int8_skip_modules` bersama dengan modul `vision_tower`.

Konfigurasi kuantisasi 4-bit NF4 yang aman dan benar untuk T5Gemma-2 adalah:
```python
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_use_double_quant=True,
    bnb_4bit_compute_dtype=torch.bfloat16,
    llm_int8_skip_modules=[
        'model.encoder.vision_tower',  # Mempertahankan presisi vision
        'lm_head',                     # Melewati output head (tied weights)
        'embed_tokens'                 # Melewati input embeddings (tied weights)
    ]
)
```
*Catatan: String `'embed_tokens'` bertindak sebagai pencocokan substring yang akan otomatis melewati `model.encoder.text_model.embed_tokens` dan `model.decoder.embed_tokens` secara bersamaan.*

---

## 5. Hasil Verifikasi & Uji Coba Lokal
Kami menguji solusi ini menggunakan model `google/t5gemma-2-270m-270m` secara lokal di GPU laptop:
* **Kompatibilitas:** Model berhasil dimuat kembali untuk inferensi setelah dikuantisasi tanpa ada error dimensi.
* **Konsumsi VRAM Lokal (Model 270M):** 
  * VRAM Terpakai: **~1216.10 MB** (Sudah termasuk bobot model non-kuantisasi seperti vision tower & embeddings + overhead CUDA).
  * Dengan skema ini, model T5Gemma-2 4B-4B terkuantisasi diestimasikan hanya membutuhkan **~3.0 - 3.5 GB VRAM**, sehingga sangat aman dijalankan di GPU 6 GB VRAM.
* **Inferensi:** Model dapat menghasilkan respons secara normal dan stabil.

---

## 6. Penerapan pada Kode Project
Solusi ini telah diimplementasikan penuh pada file:
1. Skrip Verifikasi Lokal: [test_quant_pipeline.py](file:///d:/Codings/unsloth/t5-gemma-2/instruct/test_quant_pipeline.py)
2. Skrip Pipeline Cloud V5: [working-molab-v5.py](file:///d:/Codings/unsloth/t5-gemma-2/instruct/working-molab-v5.py)
