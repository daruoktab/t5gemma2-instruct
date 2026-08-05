# Analisis Token Gambar (Image Token) pada T5-Gemma-2

Dokumen ini merangkum analisis dan temuan mengenai bagaimana model `T5-Gemma-2` (khususnya `google/t5gemma-2-270m-270m`) memproses input multimodal berupa gambar, serta token placeholder yang tepat untuk digunakan dalam prompt teks.

## 1. Pemetaan Processor
Saat memuat processor untuk model `google/t5gemma-2-270m-270m` menggunakan Hugging Face `AutoProcessor`:
```python
from transformers import AutoProcessor
processor = AutoProcessor.from_pretrained("google/t5gemma-2-270m-270m")
print(processor.__class__)
# Output: <class 'transformers.models.gemma3.processing_gemma3.Gemma3Processor'>
```
Ternyata, Hugging Face **tidak mendefinisikan processor khusus** untuk model `t5gemma2`. Di dalam kode internal `transformers`, konfigurasi `T5Gemma2Config` secara eksplisit diarahkan untuk menggunakan **`Gemma3Processor`**. Oleh karena itu, perilaku pemrosesan gambar dan token placeholder pada T5-Gemma-2 sepenuhnya mengikuti spesifikasi multimodal dari keluarga model Gemma 3.

## 2. Token Awal Gambar (Beginning of Image Token)
Karena menggunakan `Gemma3Processor`, processor ini akan mencari token penanda awal gambar (*Beginning of Image* / `boi_token`) di dalam teks prompt untuk diselaraskan dengan input gambar yang diterima.

Berdasarkan inspeksi terhadap tokenizer model:
*   `boi_token` (*Beginning of Image*): **`📷`** (Unicode character: `\uf400`, ID: `255999`)
*   `eoi_token` (*End of Image*): **`<end_of_image>`** (ID: `256000`)
*   `image_token` (*Image Soft Token*): **`<image_soft_token>`** (ID: `256001`)

### Penting: Perilaku Pencarian Token
Processor mendeteksi posisi gambar berdasarkan keberadaan karakter **`\uf400`** (`📷`) di dalam prompt teks.
Jika kita menggunakan string placeholder lain seperti `"<img>"`, `"<image>"`, atau `"<image_soft_token>"`, processor akan gagal mencocokkannya dan menghasilkan error:
```text
ValueError: Prompt contained 0 image tokens but received 1 images.
```

Oleh karena itu, setiap gambar yang di-input ke model harus ditandai dengan menyisipkan `processor.boi_token` secara dinamis di dalam prompt teks.

## 3. Contoh Implementasi & Pengujian
Berikut adalah kode pengujian yang berhasil dijalankan untuk mendeskripsikan gambar:

```python
import requests
from PIL import Image
from transformers import AutoProcessor, AutoModelForSeq2SeqLM

# 1. Muat processor dan model
processor = AutoProcessor.from_pretrained("google/t5gemma-2-4b-4b")
model = AutoModelForSeq2SeqLM.from_pretrained("google/t5gemma-2-4b-4b")

# 2. Ambil gambar uji coba
image = Image.open("d:/Codings/unsloth/t5-gemma-2/embedding/assets/batik.png")

# 3. Definisikan prompt dengan menyisipkan processor.boi_token
prompt = f"{processor.boi_token} ini adalah gambar batik."

# 4. Pemrosesan input dan inferensi
model_inputs = processor(text=prompt, images=image, return_tensors="pt")
generation = model.generate(**model_inputs, max_new_tokens=64, do_sample=False)

# 5. Tampilkan hasil
print("Output:", processor.decode(generation[0]))
```
### Hasil Eksperimen
*   **Prompt**: `f"{processor.boi_token} ini adalah gambar batik."`
*   **Decoded Inputs**: `<bos>\n\n📷<image_soft_token>... (256x) ...<end_of_image>\n\n ini adalah gambar batik.`
*   **Kesimpulan**: Model berhasil memproses gambar `batik.png` dari aset embedding dengan mengonversi token `📷` (atau `\uf400`) menjadi urutan 256 token visual yang dapat dipahami oleh T5-Gemma-2.

### 4. Analisis Perilaku Pengulangan (Repetition)
Dalam pengujian dengan parameter generasi bertipe *greedy* (`do_sample=False`), model dengan ukuran kecil (seperti 270M) atau model yang belum dilatih SFT dengan penalaran terstruktur rentan terhadap pengulangan kata (*repetition loop*). 

Untuk mengatasi hal ini, saat melakukan inferensi dianjurkan untuk menggunakan parameter:
*   `repetition_penalty=1.2` atau lebih tinggi.
*   `no_repeat_ngram_size=3`.
*   `do_sample=True` with `temperature=0.7`.

## 5. Pemrosesan Banyak Gambar (Multiple Images)
Model Gemma 3 (dan T5Gemma-2) mendukung pemrosesan banyak gambar secara natif dalam satu prompt percakapan.

### Cara Penggunaan:
1. Setiap gambar yang dikirimkan harus diwakili oleh **satu token** `processor.boi_token` (`📷` / `\uf400`) di lokasi penempatan yang diinginkan di dalam teks prompt.
2. Jumlah gambar yang dimasukkan ke dalam daftar `images` harus **sama persis** dengan jumlah token `boi_token` yang dideklarasikan di dalam teks prompt. Jika tidak sama, processor akan melempar error:
   `ValueError: Prompt contained X image tokens but received Y images.`

### Contoh Kode Implementasi:
```python
from PIL import Image
from transformers import AutoProcessor

processor = AutoProcessor.from_pretrained("google/t5gemma-2-4b-4b")

# Load beberapa gambar
img1 = Image.open("path/to/batik.png")
img2 = Image.open("path/to/nasi-goreng.png")
images = [img1, img2]

# Konstruksi prompt dengan dua placeholder boi_token
prompt = f"Ini halaman pertama: {processor.boi_token} dan ini halaman kedua: {processor.boi_token}. Tolong jelaskan."

# Proses inputs
inputs = processor(text=prompt, images=images, return_tensors="pt")
# pixel_values akan menghasilkan shape: [2, 3, 896, 896]
# input_ids akan memiliki panjang token visual ter-ekspansi sebesar 512 token (256 * 2) ditambah token teks.
```
