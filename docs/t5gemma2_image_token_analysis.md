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
*   `boi_token` (*Beginning of Image*): **`<img>`** (Unicode points: `[60, 115, 116, 97, 114, 116, 95, 111, 102, 95, 105, 109, 97, 103, 101, 62]`)
*   `eoi_token` (*End of Image*): **`<end_of_image>`** (Unicode points: `[60, 101, 110, 100, 95, 111, 102, 95, 105, 109, 97, 103, 101, 62]`)

### Penting: Perilaku Pencarian Token
Processor mendeteksi posisi gambar berdasarkan keberadaan token **`<img>`** di dalam prompt teks. 
Jika kita menggunakan token placeholder lain (misalnya `<image>` atau `<img>`), processor akan gagal mencocokkannya dan menghasilkan error berikut:
```text
ValueError: Prompt contained 0 image tokens but received 1 images.
```

Oleh karena itu, setiap gambar yang di-input ke model harus ditandai dengan token `<img>` di dalam prompt teks.

## 3. Contoh Implementasi & Pengujian
Berikut adalah kode pengujian yang berhasil dijalankan untuk mendeskripsikan gambar seekor lebah:

```python
import requests
from PIL import Image
from transformers import AutoProcessor, AutoModelForSeq2SeqLM

# 1. Muat processor dan model
processor = AutoProcessor.from_pretrained("google/t5gemma-2-270m-270m")
model = AutoModelForSeq2SeqLM.from_pretrained("google/t5gemma-2-270m-270m")

# 2. Unduh gambar lebah sebagai input
url = "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/bee.jpg"
image = Image.open(requests.get(url, stream=True).raw)

# 3. Definisikan prompt dengan token penanda <img>
prompt = "<img> in this image, there is"

# 4. Pemrosesan input dan inferensi
model_inputs = processor(text=prompt, images=image, return_tensors="pt")
generation = model.generate(**model_inputs, max_new_tokens=20, do_sample=False)

# 5. Tampilkan hasil
print("Output:", processor.decode(generation[0]))
# Output: <bos> a bumble bee in a flower bed.<eos>
```

### Hasil Eksperimen
*   **Prompt**: `"<img> in this image, there is"`
*   **Output Model**: `<bos> a bumble bee in a flower bed.<eos>`
*   **Kesimpulan**: Model berhasil memahami representasi gambar lebah yang disisipkan pada posisi token `<img>` dan mendeskripsikannya dengan sangat akurat ("a bumble bee in a flower bed").

### 4. Hasil Eksperimen dengan Bahasa Indonesia (`max_new_tokens=64`)
Kami juga menguji pemrosesan dengan prompt berbahasa Indonesia untuk mendeskripsikan gambar yang sama, dengan memperpanjang batas token output (`max_new_tokens=64`):

*   **Prompt**: `"<img> di dalam gambar ini terdapat"` (Dikonstruksi di Python sebagai: `"<" + "start_of_image" + "> di dalam gambar ini terdapat"`)
*   **Output Model**:
    ```text
    <bos> sebuah bunga yang berwarna merah muda.

     bunga ini merupakan bunga yang berwarna merah muda.

     bunga ini merupakan bunga yang berwarna merah muda.

     bunga ini merupakan bunga yang berwarna merah muda.

     bunga ini merupakan bunga yang berwarna merah muda.

     bunga ini merupakan bunga yang berwarna merah muda.

     bunga ini merupakan bunga yang berwarna
    ```
*   **Analisis**:
    1.  **Kemampuan Pemahaman Gambar**: Model berhasil mendeteksi objek bunga berwarna merah muda/keunguan (tempat lebah hinggap) yang ada di dalam gambar `bee.jpg`.
    2.  **Repetisi**: Karena menggunakan parameter dekoding *greedy* (`do_sample=False`) pada model yang relatif kecil (270M), model mengalami perulangan teks (*repetition loop*). Ini adalah perilaku yang wajar untuk model berukuran kecil jika tidak ditambahkan penalti repetisi (*repetition penalty*).

