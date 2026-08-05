# [DIAGNOSTICS] Analisis Logit Masking, Gradient Accumulation, dan Loss Divergence

**Last Updated:** 5 Agustus 2026  
**Context:** Diagnosis mendalam fenomena "loss terbang" (loss divergence), error numerical instability (`NaN`/`inf`), dan solusi konfigurasi gradient accumulation pada lingkungan Marimo Molab / PyTorch.

---

## 1. Patofisiologi "Loss Terbang" (Loss Divergence)

Fenomena "loss terbang" ditandai dengan lonjakan nilai loss yang sangat mendadak atau nilai loss inisial yang tidak masuk akal (misalnya loss bernilai 800+ pada langkah awal, atau mendadak `inf` / `NaN`).

### 3 Skenario Utama Keruntuhan Numerik:

#### Skenario 1: Masking Buta Terhadap Penanda Target Utama (Ground Truth Label)
- **Masalah:** Jika mask boolean menindih posisi target token aktual dan mengisinya dengan `float('-inf')`, fungsi Softmax mengonversi probabilitas target tersebut menjadi tepat `0.0`.
- **Dampak:** `CrossEntropyLoss` mengalkulasi $-\log(0.0) \to +\infty$. Saat Autograd menghitung turunan $-1/x$, terjadi pembagian $-1/0.0$ yang memicu pencemaran `NaN` pada seluruh matriks gradien dan merusak parameter optimizer.

#### Skenario 2: Amputasi Baris Vektor Kosakata Total (Total Vector Masking)
- **Masalah:** Semua token dalam satu posisi sekuens terisi `float('-inf')` akibat kesalahan *broadcasting* dimensi (misalnya lupa `.unsqueeze(dim=X)`).
- **Dampak:** Penyebut Softmax $\sum e^{z_j} = 0.0$, menghasilkan kalkulasi $0.0 / 0.0 \to \text{NaN}$.

#### Skenario 3: Pemilihan Skalar Masking yang Suboptimal
- **Masalah:** Penggunaan skalar netral (seperti `0` atau angka negatif kecil `-100`) alih-alih `float('-inf')` menyebabkan token yang seharusnya diabaikan tetap mendapat alokasi probabilitas Softmax.

---

## 2. Investigasi Gradient Accumulation Loss Anomalies

Pada Seq2Seq / Hugging Face Trainer:
- Jika `gradient_accumulation_steps` > 1 dan loss pada tiap micro-step di-sum tanpa pembagian yang tepat terhadap jumlah token aktif, loss akan tampak tereskalasi proporsional terhadap ukuran accumulation.
- **Solusi:** Pastikan `DataCollatorForSeq2Seq` digunakan dan `label_pad_token_id = -100` dikonfigurasi secara konsisten agar token padding diabaikan dari komputasi loss rata-rata.

---

## 3. Matriks Diagnostik dan Solusi Cepat

| Gejala | Penyebab Akar | Tindakan Perbaikan |
|---|---|---|
| Loss mendadak `NaN` di Step N | Masking menutupi ground truth label atau terjadi division by zero | Periksa boolean mask mask_fill; pastikan target label != masked index |
| Loss awal bernilai 800+ | Pilihan reduction loss keliru (`sum` vs `mean`) atau logit terdistorsi | Gunakan `reduction='mean'` pada CrossEntropyLoss |
| Memory OOM saat Gradient Accumulation | Cache autograd retain graph menumpuk | Panggil `torch.cuda.empty_cache()` dan pastikan `zero_grad()` dipanggil tepat waktu |
