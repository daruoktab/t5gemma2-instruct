# Laporan Analisis Lengkap Metrik SFT: `t5gemma-2-4b-4b-instruct-chat-indo-v4-unsloth`

Laporan ini menyajikan analisis mendalam dari **seluruh metrik** evaluasi teks (*text generation*) yang dicatat selama proses training SFT hingga `checkpoint-1000` (dari total 1232 step).

## Ringkasan Metadata Training
* **Model ID:** [`daruokta/t5gemma-2-4b-4b-instruct-chat-indo-v4-unsloth`](https://huggingface.co/daruokta/t5gemma-2-4b-4b-instruct-chat-indo-v4-unsloth)
* **Status Progres:** Step 1000 / 1232 (~81.17% selesai)
* **Epoch Terlatih:** 3.25 / 4.00
* **Interval Evaluasi:** Setiap 200 step

---

## Tabel Log Metrik Lengkap

Berikut adalah tabel lengkap riwayat training dan metrik evaluasi pada set validasi:

| Step | Epoch | Train Loss | Eval Loss | Perplexity (PPL) | ROUGE-1 (%) | ROUGE-2 (%) | ROUGE-L (%) | BLEU (%) | METEOR (%) | BERTScore F1 (%) | Exact Match (%) |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **100** | 0.3255 | 3.5703 | - | - | - | - | - | - | - | - | - |
| **200** | 0.6510 | 3.1000 | 2.9584 | 19.2677 | 53.3815 | 34.2479 | 49.0505 | 12.5724 | 49.8822 | 82.7966 | 21.4929 |
| **300** | 0.9765 | 3.0214 | - | - | - | - | - | - | - | - | - |
| **400** | 1.2995 | 2.9388 | 2.8967 | 18.1140 | 56.6260 | 37.1781 | 52.1181 | 14.4600 | 53.4153 | 83.9299 | 23.4663 |
| **500** | 1.6250 | 2.9216 | - | - | - | - | - | - | - | - | - |
| **600** | 1.9505 | 2.9090 | 2.8644 | 17.5393 | 58.0214 | 37.9609 | 53.3616 | 14.4889 | 53.6423 | 84.3886 | 24.5817 |
| **700** | 2.2734 | 2.8281 | - | - | - | - | - | - | - | - | - |
| **800** | 2.5989 | 2.8028 | 2.8566 | 17.4020 | 59.0132 | 39.0701 | 54.4187 | 15.4393 | 54.8789 | 84.9006 | 26.0403 |
| **900** | 2.9244 | 2.7943 | - | - | - | - | - | - | - | - | - |
| **1000** | 3.2474 | 2.7295 | 2.8646 | 17.5417 | 60.0708 | 40.1707 | 55.5142 | 15.9385 | 56.2059 | 85.2262 | 27.3273 |

---

## Visualisasi Dashboard Metrik SFT

![SFT Metrics Dashboard](file:///C:/Users/daru/.gemini/antigravity-ide/brain/41396469-ed2d-4796-b27b-62ac998ab7c4/sft_all_metrics_plots.png)

---

## Analisis Komprehensif

### 1. Dinamika Loss dan Perplexity (PPL)
* **Training Loss vs Validation Loss:** Training loss menurun terus menerus (dari 3.57 ke 2.73), yang membuktikan proses belajar berjalan lancar. Namun, Validation Loss mencapai titik terendah pada step 800 (`2.8566`) sebelum naik tipis ke `2.8646` pada step 1000.
* **Perplexity (PPL):** Tren Perplexity berbanding lurus dengan Validation Loss. Nilai PPL terbaik dicapai di step 800 (`17.4020`) dan mengalami sedikit peningkatan di step 1000 (`17.5417`). Kenaikan tipis ini menunjukkan model mulai memasuki wilayah jenuh/overfitting ringan pada epoch ke-3.25.

### 2. Kualitas Generasi Teks (NLG Metrics)
Meskipun Validation Loss mengalami sedikit kenaikan pada step 1000, **seluruh metrik kualitas generasi teks justru menunjukkan peningkatan yang sangat konsisten**:
* **ROUGE (F1-score):**
  * **ROUGE-1 (Unigram):** Meningkat konsisten dari **53.38%** -> **60.07%** (peningkatan +6.69%).
  * **ROUGE-2 (Bigram):** Meningkat dari **34.25%** -> **40.17%** (+5.92%).
  * **ROUGE-L (LCS):** Meningkat dari **49.05%** -> **55.51%** (+6.46%).
* **BLEU & METEOR:**
  * **BLEU:** Meningkat dari **12.57%** -> **15.94%** (+3.37%).
  * **METEOR:** Meningkat signifikan dari **49.88%** -> **56.21%** (+6.33%).
* **Semantic Similarity (BERTScore F1):**
  * Meningkat dari **82.80%** ke **85.23%**, menunjukkan keselarasan semantik jawaban model dengan target semakin akurat.
* **Exact Match (EM):**
  * Meningkat dari **21.49%** ke **27.33%**, menandakan semakin banyak jawaban model yang sama persis secara literal dengan target teks referensi.

### Kesimpulan & Rekomendasi
* **Penyimpangan Loss vs Metrik Teks:** Terkadang, sedikit kenaikan pada cross-entropy loss (Validation Loss) tidak berarti kualitas model menurun. Hal ini biasanya terjadi karena model menjadi kurang yakin (penurunan probabilitas softmax pada token yang benar), tetapi token dengan probabilitas tertinggi yang dihasilkan tetaplah token yang benar secara literal (terbukti dari kenaikan ROUGE, BLEU, dan EM).
* **Rekomendasi:** Lanjutkan proses training SFT hingga selesai di step 1232 (4 epoch). Peningkatan kualitas teks di step 1000 membuktikan model masih membaik dalam aspek generasi teks nyata.
