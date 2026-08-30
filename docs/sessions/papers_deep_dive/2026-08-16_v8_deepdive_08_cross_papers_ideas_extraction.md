# 📚 Deep-Dive Ekstraksi Paper: `paper-usingnormention-t5gemma2.csv` & `encdec_llm_2025_2026_filtered.csv`
> **Tanggal Sesi:** 16 Agustus 2026  
> **Tujuan:** Menggali ide-ide arsitektur, training, dan evaluasi mutakhir di luar paper yang sudah di-apply di V7/V8, untuk roadmap rilis bertahap (V8, V9, V10+).

---

## 1. Taksonomi Temuan Kunci dari Kedua CSV Paper

Setelah menganalisis seluruh entri dan membaca dokumen PDF lengkap (`Return of the Encoder`, `Enhance Unlearning / GRUN`, `Gemma Encoder`, `Encoder-Decoder Gemma`, `Gemma 3 TR`, `Layer-Aware Task Arithmetic`, dan `Cross-Model Transfer`), ditemukan 6 pilar inovasi yang dapat diterapkan:

---

### 🏛️ Pilar 1: Cross-Architecture Knowledge Distillation (KD Decoder $\to$ Encoder-Decoder)
* **Sumber Paper:** 
  - `Return of the Encoder: Maximizing Parameter Efficiency for SLMs` (Microsoft, Jan 2025)
  - `Encoder-Decoder Gemma: Improving the Quality-Efficiency Trade-Off via Adaptation` (Google, Apr 2025)
* **Temuan Kritis:**
  - Model Encoder-Decoder dengan rasio parameter **2/3 Encoder & 1/3 Decoder** secara konsisten mengalahkan arsitektur seimbang (1/2-1/2) dan Decoder-Only pada parameter budget tetap $\le 1\text{B}$ s.d. $4\text{B}$.
  - Encoder menangani pemahaman bidirectional penuh (tanpa KV-cache grow), sedangkan Decoder hanya menghasilkan autoregressive tokens.
  - **Formula KD Decoder-only $\to$ Enc-Dec:**
    Offset logit guru diselaraskan:
    $$\mathcal{L}_{\text{KD}} = \alpha \cdot \tau^2 \cdot D_{\text{KL}}(p_{\text{teacher}} \parallel p_{\text{student}}) + (1 - \alpha) \mathcal{L}_{\text{CE}}(y, y_{\text{student}})$$
* **Peluang Penerapan (V8 / V9):**
  - Pada tahap SFT dan ORPO V8/V9, output logit dari `Gemma 3 4B IT` (sebagai teacher) dapat disuntikkan sebagai soft targets untuk `T5Gemma 2` (sebagai student), meningkatkan penalaran nalar kompleks (SQuAD +8%, Math +12%).

---

### ⚡ Pilar 2: Gated Representation Tuning & Precision Unlearning (GRUN / ReFT)
* **Sumber Paper:** 
  - `A General Framework to Enhance Fine-tuning-based LLM Unlearning (GRUN)` (Amazon & MSU, Mar 2025)
* **Temuan Kritis:**
  - Metode penolakan / forgetting berbasis Gradient Ascent konvensional sering merusak *general utility* model karena mendistorsi bobot global.
  - **GRUN (Gated Representation UNlearning)** membekukan bobot model dan menyuntikkan soft-gate ganda pada representasi token terakhir di layer-layer tertentu:
    $$\Phi_{\text{GRUN}}(h_i^{(\ell)}) = h_i^{(\ell)} + g(h_i^{(\ell)}) \cdot R^T(W h_i^{(\ell)} + b - R h_i^{(\ell)})$$
  - Soft gate $g(h) \in (0, 1)$ hanya aktif ketika mendeteksi kueri yang harus ditolak/dihapus (misal: halusinasi bahasa atau instruksi berbahaya), sementara untuk kueri umum nilainya $\approx 0$.
* **Peluang Penerapan (V8 / V9):**
  - Menjadi mekanisme pelindung saat Alignment ORPO: mencegah penurunan skor factualitas & bahasa saat model dipaksa menolak jawaban buruk (*rejected*).

---

### 🎛️ Pilar 3: Layer-Aware Task Arithmetic (LATA) & Orthogonal Alignment
* **Sumber Paper:** 
  - `Layer-Aware Task Arithmetic (LATA)` (Feb 2025)
  - `Cross-Model Transfer of Task Vectors via Few-Shot Orthogonal Alignment` (May 2025)
* **Temuan Kritis:**
  - Task vector pada layer-layer awal ($<25\%$) memiliki cosine similarity yang sangat tinggi dengan *instruction vector*, sedangkan layer-layer dalam ($>75\%$) lebih memuat representasi spesifik tugas (*task-specific*).
  - Mengalikan task vector dengan bobot terbobot rank/logaritmik per layer:
    $$\tau'_i = \log_L(r_i) \cdot \tau_i$$
  - Menyaring task vector via *orthogonal similarity transformation* $U^T \Delta W U$ menjaga norm Frobenius dan rank LoRA tetap utuh saat ditransfer antar model yang berbeda pre-training.
* **Peluang Penerapan (V8):**
  - Sudah diintegrasikan langsung pada **Phase 0.5 Steering V8** (DeVec SVD + Layer-Wise Ramp-Up Alpha Scheduling).

---

### 👁️ Pilar 4: Vision Token Compression & Multi-Resolution Pan & Scan (P&S)
* **Sumber Paper:** 
  - `Gemma 3 Technical Report` (Google DeepMind, Mar 2025)
  - `Return of the Encoder` (Multimodal Extension, 2025)
* **Temuan Kritis:**
  - Memproses gambar resolusi tinggi tanpa kompresi membebani $5\text{k}-10\text{k}$ token visual.
  - SigLIP 400M dengan **4x4 Average Pooling** mengompresi fitur visual resolusi $896 \times 896$ menjadi tepat **256 soft tokens** (`<image_soft_token>`, ID `256001`).
  - **Pan & Scan (P&S):** Pemotongan non-overlapping tiles untuk gambar non-square / rasio ekstrem dilakukan hanya saat inferensi, menjaga dimensi input transformer tetap konstan.
* **Peluang Penerapan (V8):**
  - Menjaga kompresi 256 soft tokens tetap konsisten pada `Seq2SeqVisionCollator` dan `VisionORPOCollator`.

---

### 🧩 Pilar 5: Bidirectional Masking & Optimal Pooling untuk Task Non-Generatif
* **Sumber Paper:** 
  - `Adapting Decoder-Based Language Models for Diverse Encoder Downstream Tasks (Gemma Encoder)` (Google, Mar 2025)
* **Temuan Kritis:**
  - Mengubah causal masking menjadi bidirectional masking pada model berbasis Gemma secara instan menaikkan skor NLI, Klasifikasi, dan Ranking sebesar **+5% s.d. +10%**.
  - Right padding terbukti setara dengan left padding pada model encoder bidirectional, namun tidak menimbulkan distorsi offset posisi.
  - Dropout 0.1 pada feedforward dan attention softmax meningkatkan generalisasi saat data downstream terbatas.
* **Peluang Penerapan (V8):**
  - Memastikan encoder T5Gemma-2 beroperasi dalam mode full bidirectional attention dan right padding.

---

## 🗺️ Matriks Alokasi Inovasi per Versi (V8 vs V9 vs V10)

| Fitur / Teknik | Target Versi | Status Implementasi | Dampak Utama |
|---|:---:|:---:|---|
| **DeVec SVD Task Vector Steering** | **V8** | Ready in Blueprint | Mengisolasi instruction vector murni tanpa noise |
| **Layer-Wise Ramp-Up Alpha** | **V8** | Ready in Blueprint | Stabilkan FFN & Norm decoder ($0.05 \to 0.25 \to 0.08$) |
| **OrScale-LM Matrix Optimizer** | **V8** | Ready in Blueprint | Trust ratio scaling mencegah unit mismatch & grad explosion |
| **TLPO Confusion Point Loss** | **V8** | Ready in Blueprint | Mencegah degradasi bahasa Indonesia saat ORPO |
| **MTO Task Prefix Routing** | **V8** | Ready in Blueprint | Formatting prompt berbasis kategori tugas `<unused1>`..`<unused6>` |
| **Cross-Architecture KD (Phi-3.5/Gemma3 IT Teacher)** | **V9** | Planned (Next) | Distilasi probabilitas token dari teacher decoder-only ke encoder-decoder |
| **GRUN Gated Representation Tuning** | **V9** | Planned (Next) | Soft gate ReFT untuk selective unlearning / rejection guard |
| **INTRA Reverse-QWK KV Compression** | **V10** | Planned (Advanced) | Penghematan 30x KV cache cross-attention pada long-context |
| **Cassandra Self-Speculative Decoding** | **V10** | Planned (Advanced) | Akselerasi inferensi 2.5x tanpa model draft eksternal |

---

Dokumen ini menjadi referensi komparatif resmi untuk pengembangan lanjutan pipeline T5Gemma-2 Instruct.
