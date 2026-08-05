# [REVERSE ENGINEERING] T5Gemma Merged Attention & Task Vector Arithmetic

**Last Updated:** 5 Agustus 2026  
**Context:** Deep dive mekanisme *Merged Attention* T5Gemma 2, penyebab kegagalan *Weight Grafting* mentah, dan solusi ilmiah menggunakan *Task Vectors (Model Arithmetic)*.

---

## 1. Patofisiologi Kegagalan "Cangkok" (Weight Grafting) Direct Decoder

Mencangkok bobot (*raw weight replacement*) dari model Decoder-only (`Gemma Instruct`) secara langsung ke Decoder `T5Gemma 2` menyebabkan luaran model menjadi **hancur / garbage (loss terbang)**.

### Penyebab Utama Berdasarkan Arsitektur:

1. **Satu Matriks Proyeksi ($\mathbf{W}_k, \mathbf{W}_v$) untuk Dua Ruang Vektor Berbeda:**
   Di Gemma-Instruct (decoder-only), matriks $\mathbf{W}_k$ dan $\mathbf{W}_v$ hanya dilatih untuk memproyeksikan hidden state decoder ($\mathbf{X}$).
   Sedangkan di T5Gemma 2, matriks $\mathbf{W}_k$ dan $\mathbf{W}_v$ memproyeksikan matriks gabungan $[\mathbf{X}; \mathbf{H}]$. Vektor $\mathbf{H}$ berasal dari Encoder yang dilatih dengan **Bidirectional Attention & UL2 Denoising**, sehingga bentuk dan distribusi vektor $\mathbf{H}$ sangat berbeda dengan $\mathbf{X}$. Weight Gemma-IT tidak pernah dilatih untuk memproses $\mathbf{H}$.

2. **Joint Softmax Normalization Miscalibration:**
   - Di Gemma-IT: Softmax dinormalisasi hanya sepanjang sekuens decoder ($m$).
   - Di T5Gemma Merged Attention: Softmax dinormalisasi secara **bersama-sama (jointly)** sepanjang sekuens total $(m + n)$.
   - Ketika weight Gemma-IT dipasang mentah-mentah, nilai dot-product $\mathbf{Q}\mathbf{K}^T$ untuk bagian encoder ($\mathbf{H}$) menghasilkan angka yang tidak terkalibrasi. Perhatian (*attention weight*) dominan tersedot ke noise atau bernilai ekstrem.

3. **Perubahan Perilaku Masking ($\mathbf{M}$):**
   Masking $\mathbf{M} \in \mathbb{R}^{m \times (m+n)}$ di T5Gemma 2 mengizinkan token decoder melihat seluruh token encoder secara penuh (bidirectional), sementara Gemma-IT hanya dilatih dengan *causal mask*.

---

## 2. Solusi Ilmiah: Task Vector / Model Arithmetic ($\Delta_{\text{instruct}}$)

Daripada melakukan pengantian bobot secara mentah (*raw replacement*), kita mengisolasi **"Vektor Pengetahuan Instruksi"** ($\Delta_{\text{instruct}}$) dari Gemma-IT, lalu menyuntikkannya ke T5Gemma Base sebagai titik awal (*starting point / warm-start*) sebelum Supervised Fine-Tuning (SFT).

### Formula Matematika:

1. **Hitung Task Vector (Weight Delta):**
   $$\Delta_{\text{instruct}} = W_{\text{Gemma-IT}} - W_{\text{Gemma-Base}}$$

2. **Injeksikan ke T5Gemma Base:**
   $$W_{\text{T5Gemma-Init}} = W_{\text{T5Gemma-Base}} + \lambda \cdot \Delta_{\text{instruct}}$$
   *(dengan $\lambda$ biasanya bernilai $0.5$ hingga $1.0$)*.

### Mengapa Pendekatan Ini Berhasil?

- **Layer MLP / FFN (100% Identik):** Knowledge, penafsiran instruksi, dan gaya penalaran tersimpan dominan di layer FFN (`gate_proj`, `up_proj`, `down_proj`). Dimensi dan struktur FFN antara Gemma-Base, Gemma-IT, dan T5Gemma **sama persis**.
- **Preservasi Merged Attention:** Menambahkan $\Delta_{\text{instruct}}$ tidak merusak matriks Merged Attention yang dipelajari T5Gemma Base selama pretraining 2T token UL2, tetapi hanya mengarahkan (*steering*) aktivasi ke gaya percakapan instruksi.
