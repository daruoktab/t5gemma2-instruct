# T5-Gemma-2 Instruct Training Strategy

**Tanggal:** 15 Mei 2026  
**Versi:** 2.2 (Updated — Vision Architecture & Weight Transplant Analysis)

## 1. Intensi & Tujuan
- Membangun chatbot bilingual (Indonesia-first) berbasis T5-Gemma-2 4B yang unggul di percakapan *multi-turn* dan mahir mengeksekusi *task* spesifik (seperti *summarization*, terjemahan, ekstraksi) secara langsung di tengah obrolan.
- Memanfaatkan secara maksimal arsitektur Encoder-Decoder untuk mempertahankan pemahaman konteks yang panjang (di Encoder) tanpa mengalami degradasi atensi yang sering terjadi pada model *decoder-only*.

## 2. Kondisi Saat Ini (Aset & Infrastruktur)
- **Model Target:** `google/t5gemma-2-4b-4b` (Base Model).
- **Infrastruktur / GPU:** Fleksibel (dapat disesuaikan dengan ketersediaan). Efisiensi tinggi akan dicapai melalui kombinasi metode *reference-free alignment* dan LoRA.
- **Kapasitas Konteks (Sequence Length):** Menargetkan `max_source_length=8192` dan `max_target_length=2048` untuk dapat mengakomodasi data obrolan yang sangat panjang.
- **Dataset Tersedia:**
  1. `chat_multiturn`: 2.500 percakapan *multi-turn* berkualitas tinggi yang sudah mencakup instruksi spesifik.
  2. `indoqa_documents`: ~4.400 data *single-turn* ekstraksi dokumen untuk memperkuat kemampuan *reading comprehension* dan *grounding*.

## 3. Catatan Arsitektur T5Gemma-2 (Kritis untuk Fine-tuning)

T5Gemma 2 memiliki dua perubahan arsitektur signifikan dibandingkan T5Gemma v1 yang **langsung berdampak ke konfigurasi training dan LoRA**:

### A. Tied Word Embeddings
Semua word embeddings (encoder input, decoder input, decoder output/softmax) di-*tie* menjadi satu shared matrix (`model.shared`). Ini mengurangi parameter ~10.5% dengan hampir tanpa degradasi kualitas. [Sumber: arXiv 2512.14856]

**Implikasi ke LoRA:** Jangan apply LoRA ke `embed_tokens` atau `lm_head` secara independen — akan memutus konsistensi tied weights. LoRA hanya boleh diterapkan ke projection layers di dalam attention dan FFN.

### B. Merged Attention di Decoder
Self-attention dan cross-attention decoder digabung menjadi **satu modul tunggal**. K dan V dibentuk dengan mengkonkatenasi encoder outputs dan decoder states, dengan masking yang mempertahankan causal visibility. Tidak ada lagi layer `EncDecAttention` / `encoder_attn` terpisah seperti di T5 klasik.

**Implikasi ke LoRA Target Modules:** List `["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]` kemungkinan masih valid, tapi **wajib diverifikasi** dengan inspeksi nama layer aktual sebelum training:

```python
from transformers import AutoModelForSeq2SeqLM
model = AutoModelForSeq2SeqLM.from_pretrained("google/t5gemma-2-4b-4b")
for name, module in model.named_modules():
    if "proj" in name or "dense" in name:
        print(name)
```

### C. Total Parameter Efektif
Model 4B-4B memiliki total **~7B parameter** (tidak termasuk vision encoder SigLIP). Kalkulasi VRAM untuk training harus berpatokan ke angka ini, bukan 4B.

### D. Konfirmasi dari Paper: SFT Ringan Sudah Cukup Powerful
Google sendiri hanya melakukan *lightweight SFT* (tanpa RL) pada T5Gemma 2 post-training dan hasilnya sudah melampaui Gemma 3 di banyak benchmark. Ini mengkonfirmasi bahwa dataset 6.9K (2.5K chat + 4.4K IndoQA) berpotensi menghasilkan model kompetitif tanpa perlu skala besar.

---

## 4. Analisis Baseline (Eksperimen Model 270M)
Hal ini memperkuat argumen untuk beralih ke **Model 4B** yang memiliki "otak" lebih besar dan pengetahuan *pre-trained* yang lebih luas, serta perlunya penyeimbangan dataset yang lebih baik.

### Bukti Empiris: Log Inferensi (Model 270M)
Berikut adalah hasil pengujian langsung menggunakan skrip `test_270m_inference.py` dengan format `<start_of_turn>` yang konsisten:

| Prompt (Input) | Respons Model (Output) | Analisis Kegagalan |
| :--- | :--- | :--- |
| "Halo, siapa kamu?" | "Halo! Selamat datang di dunia. Mengapa kamu ingin memanggil saya?" | Respons terlalu generik dan sedikit defensif/aneh. |
| "Tolong terjemahkan... 'Kucing itu sedang tidur di atas meja.'" | "Waduh, ini kalimatnya yang sangat sulit untuk diluaskan. Ingat cara mengolah kata dalam bahasa Inggris: 'To be or not to do'..." | **Gagal total.** Model malah berhalusinasi tentang idiom Inggris dan mengeluh kesulitan. |
| "Bisa jelaskan apa itu fotosintesis dengan singkat?" | "Iya, itu namanya fotosintesis." | **Gagal total.** Model hanya mengulang terminologi tanpa memberikan penjelasan apa pun. |

**Kode Inferensi yang Digunakan:**
```python
# Format prompt yang digunakan (konsisten dengan training 270M)
formatted_prompt = f"<start_of_turn>user\n{system_prompt}\n\n{prompt}<end_of_turn>\n<start_of_turn>model\n"

# Generation config
outputs = model.generate(
    **inputs, 
    max_new_tokens=256, 
    do_sample=True, 
    temperature=0.7, 
    repetition_penalty=1.2,
    eos_token_id=tokenizer.convert_tokens_to_ids("<end_of_turn>")
)
```

---

## 5. Strategi Pelatihan (Training Pipeline)
Berdasarkan riset literatur terkini mengenai tuning arsitektur Encoder-Decoder, pelatihan akan dibagi menjadi dua fase utama:

### A. Tahap 1: Supervised Fine-Tuning (SFT) pada Base Model
Pelatihan SFT dilakukan langsung pada **Base Model** (`google/t5gemma-2-4b-4b`).
- **Alasan:** Melatih *base model* terbukti memberikan fleksibilitas lebih tinggi dalam menyerap pengetahuan baru dan *style* instruksi dibandingkan model yang sudah melalui *instruction tuning*. Dikonfirmasi oleh ORPO paper (Hong et al., 2024) bahwa SFT tetap *imperative* untuk konvergensi yang berhasil.
- **Data yang Digunakan:** Campuran dari `chat_multiturn` dan `indoqa_documents`, dengan urutan Curriculum Learning (lihat §6).

**Setup teknis penting (dari tutorial resmi DataCamp, Jan 2026 + Transformers docs):**
- Gunakan `Seq2SeqTrainer` + `Seq2SeqTrainingArguments`
- `bf16=True`, `tf32=True`, `gradient_checkpointing=True`
- `tokenizer.padding_side = "right"` wajib untuk Seq2Seq
- Labels padding di-mask ke `-100` agar tidak masuk ke loss computation
- Jangan truncate sisi encoder input untuk mencegah token mismatch

### B. Tahap 2: Alignment dengan Metode Reference-Free (ORPO / SimPO)
Setelah SFT selesai, dilanjutkan dengan *alignment* menggunakan **ORPO** atau **SimPO**.
- **Alasan:** Metode *reference-free* signifikan lebih efisien VRAM karena tidak perlu load *reference model* secara bersamaan. Krusial untuk encoder-decoder yang membutuhkan alokasi memori lebih besar.
- **ORPO (Hong et al., 2024):** Menggabungkan SFT dan alignment jadi satu *monolithic objective*. Terbukti efektif di model 125M–7B. Cocok jika ingin pipeline satu tahap.
- **SimPO (Meng et al., 2024):** Menggunakan *average log probability* sequence sebagai implicit reward tanpa reference model. Outperform DPO hingga 6.4 poin di AlpacaEval 2. Namun perlu diwaspadai **URSLA shortcut** — model belajar truncate respons buruk lebih awal daripada benar-benar belajar kontennya (REFA paper, 2024).
- **Alternatif tambahan — Curry-DPO (Pattnaik et al., 2024):** Mengkonstruksi multiple preference pairs per prompt dengan kualitas berbeda, lalu melatih dengan curriculum easy-to-hard. Konsisten meningkatkan performa di MT-bench. Bisa dikombinasikan dengan ORPO/SimPO.

**Kebutuhan Data Preferensi & Cara Generate *Rejected* Responses:**
Saat ini dataset hanya punya jawaban *chosen*. Strategi generate *rejected* yang aman berdasarkan paper:
- **Gunakan self-generated rejected** dari SFT model sendiri (temperature tinggi atau degraded prompt) — JANGAN gunakan model eksternal yang lebih kuat (GPT-4o dll.) sebagai *chosen* dipasangkan dengan self-generated *rejected*, karena terbukti menyebabkan reward hacking dan high attack success rate pada jailbreak (Wang et al., 2025 — "More is Less").
- Pola terbaik: SFT selesai → sample k responses per prompt dari SFT model → pilih yang terbaik sebagai *chosen*, yang terburuk sebagai *rejected* → alignment (RS-DPO pattern, Khaki et al., 2024).

## 6. Mekanisme Kerja & Penanganan Teknis
- **Data Flattening & EOS Token:** Dataset *nested* (2.500 percakapan) akan diurai (*flatten*) dengan penanda format percakapan standar: `<start_of_turn>user\n` dan `<start_of_turn>model\n` sebelum diproses ke dalam *trainer*. Di akhir setiap target respons, **wajib ditambahkan token EOS** (`<end_of_turn>`). Ini krusial karena *base model* secara natural dilatih untuk terus memprediksi kata (melengkapi teks) tanpa henti; token EOS inilah yang mengajari model kapan harus berhenti berbicara setelah instruksi selesai.
- **Konfigurasi LoRA Ekstensif:** Pelatihan model 4B menggunakan konfigurasi LoRA yang lebih "lebar" dibanding eksperimen 270M:
  - **Rank (r):** `32` (dengan `alpha=64`) untuk kapasitas lebih besar menyerap *instruction-following patterns*.
  - **Target Modules:** Semua linear layers: `["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]`. Perlu diverifikasi ulang karena merged attention T5Gemma-2 mengubah nama/struktur layer dibanding T5 klasik (lihat §3).
  - **Upgrade yang Direkomendasikan — DoRA (Liu et al., 2024):** Mendekomposisi pre-trained weight menjadi komponen *magnitude* dan *direction*, menggunakan LoRA hanya untuk directional updates. Terbukti meningkatkan learning capacity dan training stability tanpa tambahan inference overhead. Drop-in replacement untuk LoRA standar.
  - **Upgrade yang Direkomendasikan — rsLoRA (Kalajdzievski, 2023):** Scaling factor standar LoRA (dibagi rank) memperlambat learning di rank tinggi. rsLoRA menggunakan faktor `sqrt(rank)` sehingga rank lebih tinggi bisa dipakai secara efektif. Sangat relevan untuk r=32+.
- **Manajemen Unused Tokens:** *(Diperbarui berdasarkan analisis tokenizer langsung, 12 Mei 2026 — lihat §7 untuk detail lengkap.)* T5Gemma2 mewarisi tokenizer Gemma 3 dengan total **6.241 unused tokens** tersebar di dua blok ID, bukan satu token seperti yang sempat diasumsikan. Strategi mitigasi berlapis:
  1. **Sebelum training — Reinitialize embeddings:** Set embedding unused tokens ke `mean_embedding + small_noise` dari token normal, agar model punya entry-point bermakna bukan random noise.
  2. **Freeze unused embeddings selama training:** Gradient hook agar embedding 6.241 token ini tidak ikut diupdate — mencegah mereka "mencuri" gradient dari token bermakna.
  3. **Saat inferensi — LogitsProcessor:** Suppress probabilitas ke `-inf` sebagai safety net. Termasuk juga suppress vision tokens (`<v_patch>`, `<end_of_image>`, `<image_soft_token>`) karena training ini pure text.

  ```python
  # Definisi lengkap IDs yang perlu disuppress (pure text inference)
  SUPPRESS_IDS = list(range(6, 105))           # <unused0>–<unused98> (Blok 1, 99 token)
  SUPPRESS_IDS += list(range(256002, 262144))  # <unused100>–<unused6241> (Blok 2, 6142 token)
  SUPPRESS_IDS += [255999, 256000, 256001]     # <v_patch>, <end_of_image>, <image_soft_token>
  # Total: 6.244 token disuppress

  # Layer 1: Reinitialize sebelum training
  embedding_matrix = model.shared.weight.data
  normal_ids = [i for i in range(tokenizer.vocab_size)
                if i not in set(SUPPRESS_IDS)]
  mean_emb = embedding_matrix[normal_ids].mean(dim=0)
  for uid in SUPPRESS_IDS:
      embedding_matrix[uid] = mean_emb + torch.randn_like(mean_emb) * 1e-4

  # Layer 2: Freeze via gradient hook
  suppress_tensor = torch.tensor(SUPPRESS_IDS, dtype=torch.long)
  def hook(grad):
      grad[suppress_tensor] = 0
      return grad
  model.shared.weight.register_hook(hook)  # cukup satu hook karena tied embeddings

  # Layer 3: LogitsProcessor saat inferensi
  class SuppressUnwantedTokensProcessor(LogitsProcessor):
      def __init__(self, suppress_ids: list[int]):
          self.suppress_ids = torch.tensor(suppress_ids, dtype=torch.long)
      def __call__(self, input_ids, scores):
          scores[:, self.suppress_ids] = float('-inf')
          return scores
  ```
- **Curriculum Learning:** Urutan data training diatur secara metodis berdasarkan temuan paper Data-CUBE (Min et al., 2024) yang secara langsung relevan: mengatur urutan data multi-task untuk meminimalkan cross-task interference, baik di level task maupun level instance (easy-to-hard per task).
  - **Urutan Task:** IndoQA dulu (*reading comprehension* & grounding faktual) → Chat Multi-turn (gaya interaksi sosial). Tujuan: model memantapkan akurasi faktual sebelum belajar flexibilitas gaya.
  - **Implikasi empiris:** Paper ordering study (Chen et al., 2024) menemukan task ordering berdampak hingga +6% performance gain atau -4% loss. Urutan ini perlu divalidasi dengan *ablation* setelah training selesai.
  - **Curry-DPO extension:** Pada tahap alignment, multiple preference pairs per prompt bisa diurutkan juga dari easy-to-hard berdasarkan reward gap antar *chosen*-*rejected*.

---

## 7. Analisis Tokenizer T5Gemma2 vs Gemma 3 (Baru, 12 Mei 2026)

Analisis langsung terhadap file `tokenizer.json` dari kedua model menghasilkan temuan penting:

### Struktur Tokenizer (Perbandingan)

| Metrik | Gemma 3 | T5Gemma2 |
|---|---|---|
| `vocab_size` | 262.144 | 262.144 |
| `added_tokens` | 6.415 | 6.414 |
| Unused tokens | **6.242** | **6.241** |
| Non-unused added tokens | 173 | 173 |

### Satu-satunya Perbedaan

Kedua tokenizer **identik persis** kecuali satu hal: posisi `<image_soft_token>`.

```
<image_soft_token>:
  Gemma 3   → id = 262.144  (di luar vocab range utama, slot "extra")
  T5Gemma2  → id = 256.001  (menggantikan 1 slot unused, masuk dalam vocab)
```

T5Gemma2 mengambil satu slot unused (`<unused99>` — itulah kenapa ada gap di Blok 1, `<unused99>` tidak ada) dan menggantinya dengan `<image_soft_token>` di id 256.001. Token ini dipakai sebagai per-patch vision token saat processing gambar.

### Peta Lengkap Unused Tokens T5Gemma2

```
Blok 1: id 6–104     → <unused0>–<unused98>    (99 token, low IDs — PALING BERBAHAYA)
         [GAP: id 105 = <start_of_turn>]
Blok 2: id 256.002–262.143 → <unused100>–<unused6241> (6.142 token, high IDs)

Note: <unused99> TIDAK ADA — slotnya dipakai oleh <image_soft_token> (id 256.001)
```

### Mengapa Blok 1 Lebih Berbahaya

Blok 1 (id 6–104) berada di low IDs yang berdekatan dengan special tokens penting (`<pad>=0`, `<eos>=1`, `<bos>=2`). Selama decoding, model yang belum di-finetune cenderung ter-sample ke token-token low-ID ini karena distribusi pre-training. Ini dikonfirmasi oleh bug nyata di llama.cpp di mana Gemma 3 tanpa mitigasi menghasilkan output berupa spam `<unused32>` berulang-ulang.

### Tujuan Resmi Unused Tokens (dari Google)

Menurut Google DeepMind, unused tokens ini **sengaja direservasi** sebagai slot kosong agar fine-tuner bisa mendefinisikan custom tokens tanpa harus meresize vocab (yang mahal dan berisiko di BPE tokenizer). Untuk proyek ini (pure text instruct tuning), seluruh 6.241 unused tokens + 3 vision tokens = **6.244 token total yang perlu disuppress**.

---

## 8. Ringkasan Rekomendasi Tambahan (dari Literature Review)

| Komponen | Rekomendasi | Paper / Sumber |
|---|---|---|
| LoRA variant | Ganti ke **DoRA** atau aktifkan **rsLoRA** | DoRA (Liu et al., 2024), rsLoRA (2023) |
| Alignment | Pertimbangkan **Curry-DPO** (curriculum + ranked pairs) | Pattnaik et al., 2024 |
| Rejected gen | Self-generate dari SFT model, **bukan** model eksternal lebih kuat | Wang et al., 2025; Khaki et al., 2024 |
| Curriculum | Formalize task ordering + instance difficulty sorting | Min et al., 2024 (Data-CUBE) |
| Unused tokens | Reinitialize + freeze gradient + LogitsProcessor (3 layer) | Chen et al., 2025; Paech et al., 2025; analisis langsung 12 Mei 2026 |
| Vision tokens | Suppress `<v_patch>`, `<end_of_image>`, `<image_soft_token>` di pure text training | Analisis tokenizer 12 Mei 2026 |
| Length bias | Monitor URSLA shortcut pada SimPO/ORPO | Gupta et al., 2024 (REFA) |
| LoRA target modules | Verifikasi ulang karena merged attention T5Gemma-2 | arXiv 2512.14856 |
| Suppress list | Total **6.244 token** yang perlu disuppress — lihat §6 untuk daftar lengkap | Analisis tokenizer 12 Mei 2026 |
| Vision Transplant | Cangkok bobot SigLIP & Projector dari Gemma 3 IT | Analisis 15 Mei 2026 |

---

## 9. Strategi "Cangkok" Vision & Weight Transplant (Baru, 15 Mei 2026)

Berdasarkan analisis arsitektur vision pada Gemma 3 dan T5Gemma-2, terdapat peluang untuk melakukan **Weight Transplant** (Cangkok Bobot) dari model Gemma 3 Instruct (`gemma-3-4b-it`) ke model T5Gemma-2 sebelum tahap SFT dilakukan.

### A. Identitas Vision Tower (SigLIP)
Inspeksi terhadap `vision_config` menunjukkan bahwa kedua model menggunakan komponen yang **identik secara struktural**:
- **Model:** `SiglipVisionModel`
- **Konfigurasi:** 27 layers, 1152 hidden size, 16 attention heads, 14 patch size.
- **Kesimpulan:** Bobot dari SigLIP Gemma 3 IT dapat langsung dipindahkan ke SigLIP T5Gemma-2. Hal ini memberikan ekstraksi fitur visual yang sudah "terlatih" untuk tugas-tugas instruksi.

### B. Analisis Vision Projector (Adapter)
Bagian *projector* (yang menghubungkan vision tower ke model bahasa) juga menunjukkan kompatibilitas tinggi:
- **Arsitektur:** Keduanya menggunakan MLP (Linear -> GeLU -> Linear).
- **Dimensi Output:** Keduanya memiliki output dimensi **2560** (cocok dengan hidden size model bahasa 4B).
- **Strategi IT:** Pada model Instruct, projector ini biasanya telah di-align untuk menyajikan informasi visual dalam bentuk yang lebih "paham instruksi". 

### C. Langkah Pelaksanaan "Cangkok"
1. **Mapping State Dict:** Melakukan mapping manual dari `model.vision_tower` (Gemma 3) ke `model.encoder.vision_tower` (T5Gemma-2) dan `model.multi_modal_projector` ke `model.encoder.multi_modal_projector`.
2. **Synchronized Decoder:** Karena strategi ini juga melibatkan penggunaan bobot decoder Gemma 3 IT, maka "pemahaman" decoder terhadap sinyal visual dari encoder akan tetap sinkron (karena keduanya berasal dari model keluarga yang sama).
3. **Efisiensi:** Dengan metode ini, model tidak perlu belajar dari nol cara "melihat", melainkan hanya perlu melakukan adaptasi ringan terhadap format Encoder-Decoder T5Gemma-2.

---
