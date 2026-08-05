# Penjelasan Lengkap Metrik Training SFT — Bahasa Sederhana

Dokumen ini menjelaskan **setiap metrik** yang muncul di laporan analisis training SFT Anda. Penjelasan dirancang **tanpa rumus rumit** — cukup logika dan analogi yang mudah dipahami.

---

## Daftar Metrik yang Dijelaskan

| Kategori | Metrik |
|:---|:---|
| **Metrik Loss** | Train Loss, Eval Loss |
| **Metrik Probabilitas** | Perplexity (PPL) |
| **Metrik Kemiripan Kata (N-gram)** | ROUGE-1, ROUGE-2, ROUGE-L |
| **Metrik Kualitas Terjemahan** | BLEU |
| **Metrik Kesesuaian Makna** | METEOR |
| **Metrik Semantik (AI-based)** | BERTScore F1 |
| **Metrik Kecocokan Eksak** | Exact Match (EM) |

---

## 1. Train Loss & Eval Loss (Cross-Entropy Loss)

### 📌 Ngukur Apa?
**Loss** mengukur **seberapa "salah" tebakan model** dibandingkan jawaban yang seharusnya. Semakin rendah loss = semakin tepat tebakan model.

### 🧠 Logika Pengukuran
Bayangkan model sedang menebak kata berikutnya dalam kalimat:

> Referensi: "Ibu pergi ke **pasar** pagi ini"
> Tebakan model: "Ibu pergi ke **toko** pagi ini"

Model memberikan **skor keyakinan** (probabilitas) untuk setiap kata kosakata. Kalau model memberi skor keyakinan tinggi ke kata "toko" padahal jawaban benarnya "pasar", maka **loss-nya besar**.

- **Train Loss**: Dihitung dari data **latih** (data yang model pelajari langsung)
- **Eval Loss**: Dihitung dari data **validasi** (data yang model **belum pernah lihat** untuk testing)

### 💡 Analogi
- **Train Loss** = Nilai ulangan harian (soal yang sudah dipelajari)
- **Eval Loss** = Nilai ujian akhir (soal baru, untuk ukur pemahaman sungguhan)

### 📊 Cara Baca
| Kondisi | Arti |
|:---|:---|
| Train Loss ↓ terus, Eval Loss ↓ | ✅ Model belajar dengan baik |
| Train Loss ↓, Eval Loss ↑ | ⚠️ **Overfitting** — model hafal soal latihan tapi gagal di soal baru |
| Train Loss ↓, Eval Loss sedikit naik lalu turun lagi | 🔄 Fluktuasi normal, masih dalam batas wajar |

### 🎯 Di Laporan Anda
- Train Loss turun konsisten: **3.57 → 2.73** ✅
- Eval Loss terendah di step 800 (**2.8566**), naik tipis di step 1000 (**2.8646**)
- Kenaikan tipis ini = **overfitting ringan**, tapi masih aman karena metrik teks tetap membaik

---

## 2. Perplexity (PPL)

### 📌 Ngukur Apa?
PPL mengukur **seberapa "bingung" model** saat menebak kata berikutnya. Semakin rendah PPL = semakin yakin dan tepat model.

### 🧠 Logika Pengukuran
PPL adalah **transformasi dari Eval Loss** dengan rumus sederhana: `PPL = e^(Eval Loss)` (e dipangkatkan Eval Loss).

Intinya: PPL menjawab pertanyaan **"Kalau model harus menebak, rata-rata ada berapa kata yang sama-sama mungkin di pikiran model?"**

- PPL = 1 → Model 100% yakin jawabannya (sempurna)
- PPL = 17 → Rata-rata model ragu antara ~17 kata yang sama-sama mungkin
- PPL = 100 → Model sangat bingung, banyak kata yang sama-sama mungkin

### 💡 Analogi
Bayangkan Anda menebak kata dalam teka-teki silang:
- PPL rendah = Anda hanya ragu di 2-3 kata (sudah hampir yakin)
- PPL tinggi = Anda ragu di 50+ kata (masih bingung)

### 📊 Cara Baca
| Nilai PPL | Interpretasi |
|:---|:---|
| < 20 | ✅ Model cukup yakin |
| 20 - 50 | 🔄 Wajar untuk model generative |
| > 100 | ⚠️ Model masih bingung |

### 🎯 Di Laporan Anda
- PPL terbaik di step 800: **17.40** ✅
- Naik tipis di step 1000: **17.54** (karena Eval Loss naik tipis)

---

## 3. ROUGE-1, ROUGE-2, ROUGE-L (Recall-Oriented Understudy for Gisting Evaluation)

### 📌 Ngukur Apa?
ROUGE mengukur **seberapa banyak kata/frasa penting dari jawaban referensi yang berhasil muncul di jawaban model**. Fokusnya: **kelengkapan** (recall).

### 🧠 Logika Pengukuran

#### ROUGE-1 (Unigram)
Menghitung **kata per kata** yang cocok.

> Referensi: "Ibu pergi ke pasar"
> Model: "Ibu pergi ke toko"
> 
> Kata cocok: "Ibu", "pergi", "ke" = 3 dari 4 kata
> ROUGE-1 = 3/4 = 75%

#### ROUGE-2 (Bigram)
Menghitung **pasangan dua kata berurutan** yang cocok.

> Referensi: "Ibu pergi ke pasar"
> Pasangan: ("Ibu pergi"), ("pergi ke"), ("ke pasar")
> 
> Model: "Ibu pergi ke toko"
> Pasangan: ("Ibu pergi"), ("pergi ke"), ("ke toko")
> 
> Pasangan cocok: ("Ibu pergi"), ("pergi ke") = 2 dari 3
> ROUGE-2 = 2/3 = 67%

**Kenapa ROUGE-2 lebih ketat?** Karena urutan kata juga harus pas, bukan cuma kata individualnya.

#### ROUGE-L (Longest Common Subsequence)
Mencari **urutan kata terpanjang yang sama** antara referensi dan model (tidak harus berurutan langsung, tapi urutannya harus sama).

> Referensi: "Ibu pergi ke pasar pagi ini"
> Model: "Ibu pagi ini pergi ke pasar"
> 
> Subsequence terpanjang yang sama urutannya: "Ibu pergi ke pasar" (4 kata)
> ROUGE-L mengukur panjang subsequence ini relatif terhadap total kata

### 💡 Analogi
- **ROUGE-1** = "Dari 10 kata kunci jawaban benar, model berhasil menyebut 8" (kelengkapan kata)
- **ROUGE-2** = "Dari 9 frasa 2-kata jawaban benar, model berhasil menyebut 6" (kelengkapan frasa)
- **ROUGE-L** = "Seberapa panjang kalimat model mengikuti struktur kalimat referensi" (kelengkapan struktur)

### 📊 Cara Baca
| Nilai ROUGE | Interpretasi |
|:---|:---|
| > 60% | ✅ Sangat baik |
| 40-60% | 🔄 Cukup baik |
| < 30% | ⚠️ Perlu perbaikan |

### 🎯 Di Laporan Anda
- ROUGE-1: 53.38% → 60.07% ✅ (naik 6.69%)
- ROUGE-2: 34.25% → 40.17% ✅ (naik 5.92%)
- ROUGE-L: 49.05% → 55.51% ✅ (naik 6.46%)

---

## 4. BLEU (Bilingual Evaluation Understudy)

### 📌 Ngukur Apa?
BLEU mengukur **seberapa tepat jawaban model meniru jawaban referensi**. Fokusnya: **presisi** (ketepatan), kebalikan dari ROUGE yang fokus recall.

### 🧠 Logika Pengukuran
BLEU menghitung **persentase kata/frasa di jawaban model yang BENAR-BENAR ada di referensi**.

> Referensi: "Ibu pergi ke pasar"
> Model: "Ibu pergi ke pasar pagi ini"
> 
- Model menyebut 6 kata, 4 di antaranya ada di referensi
- Presisi = 4/6 = 67%
- **Tapi** model menambahkan kata berlebihan ("pagi ini") → BLEU memberi **penalti** karena model "terlalu banyak ngomong"

BLEU juga menggabungkan presisi untuk unigram, bigram, trigram, dan 4-gram, lalu memberi penalti jika jawaban model terlalu pendek atau panjang.

### 💡 Analogi
- **ROUGE** = "Apakah jawaban model LENGKAP?" (semua poin penting disebut?)
- **BLEU** = "Apakah jawaban model TEPAT?" (tidak menambah-nambah hal yang tidak perlu?)

BLEU seperti penilai yang **ketat dengan kata berlebihan**. Kalau Anda jawab pertanyaan dengan banyak omong kosong, BLEU akan memberi nilai rendah meskipun jawaban benarnya ada di dalamnya.

### 📊 Cara Baca
| Nilai BLEU | Interpretasi |
|:---|:---|
| > 30% | ✅ Sangat baik (untuk generative) |
| 15-30% | 🔄 Cukup baik |
| < 10% | ⚠️ Perlu perbaikan |

**Catatan:** BLEU untuk generative AI biasanya lebih rendah daripada ROUGE karena BLEU ketat dengan penalti. Jangan terkejut kalau BLEU 15% itu sudah bagus.

### 🎯 Di Laporan Anda
- BLEU: 12.57% → 15.94% ✅ (naik 3.37%)

---

## 5. METEOR (Metric for Evaluation of Translation with Explicit ORdering)

### 📌 Ngukur Apa?
METEOR mengukur **kesesuaian makna dan urutan** antara jawaban model dan referensi. Lebih **cerdas** dari ROUGE/BLEU karena mempertimbangkan **sinonim dan variasi kata**.

### 🧠 Logika Pengukuran
METEOR bekerja dalam beberapa tahap:

1. **Exact match**: Kata yang sama persis
2. **Stem match**: Kata dengan akar yang sama (misal: "berlari" vs "lari")
3. **Synonym match**: Kata sinonim (misal: "besar" vs "luas")

Lalu METEOR memberi **bonus** jika kata-kata yang cocok muncul dalam **urutan yang rapi** (tidak acak), dan memberi **penalti** jika urutannya berantakan.

### 💡 Analogi
Bayangkan dua siswa menjawab pertanyaan:
- Siswa A: "Ibu pergi ke pasar" (sama persis dengan kunci)
- Siswa B: "Ibu menuju ke pasar" (sinonim "pergi" → "menuju")

**ROUGE/BLEU** akan memberi Siswa B nilai lebih rendah karena katanya tidak sama persis.
**METEOR** akan memberi Siswa B nilai tinggi karena "pergi" dan "menuju" itu **sinonim** — maknanya sama.

METEOR seperti guru yang **memahami makna**, bukan cuma mencocokkan kata.

### 📊 Cara Baca
| Nilai METEOR | Interpretasi |
|:---|:---|
| > 50% | ✅ Sangat baik |
| 30-50% | 🔄 Cukup baik |
| < 20% | ⚠️ Perlu perbaikan |

### 🎯 Di Laporan Anda
- METEOR: 49.88% → 56.21% ✅ (naik 6.33%) — **sangat baik!**

---

## 6. BERTScore F1

### 📌 Ngukur Apa?
BERTScore mengukur **kesamaan makna (semantik)** antara jawaban model dan referensi menggunakan **AI (model BERT)** sebagai "hakim". Ini metrik paling cerdas karena memahami konteks.

### 🧠 Logika Pengukuran
Berbeda dengan ROUGE/BLEU yang cuma mencocokkan kata, BERTScore:

1. Mengubah setiap kata menjadi **vektor embedding** (representasi numerik makna kata)
2. Menghitung **kesamaan makna** antara kata di jawaban model vs referensi
3. Menggabungkan menjadi skor **Precision** (ketepatan), **Recall** (kelengkapan), dan **F1** (rata-rata harmonik keduanya)

### 💡 Analogi
- **ROUGE/BLEU** = Mencocokkan kata seperti **mesin fotokopi** (sama persis atau tidak)
- **METEOR** = Mencocokkan kata seperti **kamus sinonim** (sinonim dianggap cocok)
- **BERTScore** = Mencocokkan makna seperti **manusia yang paham konteks** (memahami bahwa "mobil" dan "kendaraan" terkait, meskipun bukan sinonim langsung)

Contoh:
> Referensi: "Anak itu tertawa gembira"
> Model: "Sang bocah terbahak-bahagia"

- ROUGE/BLEU: Skor rendah (kata berbeda)
- METEOR: Skor menengah (beberapa sinonim cocok)
- BERTScore: Skor **tinggi** (makna kedua kalimat sama — kebahagiaan)

### 📊 Cara Baca
| Nilai BERTScore F1 | Interpretasi |
|:---|:---|
| > 85% | ✅ Sangat baik (makna sangat selaras) |
| 75-85% | 🔄 Cukup baik |
| < 70% | ⚠️ Makna kurang selaras |

### 🎯 Di Laporan Anda
- BERTScore F1: 82.80% → 85.23% ✅ (naik 2.43%) — **makna jawaban sangat selaras!**

---

## 7. Exact Match (EM)

### 📌 Ngukur Apa?
EM mengukur **persentase jawaban model yang SAMA PERSIS 100%** dengan referensi — huruf demi huruf, spasi demi spasi.

### 🧠 Logika Pengukuran
Sangat sederhana: untuk setiap pertanyaan, jawaban model dibandingkan dengan referensi.

- Jika **sama persis** → skor 1 (benar)
- Jika **ada perbedaan 1 huruf saja** → skor 0 (salah)

EM = (jumlah jawaban sama persis) / (total pertanyaan) × 100%

### 💡 Analogi
EM seperti ujian pilihan ganda yang **harus diisi dengan jawaban kunci persis**. Kalau kunci jawaban "Jakarta" dan Anda tulis "jakarta" (huruf kecil), dianggap **salah**.

### 📊 Cara Baca
| Nilai EM | Interpretasi |
|:---|:---|
| > 30% | ✅ Sangat baik (untuk generative AI) |
| 15-30% | 🔄 Cukup baik |
| < 10% | ⚠️ Banyak jawaban tidak persis sama |

**Catatan Penting:** EM **sangat ketat**. Untuk model generative AI, EM rendah itu **normal** karena model sering memparafrase jawaban (maknanya sama tapi kata-katanya berbeda). EM tinggi menunjukkan model sangat patuh pada format jawaban referensi.

### 🎯 Di Laporan Anda
- EM: 21.49% → 27.33% ✅ (naik 5.84%) — **naik signifikan, model makin patuh format**

---

## 📊 Ringkasan: Hubungan Antar Metrik

```
┌─────────────────────────────────────────────────────────────┐
│                    KATEGORI METRIK                          │
├──────────────────┬──────────────────────────────────────────┤
│                  │  Train Loss → Belajar dari data latih    │
│  LOSS &          │  Eval Loss  → Uji di data baru           │
│  PROBABILITAS    │  PPL         → Seberapa bingung model     │
│                  │  (Semua ini saling terkait)              │
├──────────────────┼──────────────────────────────────────────┤
│                  │  ROUGE-1 → Kelengkapan kata              │
│  KEMIRIPAN KATA  │  ROUGE-2 → Kelengkapan frasa             │
│  (N-gram)        │  ROUGE-L → Kelengkapan struktur          │
│                  │  BLEU    → Ketepatan (dengan penalti)    │
├──────────────────┼──────────────────────────────────────────┤
│  KESESUAIAN      │  METEOR → Sinonim & urutan              │
│  MAKNA           │  BERTScore → Makna semantik (AI judge)   │
├──────────────────┼──────────────────────────────────────────┤
│  KECOCOKAN       │  Exact Match → Sama persis 100%          │
│  EKSAK           │  (Paling ketat, paling sulit)            │
└──────────────────┴──────────────────────────────────────────┘
```

### 🎯 Urutan dari Paling Ketat ke Paling Fleksibel

1. **Exact Match** — Paling ketat (harus sama persis)
2. **BLEU** — Ketat (presisi + penalti kata berlebih)
3. **ROUGE-2** — Ketat (bigram harus cocok)
4. **ROUGE-1** — Cukup ketat (kata per kata)
5. **ROUGE-L** — Cukup fleksibel (subsequence)
6. **METEOR** — Fleksibel (sinonim diakui)
7. **BERTScore** — Paling fleksibel (makna semantik)

---

## 🔍 Kenapa Loss Naik Tapi Metrik Teks Membagi? (Fenomena di Laporan Anda)

Ini pertanyaan bagus! Berikut penjelasan sederhananya:

### Skenario
- Step 800: Eval Loss = 2.8566, ROUGE-1 = 59.01%
- Step 1000: Eval Loss = 2.8646 (naik), ROUGE-1 = 60.07% (naik)

### Penjelasan Logis

**Loss** mengukur **keyakinan probabilitas** model, bukan ketepatan akhir.

Bayangkan model menebak kata "pasar":
- **Step 800**: Model yakin 80% ke "pasar", 15% ke "toko", 5% ke "mall" → Loss rendah
- **Step 1000**: Model yakin 60% ke "pasar", 25% ke "toko", 15% ke "mall" → Loss naik (kurang yakin)

**Tapi** saat generate teks, model tetap memilih kata dengan probabilitas tertinggi → "pasar" tetap muncul di kedua step.

Jadi:
- **Loss naik** = Model jadi **kurang yakin** (probabilitas menyebar)
- **Metrik teks naik** = Tapi **kata yang dipilih tetap benar** (karena masih yang tertinggi)

### 💡 Analogi
Seperti siswa yang ujian:
- Sebelumnya: yakin 90% jawaban A (benar)
- Sekarang: yakin 60% jawaban A (benar), 40% jawaban B (salah)

Keyakinannya turun, **tapi jawaban yang dipilih tetap benar**. Makanya nilai ujiannya (metrik teks) tetap bagus meskipun rasa yakinnya (loss) berkurang.

---

## ✅ Kesimpulan untuk Training Anda

| Aspek | Status | Keterangan |
|:---|:---:|:---|
| Train Loss | ✅ | Turun konsisten, model belajar |
| Eval Loss | 🔄 | Naik tipis di step 1000, overfitting ringan |
| Perplexity | ✅ | Stabil di ~17, model cukup yakin |
| ROUGE (semua) | ✅ | Naik konsisten, kelengkapan jawaban membaik |
| BLEU | ✅ | Naik, ketepatan membaik |
| METEOR | ✅ | Naik signifikan, kesesuaian makna membaik |
| BERTScore | ✅ | Naik, makna semantik sangat selaras |
| Exact Match | ✅ | Naik, model makin patuh format |

**Verdict:** Training Anda berjalan **sangat baik**. Metrik teks yang terus membaik membuktikan model masih berkembang meskipun loss sedikit naik. Lanjutkan hingga selesai! 🚀