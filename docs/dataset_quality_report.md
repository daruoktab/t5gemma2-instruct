# Laporan Analisis Kualitas Dataset (SFT & Validation)

## 1. Statistik Dasar Dataset
| Dataset | File Size (MB) | Total Rows | Malformed Rows | Duplicate Rows (Exact) | Duplicate Inputs (Different Targets) | Empty Input/Target |
|---|---|---|---|---|---|---|
| Chat Train | 77.45 MB | 27,990 | 0 | 0 | 5 | 0/0 |
| Chat Val | 1.24 MB | 464 | 0 | 0 | 0 | 0/0 |
| IndoQA Train | 2.88 MB | 3,309 | 0 | 0 | 0 | 0/1 |
| IndoQA Val | 0.96 MB | 1,104 | 0 | 0 | 0 | 0/0 |

## 2. Analisis Panjang Teks (Jumlah Kata)
| Dataset | Input (Min/Mean/Median/Max) | Target (Min/Mean/Median/Max) | Rasio Target-to-Input |
|---|---|---|---|
| Chat Train | 70 / 381.1 / 372 / 2157 | 2 / 33.9 / 31 / 346 | 0.09x |
| Chat Val | 74 / 370.2 / 372 / 812 | 4 / 30.4 / 29 / 87 | 0.08x |
| IndoQA Train | 74 / 112.6 / 108 / 182 | 0 / 5.0 / 3 / 54 | 0.04x |
| IndoQA Val | 70 / 112.2 / 107 / 183 | 1 / 5.0 / 3 / 36 | 0.04x |

## 3. Analisis Kebocoran Data (Data Leakage)
Data leakage terjadi jika prompt evaluasi (validation) juga terdapat pada training set. Ini membuat evaluasi kurang valid karena model hanya 'menghafal'.

### Chat Dataset Leakage:
- **Prompt Leakage:** 0 dari 464 prompt validasi ada di training set (0.00%)
- **Exact Row Leakage:** 0 baris ada secara identik di train dan val (0.00%)

### IndoQA Dataset Leakage:
- **Prompt Leakage:** 0 dari 1104 prompt validasi ada di training set (0.00%)
- **Exact Row Leakage:** 0 baris ada secara identik di train dan val (0.00%)

## 4. Distribusi Kata Tanya (Intent Analysis)
Menunjukkan variasi dan sebaran tipe instruksi/pertanyaan dalam Bahasa Indonesia:

### Tipe Pertanyaan di IndoQA Train:
**apa**: 10 (0.3%), **apakah**: 3 (0.1%), **di mana**: 3 (0.1%), **mengapa**: 2 (0.1%)

### Tipe Pertanyaan di IndoQA Val:
**mengapa**: 1 (0.1%), **apa**: 1 (0.1%)

## 5. Kesimpulan Kualitas Dataset & Temuan Kunci
- ℹ️ **Chat Train** memiliki **5 prompt yang sama tetapi dengan target jawaban berbeda**. Ini wajar untuk skenario percakapan terbuka, tetapi perlu dipantau jika berupa pertanyaan faktual.
- ℹ️ **Chat Train** memiliki prompt sangat panjang (maksimal **2157 kata**). Pastikan batas `MAX_SOURCE_LENGTH` tidak memotong informasi penting.
