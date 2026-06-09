# [MASTER] Research Motivation & Philosophical Background

**Last Updated:** 15 Mei 2026

---

## 1. Mengapa Encoder-Decoder untuk Chatbot?

Meskipun tren saat ini didominasi oleh arsitektur *decoder-only* (seperti GPT), arsitektur *encoder-decoder* (seperti T5Gemma-2) menawarkan keunggulan struktural yang unik untuk sistem percakapan:

- **Asymmetric Processing:** Sangat efisien untuk tugas di mana input panjang (konteks/dokumen) harus menghasilkan output pendek (respons/ringkasan).
- **Explicit Cross-Attention:** Encoder memproses seluruh input secara bidirectional sebelum decoder mulai bekerja. Ini mencegah masalah *attention degeneration* di mana model mulai "lupa" detail awal dari input yang panjang.
- **Task Flexibility:** Sangat unggul dalam tugas-tugas "as-you-go" seperti terjemahan, peringkasan, dan ekstraksi informasi yang sering muncul secara organik dalam sebuah percakapan.

---

## 2. Landasan Riset & Paper Pendukung (2025)

Berikut adalah beberapa paper kunci yang mendasari keputusan desain proyek ini:

1. **"Return of the Encoder: Maximizing Parameter Efficiency for SLMs" (Elfeki et al., Jan 2025)**
   - Membuktikan secara empiris bahwa arsitektur encoder-decoder lebih efisien dalam penggunaan parameter untuk tugas-tugas asimetris dibanding decoder-only.
2. **"RedLLM: Encoder-Decoder or Decoder-Only? Revisiting Encoder-Decoder LLM" (Zhang et al., Oct 2025)**
   - Menunjukkan bahwa setelah *instruction tuning*, model encoder-decoder dapat menyamai atau melampaui model decoder-only dengan efisiensi inferensi yang lebih tinggi.
3. **"Task Arithmetic" (Ilharco et al., 2023)**
   - Dasar teori untuk metode **Task Vector Transfer**, di mana kita bisa "menambahkan" kemampuan instruksi melalui operasi aritmatika pada bobot model.

---

## 3. Visi Proyek: T5Gemma-2 Instruct

Visi kita adalah menciptakan asisten AI yang:
- **Indonesia-first:** Mengutamakan keluwesan bahasa Indonesia sehari-hari.
- **Bilingual & Natural:** Mampu melakukan *code-switching* dengan mulus.
- **Multi-task Capable:** Bukan hanya "tukang ngobrol", tapi asisten yang secara alami mahir meringkas, menerjemahkan, dan mengekstrak data di tengah obrolan tanpa perlu instruksi kaku.

---
