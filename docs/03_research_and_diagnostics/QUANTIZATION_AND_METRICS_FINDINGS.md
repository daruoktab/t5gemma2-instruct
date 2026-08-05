# Quantization Findings & Training Metrics Breakdown

**Last Updated:** 5 Agustus 2026  
**Context:** Temuan kuantisasi (BitsAndBytes 4-bit / 8-bit / GGUF) dan panduan interpretasi metrik pelatihan T5Gemma 2.

---

## 1. Temuan Kuantisasi (Quantization Findings)

- **4-bit NF4 (BitsAndBytes):** Sangat efektif memotong konsumsi VRAM pada T5Gemma 2 4B dari ~15GB menjadi ~5.2GB. Namun pada Seq2Seq Merged Attention, pastikan `bnb_4bit_compute_dtype = torch.bfloat16` untuk mencegah pembulatan numerik ekstrem pada QK^T product.
- **8-bit Quantization:** Memberikan degradasi keakuratan yang hampir nol (<0.5%), tetapi hanya menghemat VRAM ~40%.
- **GGUF / Llama.cpp Exporters:** Konversi T5Gemma2 Merged Attention ke format GGUF membutuhkan penyesuaian arsitektur `t5gemma2` pada backend llama.cpp.

---

## 2. Penjelasan Metrik Pelatihan (Training Metrics)

1. **Train Loss (Cross Entropy):** Mengukur ketidakpastian prediksi token berikutnya pada decoder. Target nilai sehat: `1.0 - 2.5` tergantung kompleksitas data.
2. **Perplexity (PPL):** Formulasi $e^{\text{loss}}$. Menunjukkan berapa banyak kandidat token yang dibingungkan oleh model di setiap langkah.
3. **Grad Norm:** Ukuran norma Vektor Gradien. Norma yang normal berkisar `0.1 - 5.0`. Jika Grad Norm > 50+, gunakan `max_grad_norm = 1.0` (gradient clipping).
4. **Learning Rate Schedule:** Recomended warmup 5-10% dari total step dengan cosine decay schedule.
