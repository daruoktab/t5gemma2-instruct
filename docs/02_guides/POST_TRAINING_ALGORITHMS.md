# Post-Training Algorithms & Dataset Preparation Strategy

**Last Updated:** 5 Agustus 2026  
**Scope:** SFT, DPO, ORPO, GRPO, Task Arithmetic, dan Persiapan Dataset Pasca-SFT untuk T5-Gemma-2.

---

## 1. Taksonomi Algoritma Post-Training

Post-training bertujuan mengubah model bahasa dasar (*base model*) menjadi asisten instruksi yang aman, presisi, dan dapat dituntun.

```
Post-Training Pipeline
├── Supervised Fine-Tuning (SFT)        ← Format instruksi & multiturn
├── Preference Optimization
│   ├── Direct Preference Optimization (DPO)   ← Membutuhkan reference model
│   ├── Odds Ratio Preference Optimization (ORPO) ← Tanpa reference model (SFT + Pref)
│   └── Group Relative Policy Optimization (GRPO) ← Reinforcement Learning via Group Normalization
└── Model Merging / Task Arithmetic     ← Penggabungan vektor bobot (W_IT - W_Base)
```

---

## 2. Perbandingan Algoritma Preference Optimization

| Algoritma | Membutuhkan Reference Model? | Membutuhkan SFT terpisah? | Kekuatan Utama |
|---|---|---|---|
| **DPO** | Ya | Ya (idealnya setelah SFT) | Stabil, tidak membutuhkan Reward Model eksplisit |
| **ORPO** | **Tidak** | **Tidak** (menggabungkan NLL loss SFT & Odds Ratio) | Sangat hemat memori GPU, cocok untuk notebook / Marimo |
| **GRPO** | Tidak | Ya | Sangat baik untuk penalaran matematika / koding (DeepSeek R1 style) |

> [!TIP]
> **Rekomendasi untuk T5Gemma 2 Multimodal:**
> Gunakan **ORPO** (`Vanilla ORPO Vision`) karena tidak membutuhkan alokasi memori tambahan untuk *reference model* saat training vision tower SigLIP + T5Gemma Decoder.

---

## 3. Persiapan Dataset Pasca-SFT

Untuk mempertahankan performa generalisasi dan mencegah *catastrophic forgetting* setelah fase SFT awal:

1. **Replay Mix Ratio:** Campurkan 10–15% data instruksi umum (multilingual / Indo-English) ke dalam dataset spesifik domain.
2. **Quality Filtering:** Bersihkan token berulang, kalimat terpotong, dan respons yang terlalu pendek (< 5 token).
3. **Format Chat Standard:** Pastikan format prompt mengikuti templat percakapan `<start_of_turn>user\n...<end_of_turn>\n<start_of_turn>model\n...<end_of_turn>`.
