# [RESEARCH] Vision Training Analysis, SigLIP Cangkok & Debugging Strategy

**Last Updated:** 5 Agustus 2026  
**Status:** Cangkok selesai & tervalidasi  
**Target Models:** `google/t5gemma-2-4b-4b` | `google/gemma-3-4b-it` | `daruokta/t5gemma-2-4b-4b-instruct-chat-indo-v4-vision-cangkok`

---

## 1. Mekanisme Vision Encoder (SigLIP) & Soft Tokens

Vision Tower pada T5Gemma 2 menggunakan **SigLIP 400M** (27 layers, hidden 1152, 16 heads).

### Spesifikasi Image Pipeline:
- **Input Image Size:** 896 × 896 (Fixed).
- **Patch Size:** 14 × 14 → 64 × 64 = 4.096 patch.
- **Spatial Merge (4×4):** 16 patch digabung menjadi 1 token → **256 Soft Tokens per gambar**.
- **Location:** Vision tower berada di Encoder (`model.encoder.vision_tower`).
- **Projector:** `T5Gemma2MultiModalProjector` memproyeksikan fitur 1152-dim ke 2560-dim (sesuai text embedding T5Gemma 4B).
- **Placeholder Token:** `<image_soft_token>` (ID 256001).

```
Gambar (896×896)
   ↓ SigLIP Conv2d + Encoder
4096 Patches (1152-dim)
   ↓ Spatial Merge 4×4
256 Feature Vectors (1152-dim)
   ↓ MultiModal Projector (1152 → 2560)
256 Vectors (2560-dim)  →  Disuntikkan ke posisi ID 256001
```

---

## 2. Strategi "Cangkok" (Transplant) SigLIP & Projector

### Mengapa Cangkok Vision Dibutuhkan?
Gemma 3 4B IT memiliki bobot SigLIP dan MultiModal Projector yang sudah terlatih secara matang pada jutaan pasang gambar-teks instruksional. Kita melakukan *cangkok* (transplantasi bobot) dari Gemma 3 4B IT ke T5Gemma 2 4B Base untuk memperoleh kemampuan multimodal *zero-shot* yang kuat sebelum SFT.

### Langkah Cangkok Bobot:
1. Ambil bobot `vision_tower` SigLIP dan `multi_modal_projector` dari `google/gemma-3-4b-it`.
2. Lakukan penyesuaian RMSNorm / proyeksi jika terdapat perbedaan dimensi (1152 → 2560).
3. Simpan checkpoint hasil cangkokan ke Hugging Face Hub: `daruokta/t5gemma-2-4b-4b-instruct-chat-indo-v4-vision-cangkok`.

---

## 3. Perbaikan Quality Issues & Debugging Vision Model

### Isu Kunci & Solusi:
1. **EOS Ganda & Image Token Index:** Pastikan `image_token_index` dikonfigurasi tepat di `256001`. Mismatch index dapat menyebabkan vision token dianggap sebagai text token biasa.
2. **Cegah "Bocor" LoRA ke Vision Tower:** Secara bawaan, jangan tambahkan modul `vision_tower` ke dalam `target_modules` LoRA (kecuali sedang melakukan full multimodal fine-tuning), agar bobot SigLIP yang sudah paten tidak terkontaminasi.
3. **Resolusi Gambar & Aspect Ratio Distortion:** Gambar di-resize ke 896x896. Gambar dengan aspect ratio ekstrem wajib menggunakan padding netral alih-alih stretching paksa.
