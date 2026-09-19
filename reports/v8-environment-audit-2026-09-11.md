# Audit v8 / unsloth-env — 2026-09-11

> Audit awal di bawah merekam keadaan sebelum update fork. Lihat `v8-runtime-audit-2026-09-11.md` untuk hasil runtime dan baseline terbaru setelah pengguna mengonfirmasi update selesai: Unsloth 2026.9.4 / Zoo 2026.9.3. Path site-packages kedua paket adalah symlink ke fork lokal, bukan salinan terpisah.

Audit read-only terhadap notebook v8, v7, metadata environment conda, dan source fork editable. Tidak menjalankan notebook karena memuat training, download model, dan upload Hub.

## Versi

| Paket | v7 | v8 | unsloth-env |
|---|---|---|---|
| torch | 2.12.1 | 2.12.1 | 2.14.0+cu130 |
| torchvision | 0.27.1 | 0.27.1 | 0.29.0+cu130 |
| transformers | 5.14.1 | 5.15.0 | 5.17.0 |
| trl | 1.9.2 | 1.10.0 | 1.13.0 |
| peft | 0.19.1 | 0.20.0 | 0.20.0 |
| accelerate | 1.14.0 | 1.14.0 | 1.15.0 |
| datasets | 5.0.0 | 5.0.1 | 5.0.1 |
| bitsandbytes | 0.50.0 | 0.50.1 | 0.50.2 |
| huggingface-hub | 1.25.1 | 1.27.0 | 1.30.0 |
| numpy | 2.5.1 | 2.5.1 | 2.5.3 |
| marimo | 0.23.15 | 0.23.16 | 0.24.0 |
| pytorch-optimizer | unpinned | 3.10.1 | 3.10.1 |
| sacrebleu | unpinned | unpinned | tidak terpasang |

Pin v8 evaluate 0.4.6, absl-py 2.4.0, nltk 3.10.3, pillow 12.3.0, pymupdf 1.28.2, bert-score 0.3.13, hf-transfer 0.1.9, rouge-score 0.1.2 sesuai metadata lokal.

## Provenance dan patch seq2seq

- Unsloth 2026.9.2 editable: `D:/Codings/unsloth-fork`, origin `https://github.com/daruoktab/unsloth.git`, HEAD `d8eff9b6ccde2e75ab62a74ab6d7630f8e1f6429`.
- Zoo 2026.9.1 editable: `D:/Codings/unsloth-zoo-fork`, origin `https://github.com/daruoktab/unsloth-zoo.git`, HEAD `ca0b2daea4c827a6d37ce4717bb161d58a1c05a2`.
- HEAD cocok dengan cached `origin/main` di kedua checkout. Ini bukan verifikasi remote live.
- Tidak ada tracked modifications; ada file patch/report untracked.
- Patch loader memilih AutoModelForSeq2SeqLM masih ada (`loader.py:1976`).
- Patch AutoProcessor untuk seq2seq multimodal masih ada (`vision.py:1199`).
- TaskType.SEQ_2_SEQ_LM dan PeftModelForSeq2SeqLM masih ditangani (`vision.py:2193`, juga `llama.py`).
- Bypass batching patch causal untuk encoder-decoder masih ada (`_utils.py:3174`).
- Remapping encoder.text_model versus encoder saat merge masih ada (`saving_utils.py:4312`).

Jadi patch tidak hilang dari source yang ditunjuk instalasi editable. Ini belum membuktikan load 4-bit, forward/backward, resume, dan merge berhasil pada seluruh kombinasi paket terbaru.

## Temuan prioritas

1. **Konflik Torch langsung:** `pip check` melaporkan Zoo mensyaratkan torch >=2.4,<2.13 pada Windows, tetapi terpasang 2.14.0. Jangan menjadikan environment ini baseline terverifikasi hanya dengan menyalin versinya ke v8. Tentukan stack Torch yang sesuai atau validasi/port fork terlebih dahulu.
2. **ORPO kehilangan source mask:** v8 sekitar baris 3411 meneruskan mask ke encoder tetapi tidak ke dua pemanggilan `model(encoder_outputs=..., labels=...)`. Source Transformers lokal `modeling_t5gemma2.py:1147` mengambil `encoder_attention_mask` dari argumen `attention_mask`. Tambahkan `attention_mask=inputs['attention_mask']` pada kedua decoder forward sebelum training. Masalah serupa ada di v7 sekitar 2732; bukan regresi versi baru.
3. **Fork belum dipin SHA di notebook:** URL Git tanpa revision tidak mereproduksi checkout audit. Pin SHA kedua fork saat membuat baseline Molab yang tervalidasi. Local editable dan instalasi dari URL Git adalah dua provenance berbeda.
4. **Dependency environment tidak bersih:** pip check juga menemukan marimo/jedi (0.24.0 membutuhkan jedi <0.20; lokal 0.20.0), numba/numpy/llvmlite, banyak batas versi LLaMAFactory, dan paket lain. Konflik paket lain tidak otomatis membuktikan jalur training gagal, tetapi environment ini bukan lock yang konsisten.
5. **TLPO v8:** snapshot log-prob batch sebelumnya dipakai ulang hanya berdasarkan kesamaan shape. Shape sama tidak berarti konteks/token positions sama. Pada snapshot pertama atau shape berbeda, `ones_like` membuat cabang loss ini tidak bergantung pada logits. Jangan menganggap regularizer tersebut sebagai PPO policy ratio yang tervalidasi.

## Pemeriksaan

- Metadata paket dan direct_url.json dibaca tanpa mengimpor paket.
- Diff commit fork dan source terpasang diperiksa langsung.
- AST v8 berhasil diparse; seluruh keyword dua konstruksi Seq2SeqTrainingArguments ditemukan pada source training arguments lokal. Pemeriksaan ini tidak memvalidasi nilai maupun runtime.
- SFT dan ORPO memakai custom subclass Transformers Seq2SeqTrainer, bukan TRL ORPOTrainer. Upgrade TRL tidak dengan sendirinya menghapus jalur seq2seq notebook.
- `pip check` gagal karena konflik yang disebutkan di atas.
- Probe import Torch/Transformers/Unsloth dihentikan setelah beberapa menit tanpa hasil diagnostik utama; proses keluar dengan kode 1. Hanya warning Torch yang sempat muncul. Ini bukan bukti import berhasil ataupun diagnosis penyebab kegagalan; CUDA dan mapping runtime belum terverifikasi.
- Training model besar, akses remote terbaru, merge, dan upload tidak diuji.

Notebook dan paket environment belum diubah. Langkah berikutnya adalah memperbaiki forward mask, menetapkan stack yang memenuhi metadata fork, lalu smoke test kecil encoder-decoder/LoRA dan merge sebelum menjalankan pipeline lengkap.
