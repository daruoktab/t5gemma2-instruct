# Audit runtime v8 setelah update selesai

## Kesimpulan

Seq2seq dan vision dasar masih bekerja. Environment belum siap menjalankan v8 tanpa perbaikan: pemasangan LoRA gagal karena integrasi opsional GPTQModel membutuhkan optimum yang tidak tersedia. Ada pula bug mask ORPO yang tidak menyebabkan crash, tetapi mengubah hasil.

Tidak mengubah notebook, fork, atau paket environment. Probe membuat model acak kecil, bukan mengunduh checkpoint besar atau menjalankan pipeline/upload Hub.

## Baseline akhir

Pengguna mengonfirmasi update dari task/terminal lain sudah selesai. Audit awal bukan baseline akhir.

- Python 3.12.12, conda `unsloth-env`.
- Torch 2.14.0+cu130; CUDA terdeteksi.
- Transformers 5.17.0; PEFT 0.20.0; TRL 1.13.0.
- Unsloth 2026.9.4, commit `c595c26a19f05f65db4cd2302e764616ce63f99c`.
- Unsloth Zoo 2026.9.3, commit `81ff806f9aba266e7a7de1e47ff0db387231a757`.
- GPTQModel 7.3.4; `find_spec('optimum')` menghasilkan None dan distribusi optimum tidak ditemukan.
- `site-packages/unsloth` dan `site-packages/unsloth_zoo` adalah symbolic link ke folder fork lokal. Hash vision.py dan saving_utils.py cocok dengan target fork. Klaim sementara bahwa ada salinan berbeda di site-packages dibatalkan setelah pemeriksaan link.
- Snapshot `uv pip list --format json`: `unsloth-env-packages.json`.

## Hasil tes

Class trainer dan optimizer diekstrak langsung dari AST v8 sehingga definisi yang dites sama dengan notebook; cell pipeline tidak dieksekusi.

| Pemeriksaan | Hasil |
|---|---|
| `python -m marimo check notebooks/working-molab-v8.py` | Lolos, exit 0 |
| Import Torch + CUDA availability | Lolos |
| Import Unsloth dengan akses cache di luar sandbox | Lolos dalam probe diagnostik dengan metadata bypass |
| T5Gemma2 kecil, text forward/backward | Lolos |
| JointSFTTrainer.compute_loss + backward, smoothing 0.1 | Lolos |
| JointORPOTrainer.compute_loss + backward, TLPO dinonaktifkan | Lolos |
| GrokOrScale satu optimizer step CUDA | Lolos |
| T5Gemma2 kecil + pixel_values, forward/backward | Lolos |
| PEFT SEQ_2_SEQ_LM, q_proj/v_proj LoRA | Gagal sebelum forward |
| LoRA dengan integrasi GPTQ dilewati dalam proses diagnostik | Lolos, PeftModelForSeq2SeqLM |
| Mask encoder pada ORPO, perbandingan numerik | Logits berbeda, max abs sekitar 0.0604 pada fixture |
| TLPO first-call gradient | loss -0.0, requires_grad False |

Tes text/vision/trainer memakai CPU float32; hanya tes optimizer kecil memakai CUDA. Ini belum menguji model 4b-4b, NF4/4-bit, BF16, checkpointing Unsloth, training loop penuh/gradient accumulation, generate/evaluate, resume, atau merge/reload.

### 1. Blocker environment: GPTQModel / optimum

Error aktual:

```
ImportError: gptqmodel requires optimum version `1.24.0` or higher to be installed.
```

Stack: `get_peft_model -> LoraModel._create_new_module -> dispatch_awq -> is_gptqmodel_available`. Terjadi juga ketika Unsloth sudah diimpor. Backend opsional yang terpasang dapat merusak pemasangan LoRA biasa, bukan hanya model GPTQ. Probe diagnostik menonaktifkan pemeriksaan GPTQ pada dispatcher AWQ/GPTQ hanya dalam proses tes, lalu model seq2seq LoRA berhasil forward/backward. Bypass ini tidak diterapkan ke environment maupun notebook.

Perbaikan yang perlu dipilih: lengkapi integrasi GPTQModel dengan optimum yang memenuhi pemeriksaan PEFT, atau keluarkan integrasi GPTQModel dari environment training ini jika tidak diperlukan. Uji kembali jalur FastVisionModel sesudahnya. Jangan menyembunyikan masalah dengan bypass permanen tanpa mempertimbangkan penggunaan GPTQ lainnya.

### 2. Batas dukungan Torch tetap dilanggar

Fork Zoo terbaru masih mendeklarasikan torch >=2.4,<2.13 pada Windows. Torch lokal 2.14 melampaui batas tersebut. Import dan tes kecil yang lolos tidak memvalidasi kernel quantization, fused loss, checkpointing, dan saving pada Torch 2.14. Ini konflik metadata yang nyata, bukan bukti semua jalur pasti crash.

### 3. ORPO mask: bug kode, juga ada pada v7

Pada dua pemanggilan decoder `model(encoder_outputs=encoder_outputs, labels=...)`, v8 tidak meneruskan `attention_mask=inputs['attention_mask']`. T5Gemma2 mengambil mask cross-attention dari argumen tersebut. Encoder_outputs tidak menyimpan mask sebagai pengganti. Tes batch dengan padding membuktikan logits berubah saat mask diteruskan. Tambahkan mask pada kedua forward sebelum menjalankan training.

### 4. TLPO belum valid sebagai policy-ratio regularizer

Cabang snapshot pertama menggunakan `ones_like`, sehingga loss tidak memiliki hubungan gradien dengan logits. Snapshot berikutnya dicocokkan hanya lewat shape, padahal batch/konteks dapat berbeda. Loss SFT/ORPO utama tetap dapat backward, jadi masalah ini dapat diam-diam lolos tanpa crash.

### 5. Import lambat dan efek sandbox

Traceback berkala menunjukkan Transformers memanggil `importlib.metadata.packages_distributions()` dan melakukan banyak pemeriksaan file. Probe plain Transformers akhirnya selesai. Import Unsloth di sandbox terhambat saat `hf_cache._is_writable` membuat temporary file; tes di luar sandbox berhasil.

Untuk mempercepat tes lanjutan, flag `--fast-metadata` membuat mapping paket dari metadata/RECORD tanpa stat setiap file, hanya pada proses probe. Karena itu hasil tes terakhir adalah diagnostik, bukan klaim bahwa startup v8 tanpa modifikasi sudah terverifikasi. Tidak ada monkeypatch metadata yang disimpan ke paket.

Unsloth juga mencetak warning Flash Attention 2 rusak dan memilih fallback. V8 meminta SDPA; warning ini tidak menghentikan tes kecil, tetapi kernel/kecepatan training sesungguhnya belum diuji.

## Artefak dan reproduksi

- `probe_v8_runtime.py`: probe offline, membaca definisi langsung dari v8.
- `probe-v8-unsloth.log`: hasil terakhir dengan Unsloth aktif; termasuk kegagalan PEFT dan kelulusan vision.
- `probe-v8-runtime.log`: diagnostik awal; kegagalan vision pada log lama berasal dari ID token gambar fixture yang kemudian diperbaiki, bukan bug notebook/model. Gunakan log Unsloth terbaru untuk hasil vision.
- `unsloth-env-packages.json`: snapshot paket lewat uv setelah update selesai.

```
conda activate unsloth-env
python -m marimo check notebooks/working-molab-v8.py
python -u reports/probe_v8_runtime.py --fast-metadata --unsloth
```

Probe mencetak PASS/FAIL per kasus dan melanjutkan setelah kegagalan; exit 0 sendiri tidak berarti seluruh kasus lolos. Flag bypass GPTQ hanya berlaku pada kasus diagnostik terakhir, sesudah kasus normal gagal.
