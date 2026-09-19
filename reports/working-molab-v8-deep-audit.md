# Audit Teknis `working-molab-v8.py`

## Penilaian eksekutif

`working-molab-v8.py` sudah layak dipakai untuk menyelesaikan **Joint SFT yang sedang berjalan**, tetapi belum layak dijalankan tanpa pengawasan sebagai pipeline penuh sampai ORPO dan final merge. Fondasi Seq2Seq masih ada: model diperlakukan sebagai encoder-decoder, target diberikan melalui `labels`, collator membangun encoder input dan decoder target secara terpisah, dan `Seq2SeqTrainer` digunakan secara langsung. Fakta bahwa run Molab berhasil membangun dataset 55.522 baris dan masuk ke training 1.736 update juga merupakan bukti runtime yang lebih kuat daripada sekadar lolos impor.

Keputusan operasional yang disarankan adalah:

1. Biarkan SFT saat ini selesai selama loss tetap finite dan tidak menunjukkan divergensi.
2. Jangan aktifkan `RUN_ORPO=True` sebelum jalur pemuatan base `cangkok/` + adapter SFT diperbaiki.
3. Jangan menilai kualitas final hanya dari eval periodik saat ini. Jalankan evaluasi penuh dan terpisah setelah SFT.
4. Perlakukan DeVec, LATA, MTO, dan kombinasi OrScale–GrokFast–AdEMAMix sebagai eksperimen internal sampai ada ablation terhadap baseline yang lebih sederhana.

| Area | Penilaian | Status |
|---|---|---|
| Joint SFT saat ini | Layak diteruskan | Hijau bersyarat |
| Integritas Seq2Seq | Terjaga | Hijau |
| Reproduksibilitas environment | Baik, tetapi dataset belum dipin | Kuning |
| Dataset dan validation | Train benar; evaluasi belum mencakup split resmi | Kuning |
| Efisiensi training | Banyak overhead evaluasi dan cache clearing | Kuning |
| Dasar riset fitur v8 | Beberapa fitur hanya “paper-inspired” | Kuning-merah |
| Resume checkpoint | Cukup baik untuk SFT yang sama | Kuning |
| ORPO/TLPO | Belum siap dinyalakan | Merah |
| Final merge/deployment | Memiliki blocker pada base adapter | Merah |

## Lingkup dan bukti yang diperiksa

Audit mencakup notebook v8 sepanjang 5.518 baris, perbandingan terhadap v7, dataset publik yang dipakai, environment Conda lokal, fork Unsloth, dan dokumentasi primer untuk T5Gemma 2, Transformers Trainer, DeVec, LATA, MTO, OrScale, TLPO, dan ORPO.

Pemeriksaan statis berhasil:

- `python -m py_compile notebooks/working-molab-v8.py`
- `marimo check notebooks/working-molab-v8.py`

Environment lokal `unsloth-env` cocok dengan pin utama notebook: Transformers 5.17.0, Accelerate 1.15.0, Datasets 5.0.1, TRL 1.13.0, PEFT 0.20.0, bitsandbytes 0.50.2, Unsloth 2026.9.4, Unsloth Zoo 2026.9.3, Optimum 2.3.0, SacreBLEU 2.6.0, dan marimo 0.24.2. Kedua fork berada tepat pada commit yang dipin notebook:

- `unsloth`: `da2bc8e841977276655bb8530405200f8128f2b6`
- `unsloth-zoo`: `9a44e54f3fd8878cee0cc9d062513f084cca1bf3`

Environment lokal memakai Python 3.12 dan Torch `2.14.0+cu130`, sedangkan notebook Molab meminta Python 3.13 dan memasang Torch `2.14.0+cu132`. Perbedaan tersebut disengaja dan ditangani oleh bootstrap notebook. Wheel Flash Attention 3 memakai ABI3 Linux x86_64 dan hanya dipilih pada GPU capability 9.x atau lebih tinggi; jika probe gagal, backend turun ke SDPA. Mekanisme fallback ini merupakan perbaikan yang baik dibanding kegagalan impor `flash_attn_3.flash_attn_interface` sebelumnya.[^1]

## Dataset yang benar-benar dipakai

Revisi dataset publik yang diperiksa adalah `db22b607a359c7781777f3d640fc6026bea66384`, diperbarui 10 September 2026. Dataset Server mencatat split berikut:[^2]

| Config | Train resmi | Validation resmi |
|---|---:|---:|
| `chat_sft` | 36.239 row | 8.869 row |
| `indoqa_sft` | 3.200 row | 800 row |
| `sea_general_sft` | 81.661 row | 21.027 row |
| `vision_sft` | 1.600 percakapan | 400 percakapan |

Run v8 tidak memakai seluruh tabel tersebut. Dataset train aktual 55.522 baris tersusun sebagai berikut:

| Sumber | Baris training aktual |
|---|---:|
| Chat train resmi | 36.239 |
| IndoQA train resmi | 3.200 |
| SEA-Instruct hasil sampling | 10.000 |
| Vision assistant-turn setelah holdout | 6.083 |
| **Total** | **55.522** |

Vision train 1.600 percakapan di-unroll menjadi assistant-turn, lalu 5% percakapan—80 percakapan atau 321 turn-row pada run ini—ditahan untuk evaluasi. Dengan batch 4, gradient accumulation 16, dan dua epoch, Trainer menghitung `ceil(55.522 / 64) × 2 = 1.736` optimizer update.

Koreksi atas perhitungan awal: angka komponen v7 lama, 36.015 chat dan 3.309 IndoQA, tidak boleh dipakai sebagai rincian v8. Dataset mono-repo terbaru memiliki 36.239 dan 3.200 untuk split train. Total v8 55.522 yang dibaca dari kernel tetap benar.

### Validation 20% belum dipakai penuh

Notebook memuat seluruh train split untuk chat dan IndoQA, tetapi hanya mengambil sekitar 100 validation row dari masing-masing config untuk evaluasi periodik. Split validation SEA sebanyak 21.027 row tidak dipakai. Split validation vision resmi sebanyak 400 percakapan juga tidak dipakai; notebook justru membuat holdout baru dari 5% vision train.[^3]

Konsekuensinya:

- model vision berlatih pada kira-kira 76% dari seluruh percakapan vision (`80% × 95%`);
- 400 percakapan validation resmi tetap belum disentuh;
- loss periodik murah, tetapi estimasinya berisik dan belum mewakili validation resmi;
- SEA-Instruct tidak mempunyai sinyal validation dalam run;
- evaluasi lintas sumber belum bisa menunjukkan apakah tambahan SEA memperbaiki generalisasi atau menggeser kemampuan chat/QA.

Holdout berbasis percakapan merupakan keputusan yang benar untuk mencegah satu percakapan terpotong antara train dan eval. Masalahnya bukan holdout itu sendiri, melainkan keberadaannya di samping split 80:20 yang sudah tersedia.

### Campuran data sangat text-heavy

Komposisi aktual adalah sekitar 49.439 row teks dan 6.083 row vision, atau 89,0% teks dan 11,0% vision. Ini masuk akal jika tujuan utama adalah asisten Bahasa Indonesia dengan kemampuan visual tambahan. Namun shuffle biasa tidak menjamin komposisi setiap batch, dan row count bukan ukuran kontribusi gradient: target panjang memberi lebih banyak token loss daripada target pendek.

Untuk eksperimen yang dapat dijelaskan, log berikut perlu disimpan per run: jumlah row, jumlah target token per sumber, persentase batch yang mengandung gambar, dan dataset revision SHA.

## Integritas Seq2Seq dan multimodal

T5Gemma 2 memang merupakan model encoder-decoder multimodal dengan tied embeddings dan merged self/cross-attention pada decoder.[^4] Notebook menghormati sifat ini melalui beberapa keputusan yang benar:

- `Seq2SeqTrainer` dan `Seq2SeqTrainingArguments` dipakai;
- encoder menerima prompt dan gambar, decoder menerima label target;
- `shift_labels=False` pada label smoothing sesuai karena model encoder-decoder menyiapkan decoder inputs dari labels;
- gambar di-load secara lazy berdasarkan `dataset_idx` dan `image_indices`;
- vision tower dibekukan, sementara LoRA bahasa dan projector dilatih;
- attention Q/K/V tidak disentuh saat steering untuk mengurangi risiko merusak merged attention.

Ada dua kelemahan pada collator.[^5]

Pertama, `MAX_SOURCE_LENGTH=16384` disimpan tetapi tidak pernah diterapkan. Pemanggilan processor tidak memakai `truncation=True` atau `max_length=self.max_src`. Satu contoh yang terlalu panjang dapat menghasilkan lonjakan VRAM atau OOM, terutama ketika bercampur dengan gambar.

Kedua, exception saat mengambil gambar ditelan dengan `except Exception: pass`. Ini menghilangkan bukti apakah suatu contoh vision benar-benar membawa pixel values. Kegagalan perlu dihitung dan dibuat fail-fast setelah ambang kecil; minimal log `dataset_idx`, exception type, dan jumlah gambar yang diharapkan.

## Training objective dan optimizer

### SFT dasar cukup koheren

SFT menggunakan QLoRA 4-bit, rsLoRA rank 64, alpha 16, dropout 0,2, effective batch 64, dua epoch, cosine schedule, warmup 100 step, label smoothing 0,1, dan NEFTune alpha 5. Transformers mendokumentasikan NEFTune pada rentang 5–15, sehingga alpha 5 bukan nilai yang janggal.[^6]

Selective label smoothing mengecualikan token yang dimask dan tetap menghitung NLL pada label aktif. Implementasinya masuk akal untuk vocabulary besar, walaupun label-smoothed eval loss kemudian tidak boleh diberi nama “perplexity” tanpa kualifikasi: `exp(smoothed_loss)` bukan perplexity NLL standar.

### Scheduler berbeda dua step dari Trainer

Scheduler custom memakai floor division:

```python
len(dataset) // (batch_size * gradient_accumulation)
```

Trainer memakai jumlah batch/update yang dibulatkan ke atas. Untuk data saat ini, scheduler dibuat untuk 1.734 step sementara Trainer menjalankan 1.736 step.[^7] Dampak run sekarang kecil—dua update terakhir berada setelah ujung cosine schedule—tetapi rumus harus diganti dengan `math.ceil`, dan pada multi-GPU perlu memperhitungkan world size serta semantics dataloader Trainer.

### OrScale cukup dekat dengan paper, tetapi eksperimennya melebar

Bagian OrScale mengikuti elemen inti paper: momentum, polar factor Newton–Schulz, Moonlight shape factor, coupled weight decay, one-time calibration, dan clipped trust ratio. Paper mendefinisikan default LM `s_l = 0.2 sqrt(max(m,n))`, calibration aktif, serta batas ratio 0,1–5,0; notebook mengikuti nilai tersebut.[^8]

Namun optimizer aktual bukan OrScale-LM yang dievaluasi paper. Notebook menambahkan GrokFast, memakai AdEMAMix untuk parameter non-2D, menerapkan clipping per parameter, menggunakannya pada matriks LoRA dan projector, serta melatih model terkuantisasi. Paper melaporkan dense pretraining hingga 16B-A3B, bukan QLoRA instruction tuning.[^9] Hasilnya sah sebagai optimizer eksperimental, tetapi peningkatan tidak dapat diasumsikan tanpa baseline.

Implementasi juga mengalokasikan `m`, `v`, dan `n` untuk matriks 2D walaupun cabang OrScale tidak memakainya, lalu mengalokasikan momentum buffer tambahan. Ini memboroskan optimizer state. State seharusnya dibuat per cabang: `grok_slow_grad + momentum_buf + c_denom` untuk OrScale, dan `grok_slow_grad + m + v + n` untuk AdEMAMix.

Trainer melakukan global gradient clipping dengan `max_grad_norm=5`, sedangkan optimizer kembali melakukan clipping per parameter pada 1. Dua mekanisme ini mengubah geometri update dan membuat hyperparameter lebih sulit ditafsirkan.

### Terlalu banyak intervensi sekaligus

Run menggabungkan steering full-weight, vision grafting, MTO prefix, QLoRA, rsLoRA, label smoothing, NEFTune, logit masking, GrokFast, OrScale, AdEMAMix, coupled weight decay, dan split learning rate. Model mungkin tetap membaik, tetapi tanpa ablation tidak mungkin diketahui fitur mana yang membantu atau merusak.

Minimum baseline untuk v8 sebaiknya:

1. base `cangkok/` + rsLoRA + AdamW;
2. tambah MTO;
3. ganti AdamW dengan OrScale tanpa GrokFast;
4. tambah GrokFast;
5. bandingkan dengan dan tanpa steering.

## Klaim riset: mana yang faithful dan mana yang inspired

### DeVec

Paper DeVec membangun shared subspace dari **beberapa task vector**, menggunakan proyeksi column space, produk projection matrices, eigendecomposition, lalu memisahkan shared dan unique components.[^10] Notebook hanya mempunyai satu delta `Gemma3-IT − Gemma3-Base` dan melakukan truncated SVD berdasarkan 85% energi singular.

Kode sendiri sudah mengakui perbedaan ini. Nama yang tepat adalah **energy-truncated task-vector SVD inspired by DeVec**, bukan implementasi DeVec. `tau=0.85` pada paper adalah threshold eigenvalue shared-subspace; di notebook ia menjadi cumulative singular-energy threshold. Kedua angka sama tetapi maknanya berbeda.

### LATA

LATA asli membandingkan layer vector “complex” terhadap instruction vector, merangking layer berdasarkan cosine similarity, lalu memakai linear/logarithmic rank weighting atau threshold dropping.[^11] Notebook menghitung `abs(cos(delta, base_weight))` per tensor dan mengubahnya menjadi skala kontinu `clamp(1.5 - cos, 0.5, 1.5)`.

Ini juga heuristic yang terinspirasi LATA, bukan LATA. Label UI dan model card perlu mengatakan demikian agar provenance ilmiahnya akurat.

### MTO

Paper MTO membahas pencocokan jenis tugas dengan pretraining/adaptation objectives dan template fine-tuning. Mask-filling ditujukan terutama untuk knowledge completion dengan target pendek, sedangkan QA dibahas sebagai kategori tersendiri; paper juga menekankan adaptation stage dan template placement.[^12]

Notebook menyuntik token `<unused1>`–`<unused6>` pada target, memasukkan `<extra_id_0>` ke encoder untuk kategori tertentu, tetapi target tidak memakai pasangan sentinel denoising. Selain itu, `Reasoning`, coding, math, dan QA semuanya dipetakan menjadi `qa/mask_filling`. Target coding dan reasoning panjang tidak cocok secara alami dengan asumsi mask-filling target pendek.

Mekanisme ini sebaiknya dinamai **task-prefix routing with MTO-inspired templates**. Sebelum run selanjutnya, audit per-task diperlukan untuk memastikan sentinel meningkatkan kualitas dan tidak sekadar menjadi token asing yang dipelajari model.

### TLPO

TLPO asli mengidentifikasi posisi rawan bahasa, mengeksplorasi kandidat token, dan melakukan localized token-level policy update.[^13] Implementasi notebook memakai argmax pada daftar token suppressed, top-k heuristic, dan “old policy” berupa log-prob tensor dari batch sebelumnya.

Snapshot batch sebelumnya bukan policy lama pada state/token yang sama. Jumlah posisi bingung juga berubah antar-batch, sehingga shape sering tidak cocok dan ratio kembali menjadi satu. Dengan demikian kode ini belum mengimplementasikan importance ratio PPO/TLPO yang bermakna. Komentar notebook yang menyebutnya proxy heuristic sudah benar; fitur harus tetap nonaktif sampai dirancang ulang.

## Evaluasi: benar secara API, mahal dan kurang representatif

Transformers mendukung `eval_dataset` berupa dictionary dan memberi prefix berbeda untuk tiap dataset. Penggunaan `{"multimodal": ..., "text_only": ...}` sudah tepat.[^14]

Masalah utama adalah evaluasi generatif ganda:

- `predict_with_generate=True` membuat `Seq2SeqTrainer` memanggil generation untuk seluruh 321 multimodal + 200 text eval rows;
- `generation_max_length=2048` memberi batas sangat tinggi;
- dua callback tambahan masing-masing menghasilkan sampai 20 contoh dengan `max_new_tokens=1024` setiap 100 step.[^15]

Transformers menjelaskan bahwa `predict_with_generate` memang ditujukan untuk menghitung ROUGE/BLEU dari generation.[^16] Di sini generation Trainer dan callback kualitatif mengulang pekerjaan. Rekomendasi:

- periodic eval: `predict_with_generate=False`, hitung loss saja setiap 100 step;
- qualitative generation: 8–12 contoh tetap per modality setiap 200–400 step, `max_new_tokens` 256–512;
- full generative metrics: satu kali pada checkpoint kandidat dan seluruh validation resmi;
- simpan metrik per sumber dan kategori tugas;
- tambah language-consistency rate, instruction-following checks, dan evaluasi vision grounding.

`torch_empty_cache_steps=10` juga memiliki biaya. Dokumentasi Transformers menyebut sekitar 10% perlambatan sebagai tradeoff untuk menurunkan peak VRAM.[^6] Nilai 10 sebaiknya hanya dipertahankan jika telemetry menunjukkan fragmentasi/OOM; bila training stabil, 50–100 atau `None` lebih efisien.

Evaluator `CultureTalk-ID` didefinisikan tetapi tidak pernah dipanggil. Selain itu, ia memberi `labels=input_ids` pada model encoder-decoder untuk setiap opsi. Itu mengukur reconstruction loss atas gabungan konteks–pertanyaan–opsi, bukan conditional answer score yang bersih. Evaluator perlu dibangun ulang sebelum hasilnya dipublikasikan.

## Checkpoint, resume, dan artefak Hub

Notebook mempunyai beberapa praktik bagus:

- mendeteksi checkpoint lengkap melalui adapter, trainer state, dan optimizer state;
- menghapus upload parsial;
- menyimpan dua checkpoint terbaru;
- fail-hard jika upload gagal;
- menulis marker completion dan provenance lokal;
- memakai fork Unsloth dengan commit tetap.

Ada empat risiko operasional.

### Blocker: adapter menunjuk ke base yang salah

Fungsi sanitasi menulis ulang `base_model_name_or_path` menjadi `google/t5gemma-2-4b-4b`.[^17] Padahal adapter SFT dilatih di atas `cangkok/`, yang memuat perubahan full-weight dari steering dan vision grafting. Adapter LoRA tidak menyimpan seluruh perubahan decoder hasil steering; `modules_to_save` hanya mencakup projector.

Resume SFT saat ini aman karena notebook lebih dulu membangun model dari `cangkok/`, lalu Trainer memuat bobot checkpoint. Jalur setelah SFT berbeda: notebook memanggil `FastVisionModel.from_pretrained()` langsung pada folder adapter untuk ORPO dan merge.[^18] Jika loader mengikuti metadata yang sudah disanitasi, ia membangun base Google asli dan kehilangan perubahan `steered/`/`cangkok/`.

Perbaikan wajib sebelum ORPO:

1. load base secara eksplisit dari repo v8 subfolder `cangkok/`;
2. attach adapter SFT menggunakan PEFT/Unsloth API yang menerima base object + adapter path;
3. verifikasi hash beberapa tensor steering dan vision tower sebelum ORPO;
4. merge smoke test dan bandingkan output dengan model in-memory sebelum menyatakan final.

### Logit mask tidak menjadi bagian bobot

Logit masking dipasang sebagai Python forward hook. Hook tidak tersimpan di safetensors. Setelah model di-merge dan dimuat pada proses baru, suppression hilang kecuali consumer memasangnya kembali. Lebih aman menyimpan daftar tersebut dalam generation config sebagai `bad_words_ids`, menyediakan loader resmi, dan menguji bahwa token terlarang tidak muncul setelah reload.

### Dataset dan config belum menjadi provenance checkpoint

`load_dataset()` tidak menerima `revision`; resume dapat mengambil dataset yang sudah berubah. Marker upload hanya menyimpan path, jumlah file, byte lokal, dan timestamp. Ia belum menyimpan:

- dataset commit SHA dan fingerprints;
- notebook/git commit;
- commit kedua fork;
- seluruh hyperparameter;
- CUDA/GPU dan attention backend;
- daftar jumlah row/token per sumber;
- seed dan checksum split conversation IDs.

Tanpa manifest tersebut, `sft_done=True` dapat membuat notebook melewati training walaupun data atau kode sudah berubah.

### Multi-process dan network failure

Callback generation, file append, upload Hub, dan remote pruning tidak dilindungi `state.is_world_process_zero`. Pada multi-GPU, setiap rank dapat menulis atau mengunggah artefak yang sama. Tidak ada lock antar-session, sehingga dua notebook yang menunjuk repo sama juga dapat menghapus checkpoint satu sama lain.

Upload gagal langsung menghentikan training tanpa retry/backoff. Prinsip fail-hard baik untuk integritas, tetapi network transient sebaiknya dicoba ulang beberapa kali; checkpoint lokal harus dipertahankan dan error baru dilempar setelah retry gagal.

## ORPO belum siap walaupun dataset config sekarang terlihat

Dataset metadata publik sekarang mencantumkan `chat_orpo` dan `vision_orpo`, tetapi Dataset Server belum menghasilkan statistik ukuran untuk keduanya saat audit. Menjaga `RUN_ORPO=False` tetap merupakan keputusan yang benar.

Masalah yang harus diselesaikan sebelum ORPO:

1. jalur base + SFT adapter yang dijelaskan di atas;
2. TLPO dinonaktifkan atau diimplementasikan ulang;
3. evaluation preference memakai pasangan chosen/rejected nyata;
4. log reward chosen/rejected, reward accuracy, margin, NLL, dan odds-ratio loss.

Saat ini ORPO eval dibentuk dengan `chosen_text == rejected_text`. Karena kedua respons identik, bagian preference tidak mengukur ranking apa pun. Dokumentasi TRL justru menekankan reward chosen/rejected, accuracy, margin, log odds, dan NLL sebagai metrik ORPO.[^19]

ORPO asli adalah objective monolitik berupa NLL chosen ditambah penalti odds-ratio terhadap rejected.[^20] Custom trainer notebook mengikuti bentuk umum ini, tetapi perlu dibandingkan numerik dengan implementasi referensi pada batch kecil sebelum digunakan untuk training mahal.

## Prioritas perubahan

### P0 — sebelum ORPO atau merge

1. Load `cangkok/` secara eksplisit lalu attach adapter; jangan bergantung pada `base_model_name_or_path` yang disanitasi.
2. Buat preference validation asli; hapus eval dengan chosen dan rejected identik.
3. Matikan TLPO heuristic sampai ada old-policy/reference semantics yang valid.
4. Uji reload adapter dan hasil merge terhadap model in-memory pada prompt teks dan vision yang sama.
5. Persist logit suppression untuk deployment.

### P1 — sebelum run SFT berikutnya

1. Pakai seluruh vision train 1.600 percakapan dan validation resmi 400 percakapan; hapus holdout 5% tambahan.
2. Gunakan periodic loss-only eval, lalu full generative eval terpisah.
3. Ubah scheduler ke ceiling-aware step calculation.
4. Terapkan source truncation dan log/fail-fast untuk image loading.
5. Pin dataset revision SHA dan tulis run manifest.
6. Buat callback rank-zero dan tambah retry upload.
7. Kurangi optimizer state yang tidak dipakai.

### P2 — kualitas eksperimen

1. Ganti label DeVec/LATA/MTO menjadi “inspired” sesuai implementasi.
2. Jalankan ablation AdamW vs OrScale, dengan/tanpa GrokFast, dengan/tanpa steering, dan dengan/tanpa task prefix.
3. Ukur token balance per sumber dan pertimbangkan modality/source sampler.
4. Bangun test suite kecil untuk collator text-only, single-image, multi-image, truncation, missing image, resume, dan reload merge.
5. Gunakan loader deployment multimodal resmi. Model card T5Gemma 2 mendemonstrasikan `AutoModelForMultimodalLM`; snippet notebook saat ini memakai `AutoModelForSeq2SeqLM`.[^4]

## Putusan akhir

V8 merupakan peningkatan nyata dari v7 dalam pengelolaan environment, pemisahan modul, gating phase, fallback attention, data mixing, dan checkpoint hygiene. Seq2Seq tidak hilang. Current SFT bukan run yang sia-sia dan tidak perlu dihentikan hanya karena temuan audit ini.

Kualitas pipeline saat ini paling tepat disebut **research prototype yang mampu menjalankan Joint SFT**, bukan pipeline SOTA yang sudah tervalidasi end-to-end. Risiko terbesar bukan pada step SFT yang sedang berjalan, melainkan pada perpindahan dari final adapter SFT ke ORPO/merge. Setelah blocker base adapter, validation, dan evaluasi preference diperbaiki, struktur notebook sudah cukup kuat untuk menjadi pipeline v8 yang dapat diulang dan dinilai secara ilmiah.

## Sources

[^1]: Notebook lokal, [`working-molab-v8.py`](../notebooks/working-molab-v8.py), bootstrap environment dan FA3 probe, baris 432–651. Fork lokal diperiksa pada commit yang dipin notebook.
[^2]: Hugging Face Dataset Server, [size metadata for `daruokta/t5gemma2-indonesia-instruct-v1`](https://datasets-server.huggingface.co/size?dataset=daruokta%2Ft5gemma2-indonesia-instruct-v1), diakses 14 September 2026. Dataset revision diperoleh dari [Hugging Face dataset API](https://huggingface.co/api/datasets/daruokta/t5gemma2-indonesia-instruct-v1).
[^3]: Notebook lokal, [`working-molab-v8.py`](../notebooks/working-molab-v8.py), data loading dan split, baris 2841–2847 dan 3030–3214.
[^4]: Google DeepMind, [T5Gemma 2 4B-4B model card](https://huggingface.co/google/t5gemma-2-4b-4b); Hugging Face, [T5Gemma 2 documentation](https://huggingface.co/docs/transformers/v5.17.0/model_doc/t5gemma2).
[^5]: Notebook lokal, [`working-molab-v8.py`](../notebooks/working-molab-v8.py), `Seq2SeqVisionCollator`, baris 3227–3280.
[^6]: Hugging Face, [Transformers Trainer documentation](https://huggingface.co/docs/transformers/main_classes/trainer), bagian NEFTune dan `torch_empty_cache_steps`.
[^7]: Notebook lokal, [`working-molab-v8.py`](../notebooks/working-molab-v8.py), scheduler construction, baris 4432–4446.
[^8]: Yuxuan Lou dan Yang You, “[OrScale: Orthogonalized Optimization with Layer-Wise Trust-Ratio Scaling](https://arxiv.org/abs/2605.07815),” arXiv:2605.07815v2, 30 Agustus 2026, Algorithm 1.
[^9]: Lou dan You, [OrScale paper](https://arxiv.org/html/2605.07815v2), eksperimen language-model pretraining dan 16B-A3B configuration.
[^10]: Hamed Damirchi et al., “[Decomposing Task Vectors for Refined Model Editing](https://arxiv.org/abs/2512.22511),” arXiv:2512.22511, 27 Desember 2025, §3.1–3.2.
[^11]: Yan-Lun Chen et al., “[Layer-Aware Task Arithmetic: Disentangling Task-Specific and Instruction-Following Knowledge](https://arxiv.org/abs/2502.20186),” arXiv:2502.20186, 27 Februari 2025, §3.
[^12]: Ahmad Pouramini dan Hesham Faili, “[Matching Tasks to Objectives: Fine-Tuning and Prompt-Tuning Strategies for Encoder-Decoder Pre-trained Language Models](https://arxiv.org/abs/2606.24841),” arXiv:2606.24841, 23 Juni 2026, §4 dan §6.
[^13]: Jinho Choo et al., “[TLPO: Token-Level Policy Optimization for Mitigating Language Confusion in Large Language Models](https://arxiv.org/abs/2604.26553),” arXiv:2604.26553, 29 April 2026.
[^14]: Hugging Face, [Trainer `evaluate` documentation](https://huggingface.co/docs/transformers/main_classes/trainer), dukungan dictionary of datasets dan metric prefixes.
[^15]: Notebook lokal, [`working-molab-v8.py`](../notebooks/working-molab-v8.py), generation callback dan SFT arguments, baris 4127–4216 dan 4491–4524.
[^16]: Hugging Face, [Seq2SeqTrainingArguments](https://huggingface.co/docs/transformers/main_classes/trainer), `predict_with_generate` dan `generation_max_length`.
[^17]: Notebook lokal, [`working-molab-v8.py`](../notebooks/working-molab-v8.py), adapter metadata sanitization, baris 2016–2032.
[^18]: Notebook lokal, [`working-molab-v8.py`](../notebooks/working-molab-v8.py), direct adapter loading untuk ORPO dan merge, baris 2892–2930 dan 5232–5262.
[^19]: Hugging Face TRL, [ORPO Trainer documentation](https://huggingface.co/docs/trl/orpo_trainer), logged preference metrics dan expected dataset type.
[^20]: Jiwoo Hong, Noah Lee, dan James Thorne, “[ORPO: Monolithic Preference Optimization without Reference Model](https://aclanthology.org/2024.emnlp-main.626/),” EMNLP 2024.
