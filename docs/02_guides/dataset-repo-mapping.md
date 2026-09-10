# Katalog Sumber & Format Data → Mono-Repo `daruokta/t5gemma2-indonesia-instruct-v1`

> Dokumen ini merekam **sumber** dan **format** setiap data (lama & baru) sehingga
> struktur mono-repo bisa dibuat rapih dan jelas. Sebelum menulis repo, penting
> diketahui: **sumber & format tiap file berbeda-beda**, dan ada **temuan duplikasi** di data vision.

## 0. Ringkasan temuan (penting)

| Data | Format | Lama / Baru | Sumber asli | Duplikasi vs data lain |
| :-- | :-- | :-- | :-- | :-- |
| `chat_sft` (36015, HF) | per-turn `input`/`target` | **Lama** | sintetis topik + `jakartaresearch/indoqa` | — |
| `indoqa_sft` (3309, HF) | `input`/`target` | **Lama** | `jakartaresearch/indoqa` | — |
| `chat_multiturn` (3000, HF) | `messages` | **Lama** | `t5-gemma-2-chat-instruct-dataset-v2.jsonl` (persona santai) | — |
| `vision_sft` (1000, HF) | `id`+`images`+`messages` | **Lama** | SEACrowd / KORIKA SEA-VL + PDF `doc_scraped_*` | — |
| `chat_orpo` (1000, HF) | `prompt`/`chosen`/`rejected` | **Lama** | LLM-generated dari chat SFT | — |
| `vision_orpo` (200, HF) | `id`+`images`+`prompt`/.. | **Lama** | LLM-generated dari vision SFT | — |
| `generated_conv_agent.jsonl` teks (2000) | `messages` + `source` | **BARU** | Cendol, IndoMMLU, Wikipedia, IndoCareer, IndoCulture, LFQA-ID, QQPR, dll | **0%** overlap dgn chat lama |
| `generated_conv_agent.jsonl` vision (1000) | `messages` + `images` | **BARU** | PDF `doc_scraped_*` (sama dgn vision lama) | **~99% DUPLIKAT vision_sft** |

### Temuan duplikasi vision (kritis)
- 1000 baris vision "baru": **993** punya `id` & signature user pertama yang **sama persis** dgn `vision_sft` (1000).
- Semua 200 dokumen `doc_scraped_*` vision baru = **200/200** juga dipakai vision lama; 0 dokumen baru.
- `source` vision baru sebagian besar **kosong** (993/1000 tidak ada `source`).

> **Kesimpulan:** `generated_conv_agent.jsonl` yang "3000" sebenarnya berisi
> **2000 teks yang benar-benar baru** + **1000 vision yang = vision_sft lama (raw source-nya)**.
> Jadi yang layak ditambahkan ke repo baru adalah **2000 teks**; vision sebaiknya
> **tidak ditambah dua kali** (atau hanya 7 baris yang benar-benar baru).

## 1. Format file lokal (schema per file)

| File | Baris | Kolom | Keterangan |
| :-- | :-- | :-- | :-- |
| `data/synthetic/generated_conv_agent.jsonl` | 3000 | `id`, `source`, `category`, `num_turns`, `num_pairs`, `reviewed`, `edited_turns`, `messages` | 2000 `text_nlu_chat` + 1000 `vision_chat`. `source` ada di teks (2000), vision kebanyakan kosong. |
| `data/sft/chat_train.jsonl` | 27990 | `input`, `target`, `chat_idx`, `turn_idx`, `input_tokens`, `target_tokens` | versi lama (superseded oleh v2) |
| `data/sft/chat_train_v2.jsonl` | 36015 | sama | sama dgn HF `chat_sft` train |
| `data/sft/chat_val_v2.jsonl` | 1227 | sama | HF `chat_sft` validation |
| `data/sft/t5-gemma-2-chat-instruct-dataset.jsonl` | 2500 | `id`, `topik`, `num_turns`, `tokens`, `conversations` | lama, persona santai |
| `data/sft/t5-gemma-2-chat-instruct-dataset-v2.jsonl` | 3000 | `id`, `topik`, `num_turns`, `tokens`, `rationale`, `conversations` | = HF `chat_multiturn` (source) |
| `data/preference/orpo_train.jsonl` | 1000 | `id`, `prompt`, `chosen`, `rejected`, `flaw`, `rationale` | = HF `chat_orpo` |
| `data/preference/orpo_multimodal.jsonl` | 200 | `id`, `images`, `prompt`, `chosen`, `rejected`, `flaw`, `rationale` | = HF `vision_orpo` |
| `data/preference/preferences_dpo_light.jsonl` | 100 | `input`, `chosen`, `rejected`, `flaw_type`, `rationale` | DPO ringan (belum dipakai v8) |
| `data/multimodal/train_vision.jsonl` | 1000 | `id`, `images`, `num_turns`, `prefix_usage`, `rationale`, `messages` | = HF `vision_sft` |
| `data/grounded_qa/indoqa_train.jsonl` | 3309 | `input`, `target`, `input_tokens`, `target_tokens` | = HF `indoqa_sft` train |
| `data/grounded_qa/indoqa_val.jsonl` | 1104 | sama | = HF `indoqa_sft` validation |

## 2. Sumber asli (provenance)

| Kumpulan | Sumber HF asli | Kolom sumber |
| :-- | :-- | :-- |
| chat_sft / chat_multiturn | topik manual + `indonlp/cendol_collection_v2`? + `jakartaresearch/indoqa` | tidak ada kolom source di format akhir |
| indoqa_sft | `jakartaresearch/indoqa` | tidak ada |
| vision_sft | `SEACrowd/sea-vl_crowdsourcing`, `KORIKA-AI/sea-vl_crowdsourcing_id` + PDF `doc_scraped_*` | tidak ada (hanya `id`) |
| chat_orpo / vision_orpo | LLM-generated (flaw: hallucination, repetitive, dll) | tidak ada (`flaw`) |
| **text BARU** | `indonlp/cendol_collection_v2`, `indolem/IndoMMLU`, `indonesian-nlp/wikipedia`, `indolem/IndoCareer`, `indolem/IndoCulture`, `id-lfqa/LFQA-ID`, `robinsyihab/QQPR-triplets-ID`, `OpenWebText-Indonesia-10k`, `FineWeb-Edu-25K`, `Indonesian Simple Summaries` | **`source` (2000/2000 terisi)** |
| **vision BARU** | PDF `doc_scraped_*` (sama dgn vision lama); sebagian `SEACrowd`/`KORIKA` | `source` kosong (993/1000) |

## 3. Struktur mono-repo yang disarankan (hasil temuan)

Karena data baru teks **formatnya beda** (raw `messages`) dari `chat_sft` (per-turn `input/target`),
dan vision baru **>99% duplikat** `vision_sft`, struktur rapihnya sbb:

```
daruokta/t5gemma2-indonesia-instruct-v1/
├── README.md
├── manifest.json
├── sft/
│   ├── text/
│   │   ├── chat_hf/       config: chat_sft      (HF lama 36015 + unroll 2000 teks BARU, kolom source nullable)
│   │   └── indoqa/        config: indoqa_sft    (HF lama 3309)
│   └── vision/
│       └── hf/            config: vision_sft    (HF lama 1000 — vision BARU TIDAK ditambah; 99% dup)
└── orpo/
    ├── text/chat_orpo/    config: chat_orpo     (HF lama 1000 + adds)
    └── vision/vision_orpo/ config: vision_orpo  (HF lama 200 + adds, target 1000)
```

- **CHANGE vs draft sebelumnya:** subfolder `synthetic_text` & `synthetic_vision` **dihapus**.
  Teks baru di-**unroll & digabung** ke `chat_hf` (kolom `source` diisi, nullable utk baris HF lama).
  Vision baru **tidak dipakai** (duplikat), kecuali 7 baris unik bila diinginkan.
- Opsional: simpan raw teks baru (dgn `source`) sebagai config `multiturn_text` untuk provenance,
  terpisah dari training config — tapi tidak wajib karena user ingin "tidak terpisah".

## 4. Keputusan yang perlu dikonfirmasi

1. **Vision BARU (1000)**: 99% duplikat `vision_sft`.
   - (A) **Dapat — jangan ditambah** (rekomendasi). Repo tetap bersih, tidak double-count.
   - (B) Tetap tambahkan sebagai `multiturn_vision` (raw) untuk provenance, lalu dedupe di training.
   - (C) Tambahkan hanya 7 baris yang benar-benar baru.
2. **Teks BARU (2000)**: mau digabung ke config mana?
   - (A) **Unroll → `chat_sft`** (per-turn, terpakai langsung oleh training v8) — rekomendasi.
   - (B) Simpan raw `messages` di config terpisah `multiturn_text` (biar `source` tetap utuh), lalu v8 unroll.
3. Kolom `source` pada `chat_sft`: tambahkan (nullable) untuk provenance teks baru? **Ya** (disarankan).

Lihat `scripts/dataset/build_mono_repo.py` untuk build memory-safe; `scripts/dataset/plan_orpo_balance.py` untuk budget ORPO.
