/*
 * Paper template for JACoW conference proceedings
 * T5-Gemma-2 Instruct Paper
 */

#import "@preview/accelerated-jacow:0.14.1": jacow, jacow-table
#import "@preview/cetz:0.3.4": canvas, draw

#show: jacow.with(
  // Paper title
  title: [
    T5-Gemma-2 Instruct: Pembelajaran Instruksi dan Percakapan\
    Multimodal Berbahasa Indonesia Berbasis\
    Arsitektur Encoder-Decoder Seq2Seq
  ],

  // Author list
  authors: (
    (name: "Daru Okta Buana", at: "ind", email: "daruoktab@gmail.com"),
  ),
  affiliations: (
    ind: "Independent Researcher, Yogyakarta, Indonesia",
  ),

  // Paper abstract
  abstract: [
    #set text(size: 10pt)
    Makalah ini mempresentasikan pengembangkan *T5-Gemma-2 Instruct* (4B-4B), model bahasa berbasis arsitektur *Encoder-Decoder* (Seq2Seq) yang diadaptasi dari Gemma 3 untuk percakapan multi-turn multimodal berbahasa Indonesia. Dialog multi-turn panjang bersifat asimetris: input konteks yang sangat panjang diikuti oleh respons asisten yang pendek. Pada model *decoder-only*, pemrosesan konteks panjang ini rentan mengalami degradasi atensi. Kami memanfaatkan kemampuan representasi dua arah (*bidirectional*) pada *Encoder* T5-Gemma-2 sebagai *contextual long-term memory* untuk memahami seluruh konteks sebelum didekode secara kausal oleh *Decoder*. Selain itu, kami memperkenalkan metode *Implicit Task Steering* menggunakan *unused tokens* sebagai *prefix* tugas implisit untuk mengarahkan ragam tugas generasi tanpa mengubah prompt pengguna. Evaluasi pada SFT teks menunjukkan peningkatan kualitas respons yang konsisten (ROUGE-1 mencapai 60.07% dan BERTScore F1 mencapai 85.23%), dengan transisi ke pembelajaran multimodal berbasis visual menggunakan dataset *SEACrowd* dan scraping PDF dokumen Indonesia.
  ],
)

// Spacing optimization to balance columns
#show figure: set block(spacing: 10pt)
#show heading: set block(above: 12pt, below: 4pt)
#show bibliography: set text(size: 8pt)
#show bibliography: set par(leading: 0.40em)
#set par(justify: true, leading: 0.50em)
#show text: set text(font: "Times New Roman")


// ==========================================
// 1. PENDAHULUAN
// ==========================================
= Pendahuluan

Dominasi arsitektur _decoder-only_ pada model bahasa besar (LLM) modern menghasilkan performa tinggi pada tugas generasi bebas. Namun, arsitektur ini memiliki keterbatasan inheren pada skenario pemrosesan asimetris (_asymmetric processing_) @return_encoder_2025. Skenario dialog multi-turn, misalnya, ditandai dengan riwayat percakapan yang sangat panjang (ratusan hingga ribuan token) namun menghasilkan respons asisten yang relatif pendek.

Pada model _decoder-only_, pemrosesan masukan panjang ini rentan terhadap degradasi atensi (_attention degeneration_), di mana model kesulitan mempertahankan jangkar informasi awal karena terdistraksi oleh token-token perantara. Arsitektur _Sequence-to-Sequence_ (Seq2Seq) yang ditawarkan oleh keluarga model T5 @raffel2020t5 memberikan alternatif solusi: modul _Encoder_ memproses masukan secara dua arah (_bidirectional_) untuk menghasilkan representasi konteks yang kokoh, sementara modul _Decoder_ murni bertugas menghasilkan teks respons secara terarah.

_T5-Gemma-2_ @t5gemma2_2025 mengadaptasi model Gemma 3 @gemma3_2025 ke paradigma _Encoder-Decoder_ via metode UL2 @encoder_decoder_gemma_2025. Model ini mempertahankan bobot pra-latih Gemma 3 yang sangat kaya, sembari memperkenalkan mekanisme _Merged Attention_ pada modul decoder untuk efisiensi parameter.

Kontribusi utama dari penelitian ini adalah:
- Pengembangan *T5-Gemma-2 Instruct* sebagai model instruksi multi-turn Seq2Seq pertama untuk Bahasa Indonesia (dengan dukungan Bahasa Inggris sebagai bahasa sekunder).
- Penerapan konsep *contextual long-term memory* pada encoder untuk mengatasi degradasi pemrosesan dokumen panjang.
- Usulan metode *Implicit Task Steering* menggunakan token tak terpakai (_unused tokens_) sebagai prefix tugas implisit pada decoder.
- Rencana pengembangan *Vision SFT* dua tahap dengan dataset visual dokumen berbahasa Indonesia yang dikonstruksi dari scraping PDF.


// ==========================================
// 2. ARSITEKTUR MODEL DAN REPRESENTASI MEMORI
// ==========================================
= Arsitektur Model dan Memori Konseptual

== Struktur T5-Gemma-2

T5-Gemma-2 memiliki total parameter sekitar 7.51B untuk varian 4B-4B @encoder_decoder_gemma_2025. Model ini terdiri dari encoder 3.88B (pemrosesan teks) ditambah model vision SigLIP 400M, serta decoder sebesar 3.88B. Hubungan antara token embedding pada encoder dan decoder saling diikat (_tied embeddings_) untuk efisiensi representasi kosakata sebesar 262.144 token.

== Mekanisme Merged Attention

Berbeda dengan arsitektur T5 klasik yang menggunakan modul _cross-attention_ terpisah di setiap layer decoder, T5-Gemma-2 menerapkan mekanisme _Merged Attention_ @t5gemma2_2025. Modul ini menggabungkan _self-attention_ (terhadap respons decoder sebelumnya) dan _cross-attention_ (terhadap representasi encoder) ke dalam satu modul tunggal:

$ K, V = [X; H] $

Di sini, $X$ merupakan _hidden states_ dari decoder pada langkah saat ini, sedangkan $H$ adalah representasi keluaran dari encoder. Mekanisme masking yang digunakan bersifat hibrida: _bidirectional_ untuk bagian token $H$ (sehingga informasi konteks saling berinteraksi bebas) dan _causal_ untuk token $X$ (guna mempertahankan sifat generasi auto-regresif).
#figure(
  image("t5gemma2_merged_attention.png", width: 100%),
  caption: [Mekanisme Merged Attention pada T5-Gemma-2.],
) <fig:merged-attn>

== Encoder sebagai Contextual Long-Term Memory

Peran utama encoder dalam dialog multi-turn adalah mengompresi riwayat pembicaraan yang panjang menjadi representasi semantik yang stabil. Karena encoder memproses informasi secara _bidirectional_ tanpa masking kausal, setiap token dalam riwayat dapat saling memperhatikan. Hal ini memberikan kestabilan konteks yang bertindak mirip dengan _long-term memory_. Representasi $H$ yang dihasilkan encoder kemudian diinjeksikan secara berulang pada modul _Merged Attention_ di setiap lapisan decoder. Pola ini mencegah hilangnya informasi penting di tengah jalan (_lost in the middle_), suatu masalah yang kerap dialami oleh LLM _decoder-only_.


// ==========================================
// 3. PIPELINE PELATIHAN DUA TAHAP
// ==========================================
= Pipeline Pelatihan Dua Tahap

Untuk mewujudkan asisten multimodal berbahasa Indonesia yang andal, kami menerapkan metodologi pelatihan bertahap seperti yang diilustrasikan pada @fig:pipeline.

#figure(
  canvas(length: 1pt, {
    import draw: *
    let nd(cx, cy, w, h, lbl, fc, sc) = {
      rect((cx - w / 2, cy + h / 2), (cx + w / 2, cy - h / 2), fill: fc, stroke: 1.2pt + sc, radius: 3pt)
      content((cx, cy), text(7pt, align(center + horizon, lbl)))
    }
    let ard(x, y1, y2) = line(
      (x, y1),
      (x, y2),
      mark: (end: "stealth", fill: rgb("#1f2937"), size: 4.5pt),
      stroke: 1.2pt + rgb("#4b5563"),
    )

    nd(
      0,
      0,
      140,
      24,
      [
        #text(7pt, weight: "bold")[google/t5gemma-2-4b-4b] \
        #text(5.5pt)[Base Model · ~7.51B parameter]
      ],
      rgb("#fef9c3"),
      rgb("#b45309"),
    )

    ard(0, -12, -22)

    nd(
      0,
      -40,
      140,
      36,
      [
        #text(7.5pt, weight: "bold", fill: rgb("#1e40af"))[Tahap 1: SFT Teks] \
        #text(5.5pt)[LoRA $r=128$, $alpha=256$ · GrokAdEMAMix] \
        #text(5pt, fill: rgb("#374151"))[Chat 36K + IndoQA 3.3K sampel]
      ],
      rgb("#eff6ff"),
      rgb("#2563eb"),
    )

    ard(0, -58, -68)

    nd(
      0,
      -80,
      140,
      24,
      [
        #text(7pt, weight: "bold")[SFT Teks Model (v4-unsloth)] \
        #text(5.5pt)[Konvergensi awal bahasa Indonesia]
      ],
      rgb("#dcfce7"),
      rgb("#16a34a"),
    )

    ard(0, -92, -102)

    nd(
      0,
      -124,
      140,
      44,
      [
        #text(7.5pt, weight: "bold", fill: rgb("#5b21b6"))[Tahap 2: Vision SFT] \
        #text(5.5pt)[LoRA $r=64$, $alpha=128$ · FastVisionModel] \
        #text(5pt)[SEACrowd (800) + PDF scraping ID (200)] \
        #text(5pt, fill: rgb("#7c3aed"), style: "italic")[Dalam Pengembangan Dataset]
      ],
      rgb("#fdf4ff"),
      rgb("#7c3aed"),
    )

    ard(0, -146, -156)

    nd(
      0,
      -168,
      140,
      24,
      [
        #text(7pt, weight: "bold")[Multimodal Chatbot] \
        #text(5.5pt)[LoRA adapter · coming soon]
      ],
      rgb("#e0e7ff"),
      rgb("#4f46e5"),
    )
  }),
  caption: [Alur pelatihan bertahap T5-Gemma-2 Instruct.],
) <fig:pipeline>

== Tahap 1: SFT Teks

Tahap awal difokuskan pada penyelarasan kemampuan berbahasa Indonesia menggunakan metode LoRA @hu2021lora. Adapter LoRA diaplikasikan pada 7 modul proyeksi utama (`q_proj`, `k_proj`, `v_proj`, `o_proj`, `gate_proj`, `up_proj`, `down_proj`) untuk melatih sekitar 755 juta parameter adaptif.

Untuk mempercepat konvergensi dan mengatasi masalah _grokking_, kami menerapkan kombinasi optimizer *GrokAdEMAMix*. GrokFast melacak komponen gradien yang konsisten melalui EMA lambat, sementara AdEMAMix memelihara dua tingkat memori momentum jangka pendek dan jangka panjang untuk pembaharuan bobot yang lebih dinamis. Pelatihan diperkuat dengan teknik label smoothing sebesar 0.1 dan NEFTune noise sebesar 5.0 untuk mencegah _overfitting_ pada dataset instruksi.

== Tahap 2: Vision SFT (Sedang Dikembangkan)

Setelah model teks stabil, tahap berikutnya adalah pelatihan multimodal berbasis visual menggunakan `FastVisionModel`. Kami mengintegrasikan adapter baru ($r=64$, $alpha=128$) pada seluruh lapisan atensi dan MLP untuk menghubungkan kompresi gambar SigLIP dengan generasi teks decoder.

Dataset visual saat ini sedang dikonstruksi secara hibrida:
- *SEACrowd/sea-vl_crowdsourcing (800 gambar)*: Kumpulan gambar visual di kawasan regional Asia Tenggara yang difilter khusus untuk interaksi berbahasa Indonesia.
- *PDF Scraping (200 gambar)*: Menggunakan skrip berbasis PyMuPDF untuk memotong per-halaman dokumen resmi berbahasa Indonesia (panduan pengguna, dokumen kebijakan, brosur komersial) menjadi format gambar resolusi tinggi.

Dataset visual ini diubah menjadi format percakapan _visually-grounded_ dibantu oleh LLM generator, di mana pertanyaan dari pengguna merujuk langsung ke komponen visual dokumen (misalnya layout tabel atau diagram).


// ==========================================
// 4. METODOLOGI IMPLICIT TASK STEERING
// ==========================================
= Metodologi Implicit Task Steering

== Konsep Task Prefix Implisit

Untuk mengoptimalkan kinerja model pada beragam jenis tugas (peringkasan, ekstraksi informasi, tanya-jawab) tanpa mengotori prompt alami pengguna, kami memanfaatkan token tak terpakai (_unused tokens_) pada kosakata asli Gemma (ID 7 hingga 12). Kami menetapkan pemetaan token tersebut sebagai penunjuk tugas (_task steering_) seperti yang ditunjukkan pada @table:prefix.

#figure(
  jacow-table(
    "lc",
    [Token ID],
    [Tipe Tugas Utama (_Task Type_)],
    [`<unused1>`],
    [Summarize (Peringkasan)],
    [`<unused2>`],
    [Translate (Terjemahan)],
    [`<unused3>`],
    [NER (Ekstraksi Entitas)],
    [`<unused4>`],
    [QA (Menjawab Pertanyaan)],
    [`<unused5>`],
    [Paraphrase (Parafrase)],
    [`<unused6>`],
    [General Chat (Percakapan Umum)],
  ),
  caption: [Pemetaan token implisit untuk penunjuk tugas.],
) <table:prefix>

Dalam proses inferensi, decoder dilatih untuk memprediksi token prefix ini langsung sesaat setelah token BOS (Beginning of Sequence), sebelum menghasilkan kalimat respons utama. Melalui representasi dua arah pada encoder, model secara mandiri menginferensi intensi teks masukan dan memilih prefix tugas yang sesuai secara otomatis.

#figure(
  canvas(length: 1pt, {
    import draw: *
    let nd(cx, cy, w, h, lbl, fc, sc) = {
      rect((cx - w / 2, cy + h / 2), (cx + w / 2, cy - h / 2), fill: fc, stroke: 1.2pt + sc, radius: 3pt)
      content((cx, cy), text(7pt, align(center + horizon, lbl)))
    }
    let ard(x, y1, y2) = line(
      (x, y1),
      (x, y2),
      mark: (end: "stealth", fill: rgb("#1f2937"), size: 4.5pt),
      stroke: 1.2pt + rgb("#4b5563"),
    )

    nd(
      0,
      0,
      160,
      18,
      [
        #text(6.5pt, style: "italic")["Tolong ringkas dokumen ini untuk saya..."]
      ],
      rgb("#fef9c3"),
      rgb("#b45309"),
    )

    ard(0, -9, -21)

    nd(
      0,
      -32,
      160,
      22,
      [
        #text(7.5pt, weight: "bold")[Encoder (Bidireksional)] \
        #text(5.5pt)[Menginfer intent dari seluruh konteks #sym.arrow $H$]
      ],
      rgb("#dbeafe"),
      rgb("#2563eb"),
    )

    ard(0, -43, -55)

    nd(
      0,
      -66,
      160,
      22,
      [
        #text(7.5pt, weight: "bold", fill: rgb("#15803d"))[\<unused1\>] #h(2pt) #text(6.5pt)["Berikut ringkasan ..."]
      ],
      rgb("#dcfce7"),
      rgb("#16a34a"),
    )

    content((0, -92), text(5.5pt, fill: rgb("#6b7280"), style: "italic", align(
      center,
    )[Decoder mendeklarasikan tipe tugas secara implisit \ tanpa mengubah instruksi prompt pengguna]))
  }),
  caption: [Alur kerja mekanisme Implicit Task Steering.],
) <fig:task-steering>

== Logit Masking Non-Destruktif

untuk menjamin model tidak mengalami halusinasi dengan memunculkan token tak relevan (seperti token visual SigLIP atau token tak terpakai lainnya) selama fase generasi bebas, kami menerapkan mekanisme _Logit Masking_. Masking ini disuntikkan secara dinamis menggunakan PyTorch `forward_hook` pada komponen `lm_head`. Kelebihan teknik ini adalah sifatnya yang non-destruktif (tidak mengubah bobot asli model) dan kompatibel penuh dengan fitur _gradient checkpointing_ saat pelatihan.


// ==========================================
// 5. EKSPERIMEN DAN HASIL EVALUASI
// ==========================================
= Eksperimen dan Hasil Evaluasi

== Konfigurasi Pelatihan

Model dilatih secara mandiri pada platform *molab* (marimo notebook environment) menggunakan GPU *NVIDIA Blackwell Pro 6000* dengan *96 GB VRAM*. Kami menggunakan ukuran batch efektif sebesar 128 dengan laju pembelajaran yang diatur menggunakan _cosine scheduler_ dan 100 langkah _warmup_. Dataset SFT terdiri dari gabungan Chat berbahasa Indonesia (36.000 sampel) dan IndoQA (3.300 sampel).

Setelah fase SFT, model diselaraskan menggunakan metode ORPO (_Odds Ratio Preference Optimization_) selama 2 epoch (16 langkah evaluasi). Kami menggunakan dataset preferensi dengan laju pembelajaran $5 times 10^(-6)$, ukuran batch efektif 128, dan koefisien penalti preference odds-ratio ($alpha$) sebesar 0.05.

== Hasil Kuantitatif SFT dan Penyelarasan ORPO

Evaluasi berkala dilakukan setiap 200 langkah pada fase SFT dan pada akhir fase ORPO. Hasil pengukuran metrik disajikan pada @table:results.

Meskipun pelatihan SFT dilakukan hingga 1200 langkah, kami mengamati gejala _overfitting_ ringan setelah langkah ke-1000 (di mana metrik evaluasi mulai mendatar atau menurun secara perlahan). Oleh karena itu, checkpoint langkah ke-1000 dipilih sebagai *final SFT adapter* yang kemudian digunakan sebagai inisialisasi awal untuk tahap penyelarasan preferensi ORPO.

Penyelarasan preferensi menggunakan ORPO memberikan dampak positif yang sangat signifikan pada aspek kalibrasi model. Evaluasi pada akhir fase ORPO (step 16) mencatat penurunan *evaluation loss* secara drastis sebesar -55% (turun dari 2.865 menjadi 1.287) dan *perplexity* yang terpangkas menjadi 3.62. Penurunan drastis ini menunjukkan bahwa ORPO secara efektif menekan probabilitas token negatif (rejected) dan melatih model untuk menaruh probabilitas tinggi pada respons pilihan (chosen) tanpa memerlukan model reward terpisah. Selain itu, kualitas generatif model tetap terjaga stabil dengan metrik ROUGE-1 naik tipis ke 60.41% dan Exact Match mencapai 27.37%.

#figure(
  jacow-table(
    "ccccccc",
    [Step],
    [Loss],
    [PPL],
    [R-1],
    [R-L],
    [BLEU],
    [BERT],
    [SFT-200], [2.958], [19.27], [53.38], [49.05], [12.57], [82.80],
    [SFT-400], [2.897], [18.11], [56.63], [52.12], [14.46], [83.93],
    [SFT-600], [2.864], [17.54], [58.02], [53.36], [14.49], [84.39],
    [SFT-800], [2.857], [17.40], [59.01], [54.42], [15.44], [84.90],
    [SFT-1000], [2.865], [17.54], [60.07], [55.51], [*15.94*], [*85.23*],
    [ORPO-16], [*1.287*], [*3.62*], [*60.41*], [*55.56*], [15.89], [*85.23*],
  ),
  caption: [Metrik evaluasi model T5-Gemma-2 Instruct pada fase SFT dan penyelarasan ORPO. R-1, R-L, BLEU, dan BERT dalam %. R-1=ROUGE-1 @lin2004rouge, R-L=ROUGE-L, BLEU=BLEU-4 @papineni2002bleu, BERT=BERTScore F1 @zhang2019bertscore (dengan *embeddinggemma-300m* @vera2025embeddinggemmapowerfullightweighttext).],
) <table:results>

== Fenomena Loss-Quality Divergence

Kami mengamati fenomena menarik pada langkah ke-1000: meskipun _validation loss_ mengalami kenaikan tipis (dari 2.857 ke 2.865), seluruh metrik evaluasi kualitas teks (ROUGE, BLEU, dan BERTScore) tetap mengalami kenaikan. Hal ini menunjukkan fenomena _beneficial generalization_, di mana meskipun model secara statistik kurang terkalibrasi (_calibrated_) pada distribusi token mentah, penentuan token dengan probabilitas tertinggi tetap menghasilkan kalimat yang secara semantik benar. Pengukuran metrik METEOR yang naik sebesar +6.33% juga mengonfirmasi peningkatan penguasaan aspek morfologi Bahasa Indonesia oleh model.

Perkembangan kurva training untuk Loss & Perplexity (PPL) disajikan pada @fig:loss-ppl-chart.

#figure(
  image("loss_ppl_chart.png", width: 95%),
  caption: [Kurva evaluasi Loss dan Perplexity (PPL) antara model SFT (fase training) dan model ORPO.],
) <fig:loss-ppl-chart>


// ==========================================
// 6. PEMBAHASAN DAN PEKERJAAN MASA DEPAN
// ==========================================
= Pembahasan dan Kerja Selanjutnya

== Keunggulan Seq2Seq untuk Bahasa Indonesia

Rasio panjang target-ke-input pada dataset kami sangat rendah (0.09x untuk Chat, 0.04x untuk IndoQA), menegaskan sifat asimetris dari tugas chatbot multi-turn. Penggunaan encoder dua arah terbukti sangat efisien dalam memadatkan konteks dokumen tebal berbahasa Indonesia. Hal ini meminimalkan biaya komputasi decoder yang hanya perlu fokus pada pembangkitan respons ringkas.

Kurva perbandingan metrik NLG lexical (ROUGE-1, ROUGE-2, ROUGE-L, dan BLEU) disajikan pada @fig:nlg-metrics-chart, sedangkan metrik semantik (BERTScore F1) dan klasifikasi (METEOR dan Exact Match) disajikan pada @fig:semantic-em-chart.

#figure(
  image("nlg_metrics_chart.png", width: 95%),
  caption: [Perbandingan metrik kualitas generasi NLG (ROUGE-1, ROUGE-2, ROUGE-L, dan BLEU) antara SFT dan ORPO.],
) <fig:nlg-metrics-chart>

#figure(
  image("semantic_em_chart.png", width: 95%),
  caption: [Perbandingan metrik semantik (BERTScore F1) dan klasifikasi (METEOR dan Exact Match) antara SFT dan ORPO.],
) <fig:semantic-em-chart>

== Keterbatasan dan Rencana Kerja Selanjutnya

Keterbatasan utama penelitian saat ini adalah ukuran dataset visual untuk Tahap 2 yang masih relatif terbatas (~1.000 sampel). Selain itu, kami belum menyertakan evaluasi manusia secara masif untuk menilai kealamian dialog.

Rencana pekerjaan berikutnya meliputi:
1. Menyelesaikan pembuatan pasangan percakapan visual berbasis 1.000 gambar (SEACrowd + PDF scraping).
2. Menjalankan penyelarasan preferensi manusia menggunakan metode ORPO @rafailov2023dpo.
3. Melakukan kuantisasi model ke format INT4/INT8 agar dapat dideploy pada perangkat lokal dengan memori terbatas.


// ==========================================
// 7. KESIMPULAN
// ==========================================
= Kesimpulan

Penelitian ini mempresentasikan rancangan model *T5-Gemma-2 Instruct* yang memanfaatkan kekuatan arsitektur encoder-decoder Seq2Seq. Dengan menerapkan konsep _contextual long-term memory_ pada encoder and mekanisme _Implicit Task Steering_ via token tidak terpakai, model ini menawarkan stabilitas pemahaman konteks percakapan multi-turn berbahasa Indonesia yang unggul dibandingkan pendekatan _decoder-only_. Evaluasi awal pada SFT teks mencatat performa menjanjikan dengan nilai ROUGE-1 sebesar 60.07% dan BERTScore F1 sebesar 85.23%.


#bibliography("references.bib")
