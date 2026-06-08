"""
Real-World Multi-Usecase Evaluation — Bahasa Indonesia
=======================================================

Evaluasi komprehensif model sentence embedding untuk berbagai
use case nyata dalam bahasa Indonesia. Semua data di-hardcode
untuk reproducibility dan kemudahan penggunaan.

Use Cases:
1. Semantic Search (multi-domain)
2. Paraphrase Detection
3. FAQ Matching (customer service)
4. Document Clustering
5. Cross-lingual (EN↔ID)
6. Duplicate Question Detection
7. Intent Classification
8. Short vs Long Text Matching

Usage:
    python eval_realworld.py
    python eval_realworld.py --model t5gemma2-embedding-v1/final
"""

import argparse
import logging
import torch
import numpy as np
from typing import List, Dict, Tuple, TypedDict
from sentence_transformers import SentenceTransformer
from sklearn.metrics import accuracy_score, f1_score
from collections import defaultdict

logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)


# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

def cosine_similarity(a, b):
    """Hitung cosine similarity antara dua vektor."""
    a = np.array(a, dtype=np.float32)
    b = np.array(b, dtype=np.float32)
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-8))


def encode_texts(model, texts: List[str]) -> np.ndarray:
    """Encode texts dengan model."""
    with torch.no_grad():
        embeddings = model.encode(texts, convert_to_numpy=True, show_progress_bar=False)
    return embeddings


def print_section(title: str):
    """Print section header."""
    print(f"\n{'='*80}")
    print(f"  {title}")
    print(f"{'='*80}\n")


def print_subsection(title: str):
    """Print subsection header."""
    print(f"\n  {'─'*60}")
    print(f"  {title}")
    print(f"  {'─'*60}")


# ============================================================================
# 1. SEMANTIC SEARCH — Multi-Domain
# ============================================================================

class SemanticSearchTest(TypedDict):
    domain: str
    query: str
    corpus: List[str]
    expected_top: List[int]

SEMANTIC_SEARCH_TESTS: List[SemanticSearchTest] = [
    # ── E-Commerce ──
    {
        "domain": "E-Commerce",
        "query": "sepatu lari yang nyaman untuk marathon",
        "corpus": [
            "Nike Air Zoom Pegasus - sepatu running ringan dan empuk untuk pelari jarak jauh",
            "Adidas Ultraboost - sepatu lari premium dengan teknologi Boost untuk kenyamanan maksimal",
            "Converse Chuck Taylor - sepatu sneaker klasik untuk gaya casual sehari-hari",
            "Sandal Birkenstock - alas kaki santai dengan footbed cork yang ergonomis",
            "Sepatu safety Caterpillar - boots kerja tahan air untuk konstruksi",
            "Crocs classic clog - sandal karet ringan untuk di rumah",
        ],
        "expected_top": [0, 1],  # Nike dan Adidas harus di atas
    },
    {
        "domain": "E-Commerce",
        "query": "laptop untuk editing video dan desain grafis",
        "corpus": [
            "MacBook Pro M3 - laptop profesional dengan chip M3 untuk kreator konten",
            "ASUS ROG Zephyrus - laptop gaming dengan RTX 4070 dan layar 240Hz",
            "Lenovo ThinkPad X1 Carbon - ultrabook bisnis ringan untuk produktivitas",
            "Dell XPS 15 - laptop premium dengan layar OLED 3.5K untuk desainer",
            "Acer Aspire 3 - laptop entry-level untuk pelajar dan tugas ringan",
            "Samsung Galaxy Tab S9 - tablet Android untuk hiburan dan browsing",
        ],
        "expected_top": [0, 3],  # MacBook Pro dan Dell XPS
    },
    # ── Kuliner ──
    {
        "domain": "Kuliner",
        "query": "resep masakan pedas khas Padang",
        "corpus": [
            "Rendang daging sapi - masakan Minangkabau yang dimasak dengan santan dan rempah hingga kering",
            "Gulai ayam Padang - sayur kuah kuning kental dengan bumbu rempah khas Sumatera Barat",
            "Sushi salmon - makanan Jepang dengan nasi dan ikan mentah",
            "Nasi goreng kampung - nasi goreng sederhana dengan telur dan kecap",
            "Dendeng balado - daging tipis goreng dengan sambal merah pedas khas Minang",
            "Pizza margherita - pizza klasik Italia dengan saus tomat dan mozzarella",
        ],
        "expected_top": [0, 1, 4],  # Rendang, Gulai, Dendeng
    },
    # ── Kesehatan ──
    {
        "domain": "Kesehatan",
        "query": "gejala diabetes dan cara pencegahannya",
        "corpus": [
            "Diabetes melitus ditandai dengan sering haus, sering buang air kecil, dan penurunan berat badan",
            "Pencegahan diabetes dapat dilakukan dengan pola makan sehat, olahraga teratur, dan menjaga berat badan",
            "Demam berdarah dengue disebabkan oleh nyamuk Aedes aegypti dan ditandai bintik merah",
            "Asam urat menyebabkan nyeri sendi yang disebabkan penumpukan kristal asam urat",
            "Hipertensi atau tekanan darah tinggi sering disebut silent killer karena tanpa gejala",
            "Cara menurunkan kolesterol dengan menghindari makanan berlemak dan rutin berolahraga",
        ],
        "expected_top": [0, 1],
    },
    # ── Teknologi ──
    {
        "domain": "Teknologi",
        "query": "cara membuat aplikasi mobile dengan React Native",
        "corpus": [
            "Tutorial React Native untuk pemula - membuat aplikasi Android dan iOS dengan JavaScript",
            "Flutter vs React Native - perbandingan framework cross-platform mobile development",
            "Panduan belajar Python untuk data science dan machine learning",
            "Cara membuat website dengan WordPress tanpa coding",
            "Tutorial Docker untuk deployment aplikasi di cloud server",
            "Belajar SQL untuk mengelola database relasional",
        ],
        "expected_top": [0, 1],
    },
    # ── Hukum ──
    {
        "domain": "Hukum",
        "query": "prosedur cerai di Indonesia menurut undang-undang",
        "corpus": [
            "Gugatan perceraian diajukan ke Pengadilan Agama bagi yang beragama Islam sesuai UU Perkawinan",
            "Proses perceraian di Pengadilan Negeri untuk non-Muslim dengan mengajukan permohonan cerai talak atau gugat",
            "Pembuatan akta kelahiran anak di kantor Disdukcapil dengan membawa surat keterangan lahir",
            "Pendaftaran NPWP secara online melalui situs resmi DJP pajak",
            "Cara mengurus paspor baru di kantor imigrasi dengan membawa dokumen yang diperlukan",
            "Prosedur pelaporan LHKPN bagi pejabat negara sesuai UU KPK",
        ],
        "expected_top": [0, 1],
    },
    # ── Pendidikan ──
    {
        "domain": "Pendidikan",
        "query": "beasiswa S2 luar negeri untuk mahasiswa Indonesia",
        "corpus": [
            "LPDP memberikan beasiswa penuh untuk studi magister dan doktoral di universitas top dunia",
            "Beasiswa Chevening dari pemerintah Inggris untuk program master satu tahun di UK",
            "Program magang MSIB Kampus Merdeka untuk mahasiswa S1 semester 6-7",
            "Jadwal pendaftaran UTBK SNBT 2024 untuk masuk perguruan tinggi negeri",
            "Beasiswa Fulbright untuk studi S2 dan S3 di Amerika Serikat",
            "Kursus bahasa Inggris online gratis dari British Council",
        ],
        "expected_top": [0, 1, 4],  # LPDP, Chevening, Fulbright
    },
]


def eval_semantic_search(model) -> Dict:
    """Evaluasi kemampuan semantic search multi-domain."""
    print_section("1. SEMANTIC SEARCH (Multi-Domain)")

    total_tests = 0
    correct_top1 = 0
    correct_top3 = 0
    domain_results = {}

    for test in SEMANTIC_SEARCH_TESTS:
        # Cast eksplisit agar type checker tahu tipe setiap field
        domain: str = test["domain"]
        query: str = test["query"]
        corpus: List[str] = list(test["corpus"])
        expected: List[int] = test["expected_top"]

        # Encode
        q_emb = encode_texts(model, [query])
        c_embs = encode_texts(model, corpus)

        # Calculate similarities
        sims = [cosine_similarity(q_emb[0], c_embs[i]) for i in range(len(corpus))]
        ranked = sorted(enumerate(sims), key=lambda x: x[1], reverse=True)

        # Check accuracy
        top1_idx = ranked[0][0]
        top3_idxs = [r[0] for r in ranked[:3]]

        is_top1 = top1_idx in expected
        is_top3 = any(idx in expected for idx in top3_idxs)

        correct_top1 += is_top1
        correct_top3 += is_top3
        total_tests += 1

        # Print results
        print_subsection(f"[{domain}] Query: \"{query}\"")
        for rank, (idx, score) in enumerate(ranked[:4], 1):
            marker = "✅" if idx in expected else "  "
            print(f"    {rank}. [{score:.4f}] {marker} {corpus[idx][:70]}...")
        print(f"    → Top-1 {'✅ BENAR' if is_top1 else '❌ SALAH'} | Top-3 {'✅' if is_top3 else '❌'}")

        domain_results[domain] = {"top1": is_top1, "top3": is_top3}

    acc_top1 = correct_top1 / total_tests
    acc_top3 = correct_top3 / total_tests
    print(f"\n  📊 SEMANTIC SEARCH RESULTS:")
    print(f"     Top-1 Accuracy: {acc_top1:.1%} ({correct_top1}/{total_tests})")
    print(f"     Top-3 Accuracy: {acc_top3:.1%} ({correct_top3}/{total_tests})")

    return {"search_top1_acc": acc_top1, "search_top3_acc": acc_top3}


# ============================================================================
# 2. PARAPHRASE DETECTION
# ============================================================================

PARAPHRASE_PAIRS = [
    # ── True Paraphrases (label=1) ──
    # Formal ↔ Informal
    ("Bagaimana cara mendaftar akun?", "Gimana caranya bikin akun?", 1),
    ("Berapa harga tiket pesawat ke Bali?", "Tiket flight ke Bali berapa?", 1),
    ("Apakah produk ini masih tersedia?", "Barang ini masih ready stock ga?", 1),
    ("Kapan pengiriman akan tiba?", "Paket nyampe kapan ya?", 1),
    ("Bagaimana prosedur pengembalian barang?", "Cara return barang gimana?", 1),
    # Synonym substitution
    ("Saya ingin membatalkan pesanan", "Saya mau cancel order", 1),
    ("Dokter spesialis jantung terbaik", "Kardiologis paling bagus", 1),
    ("Cuaca hari ini sangat panas", "Hari ini hawanya gerah banget", 1),
    ("Anak saya demam tinggi", "Anak saya panasnya tinggi", 1),
    ("Restoran ini menyajikan makanan enak", "Resto ini sajiannya lezat", 1),
    # Restructuring
    ("Universitas Indonesia terletak di Depok", "Di Depok terdapat Universitas Indonesia", 1),
    ("Harga BBM naik lagi bulan ini", "Bulan ini terjadi kenaikan harga BBM", 1),
    ("Pemerintah membangun infrastruktur di desa", "Infrastruktur desa dibangun oleh pemerintah", 1),
    # Code-switching (common in Indonesian)
    ("Meeting besok jam 10 pagi", "Rapat besok pukul 10.00", 1),
    ("Deadline project akhir bulan", "Tenggat waktu proyek di penghujung bulan", 1),

    # ── Not Paraphrases (label=0) ──
    ("Bagaimana cara membuat kue?", "Siapa penemu lampu pijar?", 0),
    ("Berapa harga laptop?", "Dimana lokasi rumah sakit?", 0),
    ("Apa manfaat olahraga?", "Kapan musim hujan dimulai?", 0),
    ("Cara install Windows 11", "Resep nasi goreng spesial", 0),
    ("Jadwal keberangkatan kereta", "Harga emas hari ini", 0),
    ("Gejala penyakit flu", "Cara menanam padi", 0),
    ("Tips menghemat uang", "Sejarah kerajaan Majapahit", 0),
    ("Cara merawat kucing", "Jadwal sholat hari ini", 0),
    # Tricky negatives (same topic, different meaning)
    ("Harga saham naik", "Harga saham turun", 0),
    ("Cuaca hari ini cerah", "Hari ini hujan lebat", 0),
    ("Indonesia menang 3-0", "Indonesia kalah 0-3", 0),
    ("Toko buka jam 8 pagi", "Toko tutup jam 8 malam", 0),
    ("Produk ini mahal sekali", "Produk ini sangat murah", 0),
    ("Dia lulus ujian", "Dia gagal ujian", 0),
    ("Pasien sembuh total", "Pasien meninggal dunia", 0),
]


def eval_paraphrase_detection(model, threshold=0.5) -> Dict:
    """Evaluasi kemampuan deteksi parafrase."""
    print_section("2. PARAPHRASE DETECTION")

    predictions = []
    labels = []
    all_sims = []

    for s1, s2, label in PARAPHRASE_PAIRS:
        embs = encode_texts(model, [s1, s2])
        sim = cosine_similarity(embs[0], embs[1])
        pred = 1 if sim >= threshold else 0

        predictions.append(pred)
        labels.append(label)
        all_sims.append(sim)

        marker = "✅" if pred == label else "❌"
        label_str = "PARA" if label == 1 else "NOT "
        print(f"    {marker} [{sim:.4f}] [{label_str}] {s1[:35]:35s} ↔ {s2[:35]}")

    acc = accuracy_score(labels, predictions)
    f1 = f1_score(labels, predictions, average="binary")

    # Per-class analysis
    true_para_sims = [s for s, l in zip(all_sims, labels) if l == 1]
    false_para_sims = [s for s, l in zip(all_sims, labels) if l == 0]

    print(f"\n  📊 PARAPHRASE DETECTION RESULTS (threshold={threshold}):")
    print(f"     Accuracy:  {acc:.1%}")
    print(f"     F1 Score:  {f1:.4f}")
    print(f"     Avg similarity (paraphrase):     {np.mean(true_para_sims):.4f}")
    print(f"     Avg similarity (non-paraphrase): {np.mean(false_para_sims):.4f}")
    print(f"     Separation gap:                  {np.mean(true_para_sims) - np.mean(false_para_sims):.4f}")

    return {"para_accuracy": acc, "para_f1": f1, "para_gap": float(np.mean(true_para_sims) - np.mean(false_para_sims))}


# ============================================================================
# 3. FAQ MATCHING (Customer Service)
# ============================================================================

FAQ_DATABASE = {
    # E-Commerce
    "Berapa lama waktu pengiriman?": "Pengiriman 2-3 hari untuk Jawa, 5-7 hari untuk luar Jawa.",
    "Bagaimana cara melakukan pembayaran?": "Kami menerima transfer bank, e-wallet (GoPay, OVO, Dana), dan COD.",
    "Apakah ada garansi produk?": "Ya, semua produk mendapat garansi resmi 1 tahun.",
    "Bagaimana cara mengembalikan barang?": "Ajukan pengembalian dalam 7 hari melalui menu 'Pesanan Saya'.",
    "Apakah bisa tukar ukuran?": "Bisa, hubungi CS kami dalam 3 hari setelah barang diterima.",
    # Banking
    "Bagaimana cara membuka rekening baru?": "Kunjungi cabang terdekat dengan KTP dan NPWP.",
    "Berapa saldo minimum rekening?": "Saldo minimum Rp 50.000 untuk tabungan reguler.",
    "Bagaimana cara mengaktifkan mobile banking?": "Download aplikasi di Play Store, daftar dengan nomor rekening.",
    "Berapa limit transfer per hari?": "Transfer sesama bank Rp 50 juta, antar bank Rp 25 juta per hari.",
    "Bagaimana jika kartu ATM tertelan mesin?": "Hubungi call center 1500-xxx atau kunjungi cabang terdekat.",
    # Kesehatan
    "Bagaimana cara membuat janji dengan dokter?": "Booking via aplikasi atau telepon ke nomor registrasi rumah sakit.",
    "Apakah BPJS bisa digunakan di sini?": "Ya, kami menerima BPJS Kesehatan untuk rawat jalan dan rawat inap.",
    "Jam operasional klinik kapan?": "Senin-Jumat 08:00-20:00, Sabtu 08:00-14:00, Minggu tutup.",
    "Bagaimana prosedur rawat inap?": "Datang ke IGD, dokter akan menentukan perlu rawat inap atau tidak.",
    "Apakah ada layanan telemedicine?": "Ya, konsultasi online tersedia via aplikasi kami 24 jam.",
}

FAQ_USER_QUERIES = [
    # E-Commerce queries (informal/varied)
    ("Kapan paket saya sampai?", "Berapa lama waktu pengiriman?"),
    ("Bayarnya bisa pake gopay?", "Bagaimana cara melakukan pembayaran?"),
    ("Garansi berapa lama?", "Apakah ada garansi produk?"),
    ("Mau return barang gimana?", "Bagaimana cara mengembalikan barang?"),
    ("Ukurannya kegedean bisa ganti?", "Apakah bisa tukar ukuran?"),
    ("Berapa hari ya sampainya?", "Berapa lama waktu pengiriman?"),
    ("COD bisa ga?", "Bagaimana cara melakukan pembayaran?"),
    # Banking queries
    ("Mau bikin rekening baru gimana?", "Bagaimana cara membuka rekening baru?"),
    ("Minimal saldo di rekening berapa?", "Berapa saldo minimum rekening?"),
    ("Cara daftar m-banking", "Bagaimana cara mengaktifkan mobile banking?"),
    ("Sehari bisa transfer berapa?", "Berapa limit transfer per hari?"),
    ("ATM saya ketelen mesin gimana?", "Bagaimana jika kartu ATM tertelan mesin?"),
    # Healthcare queries
    ("Mau booking dokter gimana?", "Bagaimana cara membuat janji dengan dokter?"),
    ("Bisa pakai BPJS?", "Apakah BPJS bisa digunakan di sini?"),
    ("Klinik buka sampe jam berapa?", "Jam operasional klinik kapan?"),
    ("Kalau harus opname prosedurnya gimana?", "Bagaimana prosedur rawat inap?"),
    ("Ada konsultasi online ga?", "Apakah ada layanan telemedicine?"),
    # Edge cases — queries yang lebih jauh dari FAQ
    ("Barang rusak saat diterima", "Bagaimana cara mengembalikan barang?"),
    ("Nomor rekening salah saat transfer", "Bagaimana jika kartu ATM tertelan mesin?"),
    ("Dokter anak ada jadwal kapan?", "Bagaimana cara membuat janji dengan dokter?"),
]


def eval_faq_matching(model) -> Dict:
    """Evaluasi FAQ matching accuracy."""
    print_section("3. FAQ MATCHING (Customer Service)")

    faq_questions = list(FAQ_DATABASE.keys())
    faq_embs = encode_texts(model, faq_questions)

    correct = 0
    total = len(FAQ_USER_QUERIES)

    for user_query, expected_faq in FAQ_USER_QUERIES:
        q_emb = encode_texts(model, [user_query])
        sims = [cosine_similarity(q_emb[0], faq_embs[i]) for i in range(len(faq_questions))]
        best_idx = int(np.argmax(sims))
        best_faq = faq_questions[best_idx]
        best_score = sims[best_idx]

        is_correct = best_faq == expected_faq
        correct += int(is_correct)

        marker = "✅" if is_correct else "❌"
        print(f"    {marker} [{best_score:.4f}] \"{user_query}\"")
        print(f"       → {best_faq[:65]}")
        if not is_correct:
            print(f"       ✗ Expected: {expected_faq[:65]}")

    acc = correct / total
    print(f"\n  📊 FAQ MATCHING RESULTS:")
    print(f"     Accuracy: {acc:.1%} ({correct}/{total})")

    return {"faq_accuracy": acc}


# ============================================================================
# 4. INTENT CLASSIFICATION
# ============================================================================

INTENT_TEMPLATES = {
    "pembelian": [
        "Saya mau beli produk ini",
        "Cara order barang",
        "Mau checkout pesanan saya",
        "Tambahkan ke keranjang",
        "Saya ingin membeli 2 unit",
    ],
    "pembayaran": [
        "Mau bayar pesanan",
        "Cara transfer pembayaran",
        "Metode pembayaran apa saja?",
        "Bisa bayar pakai e-wallet?",
        "Invoice saya mana?",
    ],
    "pengiriman": [
        "Kapan barang saya sampai?",
        "Cek status pengiriman",
        "Paket belum diterima",
        "Nomor resi pengiriman",
        "Kenapa pengiriman lama?",
    ],
    "pengembalian": [
        "Mau return barang",
        "Produk tidak sesuai deskripsi",
        "Barang rusak saat diterima",
        "Minta refund dong",
        "Cara komplain barang cacat",
    ],
    "informasi_produk": [
        "Spesifikasi produk ini apa?",
        "Ukuran yang tersedia apa saja?",
        "Warna apa yang ready?",
        "Bahan material produk ini?",
        "Berat produk berapa gram?",
    ],
}

INTENT_TEST_QUERIES = [
    ("Saya mau pesan 3 buah", "pembelian"),
    ("Gimana cara bayar pake gopay?", "pembayaran"),
    ("Paket sudah dikirim belum?", "pengiriman"),
    ("Barang salah kirim, mau tukar", "pengembalian"),
    ("Produk ini tahan air ga?", "informasi_produk"),
    ("Tambah ke cart dong", "pembelian"),
    ("Tagihan order saya berapa?", "pembayaran"),
    ("Estimasi sampai berapa hari?", "pengiriman"),
    ("Mau kembalikan produk cacat", "pengembalian"),
    ("Ada size XL ga?", "informasi_produk"),
    ("Order sekarang bisa?", "pembelian"),
    ("Mau transfer via BCA", "pembayaran"),
    ("Resi JNE saya berapa?", "pengiriman"),
    ("Refund prosesnya berapa lama?", "pengembalian"),
    ("Laptop ini RAM berapa GB?", "informasi_produk"),
]


def eval_intent_classification(model) -> Dict:
    """Evaluasi intent classification berbasis embedding similarity."""
    print_section("4. INTENT CLASSIFICATION")

    # Create intent centroids
    intent_centroids = {}
    for intent, examples in INTENT_TEMPLATES.items():
        embs = encode_texts(model, examples)
        intent_centroids[intent] = np.mean(embs, axis=0)

    correct = 0
    total = len(INTENT_TEST_QUERIES)

    for query, expected_intent in INTENT_TEST_QUERIES:
        q_emb = encode_texts(model, [query])[0]

        # Find closest intent
        best_intent = None
        best_sim = -1
        for intent, centroid in intent_centroids.items():
            sim = cosine_similarity(q_emb, centroid)
            if sim > best_sim:
                best_sim = sim
                best_intent = intent

        is_correct = best_intent == expected_intent
        correct += int(is_correct)

        marker = "✅" if is_correct else "❌"
        print(f"    {marker} [{best_sim:.4f}] \"{query}\"")
        print(f"       → {best_intent}" + (f" (expected: {expected_intent})" if not is_correct else ""))

    acc = correct / total
    print(f"\n  📊 INTENT CLASSIFICATION RESULTS:")
    print(f"     Accuracy: {acc:.1%} ({correct}/{total})")

    return {"intent_accuracy": acc}


# ============================================================================
# 5. CROSS-LINGUAL SIMILARITY (EN ↔ ID)
# ============================================================================

CROSSLINGUAL_PAIRS = [
    # High similarity (same meaning)
    ("How to cook fried rice", "Cara memasak nasi goreng", 1),
    ("What is artificial intelligence?", "Apa itu kecerdasan buatan?", 1),
    ("The weather is very hot today", "Cuaca hari ini sangat panas", 1),
    ("I want to book a hotel room", "Saya ingin memesan kamar hotel", 1),
    ("Where is the nearest hospital?", "Dimana rumah sakit terdekat?", 1),
    ("How much does this cost?", "Berapa harganya?", 1),
    ("Please send me the document", "Tolong kirimkan dokumennya", 1),
    ("The meeting is at 3 PM", "Rapat jam 3 sore", 1),
    ("Thank you very much", "Terima kasih banyak", 1),
    ("I need help with my computer", "Saya butuh bantuan dengan komputer saya", 1),
    # Low similarity (different meaning)
    ("I love playing basketball", "Saya suka memasak rendang", 0),
    ("The stock market crashed today", "Hari ini cuaca sangat cerah", 0),
    ("My car needs new tires", "Kucing saya sakit perlu ke dokter hewan", 0),
    ("The concert starts at 8 PM", "Harga beras naik lagi bulan ini", 0),
    ("I finished reading the book", "Dia baru saja lulus dari universitas", 0),
]


def eval_crosslingual(model, threshold=0.4) -> Dict:
    """Evaluasi cross-lingual similarity EN↔ID."""
    print_section("5. CROSS-LINGUAL (English ↔ Indonesian)")

    predictions = []
    labels = []
    sims_list = []

    for en, id_text, label in CROSSLINGUAL_PAIRS:
        embs = encode_texts(model, [en, id_text])
        sim = cosine_similarity(embs[0], embs[1])
        pred = 1 if sim >= threshold else 0

        predictions.append(pred)
        labels.append(label)
        sims_list.append(sim)

        marker = "✅" if pred == label else "❌"
        label_str = "SAME" if label == 1 else "DIFF"
        print(f"    {marker} [{sim:.4f}] [{label_str}] EN: {en[:30]:30s} ↔ ID: {id_text[:30]}")

    acc = accuracy_score(labels, predictions)
    pos_sims = [s for s, l in zip(sims_list, labels) if l == 1]
    neg_sims = [s for s, l in zip(sims_list, labels) if l == 0]

    print(f"\n  📊 CROSS-LINGUAL RESULTS (threshold={threshold}):")
    print(f"     Accuracy: {acc:.1%}")
    print(f"     Avg similarity (same meaning):  {np.mean(pos_sims):.4f}")
    print(f"     Avg similarity (diff meaning):  {np.mean(neg_sims):.4f}")

    return {"crosslingual_accuracy": acc}


# ============================================================================
# 6. DUPLICATE QUESTION DETECTION
# ============================================================================

DUPLICATE_QUESTIONS = [
    # True duplicates
    ("Apa itu machine learning?", "Machine learning itu apa sih?", 1),
    ("Cara membuat KTP baru", "Bagaimana prosedur pembuatan KTP?", 1),
    ("Kenapa internet lemot?", "Mengapa koneksi internet saya lambat?", 1),
    ("Kapan CPNS dibuka?", "Pendaftaran CPNS kapan ya?", 1),
    ("Berapa gaji UMR Jakarta?", "UMR Jakarta berapa per bulan?", 1),
    ("Cara nego harga rumah", "Tips menawar harga properti", 1),
    ("Syarat bikin SIM A", "Persyaratan pembuatan SIM mobil", 1),
    ("HP murah RAM besar", "Handphone budget dengan RAM banyak", 1),
    ("Cara menghilangkan jerawat", "Tips mengatasi jerawat membandel", 1),
    ("Penyebab rambut rontok", "Kenapa rambut saya rontok terus?", 1),
    # Not duplicates (same topic but different questions)
    ("Apa itu kredit motor?", "Berapa cicilan motor per bulan?", 0),
    ("Kapan musim durian?", "Dimana beli durian enak?", 0),
    ("Siapa presiden pertama RI?", "Kapan Indonesia merdeka?", 0),
    ("Cara masak rendang", "Berapa lama rendang tahan?", 0),
    ("Apa gejala COVID?", "Vaksin COVID yang bagus apa?", 0),
    ("Cara belajar gitar", "Gitar akustik yang bagus merk apa?", 0),
    ("Penyebab banjir Jakarta", "Cara mengungsi saat banjir", 0),
    ("Apa itu cryptocurrency?", "Cara beli Bitcoin di Indonesia", 0),
    ("Manfaat minum air putih", "Berapa harga air mineral?", 0),
    ("Cara daftar ojek online", "Berapa penghasilan driver ojol?", 0),
]


def eval_duplicate_detection(model, threshold=0.5) -> Dict:
    """Evaluasi duplicate question detection."""
    print_section("6. DUPLICATE QUESTION DETECTION")

    predictions = []
    labels = []

    for q1, q2, label in DUPLICATE_QUESTIONS:
        embs = encode_texts(model, [q1, q2])
        sim = cosine_similarity(embs[0], embs[1])
        pred = 1 if sim >= threshold else 0

        predictions.append(pred)
        labels.append(label)

        marker = "✅" if pred == label else "❌"
        label_str = "DUP " if label == 1 else "UNIQ"
        print(f"    {marker} [{sim:.4f}] [{label_str}] {q1[:30]:30s} ↔ {q2[:30]}")

    acc = accuracy_score(labels, predictions)
    f1 = f1_score(labels, predictions, average="binary")

    print(f"\n  📊 DUPLICATE DETECTION RESULTS (threshold={threshold}):")
    print(f"     Accuracy: {acc:.1%}")
    print(f"     F1 Score: {f1:.4f}")

    return {"dup_accuracy": acc, "dup_f1": f1}


# ============================================================================
# 7. SHORT ↔ LONG TEXT MATCHING
# ============================================================================

SHORT_LONG_PAIRS = [
    {
        "short": "vaksin COVID anak",
        "long_pos": "Vaksinasi COVID-19 untuk anak usia 6-11 tahun sudah diperbolehkan oleh BPOM. Jenis vaksin yang digunakan adalah Sinovac dengan dosis 3 mcg. Orang tua dapat mendaftarkan anak ke puskesmas atau rumah sakit terdekat untuk mendapatkan vaksinasi.",
        "long_neg": "Resep masakan rendang Padang yang autentik memerlukan daging sapi berkualitas, santan kental, dan berbagai rempah seperti kunyit, lengkuas, serai, dan cabai merah.",
    },
    {
        "short": "inflasi Indonesia 2024",
        "long_pos": "Bank Indonesia melaporkan tingkat inflasi tahun 2024 berada di kisaran 2.5-3.0 persen, sesuai dengan target sasaran yang ditetapkan. Inflasi inti tetap terkendali berkat kebijakan moneter yang prudent dan koordinasi dengan pemerintah.",
        "long_neg": "Pembangunan infrastruktur jalan tol Trans-Jawa telah menghubungkan berbagai kota di pulau Jawa mulai dari Merak hingga Surabaya, memudahkan mobilitas masyarakat.",
    },
    {
        "short": "tips wawancara kerja",
        "long_pos": "Persiapan interview kerja meliputi riset perusahaan, latihan menjawab pertanyaan umum, berpakaian profesional, datang tepat waktu, dan menyiapkan pertanyaan untuk pewawancara. Jangan lupa membawa CV dan portofolio.",
        "long_neg": "Gunung Bromo terletak di Jawa Timur dan merupakan salah satu destinasi wisata alam paling populer di Indonesia. Wisatawan dapat menikmati sunrise dari puncak Penanjakan.",
    },
    {
        "short": "penyakit jantung koroner",
        "long_pos": "Penyakit jantung koroner (PJK) terjadi ketika pembuluh darah koroner mengalami penyempitan akibat penumpukan plak aterosklerosis. Gejala utama meliputi nyeri dada, sesak napas, dan kelelahan. Faktor risiko termasuk merokok, hipertensi, diabetes, dan kolesterol tinggi.",
        "long_neg": "Teknik budidaya ikan lele di kolam terpal cukup mudah dan tidak membutuhkan lahan luas. Persiapan kolam meliputi pemilihan lokasi, pemasangan terpal, dan pengisian air yang sudah diendapkan selama 3 hari.",
    },
    {
        "short": "hukum waris Indonesia",
        "long_pos": "Hukum waris di Indonesia diatur dalam tiga sistem: hukum waris adat, hukum waris Islam (faraid), dan hukum waris perdata (BW). Pembagian warisan tergantung pada agama dan adat istiadat ahli waris. Pengadilan Agama berwenang menyelesaikan sengketa waris bagi Muslim.",
        "long_neg": "Teknik fotografi landscape memerlukan pemahaman tentang komposisi, pencahayaan, dan penggunaan tripod untuk menghasilkan gambar yang tajam. Golden hour adalah waktu terbaik untuk memotret pemandangan.",
    },
]


def eval_short_long_matching(model) -> Dict:
    """Evaluasi kemampuan matching teks pendek dengan teks panjang."""
    print_section("7. SHORT ↔ LONG TEXT MATCHING")

    correct = 0
    total = len(SHORT_LONG_PAIRS)

    for test in SHORT_LONG_PAIRS:
        short = test["short"]
        long_pos = test["long_pos"]
        long_neg = test["long_neg"]

        embs = encode_texts(model, [short, long_pos, long_neg])
        sim_pos = cosine_similarity(embs[0], embs[1])
        sim_neg = cosine_similarity(embs[0], embs[2])

        is_correct = sim_pos > sim_neg
        correct += int(is_correct)

        marker = "✅" if is_correct else "❌"
        print(f"    {marker} Query: \"{short}\"")
        print(f"       Positive: [{sim_pos:.4f}] {long_pos[:65]}...")
        print(f"       Negative: [{sim_neg:.4f}] {long_neg[:65]}...")

    acc = correct / total
    print(f"\n  📊 SHORT↔LONG MATCHING RESULTS:")
    print(f"     Accuracy: {acc:.1%} ({correct}/{total})")

    return {"short_long_accuracy": acc}


# ============================================================================
# 8. SENTIMENT-AWARE SIMILARITY
# ============================================================================

SENTIMENT_TESTS = [
    # Produk positif harus lebih mirip sesama positif
    {
        "anchor": "Produk ini sangat bagus, saya puas sekali",
        "positive": "Barangnya berkualitas tinggi, recommended banget",
        "negative": "Barang jelek, menyesal beli, kapok!",
    },
    {
        "anchor": "Pelayanan rumah sakit ini ramah dan profesional",
        "positive": "Dokter dan perawatnya baik, penanganan cepat",
        "negative": "Pelayanan buruk, antri lama, perawat judes",
    },
    {
        "anchor": "Makanannya enak banget, porsi banyak, harga terjangkau",
        "positive": "Resto ini top markotop, murah meriah dan lezat",
        "negative": "Makanan hambar, porsi sedikit, kemahalan",
    },
    {
        "anchor": "Aplikasi ini user-friendly, fitur lengkap, jarang error",
        "positive": "App-nya mantap, gampang dipake, stabil banget",
        "negative": "Aplikasi sering crash, ribet, banyak bug",
    },
    {
        "anchor": "Guru di sekolah ini sangat kompeten dan perhatian",
        "positive": "Pengajarnya berkualitas, metode mengajar efektif",
        "negative": "Guru-gurunya tidak peduli, cara mengajarnya membosankan",
    },
]


def eval_sentiment_similarity(model) -> Dict:
    """Evaluasi apakah model bisa membedakan sentimen."""
    print_section("8. SENTIMENT-AWARE SIMILARITY")

    correct = 0
    total = len(SENTIMENT_TESTS)

    for test in SENTIMENT_TESTS:
        anchor = test["anchor"]
        positive = test["positive"]
        negative = test["negative"]

        embs = encode_texts(model, [anchor, positive, negative])
        sim_pos = cosine_similarity(embs[0], embs[1])
        sim_neg = cosine_similarity(embs[0], embs[2])

        is_correct = sim_pos > sim_neg
        correct += int(is_correct)

        marker = "✅" if is_correct else "❌"
        print(f"    {marker} Anchor: \"{anchor[:55]}...\"")
        print(f"       Positive: [{sim_pos:.4f}] {positive[:55]}...")
        print(f"       Negative: [{sim_neg:.4f}] {negative[:55]}...")

    acc = correct / total
    print(f"\n  📊 SENTIMENT SIMILARITY RESULTS:")
    print(f"     Accuracy: {acc:.1%} ({correct}/{total})")

    return {"sentiment_accuracy": acc}


# ============================================================================
# MAIN — Run All Evaluations
# ============================================================================

def run_all_evaluations(model_path: str):
    """Jalankan semua evaluasi dan tampilkan ringkasan."""

    print("\n" + "█"*80)
    print("█  REAL-WORLD EVALUATION — T5 Gemma 2 Indonesian Sentence Embeddings")
    print("█"*80)
    print(f"\n  Model: {model_path}")

    # Load model
    logger.info(f"Loading model: {model_path}")
    model = SentenceTransformer(model_path)
    dim = model.get_embedding_dimension()
    print(f"  Embedding Dimension: {dim}")

    if torch.cuda.is_available():
        gpu = torch.cuda.get_device_name(0)
        mem = torch.cuda.get_device_properties(0).total_memory / 1024**3
        print(f"  GPU: {gpu} ({mem:.1f} GB)")

    # Run all evaluations
    results = {}
    results.update(eval_semantic_search(model))
    results.update(eval_paraphrase_detection(model))
    results.update(eval_faq_matching(model))
    results.update(eval_intent_classification(model))
    results.update(eval_crosslingual(model))
    results.update(eval_duplicate_detection(model))
    results.update(eval_short_long_matching(model))
    results.update(eval_sentiment_similarity(model))

    # ── Final Summary ──
    print("\n" + "█"*80)
    print("█  FINAL SUMMARY")
    print("█"*80)
    print()

    summary_items = [
        ("Semantic Search (Top-1)",   results["search_top1_acc"]),
        ("Semantic Search (Top-3)",   results["search_top3_acc"]),
        ("Paraphrase Detection",      results["para_accuracy"]),
        ("Paraphrase F1",             results["para_f1"]),
        ("FAQ Matching",              results["faq_accuracy"]),
        ("Intent Classification",     results["intent_accuracy"]),
        ("Cross-lingual (EN↔ID)",     results["crosslingual_accuracy"]),
        ("Duplicate Detection",       results["dup_accuracy"]),
        ("Duplicate F1",              results["dup_f1"]),
        ("Short↔Long Matching",       results["short_long_accuracy"]),
        ("Sentiment Similarity",      results["sentiment_accuracy"]),
    ]

    total_score = 0
    total_count = 0

    for name, score in summary_items:
        bar_len = int(score * 30)
        bar = "█" * bar_len + "░" * (30 - bar_len)
        grade = "🟢" if score >= 0.8 else ("🟡" if score >= 0.6 else "🔴")
        print(f"    {grade} {name:30s} {bar} {score:.1%}")
        total_score += score
        total_count += 1

    avg_score = total_score / total_count
    print(f"\n    {'─'*65}")
    grade = "🟢" if avg_score >= 0.8 else ("🟡" if avg_score >= 0.6 else "🔴")
    print(f"    {grade} {'OVERALL AVERAGE':30s} {'':30s}  {avg_score:.1%}")

    # Total test cases
    total_cases = (
        len(SEMANTIC_SEARCH_TESTS) +
        len(PARAPHRASE_PAIRS) +
        len(FAQ_USER_QUERIES) +
        len(INTENT_TEST_QUERIES) +
        len(CROSSLINGUAL_PAIRS) +
        len(DUPLICATE_QUESTIONS) +
        len(SHORT_LONG_PAIRS) +
        len(SENTIMENT_TESTS)
    )
    print(f"\n    Total test cases evaluated: {total_cases}")
    print()

    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Real-World Evaluation for Indonesian Embeddings")
    parser.add_argument(
        "--model",
        type=str,
        default="t5gemma2-embedding-v1/final",
        help="Path to model (default: t5gemma2-embedding-v1/final)",
    )
    args = parser.parse_args()

    results = run_all_evaluations(args.model)
