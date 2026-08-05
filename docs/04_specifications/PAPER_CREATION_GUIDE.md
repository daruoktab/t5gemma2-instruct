# Pedoman Pembuatan Paper T5-Gemma-2 Instruct

Dokumen ini berfungsi sebagai pedoman standar (guidelines) untuk penulisan, modifikasi, dan kompilasi makalah ilmiah (paper) di dalam repositori ini menggunakan format **JACoW (Joint Accelerator Conferences Website)** berbasis **Typst** dan diagram **CeTZ**.

---

## 1. Prinsip Utama Pengisian Konten

> [!IMPORTANT]
> **Hanya Tulis Informasi Berdasarkan Keinginan & Data Eksperimen Riil**
> - **DILARANG KERAS** melakukan pencarian (web search) secara random lalu memasukkan teori atau angka fiktif ke dalam paper. Hal ini akan menyulitkan proses koreksi data.
> - Setiap perubahan metrik evaluasi, arsitektur model, dan pipeline pelatihan harus dikonfirmasi langsung dengan data riil hasil training.

---

## 2. Struktur Dokumen JACoW (Typst)

Gunakan template `@preview/accelerated-jacow:0.14.1` dengan struktur dasar berikut:

```typst
#import "@preview/accelerated-jacow:0.14.1": jacow, jacow-table
#import "@preview/cetz:0.3.4": canvas, draw

#show: jacow.with(
  title: [Judul Paper dalam Bahasa Indonesia/Inggris],
  authors: (
    (name: "Nama Penulis", at: "affil_id", email: "email@domain.com"),
  ),
  affiliations: (
    affil_id: "Nama Institusi, Kota, Negara",
  ),
  funding: "Penelitian ini didukung secara mandiri.",
  abstract: [
    Ringkasan singkat makalah (maksimal 200 kata) yang mencakup latar belakang, metodologi, hasil utama (ROUGE, BERTScore), dan kesimpulan.
  ],
)
```

### Aturan Formatting Halaman & Kolom
Untuk memastikan paper pas dengan batas halaman (misalnya tepat 3 halaman):
- Atur spacing gambar dan heading secara ketat:
  ```typst
  #show figure: set block(spacing: 3pt)
  #show heading: set block(above: 6pt, below: 2pt)
  #show bibliography: set text(size: 6.8pt)
  ```
- Gunakan font **Times New Roman** dengan rata kanan-kiri (justify):
  ```typst
  #set par(justify: true, leading: 0.49em)
  #show text: set text(font: "Times New Roman")
  ```

---

## 3. Standardisasi Diagram CeTZ

Agar diagram alur dan arsitektur terlihat profesional, rapi, dan seimbang (tidak menggantung/hancur), ikuti aturan berikut:

### A. Gunakan Model Node Berbasis Center (Pusat)
Jangan mendefinisikan posisi koordinat sudut kiri atas secara manual. Gunakan fungsi pembantu `nd` yang menaruh posisi box tepat pada titik tengah `(cx, cy)` agar semua elemen sejajar pada sumbu horizontal/vertikal:

```typst
let nd(cx, cy, w, h, lbl, fc, sc) = {
  rect((cx - w/2, cy + h/2), (cx + w/2, cy - h/2), fill: fc, stroke: 1.2pt + sc, radius: 3pt)
  content((cx, cy), text(7pt, align(center + horizon, lbl)))
}
```

### B. Hubungkan Panah antar Node Tepat di Tengah
Panah penunjuk arah (`arh`) harus menghubungkan sisi luar box secara presisi. Hitung koordinat x/y berdasarkan lebar (`w`) dan tinggi (`h`) node:

```typst
let arh(x1, x2, y) = line(
  (x1, y),
  (x2, y),
  mark: (end: "stealth", fill: rgb("#1f2937"), size: 5pt),
  stroke: 1.2pt + rgb("#4b5563"),
)
```

### C. Contoh Implementasi Rantai Blok Horizontal (Figure 1)
```typst
nd(30, 0, 60, 32, [*Masukan*], rgb("#fef9c3"), rgb("#b45309"))
nd(110, 0, 70, 32, [*Encoder*], rgb("#dbeafe"), rgb("#2563eb"))
nd(205, 0, 80, 48, [*Merged Attn*], rgb("#f3e8ff"), rgb("#7c3aed")) // Taller but centered!
nd(290, 0, 60, 32, [*Keluaran*], rgb("#dcfce7"), rgb("#16a34a"))

// Panah berada tepat di sumbu tengah y=0
arh(60, 75, 0)
arh(145, 165, 0)
arh(245, 260, 0)
```

---

## 4. Alur Kerja Kompilasi & Analisis Layout

Karena path kompilasi CLI di sistem ini bersifat spesifik, ikuti perintah berikut untuk melakukan build dan peninjauan:

### A. Kompilasi Typst ke PDF
Gunakan executable `typst` dari absolute path WinGet Links:
```powershell
& "C:\Users\daru\AppData\Local\Microsoft\WinGet\Links\typst.exe" compile accelerated-jacow/paper.typ accelerated-jacow/paper.pdf
```

### B. Validasi Layout & Visual PDF
- Setelah kompilasi berhasil, gunakan analyze tool (seperti `view_file` pada agent) untuk memeriksa file PDF secara langsung.
- Perhatikan keselarasan teks di batas kolom kiri dan kanan.
- Periksa diagram untuk memastikan tidak ada teks yang keluar dari garis kotak (overlapping) atau garis panah yang terputus.

---

## 5. Referensi & Tabel Data

### Penulisan Tabel
Gunakan helper `jacow-table` agar format tabel otomatis menyesuaikan regulasi JACoW:
```typst
#figure(
  jacow-table(
    "ccccccc", // Format kolom (center/left)
    [Step], [Loss], [PPL], [R-1], [R-L], [BLEU], [BERT],
    [200], [2.958], [19.27], [53.38], [49.05], [12.57], [82.80],
    [1000], [2.865], [17.54], [*60.07*], [*55.51*], [*15.94*], [*85.23*],
  ),
  caption: [Metrik evaluasi model SFT teks (%).],
) <table:results>
```

### Sitasi & Bibliografi
- Letakkan daftar referensi pada file `references.bib`.
- Gunakan `@label` untuk melakukan sitasi langsung di dalam teks.
- Pastikan modul bibliografi dipanggil di akhir halaman:
  ```typst
  #bibliography("references.bib")
  ```
