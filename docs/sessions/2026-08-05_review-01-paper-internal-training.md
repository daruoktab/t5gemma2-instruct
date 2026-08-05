# 📄 Review 01 — Paper Level Internal Training (Optimizer, Loss, Steering)

**Tanggal review:** 2026-08-05 · **Sumber:** arXiv API (diverifikasi eksis)
**Relevansi:** Pipeline T5Gemma-2 4B chatbot Indonesia (v7) — Phase 0.5 steering,
Phase 1 joint SFT, Phase 2 joint ORPO, optimizer GrokMuonAdEMA.

> 📂 File ini bagian dari `docs/sessions/2026-08-05_*` — lihat
> [`2026-08-05_INDEX.md`](2026-08-05_INDEX.md) untuk daftar lengkap.

---

## 1. OrScale — Orthogonalised Optimization with Layer-Wise Trust-Ratio Scaling
**arXiv:** [2605.07815](https://arxiv.org/abs/2605.07815) · Mei 2026

**Esensi abstract:** Muon meng-orthogonalisasi update matriks tapi magnitude per layer
dikontrol hampir seluruhnya oleh global LR. OrScale = ekstensi trust-ratio: *denominator
rasio per-layer harus mengukur Frobenius norm dari arah parameter-space aktual yang
diaplikasikan*. OrScale-LM: Moonlight shape scaling + kalibrasi sekali per layer sehingga
semua trust ratio mulai dari 1. Menganalisis **kenapa 3 varian Muon-LAMB hybrid gagal**:
(1) shape-degenerate denominators, (2) raw-momentum clip saturation, (3) **decoupled
weight-decay runaway**. Solusi: real-update-direction denominator **dengan coupled
weight decay**. Ada jaminan konvergensi O(1/√T) nonconvex.

**Implikasi untuk GrokMuonAdEMA:**
- MuonClip global (`MUON_MAX_GRAD_NORM=1.0`) memotong *gradien hasil filter*, bukan
  update aktual → persis zona "raw-momentum clip saturation" yang disebut paper.
- Pipeline memakai **decoupled weight decay** → paper menyebut ini salah satu failure mode.
- Sketsa adaptasi (eksperimen, bukan implementasi OrScale utuh):

```python
# OrScale-inspired: ganti MuonClip global dengan trust-ratio per layer.
# Prinsip: ukur ||update aktual (g_ortho)|| terhadap ||param||, lalu skala agar
# tiap layer bergerak dengan rasio yang dikalibrasi SEKALI di step pertama.
# (di __init__: simpan mapping param -> layer key dari group name)
# state per-param tambahan:
#   state["calib_ratio"] = None   # diisi step pertama
#   state["target_trust"] = 1.0   # "every trust ratio starts at one"

# di cabang 2D, ganti blok:
#   p.data.add_(g_ortho.to(p.dtype), alpha=-lr)
# dengan:
if state["calib_ratio"] is None:
    state["calib_ratio"] = (g_ortho.norm() / (p.data.norm() + 1e-8)).item()
    _trust = 1.0
else:
    _obs = (g_ortho.norm() / (p.data.norm() + 1e-8)).item()
    _trust = state["target_trust"] / (_obs + 1e-8)   # normalisasi ke target
p.data.add_(g_ortho.to(p.dtype), alpha=-lr * _trust)

# ⚠️ Catatan: coba juga coupled WD (wd langsung di update, bukan decay p.data)
# — paper menunjukkan ini menghindari runaway. Eksperimen A/B: decoupled vs coupled.
```

> ⚠️ Ini adaptasi *inspired-by*, bukan implementasi OrScale penuh. Jalankan sebagai flag
> eksperimen (`ORSCALE_MODE`) dan A/B-kan dulu di run kecil.

---

## 2. TLPO — Token-Level Policy Optimization (Language Confusion)
**arXiv:** [2604.26553](https://arxiv.org/abs/2604.26553) · April 2026

**Esensi abstract:** LLM multilingual sering gagal konsisten berbahasa sesuai intent
(language confusion). Pendekatan sequence-level (DPO, ORPO, GRPO) *"can lead to unintended
degradation of general model capabilities"* → motivasi pendekatan token-level. TLPO:
identifikasi posisi rawan error → eksplorasi kandidat token alternatif → update policy
dengan objective yang menekan output pemicu error **di level token**, tanpa mengorbankan
kemampuan umum.

**Implikasi:** `get_batch_logps` di pipeline menghitung rata-rata **sequence-level**.
Sketsa adaptasi — bobot token ∝ divergensi chosen/rejected (posisi error-prone):

```python
def get_batch_logps_tlpo(self, logits, labels, contrast_logits=None,
                         contrast_labels=None, focus_strength=2.0,
                         average_log_prob=True):
    """TLPO-inspired: bobot token ∝ |Δlogp| di posisi chosen/rejected divergen.
    Posisi error-prone (selisih log-prob besar) berbobot lebih — menekan output
    pemicu error secara granular tanpa menyentuh token lain."""
    labels = labels.clone()
    mask = labels != -100
    labels[labels == -100] = 0
    lps = torch.gather(logits.log_softmax(-1), dim=2,
                       index=labels.unsqueeze(2)).squeeze(2)
    if contrast_logits is not None:
        clabels = contrast_labels.clone()
        cmask = clabels != -100
        clabels[clabels == -100] = 0
        clps = torch.gather(contrast_logits.log_softmax(-1), dim=2,
                            index=clabels.unsqueeze(2)).squeeze(2)
        divergence = (lps - clps).abs() * mask.float()
        w = (divergence + 1e-6) ** focus_strength
        w = w * mask.float()
        if average_log_prob:
            return (lps * w).sum(-1) / w.sum(-1).clamp(min=1)
    return (lps * mask).sum(-1) / mask.sum(-1).clamp(min=1)

# di compute_loss: pass logits rejected sebagai contrast
# clp = self.get_batch_logps_tlpo(co.logits, cl, contrast_logits=ro.logits, contrast_labels=rl)
```

> ⚠️ Ini mengubah definisi loss ORPO (odds-ratio tidak lagi murni sequence-level).
> Validasi empiris wajib: BLEU/ROUGE/BERTScore & kualitas bahasa harus naik, bukan cuma loss.

---

## 3. When Does Muon Help Agentic Reinforcement Learning?
**arXiv:** [2607.16169](https://arxiv.org/abs/2607.16169) · Juli 2026 (v4)

**Esensi abstract:** Muon kompetitif dengan AdamW di pre-training skala besar, tapi regime
di RL post-training belum jelas. Dipetakan di ALFWorld (sparse-reward agentic), 3 objective,
Qwen2.5 0.5B–3B. Temuan: AdamW merespons non-monotonik terhadap rate; **fan-in Muon stabil
di effective step lebih agresif**: di `3e-5` memperbaiki late success vs AdamW `1e-6`.
Tapi: tuned AdamW hampir menyamai high-rate Muon di 3B, dan **RMS-matched control
menghilangkan gain** — high-rate Muon mengaplikasikan `3.53×` update RMS AdamW.

**Implikasi:** Keunggulan Muon di post-training ≈ kemampuan beroperasi di effective step
lebih besar, bukan sihir algoritma.
- `ORPO_MUON_LR_SCALE=5.0` → decoder ORPO LR ≈ `5e-6 × 1.0 × 5 = 2.5e-5` → **di sweet spot
  paper (≈3e-5)** ✅
- Rekomendasi: ukur **update RMS aktual** branch Muon vs AdamW (log `g_ortho.norm()`
  per step); kalau rasio ≈3.5×, konsisten dengan paper. Kalau ORPO tidak stabil,
  turunkan scale — bukan ganti optimizer.
- Catatan: domain paper = agentic RL sparse-reward; untuk chatbot ini **indikasi**, bukan hukum.

---

## 4. Decomposing Task Vectors for Refined Model Editing
**arXiv:** [2512.22511](https://arxiv.org/abs/2512.22511) · Desember 2025

**Esensi abstract:** Task vector (selisih fine-tuned − pretrained) untuk steering perilaku,
tapi vektor sering mengandung konsep tumpang-tindih yang saling interferensi saat aritmetika.
Solusi: dekomposisi prinsipil — komponen **shared knowledge** (lintas task vector) vs
komponen **task-unique** (invarian per task), lewat invariant subspaces.

**Implikasi:** Δ = Gemma3-IT − Gemma3-Base (Phase 0.5) adalah vektor campuran.
Ide: dekomposisi dulu, steer hanya komponen task-unique → interferensi dengan UL2
pre-training T5Gemma-2 lebih kecil.

---

## 5. Task Vector Geometry Underlies Dual Modes of Task Inference
**arXiv:** [2605.03780](https://arxiv.org/abs/2605.03780) · Mei 2026

**Esensi abstract:** Task-specific directions (task vectors) hidup di **middle-layer
representations**. Dua mode inferensi bisa hidup bersama: in-distribution (Bayesian task
retrieval, convex combination dari task direction) vs OOD (adaptasi ke task baru).
Geometri dibentuk distribusi training.

**Implikasi:** ✅ **Validasi desain steering**: α_FFN_MID=0.25 di 25–80% depth persis
di zona middle-layer yang paper identifikasi paling bermakna untuk task inference.

---

## 6. First-Order Predictable but Pairwise Fragile
**arXiv:** [2607.16821](https://arxiv.org/abs/2607.16821) · Juli 2026

**Esensi abstract:** Task arithmetic, sequential fine-tuning, activation steering bekerja
lewat perturbasi kecil di sekitar checkpoint. Diukur 8 properti pada 9 transformer (82M–7B):
satu arah punya **validity window hingga skala ~1e-2**; **tidak ada radius universal untuk
komposisi berpasangan** atau urutan update; efek perturbasi terhadap loss ≈ proyeksinya
di sepanjang arah tersebut.

**Implikasi (⚠️):**
- α=0.25 (FFN mid) kemungkinan **di luar window first-order** (1e-2) — perlu validasi
  empiris dengan probe-loss sebelum/sesudah steering.
- Pipeline mengkomposisikan 4 update berurutan (steer → graft → SFT → ORPO) — paper
  memperingatkan interferensi komposisi. Monitor probe-loss per fase (eval set yang sama).

---

## 7. Recoverable but Not Stationary: Local Linear Structures
**arXiv:** [2606.10929](https://arxiv.org/abs/2606.10929) · Juni 2026

**Esensi abstract:** Task vectors, LoRA, activation steering mengindikasikan perilaku
dikontrol arah linear. Temuan: struktur task-gradient lokal low-rank kuat, tapi **hipotesis
fixed-task-plane ditolak**: basis statis meleset dari arah recovery; basis berguna
**drift substansial dalam ~100 step**. Trajectory-prefix basis menangkap 77% displacement
recovery LoRA.

**Implikasi:** Arah task vector **tidak statis** → steering sekali di awal lalu SFT 2 epoch
bisa membuat arah steering basi. **Ide eksperimen:** re-steer ringan (α kecil) *setelah*
SFT, sebelum ORPO — sinkronkan arah IT dengan state pasca-SFT.

---

## 8. Objective Matters: Fine-Tuning Objectives Shape Safety
**arXiv:** [2601.12639](https://arxiv.org/abs/2601.12639) · Januari 2026

**Esensi abstract:** Perbandingan terkontrol 6 objective (SFT, DPO, Conditional FT,
Inoculation Prompting, ORPO, KL-regularized FT) dengan data/domain/arsitektur/optimizer
tetap. Di budget kecil: robustness serupa, capability beda. Di budget besar: **SFT &
preference-based tuning mengikat gain capability dengan peningkatan kerentanan adversarial
dan persona drift**; objective yang meng-constrain sinyal belajar (terutama KL-regularized)
lebih aman.

**Implikasi:** Pipeline = SFT lalu ORPO — dua objective yang paper kaitkan dengan persona
drift di budget besar. Aksi: tambah **KL-regularization** kecil di Phase 2, dan **monitor
drift persona** — bandingkan output pasca-SFT vs pasca-ORPO pada prompt sama (eval set
sudah tersedia di `VisionSampleGenerationCallback`).

---

## 9. Mechanistic Analysis of Alignment Algorithms
**arXiv:** [2606.09850](https://arxiv.org/abs/2606.09850) · Juni 2026

**Esensi abstract:** Analisis mekanistik 6 metode (PPO, DPO, SimPO, ORPO, GRPO, KTO) pada
3 keluarga model: linear probing per layer + SAE + crosscoders. Temuan: sinyal preferensi
konsisten di **early-mid atau mid-late layers**; **DPO & ORPO menurunkan separability**
lewat rotasi geometri non-konstruktif & feature attenuation; KTO/GRPO **meningkatkan**
separability (feature sharing konstruktif); PPO/SimPO mempertahankan geometri baseline.

**Implikasi:** ORPO (Phase 2 saat ini) adalah salah satu yang paling "merusak geometri".
Bukan berarti gagal — eval SFT-vs-ORPO apple-to-apple tetap cara pembuktian yang benar.
Untuk iterasi berikutnya: pertimbangkan **SimPO** (preserve geometry, tanpa reference
model = hemat VRAM) atau **KTO**. Temuan "preferensi di early-mid layers" nyambung dengan
desain layer-wise steering — sinyal preferensi bisa disuntikkan terarah di layer itu.

---

## 10. CultureTalk-ID — Cultural Commonsense Indonesia
**arXiv:** [2607.21016](https://arxiv.org/abs/2607.21016) · Juli 2026

**Esensi abstract:** Benchmark **dialogue-based pertama** untuk cultural commonsense
Bahasa Indonesia & bahasa lokal: **4.496 dialog, 11 bahasa, 13 topik**, kurasi human
multi-stage oleh native speakers. Tiga task: (1) dialogue-based MCQ cultural reasoning,
(2) machine translation culturally faithful, (3) **language steering** — apakah model bisa
memahami, mentransfer, dan menghasilkan bahasa yang grounded budaya.

**Implikasi:** Task (3) language steering persis menguji masalah language confusion yang
TLPO tangani. Tambahkan sebagai **eval pasca-training** (bukan train set). Repo resmi
belum diverifikasi — cek HuggingFace sebelum digunakan.

---

## Ringkasan Eksekusi (urutan yang disarankan)

1. **TLPO-weighted patch** → flag eksperimen di `get_batch_logps` (effort rendah)
2. **Ukur update RMS** Muon vs AdamW → logging `g_ortho.norm()` (effort sangat rendah)
3. **Cek window first-order** → log probe-loss per fase (effort rendah)
4. **OrScale trust-ratio** → flag eksperimen, A/B terisolasi (effort sedang, risiko tinggi)
5. **Re-steer pasca-SFT** → eksperimen (effort sedang)
6. **SimPO/KTO** untuk iterasi v8 (effort sedang)
7. **KL-reg / persona drift monitoring** (effort rendah)
8. **Eval CultureTalk-ID** pasca-training (effort rendah)

> Roadmap lengkap: [`2026-08-05_rekomendasi-prioritas-dan-roadmap.md`](2026-08-05_rekomendasi-prioritas-dan-roadmap.md)
