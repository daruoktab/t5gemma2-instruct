# 🔬 Deep-Dive 05 — Objectives, Data Formatting & Domain Adaptation (V8 Roadmap)

**Tanggal Analysis:** 5 Agustus 2026  
**Analisis Visual PDF & Rendered Pages:**
- `MTO` (2606.24841): [pdfs/MTO_2606.24841.pdf](file:///d:/Codings/unsloth-porto/t5-gemma-2/instruct/docs/sessions/papers_deep_dive/pdfs/MTO_2606.24841.pdf) (`pages/MTO/p01.png` s.d. `p43.png`)
- `FLEXTAB` (2606.30336): [pdfs/FLEXTAB_2606.30336.pdf](file:///d:/Codings/unsloth-porto/t5-gemma-2/instruct/docs/sessions/papers_deep_dive/pdfs/FLEXTAB_2606.30336.pdf) (`pages/FLEXTAB/p01.png` s.d. `p28.png`)
- `STYLE_TRANSFER` (2604.11687): [pdfs/STYLE_TRANSFER_2604.11687.pdf](file:///d:/Codings/unsloth-porto/t5-gemma-2/instruct/docs/sessions/papers_deep_dive/pdfs/STYLE_TRANSFER_2604.11687.pdf) (`pages/STYLE_TRANSFER/p01.png` s.d. `p12.png`)
- `DOMAIN_ADAPTATION` (2607.06613): [pdfs/DOMAIN_ADAPTATION_2607.06613.pdf](file:///d:/Codings/unsloth-porto/t5-gemma-2/instruct/docs/sessions/papers_deep_dive/pdfs/DOMAIN_ADAPTATION_2607.06613.pdf) (`pages/DOMAIN_ADAPTATION/p01.png` s.d. `p12.png`)
- `LORA_TRADEOFFS` (2607.25583): [pdfs/LORA_TRADEOFFS_2607.25583.pdf](file:///d:/Codings/unsloth-porto/t5-gemma-2/instruct/docs/sessions/papers_deep_dive/pdfs/LORA_TRADEOFFS_2607.25583.pdf) (`pages/LORA_TRADEOFFS/p01.png` s.d. `p08.png`)

---

## 1. Matching Tasks to Objectives (MTO - TEHRAN, Jun 2026)
**arXiv:** [2606.24841](https://arxiv.org/abs/2606.24841) · Univ. of Tehran

### A. Masalah & Motivasi Utama
Inefisiensi prompt-tuning / fine-tuning standar pada model Encoder-Decoder karena ketidaksesuaian (*mismatch*) antara format prompt downstream dengan objective pre-training asli model (PrefixLM vs Span Denoising).

### B. Taksonomi Task & Template MTO
1. **Mask-Filling Tasks (Denoising Objective):**
   - Untuk tugas ekstraksi fakta / KBC serba pendek.
   - Template: `MaskedPrompting` (Input: `"Subject relation X"`, Target: `"X answer"`).
2. **Map-Phrasal Tasks (Language Modeling Objective):**
   - Untuk tugas penalaran klausa / generasi panjang.
   - Template: `Prompting` (Input: `"Sentence prefix because..."`, Target: `"continuation"`).

### C. Temuan Kunci
- Menyesuaikan format prompt dengan pre-training objective meningkatkan performa hingga **>120%** pada beberapa tugas few-shot.

---

## 2. FlexTab: Flexible Enc-Dec Architecture for Tabular ICL (SAP SE, Jun 2026)
**arXiv:** [2606.30336](https://arxiv.org/abs/2606.30336) · SAP SE & SAP France

### A. Masalah & Motivasi Utama
Tabular In-Context Learning (TabPFN, TabICL) menyatukan ekstraksi fitur dengan target prediksi tunggal. FlexTab memisahkan Enkoder yang *target-agnostic* dari set Dekoder spesifik-tugas (Classification, Regression, Anomaly Detection, Entity Matching).

### B. Formulasi Enkoder Agregasi Layer
Row embedding dikumpulkan dari **seluruh** layer enkoder (bukan hanya layer terakhir):
$$h_{\text{row}} = W_{\text{out}} \sum_{\ell=1}^L W_\ell h_{\text{row}}^{(\ell)}$$

---

## 3. Enc-Dec vs Dec-Only for AI-to-Human Text Style Transfer (Apr 2026)
**arXiv:** [2604.11687](https://arxiv.org/abs/2604.11687)

### A. Keunggulan Mutlak Seq2Seq pada Penyuntingan Teks
- BART-large (406M) mengalahkan Mistral-7B QLoRA (7B, 17x lebih besar) pada seluruh metrik kemiripan referensi (BERTScore 0.924 vs 0.898, ROUGE-L 0.566 vs 0.464).
- Model Decoder-only cenderung *overshoot* (berlebihan mengubah gaya hingga merusak tanda baca dan frekuensi koma). Pre-training denoising Seq2Seq memberikan keuntungan struktural bawaan untuk tugas *constrained rewriting*.

---

## 4. Pre-Training on SE Texts (Domain Adaptation CPT, Jul 2026)
**arXiv:** [2607.06613](https://arxiv.org/abs/2607.06613)

### A. Strategi Continual Pre-Training (CPT)
- Menjalankan CPT unsupervised pada korpus domain spesifik menggunakan kombinasi Span Corruption + PrefixLM sebelum fine-tuning SFT.

---

## 5. Parameter Trade-offs in Modern LoRA Variants (Jul 2026)
**arXiv:** [2607.25583](https://arxiv.org/abs/2607.25583)

### A. Saturation Rank & Module Efficiency
- Rank LoRA mengalami kejenuhan (*rank saturation*) awal pada $r=16$. Menikkan rank ke $r=32$ hanya memberikan kenaikan akurasi <0.8% namun menggelembungkan parameter.
- Menargetkan modul proyeksi attention $\{q, v\}$ jauh lebih efisien parameter-per-gain daripada menargetkan seluruh modul linear termasuk MLP.

---

## 🛠️ Rencana Ubahan Kode V8 (`working-molab-v7-combined-unsloth.py`)

### 1. Serializer Data MTO & Task Prefix Mapped Collator
Di `working-molab-v7-combined-unsloth.py`, update formatting data SFT berdasarkan kategori MTO:

```python
def format_mto_task_prompt(task_type, text_input, text_target=None):
    """
    Format task input based on Tehran MTO principles for T5Gemma 2.
    """
    if task_type in ["summarize", "qa", "ner"]:
        # Mask-filling / short target -> Denoising style
        prompt = f"<unused1> Input: {text_input} Target: <unused0>"
    elif task_type in ["paraphrase", "chat", "style_transfer"]:
        # Map-Phrasal -> PrefixLM style
        prompt = f"<unused6> Rewrite the following naturally: {text_input}"
    else:
        prompt = text_input
    return prompt
```

### 2. Set Up Optimal LoRA Configuration (Rank 128 / Target Modules)
Konfigurasi PEFT Unsloth di V8 disesuaikan dengan rekomendasi trade-off $r=128$, $\alpha=128$, target all linear:

```python
model = FastLanguageModel.get_peft_model(
    model,
    r=128,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    lora_alpha=128,
    lora_dropout=0,
    bias="none",
    use_gradient_checkpointing="unsloth",
    random_state=3407,
)
```
