# 🔬 Deep-Dive 04 — Architecture, Internal Retrieval & Cross-Model Multimodal (V8 Roadmap)

**Tanggal Analysis:** 5 Agustus 2026  
**Analisis Visual PDF & Rendered Pages:**
- `INTRA` (2605.05806): [pdfs/INTRA_2605.05806.pdf](file:///d:/Codings/unsloth-porto/t5-gemma-2/instruct/docs/sessions/papers_deep_dive/pdfs/INTRA_2605.05806.pdf) (`pages/INTRA/p01.png` s.d. `p20.png`)
- `XBRIDGE` (2603.17512): [pdfs/XBRIDGE_2603.17512.pdf](file:///d:/Codings/unsloth-porto/t5-gemma-2/instruct/docs/sessions/papers_deep_dive/pdfs/XBRIDGE_2603.17512.pdf) (`pages/XBRIDGE/p01.png` s.d. `p23.png`)
- `T5GEMMA2` (2512.14856): [pdfs/T5GEMMA2_2512.14856.pdf](file:///d:/Codings/unsloth-porto/t5-gemma-2/instruct/docs/sessions/papers_deep_dive/pdfs/T5GEMMA2_2512.14856.pdf) (`pages/T5GEMMA2/p01.png` s.d. `p13.png`)
- `UL2R` (2210.11399): [pdfs/UL2R_2210.11399.pdf](file:///d:/Codings/unsloth-porto/t5-gemma-2/instruct/docs/sessions/papers_deep_dive/pdfs/UL2R_2210.11399.pdf) (`pages/UL2R/p01.png` s.d. `p21.png`)
- `PM_ROPE_TTS` (2604.01760): [pdfs/PM_ROPE_TTS_2604.01760.pdf](file:///d:/Codings/unsloth-porto/t5-gemma-2/instruct/docs/sessions/papers_deep_dive/pdfs/PM_ROPE_TTS_2604.01760.pdf) (`pages/PM_ROPE_TTS/p01.png` s.d. `p20.png`)

---

## 1. INTRA: In-Model Retrieval via Reverse-QWK (NVIDIA, Mei 2026)
**arXiv:** [2605.05806](https://arxiv.org/abs/2605.05806) · NVIDIA & Technion

### A. Masalah & Motivasi Utama
Retrieval-Augmented Generation (RAG) konvensional memisahkan retriever (seperti BM25, BGE, atau ColBERT) dan generator. Hal ini menimbulkan *retriever-generator representation mismatch* dan memaksakan re-encoding ulang teks dokumen yang ditarik pada setiap query. INTRA menyatukan retrieval dan generasi dalam satu arsitektur Encoder-Decoder (seperti T5Gemma 2).

### B. Formulasi Matematika Reverse-QWK (Reversed Query-Key Projection)
1. **Penyimpanan Matriks Encoder Tunggal ($d$-dimensional, Head-Agnostic):**
   Alih-alih menyimpan $K$ layer-wise & head-wise untuk setiap token encoder:
   $$\bar{K}(S) = \text{RMSNorm}(K(S)) \in \mathbb{R}^{N \times d}$$
2. **Transformasi Proyeksi pada Sisi Query Decoder Layer $\ell$:**
   Pindahkan bobot proyeksi key $W_{K, \ell}$ dan per-head RMSNorm scale $\gamma_{K, \ell}$ ke sisi query decoder $q_\ell$:
   $$\tilde{q}_\ell = (q_\ell W_{K, \ell}^T) \odot \gamma_{K, \ell}$$
3. **Persamaan Ekuivalen Cross-Attention & MaxSim Scoring:**
   $$z_\ell = \text{Attention}_{\text{RQWK}}(\tilde{q}_\ell, \bar{K}(S), \bar{K}(S)) = \text{softmax}\left( \frac{\tilde{q}_\ell \bar{K}(S)^T}{\sqrt{d}} \right) \bar{K}(S)$$
4. **Efisiensi Kompresi:**
   Mereduksi memori KV cache prefill hingga **~30x lipat** ($2L n_{\text{kv}} d_h / d$) dan murni meniadakan re-encoding dokumen saat inferensi.

---

## 2. XBridge: Composing LLMs with Enc-Dec Translation Models (Mei 2026)
**arXiv:** [2603.17512](https://arxiv.org/abs/2603.17512) · CAS ICT

### A. Masalah & Motivasi Utama
LLM seperti T5Gemma/LLaMA memiliki kemampuan penalaran dan pengetahuan yang kuat dalam Bahasa Inggris, tetapi lemah pada bahasa berdasar rendah (*low-resource languages*). Sebaliknya, model NLLB-200 sangat ahli dalam menerjemahkan 200+ bahasa tetapi kurang dalam penalaran kompleks.

### B. Arsitektur Encoder-LLM-Decoder & Loss Optimal Transport (OT)
1. **Layer Pemetaan (Mapping Layers):**
   - Enkoder Multilingual $\to$ LLM: $\tilde{H}^x = \text{Mapping}_{\text{enc}}(H^x) \in \mathbb{R}^{n \times d_l}$
   - LLM $\to$ Dekoder Multilingual: $\tilde{H}^{z'} = \text{Mapping}_{\text{dec}}(H^{z'}) \in \mathbb{R}^{m \times d_d}$
2. **Loss Alignment Optimal Transport (OT):**
   Mencocokkan representasi token LLM $H^{z'}$ dengan enkoder $H^z$ secara fleksibel meskipun terdapat perbedaan panjang sekuens ($k \neq m$):
   $$\mathcal{D}^*(H^z, \tilde{H}^{z'}) = \min_{T \ge 0} \sum_{i,j} T_{ij} c(H_i^z, \tilde{H}_j^{z'}) \quad \text{s.t.} \quad \sum_{j=1}^m T_{ij} = m_i^z$$
3. **Strategi Pelatihan 3-Tahap:**
   - **Tahap 1:** Warm-up Cross-Model Mapping ($\mathcal{L}_{\text{CE\_LLM}} + \mathcal{L}_{\text{CE\_Dec}} + \lambda \mathcal{L}_{\text{OT}}$).
   - **Tahap 2:** Adaptasi Sisi Enkoder (Fine-tune $\text{Mapping}_{\text{enc}}$ pada data instruksi).
   - **Tahap 3:** Adaptasi Sisi Dekoder (Fine-tune $\text{Mapping}_{\text{dec}}$ & Cross-Attention dekoder).

---

## 3. T5Gemma 2 Base Model & Multimodal SigLIP Integration (Des 2025)
**arXiv:** [2512.14856](https://arxiv.org/abs/2512.14856) · Google DeepMind

### A. Komponen Utama V7 yang Dipertahankan & Ditingkatkan ke V8
- **Seq2Seq 4B-4B Backbone:** Enkoder 4B + Dekoder 4B dengan *Tied Word Embeddings* dan *Merged Attention* ($K,V = [X; H]$).
- **Integrasi Vision SigLIP 400M:** Gambar di-encode menjadi 256 soft tokens disuntikkan langsung ke Enkoder T5Gemma pada token placeholder `<image_soft_token>` (`256001`).

---

## 4. UL2R: Transcending Scaling Laws with 0.1% Extra Compute (Okt 2022)
**arXiv:** [2210.11399](https://arxiv.org/abs/2210.11399) · Google

### A. Mixture-of-Denoisers Framework
- **R-Denoiser (Regular Span Corruption):** Membantu NLU / pemahaman bahasa.
- **S-Denoiser (Sequential PrefixLM):** Membantu generasi percakapan dan penalaran berurutan.
- **X-Denoiser (Extreme Span Corruption):** Membantu ekstrapolasi konteks panjang.

---

## 5. PM-RoPE: Pattern-Masked RoPE untuk Modul Voice/Speech (Apr 2026)
**arXiv:** [2604.01760](https://arxiv.org/abs/2604.01760)

### A. Integrasi Multimodal Positional Encoding
- Masking pola frekuensi RoPE untuk memisahkan posisi token teks 1D dan token audio/speech 2D pada T5Gemma.

---

## 🛠️ Rencana Ubahan Kode V8 (`working-molab-v7-combined-unsloth.py`)

### 1. Implementasi Custom Cross-Attention `Reverse-QWK` (INTRA Module)
Di `working-molab-v7-combined-unsloth.py`, tambahkan class `ReverseQWKCrossAttention`:

```python
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

class ReverseQWKCrossAttention(nn.Module):
    """
    Reverse Query-Key Projection (Reverse-QWK) module for T5Gemma 2.
    Allows reusing single RMSNorm-ed encoder representations K_bar across all decoder layers.
    """
    def __init__(self, hidden_dim, num_heads, head_dim):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.hidden_dim = hidden_dim
        
        self.q_proj = nn.Linear(hidden_dim, num_heads * head_dim, bias=False)
        self.k_proj = nn.Linear(hidden_dim, num_heads * head_dim, bias=False)
        self.v_proj = nn.Linear(hidden_dim, num_heads * head_dim, bias=False)
        self.o_proj = nn.Linear(num_heads * head_dim, hidden_dim, bias=False)
        
        self.gamma_k = nn.Parameter(torch.ones(head_dim))
        
    def forward(self, x_decoder, k_bar_encoder):
        # x_decoder: [B, L_q, d_model]
        # k_bar_encoder: [B, L_kv, d_model] (already RMSNorm-ed static pool)
        B, L_q, _ = x_decoder.shape
        _, L_kv, _ = k_bar_encoder.shape
        
        # 1. Project decoder query
        q = self.q_proj(x_decoder).view(B, L_q, self.num_heads, self.head_dim)
        
        # 2. Reverse-QWK transformation: apply key weights W_k and gamma_k to Query side
        W_k_heads = self.k_proj.weight.view(self.num_heads, self.head_dim, self.hidden_dim)
        # q_tilde = (q * gamma_k) @ W_k
        q_scaled = q * self.gamma_k
        q_tilde = torch.einsum('bqhd,hdm->bqhm', q_scaled, W_k_heads) # [B, L_q, num_heads, d_model]
        
        # 3. Compute dot product directly against shared head-agnostic encoder pool k_bar
        # k_bar: [B, 1, L_kv, d_model]
        scores = torch.einsum('bqhm,bkm->bhqk', q_tilde, k_bar_encoder) / math.sqrt(self.head_dim)
        attn_weights = F.softmax(scores, dim=-1)
        
        # 4. Standard V projection
        v = self.v_proj(k_bar_encoder).view(B, L_kv, self.num_heads, self.head_dim)
        out = torch.einsum('bhqk,bkhd->bqhd', attn_weights, v)
        out = out.reshape(B, L_q, self.hidden_dim)
        return self.o_proj(out)
```

### 2. Integrasi XBridge 2-Layer MLP Bridge & OT Loss
Di `working-molab-v7-combined-unsloth.py`:

```python
def compute_relaxed_ot_loss(h_llm, h_enc):
    """
    Relaxed Optimal Transport (OT) alignment loss for XBridge (Eq 9 in INTRA/XBridge).
    h_llm: [B, M, d_model]
    h_enc: [B, K, d_model]
    """
    # Cosine distance cost matrix: C_ij = 1 - cos(h_llm_i, h_enc_j)
    h_llm_norm = F.normalize(h_llm, p=2, dim=-1)
    h_enc_norm = F.normalize(h_enc, p=2, dim=-1)
    
    sim_matrix = torch.bmm(h_llm_norm, h_enc_norm.transpose(1, 2)) # [B, M, K]
    cost_matrix = 1.0 - sim_matrix
    
    # Relaxed OT: for each token in h_llm, find min cost in h_enc
    min_cost, _ = torch.min(cost_matrix, dim=-1)
    return min_cost.mean()
```
