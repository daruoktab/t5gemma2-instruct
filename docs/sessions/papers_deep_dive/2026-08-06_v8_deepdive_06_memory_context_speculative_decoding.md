# 🔬 Deep-Dive 06 — Latent Memory, Context Window Extension & Speculative Decoding (V8 Roadmap)

**Tanggal Analysis:** 5 Agustus 2026  
**Analisis Visual PDF & Rendered Pages:**
- `LATENT_MEMORY` (2606.20911): [pdfs/LATENT_MEMORY_2606.20911.pdf](file:///d:/Codings/unsloth-porto/t5-gemma-2/instruct/docs/sessions/papers_deep_dive/pdfs/LATENT_MEMORY_2606.20911.pdf) (`pages/LATENT_MEMORY/p01.png` s.d. `p17.png`)
- `STACKED_CONTEXT` (2603.04759): [pdfs/STACKED_CONTEXT_2603.04759.pdf](file:///d:/Codings/unsloth-porto/t5-gemma-2/instruct/docs/sessions/papers_deep_dive/pdfs/STACKED_CONTEXT_2603.04759.pdf) (`pages/STACKED_CONTEXT/p01.png` s.d. `p20.png`)
- `CASSANDRA_SPECULATIVE` (2605.26558): [pdfs/CASSANDRA_SPECULATIVE_2605.26558.pdf](file:///d:/Codings/unsloth-porto/t5-gemma-2/instruct/docs/sessions/papers_deep_dive/pdfs/CASSANDRA_SPECULATIVE_2605.26558.pdf) (`pages/CASSANDRA_SPECULATIVE/p01.png` s.d. `p15.png`)

---

## 1. Latent Personal Memory (LPM - UMASS & Samsung, Jun 2026)
**arXiv:** [2606.20911](https://arxiv.org/abs/2606.20911) · UMass Amherst & Samsung Research

### A. Masalah & Motivasi Utama
Personalisasi LLM jangka panjang membutuhkan retensi riwayat pengguna tanpa membengkakkan KV-cache atau merusak kemampuan penalaran umum (yang sering terjadi pada fine-tuning LoRA per-user).

### B. Formulasi Latent Memory Slots & Dynamic Soft Prompts
1. **Per-User Latent Slot Matrix:**
   Setiap pengguna $u$ memiliki matriks slot memori terkompresi $M_u \in \mathbb{R}^{N \times d_{\text{mem}}}$.
2. **Cross-Attention Soft Prompt Retrieval:**
   Untuk query/input $x$ dengan embedding $\epsilon(x)$:
   $$\tilde{P}_h = \text{softmax}\left( \frac{W_{Qh}\epsilon(x) (W_{Kh} M_{uh})^T}{\sqrt{d}} \right) W_{Vh} M_{uh}$$
   $$P_h(x, M_{uh}) = \text{LN}_h(\tilde{P}_h + \text{MLP}(\tilde{P}_h))$$
3. **Penyuntikan ke LLM (Prepended Soft Prompts):**
   Soft prompt dinamis berjarak $H$ head digabungkan dan disuntikkan ke system prompt sebelum token pengguna:
   $$\hat{y} = \text{LLM}_{\text{frozen}}([P_1(x, M_{u1}); \dots; P_H(x, M_{uH}); x])$$
4. **Efisiensi:**
   Mereduksi ukuran KV-cache sebesar **>64x lipat** dibandingkan full-context prompting.

---

## 2. SHAREDLLM: Multi-Scale Self-Injection for Context Window Extension (Apr 2026)
**arXiv:** [2603.04759](https://arxiv.org/abs/2603.04759) · SUTD, SMU, NTU, NUS

### A. Masalah & Motivasi Utama
Memperpanjang konteks LLM ke 128K+ token secara mahal membutuhkan continual pre-training pada sekuens panjang. SHAREDLLM menumpuk dua instance dari LLM konteks pendek yang sama (*self-injection*).

### B. Arsitektur Compressor & Tree-based Cross-Attention
1. **Lower Model (Compressor):** Menggunakan $M$ layer awal dari LLM untuk mengompresi chunk konteks $C_i$ secara paralel menjadi *Context Tree* multi-skala.
2. **Upper Model (Decoder):** Menggunakan $N-M$ layer atas. Query $Q$ mengakses KV state terkompresi dari *Context Tree* melalui layer cross-attention khusus.
3. **Ekstrapolasi:** Model yang dilatih hanya pada 8K token berhasil mengabstraksi konteks hingga **128K+ token** dengan 2x-3x speedup inferensi.

---

## 3. Cassandra: Self-Speculative Decoding at Edge (KAIST, Mei 2026)
**arXiv:** [2605.26558](https://arxiv.org/abs/2605.26558) · KAIST

### A. Masalah & Motivasi Utama
Speculative decoding standar membutuhkan model draft terpisah yang dilatih khusus. Cassandra membuat draft model tanpa pelatihan (*training-free*) langsung dari model target melalui bit-level pruning & truncation.

### B. Metode Format Transformation
1. **Unstructured Value Pruning & Mantissa Truncation:**
   Pruning elemen bobot/KV-cache beresolusi rendah dan truncation bit mantissa FP16/BF16.
2. **Lossless Exponent Compression (Unary Coding):**
   Kompresi bit eksponen menggunakan Unary Coding (rata-rata eksponen terkompresi menjadi ~2.85 bit).
3. **Dua Mode Operasi:**
   - **Cassandra-1:** Lossless exponent compression (Akurasi 100% sama dengan BF16).
   - **Cassandra-2:** Lossy MX-format exponent compression untuk speedup maksimal (hingga 2.41x speedup pada RTX 4090 / Edge NPU).

---

## 🛠️ Rencana Ubahan Kode V8 (`working-molab-v7-combined-unsloth.py`)

### 1. Integrasi Dynamic Soft Prompt Injector (LPM) untuk User Profile
Di `working-molab-v7-combined-unsloth.py`, tambahkan class `LatentPersonalMemoryModule`:

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class LatentPersonalMemoryModule(nn.Module):
    """
    Latent Personal Memory (LPM) for dynamic soft prompt injection.
    """
    def __init__(self, num_users, num_slots=32, mem_dim=256, llm_dim=2560, num_heads=8):
        super().__init__()
        self.num_slots = num_slots
        self.mem_dim = mem_dim
        self.llm_dim = llm_dim
        self.num_heads = num_heads
        
        # Per-user persistent memory slots
        self.user_memory = nn.Parameter(torch.randn(num_users, num_heads, num_slots, mem_dim) * 0.02)
        
        # Projection heads
        self.w_q = nn.Linear(llm_dim, num_heads * mem_dim, bias=False)
        self.w_k = nn.Linear(mem_dim, mem_dim, bias=False)
        self.w_v = nn.Linear(mem_dim, llm_dim // num_heads, bias=False)
        self.mlp_out = nn.Sequential(
            nn.Linear(llm_dim, llm_dim),
            nn.GELU(),
            nn.Linear(llm_dim, llm_dim)
        )
        self.layer_norm = nn.LayerNorm(llm_dim)

    def forward(self, user_id, input_embeddings):
        # input_embeddings: [B, L, llm_dim]
        # user_id: [B]
        B, L, _ = input_embeddings.shape
        q_emb = input_embeddings.mean(dim=1) # [B, llm_dim]
        
        q = self.w_q(q_emb).view(B, self.num_heads, 1, self.mem_dim)
        M_u = self.user_memory[user_id] # [B, num_heads, num_slots, mem_dim]
        
        k = self.w_k(M_u)
        v = self.w_v(M_u)
        
        scores = torch.matmul(q, k.transpose(-2, -1)) / (self.mem_dim ** 0.5)
        attn = F.softmax(scores, dim=-1)
        
        soft_prompts_per_head = torch.matmul(attn, v) # [B, num_heads, 1, llm_dim // num_heads]
        soft_prompts = soft_prompts_per_head.view(B, 1, self.llm_dim)
        
        # Pass through MLP + LayerNorm
        out = self.layer_norm(soft_prompts + self.mlp_out(soft_prompts))
        return out # [B, 1, llm_dim] soft prompt token to prepend
```

### 2. Formulasi Self-Speculative Draft Verification Loop (Cassandra-style)
Menyediakan pipeline inferensi cepat untuk dekoding V8.
