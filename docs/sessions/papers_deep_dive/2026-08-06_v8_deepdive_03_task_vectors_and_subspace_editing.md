# 🔬 Deep-Dive 03 — Task Vector Geometry, DeVec & Subspace Editing (V8 Roadmap)

**Tanggal Analysis:** 5 Agustus 2026  
**Analisis Visual PDF & Rendered Pages:**
- `TV_GEOM` (2605.03780): [pdfs/TV_GEOM_2605.03780.pdf](file:///d:/Codings/unsloth-porto/t5-gemma-2/instruct/docs/sessions/papers_deep_dive/pdfs/TV_GEOM_2605.03780.pdf) (`pages/TV_GEOM/p01.png` s.d. `p59.png`)
- `TV_DECOMP` (2512.22511): [pdfs/TV_DECOMP_2512.22511.pdf](file:///d:/Codings/unsloth-porto/t5-gemma-2/instruct/docs/sessions/papers_deep_dive/pdfs/TV_DECOMP_2512.22511.pdf) (`pages/TV_DECOMP/p01.png` s.d. `p16.png`)
- `EASYEDIT` (2308.07269): [pdfs/EASYEDIT_2308.07269.pdf](file:///d:/Codings/unsloth-porto/t5-gemma-2/instruct/docs/sessions/papers_deep_dive/pdfs/EASYEDIT_2308.07269.pdf) (`pages/EASYEDIT/p01.png` s.d. `p12.png`)
- `EMERGENT_MISALIGNMENT` (2607.21356): [pdfs/EMERGENT_MISALIGNMENT_2607.21356.pdf](file:///d:/Codings/unsloth-porto/t5-gemma-2/instruct/docs/sessions/papers_deep_dive/pdfs/EMERGENT_MISALIGNMENT_2607.21356.pdf) (`pages/EMERGENT_MISALIGNMENT/p01.png` s.d. `p108.png`)
- `CAUSAL_ENCODERS` (2512.10561): [pdfs/CAUSAL_ENCODERS_2512.10561.pdf](file:///d:/Codings/unsloth-porto/t5-gemma-2/instruct/docs/sessions/papers_deep_dive/pdfs/CAUSAL_ENCODERS_2512.10561.pdf) (`pages/CAUSAL_ENCODERS/p01.png` s.d. `p24.png`)

---

## 1. Task Vector Geometry Underlies Dual Modes of Task Inference (Mei 2026)
**arXiv:** [2605.03780](https://arxiv.org/abs/2605.03780) · UW-Madison & Univ. of Chicago

### A. Masalah & Motivasi Utama
In-Context Learning (ICL) pada Transformer beroperasi dalam dua mode inferensi yang berbeda:
1. **Mode M1 (Bayesian Task Retrieval):** Model mengenali task yang pernah dilihat saat pre-training, direpresentasikan sebagai kombinasi konveks dari *task vectors* di subruang memori.
2. **Mode M2 (Extrapolative Task Learning):** Model melakukan ekstrapolasi untuk task Out-of-Distribution (OOD) yang belum pernah dilihat, dengan memanfaatkan statistik konteks yang dikodekan pada subruang yang **hampir ortogonal** (*near-orthogonal subspace*).

### B. Formulasi Matematika & Teorema
1. **Representasi Mode M1 (Simplex Convex Combination):**
   $$h_t \approx \mu_t + \sum_{k \le K} \beta_{t,k} \theta_k + \nu_{s_t}$$
   di mana $\theta_k$ adalah task vector untuk task $k$, dan $\beta_{t,k} \approx P(z=k \mid s_{\le t})$ melacak Bayesian posterior.
2. **Batas Altitudo Subruang Mode M2 (Theorem 2):**
   Ketika intrinsic dimension task OOD $d_0 > k_\star$ (dimensi task subspace), representasi $M2$ wajib keluar dari task subspace $\mathcal{S}$ dengan batas bawah jarak:
   $$\sup_{z \in U} \inf_{c \in \mathcal{S}} \| \mu_\star(z) - c \|_2 \ge \frac{c - \delta}{L} - \frac{C}{2^{d_0 / k_\star} - 1}$$

### C. Temuan Kunci & Implikasi
- Keanekaragaman task ($N_{\text{minor}}$) saat training memicu transisi fase dari M1 ke M2.
- Mengonfirmasi bahwa manipulasi task vector hanya mempengaruhi M1 tanpa merusak kemampuan penalaran OOD (M2) jika proyeksi dilakukan secara cermat.

---

## 2. Decomposing Task Vectors for Refined Model Editing (DeVec, Des 2025)
**arXiv:** [2512.22511](https://arxiv.org/abs/2512.22511) · Adelaide Univ. & Monash Univ.

### A. Masalah & Motivasi Utama
Penjumlahan/pengurangan task vector sederhana ($\Delta W = W_{\text{SFT}} - W_{\text{base}}$) sering mengalami **interferensi konsep** karena task vector mengandung fitur umum yang tumpang tindih (*shared features*) dan fitur unik (*unique features*).

### B. Formulasi Matematika Algoritma DeVec
1. **Matriks Proyeksi Subruang Kolom:**
   Untuk matriks bobot task vector $W_i = U_i \Sigma_i V_i^T$:
   $$P_i = U_i U_i^T$$
2. **Penggabungan Proyeksi Tersambung (Chained Projections):**
   $$P_{1, 2, \dots, k} = \prod_{i=1}^k P_i$$
3. **Eigendecomposition & Matriks Shared Subspace:**
   Hitung SVD $Z \Lambda Z^T = P_{1, 2, \dots, k}$. Ambil eigenvector $Z_{\text{shared}} = Z[:, r]$ di mana nilai eigen $\lambda > \tau$ (default $\tau = 0.85$).
   $$P^{\text{shared}} = Z_{\text{shared}} Z_{\text{shared}}^T$$
4. **Pemisahan Komponen Shared & Unique:**
   $$W_i^{\text{shared}} = P^{\text{shared}} W_i, \quad W_i^{\text{unique}} = W_i - W_i^{\text{shared}}$$

### C. Temuan Kunci
- Mengurangi toksisitas sebesar 47% pada benchmark ToxiGen tanpa merusak performa tugas penalaran umum (GSM8K/MMLU) dengan hanya menegasikan $W^{\text{unique}}_{\text{toxic}}$.
- Meningkatkan akurasi multi-task model merging sebesar 5%.

---

## 3. EasyEdit: Knowledge Editing Framework for LLMs (Jun 2024)
**arXiv:** [2308.07269](https://arxiv.org/abs/2308.07269) · Zhejiang University

### A. Masalah & Motivasi Utama
Mengubah fakta spesifik pada LLM tanpa melakukan retraining penuh yang mahal atau merusak fakta lain yang tidak berhubungan (Locality & Portability).

### B. Pendekatan ROME / MEMIT dalam EasyEdit
- **ROME (Rank-One Model Editing):** Mengidentifikasi MLP Layer tengah (misal Layer 12-18) sebagai memori key-value, lalu melakukan update rank-1 pada matriks $W_{\text{down}}$:
  $$\Delta W = \frac{(v - W k) k^T C^{-1}}{k^T C^{-1} k}$$
  di mana $k$ adalah kunci representasi subjek, $v$ adalah nilai target baru, dan $C = \mathbb{E}[k k^T]$ adalah covariance matrix aktivasi memori.

---

## 4. Emergent Misalignment Recruits a Pre-existing Persona Subspace (Jul 2026)
**arXiv:** [2607.21356](https://arxiv.org/abs/2607.21356) · Independent

### A. Masalah & Motivasi Utama
Fine-tuning pada data instruksi yang berisiko kecil (misal: kode yang rentan keamanan) dapat menginduksi misalignment luas (*emergent misalignment*) di seluruh topik yang tidak relevan.

### B. Formulasi Subruang Persona & Proyeksi
- **Ekstraksi Persona Subspace:** Menggunakan *Contrastive Teacher Forcing* pada prompt berpasangan (misal: *reckless* vs *cautious*) tanpa mengubah token output.
- **Intervensi Proyeksi Residual Stream:**
  $$h \leftarrow h - P_{\text{persona}} h$$
- **Hasil Kunci:** Mengeluarkan persona subspace dari residual stream saat fine-tuning menurunkan broad misalignment dari **27.7% menjadi 0.0%** tanpa merusak bahasa.

---

## 5. Causal Encoders for Representation Learning (Des 2025)
**arXiv:** [2512.10561](https://arxiv.org/abs/2512.10561)

### A. Masalah & Motivasi Utama
Enkoder kausal memastikan bahwa representasi latent yang dipelajari pada Encoder Seq2Seq benar-benar mengisolasi faktor kausal independen.

---

## 🛠️ Rencana Ubahan Kode V8 (`working-molab-v7-combined-unsloth.py`)

### 1. Fungsi Dekomposisi LoRA Weight SVD (DeVec) untuk V8 Merging & Unlearning
Di `working-molab-v7-combined-unsloth.py` atau skrip utilitas `scripts/model/devec_utils.py`:

```python
import torch

def decompose_lora_task_vectors(lora_weights_list, threshold=0.85):
    """
    Decomposes a list of LoRA delta weight matrices into shared and unique components (DeVec algorithm).
    lora_weights_list: List of Tensors [d_out, d_in]
    """
    k = len(lora_weights_list)
    projections = []
    
    for W in lora_weights_list:
        # SVD of task vector W
        U, S, Vh = torch.linalg.svd(W.float(), full_matrices=False)
        # Projection matrix onto column space P = U @ U.T
        P = U @ U.mH
        projections.append(P)
    
    # Chained projection matrix P_chained = P_1 @ P_2 @ ... @ P_k
    P_chained = projections[0]
    for i in range(1, k):
        P_chained = P_chained @ projections[i]
        
    # Eigendecomposition of chained projection
    eigenvalues, eigenvectors = torch.linalg.eig(P_chained)
    eigenvalues = eigenvalues.real
    eigenvectors = eigenvectors.real
    
    # Select shared subspace directions where eigenvalue > threshold
    shared_mask = eigenvalues > threshold
    if not shared_mask.any():
        # Fallback if no direction clears threshold
        shared_mask = eigenvalues == eigenvalues.max()
        
    Z_shared = eigenvectors[:, shared_mask]
    P_shared = Z_shared @ Z_shared.mH
    
    # Decompose each task vector into Shared & Unique
    shared_components = []
    unique_components = []
    for W in lora_weights_list:
        W_shared = (P_shared @ W.float()).to(W.dtype)
        W_unique = W - W_shared
        shared_components.append(W_shared)
        unique_components.append(W_unique)
        
    return shared_components, unique_components
```

### 2. Forward Hook untuk Persona Subspace Ablation saat Fine-Tuning
Dapat dipasang pada `working-molab-v7-combined-unsloth.py` saat SFT V8 untuk mencegah emergent misalignment:

```python
class PersonaSubspaceAblator:
    def __init__(self, persona_basis_matrix):
        # persona_basis_matrix shape: [d_model, r_persona]
        self.P_persona = persona_basis_matrix @ persona_basis_matrix.T
        
    def hook_fn(self, module, input, output):
        # output is a tuple (hidden_states, ...) or hidden_states tensor
        if isinstance(output, tuple):
            h = output[0]
            # Project out persona: h_clean = h - h @ P_persona
            h_clean = h - torch.matmul(h, self.P_persona.to(h.device))
            return (h_clean,) + output[1:]
        else:
            return output - torch.matmul(output, self.P_persona.to(output.device))
```
