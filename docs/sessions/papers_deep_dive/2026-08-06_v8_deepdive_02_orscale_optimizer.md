# 🔬 Deep-Dive 02 — OrScale Optimizer & Non-Linear Training Dynamics (V8 Roadmap)

**Tanggal Analysis:** 6 Agustus 2026  
**Analisis Visual PDF & Rendered Pages:**
- `ORSCALE` (2605.07815): [pdfs/ORSCALE_2605.07815.pdf](file:///d:/Codings/unsloth-porto/t5-gemma-2/instruct/docs/sessions/papers_deep_dive/pdfs/ORSCALE_2605.07815.pdf) (`pages/ORSCALE/p01.png` s.d. `p16.png`)
- `MUON_RL` (2607.16169): [pdfs/MUON_RL_2607.16169.pdf](file:///d:/Codings/unsloth-porto/t5-gemma-2/instruct/docs/sessions/papers_deep_dive/pdfs/MUON_RL_2607.16169.pdf) (`pages/MUON_RL/p01.png` s.d. `p18.png`)
- `PAIRWISE_FRAGILE` (2607.16821): [pdfs/PAIRWISE_FRAGILE_2607.16821.pdf](file:///d:/Codings/unsloth-porto/t5-gemma-2/instruct/docs/sessions/papers_deep_dive/pdfs/PAIRWISE_FRAGILE_2607.16821.pdf) (`pages/PAIRWISE_FRAGILE/p01.png` s.d. `p36.png`)
- `LOCAL_LINEAR` (2606.10929): [pdfs/LOCAL_LINEAR_2606.10929.pdf](file:///d:/Codings/unsloth-porto/t5-gemma-2/instruct/docs/sessions/papers_deep_dive/pdfs/LOCAL_LINEAR_2606.10929.pdf) (`pages/LOCAL_LINEAR/p01.png` s.d. `p23.png`)

---

## 1. OrScale: Orthogonalised Optimization with Layer-Wise Trust-Ratio Scaling (NUS, Mei 2026)
**arXiv:** [2605.07815](https://arxiv.org/abs/2605.07815)

### A. Masalah & Bottleneck Muon Standar
Optimizer **Muon** (Keller Jordan et al., 2024) mengacak/meratakan spektrum nilai singular momentum matriks 2D melalui iterasi Newton-Schulz (NS). Namun, Muon standar menerapkan *global learning rate* yang sama untuk seluruh matriks 2D, mengabaikan fakta bahwa proyeksi attention (`q,k,v,o`) dan MLP (`gate,up,down`) memiliki distribusi skala gradien yang jauh berbeda. Modifikasi naif seperti menyatukan LAMB dengan Muon sering gagal karena *unit mismatch* (norm momentum $M_\ell$ berbasis gradien, bukan satuan parameter space).

### B. Formulasi OrScale & OrScale-LM
1. **Real-Update-Direction Trust Ratio:**
   Denominator trust-ratio wajib mengukur norma Frobenius dari *real update direction* $D_{\ell,t}^{\text{upd}}$ yang akan dikurangi dari bobot parameter:
   $$D_{\ell,t}^{\text{upd}} = \lambda W_{\text{l},t} + s_\ell Q_{\ell,t}$$
   $$r_{\ell,t} = \frac{\|W_{\text{l},t}\|_F}{\|D_{\text{l},t}^{\text{upd}}\|_F + \varepsilon}$$
2. **Moonlight Shape Factor:**
   $$s_\ell = 0.2 \sqrt{\max(m_\ell, n_\ell)}$$
   Faktor ini menyelaraskan RMS per-elemen dari polar factor $Q_\ell$ di seluruh bentuk matriks agar sesuai dengan skala AdamW.
3. **Lazy Per-Layer Calibration Constant ($c_{\text{denom},\ell}$):**
   Pada step optimizer pertama ($t=1$):
   $$c_{\text{denom},\ell} = \frac{\|W_{\ell,0}\|_F}{\|\lambda W_{\ell,0} + s_\ell Q_{\ell,0}\|_F + \varepsilon}$$
   Konstanta $c_{\text{denom},\ell}$ disimpan sekali (fp32) dan dipakai ulang pada tiap step berikutnya:
   $$\hat{r}_{\ell,t}^{\text{LM}} = \text{clip}\left( \frac{\|W_{\ell,t}\|_F}{c_{\text{denom},\ell} \|\lambda W_{\ell,t} + s_\ell Q_{\ell,t}\|_F + \varepsilon}, r_{\min}, r_{\max} \right)$$
   $$W_{\ell,t+1} = W_{\text{l},t} - \eta_t \hat{r}_{\ell,t}^{\text{LM}} \left( \lambda W_{\text{l},t} + s_\ell Q_{\ell,t} \right)$$

---

## 2. When Does Muon Help Agentic RL? (RUC & CAS ICT, Jul 2026)
**arXiv:** [2607.16169](https://arxiv.org/abs/2607.16169)

### A. Dynamic & Operating Regime
- Menguji Muon pada post-training RL (GRPO / GiGPO / GraphGPO).
- Mengonfirmasi bahwa **fan-in Muon** mempertahankan kestabilan pada step size yang lebih agresif ($3 \times 10^{-5}$ vs AdamW $10^{-6}$) ketika masih terdapat *headroom optimization*.
- Penggunaan *decoupled weight decay* dikombinasikan dengan Nesterov momentum ($\mu=0.95$, 5 step NS) terbukti paling konsisten mencegah degradasi pada tugas percakapan multi-turn.

---

## 3. First-Order Predictable but Pairwise Fragile & Local Linear Structures (Jul/Jun 2026)
**arXiv:** [2607.16821](https://arxiv.org/abs/2607.16821) & [2606.10929](https://arxiv.org/abs/2606.10929)

### A. Lie Bracket & Update Non-Commutativity
- Perturbasi tunggal bersifat linier (dapat diprediksi via $g^T \delta$). Namun, urutan dua update berturut-turut ($A \to B$ vs $B \to A$) menimbulkan cacat urutan (*commutator defect*) yang diatur oleh **Lie Bracket**:
  $$c(\eta) = \eta \kappa + O(\eta^2), \quad \kappa = \frac{\|H_B g_A - H_A g_B\|}{\|g_A + g_B\|}$$
- **Trajectory-Prefix Subspace:** Trajektori retraining berada di dalam *Krylov subspace* $\operatorname{span}\{g_0, H g_0, H^2 g_0\}$. Trajektori 10 step pertama menangkap 77% dari total pergeseran recovery, jauh melebihi *static task plane* (hanya 15%).

---

## 🛠️ Rencana Kode Presisi V8 (`working-molab-v7-combined-unsloth.py`)

### 1. Implementasi Class `OrScaleLM` Optimizer untuk Matrix Layers
Di `working-molab-v7-combined-unsloth.py`, tambahkan PyTorch optimizer `OrScaleLM` untuk menangani proyeksi matriks 2D LoRA:

```python
import math
import torch

class OrScaleLM(torch.optim.Optimizer):
    def __init__(self, params, lr=1e-4, weight_decay=0.01, momentum=0.95, ns_steps=5, r_min=0.1, r_max=5.0):
        defaults = dict(lr=lr, weight_decay=weight_decay, momentum=momentum, ns_steps=ns_steps, r_min=r_min, r_max=r_max)
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            lr = group['lr']
            wd = group['weight_decay']
            mu = group['momentum']
            steps = group['ns_steps']
            r_min = group['r_min']
            r_max = group['r_max']

            for p in group['params']:
                if p.grad is None:
                    continue
                g = p.grad
                if g.ndim < 2:
                    # Fallback untuk bias / LayerNorm / 1D tensors (AdamW update)
                    continue

                state = self.state[p]
                if 'momentum_buffer' not in state:
                    state['momentum_buffer'] = torch.clone(g).detach()
                    state['c_denom'] = None
                else:
                    state['momentum_buffer'].mul_(mu).add_(g)

                buf = state['momentum_buffer']
                # Nesterov lookahead momentum
                M_tilde = buf.mul(mu).add(g)

                # Newton-Schulz Iterations (approximate polar factor)
                Q = M_tilde
                norm_Q = Q.norm(p='fro')
                if norm_Q > 0:
                    Q = Q / norm_Q
                for _ in range(steps):
                    Q = 1.5 * Q - 0.5 * Q @ (Q.transpose(-2, -1) @ Q)

                # Moonlight shape factor: 0.2 * sqrt(max(m, n))
                m, n = p.shape[-2], p.shape[-1]
                s_l = 0.2 * math.sqrt(max(m, n))

                # Real update direction D_l = wd * P + s_l * Q
                D_l = wd * p + s_l * Q

                # Lazy per-layer calibration constant at t=1
                p_norm = p.norm(p='fro')
                D_norm = D_l.norm(p='fro') + 1e-6
                if state['c_denom'] is None:
                    state['c_denom'] = (p_norm / D_norm).item()

                # Calibrated trust ratio
                r_raw = p_norm / (state['c_denom'] * D_norm + 1e-6)
                r_hat = torch.clamp(r_raw, r_min, r_max)

                # Coupled update
                p.sub_(lr * r_hat * D_l)

        return loss
```

### 2. Konfigurasi Hybrid Optimizer di V8 Training Script:
Pisahkan parameter matriks 2D LoRA ke `OrScaleLM` dan 1D parameters ke `AdamW8bit`:
```python
matrix_params = [p for n, p in model.named_parameters() if p.requires_grad and p.ndim >= 2]
non_matrix_params = [p for n, p in model.named_parameters() if p.requires_grad and p.ndim < 2]

optimizer = OrScaleLM(matrix_params, lr=3e-4, weight_decay=0.01)
# Add non-matrix params to standard AdamW fallback
```
