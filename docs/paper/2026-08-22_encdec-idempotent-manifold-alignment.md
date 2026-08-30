# Encoder-Decoder Manifold Alignment for Idempotent Generation

**arXiv:** 2606.22304v1 · **Date:** 2026-06-21 · **Category:** cs.LG
**Authors:** Dareen Alharthi, Abdul Waheed, Bhiksha Raj (Carnegie Mellon University)
**Link:** https://arxiv.org/abs/2606.22304

---

## Abstract

Recent work enforces *idempotency* in generative models: applying the model multiple times should leave a sample unchanged once it lies on the target data manifold. These approaches often fail to reach exact fixed points, causing **instability and drift**. The paper argues the root cause is a **geometric mismatch between the manifolds learned by the encoder and the decoder**. It proposes a training framework that closes this gap by forcing encoder and decoder to learn consistent representations of the same underlying data manifold. Result: lower idempotency error and identical outputs under repeated application (image generation + image editing).

---

## Theory

- Model f = D_θ ∘ E_ϕ : X → X. Reconstruction loss L_rec = ||D(E(x)) − x||.
- A well-trained encoder-decoder implements a **projection** P(x) = D(E(x)) onto a learned manifold M. Projection operators are **idempotent by definition** (P² = P).
- **Proposition 1 (necessity):** If f acts as a projection, then f must be idempotent: f(f(x)) = f(x). Contrapositive: if f(f(x)) ≠ f(x), the representation is **suboptimal in the projection sense**.
- **Proposition 2 (sufficiency):** If encoder satisfies latent consistency E(D(E(x))) = E(x), output idempotency follows.
- **Proposition 3 (error bound):** If D is L-Lipschitz, output-space drift ≤ L · (latent-space idempotency error). So **minimizing latent error is a principled surrogate**.
- Idempotency is the goal; **latent-space consistency** is the practical, sufficient mechanism.

## Method

Idempotency loss (latent space), with **stop-gradient:**

```
z  = E_ϕ(x)
ẑ  = E_ϕ(D_θ(sg(z)))
L_idem = E_x[ ||ẑ − sg(z)||² ]
L total = L_base + λ · L_idem
```

**Why the stop-gradient is critical:** without it, the gradient is (f(x) − f(f(x)))(1 − f'(f(x)))f'(x), which can be trivially zeroed by setting f'(z) = 1 everywhere — collapsing f to the **identity** (useless but loss-minimizing). With `sg`, this degenerate fixed point is eliminated. Empirically, removing it gives intermediate results.

**Manifold-alignment interpretation:** the objective implicitly aligns the *data-induced manifold* {E(x)} with the *generative manifold* {D(z)}. Sampling z ~ N(0, I) and re-encoding decoded samples can further close the OOD gap (not explored, flagged as future work).

## Results (highlights)

| Metric | VAE+Ours | VAE+IGN | VAE baseline |
|---|---|---|---|
| MNIST FID ↓ | **1.12** | 30.50 | 1.70 |
| CelebA FID ↓ | **14.31** | 54.68 | — |
| LFW MSE ↓ | **0.0038** | 0.0071 | 0.0073 |

- **Idempotency:** stable, artifact-free over 40 iterations; baselines/IGN drift.
- **Editing (MagicBrush):** idempotent VAE → higher DINO (0.828 vs 0.796), lower LPIPS (0.257 vs 0.287), higher SSIM (0.720 vs 0.680).
- **Synthetic (dSprites CAS):** 0.68 vs 0.39 (IGN) / 0.35.
- **λ is robust** across a wide range (higher λ generally improves FID while keeping competitive LPIPS).

## Limitations (important)

1. Extra **decode-encode cycle** increases training time & memory (relevant for LLM scale).
2. Prior-sampling direction explored, not fully validated.
3. Only evaluated on **small/medium image models**; whether benefits **scale to large architectures / training regimes is an open question**.
4. Some λ tuning may be needed for new architectures.

## Why it matters for encoder-decoder *language* models

The framework is **model-agnostic** and stated in terms of any encoder-decoder composition (E, D). T5Gemma-2 is exactly such a model. The repeated-application drift / geometric mismatch it targets maps to real seq2seq failure modes: **degenerate repetition**, **self-consistency drift**, and **identity/stability loss** across iterative decoding. This is a candidate *latent-consistency regularizer*, not a guaranteed win — the paper itself flags LLM scaling as untested.
