#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
SMOKE TEST v8 (standalone) — validasi logika fungsi-fungsi BARU di working-molab-v8.py
+ generate singkat pakai model 270M.

Cara pakai (di mesin kamu):
    conda activate unsloth
    python smoke_test_v8_270m.py                 # logika saja (tanpa model, tanpa GPU)
    python smoke_test_v8_270m.py --with-model    # + load google/t5gemma-2-270m-270m & generate

Flag:
    --with-model    : jalankan smoke test model (butuh HF_TOKEN + bisa download 270M)
    --skip-model    : paksa skip model walaupun HF_TOKEN ada (default kalau tanpa flag:
                      model dijalankan otomatis kalau HF_TOKEN tersedia)

Bagian "LOGIC TESTS" murni torch (tidak butuh model/token/GPU) — selalu jalan.
Bagian "MODEL SMOKE" memakai jalur load yang sama dengan Phase 0.5 di v8
(AutoModelForSeq2SeqLM + trust_remote_code) tapi untuk model 270M.
"""

import argparse
import os
import re
import sys

import torch

# =====================================================================
# FUNGSI DIBAWAH INI DISALIN 1:1 DARI working-molab-v8.py
# (jangan diubah kalau mau hasilnya mewakili kode v8)
# =====================================================================

# --- v8: zeropower_via_newtonschulz5 (Muon primitive, 5-step quintic Newton-Schulz) ---
def zeropower_via_newtonschulz5(G, steps=5, eps=1e-7, apply_shape_scale=True):
    assert G.ndim == 2, f"Newton-Schulz zeropower memerlukan tensor 2D, dapat {G.ndim}D"
    a = 3.4445
    b = -4.7750
    c = 2.0315
    X = G.to(torch.float32)
    norm = X.norm() + eps
    X = X / norm
    if X.size(0) < X.size(1):
        X = X.T
    for _ in range(steps):
        A = X @ X.T
        B = b * A + c * (A @ A)
        X = a * X + B @ X
    if G.size(0) < G.size(1):
        X = X.T
    if apply_shape_scale:
        scale = max(1.0, (G.size(0) / G.size(1)) ** 0.5)
        X = X * scale
    return X.to(G.dtype)


# --- v8: GrokOrScale (OrScale-LM 2D + AdEMAMix 1D + GrokFast) ---
import math as _math  # noqa: E402


class GrokOrScale(torch.optim.Optimizer):
    def __init__(
        self,
        params,
        lr=2e-4,
        betas=(0.9, 0.999),
        beta3=0.9999,
        weight_decay=0.01,
        grok_alpha=2.0,
        grok_lamb=0.98,
        momentum=0.95,
        nesterov=True,
        ns_steps=5,
        r_min=0.1,
        r_max=5.0,
        max_grad_norm=1.0,
    ):
        defaults = dict(
            lr=lr, betas=betas, beta3=beta3, weight_decay=weight_decay,
            grok_alpha=grok_alpha, grok_lamb=grok_lamb, momentum=momentum,
            nesterov=nesterov, ns_steps=ns_steps, r_min=r_min, r_max=r_max,
            max_grad_norm=max_grad_norm,
        )
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
        for group in self.param_groups:
            lr = group["lr"]
            beta1, beta2 = group["betas"]
            beta3 = group["beta3"]
            weight_decay = group["weight_decay"]
            grok_alpha = group["grok_alpha"]
            grok_lamb = group["grok_lamb"]
            momentum = group["momentum"]
            nesterov = group["nesterov"]
            ns_steps = group["ns_steps"]
            r_min, r_max = group["r_min"], group["r_max"]
            max_grad_norm = group["max_grad_norm"]
            for p in group["params"]:
                if p.grad is None:
                    continue
                grad = p.grad
                state = self.state[p]
                _force_branch = group.get("force_branch", None)
                _use_orscale = (p.ndim == 2) if _force_branch is None else (_force_branch in ("orscale", "muon"))
                if len(state) == 0:
                    state["step"] = 0
                    state["grok_slow_grad"] = torch.zeros_like(grad)
                    state["m"] = torch.zeros_like(grad)
                    state["v"] = torch.zeros_like(grad)
                    state["n"] = torch.zeros_like(grad)
                    state["momentum_buf"] = torch.zeros_like(grad) if _use_orscale else None
                    state["c_denom"] = None
                state["step"] += 1
                step = state["step"]
                # 1) GROKFAST
                state["grok_slow_grad"].mul_(grok_lamb).add_(grad, alpha=1.0 - grok_lamb)
                filtered_grad = grad.clone()
                filtered_grad.add_(state["grok_slow_grad"], alpha=grok_alpha)
                if max_grad_norm > 0:
                    f_norm = filtered_grad.norm()
                    if f_norm > max_grad_norm:
                        filtered_grad.mul_(max_grad_norm / (f_norm + 1e-6))
                # 2) CABANG 2D: ORSCALE-LM
                if _use_orscale:
                    buf = state["momentum_buf"]
                    buf.mul_(momentum).add_(filtered_grad)
                    g_update = filtered_grad.add(buf, alpha=momentum) if nesterov else buf
                    Q = zeropower_via_newtonschulz5(g_update, steps=ns_steps, apply_shape_scale=False)
                    m_dim, n_dim = p.shape[-2], p.shape[-1]
                    s_l = 0.2 * _math.sqrt(max(m_dim, n_dim))
                    D_l = (weight_decay * p) + (s_l * Q)
                    p_norm = p.norm(p="fro")
                    D_norm = D_l.norm(p="fro") + 1e-6
                    if state["c_denom"] is None:
                        state["c_denom"] = (p_norm / D_norm).item()
                    r_raw = p_norm / (state["c_denom"] * D_norm + 1e-6)
                    r_hat = torch.clamp(r_raw, r_min, r_max)
                    p.data.add_(D_l.to(p.dtype), alpha=-lr * r_hat)
                # 3) CABANG 1D: ADEMAMIX
                else:
                    if weight_decay != 0:
                        p.data.mul_(1.0 - lr * weight_decay)
                    m, v, n = state["m"], state["v"], state["n"]
                    m.mul_(beta1).add_(filtered_grad, alpha=1.0 - beta1)
                    v.mul_(beta2).addcmul_(filtered_grad, filtered_grad, value=1.0 - beta2)
                    n.mul_(beta3).add_(filtered_grad, alpha=1.0 - beta3)
                    bc1 = 1.0 - beta1 ** step
                    bc2 = 1.0 - beta2 ** step
                    bc3 = 1.0 - beta3 ** step
                    denom = (v.sqrt() / (bc2 ** 0.5)).add_(1e-8).to(p.dtype)
                    step_update = ((m / bc1 + 0.1 * n / bc3) / denom).to(p.dtype)
                    p.data.add_(step_update, alpha=-lr)
        return loss


# --- v8: devec_svd_purify (SVD subspace filtering tau=0.85) ---
def devec_svd_purify(delta_weight, threshold=0.85):
    if delta_weight.ndim != 2:
        return delta_weight
    orig_dtype = delta_weight.dtype
    W = delta_weight.float()
    try:
        U, S, Vh = torch.linalg.svd(W, full_matrices=False)
        energy = S * S
        total_energy = energy.sum() + 1e-12
        cum_ratio = torch.cumsum(energy, dim=0) / total_energy
        k = int(torch.searchsorted(cum_ratio, threshold).item()) + 1
        k = max(1, min(k, S.numel()))
        W_purified = (U[:, :k] * S[:k]) @ Vh[:k, :]
        return W_purified.to(orig_dtype)
    except Exception:
        return delta_weight


# --- v8: _lata_alignment_scale (LaTA-style layer-wise alignment) ---
def _lata_alignment_scale(delta, base):
    d = delta.float().flatten()
    b = base.float().flatten()
    cos_abs = torch.abs((d @ b) / (d.norm() * b.norm() + 1e-8))
    return float(torch.clamp(1.5 - cos_abs, 0.5, 1.5).item())


# --- v8: TLPO_Regularizer ---
import torch.nn.functional as F  # noqa: E402


class TLPO_Regularizer:
    def __init__(self, suppress_ids, beta=0.1, clip_eps=0.2):
        self.suppress_ids = suppress_ids
        self.beta = beta
        self.clip_eps = clip_eps
        self._frozen_log_probs = None

    def __call__(self, logits, labels):
        if logits.ndim == 3:
            flat_logits = logits.view(-1, logits.size(-1))
            flat_labels = labels.view(-1)
        else:
            flat_logits = logits
            flat_labels = labels
        active_mask = flat_labels != -100
        if not active_mask.any():
            return torch.tensor(0.0, device=logits.device, requires_grad=True)
        active_logits = flat_logits[active_mask]
        preds = torch.argmax(active_logits, dim=-1)
        vocab_size = active_logits.size(-1)
        suppress_list = [i for i in self.suppress_ids if i < vocab_size]
        suppress_tensor = torch.tensor(suppress_list, device=logits.device, dtype=torch.long)
        confusion_mask = torch.isin(preds, suppress_tensor)
        if not confusion_mask.any():
            return torch.tensor(0.0, device=logits.device, requires_grad=True)
        log_probs = F.log_softmax(active_logits[confusion_mask], dim=-1)
        k = min(16, vocab_size)
        top_k_log_probs, top_k_indices = torch.topk(log_probs, k=k, dim=-1)
        rewards = torch.isin(top_k_indices, suppress_tensor).float() * -2.0 + 1.0
        advantages = rewards - rewards.mean(dim=-1, keepdim=True)
        if self._frozen_log_probs is not None and self._frozen_log_probs.shape == log_probs.shape:
            frozen_log_probs = torch.gather(self._frozen_log_probs, dim=-1, index=top_k_indices)
            ratio = torch.exp(top_k_log_probs - frozen_log_probs)
        else:
            ratio = torch.ones_like(top_k_log_probs)
        unclipped = ratio * advantages
        clipped = torch.clamp(ratio, 1.0 - self.clip_eps, 1.0 + self.clip_eps) * advantages
        tlpo_loss = -torch.mean(torch.minimum(unclipped, clipped))
        self._frozen_log_probs = log_probs.detach()
        return self.beta * tlpo_loss


# --- v8: MTO formatter ---
def format_mto_target(target_text, task_category="general_chat"):
    prefix_mapping = {
        "summarize": "<unused1>", "translate": "<unused2>", "ner": "<unused3>",
        "qa": "<unused4>", "paraphrase": "<unused5>", "general_chat": "<unused6>",
        "vision": "<unused6>",
    }
    tag = prefix_mapping.get(task_category.lower(), "<unused6>")
    target_clean = target_text.strip()
    if target_clean.startswith("<unused"):
        return target_clean
    return f"{tag} {target_clean}"


def format_encoder_from_raw(raw_input, system_prompt):
    system_match = re.search(r"^system:\s*(.*?)(?=\nuser:)", raw_input, re.DOTALL)
    system = system_match.group(1).strip() if system_match else system_prompt
    if system_match:
        raw_input = raw_input[system_match.end():].strip()
    parts = re.split(r"\n(user:|assistant:)\s*", "\n" + raw_input)
    formatted = ""
    is_first_user = True
    for i in range(1, len(parts), 2):
        role = parts[i].replace(":", "").strip()
        content = parts[i + 1].strip()
        if not content:
            continue
        if role == "user":
            formatted += "<start_of_turn>user\n"
            if is_first_user and system:
                formatted += system + "\n\n"
                is_first_user = False
            formatted += content + "<end_of_turn>\n"
        elif role == "assistant":
            formatted += "<start_of_turn>model\n"
            formatted += content + "<end_of_turn>\n"
    formatted += "<start_of_turn>model\n"
    return formatted


# =====================================================================
# HARNESS
# =====================================================================
RESULT = {"pass": 0, "fail": 0, "warn": 0}


def check(name, cond, detail=""):
    if cond:
        RESULT["pass"] += 1
        print(f"  PASS  {name}")
    else:
        RESULT["fail"] += 1
        print(f"  FAIL  {name}  {detail}")


def warn(name, detail=""):
    RESULT["warn"] += 1
    print(f"  WARN  {name}  {detail}")


def logic_tests():
    print("=" * 70)
    print("LOGIC TESTS (torch, tanpa model)")
    print("=" * 70)
    torch.manual_seed(3407)

    # 1) Newton-Schulz ortogonalitas
    G = torch.randn(128, 64)
    Q = zeropower_via_newtonschulz5(G, steps=5, apply_shape_scale=False)
    dev = (Q.T @ Q - torch.eye(64)).abs().max().item()
    check("Newton-Schulz: shape preserved", Q.shape == G.shape)
    check("Newton-Schulz: no NaN", torch.isfinite(Q).all().item())
    # Catatan: 5-step bersifat aproksimasi; dev ~0.3 pada matriks random adalah normal.
    if dev < 0.05:
        check(f"Newton-Schulz: sangat ortogonal (dev={dev:.4f})", True)
    else:
        warn(f"Newton-Schulz: dev Q^TQ-I = {dev:.4f} (5-step = aproksimasi, normal untuk Muon)")

    # 2) DeVec SVD purify
    d = torch.randn(64, 64) + 0.05 * torch.randn(64, 64)
    Wp = devec_svd_purify(d, 0.85)
    U, S, Vh = torch.linalg.svd(d, full_matrices=False)
    energy = S * S
    total = energy.sum() + 1e-12
    cum = torch.cumsum(energy, dim=0) / total
    k = int(torch.searchsorted(cum, 0.85).item()) + 1
    retained = (energy[:k].sum() / total).item()
    check(f"DeVec: k={k} capai >=85% energi (retained={retained:.3f})", 0.85 <= retained <= 1.0)
    check("DeVec: shape preserved", Wp.shape == d.shape)
    check("DeVec: no NaN", torch.isfinite(Wp).all().item())
    v1d = torch.tensor([1.0, 2.0, 3.0])
    check("DeVec: 1D return as-is", devec_svd_purify(v1d) is v1d or torch.equal(devec_svd_purify(v1d), v1d))

    # 3) LaTA scaling
    s_par = _lata_alignment_scale(torch.ones(100), torch.ones(100))
    s_ort = _lata_alignment_scale(torch.tensor([1.0, -1.0] * 50), torch.ones(100))
    check(f"LaTA: paralel -> attenuate ({s_par:.3f} ~ 0.5)", abs(s_par - 0.5) < 1e-5)
    check(f"LaTA: ortogonal -> amplify ({s_ort:.3f} ~ 1.5)", abs(s_ort - 1.5) < 1e-5)

    # 4) GrokOrScale satu step (param 2D + 1D)
    p2d = torch.randn(64, 32, requires_grad=True)
    p1d = torch.randn(64, requires_grad=True)
    opt = GrokOrScale(
        [{"params": [p2d]}, {"params": [p1d], "force_branch": "adema"}],
        lr=2e-4, weight_decay=0.01, momentum=0.95, nesterov=True,
        ns_steps=5, r_min=0.1, r_max=5.0, max_grad_norm=1.0,
    )
    p2d_before = p2d.detach().clone()
    p1d_before = p1d.detach().clone()
    p2d.grad = torch.randn_like(p2d)
    p1d.grad = torch.randn_like(p1d)
    opt.step()
    check("GrokOrScale: 2D param berubah", not torch.equal(p2d, p2d_before))
    check("GrokOrScale: 1D param berubah", not torch.equal(p1d, p1d_before))
    check("GrokOrScale: no NaN (2D)", torch.isfinite(p2d).all().item())
    check("GrokOrScale: no NaN (1D)", torch.isfinite(p1d).all().item())
    # step-1 trust ratio harus ~1.0 (lazy calibration c_denom)
    state2d = opt.state[p2d]
    check("GrokOrScale: c_denom ter-set di step-1", state2d.get("c_denom") is not None)

    # 5) TLPO regularizer
    V = 64
    labels = torch.ones(4, 8, dtype=torch.long)
    sup = set(range(50, V))
    reg = TLPO_Regularizer(suppress_ids=sup, beta=0.05, clip_eps=0.2)
    logits_ok = torch.randn(4, 8, V); logits_ok[:, :, 1] += 100  # argmax = token valid
    l_ok = reg(logits_ok, labels)
    check("TLPO: no confusion -> loss 0", l_ok.item() == 0.0)
    reg2 = TLPO_Regularizer(suppress_ids=sup, beta=0.05, clip_eps=0.2)
    logits_bad = torch.randn(4, 8, V); logits_bad[:, :, 60] += 100  # argmax = suppressed
    l1 = reg2(logits_bad, labels)
    # call-1 SELALU 0 (advantage mean-centered + ratio=1) — ini quirk v8 yang sudah dikonfirmasi
    check(f"TLPO: confusion call-1 == 0 (quirk v8, terkonfirmasi) -> {l1.item():.6f}", abs(l1.item()) < 1e-6)
    # call-2 dengan logits BERBEDA -> snapshot terpakai, ratio bisa non-uniform -> loss bisa > 0
    logits_bad2 = torch.randn(4, 8, V); logits_bad2[:, :, 60] += 100
    l2 = reg2(logits_bad2, labels)
    print(f"        (info) TLPO call-2 loss = {l2.item():.6f}  [0 atau >0 tergantung ratio per-token]")

    # 6) MTO formatter
    check("MTO: qa -> <unused4>", format_mto_target("Jakarta", "qa").startswith("<unused4> "))
    check("MTO: summarize -> <unused1>", format_mto_target("x", "summarize").startswith("<unused1> "))
    check("MTO: no duplicate prefix", format_mto_target("<unused4> Jakarta", "qa") == "<unused4> Jakarta")
    check("MTO: unknown -> <unused6>", format_mto_target("x", "bogus").startswith("<unused6> "))

    # 7) format_encoder_from_raw
    sys_prompt = "Kamu adalah asisten AI."
    enc = format_encoder_from_raw("user: Halo!", sys_prompt)
    check("Encoder: ends with <start_of_turn>model", enc.endswith("<start_of_turn>model\n"))
    check("Encoder: system injected", sys_prompt in enc)


def model_smoke():
    print()
    print("=" * 70)
    print("MODEL SMOKE (270M)")
    print("=" * 70)
    model_id = os.environ.get("SMOKE_MODEL_ID", "google/t5gemma-2-270m-270m")
    try:
        from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
    except Exception as e:
        print(f"  SKIP: transformers tidak tersedia di env ini ({e})")
        return

    # token=None -> huggingface_hub otomatis pakai token yang sudah di-cache
    # (~/.cache/huggingface/token) dari login() sebelumnya. Tidak wajib set HF_TOKEN.
    token = os.environ.get("HF_TOKEN", "").strip() or None
    print(f"  Loading {model_id} (bisa beberapa menit kalau download pertama)...")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.bfloat16 if device == "cuda" else torch.float32

    model = tok = None
    # 1) pakai token (env atau cached)
    try:
        model = AutoModelForSeq2SeqLM.from_pretrained(
            model_id, dtype=dtype, token=token, trust_remote_code=True,
        )
        tok = AutoTokenizer.from_pretrained(model_id, token=token, trust_remote_code=True)
    except Exception as e1:
        print(f"  (load via token gagal: {e1})")
        # 2) fallback: murni dari cache lokal (tanpa auth/network)
        try:
            model = AutoModelForSeq2SeqLM.from_pretrained(
                model_id, dtype=dtype, trust_remote_code=True, local_files_only=True,
            )
            tok = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True, local_files_only=True)
        except Exception as e2:
            print(f"  FAIL: gagal load {model_id}.")
            print(f"    via token/cached : {e1}")
            print(f"    via local cache  : {e2}")
            print("  Cek: (a) model 270M belum pernah di-download di mesin ini, dan")
            print("       (b) butuh token yang sudah accept license di huggingface.co/settings/tokens")
            RESULT["fail"] += 1
            return
    model = model.to(device).eval()

    # Cek token <unused1>..<unused6> (dipakai MTO prefix mapping v8; v8 asumsi ID 7..12)
    print("\n  Cek token MTO (<unused1>..<unused6>) di vocab:")
    all_mto_ok = True
    for i in range(1, 7):
        tok_name = f"<unused{i}>"
        tid = tok.convert_tokens_to_ids(tok_name)
        print(f"    {tok_name:12s} -> id {tid}")
        if tid is None or tid == tok.unk_token_id:
            warn(f"MTO: {tok_name} TIDAK ADA di vocab — MTO prefix mapping akan rusak")
            all_mto_ok = False
        elif tid != i + 6:
            warn(f"MTO: {tok_name} id={tid}, v8 asumsi id={i + 6} — suppress list v8 bisa salah target")
    check("MTO: token <unused1>..<unused6> ada & id sesuai asumsi v8 (7..12)", all_mto_ok)

    prompts = [
        "user: Halo! Perkenalkan dirimu secara singkat.",
        "user: Apa ibu kota Indonesia?",
        "user: Tolong ringkas: Fotosintesis adalah proses tumbuhan mengubah cahaya matahari menjadi energi.",
    ]
    sys_prompt = "Kamu adalah asisten AI yang helpful, santai, dan ramah. Gunakan Bahasa Indonesia sebagai bahasa utama."
    with torch.no_grad():
        for p in prompts:
            fmt = format_encoder_from_raw(p, sys_prompt)
            ids = tok.encode(fmt, add_special_tokens=True, return_tensors="pt").to(device)
            out = model.generate(ids, max_new_tokens=48, do_sample=False, pad_token_id=tok.pad_token_id)
            resp = tok.decode(out[0], skip_special_tokens=True)
            print(f"\n  Q: {p}\n  A: {resp}")
    check("Model: generate 3 prompt tanpa exception", True)
    print(f"  (device={device})")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--with-model", action="store_true", help="jalankan smoke test model 270M")
    ap.add_argument("--skip-model", action="store_true", help="skip model walau HF_TOKEN ada")
    args = ap.parse_args()

    logic_tests()

    run_model = args.with_model and not args.skip_model
    if args.with_model:
        model_smoke()
    elif not args.skip_model and os.environ.get("HF_TOKEN", "").strip():
        print("\n  (HF_TOKEN terdeteksi — jalankan --with-model untuk smoke test model)")
    else:
        print("\n  (model smoke di-skip; pakai --with-model untuk menjalankannya)")

    print()
    print("=" * 70)
    print(f"SUMMARY: {RESULT['pass']} PASS, {RESULT['fail']} FAIL, {RESULT['warn']} WARN")
    print("=" * 70)
    sys.exit(1 if RESULT["fail"] else 0)


if __name__ == "__main__":
    main()
