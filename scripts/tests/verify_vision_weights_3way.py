# /// script
# requires-python = ">=3.10"
# dependencies = [
#     "torch", "transformers==5.12.1", "huggingface-hub",
#     "unsloth_zoo @ git+https://github.com/daruoktab/unsloth-zoo.git",
#     "unsloth @ git+https://github.com/daruoktab/unsloth.git",
# ]
# ///

import marimo
__generated_with = "0.23.13"
app = marimo.App(width="full", css_file="/usr/local/_marimo/custom.css", auto_download=["html"])


@app.cell
def _():
    import marimo as mo
    return (mo,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # 🔍 Verifikasi 3-Arah SigLIP Vision Tower & Projector

    Compare weights `vision_tower` (SigLIP) + `multi_modal_projector` dari 3 sumber:

    - **[A]** `daruokta/t5gemma-2-4b-4b-instruct-chat-indo-v4-unsloth/merged_bf16` (v6 text hasil)
    - **[B]** `google/t5gemma-2-4b-4b` (original T5Gemma2)
    - **[C]** `google/gemma-3-4b-it` (Gemma 3 IT, kandidat cangkok)

    **Keputusan dari output:**
    - A vs B ≈ 0 → v6 text tidak merusak vision (bocor tidak ada efek)
    - A vs B > 1e-4 → ada perubahan, perlu reset
    - B vs C beda signifikan → cangkok Gemma 3 IT worth it
    - B vs C ≈ 0 → skip cangkok
    """)
    return


@app.cell
def _():
    import os, sys, torch
    from transformers import AutoModelForSeq2SeqLM, AutoModelForCausalLM

    V6_REPO = "daruokta/t5gemma-2-4b-4b-instruct-chat-indo-v4-unsloth"
    V6_SUBFOLDER = "merged_bf16"
    ORIG_T5GEMMA2 = "google/t5gemma-2-4b-4b"
    GEMMA3_IT = "google/gemma-3-4b-it"
    VISION_KEYWORDS = ["vision_tower", "multi_modal_projector"]
    return (os, sys, torch, AutoModelForSeq2SeqLM, AutoModelForCausalLM,
            V6_REPO, V6_SUBFOLDER, ORIG_T5GEMMA2, GEMMA3_IT, VISION_KEYWORDS)


@app.cell
def _(mo):
    hf_token_input = mo.ui.text(
        label="Hugging Face Token (HF_TOKEN)", value="", full_width=True)
    hf_token_input
    return (hf_token_input,)


@app.cell
def _(hf_token_input, mo, os):
    from huggingface_hub import login
    mo.stop(
        not hf_token_input.value,
        mo.md("⚠️ *Please enter your Hugging Face token in the input above to authenticate and load gated models.*"),
    )
    try:
        os.environ["HF_TOKEN"] = hf_token_input.value
        login(token=hf_token_input.value)
        status = mo.md("✅ **Successfully authenticated with Hugging Face Hub!** You can now load gated models.")
    except Exception as e:
        status = mo.md(f"❌ **Authentication failed:** {e}")
    status


@app.cell
def _(VISION_KEYWORDS, torch):
    def extract_vision_params(model, model_type="t5gemma2"):
        """Extract vision_tower + projector params. Normalisasi path prefix."""
        params = {}
        for name, param in model.named_parameters():
            if not any(k in name for k in VISION_KEYWORDS):
                continue
            clean = name
            if model_type == "t5gemma2":
                if name.startswith("model.encoder."):
                    clean = name[len("model.encoder."):]
                elif name.startswith("encoder."):
                    clean = name[len("encoder."):]
            elif model_type == "gemma3":
                if name.startswith("model."):
                    clean = name[len("model."):]
            params[clean] = param.detach().cpu().clone()
        return params

    def compare_params(set_a, set_b, label_a, label_b):
        results = []
        all_names = sorted(set(set_a.keys()) | set(set_b.keys()))
        for name in all_names:
            pa = set_a.get(name); pb = set_b.get(name)
            if pa is None:
                results.append((name, None, tuple(pb.shape), None, None, f"Hanya di {label_b}")); continue
            if pb is None:
                results.append((name, tuple(pa.shape), None, None, None, f"Hanya di {label_a}")); continue
            if pa.shape != pb.shape:
                results.append((name, tuple(pa.shape), tuple(pb.shape), None, None, "SHAPE MISMATCH")); continue
            diff = (pa.float() - pb.float()).abs()
            mx = diff.max().item(); mn = diff.mean().item()
            v = "IDENTIK" if mx < 1e-7 else ("≈sama" if mx < 1e-4 else "BERBEDA")
            results.append((name, tuple(pa.shape), tuple(pb.shape), mx, mn, v))
        return results

    def print_comparison(results, title):
        print(f"\n{'='*100}")
        print(f"  {title}")
        print(f"{'='*100}")
        print(f"{'Module':55s} {'Shape A':15s} {'Shape B':15s} {'MaxDiff':>10s} {'MeanDiff':>10s}  Verdict")
        print("-"*100)
        for name, sa, sb, mx, mn, v in results:
            sa_s = str(sa) if sa else "-"; sb_s = str(sb) if sb else "-"
            mx_s = f"{mx:.2e}" if mx is not None else "-"; mn_s = f"{mn:.2e}" if mn is not None else "-"
            print(f"{name:55s} {sa_s:15s} {sb_s:15s} {mx_s:>10s} {mn_s:>10s}  {v}")
        identical = sum(1 for r in results if r[5] == "IDENTIK")
        similar = sum(1 for r in results if r[5] == "≈sama")
        diff = sum(1 for r in results if r[5] == "BERBEDA")
        mismatch = sum(1 for r in results if r[5] == "SHAPE MISMATCH")
        only = sum(1 for r in results if "Hanya" in str(r[5]))
        print("-"*100)
        print(f"  Ringkasan: IDENTIK={identical}, ≈sama={similar}, BERBEDA={diff}, "
              f"SHAPE_MISMATCH={mismatch}, HANYA_DI_SATU={only}")
    return (extract_vision_params, compare_params, print_comparison)


@app.cell
def _(V6_REPO, V6_SUBFOLDER, ORIG_T5GEMMA2, GEMMA3_IT,
      AutoModelForSeq2SeqLM, AutoModelForCausalLM, torch, os,
      extract_vision_params, mo):
    # Load 3 model + extract vision params
    _token = os.environ.get("HF_TOKEN")
    if not _token:
        print("⚠️ Masukkan HF token di cell atas dulu!")
        mo.stop(True, mo.md("❌ Masukkan token HF di cell atas."))

    print("="*90)
    print("  LOADING 3 MODEL (butuh VRAM besar — Molab 96GB)")
    print("="*90)

    print(f"\n[A] Loading v6 text merged: {V6_REPO}/{V6_SUBFOLDER} ...")
    model_a = AutoModelForSeq2SeqLM.from_pretrained(
        V6_REPO, subfolder=V6_SUBFOLDER, torch_dtype=torch.bfloat16, token=_token)
    print(f"    ✅ {model_a.__class__.__name__}")

    print(f"\n[B] Loading original T5Gemma2: {ORIG_T5GEMMA2} ...")
    model_b = AutoModelForSeq2SeqLM.from_pretrained(
        ORIG_T5GEMMA2, torch_dtype=torch.bfloat16, token=_token)
    print(f"    ✅ {model_b.__class__.__name__}")

    print(f"\n[C] Loading Gemma 3 4B IT: {GEMMA3_IT} ...")
    model_c = AutoModelForCausalLM.from_pretrained(
        GEMMA3_IT, torch_dtype=torch.bfloat16, token=_token)
    print(f"    ✅ {model_c.__class__.__name__}")

    print("\n" + "="*90)
    print("  EXTRACTING VISION_TOWER (SigLIP) + PROJECTOR PARAMS")
    print("="*90)

    print("\n[A] v6 text merged — extracting ...")
    vt_a = extract_vision_params(model_a, "t5gemma2")
    print(f"    vision_tower + projector: {len(vt_a)} params")

    print("\n[B] original T5Gemma2 — extracting ...")
    vt_b = extract_vision_params(model_b, "t5gemma2")
    print(f"    vision_tower + projector: {len(vt_b)} params")

    print("\n[C] Gemma 3 4B IT — extracting ...")
    vt_c = extract_vision_params(model_c, "gemma3")
    print(f"    vision_tower + projector: {len(vt_c)} params")

    # Free GPU memory
    del model_a, model_b, model_c
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    print("\n✅ Extract selesai, model di-free dari GPU.")
    return (vt_a, vt_b, vt_c)


@app.cell
def _(vt_a, vt_b, vt_c, compare_params, print_comparison):
    # COMPARISON 1: v6 merged vs original T5Gemma2
    print("\n\n" + "#"*90)
    print("# COMPARISON 1: v6 TEXT MERGED [A] vs ORIGINAL T5GEMMA2 [B]")
    print("# → Apakah training text-only v6 mengubah vision_tower/projector?")
    print("#"*90)
    r_ab = compare_params(vt_a, vt_b, "v6 merged", "original T5Gemma2")
    print_comparison(r_ab, "VISION_TOWER + PROJECTOR: v6 merged [A] vs original T5Gemma2 [B]")

    # COMPARISON 2: original T5Gemma2 vs Gemma 3 IT
    print("\n\n" + "#"*90)
    print("# COMPARISON 2: ORIGINAL T5GEMMA2 [B] vs GEMMA 3 4B IT [C]")
    print("# → Apakah SigLIP Gemma 3 IT berbeda (lebih instruct-ready)?")
    print("#"*90)
    r_bc = compare_params(vt_b, vt_c, "original T5Gemma2", "Gemma 3 IT")
    print_comparison(r_bc, "VISION_TOWER + PROJECTOR: original T5Gemma2 [B] vs Gemma 3 IT [C]")

    # COMPARISON 3: v6 merged vs Gemma 3 IT
    print("\n\n" + "#"*90)
    print("# COMPARISON 3: v6 TEXT MERGED [A] vs GEMMA 3 4B IT [C]")
    print("# → Beda total kalau langsung timpa v6 dengan Gemma 3 IT")
    print("#"*90)
    r_ac = compare_params(vt_a, vt_c, "v6 merged", "Gemma 3 IT")
    print_comparison(r_ac, "VISION_TOWER + PROJECTOR: v6 merged [A] vs Gemma 3 IT [C]")
    return (r_ab, r_bc, r_ac)


@app.cell
def _(r_ab, r_bc, mo):
    # RINGKASAN & REKOMENDASI
    print("\n\n" + "="*90)
    print("  RINGKASAN & REKOMENDASI")
    print("="*90)

    vt_ab_max = max((r[3] for r in r_ab if r[3] is not None), default=0)
    print(f"\n1. v6 merged vs original T5Gemma2:")
    print(f"   max_diff = {vt_ab_max:.2e}")
    if vt_ab_max < 1e-7:
        print("   → ✅ IDENTIK — training text v6 TIDAK mengubah vision (bocor tidak ada efek)")
        print("   → Tidak perlu menambat/reset vision_tower")
    elif vt_ab_max < 1e-4:
        print("   → ⚠️ ≈sama — numerical noise negligible, tidak perlu reset")
    else:
        print("   → ❌ BERBEDA — training text v6 mengubah vision, perlu reset ke original")

    vt_bc_max = max((r[3] for r in r_bc if r[3] is not None), default=0)
    print(f"\n2. original T5Gemma2 vs Gemma 3 IT:")
    print(f"   max_diff = {vt_bc_max:.2e}")
    if vt_bc_max < 1e-7:
        print("   → SigLIP IDENTIK — cangkok Gemma 3 IT TIDAK ada manfaat")
    elif vt_bc_max < 1e-4:
        print("   → SigLIP ≈sama — cangkok minim manfaat")
    else:
        print("   → SigLIP BERBEDA — cangkok Gemma 3 IT WORTH IT (lebih instruct-ready)")

    print("\n" + "="*90)
    print("  Verifikasi selesai. Tidak ada file disimpan.")
    print("  Gunakan output di atas untuk keputusan: cangkok / reset / skip.")
    print("="*90)
    return


@app.cell
def _(V6_REPO, V6_SUBFOLDER, GEMMA3_IT, ORIG_T5GEMMA2,
      AutoModelForSeq2SeqLM, AutoModelForCausalLM, torch, os):
    # =====================================================================
    # CELL CANGKOK: Vision Tower + Projector dari Gemma 3 IT → v6 merged
    # =====================================================================
    CANGKOK_REPO = "daruokta/t5gemma-2-4b-4b-instruct-chat-indo-v4-vision-cangkok"
    _token = os.environ.get("HF_TOKEN")

    print("="*90)
    print("  CANGKOK: SigLIP + Projector dari Gemma 3 4B IT → v6 merged_bf16")
    print("="*90)

    # 1. Load v6 merged (target)
    print(f"\n[A] Loading v6 merged: {V6_REPO}/{V6_SUBFOLDER} ...")
    model_tgt = AutoModelForSeq2SeqLM.from_pretrained(
        V6_REPO, subfolder=V6_SUBFOLDER, torch_dtype=torch.bfloat16, token=_token)
    print(f"    ✅ {model_tgt.__class__.__name__}")

    # 2. Load Gemma 3 IT (source)
    print(f"\n[C] Loading Gemma 3 4B IT: {GEMMA3_IT} ...")
    model_src = AutoModelForCausalLM.from_pretrained(
        GEMMA3_IT, torch_dtype=torch.bfloat16, token=_token)
    print(f"    ✅ {model_src.__class__.__name__}")

    # 3. Build source param dict (gemma3 path → normalized)
    src_params = {}
    for name, param in model_src.named_parameters():
        if "vision_tower" in name or "multi_modal_projector" in name:
            clean = name[len("model."):] if name.startswith("model.") else name
            src_params[clean] = param.detach().cpu()

    print(f"\n  Source (Gemma 3 IT): {len(src_params)} vision params")

    # 4. Cangkok: copy source → target (path mapping t5gemma2)
    print("\n  Melakukan cangkok...")
    cangkok_count = 0
    skip_count = 0
    for name, param in model_tgt.named_parameters():
        if "vision_tower" not in name and "multi_modal_projector" not in name:
            continue
        # Normalisasi target path
        clean = name
        if name.startswith("model.encoder."):
            clean = name[len("model.encoder."):]
        elif name.startswith("encoder."):
            clean = name[len("encoder."):]

        if clean in src_params:
            src = src_params[clean]
            if src.shape == param.shape:
                param.data.copy_(src.to(param.device, param.dtype))
                cangkok_count += 1
            else:
                print(f"    ⚠️ SHAPE MISMATCH {clean}: {src.shape} vs {param.shape}")
                skip_count += 1
        else:
            print(f"    ⚠️ Tidak ditemukan di source: {clean}")
            skip_count += 1

    print(f"  ✅ Cangkok: {cangkok_count} params, skip: {skip_count}")

    # 5. Verify cangkok berhasil (compare target vs source, harus ≈ 0 sekarang)
    print("\n  Verifikasi cangkok...")
    verify_ok = 0
    verify_fail = 0
    for name, param in model_tgt.named_parameters():
        if "vision_tower" not in name and "multi_modal_projector" not in name:
            continue
        clean = name
        if name.startswith("model.encoder."):
            clean = name[len("model.encoder."):]
        elif name.startswith("encoder."):
            clean = name[len("encoder."):]
        if clean in src_params:
            src = src_params[clean]
            diff = (param.detach().cpu().float() - src.float()).abs().max().item()
            if diff < 1e-6:
                verify_ok += 1
            else:
                print(f"    ❌ Verify fail {clean}: diff={diff:.2e}")
                verify_fail += 1
    print(f"  ✅ Verify: {verify_ok} OK, {verify_fail} fail")

    # 6. Save lokal lalu upload ke HF (publik)
    print(f"\n  Saving & uploading ke HF: {CANGKOK_REPO} (PUBLIC)...")
    from huggingface_hub import HfApi, create_repo
    api = HfApi(token=_token)
    create_repo(repo_id=CANGKOK_REPO, repo_type="model", private=False, exist_ok=True, token=_token)

    local_save = "/tmp/v6_vision_cangkok"
    os.makedirs(local_save, exist_ok=True)
    print(f"  Saving lokal ke {local_save}...")
    model_tgt.save_pretrained(local_save, safe_serialization=True)

    # Upload processor dari original T5Gemma2 (punya full preprocessor_config.json)
    # v6 merged_bf16 hanya punya tokenizer (text-only), tidak ada image processor
    from transformers import AutoProcessor
    processor = AutoProcessor.from_pretrained(ORIG_T5GEMMA2, token=_token)
    processor.save_pretrained(local_save)

    print(f"  Uploading ke {CANGKOK_REPO}...")
    api.upload_folder(
        folder_path=local_save,
        repo_id=CANGKOK_REPO,
        repo_type="model",
        commit_message="Cangkok SigLIP + projector dari Gemma 3 4B IT ke v6 merged_bf16",
    )
    print(f"\n  ✅ BERHASIL! Model cangkok tersimpan di: {CANGKOK_REPO}")
    print(f"     Gunakan repo ini sebagai MODEL_NAME + SUBFOLDER='' di vision training.")

    # Cleanup
    del model_tgt, model_src
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return


if __name__ == "__main__":
    app.run()