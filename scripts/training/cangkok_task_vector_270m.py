"""
Cangkok Task Vector: T5Gemma-2 270M + Gemma 3 IT
=================================================
Strategi: Task Arithmetic (Ilharco 2023)
  τ = θ_IT - θ_PT  →  θ_target += α * τ

Temuan dari analisis (test_tokenizer_template.py):
  - Tokenizer IT dan T5Gemma2 IDENTIK (vocab 262144, token IDs sama)
  - Satu perbedaan: ID 256001 = <unused99> di IT, = <image_soft_token> di T5G
  - Pakai tokenizer T5Gemma2 native (bukan IT) agar vision token ID konsisten

Layer-wise alpha (dari insight LATA paper):
  - Early layers (0-33%): task knowledge sedikit → alpha * 0.5
  - Mid   layers (33-65%): transisi → alpha * 0.75
  - Later layers (65-100%): task knowledge dominan → alpha * 1.0

Skip k_proj/v_proj (dari analisis Merged Attention):
  - T5Gemma2 Merged Attention: K, V menerima concat(decoder, encoder)
  - Task vector dari Gemma 3 IT hanya training K/V untuk decoder context
  - Inject k/v ke merged attention = distribusi input mismatch → skip
"""

import torch
from transformers import (
    AutoModelForCausalLM,
    AutoModelForSeq2SeqLM,
    AutoTokenizer,
    PreTrainedTokenizerFast,
)
import os
from typing import cast

# ─────────────────────────────────────────────
# KONFIGURASI
# ─────────────────────────────────────────────
T5G_BASE = "google/t5gemma-2-270m-270m"
G3_PT    = "google/gemma-3-270m"
G3_IT    = "google/gemma-3-270m-it"
OUTPUT_PATH = "models/t5gemma2-270m-task-vector"

BASE_ALPHA = 0.3  # Koefisien suntikan global

# Parameter yang SKIP karena merged attention mismatch
# k_proj dan v_proj menerima concat(decoder, encoder) di T5Gemma2
# Task vector IT hanya training untuk pure decoder input → skip
SKIP_PARAM_SUFFIXES = {"k_proj.weight", "k_proj.bias", "v_proj.weight", "v_proj.bias"}


def get_layer_alpha(layer_idx: int, num_layers: int, base_alpha: float = 0.3) -> float:
    """
    Layer-wise alpha berdasarkan insight LATA paper.
    Task-specific knowledge (instruction following) dominan di layer akhir.
    Layer awal lebih banyak encode input processing (dekat instruction vector).
    """
    ratio = layer_idx / max(num_layers - 1, 1)
    if ratio < 0.33:
        return base_alpha * 0.5   # Early: 0.15
    elif ratio < 0.65:
        return base_alpha * 0.75  # Mid:   0.225
    else:
        return base_alpha * 1.0   # Later: 0.3  ← most task-specific


def task_vector_transplant() -> None:
    print(f"--- Task Vector Transplant (Base Alpha={BASE_ALPHA}) ---")
    print(f"Skip params: {SKIP_PARAM_SUFFIXES}")

    # ── Load Models ──────────────────────────────────────────────────────
    print(f"\nLoading T5Gemma-2 Base: {T5G_BASE}...")
    target_model = AutoModelForSeq2SeqLM.from_pretrained(
        T5G_BASE,
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,
        device_map="cpu",
    )

    print(f"Loading Gemma 3 PT (Base): {G3_PT}...")
    pt_model = AutoModelForCausalLM.from_pretrained(
        G3_PT,
        torch_dtype=torch.bfloat16,
        device_map="cpu",
    )

    print(f"Loading Gemma 3 IT (Instruct): {G3_IT}...")
    it_model = AutoModelForCausalLM.from_pretrained(
        G3_IT,
        torch_dtype=torch.bfloat16,
        device_map="cpu",
    )

    # ── Injection ke Decoder Layers ───────────────────────────────────────
    num_layers: int = target_model.config.decoder.num_hidden_layers
    print(f"\nDecoder layers: {num_layers}")
    print("Injecting Task Vector into Decoder layers (skip k/v, layer-wise alpha)...\n")

    skipped_names: set[str] = set()
    shape_mismatch_names: set[str] = set()

    with torch.no_grad():
        for i in range(num_layers):
            layer_alpha = get_layer_alpha(i, num_layers, BASE_ALPHA)

            it_layer     = it_model.model.layers[i]
            pt_layer     = pt_model.model.layers[i]
            target_layer = target_model.model.decoder.layers[i]

            pt_params     = dict(pt_layer.named_parameters())
            target_params = dict(target_layer.named_parameters())

            n_injected = 0
            n_skipped  = 0

            for name, it_param in it_layer.named_parameters():
                # 1. Skip jika tidak ada di PT atau target
                if name not in pt_params or name not in target_params:
                    continue

                # 2. Skip k_proj / v_proj karena merged attention mismatch
                if any(name.endswith(suf) for suf in SKIP_PARAM_SUFFIXES):
                    skipped_names.add(name)
                    n_skipped += 1
                    continue

                # 3. Shape guard
                pt_shape  = pt_params[name].shape
                tgt_shape = target_params[name].shape
                if it_param.shape != pt_shape or it_param.shape != tgt_shape:
                    shape_mismatch_names.add(f"L{i}.{name}")
                    continue

                # 4. Inject: target += layer_alpha * (IT - PT)
                delta = it_param - pt_params[name]
                target_params[name].add_(delta, alpha=layer_alpha)
                n_injected += 1

            if i % 5 == 0 or i == num_layers - 1:
                print(
                    f"  Layer {i:>2}/{num_layers} | α={layer_alpha:.3f} | "
                    f"injected={n_injected} skipped_kv={n_skipped}"
                )

    print(f"\nSkipped params (k/v mismatch): {sorted(skipped_names)[:5]}...")
    if shape_mismatch_names:
        print(f"Shape mismatches: {sorted(shape_mismatch_names)[:5]}")

    # ── Injection ke Embeddings ───────────────────────────────────────────
    print("\nInjecting into Shared Embeddings...")
    with torch.no_grad():
        it_embed     = it_model.model.embed_tokens.weight      # [262208, 640] Gemma 3
        pt_embed     = pt_model.model.embed_tokens.weight      # [262208, 640]
        target_embed = target_model.get_input_embeddings().weight  # [262144, 640] T5Gemma2

        target_vocab = target_embed.shape[0]

        if it_embed.shape[0] != target_vocab or pt_embed.shape[0] != target_vocab:
            print(
                f"  [INFO] Vocab mismatch: IT={it_embed.shape[0]}, "
                f"PT={pt_embed.shape[0]}, target={target_vocab}. "
                f"Slicing delta ke [:{target_vocab}]."
            )
            # Slice — 64 token ekstra Gemma 3 (ID 262144-262207) tidak dipakai di T5G
            delta_embed = it_embed[:target_vocab] - pt_embed[:target_vocab]
        else:
            delta_embed = it_embed - pt_embed

        # Embed pakai BASE_ALPHA (uniform, bukan layer-wise)
        target_embed.add_(delta_embed, alpha=BASE_ALPHA)
        print(f"  Embed injected: shape={target_embed.shape}, alpha={BASE_ALPHA}")

    # ── Injection ke Final Norm ───────────────────────────────────────────
    print("Injecting into Final Norm (decoder)...")
    with torch.no_grad():
        it_norm     = it_model.model.norm.weight
        pt_norm     = pt_model.model.norm.weight
        target_norm = target_model.model.decoder.norm.weight

        if it_norm.shape != target_norm.shape:
            print(f"  [SKIP] Norm shape mismatch: IT={it_norm.shape}, target={target_norm.shape}")
        else:
            delta_norm = it_norm - pt_norm
            target_norm.add_(delta_norm, alpha=BASE_ALPHA)
            print(f"  Norm injected: shape={target_norm.shape}")

    # ── Save Model + Tokenizer ────────────────────────────────────────────
    print(f"\nSaving model to {OUTPUT_PATH}...")
    target_model.save_pretrained(OUTPUT_PATH)

    # PENTING: Pakai tokenizer T5Gemma2 native (bukan Gemma 3 IT)
    # Alasan:
    #   1. Vocab identik untuk teks biasa (token IDs sama persis)
    #   2. ID 256001: di T5G = <image_soft_token>, di IT = <unused99>
    #      Pakai T5G agar vision token suppression bekerja dengan benar
    #   3. T5G tidak punya chat_template (kita format manual anyway)
    print("Loading T5Gemma2 native tokenizer (bukan IT)...")
    tokenizer = AutoTokenizer.from_pretrained(T5G_BASE, trust_remote_code=True)
    assert isinstance(tokenizer, PreTrainedTokenizerFast)

    # Validasi: pastikan special tokens punya ID yang benar
    eot_id = tokenizer.convert_tokens_to_ids("<end_of_turn>")
    sot_id = tokenizer.convert_tokens_to_ids("<start_of_turn>")
    vis_id = tokenizer.convert_tokens_to_ids("<image_soft_token>")
    print(f"  <start_of_turn>={sot_id}, <end_of_turn>={eot_id}, <image_soft_token>={vis_id}")
    assert eot_id == 106, f"<end_of_turn> ID salah: {eot_id}"
    assert sot_id == 105, f"<start_of_turn> ID salah: {sot_id}"
    assert vis_id == 256001, f"<image_soft_token> ID salah: {vis_id}"

    tokenizer.save_pretrained(OUTPUT_PATH)
    print(f"\n✅ Success! Model + T5G tokenizer saved to {OUTPUT_PATH}")
    print(f"   Vocab size: {tokenizer.vocab_size}")


if __name__ == "__main__":
    task_vector_transplant()
