"""
Perbandingan lengkap arsitektur google/gemma-3-4b-pt vs google/t5gemma-2-4b-4b.
- Config comparison
- Architecture tree
- Parameter counts per component
- Layer naming differences
- Attention mechanism details
"""
import json
import torch
from collections import OrderedDict
from typing import Any, cast, Iterable
from transformers import AutoConfig, AutoModel, AutoModelForCausalLM, AutoModelForSeq2SeqLM

GEMMA3 = "google/gemma-3-4b-pt"
T5GEMMA2 = "google/t5gemma-2-4b-4b"

# ============================================================
# PART 1: Config Comparison
# ============================================================
def compare_configs():
    print("=" * 80)
    print("  PART 1: CONFIG COMPARISON")
    print("=" * 80)

    cfg_g3 = AutoConfig.from_pretrained(GEMMA3, trust_remote_code=True)
    cfg_t5 = AutoConfig.from_pretrained(T5GEMMA2, trust_remote_code=True)

    g3_dict = cfg_g3.to_dict()
    t5_dict = cfg_t5.to_dict()

    # T5Gemma2 may have nested configs (text_config, etc.)
    # Flatten if needed
    print(f"\n  Gemma 3 config type: {g3_dict.get('model_type', 'N/A')}")
    print(f"  T5Gemma2 config type: {t5_dict.get('model_type', 'N/A')}")

    # Check if T5Gemma2 has sub-configs
    sub_configs = {}
    for key, val in t5_dict.items():
        if isinstance(val, dict) and "model_type" in val:
            sub_configs[key] = val
            print(f"  T5Gemma2 sub-config '{key}': model_type={val.get('model_type')}")

    # --- Direct config key comparison ---
    all_keys = sorted(set(list(g3_dict.keys()) + list(t5_dict.keys())))

    print(f"\n{'Key':<45} {'Gemma 3':>20} {'T5Gemma2':>20}  {'Status'}")
    print("-" * 110)

    for key in all_keys:
        g3_val = g3_dict.get(key, "—")
        t5_val = t5_dict.get(key, "—")

        # Skip large nested dicts in display
        if isinstance(g3_val, dict) and len(str(g3_val)) > 50:
            g3_val = f"{{dict, {len(g3_val)} keys}}"
        if isinstance(t5_val, dict) and len(str(t5_val)) > 50:
            t5_val = f"{{dict, {len(t5_val)} keys}}"
        if isinstance(g3_val, list) and len(str(g3_val)) > 50:
            g3_val = f"[list, {len(g3_val)} items]"
        if isinstance(t5_val, list) and len(str(t5_val)) > 50:
            t5_val = f"[list, {len(t5_val)} items]"

        status = "✅" if g3_val == t5_val else "⚠️"
        if g3_val == "—":
            status = "🆕 T5 only"
        elif t5_val == "—":
            status = "🆕 G3 only"

        print(f"  {key:<43} {str(g3_val):>20} {str(t5_val):>20}  {status}")

    # --- If T5Gemma2 has text_config, compare with Gemma 3 directly ---
    if "text_config" in sub_configs:
        print(f"\n\n{'='*80}")
        print("  PART 1b: Gemma 3 config vs T5Gemma2 text_config (apple-to-apple)")
        print("=" * 80)

        text_cfg = sub_configs["text_config"]
        all_text_keys = sorted(set(list(g3_dict.keys()) + list(text_cfg.keys())))

        same_count = 0
        diff_count = 0
        only_g3 = 0
        only_t5 = 0

        print(f"\n{'Key':<45} {'Gemma 3':>20} {'T5 text_cfg':>20}  {'Status'}")
        print("-" * 110)

        for key in all_text_keys:
            # Skip meta keys
            if key in ("_name_or_path", "transformers_version", "architectures", "model_type", "auto_map"):
                continue

            g3_val = g3_dict.get(key, "—")
            t5_val = text_cfg.get(key, "—")

            if isinstance(g3_val, (dict, list)) and len(str(g3_val)) > 50:
                g3_val = f"<complex>"
            if isinstance(t5_val, (dict, list)) and len(str(t5_val)) > 50:
                t5_val = f"<complex>"

            if g3_val == "—":
                status = "🆕 T5 only"
                only_t5 += 1
            elif t5_val == "—":
                status = "🆕 G3 only"
                only_g3 += 1
            elif g3_val == t5_val:
                status = "✅"
                same_count += 1
            else:
                status = "⚠️ DIFF"
                diff_count += 1

            if status != "✅":  # Only show differences
                print(f"  {key:<43} {str(g3_val):>20} {str(t5_val):>20}  {status}")

        print(f"\n  Summary: {same_count} identical, {diff_count} different, {only_g3} G3-only, {only_t5} T5-only")

    return cfg_g3, cfg_t5


# ============================================================
# PART 2: Architecture Tree & Named Modules
# ============================================================
def compare_architectures():
    print(f"\n\n{'='*80}")
    print("  PART 2: ARCHITECTURE TREE (loaded on meta device)")
    print("=" * 80)

    # Load on meta device to avoid downloading weights — use from_config
    from transformers import Gemma3ForCausalLM
    from transformers import T5Gemma2ForConditionalGeneration

    cfg_g3 = AutoConfig.from_pretrained(GEMMA3, trust_remote_code=True)
    cfg_t5 = AutoConfig.from_pretrained(T5GEMMA2, trust_remote_code=True)

    # Gemma3Config wraps text_config — extract it for CausalLM
    # Use text_config which is what Gemma3ForCausalLM actually needs
    g3_text_cfg = cfg_g3.text_config if hasattr(cfg_g3, "text_config") else cfg_g3

    print("\n  Loading Gemma 3 on meta device (from_config)...")
    with torch.device("meta"):
        model_g3 = Gemma3ForCausalLM._from_config(g3_text_cfg)

    print("  Loading T5Gemma2 on meta device (from_config)...")
    with torch.device("meta"):
        model_t5 = T5Gemma2ForConditionalGeneration._from_config(cfg_t5)

    # --- Print architecture ---
    print(f"\n{'='*80}")
    print("  GEMMA 3 ARCHITECTURE")
    print("=" * 80)
    print(model_g3)

    print(f"\n{'='*80}")
    print("  T5GEMMA2 ARCHITECTURE")
    print("=" * 80)
    print(model_t5)

    # --- Named modules comparison ---
    print(f"\n\n{'='*80}")
    print("  PART 3: PARAMETER COUNT PER COMPONENT")
    print("=" * 80)

    def count_params(model, name_prefix=""):
        """Count params per top-level component."""
        components = OrderedDict()
        total = 0
        for name, param in model.named_parameters():
            top = name.split(".")[0]
            if top not in components:
                components[top] = {"count": 0, "params": 0}
            components[top]["count"] += 1
            components[top]["params"] += param.numel()
            total += param.numel()
        return components, total

    g3_comp, g3_total = count_params(model_g3)
    t5_comp, t5_total = count_params(model_t5)

    print(f"\n  --- Gemma 3 ({g3_total:,} total parameters) ---")
    for comp, info in g3_comp.items():
        pct = info["params"] / g3_total * 100
        print(f"    {comp:<30} {info['params']:>15,} params ({info['count']:>4} tensors) [{pct:5.1f}%]")

    print(f"\n  --- T5Gemma2 ({t5_total:,} total parameters) ---")
    for comp, info in t5_comp.items():
        pct = info["params"] / t5_total * 100
        print(f"    {comp:<30} {info['params']:>15,} params ({info['count']:>4} tensors) [{pct:5.1f}%]")

    # --- Detailed layer-by-layer for first encoder/decoder layer ---
    print(f"\n\n{'='*80}")
    print("  PART 4: DETAILED LAYER STRUCTURE (first layer)")
    print("=" * 80)

    print("\n  --- Gemma 3: Layer 0 modules ---")
    for name, module in model_g3.named_modules():
        if name.startswith("model.layers.0.") and name.count(".") <= 3:
            print(f"    {name}: {module.__class__.__name__}")

    print("\n  --- T5Gemma2: Encoder Layer 0 modules ---")
    for name, module in model_t5.named_modules():
        if "encoder" in name and "layer" in name and ".0." in name and name.count(".") <= 5:
            if "layers.0." in name:
                print(f"    {name}: {module.__class__.__name__}")

    print("\n  --- T5Gemma2: Decoder Layer 0 modules ---")
    for name, module in model_t5.named_modules():
        if "decoder" in name and "layer" in name and ".0." in name and name.count(".") <= 5:
            if "layers.0." in name:
                print(f"    {name}: {module.__class__.__name__}")

    # --- Projection layers (important for LoRA) ---
    print(f"\n\n{'='*80}")
    print("  PART 5: ALL PROJECTION LAYERS (LoRA targets)")
    print("=" * 80)

    print("\n  --- Gemma 3: Unique projection layer names ---")
    g3_proj_names = set()
    for name, module in model_g3.named_modules():
        if hasattr(module, "weight") and ("proj" in name or "gate" in name or "dense" in name):
            # Get relative name (remove layer number)
            import re
            generic = re.sub(r'layers\.\d+', 'layers.X', name)
            shape_str = str(tuple(cast(Any, module.weight).shape)) if hasattr(module.weight, 'shape') else "?"
            g3_proj_names.add((generic, shape_str))
    for name, shape in sorted(g3_proj_names):
        print(f"    {name:<60} {shape}")

    print("\n  --- T5Gemma2: Unique projection layer names ---")
    t5_proj_names = set()
    for name, module in model_t5.named_modules():
        if hasattr(module, "weight") and ("proj" in name or "gate" in name or "dense" in name):
            import re
            generic = re.sub(r'layers\.\d+', 'layers.X', name)
            shape_str = str(tuple(cast(Any, module.weight).shape)) if hasattr(module.weight, 'shape') else "?"
            t5_proj_names.add((generic, shape_str))
    for name, shape in sorted(t5_proj_names):
        print(f"    {name:<60} {shape}")

    # --- Embedding comparison ---
    print(f"\n\n{'='*80}")
    print("  PART 6: EMBEDDING & HEAD COMPARISON")
    print("=" * 80)

    print("\n  --- Gemma 3 ---")
    for name, module in model_g3.named_modules():
        if "embed" in name.lower() or "lm_head" in name.lower():
            if hasattr(module, "weight"):
                print(f"    {name}: {module.__class__.__name__}, shape={tuple(cast(Iterable, cast(Any, module.weight).shape))}")

    print("\n  --- T5Gemma2 ---")
    for name, module in model_t5.named_modules():
        if "embed" in name.lower() or "lm_head" in name.lower() or name == "shared":
            if hasattr(module, "weight"):
                print(f"    {name}: {module.__class__.__name__}, shape={tuple(cast(Iterable, cast(Any, module.weight).shape))}")

    # Check tied weights
    print("\n  --- Tied Weights Check (T5Gemma2) ---")
    tied_params = set()
    param_ids = {}
    for name, param in model_t5.named_parameters():
        pid = id(param)
        if pid in param_ids:
            print(f"    TIED: {param_ids[pid]} ↔ {name}")
            tied_params.add(name)
        else:
            param_ids[pid] = name

    # --- Number of layers ---
    print(f"\n\n{'='*80}")
    print("  PART 7: LAYER COUNTS")
    print("=" * 80)

    g3_layers = [name for name, _ in model_g3.named_modules() if name.endswith(".self_attn")]
    print(f"\n  Gemma 3: {len(g3_layers)} attention layers (decoder-only)")

    t5_enc_layers = [name for name, _ in model_t5.named_modules() 
                     if "encoder" in name and name.endswith(".self_attn")]
    t5_dec_layers = [name for name, _ in model_t5.named_modules() 
                     if "decoder" in name and (name.endswith(".self_attn") or name.endswith(".temporal_self_attn"))]
    print(f"  T5Gemma2 encoder: {len(t5_enc_layers)} self-attention layers")
    print(f"  T5Gemma2 decoder: {len(t5_dec_layers)} attention layers")

    # Also check for cross attention / merged attention
    t5_cross = [name for name, _ in model_t5.named_modules()
                if "cross" in name.lower() or "encoder_attn" in name.lower()]
    if t5_cross:
        print(f"  T5Gemma2 cross-attention modules: {len(t5_cross)}")
        for n in t5_cross[:5]:
            print(f"    {n}")
    else:
        print(f"  T5Gemma2: NO separate cross-attention (merged attention confirmed)")

    return model_g3, model_t5


# ============================================================
# MAIN
# ============================================================
if __name__ == "__main__":
    cfg_g3, cfg_t5 = compare_configs()
    model_g3, model_t5 = compare_architectures()
    
    print(f"\n\n{'='*80}")
    print("  SELESAI!")
    print("=" * 80)
