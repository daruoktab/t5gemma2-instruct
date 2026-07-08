"""Verifikasi nama module T5Gemma2 270m: projector, vision_tower, encoder, decoder."""
import os, sys
os.environ["UNSLOTH_COMPILE_DISABLE"] = "1"
import torch
from transformers import AutoModelForSeq2SeqLM, AutoProcessor

MODEL_NAME = "google/t5gemma-2-270m-270m"
print(f"Loading {MODEL_NAME}...")
model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_NAME, torch_dtype=torch.bfloat16)
print(f"Model: {model.__class__.__name__}")
print(f"is_encoder_decoder: {model.config.is_encoder_decoder}")
print()

# Top-level children
print("=== TOP-LEVEL CHILDREN ===")
for name, mod in model.named_children():
    nparams = sum(p.numel() for p in mod.parameters())
    print(f"  {name:30s} {mod.__class__.__name__:40s} {nparams/1e6:.2f}M")
print()

# Dump semua module linear + projector + vision ke file
out_path = "results/t5gemma2_modules_dump.txt"
os.makedirs("results", exist_ok=True)
with open(out_path, "w", encoding="utf-8") as f:
    f.write(f"Model: {model.__class__.__name__}\n")
    f.write(f"Total params: {sum(p.numel() for p in model.parameters())/1e6:.1f}M\n\n")
    f.write("=== ALL MODULES (name | class | shape) ===\n")
    for name, mod in model.named_modules():
        # Hanya tampilkan yang punya parameter (skip container kosong kecuali root)
        params = list(mod.parameters())
        if not params and name:
            continue
        cls = mod.__class__.__name__
        # Untuk Linear, tampilkan shape
        shape_str = ""
        if hasattr(mod, "weight") and mod.weight is not None:
            shape_str = f" {tuple(mod.weight.shape)}"
        f.write(f"{name:70s} {cls:35s}{shape_str}\n")

print(f"=== Module dump saved to {out_path} ===")
print()

# Filter khusus: projector, vision_tower structure, embed, lm_head
print("=== PROJECTOR / VISION / EMBED / HEAD ===")
keywords = ["projector", "vision", "embed", "lm_head", "patch", "conv", "norm"]
for name, mod in model.named_modules():
    if any(k in name.lower() for k in keywords):
        cls = mod.__class__.__name__
        shape_str = ""
        if hasattr(mod, "weight") and mod.weight is not None:
            shape_str = f" {tuple(mod.weight.shape)}"
        nparams = sum(p.numel() for p in mod.parameters())
        print(f"  {name:70s} {cls:35s}{shape_str:30s} {nparams/1e6:.3f}M")

print()
print("=== SAMPLE: layer 0 dari encoder text, decoder, vision tower ===")
for name, mod in model.named_modules():
    if any(p in name for p in ["encoder.text_model.layers.0", "decoder.layers.0", "vision_tower.encoder.layers.0"]):
        if "." in name and name.count(".") >= 4:
            cls = mod.__class__.__name__
            shape_str = ""
            if hasattr(mod, "weight") and mod.weight is not None:
                shape_str = f" {tuple(mod.weight.shape)}"
            print(f"  {name:75s} {cls:30s}{shape_str}")
print("\nDONE")
