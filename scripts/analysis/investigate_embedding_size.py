"""
Investigasi: Kenapa Gemma 3 embedding = 262.208, bukan 262.144?
"""
from transformers import AutoConfig

GEMMA3 = "google/gemma-3-4b-pt"
T5GEMMA2 = "google/t5gemma-2-4b-4b"

cfg_g3 = AutoConfig.from_pretrained(GEMMA3, trust_remote_code=True)
cfg_t5 = AutoConfig.from_pretrained(T5GEMMA2, trust_remote_code=True)

# ---- Gemma 3 ----
g3_text = cfg_g3.text_config
print("=" * 60)
print("  GEMMA 3")
print("=" * 60)
print(f"  text_config.vocab_size       = {g3_text.vocab_size}")
print(f"  image_token_index            = {cfg_g3.image_token_index}")
print(f"  boi_token_index (start_img)  = {cfg_g3.boi_token_index}")
print(f"  eoi_token_index (end_img)    = {cfg_g3.eoi_token_index}")
print(f"  mm_tokens_per_image          = {cfg_g3.mm_tokens_per_image}")

# Check: Is image_token_index OUTSIDE vocab range?
print(f"\n  image_token_index >= vocab_size?")
print(f"    {cfg_g3.image_token_index} >= {g3_text.vocab_size}?")
print(f"    = {cfg_g3.image_token_index >= g3_text.vocab_size}")

# Check alignment
embed_size = 262208
vocab_size = 262144
extra = embed_size - vocab_size
print(f"\n  Embedding rows:  {embed_size}")
print(f"  Vocab size:      {vocab_size}")
print(f"  Extra:           {extra}")

# What alignment is 262208?
for align in [8, 16, 32, 64, 128, 256, 512]:
    if embed_size % align == 0:
        print(f"  262208 % {align} = 0  ✅ (divisible)")
    
# Check: vocab_size (262144) + 1 for image_soft_token = 262145
# Rounded up to nearest 64 = ?
import math
needed = cfg_g3.image_token_index + 1  # Need at least this many rows
rounded = math.ceil(needed / 64) * 64
print(f"\n  image_token_index + 1 = {needed} (minimum rows needed)")
print(f"  Rounded up to nearest 64 = {rounded}")
print(f"  Matches embedding size? {rounded == embed_size}")

# ---- T5Gemma2 ----
print(f"\n{'=' * 60}")
print("  T5GEMMA2")
print("=" * 60)
print(f"  config.vocab_size            = {cfg_t5.vocab_size}")
print(f"  image_token_index            = {cfg_t5.image_token_index}")
print(f"  eoi_token_index              = {cfg_t5.eoi_token_index}")

# Check: Is image_token_index INSIDE vocab range?
print(f"\n  image_token_index < vocab_size?")
print(f"    {cfg_t5.image_token_index} < {cfg_t5.vocab_size}?")
print(f"    = {cfg_t5.image_token_index < cfg_t5.vocab_size}")
print(f"    → Tidak perlu expand embedding!")

# ---- Summary ----
print(f"\n{'=' * 60}")
print("  KESIMPULAN")
print("=" * 60)
print(f"""
  Gemma 3:
    - <image_soft_token> di id {cfg_g3.image_token_index} (= vocab_size, di LUAR range 0-262143)
    - Maka embedding harus minimal {cfg_g3.image_token_index + 1} baris
    - Dibulatkan ke kelipatan 64 → {rounded} baris (untuk GPU alignment)
    - Extra {rounded - vocab_size} slot = padding kosong, tidak ada token yang memakai

  T5Gemma2:
    - <image_soft_token> di id {cfg_t5.image_token_index} (DALAM range 0-262143)
    - Menggantikan slot <unused99> yang ada
    - Embedding tetap {cfg_t5.vocab_size} baris, tidak perlu expand
    - Lebih efisien memori!
""")
