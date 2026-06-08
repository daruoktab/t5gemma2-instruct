"""
Compare Tokenizers of Gemma 3 (Base), Gemma 3 IT, and T5Gemma 2 (Base)
"""
import torch
from transformers import AutoTokenizer, PreTrainedTokenizerFast

G3_BASE = "google/gemma-3-270m"
G3_IT   = "google/gemma-3-270m-it"
T5G     = "google/t5gemma-2-270m-270m"

def main():
    print("Loading tokenizers...")
    tok_g3_base = AutoTokenizer.from_pretrained(G3_BASE)
    tok_g3_it   = AutoTokenizer.from_pretrained(G3_IT)
    tok_t5g     = AutoTokenizer.from_pretrained(T5G, trust_remote_code=True)

    print("\n=== VOCABULARY SIZES ===")
    print(f"Gemma 3 Base : {tok_g3_base.vocab_size}")
    print(f"Gemma 3 IT   : {tok_g3_it.vocab_size}")
    print(f"T5Gemma 2    : {tok_t5g.vocab_size}")

    special_tokens = [
        "<bos>", "<eos>", "<pad>", "<unk>",
        "<start_of_turn>", "<end_of_turn>",
        "<image_soft_token>", "<img>", "<end_of_image>",
    ]

    print("\n=== SPECIAL TOKEN COMPARISON ===")
    print(f"{'Token':<25} | {'G3 Base':>10} | {'G3 IT':>10} | {'T5G':>10} | Consistent?")
    print("-" * 75)
    for t in special_tokens:
        id_base = tok_g3_base.convert_tokens_to_ids(t)
        id_it   = tok_g3_it.convert_tokens_to_ids(t)
        id_t5g  = tok_t5g.convert_tokens_to_ids(t)
        
        # unk token id to check if token exists
        unk_base = tok_g3_base.unk_token_id or -1
        unk_it   = tok_g3_it.unk_token_id or -1
        unk_t5g  = tok_t5g.unk_token_id or -1

        base_str = "UNK" if id_base == unk_base else str(id_base)
        it_str   = "UNK" if id_it == unk_it else str(id_it)
        t5g_str  = "UNK" if id_t5g == unk_t5g else str(id_t5g)
        
        consistent = "YES" if (id_base == id_it == id_t5g) and id_base != unk_base else "NO"
        print(f"{t:<25} | {base_str:>10} | {it_str:>10} | {t5g_str:>10} | {consistent}")

    # Check contiguous unused ranges
    print("\n=== UNUSED TOKEN CHECK (T5Gemma 2) ===")
    # Check ID 6 to 104
    all_unused_block1 = True
    for idx in range(6, 105):
        tok_name = tok_t5g.convert_ids_to_tokens(idx)
        if tok_name is not None and not tok_name.startswith("<unused"):
            all_unused_block1 = False
            print(f"  Found non-unused in Block 1: {idx} -> {tok_name}")
            break
    if all_unused_block1:
        print("  ✅ All tokens in range [6, 104] are indeed '<unused>' tokens.")

    # Check ID 256002 to 262143
    all_unused_block2 = True
    for idx in range(256002, 256050): # check first 50 of block 2 to be fast
        tok_name = tok_t5g.convert_ids_to_tokens(idx)
        if tok_name is not None and not tok_name.startswith("<unused"):
            all_unused_block2 = False
            print(f"  Found non-unused in Block 2 start: {idx} -> {tok_name}")
            break
    if all_unused_block2:
        print("  ✅ Start of range [256002, 262143] are indeed '<unused>' tokens.")

if __name__ == "__main__":
    main()
