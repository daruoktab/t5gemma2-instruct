"""Deep check: vocab sentinel <extra_id_N> & prefix <unusedX> pada tokenizer t5gemma."""
import json
import sys

from transformers import AutoTokenizer

tok = AutoTokenizer.from_pretrained("google/t5gemma-2-270m-270m")
print("vocab_size:", len(tok), flush=True)
print("bos/eos/unk/pad:", tok.bos_token_id, tok.eos_token_id, tok.unk_token_id, tok.pad_token_id, flush=True)

# 1) Apakah <extra_id_N> ada di vocab & id-nya?
vocab = tok.get_vocab()
for t in ["<unused1>", "<unused2>", "<unused3>", "<unused4>", "<unused5>", "<unused6>",
          "<extra_id_0>", "<extra_id_1>", "<extra_id_2>", "<extra_id_10>", "<extra_id_63>"]:
    print(f"vocab[{t!r}] = {vocab.get(t)}", flush=True)

# 2) Semua special token yang dikenal tokenizer
print("special_tokens[:25]:", tok.all_special_tokens[:25], flush=True)
print("special_ids[:25]:", tok.all_special_ids[:25], flush=True)

# 3) Decode id 3, 4, 5, 6, 7, dan token sekitar sentinel
ids = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 236779, 236771, 236813, 236770]
print("id->token:", {i: tok.convert_ids_to_tokens(i) for i in ids}, flush=True)

# 4) Encode per sentinel (tanpa bos/eos) — bandingkan
for s in ["<extra_id_0>", "<extra_id_1>", "<extra_id_2>", "<unused1>"]:
    print(f"encode({s!r}, no_special):", tok.encode(s, add_special_tokens=False), flush=True)
    print(f"  tokenize: {tok.tokenize(s)}", flush=True)

# 5) added_tokens_decoder (lihat sentinel yang ditambahkan runtime)
adt = tok.added_tokens_decoder
print("added_tokens count:", len(adt), flush=True)
for k in sorted(adt)[:20]:
    print("  added:", k, adt[k], flush=True)

# 6) Info config sentinel
print("extra_ids attr:", getattr(tok, "extra_ids", "TIDAK ADA"), flush=True)
init = tok.init_kwargs
print("init_kwargs keys:", list(init.keys()), flush=True)
for key in ["extra_ids", "additional_special_tokens", "model_max_length"]:
    if key in init:
        print(f"  init[{key}] = {init[key]}", flush=True)

# 7) Uji round-trip sentinel dalam teks
text = "T5Gemma <extra_id_0> model <extra_id_1>."
enc = tok.encode(text)
print("encode:", enc, flush=True)
print("decode:", tok.decode(enc), flush=True)
