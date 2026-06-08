"""Quick tokenizer comparison script."""
from transformers import AutoTokenizer, PreTrainedTokenizerFast

G3_IT = "google/gemma-3-270m-it"
T5G   = "google/t5gemma-2-270m-270m"

_raw_it  = AutoTokenizer.from_pretrained(G3_IT)
_raw_t5g = AutoTokenizer.from_pretrained(T5G, trust_remote_code=True)

assert isinstance(_raw_it, PreTrainedTokenizerFast), f"tok_it bukan PTF: {type(_raw_it)}"
assert isinstance(_raw_t5g, PreTrainedTokenizerFast), f"tok_t5g bukan PTF: {type(_raw_t5g)}"

tok_it: PreTrainedTokenizerFast  = _raw_it
tok_t5g: PreTrainedTokenizerFast = _raw_t5g

SPECIAL = [
    "<start_of_turn>", "<end_of_turn>", "<eos>", "<bos>", "<pad>", "<unk>",
    "<image_soft_token>", "<start_of_image>", "<end_of_image>",
]

print("=== VOCAB SIZE ===")
print(f"Gemma3-IT : {tok_it.vocab_size}")
print(f"T5Gemma2  : {tok_t5g.vocab_size}")

print()
print(f"{'Token':<25} | {'IT':>8} | {'T5G':>8} | SAMA?")
print("-" * 55)
for t in SPECIAL:
    iit = tok_it.convert_tokens_to_ids(t)
    it5 = tok_t5g.convert_tokens_to_ids(t)
    same = "YES" if iit == it5 else "NO <--"
    print(f"{t:<25} | {iit:>8} | {it5:>8} | {same}")

print()
print("=== ENCODING IDENTIK? ===")
SYSTEM = "Kamu adalah asisten AI yang helpful."
Q = "Halo, siapa kamu?"
prompt = f"<start_of_turn>user\n{SYSTEM}\n\n{Q}<end_of_turn>\n<start_of_turn>model\n"
ids_it  = tok_it.encode(prompt, add_special_tokens=False)
ids_t5g = tok_t5g.encode(prompt, add_special_tokens=False)
print(f"IT  ({len(ids_it)} tokens): {ids_it[:25]}")
print(f"T5G ({len(ids_t5g)} tokens): {ids_t5g[:25]}")
print(f"IDENTIK: {ids_it == ids_t5g}")

print()
print("=== VISION TOKEN CHECK ===")
for name, tok in [("IT", tok_it), ("T5G", tok_t5g)]:
    s = tok.convert_ids_to_tokens([255999, 256000, 256001])
    print(f"{name}: {list(zip([255999, 256000, 256001], s))}")

print()
print("=== APPLY_CHAT_TEMPLATE (IT) ===")
msgs = [{"role": "user", "content": f"{SYSTEM}\n\n{Q}"}]
try:
    t = tok_it.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
    print(repr(t))
except Exception as e:
    print(f"Error: {e}")

print()
print("=== APPLY_CHAT_TEMPLATE (T5G) ===")
try:
    t2 = tok_t5g.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
    print(repr(t2))
except Exception as e:
    print(f"T5G Error: {e}")

print()
print("=== BOS TOKEN BEHAVIOUR ===")
prompt_bos = f"<start_of_turn>user\n{Q}<end_of_turn>\n<start_of_turn>model\n"
ids_with    = tok_it.encode(prompt_bos, add_special_tokens=True)
ids_without = tok_it.encode(prompt_bos, add_special_tokens=False)
print(f"add_special_tokens=True  : {ids_with[:8]}  (first: {tok_it.convert_ids_to_tokens([ids_with[0]])})")
print(f"add_special_tokens=False : {ids_without[:8]}  (first: {tok_it.convert_ids_to_tokens([ids_without[0]])})")
