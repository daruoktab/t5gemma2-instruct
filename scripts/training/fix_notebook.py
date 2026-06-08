import json
import os
import re

NOTEBOOK_PATH = "d:/Codings/unsloth/t5-gemma-2/instruct/scripts/training/training_cloud.ipynb"

with open(NOTEBOOK_PATH, "r", encoding="utf-8") as f:
    nb = json.load(f)

for cell in nb.get("cells", []):
    if cell["cell_type"] == "code":
        source = cell["source"]
        for i, line in enumerate(source):
            # 1. Bug query decode in SampleGenerationCallback
            if "query = self.tokenizer.decode(sample[\"input_ids\"], skip_special_tokens=True).strip()" in line:
                source[i] = line.replace('query = self.tokenizer.decode(sample["input_ids"], skip_special_tokens=True).strip()', 'query = sample.get("raw_query", "<unknown query>")')
            
            if "eval_generation_samples = [" in line:
                # the user's cell has something like:
                # eval_generation_samples = val_rows[:SAMPLE_EVAL_GENERATION]
                # We want to insert raw_query instead.
                pass
            if "eval_generation_samples = val_rows[:SAMPLE_EVAL_GENERATION]" in line:
                indent = line[:len(line) - len(line.lstrip())]
                source[i] = indent + "eval_generation_samples = [{**row, 'raw_query': row.get('input', '')} for row in val_rows[:SAMPLE_EVAL_GENERATION]]\n"
                
            # Type checking issue with tokenizer.decode().strip()
            if "_resp_raw = self.tokenizer.decode(out[0], skip_special_tokens=True)" in line:
                # The next lines might be doing .strip() or something else.
                pass
            if "response = self.tokenizer.decode(out[0], skip_special_tokens=True).strip()" in line:
                indent = line[:len(line) - len(line.lstrip())]
                replacement = indent + "raw_resp = self.tokenizer.decode(out[0], skip_special_tokens=True)\n"
                replacement += indent + "response = raw_resp.strip() if isinstance(raw_resp, str) else \" \".join(raw_resp).strip()\n"
                source[i] = replacement
                
            # Type checking issue with encode: add assert isinstance(tokenizer, PreTrainedTokenizerFast)
            if "tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME" in line:
                indent = line[:len(line) - len(line.lstrip())]
                source[i] = line + indent + "assert isinstance(tokenizer, PreTrainedTokenizerFast)\n"
            if "from transformers import (" in line:
                # check if PreTrainedTokenizerFast is there, it usually is
                pass

            # 2. Logit mask dipanggil dua kali
            if "apply_logit_mask(model, ALL_SUPPRESS_IDS)" in line:
                if i > 0 and "Re-apply" in source[i-1]:
                    source[i] = line.replace("apply_logit_mask", "# apply_logit_mask")
            
            # 3. Regex escaped \\n vs \n
            if r"\\\\nuser:" in line:
                source[i] = line.replace(r"\\\\nuser:", r"\\nuser:")
            if r"\\\\n(user:|assistant:)" in line:
                source[i] = line.replace(r"\\\\n(user:|assistant:)", r"\\n(user:|assistant:)")

            # 4. Conversation boundary fragile
            if r"\\n(user:|assistant:)\\s*" in line:
                source[i] = line.replace(r"\\n(user:|assistant:)\\s*", r"\\n(user:|assistant:)\\n")
            if r"\n(user:|assistant:)\s*" in line:
                source[i] = line.replace(r"\n(user:|assistant:)\s*", r"\n(user:|assistant:)\n")
                
            # 5. Early stopping patience
            if "early_stopping_patience=10" in line:
                source[i] = line.replace("early_stopping_patience=10", "early_stopping_patience=3")
                
            # 6. LR
            if "LEARNING_RATE = 5e-5" in line:
                source[i] = line.replace("LEARNING_RATE = 5e-5", "LEARNING_RATE = 2e-4")
                
            # 7. plt.show()
            if "plt.show()" in line:
                indent = line[:len(line) - len(line.lstrip())]
                replacement = f"{indent}import matplotlib\n{indent}if matplotlib.get_backend() != 'agg':\n{indent}    plt.show()\n"
                source[i] = replacement
                
            # F-string without placeholders
            if 'print(f"\\nLoading Model from {MODEL_NAME}...")' in line:
                pass
            if 'print(f"\\nStarting Clean SFT...")' in line:
                source[i] = line.replace('print(f"\\nStarting', 'print("\\nStarting')

        cell["source"] = source

with open(NOTEBOOK_PATH, "w", encoding="utf-8") as f:
    json.dump(nb, f, indent=1, ensure_ascii=False)
print("Notebook updated.")
