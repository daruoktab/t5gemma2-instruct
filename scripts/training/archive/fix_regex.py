import json

NOTEBOOK_PATH = "d:/Codings/unsloth/t5-gemma-2/instruct/scripts/training/training_cloud.ipynb"

with open(NOTEBOOK_PATH, "r", encoding="utf-8") as f:
    nb = json.load(f)

for cell in nb.get("cells", []):
    if cell["cell_type"] == "code":
        source = cell["source"]
        for i, line in enumerate(source):
            if r"\\n(user:|assistant:)\\n" in line:
                source[i] = line.replace(r"\\n(user:|assistant:)\\n", r"\\n(user:|assistant:)\\s*")
            if r"\n(user:|assistant:)\n" in line:
                source[i] = line.replace(r"\n(user:|assistant:)\n", r"\n(user:|assistant:)\s*")
        cell["source"] = source

with open(NOTEBOOK_PATH, "w", encoding="utf-8") as f:
    json.dump(nb, f, indent=1, ensure_ascii=False)
print("Notebook regex fixed.")
