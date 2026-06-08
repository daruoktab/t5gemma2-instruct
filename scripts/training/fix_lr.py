import json

path = "d:/Codings/unsloth/t5-gemma-2/instruct/scripts/training/training_cloud.ipynb"
with open(path, "r", encoding="utf-8") as f:
    nb = json.load(f)

for cell in nb["cells"]:
    if cell["cell_type"] == "code":
        for i, line in enumerate(cell["source"]):
            if "LEARNING_RATE = 2e-5" in line:
                cell["source"][i] = line.replace("LEARNING_RATE = 2e-5", "LEARNING_RATE = 2e-4")
                print("Fixed LEARNING_RATE: 2e-5 -> 2e-4")

with open(path, "w", encoding="utf-8") as f:
    json.dump(nb, f, indent=1, ensure_ascii=False)

print("Done.")
