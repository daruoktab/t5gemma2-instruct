"""
Diagnostic script: extract the EXACT code from training_cloud.ipynb,
execute it, and test whether format_encoder_from_raw actually works.
"""
import json
import re

NOTEBOOK_PATH = "d:/Codings/unsloth/t5-gemma-2/instruct/scripts/training/training_cloud.ipynb"

with open(NOTEBOOK_PATH, "r", encoding="utf-8") as f:
    nb = json.load(f)

# ============================================================
# STEP 1: Find and print the raw source of the formatting cell
# ============================================================
func_source = None
for cell in nb["cells"]:
    if cell["cell_type"] == "code":
        source = "".join(cell["source"])
        if "def format_encoder_from_raw" in source:
            func_source = source
            break

if func_source is None:
    print("ERROR: format_encoder_from_raw not found in notebook!")
    exit(1)

print("=" * 60)
print("EXACT SOURCE CODE FROM NOTEBOOK (repr of each line):")
print("=" * 60)
for i, line in enumerate(func_source.split("\n")):
    if "re.split" in line or "re.search" in line or "formatted +=" in line:
        print(f"  LINE {i}: {repr(line)}")

# ============================================================
# STEP 2: Check for double-escaped backslashes
# ============================================================
print("\n" + "=" * 60)
print("DOUBLE-ESCAPE CHECK:")
print("=" * 60)

issues = []
for line in func_source.split("\n"):
    if "\\\\n" in line:
        issues.append(f"  FOUND \\\\n (double-escaped newline): {repr(line.strip())}")
    if "\\\\s" in line:
        issues.append(f"  FOUND \\\\s (double-escaped whitespace): {repr(line.strip())}")

if issues:
    print("⚠️  CRITICAL BUG DETECTED! Double-escaped backslashes found:")
    for iss in issues:
        print(iss)
else:
    print("✅ No double-escaping detected.")

# ============================================================
# STEP 3: Actually execute the function and test it
# ============================================================
print("\n" + "=" * 60)
print("EXECUTION TEST:")
print("=" * 60)

SYSTEM_PROMPT = "Kamu adalah asisten AI yang helpful, santai, dan ramah. Gunakan Bahasa Indonesia sebagai bahasa utama."

# Execute the function source
exec(func_source, globals())

test_cases = [
    "system: Test system prompt\nuser: Hai, apa kabar?",
    "user: Siapa namamu?",
    "user: halo\nassistant: hai juga\nuser: gimana kabarmu?",
]

for i, test_input in enumerate(test_cases):
    print(f"\n--- Test Case {i+1} ---")
    print(f"INPUT: {repr(test_input)}")
    result = format_encoder_from_raw(test_input)
    print(f"OUTPUT repr: {repr(result)}")
    print(f"OUTPUT display:\n{result}")
    
    # Check if output is basically empty (only <start_of_turn>model)
    stripped = result.strip()
    if stripped == "<start_of_turn>model" or stripped.endswith("model") and "user" not in stripped:
        print("⚠️  CRITICAL: Output is EMPTY! Encoder gets no real content!")
    else:
        print("✅ Output contains user content.")

# ============================================================
# STEP 4: Check the SECOND regex in the non-raw string prefix
# ============================================================
print("\n" + "=" * 60)
print("STRING PREFIX CHECK:")
print("=" * 60)

for line in func_source.split("\n"):
    if '"\\n" + raw_input' in line or '"\\\\n" + raw_input' in line:
        print(f"  Found prefix line: {repr(line.strip())}")
        # Test what the prefix actually is
        prefix_test = eval('"' + line.split('"')[1] + '"') if '"' in line else "?"
        print(f"  Prefix evaluates to: {repr(prefix_test)}")
