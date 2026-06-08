"""
DEFINITIVE FIX for training_cloud.ipynb
Fixes:
1. Double-escaped backslashes in ALL cells (the ROOT CAUSE of mode collapse)
2. LEARNING_RATE: 5e-5 → 2e-4
3. early_stopping_patience: 10 → 3
4. Double logit mask: comment out the second apply_logit_mask call
"""
import json

NOTEBOOK_PATH = "d:/Codings/unsloth/t5-gemma-2/instruct/scripts/training/training_cloud.ipynb"

with open(NOTEBOOK_PATH, "r", encoding="utf-8") as f:
    nb = json.load(f)

fixes_applied = []

for cell_idx, cell in enumerate(nb.get("cells", [])):
    if cell["cell_type"] != "code":
        continue
    
    new_source = []
    for line in cell["source"]:
        original = line
        
        # ============================================================
        # FIX 1: Double-escaped backslashes → single-escaped
        # After json.load(), source lines that should have \n have \\n
        # We need to replace \\\\ with \\ in the actual string data
        # In Python: '\\\\' (the string \\) → '\\' (the string \)
        # ============================================================
        if '\\\\' in line:
            line = line.replace('\\\\', '\\')
            if line != original:
                fixes_applied.append(f"  Cell {cell_idx}: Fixed double-escape in: {repr(original.strip()[:80])}")
        
        # ============================================================
        # FIX 2: LEARNING_RATE 5e-5 → 2e-4
        # ============================================================
        if 'LEARNING_RATE = 5e-5' in line:
            line = line.replace('LEARNING_RATE = 5e-5', 'LEARNING_RATE = 2e-4')
            fixes_applied.append(f"  Cell {cell_idx}: Fixed LEARNING_RATE: 5e-5 → 2e-4")
        
        # ============================================================
        # FIX 3: early_stopping_patience 10 → 3
        # ============================================================
        if 'early_stopping_patience=10' in line:
            line = line.replace('early_stopping_patience=10', 'early_stopping_patience=3')
            fixes_applied.append(f"  Cell {cell_idx}: Fixed early_stopping_patience: 10 → 3")
        
        # ============================================================
        # FIX 4: Comment out second apply_logit_mask (after PEFT wrapping)
        # ============================================================
        if line.strip() == 'apply_logit_mask(model, ALL_SUPPRESS_IDS)':
            # Check if previous lines indicate this is the second call (after PEFT)
            joined_prev = ''.join(new_source[-5:]) if len(new_source) >= 5 else ''
            if 'get_peft_model' in joined_prev or 'Re-apply' in joined_prev or 'print_trainable' in joined_prev:
                line = line.replace('apply_logit_mask', '# apply_logit_mask')
                fixes_applied.append(f"  Cell {cell_idx}: Commented out second apply_logit_mask (double mask bug)")
        
        new_source.append(line)
    
    cell["source"] = new_source

# Write back
with open(NOTEBOOK_PATH, "w", encoding="utf-8") as f:
    json.dump(nb, f, indent=1, ensure_ascii=False)

print(f"✅ Applied {len(fixes_applied)} fixes to {NOTEBOOK_PATH}")
for fix in fixes_applied:
    print(fix)

# ============================================================
# VERIFY: Re-read and test format_encoder_from_raw
# ============================================================
import re

SYSTEM_PROMPT = "Kamu adalah asisten AI yang helpful, santai, dan ramah. Gunakan Bahasa Indonesia sebagai bahasa utama."

with open(NOTEBOOK_PATH, "r", encoding="utf-8") as f:
    nb2 = json.load(f)

func_source = None
for cell in nb2["cells"]:
    if cell["cell_type"] == "code":
        source = "".join(cell["source"])
        if "def format_encoder_from_raw" in source:
            func_source = source
            break

if func_source:
    print("\n" + "=" * 60)
    print("VERIFICATION: Testing fixed format_encoder_from_raw")
    print("=" * 60)
    
    # Check for remaining double escapes
    has_double = False
    for line in func_source.split("\n"):
        if '\\\\' in line:
            has_double = True
            print(f"  ⚠️ STILL HAS DOUBLE ESCAPE: {repr(line.strip()[:80])}")
    
    if not has_double:
        print("  ✅ No double-escaping remaining!")
    
    exec(func_source, globals())
    
    test_cases = [
        ("system: Test\nuser: Hai, apa kabar?", "Single turn with system"),
        ("user: Siapa namamu?", "Single turn no system"),
        ("user: halo\nassistant: hai juga\nuser: gimana kabarmu?", "Multi-turn"),
    ]
    
    for test_input, desc in test_cases:
        result = format_encoder_from_raw(test_input)
        is_empty = result.strip() in ("<start_of_turn>model", "<start_of_turn>model\\n")
        status = "⚠️ EMPTY!" if is_empty else "✅ OK"
        print(f"\n  [{desc}] {status}")
        print(f"    Input:  {repr(test_input[:60])}")
        print(f"    Output: {repr(result[:120])}")
