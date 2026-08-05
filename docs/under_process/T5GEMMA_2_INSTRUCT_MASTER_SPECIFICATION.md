# 📊 MASTER ARCHITECTURAL COMPARISON & SPECIFICATION MANUAL
## Project: T5Gemma-2 Instruct (v7 Joint Multimodal Pipeline)

> **FOR USER & CODING AGENT**: This document provides a complete side-by-side technical comparison of **Gemma 3 IT**, **T5Gemma 1 IT**, **User v6 Text & Vision Implementations**, and the **Proposed v7 Joint Pipeline**. It synthesizes all scratch experiments, Hugging Face configurations, diagnostic bug fixes, Tokenicer integration, Task Vector Steering evolution (Before vs. After code + GPU benchmarks), and explicit decision choices so you and your coding agent can make informed architectural choices.

---

## 📑 TABLE OF CONTENTS
1. [Executive Summary & Purpose](#1-executive-summary--purpose)
2. [Task Vector Steering Evolution: Before vs. After Code & GPU Benchmarks](#2-task-vector-steering-evolution-before-vs-after-code--gpu-benchmarks)
   - [2.1 The Problem & Initial State (Before Code)](#21-the-problem--initial-state-before-code)
   - [2.2 The Solution: SOTA Layer-Wise Ramp-Up Steering (After Code)](#22-the-solution-sota-layer-wise-ramp-up-steering-after-code)
   - [2.3 Empirical GPU Benchmark Evidence across 6 Steering Variants](#23-empirical-gpu-benchmark-evidence-across-6-steering-variants)
3. [Comprehensive Architectural Comparison Matrix](#3-comprehensive-architectural-comparison-matrix)
4. [Deep-Dive Analysis of the 4 Architectures](#4-deep-dive-analysis-of-the-4-architectures)
   - [4.1 Model 1: Google Gemma 3 4B IT (`google/gemma-3-4b-it`)](#41-model-1-google-gemma-3-4b-it-googlegemma-3-4b-it)
   - [4.2 Model 2: Google T5Gemma 1 IT (`google/t5gemma-2b-2b-prefixlm-it` / `ul2-it`)](#42-model-2-google-t5gemma-1-it-googlet5gemma-2b-2b-prefixlm-it--ul2-it)
   - [4.3 Model 3: Historical User v6 Implementations (Text vs Vision)](#43-model-3-historical-user-v6-implementations-text-vs-vision)
   - [4.4 Model 4: Proposed v7 Joint Multimodal Pipeline (`working-molab-v7-combined-unsloth.py`)](#44-model-4-proposed-v7-joint-multimodal-pipeline-working-molab-v7-combined-unslothpy)
5. [Deep-Dive Tokenizer Comparison & `Tokenicer.load()` Integration](#5-deep-dive-tokenizer-comparison--tokenicerload-integration)
6. [Synthesized Lessons & Root Cause Fixes from Scratch Experiments](#6-synthesized-lessons--root-cause-fixes-from-scratch-experiments)
7. [Explicit Architectural Decisions & Questions for Coding Agent](#7-explicit-architectural-decisions--questions-for-coding-agent)
8. [Complete Code Block Library for v7 Implementation](#8-complete-code-block-library-for-v7-implementation)

---

## 1. EXECUTIVE SUMMARY & PURPOSE

This document answers the core questions: **"How should we structure the task vector steering, chat template, tokenizer behavior, vision mechanism, and training protocol for T5Gemma-2 Instruct in v7 compared to what Google did in Gemma 3 / T5Gemma 1 and what we did in v6 Text / v6 Vision?"**

It details the exact code changes made to `working-molab-v7-combined-unsloth.py`, providing empirical GPU benchmark proof for why the new **Layer-Wise Ramp-Up Steering** and **Tokenicer auto-patching** were integrated.

---

## 2. TASK VECTOR STEERING EVOLUTION: BEFORE VS. AFTER CODE & GPU BENCHMARKS

### 2.1 The Problem & Initial State (Before Code)

#### **Original Task Vector Implementation (Uniform Alpha Scaling)**:
In initial versions of the script, Task Vector Steering was applied uniformly across all layers or modules using a constant scaling factor ($\alpha = 0.15$ or $\alpha = 0.35$):

```python
# ❌ BEFORE (Original Implementation in working-molab-v7-combined-unsloth.py)
# Problem: Uniform alpha across all layers caused token embedding distortion in early layers
# and logit output explosion in late layers, leading to repeating loops ("88888888").

def apply_steering_original(t5_model, donor_base, donor_it, alpha=0.35):
    t5_sd = t5_model.state_dict()
    db_sd = donor_base.state_dict()
    di_sd = donor_it.state_dict()

    for k in t5_sd.keys():
        if "mlp" in k and k in di_sd and k in db_sd:
            t5_sd[k] += alpha * (di_sd[k] - db_sd[k])
    return t5_model
```

---

### 2.2 The Solution: SOTA Layer-Wise Ramp-Up Steering (After Code)

```python
# ✅ AFTER (SOTA Implementation in working-molab-v7-combined-unsloth.py)
# Feature: Layer-Wise Ramp-Up Alpha Scaling targeting FFN (gate, up, down) and RMSNorm tensors

def apply_layer_ramp_steering(t5_model, gemma_base_model, gemma_it_model):
    t5_sd = t5_model.state_dict()
    gb_sd = gemma_base_model.state_dict()
    gi_sd = gemma_it_model.state_dict()

    L = getattr(t5_model.config.decoder, "num_hidden_layers", 34)

    def find_key(sd, suffix):
        for k in sd.keys():
            if k.endswith(suffix):
                return k
        return None

    counts = 0
    for l in range(L):
        depth_ratio = l / float(L)

        # Layer-Wise Ramp-Up Alpha Profile
        if depth_ratio < 0.25:
            alpha_ffn = 0.05   # Early layers: subtle
            alpha_norm = 0.02
        elif depth_ratio < 0.80:
            alpha_ffn = 0.25   # Middle layers: PEAK instruction reasoning
            alpha_norm = 0.08
        else:
            alpha_ffn = 0.08   # Late layers: subtle
            alpha_norm = 0.03

        # Steer FFN projection weights (gate_proj, up_proj, down_proj)
        for proj in ("gate_proj", "up_proj", "down_proj"):
            g_key = find_key(gi_sd, f"layers.{l}.mlp.{proj}.weight")
            t_key = find_key(t5_sd, f"decoder.layers.{l}.mlp.{proj}.weight")
            if g_key and g_key in gb_sd and t_key:
                if t5_sd[t_key].shape == gi_sd[g_key].shape == gb_sd[g_key].shape:
                    t5_sd[t_key] += alpha_ffn * (gi_sd[g_key] - gb_sd[g_key])
                    counts += 1

        # Steer RMSNorm weights
        for g_suf, t_suf in (
            ("input_layernorm", "pre_self_attn_layernorm"),
            ("post_attention_layernorm", "post_self_attn_layernorm"),
            ("pre_feedforward_layernorm", "pre_feedforward_layernorm"),
            ("post_feedforward_layernorm", "post_feedforward_layernorm"),
        ):
            g_key = find_key(gi_sd, f"layers.{l}.{g_suf}.weight")
            t_key = find_key(t5_sd, f"decoder.layers.{l}.{t_suf}.weight")
            if g_key and g_key in gb_sd and t_key:
                if t5_sd[t_key].shape == gi_sd[g_key].shape == gb_sd[g_key].shape:
                    t5_sd[t_key] += alpha_norm * (gi_sd[g_key] - gb_sd[g_key])
                    counts += 1

    print(f"✅ Layer-Wise Ramp-Up Steering applied to {counts} tensors across {L} decoder layers.")
    return t5_model
```

---

### 2.3 Empirical GPU Benchmark Evidence across 6 Steering Variants

| Benchmark Variant | Alpha Profile & Strategy | Zero-Shot Generation Behavior | Evaluation Result |
|---|---|---|---|
| **Variant 1 (Pure Base)** | $\alpha = 0.0$ (No steering) | Looped `88888888` and `Incoming User` endlessly | ❌ Failed (No instruction following) |
| **Variant 2 (Subtle Uniform)** | Constant $\alpha = 0.15$ | Echoed system prompt verbatim | ❌ Failed (Poor instruction following) |
| **Variant 3 (Layer-Wise Ramp-Up)** | **Early=0.05, Mid=0.25, Late=0.08** | **Coherent, rich multimodal Bahasa Indonesia text zero-shot** (*"dapat mengubah gambar, teks, dan gambar-gambaranya..."*) | ✅ **WINNER (Selected for v7)** |
| **Variant 4 (DARE Top 20%)** | Top 20% magnitude pruning, $\alpha = 0.35$ | Lost structural coherence at $\alpha = 0.35$ | ❌ Failed (Disrupted layer continuity) |
| **Variant 5 (Hybrid DARE + Ramp)** | DARE pruning + Layer ramp | DARE pruning broke smooth layer continuity | ❌ Failed (Degraded text fluency) |
| **Variant 6 (Target Donor Gemma 3)** | Reference Donor (`google/gemma-3-270m-it`) | Perfect reference response after prompt slice fix | 🎯 Reference Standard |

---

## 3. COMPREHENSIVE ARCHITECTURAL COMPARISON MATRIX

| Feature / Dimension | Google Gemma 3 IT (`google/gemma-3-4b-it`) | Google T5Gemma 1 IT (`t5gemma-2b-2b-prefixlm-it`) | User v6 Text Only (`working-molab-v6-unsloth.py`) | User v6 Vision & v4-Vision Repo (`daruokta/.../v4-vision`) | Proposed v7 Joint Pipeline (`working-molab-v7-combined-unsloth.py`) |
|---|---|---|---|---|---|
| **Architecture Family** | Causal LM (Decoder-Only) | Seq2Seq (Encoder-Decoder) | Seq2Seq (Encoder-Decoder) | Seq2Seq (Encoder-Decoder + SigLIP Graft) | Seq2Seq (Encoder-Decoder + SigLIP Graft) |
| **Tokenizer Class** | `GemmaTokenizer` | `GemmaTokenizer` | `GemmaTokenizer` | `GemmaTokenizer` / `Gemma3Processor` | `GemmaTokenizer` / `Gemma3Processor` + **`Tokenicer`** |
| **Default `add_bos_token`** | `true` | **`false`** (Official Google Fix) | `true` | ⚠️ `true` (Caused double BOS `[<bos>, <bos>]`) | ✅ **`false`** (Explicitly patched in v7) |
| **Padding Side** | `"right"` | `"right"` | `"right"` | `"right"` | `"right"` |
| **Chat Template Source** | `chat_template.json` in HF config | Standalone `chat_template.jinja` in repo root | Custom string formatting (`format_encoder_from_raw`) | `get_chat_template(..., "gemma-3")` saved to `chat_template.jinja` | Verified Jinja2 template (`chat_template.jinja`) |
| **System Prompt Handling** | Embedded inside 1st `<start_of_turn>user` turn | **Disabled** (`raise_exception('System role not supported')`) | Embedded inside `<start_of_turn>user` turn | Embedded inside 1st `<start_of_turn>user` turn | Embedded inside 1st `<start_of_turn>user` turn |
| **Turn Markers** | `<start_of_turn>user\n...<end_of_turn>\n<start_of_turn>model\n` | `<start_of_turn>user\n...<end_of_turn>\n<start_of_turn>model\n` | `<start_of_turn>user\n...<end_of_turn>\n<start_of_turn>model\n` | `<start_of_turn>user\n...<end_of_turn>\n<start_of_turn>model\n` | `<start_of_turn>user\n...<end_of_turn>\n<start_of_turn>model\n` |
| **Vision Backbone & Resolution** | SigLIP ($896 \times 896$ pixels) | None (Text Only) | None (Text Only) | SigLIP ($896 \times 896$ pixels) | SigLIP ($896 \times 896$ pixels) |
| **Image Soft Token Count** | **256 soft tokens** per image | N/A | N/A | **256 soft tokens** per image | **256 soft tokens** per image |
| **Image Token ID & Boundary** | Index `262144` (`<image_soft_token>`) | N/A | N/A | Index `256001` (`<image_soft_token>`) | Index `256001` (`<image_soft_token>`) |

---

## 4. DEEP-DIVE ANALYSIS OF THE 4 ARCHITECTURES

### 4.1 Model 1: Google Gemma 3 4B IT (`google/gemma-3-4b-it`)
* **Chat Template Strategy**: Gemma 3 IT introduced system prompt embedding inside the first user turn.
* **Vision Mechanism**: Gemma 3 IT pairs a 27-layer SigLIP vision transformer ($896 \times 896$ resolution) with a `multi_modal_projector`. Each image is projected into exactly **256 soft tokens**.

### 4.2 Model 2: Google T5Gemma 1 IT (`google/t5gemma-2b-2b-prefixlm-it` / `ul2-it`)
* **Chat Template Strategy**: Google released standalone `chat_template.jinja` files in the repository root of T5Gemma 1 IT. Google **explicitly disabled system roles** by adding `{% if messages[0]['role'] == 'system' %}{{ raise_exception('System role not supported') }}{% endif %}`.

### 4.3 Model 3: Historical User v6 Implementations (Text vs Vision)
* **v6 Text-Only (`working-molab-v6-unsloth.py`)**: Used manual string formatting (`format_encoder_from_raw`). Maintained task prefixes (`<unused1>` for IndoQA, `<unused2>` for Summarization, `<unused6>` for Chat).
* **v6 Vision (`working-molab-v6-vision-unsloth.py`)**: Attached Unsloth `gemma-3` chat template. Unrolled `📷` into `{"type": "image"}`.

### 4.4 Model 4: Proposed v7 Joint Multimodal Pipeline (`working-molab-v7-combined-unsloth.py`)
* **Joint Training Protocol**: Integrates Stage 1 Task Vector Steering (Layer-Wise Ramp-Up from `gemma-3-270m-it`), Stage 2 Joint SFT (Text Retention + Vision Instruction), Stage 3 Joint ORPO, and **Tokenicer Auto-Patching**.

---

## 5. DEEP-DIVE TOKENIZER COMPARISON & `Tokenicer.load()` INTEGRATION

### A. Applied Code Change in `working-molab-v7-combined-unsloth.py` (Cell 15)

We have directly applied `Tokenicer.load()` inside `working-molab-v7-combined-unsloth.py`:

```python
# ✅ APPLIED IN CELL 15 OF working-molab-v7-combined-unsloth.py
# Load processor & apply Tokenicer auto-patching for robust pad_token normalization & cache management
try:
    from tokenicer import Tokenicer
    print("[MODEL] Applying Tokenicer for robust tokenizer loading & pad_token normalization...")
    processor = AutoProcessor.from_pretrained(
        UNIFIED_HF_REPO, subfolder=CANGKOK_SUBFOLDER, token=_token
    )
    if hasattr(processor, "tokenizer"):
        processor.tokenizer = Tokenicer.load(processor.tokenizer)
except Exception as _e_tok:
    print(f"ℹ️ [MODEL] Tokenicer info: {_e_tok}. Using standard AutoProcessor.")
    processor = AutoProcessor.from_pretrained(
        UNIFIED_HF_REPO, subfolder=CANGKOK_SUBFOLDER, token=_token
    )

from unsloth.chat_templates import get_chat_template
tokenizer = get_chat_template(tokenizer, chat_template="gemma-3")
processor.chat_template = tokenizer.chat_template
if hasattr(processor, "tokenizer"):
    processor.tokenizer.chat_template = tokenizer.chat_template

# Nonaktifkan penambahan bos_token otomatis untuk menghindari bos ganda saat inferensi
tokenizer.add_bos_token = False
if hasattr(processor, "tokenizer"):
    processor.tokenizer.add_bos_token = False
```

### B. Why Tokenicer is Integrated:
1. **Auto-Patch Missing/Wrong `pad_token`**: Automatically prevents Hugging Face tokenizers from defaulting `pad_token` to `eos_token` during batching and SFT/ORPO training.
2. **Safe HF Cache Operations**: Ensures remote code execution and cache writes are routed to writable paths (`HF_HOME`, `HF_HUB_CACHE`).

---

## 6. SYNTHESIZED LESSONS & ROOT CAUSE FIXES FROM SCRATCH EXPERIMENTS

Diagnostic review of historical vision degradation (`Debugging Vision Model Performance.md`) uncovered 4 critical root causes enforced in v7:

1. **Root Cause #1: Double BOS Tokens (`[<bos>, <bos>]`)**:
   - `processor.apply_chat_template()` prepends `<bos>`. Collators manually prepending `bos_token_id` generated double `<bos>` tokens.
   - **Fix**: Set `tokenizer.add_bos_token = False` and `processor.tokenizer.add_bos_token = False`.
2. **Root Cause #2: Unsloth Kernel State Locking**:
   - Calling `trainer.evaluate()` without calling `FastVisionModel.for_training(model)` locked Unsloth kernels in evaluation mode.
   - **Fix**: Wrap `evaluate()` with `FastVisionModel.for_inference(model)` before evaluation and `FastVisionModel.for_training(model)` after evaluation.
3. **Root Cause #3: Uniform Learning Rate Collapse**:
   - **Fix**: Apply GrokAdEMAMix split learning rates: Encoder/Decoder `5e-6` (`0.2x`), Projector `1.25e-6` (`0.05x`), Vision Tower `0.0` (Frozen).
4. **Root Cause #4: RSLoRA Scaling Explosion**:
   - **Fix**: Use standard LoRA scaling ($r=64, \alpha=128$, multiplier $2.0$) for stable updates.

---

## 7. EXPLICIT ARCHITECTURAL DECISIONS & QUESTIONS FOR CODING AGENT

> [!IMPORTANT]
> Use the following 4 decision choices to guide implementation:

- [x] **Decision 1**: Standardize on `gemma-3` Jinja2 chat template with System Prompt prepended in first user turn.
- [x] **Decision 2**: Integrate `Tokenicer.load()` + `add_bos_token = False` in processor setup to prevent double BOS tokens and pad token mismatch.
- [x] **Decision 3**: Use Layer-Wise Ramp-Up Task Vector Steering ($\alpha_{\text{FFN}}$ Early=0.05, Mid=0.25, Late=0.08).
- [x] **Decision 4**: Maintain 256 soft tokens per image at $896 \times 896$ resolution via `Gemma3Processor`.

---

## 8. COMPLETE CODE BLOCK LIBRARY FOR V7 IMPLEMENTATION

### A. Official Verified Jinja2 Template (`chat_template.jinja`)

```jinja2
{{ bos_token }}
{%- if messages[0]['role'] == 'system' -%}
    {%- if messages[0]['content'] is string -%}
        {%- set first_user_prefix = messages[0]['content'] + '\n\n' -%}
    {%- else -%}
        {%- set first_user_prefix = messages[0]['content'][0]['text'] + '\n\n' -%}
    {%- endif -%}
    {%- set loop_messages = messages[1:] -%}
{%- else -%}
    {%- set first_user_prefix = "" -%}
    {%- set loop_messages = messages -%}
{%- endif -%}
{%- for message in loop_messages -%}
    {%- if (message['role'] == 'user') != (loop.index0 % 2 == 0) -%}
        {{ raise_exception("Conversation roles must alternate user/assistant/user/assistant/...") }}
    {%- endif -%}
    {%- if (message['role'] == 'assistant') -%}
        {%- set role = "model" -%}
    {%- else -%}
        {%- set role = message['role'] -%}
    {%- endif -%}
    {{ '<start_of_turn>' + role + '\n' + (first_user_prefix if loop.first else "") }}
    {%- if message['content'] is string -%}
        {{ message['content'] | trim }}
    {%- elif message['content'] is iterable -%}
        {%- for item in message['content'] -%}
            {%- if item['type'] == 'image' -%}
                {{ '📷' }}
            {%- elif item['type'] == 'text' -%}
                {{ item['text'] | trim }}
            {%- endif -%}
        {%- endfor -%}
    {%- else -%}
        {{ raise_exception("Invalid content type") }}
    {%- endif -%}
    {{ '<end_of_turn>\n' }}
{%- endfor -%}
{%- if add_generation_prompt -%}
    {{ '<start_of_turn>model\n' }}
{%- endif -%}
```

### B. Processor & Tokenizer Setup Function with Tokenicer

```python
from transformers import AutoProcessor, AutoTokenizer
from unsloth.chat_templates import get_chat_template

def setup_t5gemma2_processor(model_name_or_path, hf_token=None):
    processor = AutoProcessor.from_pretrained(model_name_or_path, token=hf_token, trust_remote_code=True)
    
    # Auto-patch tokenizer using Tokenicer if available
    try:
        from tokenicer import Tokenicer
        if hasattr(processor, "tokenizer"):
            processor.tokenizer = Tokenicer.load(processor.tokenizer)
    except Exception:
        pass

    tokenizer = get_chat_template(processor.tokenizer, chat_template="gemma-3")
    processor.chat_template = tokenizer.chat_template
    if hasattr(processor, "tokenizer"):
        processor.tokenizer.chat_template = tokenizer.chat_template

    # CRITICAL: Prevent double BOS tokens [<bos>, <bos>, ...]
    tokenizer.add_bos_token = False
    if hasattr(processor, "tokenizer"):
        processor.tokenizer.add_bos_token = False

    return processor, tokenizer
```

### C. Layer-Wise Ramp-Up Steering Function

```python
def apply_layer_ramp_steering(t5_model, gemma_base_model, gemma_it_model):
    t5_sd = t5_model.state_dict()
    gb_sd = gemma_base_model.state_dict()
    gi_sd = gemma_it_model.state_dict()

    L = getattr(t5_model.config.decoder, "num_hidden_layers", 34)

    def find_key(sd, suffix):
        for k in sd.keys():
            if k.endswith(suffix):
                return k
        return None

    counts = 0
    for l in range(L):
        depth_ratio = l / float(L)

        if depth_ratio < 0.25:
            alpha_ffn = 0.05
            alpha_norm = 0.02
        elif depth_ratio < 0.80:
            alpha_ffn = 0.25
            alpha_norm = 0.08
        else:
            alpha_ffn = 0.08
            alpha_norm = 0.03

        for proj in ("gate_proj", "up_proj", "down_proj"):
            g_key = find_key(gi_sd, f"layers.{l}.mlp.{proj}.weight")
            t_key = find_key(t5_sd, f"decoder.layers.{l}.mlp.{proj}.weight")
            if g_key and g_key in gb_sd and t_key:
                if t5_sd[t_key].shape == gi_sd[g_key].shape == gb_sd[g_key].shape:
                    t5_sd[t_key] += alpha_ffn * (gi_sd[g_key] - gb_sd[g_key])
                    counts += 1

        for g_suf, t_suf in (
            ("input_layernorm", "pre_self_attn_layernorm"),
            ("post_attention_layernorm", "post_self_attn_layernorm"),
            ("pre_feedforward_layernorm", "pre_feedforward_layernorm"),
            ("post_feedforward_layernorm", "post_feedforward_layernorm"),
        ):
            g_key = find_key(gi_sd, f"layers.{l}.{g_suf}.weight")
            t_key = find_key(t5_sd, f"decoder.layers.{l}.{t_suf}.weight")
            if g_key and g_key in gb_sd and t_key:
                if t5_sd[t_key].shape == gi_sd[g_key].shape == gb_sd[g_key].shape:
                    t5_sd[t_key] += alpha_norm * (gi_sd[g_key] - gb_sd[g_key])
                    counts += 1

    print(f"✅ Layer-Wise Ramp-Up Steering applied to {counts} tensors across {L} decoder layers.")
    return t5_model
```

### D. Multimodal Seq2Seq Vision Collator

```python
import torch
from PIL import Image

class Seq2SeqVisionCollator:
    def __init__(self, processor, max_source_length=16384, max_target_length=2048, vision_dataset=None):
        self.processor = processor
        self.max_source_length = max_source_length
        self.max_target_length = max_target_length
        self.vision_dataset = vision_dataset

    def __call__(self, batch):
        prompts = [item["prompt_text"] for item in batch]
        targets = [item["target_text"] for item in batch]

        batch_images = []
        for item in batch:
            if "images" in item and item["images"]:
                batch_images.append(item["images"])
            elif "dataset_idx" in item and item["dataset_idx"] >= 0 and self.vision_dataset is not None:
                idx = item["dataset_idx"]
                full_imgs = self.vision_dataset[idx]["images"]
                indices = item.get("image_indices", [])
                subset = [full_imgs[i] for i in indices if i < len(full_imgs)]
                batch_images.append(subset)
            else:
                batch_images.append([])

        has_images = any(len(imgs) > 0 for imgs in batch_images)

        if has_images:
            flat_images = []
            for imgs in batch_images:
                if imgs:
                    flat_images.extend(imgs)
                else:
                    flat_images.append(Image.new("RGB", (896, 896), (0, 0, 0)))

            model_inputs = self.processor(
                text=prompts,
                images=flat_images,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=self.max_source_length,
            )
        else:
            model_inputs = self.processor.tokenizer(
                text=prompts,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=self.max_source_length,
            )

        labels = self.processor.tokenizer(
            text=targets,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=self.max_target_length,
        ).input_ids

        labels[labels == self.processor.tokenizer.pad_token_id] = -100
        model_inputs["labels"] = labels
        return model_inputs
```

### E. Checkpoint Save & HF Upload Protocol

```python
import os
from huggingface_hub import HfApi

def export_checkpoint_with_chat_template(output_dir, processor, repo_id, path_in_repo, hf_token):
    processor.save_pretrained(output_dir)

    jinja_path = os.path.join(output_dir, "chat_template.jinja")
    with open(jinja_path, "w", encoding="utf-8") as f:
        f.write(JINJA_TEMPLATE_CONTENT)
    print(f"✅ chat_template.jinja saved to {jinja_path}")

    if hf_token:
        api = HfApi(token=hf_token)
        api.create_repo(repo_id=repo_id, repo_type="model", exist_ok=True)
        api.upload_folder(
            folder_path=output_dir,
            repo_id=repo_id,
            path_in_repo=path_in_repo,
            repo_type="model"
        )
        print(f"📤 Uploaded checkpoint to {repo_id}/{path_in_repo}")
```
