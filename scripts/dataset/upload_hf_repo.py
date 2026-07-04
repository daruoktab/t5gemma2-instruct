"""
Script to upload chat_template.jinja and README.md to HuggingFace repo
daruokta/t5gemma-2-4b-4b-instruct-chat-indo-v3-unsloth
"""
import os
import json
import tempfile
from huggingface_hub import HfApi, login

# Login using token from .env or cached credentials
api = HfApi()

REPO_ID = "daruokta/t5gemma-2-4b-4b-instruct-chat-indo-v3-unsloth"

# =====================================================================
# 1. Chat Template (Jinja2)
# =====================================================================
CHAT_TEMPLATE = """{%- set ns = namespace(system_message='', found_system=false, first_user=true) -%}
{%- for message in messages -%}
    {%- if message['role'] == 'system' -%}
        {%- set ns.system_message = message['content'] -%}
        {%- set ns.found_system = true -%}
    {%- endif -%}
{%- endfor -%}
{{- bos_token -}}
{%- for message in messages -%}
    {%- if message['role'] == 'system' -%}
        {%- continue -%}
    {%- endif -%}
    {%- if message['role'] == 'user' -%}
        {{- '<start_of_turn>user\\n' -}}
        {%- if ns.first_user and ns.found_system -%}
            {{- ns.system_message + '\\n\\n' -}}
            {%- set ns.first_user = false -%}
        {%- elif ns.first_user -%}
            {%- set ns.first_user = false -%}
        {%- endif -%}
        {{- message['content'] + '<end_of_turn>\\n' -}}
    {%- elif message['role'] == 'model' or message['role'] == 'assistant' -%}
        {{- '<start_of_turn>model\\n' + message['content'] + '<end_of_turn>\\n' -}}
    {%- endif -%}
{%- endfor -%}
{%- if add_generation_prompt -%}
    {{- '<start_of_turn>model\\n' -}}
{%- endif -%}"""

# =====================================================================
# 2. README.md
# =====================================================================
README_CONTENT = r"""---
base_model: google/t5gemma-2-4b-4b
library_name: transformers
datasets:
- daruokta/t5gemma2-indonesia-chat-formatted
tags:
- t5-gemma-2
- conversational
- text-generation
- indonesian
- unsloth
- seq2seq
- encoder-decoder
- chat-template
license: gemma
language:
- id
- en
pipeline_tag: text-generation
model-index:
- name: T5-Gemma-2-4B-Instruct-Chat-Indo-v3-unsloth
  results:
  - task:
      type: text-generation
      name: Conversational & Instruction Following
    dataset:
      name: T5-Gemma-2 Indonesia Chat Formatted
      type: daruokta/t5gemma2-indonesia-chat-formatted
      args: chat_sft + indoqa_sft
    metrics:
    - type: rougeL
      value: 69.05
      name: ROUGE-L (Peak)
    - type: loss
      value: 2.53
      name: Eval Loss (Min)
---

# T5-Gemma-2-4B-4B-Instruct-Chat-Indo-v3 (Unsloth)

Model ini merupakan **fine-tuned merged model** dari [google/t5gemma-2-4b-4b](https://huggingface.co/google/t5gemma-2-4b-4b) menggunakan Supervised Fine-Tuning (SFT) dengan framework Unsloth. Model ini dioptimalkan untuk tugas pemahaman instruksi dan kemampuan dialog multi-turn secara natural dalam **Bahasa Indonesia** dan **Bahasa Inggris**.

> **Catatan Penting:** Ini adalah model **encoder-decoder** (Seq2Seq), bukan decoder-only. Model ini dilengkapi dengan **chat template** khusus yang dirancang untuk arsitektur encoder-decoder T5Gemma-2.

## 📈 Hasil Evaluasi (Training)

Pelatihan dilakukan selama 1225 langkah. Model mencapai **skor ROUGE-L tertinggi sekitar 69.05%** pada langkah 1000, dengan Loss Evaluasi terendah di kisaran **2.53** (pada langkah 600).

![Training & Evaluation Curves](https://huggingface.co/daruokta/t5gemma-2-4b-4b-instruct-chat-indo-v3-unsloth/resolve/main/training_chart-v5-unsloth.png)

*(Catatan: Laporan evaluasi komparatif lengkap dengan sampel generasi masih dalam tahap pengumpulan data dan akan di-update kemudian).*

## ✨ Fitur Utama

- 🔄 **Encoder-Decoder Architecture:** Menggunakan arsitektur Seq2Seq yang efisien untuk memetakan representasi encoder secara implisit sesuai kategori tugas.
- 💬 **Custom Chat Template:** Dilengkapi dengan Jinja2 chat template (`chat_template.jinja`) yang mendukung `tokenizer.apply_chat_template()` out-of-the-box.
- 🇮🇩 **Bilingual (ID/EN):** Dioptimalkan untuk percakapan natural dalam Bahasa Indonesia dengan kemampuan Bahasa Inggris.
- 🎯 **Multi-Task SFT:** Dilatih pada tugas percakapan multi-turn, QA berbasis dokumen, ringkasan, terjemahan, dan parafrase.
- 🛡️ **Logit Masking:** Menerapkan logit masking untuk memblokir token yang tidak digunakan (unused tokens) dan token visual guna meminimalkan kesalahan pembentukan token.

## 📋 Cara Penggunaan

### Cara 1: Menggunakan `apply_chat_template` (Direkomendasikan)

```python
import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

model_id = "daruokta/t5gemma-2-4b-4b-instruct-chat-indo-v3-unsloth"

tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
model = AutoModelForSeq2SeqLM.from_pretrained(
    model_id,
    trust_remote_code=True,
    torch_dtype=torch.bfloat16,
    device_map="auto"
)

# Siapkan pesan percakapan
messages = [
    {"role": "system", "content": "Kamu adalah asisten AI yang helpful, santai, dan ramah. Gunakan Bahasa Indonesia sebagai bahasa utama."},
    {"role": "user", "content": "Jelaskan secara singkat apa itu fotosintesis."}
]

# Gunakan chat template bawaan
prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
inputs = tokenizer(prompt, return_tensors="pt", add_special_tokens=False).to(model.device)

with torch.no_grad():
    outputs = model.generate(
        **inputs,
        max_new_tokens=256,
        temperature=0.7,
        do_sample=True,
        eos_token_id=[
            tokenizer.convert_tokens_to_ids("<end_of_turn>"),
            tokenizer.eos_token_id
        ]
    )

response = tokenizer.decode(outputs[0], skip_special_tokens=True)
print(response.strip())
```

### Cara 2: Multi-Turn Conversation

```python
messages = [
    {"role": "system", "content": "Kamu adalah asisten AI yang helpful dan ramah."},
    {"role": "user", "content": "Apa itu machine learning?"},
    {"role": "assistant", "content": "Machine learning adalah cabang AI di mana komputer belajar dari data tanpa diprogram secara eksplisit."},
    {"role": "user", "content": "Bisa kasih contoh penerapannya?"}
]

prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
# ... (lanjutkan dengan generate seperti di atas)
```

### Cara 3: Dengan Logit Masking (Opsional, untuk Kualitas Lebih Baik)

```python
import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

model_id = "daruokta/t5gemma-2-4b-4b-instruct-chat-indo-v3-unsloth"

tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
model = AutoModelForSeq2SeqLM.from_pretrained(
    model_id,
    trust_remote_code=True,
    torch_dtype=torch.bfloat16,
    device_map="auto"
)

# Logit Masking untuk memblokir unused & vision tokens
vocab_size = model.config.vocab_size
suppress_block1 = list(range(6, 105))         # <unused0>–<unused98>
suppress_block2 = list(range(256002, 262144))  # <unused100>–<unused6241>
suppress_vision = [255999, 256000, 256001]     # <end_of_image>, <image_soft_token>
suppress_ids = [i for i in (suppress_block1 + suppress_block2 + suppress_vision) if i < vocab_size]

mask = torch.zeros(vocab_size, dtype=torch.bfloat16)
mask[suppress_ids] = -10000.0

def forward_hook(module, inputs, outputs):
    if isinstance(outputs, torch.Tensor):
        return outputs + mask.to(outputs.device)
    elif hasattr(outputs, "logits"):
        outputs.logits = outputs.logits + mask.to(outputs.logits.device)
        return outputs
    return outputs

# Pasang hook di lm_head
if hasattr(model, "lm_head"):
    model.lm_head.register_forward_hook(forward_hook)
else:
    model.register_forward_hook(forward_hook)

# Sekarang generate seperti biasa
messages = [
    {"role": "system", "content": "Kamu adalah asisten AI yang helpful."},
    {"role": "user", "content": "Buatkan ringkasan tentang sejarah Indonesia."}
]

prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
inputs = tokenizer(prompt, return_tensors="pt", add_special_tokens=False).to(model.device)

with torch.no_grad():
    outputs = model.generate(
        **inputs,
        max_new_tokens=256,
        temperature=0.7,
        do_sample=True,
        repetition_penalty=1.2,
        no_repeat_ngram_size=3,
        eos_token_id=[
            tokenizer.convert_tokens_to_ids("<end_of_turn>"),
            tokenizer.eos_token_id
        ]
    )

response = tokenizer.decode(outputs[0], skip_special_tokens=True)
print(response.strip())
```

## 🔧 Chat Template Format

Template ini mengikuti format Gemma family yang disesuaikan untuk encoder-decoder:

```
<bos><start_of_turn>user
{system_prompt}

{user_message}<end_of_turn>
<start_of_turn>model
{model_response}<end_of_turn>
<start_of_turn>user
{next_user_message}<end_of_turn>
<start_of_turn>model
```

**Catatan:**
- System prompt di-prepend ke first user message (tidak ada token khusus untuk system role)
- Role `assistant` otomatis di-map ke `model` untuk kompatibilitas format OpenAI
- Template ini hanya menghasilkan **encoder input** — response dihasilkan oleh decoder secara terpisah

## 📊 Spesifikasi & Hyperparameter Pelatihan

| Parameter | Nilai |
| :--- | :--- |
| **Base Model** | `google/t5gemma-2-4b-4b` |
| **Framework** | Unsloth + Transformers |
| **Dataset** | `daruokta/t5gemma2-indonesia-chat-formatted` |
| **Konfigurasi** | `chat_sft` + `indoqa_sft` (multi-task) |
| **LoRA Rank (r)** | `256` |
| **LoRA Alpha (α)** | `512` |
| **Target Modules** | `q_proj, k_proj, v_proj, o_proj, gate_proj, up_proj, down_proj` |
| **LoRA Dropout** | `0.2` |
| **Epochs** | `5` |
| **Optimizer** | `GrokAdEMAMix` (grok_alpha=2.0, grok_lamb=0.98) |
| **Learning Rate** | `1e-5` (Cosine decay, 100 warmup steps) |
| **Batch Size** | `4` per device × `32` gradient accumulation |
| **Precision** | Mixed `bfloat16` |
| **Context Length** | Source: `2048` / Target: `512` |
| **Label Smoothing** | `0.1` |
| **NEFTune Noise Alpha** | `5.0` |
| **Weight Decay** | `0.1` |

## 🏗️ Arsitektur

T5Gemma-2 menggunakan arsitektur **encoder-decoder** (Seq2Seq) yang berbeda dari model Gemma standar (decoder-only):

```
┌─────────────────────┐    ┌─────────────────────┐
│      ENCODER        │    │      DECODER        │
│                     │    │                     │
│  Conversation       │───▶│  Response           │
│  Context + System   │    │  Generation         │
│  Prompt             │    │  (Autoregressive)   │
│                     │    │                     │
│  (Bidirectional     │    │  (Causal            │
│   Attention)        │    │   Attention)        │
└─────────────────────┘    └─────────────────────┘
```

**Keunggulan:**
- Encoder memproses seluruh konteks percakapan secara bidirectional
- Decoder fokus pada generasi response secara autoregressive
- Lebih efisien dalam parameter dibandingkan decoder-only untuk tugas instruksi

## ⚠️ Batasan & Lisensi

- **Lisensi:** Mengikuti [Gemma License](https://ai.google.dev/gemma/terms) dari Google.
- **Keterbatasan:**
  - Performa terbaik difokuskan pada Bahasa Indonesia dan Bahasa Inggris
  - Model masih bisa menghasilkan output repetitif pada kasus tertentu
  - Tidak dirancang untuk tugas visual/multimodal (token vision di-mask)

## 📚 Referensi & Publikasi Ilmiah

### Makalah Rujukan Utama

1. **T5Gemma 2: Seeing, Reading, and Understanding Longer**
   * Biao Zhang, et al. (Google DeepMind, 2025).
   * arXiv: [arXiv:2512.14856](https://arxiv.org/abs/2512.14856)

2. **Return of the Encoder: Maximizing Parameter Efficiency for SLMs**
   * Mohamed Elfeki, et al. (Microsoft, 2025).
   * arXiv: [arXiv:2501.16273](https://arxiv.org/abs/2501.16273)

3. **The AdEMAMix Optimizer: Better, Faster, Older**
   * Matteo Pagliardini, Pierre Ablin, and David Grangier (2024).
   * arXiv: [arXiv:2409.03137](https://arxiv.org/abs/2409.03137)

4. **Gemma 3 Technical Report**
   * Google DeepMind (2025).
   * arXiv: [arXiv:2503.19786](https://arxiv.org/abs/2503.19786)

5. **Cendol: Open Instruction-tuned Generative Large Language Models for Indonesian Languages**
   * Fajri Koto, et al. (2024).
   * arXiv: [arXiv:2404.06138](https://arxiv.org/abs/2404.06138)

### Sitasi BibTeX

```bibtex
@article{zhang2025t5gemma2,
  title={T5Gemma 2: Seeing, Reading, and Understanding Longer},
  author={Zhang, Biao and others},
  journal={arXiv preprint arXiv:2512.14856},
  year={2025}
}

@article{elfeki2025return,
  title={Return of the Encoder: Maximizing Parameter Efficiency for SLMs},
  author={Elfeki, Mohamed and others},
  journal={arXiv preprint arXiv:2501.16273},
  year={2025}
}

@article{pagliardini2024ademamix,
  title={The AdEMAMix Optimizer: Better, Faster, Older},
  author={Pagliardini, Matteo and Ablin, Pierre and Grangier, David},
  journal={arXiv preprint arXiv:2409.03137},
  year={2024}
}

@article{gemma3report,
  title={Gemma 3 Technical Report},
  author={DeepMind, Google},
  journal={arXiv preprint arXiv:2503.19786},
  year={2025}
}

@article{koto2024cendol,
  title={Cendol: Open Instruction-tuned Generative Large Language Models for Indonesian Languages},
  author={Koto, Fajri and others},
  journal={arXiv preprint arXiv:2404.06138},
  year={2024}
}
```
"""

# =====================================================================
# 3. Upload to HuggingFace
# =====================================================================

def main():
    print(f"Target repo: {REPO_ID}")
    
    # Ensure repo exists
    try:
        api.repo_info(repo_id=REPO_ID, repo_type="model")
        print(f"✅ Repo {REPO_ID} exists.")
    except Exception:
        print(f"Creating repo {REPO_ID}...")
        api.create_repo(repo_id=REPO_ID, repo_type="model", private=False, exist_ok=True)
        print(f"✅ Repo created.")
    
    # List existing files
    print("\nExisting files in repo:")
    try:
        files = api.list_repo_files(repo_id=REPO_ID, repo_type="model")
        for f in files:
            print(f"  - {f}")
    except Exception as e:
        print(f"  (Could not list files: {e})")
    
    with tempfile.TemporaryDirectory() as tmpdir:
        # Write chat_template.jinja
        template_path = os.path.join(tmpdir, "chat_template.jinja")
        with open(template_path, "w", encoding="utf-8") as f:
            f.write(CHAT_TEMPLATE)
        print(f"\n✅ Created: chat_template.jinja ({os.path.getsize(template_path)} bytes)")
        
        # Write README.md
        readme_path = os.path.join(tmpdir, "README.md")
        with open(readme_path, "w", encoding="utf-8") as f:
            f.write(README_CONTENT)
        print(f"✅ Created: README.md ({os.path.getsize(readme_path)} bytes)")
        
        # Upload chat_template.jinja
        print("\nUploading chat_template.jinja...")
        api.upload_file(
            path_or_fileobj=template_path,
            path_in_repo="chat_template.jinja",
            repo_id=REPO_ID,
            repo_type="model",
            commit_message="Add custom Jinja2 chat template for T5Gemma-2 encoder-decoder"
        )
        print("✅ chat_template.jinja uploaded!")
        
        # Upload Training Chart
        chart_path = r"d:\Codings\unsloth\t5-gemma-2\instruct\training_chart-v5-unsloth.png"
        if os.path.exists(chart_path):
            print("Uploading training chart...")
            api.upload_file(
                path_or_fileobj=chart_path,
                path_in_repo="training_chart-v5-unsloth.png",
                repo_id=REPO_ID,
                repo_type="model",
                commit_message="Add training and evaluation loss/ROUGE curve chart"
            )
            print("✅ Training chart uploaded!")
        else:
            print("⚠️ Training chart not found locally, skipping image upload.")
        
        # Upload README.md
        print("Uploading README.md...")
        api.upload_file(
            path_or_fileobj=readme_path,
            path_in_repo="README.md",
            repo_id=REPO_ID,
            repo_type="model",
            commit_message="Add comprehensive README with usage examples and chat template documentation"
        )
        print("✅ README.md uploaded!")
    
    print(f"\n🎉 Done! Check: https://huggingface.co/{REPO_ID}")


if __name__ == "__main__":
    main()
