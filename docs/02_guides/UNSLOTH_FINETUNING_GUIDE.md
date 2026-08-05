# Panduan Finetuning T5-Gemma-2 Menggunakan Unsloth (Seq2Seq)

**Last Updated:** 5 Agustus 2026  
**Target:** `google/t5gemma-2-270m-270m` | `google/t5gemma-2-4b-4b`

---

## 1. Status Dukungan Seq2Seq di Unsloth

Secara resmi, Unsloth versi rilis utama belum menggabungkan (*merge*) penuh dukungan Seq2Seq. Namun, creator Unsloth (`danielhanchen`) membuat **Pull Request #4226** pada branch **`dh/recover-3153-seq2seq`** untuk menambahkan dukungan `AutoModelForSeq2SeqLM`.

### Instalasi Branch Seq2Seq
Pasang menggunakan flag `--no-deps` agar tidak mengacaukan dependensi CUDA, `torch`, atau `transformers` yang sudah stabil:

```bash
pip install --force-reinstall --no-deps "git+https://github.com/unslothai/unsloth.git@dh/recover-3153-seq2seq"
```

---

## 2. Perbaikan & Patch Wajib (Environment Fixes)

Berikut adalah patch kritis yang telah teruji untuk mengatasi error pertentangan versi library:

### A. Isu `NameError` saat Memuat Model (Unsloth `_utils.py`)
Versi `transformers` baru menggunakan decorator `@strict` dan `@auto_docstring`. Tambahkan impor dinamis di `unsloth/models/_utils.py`:

```python
try:
    from transformers.utils import auto_docstring
except:
    auto_docstring = lambda x: x
try:
    from transformers.utils.validation import strict
except:
    strict = lambda x: x
try:
    from transformers.utils.type_validators import *
except:
    pass
```

### B. Isu `ImportError` dari PEFT (`import_utils.py`)
Pada `peft` (misal v0.19+), `is_gptqmodel_available()` melempar `ImportError` jika `optimum` tidak terinstal. Patch `peft/import_utils.py`:

```python
@lru_cache
def is_gptqmodel_available():
    return False
```

### C. Isu Mismatch Dimensi Loss pada Trainer (Unsloth `_utils.py`)
Batch sampler Unsloth mengasumsikan model *Causal LM* di mana panjang *attention mask* (encoder) dan *labels* (decoder) selalu sama. Pada Seq2Seq, panjang tensor ini berbeda, menyebabkan `RuntimeError: The size of tensor a must match the size of tensor b`.

Solusi: Bungkus `Trainer.get_batch_samples` di `_utils.py` agar me-bypass optimasi batch sampler Unsloth jika `is_encoder_decoder == True`:

```python
def _wrapped_get_batch_samples(self, epoch_iterator, num_batches, device):
    model = getattr(self, "model", None)
    is_encoder_decoder = False
    if model is not None:
        config = getattr(model, "config", None)
        is_encoder_decoder = getattr(config, "is_encoder_decoder", False)
        if not is_encoder_decoder and hasattr(model, "base_model"):
            base_config = getattr(model.base_model, "config", None)
            is_encoder_decoder = getattr(base_config, "is_encoder_decoder", False)
            if not is_encoder_decoder and hasattr(model.base_model, "model"):
                model_config = getattr(model.base_model.model, "config", None)
                is_encoder_decoder = getattr(model_config, "is_encoder_decoder", False)
    
    if is_encoder_decoder:
        return self._original_get_batch_samples(epoch_iterator, num_batches, device)
    else:
        return _unsloth_get_batch_samples(self, epoch_iterator, num_batches, device)
```

---

## 3. Script Finetuning Teruji (Minimal Working Example)

```python
import torch
from unsloth import FastLanguageModel
from transformers import Trainer, TrainingArguments, DataCollatorForSeq2Seq
from datasets import Dataset

# 1. Load model dan tokenizer
model_name = "google/t5gemma-2-270m-270m"
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name = model_name,
    max_seq_length = 512,
    load_in_4bit = False,
    trust_remote_code = True,
)

# 2. Terapkan LoRA
model = FastLanguageModel.get_peft_model(
    model,
    r = 16,
    target_modules = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    lora_alpha = 16,
    lora_dropout = 0,
    bias = "none",
    use_gradient_checkpointing = "unsloth",
    random_state = 3407,
)

# 3. Training setup
trainer = Trainer(
    model = model,
    tokenizer = tokenizer,
    data_collator = DataCollatorForSeq2Seq(tokenizer = tokenizer, model = model),
    args = TrainingArguments(
        per_device_train_batch_size = 2,
        gradient_accumulation_steps = 4,
        warmup_steps = 5,
        max_steps = 60,
        learning_rate = 2e-4,
        fp16 = not torch.cuda.is_bf16_supported(),
        bf16 = torch.cuda.is_bf16_supported(),
        logging_steps = 1,
        output_dir = "outputs",
    ),
)
```
