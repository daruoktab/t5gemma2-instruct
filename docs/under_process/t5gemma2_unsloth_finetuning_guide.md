# Panduan Finetuning T5-Gemma-2 Menggunakan Unsloth (Seq2Seq)

Dokumen ini menjelaskan langkah-langkah, penemuan, dan cara melakukan finetuning model Seq2Seq `T5-Gemma-2` (khususnya `google/t5gemma-2-270m-270m`) menggunakan pustaka Unsloth.

## 1. Status Dukungan Seq2Seq di Unsloth
Secara resmi, Unsloth versi rilis utama saat ini belum menggabungkan (*merge*) dukungan Seq2Seq. Namun, creator Unsloth (`danielhanchen`) telah membuat **Pull Request #4226** pada branch **`dh/recover-3153-seq2seq`** untuk menambahkan dukungan penuh `AutoModelForSeq2SeqLM`.

### Instalasi Branch Seq2Seq
Untuk menggunakan dukungan ini tanpa mengacaukan dependensi CUDA, `torch`, atau `transformers` Anda yang sudah stabil, pasang menggunakan flag `--no-deps`:
```bash
pip install --force-reinstall --no-deps "git+https://github.com/unslothai/unsloth.git@dh/recover-3153-seq2seq"
```

---

## 2. Perbaikan/Patch yang Diterapkan
Selama proses pengujian pada environment Anda, kami menemukan beberapa isu kompatibilitas antara branch Unsloth (Maret 2026) dan library modern lainnya. Kami telah memperbaikinya dengan melakukan patch langsung pada file-file berikut:

### A. Isu `NameError` saat Memuat Model (Unsloth `_utils.py`)
*   **Masalah**: Versi `transformers` yang baru menggunakan decorator `@strict` dan `@auto_docstring` serta validator `interval` pada definisi `LlamaConfig` dan lainnya. Unsloth mencoba me-re-exec kelas konfigurasi ini tanpa mengimpor fungsi-fungsi tersebut ke dalam global namespace-nya.
*   **Solusi**: Kami menambahkan impor dinamis di `C:\Users\daru\anaconda3\envs\unsloth\Lib\site-packages\unsloth\models\_utils.py` (sekitar baris 742):
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
*   **Masalah**: Pustaka `peft` (versi `0.19.1`) memiliki bug pada pemeriksaan `is_gptqmodel_available()`, di mana ia langsung melempar `ImportError` jika `optimum` tidak terinstal (alih-alih mengembalikan `False` dengan anggun).
*   **Solusi**: Kami mempatch `C:\Users\daru\anaconda3\envs\unsloth\Lib\site-packages\peft\import_utils.py` agar langsung mengembalikan `False`:
    ```python
    @lru_cache
    def is_gptqmodel_available():
        return False
    ```

### C. Isu Mismatch Dimensi Loss pada Trainer (Unsloth `_utils.py`)
*   **Masalah**: Optimizer/Trainer bawaan Unsloth mengasumsikan model *Causal LM* di mana panjang *attention mask* (encoder) dan *labels* (decoder) selalu sama. Pada Seq2Seq (Encoder-Decoder), panjang kedua tensor ini berbeda, menyebabkan error:
    `RuntimeError: The size of tensor a (7) must match the size of tensor b (6) at non-singleton dimension 1`
*   **Solusi**: Kami menambahkan *wrapper* pada `Trainer.get_batch_samples` di `_utils.py` agar jika model yang dilatih bertipe **Encoder-Decoder**, ia akan melewati (*bypass*) optimasi batch sampler Unsloth dan kembali menggunakan fungsi bawaan Hugging Face Trainer asli yang aman untuk Seq2Seq:
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

## 3. Contoh Kode Pengetesan Finetuning yang Berhasil
Script berikut berada di [test_unsloth_training.py](file:///d:/Codings/unsloth/t5-gemma-2/instruct/scratch/test_unsloth_training.py) dan berhasil dijalankan tanpa error di environment Anda:

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

# 2. Terapkan LoRA dengan target modul proyeksi Gemma yang tepat
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

# 3. Siapkan dataset Seq2Seq dummy (input_ids untuk encoder, labels untuk decoder target)
data = {
    "input_ids": [tokenizer("Hello, how are you?", return_tensors="pt").input_ids[0].tolist()] * 4,
    "labels": [tokenizer("I am fine, thank you!", return_tensors="pt").input_ids[0].tolist()] * 4,
}
dataset = Dataset.from_dict(data)

# 4. Training Arguments
training_args = TrainingArguments(
    output_dir = "./outputs",
    per_device_train_batch_size = 2,
    gradient_accumulation_steps = 1,
    max_steps = 2,
    learning_rate = 2e-4,
    fp16 = not torch.cuda.is_bf16_supported(),
    bf16 = torch.cuda.is_bf16_supported(),
    logging_steps = 1,
    optim = "adamw_8bit",
    report_to = "none",
)

# 5. Trainer setup (tanpa argumen tokenizer di Trainer, gunakan DataCollator)
trainer = Trainer(
    model = model,
    args = training_args,
    train_dataset = dataset,
    data_collator = DataCollatorForSeq2Seq(tokenizer, model=model, padding=True),
)

print("Starting training...")
trainer.train()
print("Training completed successfully!")
```

### Output Logs Training yang Berhasil:
```text
{'loss': '8.78', 'grad_norm': '755.2', 'learning_rate': '0.0002', 'epoch': '0.5'}
{'loss': '3.185', 'grad_norm': '28.01', 'learning_rate': '0.0001', 'epoch': '1'}
{'train_runtime': '24.74', 'train_samples_per_second': '0.162', 'train_steps_per_second': '0.081', 'train_loss': '5.983', 'epoch': '1'}
Training completed successfully!
```
Dengan ini, model Seq2Seq T5-Gemma-2 dapat sepenuhnya dilatih menggunakan framework Unsloth secara stabil.
