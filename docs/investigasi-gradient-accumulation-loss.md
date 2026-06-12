# Investigasi: Gradient Accumulation Loss Normalization di Transformers T5/Gemma-2

**Tanggal Investigasi:** 2026-06-10
**Investigator:** Kimi (AI Assistant)
**Workspace:** `D:\Codings\unsloth\t5-gemma-2\instruct`
**Environment:** `conda env unsloth` (Windows, Anaconda)
**Transformers Version:** ~5.11.0 (sesuai dependency di script)

---

## 1. Konteks Permasalahan

Pengguna melaporkan bahwa saat training T5-Gemma-2 dengan `GRADIENT_ACCUMULATION_STEPS = 32`, **loss yang dicatat di log mencapai ~100**, padahal seharusnya loss per step sekitar ~2-5. Pengguna kemudian menambahkan pembagian manual di `compute_loss`:

```python
if self.args.gradient_accumulation_steps > 1:
    loss = loss / self.args.gradient_accumulation_steps
```

Pengguna bertanya: **Apakah fix ini benar? Apakah seharusnya `logs["loss"]` di callback juga perlu dibagi?**

---

## 2. File yang Diperiksa

### 2.1 File Script Training (User)

#### File 1: `working-molab-v4.py`
- **Lokasi:** `D:\Codings\unsloth\t5-gemma-2\instruct\working-molab-v4.py`
- **Bagian yang diinspeksi:**
  - Baris 1044-1079: `CustomSeq2SeqTrainer` dan `compute_loss` override
  - Baris 1073-1077: Fix pembagian manual `loss / gradient_accumulation_steps`
  - Baris 455-563: `TrainingPlotCallback` (logika `logs["loss"]`)
  - Baris 990-1041: `SelectiveLabelSmoother` custom

#### File 2: `working-molab-v4-exp.py`
- **Lokasi:** `D:\Codings\unsloth\t5-gemma-2\instruct\working-molab-v4-exp.py`
- **Bagian yang diinspeksi:**
  - Baris 1141-1177: `CustomSeq2SeqTrainer` dan `compute_loss` override
  - Baris 1171-1175: Fix pembagian manual `loss / gradient_accumulation_steps`
  - Baris 459-567: `TrainingPlotCallback`
  - Baris 936-1023: `GrokAdEMAMix` custom optimizer
  - Baris 1287-1306: Inisialisasi scheduler manual

### 2.2 File Source Transformers (Environment `unsloth`)

#### File 3: `transformers/trainer.py`
- **Lokasi:** `C:\Users\daru\anaconda3\envs\unsloth\Lib\site-packages\transformers\trainer.py`
- **Bagian krusial yang diinspeksi:**

| Baris | Method / Bagian | Keterangan |
|-------|-----------------|------------|
| ~496-506 | `model_accepts_loss_kwargs` assignment | Introspeksi model forward signature |
| ~1876-1940 | `training_step()` | Core training loop, gradient accumulation handling |
| ~1934-1936 | Pembagian loss otomatis | `loss / self.current_gradient_accumulation_steps` |
| ~1952-2030 | `compute_loss()` | Default HF loss computation logic |
| ~1984-1985 | `num_items_in_batch` passing | Forward kwargs handling |
| ~2050-2090 | `_maybe_log_save_evaluate()` | Loss logging normalization |
| ~2074-2079 | `logs["loss"] = tr_loss_scalar / ...` | Log loss per-step calculation |
| ~2123-2180 | `_get_num_items_in_batch()` | Batch item counting for loss scaling |
| ~3874-3890 | `log()` | Callback dispatch, logs ke `on_log()` |

---

## 3. Detail Inspeksi per Bagian

### 3.1 `model_accepts_loss_kwargs` (Baris ~496-506)

```python
# Di trainer.py baris ~496-506
if hasattr(unwrapped_model, "accepts_loss_kwargs"):
    self.model_accepts_loss_kwargs = unwrapped_model.accepts_loss_kwargs
else:
    forward_params = inspect.signature(unwrapped_model.forward).parameters
    self.model_accepts_loss_kwargs = any(
        k.kind == inspect.Parameter.VAR_KEYWORD for k in forward_params.values()
    )
```

**Temuan:**
- Untuk model T5 (`AutoModelForSeq2SeqLM`) seperti `google/t5gemma-2-1b-1b`, method `forward()` menerima `**kwargs` (var keyword parameter).
- Akibatnya: `model_accepts_loss_kwargs = True`.
- **Ini adalah kunci utama.** Transformers menentukan behavior gradient accumulation berdasarkan flag ini.

### 3.2 `training_step()` - Baris ~1876-1940

Method `training_step()` memanggil `compute_loss()` dan kemudian menormalisasi loss:

```python
# trainer.py ~1934-1936
if (not self.model_accepts_loss_kwargs or num_items_in_batch is None) and self.compute_loss_func is None:
    loss = loss / self.current_gradient_accumulation_steps
```

**Temuan:**
- Kondisi untuk pembagian otomatis: `(not self.model_accepts_loss_kwargs OR num_items_in_batch is None) AND self.compute_loss_func is None`.
- Untuk kasus kita:
  - `self.model_accepts_loss_kwargs = True` → kondisi `not self.model_accepts_loss_kwargs = False`.
  - `num_items_in_batch` dihitung dari labels (batch punya `labels`) → **tidak None**.
  - `self.compute_loss_func = None` → True.
- **Karena `model_accepts_loss_kwargs = True` dan `num_items_in_batch` tidak None, pembagian otomatis DILEWATI.**
- Loss yang dikembalikan `compute_loss()` (yang tidak dibagi) langsung di-`backward()` tanpa normalisasi accumulation.

### 3.3 `_get_num_items_in_batch()` - Baris ~2123-2180

```python
def _get_num_items_in_batch(self, batch_samples: list, device: torch.device):
    num_items_in_batch = None
    count_num_items_in_batch = (
        len(batch_samples) > 0
        and "labels" in batch_samples[0]
        and (
            self.model_accepts_loss_kwargs  # <-- True untuk T5
            or self.compute_loss_func is not None
        )
    )
    if count_num_items_in_batch:
        labels_for_count = [...]
        num_items_in_batch = sum(labels.ne(-100).sum() for labels in labels_for_count)
```

**Temuan:**
- Karena dataset SFT memiliki key `labels` di setiap batch, dan `model_accepts_loss_kwargs = True`, maka `num_items_in_batch` dihitung.
- `num_items_in_batch` jadi scalar tensor (bukan `None`).
- Ini menyebabkan `training_step()` melewati pembagian otomatis.

### 3.4 `compute_loss()` - Baris ~1952-2030

```python
def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
    if self.model_accepts_loss_kwargs:
        kwargs = {}
        if num_items_in_batch is not None:
            kwargs["num_items_in_batch"] = num_items_in_batch
        inputs = {**inputs, **kwargs}
    outputs = model(**inputs)
    # ... label smoother logic ...
    return (loss, outputs) if return_outputs else loss
```

**Temuan:**
- `compute_loss` default di `Trainer` akan pass `num_items_in_batch` ke `model.forward()` jika model menerimanya.
- T5/Gemma-2 tidak menggunakan `num_items_in_batch` untuk loss calculation (model return loss dari internal `forward`).
- Namun karena `num_items_in_batch` ada, `training_step` menganggap "sudah ada mekanisme loss scaling" dan tidak membagikan lagi.
- **Padahal model tidak benar-benar memanfaatkan `num_items_in_batch` untuk scale loss.**

### 3.5 `_maybe_log_save_evaluate()` - Baris ~2050-2090

```python
# trainer.py ~2074-2079
if self.control.should_log and self.state.global_step > self._globalstep_last_logged:
    tr_loss_scalar = nested_gather(tr_loss, self.args.parallel_mode).mean().item()
    tr_loss -= tr_loss  # reset
    logs["loss"] = tr_loss_scalar / (self.state.global_step - self._globalstep_last_logged)
    # ... other logs ...
    self.log(logs, start_time)
```

**Temuan:**
- `logs["loss"]` yang sampai ke `on_log()` callback adalah **rata-rata per global step**.
- Log sudah dibagi dengan selisih step (`self.state.global_step - self._globalstep_last_logged`).
- **Tapi `tr_loss` itu sendiri sudah terakumulasi mentah dari `training_step()` yang tidak dibagi.**
- Jadi `logs["loss"]` = `sum_of_unnormalized_losses / num_steps` = masih ~80-100.

### 3.6 `log()` - Baris ~3874-3890

```python
def log(self, logs: dict[str, float], start_time: float | None = None):
    if self.state.epoch is not None:
        logs["epoch"] = self.state.epoch
    # ... speed metrics ...
    output = {**logs, "step": self.state.global_step}
    self.state.log_history.append(output)
    self.control = self.callback_handler.on_log(self.args, self.state, self.control, logs)
```

**Temuan:**
- `log()` hanya meneruskan `logs` (yang sudah berisi `loss`) ke callback handler.
- Tidak ada normalisasi tambahan di sini.
- `TrainingPlotCallback.on_log()` menerima `logs["loss"]` yang sudah ada.

---

## 4. Logika Matematika yang Terjadi

### Tanpa Fix (Sebelumnya)

```
Per micro-batch loss: ~2.5
Micro-batch per update: 32 (GRADIENT_ACCUMULATION_STEPS)
tr_loss per step (diakumulasi): 32 × 2.5 = 80.0
logs["loss"] = 80.0 / 1 = 80.0  (karena 1 log per step)
```

### Dengan Fix Manual (Saat Ini)

```python
# Di CustomSeq2SeqTrainer.compute_loss()
if self.args.gradient_accumulation_steps > 1:
    loss = loss / self.args.gradient_accumulation_steps
```

```
Per micro-batch loss: ~2.5
Setelah pembagian: 2.5 / 32 = 0.078
tr_loss per step (diakumulasi): 32 × 0.078 = 2.5
logs["loss"] = 2.5 / 1 = 2.5  ✅ Benar!
```

### Catatan: `logs["loss"]` di Callback

Di `TrainingPlotCallback`:

```python
actual_loss = float(logs["loss"])
self.train_losses.append(actual_loss)
```

**Ini sudah benar.** `logs["loss"]` sudah di-normalisasi per step (dibagi selisih step). Yang salah bukan di callback, tapi di `compute_loss()` yang tidak membagikan loss sebelum akumulasi.

---

## 5. Analisis `SelectiveLabelSmoother` (Custom)

File: `working-molab-v4.py` baris 990-1041

```python
class SelectiveLabelSmoother:
    def __init__(self, epsilon, suppress_ids):
        self.epsilon = epsilon
        self.suppress_ids = suppress_ids

    def __call__(self, model_output, labels, shift_labels=False):
        # ... logit extraction ...
        # ... valid mask buat suppressed tokens ...
        log_probs = torch.nn.functional.log_softmax(active_logits, dim=-1)
        nll_loss = -log_probs.gather(dim=-1, index=active_labels.unsqueeze(-1)).squeeze(-1)
        valid_log_probs = log_probs * valid_mask.to(log_probs.dtype)
        smooth_loss = -valid_log_probs.sum(dim=-1) / num_valid_tokens
        token_losses = (1.0 - self.epsilon) * nll_loss + self.epsilon * smooth_loss
        return token_losses.mean()  # <-- Mean per token
```

**Validasi:**
- `token_losses.mean()` menghitung **rata-rata per token**, bukan sum total.
- Ini konsisten dengan behavior default `CrossEntropyLoss(reduction="mean")`.
- Loss yang dikembalikan ~2.5 adalah realistic untuk model SFT di awal training.
- `SelectiveLabelSmoother` tidak menyebabkan loss bengkak; yang menyebabkan adalah **gradient accumulation tanpa normalisasi**.

---

## 6. Analisis `GrokAdEMAMix` (Custom Optimizer, v4-exp)

File: `working-molab-v4-exp.py` baris 936-1023

Optimizer custom ini menggabungkan:
- **Grokfast**: Grok-knowledge slow gradient dengan EMA (`grok_slow_grad`).
- **AdEMAMix**: Second-order moment `n` (beta3) untuk long-term memory.
- **AdamW**: Standard first-order dengan bias correction.

Logika update:
```python
state["grok_slow_grad"].mul_(grok_lamb).add_(grad, alpha=1.0 - grok_lamb)
filtered_grad = grad.clone().add_(state["grok_slow_grad"], alpha=grok_alpha)
# ... m, v, n updates ...
# ... step update ...
```

**Validasi:**
- Logika Grokfast (`filtered_grad = grad + alpha * slow_grad`) sudah sesuai paper.
- AdEMAMix (`n` term dengan beta3=0.9999) sudah sesuai.
- Weight decay applied via `p.data.mul_(1.0 - lr * weight_decay)`.
- **Tidak ada interaksi dengan loss normalization** — optimizer hanya menerima gradient, tidak peduli loss dari mana.

---

## 7. Perbandingan Kedua File (v4 vs v4-exp)

| Aspek | `working-molab-v4.py` | `working-molab-v4-exp.py` | Catatan |
|-------|----------------------|--------------------------|---------|
| Optimizer | `paged_adamw_8bit` (via `OPTIM`) | `GrokAdEMAMix` (custom) | v4-exp pakai optimizer custom |
| Scheduler | Auto oleh `Trainer` | Manual `get_scheduler()` | v4-exp pass `(optimizer, lr_scheduler)` ke `Trainer` |
| Fix `loss / GAS` | ✅ Ada | ✅ Ada | Keduanya sama |
| `SelectiveLabelSmoother` | ✅ Ada | ✅ Ada | Keduanya sama |
| Logit masking | ✅ Ada | ✅ Ada | Keduanya sama |
| `max_steps` calculation | N/A (auto) | Manual di script | v4-exp: `len(train_ds) // (batch_size * GAS) * epochs` |
| `model_accepts_loss_kwargs` | Default (True) | Default (True) | Keduanya sama |

**Catatan `max_steps` di v4-exp:**
```python
num_update_steps_per_epoch = max(1, len(train_ds) // (PER_DEVICE_TRAIN_BATCH_SIZE * GRADIENT_ACCUMULATION_STEPS))
max_steps = num_update_steps_per_epoch * NUM_EPOCHS
```
- Untuk single GPU: **Valid dan aman**.
- Untuk multi-GPU (DataParallel / Distributed): Perlu `// world_size` atau biarkan `Trainer` yang hitung.

---

## 8. Kesimpulan Akhir

### 8.1 Kenapa Loss 100an?

**Root cause bukan bug di kode user, melainkan behavior internal Transformers yang "counter-intuitive":**

1. Transformers `Trainer` introspect `model.forward()` signature.
2. T5/Gemma-2 `forward()` menerima `**kwargs` → `model_accepts_loss_kwargs = True`.
3. Karena `model_accepts_loss_kwargs = True` dan `num_items_in_batch` dihitung (batch punya `labels`), `Trainer` **melewati pembagian otomatis** `loss / gradient_accumulation_steps`.
4. `Trainer` menganggap model "sudah handle loss scaling via `num_items_in_batch`", padahal T5/Gemma-2 tidak memanfaatkan `num_items_in_batch` untuk scale loss.
5. Akibatnya: loss mentah per micro-batch diakumulasi 32× tanpa normalisasi → log loss ~80-100.

### 8.2 Apakah Fix `loss / gradient_accumulation_steps` Benar?

**YA, fix ini benar dan diperlukan.**

- Fix memastikan loss yang diakumulasi dan di-log adalah **loss per update step** (bukan total akumulasi).
- Gradient scale juga jadi benar karena `loss / 32` → `backward()` mendapat gradient yang lebih kecil sesuai expected.

### 8.3 Apakah `logs["loss"]` di Callback Perlu Dibagi Lagi?

**TIDAK.** `logs["loss"]` yang sampai ke `on_log()` sudah di-normalisasi per step oleh `Trainer._maybe_log_save_evaluate()`. Yang salah adalah `tr_loss` sebelumnya yang sudah bengkak.

```python
# Di TrainingPlotCallback — SUDAH BENAR
def on_log(self, args, state, control, logs=None):
    if "loss" in logs:
        actual_loss = float(logs["loss"])  # ✅ Tidak perlu dibagi lagi
        self.train_losses.append(actual_loss)
```

### 8.4 Alternatif Fix (Lebih Clean)

Daripada bagi manual di `compute_loss`, cara yang lebih "proper" menurut Transformers:

```python
class CustomSeq2SeqTrainer(Seq2SeqTrainer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.model_accepts_loss_kwargs = False  # <-- Force disable
        if self.args.label_smoothing_factor > 0:
            self.label_smoother = SelectiveLabelSmoother(...)
```

Dengan `model_accepts_loss_kwargs = False`, `Trainer` akan otomatis membagi `loss / current_gradient_accumulation_steps` di `training_step()` (baris ~1934), dan pembagian manual di `compute_loss` bisa dihapus.

---

## 9. Referensi Source Transformers

- File: `C:\Users\daru\anaconda3\envs\unsloth\Lib\site-packages\transformers\trainer.py`
- Baris krusial: 496, 1876, 1934, 1952, 2050, 2074, 2123, 3874
- Transformers dependency: `transformers==5.11.0` (di script)

---

*Dokumen ini dibuat untuk arsip investigasi dan referensi maintenance di masa depan.*
