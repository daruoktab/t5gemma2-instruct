"""
=======================================================================
VISION TRAINING MECHANISM DIAGNOSTIC SCRIPT
=======================================================================
Tujuan: Menganalisa semua mekanisme kode working-molab-v6-vision-unsloth.py
        tanpa GPU, menggunakan mock model + synthetic data.

Setiap TEST section menganalisa satu mekanisme spesifik dengan logging
verbose agar bug terlihat dengan jelas.

Jalankan dengan:
    conda activate unsloth-env
    python vision_diagnostic.py
=======================================================================
"""

import sys
import math
import traceback
import warnings
warnings.filterwarnings("ignore")

import torch
import torch.nn as nn
import torch.nn.functional as F

# =====================================================================
# LOGGING UTILITIES
# =====================================================================

BOLD = "\033[1m"
RED = "\033[91m"
GREEN = "\033[92m"
YELLOW = "\033[93m"
BLUE = "\033[94m"
CYAN = "\033[96m"
RESET = "\033[0m"
DIM = "\033[2m"

_pass_count = 0
_fail_count = 0
_warn_count = 0

def header(title):
    print(f"\n{'='*70}")
    print(f"{BOLD}{CYAN}  {title}{RESET}")
    print(f"{'='*70}")

def subheader(title):
    print(f"\n{BOLD}{BLUE}--- {title} ---{RESET}")

def log(msg, indent=2):
    print(" " * indent + msg)

def log_ok(msg):
    global _pass_count
    _pass_count += 1
    print(f"  {GREEN}[PASS]{RESET} {msg}")

def log_fail(msg, detail=None):
    global _fail_count
    _fail_count += 1
    print(f"  {RED}[FAIL]{RESET} {msg}")
    if detail:
        print(f"     {DIM}{detail}{RESET}")

def log_warn(msg, detail=None):
    global _warn_count
    _warn_count += 1
    print(f"  {YELLOW}[WARN]{RESET} {msg}")
    if detail:
        print(f"     {DIM}{detail}{RESET}")

def log_info(msg):
    print(f"  {DIM}[INFO] {msg}{RESET}")

def log_data(label, value, width=60):
    val_str = str(value)
    if len(val_str) > width:
        val_str = val_str[:width] + "..."
    print(f"     {BOLD}{label}:{RESET} {val_str}")

def summary():
    total = _pass_count + _fail_count + _warn_count
    print(f"\n{'='*70}")
    print(f"{BOLD}DIAGNOSTIC SUMMARY{RESET}")
    print(f"{'='*70}")
    print(f"  Total checks : {total}")
    print(f"  {GREEN}Passed{RESET}       : {_pass_count}")
    print(f"  {RED}Failed{RESET}       : {_fail_count}")
    print(f"  {YELLOW}Warnings{RESET}     : {_warn_count}")
    if _fail_count > 0:
        print(f"\n  {RED}{BOLD}>>> {_fail_count} BUG(S) CONFIRMED!{RESET}")
    else:
        print(f"\n  {GREEN}{BOLD}>>> No critical bugs detected.{RESET}")
    print(f"{'='*70}\n")


# =====================================================================
# MOCK COMPONENTS (mirror exact interfaces from vision code)
# =====================================================================

class MockTokenizer:
    """
    Mock tokenizer behaving like Gemma3Tokenizer.
    KEY: bos_token_id=2, eos_token_id=1, add_bos_token controls encode()
    """
    def __init__(self, add_bos_token=True):
        self.bos_token_id = 2
        self.eos_token_id = 1
        self.pad_token_id = 0
        self.bos_token = "<bos>"
        self.eos_token = "<eos>"
        self.pad_token = "<pad>"
        self.add_bos_token = add_bos_token
        self.chat_template = None
        self._vocab = {
            "<pad>": 0, "<eos>": 1, "<bos>": 2,
            "<end_of_turn>": 107, "<start_of_turn>": 106,
            "Halo": 500, "apa": 501, "kabar": 502, "kamu": 503,
            "adalah": 504, "model": 505, "AI": 506, "yang": 507,
            "helpful": 508, "user": 509, "sebuah": 510,
            "gambar": 511, "menunjukkan": 512, "ini": 513, "foto": 515,
        }
        self._inv_vocab = {v: k for k, v in self._vocab.items()}

    def encode(self, text, add_special_tokens=True):
        tokens = []
        if add_special_tokens and self.add_bos_token:
            tokens.append(self.bos_token_id)
        for word in text.split():
            tokens.append(self._vocab.get(word, 999))
        if add_special_tokens:
            tokens.append(self.eos_token_id)
        return tokens

    def decode(self, ids, skip_special_tokens=True):
        skip_ids = {self.bos_token_id, self.eos_token_id, self.pad_token_id} if skip_special_tokens else set()
        words = []
        for id_ in (ids.tolist() if hasattr(ids, "tolist") else ids):
            if id_ in skip_ids:
                continue
            words.append(self._inv_vocab.get(id_, f"<unk_{id_}>"))
        return " ".join(words)

    def batch_decode(self, ids_list, skip_special_tokens=True):
        return [self.decode(ids, skip_special_tokens) for ids in ids_list]

    def convert_tokens_to_ids(self, token):
        return self._vocab.get(token, 999)


class MockProcessor:
    """
    Mock processor mimicking Gemma3Processor.

    KEY BEHAVIOR to simulate:
    - Gemma3Processor.__call__() ALWAYS adds BOS token in output
      (hardcoded in chat template logic, NOT via tokenizer.add_bos_token)
    - Gemma3Processor.__call__() ALWAYS adds EOS token at end
    - If add_bos_token=False on the tokenizer, it only affects
      tokenizer.encode() calls, NOT processor.__call__() output
    """
    def __init__(self):
        self.tokenizer = MockTokenizer(add_bos_token=True)
        self.chat_template = "gemma-3"

    def __call__(self, text, images=None, return_tensors=None):
        """
        Simulate Gemma3Processor behavior:
        BOS is ALWAYS added regardless of tokenizer.add_bos_token
        """
        tokens = []

        # Gemma3Processor ALWAYS adds BOS (hardcoded in processor internals)
        tokens.append(self.tokenizer.bos_token_id)  # ID=2

        # Add image tokens if images present
        if images is not None:
            img_list = images if isinstance(images, list) else [images]
            for _ in img_list:
                tokens.append(255999)         # BOI token
                tokens.extend([256001] * 4)   # 4 image tokens (256 in real)
                tokens.append(256000)          # EOI token

        # Add text tokens
        for word in text.split():
            tokens.append(self.tokenizer._vocab.get(word, 999))

        # Gemma3Processor ALWAYS adds EOS
        tokens.append(self.tokenizer.eos_token_id)  # ID=1

        attention_mask = [1] * len(tokens)
        result = {
            "input_ids": torch.tensor([tokens], dtype=torch.long),
            "attention_mask": torch.tensor([attention_mask], dtype=torch.long),
        }
        if images is not None:
            n_imgs = len(images) if isinstance(images, list) else 1
            result["pixel_values"] = torch.randn(n_imgs, 3, 224, 224)
        return result

    def apply_chat_template(self, messages, tokenize=False, add_generation_prompt=True):
        result = ""
        for msg in messages:
            role = msg.get("role", "user")
            content = msg.get("content", "")
            if isinstance(content, list):
                text_parts = [b.get("text", "") for b in content
                              if isinstance(b, dict) and "text" in b]
                has_image = any(b.get("type") == "image"
                                for b in content if isinstance(b, dict))
                content = ("IMG " if has_image else "") + " ".join(text_parts)
            result += f"<start_of_turn>{role}\n{content}<end_of_turn>\n"
        if add_generation_prompt:
            result += "<start_of_turn>model\n"
        return result

    def decode(self, ids, skip_special_tokens=True):
        return self.tokenizer.decode(ids, skip_special_tokens)

    def save_pretrained(self, path):
        pass


# =====================================================================
# MOCK MODEL
# =====================================================================

class MockEncoder(nn.Module):
    """Tracks whether pixel_values was received"""
    def __init__(self, hidden_size=32):
        super().__init__()
        self.hidden_size = hidden_size
        self.embed = nn.Embedding(262144, hidden_size, padding_idx=0)
        self._pixel_values_received = False
        self._pixel_values_shape = None
        self._call_count = 0

    def forward(self, input_ids=None, attention_mask=None, pixel_values=None, **kwargs):
        self._call_count += 1
        self._pixel_values_received = pixel_values is not None
        self._pixel_values_shape = pixel_values.shape if pixel_values is not None else None
        safe_ids = input_ids.clamp(0, 262143) if input_ids is not None else None
        hidden = self.embed(safe_ids) if safe_ids is not None else torch.zeros(1, 1, self.hidden_size)
        return type("EncOut", (), {"last_hidden_state": hidden})()

    def get_encoder(self):
        return self


class MockModelOutput:
    def __init__(self, loss, logits):
        self.loss = loss
        self.logits = logits


class MockBaseModelWrapper:
    """Non-nn.Module wrapper to avoid circular registration in MockModel.base_model"""
    def __init__(self, model):
        # Use plain object to avoid nn.Module submodule registration
        object.__setattr__(self, "model", model)

    def __getattr__(self, name):
        return getattr(object.__getattribute__(self, "model"), name)


class MockModel(nn.Module):
    TRAINING_KERNELS_ACTIVE = False

    def __init__(self, vocab_size=262144, hidden_size=32):
        super().__init__()
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.encoder = MockEncoder(hidden_size)
        self.decoder_embed = nn.Embedding(vocab_size, hidden_size, padding_idx=0)
        self.lm_head = nn.Linear(hidden_size, vocab_size, bias=False)
        self.config = type("Cfg", (), {
            "vocab_size": vocab_size,
            "decoder_start_token_id": 2,
            "is_encoder_decoder": True,
            "use_cache": False,
            "max_length": None,
        })()
        self.generation_config = type("GenCfg", (), {"max_length": None})()
        self._forward_calls = []
        self._training_mode_log = []
        # CRITICAL FIX: Store base_model as plain Python attribute (NOT as nn.Module),
        # otherwise nn.Module registers it as submodule -> circular train() recursion.
        # Use object.__setattr__ to bypass nn.Module.__setattr__.
        object.__setattr__(self, "base_model", MockBaseModelWrapper(self))

    def get_encoder(self):
        return self.encoder

    def forward(self, input_ids=None, attention_mask=None, labels=None,
                pixel_values=None, encoder_outputs=None, **kwargs):
        call_info = {
            "pixel_values_received": pixel_values is not None,
            "pixel_values_shape": pixel_values.shape if pixel_values is not None else None,
            "encoder_outputs_received": encoder_outputs is not None,
            "training": self.training,
            "unsloth_training_kernels": self.TRAINING_KERNELS_ACTIVE,
        }
        self._forward_calls.append(call_info)

        if encoder_outputs is None:
            enc_out = self.encoder(input_ids=input_ids,
                                   attention_mask=attention_mask,
                                   pixel_values=pixel_values)
        else:
            enc_out = encoder_outputs

        if labels is not None:
            safe_labels = labels.clamp(0, self.vocab_size - 1)
            decoder_hidden = self.decoder_embed(safe_labels)
            logits = self.lm_head(decoder_hidden)
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            flat_labels = shift_labels.reshape(-1).clamp(-100, self.vocab_size - 1)
            loss = nn.CrossEntropyLoss(ignore_index=-100)(
                shift_logits.reshape(-1, self.vocab_size), flat_labels)
        else:
            seq_len = input_ids.shape[1] if input_ids is not None else 8
            logits = torch.randn(1, seq_len, self.vocab_size)
            loss = torch.tensor(0.0)

        return MockModelOutput(loss=loss, logits=logits)

    def eval(self):
        # Only log once at top level (not for every child module call)
        if self.training:  # only if state actually changes
            self._training_mode_log.append("eval()")
        return super().eval()

    def train(self, mode=True):
        # Only log once at top level (not for every child module call)
        if not self.training and mode:  # going from eval -> train
            self._training_mode_log.append(f"train({mode})")
        return super().train(mode)

    def zero_grad(self, set_to_none=True):
        super().zero_grad(set_to_none=set_to_none)

    def print_trainable_parameters(self):
        n = sum(p.numel() for p in self.parameters())
        log_info(f"Mock model: {n:,} params")

    def save_pretrained(self, path):
        pass


# =====================================================================
# EXACT COPY OF VISION CODE CLASSES
# =====================================================================

from torch.nn.utils.rnn import pad_sequence

class Seq2SeqVisionCollator_ORIGINAL:
    """EXACT copy dari kode asli line 489-545"""
    def __init__(self, processor, max_src, max_tgt, train_dataset=None):
        self.processor = processor
        self.tok = processor.tokenizer
        self.pad_id = self.tok.pad_token_id
        self.eos_id = self.tok.eos_token_id
        self.max_src = max_src
        self.max_tgt = max_tgt
        self.train_dataset = train_dataset

    def __call__(self, batch):
        iids, amasks, pvals, labs = [], [], [], []
        for item in batch:
            images = None
            if "images" in item and item["images"]:
                images = item["images"]

            enc = self.processor(
                text=item["prompt_text"],
                images=images if images else None,
                return_tensors="pt")

            input_ids = enc["input_ids"][0].tolist()
            attention_mask = enc["attention_mask"][0].tolist()

            # === SUSPECTED BUG #1 ===
            if self.tok.bos_token_id is not None and (
                    not input_ids or input_ids[0] != self.tok.bos_token_id):
                input_ids = [self.tok.bos_token_id] + input_ids
                attention_mask = [1] + attention_mask

            if self.tok.eos_token_id is not None and (
                    not input_ids or input_ids[-1] != self.tok.eos_token_id):
                input_ids = input_ids + [self.tok.eos_token_id]
                attention_mask = attention_mask + [1]

            iids.append(torch.tensor(input_ids, dtype=torch.long))
            amasks.append(torch.tensor(attention_mask, dtype=torch.long))

            if "pixel_values" in enc:
                pvals.append(enc["pixel_values"])

            target_formatted = item["target_text"].strip() + "<end_of_turn>"
            tids = self.tok.encode(target_formatted, add_special_tokens=False)
            tids = tids[:self.max_tgt - 1] + [self.eos_id]
            labs.append(torch.tensor(tids, dtype=torch.long))

        ii = pad_sequence(iids, batch_first=True, padding_value=self.pad_id)
        am = pad_sequence(amasks, batch_first=True, padding_value=0)
        lb = pad_sequence(labs, batch_first=True, padding_value=-100)
        out = {"input_ids": ii, "attention_mask": am, "labels": lb}
        if pvals:
            out["pixel_values"] = (torch.cat(pvals, dim=0)
                                   if pvals[0].ndim == 4
                                   else torch.stack(pvals, dim=0))
        return out


class VisionORPOTrainer_ComputeLoss_ORIGINAL:
    """Reproduksi exact VisionORPOTrainer.compute_loss (line 620-655)"""
    def __init__(self, beta=0.1):
        self.beta = beta

    def get_batch_logps(self, logits, labels, average_log_prob=True):
        labels = labels.clone()
        mask = labels != -100
        labels[labels == -100] = 0
        lps = torch.gather(
            logits.log_softmax(-1), dim=2,
            index=labels.unsqueeze(2)).squeeze(2)
        if average_log_prob:
            return (lps * mask).sum(-1) / mask.sum(-1).clamp(min=1)
        return (lps * mask).sum(-1)

    def compute_loss_original(self, model, inputs):
        """EXACT logic dari kode asli"""
        cl = inputs.get("chosen_labels")
        rl = inputs.get("rejected_labels")
        if cl is None or rl is None:
            raise ValueError("Missing chosen/rejected labels")

        # === SUSPECTED BUG #2: Encoder dipanggil langsung ===
        base_model = (model.base_model.model
                      if hasattr(model, "base_model") and hasattr(model.base_model, "model")
                      else model)

        if hasattr(base_model, "get_encoder"):
            encoder = base_model.get_encoder()
        elif (hasattr(base_model, "model") and
              hasattr(base_model.model, "encoder")):
            encoder = base_model.model.encoder
        else:
            encoder = base_model.encoder

        pixel_values = inputs.get("pixel_values")
        print(f"     {DIM}[VisionORPOTrainer.compute_loss] calling encoder directly...")
        print(f"     pixel_values present in inputs: {pixel_values is not None}")
        print(f"     encoder type: {type(encoder).__name__}{RESET}")

        encoder_outputs = encoder(
            input_ids=inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
            pixel_values=pixel_values,  # passed here — but does it reach the fused encoder?
        )

        co = model(encoder_outputs=encoder_outputs, labels=cl)
        ro = model(encoder_outputs=encoder_outputs, labels=rl)

        clp = self.get_batch_logps(co.logits, cl)
        rlp = self.get_batch_logps(ro.logits, rl)
        cp = clp.exp().clamp(1e-7, 1 - 1e-7)
        rp = rlp.exp().clamp(1e-7, 1 - 1e-7)
        clo = torch.log(cp / (1 - cp))
        rlo = torch.log(rp / (1 - rp))
        or_loss = -F.logsigmoid(clo - rlo).mean()
        loss = co.loss + self.beta * or_loss
        return loss


class CustomSeq2SeqTrainer_Evaluate_VISION:
    """Exact logic dari vision CustomSeq2SeqTrainer.evaluate(), line 863-898"""
    def __init__(self, model):
        self.model = model
        self._evaluate_log = []

    def evaluate(self, metric_key_prefix="eval"):
        import math, gc
        entry = {
            "for_inference_called_before": False,
            "for_training_called_after": False,
            "model_training_before": self.model.training,
        }

        gc.collect()

        # === SUSPECTED BUG #3: Hanya .eval(), tidak ada for_inference() ===
        self.model.eval()  # text-only: FastLanguageModel.for_inference(self.model) dipanggil

        fake_metrics = {f"{metric_key_prefix}_loss": 2.5}
        # === BUG #3 cont: tidak ada model.train() / for_training() setelah ini ===
        torch._dynamo.reset()
        gc.collect()

        try:
            fake_metrics[f"{metric_key_prefix}_perplexity"] = math.exp(
                fake_metrics[f"{metric_key_prefix}_loss"])
        except OverflowError:
            fake_metrics[f"{metric_key_prefix}_perplexity"] = float("inf")

        entry["model_training_after"] = self.model.training
        self._evaluate_log.append(entry)
        return fake_metrics


class CustomSeq2SeqTrainer_Evaluate_TEXTONLY:
    """Exact logic dari text-only CustomSeq2SeqTrainer.evaluate(), line 1473-1500"""
    def __init__(self, model):
        self.model = model
        self._evaluate_log = []
        self._for_inference_calls = 0
        self._for_training_calls = 0

    def _for_inference(self):
        """Simulate FastLanguageModel.for_inference()"""
        self._for_inference_calls += 1
        MockModel.TRAINING_KERNELS_ACTIVE = False

    def _for_training(self):
        """Simulate FastLanguageModel.for_training()"""
        self._for_training_calls += 1
        MockModel.TRAINING_KERNELS_ACTIVE = True

    def evaluate(self, metric_key_prefix="eval"):
        import math, gc
        entry = {
            "for_inference_called_before": True,
            "model_training_before": self.model.training,
        }

        self._for_inference()  # text-only calls this
        self.model.eval()

        fake_metrics = {f"{metric_key_prefix}_loss": 2.5}

        self._for_training()   # text-only calls this after
        self.model.train()

        gc.collect()
        entry["for_training_called_after"] = True
        entry["model_training_after"] = self.model.training
        self._evaluate_log.append(entry)
        return fake_metrics


class GrokAdEMAMix(torch.optim.Optimizer):
    """EXACT copy dari kode vision line 683-770"""
    def __init__(self, params, lr=3e-5, betas=(0.9, 0.999), beta3=0.9999,
                 weight_decay=0.05, grok_alpha=2.0, grok_lamb=0.98):
        defaults = dict(lr=lr, betas=betas, beta3=beta3,
                        weight_decay=weight_decay,
                        grok_alpha=grok_alpha, grok_lamb=grok_lamb)
        super().__init__(params, defaults)
        self.step_count = 0

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
        self.step_count += 1
        for group in self.param_groups:
            lr = group["lr"]
            beta1, beta2 = group["betas"]
            beta3 = group["beta3"]
            wd = group["weight_decay"]
            galpha = group["grok_alpha"]
            gklamb = group["grok_lamb"]
            for p in group["params"]:
                if p.grad is None:
                    continue
                g = p.grad
                st = self.state[p]
                if len(st) == 0:
                    st["step"] = 0
                    st["grok_slow_grad"] = torch.zeros_like(g)
                    st["m"] = torch.zeros_like(g)
                    st["v"] = torch.zeros_like(g)
                    st["n"] = torch.zeros_like(g)
                st["step"] += 1
                s = st["step"]
                st["grok_slow_grad"].mul_(gklamb).add_(g, alpha=1.0 - gklamb)
                fg = g.clone().add_(st["grok_slow_grad"], alpha=galpha)
                if wd != 0:
                    p.data.mul_(1.0 - lr * wd)
                m, v, n = st["m"], st["v"], st["n"]
                m.mul_(beta1).add_(fg, alpha=1.0 - beta1)
                v.mul_(beta2).addcmul_(fg, fg, value=1.0 - beta2)
                n.mul_(beta3).add_(fg, alpha=1.0 - beta3)
                bc1 = 1.0 - beta1**s
                bc2 = 1.0 - beta2**s
                bc3 = 1.0 - beta3**s
                denom = (v.sqrt() / (bc2**0.5)).add_(1e-8).to(p.dtype)
                upd = (m / bc1 + 0.1 * n / bc3) / denom
                p.data.add_(upd.to(p.dtype), alpha=-lr)
        return loss


# =====================================================================
# TEST 1: BOS/EOS DOUBLE ADDITION IN COLLATOR
# =====================================================================

def test_bos_eos_collator():
    header("TEST 1: BOS/EOS Double-Addition di Seq2SeqVisionCollator")

    processor = MockProcessor()

    subheader("1.1 — Verifikasi output processor (Gemma3Processor behavior)")

    text_sample = "Halo apa kabar kamu"
    enc_no_img = processor(text=text_sample, images=None)
    ids_no_img = enc_no_img["input_ids"][0].tolist()

    log_data("Processor output IDs (no image)", ids_no_img)
    log_data("IDs[0]", f"{ids_no_img[0]}  (BOS = {processor.tokenizer.bos_token_id})")
    log_data("IDs[-1]", f"{ids_no_img[-1]}  (EOS = {processor.tokenizer.eos_token_id})")

    proc_adds_bos = ids_no_img[0] == processor.tokenizer.bos_token_id
    proc_adds_eos = ids_no_img[-1] == processor.tokenizer.eos_token_id

    if proc_adds_bos:
        log_warn("Processor output SUDAH ada BOS (ini perilaku Gemma3Processor sebenarnya!)",
                 "Gemma3Processor hardcodes BOS in __call__, tidak tergantung tokenizer.add_bos_token")
    if proc_adds_eos:
        log_warn("Processor output SUDAH ada EOS")

    subheader("1.2 — Simulate exactly what collator does with that output")

    tok = processor.tokenizer
    bos = tok.bos_token_id
    eos = tok.eos_token_id

    input_ids = ids_no_img.copy()
    log_data("Before collator check", input_ids)

    # Collator BOS check (EXACT dari kode)
    bos_added = False
    if tok.bos_token_id is not None and (not input_ids or input_ids[0] != tok.bos_token_id):
        input_ids = [tok.bos_token_id] + input_ids
        bos_added = True

    # Collator EOS check (EXACT dari kode)
    eos_added = False
    if tok.eos_token_id is not None and (not input_ids or input_ids[-1] != tok.eos_token_id):
        input_ids = input_ids + [tok.eos_token_id]
        eos_added = True

    log_data("After collator check", input_ids)
    log_data("BOS added by collator", bos_added)
    log_data("EOS added by collator", eos_added)

    if proc_adds_bos and not bos_added:
        log_ok("BUG #1 (BOS): Collator tidak menambah BOS ganda karena cek input_ids[0] == bos benar")
        log_info("Kondisi check 'input_ids[0] != bos_token_id' -> False -> BOS tidak ditambah lagi")
    elif not proc_adds_bos and bos_added:
        log_warn("Collator menambah BOS karena processor tidak menambahkannya",
                 "Ini mungkin perilaku yang benar jika processor tidak menambah BOS")
    elif proc_adds_bos and bos_added:
        log_fail("BUG #1 (BOS) CONFIRMED: Collator MENAMBAH BOS padahal processor sudah menambahkan!",
                 f"Result: [{bos}, {bos}, ...] - double BOS!")

    if proc_adds_eos and not eos_added:
        log_ok("BUG #1 (EOS): Collator tidak menambah EOS ganda")
    elif proc_adds_eos and eos_added:
        log_fail("BUG #1 (EOS) CONFIRMED: Double EOS!")

    subheader("1.3 — Full collator run dengan batch (text + multimodal)")

    try:
        from PIL import Image as PILImage
        dummy_img = PILImage.new("RGB", (224, 224), color=(128, 128, 128))
        has_pil = True
    except ImportError:
        has_pil = False
        log_warn("PIL not available, skipping image test")

    collator = Seq2SeqVisionCollator_ORIGINAL(processor, max_src=256, max_tgt=64)

    batch_text = [
        {"prompt_text": "Halo apa kabar kamu", "target_text": "Halo sebuah gambar", "images": []},
        {"prompt_text": "model AI yang helpful", "target_text": "ini adalah foto", "images": []},
    ]

    print()
    log("Running collator on text-only batch...")
    try:
        result_text = collator(batch_text)
        log_data("  input_ids shape", result_text["input_ids"].shape)
        log_data("  labels shape", result_text["labels"].shape)
        log_data("  pixel_values present", "pixel_values" in result_text)

        for i, row in enumerate(result_text["input_ids"]):
            row_list = row[row != tok.pad_token_id].tolist()
            double_bos = len(row_list) >= 2 and row_list[0] == bos and row_list[1] == bos
            double_eos = row_list[-2:].count(eos) > 1 if len(row_list) >= 2 else False
            log_data(f"  Row {i} first 6 tokens", row_list[:6])
            log_data(f"  Row {i} last 3 tokens", row_list[-3:])
            if double_bos:
                log_fail(f"Row {i}: DOUBLE BOS! ids[0]={row_list[0]}, ids[1]={row_list[1]}")
            else:
                log_ok(f"Row {i}: No double BOS (first token: {row_list[0]})")
            if double_eos:
                log_fail(f"Row {i}: DOUBLE EOS!")
            else:
                log_ok(f"Row {i}: No double EOS")
    except Exception as e:
        log_fail(f"Text collator raised: {e}")
        traceback.print_exc()

    if has_pil:
        batch_mm = [
            {"prompt_text": "Halo apa gambar", "target_text": "ini adalah foto",
             "images": [dummy_img]},
        ]
        print()
        log("Running collator on multimodal batch...")
        try:
            result_mm = collator(batch_mm)
            log_data("  input_ids shape", result_mm["input_ids"].shape)
            log_data("  pixel_values present", "pixel_values" in result_mm)
            if "pixel_values" in result_mm:
                log_data("  pixel_values shape", result_mm["pixel_values"].shape)
            row_mm = result_mm["input_ids"][0][result_mm["input_ids"][0] != tok.pad_token_id].tolist()
            log_data("  First 8 tokens", row_mm[:8])
            double_bos_mm = len(row_mm) >= 2 and row_mm[0] == bos and row_mm[1] == bos
            if double_bos_mm:
                log_fail("MULTIMODAL: DOUBLE BOS detected!")
            else:
                log_ok("MULTIMODAL: No double BOS")
        except Exception as e:
            log_fail(f"Multimodal collator raised: {e}")
            traceback.print_exc()

    subheader("1.4 — CRITICAL: add_bos_token=False vs Gemma3Processor hardcoded BOS")

    log("Kode vision line 1432-1434:")
    log("   tokenizer.add_bos_token = False")
    log("   processor.tokenizer.add_bos_token = False")
    log("")
    log("TAPI: Gemma3Processor.__call__() TIDAK cek add_bos_token dari tokenizer.")
    log("      Processor menambah BOS via hardcoded logic di dalam __call__.")
    log("      Jadi BOS TETAP ada di output processor meskipun add_bos_token=False!")
    log("")

    proc2 = MockProcessor()
    proc2.tokenizer.add_bos_token = False  # Seperti di kode setelah get_chat_template

    enc2 = proc2(text="Halo apa", images=None)
    ids2 = enc2["input_ids"][0].tolist()
    ids_via_tok = proc2.tokenizer.encode("Halo apa", add_special_tokens=True)

    log_data("  processor.__call__() output", ids2)
    log_data("  processor.tokenizer.encode() output", ids_via_tok)
    log_data("  processor output has BOS", ids2[0] == proc2.tokenizer.bos_token_id)
    log_data("  tokenizer.encode output has BOS", ids_via_tok[0] == proc2.tokenizer.bos_token_id if ids_via_tok else False)

    if ids2[0] == proc2.tokenizer.bos_token_id and (
            not ids_via_tok or ids_via_tok[0] != proc2.tokenizer.bos_token_id):
        log_fail("BUG #4 (Distribution Mismatch): Training via processor() punya BOS,"
                 " Validation via tokenizer.encode() TIDAK punya BOS!",
                 "Ini karena add_bos_token=False hanya mempengaruhi tokenizer.encode(),"
                 " bukan Gemma3Processor.__call__()")
    elif ids2[0] == proc2.tokenizer.bos_token_id and (
            ids_via_tok and ids_via_tok[0] == proc2.tokenizer.bos_token_id):
        log_ok("Konsisten: keduanya punya BOS")
    else:
        log_warn("Unexpected BOS behavior")


# =====================================================================
# TEST 2: PIXEL_VALUES ROUTING IN VISIONORPOTRAINER
# =====================================================================

def test_pixel_values_routing():
    header("TEST 2: Pixel Values Routing di VisionORPOTrainer.compute_loss")

    model = MockModel(vocab_size=262144, hidden_size=32)
    model.eval()

    orpo_trainer = VisionORPOTrainer_ComputeLoss_ORIGINAL(beta=0.1)

    inputs_mm = {
        "input_ids": torch.randint(3, 1000, (1, 8)),
        "attention_mask": torch.ones(1, 8, dtype=torch.long),
        "pixel_values": torch.randn(1, 3, 224, 224),
        "chosen_labels": torch.randint(3, 1000, (1, 4)),
        "rejected_labels": torch.randint(3, 1000, (1, 4)),
    }

    subheader("2.1 — Apakah pixel_values sampai ke encoder?")
    model.encoder._call_count = 0
    model.encoder._pixel_values_received = False
    model._forward_calls = []

    try:
        loss = orpo_trainer.compute_loss_original(model, inputs_mm)
        log_data("  Loss value", f"{loss.item():.4f}")
        log_data("  Encoder call count", model.encoder._call_count)
        log_data("  Encoder received pixel_values", model.encoder._pixel_values_received)
        log_data("  Encoder pixel_values shape", model.encoder._pixel_values_shape)
        log_data("  Model forward calls total", len(model._forward_calls))

        if model.encoder._pixel_values_received:
            log_ok("BUG #2: pixel_values MASUK ke encoder (via direct encoder call)")
            log_info("Encoder menerima pixel_values di forward() — OK untuk mock T5Gemma2Encoder")
            log_warn("Tapi di PEFT-wrapped model: PEFT LoRA wrapper mungkin tidak forward pixel_values ke base encoder",
                     "PEFT hanya forward arg yang ada di LoRA module signature, bukan arbitrary kwargs")
        else:
            log_fail("BUG #2 CONFIRMED: pixel_values TIDAK diterima oleh encoder!")

        log("")
        for i, call in enumerate(model._forward_calls):
            log(f"  model.forward() call #{i+1}:")
            log_data(f"    encoder_outputs_received", call["encoder_outputs_received"])
            log_data(f"    pixel_values_received", call["pixel_values_received"])

    except Exception as e:
        log_fail(f"compute_loss_original raised: {e}")
        traceback.print_exc()

    subheader("2.2 — Bandingkan: SFT path (model(**inputs)) vs ORPO path (encoder() then model())")

    model2 = MockModel(vocab_size=262144, hidden_size=32)
    model2.eval()
    model2.encoder._call_count = 0
    model2.encoder._pixel_values_received = False
    model2._forward_calls = []

    sft_inputs = {
        "input_ids": torch.randint(3, 1000, (1, 8)),
        "attention_mask": torch.ones(1, 8, dtype=torch.long),
        "pixel_values": torch.randn(1, 3, 224, 224),
        "labels": torch.randint(3, 1000, (1, 4)),
    }

    # SFT path: model(**inputs) langsung
    with torch.no_grad():
        out_sft = model2(**sft_inputs)

    log_data("  SFT encoder received pixel_values", model2.encoder._pixel_values_received)
    log_data("  SFT model forward calls", len(model2._forward_calls))

    if model2.encoder._pixel_values_received:
        log_ok("SFT path: pixel_values masuk ke encoder via model(**inputs)")
    else:
        log_fail("SFT path: pixel_values tidak masuk ke encoder")

    log("")
    log_data("ORPO encoder received pixel_values", model.encoder._pixel_values_received)
    log_data("SFT encoder received pixel_values", model2.encoder._pixel_values_received)

    if model.encoder._pixel_values_received == model2.encoder._pixel_values_received:
        log_ok("Kedua path menghasilkan perilaku pixel_values yang sama (di mock ini)")
        log_warn("Di model real dengan PEFT: ORPO path lebih berisiko karena encoder dipanggil langsung",
                 "PEFT LoRA wrapper tidak meneruskan pixel_values di encoder-only call")
    else:
        log_fail("INCONSISTENCY antara SFT dan ORPO dalam handling pixel_values!")

    subheader("2.3 — Analisa PEFT wrapper impact")

    log("Ketika model di-wrap PEFT:")
    log("   model = FastVisionModel.get_peft_model(model, ...)")
    log("   → model menjadi PeftModel")
    log("   → model.base_model.model = original T5Gemma2Model")
    log("")
    log("VisionORPOTrainer.compute_loss (line 627-633):")
    log("   base_model = model.base_model.model  (unwrap PEFT)")
    log("   encoder = base_model.get_encoder()")
    log("")
    log("   Ini mestinya benar — encoder adalah T5Gemma2Encoder yang asli.")
    log("   Tapi pixel_values harus di-forward via T5Gemma2Encoder.forward(pixel_values=...)")
    log("   Dan T5Gemma2Encoder harus ada di LoRA target modules atau di modules_to_save")
    log("   supaya gradient tetap mengalir dengan benar.")
    log("")
    log("   ISSUE: finetune_vision_layers=False → vision tower di-freeze")
    log("   Tapi pixel_values masuk ke vision tower (SigLIP) di dalam encoder!")
    log("   → Vision tower tidak belajar → representasi visual tidak optimal")
    log_warn("Vision tower frozen (finetune_vision_layers=False) menyebabkan SigLIP tidak belajar",
             "Meskipun datanya ada, model tidak bisa update representasi visual")


# =====================================================================
# TEST 3: FOR_INFERENCE/FOR_TRAINING STATE MACHINE
# =====================================================================

def test_inference_training_state():
    header("TEST 3: for_inference/for_training State Machine")

    model_vision = MockModel(vocab_size=262144, hidden_size=32)
    model_textonly = MockModel(vocab_size=262144, hidden_size=32)
    MockModel.TRAINING_KERNELS_ACTIVE = True

    subheader("3.1 — Vision CustomSeq2SeqTrainer.evaluate() (ORIGINAL code)")

    evaluator_vision = CustomSeq2SeqTrainer_Evaluate_VISION(model_vision)
    model_vision.train()
    MockModel.TRAINING_KERNELS_ACTIVE = True

    log_data("State SEBELUM evaluate (vision)", f"model.training={model_vision.training}")

    metrics_v = evaluator_vision.evaluate(metric_key_prefix="eval")

    log_data("State SETELAH evaluate (vision)", f"model.training={model_vision.training}")
    log_data("for_inference called", evaluator_vision._evaluate_log[-1]["for_inference_called_before"])
    log_data("for_training called", evaluator_vision._evaluate_log[-1]["for_training_called_after"])
    log_data("model.training after", evaluator_vision._evaluate_log[-1]["model_training_after"])

    if not model_vision.training:
        log_fail("BUG #3 CONFIRMED: Model STUCK IN EVAL MODE setelah evaluate()!",
                 "Vision CustomSeq2SeqTrainer tidak memanggil model.train() atau for_training() "
                 "setelah evaluate selesai. Model tetap dalam eval state!")
        log_warn("Di Unsloth: tanpa for_training() → training kernels TIDAK diaktifkan kembali",
                 "Konsekuensi: gradient update di step berikutnya rusak / tidak optimal")
    else:
        log_ok("Model kembali ke training mode")

    log_data("Unsloth training kernels active", MockModel.TRAINING_KERNELS_ACTIVE)

    subheader("3.2 — Text-Only CustomSeq2SeqTrainer.evaluate() (CORRECT reference)")

    evaluator_textonly = CustomSeq2SeqTrainer_Evaluate_TEXTONLY(model_textonly)
    model_textonly.train()
    MockModel.TRAINING_KERNELS_ACTIVE = True

    log_data("State SEBELUM evaluate (text-only)", f"model.training={model_textonly.training}")

    metrics_t = evaluator_textonly.evaluate(metric_key_prefix="eval")

    log_data("State SETELAH evaluate (text-only)", f"model.training={model_textonly.training}")
    log_data("for_inference called", len(evaluator_textonly._for_inference_calls) > 0
             if isinstance(evaluator_textonly._for_inference_calls, list)
             else evaluator_textonly._for_inference_calls > 0)
    log_data("for_training called", evaluator_textonly._for_training_calls > 0)
    log_data("model.training after", model_textonly.training)

    if model_textonly.training and MockModel.TRAINING_KERNELS_ACTIVE:
        log_ok("Text-only: model kembali ke training mode dengan Unsloth kernels aktif")
    else:
        log_fail("Text-only evaluate juga ada masalah")

    subheader("3.3 — Training loop simulation: eval diselingi antara training steps")

    log("Simulasi 3 training steps dengan eval di tengah (vision behavior):")

    param = nn.Parameter(torch.randn(5, 5))
    optim = torch.optim.SGD([param], lr=0.1)
    grad_log = []

    for step in range(3):
        # Training step
        optim.zero_grad()
        loss = (param * 2).sum()
        loss.backward()
        grad_norm_before = param.grad.norm().item() if param.grad is not None else 0.0
        optim.step()

        # Simulate evaluate after step 1
        if step == 1:
            log(f"\n  [Step {step}] Evaluate dipanggil (vision bug: tanpa for_training() after):")
            # Vision bug: hanya eval(), tidak ada train() setelah itu
            param_as_model = type("M", (), {
                "training": True, "eval": lambda self: setattr(self, "training", False),
                "_training_mode_log": []
            })()
            param_as_model.eval()
            log_data(f"  After evaluate, model.training", param_as_model.training)
            log_data(f"  Training would continue with model.training=False", True)

        grad_log.append(f"Step {step}: grad_norm={grad_norm_before:.4f}")

    for g in grad_log:
        log(f"  {g}")

    log("")
    log(f"  {YELLOW}IMPORTANT:{RESET} Di Unsloth, model.eval() + tidak ada for_training():")
    log("  1. Gradient checkpointing dari Unsloth TIDAK aktif kembali")
    log("  2. Custom triton attention kernels tetap dalam eval/inference mode")
    log("  3. Untuk vision model: SigLIP vision tower kernels stuck in eval")
    log("  4. Efek: training TIDAK crash, tapi gradient tidak dihitung dengan benar")
    log("     → Loss bisa terus turun tapi kualitas output TIDAK membaik")

    log_warn("Unsloth for_training() diperlukan setelah setiap evaluate() untuk vision model",
             "Tanpa ini, training lanjut tapi Unsloth kernels tidak aktif → silent degradation")


# =====================================================================
# TEST 4: TOKENIZER CONSISTENCY
# =====================================================================

def test_tokenizer_consistency():
    header("TEST 4: Tokenizer Consistency (Training vs Validation Pipeline)")

    subheader("4.1 — Training input path vs Validation input path")

    log("Training (via Seq2SeqVisionCollator):")
    log("   enc = processor(text=prompt_text, images=..., return_tensors='pt')")
    log("   → Gemma3Processor.__call__() yang menambah BOS+EOS")
    log("")
    log("Validation TEXT-ONLY (via run_eval(), line 2513-2518):")
    log("   inp_ids = tokenizer.encode(inp_f, add_special_tokens=True)")
    log("   → tokenizer.encode() dengan add_bos_token=False")
    log("")

    proc = MockProcessor()
    proc.tokenizer.add_bos_token = False  # Sesuai line 1432

    text = "Halo apa kabar"

    # Training path
    enc_train = proc(text=text, images=None)
    ids_train = enc_train["input_ids"][0].tolist()

    # Validation path (process_sft_rows menggunakan tokenizer.encode)
    ids_val = proc.tokenizer.encode(text, add_special_tokens=True)

    log_data("  Training path (processor.__call__)", ids_train[:6])
    log_data("  Validation path (tokenizer.encode)", ids_val[:6])
    log_data("  Training has BOS at start", ids_train[0] == proc.tokenizer.bos_token_id)
    log_data("  Validation has BOS at start", bool(ids_val) and ids_val[0] == proc.tokenizer.bos_token_id)

    if ids_train[0] == proc.tokenizer.bos_token_id and (
            not ids_val or ids_val[0] != proc.tokenizer.bos_token_id):
        log_fail("BUG #4 CONFIRMED: Training input punya BOS, Validation input TIDAK punya BOS!",
                 "Training distribution != Validation distribution")
        log_info("Ini menyebabkan validation metrics underestimate kemampuan model sebenarnya")
        log_info("LEBIH PARAH: model belajar expect BOS di input, tapi eval test tanpa BOS")
    elif ids_train[0] == proc.tokenizer.bos_token_id and (
            ids_val and ids_val[0] == proc.tokenizer.bos_token_id):
        log_ok("Konsisten: keduanya punya BOS (add_bos_token mungkin tidak berpengaruh di encode juga)")
    else:
        log_warn("BOS behavior tidak konsisten atau unexpected")

    subheader("4.2 — Seq length mismatch karena BOS")

    train_len = len(ids_train)
    val_len = len(ids_val)
    log_data("  Training seq len", train_len)
    log_data("  Validation seq len", val_len)

    if train_len != val_len:
        diff = train_len - val_len
        log_warn(f"Seq length mismatch: training={train_len}, val={val_len} (diff={diff:+d})",
                 "Karena BOS ada di training tapi tidak di validation")
    else:
        log_ok("Seq length sama")

    subheader("4.3 — Label encoding consistency")

    log("Di collator (line 535-537):")
    log("   target_formatted = item['target_text'].strip() + '<end_of_turn>'")
    log("   tids = self.tok.encode(target_formatted, add_special_tokens=False)")
    log("   → add_special_tokens=False, jadi add_bos_token tidak berpengaruh di sini")
    log("")
    log("Di process_sft_rows (line 920):")
    log("   tgt_ids = tokenizer.encode(tgt_f, add_special_tokens=False)")
    log("   → Konsisten: add_special_tokens=False")
    log("")

    target = "Halo sebuah gambar yang helpful"
    tgt_formatted = target + "<end_of_turn>"

    tids_collator = proc.tokenizer.encode(tgt_formatted, add_special_tokens=False)
    tids_eval = proc.tokenizer.encode(tgt_formatted, add_special_tokens=False)

    if tids_collator == tids_eval:
        log_ok("Label encoding konsisten antara training dan validation")
    else:
        log_fail("Label encoding tidak konsisten!")


# =====================================================================
# TEST 5: RSLORA SCALING
# =====================================================================

def test_rslora_scaling():
    header("TEST 5: RSLoRA Scaling & Hyperparameter Analysis")

    LORA_RANK = 256
    LORA_ALPHA = 512

    subheader("5.1 — Scaling factor")

    std_scaling = LORA_ALPHA / LORA_RANK
    rslora_scaling = LORA_ALPHA / math.sqrt(LORA_RANK)

    log_data("  r", LORA_RANK)
    log_data("  alpha", LORA_ALPHA)
    log_data("  Standard LoRA scaling (alpha/r)", f"{std_scaling:.3f}")
    log_data("  RSLoRA scaling (alpha/sqrt(r))", f"{rslora_scaling:.3f}")
    log_data("  RSLoRA vs Standard ratio", f"{rslora_scaling / std_scaling:.1f}x")

    if rslora_scaling > 10:
        log_warn(f"RSLoRA scaling BESAR: {rslora_scaling:.1f}",
                 f"Dengan r={LORA_RANK}, alpha={LORA_ALPHA}: scaling = {rslora_scaling:.1f} "
                 f"(standard = {std_scaling:.1f})")
    else:
        log_ok(f"RSLoRA scaling dalam batas: {rslora_scaling:.2f}")

    subheader("5.2 — Effective learning rate comparison")

    LR_vision = 2e-5
    LR_text = 1e-5
    ACCUM_vision = 16
    ACCUM_text = 64

    eff_lr_vision = LR_vision / ACCUM_vision
    eff_lr_text = LR_text / ACCUM_text
    ratio = eff_lr_vision / eff_lr_text

    log_data("  Vision: LR", LR_vision)
    log_data("  Vision: grad_accum", ACCUM_vision)
    log_data("  Vision: effective LR", f"{eff_lr_vision:.2e}")
    log_data("  Text-only: LR", LR_text)
    log_data("  Text-only: grad_accum", ACCUM_text)
    log_data("  Text-only: effective LR", f"{eff_lr_text:.2e}")
    log_data("  Ratio effective LR (vision/text)", f"{ratio:.1f}x")

    if ratio > 5:
        log_fail(f"Vision training {ratio:.0f}x lebih agresif dari text-only!",
                 f"eff_LR_vision={eff_lr_vision:.2e} vs eff_LR_text={eff_lr_text:.2e}")
    elif ratio > 2:
        log_warn(f"Vision training {ratio:.1f}x lebih agresif dari text-only",
                 "Ini bisa menyebabkan instabilitas dan forgetting yang lebih cepat")
    else:
        log_ok(f"Effective LR ratio acceptable: {ratio:.1f}x")

    subheader("5.3 — GrokAdEMAMix dengan split LR: apakah language decoder terlindungi?")

    log("Split LR di vision code:")
    log(f"   encoder_params   : LR = {LR_vision * 0.5:.2e} (0.5x)")
    log(f"   decoder_params   : LR = {LR_vision:.2e} (1.0x)  ← PENUH!")
    log(f"   projector_params : LR = {LR_vision:.2e} (1.0x)")
    log(f"   vision_tower     : LR = {LR_vision * 0.5:.2e} (0.5x)")
    log("")
    log("  Decoder (language model) mendapat LR PENUH 2e-5.")
    log("  Ini 8x lebih besar dari text-only training (1e-5 / 64 per step).")
    log("  → Decoder langsung mendapat update besar dari multimodal signal")
    log("  → Jika projector outputnya noise (karena cangkok alignment belum sempurna),")
    log("    decoder belajar mengakomodasi noise tersebut → language quality turun!")

    if LR_vision >= LR_text:
        log_warn("Decoder LR di vision training sama atau lebih besar dari text-only",
                 "Kombinasi dengan gradient yang mungkin noisy dari projector = language degradation")


# =====================================================================
# TEST 6: MINI FORWARD PASS
# =====================================================================

def test_mini_forward_pass():
    header("TEST 6: Mini End-to-End Forward Pass")

    processor = MockProcessor()
    model = MockModel(vocab_size=262144, hidden_size=32)
    model.train()

    subheader("6.1 — SFT batch forward pass")

    try:
        from PIL import Image as PILImg
        dummy_img = PILImg.new("RGB", (224, 224), color=(100, 150, 200))
        has_pil = True
    except ImportError:
        has_pil = False

    sft_batch = [
        {
            "prompt_text": "Halo apa kabar kamu model",
            "target_text": "Halo sebuah gambar yang helpful",
            "images": [dummy_img] if has_pil else [],
        },
        {
            "prompt_text": "AI yang helpful user",
            "target_text": "ini adalah foto kamu",
            "images": [],
        },
    ]

    collator = Seq2SeqVisionCollator_ORIGINAL(processor, max_src=128, max_tgt=32)

    try:
        batch = collator(sft_batch)
        log_data("  input_ids shape", batch["input_ids"].shape)
        log_data("  labels shape", batch["labels"].shape)
        log_data("  pixel_values present", "pixel_values" in batch)

        iids = batch["input_ids"]
        bos = processor.tokenizer.bos_token_id
        eos = processor.tokenizer.eos_token_id

        log("\n  === Token Analysis ===")
        for i in range(len(iids)):
            row = iids[i][iids[i] != processor.tokenizer.pad_token_id].tolist()
            double_bos = len(row) >= 2 and row[0] == bos and row[1] == bos
            log(f"  Row {i}: {row[:6]} ... {row[-2:]}")
            log_data(f"    len={len(row)}, double_bos={double_bos}",
                     "BUG!" if double_bos else "OK")

        model._forward_calls = []
        model.encoder._pixel_values_received = False

        with torch.enable_grad():
            out = model(
                input_ids=batch["input_ids"],
                attention_mask=batch["attention_mask"],
                labels=batch["labels"],
                pixel_values=batch.get("pixel_values"),
            )

        log("\n  === Forward Pass ===")
        log_data("  Loss", f"{out.loss.item():.4f}")
        log_data("  Loss is NaN", torch.isnan(out.loss).item())
        log_data("  Encoder received pixel_values", model.encoder._pixel_values_received)

        out.loss.backward()

        grad_norms = {n: p.grad.norm().item()
                      for n, p in model.named_parameters()
                      if p.grad is not None}

        log("\n  === Gradient Norms ===")
        for name, gnorm in list(grad_norms.items())[:5]:
            log_data(f"  {name}", f"{gnorm:.4f}")

        if grad_norms:
            max_g = max(grad_norms.values())
            if max_g > 1000:
                log_fail(f"Gradient EXPLOSION! max_grad_norm={max_g:.2f}")
            elif max_g < 1e-10:
                log_fail(f"Vanishing gradient! max_grad_norm={max_g:.2e}")
            else:
                log_ok(f"Gradients normal: max={max_g:.4f}")
        else:
            log_fail("No gradients found!")

    except Exception as e:
        log_fail(f"SFT forward pass failed: {e}")
        traceback.print_exc()

    subheader("6.2 — ORPO batch forward pass")

    model2 = MockModel(vocab_size=262144, hidden_size=32)
    model2.eval()
    orpo_trainer = VisionORPOTrainer_ComputeLoss_ORIGINAL(beta=0.1)

    orpo_inputs = {
        "input_ids": torch.randint(3, 1000, (1, 8)),
        "attention_mask": torch.ones(1, 8, dtype=torch.long),
        "pixel_values": torch.randn(1, 3, 224, 224),
        "chosen_labels": torch.randint(3, 1000, (1, 4)),
        "rejected_labels": torch.randint(3, 1000, (1, 4)),
    }

    try:
        loss_orpo = orpo_trainer.compute_loss_original(model2, orpo_inputs)
        log_data("  ORPO Loss", f"{loss_orpo.item():.4f}")
        log_data("  Encoder pixel_values", model2.encoder._pixel_values_received)
        log_data("  Model forward calls", len(model2._forward_calls))

        for i, call in enumerate(model2._forward_calls):
            log(f"  Call {i+1}: encoder_outputs={call['encoder_outputs_received']}, "
                f"pv={call['pixel_values_received']}")

        loss_orpo.backward()
        log_ok("ORPO backward pass OK")

    except Exception as e:
        log_fail(f"ORPO forward pass failed: {e}")
        traceback.print_exc()


# =====================================================================
# TEST 7: SELECTIVE LABEL SMOOTHER
# =====================================================================

def test_label_smoothing():
    header("TEST 7: SelectiveLabelSmoother Behavior")

    SUPPRESS_BLOCK1 = [6] + list(range(13, 105))
    SUPPRESS_VISION = [255999, 256000, 256001]
    ALL_SUPPRESS_IDS = set(SUPPRESS_BLOCK1 + SUPPRESS_VISION)

    class SelectiveLabelSmoother:
        def __init__(self, epsilon, suppress_ids):
            self.epsilon = epsilon
            self.suppress_ids = suppress_ids

        def __call__(self, model_output, labels, shift_labels=False):
            logits = model_output.logits if hasattr(model_output, "logits") else model_output["logits"]
            if shift_labels:
                logits = logits[..., :-1, :].contiguous()
                labels = labels[..., 1:].contiguous()

            vocab_size = logits.size(-1)
            suppress_list = [i for i in self.suppress_ids if i < vocab_size]
            valid_mask = torch.ones(vocab_size, dtype=torch.bool)
            valid_mask[suppress_list] = False
            num_valid = valid_mask.sum().item()

            flat_logits = logits.view(-1, vocab_size)
            flat_labels = labels.view(-1)
            active_mask = flat_labels != -100

            if active_mask.sum() == 0:
                return torch.tensor(0.0, requires_grad=True)

            act_l = flat_logits[active_mask]
            act_lb = flat_labels[active_mask]
            total_loss = torch.tensor(0.0)
            for i in range(0, len(act_l), 512):
                cl = act_l[i:i + 512]
                clab = act_lb[i:i + 512]
                lp = F.log_softmax(cl, dim=-1)
                nll = -lp.gather(-1, clab.unsqueeze(-1)).squeeze(-1)
                smooth = -(lp * valid_mask.to(lp.dtype)).sum(-1) / num_valid
                total_loss += ((1 - self.epsilon) * nll + self.epsilon * smooth).sum()
            return total_loss / active_mask.sum()

    smoother = SelectiveLabelSmoother(epsilon=0.1, suppress_ids=ALL_SUPPRESS_IDS)
    vocab_size = 500

    subheader("7.1 — Basic loss calculation")
    logits = torch.randn(2, 8, vocab_size)
    labels = torch.randint(3, 400, (2, 8))
    labels[0, -2:] = -100

    out = type("O", (), {"logits": logits})()
    try:
        loss = smoother(out, labels)
        log_data("  Loss", f"{loss.item():.4f}")
        if not torch.isnan(loss) and not torch.isinf(loss):
            log_ok("SelectiveLabelSmoother normal")
        else:
            log_fail("Loss NaN/Inf!")
    except Exception as e:
        log_fail(f"SelectiveLabelSmoother error: {e}")
        traceback.print_exc()

    subheader("7.2 — Task prefix tokens (ID 7-12) NOT suppressed?")
    task_prefix_ids = list(range(7, 13))
    not_suppressed = [i for i in task_prefix_ids if i not in ALL_SUPPRESS_IDS]
    suppressed = [i for i in task_prefix_ids if i in ALL_SUPPRESS_IDS]
    log_data("  Task prefix IDs (7-12)", task_prefix_ids)
    log_data("  NOT suppressed (correct)", not_suppressed)
    log_data("  SUPPRESSED (should be empty)", suppressed)
    if not suppressed:
        log_ok("Task prefix tokens (unused1-6) tidak di-suppress — benar!")
    else:
        log_fail(f"Task prefix tokens tersuppress: {suppressed}")

    subheader("7.3 — Vision tokens suppressed?")
    for vid in SUPPRESS_VISION:
        in_suppress = vid in ALL_SUPPRESS_IDS
        if in_suppress:
            log_ok(f"Vision token ID={vid} di-suppress (benar — tidak boleh jadi output)")
        else:
            log_fail(f"Vision token ID={vid} TIDAK di-suppress!")


# =====================================================================
# MAIN
# =====================================================================

def main():
    print(f"\n{BOLD}{CYAN}{'='*70}")
    print("  VISION TRAINING MECHANISM DIAGNOSTIC")
    print("  working-molab-v6-vision-unsloth.py")
    print(f"{'='*70}{RESET}")
    print(f"  Python  : {sys.version.split()[0]}")
    print(f"  PyTorch : {torch.__version__}")
    print(f"  Device  : CPU only (diagnostic/mock mode)")
    print(f"  NOTE    : Uses mock model mirroring T5Gemma2 interface")
    print()

    tests = [
        ("BOS/EOS Collator", test_bos_eos_collator),
        ("Pixel Values Routing", test_pixel_values_routing),
        ("for_inference/for_training State", test_inference_training_state),
        ("Tokenizer Consistency", test_tokenizer_consistency),
        ("RSLoRA Scaling", test_rslora_scaling),
        ("Mini Forward Pass", test_mini_forward_pass),
        ("Label Smoothing", test_label_smoothing),
    ]

    failed_tests = []
    for name, fn in tests:
        try:
            fn()
        except Exception as e:
            log_fail(f"Test '{name}' CRASHED: {e}")
            traceback.print_exc()
            failed_tests.append(name)

    summary()

    if failed_tests:
        print(f"  {RED}Tests yang crash (bukan FAIL, tapi exception):{RESET}")
        for t in failed_tests:
            print(f"    - {t}")

    return _fail_count


if __name__ == "__main__":
    result = main()
    sys.exit(0 if result == 0 else 1)
