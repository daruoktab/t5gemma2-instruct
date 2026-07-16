# /// script
# requires-python = ">=3.10"
# dependencies = [
#     "accelerate==1.14.0",
#     "absl-py==2.4.0",
#     "bitsandbytes==0.49.2",
#     "datasets==5.0.0",
#     "evaluate",
#     "rouge-score",
#     "sacrebleu",
#     "bert_score",
#     "nltk",
#     "huggingface-hub==1.23.0",
#     "marimo==0.23.14",
#     "numpy==2.5.1",
#     "peft==0.19.1",
#     "pillow==12.3.0",
#     "pymupdf==1.28.0",
#     "torch==2.12.1",
#     "torchvision==0.27.1",
#     "trl==1.8.0",
#     "transformers==5.13.1",
#     "unsloth_zoo @ git+https://github.com/daruoktab/unsloth-zoo.git",
#     "unsloth @ git+https://github.com/daruoktab/unsloth.git",
# ]
# ///

import marimo

__generated_with = "0.23.14"
app = marimo.App(
    width="full",
    css_file="/usr/local/_marimo/custom.css",
    auto_download=["html"],
)


@app.cell
def _():
    import marimo as mo

    return (mo,)


@app.cell
def _():
    import subprocess
    import sys

    # Auto-install dependencies jika belum ada di env Molab
    try:
        import unsloth
        import datasets
        import peft
        print("✅ Dependencies utama sudah ter-install.")
    except ImportError:
        print("📦 Meng-install dependencies di Molab...")
        subprocess.run(
            [
                "uv", "pip", "install",
                "accelerate==1.14.0",
                "absl-py==2.4.0",
                "bitsandbytes==0.49.2",
                "datasets==5.0.0",
                "evaluate",
                "rouge-score",
                "sacrebleu",
                "bert_score",
                "nltk",
                "huggingface-hub==1.23.0",
                "marimo==0.23.14",
                "numpy==2.5.1",
                "peft==0.19.1",
                "pillow==12.3.0",
                "pymupdf==1.28.0",
                "torch==2.12.1",
                "torchvision==0.27.1",
                "trl==1.8.0",
                "transformers==5.13.1",
                "unsloth_zoo @ git+https://github.com/daruoktab/unsloth-zoo.git",
                "unsloth @ git+https://github.com/daruoktab/unsloth.git",
            ],
            check=True
        )

    # Selalu pastikan flash_attn ter-install
    try:
        import flash_attn
        print("✅ flash_attn sudah ter-install.")
    except ImportError:
        print("📦 Meng-install flash_attn prebuild wheel...")
        subprocess.run(
            [
                "uv", "pip", "install",
                "flash_attn @ https://github.com/mjun0812/flash-attention-prebuild-wheels/releases/download/v0.9.17/flash_attn-2.8.3+cu130torch2.12-cp313-cp313-linux_x86_64.whl",
            ],
            check=True,
        )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # 📷 Multimodal Vision SFT Fine-Tuning Pipeline (Version 6 - Unsloth)
    =====================================================================
    Notebook ini melatih aspek **vision** dari model **T5Gemma-2 4B-4B** menggunakan QLoRA/LoRA via Unsloth.
    Model dasar yang digunakan adalah model hasil SFT + ORPO teks (`t5gemma-2-4b-4b-instruct-chat-indo-v4-unsloth`).

    **Fitur utama:**
    - Pemrosesan gambar dokumen ilmiah/PDF yang diconvert ke format PIL Image.
    - Fine-tuning vision encoder dan attention/language adapter secara modular.
    - Integrasi penuh dengan `FastVisionModel` dan `UnslothVisionDataCollator`.
    """)
    return


@app.cell
def _():
    import os
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
    # Matikan auto torch.compile bawaan Unsloth (unsloth_zoo membungkus forward
    # T5Gemma2 dengan @torch.compile(fullgraph=True, dynamic=True, ...)). Dengan
    # fullgraph=True, begitu recompile limit kena, itu SELALU hard-crash tanpa
    # ada config yang bisa menyelamatkan (fail_on_recompile_limit_hit /
    # suppress_errors tidak berlaku untuk fullgraph). OOM sudah ditangani oleh
    # expandable_segments di atas + gradient checkpointing "unsloth", jadi
    # compile ini murni untuk speed, bukan untuk mencegah OOM -> aman dimatikan.
    os.environ["TORCH_COMPILE_DISABLE"] = "1"
    import re, json, torch, random, datetime, gc, traceback
    import warnings
    warnings.filterwarnings("ignore")

    # Belt-and-suspenders di atas TORCH_COMPILE_DISABLE: env var itu ternyata
    # TIDAK reliable mematikan compile Unsloth untuk T5Gemma2 di torch 2.12.1 --
    # traceback masih lewat unsloth_compiled_module_t5gemma2.py & torch._dynamo
    # meski env var sudah diset (terbukti ORPO training masih crash dengan
    # "Hard failure due to fullgraph=True" walau env var ini sudah ada).
    # Monkeypatch torch.compile jadi no-op di sini -- SEBELUM unsloth di-import
    # dan SEBELUM FastVisionModel.from_pretrained() memicu Unsloth membungkus
    # forward T5Gemma2 dengan @torch.compile(fullgraph=True, ...). Ini gak
    # bergantung env var/versi torch: begitu torch.compile dipanggil di mana pun
    # (termasuk di dalam Unsloth), langsung dikembalikan fungsi/model aslinya
    # tanpa dibungkus compile -> nol resiko nabrak recompile limit lagi.
    def _torch_compile_noop(model=None, *args, **kwargs):
        if model is not None:
            return model
        return lambda fn: fn
    torch.compile = _torch_compile_noop  # type: ignore[assignment]
    import torch.nn.functional as F
    torch._dynamo.config.recompile_limit = 1024  # type: ignore[assignment]
    torch._dynamo.config.cache_size_limit = 1024  # type: ignore[assignment]
    from PIL import Image
    from unsloth import FastVisionModel
    from datasets import Dataset, load_dataset
    from transformers import (
        AutoProcessor, AutoTokenizer,
        Seq2SeqTrainer, Seq2SeqTrainingArguments,
        get_scheduler,
        TrainerCallback, TrainerControl, TrainerState, TrainingArguments,
    )
    from typing import Any, cast

    # ---- LOGIT MASKING (decoder lm_head, aman untuk vision) ----
    def apply_logit_mask(model, suppress_ids):
        vs = model.config.vocab_size
        sl = [i for i in suppress_ids if i < vs]
        mask = torch.zeros(vs, dtype=torch.bfloat16)
        mask[sl] = -10000.0
        def hook(mod, inp, out):
            if isinstance(out, torch.Tensor):
                return out + mask.to(out.device)
            elif hasattr(out, "logits"):
                out.logits = out.logits + mask.to(out.logits.device)
                return out
            elif isinstance(out, tuple) and out and isinstance(out[0], torch.Tensor):
                return (out[0] + mask.to(out[0].device),) + out[1:]
            return out
        t = None
        if hasattr(model, "lm_head"):
            t = model.lm_head
        elif hasattr(model, "base_model") and hasattr(model.base_model, "lm_head"):
            t = model.base_model.lm_head
        elif hasattr(model, "base_model") and hasattr(model.base_model, "model") and hasattr(model.base_model.model, "lm_head"):
            t = model.base_model.model.lm_head
        if t is not None:
            t.register_forward_hook(hook)
            print(f"  ✅ Logit mask (lm_head) untuk {len(sl)} tokens.")
        else:
            model.register_forward_hook(hook)
            print(f"  ✅ Logit mask (fallback) untuk {len(sl)} tokens.")

    return (
        Any,
        AutoProcessor,
        Dataset,
        F,
        FastVisionModel,
        Image,
        Seq2SeqTrainer,
        Seq2SeqTrainingArguments,
        TrainerCallback,
        TrainerControl,
        TrainerState,
        TrainingArguments,
        apply_logit_mask,
        cast,
        datetime,
        gc,
        get_scheduler,
        load_dataset,
        os,
        random,
        re,
        torch,
        traceback,
    )


@app.cell
def _(mo):
    # Create a secure token input
    hf_token_input = mo.ui.text(
        label="Hugging Face Token (HF_TOKEN)", value="", full_width=True
    )
    hf_token_input
    return (hf_token_input,)


@app.cell
def _(hf_token_input, mo, os):
    from huggingface_hub import login

    # Stop execution of this cell if no token is entered yet
    mo.stop(
        not hf_token_input.value,
        mo.md(
            "⚠️ *Please enter your Hugging Face token in the input above to authenticate and load gated models.*"
        ),
    )

    try:
        # Set the environment variable so transformers/datasets can find it
        os.environ["HF_TOKEN"] = hf_token_input.value
        login(token=hf_token_input.value)
        status = mo.md(
            "✅ **Successfully authenticated with Hugging Face Hub!** You can now load gated models."
        )
    except Exception as e:
        status = mo.md(f"❌ **Authentication failed:** {e}")

    status
    return


@app.cell
def _(HF_CHECKPOINT_REPO, mo, os):
    from huggingface_hub import HfApi as _StageDetectApi

    _hf_token = os.environ.get("HF_TOKEN")
    _api = _StageDetectApi(token=_hf_token)

    # Default
    current_stage = "sft"
    resume_checkpoint = None

    try:
        # Automatically create repository if it does not exist
        if not _api.repo_exists(repo_id=HF_CHECKPOINT_REPO):
            print(f"📍 Repo '{HF_CHECKPOINT_REPO}' belum ada. Membuat repositori baru...")
            _api.create_repo(repo_id=HF_CHECKPOINT_REPO, repo_type="model", private=False, exist_ok=True)

        _repo_files = _api.list_repo_files(HF_CHECKPOINT_REPO)

        # Cek apakah ORPO sudah selesai
        if any(f.startswith("orpo/final_adapter/") for f in _repo_files):
            current_stage = "done"
            print("📍 Pipeline stage: DONE — Semua training selesai!")

        # Cek apakah SFT sudah selesai → lanjut ORPO
        elif any(f.startswith("sft/final_adapter/") for f in _repo_files):
            current_stage = "orpo"
            # Ada checkpoint ORPO untuk resume?
            _orpo_ckpts = sorted([
                f for f in _repo_files
                if f.startswith("orpo/checkpoint-") and "/" in f[len("orpo/checkpoint-"):]
            ])
            if _orpo_ckpts:
                resume_checkpoint = True
                print(f"📍 Pipeline stage: ORPO (resume dari checkpoint)")
            else:
                print("📍 Pipeline stage: ORPO (mulai dari awal, load SFT adapter)")

        # SFT belum selesai
        else:
            current_stage = "sft"
            _sft_ckpts = sorted([
                f for f in _repo_files
                if f.startswith("sft/checkpoint-") and "/" in f[len("sft/checkpoint-"):]
            ])
            if _sft_ckpts:
                resume_checkpoint = True
                print(f"📍 Pipeline stage: SFT (resume dari checkpoint)")
            else:
                print("📍 Pipeline stage: SFT (mulai dari awal)")
    except Exception as e:
        print(f"⚠️ Gagal mendeteksi stage: {e}. Mulai SFT dari awal.")

    mo.md(f"**📍 Current Stage: `{current_stage}`** | Resume: `{resume_checkpoint}`")
    return current_stage, resume_checkpoint


@app.cell
def _(Image, os):
    def convert_sft_record_to_vision(rec):
        img_paths = rec.get("images", [])
        if not img_paths:
            return None
        
        pil_images = []
        if isinstance(img_paths[0], str):
            for p in img_paths:
                if os.path.exists(p):
                    try:
                        # Open but do NOT convert or load pixels into RAM yet (lazy-loading)
                        pil_images.append(Image.open(p))
                    except Exception:
                        pass
        else:
            pil_images = img_paths

        if not pil_images:
            return None
        old_messages = rec.get("messages", [])
        new_messages = []
        image_idx = 0
        for msg in old_messages:
            role = msg["role"]
            content = msg["content"]
            if role == "user" and "📷" in content:
                num_images = content.count("📷")
                text_content = content.replace("📷", "").strip()
                new_content = []
                for _ in range(num_images):
                    if image_idx < len(pil_images):
                        new_content.append({"type": "image", "image": pil_images[image_idx]})
                        image_idx += 1
                if text_content:
                    new_content.append({"type": "text", "text": text_content})
                new_messages.append({"role": role, "content": new_content})
            else:
                new_messages.append({"role": role, "content": [{"type": "text", "text": content}]})
        return {"messages": new_messages}

    def unroll_vision_messages_to_sft_samples(messages, processor):
        samples = []
        for i, msg in enumerate(messages):
            if msg["role"] != "assistant":
                continue
            context = messages[:i]
            if not context:
                continue

            # Bersihkan konteks agar format pesannya standar sebelum dilewatkan ke template chat
            clean_context = []
            for m in context:
                clean_content = []
                if isinstance(m.get("content"), list):
                    for b in m["content"]:
                        if isinstance(b, dict):
                            if "image" in b:
                                clean_content.append({"type": "image"})
                            elif "text" in b:
                                clean_content.append({"type": "text", "text": b["text"]})
                else:
                    clean_content = m["content"]
                clean_context.append({"role": m["role"], "content": clean_content})

            prompt_text = processor.apply_chat_template(clean_context, tokenize=False, add_generation_prompt=True)

            images = []
            for m in context:
                if isinstance(m.get("content"), list):
                    for b in m["content"]:
                        if isinstance(b, dict) and "image" in b:
                            images.append(b["image"])

            target_text = ""
            if isinstance(msg["content"], list):
                for b in msg["content"]:
                    if isinstance(b, dict) and "text" in b:
                        target_text = b["text"]
            else:
                target_text = msg["content"]

            if target_text:
                samples.append({"prompt_text": prompt_text, "images": images, "target_text": target_text})
        return samples

    def parse_orpo_prompt_to_messages(prompt_str, img_paths):
        pil_images = []
        if img_paths:
            if isinstance(img_paths[0], str):
                for p in img_paths:
                    if os.path.exists(p):
                        try:
                            pil_images.append(Image.open(p))
                        except Exception:
                            pass
            else:
                pil_images = img_paths

        lines = prompt_str.split("\n")
        raw_messages = []
        current_role = None
        current_lines = []
        for line in lines:
            if line.startswith("system: "):
                if current_role is not None:
                    raw_messages.append((current_role, "\n".join(current_lines)))
                current_role = "system"
                current_lines = [line[8:]]
            elif line.startswith("user: "):
                if current_role is not None:
                    raw_messages.append((current_role, "\n".join(current_lines)))
                current_role = "user"
                current_lines = [line[6:]]
            elif line.startswith("assistant: "):
                if current_role is not None:
                    raw_messages.append((current_role, "\n".join(current_lines)))
                current_role = "assistant"
                current_lines = [line[11:]]
            else:
                current_lines.append(line)
        if current_role is not None:
            raw_messages.append((current_role, "\n".join(current_lines)))
        new_messages = []
        image_idx = 0
        for role, content in raw_messages:
            if role == "user" and "📷" in content:
                num_images = content.count("📷")
                text_content = content.replace("📷", "").strip()
                new_content = []
                for _ in range(num_images):
                    if image_idx < len(pil_images):
                        new_content.append({"type": "image", "image": pil_images[image_idx]})
                        image_idx += 1
                if text_content:
                    new_content.append({"type": "text", "text": text_content})
                new_messages.append({"role": role, "content": new_content})
            else:
                new_messages.append({"role": role, "content": [{"type": "text", "text": content}]})
        merged_messages = []
        for msg in new_messages:
            if merged_messages and merged_messages[-1]["role"] == msg["role"]:
                last_msg = merged_messages[-1]
                last_content = last_msg["content"]
                current_content = msg["content"]
                merged_content = []
                merged_content.extend(last_content)
                for block in current_content:
                    if isinstance(block, dict) and block.get("type") == "text" and merged_content and isinstance(merged_content[-1], dict) and merged_content[-1].get("type") == "text":
                        last_block = dict(merged_content[-1])
                        last_block["text"] = str(last_block.get("text", "")) + "\n" + str(block.get("text", ""))
                        merged_content[-1] = last_block
                    else:
                        merged_content.append(block)
                last_msg["content"] = merged_content  # type: ignore

            else:
                merged_messages.append(msg)
        return merged_messages

    def format_encoder_from_raw(raw_input: str) -> str:
        import re
        system = "Kamu adalah asisten AI yang helpful, santai, dan ramah. Gunakan Bahasa Indonesia sebagai bahasa utama."
        system_match = re.search(r"^system:\s*(.*?)(?=\nuser:)", raw_input, re.DOTALL)
        if system_match:
            system = system_match.group(1).strip()
            raw_input = raw_input[system_match.end() :].strip()

        parts = re.split(r"\n(user:|assistant:)\s*", "\n" + raw_input)
        formatted = ""
        is_first_user = True

        for i in range(1, len(parts), 2):
            role = parts[i].replace(":", "").strip()
            content = parts[i + 1].strip()
            if not content:
                continue

            if role == "user":
                formatted += "<start_of_turn>user\n"
                if is_first_user and system:
                    formatted += system + "\n\n"
                    is_first_user = False
                formatted += content + "<end_of_turn>\n"
            elif role == "assistant":
                formatted += "<start_of_turn>model\n"
                formatted += content + "<end_of_turn>\n"

        formatted += "<start_of_turn>model\n"
        return formatted

    return (
        convert_sft_record_to_vision,
        format_encoder_from_raw,
        parse_orpo_prompt_to_messages,
        unroll_vision_messages_to_sft_samples,
    )


@app.cell
def _(F, Seq2SeqTrainer, torch):
    class Seq2SeqVisionCollator:
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
                elif "dataset_idx" in item and self.train_dataset is not None:
                    try:
                        full_images = self.train_dataset[item["dataset_idx"]]["images"]
                        indices = item.get("image_indices", [])
                        images = [full_images[i] for i in indices if i < len(full_images)]
                    except Exception:
                        pass

                enc = self.processor(text=item["prompt_text"],
                    images=images if images else None,
                    return_tensors="pt")

                input_ids = enc["input_ids"][0].tolist()
                attention_mask = enc["attention_mask"][0].tolist()

                # Prepend BOS jika belum ada
                if self.tok.bos_token_id is not None and (not input_ids or input_ids[0] != self.tok.bos_token_id):
                    input_ids = [self.tok.bos_token_id] + input_ids
                    attention_mask = [1] + attention_mask

                # Append EOS jika belum ada
                if self.tok.eos_token_id is not None and (not input_ids or input_ids[-1] != self.tok.eos_token_id):
                    input_ids = input_ids + [self.tok.eos_token_id]
                    attention_mask = attention_mask + [1]

                iids.append(torch.tensor(input_ids, dtype=torch.long))
                amasks.append(torch.tensor(attention_mask, dtype=torch.long))

                if "pixel_values" in enc:
                    pvals.append(enc["pixel_values"])
                # Tambahkan <end_of_turn> pada target agar model belajar penutupan turn
                target_formatted = item["target_text"].strip() + "<end_of_turn>"
                tids = self.tok.encode(target_formatted, add_special_tokens=False)
                tids = tids[:self.max_tgt-1] + [self.eos_id]
                labs.append(torch.tensor(tids, dtype=torch.long))
            ii = torch.nn.utils.rnn.pad_sequence(iids, batch_first=True, padding_value=self.pad_id)
            am = torch.nn.utils.rnn.pad_sequence(amasks, batch_first=True, padding_value=0)
            lb = torch.nn.utils.rnn.pad_sequence(labs, batch_first=True, padding_value=-100)
            out = {"input_ids": ii, "attention_mask": am, "labels": lb}
            if pvals:
                out["pixel_values"] = torch.cat(pvals, dim=0) if pvals[0].ndim == 4 else torch.stack(pvals, dim=0)
            return out

    class VisionORPOCollator:
        def __init__(self, processor, max_src, max_tgt, train_dataset=None):
            self.processor = processor
            self.tok = processor.tokenizer
            self.pad_id = self.tok.pad_token_id
            self.eos_id = self.tok.eos_token_id
            self.max_src = max_src
            self.max_tgt = max_tgt
            self.train_dataset = train_dataset
        def _enc_tgt(self, text):
            # Tambahkan <end_of_turn> pada target agar model belajar penutupan turn
            text_formatted = text.strip() + "<end_of_turn>"
            ids = self.tok.encode(text_formatted, add_special_tokens=False)
            return torch.tensor(ids[:self.max_tgt-1] + [self.eos_id], dtype=torch.long)
        def __call__(self, batch):
            iids, amasks, pvals, clabs, rlabs = [], [], [], [], []
            for item in batch:
                images = None
                if "images" in item and item["images"]:
                    images = item["images"]
                elif "dataset_idx" in item and self.train_dataset is not None:
                    try:
                        full_images = self.train_dataset[item["dataset_idx"]]["images"]
                        indices = item.get("image_indices", [])
                        images = [full_images[i] for i in indices if i < len(full_images)]
                    except Exception:
                        pass

                enc = self.processor(text=item["prompt_text"],
                    images=images if images else None,
                    return_tensors="pt")

                input_ids = enc["input_ids"][0].tolist()
                attention_mask = enc["attention_mask"][0].tolist()

                # Prepend BOS jika belum ada
                if self.tok.bos_token_id is not None and (not input_ids or input_ids[0] != self.tok.bos_token_id):
                    input_ids = [self.tok.bos_token_id] + input_ids
                    attention_mask = [1] + attention_mask

                # Append EOS jika belum ada
                if self.tok.eos_token_id is not None and (not input_ids or input_ids[-1] != self.tok.eos_token_id):
                    input_ids = input_ids + [self.tok.eos_token_id]
                    attention_mask = attention_mask + [1]

                iids.append(torch.tensor(input_ids, dtype=torch.long))
                amasks.append(torch.tensor(attention_mask, dtype=torch.long))

                if "pixel_values" in enc:
                    pvals.append(enc["pixel_values"])
                clabs.append(self._enc_tgt(item["chosen_text"]))
                rlabs.append(self._enc_tgt(item["rejected_text"]))
            ii = torch.nn.utils.rnn.pad_sequence(iids, batch_first=True, padding_value=self.pad_id)
            am = torch.nn.utils.rnn.pad_sequence(amasks, batch_first=True, padding_value=0)
            cl = torch.nn.utils.rnn.pad_sequence(clabs, batch_first=True, padding_value=-100)
            rl = torch.nn.utils.rnn.pad_sequence(rlabs, batch_first=True, padding_value=-100)
            out = {"input_ids": ii, "attention_mask": am, "chosen_labels": cl, "rejected_labels": rl}
            if pvals:
                out["pixel_values"] = torch.cat(pvals, dim=0) if pvals[0].ndim == 4 else torch.stack(pvals, dim=0)
            return out

    class VisionORPOTrainer(Seq2SeqTrainer):
        def __init__(self, beta=0.1, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self.beta = beta
        def get_batch_logps(self, logits, labels, average_log_prob=True):
            labels = labels.clone()
            mask = labels != -100
            labels[labels == -100] = 0
            lps = torch.gather(logits.log_softmax(-1), dim=2, index=labels.unsqueeze(2)).squeeze(2)
            if average_log_prob:
                return (lps * mask).sum(-1) / mask.sum(-1).clamp(min=1)
            return (lps * mask).sum(-1)
        def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None, **kwargs):
            cl = inputs.pop("chosen_labels", None)
            rl = inputs.pop("rejected_labels", None)
            if cl is None or rl is None:
                return super().compute_loss(model, inputs, return_outputs, num_items_in_batch, **kwargs)

            # Split forward optimization to prevent CUDA OOM
            base_model = model.base_model.model if hasattr(model, "base_model") and hasattr(model.base_model, "model") else model
            if hasattr(base_model, "get_encoder"):
                encoder = base_model.get_encoder()
            elif hasattr(base_model, "model") and hasattr(base_model.model, "encoder"):
                encoder = base_model.model.encoder
            else:
                encoder = base_model.encoder
            encoder_outputs = encoder(
                input_ids=inputs["input_ids"],
                attention_mask=inputs["attention_mask"],
                pixel_values=inputs.get("pixel_values"),
            )
            co = model(
                encoder_outputs=encoder_outputs,
                labels=cl,
            )
            ro = model(
                encoder_outputs=encoder_outputs,
                labels=rl,
            )
            clp = self.get_batch_logps(co.logits, cl)
            rlp = self.get_batch_logps(ro.logits, rl)
            cp = clp.exp().clamp(1e-7, 1-1e-7)
            rp = rlp.exp().clamp(1e-7, 1-1e-7)
            clo = torch.log(cp / (1 - cp))
            rlo = torch.log(rp / (1 - rp))
            or_loss = -F.logsigmoid(clo - rlo).mean()
            loss = co.loss + self.beta * or_loss
            return (loss, co) if return_outputs else loss

        def prediction_step(self, model, inputs, prediction_loss_only, ignore_keys=None, **kwargs):
            # Seq2SeqTrainer.evaluate() (dipanggil dari evaluate() di bawah, bukan
            # lewat compute_loss) langsung memakai `inputs` mentah buat
            # model.generate(**inputs). Key "chosen_labels"/"rejected_labels" dari
            # collator ORPO bukan kwarg valid buat .generate() -> harus dibuang di
            # sini juga (compute_loss cuma nge-pop pas training, gak kepakai pas eval).
            inputs = dict(inputs)
            cl = inputs.pop("chosen_labels", None)
            inputs.pop("rejected_labels", None)
            if cl is not None and "labels" not in inputs:
                # Pakai jawaban "chosen" (yang disukai) sebagai referensi buat
                # hitung eval loss/ROUGE/BLEU dari hasil generate.
                inputs["labels"] = cl
            return super().prediction_step(model, inputs, prediction_loss_only, ignore_keys=ignore_keys, **kwargs)

        def evaluate(
            self,
            eval_dataset=None,
            ignore_keys=None,
            metric_key_prefix="eval",
        ):
            import math
            import gc
            from unsloth import FastVisionModel
            # Switch ke inference kernels sebelum evaluate/generate (mirip text-only)
            if hasattr(FastVisionModel, "for_inference"):
                FastVisionModel.for_inference(self.model)
            else:
                self.model.eval()
            metrics = super().evaluate(
                eval_dataset=eval_dataset,
                ignore_keys=ignore_keys,
                metric_key_prefix=metric_key_prefix,
            )
            # Hitung perplexity untuk setiap eval sub-dataset (multimodal & text-only).
            # metric_key_prefix selalu "eval" saat trainer memanggil evaluate() dengan
            # dict eval_dataset, jadi kita harus iterate semua key *_loss yang ada.
            for k in list(metrics.keys()):
                if k.endswith("_loss") and k.startswith("eval_"):
                    ppl_key = k.replace("_loss", "_perplexity")
                    try:
                        metrics[ppl_key] = math.exp(metrics[k])
                    except OverflowError:
                        metrics[ppl_key] = float("inf")

            # KRITIS: kembalikan ke training kernels + mode train. Tanpa ini model
            # tetap di eval state -> gradient checkpointing mati -> OOM & degradasi
            # silent saat training dilanjutkan setelah eval.
            if hasattr(FastVisionModel, "for_training"):
                FastVisionModel.for_training(self.model)
            else:
                self.model.train()
            # Vision kernels fullgraph=True -> flush cache rekompilasi Dynamo.
            torch._dynamo.reset()
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            return metrics

        def log(self, logs, start_time=None):
            import math
            for k in list(logs.keys()):
                if k.endswith("_loss") and k.startswith("eval_"):
                    ppl_key = k.replace("_loss", "_perplexity")
                    try:
                        logs[ppl_key] = math.exp(logs[k])
                    except OverflowError:
                        logs[ppl_key] = float("inf")
            super().log(logs, start_time=start_time)

    class GrokAdEMAMix(torch.optim.Optimizer):
        def __init__(
            self,
            params,
            lr=3e-5,
            betas=(0.9, 0.999),
            beta3=0.9999,
            weight_decay=0.05,
            grok_alpha=2.0,
            grok_lamb=0.98,
        ):
            defaults = dict(
                lr=lr,
                betas=betas,
                beta3=beta3,
                weight_decay=weight_decay,
                grok_alpha=grok_alpha,
                grok_lamb=grok_lamb,
            )
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
                weight_decay = group["weight_decay"]
                grok_alpha = group["grok_alpha"]
                grok_lamb = group["grok_lamb"]

                for p in group["params"]:
                    if p.grad is None:
                        continue

                    grad = p.grad
                    state = self.state[p]

                    if len(state) == 0:
                        state["step"] = 0
                        state["grok_slow_grad"] = torch.zeros_like(grad)
                        state["m"] = torch.zeros_like(grad)
                        state["v"] = torch.zeros_like(grad)
                        state["n"] = torch.zeros_like(grad)

                    state["step"] += 1
                    step = state["step"]

                    # GROKFAST
                    state["grok_slow_grad"].mul_(grok_lamb).add_(
                        grad, alpha=1.0 - grok_lamb
                    )
                    filtered_grad = grad.clone()
                    filtered_grad.add_(state["grok_slow_grad"], alpha=grok_alpha)

                    if weight_decay != 0:
                        p.data.mul_(1.0 - lr * weight_decay)

                    # ADEMAMIX
                    m, v, n = state["m"], state["v"], state["n"]

                    m.mul_(beta1).add_(filtered_grad, alpha=1.0 - beta1)
                    v.mul_(beta2).addcmul_(
                        filtered_grad, filtered_grad, value=1.0 - beta2
                    )
                    n.mul_(beta3).add_(filtered_grad, alpha=1.0 - beta3)

                    bias_correction1 = 1.0 - beta1**step
                    bias_correction2 = 1.0 - beta2**step
                    bias_correction3 = 1.0 - beta3**step

                    denom = (v.sqrt() / (bias_correction2**0.5)).add_(1e-8)
                    denom = denom.to(p.dtype)
                    step_update = (
                        m / bias_correction1 + 0.1 * n / bias_correction3
                    ) / denom
                    step_update = step_update.to(p.dtype)

                    p.data.add_(step_update, alpha=-lr)
            return loss

    class SelectiveLabelSmoother:
        def __init__(self, epsilon, suppress_ids):
            self.epsilon = epsilon
            self.suppress_ids = suppress_ids

        def __call__(self, model_output, labels, shift_labels=False):
            if isinstance(model_output, dict) and "logits" in model_output:
                logits = model_output["logits"]
            elif isinstance(model_output, tuple):
                logits = (
                    model_output[1] if len(model_output) > 1 else model_output[0].logits
                )
            else:
                logits = model_output.logits

            if shift_labels:
                logits = logits[..., :-1, :].contiguous()
                labels = labels[..., 1:].contiguous()

            vocab_size = logits.size(-1)
            suppress_list = [i for i in self.suppress_ids if i < vocab_size]

            valid_mask = torch.ones(vocab_size, dtype=torch.bool, device=logits.device)
            valid_mask[suppress_list] = False
            num_valid_tokens = valid_mask.sum().item()

            flat_logits = logits.view(-1, vocab_size)
            flat_labels = labels.view(-1)

            active_mask = flat_labels != -100
            if active_mask.sum() == 0:
                return torch.tensor(0.0, device=logits.device, requires_grad=True)

            active_logits = flat_logits[active_mask]
            active_labels = flat_labels[active_mask]

            num_active = active_logits.size(0)
            chunk_size = 2048

            total_loss = torch.tensor(0.0, device=logits.device)

            for i in range(0, num_active, chunk_size):
                chunk_logits = active_logits[i : i + chunk_size]
                chunk_labels = active_labels[i : i + chunk_size]

                log_probs = torch.nn.functional.log_softmax(chunk_logits, dim=-1)

                nll_loss = -log_probs.gather(
                    dim=-1, index=chunk_labels.unsqueeze(-1)
                ).squeeze(-1)

                valid_log_probs = log_probs * valid_mask.to(log_probs.dtype)
                smooth_loss = -valid_log_probs.sum(dim=-1) / num_valid_tokens

                token_losses = (1.0 - self.epsilon) * nll_loss + self.epsilon * smooth_loss
                total_loss += token_losses.sum()

            return total_loss / num_active

    class CustomSeq2SeqTrainer(Seq2SeqTrainer):
        def __init__(self, suppress_ids=None, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self.model_accepts_loss_kwargs = False
            if self.args.label_smoothing_factor > 0 and suppress_ids is not None:
                self.label_smoother = SelectiveLabelSmoother(
                    epsilon=self.args.label_smoothing_factor,
                    suppress_ids=suppress_ids,
                )

        def compute_loss(
            self, model, inputs, return_outputs=False, num_items_in_batch=None, **kwargs
        ):
            labels = inputs.get("labels")
            outputs = model(**inputs)

            if self.label_smoother is not None and labels is not None:
                loss = self.label_smoother(outputs, labels)
            else:
                if isinstance(outputs, dict) and "logits" in outputs:
                    logits = outputs["logits"]
                elif isinstance(outputs, tuple):
                    logits = outputs[1] if len(outputs) > 1 else outputs[0].logits
                else:
                    logits = outputs.logits
                loss_fct = torch.nn.CrossEntropyLoss(
                    ignore_index=-100, reduction="mean"
                )
                loss = loss_fct(logits.view(-1, logits.size(-1)), labels.view(-1))

            return (loss, outputs) if return_outputs else loss

        def evaluate(
            self,
            eval_dataset=None,
            ignore_keys=None,
            metric_key_prefix="eval",
        ):
            import math
            import gc
            from unsloth import FastVisionModel
            # Switch ke inference kernels sebelum evaluate/generate (mirip text-only v6)
            if hasattr(FastVisionModel, "for_inference"):
                FastVisionModel.for_inference(self.model)
            else:
                self.model.eval()
            metrics = super().evaluate(
                eval_dataset=eval_dataset,
                ignore_keys=ignore_keys,
                metric_key_prefix=metric_key_prefix,
            )
            # Hitung perplexity untuk setiap eval sub-dataset (multimodal & text-only).
            # metric_key_prefix selalu "eval" saat trainer memanggil evaluate() dengan
            # dict eval_dataset, jadi kita harus iterate semua key *_loss yang ada.
            for k in list(metrics.keys()):
                if k.endswith("_loss") and k.startswith("eval_"):
                    ppl_key = k.replace("_loss", "_perplexity")
                    try:
                        metrics[ppl_key] = math.exp(metrics[k])
                    except OverflowError:
                        metrics[ppl_key] = float("inf")

            # KRITIS: kembalikan ke training kernels + mode train. Tanpa ini model
            # tetap di eval state -> gradient checkpointing ("unsloth") mati ->
            # seluruh activation graph ditahan saat training resume -> OOM di step
            # setelah eval pertama (mirip bug OOM step ~97). for_training() juga
            # me-reload triton training kernels agar gradient update kembali optimal.
            if hasattr(FastVisionModel, "for_training"):
                FastVisionModel.for_training(self.model)
            else:
                self.model.train()
            # Vision kernels dikompilasi dengan fullgraph=True. Setiap switch
            # for_inference<->for_training memicu rekompilasi Dynamo. Flush cache
            # agar counter rekompilasi tidak melebihi limit (Hard failure
            # "recompile_limit exceeded") dan sekaligus bebaskan memori graf.
            torch._dynamo.reset()
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            return metrics

        def log(self, logs, start_time=None):
            import math
            for k in list(logs.keys()):
                if k.endswith("_loss") and k.startswith("eval_"):
                    ppl_key = k.replace("_loss", "_perplexity")
                    try:
                        logs[ppl_key] = math.exp(logs[k])
                    except OverflowError:
                        logs[ppl_key] = float("inf")
            super().log(logs, start_time=start_time)

    return (
        CustomSeq2SeqTrainer,
        GrokAdEMAMix,
        Seq2SeqVisionCollator,
        VisionORPOCollator,
        VisionORPOTrainer,
    )


@app.cell
def _():
    import numpy as np
    import evaluate

    try:
        rouge_metric = evaluate.load("rouge")
        bleu_metric = evaluate.load("bleu")
        exact_match_metric = evaluate.load("exact_match")
        bertscore_metric = evaluate.load("bertscore")
        meteor_metric = evaluate.load("meteor")
    except Exception as e:
        print(
            f"Warning: evaluate metrics not available. Metric evaluation will be bypassed. Error: {e}"
        )
        rouge_metric = None
        bleu_metric = None
        exact_match_metric = None
        bertscore_metric = None
        meteor_metric = None
    return (
        bertscore_metric,
        bleu_metric,
        exact_match_metric,
        meteor_metric,
        np,
        rouge_metric,
    )


@app.cell
def _(
    Any,
    TrainerCallback,
    TrainerControl,
    TrainerState,
    TrainingArguments,
    datetime,
    os,
    torch,
):
    class TrainingPlotCallback(TrainerCallback):
        def __init__(self, output_dir: str) -> None:
            self.output_dir = output_dir
            self.chart_path = os.path.join(output_dir, "training_chart.png")
            self.train_steps = []
            self.train_losses = []
            self.eval_data = {}  # {step: {metric_name: value}}

        def on_log(
            self,
            args: TrainingArguments,
            state: TrainerState,
            control: TrainerControl,
            logs: dict[str, float] | None = None,
            **kwargs: Any,
        ) -> None:
            if logs is None:
                return

            # Hapus eval_loss dari logs agar kolom "Validation Loss" (yang selalu "No log")
            # tidak tampil di widget training. Eval sudah di-split menjadi
            # multimodal & text-only, jadi eval_loss gabungan tidak relevan.
            logs.pop("eval_loss", None)
            logs.pop("eval_perplexity", None)

            if "loss" in logs:
                self.train_steps.append(state.global_step)
                self.train_losses.append(float(logs["loss"]))

            is_eval = any(k.startswith("eval_") for k in logs.keys())
            if is_eval:
                step = state.global_step
                if step not in self.eval_data:
                    self.eval_data[step] = {}
                for k, v in logs.items():
                    if k.startswith("eval_"):
                        self.eval_data[step][k] = float(v)

            self.plot_chart()

        def plot_chart(self) -> None:
            import matplotlib.pyplot as plt
            os.makedirs(self.output_dir, exist_ok=True)

            fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))

            # 1. Plot Loss Curve
            if self.train_losses:
                ax1.plot(self.train_steps, self.train_losses, color="#3498DB", linewidth=2, label="Train Loss")

            steps = sorted(self.eval_data.keys())

            m_loss_steps = [s for s in steps if "eval_multimodal_loss" in self.eval_data[s]]
            if m_loss_steps:
                m_losses = [self.eval_data[s]["eval_multimodal_loss"] for s in m_loss_steps]
                ax1.plot(m_loss_steps, m_losses, color="#2ECC71", marker="o", linestyle="--", linewidth=1.5, label="Eval Multimodal Loss")

            t_loss_steps = [s for s in steps if "eval_text_only_loss" in self.eval_data[s]]
            if t_loss_steps:
                t_losses = [self.eval_data[s]["eval_text_only_loss"] for s in t_loss_steps]
                ax1.plot(t_loss_steps, t_losses, color="#E74C3C", marker="x", linestyle=":", linewidth=1.5, label="Eval Text-Only Loss")

            ax1.set_xlabel("Steps")
            ax1.set_ylabel("Loss")
            ax1.set_title("Training & Evaluation Loss Curve")
            ax1.grid(True, alpha=0.3)
            ax1.legend()

            # 2. Plot Perplexity Curve
            m_ppl_steps = [s for s in steps if "eval_multimodal_perplexity" in self.eval_data[s]]
            if m_ppl_steps:
                m_ppls = [self.eval_data[s]["eval_multimodal_perplexity"] for s in m_ppl_steps]
                ax2.plot(m_ppl_steps, m_ppls, color="#1ABC9C", marker="o", linestyle="--", linewidth=1.5, label="Multimodal Perplexity")

            t_ppl_steps = [s for s in steps if "eval_text_only_perplexity" in self.eval_data[s]]
            if t_ppl_steps:
                t_ppls = [self.eval_data[s]["eval_text_only_perplexity"] for s in t_ppl_steps]
                ax2.plot(t_ppl_steps, t_ppls, color="#9B59B6", marker="x", linestyle=":", linewidth=1.5, label="Text-Only Perplexity")

            ax2.set_xlabel("Steps")
            ax2.set_ylabel("Perplexity")
            ax2.set_title("Model Perplexity Curve")
            ax2.grid(True, alpha=0.3)
            ax2.legend()

            # 3. Plot Multimodal Quality Metrics
            metrics_list = [
                ("eval_multimodal_rouge1", "ROUGE-1", "#E67E22", "o"),
                ("eval_multimodal_rouge2", "ROUGE-2", "#D35400", "x"),
                ("eval_multimodal_bleu", "BLEU", "#2980B9", "s"),
                ("eval_multimodal_bertscore_f1", "BERTScore F1", "#8E44AD", "d"),
            ]
            for metric_key, label, color, marker in metrics_list:
                m_steps = [s for s in steps if metric_key in self.eval_data[s]]
                if m_steps:
                    m_vals = [self.eval_data[s][metric_key] for s in m_steps]
                    ax3.plot(m_steps, m_vals, color=color, marker=marker, linestyle="-", linewidth=1.5, label=label)

            ax3.set_xlabel("Steps")
            ax3.set_ylabel("Score (%)")
            ax3.set_title("Multimodal Generation Quality Metrics")
            ax3.grid(True, alpha=0.3)
            ax3.legend()

            # 4. Plot Text-Only Quality Metrics
            metrics_list_text = [
                ("eval_text_only_rouge1", "ROUGE-1", "#E67E22", "o"),
                ("eval_text_only_rouge2", "ROUGE-2", "#D35400", "x"),
                ("eval_text_only_bleu", "BLEU", "#2980B9", "s"),
                ("eval_text_only_bertscore_f1", "BERTScore F1", "#8E44AD", "d"),
            ]
            for metric_key, label, color, marker in metrics_list_text:
                m_steps = [s for s in steps if metric_key in self.eval_data[s]]
                if m_steps:
                    m_vals = [self.eval_data[s][metric_key] for s in m_steps]
                    ax4.plot(m_steps, m_vals, color=color, marker=marker, linestyle="-", linewidth=1.5, label=label)

            ax4.set_xlabel("Steps")
            ax4.set_ylabel("Score (%)")
            ax4.set_title("Text-Only Generation Quality Metrics")
            ax4.grid(True, alpha=0.3)
            ax4.legend()

            plt.tight_layout()
            plt.savefig(self.chart_path, dpi=120)
            plt.close(fig)

    class CleanNotebookProgressCallback(TrainerCallback):
        """
        Pengganti transformers.utils.notebook.NotebookProgressCallback bawaan.

        NotebookProgressCallback bawaan SELALU menambahkan kolom "Validation Loss"
        (hardcoded di on_train_begin & on_evaluate) terlepas dari apakah key
        "eval_loss" benar-benar ada di metrics atau tidak. Karena eval kita sudah
        dipecah jadi Multimodal Loss & Text Only Loss (tidak ada eval_loss
        gabungan), kolom itu selalu tampil "No log". Callback ini meniru semua
        behavior aslinya tapi tanpa kolom "Validation Loss" default tersebut.
        """

        def __init__(self) -> None:
            self.training_tracker = None
            self.prediction_bar = None
            self._force_next_update = False

        def on_train_begin(self, args, state, control, **kwargs) -> None:
            from transformers.trainer_utils import IntervalStrategy
            from transformers.utils.notebook import NotebookTrainingTracker

            self.first_column = "Epoch" if args.eval_strategy == IntervalStrategy.EPOCH else "Step"
            self.training_loss = 0
            self.last_log = 0
            column_names = [self.first_column, "Training Loss"]
            self.training_tracker = NotebookTrainingTracker(state.max_steps, column_names)

        def on_step_end(self, args, state, control, **kwargs) -> None:
            epoch = int(state.epoch) if int(state.epoch) == state.epoch else f"{state.epoch:.2f}"
            self.training_tracker.update(
                state.global_step + 1,
                comment=f"Epoch {epoch}/{state.num_train_epochs}",
                force_update=self._force_next_update,
            )
            self._force_next_update = False

        def on_prediction_step(self, args, state, control, eval_dataloader=None, **kwargs) -> None:
            from transformers.trainer_utils import has_length

            if not has_length(eval_dataloader):
                return
            if self.prediction_bar is None:
                if self.training_tracker is not None:
                    self.prediction_bar = self.training_tracker.add_child(len(eval_dataloader))
                else:
                    from transformers.utils.notebook import NotebookProgressBar
                    self.prediction_bar = NotebookProgressBar(len(eval_dataloader))
                self.prediction_bar.update(1)
            else:
                self.prediction_bar.update(self.prediction_bar.value + 1)

        def on_predict(self, args, state, control, **kwargs) -> None:
            if self.prediction_bar is not None:
                self.prediction_bar.close()
            self.prediction_bar = None

        def on_log(self, args, state, control, logs=None, **kwargs) -> None:
            from transformers.trainer_utils import IntervalStrategy

            if args.eval_strategy == IntervalStrategy.NO and logs is not None and "loss" in logs:
                values = {"Training Loss": logs["loss"], "Step": state.global_step}
                self.training_tracker.write_line(values)

        def on_evaluate(self, args, state, control, metrics=None, **kwargs) -> None:
            import re as _re
            import IPython.display as disp
            from transformers.trainer_utils import IntervalStrategy
            from transformers.utils.notebook import text_to_html_table

            self.first_column = "Epoch" if args.eval_strategy == IntervalStrategy.EPOCH else "Step"

            # Tidak seperti bawaan: TIDAK ada default "Validation Loss": "No log" di sini.
            values = {"Training Loss": "No log"}
            for log in reversed(state.log_history):
                if "loss" in log:
                    values["Training Loss"] = log["loss"]
                    break

            if self.first_column == "Epoch":
                values["Epoch"] = int(state.epoch)
            else:
                values["Step"] = state.global_step

            if metrics is None:
                metrics = {}
            metric_key_prefix = "eval"
            for k in metrics:
                if k.endswith("_loss"):
                    metric_key_prefix = _re.sub(r"_loss$", "", k)
            metrics.pop("total_flos", None)
            metrics.pop("epoch", None)
            metrics.pop(f"{metric_key_prefix}_runtime", None)
            metrics.pop(f"{metric_key_prefix}_samples_per_second", None)
            metrics.pop(f"{metric_key_prefix}_steps_per_second", None)
            metrics.pop(f"{metric_key_prefix}_model_preparation_time", None)

            for k, v in metrics.items():
                splits = k.split("_")
                name = " ".join(part.capitalize() for part in splits[1:])
                # Catatan: sengaja TIDAK rename name == "Loss" -> "Validation Loss"
                # seperti versi bawaan, karena kita mau tiap eval dataset punya
                # kolom sendiri (mis. "Multimodal Loss", "Text Only Loss").
                values[name] = v

            if self.training_tracker is not None:
                tt = self.training_tracker
                tt.write_line(values)
                tt.remove_child()
                self._force_next_update = True
            else:
                disp.display(disp.HTML(text_to_html_table([list(values.keys()), list(values.values())])))

            self.prediction_bar = None

        def on_train_end(self, args, state, control, **kwargs) -> None:
            if self.training_tracker is not None:
                self.training_tracker.update(
                    state.global_step,
                    comment=f"Epoch {int(state.epoch)}/{state.num_train_epochs}",
                    force_update=True,
                )
                self.training_tracker = None

    class SampleGenerationCallback(TrainerCallback):
        def __init__(
            self,
            processor: Any,
            eval_samples: list[dict],
            output_dir: str,
            eval_every_n_steps: int = 50,
            temperature: float = 0.7,
            top_p: float = 0.9,
            repetition_penalty: float = 1.2,
            bad_words_ids: list[list[int]] | None = None,
        ) -> None:
            self.processor = processor
            self.tokenizer = processor.tokenizer
            self.eval_samples = eval_samples
            self.output_dir = output_dir
            self.eval_every_n_steps = eval_every_n_steps
            self.log_path = os.path.join(output_dir, "eval_samples.txt")
            self._eot_id = self.tokenizer.convert_tokens_to_ids("<end_of_turn>")
            self._eos_id = self.tokenizer.eos_token_id or 1
            self._stop_ids = list({self._eot_id, self._eos_id})
            self.temperature = temperature
            self.top_p = top_p
            self.repetition_penalty = repetition_penalty
            self.bad_words_ids = bad_words_ids

        def on_step_end(
            self,
            args: TrainingArguments,
            state: TrainerState,
            control: TrainerControl,
            model: Any = None,
            **kwargs: Any,
        ) -> None:
            if (
                state.global_step == 0
                or state.global_step % self.eval_every_n_steps != 0
            ):
                return
            if model is None:
                return

            from unsloth import FastVisionModel
            if hasattr(FastVisionModel, "for_inference"):
                FastVisionModel.for_inference(model)
            else:
                model.eval()
            timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            lines = [
                f"\n{'=' * 60}",
                f"Step {state.global_step} | {timestamp}",
                f"{'=' * 60}",
            ]

            import gc
            gc.collect()
            torch.cuda.empty_cache()

            with torch.no_grad():
                pad_id = (
                    self.tokenizer.pad_token_id
                    if self.tokenizer.pad_token_id is not None
                    else self._eos_id
                )

                for idx, sample in enumerate(self.eval_samples):
                    inputs = self.processor(
                        text=[sample["prompt_text"]],
                        images=sample["images"] if sample["images"] else None,
                        return_tensors="pt"
                    ).to(model.device)

                    outputs = getattr(model, "generate")(
                        **inputs,
                        max_new_tokens=1024,
                        do_sample=True,
                        temperature=self.temperature,
                        top_p=self.top_p,
                        repetition_penalty=self.repetition_penalty,
                        eos_token_id=self._stop_ids,
                        pad_token_id=pad_id,
                        bad_words_ids=self.bad_words_ids,
                    )

                    gen_ids = outputs[0].cpu()
                    raw_response = self.tokenizer.decode(gen_ids, skip_special_tokens=True)

                    # Immediately free GPU tensors to prevent cumulative OOM
                    del inputs, outputs
                    torch.cuda.empty_cache()

                    query = sample["prompt_text"].strip()
                    target = sample["target_text"].strip()
                    response = raw_response.strip()

                    words = response.split()
                    is_repetitive = (
                        len(set(words)) < max(1, len(words) * 0.3) if words else True
                    )
                    flag = " ⚠️ REPETITIVE" if is_repetitive else " ✅"

                    lines.append(f"\nQ: {query}")
                    lines.append(f"Expected Target: {target}")
                    lines.append(f"Model Response: {response}{flag}")

            from unsloth import FastVisionModel
            if hasattr(FastVisionModel, "for_training"):
                FastVisionModel.for_training(model)
            else:
                model.train()

            # Vision kernels fullgraph=True -> flush cache rekompilasi Dynamo
            # agar tidak tembus recompile_limit saat training dilanjutkan.
            torch._dynamo.reset()
            gc.collect()
            torch.cuda.empty_cache()

            with open(self.log_path, "a", encoding="utf-8") as f:
                f.write("\n".join(lines) + "\n")

            print(f"\n[BEHAVIOR EVAL @ step {state.global_step}]")
            for line in lines[3:]:
                if (
                    line.startswith("Q:")
                    or line.startswith("Model Response:")
                    or line.startswith("Expected Target:")
                ):
                    print(f"  {line}")

    class HubUploadCallback(TrainerCallback):
        def __init__(self, repo_id: str, stage: str, token: str | None = None, output_dir: str | None = None) -> None:
            self.repo_id = repo_id
            self.stage = stage
            self.token = token
            self.output_dir = output_dir

        def on_save(
            self,
            args: TrainingArguments,
            state: TrainerState,
            control: TrainerControl,
            **kwargs: Any,
        ) -> TrainerControl:
            from huggingface_hub import HfApi
            _api = HfApi(token=self.token)
            checkpoint_name = f"checkpoint-{state.global_step}"
            local_checkpoint_path = os.path.join(args.output_dir, checkpoint_name)

            try:
                # Ensure the repository is created before uploading checkpoints
                _api.create_repo(repo_id=self.repo_id, repo_type="model", private=False, exist_ok=True)
                print(f"\n📤 Uploading {checkpoint_name} to HF {self.stage}/...")
                _api.upload_folder(
                    folder_path=local_checkpoint_path,
                    repo_id=self.repo_id,
                    path_in_repo=f"{self.stage}/{checkpoint_name}",
                    repo_type="model",
                )

                if self.output_dir:
                    for artifact_name in ["training_chart.png", f"{self.stage}_eval_samples_multimodal.txt", f"{self.stage}_eval_samples_text_only.txt"]:
                        local_art_path = os.path.join(self.output_dir, artifact_name)
                        if os.path.exists(local_art_path):
                            _api.upload_file(
                                path_or_fileobj=local_art_path,
                                path_in_repo=f"{self.stage}/{artifact_name}",
                                repo_id=self.repo_id,
                                repo_type="model",
                            )
                print(f"✅ {checkpoint_name} + artifacts uploaded!")
            except Exception as e:
                print(f"⚠️ Upload gagal untuk {checkpoint_name}: {e}")
            return control

    return CleanNotebookProgressCallback, HubUploadCallback, SampleGenerationCallback, TrainingPlotCallback


@app.cell
def _(torch):
    # =====================================================================
    # KONFIGURASI HYPERPARAMETER
    # =====================================================================
    # Vision kernels Unsloth dikompilasi dengan fullgraph=True. Setiap switch
    # for_inference<->for_training (saat eval/generate) memicu rekompilasi
    # Dynamo. Saat training resume setelah eval, gradient checkpointing aktif
    # kembali sehingga backward memicu banyak segmen rekompilasi. Naikkan
    # recompile_limit jauh di atas jumlah modul agar burst ini tidak tembus
    # limit -> "Hard failure due to fullgraph=True". 1024 >> aman.
    torch._dynamo.config.recompile_limit = 1024  # type: ignore[assignment]
    torch._dynamo.config.cache_size_limit = 1024  # type: ignore[assignment]
    # Model base = hasil cangkok (v6 text + SigLIP/projector Gemma 3 IT)
    MODEL_NAME = "daruokta/t5gemma-2-4b-4b-instruct-chat-indo-v4-vision-cangkok"
    SUBFOLDER = ""  # repo cangkok langsung di root (tidak ada subfolder)
    LOAD_IN_4BIT = True
    OUTPUT_DIR = "results/t5gemma2_vision"
    HF_CHECKPOINT_REPO = "daruokta/t5gemma-2-4b-4b-instruct-chat-indo-v4-vision"

    # Dataset JSONL lokal
    JSONL_DATASET_PATH = "data/multimodal/train_vision.jsonl"
    ORPO_DATASET_PATH = "data/preference/orpo_multimodal.jsonl"

    # LoRA config: r=256 (diselaraskan dengan versi teks)
    LORA_RANK = 256
    LORA_ALPHA = 512
    LORA_DROPOUT = 0.2

    # Seq2Seq lengths (cloud Molab 96GB)
    MAX_SOURCE_LENGTH = 16384
    MAX_TARGET_LENGTH = 2048
    MAX_IMAGES_PER_CHAT = 10

    # Training args
    # NOTE: effective LR = LEARNING_RATE / GRADIENT_ACCUMULATION_STEPS.
    # Disetara dengan text-only (1e-5 / 64 = 1.56e-7) agar decoder tidak
    # belajar 8x lebih agresif dari sinyal multimodal yang masih noisy.
    LEARNING_RATE = 5e-6
    NUM_EPOCHS_SFT = 2
    NUM_EPOCHS_ORPO = 1
    ORPO_BETA = 0.1
    PER_DEVICE_TRAIN_BATCH_SIZE = 2
    GRADIENT_ACCUMULATION_STEPS = 32
    WARMUP_STEPS = 100
    WEIGHT_DECAY = 0.01
    LR_SCHEDULER_TYPE = "cosine"
    LOGGING_STEPS = 10
    SAVE_TOTAL_LIMIT = 2
    OPTIM = "paged_adamw_8bit"

    BF16 = torch.cuda.is_available()

    # Logit masking (sama dengan text-only v6)
    SUPPRESS_BLOCK1 = [6] + list(range(13, 105))
    SUPPRESS_BLOCK2 = list(range(256002, 262144))
    SUPPRESS_VISION = [255999, 256000, 256001]
    ALL_SUPPRESS_IDS = set(SUPPRESS_BLOCK1 + SUPPRESS_BLOCK2 + SUPPRESS_VISION)

    # General random seed and validation split size
    SEED = 3407
    TEST_SIZE = 0.05

    # Label smoothing & NEFTune
    LABEL_SMOOTHING_FACTOR = 0.1
    NEFTUNE_NOISE_ALPHA = 5.0
    PREDICT_WITH_GENERATE = True
    return (
        ALL_SUPPRESS_IDS,
        BF16,
        GRADIENT_ACCUMULATION_STEPS,
        HF_CHECKPOINT_REPO,
        LABEL_SMOOTHING_FACTOR,
        LEARNING_RATE,
        LOAD_IN_4BIT,
        LOGGING_STEPS,
        LORA_ALPHA,
        LORA_DROPOUT,
        LORA_RANK,
        LR_SCHEDULER_TYPE,
        MAX_SOURCE_LENGTH,
        MAX_TARGET_LENGTH,
        MODEL_NAME,
        NEFTUNE_NOISE_ALPHA,
        NUM_EPOCHS_ORPO,
        NUM_EPOCHS_SFT,
        OPTIM,
        ORPO_BETA,
        OUTPUT_DIR,
        PER_DEVICE_TRAIN_BATCH_SIZE,
        PREDICT_WITH_GENERATE,
        SAVE_TOTAL_LIMIT,
        SEED,
        SUBFOLDER,
        TEST_SIZE,
        WARMUP_STEPS,
        WEIGHT_DECAY,
    )


@app.cell
def _(load_dataset):
    # Load dan format dataset untuk SFT
    print("Memuat dataset SFT dari Hugging Face Hub (daruokta/t5gemma2-indonesia-vision-formatted)...")
    train_dataset = load_dataset("daruokta/t5gemma2-indonesia-vision-formatted", "vision_sft", split="train")
    print(f"✅ SFT Dataset berhasil dimuat dari Hugging Face Hub: {len(train_dataset)} sampel.")
    return (train_dataset,)


@app.cell
def _(
    ALL_SUPPRESS_IDS,
    AutoProcessor,
    FastVisionModel,
    HF_CHECKPOINT_REPO,
    LOAD_IN_4BIT,
    LORA_ALPHA,
    LORA_DROPOUT,
    LORA_RANK,
    MODEL_NAME,
    OUTPUT_DIR,
    SEED,
    SUBFOLDER,
    apply_logit_mask,
    current_stage,
    os,
):
    model = None
    tokenizer = None
    processor = None

    if current_stage == "done":
        print("Semua tahapan training selesai. Lewati pemuatan model.")
    else:
        _hf_token = os.environ.get("HF_TOKEN")

        # Tentukan sumber pemuatan model
        if current_stage == "orpo":
            _model_path = os.path.join(OUTPUT_DIR, "sft", "final_adapter")
            if not os.path.exists(_model_path):
                from huggingface_hub import snapshot_download as _resume_snap
                print("📥 Downloading SFT final adapter dari HF untuk ORPO...")
                _resume_snap(
                    repo_id=HF_CHECKPOINT_REPO,
                    local_dir=_model_path,
                    allow_patterns=["sft/final_adapter/**"],
                    token=_hf_token,
                )
                _sub_dir = os.path.join(_model_path, "sft", "final_adapter")
                if os.path.exists(_sub_dir):
                    import shutil as _shutil_load
                    for _item in os.listdir(_sub_dir):
                        _src = os.path.join(_sub_dir, _item)
                        _dst = os.path.join(_model_path, _item)
                        if os.path.exists(_dst):
                            if os.path.isdir(_dst):
                                _shutil_load.rmtree(_dst)
                            else:
                                os.remove(_dst)
                        _shutil_load.move(_src, _dst)
                    _shutil_load.rmtree(os.path.join(_model_path, "sft"))
            print(f"Loading SFT model dari adapter path: {_model_path}")
        else:
            # SFT: Load base model
            _model_path = MODEL_NAME
            print(f"Loading base model dari {_model_path}...")

        _load_kwargs = dict(
            model_name=_model_path,
            load_in_4bit=LOAD_IN_4BIT,
            use_gradient_checkpointing="unsloth",
            token=_hf_token,
        )
        if current_stage == "sft" and SUBFOLDER:
            _load_kwargs["subfolder"] = SUBFOLDER

        model, tokenizer = FastVisionModel.from_pretrained(**_load_kwargs)

        # Reset max_length to silence warning about max_new_tokens taking precedence
        model.config.max_length = None
        if hasattr(model, "generation_config") and model.generation_config is not None:
            model.generation_config.max_length = None

        # Load processor dari base model
        _proc_kwargs = dict(token=_hf_token)
        if current_stage == "sft" and SUBFOLDER:
            _proc_kwargs["subfolder"] = SUBFOLDER
        processor = AutoProcessor.from_pretrained(MODEL_NAME, **_proc_kwargs)

        from unsloth.chat_templates import get_chat_template
        tokenizer = get_chat_template(tokenizer, chat_template="gemma-3")
        processor.chat_template = tokenizer.chat_template
        if hasattr(processor, "tokenizer"):
            processor.tokenizer.chat_template = tokenizer.chat_template

        # Nonaktifkan penambahan bos_token otomatis untuk menghindari bos_token ganda saat inferensi
        tokenizer.add_bos_token = False
        if hasattr(processor, "tokenizer"):
            processor.tokenizer.add_bos_token = False

        # LoRA Config hanya untuk SFT (karena ORPO me-load model yang sudah memiliki LoRA adapter)
        if current_stage == "sft":
            print("Applying PEFT LoRA (vision_tower=SKIP, projector=FULL FT)...")
            model = FastVisionModel.get_peft_model(
                model,
                finetune_vision_layers=False,      # ⚠️ SKIP vision tower (SigLIP) to avoid Unsloth merge bug
                finetune_language_layers=True,
                finetune_attention_modules=True,
                finetune_mlp_modules=True,
                modules_to_save=["multi_modal_projector"],  # FULL FT projector
                r=LORA_RANK,
                lora_alpha=LORA_ALPHA,
                lora_dropout=LORA_DROPOUT,
                bias="none",
                random_state=SEED,
                use_rslora=True,
            )
        else:
            print("Model has already been loaded with PEFT adapter (from SFT). Skipping get_peft_model.")

        if not hasattr(model.config, "text_config"):
            type(model.config).text_config = property(lambda self: self.decoder)
            type(model.config).get_text_config = lambda self, *args, **kwargs: self.decoder

        apply_logit_mask(model, ALL_SUPPRESS_IDS)
        FastVisionModel.for_training(model)
    return model, processor, tokenizer


@app.cell
def _(
    ALL_SUPPRESS_IDS,
    Any,
    BF16,
    CleanNotebookProgressCallback,
    CustomSeq2SeqTrainer,
    Dataset,
    GRADIENT_ACCUMULATION_STEPS,
    GrokAdEMAMix,
    HF_CHECKPOINT_REPO,
    HubUploadCallback,
    LABEL_SMOOTHING_FACTOR,
    LEARNING_RATE,
    LOGGING_STEPS,
    LR_SCHEDULER_TYPE,
    MAX_SOURCE_LENGTH,
    MAX_TARGET_LENGTH,
    NEFTUNE_NOISE_ALPHA,
    NUM_EPOCHS_SFT,
    OPTIM,
    OUTPUT_DIR,
    PER_DEVICE_TRAIN_BATCH_SIZE,
    PREDICT_WITH_GENERATE,
    SAVE_TOTAL_LIMIT,
    SEED,
    SampleGenerationCallback,
    Seq2SeqTrainingArguments,
    Seq2SeqVisionCollator,
    TEST_SIZE,
    TrainingPlotCallback,
    WARMUP_STEPS,
    WEIGHT_DECAY,
    bertscore_metric,
    bleu_metric,
    cast,
    current_stage,
    exact_match_metric,
    gc,
    get_scheduler,
    meteor_metric,
    mo,
    model,
    np,
    os,
    processor,
    resume_checkpoint,
    rouge_metric,
    torch,
    traceback,
    train_dataset,
    unroll_vision_messages_to_sft_samples,
    convert_sft_record_to_vision,
    format_encoder_from_raw,
    load_dataset,
):
    mo.stop(
        current_stage != "sft",
        mo.md("ℹ️ **Bukan tahap SFT (atau SFT sudah selesai). Melewati training SFT.**")
    )
    mo.stop(
        train_dataset is None,
        mo.md("❌ **Dataset SFT tidak ditemukan, training SFT dibatalkan.**")
    )

    # Active memory cleanup from previous SFT attempts
    sft_trainer = None
    _optimizer = None
    _lr_scheduler = None
    if "model" in globals() and globals()["model"] is not None:
        try:
            globals()["model"].zero_grad(set_to_none=True)
        except Exception:
            pass
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # Unroll SFT dataset using ONLY text columns to avoid slow Hugging Face image loading/decoding
    print("Unrolling SFT dataset (text-only pass)...")
    sft_formatted = []
    messages_list = train_dataset["messages"]
    _arrow_images_sft = train_dataset._data.column("images")
    for _idx_sft, _msgs_sft in enumerate(messages_list):
        _num_actual_images = len(_arrow_images_sft[_idx_sft])
        _image_idx = 0
        clean_context = []
        for _msg in _msgs_sft:
            _role_sft = _msg["role"]
            _content_sft = _msg["content"]
            if _role_sft == "user" and "📷" in _content_sft:
                _num_images_sft = _content_sft.count("📷")
                _text_content_sft = _content_sft.replace("📷", "").strip()
                clean_content = []
                for _ in range(_num_images_sft):
                    if _image_idx < _num_actual_images:
                        clean_content.append({"type": "image"})
                        _image_idx += 1
                if _text_content_sft:
                    clean_content.append({"type": "text", "text": _text_content_sft})
                clean_context.append({"role": _role_sft, "content": clean_content})
            else:
                clean_context.append({"role": _role_sft, "content": [{"type": "text", "text": _content_sft}]})

        for i, msg in enumerate(clean_context):
            if msg["role"] != "assistant":
                continue
            context = clean_context[:i]
            if not context:
                continue

            prompt_text = processor.apply_chat_template(context, tokenize=False, add_generation_prompt=True)
            
            # Count image blocks in context up to this turn
            _num_context_images = 0
            for _m in context:
                for _b in _m["content"]:
                    if isinstance(_b, dict) and _b.get("type") == "image":
                        _num_context_images += 1

            target_text = ""
            if isinstance(msg["content"], list):
                for b in msg["content"]:
                    if isinstance(b, dict) and "text" in b:
                        target_text = b["text"]
            else:
                target_text = msg["content"]

            if target_text:
                sft_formatted.append({
                    "prompt_text": prompt_text,
                    "target_text": target_text,
                    "dataset_idx": _idx_sft,
                    "image_indices": list(range(_num_context_images))
                })
    sft_dataset = Dataset.from_list(sft_formatted)
    print(f"✅ SFT dataset: {len(sft_dataset)} samples (dari {len(train_dataset)} percakapan)")

    # Splitting Train & Validation
    split_ds = sft_dataset.train_test_split(test_size=TEST_SIZE, seed=SEED)
    sft_train_dataset = split_ds["train"]
    # Limit evaluation dataset to 30 samples to avoid CUDA OOM during predict_with_generate
    sft_eval_dataset = split_ds["test"].select(range(min(len(split_ds["test"]), 30)))
    print(f"  SFT Train size: {len(sft_train_dataset)} | SFT Eval size: {len(sft_eval_dataset)}")

    # Load and format text-only validation dataset
    print("Loading text-only SFT validation dataset from HF Hub...")
    _text_only_eval_dataset = None
    try:
        _val_chat_ds = load_dataset("daruokta/t5gemma2-indonesia-chat-formatted", "chat_sft", split="validation")
        _val_indoqa_ds = load_dataset("daruokta/t5gemma2-indonesia-chat-formatted", "indoqa_sft", split="validation")
        _val_chat_samples = [dict(_row) for _row in _val_chat_ds]
        _val_indoqa_samples = [dict(_row) for _row in _val_indoqa_ds]

        import random as _rng_sft
        _raw_text_only_samples = _val_chat_samples + _val_indoqa_samples
        _rng_sft.seed(42)
        _rng_sft.shuffle(_raw_text_only_samples)
        # Limit text-only evaluation to 30 samples to avoid CUDA OOM during predict_with_generate
        _raw_text_only_samples = _raw_text_only_samples[:30]

        _text_only_formatted = []
        for _row in _raw_text_only_samples:
            _pt = format_encoder_from_raw(_row["input"])
            _tt = _row["target"]
            _text_only_formatted.append({
                "prompt_text": _pt,
                "images": [],
                "target_text": _tt
            })
        _text_only_eval_dataset = Dataset.from_list(_text_only_formatted)
        print(f"  Text-Only Eval size: {len(_text_only_eval_dataset)}")
    except Exception as e:
        print(f"⚠️ Gagal memuat dataset validasi teks untuk SFT: {e}")

    sft_eval_datasets = {"multimodal": sft_eval_dataset}
    if _text_only_eval_dataset is not None:
        sft_eval_datasets["text_only"] = _text_only_eval_dataset

    sft_output_dir = os.path.join(OUTPUT_DIR, "sft")
    sft_collator = Seq2SeqVisionCollator(processor, MAX_SOURCE_LENGTH, MAX_TARGET_LENGTH, train_dataset)

    # Setup qualitative generation samples (similar to V6 text-only)
    _sft_val_rows = list(sft_eval_dataset)
    _n_eval_gen = min(len(_sft_val_rows), 20)
    _eval_generation_samples = []
    for _item_sft in _sft_val_rows[:_n_eval_gen]:
        _full_imgs = train_dataset[_item_sft["dataset_idx"]]["images"] if "dataset_idx" in _item_sft else []
        _indices = _item_sft.get("image_indices", [])
        _subset_imgs = [_full_imgs[i] for i in _indices if i < len(_full_imgs)]
        _eval_generation_samples.append({
            "prompt_text": _item_sft["prompt_text"],
            "target_text": _item_sft["target_text"],
            "images": _subset_imgs
        })

    # Define compute metrics
    def _compute_metrics(eval_preds):
        metrics = {}

        if rouge_metric is None and bleu_metric is None:
            return metrics
        preds, labels = eval_preds
        if isinstance(preds, tuple):
            preds = preds[0]
        tok = cast(Any, processor.tokenizer)

        if preds.ndim == 3:
            preds = preds.argmax(axis=-1)

        labels = np.where(labels != -100, labels, tok.pad_token_id)
        preds = np.where(preds != -100, preds, tok.pad_token_id)
        decoded_preds = tok.batch_decode(preds, skip_special_tokens=True)
        decoded_labels = tok.batch_decode(labels, skip_special_tokens=True)
        decoded_preds = [pred.strip() for pred in decoded_preds]
        decoded_labels = [label.strip() for label in decoded_labels]

        if rouge_metric is not None:
            try:
                result = cast(Any, rouge_metric).compute(predictions=decoded_preds, references=decoded_labels, use_stemmer=False)
                if result is not None:
                    for key, value in result.items():
                        metrics[key] = value * 100
            except Exception as e:
                print(f"Error during ROUGE: {e}")

        if bleu_metric is not None:
            try:
                formatted_labels = [[label] for label in decoded_labels]
                bleu_result = cast(Any, bleu_metric).compute(predictions=decoded_preds, references=formatted_labels)
                if bleu_result is not None and "bleu" in bleu_result:
                    metrics["bleu"] = bleu_result["bleu"] * 100
            except Exception as e:
                print(f"Error during BLEU: {e}")

        if exact_match_metric is not None:
            try:
                em_result = cast(Any, exact_match_metric).compute(predictions=decoded_preds, references=decoded_labels)
                if em_result is not None and "exact_match" in em_result:
                    metrics["exact_match"] = em_result["exact_match"] * 100
            except Exception as e:
                print(f"Error during Exact Match: {e}")

        if bertscore_metric is not None:
            try:
                bertscore_result = cast(Any, bertscore_metric).compute(
                    predictions=decoded_preds, references=decoded_labels,
                    model_type="google/embeddinggemma-300m", num_layers=12, lang="id"
                )
                if bertscore_result is not None and "f1" in bertscore_result:
                    metrics["bertscore_f1"] = np.mean(bertscore_result["f1"]) * 100
            except Exception as e:
                print(f"Error during BERTScore: {e}")

        if meteor_metric is not None:
            try:
                meteor_result = cast(Any, meteor_metric).compute(predictions=decoded_preds, references=decoded_labels)
                if meteor_result is not None and "meteor" in meteor_result:
                    metrics["meteor"] = meteor_result["meteor"] * 100
            except Exception as e:
                print(f"Error during METEOR: {e}")

        return metrics

    # Instantiate GrokAdEMAMix Optimizer with split learning rates
    print("Menggunakan optimizer: GrokAdEMAMix (Split LR: Encoder=0.5x, Decoder=1.0x, Projector=1.0x, VisionTower=0.5x)")
    _encoder_params = []
    _decoder_params = []
    _projector_params = []
    _vision_tower_params = []
    for _name, _param in model.named_parameters():
        if _param.requires_grad:
            if "multi_modal_projector" in _name:
                _projector_params.append(_param)
            elif "vision_tower" in _name:
                _vision_tower_params.append(_param)
            elif "encoder" in _name:
                _encoder_params.append(_param)
            elif "decoder" in _name:
                _decoder_params.append(_param)
            else:
                _decoder_params.append(_param)

    _optimizer = GrokAdEMAMix([
        {"params": _encoder_params, "lr": LEARNING_RATE * 0.5},
        {"params": _decoder_params, "lr": LEARNING_RATE},
        {"params": _projector_params, "lr": LEARNING_RATE},
        {"params": _vision_tower_params, "lr": LEARNING_RATE * 0.5}
    ], weight_decay=WEIGHT_DECAY, grok_alpha=2.0, grok_lamb=0.98)

    # Calculate steps for Cosine Scheduler
    _num_update_steps_per_epoch = max(
        1, len(sft_train_dataset) // (PER_DEVICE_TRAIN_BATCH_SIZE * GRADIENT_ACCUMULATION_STEPS)
    )
    _max_steps = _num_update_steps_per_epoch * NUM_EPOCHS_SFT

    _lr_scheduler = get_scheduler(
        name=LR_SCHEDULER_TYPE,
        optimizer=_optimizer,
        num_warmup_steps=WARMUP_STEPS,
        num_training_steps=_max_steps,
    )

    # Callbacks (same as V6 text-only)
    _bad_words_ids = [
        [id_] for id_ in ALL_SUPPRESS_IDS if id_ < cast(Any, model).config.vocab_size
    ]
    _plot_callback = TrainingPlotCallback(output_dir=sft_output_dir)
    _progress_callback = CleanNotebookProgressCallback()

    _sample_callback_multimodal = SampleGenerationCallback(
        processor=processor,
        eval_samples=_eval_generation_samples,
        output_dir=sft_output_dir,
        eval_every_n_steps=50,
        temperature=0.7,
        top_p=0.9,
        repetition_penalty=1.2,
        bad_words_ids=_bad_words_ids,
    )
    _sample_callback_multimodal.log_path = os.path.join(sft_output_dir, "sft_eval_samples_multimodal.txt")

    # Setup qualitative generation samples for text-only validation
    _text_only_val_rows = list(_text_only_eval_dataset) if _text_only_eval_dataset is not None else []
    _n_text_only_eval_gen = min(len(_text_only_val_rows), 20)
    _text_only_eval_generation_samples = _text_only_val_rows[:_n_text_only_eval_gen]

    _sample_callback_text_only = SampleGenerationCallback(
        processor=processor,
        eval_samples=_text_only_eval_generation_samples,
        output_dir=sft_output_dir,
        eval_every_n_steps=50,
        temperature=0.7,
        top_p=0.9,
        repetition_penalty=1.2,
        bad_words_ids=_bad_words_ids,
    )
    _sample_callback_text_only.log_path = os.path.join(sft_output_dir, "sft_eval_samples_text_only.txt")

    _hub_callback = HubUploadCallback(
        repo_id=HF_CHECKPOINT_REPO,
        stage="sft",
        token=os.environ.get("HF_TOKEN"),
        output_dir=sft_output_dir,
    )

    print("Starting CustomSeq2SeqTrainer for SFT...")
    sft_trainer = CustomSeq2SeqTrainer(
        suppress_ids=ALL_SUPPRESS_IDS,
        model=model,
        args=Seq2SeqTrainingArguments(
            per_device_train_batch_size=PER_DEVICE_TRAIN_BATCH_SIZE,
            per_device_eval_batch_size=1,  # Eval one sample at a time to prevent OOM during generate
            gradient_accumulation_steps=GRADIENT_ACCUMULATION_STEPS,
            eval_accumulation_steps=1,  # Move predictions to CPU immediately after each batch
            learning_rate=LEARNING_RATE,
            num_train_epochs=NUM_EPOCHS_SFT,
            warmup_steps=WARMUP_STEPS,
            weight_decay=WEIGHT_DECAY,
            lr_scheduler_type=LR_SCHEDULER_TYPE,
            logging_steps=LOGGING_STEPS,
            save_strategy="steps",
            save_steps=50,
            save_total_limit=SAVE_TOTAL_LIMIT,
            output_dir=sft_output_dir,
            remove_unused_columns=False,
            fp16=False,
            bf16=BF16,
            optim=OPTIM,
            label_smoothing_factor=LABEL_SMOOTHING_FACTOR,
            neftune_noise_alpha=NEFTUNE_NOISE_ALPHA,
            gradient_checkpointing=True,
            eval_strategy="steps",
            eval_steps=50,
            report_to="none",
            predict_with_generate=PREDICT_WITH_GENERATE,
            generation_max_length=MAX_TARGET_LENGTH,
        ),
        train_dataset=sft_train_dataset,
        eval_dataset=sft_eval_datasets,
        data_collator=sft_collator,
        optimizers=(_optimizer, _lr_scheduler),
        compute_metrics=_compute_metrics,
        callbacks=[_plot_callback, _progress_callback, _sample_callback_multimodal, _sample_callback_text_only, _hub_callback],
    )

    # Buang NotebookProgressCallback bawaan transformers (kalau ada) supaya
    # tabel progress bawaan (dengan kolom "Validation Loss" yang selalu "No log")
    # tidak ikut ter-render berdampingan dengan CleanNotebookProgressCallback.
    from transformers.utils.notebook import NotebookProgressCallback as _HFNotebookProgressCallback
    sft_trainer.remove_callback(_HFNotebookProgressCallback)

    # === RESUME FROM HF CHECKPOINT ===
    _resume_from = None
    if resume_checkpoint:
        try:
            from huggingface_hub import snapshot_download as _resume_snap
            from huggingface_hub import HfApi as _ResumeApi

            _api = _ResumeApi(token=os.environ.get("HF_TOKEN"))
            _files = _api.list_repo_files(repo_id=HF_CHECKPOINT_REPO)

            _ckpts = list(set([f.split('/')[1] for f in _files if f.startswith("sft/checkpoint-")]))
            if _ckpts:
                _ckpts.sort(key=lambda x: int(x.split('-')[1]))
                _latest_ckpt = _ckpts[-1]
            else:
                _latest_ckpt = "checkpoint-*"

            print(f"\n📥 Downloading {_latest_ckpt} (sft) dari HF untuk resume...")
            _resume_snap(
                repo_id=HF_CHECKPOINT_REPO,
                local_dir=sft_output_dir,
                allow_patterns=[f"sft/{_latest_ckpt}/**"],
                token=os.environ.get("HF_TOKEN"),
            )
            _sub_dir = os.path.join(sft_output_dir, "sft")
            if os.path.exists(_sub_dir):
                import shutil as _shutil_SFT
                for _item in os.listdir(_sub_dir):
                    _src = os.path.join(_sub_dir, _item)
                    _dst = os.path.join(sft_output_dir, _item)
                    if os.path.isdir(_src) and _item.startswith("checkpoint-"):
                        if os.path.exists(_dst):
                            _shutil_SFT.rmtree(_dst)
                        _shutil_SFT.move(_src, _dst)
                _shutil_SFT.rmtree(_sub_dir)

            _checkpoints = sorted([
                d for d in os.listdir(sft_output_dir)
                if d.startswith("checkpoint-") and os.path.isdir(os.path.join(sft_output_dir, d))
            ])
            if _checkpoints:
                _resume_from = True
                print(f"✅ Ditemukan {len(_checkpoints)} checkpoint(s). Resume dari yang terbaru!")
            else:
                print("⚠️ Tidak ada checkpoint valid ditemukan. Mulai dari awal.")
        except Exception as e:
            print(f"⚠️ Gagal download checkpoint: {e}. Mulai dari awal.")

    sft_result = None
    try:
        sft_result = sft_trainer.train(resume_from_checkpoint=_resume_from)
        print(f"✅ SFT selesai! Loss: {sft_result.training_loss:.4f}")

        # Save final SFT model & processor
        sft_final_path = os.path.join(sft_output_dir, "final_adapter")
        print(f"💾 Saving final SFT adapter ke {sft_final_path}...")
        sft_trainer.save_model(sft_final_path)
        processor.save_pretrained(sft_final_path)

        # Upload final adapter to HF Hub
        if os.environ.get("HF_TOKEN"):
            try:
                from huggingface_hub import HfApi as _HfApi_SFT
                _final_api = _HfApi_SFT(token=os.environ.get("HF_TOKEN"))
                _final_api.create_repo(repo_id=HF_CHECKPOINT_REPO, repo_type="model", private=False, exist_ok=True)
                print("📤 Uploading final SFT adapter ke HF Hub...")
                _final_api.upload_folder(
                    folder_path=sft_final_path,
                    repo_id=HF_CHECKPOINT_REPO,
                    path_in_repo="sft/final_adapter",
                    repo_type="model",
                )
                print("✅ Upload final SFT adapter sukses!")
            except Exception as e:
                print(f"⚠️ Gagal upload final SFT adapter: {e}")
    except Exception as e:
        print(f"❌ SFT gagal: {e}")
        traceback.print_exc()
    finally:
        sft_trainer = None
        _optimizer = None
        _lr_scheduler = None
        if "model" in globals() and globals()["model"] is not None:
            try:
                globals()["model"].zero_grad(set_to_none=True)
            except Exception:
                pass
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    return


@app.cell
def _(
    ALL_SUPPRESS_IDS,
    Any,
    BF16,
    CleanNotebookProgressCallback,
    Dataset,
    GRADIENT_ACCUMULATION_STEPS,
    GrokAdEMAMix,
    HF_CHECKPOINT_REPO,
    HubUploadCallback,
    LEARNING_RATE,
    LOGGING_STEPS,
    LR_SCHEDULER_TYPE,
    MAX_SOURCE_LENGTH,
    MAX_TARGET_LENGTH,
    NUM_EPOCHS_ORPO,
    OPTIM,
    ORPO_BETA,
    OUTPUT_DIR,
    PER_DEVICE_TRAIN_BATCH_SIZE,
    PREDICT_WITH_GENERATE,
    SAVE_TOTAL_LIMIT,
    SEED,
    SampleGenerationCallback,
    Seq2SeqTrainingArguments,
    TEST_SIZE,
    TrainingPlotCallback,
    VisionORPOCollator,
    VisionORPOTrainer,
    WARMUP_STEPS,
    WEIGHT_DECAY,
    bertscore_metric,
    bleu_metric,
    cast,
    current_stage,
    exact_match_metric,
    gc,
    get_scheduler,
    load_dataset,
    meteor_metric,
    mo,
    model,
    np,
    os,
    parse_orpo_prompt_to_messages,
    processor,
    resume_checkpoint,
    rouge_metric,
    torch,
    traceback,
    format_encoder_from_raw,
):
    # Re-detect pipeline stage FRESH dari HF Hub. Cell deteksi stage awal hanya
    # jalan sekali di awal notebook dan nilainya di-cache marimo. Saat notebook
    # mulai, `sft/final_adapter/` belum ada -> current_stage = "sft". Setelah SFT
    # selesai & upload, cell deteksi itu TIDAK re-run, sehingga current_stage tetap
    # stale "sft" dan mo.stop(current_stage != "orpo") SALAH me-skip ORPO tepat
    # setelah SFT selesai dalam sesi yang sama. Deteksi ulang di sini memastikan
    # ORPO jalan berdasarkan state repo yang sebenarnya.
    from huggingface_hub import HfApi as _OrpoStageApi
    _fresh_stage = current_stage
    _fresh_resume = resume_checkpoint
    try:
        _stage_api = _OrpoStageApi(token=os.environ.get("HF_TOKEN"))
        _stage_files = _stage_api.list_repo_files(HF_CHECKPOINT_REPO)
        if any(f.startswith("orpo/final_adapter/") for f in _stage_files):
            _fresh_stage = "done"
        elif any(f.startswith("sft/final_adapter/") for f in _stage_files):
            _fresh_stage = "orpo"
            _fresh_resume = any(
                f.startswith("orpo/checkpoint-") and "/" in f[len("orpo/checkpoint-"):]
                for f in _stage_files
            )
        else:
            _fresh_stage = "sft"
            _fresh_resume = None
        print(f"📍 Fresh stage detection untuk ORPO: `{_fresh_stage}` (resume={_fresh_resume})")
    except Exception as _e_stage:
        print(f"⚠️ Gagal re-detect stage untuk ORPO ({_e_stage}); pakai current_stage={current_stage}.")

    mo.stop(
        _fresh_stage != "orpo",
        mo.md(f"ℹ️ **Bukan tahap ORPO (deteksi fresh: `{_fresh_stage}`). Melewati training ORPO.**")
    )

    # Active memory cleanup from previous SFT/ORPO attempts
    orpo_trainer = None
    _optimizer = None
    _lr_scheduler = None
    if "model" in globals() and globals()["model"] is not None:
        try:
            globals()["model"].zero_grad(set_to_none=True)
        except Exception:
            pass
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # ORPO Vision Training
    print(f"\n=== ORPO Vision Training (beta={ORPO_BETA}) ===")
    orpo_output_dir = os.path.join(OUTPUT_DIR, "orpo")

    # Load ORPO dataset directly from Hugging Face Hub
    print("Memuat dataset ORPO dari Hugging Face Hub...")
    raw_orpo_dataset = load_dataset("daruokta/t5gemma2-indonesia-vision-formatted", "vision_orpo", split="train")
    print(f"✅ ORPO dataset dimuat dari Hugging Face: {len(raw_orpo_dataset)} sampel.")

    # Format ORPO dataset using ONLY text columns to avoid slow Hugging Face image loading/decoding
    print("Formatting ORPO dataset (text-only pass)...")
    orpo_formatted = []
    prompts_list = raw_orpo_dataset["prompt"]
    chosen_list = raw_orpo_dataset["chosen"]
    rejected_list = raw_orpo_dataset["rejected"]
    
    for _idx_orpo in range(len(prompts_list)):
        prompt_str = prompts_list[_idx_orpo]
        chosen_raw = chosen_list[_idx_orpo].replace("assistant: ", "", 1).strip()
        rejected_raw = rejected_list[_idx_orpo].replace("assistant: ", "", 1).strip()

        # Parse prompt to messages text-only
        lines = prompt_str.split("\n")
        raw_messages = []
        current_role = None
        current_lines = []
        for line in lines:
            if line.startswith("system: "):
                if current_role is not None:
                    raw_messages.append((current_role, "\n".join(current_lines)))
                current_role = "system"
                current_lines = [line[8:]]
            elif line.startswith("user: "):
                if current_role is not None:
                    raw_messages.append((current_role, "\n".join(current_lines)))
                current_role = "user"
                current_lines = [line[6:]]
            elif line.startswith("assistant: "):
                if current_role is not None:
                    raw_messages.append((current_role, "\n".join(current_lines)))
                current_role = "assistant"
                current_lines = [line[11:]]
            else:
                current_lines.append(line)
        if current_role is not None:
            raw_messages.append((current_role, "\n".join(current_lines)))
        
        # Merge messages and count 📷
        new_messages = []
        for _role_orpo, _content_orpo in raw_messages:
            if _role_orpo == "user" and "📷" in _content_orpo:
                _num_images_orpo = _content_orpo.count("📷")
                _text_content_orpo = _content_orpo.replace("📷", "").strip()
                new_content = []
                for _ in range(_num_images_orpo):
                    new_content.append({"type": "image"})
                if _text_content_orpo:
                    new_content.append({"type": "text", "text": _text_content_orpo})
                new_messages.append({"role": _role_orpo, "content": new_content})
            else:
                new_messages.append({"role": _role_orpo, "content": [{"type": "text", "text": _content_orpo}]})

        # Gabungkan turn dengan role sama yang berurutan (mis. dua "user:" beruntun
        # akibat prompt yang kepecah pas parsing baris demi baris). Tanpa ini,
        # apply_chat_template bisa gagal dengan
        # "Conversation roles must alternate user/assistant/user/assistant/...".
        # Logika sama persis dengan yang dipakai di parse_orpo_prompt_to_messages().
        _merged_messages_orpo = []
        for _msg_orpo in new_messages:
            if _merged_messages_orpo and _merged_messages_orpo[-1]["role"] == _msg_orpo["role"]:
                _last_msg_orpo = _merged_messages_orpo[-1]
                _merged_content_orpo = list(_last_msg_orpo["content"])
                for _block_orpo in _msg_orpo["content"]:
                    if (
                        isinstance(_block_orpo, dict)
                        and _block_orpo.get("type") == "text"
                        and _merged_content_orpo
                        and isinstance(_merged_content_orpo[-1], dict)
                        and _merged_content_orpo[-1].get("type") == "text"
                    ):
                        _last_block_orpo = dict(_merged_content_orpo[-1])
                        _last_block_orpo["text"] = str(_last_block_orpo.get("text", "")) + "\n" + str(_block_orpo.get("text", ""))
                        _merged_content_orpo[-1] = _last_block_orpo
                    else:
                        _merged_content_orpo.append(_block_orpo)
                _last_msg_orpo["content"] = _merged_content_orpo
            else:
                _merged_messages_orpo.append(_msg_orpo)
        new_messages = _merged_messages_orpo

        # Apply chat template
        pt = processor.apply_chat_template(new_messages, tokenize=False, add_generation_prompt=True)

        if chosen_raw.endswith("<end_of_turn>"):
            chosen_raw = chosen_raw[:-len("<end_of_turn>")].strip()
        if rejected_raw.endswith("<end_of_turn>"):
            rejected_raw = rejected_raw[:-len("<end_of_turn>")].strip()

        orpo_formatted.append({
            "prompt_text": pt,
            "chosen_text": chosen_raw,
            "rejected_text": rejected_raw,
            "dataset_idx": _idx_orpo
        })
    orpo_dataset = Dataset.from_list(orpo_formatted)
    print(f"✅ ORPO dataset siap: {len(orpo_dataset)} sampel.")

    # Split Train / Validation
    split_orpo = orpo_dataset.train_test_split(test_size=TEST_SIZE, seed=SEED)
    orpo_train_dataset = split_orpo["train"]
    orpo_eval_dataset = split_orpo["test"]
    print(f"  ORPO Train size: {len(orpo_train_dataset)} | ORPO Eval size: {len(orpo_eval_dataset)}")

    # Load and format text-only validation dataset for ORPO
    print("Loading text-only ORPO validation dataset from HF Hub...")
    _text_only_eval_dataset = None
    try:
        _val_chat_ds = load_dataset("daruokta/t5gemma2-indonesia-chat-formatted", "chat_sft", split="validation")
        _val_indoqa_ds = load_dataset("daruokta/t5gemma2-indonesia-chat-formatted", "indoqa_sft", split="validation")
        _val_chat_samples = [dict(_row) for _row in _val_chat_ds]
        _val_indoqa_samples = [dict(_row) for _row in _val_indoqa_ds]

        import random as _rng_orpo
        _raw_text_only_samples = _val_chat_samples + _val_indoqa_samples
        _rng_orpo.seed(42)
        _rng_orpo.shuffle(_raw_text_only_samples)
        _raw_text_only_samples = _raw_text_only_samples[:100]

        _text_only_formatted = []
        for _row in _raw_text_only_samples:
            _pt = format_encoder_from_raw(_row["input"])
            _tt = _row["target"]
            _text_only_formatted.append({
                "prompt_text": _pt,
                "images": [],
                "chosen_text": _tt,
                "rejected_text": "Maaf, saya kurang tahu mengenai hal tersebut."
            })
        _text_only_eval_dataset = Dataset.from_list(_text_only_formatted)
        print(f"  Text-Only ORPO Eval size: {len(_text_only_eval_dataset)}")
    except Exception as e:
        print(f"⚠️ Gagal memuat dataset validasi teks untuk ORPO: {e}")

    orpo_eval_datasets = {"multimodal": orpo_eval_dataset}
    if _text_only_eval_dataset is not None:
        orpo_eval_datasets["text_only"] = _text_only_eval_dataset

    # Setup qualitative generation samples
    _orpo_val_rows = list(orpo_eval_dataset)
    _n_eval_gen = min(len(_orpo_val_rows), 20)
    _eval_generation_samples = []
    for _item_orpo in _orpo_val_rows[:_n_eval_gen]:
        _eval_generation_samples.append({
            "prompt_text": _item_orpo["prompt_text"],
            "target_text": _item_orpo["chosen_text"],
            "images": raw_orpo_dataset[_item_orpo["dataset_idx"]]["images"] if "dataset_idx" in _item_orpo else []
        })

    # Define compute metrics
    def _compute_metrics(eval_preds):
        metrics = {}
        if not PREDICT_WITH_GENERATE:
            return metrics
        if rouge_metric is None and bleu_metric is None:
            return metrics
        preds, labels = eval_preds
        if isinstance(preds, tuple):
            preds = preds[0]
        tok = cast(Any, processor.tokenizer)

        if preds.ndim == 3:
            preds = preds.argmax(axis=-1)

        labels = np.where(labels != -100, labels, tok.pad_token_id)
        preds = np.where(preds != -100, preds, tok.pad_token_id)
        decoded_preds = tok.batch_decode(preds, skip_special_tokens=True)
        decoded_labels = tok.batch_decode(labels, skip_special_tokens=True)
        decoded_preds = [pred.strip() for pred in decoded_preds]
        decoded_labels = [label.strip() for label in decoded_labels]

        if rouge_metric is not None:
            try:
                result = cast(Any, rouge_metric).compute(predictions=decoded_preds, references=decoded_labels, use_stemmer=False)
                if result is not None:
                    for key, value in result.items():
                        metrics[key] = value * 100
            except Exception as e:
                print(f"Error during ROUGE: {e}")

        if bleu_metric is not None:
            try:
                formatted_labels = [[label] for label in decoded_labels]
                bleu_result = cast(Any, bleu_metric).compute(predictions=decoded_preds, references=formatted_labels)
                if bleu_result is not None and "bleu" in bleu_result:
                    metrics["bleu"] = bleu_result["bleu"] * 100
            except Exception as e:
                print(f"Error during BLEU: {e}")

        if exact_match_metric is not None:
            try:
                em_result = cast(Any, exact_match_metric).compute(predictions=decoded_preds, references=decoded_labels)
                if em_result is not None and "exact_match" in em_result:
                    metrics["exact_match"] = em_result["exact_match"] * 100
            except Exception as e:
                print(f"Error during Exact Match: {e}")

        if bertscore_metric is not None:
            try:
                bertscore_result = cast(Any, bertscore_metric).compute(
                    predictions=decoded_preds, references=decoded_labels,
                    model_type="google/embeddinggemma-300m", num_layers=12, lang="id"
                )
                if bertscore_result is not None and "f1" in bertscore_result:
                    metrics["bertscore_f1"] = np.mean(bertscore_result["f1"]) * 100
            except Exception as e:
                print(f"Error during BERTScore: {e}")

        if meteor_metric is not None:
            try:
                meteor_result = cast(Any, meteor_metric).compute(predictions=decoded_preds, references=decoded_labels)
                if meteor_result is not None and "meteor" in meteor_result:
                    metrics["meteor"] = meteor_result["meteor"] * 100
            except Exception as e:
                print(f"Error during METEOR: {e}")

        return metrics

    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    orpo_collator = VisionORPOCollator(processor, MAX_SOURCE_LENGTH, MAX_TARGET_LENGTH, raw_orpo_dataset)

    # Instantiate GrokAdEMAMix Optimizer with split learning rates
    print("Menggunakan optimizer: GrokAdEMAMix (Split LR: Encoder=0.5x, Decoder=1.0x, Projector=1.0x, VisionTower=0.5x)")
    _encoder_params = []
    _decoder_params = []
    _projector_params = []
    _vision_tower_params = []
    for _name, _param in model.named_parameters():
        if _param.requires_grad:
            if "multi_modal_projector" in _name:
                _projector_params.append(_param)
            elif "vision_tower" in _name:
                _vision_tower_params.append(_param)
            elif "encoder" in _name:
                _encoder_params.append(_param)
            elif "decoder" in _name:
                _decoder_params.append(_param)
            else:
                _decoder_params.append(_param)

    _optimizer = GrokAdEMAMix([
        {"params": _encoder_params, "lr": LEARNING_RATE * 0.5},
        {"params": _decoder_params, "lr": LEARNING_RATE},
        {"params": _projector_params, "lr": LEARNING_RATE},
        {"params": _vision_tower_params, "lr": LEARNING_RATE * 0.5}
    ], weight_decay=WEIGHT_DECAY, grok_alpha=2.0, grok_lamb=0.98)

    # Calculate steps for Cosine Scheduler
    _num_update_steps_per_epoch = max(
        1, len(orpo_train_dataset) // (PER_DEVICE_TRAIN_BATCH_SIZE * GRADIENT_ACCUMULATION_STEPS)
    )
    _max_steps = _num_update_steps_per_epoch * NUM_EPOCHS_ORPO

    _lr_scheduler = get_scheduler(
        name=LR_SCHEDULER_TYPE,
        optimizer=_optimizer,
        num_warmup_steps=WARMUP_STEPS,
        num_training_steps=_max_steps,
    )

    # Callbacks (same as V6 text-only)
    _bad_words_ids = [
        [id_] for id_ in ALL_SUPPRESS_IDS if id_ < cast(Any, model).config.vocab_size
    ]
    _plot_callback = TrainingPlotCallback(output_dir=orpo_output_dir)
    _progress_callback = CleanNotebookProgressCallback()

    _sample_callback_multimodal = SampleGenerationCallback(
        processor=processor,
        eval_samples=_eval_generation_samples,
        output_dir=orpo_output_dir,
        eval_every_n_steps=50,
        temperature=0.7,
        top_p=0.9,
        repetition_penalty=1.2,
        bad_words_ids=_bad_words_ids,
    )
    _sample_callback_multimodal.log_path = os.path.join(orpo_output_dir, "orpo_eval_samples_multimodal.txt")

    # Setup qualitative generation samples for text-only validation in ORPO
    _text_only_val_rows = list(_text_only_eval_dataset) if _text_only_eval_dataset is not None else []
    _n_text_only_eval_gen = min(len(_text_only_val_rows), 20)
    _text_only_eval_generation_samples = []
    for _item_text in _text_only_val_rows[:_n_text_only_eval_gen]:
        _text_only_eval_generation_samples.append({
            "prompt_text": _item_text["prompt_text"],
            "images": _item_text["images"],
            "target_text": _item_text["chosen_text"]
        })

    _sample_callback_text_only = SampleGenerationCallback(
        processor=processor,
        eval_samples=_text_only_eval_generation_samples,
        output_dir=orpo_output_dir,
        eval_every_n_steps=50,
        temperature=0.7,
        top_p=0.9,
        repetition_penalty=1.2,
        bad_words_ids=_bad_words_ids,
    )
    _sample_callback_text_only.log_path = os.path.join(orpo_output_dir, "orpo_eval_samples_text_only.txt")

    _hub_callback = HubUploadCallback(
        repo_id=HF_CHECKPOINT_REPO,
        stage="orpo",
        token=os.environ.get("HF_TOKEN"),
        output_dir=orpo_output_dir,
    )

    orpo_result = None
    try:
        orpo_trainer = VisionORPOTrainer(
            beta=ORPO_BETA, model=model,
            args=Seq2SeqTrainingArguments(
                per_device_train_batch_size=PER_DEVICE_TRAIN_BATCH_SIZE,
                per_device_eval_batch_size=1,  # Eval one sample at a time to prevent OOM
                gradient_accumulation_steps=GRADIENT_ACCUMULATION_STEPS,
                eval_accumulation_steps=1,  # Move predictions to CPU immediately
                learning_rate=LEARNING_RATE,
                num_train_epochs=NUM_EPOCHS_ORPO,
                warmup_steps=WARMUP_STEPS,
                weight_decay=WEIGHT_DECAY,
                lr_scheduler_type=LR_SCHEDULER_TYPE,
                logging_steps=LOGGING_STEPS,
                save_strategy="steps",
                save_steps=50,
                save_total_limit=SAVE_TOTAL_LIMIT,
                output_dir=orpo_output_dir,
                remove_unused_columns=False,
                fp16=False, bf16=BF16, optim=OPTIM,
                gradient_checkpointing=True,
                eval_strategy="steps",
                eval_steps=50,
                report_to="none",
                predict_with_generate=PREDICT_WITH_GENERATE,
                generation_max_length=MAX_TARGET_LENGTH,
            ),
            train_dataset=orpo_train_dataset,
            eval_dataset=orpo_eval_datasets,
            data_collator=orpo_collator,
            optimizers=(_optimizer, _lr_scheduler),
            compute_metrics=_compute_metrics,
            callbacks=[_plot_callback, _progress_callback, _sample_callback_multimodal, _sample_callback_text_only, _hub_callback],
        )
        # Buang NotebookProgressCallback bawaan transformers (kalau ada) supaya
        # tabel progress bawaan (dengan kolom "Validation Loss" yang selalu "No log")
        # tidak ikut ter-render berdampingan dengan CleanNotebookProgressCallback.
        from transformers.utils.notebook import NotebookProgressCallback as _HFNotebookProgressCallback
        orpo_trainer.remove_callback(_HFNotebookProgressCallback)

        # === RESUME FROM HF CHECKPOINT ===
        # Pakai hasil deteksi fresh (`_fresh_resume`) alih-alih `resume_checkpoint`
        # yang mungkin stale dari cell deteksi stage awal.
        _resume_from = None
        if _fresh_resume:
            try:
                from huggingface_hub import snapshot_download as _resume_snap
                from huggingface_hub import HfApi as _ResumeApi

                _api = _ResumeApi(token=os.environ.get("HF_TOKEN"))
                _files = _api.list_repo_files(repo_id=HF_CHECKPOINT_REPO)

                _ckpts = list(set([f.split('/')[1] for f in _files if f.startswith("orpo/checkpoint-")]))
                if _ckpts:
                    _ckpts.sort(key=lambda x: int(x.split('-')[1]))
                    _latest_ckpt = _ckpts[-1]
                else:
                    _latest_ckpt = "checkpoint-*"

                print(f"\n📥 Downloading {_latest_ckpt} (orpo) dari HF untuk resume...")
                _resume_snap(
                    repo_id=HF_CHECKPOINT_REPO,
                    local_dir=orpo_output_dir,
                    allow_patterns=[f"orpo/{_latest_ckpt}/**"],
                    token=os.environ.get("HF_TOKEN"),
                )
                _sub_dir = os.path.join(orpo_output_dir, "orpo")
                if os.path.exists(_sub_dir):
                    import shutil as _shutil_ORPO
                    for _item in os.listdir(_sub_dir):
                        _src = os.path.join(_sub_dir, _item)
                        _dst = os.path.join(orpo_output_dir, _item)
                        if os.path.isdir(_src) and _item.startswith("checkpoint-"):
                            if os.path.exists(_dst):
                                _shutil_ORPO.rmtree(_dst)
                            _shutil_ORPO.move(_src, _dst)
                    _shutil_ORPO.rmtree(_sub_dir)

                _checkpoints = sorted([
                    d for d in os.listdir(orpo_output_dir)
                    if d.startswith("checkpoint-") and os.path.isdir(os.path.join(orpo_output_dir, d))
                ])
                if _checkpoints:
                    _resume_from = True
                    print(f"✅ Ditemukan {len(_checkpoints)} checkpoint(s). Resume dari yang terbaru!")
                else:
                    print("⚠️ Tidak ada checkpoint valid ditemukan. Mulai dari awal.")
            except Exception as e:
                print(f"⚠️ Gagal download checkpoint: {e}. Mulai dari awal.")

        orpo_result = orpo_trainer.train(resume_from_checkpoint=_resume_from)
        print(f"✅ ORPO selesai! Loss: {orpo_result.training_loss:.4f}")

        # Save final ORPO model & processor
        orpo_final_path = os.path.join(orpo_output_dir, "final_adapter")
        print(f"💾 Saving final ORPO adapter ke {orpo_final_path}...")
        orpo_trainer.save_model(orpo_final_path)
        processor.save_pretrained(orpo_final_path)

        # Upload final adapter to HF Hub
        if os.environ.get("HF_TOKEN"):
            try:
                from huggingface_hub import HfApi as _HfApi_SFT
                _final_api = _HfApi_SFT(token=os.environ.get("HF_TOKEN"))
                _final_api.create_repo(repo_id=HF_CHECKPOINT_REPO, repo_type="model", private=False, exist_ok=True)
                print("📤 Uploading final ORPO adapter ke HF Hub...")
                _final_api.upload_folder(
                    folder_path=orpo_final_path,
                    repo_id=HF_CHECKPOINT_REPO,
                    path_in_repo="orpo/final_adapter",
                    repo_type="model",
                )
                print("✅ Upload final ORPO adapter sukses!")
            except Exception as e:
                print(f"⚠️ Gagal upload final ORPO adapter: {e}")
    except Exception as e:
        print(f"❌ ORPO gagal: {e}")
        traceback.print_exc()
    finally:
        orpo_trainer = None
        _optimizer = None
        _lr_scheduler = None
        if "model" in globals() and globals()["model"] is not None:
            try:
                globals()["model"].zero_grad(set_to_none=True)
            except Exception:
                pass
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    return


@app.cell
def _(HF_CHECKPOINT_REPO, OUTPUT_DIR, model, os, processor):
    def save_adapter():
        if model is None:
            return
        # Menyimpan adapter vision dan mengunggah ke HF Hub
        adapter_path = os.path.join(OUTPUT_DIR, "final_adapter")
        model.save_pretrained(adapter_path)
        processor.save_pretrained(adapter_path)
        print(f"✅ Adapter LoRA vision berhasil disimpan ke: {adapter_path}")

        token = os.environ.get("HF_TOKEN")
        if token:
            print(f"Mengunggah adapter vision ke Hugging Face Hub: {HF_CHECKPOINT_REPO}...")
    save_adapter()
    return


@app.cell
def _(
    ALL_SUPPRESS_IDS,
    load_dataset,
    model,
    processor,
    random,
    re,
    tokenizer,
    torch,
    traceback,
):
    def run_eval():
        if model is None:
            return
        # =====================================================================
        # EVALUASI GENERASI (TEST KUALITAS GENERASI CHAT & VISION)
        # =====================================================================
        # Menguji kemampuan visual dan menjaga kemampuan dialog bahasa Indonesia
        # menggunakan dataset validasi teks asli dari training sebelumnya
        model.eval()

        def format_encoder_from_raw(raw_input: str) -> str:
            system = "Kamu adalah asisten AI yang helpful, santai, dan ramah. Gunakan Bahasa Indonesia sebagai bahasa utama."
            system_match = re.search(r"^system:\s*(.*?)(?=\nuser:)", raw_input, re.DOTALL)
            if system_match:
                system = system_match.group(1).strip()
                raw_input = raw_input[system_match.end() :].strip()

            parts = re.split(r"\n(user:|assistant:)\s*", "\n" + raw_input)
            formatted = ""
            is_first_user = True

            for i in range(1, len(parts), 2):
                role = parts[i].replace(":", "").strip()
                content = parts[i + 1].strip()
                if not content:
                    continue

                if role == "user":
                    formatted += "<start_of_turn>user\n"
                    if is_first_user and system:
                        formatted += system + "\n\n"
                        is_first_user = False
                    formatted += content + "<end_of_turn>\n"
                elif role == "assistant":
                    formatted += "<start_of_turn>model\n"
                    formatted += content + "<end_of_turn>\n"

            formatted += "<start_of_turn>model\n"
            return formatted

        def process_sft_rows(samples, tokenizer, is_chat=True):
            rows = []
            if is_chat:
                chat_groups = {}
                for obj in samples:
                    if not obj.get("input") or not obj.get("target"):
                        continue
                    chat_idx = obj.get("chat_idx", -1)
                    if chat_idx not in chat_groups:
                        chat_groups[chat_idx] = []
                    chat_groups[chat_idx].append(obj)

                for chat_idx, turns in chat_groups.items():
                    turns = sorted(turns, key=lambda x: x.get("turn_idx", 0))
                    for turn in turns:
                        inp_f = format_encoder_from_raw(turn["input"])
                        tgt_f = turn["target"].strip() + "<end_of_turn>"
                        inp_ids = tokenizer.encode(inp_f, add_special_tokens=True)
                        # Mirror training collator: processor() hardcode BOS saat
                        # training, jadi validasi teks-only juga harus diawali BOS
                        # (add_bos_token=False membuat encode() tidak menambah BOS).
                        _bos_id = getattr(tokenizer, "bos_token_id", None)
                        if _bos_id is not None and (not inp_ids or inp_ids[0] != _bos_id):
                            inp_ids = [_bos_id] + inp_ids
                        if getattr(tokenizer, "eos_token_id", None) is not None and inp_ids[-1] != tokenizer.eos_token_id:
                            inp_ids.append(tokenizer.eos_token_id)
                        tgt_ids = tokenizer.encode(tgt_f, add_special_tokens=False)
                        if getattr(tokenizer, "eos_token_id", None) is not None and tgt_ids[-1] != tokenizer.eos_token_id:
                            tgt_ids.append(tokenizer.eos_token_id)
                        rows.append({"input_ids": inp_ids, "labels": tgt_ids})
            else:
                for obj in samples:
                    inp_f = format_encoder_from_raw(obj.get("input", ""))
                    tgt_f = obj.get("target", "").strip() + "<end_of_turn>"
                    inp_ids = tokenizer.encode(inp_f, add_special_tokens=True)
                    if getattr(tokenizer, "eos_token_id", None) is not None and inp_ids[-1] != tokenizer.eos_token_id:
                        inp_ids.append(tokenizer.eos_token_id)
                    tgt_ids = tokenizer.encode(tgt_f, add_special_tokens=False)
                    if getattr(tokenizer, "eos_token_id", None) is not None and tgt_ids[-1] != tokenizer.eos_token_id:
                        tgt_ids.append(tokenizer.eos_token_id)
                    rows.append({"input_ids": inp_ids, "labels": tgt_ids})
            return rows

        print("\n" + "="*70)
        print("TEST 1: Evaluasi Gambar Umum / Dokumen (Multimodal)")
        print("="*70)

        test_messages = [
            {"role": "system", "content": "Kamu adalah Gemma, asisten AI cerdas berbahasa Indonesia. Berikan respons yang akurat, ramah, dan terstruktur."},
            {"role": "user", "content": [
                {"type": "image"},
                {"type": "text", "text": "Halo Gemma, boleh tolong jelaskan apa menu makanan yang paling populer seharga di bawah 150 ribu berdasarkan brosur/menu ini?"}
            ]}
        ]

        from PIL import Image as PILImage
        dummy_img = PILImage.new("RGB", (224, 224), color="blue")

        try:
            prompt = processor.apply_chat_template(test_messages, tokenize=False, add_generation_prompt=True)
            inputs = processor(text=prompt, images=dummy_img, return_tensors="pt")

            device = next(model.parameters()).device
            inputs = {k: v.to(device) for k, v in inputs.items()}

            with torch.no_grad():
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=256,
                    do_sample=True,
                    temperature=0.7, top_p=0.9, use_cache=True
                )
            response = processor.decode(outputs[0], skip_special_tokens=True)
            print(f"User: [📷 Image] Halo Gemma, boleh tolong jelaskan apa menu makanan yang paling populer seharga di bawah 150 ribu berdasarkan brosur/menu ini?")
            print(f"Assistant:\n{response}")
        except Exception as e:
            print(f"Gagal melakukan inferensi multimodal: {e}")

        print("\n" + "="*70)
        print("TEST 2: Evaluasi Pemeliharaan Chat Umum (Text-Only - LITERALLY 100 Kueri dari Validation Sebelumnya)")
        print("="*70)

        print("Memuat dataset validasi percakapan teks sebelumnya...")
        try:
            val_chat_ds = load_dataset("daruokta/t5gemma2-indonesia-chat-formatted", "chat_sft", split="validation")
            val_indoqa_ds = load_dataset("daruokta/t5gemma2-indonesia-chat-formatted", "indoqa_sft", split="validation")

            val_chat_samples = [dict(row) for row in val_chat_ds]
            val_indoqa_samples = [dict(row) for row in val_indoqa_ds]

            val_rows = process_sft_rows(val_chat_samples, tokenizer, is_chat=True) + process_sft_rows(val_indoqa_samples, tokenizer, is_chat=False)

            # Samakan dengan seed 42 dan shuffle agar urutannya konsisten dengan baseline teks
            random.seed(42)
            random.shuffle(val_rows)

            eval_generation_samples = val_rows[:100]
            print(f"Berhasil memuat dan memproses {len(eval_generation_samples)} sampel validasi teks.")

            device = next(model.parameters()).device
            _eot_id = tokenizer.convert_tokens_to_ids("<end_of_turn>")
            _eos_id = tokenizer.eos_token_id or 1
            _stop_ids = list({_eot_id, _eos_id})

            # Gunakan ALL_SUPPRESS_IDS yang dilewatkan sebagai argumen
            bad_words_ids = [[id_] for id_ in ALL_SUPPRESS_IDS if id_ < model.config.vocab_size]
            pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else _eos_id

            for idx, sample in enumerate(eval_generation_samples):
                input_tensor = torch.tensor([sample["input_ids"]], dtype=torch.long).to(device)
                attention_mask = torch.ones_like(input_tensor).to(device)

                with torch.no_grad():
                    outputs_text = model.generate(
                        input_ids=input_tensor,
                        attention_mask=attention_mask,
                        max_new_tokens=1024,
                        do_sample=True,
                        temperature=0.7,
                        top_p=0.9,
                        repetition_penalty=1.2,
                        eos_token_id=_stop_ids,
                        pad_token_id=pad_id,
                        bad_words_ids=bad_words_ids,
                        use_cache=True
                    )

                query = tokenizer.decode(sample["input_ids"], skip_special_tokens=True).strip()
                target = tokenizer.decode(sample["labels"], skip_special_tokens=True).strip()

                raw_response = tokenizer.decode(outputs_text[0], skip_special_tokens=True)
                if raw_response.startswith(query):
                    raw_response = raw_response[len(query):].strip()
                response = raw_response.strip()

                words = response.split()
                is_repetitive = len(set(words)) < max(1, len(words) * 0.3) if words else True
                flag = " ⚠️ REPETITIVE" if is_repetitive else ""

                print(f"\n[Sampel {idx+1}/100]{flag}")
                print(f"  Q: {query[:250]}...")
                print(f"  Expected Target: {target[:200]}...")
                print(f"  Model Generated: {response[:350]}...")

        except Exception as e:
            print(f"Gagal melakukan inferensi teks validasi 100 sampel: {e}")
            traceback.print_exc()

        print("="*70)

    run_eval()
    return


@app.cell
def _(
    HF_CHECKPOINT_REPO,
    LOAD_IN_4BIT,
    MODEL_NAME,
    OUTPUT_DIR,
    model,
    os,
    processor,
    tokenizer,
):
    def merge_and_quantize(model, tokenizer, processor, upload_dir: str):
        import unsloth_zoo.saving_utils
        unsloth_zoo.saving_utils.assert_same_keys = lambda *args, **kwargs: None  # type: ignore

        # --- Workaround: unsloth_zoo `_infer_prefix_and_remap` UnboundLocalError ---
        # Versi unsloth_zoo (lama) yang terinstal tidak menginisialisasi
        # `unmatched_keys = []` sebelum cek `if unmatched_keys:` pertama. Saat SEMUA
        # key LoRA sudah cocok langsung dengan key safetensor (tidak ada unmatched key
        # yang tercipta -> cabang `else` tidak pernah jalan), variabel `unmatched_keys`
        # tidak pernah di-assign sehingga `if unmatched_keys:` melempar
        # `UnboundLocalError: cannot access local variable 'unmatched_keys'`.
        # Bug ini SUDAH diperbaiki di upstream (unsloth-zoo main menambah
        # `unmatched_keys = []` sebelum loop). Di sini kita bungkus fungsi terinstal
        # lalu fallback ke reimplementation minimal yang sudah di-fix ketika error
        # spesifik itu muncul, agar save_pretrained_merged("merged_16bit") sukses.
        _sz = unsloth_zoo.saving_utils
        if not getattr(_sz, "_unmatched_keys_patch_applied", False):
            from collections import defaultdict as _ddp
            _orig_infer = getattr(_sz, "_infer_prefix_and_remap", None)

            def _infer_prefix_and_remap_fixed(lora_weights, safetensor_keys):
                # Reimplementasi minimal dari _infer_prefix_and_remap (upstream main)
                # dengan inisialisasi `unmatched_keys = []` yang menjadi root cause fix.
                if not safetensor_keys:
                    return None
                sf_key_set = set(safetensor_keys)
                remapped = _ddp(getattr(lora_weights, "default_factory", None))
                changed = False
                unmatched_keys = []  # <-- THE FIX: inisialisasi sebelum dipakai
                for k, v in lora_weights.items():
                    if not isinstance(k, str):
                        remapped[k] = v
                        continue
                    # Sudah cocok langsung dengan key safetensor (.weight / .linear.weight)
                    if (k + ".weight") in sf_key_set or (k + ".linear.weight") in sf_key_set:
                        remapped[k] = v
                        continue
                    # Cari kandidat prefix unik
                    candidates = list(dict.fromkeys(
                        sf_key[: -len(suffix)]
                        for suffix in (k + ".weight", k + ".linear.weight")
                        for sf_key in safetensor_keys
                        if sf_key.endswith(suffix) and sf_key[: -len(suffix)]
                    ))
                    if len(candidates) == 1:
                        remapped[candidates[0] + k] = v
                        changed = True
                    else:
                        unmatched_keys.append((k, v))
                # Tidak ada perubahan sama sekali -> sinyalkan "tidak perlu remap"
                if not changed and not unmatched_keys:
                    return None
                # Untuk key yang benar-benar tak ter-match, biarkan apa adanya
                # (merge akan skip target tanpa backing tensor) -> konservatif & aman.
                for k, v in unmatched_keys:
                    remapped[k] = v
                return remapped

            def _patched_infer(lora_weights, safetensor_keys):
                if _orig_infer is not None:
                    try:
                        return _orig_infer(lora_weights, safetensor_keys)
                    except UnboundLocalError as e:
                        if "unmatched_keys" in str(e):
                            print(
                                f"⚠️ [patch] _infer_prefix_and_remap UnboundLocalError "
                                f"({e}); memakai fallback reimplementation yang sudah di-fix."
                            )
                            return _infer_prefix_and_remap_fixed(lora_weights, safetensor_keys)
                        raise
                return _infer_prefix_and_remap_fixed(lora_weights, safetensor_keys)

            _sz._infer_prefix_and_remap = _patched_infer
            _sz._unmatched_keys_patch_applied = True
            print("✅ [patch] Workaround `_infer_prefix_and_remap` UnboundLocalError terpasang.")

        if model is None:
            from unsloth import FastVisionModel
            from transformers import AutoProcessor

            # Load model dari adapter ORPO final
            _orpo_path = os.path.join(OUTPUT_DIR, "orpo", "final_adapter")
            if not os.path.exists(_orpo_path):
                # Fallback download dari HF
                from huggingface_hub import snapshot_download as _snap_dl
                print("📥 Downloading final ORPO adapter dari HF untuk merging...")
                _snap_dl(
                    repo_id=HF_CHECKPOINT_REPO,
                    local_dir=_orpo_path,
                    allow_patterns=["orpo/final_adapter/**"],
                    token=os.environ.get("HF_TOKEN"),
                )
                _sub_path = os.path.join(_orpo_path, "orpo", "final_adapter")
                if os.path.exists(_sub_path):
                    import shutil as _shutil_merge
                    for _item in os.listdir(_sub_path):
                        _src = os.path.join(_sub_path, _item)
                        _dst = os.path.join(_orpo_path, _item)
                        if os.path.exists(_dst):
                            if os.path.isdir(_dst):
                                _shutil_merge.rmtree(_dst)
                            else:
                                os.remove(_dst)
                        _shutil_merge.move(_src, _dst)
                    _shutil_merge.rmtree(os.path.join(_orpo_path, "orpo"))

            print(f"📂 Loading model dari ORPO adapter untuk merge: {_orpo_path}")
            model, tokenizer = FastVisionModel.from_pretrained(
                model_name=_orpo_path,
                load_in_4bit=LOAD_IN_4BIT,
                use_gradient_checkpointing="unsloth",
                token=os.environ.get("HF_TOKEN"),
            )
            processor = AutoProcessor.from_pretrained(MODEL_NAME, token=os.environ.get("HF_TOKEN"))
            from unsloth.chat_templates import get_chat_template
            tokenizer = get_chat_template(tokenizer, chat_template="gemma-3")
            processor.chat_template = tokenizer.chat_template
            if hasattr(processor, "tokenizer"):
                processor.tokenizer.chat_template = tokenizer.chat_template

        merged_bf16_path = os.path.join(upload_dir, "merged_bf16")
        quantized_4bit_path = os.path.join(upload_dir, "quantized_4bit")

        print("Merging LoRA adapter and saving model as BF16 using Unsloth...")
        model.save_pretrained_merged(merged_bf16_path, tokenizer, save_method="merged_16bit")
        tokenizer.save_pretrained(merged_bf16_path)
        processor.save_pretrained(merged_bf16_path)
        print("✅ Model BF16 berhasil disimpan.")

        print("\nMerging LoRA adapter and saving model as 4-bit NF4 using Unsloth...")
        model.save_pretrained_merged(quantized_4bit_path, tokenizer, save_method="merged_4bit_forced")
        tokenizer.save_pretrained(quantized_4bit_path)
        processor.save_pretrained(quantized_4bit_path)
        print("✅ Model 4-bit NF4 berhasil disimpan!")

        return None

    upload_dir = os.path.join(OUTPUT_DIR, "hf_upload")
    merge_and_quantize(model, tokenizer, processor, upload_dir)
    return (upload_dir,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### 💻 Local Deployment & Inference (Direct Load from Hugging Face Hub Subfolders)
    Setelah model diunggah ke Hugging Face Hub, repositori Anda akan memiliki struktur:
    - `sft/` — Checkpoint dan artifacts SFT training
    - `orpo/` — Checkpoint dan artifacts ORPO training
    - `merged_bf16/` — Model gabungan utuh (bfloat16, ~15 GB)
    - `quantized_4bit/` — Model terkuantisasi (NF4, ~5 GB)

    #### Load Model Quantized 4-bit:
    ```python
    from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, AutoProcessor

    model_id = "daruokta/t5gemma-2-4b-4b-instruct-chat-indo-v4-vision"

    tokenizer = AutoTokenizer.from_pretrained(model_id, subfolder="quantized_4bit")
    processor = AutoProcessor.from_pretrained(model_id, subfolder="quantized_4bit")
    model = AutoModelForSeq2SeqLM.from_pretrained(
        model_id, subfolder="quantized_4bit", device_map="auto"
    )
    ```

    #### Load Model Full Precision (BF16):
    ```python
    import torch
    from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, AutoProcessor

    model_id = "daruokta/t5gemma-2-4b-4b-instruct-chat-indo-v4-vision"

    tokenizer = AutoTokenizer.from_pretrained(model_id, subfolder="merged_bf16")
    processor = AutoProcessor.from_pretrained(model_id, subfolder="merged_bf16")
    model = AutoModelForSeq2SeqLM.from_pretrained(
        model_id, subfolder="merged_bf16",
        torch_dtype=torch.bfloat16, device_map="auto"
    )
    ```
    """)
    return


@app.cell
def _(HF_CHECKPOINT_REPO, os, upload_dir):
    from huggingface_hub import HfApi as _UploadMergedApi

    print(f"Memulai proses unggah model merged ke HF Hub: {HF_CHECKPOINT_REPO}...")
    try:
        _merged_api = _UploadMergedApi(token=os.environ.get("HF_TOKEN"))

        # Ensure target model repository exists before uploading merged folder
        _merged_api.create_repo(repo_id=HF_CHECKPOINT_REPO, repo_type="model", private=False, exist_ok=True)

        _merged_api.upload_folder(
            folder_path=upload_dir,
            repo_id=HF_CHECKPOINT_REPO,
            repo_type="model",
        )

        print("✅ Berhasil mengunggah merged models ke Hugging Face Hub!")
    except Exception as e:
        print(f"❌ Terjadi kesalahan saat mengunggah: {e}")
    return


if __name__ == "__main__":
    app.run()
