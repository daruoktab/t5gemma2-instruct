"""
DPO Training: T5Gemma-2 270M Seq2Seq (Light Version)
======================================================
Melatih model preferensi DPO menggunakan DPOTrainer dari TRL.
Memuat adapter hasil dari SFT Light sebagai basis awal (SFT reference).
"""

import os
import re
import json
import torch
import random
import datetime
from datasets import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSeq2SeqLM,
    PreTrainedTokenizerFast,
    TrainerCallback,
    TrainerControl,
    TrainerState,
    TrainingArguments,
)
from peft import LoraConfig, PeftModel, TaskType, get_peft_model
from trl import DPOTrainer, DPOConfig
from trl.trainer.utils import use_adapter, selective_log_softmax
from trl.models.utils import disable_gradient_checkpointing
import torch.nn.functional as F
from typing import cast, Any
from transformers import PreTrainedTokenizerBase

def split_seq2seq_batch(inputs, pad_token_id=0):
    input_ids = inputs["input_ids"]
    attention_mask = inputs["attention_mask"]
    completion_mask = inputs["completion_mask"]
    
    batch_size = input_ids.shape[0]
    
    prompts_list = []
    prompts_mask_list = []
    completions_list = []
    completions_mask_list = []
    
    for i in range(batch_size):
        prompt_idx = (completion_mask[i] == 0).nonzero(as_tuple=True)[0]
        prompt_ids_i = input_ids[i][prompt_idx]
        prompt_mask_i = attention_mask[i][prompt_idx]
        
        completion_idx = (completion_mask[i] == 1).nonzero(as_tuple=True)[0]
        completion_ids_i = input_ids[i][completion_idx]
        completion_mask_i = attention_mask[i][completion_idx]
        
        prompts_list.append(prompt_ids_i)
        prompts_mask_list.append(prompt_mask_i)
        completions_list.append(completion_ids_i)
        completions_mask_list.append(completion_mask_i)
        
    max_prompt_len = max(len(p) for p in prompts_list)
    max_completion_len = max(len(c) for c in completions_list)
    
    padded_prompts = []
    padded_prompts_mask = []
    padded_completions = []
    padded_completions_mask = []
    
    for i in range(batch_size):
        p_len = len(prompts_list[i])
        c_len = len(completions_list[i])
        
        p_pad = torch.full((max_prompt_len - p_len,), pad_token_id, dtype=torch.long, device=input_ids.device)
        pm_pad = torch.zeros((max_prompt_len - p_len,), dtype=torch.long, device=input_ids.device)
        padded_prompts.append(torch.cat([prompts_list[i], p_pad]))
        padded_prompts_mask.append(torch.cat([prompts_mask_list[i], pm_pad]))
        
        c_pad = torch.full((max_completion_len - c_len,), pad_token_id, dtype=torch.long, device=input_ids.device)
        cm_pad = torch.zeros((max_completion_len - c_len,), dtype=torch.long, device=input_ids.device)
        padded_completions.append(torch.cat([completions_list[i], c_pad]))
        padded_completions_mask.append(torch.cat([completions_mask_list[i], cm_pad]))
        
    return {
        "input_ids": torch.stack(padded_prompts),
        "attention_mask": torch.stack(padded_prompts_mask),
        "decoder_input_ids": torch.stack(padded_completions),
        "decoder_attention_mask": torch.stack(padded_completions_mask),
    }

class Seq2SeqDPOTrainer(DPOTrainer):
    def _compute_loss(self, model, inputs, return_outputs):
        if self.processing_class is None:
            raise ValueError("processing_class (tokenizer) is required for Seq2SeqDPOTrainer")
        tokenizer = cast(PreTrainedTokenizerBase, self.processing_class)
        pad_token_id = tokenizer.pad_token_id
        
        # 1. Split batch secara dinamis menjadi encoder input dan decoder input
        split_kwargs = split_seq2seq_batch(inputs, pad_token_id=pad_token_id)
        
        model_kwargs = {
            "input_ids": split_kwargs["input_ids"],
            "attention_mask": split_kwargs["attention_mask"],
            "decoder_input_ids": split_kwargs["decoder_input_ids"],
            "decoder_attention_mask": split_kwargs["decoder_attention_mask"],
            "use_cache": False,
        }
        
        # 2. Forward pass model target
        outputs = model(**model_kwargs)
        
        # 3. Hitung log prob untuk model target
        logits = outputs.logits
        labels = split_kwargs["decoder_input_ids"]
        
        logits_shift = logits[..., :-1, :].contiguous()
        labels_shift = labels[..., 1:].contiguous()
        
        per_token_logps = selective_log_softmax(logits_shift, labels_shift)
        completion_mask = (labels_shift != pad_token_id).long()
        
        # --- Token Suppression & Priority (Hard Masking) ---
        # Mengabaikan token-token yang di-suppress agar tidak mempengaruhi DPO loss (Unlearning)
        for suppress_id in ALL_SUPPRESS_IDS:
            completion_mask[labels_shift == suppress_id] = 0
        # ---------------------------------------------------
        
        # 4. Hitung log prob untuk model referensi (frozen / disabled adapter)
        if self.precompute_ref_logps:
            # Jika precompute, kita hanya bisa pakai hard masking di atas
            per_token_logps[completion_mask == 0] = 0.0
            logps = per_token_logps.sum(dim=1)
            chosen_logps, rejected_logps = logps.chunk(2, dim=0)
            ref_chosen_logps, ref_rejected_logps = inputs["ref_chosen_logps"], inputs["ref_rejected_logps"]
        else:
            if self.model is None:
                raise ValueError("self.model is None")
            model_to_disable = cast(torch.nn.Module, self.model)
            gradient_checkpointing_kwargs = cast(Any, getattr(self.args, "gradient_checkpointing_kwargs", None))
            
            with torch.no_grad(), disable_gradient_checkpointing(model_to_disable, gradient_checkpointing_kwargs):
                if isinstance(model, PeftModel) and self.ref_model is None:
                    model_unwrapped = self.accelerator.unwrap_model(model)
                    if isinstance(model_unwrapped, PeftModel):
                        peft_config = getattr(model_unwrapped, "peft_config", {})
                        ref_adapter = "ref" if "ref" in peft_config else None
                        with cast(Any, use_adapter)(model_unwrapped, adapter_name=ref_adapter):
                            ref_outputs = cast(Any, self.model)(**model_kwargs)
                    else:
                        ref_outputs = cast(Any, self.model)(**model_kwargs)
                else:
                    if self.ref_model is None:
                        raise ValueError("ref_model is None but model is not PeftModel")
                    ref_outputs = cast(Any, self.ref_model)(**model_kwargs)
            
            ref_logits = ref_outputs.logits
            ref_logits_shift = ref_logits[..., :-1, :].contiguous()
            ref_per_token_logps = selective_log_softmax(ref_logits_shift, labels_shift)
            
            # --- Exact Token Cleaning & Token Priority Formula ---
            # Paper Token Cleaning mendefinisikan: Score = L_base - L_ref
            # Karena L = -logps, maka Score = ref_per_token_logps - per_token_logps
            # Paper Token Priority (Rho-1) menyarankan: Pertahankan token jika L_base > L_ref (Score > 0)
            token_score = ref_per_token_logps - per_token_logps
            
            dynamic_mask = completion_mask.clone()
            
            # 1. Token Cleaning / Priority (Noise Filtration)
            # Abaikan token yang uninformative atau noise (di mana base model jauh lebih overconfident dari ref model)
            # Kita filter token dengan score < 0 (atau threshold negatif kecil agar tidak terlalu agresif)
            dynamic_mask[token_score < -0.5] = 0  
            
            # 2. TS-PEFT Proxy (Redundancy / Easy Token Filter)
            # Mencegah 'Gradient Starvation' pada token yang sangat mudah (prob > 0.99)
            # Alih-alih membuat layer LoRA sparsified secara struktural (karena butuh modifikasi arsitektur), 
            # kita simulasikan dengan tidak memberikan update gradien pada token redundan tersebut.
            dynamic_mask[ref_per_token_logps > -0.01] = 0
            
            per_token_logps[dynamic_mask == 0] = 0.0
            ref_per_token_logps[dynamic_mask == 0] = 0.0
            # ----------------------------------------
            
            logps = per_token_logps.sum(dim=1)
            chosen_logps, rejected_logps = logps.chunk(2, dim=0)
            
            ref_logps = ref_per_token_logps.sum(dim=1)
            ref_chosen_logps, ref_rejected_logps = ref_logps.chunk(2, dim=0)
            
        # 5. Hitung DPO Loss
        chosen_logratios = chosen_logps - ref_chosen_logps
        rejected_logratios = rejected_logps - ref_rejected_logps
        delta_score = chosen_logratios - rejected_logratios
        
        loss = -F.logsigmoid(self.beta * delta_score).mean()
        
        # 6. Catat log metrics
        is_training = self.model.training if self.model is not None else False
        mode = "train" if is_training else "eval"
        self._metrics[mode]["loss"].append(loss.item())
        self._metrics[mode]["rewards/chosen"].append((self.beta * chosen_logratios.detach()).mean().item())
        self._metrics[mode]["rewards/rejected"].append((self.beta * rejected_logratios.detach()).mean().item())
        self._metrics[mode]["rewards/margins"].append((self.beta * delta_score.detach()).mean().item())
        self._metrics[mode]["rewards/accuracies"].append((chosen_logratios > rejected_logratios).float().mean().item())
        self._metrics[mode]["logps/chosen"].append(chosen_logps.detach().mean().item())
        self._metrics[mode]["logps/rejected"].append(rejected_logps.detach().mean().item())
        
        return (loss, outputs) if return_outputs else loss

# ==========================================
# KONFIGURASI HYPERPARAMETER
# ==========================================
MODEL_NAME = "google/t5gemma-2-270m-270m"
SFT_ADAPTER_DIR = "results/t5gemma2-270m-clean-sft/final_adapter"
OUTPUT_DIR = "results/t5gemma2-270m-dpo"

DPO_DATA_FILE = "data/preferences_dpo_light.jsonl"

MAX_SOURCE_LENGTH = 2048
MAX_TARGET_LENGTH = 1024

SUPPRESS_BLOCK1 = list(range(6, 105))             # <unused0>–<unused98>
SUPPRESS_BLOCK2 = list(range(256002, 262144))     # <unused100>–<unused6241>
SUPPRESS_VISION = [255999, 256000, 256001]        # <end_of_image>, <image_soft_token>
ALL_SUPPRESS_IDS = set(SUPPRESS_BLOCK1 + SUPPRESS_BLOCK2 + SUPPRESS_VISION)

SYSTEM_PROMPT = (
    "Kamu adalah asisten AI yang helpful, santai, dan ramah. "
    "Gunakan Bahasa Indonesia sebagai bahasa utama."
)

EVAL_QUERIES = [
    "Siapa presiden Indonesia pertama?",
    "Jelaskan apa itu fotosintesis dengan bahasa sederhana.",
    "Apa perbedaan antara simile dan metafora?",
]

def format_encoder_from_raw(raw_input: str) -> str:
    system_match = re.search(r'^system:\s*(.*?)(?=\nuser:)', raw_input, re.DOTALL)
    system = system_match.group(1).strip() if system_match else SYSTEM_PROMPT

    if system_match:
        raw_input = raw_input[system_match.end():].strip()

    parts = re.split(r'\n(user:|assistant:)\s*', '\n' + raw_input)
    formatted = ''
    is_first_user = True

    for i in range(1, len(parts), 2):
        role = parts[i].replace(':', '').strip()
        content = parts[i + 1].strip()
        if not content:
            continue

        if role == 'user':
            formatted += '<start_of_turn>user\n'
            if is_first_user and system:
                formatted += system + '\n\n'
                is_first_user = False
            formatted += content + '<end_of_turn>\n'
        elif role == 'assistant':
            formatted += '<start_of_turn>model\n'
            formatted += content + '<end_of_turn>\n'

    formatted += '<start_of_turn>model\n'
    return formatted

def suppress_unused_tokens(model: Any, suppress_ids: set[int]) -> None:
    suppress_list = sorted(suppress_ids)

    with torch.no_grad():
        embed_weight = model.get_input_embeddings().weight
        vocab_size = embed_weight.shape[0]

        valid_suppress = [i for i in suppress_list if i < vocab_size]
        valid_ids = [i for i in range(vocab_size) if i not in suppress_ids]
        valid_embeds = embed_weight[valid_ids]
        mean_val = valid_embeds.mean(dim=0)

        hidden_size = embed_weight.shape[1]
        noise = torch.randn(
            (len(valid_suppress), hidden_size), device=embed_weight.device
        ) * 0.001
        new_embeds = mean_val.unsqueeze(0) + noise

        suppress_tensor_valid = torch.tensor(valid_suppress, dtype=torch.long)
        embed_weight[suppress_tensor_valid] = new_embeds.to(embed_weight.dtype)
        print(f"  Re-initialized {len(valid_suppress)} suppressed tokens.")

    def freeze_suppressed_hook(grad: torch.Tensor) -> torch.Tensor:
        grad = grad.clone()
        grad[suppress_tensor_valid] = 0.0
        return grad

    if model.get_input_embeddings().weight.requires_grad:
        model.get_input_embeddings().weight.register_hook(freeze_suppressed_hook)
        print(f"  Gradient hook registered untuk {len(valid_suppress)} suppressed tokens.")
    else:
        print("  Gradient hook skipped karena input embeddings dibekukan (requires_grad=False).")

class SampleGenerationCallback(TrainerCallback):
    def __init__(
        self,
        tokenizer: PreTrainedTokenizerFast,
        queries: list[str],
        output_dir: str,
        eval_every_n_steps: int = 50,
    ) -> None:
        self.tokenizer = tokenizer
        self.queries = queries
        self.output_dir = output_dir
        self.eval_every_n_steps = eval_every_n_steps
        self.log_path = os.path.join(output_dir, "eval_samples.txt")
        self._eot_id = tokenizer.convert_tokens_to_ids("<end_of_turn>")
        self._eos_id = tokenizer.eos_token_id or 1
        self._stop_ids = list({self._eot_id, self._eos_id})

    def on_step_end(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        model: Any = None,
        **kwargs: Any,
    ) -> None:
        if state.global_step == 0 or state.global_step % self.eval_every_n_steps != 0:
            return
        if model is None:
            return

        model.eval()
        timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        lines = [
            f"\n{'='*60}",
            f"Step {state.global_step} | {timestamp}",
            f"{'='*60}",
        ]

        with torch.no_grad():
            for q in self.queries:
                prompt = (
                    f"<start_of_turn>user\n{SYSTEM_PROMPT}\n\n{q}<end_of_turn>\n"
                    f"<start_of_turn>model\n"
                )
                enc = self.tokenizer(
                    prompt,
                    return_tensors="pt",
                    add_special_tokens=False,  # Jangan prepend BOS di encoder
                    truncation=True,
                    max_length=512,
                )
                enc = {k: v.to(model.device) for k, v in enc.items()}

                out = getattr(model, "generate")(
                    **enc,
                    max_new_tokens=128,
                    do_sample=False,
                    eos_token_id=self._stop_ids,
                    bad_words_ids=[[i] for i in sorted(ALL_SUPPRESS_IDS)[:50]],
                )
                _resp_raw = self.tokenizer.decode(out[0], skip_special_tokens=True)
                response: str = _resp_raw if isinstance(_resp_raw, str) else " ".join(_resp_raw)

                words = response.split()
                is_repetitive = len(set(words)) < max(1, len(words) * 0.3) if words else True
                flag = " ⚠️ REPETITIVE" if is_repetitive else " ✅"

                lines.append(f"\nQ: {q}")
                lines.append(f"A: {response[:300]}{flag}")

        model.train()
        with open(self.log_path, "a", encoding="utf-8") as f:
            f.write("\n".join(lines) + "\n")

        print(f"\n[DPO BEHAVIOR EVAL @ step {state.global_step}]")
        for line in lines[3:]:
            if line.startswith("Q:") or line.startswith("A:"):
                print(f"  {line}")

def main() -> None:
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print(f"Loading Tokenizer from {MODEL_NAME}...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    assert isinstance(tokenizer, PreTrainedTokenizerFast), "Tokenizer harus PreTrainedTokenizerFast"

    # Muat DPO dataset
    print(f"\nLoading DPO dataset from {DPO_DATA_FILE}...")
    if not os.path.exists(DPO_DATA_FILE):
        raise FileNotFoundError(f"Dataset DPO {DPO_DATA_FILE} tidak ditemukan! Jalankan generate_synthetic_dpo.py dulu.")
        
    dpo_samples = []
    with open(DPO_DATA_FILE, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                dpo_samples.append(json.loads(line))
                
    print(f"Loaded {len(dpo_samples)} DPO samples.")

    # Format dataset kolom prompt, chosen, dan rejected
    formatted_samples = []
    for sample in dpo_samples:
        prompt_f = format_encoder_from_raw(sample["input"])
        chosen_f = sample["chosen"].strip() + "<end_of_turn>"
        rejected_f = sample["rejected"].strip() + "<end_of_turn>"
        
        formatted_samples.append({
            "prompt": prompt_f,
            "chosen": chosen_f,
            "rejected": rejected_f
        })

    # Train / Eval split 95/5
    random.seed(42)
    random.shuffle(formatted_samples)
    split_idx = max(1, int(len(formatted_samples) * 0.95))
    train_samples = formatted_samples[:split_idx]
    eval_samples = formatted_samples[split_idx:]

    train_ds = Dataset.from_list(train_samples)
    eval_ds  = Dataset.from_list(eval_samples)

    # Load Base Model & merge SFT Adapter
    print(f"\nLoading Base Model from {MODEL_NAME}...")
    base_model = AutoModelForSeq2SeqLM.from_pretrained(
        MODEL_NAME,
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )
    
    # Suppression base model
    suppress_unused_tokens(base_model, ALL_SUPPRESS_IDS)

    if os.path.exists(SFT_ADAPTER_DIR):
        print(f"Loading SFT Adapter dari {SFT_ADAPTER_DIR}...")
        model = PeftModel.from_pretrained(
            base_model,
            SFT_ADAPTER_DIR,
            is_trainable=True,  # Supaya LoRA tetap trainable untuk DPO
        )
    else:
        print("[WARN] SFT Adapter tidak ditemukan, menggunakan base model langsung.")
        lora_config = LoraConfig(
            task_type=TaskType.SEQ_2_SEQ_LM,
            r=32,
            lora_alpha=64,
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
            lora_dropout=0.05,
        )
        model = get_peft_model(base_model, lora_config)

    setattr(model.config, "use_cache", False)

    # Re-register hook untuk suppressed tokens
    suppress_unused_tokens(model, ALL_SUPPRESS_IDS)

    sample_callback = SampleGenerationCallback(
        tokenizer=tokenizer,
        queries=EVAL_QUERIES,
        output_dir=OUTPUT_DIR,
        eval_every_n_steps=10,  # Evaluasi DPO lebih sering (tiap 10 steps)
    )

    # DPO Training Config
    training_args = DPOConfig(
        output_dir=OUTPUT_DIR,
        per_device_train_batch_size=2,
        gradient_accumulation_steps=4,
        learning_rate=5e-5,  # Learning rate lebih rendah untuk DPO
        num_train_epochs=1,  # 1 Epoch DPO
        warmup_ratio=0.1,
        weight_decay=0.01,
        lr_scheduler_type="cosine",
        logging_steps=2,
        save_strategy="steps",
        save_steps=10,
        save_total_limit=2,
        eval_strategy="steps",
        eval_steps=10,
        optim="adamw_torch",
        report_to="none",
        bf16=torch.cuda.is_available(),
        gradient_checkpointing=True,
        max_length=MAX_SOURCE_LENGTH + MAX_TARGET_LENGTH,  # Didukung di DPOConfig
        beta=0.1,  # Didukung di DPOConfig
    )

    trainer = Seq2SeqDPOTrainer(
        model=model,
        ref_model=None,  # TRL otomatis mematikan adapter jika bernilai None untuk menghemat VRAM
        args=training_args,
        train_dataset=cast(Any, train_ds),
        eval_dataset=cast(Any, eval_ds),
        processing_class=tokenizer,
        callbacks=[sample_callback],
    )

    print("\nStarting DPO Training...")
    trainer.train()

    # Save
    final_path = os.path.join(OUTPUT_DIR, "final_dpo_adapter")
    print(f"\nSaving final DPO adapter to {final_path}...")
    trainer.save_model(final_path)
    tokenizer.save_pretrained(final_path)
    print("✅ DPO Light Selesai!")

if __name__ == "__main__":
    main()
