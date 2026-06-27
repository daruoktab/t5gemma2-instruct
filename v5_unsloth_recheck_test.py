import torch
from unsloth import FastLanguageModel
import os

# We might need to handle HF_TOKEN if gated, but t5gemma-2 is probably not gated, 
# or if it is, the user already has the token in their env or we don't need it.

MODEL_NAME = "google/t5gemma-2-270m-270m"
MAX_SOURCE_LENGTH = 2048
LOAD_IN_4BIT = True

print(f"Loading Model from {MODEL_NAME} using Unsloth...")
model, tokenizer_unsloth = FastLanguageModel.from_pretrained(
    model_name = MODEL_NAME,
    max_seq_length = MAX_SOURCE_LENGTH,
    load_in_4bit = LOAD_IN_4BIT,
    trust_remote_code = True,
)

print("Applying LoRA using Unsloth...")
model = FastLanguageModel.get_peft_model(
    model,
    r = 256,
    lora_alpha = 512,
    target_modules = [
        "q_proj",
        "k_proj",
        "v_proj",
        "o_proj",
        "gate_proj",
        "up_proj",
        "down_proj",
    ],
    lora_dropout = 0.2,
    bias = "none",
    use_gradient_checkpointing = "unsloth",
    random_state = 3407,
)

print("Test passed! Unsloth successfully loaded the model and applied LoRA.")
