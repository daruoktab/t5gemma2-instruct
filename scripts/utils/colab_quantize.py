import os
import torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, BitsAndBytesConfig
from huggingface_hub import HfApi, login

def main():
    token = os.environ.get("HF_TOKEN")
    if not token:
        raise ValueError("HF_TOKEN environment variable is not set in the Colab VM environment!")

    # Login to HF Hub
    print("Logging in to Hugging Face Hub...")
    login(token=token)

    model_id = "daruokta/t5gemma-2-4b-4b-instruct-chat-indo-v3-unsloth"
    quantized_path = "./quantized_4bit"

    print("Loading model and tokenizer in 4-bit...")
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
        llm_int8_skip_modules=['model.encoder.vision_tower', 'lm_head', 'embed_tokens']
    )

    tokenizer = AutoTokenizer.from_pretrained(model_id, subfolder="merged_bf16", token=token)
    assert tokenizer is not None
    model = AutoModelForSeq2SeqLM.from_pretrained(
        model_id,
        subfolder="merged_bf16",
        quantization_config=bnb_config,
        device_map="auto",
        token=token
    )

    print("Saving 4-bit quantized model and tokenizer locally on Colab VM...")
    os.makedirs(quantized_path, exist_ok=True)
    model.save_pretrained(quantized_path, safe_serialization=True)
    tokenizer.save_pretrained(quantized_path)

    print("Uploading quantized model to Hugging Face Hub under 'quantized_4bit' subfolder...")
    api = HfApi()
    api.upload_folder(
        folder_path=quantized_path,
        path_in_repo="quantized_4bit",
        repo_id=model_id,
        repo_type="model",
        token=token
    )
    print("✅ Quantization and upload completed successfully!")

if __name__ == "__main__":
    main()
