import os
import torch
import gc
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, BitsAndBytesConfig
from huggingface_hub import HfApi

# 1. Konfigurasi
model_id = "google/t5gemma-2-270m-270m"
repo_name = "daruokta/t5gemma-2-270m-quant-test"  # Repositori HF Hub Anda
local_quant_path = "results/t5gemma2-270m-quant-test"

# Pastikan token Hugging Face ada
token = os.getenv("HF_TOKEN")
if not token:
    print(
        "WARNING: HF_TOKEN env variable is not set. We will try using Hugging Face Hub cached credentials."
    )

# 2. Load model dalam bfloat16 & Kuantisasi ke 4-bit
print("\n=== STEP 1: Memuat model dan mengompresi ke 4-bit NF4 ===")
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_use_double_quant=True,
    bnb_4bit_compute_dtype=torch.bfloat16,
    llm_int8_skip_modules=['model.encoder.vision_tower', 'lm_head', 'embed_tokens']
)
tokenizer = AutoTokenizer.from_pretrained(model_id)
assert tokenizer is not None
model = AutoModelForSeq2SeqLM.from_pretrained(
    model_id, quantization_config=bnb_config, device_map="auto"
)

# 3. Simpan model terkuantisasi secara lokal
print("\n=== STEP 2: Menyimpan model terkuantisasi ke disk ===")
os.makedirs(os.path.dirname(local_quant_path), exist_ok=True)
model.save_pretrained(local_quant_path, safe_serialization=True)
tokenizer.save_pretrained(local_quant_path)
print(f"Model terkuantisasi disimpan di {local_quant_path}")

# Hapus model dari memori untuk simulasi download bersih
del model
gc.collect()
torch.cuda.empty_cache()

# 4. Unggah model terkuantisasi ke Hugging Face Hub
print("\n=== STEP 3: Mengunggah model terkuantisasi ke Hugging Face Hub ===")
try:
    api = HfApi()
    api.create_repo(
        repo_id=repo_name, repo_type="model", private=True, exist_ok=True, token=token
    )
    api.upload_folder(
        folder_path=local_quant_path, repo_id=repo_name, repo_type="model", token=token
    )
    print("✅ Berhasil mengunggah model terkuantisasi ke Hugging Face Hub!")
except Exception as e:
    print(f"❌ Gagal mengunggah ke Hub: {e}")
    print("Kita akan langsung menguji loading dari folder lokal sebagai alternatif.")

# 5. Tarik model langsung dari Hugging Face Hub (atau folder lokal jika upload gagal)
print("\n=== STEP 4: Memuat model terkuantisasi kembali untuk inferensi ===")
# Gunakan repo name jika token tersedia, jika tidak gunakan path lokal
load_source = repo_name if token else local_quant_path
print(f"Memuat model dari: {load_source}")

# Catat VRAM sebelum load
torch.cuda.empty_cache()
vram_before = torch.cuda.memory_allocated() / (1024**2)

try:
    local_model = AutoModelForSeq2SeqLM.from_pretrained(
        load_source, device_map="auto", token=token
    )
    local_tokenizer = AutoTokenizer.from_pretrained(load_source, token=token)
    assert local_tokenizer is not None

    # Catat VRAM setelah load
    vram_after = torch.cuda.memory_allocated() / (1024**2)
    print("Pemuatan berhasil!")
    print(f"VRAM sebelum model dimuat: {vram_before:.2f} MB")
    print(f"VRAM setelah model dimuat: {vram_after:.2f} MB")
    print(f"Estimasi VRAM yang digunakan model: {vram_after - vram_before:.2f} MB")

    # 6. Uji inferensi sederhana
    print("\n=== STEP 5: Menjalankan uji inferensi ===")
    input_text = (
        "<start_of_turn>user\nSiapakah kamu?<end_of_turn>\n<start_of_turn>model\n"
    )
    inputs = local_tokenizer(input_text, return_tensors="pt").to("cuda")
    outputs = local_model.generate(**inputs, max_new_tokens=50)
    response = local_tokenizer.decode(outputs[0], skip_special_tokens=True)
    print(f"Prompt: {input_text}")
    print(f"Respons Model: {response}")

except Exception:
    import traceback
    print("❌ Gagal memuat atau menjalankan model:")
    traceback.print_exc()
