"""
Manual Mechanistic Interpretability via PyTorch Hooks
Sebagai alternatif TransformerLens untuk T5Gemma2
"""

import torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
from typing import Callable, Dict, List

MODEL_NAME = "models/t5gemma2-270m-task-vector"

def get_activation_cache(model: torch.nn.Module, inputs: Dict[str, torch.Tensor], target_layer_names: List[str] | None = None) -> Dict[str, torch.Tensor]:
    """
    Menjalankan model dan mencatat aktivasi di setiap layer yang diinginkan.
    Ini adalah replika manual dari `model.run_with_cache()` di TransformerLens.
    """
    cache = {}
    hooks = []

    def save_activation_hook(name: str) -> Callable:
        def hook(module, module_in, module_out):
            # module_out bisa berupa tuple, biasanya elemen pertama adalah hidden states
            if isinstance(module_out, tuple):
                hidden_states = module_out[0]
            else:
                hidden_states = module_out
            # Pindahkan ke CPU untuk menghemat memori GPU jika modelnya besar
            cache[name] = hidden_states.detach().cpu()
        return hook

    # Daftarkan hooks ke modul-modul di dalam model
    for name, module in model.named_modules():
        # Jika target_layer_names kosong, ambil semua layer.
        # Jika didefinisikan, ambil yang cocok saja.
        if target_layer_names is None or any(target in name for target in target_layer_names):
            # Kita hanya tertarik mencatat layer yang relevan, misalnya block encoder/decoder
            # atau attention head / mlp. Kita filter berdasarkan namanya.
            if "block" in name and name.count(".") == 2: # Contoh: 'encoder.block.0'
                hooks.append(module.register_forward_hook(save_activation_hook(name)))

    # Jalankan model (forward pass)
    with torch.no_grad():
        model(**inputs)

    # Bersihkan hooks setelah selesai
    for h in hooks:
        h.remove()

    return cache


def run_activation_patching(model: torch.nn.Module, inputs: Dict[str, torch.Tensor], patch_layer_name: str, patch_tensor: torch.Tensor):
    """
    Menjalankan model dan MENGGANTI (mencangkok) aktivasi di layer tertentu.
    Ini adalah inti dari mechanistic interpretability.
    """
    hooks = []

    def patch_hook(module, module_in, module_out):
        print(f"\n[!] MENCANGKOK AKTIVASI DI LAYER: {patch_layer_name}")
        # Ganti hidden states dengan tensor yang sudah kita siapkan
        if isinstance(module_out, tuple):
            # Tuple reconstruction karena output tuple PyTorch bersifat immutable
            new_out = (patch_tensor.to(module_out[0].device),) + module_out[1:]
            return new_out
        else:
            return patch_tensor.to(module_out.device)

    # Pasang kail pengganti di layer spesifik
    for name, module in model.named_modules():
        if name == patch_layer_name:
            hooks.append(module.register_forward_hook(patch_hook))

    # Jalankan model (output akhir akan berubah karena cangkokan kita)
    with torch.no_grad():
        outputs = getattr(model, "generate")(
            **inputs, 
            max_new_tokens=20,
            do_sample=False
        )

    for h in hooks:
        h.remove()

    return outputs


def main():
    print(f"Loading {MODEL_NAME}...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    assert tokenizer is not None, "Gagal memuat tokenizer"
    
    model = AutoModelForSeq2SeqLM.from_pretrained(
        MODEL_NAME,
        trust_remote_code=True,
        device_map="cuda",
        torch_dtype=torch.bfloat16
    )

    text = "<start_of_turn>user\nApa perbedaan simile dan metafora?<end_of_turn>\n<start_of_turn>model\n"
    inputs = tokenizer(text, return_tensors="pt").to(model.device)

    # 1. BACA PIKIRAN MODEL (Caching)
    print("\n--- 1. CACHING ACTIVATIONS ---")
    cache = get_activation_cache(model, inputs)
    print(f"Berhasil menyimpan {len(cache)} tensors dari berbagai layer!")
    
    # Cetak 5 layer pertama yang berhasil direkam
    for layer_name, tensor in list(cache.items())[:5]:
        print(f"Layer: {layer_name:<25} | Shape: {list(tensor.shape)}")

    
    # 2. MANIPULASI PIKIRAN MODEL (Activation Patching)
    print("\n--- 2. ACTIVATION PATCHING (Cangkok Layer) ---")
    
    # Ambil hidden_state dari encoder layer ke-0 yang tadi kita simpan, lalu kita NOL-kan semua nilainya
    layer_target = list(cache.keys())[0]  # Ambil layer pertama
    original_tensor = cache[layer_target]
    
    # Bikin tensor palsu (blank / isinya nol semua)
    fake_tensor = torch.zeros_like(original_tensor)
    
    print(f"Kita akan mengganti isi pikiran model di {layer_target} menjadi NOL.")
    fake_outputs = run_activation_patching(model, inputs, layer_target, fake_tensor)
    fake_response = tokenizer.decode(fake_outputs[0], skip_special_tokens=True)
    
    print("\nHasil Akhir Setelah Otaknya Dicangkok Tensor Kosong:")
    print(fake_response)


if __name__ == "__main__":
    main()
