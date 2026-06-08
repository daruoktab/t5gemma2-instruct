import torch
import os
from typing import cast
from transformers import AutoModelForCausalLM, AutoModelForSeq2SeqLM

def transplant_270m():
    src_name = "google/gemma-3-270m-it"
    dst_name = "google/t5gemma-2-270m-270m"
    output_dir = "models/t5gemma2-270m-cangkok"
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    
    print(f"Loading Source Model (Gemma 3 IT): {src_name}...")
    src_model = AutoModelForCausalLM.from_pretrained(
        src_name, 
        trust_remote_code=True, 
        torch_dtype=torch.bfloat16, 
        device_map=device
    )
    
    print(f"Loading Target Model (T5Gemma 2): {dst_name}...")
    dst_model = AutoModelForSeq2SeqLM.from_pretrained(
        dst_name, 
        trust_remote_code=True, 
        torch_dtype=torch.bfloat16, 
        device_map=device
    )
    
    print("Starting Weight Transplantation (Decoder Only)...")
    
    # 1. Embeddings
    print("Transplanting Embeddings...")
    dst_model.model.decoder.embed_tokens.weight.data = src_model.model.embed_tokens.weight.data.clone()
    
    # 2. Final Norm
    print("Transplanting Final Norm...")
    dst_model.model.decoder.norm.weight.data = src_model.model.norm.weight.data.clone()
    
    # 3. LM Head
    print("Transplanting LM Head...")
    # T5Gemma 2 has lm_head.out_proj
    dst_model.lm_head.out_proj.weight.data = src_model.lm_head.weight.data.clone()
    
    # 4. Layers
    num_layers = len(src_model.model.layers)
    print(f"Transplanting {num_layers} Layers...")
    
    for i in range(num_layers):
        src_layer = src_model.model.layers[i]
        dst_layer = dst_model.model.decoder.layers[i]
        
        # Layernorms
        dst_layer.pre_self_attn_layernorm.weight.data = src_layer.input_layernorm.weight.data.clone()
        dst_layer.post_self_attn_layernorm.weight.data = src_layer.post_attention_layernorm.weight.data.clone()
        dst_layer.pre_feedforward_layernorm.weight.data = src_layer.pre_feedforward_layernorm.weight.data.clone()
        dst_layer.post_feedforward_layernorm.weight.data = src_layer.post_feedforward_layernorm.weight.data.clone()
        
        # Attention
        dst_layer.self_attn.q_proj.weight.data = src_layer.self_attn.q_proj.weight.data.clone()
        dst_layer.self_attn.k_proj.weight.data = src_layer.self_attn.k_proj.weight.data.clone()
        dst_layer.self_attn.v_proj.weight.data = src_layer.self_attn.v_proj.weight.data.clone()
        dst_layer.self_attn.o_proj.weight.data = src_layer.self_attn.o_proj.weight.data.clone()
        
        # Attention Norms
        dst_layer.self_attn.q_norm.weight.data = src_layer.self_attn.q_norm.weight.data.clone()
        dst_layer.self_attn.k_norm.weight.data = src_layer.self_attn.k_norm.weight.data.clone()
        
        # MLP
        dst_layer.mlp.gate_proj.weight.data = src_layer.mlp.gate_proj.weight.data.clone()
        dst_layer.mlp.up_proj.weight.data = src_layer.mlp.up_proj.weight.data.clone()
        dst_layer.mlp.down_proj.weight.data = src_layer.mlp.down_proj.weight.data.clone()
        
        if (i + 1) % 5 == 0:
            print(f"  Processed {i + 1} layers...")

    print("Transplantation Complete!")
    
    print(f"Saving model and tokenizer to {output_dir}...")
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    dst_model.save_pretrained(output_dir)
    
    from transformers import AutoTokenizer, PreTrainedTokenizerBase
    tokenizer = AutoTokenizer.from_pretrained(src_name)
    if tokenizer is not None:
        tokenizer = cast(PreTrainedTokenizerBase, tokenizer)
        tokenizer.save_pretrained(output_dir)
    
    print("Model and Tokenizer saved successfully.")

if __name__ == "__main__":
    transplant_270m()
