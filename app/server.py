import os
import re
import json
import gc
import torch
from flask import Flask, request, jsonify, render_template
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from peft import PeftModel

app = Flask(__name__, template_folder='templates')

DEFAULT_BASE_MODEL = "google/t5gemma-2-1b-1b"
DEFAULT_ADAPTER = "daruokta/t5gemma-2-1b-1b-instruct-chat-indo-v2-exp"

# Resolve paths relative to repository root
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.normpath(os.path.join(_SCRIPT_DIR, ".."))
VAL_DATA_PATH = os.path.join(ROOT_DIR, "data", "chat_val_demo.jsonl")

# Token IDs yang harus di-suppress (unused + vision)
SUPPRESS_BLOCK1 = list(range(6, 105))         # <unused0>–<unused98>
SUPPRESS_BLOCK2 = list(range(256002, 262144))  # <unused100>–<unused6241>
SUPPRESS_VISION = [255999, 256000, 256001]     # <end_of_image>, <image_soft_token>
ALL_SUPPRESS_IDS = set(SUPPRESS_BLOCK1 + SUPPRESS_BLOCK2 + SUPPRESS_VISION)

# Global model and tokenizer
tokenizer = None
model = None
current_base_model = None
current_adapter = None
device = "cuda" if torch.cuda.is_available() else "cpu"

MAX_SOURCE_LENGTH = 4096
MAX_TARGET_LENGTH = 1024

SYSTEM_PROMPT = (
    "Kamu adalah asisten AI yang helpful, santai, dan ramah. "
    "Gunakan Bahasa Indonesia sebagai bahasa utama."
)

def apply_logit_mask(model, suppress_ids):
    """
    Menerapkan logit masking secara dinamis lewat PyTorch forward hook.
    """
    vocab_size = model.config.vocab_size
    suppress_list = [i for i in suppress_ids if i < vocab_size]
    
    mask = torch.zeros(vocab_size, dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32)
    mask[suppress_list] = -10000.0
    
    def forward_hook(module, inputs, outputs):
        if hasattr(outputs, "logits"):
            outputs.logits.add_(mask.to(outputs.logits.device))
        elif isinstance(outputs, tuple):
            outputs[0].add_(mask.to(outputs[0].device))
        return outputs
        
    model.register_forward_hook(forward_hook)
    print(f"  ✅ Logit masking registered untuk {len(suppress_list)} suppressed tokens.")

def load_model_and_tokenizer(base_model_name, adapter_name_or_path=None):
    global tokenizer, model, current_base_model, current_adapter
    
    print("\nUnloading previous model (if any)...")
    model = None
    tokenizer = None
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        
    # Resolve adapter path locally if it's on HF Hub to bypass PEFT subfolder bugs
    adapter_path = adapter_name_or_path
    if adapter_name_or_path and adapter_name_or_path.strip().lower() != "none" and not os.path.exists(adapter_name_or_path):
        from huggingface_hub import snapshot_download
        print(f"Downloading adapter snapshot from HF Hub repo '{adapter_name_or_path}'...")
        try:
            local_dir = snapshot_download(
                repo_id=adapter_name_or_path,
                allow_patterns=[
                    "final_adapter/*",
                    "checkpoint-1225/*",
                    "checkpoint-1000/*",
                    "adapter_config.json",
                    "adapter_model.bin",
                    "adapter_model.safetensors",
                    "README.md"
                ]
            )
            print(f"Snapshot downloaded to: {local_dir}")
            
            # Find the best adapter directory
            final_adapter_dir = os.path.join(local_dir, "final_adapter")
            checkpoint_1225_dir = os.path.join(local_dir, "checkpoint-1225")
            checkpoint_1000_dir = os.path.join(local_dir, "checkpoint-1000")
            
            if os.path.exists(final_adapter_dir) and os.path.exists(os.path.join(final_adapter_dir, "adapter_config.json")):
                adapter_path = final_adapter_dir
                print(f"Found 'final_adapter' subfolder: {adapter_path}")
            elif os.path.exists(checkpoint_1225_dir) and os.path.exists(os.path.join(checkpoint_1225_dir, "adapter_config.json")):
                adapter_path = checkpoint_1225_dir
                print(f"Found 'checkpoint-1225' subfolder: {adapter_path}")
            elif os.path.exists(checkpoint_1000_dir) and os.path.exists(os.path.join(checkpoint_1000_dir, "adapter_config.json")):
                adapter_path = checkpoint_1000_dir
                print(f"Found 'checkpoint-1000' subfolder: {adapter_path}")
            elif os.path.exists(os.path.join(local_dir, "adapter_config.json")):
                adapter_path = local_dir
                print(f"Using snapshot root: {adapter_path}")
            else:
                # Scan for any checkpoint directory
                subdirs = [os.path.join(local_dir, d) for d in os.listdir(local_dir) if os.path.isdir(os.path.join(local_dir, d))]
                checkpoint_dirs = [d for d in subdirs if "checkpoint-" in os.path.basename(d)]
                if checkpoint_dirs:
                    def get_num(folder):
                        match = re.search(r'\d+', os.path.basename(folder))
                        return int(match.group(0)) if match else 0
                    checkpoint_dirs.sort(key=get_num)
                    adapter_path = checkpoint_dirs[-1]
                    print(f"Found auto-detected checkpoint folder: {adapter_path}")
                else:
                    adapter_path = local_dir
                    print(f"No adapter config found in subdirectories. Trying snapshot root: {adapter_path}")
        except Exception as e:
            print(f"Failed to download adapter snapshot: {e}. Will try default Hub resolution.")
            adapter_path = adapter_name_or_path

    print(f"Loading tokenizer for {base_model_name}...")
    tok_path = adapter_path if (adapter_path and os.path.exists(adapter_path)) else base_model_name
    try:
        tokenizer = AutoTokenizer.from_pretrained(tok_path, trust_remote_code=True)
    except Exception as tok_err:
        print(f"Failed to load tokenizer from {tok_path}: {tok_err}. Falling back to base model tokenizer.")
        tokenizer = AutoTokenizer.from_pretrained(base_model_name, trust_remote_code=True)
        
    print(f"Loading base model {base_model_name}...")
    base_model = AutoModelForSeq2SeqLM.from_pretrained(
        base_model_name,
        trust_remote_code=True,
        dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
        device_map="auto" if torch.cuda.is_available() else None
    )
    
    if not torch.cuda.is_available():
        base_model = base_model.to(device)
        
    # Apply logit masking before Peft
    apply_logit_mask(base_model, ALL_SUPPRESS_IDS)
        
    if adapter_name_or_path and adapter_name_or_path.strip().lower() != "none":
        assert adapter_path is not None
        print(f"Loading LoRA adapter from {adapter_path}...")
        try:
            model = PeftModel.from_pretrained(base_model, adapter_path)
        except Exception as e:
            print(f"Failed to load adapter directly from {adapter_path}: {e}. Trying fallback...")
            # If adapter_path is local snapshot and loading failed, try loading from original path with subfolder
            try:
                model = PeftModel.from_pretrained(base_model, adapter_name_or_path, subfolder="final_adapter")
            except Exception as e2:
                print(f"Failed to load adapter with subfolder final_adapter: {e2}. Trying checkpoint fallback...")
                try:
                    if "exp" in adapter_name_or_path:
                        model = PeftModel.from_pretrained(base_model, adapter_name_or_path, subfolder="checkpoint-1225")
                    else:
                        model = PeftModel.from_pretrained(base_model, adapter_name_or_path, subfolder="checkpoint-1000")
                except Exception as e3:
                    raise e3
    else:
        print("No LoRA adapter specified. Using base model directly.")
        model = base_model
        
    model.eval()
    current_base_model = base_model_name
    current_adapter = adapter_name_or_path
    print(f"✅ Loaded base={base_model_name}, adapter={adapter_name_or_path} successfully!")
    return True

def init_model():
    try:
        print("Attempting to load default 1B model + v4-exp adapter...")
        load_model_and_tokenizer(DEFAULT_BASE_MODEL, DEFAULT_ADAPTER)
    except Exception as e:
        print(f"Failed to load 1B model: {e}. Falling back to 270m base model.")
        try:
            load_model_and_tokenizer("google/t5gemma-2-270m-270m", None)
        except Exception as e2:
            print(f"Critical error loading fallback model: {e2}")

def format_chat_prompt(messages):
    formatted = ""
    system = SYSTEM_PROMPT
    
    # Try to find a custom system prompt
    for msg in messages:
        if msg["role"] == "system":
            system = msg["content"]
            break
            
    is_first_user = True
    for msg in messages:
        role = msg["role"]
        content = msg["content"]
        if not content or role == "system":
            continue
            
        if role == "user":
            formatted += "<start_of_turn>user\n"
            if is_first_user and system:
                formatted += system + "\n\n"
                is_first_user = False
            formatted += content + "<end_of_turn>\n"
        elif role in ["assistant", "model"]:
            formatted += "<start_of_turn>model\n"
            formatted += content + "<end_of_turn>\n"
            
    formatted += "<start_of_turn>model\n"
    return formatted

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/api/samples", methods=["GET"])
def get_samples():
    samples = []
    
    if os.path.exists(VAL_DATA_PATH):
        conversations = []
        current_conv = []
        with open(VAL_DATA_PATH, "r", encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                obj = json.loads(line)
                
                if "assistant:" not in obj["input"]:
                    if current_conv:
                        conversations.append(current_conv)
                    current_conv = []
                current_conv.append(obj)
            if current_conv:
                conversations.append(current_conv)
                
        # Sample 5 conversations
        for i, conv in enumerate(conversations[:5]):
            messages = []
            # Parse messages from SFT inputs
            # Turn 1
            first_turn = conv[0]
            system_match = re.search(r'^system:\s*(.*?)(?=\nuser:)', first_turn["input"], re.DOTALL)
            system = system_match.group(1).strip() if system_match else SYSTEM_PROMPT
            messages.append({"role": "system", "content": system})
            
            raw_input = first_turn["input"]
            if system_match:
                raw_input = raw_input[system_match.end():].strip()
                
            parts = re.split(r'\n(user:|assistant:)\s*', '\n' + raw_input)
            for j in range(1, len(parts), 2):
                role = parts[j].replace(':', '').strip()
                content = parts[j + 1].strip()
                if content:
                    messages.append({"role": role, "content": content})
            messages.append({"role": "assistant", "content": first_turn["target"]})
            
            # Additional turns
            for turn in conv[1:]:
                parts = re.split(r'\n(user:|assistant:)\s*', '\n' + turn["input"])
                last_user = parts[-1].strip()
                messages.append({"role": "user", "content": last_user})
                messages.append({"role": "assistant", "content": turn["target"]})
                
            samples.append({
                "id": i,
                "title": f"Sample Chat #{i+1} ({len(conv)} Turns)",
                "messages": messages
            })
            
    return jsonify(samples)

@app.route("/api/model_status", methods=["GET"])
def api_model_status():
    global current_base_model, current_adapter
    return jsonify({
        "base_model": current_base_model,
        "adapter": current_adapter,
        "device": device,
        "gpu_memory": f"{torch.cuda.memory_allocated() / 1024**3:.2f} GB" if torch.cuda.is_available() else "N/A"
    })

@app.route("/api/load_model", methods=["POST"])
def api_load_model():
    data = request.json or {}
    base_model = data.get("base_model", DEFAULT_BASE_MODEL)
    adapter = data.get("adapter", DEFAULT_ADAPTER)
    
    try:
        load_model_and_tokenizer(base_model, adapter)
        return jsonify({
            "status": "success",
            "base_model": current_base_model,
            "adapter": current_adapter,
            "message": f"Successfully loaded base={base_model} and adapter={adapter}"
        })
    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({
            "status": "error",
            "message": str(e)
        }), 500

@app.route("/api/chat", methods=["POST"])
def chat():
    global tokenizer, model
    if tokenizer is None or model is None:
        return jsonify({"error": "Model not loaded"}), 500
        
    data = request.json
    messages = data.get("messages", [])
    if not messages:
        return jsonify({"error": "No messages provided"}), 400
        
    temperature = float(data.get("temperature", 0.7))
    repetition_penalty = float(data.get("repetition_penalty", 1.0))
    do_sample = data.get("do_sample", temperature > 0.0)
    
    prompt = format_chat_prompt(messages)
    
    inputs = tokenizer(
        prompt,
        return_tensors="pt",
        add_special_tokens=False,
        truncation=True,
        max_length=MAX_SOURCE_LENGTH
    )
    inputs = {k: v.to(device) for k, v in inputs.items()}
    
    stop_eot = tokenizer.convert_tokens_to_ids("<end_of_turn>")
    stop_eos = tokenizer.eos_token_id or 1
    stop_ids = list({stop_eot, stop_eos})
    
    gen_kwargs = {
        "max_new_tokens": 256,
        "repetition_penalty": repetition_penalty,
        "eos_token_id": stop_ids,
        "do_sample": do_sample
    }
    
    if do_sample:
        gen_kwargs["temperature"] = temperature
        gen_kwargs["top_p"] = 0.9
        gen_kwargs["top_k"] = 50
    
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            **gen_kwargs
        )
        
    raw_response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    response = raw_response.strip() if isinstance(raw_response, str) else " ".join(raw_response).strip()
    
    badge = None
    last_user_content = messages[-1]["content"].lower() if messages else ""
    if "ringkas" in last_user_content or "summarize" in last_user_content or "rangkum" in last_user_content:
        badge = "SUMMARIZE"
    elif "terjemah" in last_user_content or "translate" in last_user_content or "bahasa inggris" in last_user_content or "english" in last_user_content:
        badge = "TRANSLATE"
    elif "parafrase" in last_user_content or "paraphrase" in last_user_content:
        badge = "PARAPHRASE"
    elif "grounded" in last_user_content or "ekstrak" in last_user_content or "extract" in last_user_content or "qa" in last_user_content:
        badge = "GROUNDED"
        
    return jsonify({
        "response": response,
        "badge": badge
    })

if __name__ == "__main__":
    init_model()
    app.run(host="127.0.0.1", port=5000, debug=False)
