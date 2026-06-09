import os
import re
import json
import torch
from flask import Flask, request, jsonify, render_template
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from peft import PeftModel

app = Flask(__name__, template_folder='templates')

MODEL_NAME = "google/t5gemma-2-270m-270m"

# Resolve paths relative to repository root
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.normpath(os.path.join(_SCRIPT_DIR, ".."))
ADAPTER_PATH = os.path.join(ROOT_DIR, "results", "t5gemma2-270m-clean-sft", "final_adapter")
VAL_DATA_PATH = os.path.join(ROOT_DIR, "data", "chat_val_demo.jsonl")

# Token IDs yang harus di-suppress (unused + vision)
SUPPRESS_BLOCK1 = list(range(6, 105))         # <unused0>–<unused98>
SUPPRESS_BLOCK2 = list(range(256002, 262144))  # <unused100>–<unused6241>
SUPPRESS_VISION = [255999, 256000, 256001]     # <end_of_image>, <image_soft_token>
ALL_SUPPRESS_IDS = set(SUPPRESS_BLOCK1 + SUPPRESS_BLOCK2 + SUPPRESS_VISION)

# Global model and tokenizer
tokenizer = None
model = None
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

def init_model():
    global tokenizer, model
    print(f"Loading tokenizer {MODEL_NAME}...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
    
    print(f"Loading base model {MODEL_NAME}...")
    base_model = AutoModelForSeq2SeqLM.from_pretrained(
        MODEL_NAME,
        trust_remote_code=True,
        dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
        device_map="auto" if torch.cuda.is_available() else None
    )
    
    if not torch.cuda.is_available():
        base_model = base_model.to(device)
        
    # Terapkan logit masking sebelum dimuat ke PeftModel
    apply_logit_mask(base_model, ALL_SUPPRESS_IDS)
        
    if os.path.exists(ADAPTER_PATH):
        print(f"Loading LoRA adapter from {ADAPTER_PATH}...")
        model = PeftModel.from_pretrained(base_model, ADAPTER_PATH)
    else:
        print(f"[WARN] LoRA adapter not found at {ADAPTER_PATH}. Falling back to base model.")
        model = base_model
        
    model.eval()
    print("✅ Model initialization complete!")

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

@app.route("/api/chat", methods=["POST"])
def chat():
    global tokenizer, model
    if tokenizer is None or model is None:
        return jsonify({"error": "Model not loaded"}), 500
        
    data = request.json
    messages = data.get("messages", [])
    if not messages:
        return jsonify({"error": "No messages provided"}), 400
        
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
    
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=256,
            do_sample=False,
            eos_token_id=stop_ids
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
