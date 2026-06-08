import re
from datasets import load_dataset
from transformers import AutoTokenizer

SYSTEM_PROMPT = "Kamu adalah asisten AI yang helpful, santai, dan ramah. Gunakan Bahasa Indonesia sebagai bahasa utama."

def format_encoder_from_raw(raw_input: str) -> str:
    system_match = re.search(r"^system:\s*(.*?)(?=\nuser:)", raw_input, re.DOTALL)
    system = system_match.group(1).strip() if system_match else SYSTEM_PROMPT
    
    if system_match:
        raw_input = raw_input[system_match.end():].strip()
        
    parts = re.split(r"\n(user:|assistant:)\s*", "\n" + raw_input)
    formatted = ""
    is_first_user = True
    
    for i in range(1, len(parts), 2):
        role = parts[i].replace(":", "").strip()
        content = parts[i+1].strip()
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

def main():
    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained("google/t5gemma-2-270m-270m", trust_remote_code=True)
    
    print("Loading dataset...")
    ds = load_dataset('daruokta/t5gemma2-indonesia-chat-formatted', 'chat_sft', split='train')
    
    sample = ds[0]
    print("\n--- ORIGINAL DATASET ROW ---")
    print(f"INPUT:\n{sample['input']}")
    print(f"\nTARGET:\n{sample['target']}")
    
    print("\n--- FORMATTED ENCODER INPUT ---")
    formatted_inp = format_encoder_from_raw(sample['input'])
    print(formatted_inp)
    
    print("\n--- TOKENIZATION ---")
    inp_ids = tokenizer.encode(formatted_inp, add_special_tokens=False)
    tgt_f = sample["target"].strip() + "<end_of_turn>"
    tgt_ids = tokenizer.encode(tgt_f, add_special_tokens=False)
    
    print(f"Encoder Tokens ({len(inp_ids)}): {tokenizer.convert_ids_to_tokens(inp_ids)[:20]}...")
    print(f"Decoder Tokens ({len(tgt_ids)}): {tokenizer.convert_ids_to_tokens(tgt_ids)[:20]}...")
    
    # Check what happens if there's no system prompt
    raw_no_sys = "user: halo, apa kabar?"
    print("\n--- FORMATTED ENCODER (NO SYSTEM PROMPT IN RAW) ---")
    print(format_encoder_from_raw(raw_no_sys))
    
    # Test conversational data with multiple turns
    raw_multi = "user: halo\nassistant: hai\nuser: nama kamu siapa?"
    print("\n--- FORMATTED ENCODER (MULTI-TURN) ---")
    print(format_encoder_from_raw(raw_multi))

if __name__ == "__main__":
    main()
