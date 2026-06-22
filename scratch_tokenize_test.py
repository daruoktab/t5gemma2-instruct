import json
from transformers import AutoTokenizer

def main():
    try:
        # Load the DeepSeek tokenizer specified by the user
        tokenizer = AutoTokenizer.from_pretrained("deepseek-ai/DeepSeek-V4-Flash", trust_remote_code=True)
    except Exception as e:
        print(f"Error loading DeepSeek-V4-Flash tokenizer: {e}")
        print("Mencoba fallback ke tokenizer deepseek yang umum...")
        tokenizer = AutoTokenizer.from_pretrained("deepseek-ai/deepseek-coder-33b-instruct", trust_remote_code=True)

    with open("data/generated_topics_manual.json", "r", encoding="utf-8") as f:
        data = json.load(f)

    toks = []
    for d in data:
        text = d["topik"] + "\n\n" + d.get("summary", "")
        toks.append(len(tokenizer.encode(text)))

    print(f"Total data: {len(toks)}")
    print(f"Min tokens: {min(toks)}")
    print(f"Max tokens: {max(toks)}")
    print(f"Avg tokens: {sum(toks)/len(toks):.1f}")
    
    print("\nContoh 1 (Topik + Summary):")
    example = data[0]["topik"] + "\n\n" + data[0].get("summary", "")
    print(example)
    print(f"-> Token count: {toks[0]}")

if __name__ == "__main__":
    main()
