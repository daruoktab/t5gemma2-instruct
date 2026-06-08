import json
import re
from google import genai

def extract_json(text):
    match = re.search(r'\{.*\}', text, re.DOTALL)
    if match:
        try:
            return json.loads(match.group(0))
        except:
            pass
    return {"Topik": "Error parsing", "Task": "Error parsing"}

def main():
    client = genai.Client()
    
    with open("t5gemma2_finetune_dataset_all.jsonl", "r", encoding="utf-8") as f:
        lines = f.readlines()
        
    results = []
    
    for i, line in enumerate(lines):
        data = json.loads(line)
        conversations = data.get("conversations", [])
        turns = len(conversations)
        
        # We only need the first few exchanges to understand the topic, but for tasks we might need the whole thing.
        # To avoid context limits, we can send the whole conversation as text.
        conv_text = ""
        for turn in conversations:
            conv_text += f"{turn['role']}: {turn['content']}\n"
            
        prompt = f"""
Analyze the following conversation and provide a JSON response with two keys:
1. "Topik": A short summary of the conversation topic (in Indonesian, 2-5 words).
2. "Task": A comma-separated list of tasks performed by the assistant in the conversation (in English, choose from: translation, rewriting, summarization, paraphrase, Q-generation, extraction, explanation, roleplay, calculation).

Conversation:
{conv_text[:4000]}  # limit length if needed

Respond ONLY with valid JSON. Example:
{{"Topik": "Komplain belanja online", "Task": "translation, rewriting, summarization, paraphrase"}}
"""
        
        print(f"Processing conversation {i+1}/{len(lines)}...")
        try:
            response = client.models.generate_content(
                model="gemini-2.5-flash-lite",
                contents=prompt,
            )
            parsed = extract_json(response.text)
        except Exception as e:
            try:
                response = client.models.generate_content(
                    model="gemma-4-31b-it",
                    contents=prompt,
                )
                parsed = extract_json(response.text)
            except Exception as e2:
                print(f"Error on conversation {i+1}: {e2}")
                parsed = {"Topik": "Error", "Task": "Error"}
                
        results.append({
            "Konversasi": i + 1,
            "Topik": parsed.get("Topik", ""),
            "Turns": turns,
            "Task yang di-embed": parsed.get("Task", "")
        })
        
    with open("summary_table.md", "w", encoding="utf-8") as f:
        f.write("| Konversasi | Topik | Turns | Task yang di-embed |\n")
        f.write("|---|---|---|---|\n")
        for res in results:
            f.write(f"| {res['Konversasi']} | {res['Topik']} | {res['Turns']} | {res['Task yang di-embed']} |\n")
            
    print("Table generated in summary_table.md")

if __name__ == "__main__":
    main()
