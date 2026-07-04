from huggingface_hub import DatasetCard

repo_id = "daruokta/t5gemma2-indonesia-chat-formatted"
print(f"Fetching DatasetCard from {repo_id}...")
card = DatasetCard.load(repo_id)

text = card.text

text = text.replace(
    "- **May 2026 Update:** The chat dataset has been expanded! It previously contained 1,030 conversations and has now been updated to include **2,500 curated multi-turn conversations**.",
    "- **June 2026 Update:** The chat dataset has been expanded again! It previously contained 2,500 conversations and has now been updated to include **3,000 curated multi-turn conversations**, including 500 new Agentic Prefix-Task conversations."
)

text = text.replace(
    "1. **`chat_multiturn`**: **2,500** manually curated multi-turn conversational data (Updated from 1,030).",
    "1. **`chat_multiturn`**: **3,000** manually curated multi-turn conversational data (including 500 agentic prefix-task)."
)

card.text = text
print("Pushing updated DatasetCard to Hub...")
card.push_to_hub(repo_id)
print("✅ Successfully updated the README.md on Hugging Face Hub!")
