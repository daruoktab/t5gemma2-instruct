---
language:
- id
license: mit
task_categories:
- text-generation
- question-answering
pretty_name: T5Gemma-2 Indonesian Chat & QA Dataset
size_categories:
- 10K<n<100K
configs:
  - config_name: chat_multiturn
    data_files:
      - split: train
        path: chat_multiturn/train-*
  - config_name: indoqa_documents
    data_files:
      - split: train
        path: indoqa_documents/train-*
      - split: validation
        path: indoqa_documents/validation-*
---

# T5Gemma-2 Indonesian Chat & QA Dataset

A high-quality Indonesian language multi-turn conversation and reading comprehension dataset, specifically formatted for instruction tuning of sequence-to-sequence (Seq2Seq) models like T5-Gemma / T5-Gemma-2.

## Dataset Description

This dataset contains over 6,900 multi-turn conversations and document-based Q&A in Bahasa Indonesia. It covers diverse topics including everyday life, technology, general knowledge, and structured document analysis. The dataset is pre-formatted into standard OpenAI/ChatML message lists, making it instantly ready for fine-tuning.

### 📢 Updates
- **May 2026 Update:** The chat dataset has been expanded! It previously contained 1,030 conversations and has now been updated to include **2,500 curated multi-turn conversations**. We have also simplified the configurations by removing the redundant "expanded/full" subset, as the multi-turn format natively supports sequence-to-sequence training. The main conversational subset is now named `chat_multiturn`.

## Dataset Structure

The repository is divided into two distinct subsets (configurations):

1. **`chat_multiturn`**: **2,500** manually curated multi-turn conversational data (Updated from 1,030).
2. **`indoqa_documents`**: ~4,400 reading comprehension and factual Q&A examples derived from Indonesian documents (merged train & test sets).

Each entry contains a single column `messages` which is a list of message objects:
- `role`: The speaker role (`system`, `user`, or `assistant`).
- `content`: The actual text of the dialogue.

## Usage

You can easily load a specific subset using the Hugging Face `datasets` library:

```python
from datasets import load_dataset

# Load the multi-turn conversational dataset
ds_chat = load_dataset("daruokta/t5gemma2-indonesia-chat-formatted", "chat_multiturn")
print(ds_chat['train'][0]['messages'])

# Load the IndoQA reading comprehension dataset
ds_qa = load_dataset("daruokta/t5gemma2-indonesia-chat-formatted", "indoqa_documents")
print(ds_qa['train'][0]['messages'])
```

## Statistics
- **Total examples**: ~6,900
- **Language**: Indonesian (Bahasa Indonesia)
- **Subsets**: `chat_multiturn`, `indoqa_documents`
- **Format**: Standard ChatML list of dictionaries (`role`, `content`)
- **Created**: May 2026
