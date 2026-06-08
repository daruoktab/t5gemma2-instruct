from huggingface_hub import HfApi

markdown_content = """---
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
  - config_name: chat_full
    data_files:
      - split: train
        path: chat_full/train-*
      - split: validation
        path: chat_full/validation-*
  - config_name: chat_seed
    data_files:
      - split: train
        path: chat_seed/train-*
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

This dataset contains over 15,000 multi-turn conversations and document-based Q&A in Bahasa Indonesia. It covers diverse topics including everyday life, technology, general knowledge, and structured document analysis. The dataset is pre-formatted into standard OpenAI/ChatML message lists, making it instantly ready for fine-tuning.

## Dataset Structure

The repository is divided into three distinct subsets (configurations):

1. **`chat_seed`**: The original 1,030 manually curated multi-turn conversational seed data.
2. **`chat_full`**: An expanded dataset of ~10,000 augmented multi-turn casual and formal conversations.
3. **`indoqa_documents`**: ~4,000 reading comprehension and factual Q&A examples derived from Indonesian documents.

Each entry contains a single column `messages` which is a list of message objects:
- `role`: The speaker role (`system`, `user`, or `assistant`).
- `content`: The actual text of the dialogue.

## Usage

You can easily load a specific subset using the Hugging Face `datasets` library:

```python
from datasets import load_dataset

# Load the full conversational dataset
ds_chat = load_dataset("daruokta/t5gemma2-indonesia-chat-formatted", "chat_full")
print(ds_chat['train'][0]['messages'])

# Load the IndoQA reading comprehension dataset
ds_qa = load_dataset("daruokta/t5gemma2-indonesia-chat-formatted", "indoqa_documents")
print(ds_qa['train'][0]['messages'])
```

## Statistics
- **Total examples**: ~15,000
- **Language**: Indonesian (Bahasa Indonesia)
- **Subsets**: `chat_seed`, `chat_full`, `indoqa_documents`
- **Format**: Standard ChatML list of dictionaries (`role`, `content`)
- **Created**: May 2026
"""

api = HfApi()
with open('temp_readme.md', 'w', encoding='utf-8') as f:
    f.write(markdown_content)

api.upload_file(
    path_or_fileobj='temp_readme.md',
    path_in_repo='README.md',
    repo_id='daruokta/t5gemma2-indonesia-chat-formatted',
    repo_type='dataset'
)
print('✅ README successfully uploaded with raw YAML!')
