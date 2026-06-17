#!/usr/bin/env python
# coding: utf-8

# # Welcome to Modal notebooks!
# 
# Write Python code and collaborate in real time. Your code runs in Modal's
# **serverless cloud**, and anyone in the same workspace can join.
# 
# This notebook comes with some common Python libraries installed. Run
# cells with `Shift+Enter`.

# In[8]:


# %uv pip install -U accelerate datasets pillow sentencepiece safetensors pillow transformers --force-reinstall
# %uv pip install -U torch torchvision --index-url https://download.pytorch.org/whl/cu129
# %uv pip install --no-deps trl 


# In[1]:


def get_ipython():
    class _MockIPython:
        def run_line_magic(self, *args, **kwargs):
            pass
    return _MockIPython()


get_ipython().run_line_magic('uv', 'pip list')


# In[2]:


import torch
from datasets import load_dataset
from transformers import AutoProcessor, AutoModelForSeq2SeqLM
from transformers import Seq2SeqTrainingArguments, Seq2SeqTrainer, set_seed


# In[3]:


set_seed(42)

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

print("CUDA:", torch.cuda.is_available(), torch.cuda.get_device_name(0) if torch.cuda.is_available() else None)
print("bf16 supported:", torch.cuda.is_available() and torch.cuda.is_bf16_supported())


# In[5]:


DATASET_NAME = "full"
raw_train = load_dataset("linxy/LaTeX_OCR", name=DATASET_NAME, split="train")
raw_val   = load_dataset("linxy/LaTeX_OCR", name=DATASET_NAME, split="validation")

train_ds = raw_train.shuffle(seed=42).select(range(1000))
val_ds   = raw_val.shuffle(seed=42).select(range(200)) 

print(train_ds, val_ds)
print("Columns:", train_ds.column_names)


# In[7]:


train_ds[10]["image"]


# In[8]:


train_ds[10]["text"]


# In[9]:


from IPython.display import display, Math, Latex

latex = train_ds[10]["text"]
display(Math(latex))


# In[10]:


MODEL_ID = "google/t5gemma-2-270m-270m"

processor = AutoProcessor.from_pretrained(MODEL_ID, use_fast=True)
model = AutoModelForSeq2SeqLM.from_pretrained(
    MODEL_ID,
    dtype=torch.bfloat16,   # A100 -> bf16
    device_map="auto",
)


# In[11]:


tokenizer = processor.tokenizer
if tokenizer.pad_token is None:
    tokenizer.add_special_tokens({"pad_token": tokenizer.eos_token})
    model.resize_token_embeddings(len(tokenizer))


# In[12]:


PROMPT = "<start_of_image> Convert this image to LaTeX. Output only LaTeX."
MAX_INPUT_LEN  = 128
MAX_TARGET_LEN = 256


# In[13]:


image = train_ds[20]["image"]

# prepare inputs
model_inputs = processor(text=PROMPT, images=image, return_tensors="pt")
model_inputs = {k: v.to("cuda") for k, v in model_inputs.items()}

# run inference
model = model.eval()
with torch.inference_mode():
    generation = model.generate(
        **model_inputs,
        max_new_tokens=100,
        do_sample=False,
        num_beams=3,    
        repetition_penalty=1.2,
        no_repeat_ngram_size=4,
        early_stopping=True,
    )

# decode
pred = processor.decode(generation[0], skip_special_tokens=True)

print("\n--- Model output ---")
print(pred)


# In[15]:


train_ds[20]["text"]


# In[16]:


from typing import Any, Dict, List
import torch
from PIL import Image as PILImage

tokenizer = processor.tokenizer
tokenizer.padding_side = "right"

if tokenizer.pad_token is None:
    tokenizer.add_special_tokens({"pad_token": tokenizer.eos_token})

pad_id = tokenizer.pad_token_id

PROMPT = "<start_of_image> Convert this equation image to LaTeX. Output only LaTeX."
MAX_TARGET_LEN = 256

def collate_fn(examples: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
    images, prompts, targets = [], [], []

    for ex in examples:
        im = ex["image"]
        if isinstance(im, PILImage.Image):
            im = im.convert("RGB")
        elif isinstance(im, dict) and "path" in im:
            im = PILImage.open(im["path"]).convert("RGB")
        else:
            raise ValueError(f"Unexpected image type: {type(im)}")

        # IMPORTANT: one image per sample -> nested list
        images.append([im])

        prompts.append(PROMPT)
        targets.append(ex["text"])

    # ✅ NO truncation here (prevents image-token mismatch)
    model_inputs = processor(
        text=prompts,
        images=images,
        padding=True,
        truncation=False,
        return_tensors="pt",
    )

    labels = tokenizer(
        targets,
        padding=True,
        truncation=True,
        max_length=MAX_TARGET_LEN,
        return_tensors="pt",
    )["input_ids"]

    labels[labels == pad_id] = -100
    model_inputs["labels"] = labels
    return model_inputs


# In[17]:


from transformers import Seq2SeqTrainingArguments

args = Seq2SeqTrainingArguments(
    output_dir="t5gemma2-latex-ocr-1k",

    # --- core training ---
    num_train_epochs=1,
    per_device_train_batch_size=8, 
    gradient_accumulation_steps=1,

    learning_rate=1e-4,
    warmup_steps=15,
    lr_scheduler_type="linear",

    # --- precision / speed ---
    bf16=True,    
    fp16=False,
    tf32=True,     

    # --- memory ---
    gradient_checkpointing=True,

    # --- stop extra work (this is the big speed win) ---
    eval_strategy="no",           
    predict_with_generate=False,          
    save_strategy="no",                 
    report_to="none",

    # --- dataloader ---
    dataloader_num_workers=0,         
    remove_unused_columns=False,

    # --- logging ---
    logging_steps=10,
)


# In[18]:


trainer = Seq2SeqTrainer(
    model=model,
    args=args,
    train_dataset=train_ds,
    eval_dataset=val_ds,
    data_collator=collate_fn,
)


# In[20]:


trainer.train()


# In[21]:


# pick a sample
image = train_ds[10]["image"]

# prepare inputs
model_inputs = processor(text=PROMPT, images=image, return_tensors="pt")
model_inputs = {k: v.to("cuda") for k, v in model_inputs.items()}

# run inference
model = model.eval()
with torch.inference_mode():
    generation = model.generate(
        **model_inputs,
        max_new_tokens=100,
        do_sample=False,
        num_beams=3,    
        repetition_penalty=1.2,
        no_repeat_ngram_size=4,
        early_stopping=True,
    )

# decode
pred = processor.decode(generation[0], skip_special_tokens=True)
print(pred)


# In[22]:


image = val_ds[10]["image"]

# prepare inputs
model_inputs = processor(text=PROMPT, images=image, return_tensors="pt")
model_inputs = {k: v.to("cuda") for k, v in model_inputs.items()}

# run inference
model = model.eval()
with torch.inference_mode():
    generation = model.generate(
        **model_inputs,
        max_new_tokens=100,
        do_sample=False,
        num_beams=3,    
        repetition_penalty=1.2,
        no_repeat_ngram_size=4,
        early_stopping=True,
    )

# decode
pred = processor.decode(generation[0], skip_special_tokens=True)
print(pred)


# In[23]:


print(val_ds[10]["text"])


# In[ ]:


trainer.save_model()

