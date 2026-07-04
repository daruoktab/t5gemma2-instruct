import torch
from PIL import Image
from datasets import Dataset
from transformers import AutoProcessor
from unsloth import FastLanguageModel
from transformers import Trainer, TrainingArguments

# 1. Load Model and Processor
model_name = "google/t5gemma-2-270m-270m"
print(f"Loading model {model_name}...")
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name = model_name,
    max_seq_length = 1024,
    load_in_4bit = False,
    trust_remote_code = True,
)

processor = AutoProcessor.from_pretrained(model_name)

# 2. Apply LoRA
print("Applying LoRA...")
model = FastLanguageModel.get_peft_model(
    model,
    r = 16,
    target_modules = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    lora_alpha = 16,
    lora_dropout = 0,
    bias = "none",
    use_gradient_checkpointing = "unsloth",
    random_state = 3407,
)

# 3. Create Dataset
print("Creating dataset...")
image = Image.new("RGB", (224, 224), color = "white")
prompt = "<" + "start_of_image" + "> di dalam gambar ini terdapat"
target_text = "lebah"

# Preprocess single example to find shapes and structure
inputs = processor(text=prompt, images=image, return_tensors="pt")
labels = processor(text=target_text, return_tensors="pt").input_ids[0]

# Construct a list of dicts for dataset
# The Trainer expects inputs to be standard python lists/types
data_list = []
for _ in range(4):
    sample = {
        "input_ids": inputs.input_ids[0].tolist(),
        "attention_mask": inputs.attention_mask[0].tolist(),
        "pixel_values": inputs.pixel_values[0].tolist(),
        "labels": labels.tolist(),
    }
    if "token_type_ids" in inputs:
        sample["token_type_ids"] = inputs.token_type_ids[0].tolist()
    data_list.append(sample)

dataset = Dataset.from_list(data_list)

# 4. Custom Data Collator for Seq2Seq Vision
class T5Gemma2VisionCollator:
    def __init__(self, processor):
        self.processor = processor

    def __call__(self, batch):
        input_ids = [torch.tensor(item["input_ids"]) for item in batch]
        attention_mask = [torch.tensor(item["attention_mask"]) for item in batch]
        pixel_values = [torch.tensor(item["pixel_values"]) for item in batch]
        labels = [torch.tensor(item["labels"]) for item in batch]

        input_ids_padded = torch.nn.utils.rnn.pad_sequence(
            input_ids, batch_first=True, padding_value=self.processor.tokenizer.pad_token_id
        )
        attention_mask_padded = torch.nn.utils.rnn.pad_sequence(
            attention_mask, batch_first=True, padding_value=0
        )
        labels_padded = torch.nn.utils.rnn.pad_sequence(
            labels, batch_first=True, padding_value=-100
        )

        if pixel_values[0].ndim == 4:
            pixel_values_collated = torch.cat(pixel_values, dim=0)
        else:
            pixel_values_collated = torch.stack(pixel_values, dim=0)

        collated = {
            "input_ids": input_ids_padded,
            "attention_mask": attention_mask_padded,
            "pixel_values": pixel_values_collated,
            "labels": labels_padded,
        }

        if "token_type_ids" in batch[0]:
            token_type_ids = [torch.tensor(item["token_type_ids"]) for item in batch]
            collated["token_type_ids"] = torch.nn.utils.rnn.pad_sequence(
                token_type_ids, batch_first=True, padding_value=0
            )

        return collated

# 5. Training Arguments
training_args = TrainingArguments(
    output_dir = "./outputs",
    per_device_train_batch_size = 2,
    gradient_accumulation_steps = 1,
    max_steps = 2,  # run 2 steps for validation
    learning_rate = 2e-4,
    fp16 = not torch.cuda.is_bf16_supported(),
    bf16 = torch.cuda.is_bf16_supported(),
    logging_steps = 1,
    optim = "adamw_8bit",
    report_to = "none",
)

# 6. Trainer Setup
trainer = Trainer(
    model = model,
    args = training_args,
    train_dataset = dataset,
    data_collator = T5Gemma2VisionCollator(processor),
)

print("Starting training...")
try:
    trainer.train()
    print("Training completed successfully!")
except Exception as e:
    import traceback
    traceback.print_exc()
    raise e
