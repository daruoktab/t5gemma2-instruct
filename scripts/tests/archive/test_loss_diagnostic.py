import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from transformers import DataCollatorForSeq2Seq

tokenizer = AutoTokenizer.from_pretrained('models/t5gemma2-270m-task-vector')
model = AutoModelForSeq2SeqLM.from_pretrained('models/t5gemma2-270m-task-vector', trust_remote_code=True, torch_dtype=torch.bfloat16).cuda()

if getattr(model.config, "decoder_start_token_id", None) is None:
    model.config.decoder_start_token_id = tokenizer.bos_token_id

inp = tokenizer('hello world', return_tensors='pt').to('cuda')
tgt = tokenizer('hello world', return_tensors='pt').to('cuda')

# Simulate DataCollator
collator = DataCollatorForSeq2Seq(tokenizer, model)
batch = collator([{"input_ids": inp.input_ids[0].cpu().tolist(), "labels": tgt.input_ids[0].cpu().tolist()}])
batch = {k: v.to('cuda') for k, v in batch.items()}

print("Batch keys:", batch.keys())
print("Labels:", batch["labels"])
if "decoder_input_ids" in batch:
    print("Decoder input ids:", batch["decoder_input_ids"])

with torch.no_grad():
    outputs = model(**batch)

print("Internal Loss:", outputs.loss.item() if hasattr(outputs, "loss") and outputs.loss is not None else "None")

logits = outputs.logits
print("Logits shape:", logits.shape)

loss_fct = torch.nn.CrossEntropyLoss(reduction="mean")
loss_unaligned = loss_fct(logits.view(-1, logits.size(-1)), batch["labels"].view(-1))
print("Standard CE Loss:", loss_unaligned.item())

# Shift labels like Causal LM
shift_logits = logits[..., :-1, :].contiguous()
shift_labels = batch["labels"][..., 1:].contiguous()
loss_shifted = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
print("Shifted CE Loss:", loss_shifted.item())
