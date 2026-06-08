import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, DataCollatorForSeq2Seq
from transformers.trainer_pt_utils import LabelSmoother

tokenizer = AutoTokenizer.from_pretrained("models/t5gemma2-270m-task-vector")
model = AutoModelForSeq2SeqLM.from_pretrained(
    "models/t5gemma2-270m-task-vector",
    trust_remote_code=True,
    torch_dtype=torch.bfloat16,
).cuda()
if getattr(model.config, "decoder_start_token_id", None) is None:
    model.config.decoder_start_token_id = tokenizer.bos_token_id

inp = tokenizer("hello world", return_tensors="pt").to("cuda")
tgt = tokenizer("hello world", return_tensors="pt").to("cuda")

collator = DataCollatorForSeq2Seq(tokenizer, model)
batch = collator(
    [
        {
            "input_ids": inp.input_ids[0].cpu().tolist(),
            "labels": tgt.input_ids[0].cpu().tolist(),
        }
    ]
)
batch = {k: v.to("cuda") for k, v in batch.items()}

labels = batch.pop("labels")
with torch.no_grad():
    outputs = model(**batch)

smoother = LabelSmoother(epsilon=0.1)
loss = smoother(outputs, labels)
print("LabelSmoother loss (default):", loss.item())

# what if we pass shift_labels=True?
try:
    loss2 = smoother(outputs, labels, shift_labels=True)
    print("LabelSmoother loss (shifted):", loss2.item())
except Exception:
    pass

# what about suppress tokens?
suppress = list(range(6, 105)) + list(range(256002, 262144)) + [255999, 256000, 256001]
embed = model.get_input_embeddings().weight
with torch.no_grad():
    mean_val = embed.mean(dim=0)
    embed[suppress] = mean_val

    outputs_supp = model(**batch)
loss_supp = smoother(outputs_supp, labels)
print("Loss with suppressed tokens:", loss_supp.item())
