"""Offline smoke tests extracted from v8; never runs notebook cells or Hub operations."""
import ast
import os
import sys
import math
import traceback
from pathlib import Path

os.environ['HF_HUB_OFFLINE'] = '1'
os.environ['TORCH_COMPILE_DISABLE'] = '1'
os.environ['WANDB_DISABLED'] = 'true'
print('PYTHON', sys.executable, flush=True)
if '--fast-metadata' in sys.argv:
    # Diagnostic-only: skip per-file existence checks in Python 3.12 metadata.
    # Does not change installed files. Normal import must be validated separately.
    import importlib.metadata as metadata
    import csv
    import io
    mapping = {}
    for dist in metadata.distributions():
        name = dist.metadata.get('Name')
        if not name:
            continue
        top = (dist.read_text('top_level.txt') or '').split()
        if not top:
            top = set()
            for row in csv.reader(io.StringIO(dist.read_text('RECORD') or '')):
                if not row:
                    continue
                first = row[0].split('/')[0]
                if '/' not in row[0]:
                    first = first.removesuffix('.py').removesuffix('.pyd')
                if first.isidentifier():
                    top.add(first)
        for package in top:
            mapping.setdefault(package, []).append(name)
    metadata.packages_distributions = lambda: mapping
    print('DIAGNOSTIC metadata mapping bypass enabled', flush=True)
import faulthandler
faulthandler.dump_traceback_later(90, repeat=True)
import torch
print('TORCH', torch.__version__, 'CUDA', torch.cuda.is_available(), flush=True)
if '--unsloth' in sys.argv:
    import unsloth
    print('UNSLOTH', unsloth.__file__, flush=True)
from transformers import Seq2SeqTrainer, Seq2SeqTrainingArguments, AutoModelForSeq2SeqLM, T5Gemma2Config
import torch.nn.functional as F
from peft import LoraConfig, TaskType, get_peft_model

tree = ast.parse(Path('notebooks/working-molab-v8.py').read_text(encoding='utf-8'))
ns = dict(torch=torch, F=F, Seq2SeqTrainer=Seq2SeqTrainer, _math=math)
names = {'SelectiveLabelSmoother', 'JointSFTTrainer', 'JointORPOTrainer', 'TLPO_Regularizer', 'Seq2SeqVisionCollator', 'VisionORPOCollator', 'GrokOrScale', 'zeropower_via_newtonschulz5'}
nodes = [n for cell in tree.body if isinstance(cell, ast.FunctionDef) for n in cell.body if isinstance(n, (ast.ClassDef, ast.FunctionDef)) and n.name in names]
exec(compile(ast.Module(body=nodes, type_ignores=[]), '<v8-extracted>', 'exec'), ns)

def check(name, fn):
    try:
        result = fn()
        print('PASS', name, result, flush=True)
    except Exception:
        print('FAIL', name, traceback.format_exc(), flush=True)

text = dict(vocab_size=64, hidden_size=32, intermediate_size=64, num_hidden_layers=1, num_attention_heads=2, num_key_value_heads=1, head_dim=16, layer_types=['full_attention'])
config = T5Gemma2Config(encoder=dict(text_config=text, vision_config=dict(hidden_size=32, intermediate_size=64, num_hidden_layers=1, num_attention_heads=2, image_size=16, patch_size=8), mm_tokens_per_image=4, image_token_index=60, boi_token_index=61, eoi_token_index=62), decoder=text, decoder_start_token_id=0, image_token_index=60)
model = AutoModelForSeq2SeqLM.from_config(config, attn_implementation='eager')
print('MODEL', type(model).__name__, flush=True)
inputs = dict(input_ids=torch.tensor([[2,3,4,1], [2,5,1,0]]), attention_mask=torch.tensor([[1,1,1,1],[1,1,1,0]]), labels=torch.tensor([[6,7,1],[8,1,-100]]))
def forward_backward():
    model.zero_grad()
    out = model(**inputs)
    out.loss.backward()
    return float(out.loss)
check('t5gemma2 forward/backward', forward_backward)
args = Seq2SeqTrainingArguments(output_dir='reports/probe-output', use_cpu=True, report_to='none', label_smoothing_factor=0.1, predict_with_generate=True)
def sft():
    trainer = ns['JointSFTTrainer'](model=model, args=args, suppress_ids=[63])
    loss = trainer.compute_loss(model, dict(inputs))
    loss.backward()
    return float(loss)
check('v8 SFT loss/backward', sft)
args.label_smoothing_factor = 0.0
def orpo():
    trainer = ns['JointORPOTrainer'](model=model, args=args, enable_tlpo=False)
    batch = {k:v for k,v in inputs.items() if k != 'labels'}
    batch.update(chosen_labels=inputs['labels'], rejected_labels=torch.tensor([[9,10,1],[11,1,-100]]))
    loss = trainer.compute_loss(model, batch)
    loss.backward()
    return float(loss)
check('v8 ORPO loss/backward', orpo)
def lora():
    wrapped = get_peft_model(model, LoraConfig(task_type=TaskType.SEQ_2_SEQ_LM, r=2, target_modules=['q_proj','v_proj']))
    out = wrapped(**inputs)
    out.loss.backward()
    return type(wrapped).__name__, float(out.loss)
check('PEFT seq2seq forward/backward', lora)
def tlpo():
    reg = ns['TLPO_Regularizer']({3})
    logits = torch.tensor([[[0.,1.,2.,5.]]], requires_grad=True)
    loss = reg(logits, torch.tensor([[1]]))
    return {'first_loss':float(loss), 'requires_grad':loss.requires_grad}
check('TLPO gradient diagnostic', tlpo)
def optimizer_step():
    p = torch.nn.Parameter(torch.randn(8, 4, device='cuda'))
    opt = ns['GrokOrScale']([p])
    p.square().sum().backward()
    opt.step()
    return bool(torch.isfinite(p).all())
check('v8 GrokOrScale CUDA step', optimizer_step)
def multimodal():
    fresh = AutoModelForSeq2SeqLM.from_config(config, attn_implementation='eager')
    out = fresh(input_ids=torch.tensor([[2,60,60,60,60,3,1]]), attention_mask=torch.ones(1,7,dtype=torch.long), pixel_values=torch.randn(1,3,16,16), labels=torch.tensor([[6,7,1]]))
    out.loss.backward()
    return float(out.loss.detach())
check('tiny vision forward/backward', multimodal)
def source_mask():
    fresh = AutoModelForSeq2SeqLM.from_config(config, attn_implementation='eager').eval()
    with torch.no_grad():
        enc = fresh.get_encoder()(input_ids=inputs['input_ids'], attention_mask=inputs['attention_mask'])
        a = fresh(encoder_outputs=enc, labels=inputs['labels']).logits
        b = fresh(encoder_outputs=enc, attention_mask=inputs['attention_mask'], labels=inputs['labels']).logits
    return {'max_logit_difference':float((a-b).abs().max())}
check('ORPO missing mask numerical diagnostic', source_mask)
def isolated_lora():
    # Diagnostic only: disable unrelated broken optional GPTQ integration in this process.
    import peft.tuners.lora.awq as awq
    import peft.tuners.lora.gptq as gptq
    old_a, old_g = awq.is_gptqmodel_available, gptq.is_gptqmodel_available
    try:
        awq.is_gptqmodel_available = lambda: False
        gptq.is_gptqmodel_available = lambda: False
        fresh = AutoModelForSeq2SeqLM.from_config(config, attn_implementation='eager')
        wrapped = get_peft_model(fresh, LoraConfig(task_type=TaskType.SEQ_2_SEQ_LM, r=2, target_modules=['q_proj','v_proj']))
        out = wrapped(**inputs)
        out.loss.backward()
        return type(wrapped).__name__, float(out.loss.detach())
    finally:
        awq.is_gptqmodel_available, gptq.is_gptqmodel_available = old_a, old_g
check('DIAGNOSTIC LoRA with optional GPTQ bypassed', isolated_lora)
faulthandler.cancel_dump_traceback_later()
print('DONE', flush=True)
