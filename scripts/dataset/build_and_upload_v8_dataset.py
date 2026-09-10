#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
BUILD & UPLOAD UNIFIED DATASET V8
==================================
Script ini menyatukan SELURUH sumber dataset ke dalam SATU Repositori Dataset Hugging Face Publik:
1. daruokta/t5gemma2-indonesia-chat-formatted (chat_sft, indoqa_sft, chat_orpo, chat_multiturn)
2. daruokta/t5gemma2-indonesia-vision-formatted (vision_sft, vision_orpo)
3. data/synthetic/generated_conv_agent.jsonl (3.000 percakapan: 2K text + 1K vision, gambar di-embed ke Parquet)
4. aisingapore/SEA-Instruct-2602 (subset Indonesian, filter kualitas Excellent + task routing MTO)

Output Config di Repo Target:
- `joint_sft`   : Split train & validation (siap pakai untuk Phase 1 Joint SFT di Molab/Server)
- `joint_orpo`  : Split train & validation (siap pakai untuk Phase 2 Joint ORPO di Molab/Server)
- `chat_raw`    : Arsip raw multi-turn conversation teks
- `vision_raw`  : Arsip raw multi-turn conversation visual (dengan embedded PIL Images)

Penggunaan:
    conda activate unsloth-env
    python scripts/dataset/build_and_upload_v8_dataset.py --dry-run
    python scripts/dataset/build_and_upload_v8_dataset.py --push --repo-id daruokta/t5gemma2-indonesia-joint-multimodal-v8
"""

import argparse
import ast
import json
import os
import random
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import datasets
from datasets import Dataset, DatasetDict, Features, Image as HFImage, Sequence, Value, load_dataset
from huggingface_hub import HfApi
from PIL import Image as PILImage

# Root directories
ROOT_DIR = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT_DIR / "data"

DEFAULT_TARGET_REPO = "daruokta/t5gemma2-indonesia-joint-multimodal-v8"
DEFAULT_CHAT_REPO = "daruokta/t5gemma2-indonesia-chat-formatted"
DEFAULT_VISION_REPO = "daruokta/t5gemma2-indonesia-vision-formatted"
DEFAULT_SEA_REPO = "aisingapore/SEA-Instruct-2602"
DEFAULT_LOCAL_SYNTH = DATA_DIR / "synthetic" / "generated_conv_agent.jsonl"


# =====================================================================
# 1. MTO TASK PREFIX HELPER
# =====================================================================
def format_mto_target(target_text: str, task_category: str = "general_chat") -> str:
    prefix_mapping = {
        "summarize": "<unused1>",
        "translate": "<unused2>",
        "ner": "<unused3>",
        "qa": "<unused4>",
        "paraphrase": "<unused5>",
        "general_chat": "<unused6>",
        "vision": "<unused6>",
    }
    tag = prefix_mapping.get(task_category.lower(), "<unused6>")
    target_clean = target_text.strip()
    if target_clean.startswith("<unused"):
        return target_clean
    return f"{tag} {target_clean}"


def format_encoder_from_raw(raw_input: str, system_fallback: str = "Kamu adalah asisten AI yang helpful, santai, dan ramah. Gunakan Bahasa Indonesia sebagai bahasa utama.") -> str:
    import re
    system_match = re.search(r"^system:\s*(.*?)(?=\nuser:)", raw_input, re.DOTALL)
    system = system_match.group(1).strip() if system_match else system_fallback

    if system_match:
        raw_input = raw_input[system_match.end() :].strip()

    parts = re.split(r"\n(user:|assistant:)\s*", "\n" + raw_input)
    formatted = ""
    is_first_user = True

    for i in range(1, len(parts), 2):
        role = parts[i].replace(":", "").strip()
        content = parts[i + 1].strip()
        if not content:
            continue

        if role == "user":
            formatted += "<start_of_turn>user\n"
            if is_first_user and system:
                formatted += system + "\n\n"
                is_first_user = False
            formatted += content + "<end_of_turn>\n"
        elif role == "assistant":
            formatted += "<start_of_turn>model\n"
            formatted += content + "<end_of_turn>\n"

    formatted += "<start_of_turn>model\n"
    return formatted


# =====================================================================
# 2. DATA LOADERS & UNROLLERS
# =====================================================================

def unroll_messages_to_sft_rows(
    messages: List[Dict[str, Any]],
    raw_images: Optional[List[Any]] = None,
    source_name: str = "custom",
    conv_id: str = "",
    enable_mto: bool = True,
    default_task: str = "general_chat"
) -> List[Dict[str, Any]]:
    """
    Unroll percakapan multi-turn menjadi baris SFT per-turn asisten.
    Mendukung teks maupun visual dengan gambar terlampir.
    """
    loaded_images = []
    if raw_images:
        for img in raw_images:
            if isinstance(img, PILImage.Image):
                loaded_images.append(img)
            elif isinstance(img, str) and os.path.exists(img):
                try:
                    loaded_images.append(PILImage.open(img).convert("RGB"))
                except Exception:
                    pass

    is_vision = bool(loaded_images) or any("📷" in m.get("content", "") for m in messages if m.get("role") == "user")
    clean_context = []
    image_idx = 0

    for msg in messages:
        role = msg.get("role", "user")
        content = msg.get("content", "")
        if isinstance(content, list):
            clean_context.append({"role": role, "content": content})
            continue

        if role == "user" and "📷" in content:
            n_imgs = content.count("📷")
            text_part = content.replace("📷", "").strip()
            c_blocks = []
            for _ in range(n_imgs):
                c_blocks.append({"type": "image"})
                image_idx += 1
            if text_part:
                c_blocks.append({"type": "text", "text": text_part})
            clean_context.append({"role": role, "content": c_blocks})
        else:
            clean_context.append({"role": role, "content": [{"type": "text", "text": content}]})

    rows = []
    for i, msg in enumerate(clean_context):
        if msg["role"] != "assistant":
            continue
        context = clean_context[:i]
        if not context:
            continue

        # Target text
        target_text = ""
        if isinstance(msg["content"], list):
            for b in msg["content"]:
                if isinstance(b, dict) and "text" in b:
                    target_text = b["text"]
        else:
            target_text = msg["content"]

        if not target_text:
            continue

        task_cat = "vision" if is_vision else default_task
        if enable_mto:
            target_text = format_mto_target(target_text, task_cat)

        # Build prompt_text manual (Gemma format)
        prompt_text = "<bos>"
        is_first_user = True
        system_text = ""
        for m in context:
            r = m["role"]
            if r == "system":
                if isinstance(m["content"], list):
                    system_text = m["content"][0].get("text", "")
                else:
                    system_text = str(m["content"])
            elif r == "user":
                prompt_text += "<start_of_turn>user\n"
                if is_first_user and system_text:
                    prompt_text += system_text + "\n\n"
                    is_first_user = False
                if isinstance(m["content"], list):
                    for b in m["content"]:
                        if b.get("type") == "image":
                            prompt_text += "📷"
                        elif b.get("type") == "text":
                            prompt_text += b.get("text", "")
                else:
                    prompt_text += str(m["content"])
                prompt_text += "<end_of_turn>\n"
            elif r == "assistant" or r == "model":
                prompt_text += "<start_of_turn>model\n"
                if isinstance(m["content"], list):
                    for b in m["content"]:
                        if b.get("type") == "text":
                            prompt_text += b.get("text", "")
                else:
                    prompt_text += str(m["content"])
                prompt_text += "<end_of_turn>\n"
        prompt_text += "<start_of_turn>model\n"

        num_imgs_in_ctx = sum(
            1 for m in context for b in m.get("content", [])
            if isinstance(b, dict) and b.get("type") == "image"
        )
        ctx_images = loaded_images[:num_imgs_in_ctx] if loaded_images else []

        rows.append({
            "prompt_text": prompt_text,
            "target_text": target_text.strip(),
            "images": ctx_images,
            "modality": "vision" if is_vision else "text",
            "source": source_name,
            "task_category": task_cat,
            "conv_id": conv_id,
        })

    return rows


def load_hf_chat_formatted_sft(repo_id: str, enable_mto: bool = True) -> Tuple[List[Dict], List[Dict]]:
    """Muat chat_sft & indoqa_sft dari HF repo daruokta/t5gemma2-indonesia-chat-formatted."""
    print(f"\n[1/4] Memuat dataset chat_sft & indoqa_sft dari {repo_id}...")
    train_rows, val_rows = [], []

    # 1. Chat SFT
    try:
        ds_chat = load_dataset(repo_id, "chat_sft")
        for split_name, target_list in [("train", train_rows), ("validation", val_rows)]:
            if split_name in ds_chat:
                for row in ds_chat[split_name]:
                    inp = format_encoder_from_raw(row["input"])
                    tgt = row["target"]
                    if enable_mto:
                        tgt = format_mto_target(tgt, "general_chat")
                    target_list.append({
                        "prompt_text": inp,
                        "target_text": tgt,
                        "images": [],
                        "modality": "text",
                        "source": f"hf_chat_sft_{split_name}",
                        "task_category": "general_chat",
                        "conv_id": f"chat_{row.get('chat_idx', 0)}",
                    })
        print(f"  ✅ chat_sft loaded: {len(train_rows)} train, {len(val_rows)} val")
    except Exception as e:
        print(f"  ⚠️ Error loading chat_sft: {e}")

    # 2. IndoQA SFT
    try:
        ds_indoqa = load_dataset(repo_id, "indoqa_sft")
        iq_train, iq_val = 0, 0
        for split_name, target_list in [("train", train_rows), ("validation", val_rows)]:
            if split_name in ds_indoqa:
                for row in ds_indoqa[split_name]:
                    inp = format_encoder_from_raw(row["input"])
                    tgt = row["target"]
                    if enable_mto:
                        tgt = format_mto_target(tgt, "qa")
                    target_list.append({
                        "prompt_text": inp,
                        "target_text": tgt,
                        "images": [],
                        "modality": "text",
                        "source": f"hf_indoqa_{split_name}",
                        "task_category": "qa",
                        "conv_id": "indoqa_sft",
                    })
                    if split_name == "train":
                        iq_train += 1
                    else:
                        iq_val += 1
        print(f"  ✅ indoqa_sft loaded: {iq_train} train, {iq_val} val")
    except Exception as e:
        print(f"  ⚠️ Error loading indoqa_sft: {e}")

    return train_rows, val_rows


def load_hf_vision_sft(repo_id: str, enable_mto: bool = True) -> List[Dict]:
    """Muat vision_sft dari HF repo daruokta/t5gemma2-indonesia-vision-formatted."""
    print(f"\n[2/4] Memuat dataset vision_sft dari {repo_id}...")
    rows = []
    try:
        ds = load_dataset(repo_id, "vision_sft", split="train")
        for idx, item in enumerate(ds):
            msgs = item["messages"]
            imgs = item.get("images", [])
            unrolled = unroll_messages_to_sft_rows(
                messages=msgs,
                raw_images=imgs,
                source_name="hf_vision_sft",
                conv_id=f"hf_vis_{item.get('id', idx)}",
                enable_mto=enable_mto,
                default_task="vision"
            )
            rows.extend(unrolled)
        print(f"  ✅ vision_sft loaded & unrolled: {len(rows)} turn-rows")
    except Exception as e:
        print(f"  ⚠️ Error loading vision_sft: {e}")
    return rows


def load_local_synthetic_agentic(file_path: Path, enable_mto: bool = True) -> List[Dict]:
    """Muat 3.000 percakapan dari generated_conv_agent.jsonl (text + vision with images)."""
    print(f"\n[3/4] Memuat dataset sintetis lokal dari {file_path}...")
    if not file_path.exists():
        print(f"  ⚠️ File {file_path} tidak ditemukan!")
        return []

    rows = []
    with open(file_path, "r", encoding="utf-8") as f:
        for idx, line in enumerate(f):
            if not line.strip():
                continue
            try:
                obj = json.loads(line)
            except Exception:
                continue

            cat = obj.get("category", "text_nlu_chat")
            msgs = obj.get("messages", [])
            raw_imgs = obj.get("images", [])
            task = "vision" if cat == "vision_chat" else "general_chat"

            unrolled = unroll_messages_to_sft_rows(
                messages=msgs,
                raw_images=raw_imgs,
                source_name=f"synthetic_{cat}",
                conv_id=f"synth_{obj.get('id', idx)}",
                enable_mto=enable_mto,
                default_task=task
            )
            rows.extend(unrolled)

    print(f"  ✅ generated_conv_agent.jsonl loaded & unrolled: {len(rows)} turn-rows")
    return rows


def load_sea_instruct_streaming(
    repo_id: str = "aisingapore/SEA-Instruct-2602",
    n_samples: int = 10000,
    min_quality: str = "Excellent",
    enable_mto: bool = True,
    seed: int = 42
) -> List[Dict]:
    """Download single parquet shard dan ambil sampel pilihan berkualitas dari aisingapore/SEA-Instruct-2602."""
    if n_samples <= 0:
        return []

    print(f"\n[4/4] Memuat {n_samples} sampel dari {repo_id} (subset: Indonesian, quality: {min_quality})...")
    import pyarrow.parquet as _pq
    from huggingface_hub import hf_hub_download as _hf_download

    task_mapping = {
        "Summarization": "summarize",
        "Translation": "translate",
        "Information_Extraction": "ner",
        "Math_and_Scientific_Problem_Solving": "qa",
        "Reasoning": "qa",
        "Coding_and_Debugging": "qa",
        "Question_Answering": "qa",
        "Paraphrase": "paraphrase",
        "Creative_Writing_and_Generation": "general_chat",
        "Recommendation_and_Advice": "general_chat",
    }

    try:
        shard_path = _hf_download(
            repo_id=repo_id,
            filename="Indonesian/train-00000-of-00012.parquet",
            repo_type="dataset"
        )
        table = _pq.read_table(
            shard_path,
            columns=["conversations_id", "conversations", "prompt_primary_task", "prompt_input_quality", "prompt_is_coherent", "prompt_is_natural"]
        )
        data_dict = table.to_pydict()
        total_rows = len(data_dict["conversations_id"])
        print(f"  Shard train-00000 berhasil dibaca ({total_rows} rows).")
    except Exception as e:
        print(f"  ⚠️ Gagal membaca shard SEA-Instruct: {e}")
        return []

    indices = list(range(total_rows))
    random.seed(seed)
    random.shuffle(indices)

    rows = []
    count = 0
    for idx in indices:
        if count >= n_samples:
            break

        q = data_dict["prompt_input_quality"][idx]
        if min_quality and min_quality.lower() != "all":
            if q != min_quality:
                continue
            if data_dict["prompt_is_coherent"][idx] is False or data_dict["prompt_is_natural"][idx] is False:
                continue

        raw_conv = data_dict["conversations"][idx]
        if not raw_conv:
            continue

        try:
            if isinstance(raw_conv, str):
                try:
                    conv_list = json.loads(raw_conv)
                except Exception:
                    conv_list = ast.literal_eval(raw_conv)
            else:
                conv_list = raw_conv
        except Exception:
            continue

        if not isinstance(conv_list, list) or len(conv_list) < 2:
            continue

        primary_task = data_dict["prompt_primary_task"][idx] or "general_chat"
        task_cat = task_mapping.get(primary_task, "general_chat")
        conv_id = data_dict["conversations_id"][idx] or str(count)

        unrolled = unroll_messages_to_sft_rows(
            messages=conv_list,
            raw_images=[],
            source_name="sea_instruct_2602",
            conv_id=f"sea_{conv_id[:16]}",
            enable_mto=enable_mto,
            default_task=task_cat
        )
        rows.extend(unrolled)
        count += 1

    print(f"  ✅ SEA-Instruct-2602 loaded: {len(rows)} turn-rows (dari {count} conversations)")
    return rows


# =====================================================================
# 3. ORPO DATASET BUILDER
# =====================================================================
def build_joint_orpo_dataset(
    chat_repo: str,
    vision_repo: str,
    enable_mto: bool = True
) -> Dataset:
    """Menggabungkan chat_orpo dan vision_orpo ke dalam format joint ORPO."""
    print("\n[ORPO] Membangun dataset joint ORPO (teks + vision)...")
    orpo_rows = []

    # 1. Chat ORPO (Teks)
    try:
        ds_chat_orpo = load_dataset(chat_repo, "chat_orpo", split="train")
        for row in ds_chat_orpo:
            chosen = row["chosen"].replace("assistant: ", "", 1).strip()
            rejected = row["rejected"].replace("assistant: ", "", 1).strip()
            if chosen.endswith("<end_of_turn>"):
                chosen = chosen[:-len("<end_of_turn>")].strip()
            if rejected.endswith("<end_of_turn>"):
                rejected = rejected[:-len("<end_of_turn>")].strip()

            if enable_mto:
                chosen = format_mto_target(chosen, "general_chat")
                rejected = format_mto_target(rejected, "general_chat")

            orpo_rows.append({
                "prompt_text": format_encoder_from_raw(row["prompt"]),
                "chosen_text": chosen,
                "rejected_text": rejected,
                "images": [],
                "modality": "text",
                "flaw": row.get("flaw", ""),
                "rationale": row.get("rationale", ""),
                "source": "hf_chat_orpo",
            })
        print(f"  ✅ chat_orpo loaded: {len(ds_chat_orpo)} pairs")
    except Exception as e:
        print(f"  ⚠️ Error loading chat_orpo: {e}")

    # 2. Vision ORPO
    try:
        ds_vis_orpo = load_dataset(vision_repo, "vision_orpo", split="train")
        for row in ds_vis_orpo:
            chosen = row["chosen"].strip()
            rejected = row["rejected"].strip()
            if chosen.endswith("<end_of_turn>"):
                chosen = chosen[:-len("<end_of_turn>")].strip()
            if rejected.endswith("<end_of_turn>"):
                rejected = rejected[:-len("<end_of_turn>")].strip()

            if enable_mto:
                chosen = format_mto_target(chosen, "vision")
                rejected = format_mto_target(rejected, "vision")

            raw_imgs = row.get("images", [])
            loaded_imgs = []
            for img in raw_imgs:
                if isinstance(img, PILImage.Image):
                    loaded_imgs.append(img)
                elif isinstance(img, str) and os.path.exists(img):
                    try:
                        loaded_imgs.append(PILImage.open(img).convert("RGB"))
                    except Exception:
                        pass

            orpo_rows.append({
                "prompt_text": row["prompt"],
                "chosen_text": chosen,
                "rejected_text": rejected,
                "images": loaded_imgs,
                "modality": "vision",
                "flaw": row.get("flaw", ""),
                "rationale": row.get("rationale", ""),
                "source": "hf_vision_orpo",
            })
        print(f"  ✅ vision_orpo loaded: {len(ds_vis_orpo)} pairs")
    except Exception as e:
        print(f"  ⚠️ Error loading vision_orpo: {e}")

    features_orpo = Features({
        "prompt_text": Value("string"),
        "chosen_text": Value("string"),
        "rejected_text": Value("string"),
        "images": Sequence(HFImage()),
        "modality": Value("string"),
        "flaw": Value("string"),
        "rationale": Value("string"),
        "source": Value("string"),
    })

    return Dataset.from_list(orpo_rows, features=features_orpo)


# =====================================================================
# 4. MAIN BUILD & UPLOAD PIPELINE
# =====================================================================
def main():
    parser = argparse.ArgumentParser(description="Build and Upload Unified Multimodal Dataset V8 to HuggingFace Hub")
    parser.add_argument("--repo-id", type=str, default=DEFAULT_TARGET_REPO, help="Target HuggingFace Dataset Repo ID")
    parser.add_argument("--sea-samples", type=int, default=10000, help="Jumlah sampel SEA-Instruct-2602 (0 = skip)")
    parser.add_argument("--val-size", type=float, default=0.03, help="Rasio validation split (conversation-level)")
    parser.add_argument("--push", action="store_true", help="Push langsung ke HuggingFace Hub")
    parser.add_argument("--dry-run", action="store_true", help="Cek kompilasi dan statistik tanpa push")
    parser.add_argument("--seed", type=int, default=3407, help="Random seed")

    args = parser.parse_args()
    random.seed(args.seed)

    print("=" * 70)
    print(f"🚀 MEMBANGUN UNIFIED MULTIMODAL DATASET V8 -> {args.repo_id}")
    print("=" * 70)

    # 1. Load All SFT Components
    hf_text_train, hf_text_val = load_hf_chat_formatted_sft(DEFAULT_CHAT_REPO, enable_mto=True)
    hf_vis_rows = load_hf_vision_sft(DEFAULT_VISION_REPO, enable_mto=True)
    synth_rows = load_local_synthetic_agentic(DEFAULT_LOCAL_SYNTH, enable_mto=True)
    sea_rows = load_sea_instruct_streaming(DEFAULT_SEA_REPO, n_samples=args.sea_samples, min_quality="Excellent", enable_mto=True, seed=args.seed)

    # 2. Conversation-level Validation Split for Vision & Synth
    all_new_pool = hf_vis_rows + synth_rows + sea_rows
    conv_groups = {}
    for r in all_new_pool:
        cid = r.get("conv_id", "misc")
        conv_groups.setdefault(cid, []).append(r)

    conv_keys = list(conv_groups.keys())
    random.shuffle(conv_keys)
    n_val_conv = max(10, int(len(conv_keys) * args.val_size))
    val_conv_set = set(conv_keys[:n_val_conv])

    extra_train, extra_val = [], []
    for cid in conv_keys:
        if cid in val_conv_set:
            extra_val.extend(conv_groups[cid])
        else:
            extra_train.extend(conv_groups[cid])

    final_sft_train = hf_text_train + extra_train
    final_sft_val = hf_text_val + extra_val

    random.shuffle(final_sft_train)
    random.shuffle(final_sft_val)

    # Drop conv_id from features before creating Dataset
    for r in final_sft_train:
        r.pop("conv_id", None)
    for r in final_sft_val:
        r.pop("conv_id", None)

    features_sft = Features({
        "prompt_text": Value("string"),
        "target_text": Value("string"),
        "images": Sequence(HFImage()),
        "modality": Value("string"),
        "source": Value("string"),
        "task_category": Value("string"),
    })

    ds_sft_train = Dataset.from_list(final_sft_train, features=features_sft)
    ds_sft_val = Dataset.from_list(final_sft_val, features=features_sft)
    joint_sft_dict = DatasetDict({"train": ds_sft_train, "validation": ds_sft_val})

    print("\n" + "=" * 70)
    print("📊 STATISTIK JOINT SFT V8:")
    print(f"  - Train Split: {len(ds_sft_train)} rows")
    print(f"  - Val Split  : {len(ds_sft_val)} rows")
    print("=" * 70)

    # 3. Build Joint ORPO
    ds_orpo = build_joint_orpo_dataset(DEFAULT_CHAT_REPO, DEFAULT_VISION_REPO, enable_mto=True)
    joint_orpo_dict = DatasetDict({"train": ds_orpo})

    print(f"📊 STATISTIK JOINT ORPO V8: {len(ds_orpo)} pairs")

    if args.dry_run:
        print("\n🔍 Mode --dry-run aktif. Dataset selesai dikompilasi secara lokal.")
        print("Gunakan flag --push untuk mengunggah dataset ke Hugging Face Hub.")
        return

    if args.push:
        print(f"\n🚀 Mengunggah dataset ke Hugging Face Hub: {args.repo_id}...")
        api = HfApi()
        
        # Push joint_sft
        print(f"  Pushing config 'joint_sft'...")
        joint_sft_dict.push_to_hub(args.repo_id, config_name="joint_sft")
        print("  ✅ 'joint_sft' berhasil diunggah!")

        # Push joint_orpo
        print(f"  Pushing config 'joint_orpo'...")
        joint_orpo_dict.push_to_hub(args.repo_id, config_name="joint_orpo")
        print("  ✅ 'joint_orpo' berhasil diunggah!")

        print("\n🎉 SELURUH DATASET V8 BERHASIL DISATUKAN & DIUNGGAH KE HUGGING FACE!")
        print(f"Link Dataset: https://huggingface.co/datasets/{args.repo_id}")


if __name__ == "__main__":
    main()
