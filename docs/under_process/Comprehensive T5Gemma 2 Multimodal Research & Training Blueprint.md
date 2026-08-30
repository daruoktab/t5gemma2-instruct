# 🗺️ Master Blueprint & Algoritma Eksekusi Pipeline V8 (T5Gemma-2 Instruct Multimodal)

> **Status Dokumen:** Master Reference & Implementation Blueprint for Coding Agents  
> **Arsitektur Model Target:** `google/t5gemma-2-4b-4b` (Multimodal Encoder-Decoder Seq2Seq)  
> **Base Donor Models:** `google/gemma-3-4b-pt` (Base) & `google/gemma-3-4b-it` (Instruct)  
> **Target Output Repo (HF):** `daruokta/t5gemma-2-4b-4b-instruct-chat-indo-v8-joint-unsloth`  
> **File Skrip Target:** `notebooks/working-molab-v8-combined-unsloth.py`

---

## 📑 Daftar Isi
1. [Prinsip Desain & Perubahan Kritis: V7 $\rightarrow$ V8](#1-prinsip-desain--perubahan-kritis-v7--v8)
2. [Peta Arsitektur Komponen V8](#2-peta-arsitektur-komponen-v8)
3. [Algoritma Eksekusi End-to-End (Pseudo-code Lengkap)](#3-algoritma-eksekusi-end-to-end-pseudo-code-lengkap)
   - [Phase 0: Environment Setup & Auth Gate](#phase-0-environment-setup--auth-gate)
   - [Phase 0.5: SVD-Purified Layer-Wise Task Vector Steering (DeVec + Ramp-Up)](#phase-05-svd-purified-layer-wise-task-vector-steering-devec--ramp-up)
   - [Phase 1.5: Precision Multimodal Vision Grafting (SigLIP 400M)](#phase-15-precision-multimodal-vision-grafting-siglip-400m)
   - [Phase 1: Multi-Task Joint SFT + Task Prefix + Selective Label Smoothing](#phase-1-multi-task-joint-sft--task-prefix--selective-label-smoothing)
   - [Phase 2: Hybrid Multi-Objective Alignment (Joint ORPO + TLPO Token-Level Regularization)](#phase-2-hybrid-multi-objective-alignment-joint-orpo--tlpo-token-level-regularization)
   - [Phase 3: Unified Single Merge & 4-bit Quantization](#phase-3-unified-single-merge--4-bit-quantization)
   - [Phase 4: Comprehensive Evaluation & Diagnostics](#phase-4-comprehensive-evaluation--diagnostics)
4. [Tabel Konfigurasi & Hyperparameter Eksplisit V8](#4-tabel-konfigurasi--hyperparameter-eksplisit-v8)
5. [Spesifikasi Teknis Modul Baru V8 (Untuk Coding Agent)](#5-spesifikasi-teknis-modul-baru-v8-untuk-coding-agent)
   - [Modul A: OrScaleLM Optimizer](#modul-a-orscalelm-optimizer)
   - [Modul B: DeVec SVD Subspace Filtering](#modul-b-devec-svd-subspace-filtering)
   - [Modul C: TLPO Language Confusion Penalty](#modul-c-tlpo-language-confusion-penalty)
   - [Modul D: MTO (Matching Tasks to Objectives) Prompt Formatting](#modul-d-mto-matching-tasks-to-objectives-prompt-formatting)
6. [Checklist Verifikasi & Panduan Eksekusi Agent](#6-checklist-verifikasi--panduan-eksekusi-agent)

---

## 1. Prinsip Desain & Perubahan Kritis: V7 $\rightarrow$ V8

Pipeline **V8** mempertahankan seluruh fondasi keberhasilan V7 (1-Stage Joint Co-training, Merged Attention safety, Logit Masking, SigLIP Vision Grafting, dan Tokenicer patch) sekaligus menambal batasan teknis V7 menggunakan penemuan 26 paper riset arXiv 2025–2026.

### Matriks Komparasi Fitur V7 vs V8:

| Dimensi Fitur | Pipeline V7 (`working-molab-v7-...`) | Pipeline V8 (`working-molab-v8-...`) | Landasan Riset / Justifikasi |
|---|---|---|---|
| **Phase 0.5 (Steering)** | Layer-Wise Ramp-Up FFN & Norm ($\alpha=0.05, 0.25, 0.08$) | **DeVec SVD Purified + Layer-Wise Ramp-Up** (Shared Subspace Separation $\tau=0.85$) | *TV_DECOMP* ([2512.22511](https://arxiv.org/abs/2512.22511)): Membuang noise interferensi parameter donor |
| **Attention Steering Guard** | $\alpha_{\text{QO}}=0, \alpha_{\text{KV}}=0, \alpha_{\text{QKNORM}}=0$ | $\alpha_{\text{QO}}=0, \alpha_{\text{KV}}=0, \alpha_{\text{QKNORM}}=0$ (Dipertahankan 100%) | *Merged Attention* ($K,V = [X; H]$): $W_k, W_v$ wajib murni untuk stabilitas cross-attention |
| **Vision Grafting** | SigLIP 400M + Projector dari Gemma 3 4B IT | SigLIP 400M + Projector dari Gemma 3 4B IT + FP32-to-BF16 Cast Guard | *Gemma 3 TR* ([2503.19786](https://arxiv.org/abs/2503.19786)): 256 soft tokens visual di token `<image_soft_token>` (`256001`) |
| **Optimizer 2D Matrices** | Muon standar (5-step Newton-Schulz) | **OrScale-LM** (Layer-wise Trust Ratio Scaling + Lazy Calibration $c_{\text{denom}}$) | *ORSCALE* ([2605.07815](https://arxiv.org/abs/2605.07815)): Menghindari *unit mismatch* dan gradient explosion pada matriks LoRA |
| **Optimizer 1D Params** | AdEMAMix ($\beta_1=0.9, \beta_2=0.999, \beta_3=0.9999$) | AdEMAMix (Dipertahankan untuk Norm/Bias/Embed) | *GrokAdEMAMix*: Dual EMA stabil untuk representasi 1D |
| **Data Formatting** | Raw conversation unrolling | **MTO (Matching Tasks to Objectives)** formatting | *MTO* ([2606.24841](https://arxiv.org/abs/2606.24841)): Mask-Filling untuk QA/NER, Map-Phrasal untuk Chat |
| **Task Prefix Routing** | Manual mapping di metadata | **Assistant-Driven Task Prefix** (`<unused1>` s.d. `<unused6>`) injected in decoder targets | *Token Priority* ([2602.01227](https://arxiv.org/abs/2602.01227)) & V6/V7 Spec |
| **Alignment Loss** | Standard ORPO ($\beta=0.1, \epsilon=0$) | **Joint ORPO + TLPO Regularization** (Token-Level Policy Optimization) | *TLPO* ([2604.26553](https://arxiv.org/abs/2604.26553)): Mencegah *language confusion* & *catastrophic forgetting* |
| **LoRA Layer Targeting** | All layers ($0 \dots 33$) | Target All Linear dengan Asymmetric Weighting / Focused Alignment | *Mechanistic Analysis* ([2606.09850](https://arxiv.org/abs/2606.09850)) & *LoRA Tradeoffs* ([2607.25583](https://arxiv.org/abs/2607.25583)) |
| **Logit Masking** | Suppress Block 1, Block 2, Vision tokens kecuali ID 7-12 | Masking forward hook + Vocab Safe Guard (262.144) | *Token Cleaning* ([2502.01968](https://arxiv.org/abs/2502.01968)) |
| **Evaluation Suite** | ROUGE, BLEU, Exact Match, BERTScore | ROUGE, BLEU, BERTScore, **CultureTalk-ID**, Per-Modality Loss & Perplexity | *CultureTalk-ID* ([2607.21016](https://arxiv.org/abs/2607.21016)) |

---

## 2. Peta Arsitektur Komponen V8

```
                                  ┌──────────────────────────────────────────────┐
                                  │      BASE MODEL: google/t5gemma-2-4b-4b      │
                                  │   (Multimodal Seq2Seq, Tied Embed 262.144)   │
                                  └──────────────────────┬───────────────────────┘
                                                         │
                                                         ▼
                                  ┌──────────────────────────────────────────────┐
                                  │  PHASE 0.5: SVD DeVec Task Vector Steering   │
                                  │  Δ = SVD_Filter(Gemma3_IT - Gemma3_Base)     │
                                  │  Layer Ramp: FFN (0.05->0.25->0.08), Norm    │
                                  │  Attn Q/K/V = 0.0 (Preserve Merged Attention)│
                                  └──────────────────────┬───────────────────────┘
                                                         │ (Output: steered/)
                                                         ▼
                                  ┌──────────────────────────────────────────────┐
                                  │   PHASE 1.5: Precision SigLIP 400M Grafting  │
                                  │  Vision Tower + Projector <- Gemma 3 4B IT   │
                                  │  Cast all float32 -> bfloat16, Tokenicer Fix │
                                  └──────────────────────┬───────────────────────┘
                                                         │ (Output: cangkok/)
                                                         ▼
                     ┌───────────────────────────────────┴───────────────────────────────────┐
                     │                                                                       │
                     ▼                                                                       ▼
    ┌───────────────────────────────────┐                                   ┌───────────────────────────────────┐
    │     DATASET MIXING: MULTI-TASK    │                                   │       OPTIMIZER: GrokOrScale      │
    │  - Text Chat SFT (100%)           │                                   │  - 2D Linear/LoRA: OrScaleLM (NS) │
    │  - IndoQA MTO SFT (100%)          │                                   │  - 1D Norms/Bias: AdEMAMix        │
    │  - SigLIP Vision SFT (100%)       │                                   │  - GrokFast Gradient Filter       │
    │  - Hold-out 5% Conv for Eval      │                                   │  - MuonClip Norm Threshold (1.0)  │
    └─────────────────┬─────────────────┘                                   └─────────────────┬─────────────────┘
                      │                                                                       │
                      └───────────────────────────────────┬───────────────────────────────────┘
                                                          │
                                                          ▼
                                  ┌──────────────────────────────────────────────┐
                                  │    PHASE 1: JOINT SFT CO-TRAINING (1 Loop)   │
                                  │  - Task Prefix Injection (<unused1>..<6>)    │
                                  │  - Selective Label Smoothing (eps = 0.1)     │
                                  │  - Logit Mask on ALL_SUPPRESS_IDS            │
                                  │  - Auto Checkpoint Upload & Resume           │
                                  └──────────────────────┬───────────────────────┘
                                                         │ (Output: joint/sft/final_adapter)
                                                         ▼
                                  ┌──────────────────────────────────────────────┐
                                  │     PHASE 2: JOINT ORPO + TLPO ALIGNMENT     │
                                  │  - Split Forward (Encoder -> Decoder chosen/ │
                                  │    rejected) hemat 40% VRAM                  │
                                  │  - Loss: CE(chosen) + beta * OddsRatio       │
                                  │    + lambda_tlpo * TLPO_Loss                 │
                                  │  - Label Smoothing = 0.0 (Mandatory)         │
                                  └──────────────────────┬───────────────────────┘
                                                         │ (Output: joint/orpo/final_adapter)
                                                         ▼
                                  ┌──────────────────────────────────────────────┐
                                  │   PHASE 3: UNIFIED SINGLE MERGE & EXPORT     │
                                  │  - Merge 1x: Base(cangkok) + LoRA(orpo)      │
                                  │  - Final BF16 (merged_bf16/)                 │
                                  │  - Final 4-bit NF4 (quantized_4bit/)         │
                                  └──────────────────────────────────────────────┘
```

---

## 3. Algoritma Eksekusi End-to-End (Pseudo-code Lengkap)

Berikut representasi algoritma kompresi presisi dari seluruh eksekusi pipeline V8 dari awal hingga selesai:

```python
# ==============================================================================
# PIPELINE V8: ALGORITMA EKSEKUSI RUNNABLE BLUEPRINT
# ==============================================================================

def execute_pipeline_v8():
    # --------------------------------------------------------------------------
    # PHASE 0: INITIALIZATION, ENVIRONMENT SETUP & AUTH GATE
    # --------------------------------------------------------------------------
    env.set_cuda_alloc_conf("expandable_segments:True")
    env.disable_torch_compile() # Cegah hard-crash recompile limit pada T5Gemma2
    hf_token = auth.get_write_token_or_fail()
    hf_api = HfApi(token=hf_token)
    hf_api.create_repo(repo_id=CONFIG.UNIFIED_HF_REPO, exist_ok=True)
    
    # Audit Remote Artifacts & Cleanup Incomplete Uploads
    remote_state = hf_api.inspect_repo_state(CONFIG.UNIFIED_HF_REPO)
    remote_state.cleanup_partial_artifacts() # Hapus prefix tanpa upload_complete.json

    # --------------------------------------------------------------------------
    # PHASE 0.5: SVD-PURIFIED LAYER-WISE TASK VECTOR STEERING (DeVec)
    # --------------------------------------------------------------------------
    if not remote_state.steered_complete or CONFIG.STEERING_FORCE:
        print("[PHASE 0.5] Loading models for Steering: Base T5Gemma, Gemma3-Base, Gemma3-IT (CPU)...")
        t5_base = load_model_cpu(CONFIG.BASE_T5_MODEL, dtype=torch.bfloat16)
        g_base = load_model_cpu(CONFIG.GEMMA_BASE_MODEL, dtype=torch.bfloat16)
        g_it = load_model_cpu(CONFIG.GEMMA_IT_MODEL, dtype=torch.bfloat16)
        
        L = min(t5_base.decoder.num_layers, g_it.num_layers) # 34 layers
        
        for l in range(L):
            depth_ratio = l / float(L)
            # Ramp-Up Alpha Scheduling
            if depth_ratio < 0.25:
                alpha_ffn, alpha_norm = 0.05, 0.02
            elif depth_ratio < 0.80:
                alpha_ffn, alpha_norm = 0.25, 0.08
            else:
                alpha_ffn, alpha_norm = 0.08, 0.03
                
            # Steer FFN (gate, up, down) dengan DeVec SVD Denoising
            for proj in ["gate_proj", "up_proj", "down_proj"]:
                w_it = g_it.get_layer(l).mlp.get_proj(proj).weight
                w_base = g_base.get_layer(l).mlp.get_proj(proj).weight
                delta_w = w_it - w_base
                
                # DeVec SVD Purification: pisahkan shared subspace
                delta_purified = devec_svd_purify(delta_w, threshold=0.85)
                
                t5_target = t5_base.decoder.get_layer(l).mlp.get_proj(proj).weight
                t5_target.data.add_(delta_purified, alpha=alpha_ffn)
                
            # Steer RMSNorms
            for norm_name in ["pre_self_attn_layernorm", "post_self_attn_layernorm", 
                              "pre_feedforward_layernorm", "post_feedforward_layernorm"]:
                delta_norm = g_it.get_norm(l, norm_name) - g_base.get_norm(l, norm_name)
                t5_base.decoder.get_norm(l, norm_name).data.add_(delta_norm, alpha=alpha_norm)
                
        # CRITICAL SAFETY: Attention Q/K/V/O dan QK-Norm dibiarkan alpha=0.0 (Preserve Merged Attention)
        
        # Smoke test generation (3 prompt)
        run_smoke_test(t5_base, prompts=["Halo!", "Apa ibu kota Indonesia?", "Ringkas teks ini..."])
        
        # Save, patch tokenizer_config task_prefix_mapping, and verified-atomic upload
        save_and_upload_atomic(t5_base, path_in_repo="steered/", marker="steered_marker")
        del t5_base, g_base, g_it; gc.collect(); torch.cuda.empty_cache()

    # --------------------------------------------------------------------------
    # PHASE 1.5: PRECISION MULTIMODAL VISION GRAFTING (SigLIP 400M)
    # --------------------------------------------------------------------------
    if not remote_state.cangkok_complete or CONFIG.CANGKOK_FORCE:
        print("[PHASE 1.5] Grafting SigLIP Vision Tower & Projector...")
        steered_model = load_model_cpu(CONFIG.UNIFIED_HF_REPO, subfolder="steered/")
        gemma_donor = load_model_cpu(CONFIG.GEMMA_IT_MODEL, dtype=torch.bfloat16)
        
        # Exact weight transplantation: model.encoder.vision_tower & multi_modal_projector
        transplant_vision_weights(source=gemma_donor, target=steered_model)
        
        # Cast all modules & buffers to pure bfloat16
        cast_all_to_bfloat16(steered_model)
        
        # Verification: difference check < 1e-6
        verify_graft_exactness_or_fail(steered_model, gemma_donor)
        
        # Save with full T5Gemma2 processor + patched task_prefix_mapping & atomic upload
        save_and_upload_atomic(steered_model, path_in_repo="cangkok/", marker="cangkok_marker")
        del steered_model, gemma_donor; gc.collect(); torch.cuda.empty_cache()

    # --------------------------------------------------------------------------
    # PHASE 1: MULTI-TASK JOINT SFT CO-TRAINING
    # --------------------------------------------------------------------------
    if not remote_state.sft_complete and CONFIG.RUN_SFT:
        print("[PHASE 1] Preparing Joint SFT Co-Training...")
        # 1. Load Cangkok Base Model in 4-bit with Unsloth FastVisionModel
        model, tokenizer = FastVisionModel.from_pretrained(
            model_name=f"{CONFIG.UNIFIED_HF_REPO}/cangkok",
            load_in_4bit=True,
            use_gradient_checkpointing="unsloth"
        )
        processor = AutoProcessor.from_pretrained(f"{CONFIG.UNIFIED_HF_REPO}/cangkok")
        processor.tokenizer.add_bos_token = False # Prevent Double-BOS
        
        # 2. Setup LoRA: Freeze SigLIP (avoid unsloth merge bug), Full-FT Projector, Target All Linear
        model = FastVisionModel.get_peft_model(
            model,
            finetune_vision_layers=False,
            finetune_language_layers=True,
            modules_to_save=["multi_modal_projector"],
            r=CONFIG.LORA_RANK, # 128
            lora_alpha=CONFIG.LORA_ALPHA, # 128
            lora_dropout=CONFIG.LORA_DROPOUT, # 0.0
            use_rslora=False
        )
        
        # 3. Apply Logit Masking Hook on lm_head (Suppress Unused & Vision except Task Prefixes 7-12)
        apply_logit_mask_hook(model, suppress_ids=CONFIG.ALL_SUPPRESS_IDS)
        
        # 4. Prepare Joint Dataset (Multi-Task Streaming + 5% Conversation Hold-out Eval)
        dataset_sft_train, dataset_sft_eval = build_joint_sft_dataset(
            text_chat_repo=CONFIG.DATASET_TEXT_REPO,
            vision_repo=CONFIG.DATASET_VISION_REPO,
            mto_format=True, # Apply MTO prefixing & denoising templates
            holdout_ratio=0.05,
            seed=CONFIG.SEED
        )
        
        # 5. Build Optimizer: GrokFast Filter -> OrScaleLM (2D LoRA) + AdEMAMix (1D Norms)
        optimizer_sft = build_grok_orscale_optimizer(
            model,
            base_lr=CONFIG.SFT_LEARNING_RATE, # 5e-6
            lr_mults={"encoder": 0.2, "decoder": 0.2, "projector": 0.05, "vision_tower": 0.0},
            orscale_scale=CONFIG.SFT_ORSCALE_SCALE # 20.0
        )
        
        # 6. Train SFT Loop with SelectiveLabelSmoother (eps=0.1) & Resume Protection
        trainer_sft = JointSFTTrainer(
            model=model,
            train_dataset=dataset_sft_train,
            eval_dataset=dataset_sft_eval,
            data_collator=Seq2SeqVisionCollator(processor),
            optimizer=optimizer_sft,
            label_smoothing_factor=0.1,
            callbacks=[HubUploadCallback(stage="sft"), ProgressPlotCallback()]
        )
        trainer_sft.train(resume_from_checkpoint=remote_state.sft_checkpoint)
        
        # 7. Save & Upload final adapter to joint/sft/final_adapter
        save_final_adapter_atomic(trainer_sft, path_in_repo="joint/sft/final_adapter")
        del model, trainer_sft; gc.collect(); torch.cuda.empty_cache()

    # --------------------------------------------------------------------------
    # PHASE 2: HYBRID MULTI-OBJECTIVE ALIGNMENT (JOINT ORPO + TLPO)
    # --------------------------------------------------------------------------
    if not remote_state.orpo_complete and CONFIG.RUN_ORPO:
        print("[PHASE 2] Preparing Joint ORPO + TLPO Preference Tuning...")
        # 1. Load Model with SFT Final Adapter
        model, processor = FastVisionModel.from_pretrained(
            model_name=f"{CONFIG.UNIFIED_HF_REPO}/joint/sft/final_adapter",
            load_in_4bit=True,
            use_gradient_checkpointing="unsloth"
        )
        processor.tokenizer.add_bos_token = False
        apply_logit_mask_hook(model, suppress_ids=CONFIG.ALL_SUPPRESS_IDS)
        cast_all_to_bfloat16(model)
        
        # 2. Build Joint ORPO Dataset (Vision ORPO + Chat ORPO)
        dataset_orpo_train, dataset_orpo_eval = build_joint_orpo_dataset(
            text_orpo_repo=CONFIG.DATASET_TEXT_REPO,
            vision_orpo_repo=CONFIG.DATASET_VISION_REPO,
            eval_source=dataset_sft_eval # Apple-to-apple eval benchmark
        )
        
        # 3. Build Optimizer: GrokFast + OrScaleLM (Gentle Scale for Preference)
        optimizer_orpo = build_grok_orscale_optimizer(
            model,
            base_lr=CONFIG.ORPO_LEARNING_RATE, # 5e-6
            lr_mults={"encoder": 0.5, "decoder": 1.0, "projector": 1.0, "vision_tower": 0.5},
            orscale_scale=CONFIG.ORPO_ORSCALE_SCALE # 5.0
        )
        
        # 4. Train ORPO Loop (Split Forward + Odds-Ratio + TLPO Language Confusion Penalty)
        # Note: Label smoothing MUST be 0.0
        trainer_orpo = JointORPOTLPO_Trainer(
            model=model,
            beta=CONFIG.ORPO_BETA, # 0.1
            tlpo_weight=CONFIG.TLPO_WEIGHT, # 0.05
            train_dataset=dataset_orpo_train,
            eval_dataset=dataset_orpo_eval,
            data_collator=VisionORPOCollator(processor),
            optimizer=optimizer_orpo,
            label_smoothing_factor=0.0, # WAJIB 0.0
            callbacks=[HubUploadCallback(stage="orpo"), ProgressPlotCallback()]
        )
        trainer_orpo.train(resume_from_checkpoint=remote_state.orpo_checkpoint)
        
        # 5. Save & Upload final adapter to joint/orpo/final_adapter
        save_final_adapter_atomic(trainer_orpo, path_in_repo="joint/orpo/final_adapter")
        del model, trainer_orpo; gc.collect(); torch.cuda.empty_cache()

    # --------------------------------------------------------------------------
    # PHASE 3: UNIFIED SINGLE MERGE & 4-BIT QUANTIZATION EXPORT
    # --------------------------------------------------------------------------
    if not remote_state.final_complete:
        print("[PHASE 3] Executing 1x Unified Merge & Export...")
        adapter_path = f"{CONFIG.UNIFIED_HF_REPO}/joint/orpo/final_adapter"
        model_to_merge, tokenizer = FastVisionModel.from_pretrained(
            model_name=adapter_path,
            load_in_4bit=True,
            token=hf_token
        )
        processor = AutoProcessor.from_pretrained(f"{CONFIG.UNIFIED_HF_REPO}/cangkok")
        
        # Patch unsloth_zoo unmatched_keys bug if present
        patch_unsloth_zoo_saving_utils()
        
        # Cast all remaining fp32 modules to bfloat16
        cast_all_to_bfloat16(model_to_merge)
        
        # 1. Export BF16 Merged Model
        print("  Exporting final/merged_bf16 ...")
        model_to_merge.save_pretrained_merged("results/final/merged_bf16", tokenizer, save_method="merged_16bit")
        upload_folder_atomic(hf_api, CONFIG.UNIFIED_HF_REPO, "results/final/merged_bf16", "final/merged_bf16")
        
        # 2. Export 4-bit NF4 Quantized Model
        print("  Exporting final/quantized_4bit ...")
        model_to_merge.save_pretrained_merged("results/final/quantized_4bit", tokenizer, save_method="merged_4bit_forced")
        upload_folder_atomic(hf_api, CONFIG.UNIFIED_HF_REPO, "results/final/quantized_4bit", "final/quantized_4bit")

    # --------------------------------------------------------------------------
    # PHASE 4: FINAL VALIDATION & METRIC REPORTING
    # --------------------------------------------------------------------------
    run_final_multimodal_evaluation(CONFIG.UNIFIED_HF_REPO)
    print("🎯 [SUCCESS] Pipeline V8 Complete! Artifacts deployed on HuggingFace Hub.")
```

---

## 4. Tabel Konfigurasi & Hyperparameter Eksplisit V8

| Parameter Group | Variabel Konfigurasi | Nilai Default V8 | Deskripsi & Rujukan Paper |
|---|---|---|---|
| **Repositories** | `UNIFIED_HF_REPO` | `daruokta/t5gemma-2-4b-4b-instruct-chat-indo-v8-joint-unsloth` | Unified destination repository di Hugging Face |
| | `DATASET_TEXT_REPO` | `daruokta/t5gemma2-indonesia-chat-formatted` | Dataset chat & IndoQA bilingual |
| | `DATASET_VISION_REPO` | `daruokta/t5gemma2-indonesia-vision-formatted` | Dataset visual multimodal (SigLIP-formatted) |
| **Model Sources** | `BASE_T5_MODEL` | `google/t5gemma-2-4b-4b` | Base Seq2Seq 4B-4B |
| | `GEMMA_BASE_MODEL` | `google/gemma-3-4b-pt` | Base donor model untuk steering |
| | `GEMMA_IT_MODEL` | `google/gemma-3-4b-it` | Instruct donor model untuk steering & vision graft |
| **Phase 0.5 Steering** | `STEERING_ALPHA_FFN_EARLY` | `0.05` | Layer $0 \dots 8$ ($<25\%$ depth) |
| | `STEERING_ALPHA_FFN_MID` | `0.25` | Layer $9 \dots 26$ ($25\% \dots 80\%$ depth — peak reasoning) |
| | `STEERING_ALPHA_FFN_LATE` | `0.08` | Layer $27 \dots 33$ ($>80\%$ depth — output calibration) |
| | `STEERING_ALPHA_NORM_EARLY`| `0.02` | Scale RMSNorm layer awal |
| | `STEERING_ALPHA_NORM_MID`  | `0.08` | Scale RMSNorm layer tengah |
| | `STEERING_ALPHA_NORM_LATE` | `0.03` | Scale RMSNorm layer akhir |
| | `STEERING_ALPHA_QO / KV`   | `0.0` (WAJIB) | Mencegah kerusakan proyeksi joint $[X; H]$ |
| | `DEVEC_SVD_THRESHOLD`      | `0.85` | Eigendecomposition threshold pemisahan shared subspace |
| **LoRA Configuration** | `LORA_RANK` ($r$) | `128` | Rank LoRA optimal (*LORA_TRADEOFFS* 2607.25583) |
| | `LORA_ALPHA` ($\alpha$) | `128` | Skala LoRA (rasio $\alpha/r = 1.0$) |
| | `LORA_DROPOUT` | `0.0` | Dropout 0 untuk konsistensi deterministik |
| | `LORA_TARGET_MODULES` | All Linear (`q,k,v,o,gate,up,down`)| Target semua modul linier encoder & decoder |
| | `MODULES_TO_SAVE` | `["multi_modal_projector"]` | Full fine-tuning pada projector multimodal |
| **Phase 1 (SFT)** | `SFT_LEARNING_RATE` | `5e-6` | Base learning rate SFT |
| | `SFT_NUM_EPOCHS` | `2` | Jumlah epoch training |
| | `SFT_BATCH_SIZE` (per dev)| `4` (Train), `16` (Eval) | Batch size per GPU device |
| | `SFT_GRAD_ACC_STEPS` | `16` | Effective batch size = $4 \times 16 = 64$ |
| | `SFT_LABEL_SMOOTHING` | `0.1` | Selective label smoother on valid tokens |
| | `SFT_NEFTUNE_ALPHA` | `5.0` | NEFTune embedding noise |
| | `SFT_LR_MULT_ENCODER` | `0.2` | Effective LR Encoder = $1 \times 10^{-6}$ |
| | `SFT_LR_MULT_DECODER` | `0.2` | Effective LR Decoder = $1 \times 10^{-6}$ |
| | `SFT_LR_MULT_PROJECTOR`| `0.05`| Effective LR Projector = $2.5 \times 10^{-7}$ |
| | `SFT_LR_MULT_VISION` | `0.0` (Frozen) | SigLIP vision tower di-freeze |
| **Phase 2 (ORPO + TLPO)**| `ORPO_BETA` | `0.1` | Koefisien penalti Odds-Ratio |
| | `TLPO_WEIGHT` ($\lambda$) | `0.05` | Koefisien penalti token-level language confusion |
| | `ORPO_LEARNING_RATE` | `5e-6` | Base learning rate ORPO |
| | `ORPO_NUM_EPOCHS` | `1` | Jumlah epoch ORPO |
| | `ORPO_LABEL_SMOOTHING` | `0.0` (WAJIB) | Dilarang memakai label smoothing pada ORPO |
| | `ORPO_LR_MULT_DECODER` | `1.0` | Decoder alignment LR = $5 \times 10^{-6}$ |
| **Optimizer (OrScale)** | `OPTIMIZER_TYPE` | `grokorscale` | GrokFast + OrScaleLM (2D) + AdEMAMix (1D) |
| | `GROK_ALPHA` / `LAMB` | `2.0` / `0.98` | GrokFast amplification & momentum decay |
| | `ORSCALE_NS_STEPS` | `5` | Iterasi quintic Newton-Schulz polar decomposition |
| | `SFT_ORSCALE_SCALE` | `20.0` | Effective OrScale Decoder LR = $2 \times 10^{-5}$ |
| | `ORPO_ORSCALE_SCALE` | `5.0` | Effective OrScale ORPO LR = $2.5 \times 10^{-5}$ |
| | `ORSCALE_R_MIN / R_MAX`| `0.1` / `5.0` | Clamping boundary per-layer trust ratio |
| **Token Masking** | `ALL_SUPPRESS_IDS` | Block 1 (6, 13-104), Block 2 (256002-262143), Vision (255999-256001) | Token ID 7 s.d. 12 `<unused1>`..`<unused6>` TIDAK di-mask |

---

## 5. Spesifikasi Teknis Modul Baru V8 (Untuk Coding Agent)

Dokumen ini menyediakan implementasi kelas Python konkret untuk 4 modul baru yang harus disuntikkan ke skrip `notebooks/working-molab-v8-combined-unsloth.py`.

### Modul A: `OrScaleLM` Optimizer (Matrix Layers Trust Ratio Scaling)
*Landasan: arXiv 2605.07815 (NUS)*

```python
import math
import torch
from torch.optim import Optimizer

def zeropower_via_newtonschulz5(G: torch.Tensor, steps: int = 5, eps: float = 1e-7) -> torch.Tensor:
    """Quintic Newton-Schulz iteration for polar factor approximation."""
    assert G.ndim == 2, f"Tensor 2D required, got {G.ndim}D"
    a, b, c = 3.4445, -4.7750, 2.0315
    X = G.to(torch.float32)
    norm = X.norm() + eps
    X = X / norm
    if X.size(0) < X.size(1): X = X.T
    for _ in range(steps):
        A = X @ X.T
        B = b * A + c * (A @ A)
        X = a * X + B @ X
    if G.size(0) < G.size(1): X = X.T
    scale = max(1.0, (G.size(0) / G.size(1)) ** 0.5)
    return (X * scale).to(G.dtype)

class OrScaleLM(Optimizer):
    """
    OrScale-LM: Orthogonalised Optimization with Layer-Wise Trust-Ratio Scaling.
    Applies Newton-Schulz polar updates coupled with Frobenius trust ratio scaling.
    """
    def __init__(self, params, lr=1e-4, weight_decay=0.01, momentum=0.95, nesterov=True, 
                 ns_steps=5, r_min=0.1, r_max=5.0, grok_alpha=2.0, grok_lamb=0.98, max_grad_norm=1.0):
        defaults = dict(lr=lr, weight_decay=weight_decay, momentum=momentum, nesterov=nesterov,
                        ns_steps=ns_steps, r_min=r_min, r_max=r_max, grok_alpha=grok_alpha,
                        grok_lamb=grok_lamb, max_grad_norm=max_grad_norm)
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure=None):
        loss = closure() if closure is not None else None
        for group in self.param_groups:
            lr, wd = group["lr"], group["weight_decay"]
            mu, nesterov = group["momentum"], group["nesterov"]
            steps, r_min, r_max = group["ns_steps"], group["r_min"], group["r_max"]
            grok_alpha, grok_lamb = group["grok_alpha"], group["grok_lamb"]
            max_grad_norm = group["max_grad_norm"]

            for p in group["params"]:
                if p.grad is None or p.ndim != 2:
                    continue
                grad = p.grad
                state = self.state[p]

                if len(state) == 0:
                    state["step"] = 0
                    state["grok_slow"] = torch.zeros_like(grad)
                    state["momentum_buf"] = torch.zeros_like(grad)
                    state["c_denom"] = None

                state["step"] += 1
                
                # 1. GrokFast Gradient Filtering
                state["grok_slow"].mul_(grok_lamb).add_(grad, alpha=1.0 - grok_lamb)
                filtered_grad = grad.clone().add_(state["grok_slow"], alpha=grok_alpha)

                if max_grad_norm > 0:
                    f_norm = filtered_grad.norm()
                    if f_norm > max_grad_norm:
                        filtered_grad.mul_(max_grad_norm / (f_norm + 1e-6))

                # 2. Momentum Buffer Update
                buf = state["momentum_buf"]
                buf.mul_(mu).add_(filtered_grad)
                g_update = filtered_grad.add(buf, alpha=mu) if nesterov else buf

                # 3. Newton-Schulz Polar Update Q
                Q = zeropower_via_newtonschulz5(g_update, steps=steps)

                # 4. Moonlight Shape Factor s_l = 0.2 * sqrt(max(m, n))
                m, n = p.shape[-2], p.shape[-1]
                s_l = 0.2 * math.sqrt(max(m, n))

                # 5. Real Update Direction D_l = wd * W + s_l * Q
                D_l = wd * p + s_l * Q

                # 6. Lazy Calibration Constant at t=1
                p_norm = p.norm(p="fro")
                D_norm = D_l.norm(p="fro") + 1e-6
                if state["c_denom"] is None:
                    state["c_denom"] = (p_norm / D_norm).item()

                # 7. Trust Ratio Scaling
                r_raw = p_norm / (state["c_denom"] * D_norm + 1e-6)
                r_hat = torch.clamp(r_raw, r_min, r_max)

                # 8. Parameter Update
                p.data.add_(D_l, alpha=-lr * r_hat)
        return loss
```

---

### Modul B: `DeVec` SVD Subspace Filtering (Task Vector Decomposition)
*Landasan: arXiv 2512.22511 (Adelaide & Monash)*

```python
def devec_svd_purify(delta_weight: torch.Tensor, threshold: float = 0.85) -> torch.Tensor:
    """
    Decomposes a delta weight matrix into Shared and Unique subspaces via SVD.
    Returns the purified task component with reduced interference.
    """
    if delta_weight.ndim != 2:
        return delta_weight
    
    orig_dtype = delta_weight.dtype
    W = delta_weight.float()
    
    # SVD decomposition
    U, S, Vh = torch.linalg.svd(W, full_matrices=False)
    
    # Column projection P = U @ U^T
    P = U @ U.mH
    
    # Eigendecomposition of projection subspace
    eigenvalues, eigenvectors = torch.linalg.eig(P)
    eigenvalues = eigenvalues.real
    eigenvectors = eigenvectors.real
    
    # Mask shared subspace where eigenvalue > threshold
    shared_mask = eigenvalues > threshold
    if not shared_mask.any():
        shared_mask = (eigenvalues == eigenvalues.max())
        
    Z_shared = eigenvectors[:, shared_mask]
    P_shared = Z_shared @ Z_shared.mH
    
    # Purified delta = W_shared (dominant coherent signal)
    W_purified = P_shared @ W
    return W_purified.to(orig_dtype)
```

---

### Modul C: `TLPO` Regularization (Token-Level Language Confusion Penalty)
*Landasan: arXiv 2604.26553 (Samsung SDS)*

```python
class TLPO_Regularizer:
    """
    Token-Level Policy Optimization penalty to eliminate Bahasa Indonesia language confusion.
    """
    def __init__(self, target_vocab_set: set[int], beta: float = 0.1, clip_eps: float = 0.2):
        self.target_vocab_set = target_vocab_set
        self.beta = beta
        self.clip_eps = clip_eps

    def __call__(self, logits: torch.Tensor, labels: torch.Tensor, old_logits: torch.Tensor | None = None):
        preds = torch.argmax(logits, dim=-1)
        # Confusion point mask: tokens generated outside target Indonesian vocab
        target_tensor = torch.tensor(list(self.target_vocab_set), device=logits.device, dtype=torch.long)
        confusion_mask = ~torch.isin(preds, target_tensor) & (labels != -100)
        
        if not confusion_mask.any():
            return torch.tensor(0.0, device=logits.device, requires_grad=True)
        
        probs = torch.nn.functional.softmax(logits, dim=-1)
        active_probs = probs[confusion_mask]
        
        # Penalty reward: -1.0 on confusion tokens
        top_k_probs, top_k_indices = torch.topk(active_probs, k=min(16, active_probs.size(-1)), dim=-1)
        rewards = torch.isin(top_k_indices, target_tensor).float() * 2.0 - 1.0
        
        # Advantage-weighted policy penalty
        weighted_mean = torch.sum(top_k_probs * rewards, dim=-1, keepdim=True) / (torch.sum(top_k_probs, dim=-1, keepdim=True) + 1e-8)
        advantages = top_k_probs * (rewards - weighted_mean)
        norm_factor = torch.sum(torch.abs(advantages), dim=-1, keepdim=True) + 1e-8
        advantages = advantages / norm_factor
        
        tlpo_loss = -torch.mean(top_k_probs * advantages)
        return self.beta * tlpo_loss
```

---

### Modul D: MTO (Matching Tasks to Objectives) Prompt Formatting
*Landasan: arXiv 2606.24841 (Univ. of Tehran)*

```python
def format_mto_record(record: dict, task_category: str) -> dict:
    """
    Formats input prompt and target based on Tehran MTO taxonomy:
    - Mask-Filling (Summarize, QA, NER) -> Denoising prefix format
    - Map-Phrasal (Chat, Paraphrase, Translation) -> PrefixLM format
    """
    raw_prompt = record.get("prompt_text", "")
    target = record.get("target_text", "").strip()
    
    if task_category == "SUMMARIZE":
        prefix_tag = "<unused1>"
    elif task_category == "TRANSLATE":
        prefix_tag = "<unused2>"
    elif task_category == "NER":
        prefix_tag = "<unused3>"
    elif task_category == "QA":
        prefix_tag = "<unused4>"
    elif task_category == "PARAPHRASE":
        prefix_tag = "<unused5>"
    else: # GENERAL_CHAT
        prefix_tag = "<unused6>"
        
    # Injected prefix into Assistant's target start
    formatted_target = f"{prefix_tag} {target}"
    record["target_text"] = formatted_target
    return record
```

---

## 6. Checklist Verifikasi & Panduan Eksekusi Agent

Ketika agent koding mulai memproses kode `notebooks/working-molab-v8-combined-unsloth.py`:

- [ ] **1. Header & Metadata:** Pastikan deklarasi script menggunakan `requires-python = ">=3.10"` dan paket dependencies termutakhir (Unsloth, bitsandbytes, flash_attn cu132).
- [ ] **2. Control Center Setup:** Pastikan semua konstanta V8 di cell Control Center didefinisikan dengan jelas (`UNIFIED_HF_REPO`, alpha layer-ramp, `LORA_RANK=128`, `OPTIMIZER_TYPE="grokorscale"`).
- [ ] **3. Preservasi DAG Marimo:** Jangan gunakan variable scope global yang bocor antar fungsi. Setiap cell marimo harus mengembalikan tupel variabel output secara eksplisit.
- [ ] **4. Fail-Hard Auth & Upload Gate:** Semua upload model ke Hugging Face wajib menggunakan helper `upload_folder_atomic()` dengan file marker `upload_complete.json`. Jika upload gagal di tengah, hapus prefix remote parsial dan lempar exception.
- [ ] **5. Logit Masking Hook:** Pastikan hook pada `lm_head` menolak token ID `<unused0>` dan `<unused7>` s.d. `<unused6241>`, tetapi **membiarkan ID 7 s.d. 12 (`<unused1>` hingga `<unused6>`) tetap aktif**.
- [ ] **6. No-Op Torch Compile Monkeypatch:** Wajib mematikan `torch.compile` (`os.environ["TORCH_COMPILE_DISABLE"] = "1"` dan monkeypatch `torch.compile = noop`) sebelum `FastVisionModel` di-import untuk mencegah crash `recompile_limit`.
- [ ] **7. Cast Guard Pure BF16:** Sebelum memanggil `.save_pretrained_merged()`, cast seluruh parameter dan buffer float32 ke `torch.bfloat16` untuk mencegah crash merge Unsloth.
