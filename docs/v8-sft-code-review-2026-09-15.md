# V8 SFT full-code review — 2026-09-15

Scope: every cell in `notebooks/working-molab-v8.py` was read, with v7 used as the behavioral baseline. Runtime validation used Transformers 5.17.0, Torch 2.14.0, and marimo 0.24.2 on CPU. ORPO findings are listed separately because ORPO is disabled and its dataset is not ready.

## Confirmed SFT defects fixed

1. **Accumulated loss was not normalized.** `JointSFTTrainer.compute_loss` returns a microbatch mean and ignores `num_items_in_batch`, but v8 lacked v7's `model_accepts_loss_kwargs = False` guard. Transformers consequently skipped its `loss / gradient_accumulation_steps` branch. With GA=16 this made logged loss and pre-clip gradients 16 times larger. A real Trainer regression now produces loss 1.81 for GA=1 and GA=16; the old path produces 28.95.
2. **Selective smoothing had unsafe BF16 arithmetic.** Smoothing now computes `log_softmax` in FP32, avoids `-inf * 0`, handles tuple outputs correctly, validates suppressed IDs, and fails on batches with no supervised tokens. The smoothing coefficient stays at 0.1.
3. **OrScale calibrated zero-initialized LoRA weights to zero.** Such parameters now bootstrap at trust ratio 1 and calibrate once parameter and update norms are meaningful. Zero or invalid calibration values restored from older checkpoints are repaired.
4. **Newton–Schulz formed its Gram matrix on the large axis.** Wide LoRA matrices could allocate a width-by-width matrix. It now uses the smaller axis while preserving the numerical result.
5. **Vision rows could pair shuffled logical rows with raw Arrow image rows.** Logical select/shuffle indices are resolved against the original Arrow table without materializing image data. Invalid image indices now fail explicitly. This avoids `offset overflow while concatenating arrays` on large image columns.
6. **Source length was configured but not enforced.** Oversized inputs now fail with a descriptive error because blindly truncating image token blocks can corrupt multimodal input. Target formatting preserves exactly one end-of-turn token plus EOS within the limit.
7. **Resume could select a partial or stale checkpoint.** It now chooses the latest numerically complete checkpoint, requires adapter/trainer/optimizer/scheduler/RNG state, pins the HF snapshot revision, and passes the exact checkpoint path to Trainer. The scheduler is built by Trainer using the actual distributed dataloader length.
8. **Evaluation could alter later training.** Loss evaluation and sampled generation restore model mode, cache configuration, and Torch RNG even after an exception. Post-SFT smoke evaluation now depends on SFT completion.
9. **Checkpoint uploads were unsafe on verification failure.** A failed retry no longer deletes an existing remote prefix. Upload runs only on the world-process-zero rank, and the processor is saved with Trainer checkpoints.
10. **Metrics/UI issues.** Exponentiated smoothed loss is no longer mislabeled as perplexity, evaluation logs are not mutated by the plot callback, and the progress table receives training loss during epoch-based evaluation.

## Interpretation of checkpoint-200

The checkpoint-200 log decreases from 76.69 to roughly 52.64. Dividing the historical values by GA=16 gives approximately 4.79 to 3.29. The direction is promising, but checkpoint-200 has no epoch evaluation yet, so the log alone cannot prove final quality. Because the old gradient was scaled before clipping, the checkpoint is not mathematically equivalent to a correctly normalized run even though the curve looks healthy. Resuming it is supported, but loss values before and after the code fix use different scales.

## Deferred ORPO findings

These are intentionally not changed in this SFT patch: the TLPO lagged-policy snapshot mixes unrelated batches; repetition weighting is applied batch-wide; the ORPO multimodal eval path references the training vision dataset; and adapter/base-model provenance needs a durable manifest for later merge/reload. These must be resolved before enabling ORPO.

## Validation

- `python -m unittest discover -s tests -p test_v8_sft_regressions.py -v`: 6 tests passed.
- `python -m marimo check notebooks/working-molab-v8.py`: passed.
- `git diff --check`: passed (only Git's CRLF-to-LF notice).

Full molab training was not run locally; that remains the final integration test for GPU memory, Unsloth fork behavior, and remote checkpoint upload.
