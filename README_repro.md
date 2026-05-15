# MicLog2.0 Repro Notes

This file records the main commands used to prepare data and train the current Qwen3.5 meta-training setup.

## 1. Preprocess LogHub Data

Source data root:

```bash
loghub-2.0/full_dataset
```

Main preprocessing script:

```bash
python3 dedup_content_logs.py --help
```

Important stages:

```bash
python3 dedup_content_logs.py --stage normalized-dedup
python3 dedup_content_logs.py --stage cluster
python3 dedup_content_logs.py --stage select-balanced
```

One-shot end-to-end normalized pipeline:

```bash
python3 dedup_content_logs.py --stage all-normalized
python3 dedup_content_logs.py --stage select-balanced
```

Outputs used later:

- `normalized_deduplicated/`
- `clustered/`
- `selected_balanced/`
- `selected_balanced_summary.csv`

Notes:

- `normalized_deduplicated/` is based on `normalized_content`, not raw `Content`.
- `clustered/` is also based on `normalized_content`.
- `selected_balanced/` is the balanced subset used as the retrieval bank and base training source.

## 2. Generate Meta-ICL JSONL Data

Main generation script:

```bash
python3 generate_meta_icl_jsonl.py --help
```

### 2.1 Full Variant Set

Generate the full training set variants:

```bash
python3 generate_meta_icl_jsonl.py \
  --max-shots 5 \
  --query-mode full \
  --output-root meta_incontext_data_variants
```

This produces 11 variant folders:

- `0-shot-only`
- `1-shot-only`
- `2-shot-only`
- `3-shot-only`
- `4-shot-only`
- `5-shot-only`
- `0-1-shot`
- `0-2-shot`
- `0-3-shot`
- `0-4-shot`
- `0-5-shot`

Meaning:

- `k-shot-only`: only k-shot training samples
- `0-k-shot`: progressive samples from 0-shot through k-shot

### 2.2 Lite Variant Set

Generate the lite training set variants:

```bash
python3 generate_meta_icl_jsonl.py \
  --max-shots 5 \
  --query-mode lite \
  --output-root meta_incontext_data_variants_lite
```

This is how `meta_incontext_data_variants_lite/` was generated.

Key design:

- retrieval bank stays full: all rows in `selected_balanced/`
- training queries are downsampled with lite rules

Lite query rules:

- `trivial`: keep `0`
- `single_var`: keep `1`
- `medium`: keep `1`
- `complex`: keep `2`
- `very_complex`: keep `3`

Current lite reduction:

- full query count: `6520`
- lite query count: `4817`
- ratio: `0.738804`

Summary files:

- `meta_incontext_data_variants/experiment_summary.csv`
- `meta_incontext_data_variants_lite/experiment_summary.csv`
- `meta_incontext_data_variants_lite/lite_query_summary.csv`

### 2.3 Optional Useful Arguments

Use normalized text for retrieval instead of raw content:

```bash
python3 generate_meta_icl_jsonl.py \
  --max-shots 5 \
  --query-mode full \
  --retrieval-field normalized_content \
  --output-root meta_incontext_data_variants_norm_retrieval
```

Generate only one dataset:

```bash
python3 generate_meta_icl_jsonl.py \
  --max-shots 5 \
  --query-mode lite \
  --dataset Apache \
  --output-root meta_incontext_data_variants_lite_apache
```

Generate metadata JSONL for debugging:

```bash
python3 generate_meta_icl_jsonl.py \
  --max-shots 5 \
  --query-mode lite \
  --write-metadata \
  --output-root meta_incontext_data_variants_lite_with_meta
```

## 3. Training Environment

Training requirements file:

```bash
requirements_qwen35.txt
```

Install dependencies inside the target environment:

```bash
PYTHONNOUSERSITE=1 pip install -U -r requirements_qwen35.txt
```

Why `PYTHONNOUSERSITE=1` is recommended:

- prevents accidental imports from `~/.local/lib/python3.10/site-packages`
- avoids user-site package contamination across environments
- makes training more reproducible

### 3.1 Verify CUDA Runtime

Before training, verify GPU visibility:

```bash
PYTHONNOUSERSITE=1 python3 - <<'PY'
import torch
print("torch", torch.__version__)
print("torch.version.cuda", torch.version.cuda)
print("cuda.is_available", torch.cuda.is_available())
print("device_count", torch.cuda.device_count())
if torch.cuda.is_available():
    print("device0", torch.cuda.get_device_name(0))
PY
```

## 4. Train Qwen3.5 with LoRA / QLoRA

Main training script:

```bash
PYTHONNOUSERSITE=1 python3 train_qwen35_meta.py --help
```

Supported model aliases:

- `0.8b`
- `2b`
- `4b`
- `9b`

These map to local models under:

```bash
/tempdisk2/yjb/Models/
```

### 4.1 Smoke Test

Recommended first run:

```bash
CUDA_VISIBLE_DEVICES=1 PYTHONNOUSERSITE=1 python3 train_qwen35_meta.py \
  --model 2b \
  --train-file meta_incontext_data_variants/0-shot-only/train.jsonl \
  --output-dir outputs/qwen35_2b_lora_0shot_smoke \
  --method lora \
  --dataset-format instruction \
  --max-length 1024 \
  --num-train-epochs 1 \
  --max-train-samples 200 \
  --per-device-train-batch-size 2 \
  --gradient-accumulation-steps 8
```

### 4.2 Standard 0-shot Full Training

```bash
CUDA_VISIBLE_DEVICES=1 PYTHONNOUSERSITE=1 python3 train_qwen35_meta.py \
  --model 2b \
  --train-file meta_incontext_data_variants/0-shot-only/train.jsonl \
  --output-dir outputs/qwen35_2b_lora_0shot \
  --method lora \
  --dataset-format instruction \
  --max-length 2048 \
  --num-train-epochs 1 \
  --per-device-train-batch-size 2 \
  --gradient-accumulation-steps 8
```

### 4.3 Standard 0-shot Lite Training

```bash
CUDA_VISIBLE_DEVICES=1 PYTHONNOUSERSITE=1 python3 train_qwen35_meta.py \
  --model 2b \
  --train-file meta_incontext_data_variants_lite/0-shot-only/train.jsonl \
  --output-dir outputs/qwen35_2b_lora_0shot_lite \
  --method lora \
  --dataset-format instruction \
  --max-length 2048 \
  --num-train-epochs 1 \
  --per-device-train-batch-size 2 \
  --gradient-accumulation-steps 8
```

### 4.4 Progressive 0-5-shot Lite Training

```bash
CUDA_VISIBLE_DEVICES=1 PYTHONNOUSERSITE=1 python3 train_qwen35_meta.py \
  --model 2b \
  --train-file meta_incontext_data_variants_lite/0-5-shot/train.jsonl \
  --output-dir outputs/qwen35_2b_lora_0_5shot_lite \
  --method lora \
  --dataset-format instruction \
  --max-length 2048 \
  --num-train-epochs 1 \
  --per-device-train-batch-size 2 \
  --gradient-accumulation-steps 8
```

### 4.5 Run on the Second GPU

To bind training to the second physical GPU:

```bash
CUDA_VISIBLE_DEVICES=1 PYTHONNOUSERSITE=1 python3 train_qwen35_meta.py ...
```

Important note:

- after setting `CUDA_VISIBLE_DEVICES=1`, that GPU becomes `cuda:0` inside the process

## 5. Practical Notes

### Epoch meaning

`--num-train-epochs N` means:

- the whole training dataset is traversed `N` times
- not that one sample is optimized `N` times consecutively before moving on

### Full vs Lite

- `meta_incontext_data_variants/`: full query set
- `meta_incontext_data_variants_lite/`: full retrieval bank + lite query set

### Retrieval vs Training Queries

In lite mode:

- retrieval examples still come from the full `selected_balanced/` bank
- only the query side of the training JSONL is reduced

This preserves reference diversity while reducing training time.

## 6. Use the Trained LoRA Model

After LoRA training, the main output is not a full 2B model checkpoint.
The important files are:

- `outputs/.../adapter_model.safetensors`
- `outputs/.../adapter_config.json`

That means inference needs:

- the original base model under `/tempdisk2/yjb/Models/`
- the trained adapter directory under `outputs/.../`

### 6.1 Direct Inference with Base Model + Adapter

Main inference script:

```bash
PYTHONNOUSERSITE=1 python3 infer_qwen35_meta.py --help
```

Example for the trained 2B 0-shot adapter:

```bash
CUDA_VISIBLE_DEVICES=1 PYTHONNOUSERSITE=1 python3 infer_qwen35_meta.py \
  --adapter-dir outputs/qwen35_2b_lora_0shot \
  --content "Accepted password for root from 10.0.0.1 port 22 ssh2"
```

If you want to inspect the exact prompt:

```bash
CUDA_VISIBLE_DEVICES=1 PYTHONNOUSERSITE=1 python3 infer_qwen35_meta.py \
  --adapter-dir outputs/qwen35_2b_lora_0shot \
  --content "Accepted password for root from 10.0.0.1 port 22 ssh2" \
  --print-prompt
```

### 6.2 Merge LoRA into a Standalone Model Directory

Main merge script:

```bash
PYTHONNOUSERSITE=1 python3 merge_qwen35_lora.py --help
```

Example:

```bash
CUDA_VISIBLE_DEVICES=1 PYTHONNOUSERSITE=1 python3 merge_qwen35_lora.py \
  --adapter-dir outputs/qwen35_2b_lora_0shot \
  --output-dir outputs/qwen35_2b_lora_0shot_merged
```

After merging, `outputs/qwen35_2b_lora_0shot_merged/` becomes a standalone model directory and no longer needs the adapter separately.

### 6.3 What Was Actually Saved

For a successful LoRA run, the expected output looks like:

- `adapter_model.safetensors`: trained LoRA delta weights
- `adapter_config.json`: adapter structure + base model reference
- `run_config.json`: local training run metadata
- `checkpoint-*`: intermediate checkpoints

So if you "didn't see the model weights", that is expected. What you trained is the adapter.

## 7. Batch Progressive Training

Main batch launcher:

```bash
PYTHONNOUSERSITE=1 python3 run_progressive_train_qwen35.py --help
```

Example: sequentially train `0-1-shot` through `0-5-shot` on 2B with the full variant set:

```bash
CUDA_VISIBLE_DEVICES=1 PYTHONNOUSERSITE=1 python3 run_progressive_train_qwen35.py \
  --model 2b \
  --data-root meta_incontext_data_variants \
  --output-root outputs/qwen35_2b_lora_progressive_full
```

Lite version:

```bash
CUDA_VISIBLE_DEVICES=1 PYTHONNOUSERSITE=1 python3 run_progressive_train_qwen35.py \
  --model 2b \
  --data-root meta_incontext_data_variants_lite \
  --output-root outputs/qwen35_2b_lora_progressive_lite
```

Useful flags:

- `--dry-run`: print commands only
- `--skip-existing`: skip variants whose adapter already exists
- `--continue-on-error`: do not stop the whole batch after one failure
- any unknown extra args are forwarded to `train_qwen35_meta.py`

### 7.1 Output Naming Update

The current default output naming is flat, not nested.
For example, with `--model 2b` and `--method lora`, the launcher writes to:

- `outputs/qwen35_2b_lora_0-1-shot`
- `outputs/qwen35_2b_lora_0-2-shot`
- `outputs/qwen35_2b_lora_0-3-shot`
- `outputs/qwen35_2b_lora_0-4-shot`
- `outputs/qwen35_2b_lora_0-5-shot`

So you usually do not need to pass `--output-root` anymore.
If needed, you can override the pattern with:

```bash
--output-template 'outputs/qwen35_{model_label}_{method}_{variant}'
```

## 8. Single-Dataset Meta-Train Ablation

This ablation keeps the main test/eval pipeline unchanged:

- train one LoRA model using only one dataset's meta-ICL JSONL
- test that model on the full 14-dataset benchmark
- keep the same online parsing and evaluation settings as the main line

### 8.1 Generate Per-Dataset JSONL Roots

Main helper:

```bash
PYTHONNOUSERSITE=1 python3 generate_meta_icl_jsonl_by_dataset.py --help
```

Generate 14 per-dataset roots with the same variant structure as the full setup:

```bash
PYTHONNOUSERSITE=1 python3 generate_meta_icl_jsonl_by_dataset.py \
  --max-shots 5 \
  --query-mode full \
  --output-root-parent meta_incontext_data_variants_by_dataset
```

This creates:

- `meta_incontext_data_variants_by_dataset/Apache/...`
- `meta_incontext_data_variants_by_dataset/BGL/...`
- ...
- `meta_incontext_data_variants_by_dataset/Zookeeper/...`

Each dataset root contains the same variant names as before, for example:

- `0-shot-only/train.jsonl`
- `0-1-shot/train.jsonl`
- `0-5-shot/train.jsonl`

Generate only one dataset if needed:

```bash
PYTHONNOUSERSITE=1 python3 generate_meta_icl_jsonl_by_dataset.py \
  --datasets Apache \
  --max-shots 5 \
  --query-mode full
```

### 8.2 Batch-Train One Model Per Dataset

Main helper:

```bash
PYTHONNOUSERSITE=1 python3 run_single_dataset_ablation_train_qwen35.py --help
```

Recommended main-line ablation: train one `0-5-shot` model per dataset on 4B with the same LoRA-style training settings:

```bash
CUDA_VISIBLE_DEVICES=1 PYTHONNOUSERSITE=1 python3 run_single_dataset_ablation_train_qwen35.py \
  --model 4b \
  --variants 0-5-shot \
  --data-root-parent meta_incontext_data_variants_by_dataset \
  --output-root-parent outputs_single_dataset_ablation \
  --dataset-format instruction \
  --max-length 2048 \
  --num-train-epochs 1 \
  --per-device-train-batch-size 2 \
  --gradient-accumulation-steps 8
```

Output layout:

- `outputs_single_dataset_ablation/Apache/qwen35_4b_lora_0-5-shot`
- `outputs_single_dataset_ablation/BGL/qwen35_4b_lora_0-5-shot`
- ...

Train only one dataset if needed:

```bash
CUDA_VISIBLE_DEVICES=1 PYTHONNOUSERSITE=1 python3 run_single_dataset_ablation_train_qwen35.py \
  --model 4b \
  --datasets Apache \
  --variants 0-5-shot \
  --dataset-format instruction \
  --max-length 2048 \
  --num-train-epochs 1 \
  --per-device-train-batch-size 2 \
  --gradient-accumulation-steps 8
```

### 8.3 Test / Eval Remain Unchanged

Only the adapter path changes. The online test and evaluation commands stay the same.

Example: test the Apache-only trained model on the full 14-dataset benchmark with `1-shot` online parsing:

```bash
CUDA_VISIBLE_DEVICES=0 PYTHONNOUSERSITE=1 python3 run_online_parser_batch.py \
  --model-path /tempdisk2/yjb/Models/Qwen3.5-4B \
  --adapter-dir outputs_single_dataset_ablation/Apache/qwen35_4b_lora_0-5-shot \
  --shots 1
```

Then evaluate the produced results as usual:

```bash
PYTHONNOUSERSITE=1 python3 evaluation/MicLog2_eval.py \
  --parsed-root results/<timestamp>/qwen35_4b_lora_0-5-shot_1shot
```
