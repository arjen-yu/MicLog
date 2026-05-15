# MicLog2.0

MicLog2.0 is a Meta-ICL based log parsing framework. It fine-tunes a local causal language model with LoRA and performs online log parsing with retrieval, multi-level template caching, and TA-Eval-Rep metrics.

## Repository Layout

```text
miclog2/                                      Core online parser package
evaluation/                                  TA-Eval-Rep style evaluation
sampling_ablation/                           Sampling ablation utilities
dedup_content_logs.py                        LogHub preprocessing
generate_meta_icl_jsonl.py                   Meta-ICL JSONL generation
generate_meta_icl_jsonl_by_dataset.py        Per-dataset JSONL generation
train_qwen35_meta.py                         LoRA/QLoRA training
run_progressive_train_qwen35.py              Batch training over shot variants
run_online_parser.py                         Single-dataset online parsing
run_online_parser_batch.py                   Multi-dataset online parsing
run_sequence_dataset_parser.py               Sequential dataset parsing helper
run_single_dataset_ablation_train_qwen35.py  Single-dataset ablation training
infer_qwen35_meta.py                         Single-log inference
merge_qwen35_lora.py                         Merge LoRA into base model
```

Large datasets, model weights, LoRA checkpoints, and full prediction outputs are intentionally not included in this repository.

## Requirements

Use Python 3.10+ with CUDA-enabled PyTorch for training and GPU inference.

```bash
PYTHONNOUSERSITE=1 pip install -U -r requirements_qwen35.txt
```

Check CUDA:

```bash
PYTHONNOUSERSITE=1 python3 - <<'PY'
import torch
print("torch", torch.__version__)
print("cuda", torch.cuda.is_available())
print("device_count", torch.cuda.device_count())
PY
```

## Data and Models

Place LogHub-2.0 full datasets under:

```text
loghub-2.0/full_dataset/
```

The scripts accept local HuggingFace model directories. The default aliases in `train_qwen35_meta.py` point to local paths such as:

```text
/tempdisk2/yjb/Models/Qwen3.5-0.8B
```

You can also pass an absolute model path directly:

```bash
--model /path/to/model
--model-path /path/to/model
```

## Quick Start

Assume `meta_incontext_data_variants/0-5-shot/train.jsonl` already exists.

Train a LoRA adapter:

```bash
CUDA_VISIBLE_DEVICES=0 PYTHONNOUSERSITE=1 python3 train_qwen35_meta.py \
  --model 0.8b \
  --train-file meta_incontext_data_variants/0-5-shot/train.jsonl \
  --output-dir outputs/qwen35_0.8b_lora_0-5-shot \
  --method lora \
  --dataset-format instruction \
  --max-length 2048 \
  --num-train-epochs 1 \
  --per-device-train-batch-size 2 \
  --gradient-accumulation-steps 8
```

Run 1-shot online parsing:

```bash
CUDA_VISIBLE_DEVICES=0 PYTHONNOUSERSITE=1 python3 run_online_parser_batch.py \
  --model-path /tempdisk2/yjb/Models/Qwen3.5-0.8B \
  --adapter-dir outputs/qwen35_0.8b_lora_0-5-shot \
  --shots 1 \
  --run-name qwen35_0.8b_lora_0-5-shot_1shot
```

Evaluate an existing parsing run:

```bash
PYTHONNOUSERSITE=1 python3 evaluation/MicLog2_eval.py \
  --parsed-root results/<timestamp>/qwen35_0.8b_lora_0-5-shot_1shot \
  --shots 1 \
  --run-name eval_qwen35_0.8b_lora_0-5-shot_1shot
```

Evaluation outputs are written to:

```text
results/evaluation/<timestamp>/<run-name>/
```

## Full Reproduction

### 1. Preprocess LogHub Data

```bash
PYTHONNOUSERSITE=1 python3 dedup_content_logs.py --stage normalized-dedup
PYTHONNOUSERSITE=1 python3 dedup_content_logs.py --stage cluster
PYTHONNOUSERSITE=1 python3 dedup_content_logs.py --stage select-balanced
```

This produces:

```text
normalized_deduplicated/
clustered/
selected_balanced/
selected_balanced_summary.csv
```

### 2. Generate Meta-ICL JSONL

Generate the full 0-5 shot variant set:

```bash
PYTHONNOUSERSITE=1 python3 generate_meta_icl_jsonl.py \
  --max-shots 5 \
  --query-mode full \
  --output-root meta_incontext_data_variants
```

The main training file used in the standard run is:

```text
meta_incontext_data_variants/0-5-shot/train.jsonl
```

### 3. Train LoRA

Single run:

```bash
CUDA_VISIBLE_DEVICES=0 PYTHONNOUSERSITE=1 python3 train_qwen35_meta.py \
  --model 0.8b \
  --train-file meta_incontext_data_variants/0-5-shot/train.jsonl \
  --output-dir outputs/qwen35_0.8b_lora_0-5-shot \
  --method lora \
  --dataset-format instruction \
  --max-length 2048 \
  --num-train-epochs 1 \
  --per-device-train-batch-size 2 \
  --gradient-accumulation-steps 8
```

Progressive variant training:

```bash
CUDA_VISIBLE_DEVICES=0 PYTHONNOUSERSITE=1 python3 run_progressive_train_qwen35.py \
  --model 0.8b \
  --data-root meta_incontext_data_variants \
  --variants 0-1-shot,0-2-shot,0-3-shot,0-4-shot,0-5-shot \
  --method lora \
  --dataset-format instruction \
  --max-length 2048 \
  --num-train-epochs 1 \
  --per-device-train-batch-size 2 \
  --gradient-accumulation-steps 8 \
  --cuda-visible-devices 0
```

### 4. Online Parsing

Run all default datasets:

```bash
CUDA_VISIBLE_DEVICES=0 PYTHONNOUSERSITE=1 python3 run_online_parser_batch.py \
  --model-path /tempdisk2/yjb/Models/Qwen3.5-0.8B \
  --adapter-dir outputs/qwen35_0.8b_lora_0-5-shot \
  --shots 1 \
  --run-name qwen35_0.8b_lora_0-5-shot_1shot
```

Run selected datasets:

```bash
CUDA_VISIBLE_DEVICES=0 PYTHONNOUSERSITE=1 python3 run_online_parser_batch.py \
  --datasets Apache,BGL,Hadoop \
  --model-path /tempdisk2/yjb/Models/Qwen3.5-0.8B \
  --adapter-dir outputs/qwen35_0.8b_lora_0-5-shot \
  --shots 1 \
  --run-name qwen35_0.8b_lora_0-5-shot_1shot_partial
```

Each dataset output contains:

```text
predictions.csv
summary.json
```

### 5. Evaluation

Evaluate existing `predictions.csv` outputs:

```bash
PYTHONNOUSERSITE=1 python3 evaluation/MicLog2_eval.py \
  --parsed-root results/<timestamp>/qwen35_0.8b_lora_0-5-shot_1shot \
  --shots 1 \
  --run-name eval_qwen35_0.8b_lora_0-5-shot_1shot
```

Main metrics are saved in:

```text
summary.csv
summary_average.csv
```

## Notes

- `outputs/` contains LoRA adapters and is not tracked by Git.
- `results/` contains full parsing and evaluation outputs and is not tracked by Git.
- Large generated JSONL files are not tracked by Git; regenerate them with the scripts above.
- Use `PYTHONNOUSERSITE=1` to avoid user-site package contamination.
