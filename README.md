# MicLog

MicLog is an LLM-based log parser built on progressive Meta In-Context Learning.

This repository provides a revised replication package for [AAAI 2026] [MicLog: Towards Accurate and Efficient LLM-based Log Parsing via Progressive Meta In-Context Learning](https://ojs.aaai.org/index.php/AAAI/article/view/37123). This version preserves the original ProgMeta-ICL strategy, but refines the implementation of other modules, resulting in higher parsing performance.

## Repository Organization

```text
MicLog/
├── miclog2/                    # Core parser package: retrieval, prompting, caching, validation, model runner
├── evaluation/                 # TA-Eval-Rep style evaluation metrics
├── sampling_ablation/          # Utilities for sampling-stage ablation studies
├── scripts/                    # Command-line entrypoints
│   ├── dedup_content_logs.py
│   ├── generate_meta_icl_jsonl.py
│   ├── generate_meta_icl_jsonl_by_dataset.py
│   ├── train_qwen35_meta.py
│   ├── run_progressive_train_qwen35.py
│   ├── run_online_parser.py
│   ├── run_online_parser_batch.py
│   ├── run_sequence_dataset_parser.py
│   ├── run_single_dataset_ablation_train_qwen35.py
│   ├── evaluate.py
│   ├── infer_qwen35_meta.py
│   └── merge_qwen35_lora.py
├── requirements_qwen35.txt     # Python dependencies
├── dataset_overview.csv        # Dataset overview
├── selected_balanced_summary.csv
└── cluster_dataset_summary.csv
```

Large datasets, generated JSONL files, model weights, LoRA adapters, and full prediction outputs are not tracked by Git.

## Quick Start

Quick Start assumes that LogHub-2.0 data, a local base model, a trained LoRA adapter, and `selected_balanced/` are already prepared.

Create and activate a Conda environment with Python 3.10:

```bash
conda create -n miclog python=3.10 -y
conda activate miclog
```

Install dependencies:

```bash
pip install -U -r requirements_qwen35.txt
```

Run online parsing directly:

```bash
CUDA_VISIBLE_DEVICES=0 python3 scripts/run_online_parser_batch.py \
  --model-path /path/to/base-model \
  --adapter-dir outputs/qwen35_0.8b_lora_0-5-shot \
  --shots 1 \
  --run-name miclog_1shot
```

Evaluate the parsing results:

```bash
python3 scripts/evaluate.py \
  --parsed-root results/<timestamp>/miclog_1shot \
  --shots 1 \
  --run-name eval_miclog_1shot
```

Main outputs:

```text
results/<timestamp>/<run-name>/<dataset>/predictions.csv
results/evaluation/<timestamp>/<run-name>/summary.csv
results/evaluation/<timestamp>/<run-name>/summary_average.csv
```

## Full Reproduction

### 1. Prepare Data

Download LogHub-2.0 and place the full datasets under:

```text
loghub-2.0/full_dataset/
```

Run preprocessing:

```bash
python3 scripts/dedup_content_logs.py --stage normalized-dedup
python3 scripts/dedup_content_logs.py --stage cluster
python3 scripts/dedup_content_logs.py --stage select-balanced
```

Generated directories:

```text
normalized_deduplicated/
clustered/
selected_balanced/
```

### 2. Generate Meta-ICL Training Data

```bash
python3 scripts/generate_meta_icl_jsonl.py \
  --max-shots 5 \
  --query-mode full \
  --output-root meta_incontext_data_variants
```

The standard progressive training file is:

```text
meta_incontext_data_variants/0-5-shot/train.jsonl
```

### 3. Train LoRA

Train one adapter:

```bash
CUDA_VISIBLE_DEVICES=0 python3 scripts/train_qwen35_meta.py \
  --model /path/to/base-model \
  --train-file meta_incontext_data_variants/0-5-shot/train.jsonl \
  --output-dir outputs/miclog_lora_0-5-shot \
  --method lora \
  --dataset-format instruction \
  --max-length 2048 \
  --num-train-epochs 1 \
  --per-device-train-batch-size 2 \
  --gradient-accumulation-steps 8
```

Train progressive variants:

```bash
CUDA_VISIBLE_DEVICES=0 python3 scripts/run_progressive_train_qwen35.py \
  --model /path/to/base-model \
  --data-root meta_incontext_data_variants \
  --variants 0-1-shot,0-2-shot,0-3-shot,0-4-shot,0-5-shot \
  --output-template 'outputs/miclog_{model_label}_{method}_{variant}' \
  --method lora \
  --dataset-format instruction \
  --max-length 2048 \
  --num-train-epochs 1 \
  --per-device-train-batch-size 2 \
  --gradient-accumulation-steps 8 \
  --cuda-visible-devices 0
```

### 4. Parse Logs

Run all default datasets:

```bash
CUDA_VISIBLE_DEVICES=0 python3 scripts/run_online_parser_batch.py \
  --model-path /path/to/base-model \
  --adapter-dir outputs/miclog_lora_0-5-shot \
  --shots 1 \
  --run-name miclog_lora_0-5-shot_1shot
```

Run selected datasets:

```bash
CUDA_VISIBLE_DEVICES=0 python3 scripts/run_online_parser_batch.py \
  --datasets Apache,BGL,Hadoop \
  --model-path /path/to/base-model \
  --adapter-dir outputs/miclog_lora_0-5-shot \
  --shots 1 \
  --run-name miclog_lora_0-5-shot_1shot_partial
```

### 5. Evaluate

```bash
python3 scripts/evaluate.py \
  --parsed-root results/<timestamp>/miclog_lora_0-5-shot_1shot \
  --shots 1 \
  --run-name eval_miclog_lora_0-5-shot_1shot
```

Metrics include GA, PA, FGA, PTA, RTA, and FTA.

## Notes

- The scripts accept either model aliases defined in `scripts/train_qwen35_meta.py` or absolute HuggingFace-compatible model paths.
- `outputs/`, `results/`, `loghub-2.0/`, and generated JSONL directories are intentionally ignored by Git.

## Citation

```bibtex
@article{yu2026miclog,
    title={MicLog: Towards Accurate and Efficient LLM-based Log Parsing via Progressive Meta In-Context Learning},
    volume={40},
    url={https://ojs.aaai.org/index.php/AAAI/article/view/37123},
    DOI={10.1609/aaai.v40i2.37123},
    number={2},
    journal={Proceedings of the AAAI Conference on Artificial Intelligence},
    author={Yu, Jianbo and Li, Yixuan and Xu, Hai and Xu, Kang and Xu, Junjielong and Li, Zhijing and He, Pinjia and Wang, Wanyuan},
    year={2026},
    month={Mar.},
    pages={1480-1488}
}
```
