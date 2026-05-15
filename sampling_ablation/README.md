# Sampling Ablation

This directory contains standalone tooling for sampling-stage ablations. It does not modify the main preprocessing, training, parsing, or evaluation entrypoints.

## Goal

Build alternative support roots with the same CSV schema as `selected_balanced/`, then reuse the existing pipeline unchanged:

- `scripts/generate_meta_icl_jsonl.py`
- `scripts/run_progressive_train_qwen35.py`
- `scripts/run_online_parser_batch.py`
- `scripts/evaluate.py`

## Implemented Strategies

1. `random_global_matched`

- Randomly sample the same number of rows as `selected_balanced` for each dataset.
- Cluster coverage is not preserved.

2. `random_cluster_matched`

- Keep the same cluster-level `keep_k` as the main pipeline.
- Replace representative/complex/diverse selection inside each cluster with random sampling.

3. `representative_only`

- Keep only the frequency representative for each cluster.
- Remove the extra complex/diverse rows.

## Output Layout

Running the builder creates:

- `selected_sampling_ablation/random_global_matched_seed42/...`
- `selected_sampling_ablation/random_cluster_matched_seed42/...`
- `selected_sampling_ablation/random_cluster_matched_seed43/...`
- `selected_sampling_ablation/random_cluster_matched_seed44/...`
- `selected_sampling_ablation/representative_only/...`

Each dataset CSV keeps the same columns as `selected_balanced`, including:

- `cluster_keep_k`
- `selected_rank`
- `selection_reason`

Each run root also contains `summary.csv`.

## Build The Sampling Roots

```bash
PYTHONNOUSERSITE=1 python3 sampling_ablation/build_sampling_ablation_selected.py
```

Optional subset run:

```bash
PYTHONNOUSERSITE=1 python3 sampling_ablation/build_sampling_ablation_selected.py \
  --datasets Apache,BGL,Thunderbird
```

## Generate Meta-Train JSONL From An Ablation Root

Example with `random_cluster_matched_seed42`:

```bash
PYTHONNOUSERSITE=1 python3 scripts/generate_meta_icl_jsonl.py \
  --input-root selected_sampling_ablation/random_cluster_matched_seed42 \
  --output-root meta_incontext_data_variants_sampling_ablation/random_cluster_matched_seed42 \
  --max-shots 5 \
  --query-mode full
```

## Train

Example:

```bash
CUDA_VISIBLE_DEVICES=1 PYTHONNOUSERSITE=1 python3 scripts/run_progressive_train_qwen35.py \
  --model 0.8b \
  --data-root meta_incontext_data_variants_sampling_ablation/random_cluster_matched_seed42 \
  --variants 0-5-shot
```

## Test

Use the ablation support root during online parsing:

```bash
CUDA_VISIBLE_DEVICES=0 PYTHONNOUSERSITE=1 python3 scripts/run_online_parser_batch.py \
  --model-path /tempdisk2/yjb/Models/Qwen3.5-0.8B \
  --adapter-dir outputs/qwen35_0.8b_lora_0-5-shot \
  --shots 1 \
  --support-root selected_sampling_ablation/random_cluster_matched_seed42
```

## Evaluation

Evaluation stays unchanged and still reads the parsing results directory:

```bash
PYTHONNOUSERSITE=1 python3 scripts/evaluate.py \
  --parsed-root results/<timestamp>/<run_name>
```
