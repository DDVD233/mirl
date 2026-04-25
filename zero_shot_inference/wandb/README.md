# W&B Result Export

`build_wandb_results.py` pulls metrics from the W&B runs created by the
zero-shot inference and reasoning-evaluation scripts. Those scripts log one run
per model, with summary keys shaped like `{dataset}/{metric}`.

## Zero-shot inference

```bash
python verl/zero_shot_inference/wandb/build_wandb_results.py \
  --entity <wandb-entity> \
  --project zero-shot-inference \
  --models Qwen/Qwen2.5-Omni-7B PhilipC/HumanOmniV2 \
  --task inference \
  --output_json verl/zero_shot_inference/wandb/zero_shot_results.json \
  --output_tex verl/zero_shot_inference/wandb/zero_shot_results.tex
```

Default table metrics: `accuracy`, `weighted_f1`.

## Reasoning evaluation

```bash
python verl/zero_shot_inference/wandb/build_wandb_results.py \
  --entity <wandb-entity> \
  --project reasoning-evaluation \
  --models Qwen/Qwen2.5-Omni-7B ddvd233/OmniSapiens-7B-RL PhilipC/HumanOmniV2 \
  --task reasoning \
  --table_metrics reasoning_accuracy direct_accuracy self_consistency_rate para_consistency_rate mean_reasoning_tokens \
  --output_json verl/zero_shot_inference/wandb/reasoning_results.json \
  --output_tex verl/zero_shot_inference/wandb/reasoning_results.tex
```

## Regenerate TeX from JSON

```bash
python verl/zero_shot_inference/wandb/build_wandb_results.py \
  --from_json verl/zero_shot_inference/wandb/reasoning_results.json \
  --output_tex verl/zero_shot_inference/wandb/reasoning_results.tex
```

Useful flags:

- `--datasets eatd mvsa av-asd iemocap sarcnet overall` filters and orders datasets.
- `--metrics ...` filters which metrics are kept in the JSON.
- `--table_metrics ...` selects table columns.
- `--latest_per_model` keeps only the newest matching W&B run for each `config.model`.
- `--run_ids ...` fetches exact W&B run IDs instead of scanning the project.
- `--model_alias OLD=NEW` changes display names in the TeX table.
