# Multi-Task Classification Training

This directory contains training scripts for the OmniClassifier with different training strategies using Hugging Face Accelerate.

## Training Scripts

### 1. Pre-configured Training Scripts

#### Head-Only Training
```bash
./train_head_only.sh
```
- Trains only the classification head
- Batch size: 8 (effective 16 with gradient accumulation)
- Learning rate: 1e-3
- Epochs: 5
- Wandb project: "omni-classifier-head-only"

#### LoRA Training
```bash
./train_lora.sh
```
- Uses Low-Rank Adaptation for efficient fine-tuning
- Batch size: 4 (effective 16 with gradient accumulation)
- Learning rate: 5e-5
- Epochs: 10
- Wandb project: "omni-classifier-lora"

#### Full Model Training
```bash
./train_full.sh
```
- Trains all model parameters
- Batch size: 2 (effective 16 with gradient accumulation)
- Learning rate: 1e-6
- Epochs: 15
- Wandb project: "omni-classifier-full"

### 2. Customizable Training Script

Use the customizable script to override any parameter:

```bash
./train_custom.sh --training_strategy head_only --lr 1e-4 --epochs 10 --train_batch_size 4
```

## Command-Line Arguments

The training script supports the following command-line arguments that override the YAML config:

### Training Strategy
- `--training_strategy`: Choose from `head_only`, `lora`, or `full`

### Batch Sizes and Learning Rate
- `--train_batch_size`: Training batch size
- `--val_batch_size`: Validation batch size
- `--lr`: Learning rate
- `--gradient_accumulation_steps`: Gradient accumulation steps

### Checkpoint Management
- `--save_checkpoint_dir`: Directory to save checkpoints
- `--load_checkpoint_path`: Path to load checkpoint from

### Training Duration
- `--epochs`: Number of training epochs

### Validation and Early Stopping
- `--validate_every_n_epochs`: Validate every N epochs
- `--validate_every_n_steps`: Validate every N steps
- `--early_stopping_patience`: Early stopping patience

### Logging
- `--project`: Wandb project name

### Other
- `--config_file`: Path to config YAML file (default: config_accelerate.yaml)

## Examples

### Quick head-only training with custom parameters:
```bash
./train_custom.sh \
    --training_strategy head_only \
    --train_batch_size 6 \
    --lr 2e-3 \
    --epochs 3 \
    --project "quick-test"
```

### LoRA training with step-based validation:
```bash
./train_custom.sh \
    --training_strategy lora \
    --train_batch_size 4 \
    --lr 1e-4 \
    --epochs 8 \
    --validate_every_n_steps 100 \
    --early_stopping_patience 4 \
    --project "lora-step-validation"
```

### Full training with custom checkpoint directory:
```bash
./train_custom.sh \
    --training_strategy full \
    --train_batch_size 1 \
    --lr 5e-7 \
    --epochs 20 \
    --save_checkpoint_dir "/path/to/custom/checkpoints" \
    --project "full-custom-path"
```

## Configuration Files

- `config_accelerate.yaml`: Main configuration file with default parameters
- `accelerate_config_qwen.yaml`: Accelerate configuration for multi-GPU training

## Notes

- All scripts use GPUs 2 and 3 by default (CUDA_VISIBLE_DEVICES="2,3")
- Gradient accumulation is used to increase effective batch size
- Mixed precision (fp16) is enabled for better memory efficiency
- Wandb logging is enabled by default
- Early stopping is based on validation F1 score
