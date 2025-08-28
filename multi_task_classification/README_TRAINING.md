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

### Data Parameters
- `--train_file`: Training data file path
- `--val_file`: Validation data file path
- `--test_file`: Test data file path
- `--label_map_path`: Path to label mapping JSON file

### Model Parameters
- `--tokenizer_name`: Tokenizer model name
- `--processor_name`: Processor model name
- `--training_strategy`: Choose from `head_only`, `lora`, or `full`
- `--device_map`: Device mapping (auto, cpu, or specific devices)
- `--torch_dtype`: PyTorch data type (float16, float32, bfloat16)

### LoRA Parameters
- `--lora_r`: LoRA rank
- `--lora_alpha`: LoRA alpha
- `--lora_dropout`: LoRA dropout
- `--lora_target_modules`: LoRA target modules (space-separated list)

### Training Parameters
- `--train_batch_size`: Training batch size
- `--val_batch_size`: Validation batch size
- `--lr`: Learning rate
- `--epochs`: Number of training epochs
- `--save_checkpoint_dir`: Directory to save checkpoints
- `--load_checkpoint_path`: Path to load checkpoint from
- `--save_every_n_epochs`: Save checkpoint every N epochs
- `--debug_dry_run`: Enable debug dry run mode
- `--gradient_accumulation_steps`: Gradient accumulation steps
- `--num_workers`: Number of data loader workers

### Validation Parameters
- `--validate_every_n_epochs`: Validate every N epochs (use "None" to disable)
- `--validate_every_n_steps`: Validate every N steps (use "None" to disable)
- `--early_stopping_patience`: Early stopping patience

### Dataset Parameters
- `--max_prompt_length`: Maximum prompt length
- `--modalities`: Comma-separated list of modalities
- `--prompt_key`: Prompt key in dataset
- `--image_key`: Image key in dataset
- `--video_key`: Video key in dataset
- `--audio_key`: Audio key in dataset
- `--label_key`: Label key in dataset
- `--return_multi_modal_inputs`: Return multi-modal inputs
- `--filter_overlong_prompts`: Filter overlong prompts
- `--truncation`: Truncation direction (left, right)
- `--format_prompt`: Path to format prompt template

### Wandb Parameters
- `--use_wandb`: Enable wandb logging
- `--project`: Wandb project name
- `--entity`: Wandb entity name

### System Parameters
- `--cuda_visible_devices`: CUDA visible devices
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

### Advanced LoRA configuration:
```bash
./train_custom.sh \
    --training_strategy lora \
    --lora_r 32 \
    --lora_alpha 64 \
    --lora_dropout 0.2 \
    --lora_target_modules q_proj k_proj v_proj o_proj \
    --train_batch_size 4 \
    --lr 1e-4 \
    --project "custom-lora-config"
```

### Custom dataset configuration:
```bash
./train_custom.sh \
    --max_prompt_length 8192 \
    --modalities "images,videos,audio" \
    --prompt_key "question" \
    --label_key "ground_truth" \
    --return_multi_modal_inputs \
    --filter_overlong_prompts \
    --truncation "right" \
    --project "custom-dataset-config"
```

### System-level customization:
```bash
./train_custom.sh \
    --cuda_visible_devices "0,1" \
    --device_map "auto" \
    --torch_dtype "bfloat16" \
    --num_workers 4 \
    --project "system-optimized"
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
