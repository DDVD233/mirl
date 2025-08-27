# OmniClassifier Trainer with Validation and Wandb Integration

This enhanced trainer includes comprehensive validation, testing, and experiment tracking capabilities using Weights & Biases (wandb).

## Features

### 🎯 Validation & Testing
- **Validation during training**: Automatic validation every N epochs
- **Best model saving**: Saves the best model based on validation accuracy
- **Early stopping**: Stops training if validation accuracy doesn't improve
- **Final testing**: Comprehensive testing on test set after training
- **Detailed metrics**: Accuracy, precision, recall, F1-score, confusion matrices

### 📊 Wandb Integration
- **Experiment tracking**: All training progress logged to wandb
- **Configuration logging**: Complete config saved as wandb table
- **Training curves**: Loss, accuracy, and F1 score plots over epochs
- **Confusion matrices**: Visual confusion matrices for validation and test
- **Classification reports**: Detailed per-class performance metrics
- **Multi-metric tracking**: Micro, macro, and weighted F1 scores
- **Progress bar tracking**: Tqdm progress bars automatically logged to wandb
- **Real-time monitoring**: Live progress updates with ETA and processing rates

### 💾 Checkpoint Management
- **Automatic checkpointing**: Saves model state every epoch
- **Best model preservation**: Keeps the best performing model
- **Training state recovery**: Resumes training from checkpoints
- **Complete state saving**: Model, optimizer, and training history

### 🚀 Gradient Accumulation
- **Memory efficient training**: Accumulate gradients over multiple batches
- **Effective batch size control**: Configure effective batch size via `gradient_accumulation_steps`
- **Training stability**: Improves training stability with limited GPU memory
- **Configurable accumulation**: Set accumulation steps in config.yaml

## Configuration

### Data Files
```python
DATA_FILES = ["/path/to/train.jsonl"]
VAL_DATA_FILES = ["/path/to/val.jsonl"]
TEST_DATA_FILES = ["/path/to/test.jsonl"]
```

### Validation Settings
```python
VALIDATE_EVERY_N_EPOCHS = 1  # Validate every N epochs
SAVE_BEST_MODEL = True       # Save best model based on validation accuracy
EARLY_STOPPING_PATIENCE = 5  # Stop training if no improvement for N epochs
```

### Gradient Accumulation Settings
```yaml
train:
  train_batch_size: 8
  gradient_accumulation_steps: 4  # Effective batch size = 8 * 4 = 32
  num_workers: 4  # Number of worker processes for data loading
```

### Data Loading Configuration
- **num_workers: 0**: Single-threaded data loading (good for debugging)
- **num_workers: 1-4**: Multi-threaded data loading (good for most setups)
- **num_workers: 8+**: High-performance data loading (for powerful systems)

**Note**: Higher num_workers values can improve data loading speed but may increase memory usage.

### Wandb Settings
```python
USE_WANDB = True
WANDB_PROJECT = "omni-classifier"
WANDB_ENTITY = None  # Set to your wandb username
```

## Usage

### 1. Setup Wandb (Optional)
```bash
# Install wandb
pip install wandb

# Login to wandb
wandb login

# Or run the setup script
python setup_wandb.py
```

### 2. Run Training
```bash
python train_omni_classifier.py
```

### 3. Monitor Progress
- Training progress is logged to wandb dashboard
- Checkpoints are saved locally
- Best model is automatically saved
- **Tqdm progress bars**: All progress bars are automatically logged to wandb with real-time metrics

### 4. Tqdm Integration Features
The training script now includes enhanced tqdm integration with wandb:

- **Automatic logging**: All tqdm progress bars are automatically logged to wandb
- **Real-time metrics**: Progress includes current/total, elapsed time, ETA, and processing rate
- **Nested progress**: Both epoch-level and batch-level progress are tracked
- **Smart logging**: Progress is logged at reasonable intervals to avoid overwhelming wandb
- **Fallback support**: If wandb is not initialized, regular tqdm behavior is maintained

#### Example Usage:
```python
from wandb_utils import create_wandb_tqdm, log_tqdm_progress

# Create a wandb-enabled tqdm progress bar
pbar = create_wandb_tqdm(range(100), desc="Processing", total=100)
for i in pbar:
    # Do work...
    time.sleep(0.1)
    # Progress is automatically logged to wandb
```

#### Enhanced Metrics in Wandb:
The progress tracking now includes user-friendly formatted metrics:

```json
{
  "progress/Training": {
    "current": 45,
    "total": 100,
    "progress": 0.45,
    "progress_percentage": "45.00%",
    "elapsed_time": 125.5,
    "elapsed_time_formatted": "00:02:05",
    "eta": 153.2,
    "eta_formatted": "00:02:33",
    "rate": 0.36,
    "rate_per_second": "0.36/s",
    "summary": "45.00% complete | Elapsed: 00:02:05 | ETA: 00:02:33 | Rate: 0.36/s"
  }
}
```

#### Wandb Dashboard Features:
- **Progress tracking**: Real-time progress bars in wandb dashboard
- **Time estimates**: ETA and processing rate calculations with HH:MM:SS formatting
- **Percentage display**: Progress shown as percentages (e.g., "75.50%")
- **Formatted time**: Elapsed time and ETA displayed as HH:MM:SS (e.g., "01:23:45")
- **Rate display**: Processing rate shown as items/second
- **Summary field**: Combined information in a single readable string
- **Nested progress**: Multiple progress bars for different training phases
- **Historical data**: Progress history across training runs

## Output Structure

### Local Files
```
checkpoint_dir/
├── checkpoint_epoch_1.pt
├── checkpoint_epoch_2.pt
├── ...
└── best_model.pt
```

### Wandb Logs
- **Configuration**: Complete training config as table
- **Training Metrics**: Loss and accuracy per epoch
- **Validation Metrics**: Loss, accuracy, precision, recall, F1
- **Test Results**: Final performance on test set
- **Visualizations**: Training curves, confusion matrices
- **Classification Report**: Per-class performance breakdown

## Key Methods

### `validate(dataloader, split_name)`
Performs validation on the given dataloader and returns comprehensive metrics using the new evaluation module.

### `test()`
Loads the best model and performs final testing on the test set.

## Gradient Accumulation

Gradient accumulation allows you to simulate larger batch sizes when GPU memory is limited. Here's how it works:

### How It Works
1. **Forward Pass**: Process a small batch (e.g., batch_size=8)
2. **Backward Pass**: Compute gradients for the small batch
3. **Accumulate**: Add gradients to existing gradients (don't zero them)
4. **Repeat**: Process more batches until accumulation_steps is reached
5. **Update**: Zero gradients and perform optimizer step with accumulated gradients

### Benefits
- **Memory Efficiency**: Use smaller batches to fit in GPU memory
- **Training Stability**: Larger effective batch size improves gradient estimates
- **Flexible Configuration**: Adjust accumulation steps based on available memory

### Example Configuration
```yaml
train:
  train_batch_size: 8                    # Actual batch size per forward pass
  gradient_accumulation_steps: 4         # Number of batches to accumulate
  # Effective batch size = 8 * 4 = 32
```

### Monitoring
- Wandb logs show both actual and effective batch sizes
- Training metrics are logged at accumulation boundaries
- Loss scaling is automatically handled

### `save_checkpoint(optimizer, epoch, is_best)`
Saves model checkpoint with complete training state.

### `_init_wandb()`
Initializes wandb logging with configuration table.

## Evaluation Module

The trainer uses a dedicated `multi_task_evaluation.py` module that provides:

### `evaluate_predictions()`
Comprehensive evaluation function that computes all metrics including:
- Micro, macro, and weighted precision, recall, F1, accuracy
- Per-class metrics
- Confusion matrix
- Per-dataset breakdown (if dataset information available)

### `compute_dataset_metrics()`
Computes dataset-level metrics using the same formula as `hb_evaluation.py`:
- Proper F1 calculation: `F1 = 2 * precision * recall / (precision + recall)`
- Safe division handling
- Per-class one-vs-rest evaluation

## Metrics Tracked

### Training Metrics
- Loss per epoch
- Accuracy per epoch

### Validation Metrics
- Loss
- Accuracy (micro)
- Precision (micro, macro, weighted)
- Recall (micro, macro, weighted)
- F1-score (micro, macro, weighted)
- Confusion matrix
- Per-class metrics

### Test Metrics
- All validation metrics plus:
- Per-class precision, recall, F1, accuracy
- Support counts
- Detailed classification report
- Per-dataset breakdown (if dataset information available)

## Early Stopping

The trainer implements early stopping based on validation F1 score:
- Monitors validation micro F1 score improvement
- Stops training if no improvement for `EARLY_STOPPING_PATIENCE` epochs
- Saves the best model automatically

## Checkpoint Recovery

The trainer can resume training from checkpoints:
- Automatically finds latest checkpoint
- Restores model state, optimizer state, and training history
- Continues from the last saved epoch

## Customization

### Disable Wandb
Set `USE_WANDB = False` in the configuration section.

### Change Validation Frequency
Modify `VALIDATE_EVERY_N_EPOCHS` to validate less frequently.

### Adjust Early Stopping
Change `EARLY_STOPPING_PATIENCE` to be more or less patient.

### Custom Metrics
Add custom metrics in the `validate()` method and log them to wandb.

## Troubleshooting

### Wandb Issues
1. Run `python setup_wandb.py` to check configuration
2. Ensure you're logged in: `wandb login`
3. Set `USE_WANDB = False` to disable wandb temporarily

### Data Issues
1. Check data file paths in configuration
2. Ensure validation and test files exist
3. Verify label mapping is correct

### Memory Issues
1. Reduce batch sizes (`BATCH_SIZE`, `VAL_BATCH_SIZE`)
2. Use gradient accumulation
3. Enable mixed precision training

## Dependencies

Install required packages:
```bash
pip install -r requirements.txt
```

Key dependencies:
- `torch>=2.0.0`
- `transformers>=4.30.0`
- `wandb>=0.15.0`
- `scikit-learn>=1.3.0`
- `omegaconf>=2.3.0`
