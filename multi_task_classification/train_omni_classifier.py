import os
import sys
import json
import torch
import time
from torch.utils.data import DataLoader
from torch.optim import Adam
from torch.nn import CrossEntropyLoss
from dummy_classifier import DummyClassifier
from omni_classifier import OmniClassifier
from omni_classifier_dataset import OmniClassifierDataset
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from verl.utils.dataset.rl_dataset import collate_fn
from transformers import AutoTokenizer, AutoProcessor
from omegaconf import OmegaConf
from tqdm import tqdm
from wandb_utils import log_metrics
from datetime import datetime
from multi_task_evaluation import evaluate_predictions, compute_dataset_metrics
from wandb_utils import init_wandb, log_metrics, log_line_series, finish

# ---------------------------
# CONFIG (loaded from YAML)
# ---------------------------
CFG_PATH = os.path.join(os.path.dirname(__file__), "config.yaml")

# Set CUDA_VISIBLE_DEVICES before any CUDA operations
if os.path.exists(CFG_PATH):
    import yaml
    with open(CFG_PATH, 'r') as f:
        config_data = yaml.safe_load(f)
    
    if 'system' in config_data and 'cuda_visible_devices' in config_data['system']:
        os.environ['CUDA_VISIBLE_DEVICES'] = config_data['system']['cuda_visible_devices']
        print(f"[INFO] Set CUDA_VISIBLE_DEVICES to: {config_data['system']['cuda_visible_devices']}")

cfg = OmegaConf.load(CFG_PATH)

TRAIN_DATA_FILE =  cfg.data.train_file
VAL_DATA_FILE = cfg.data.val_file
TEST_DATA_FILE = cfg.data.test_file

TOKENIZER_NAME = cfg.model.tokenizer_name
PROCESSOR_NAME = cfg.model.processor_name
TRAINING_STRATEGY = cfg.model.training_strategy
DEVICE_MAP = cfg.model.device_map
TORCH_DTYPE = cfg.model.torch_dtype

# Convert torch_dtype string to actual torch dtype
if TORCH_DTYPE == "float16":
    TORCH_DTYPE = torch.float16
elif TORCH_DTYPE == "float32":
    TORCH_DTYPE = torch.float32
elif TORCH_DTYPE == "bfloat16":
    TORCH_DTYPE = torch.bfloat16
else:
    TORCH_DTYPE = torch.float16  # default

TRAIN_BATCH_SIZE = cfg.train.train_batch_size
VAL_BATCH_SIZE = cfg.train.val_batch_size
LR = float(cfg.train.lr)
EPOCHS = int(cfg.train.epochs)
SAVE_CHECKPOINT_DIR = cfg.train.save_checkpoint_dir
LOAD_CHECKPOINT_PATH = cfg.train.load_checkpoint_path
SAVE_EVERY_N_EPOCHS = int(cfg.train.save_every_n_epochs)
DEBUG_DRY_RUN = bool(cfg.train.debug_dry_run)
GRADIENT_ACCUMULATION_STEPS = int(cfg.train.gradient_accumulation_steps)
NUM_WORKERS = int(cfg.train.num_workers)

# Validation configuration
VALIDATE_EVERY_N_EPOCHS = int(cfg.train.validate_every_n_epochs)
SAVE_BEST_MODEL = True
EARLY_STOPPING_PATIENCE = int(cfg.train.early_stopping_patience)

# Wandb configuration
USE_WANDB = bool(cfg.wandb.use)
WANDB_PROJECT = cfg.wandb.project
WANDB_ENTITY = cfg.wandb.entity

# Load label mapping from JSON file
LABEL_MAP_PATH = cfg.data.label_map_path
with open(LABEL_MAP_PATH, 'r') as f:
    label_config = json.load(f)

LABEL_MAP = label_config["label_mapping"]
NUM_CLASSES = label_config["num_classes"]

print(f"[INFO] Loaded label mapping with {NUM_CLASSES} classes from {LABEL_MAP_PATH}")
print(f"[INFO] Available datasets: {', '.join(label_config['datasets'])}")
print(f"[INFO] Gradient accumulation: {GRADIENT_ACCUMULATION_STEPS} steps (effective batch size: {TRAIN_BATCH_SIZE * GRADIENT_ACCUMULATION_STEPS})")
print(f"[INFO] Data loading: {NUM_WORKERS} worker processes (0 = single-threaded, {NUM_WORKERS}+ = multi-threaded)")

# LoRA Configuration (only used when TRAINING_STRATEGY = "lora")
LORA_CONFIG = {
    'r': int(cfg.model.lora_config.r),
    'alpha': int(cfg.model.lora_config.alpha),
    'dropout': float(cfg.model.lora_config.dropout),
    'target_modules': list(cfg.model.lora_config.target_modules),
}

# TODO: this should wrap around the huggingface loading stuff
tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_NAME)
processor = AutoProcessor.from_pretrained(PROCESSOR_NAME)

config = OmegaConf.create(dict(cfg.dataset_config))

# ---------------------------
# TRAINER
# ---------------------------
class OmniClassifierTrainer:
    def __init__(self, data_files, val_data_files, test_data_files, tokenizer, processor, config, 
                 batch_size, val_batch_size, lr, epochs, save_checkpoint_dir, load_checkpoint_path, model, device_map, gradient_accumulation_steps, num_workers=0, use_lora=False):
        self.data_files = data_files
        self.val_data_files = val_data_files
        self.test_data_files = test_data_files
        self.tokenizer = tokenizer
        self.processor = processor
        self.config = config
        self.batch_size = batch_size
        self.val_batch_size = val_batch_size
        self.gradient_accumulation_steps = gradient_accumulation_steps
        self.num_workers = num_workers
        self.lr = lr
        self.epochs = epochs
        self.model = model
        self.label_key = config.get("label_key", "answer")
        # Use the label map loaded from JSON
        self.label_map = LABEL_MAP
        # Store device_map for later use
        self.device_map = device_map
        
        # Checkpoint IO setup
        self.checkpoint_dir = save_checkpoint_dir
        self.load_checkpoint_path = load_checkpoint_path

        # Training state
        self.best_val_acc = 0.0
        self.epochs_without_improvement = 0
        self.training_history = {
            'train_loss': [],
            'train_acc': [],
            'val_loss': [],
            'val_acc': [],
            'val_precision': [],
            'val_recall': [],
            'val_f1': [],
            'val_macro_f1': [],
            'val_weighted_f1': [],
            'val_micro_f1': []
        }

        if device_map == "auto":
            print("[INFO] Using device_map='auto' — model is already distributed/sharded")
            # anchor inputs on the device of the first shard/parameter
            first_device = next(self.model.parameters()).device
            self.device = torch.device(first_device)
            print(f"[INFO] Model distributed across devices. First device: {first_device}")
            print(f"[INFO] Available GPUs: {torch.cuda.device_count()}")
            for i in range(torch.cuda.device_count()):
                print(f"[INFO] GPU {i}: {torch.cuda.get_device_name(i)}")
        else:
            self.device = torch.device(device_map)

        # Initialize wandb
        if USE_WANDB:
            self._init_wandb()
        
        # Initialize training start time
        self.start_time = time.time()

    def _init_wandb(self):
        """Initialize wandb logging via wandb_utils."""
        wandb_config = {
            "model_name": TOKENIZER_NAME,
            "training_strategy": TRAINING_STRATEGY,
            "batch_size": self.batch_size,
            "val_batch_size": self.val_batch_size,
            "gradient_accumulation_steps": self.gradient_accumulation_steps,
            "effective_batch_size": self.batch_size * self.gradient_accumulation_steps,
            "learning_rate": self.lr,
            "epochs": self.epochs,
            "num_classes": NUM_CLASSES,
            "validate_every_n_epochs": VALIDATE_EVERY_N_EPOCHS,
            "early_stopping_patience": EARLY_STOPPING_PATIENCE,
            "save_best_model": SAVE_BEST_MODEL,
            "num_workers": self.num_workers,
            "lora_config": LORA_CONFIG if TRAINING_STRATEGY == "lora" else None,
            "label_map_path": LABEL_MAP_PATH,
            "datasets": label_config['datasets']
        }
        init_wandb(
            project=WANDB_PROJECT,
            entity=WANDB_ENTITY,
            config=wandb_config,
            run_name=f"omni_classifier_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        )

    def get_dataloader(self, data_files, batch_size, shuffle=True):
        dataset = OmniClassifierDataset(
            data_files=data_files,
            tokenizer=self.tokenizer,
            config=self.config,
            processor=self.processor,
            label_key=self.label_key,
            label_map=self.label_map
        )
        return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, collate_fn=collate_fn,
                          num_workers=self.num_workers, pin_memory=False, persistent_workers=self.num_workers > 0)

    def validate(self, dataloader, split_name="validation"):
        """Validate the model on the given dataloader."""
        self.model.eval()
        total_loss = 0.0
        all_predictions = []
        all_labels = []
        all_datasets = []
        criterion = CrossEntropyLoss()
        
        with torch.no_grad():
            for batch in tqdm(dataloader, desc=f"{split_name.capitalize()}", leave=False):
                if 'input_ids' not in batch or 'labels' not in batch:
                    raise KeyError(f"Batch missing required keys. Got: {list(batch.keys())}")

                # When using device_map="auto", let the model handle device placement
                if hasattr(self, 'device_map') and self.device_map == "auto":
                    input_ids = batch['input_ids']
                    labels = batch['labels']
                    attention_mask = batch.get('attention_mask', None)
                else:
                    input_ids = batch['input_ids'].to(self.device)
                    labels = batch['labels'].to(self.device)
                    attention_mask = batch.get('attention_mask', None)
                    
                    if attention_mask is not None:
                        attention_mask = attention_mask.to(self.device)

                # Handle labels shape
                if labels.dim() != 1:
                    if labels.dim() == 2 and labels.size(1) == NUM_CLASSES:
                        labels = labels.argmax(dim=1)
                    else:
                        raise ValueError(f"Unexpected labels shape {labels.shape}")

                logits = self.model(input_ids, attention_mask=attention_mask)
                
                # Move labels to the same device as logits when using device_map="auto"
                if self.device_map == "auto":
                    labels = labels.to(logits.device)
                
                loss = criterion(logits, labels)
                
                total_loss += loss.item() * input_ids.size(0)
                preds = logits.argmax(dim=1)
                
                all_predictions.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
                
                # Extract dataset information if available
                if 'dataset' in batch:
                    # feed the datasets in
                    all_datasets.extend(batch['dataset'])

        # Calculate average loss
        avg_loss = total_loss / max(1, len(all_labels))
        
        # Use the new evaluation module
        # All predictions are included in the list
        # All labels are included in the list
        evaluation_results = evaluate_predictions(
            predictions=all_predictions,
            ground_truths=all_labels,
            datasets=all_datasets if all_datasets else None,
            num_classes=NUM_CLASSES,
            split_name=split_name,
            
        )
        
        # Extract aggregate metrics (aligned with multi_task_evaluation)
        aggregate_metrics = evaluation_results["aggregate_metrics"]
        accuracy = aggregate_metrics.get("micro_accuracy", 0.0)
        f1 = aggregate_metrics.get("micro_f1", 0.0)
        precision = aggregate_metrics.get("micro_precision", 0.0)
        recall = aggregate_metrics.get("micro_recall", 0.0)
        
        print(f"{split_name.capitalize()} - Loss: {avg_loss:.4f} - Acc: {accuracy:.4f} - F1: {f1:.4f}")
        print(f"  Macro F1: {aggregate_metrics.get('macro_f1', 0.0):.4f} - Weighted F1: {aggregate_metrics.get('weighted_f1', 0.0):.4f}")
        
        return {
            'loss': avg_loss,
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'predictions': all_predictions,
            'labels': all_labels,
            'evaluation_results': evaluation_results,
            'aggregate_metrics': aggregate_metrics
        }

    def debug_batch_loader(self, n_batches: int = 1):
        print("[DEBUG] Running batch loader probe...")
        dl = self.get_dataloader(self.data_files, self.batch_size)
        for bi, batch in enumerate(dl):
            print(f"Batch {bi} keys:", list(batch.keys()))
            # Print shapes/types for all keys
            for k, v in batch.items():
                try:
                    if hasattr(v, 'shape'):
                        print(f"  - {k}: shape {tuple(v.shape)} dtype {getattr(v, 'dtype', None)} device {getattr(v, 'device', None)}")
                    elif isinstance(v, (list, tuple)):
                        print(f"  - {k}: list(len={len(v)}) sample_type={type(v[0]) if len(v) else None}")
                    else:
                        print(f"  - {k}: type {type(v)}")
                except Exception as e:
                    print(f"  - {k}: <print error: {e}>")
            if bi + 1 >= n_batches:
                break
        print("[DEBUG] Batch loader probe complete.")

    def _find_latest_checkpoint(self):
        """Find any .pt file in the checkpoint directory."""
        if not os.path.exists(self.checkpoint_dir):
            return None
        
        checkpoint_files = []
        for file in os.listdir(self.checkpoint_dir):
            if file.endswith(".pt"):
                checkpoint_files.append(os.path.join(self.checkpoint_dir, file))
        
        if not checkpoint_files:
            return None
        
        # Return the first .pt file found (or you could implement more sophisticated logic)
        return checkpoint_files[0]

    def load_checkpoint(self, optimizer):
        """Load the latest checkpoint if available."""
        # Prefer explicit load path if provided, else auto-find latest in save dir
        latest_checkpoint = self.load_checkpoint_path if self.load_checkpoint_path else self._find_latest_checkpoint()
        if latest_checkpoint:
            try:
                print(f"Loading checkpoint from {latest_checkpoint}")
                checkpoint = torch.load(latest_checkpoint, map_location=self.device)
                
                # Standard checkpoint loading
                if 'model_state_dict' in checkpoint:
                    self.model.load_state_dict(checkpoint['model_state_dict'], strict=False)
                else:
                    self.model.load_state_dict(checkpoint, strict=False)
                
                # Load optimizer state if available
                if 'optimizer_state_dict' in checkpoint:
                    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
                
                # Load training state
                if 'best_val_acc' in checkpoint:
                    self.best_val_acc = checkpoint['best_val_acc']
                if 'epochs_without_improvement' in checkpoint:
                    self.epochs_without_improvement = checkpoint['epochs_without_improvement']
                if 'training_history' in checkpoint:
                    self.training_history = checkpoint['training_history']
                
                start_epoch = checkpoint.get('epoch', 0)
                print(f"Successfully loaded checkpoint from epoch {start_epoch}")
                return start_epoch
                
            except Exception as e:
                print(f"[WARN] Could not load checkpoint due to: {e}. Continuing fresh.")
        
        return 0

    def save_checkpoint(self, optimizer, epoch, is_best=False):
        """Save checkpoint with training state."""
        try:
            os.makedirs(self.checkpoint_dir, exist_ok=True)
            
            checkpoint_data = {
                'model_state_dict': self.model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'epoch': epoch + 1,
                'best_val_acc': self.best_val_acc,
                'epochs_without_improvement': self.epochs_without_improvement,
                'training_history': self.training_history,
                'config': {
                    'lr': self.lr,
                    'batch_size': self.batch_size,
                    'num_classes': NUM_CLASSES,
                    'freeze_backbone': TRAINING_STRATEGY
                }
            }
            
            # Save regular checkpoint
            checkpoint_path = os.path.join(self.checkpoint_dir, f"checkpoint_epoch_{epoch + 1}.pt")
            torch.save(checkpoint_data, checkpoint_path)
            
            # Save best model if this is the best so far
            if is_best:
                best_model_path = os.path.join(self.checkpoint_dir, "best_model.pt")
                torch.save(checkpoint_data, best_model_path)
                print(f"New best model saved with validation accuracy: {self.best_val_acc:.4f}")
            
            print(f"Checkpoint saved to {checkpoint_path}")
            
        except Exception as e:
            print(f"[WARN] Failed to save checkpoint: {e}")

    def train(self):
        train_dataloader = self.get_dataloader(self.data_files, self.batch_size, shuffle=True)
        val_dataloader = self.get_dataloader(self.val_data_files, self.val_batch_size, shuffle=False)
        
        optimizer = Adam(self.model.parameters(), lr=self.lr)
        criterion = CrossEntropyLoss()

        # Load checkpoint if available
        start_epoch = self.load_checkpoint(optimizer)

        for epoch in tqdm(range(start_epoch, self.epochs), desc="Epochs", position=0):
            # Training phase
            self.model.train()
            total_loss = 0.0
            correct = 0
            total = 0
            epoch_start_time = time.time()

            for batch_idx, batch in tqdm(enumerate(train_dataloader), desc="Training", total=len(train_dataloader)):
                # --- defensive checks
                if 'input_ids' not in batch or 'labels' not in batch:
                    raise KeyError(f"Batch missing required keys. Got: {list(batch.keys())}")

                # When using device_map="auto", let the model handle device placement
                if self.device_map == "auto":
                    input_ids = batch['input_ids']
                    labels = batch['labels']
                    attention_mask = batch.get('attention_mask', None)
                else:
                    input_ids = batch['input_ids'].to(self.device)
                    labels = batch['labels'].to(self.device)
                    attention_mask = batch.get('attention_mask', None)
                    
                    if attention_mask is not None:
                        attention_mask = attention_mask.to(self.device)

                # labels sanity
                if labels.dim() != 1:
                    # If your dataset sometimes emits multi-task/one-hot, squeeze or argmax here
                    if labels.dim() == 2 and labels.size(1) == NUM_CLASSES:
                        labels = labels.argmax(dim=1)
                    else:
                        raise ValueError(f"Unexpected labels shape {labels.shape} (expected [B] or [B, C])")
            
                # Gradient accumulation: only zero gradients at the start of accumulation cycle
                if batch_idx % self.gradient_accumulation_steps == 0:
                    optimizer.zero_grad(set_to_none=True)
                
                logits = self.model(input_ids, attention_mask=attention_mask)

                if not torch.isfinite(logits).all():
                    raise FloatingPointError("Non-finite logits encountered")

                # Move labels to the same device as logits when using device_map="auto"
                if self.device_map == "auto":
                    labels = labels.to(logits.device)

                loss = criterion(logits, labels)
                if not torch.isfinite(loss):
                    raise FloatingPointError("Non-finite loss encountered")

                # Scale loss for gradient accumulation
                loss = loss / self.gradient_accumulation_steps
                loss.backward()
                
                # Only step optimizer at the end of accumulation cycle
                if (batch_idx + 1) % self.gradient_accumulation_steps == 0:
                    optimizer.step()

                with torch.no_grad():
                    total_loss += loss.item() * input_ids.size(0) * self.gradient_accumulation_steps
                    preds = logits.argmax(dim=1)
                    correct += (preds == labels).sum().item()
                    total += labels.size(0)

                # Log batch information to wandb (only log at accumulation boundaries for cleaner logs)
                if USE_WANDB and (batch_idx + 1) % self.gradient_accumulation_steps == 0:
                    batch_info = {
                        'batch_loss': loss.item() * self.gradient_accumulation_steps,  # Scale back for logging
                        'batch_accuracy': (preds == labels).float().mean().item(),
                        'batch_idx': batch_idx,
                        'effective_batch_size': self.batch_size * self.gradient_accumulation_steps,
                    }
                    
                    # Log batch metrics
                    log_metrics('batch_metrics_at_effective_batch_size_step', batch_info)

                # Log training progress statistics to wandb
                if USE_WANDB:
                    # Calculate progress statistics
                    batch_progress = (batch_idx + 1) / len(train_dataloader)
                    epoch_progress = (epoch + 1) / self.epochs
                    overall_progress = ((epoch * len(train_dataloader)) + batch_idx + 1) / (self.epochs * len(train_dataloader))
                    
                    # Calculate time statistics
                    elapsed_time = time.time() - epoch_start_time
                    total_elapsed = time.time() - self.start_time if hasattr(self, 'start_time') else elapsed_time
                    
                    # Calculate ETA
                    if batch_progress > 0:
                        epoch_eta = (elapsed_time / batch_progress) * (1 - batch_progress)
                        overall_eta = (total_elapsed / overall_progress) * (1 - overall_progress) if overall_progress > 0 else 0
                    else:
                        epoch_eta = 0
                        overall_eta = 0
                    
                    # Calculate training rate
                    training_rate = (batch_idx + 1) / max(1, elapsed_time)
                    
                    # Log progress statistics (numeric values only to avoid wandb media warnings)
                    progress_stats = {
                        'batch_progress': batch_progress,  # 0.0 to 1.0
                        'epoch_progress': epoch_progress,  # 0.0 to 1.0
                        'overall_progress': overall_progress,  # 0.0 to 1.0
                        'epoch_elapsed_time_seconds': elapsed_time,  # raw seconds
                        'epoch_eta_seconds': epoch_eta,  # raw seconds
                        'overall_eta_seconds': overall_eta,  # raw seconds
                        'training_rate': training_rate,  # batches per second
                        'current_epoch': epoch + 1,
                        'total_epochs': self.epochs,
                        'current_batch': batch_idx + 1,
                        'total_batches': len(train_dataloader)
                    }
                    
                    log_metrics('training_progress', progress_stats)
                    
                    # Optionally log formatted metrics as a table (uncomment if you want formatted strings)
                    # formatted_stats = {
                    #     'batch_progress': format_percentage(batch_idx + 1, len(train_dataloader)),
                    #     'epoch_progress': format_percentage(epoch + 1, self.epochs),
                    #     'overall_progress': format_percentage((epoch * len(train_dataloader)) + batch_idx + 1, self.epochs * len(train_dataloader)),
                    #     'epoch_elapsed_time': format_time(elapsed_time),
                    #     'epoch_eta': format_time(epoch_eta),
                    #     'overall_eta': format_time(overall_eta),
                    #     'training_rate': f"{training_rate:.2f}/s"
                    # }
                    # log_formatted_metrics('training_progress', formatted_stats)

                # step completes

            # Handle any remaining gradients at the end of epoch
            if len(train_dataloader) % self.gradient_accumulation_steps != 0:
                optimizer.step()

            # Calculate training metrics
            avg_train_loss = total_loss / max(1, total)
            train_acc = correct / max(1, total)
            
            # Store training metrics
            self.training_history['train_loss'].append(avg_train_loss)
            self.training_history['train_acc'].append(train_acc)
            
            print(f"Epoch {epoch+1}/{self.epochs} - Train Loss: {avg_train_loss:.4f} - Train Acc: {train_acc:.4f}")

            # Validation phase
            is_best = False
            if (epoch + 1) % VALIDATE_EVERY_N_EPOCHS == 0:
                val_results = self.validate(val_dataloader, "validation")
                
                # Store validation metrics
                self.training_history['val_loss'].append(val_results['loss'])
                self.training_history['val_acc'].append(val_results['accuracy'])
                self.training_history['val_precision'].append(val_results['precision'])
                self.training_history['val_recall'].append(val_results['recall'])
                self.training_history['val_f1'].append(val_results['f1'])
                
                # Store additional F1 metrics from aggregate_metrics
                aggregate_metrics = val_results['aggregate_metrics']
                self.training_history['val_macro_f1'].append(aggregate_metrics.get('macro_f1', 0.0))
                self.training_history['val_weighted_f1'].append(aggregate_metrics.get('weighted_f1', 0.0))
                self.training_history['val_micro_f1'].append(aggregate_metrics.get('micro_f1', 0.0))
                
                # Check if this is the best model (using micro F1 as primary metric)
                val_f1 = val_results['f1']
                if val_f1 > self.best_val_acc:
                    self.best_val_acc = val_f1
                    self.epochs_without_improvement = 0
                    is_best = True
                else:
                    self.epochs_without_improvement += 1
                
                print(f"Validation - Loss: {val_results['loss']:.4f} - Acc: {val_results['accuracy']:.4f} - F1: {val_f1:.4f}")
                print(f"Best validation F1 so far: {self.best_val_acc:.4f}")
                print(f"Epochs without improvement: {self.epochs_without_improvement}")
                
                # Log to wandb
                if USE_WANDB:
                    # Log training metrics
                    log_metrics('train', {
                        'loss': avg_train_loss,
                        'accuracy': train_acc
                    }, epoch=epoch + 1)
                    # Log validation loss and aggregate metrics
                    vm = {'loss': val_results['loss'], 'best_val_f1': self.best_val_acc, 'epochs_without_improvement': self.epochs_without_improvement}
                    for key, value in val_results['aggregate_metrics'].items():
                        vm[key] = value
                    log_metrics('val', vm, epoch=epoch + 1)
                    # Log per-dataset metrics if available
                    if 'per_dataset_metrics' in val_results['evaluation_results']:
                        log_metrics('val', val_results['evaluation_results']['per_dataset_metrics'], epoch=epoch + 1)
            else:
                # Log only training metrics
                if USE_WANDB:
                    log_metrics('train', {'loss': avg_train_loss, 'accuracy': train_acc}, epoch=epoch + 1)

            # Save checkpoint: every N epochs and also when best
            if SAVE_EVERY_N_EPOCHS and ((epoch + 1) % SAVE_EVERY_N_EPOCHS == 0):
                self.save_checkpoint(optimizer, epoch, is_best)
            elif is_best:
                # ensure best is saved even if not on save interval
                self.save_checkpoint(optimizer, epoch, is_best)
            
            # Early stopping
            if self.epochs_without_improvement >= EARLY_STOPPING_PATIENCE:
                print(f"Early stopping triggered after {EARLY_STOPPING_PATIENCE} epochs without improvement")
                break

            # continue to next epoch

        # Log final training history
        if USE_WANDB:
            # Create training curves
            epochs_list = list(range(1, len(self.training_history['train_loss']) + 1))
            log_line_series(
                name="training_curves",
                xs=epochs_list,
                ys_series=[self.training_history['train_loss'], self.training_history['val_loss']],
                keys=["Train Loss", "Val Loss"],
                title="Training and Validation Loss",
                xname="Epoch",
            )
            log_line_series(
                name="accuracy_curves",
                xs=epochs_list,
                ys_series=[self.training_history['train_acc'], self.training_history['val_acc']],
                keys=["Train Accuracy", "Val Accuracy"],
                title="Training and Validation Accuracy",
                xname="Epoch",
            )
            log_line_series(
                name="f1_curves",
                xs=epochs_list,
                ys_series=[
                    self.training_history['val_micro_f1'],
                    self.training_history['val_macro_f1'],
                    self.training_history['val_weighted_f1'],
                ],
                keys=["Micro F1", "Macro F1", "Weighted F1"],
                title="Validation F1 Scores",
                xname="Epoch",
            )

    def test(self):
        """Test the model on the test set."""
        print("\n" + "="*50)
        print("STARTING TESTING PHASE")
        print("="*50)
        
        # Load best model if available
        best_model_path = os.path.join(self.checkpoint_dir, "best_model.pt")
        if os.path.exists(best_model_path):
            print(f"Loading best model from {best_model_path}")
            checkpoint = torch.load(best_model_path, map_location=self.device)
            self.model.load_state_dict(checkpoint['model_state_dict'], strict=False)
        else:
            print("No best model found, using current model state")
        
        test_dataloader = self.get_dataloader(self.test_data_files, self.val_batch_size, shuffle=False)
        test_results = self.validate(test_dataloader, "test")
        
        print(f"\nFINAL TEST RESULTS:")
        print(f"Test Loss: {test_results['loss']:.4f}")
        print(f"Test Accuracy: {test_results['accuracy']:.4f}")
        print(f"Test Precision: {test_results['precision']:.4f}")
        print(f"Test Recall: {test_results['recall']:.4f}")
        print(f"Test F1: {test_results['f1']:.4f}")
        
        # Print detailed aggregate metrics
        aggregate_metrics = test_results['aggregate_metrics']
        print(f"\nDetailed Test Metrics:")
        print(f"  Micro Accuracy: {aggregate_metrics.get('micro_accuracy', 0.0):.4f}")
        print(f"  Micro F1: {aggregate_metrics.get('micro_f1', 0.0):.4f}")
        print(f"  Macro F1: {aggregate_metrics.get('macro_f1', 0.0):.4f}")
        print(f"  Weighted F1: {aggregate_metrics.get('weighted_f1', 0.0):.4f}")
        
        # Log test results to wandb
        if USE_WANDB:
            # Log scalar metrics
            tm = {
                'loss': test_results['loss'],
                'accuracy': test_results['accuracy'],
                'precision': test_results['precision'],
                'recall': test_results['recall'],
                'f1': test_results['f1'],
            }
            for key, value in test_results['aggregate_metrics'].items():
                tm[key] = value
            log_metrics('test', tm)
            # Per-dataset
            if 'per_dataset_metrics' in test_results['evaluation_results']:
                log_metrics('test', test_results['evaluation_results']['per_dataset_metrics'])
            # Finish wandb run
            finish()
        
        return test_results


if __name__ == "__main__":
    print(f"[INFO] Initializing model with {NUM_CLASSES} classes")
    
    if DEBUG_DRY_RUN:
        print("[INFO] Using DummyClassifier for dry-run (no backbone loaded).")
        model = DummyClassifier(num_classes=NUM_CLASSES)
    else:
        # Initialize model with training strategy
        print(f"[INFO] Initializing OmniClassifier with training strategy: {TRAINING_STRATEGY}")
        model = OmniClassifier(
            num_classes=NUM_CLASSES, 
            freeze_backbone=TRAINING_STRATEGY,
            lora_config=LORA_CONFIG if TRAINING_STRATEGY == "lora" else None,
            device_map=DEVICE_MAP,
            torch_dtype=TORCH_DTYPE
        )
        # Print trainable parameters info
        model.get_trainable_parameters()

    trainer = OmniClassifierTrainer(
        data_files=TRAIN_DATA_FILE,
        val_data_files=VAL_DATA_FILE,
        test_data_files=TEST_DATA_FILE,
        tokenizer=tokenizer,
        processor=processor,
        config=config,
        batch_size=TRAIN_BATCH_SIZE,
        val_batch_size=VAL_BATCH_SIZE,
        lr=LR,
        epochs=EPOCHS,
        save_checkpoint_dir=SAVE_CHECKPOINT_DIR,
        load_checkpoint_path=LOAD_CHECKPOINT_PATH,
        model=model,
        device_map=DEVICE_MAP,
        gradient_accumulation_steps=GRADIENT_ACCUMULATION_STEPS,
        num_workers=NUM_WORKERS
    )

    # 1) Dataloader probe (prints batch structure)
    # trainer.debug_batch_loader()

    # 2) Training with validation
    trainer.train()
    
    # 3) Testing
    test_results = trainer.test()