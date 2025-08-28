import os
import sys
import json
import torch
import time
from torch.utils.data import DataLoader
from torch.optim import Adam
from torch.nn import CrossEntropyLoss
from others.dummy_classifier import DummyClassifier
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
from wandb_utils import init_wandb, log_metrics, finish

# Accelerate imports
from accelerate import Accelerator
from accelerate.utils import set_seed
from accelerate.logging import get_logger

logger = get_logger(__name__)

# ---------------------------
# CONFIG (loaded from YAML)
# ---------------------------
CFG_PATH = os.path.join(os.path.dirname(__file__), "config_accelerate.yaml")

# Set CUDA_VISIBLE_DEVICES before any CUDA operations
if os.path.exists(CFG_PATH):
    import yaml
    with open(CFG_PATH, 'r') as f:
        config_data = yaml.safe_load(f)
    
    # if 'system' in config_data and 'cuda_visible_devices' in config_data['system']:
    #     os.environ['CUDA_VISIBLE_DEVICES'] = config_data['system']['cuda_visible_devices']
    #     print(f"[INFO] Set CUDA_VISIBLE_DEVICES to: {config_data['system']['cuda_visible_devices']}")

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
VALIDATE_EVERY_N_EPOCHS = cfg.train.validate_every_n_epochs
VALIDATE_EVERY_N_STEPS = cfg.train.validate_every_n_steps
if VALIDATE_EVERY_N_STEPS is not None:
    VALIDATE_EVERY_N_STEPS = int(VALIDATE_EVERY_N_STEPS)
if VALIDATE_EVERY_N_EPOCHS is not None:
    VALIDATE_EVERY_N_EPOCHS = int(VALIDATE_EVERY_N_EPOCHS)
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
# ACCELERATE TRAINER
# ---------------------------
class OmniClassifierAccelerateTrainer:
    def __init__(self, data_files, val_data_files, test_data_files, tokenizer, processor, config, 
                 batch_size, val_batch_size, lr, epochs, save_checkpoint_dir, load_checkpoint_path, model, gradient_accumulation_steps, num_workers=0, use_lora=False):
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
        
        # Checkpoint IO setup
        self.checkpoint_dir = save_checkpoint_dir
        self.load_checkpoint_path = load_checkpoint_path

        # Training state
        self.best_val_acc = 0.0
        self.epochs_without_improvement = 0
        self.steps_without_improvement = 0  # For step-based early stopping

        # Initialize Accelerate
        self.accelerator = Accelerator(
            gradient_accumulation_steps=gradient_accumulation_steps,
            mixed_precision='fp16',  # Use fp16 for better memory efficiency
            log_with="wandb" if USE_WANDB else None,
            project_dir=save_checkpoint_dir if USE_WANDB else None,
        )
        
        # Set seed for reproducibility
        set_seed(42)
        
        # Initialize wandb
        if USE_WANDB and self.accelerator.is_main_process:
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
            "validate_every_n_steps": VALIDATE_EVERY_N_STEPS,
            "early_stopping_patience": EARLY_STOPPING_PATIENCE,
            "save_best_model": SAVE_BEST_MODEL,
            "num_workers": self.num_workers,
            "lora_config": LORA_CONFIG if TRAINING_STRATEGY == "lora" else None,
            "label_map_path": LABEL_MAP_PATH,
            "datasets": label_config['datasets'],
            "accelerate": True,
            "mixed_precision": "fp16"
        }
        init_wandb(
            project=WANDB_PROJECT,
            entity=WANDB_ENTITY,
            config=wandb_config,
            run_name=f"omni_classifier_accelerate_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        )

    def get_dataloader(self, data_files, batch_size, num_workers=0, shuffle=True):
        dataset = OmniClassifierDataset(
            data_files=data_files,
            tokenizer=self.tokenizer,
            config=self.config,
            processor=self.processor,
            label_key=self.label_key,
            label_map=self.label_map
        )
        return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, collate_fn=collate_fn,
                          num_workers=num_workers, pin_memory=True, persistent_workers=num_workers > 0)

    def validate(self, val_dataloader, split_name="validation"):
        """Validate the model on the given dataloader."""
        self.model.eval()
        total_loss = 0.0
        all_predictions = []
        all_labels = []
        all_datasets = []
        criterion = CrossEntropyLoss()

        with torch.no_grad():
            for batch in tqdm(val_dataloader, desc="Validating", total=len(val_dataloader), disable=not self.accelerator.is_main_process):
                if 'input_ids' not in batch or 'labels' not in batch:
                    raise KeyError(f"Batch missing required keys. Got: {list(batch.keys())}")

                input_ids = batch['input_ids']
                labels = batch['labels']
                attention_mask = batch.get('attention_mask', None)

                # Handle labels shape
                if labels.dim() != 1:
                    if labels.dim() == 2 and labels.size(1) == NUM_CLASSES:
                        labels = labels.argmax(dim=1)
                    else:
                        raise ValueError(f"Unexpected labels shape {labels.shape}")

                logits = self.model(input_ids, attention_mask=attention_mask)
                loss = criterion(logits, labels)
                
                total_loss += loss.item() * input_ids.size(0)
                preds = logits.argmax(dim=1)

                gathered_preds = self.accelerator.gather_for_metrics(preds)
        
                gathered_labels = self.accelerator.gather_for_metrics(labels)

                # Gather datasets from all processes (if available)
                gathered_datasets = None

                if 'dataset' in batch:
                    gathered_datasets = self.accelerator.gather_for_metrics(batch['dataset'])
                else:
                    # All processes must participate in gather_object, even if they don't have the data
                    gathered_datasets = self.accelerator.gather_for_metrics(None)
                
                # Only process on main process
                if self.accelerator.is_main_process:
                    all_predictions.extend(gathered_preds.cpu().numpy())
                    all_labels.extend(gathered_labels.cpu().numpy())
                    all_datasets.extend(gathered_datasets)

                # with open('/home/keaneong/human-behavior/verl/multi_task_classification/debug_batch.txt', 'a') as f: 
                #     f.write(f"stop here after gather, all_predictions: {all_predictions}\n")
                #     f.write(f"stop here after gather, all_labels: {all_labels}\n")
                #     f.write(f"stop here after gather, all_datasets: {all_datasets}\n")
                #     raise Exception(f"Stop here, all_predictions: {all_predictions}, all_labels: {all_labels}, all_datasets: {all_datasets}")

        # Calculate average loss
        avg_loss = total_loss / max(1, len(all_labels)) if self.accelerator.is_main_process else 0.0
        
        # Use the new evaluation module (only on main process)
        if self.accelerator.is_main_process:
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
            
            raise Exception(f"Stop here, all_predictions: {all_predictions}, all_labels: {all_labels}, all_datasets: {all_datasets}")

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
        else:
            return None

    def debug_batch_loader(self, n_batches: int = 1):
        if not self.accelerator.is_main_process:
            return
            
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
                checkpoint = torch.load(latest_checkpoint, map_location='cpu')
                
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
                
                
                start_epoch = checkpoint.get('epoch', 0)
                print(f"Successfully loaded checkpoint from epoch {start_epoch}")
                return start_epoch
                
            except Exception as e:
                print(f"[WARN] Could not load checkpoint due to: {e}. Continuing fresh.")
        
        return 0

    def save_checkpoint(self, optimizer, epoch, is_best=False):
        """Save checkpoint with training state."""
        if not self.accelerator.is_main_process:
            return
            
        try:
            os.makedirs(self.checkpoint_dir, exist_ok=True)
            
            checkpoint_data = {
                'model_state_dict': self.model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'epoch': epoch + 1,
                'best_val_acc': self.best_val_acc,
                'epochs_without_improvement': self.epochs_without_improvement,
    
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
        train_dataloader = self.get_dataloader(self.data_files, self.batch_size, num_workers=self.num_workers, shuffle=True)
        val_dataloader = self.get_dataloader(self.val_data_files, self.val_batch_size, num_workers=self.num_workers, shuffle=False)
        
        optimizer = Adam(self.model.parameters(), lr=self.lr)
        criterion = CrossEntropyLoss()

        # Prepare everything with Accelerate
        self.model, optimizer, train_dataloader, val_dataloader = self.accelerator.prepare(
            self.model, optimizer, train_dataloader, val_dataloader
        )

        # Load checkpoint if available
        start_epoch = self.load_checkpoint(optimizer)

        for epoch in tqdm(range(start_epoch, self.epochs), desc="Epochs", position=0, disable=not self.accelerator.is_main_process):
            # Training phase
            self.model.train()
            total_loss = 0.0
            correct = 0
            total = 0
            epoch_start_time = time.time()

            for batch_idx, batch in tqdm(enumerate(train_dataloader), desc="Training", total=len(train_dataloader), disable=not self.accelerator.is_main_process):
                # Calculate current step for validation checking
                current_step = (epoch * len(train_dataloader)) + batch_idx + 1
                # --- defensive checks
                if 'input_ids' not in batch or 'labels' not in batch:
                    raise KeyError(f"Batch missing required keys. Got: {list(batch.keys())}")

                input_ids = batch['input_ids']
                labels = batch['labels']
                attention_mask = batch.get('attention_mask', None)

                # labels sanity
                if labels.dim() != 1:
                    # If your dataset sometimes emits multi-task/one-hot, squeeze or argmax here
                    if labels.dim() == 2 and labels.size(1) == NUM_CLASSES:
                        labels = labels.argmax(dim=1)
                    else:
                        raise ValueError(f"Unexpected labels shape {labels.shape} (expected [B] or [B, C])")
            
                logits = self.model(input_ids, attention_mask=attention_mask)

                if not torch.isfinite(logits).all():
                    raise FloatingPointError("Non-finite logits encountered")

                loss = criterion(logits, labels)
                if not torch.isfinite(loss):
                    raise FloatingPointError("Non-finite loss encountered")

                # Accelerate handles gradient accumulation automatically
                self.accelerator.backward(loss)

                with torch.no_grad():
                    total_loss += loss.item() * input_ids.size(0)
                    preds = logits.argmax(dim=1)
                    
                    # Gather accuracy metrics from all processes
                    gathered_preds = self.accelerator.gather_for_metrics(preds)
                    gathered_labels = self.accelerator.gather_for_metrics(labels)
                    
                    # Only compute global accuracy on main process
                    if self.accelerator.is_main_process:
                        correct += (gathered_preds == gathered_labels).sum().item()
                        total += gathered_labels.size(0)
                        
                        # Debug prints for predictions and labels
                        if batch_idx < 3:  # Only print first 3 batches
                            print(f"\n[DEBUG] Batch {batch_idx} - Sample {0}:")
                            print(f"  Labels: {gathered_labels}")
                            print(f"  Predictions: {gathered_preds}")
                            print(f"  Correct: {(gathered_preds == gathered_labels).sum().item()}/{gathered_labels.size(0)}")
                            print(f"  Accuracy so far: {correct}/{total} = {correct/max(1, total):.4f}")
                        
                        # Check for potential issues
                        if torch.isnan(logits).any():
                            print("  WARNING: NaN values in logits!")
                        if torch.isinf(logits).any():
                            print("  WARNING: Inf values in logits!")
                        if len(torch.unique(preds)) == 1:
                            print("  WARNING: All predictions are the same!")
                        if logits.max() - logits.min() < 0.1:
                            print("  WARNING: Logits are very close to each other!")
                        

                # Log batch information to wandb (only on main process)
                if USE_WANDB and self.accelerator.is_main_process and (batch_idx + 1) % self.gradient_accumulation_steps == 0:
                    batch_info = {
                        'batch_loss': loss.item(),
                        'batch_accuracy': (preds == labels).float().mean().item(),
                        'batch_idx': batch_idx,
                        'effective_batch_size': self.batch_size * self.gradient_accumulation_steps,
                    }
                    
                    # Calculate current step for logging
                    current_step = (epoch * len(train_dataloader)) + batch_idx + 1
                    # Log batch metrics with step information
                    log_metrics('batch_metrics_at_effective_batch_size_step', batch_info, step=current_step)

                # Log training progress statistics to wandb
                if USE_WANDB and self.accelerator.is_main_process:
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
                        'batch_progress': batch_progress,  # 0.0 to 1.0 # progress of the current batches in the current epoch
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
                    
             
                    # Log progress with step information
                    log_metrics('training_progress', progress_stats, step=current_step)

                # Step-based validation (if configured)
                if VALIDATE_EVERY_N_STEPS is not None and current_step % VALIDATE_EVERY_N_STEPS == 0:
                    print(f"\n[STEP {current_step}] Running step-based validation...")
                    val_results = self.validate(val_dataloader, "validation")
                    
                    if self.accelerator.is_main_process and val_results is not None:
                        # Check if this is the best model (using micro F1 as primary metric)
                        val_f1 = val_results['f1']
                        if val_f1 > self.best_val_acc:
                            self.best_val_acc = val_f1
                            self.steps_without_improvement = 0
                            print(f"[STEP {current_step}] New best model! F1: {val_f1:.4f}")
                        else:
                            self.steps_without_improvement += 1
                        
                        print(f"[STEP {current_step}] Validation - Loss: {val_results['loss']:.4f} - Acc: {val_results['accuracy']:.4f} - F1: {val_f1:.4f}")
                        print(f"[STEP {current_step}] Best validation F1 so far: {self.best_val_acc:.4f}")
                        print(f"[STEP {current_step}] Steps without improvement: {self.steps_without_improvement}")
                        
                        # Log to wandb
                        if USE_WANDB:
                            # Log validation metrics at current step
                            vm = {'loss': val_results['loss'], 'best_val_f1': self.best_val_acc, 'steps_without_improvement': self.steps_without_improvement}
                            for key, value in val_results['aggregate_metrics'].items():
                                vm[key] = value
                            log_metrics('val', vm, step=current_step)
                            # Log per-dataset metrics if available
                            if 'per_dataset_metrics' in val_results['evaluation_results']:
                                log_metrics('val', val_results['evaluation_results']['per_dataset_metrics'], step=current_step)

            # Calculate training metrics
            avg_train_loss = total_loss / max(1, total)
            train_acc = correct / max(1, total)
            
            if self.accelerator.is_main_process:
                print(f"Epoch {epoch+1}/{self.epochs} - Train Loss: {avg_train_loss:.4f} - Train Acc: {train_acc:.4f}")

            # Epoch-based validation phase (only if step-based validation is not configured)
            is_best = False
            if VALIDATE_EVERY_N_EPOCHS is not None and (epoch + 1) % VALIDATE_EVERY_N_EPOCHS == 0:
                val_results = self.validate(val_dataloader, "validation")
                
                if self.accelerator.is_main_process and val_results is not None:
                    
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
                        # Calculate current step for logging (end of epoch)
                        current_step = (epoch + 1) * len(train_dataloader)
                        # Log training metrics
                        log_metrics('train', {
                            'loss': avg_train_loss,
                            'accuracy': train_acc
                        }, step=current_step)
                        # Log validation loss and aggregate metrics
                        vm = {'loss': val_results['loss'], 'best_val_f1': self.best_val_acc, 'epochs_without_improvement': self.epochs_without_improvement}
                        for key, value in val_results['aggregate_metrics'].items():
                            vm[key] = value
                        log_metrics('val', vm, step=current_step)
                        # Log per-dataset metrics if available
                        if 'per_dataset_metrics' in val_results['evaluation_results']:
                            log_metrics('val', val_results['evaluation_results']['per_dataset_metrics'], step=current_step)
            else:
                # Log only training metrics
                if USE_WANDB and self.accelerator.is_main_process:
                    # Calculate current step for logging (end of epoch)
                    current_step = (epoch + 1) * len(train_dataloader)
                    log_metrics('train', {'loss': avg_train_loss, 'accuracy': train_acc}, step=current_step)

            # Save checkpoint: every N epochs and also when best
            if SAVE_EVERY_N_EPOCHS and ((epoch + 1) % SAVE_EVERY_N_EPOCHS == 0):
                self.save_checkpoint(optimizer, epoch, is_best)
            elif is_best:
                # ensure best is saved even if not on save interval
                self.save_checkpoint(optimizer, epoch, is_best)
            
            # Early stopping
            if VALIDATE_EVERY_N_STEPS is not None:
                # Step-based early stopping
                if self.steps_without_improvement >= EARLY_STOPPING_PATIENCE:
                    if self.accelerator.is_main_process:
                        print(f"Early stopping triggered after {EARLY_STOPPING_PATIENCE} steps without improvement")
                    break
            else:
                # Epoch-based early stopping
                if self.epochs_without_improvement >= EARLY_STOPPING_PATIENCE:
                    if self.accelerator.is_main_process:
                        print(f"Early stopping triggered after {EARLY_STOPPING_PATIENCE} epochs without improvement")
                    break



    def test(self):
        """Test the model on the test set."""
        if not self.accelerator.is_main_process:
            return None
            
        print("\n" + "="*50)
        print("STARTING TESTING PHASE")
        print("="*50)
        
        # Load best model if available
        best_model_path = os.path.join(self.checkpoint_dir, "best_model.pt")
        if os.path.exists(best_model_path):
            print(f"Loading best model from {best_model_path}")
            checkpoint = torch.load(best_model_path, map_location='cpu')
            self.model.load_state_dict(checkpoint['model_state_dict'], strict=False)
        else:
            print("No best model found, using current model state")
        
        test_dataloader = self.get_dataloader(self.test_data_files, self.val_batch_size, shuffle=False)
        test_dataloader = self.accelerator.prepare(test_dataloader)
        test_results = self.validate(test_dataloader, "test")
        
        if test_results is not None:
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
                # Calculate final step for test logging;
                # just log to 0 for now
                # final_step = self.epochs * len(train_dataloader)
                final_step = 0
                log_metrics('test', tm, step=final_step)
                # Per-dataset
                if 'per_dataset_metrics' in test_results['evaluation_results']:
                    log_metrics('test', test_results['evaluation_results']['per_dataset_metrics'], step=final_step)
                # Finish wandb run
                finish()
            
            return test_results
        return None


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

    trainer = OmniClassifierAccelerateTrainer(
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
        gradient_accumulation_steps=GRADIENT_ACCUMULATION_STEPS,
        num_workers=NUM_WORKERS
    )

    # 1) Dataloader probe (prints batch structure)
    # trainer.debug_batch_loader()

    # 2) Training with validation
    trainer.train()
    
    # 3) Testing
    test_results = trainer.test()
