import os
import json
import torch
from torch.utils.data import DataLoader
from torch.optim import Adam
from torch.nn import CrossEntropyLoss
import sys
from dummy_classifier import DummyClassifier
from omni_classifier import OmniClassifier
from omni_classifier_dataset import OmniClassifierDataset
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from verl.utils.dataset.rl_dataset import collate_fn
from transformers import AutoTokenizer, AutoProcessor
from omegaconf import OmegaConf
from tqdm import tqdm
import wandb
import numpy as np
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, classification_report
import time
from datetime import datetime
from multi_task_evaluation import evaluate_predictions, compute_dataset_metrics

# ---------------------------
# CONFIG
# ---------------------------
DATA_FILES = ["/Users/keane/Desktop/research/human-behavior/data/splits/audio_sigs_fixed_full_train.jsonl"]
VAL_DATA_FILES = ["/Users/keane/Desktop/research/human-behavior/data/splits/audio_sigs_fixed_full_val.jsonl"]  # Validation data
TEST_DATA_FILES = ["/Users/keane/Desktop/research/human-behavior/data/splits/audio_sigs_fixed_full_test.jsonl"]  # Test data
TOKENIZER_NAME = "Qwen/Qwen2.5-Omni-7B"
PROCESSOR_NAME = "Qwen/Qwen2.5-Omni-7B"
FREEZE_BACKBONE = "lora"  # Options: "head_only", "lora", "full" (or True/False for backward compatibility)
BATCH_SIZE = 1
VAL_BATCH_SIZE = 1  # Can be different from training batch size
LR = 1e-3
EPOCHS = 3
CHECKPOINT_DIR = ""
DEBUG_DRY_RUN = False  # <<< set True to avoid loading the real model

# Validation configuration
VALIDATE_EVERY_N_EPOCHS = 1  # Validate every N epochs
SAVE_BEST_MODEL = True  # Save best model based on validation accuracy
EARLY_STOPPING_PATIENCE = 5  # Stop training if validation accuracy doesn't improve for N epochs

# Wandb configuration
USE_WANDB = True
WANDB_PROJECT = "omni-classifier"
WANDB_ENTITY = None  # Set to your wandb username if needed

# Load label mapping from JSON file
LABEL_MAP_PATH = os.path.join(os.path.dirname(__file__), "meld_label_map.json")
with open(LABEL_MAP_PATH, 'r') as f:
    label_config = json.load(f)

LABEL_MAP = label_config["label_mapping"]
NUM_CLASSES = label_config["num_classes"]

print(f"[INFO] Loaded label mapping with {NUM_CLASSES} classes from {LABEL_MAP_PATH}")
print(f"[INFO] Available datasets: {', '.join(label_config['datasets'])}")

# LoRA Configuration (only used when FREEZE_BACKBONE = "lora")
LORA_CONFIG = {
    'r': 16,
    'alpha': 32,
    'dropout': 0.1,
    'target_modules': ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
}

# TODO: this should wrap around the huggingface loading stuff
tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_NAME)
processor = AutoProcessor.from_pretrained(PROCESSOR_NAME)

config = OmegaConf.create({
    "max_prompt_length": 4096,
    "modalities": "images,videos,audio",
    "prompt_key": "problem",
    "image_key": "images",
    "video_key": "videos",
    "audio_key": "audios",
    "label_key": "answer",  # <<< specify the label key here
    "filter_overlong_prompts": False,
    "return_multi_modal_inputs": True,
    "num_workers": 0,  # for debug, avoid multiprocessing noise
    "filter_overlong_prompts_workers": 0,
    "format_prompt": "/home/keaneong/human-behavior/verl/examples/format_prompt/default.jinja"
})

# ---------------------------
# TRAINER
# ---------------------------
class OmniClassifierTrainer:
    def __init__(self, data_files, val_data_files, test_data_files, tokenizer, processor, config, 
                 batch_size, val_batch_size, lr, epochs, checkpoint_dir, model, use_lora=False):
        self.data_files = data_files
        self.val_data_files = val_data_files
        self.test_data_files = test_data_files
        self.tokenizer = tokenizer
        self.processor = processor
        self.config = config
        self.batch_size = batch_size
        self.val_batch_size = val_batch_size
        self.lr = lr
        self.epochs = epochs
        self.checkpoint_dir = checkpoint_dir
        self.model = model
        self.label_key = config.get("label_key", "answer")
        # Use the label map loaded from JSON
        self.label_map = LABEL_MAP
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
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
        
        # Initialize wandb
        if USE_WANDB:
            self._init_wandb()

    def _init_wandb(self):
        """Initialize wandb logging."""
        wandb_config = {
            "model_name": TOKENIZER_NAME,
            "freeze_backbone": FREEZE_BACKBONE,
            "batch_size": self.batch_size,
            "val_batch_size": self.val_batch_size,
            "learning_rate": self.lr,
            "epochs": self.epochs,
            "num_classes": NUM_CLASSES,
            "validate_every_n_epochs": VALIDATE_EVERY_N_EPOCHS,
            "early_stopping_patience": EARLY_STOPPING_PATIENCE,
            "save_best_model": SAVE_BEST_MODEL,
            "lora_config": LORA_CONFIG if FREEZE_BACKBONE == "lora" else None,
            "label_map_path": LABEL_MAP_PATH,
            "datasets": label_config['datasets']
        }
        
        wandb.init(
            project=WANDB_PROJECT,
            entity=WANDB_ENTITY,
            config=wandb_config,
            name=f"omni_classifier_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        )
        
        # Log the complete configuration as a table
        config_table = wandb.Table(columns=["Parameter", "Value"])
        for key, value in wandb_config.items():
            config_table.add_data(key, str(value))
        wandb.log({"configuration": config_table})

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
                          num_workers=0, pin_memory=False, persistent_workers=False)

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
                loss = criterion(logits, labels)
                
                total_loss += loss.item() * input_ids.size(0)
                preds = logits.argmax(dim=1)
                
                all_predictions.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
                
                # Extract dataset information if available
                if 'dataset' in batch:
                    all_datasets.extend(batch['dataset'])

        # Calculate average loss
        avg_loss = total_loss / max(1, len(all_labels))
        
        # Use the new evaluation module
        evaluation_results = evaluate_predictions(
            predictions=all_predictions,
            ground_truths=all_labels,
            datasets=all_datasets if all_datasets else None,
            num_classes=NUM_CLASSES,
            split_name=split_name,
            log_to_wandb=USE_WANDB
        )
        
        # Extract key metrics for backward compatibility
        dataset_metrics = evaluation_results["dataset_metrics"]
        accuracy = dataset_metrics.get("micro_accuracy", 0.0)
        f1 = dataset_metrics.get("micro_f1", 0.0)
        precision = dataset_metrics.get("micro_precision", 0.0)
        recall = dataset_metrics.get("micro_recall", 0.0)
        
        print(f"{split_name.capitalize()} - Loss: {avg_loss:.4f} - Acc: {accuracy:.4f} - F1: {f1:.4f}")
        print(f"  Macro F1: {dataset_metrics.get('macro_f1', 0.0):.4f} - Weighted F1: {dataset_metrics.get('weighted_f1', 0.0):.4f}")
        
        return {
            'loss': avg_loss,
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'predictions': all_predictions,
            'labels': all_labels,
            'evaluation_results': evaluation_results,
            'dataset_metrics': dataset_metrics
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
        latest_checkpoint = self._find_latest_checkpoint()
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
                    'freeze_backbone': FREEZE_BACKBONE
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

    def train(self, max_steps: int | None = None):
        train_dataloader = self.get_dataloader(self.data_files, self.batch_size, shuffle=True)
        val_dataloader = self.get_dataloader(self.val_data_files, self.val_batch_size, shuffle=False)
        
        self.model.train().to(self.device)
        optimizer = Adam(self.model.parameters(), lr=self.lr)
        criterion = CrossEntropyLoss()

        # Load checkpoint if available
        start_epoch = self.load_checkpoint(optimizer)

        global_step = 0
        for epoch in tqdm(range(start_epoch, self.epochs), desc="Epochs", position=0):
            # Training phase
            self.model.train()
            total_loss = 0.0
            correct = 0
            total = 0

            for batch in tqdm(train_dataloader, desc="Training", total=len(train_dataloader), leave=False):
                # --- defensive checks
                if 'input_ids' not in batch or 'labels' not in batch:
                    raise KeyError(f"Batch missing required keys. Got: {list(batch.keys())}")

                input_ids = batch['input_ids'].to(self.device)
                labels = batch['labels'].to(self.device)

                # labels sanity
                if labels.dim() != 1:
                    # If your dataset sometimes emits multi-task/one-hot, squeeze or argmax here
                    if labels.dim() == 2 and labels.size(1) == NUM_CLASSES:
                        labels = labels.argmax(dim=1)
                    else:
                        raise ValueError(f"Unexpected labels shape {labels.shape} (expected [B] or [B, C])")

                attention_mask = batch.get('attention_mask', None)

                if attention_mask is not None:
                    attention_mask = attention_mask.to(self.device)
            
                optimizer.zero_grad(set_to_none=True)
                logits = self.model(input_ids, attention_mask=attention_mask)

                if not torch.isfinite(logits).all():
                    raise FloatingPointError("Non-finite logits encountered")

                loss = criterion(logits, labels)
                if not torch.isfinite(loss):
                    raise FloatingPointError("Non-finite loss encountered")

                loss.backward()
                optimizer.step()

                with torch.no_grad():
                    total_loss += loss.item() * input_ids.size(0)
                    preds = logits.argmax(dim=1)
                    correct += (preds == labels).sum().item()
                    total += labels.size(0)

                global_step += 1
                if max_steps is not None and global_step >= max_steps:
                    break

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
                
                # Store additional F1 metrics
                dataset_metrics = val_results['dataset_metrics']
                self.training_history['val_macro_f1'].append(dataset_metrics.get('macro_f1', 0.0))
                self.training_history['val_weighted_f1'].append(dataset_metrics.get('weighted_f1', 0.0))
                self.training_history['val_micro_f1'].append(dataset_metrics.get('micro_f1', 0.0))
                
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
                    # Log all dataset metrics
                    log_dict = {
                        'epoch': epoch + 1,
                        'train/loss': avg_train_loss,
                        'train/accuracy': train_acc,
                        'val/loss': val_results['loss'],
                        'best_val_f1': self.best_val_acc,
                        'epochs_without_improvement': self.epochs_without_improvement
                    }
                    
                    # Add all dataset metrics
                    for key, value in val_results['dataset_metrics'].items():
                        log_dict[f'val/{key}'] = value
                    
                    wandb.log(log_dict)
                    
                    # Log per-dataset metrics if available
                    if 'per_dataset_metrics' in val_results['evaluation_results']:
                        for key, value in val_results['evaluation_results']['per_dataset_metrics'].items():
                            wandb.log({f'val/{key}': value})
            else:
                # Log only training metrics
                if USE_WANDB:
                    wandb.log({
                        'epoch': epoch + 1,
                        'train/loss': avg_train_loss,
                        'train/accuracy': train_acc
                    })

            # Save checkpoint
            self.save_checkpoint(optimizer, epoch, is_best)
            
            # Early stopping
            if self.epochs_without_improvement >= EARLY_STOPPING_PATIENCE:
                print(f"Early stopping triggered after {EARLY_STOPPING_PATIENCE} epochs without improvement")
                break

            if max_steps is not None and global_step >= max_steps:
                break

        # Log final training history
        if USE_WANDB:
            # Create training curves
            epochs = list(range(1, len(self.training_history['train_loss']) + 1))
            wandb.log({
                "training_curves": wandb.plot.line_series(
                    xs=epochs,
                    ys=[self.training_history['train_loss'], self.training_history['val_loss']],
                    keys=["Train Loss", "Val Loss"],
                    title="Training and Validation Loss",
                    xname="Epoch"
                )
            })
            
            wandb.log({
                "accuracy_curves": wandb.plot.line_series(
                    xs=epochs,
                    ys=[self.training_history['train_acc'], self.training_history['val_acc']],
                    keys=["Train Accuracy", "Val Accuracy"],
                    title="Training and Validation Accuracy",
                    xname="Epoch"
                )
            })
            
            wandb.log({
                "f1_curves": wandb.plot.line_series(
                    xs=epochs,
                    ys=[
                        self.training_history['val_micro_f1'],
                        self.training_history['val_macro_f1'],
                        self.training_history['val_weighted_f1']
                    ],
                    keys=["Micro F1", "Macro F1", "Weighted F1"],
                    title="Validation F1 Scores",
                    xname="Epoch"
                )
            })

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
        
        # Print detailed metrics
        dataset_metrics = test_results['dataset_metrics']
        print(f"\nDetailed Test Metrics:")
        print(f"  Micro Accuracy: {dataset_metrics.get('micro_accuracy', 0.0):.4f}")
        print(f"  Micro F1: {dataset_metrics.get('micro_f1', 0.0):.4f}")
        print(f"  Macro F1: {dataset_metrics.get('macro_f1', 0.0):.4f}")
        print(f"  Weighted F1: {dataset_metrics.get('weighted_f1', 0.0):.4f}")
        
        # Log test results to wandb
        if USE_WANDB:
            # Log all dataset metrics
            log_dict = {
                'test/loss': test_results['loss'],
                'test/accuracy': test_results['accuracy'],
                'test/precision': test_results['precision'],
                'test/recall': test_results['recall'],
                'test/f1': test_results['f1']
            }
            
            # Add all dataset metrics
            for key, value in test_results['dataset_metrics'].items():
                log_dict[f'test/{key}'] = value
            
            wandb.log(log_dict)
            
            # Log per-dataset metrics if available
            if 'per_dataset_metrics' in test_results['evaluation_results']:
                for key, value in test_results['evaluation_results']['per_dataset_metrics'].items():
                    wandb.log({f'test/{key}': value})
            
            # Log detailed classification report
            class_metrics = test_results['evaluation_results']['class_metrics']
            class_report_table = wandb.Table(columns=["Class", "Precision", "Recall", "F1-Score", "Accuracy", "Support"])
            for class_name, metrics in class_metrics.items():
                class_report_table.add_data(
                    class_name,
                    metrics.get('precision', 0),
                    metrics.get('recall', 0),
                    metrics.get('f1', 0),
                    metrics.get('accuracy', 0),
                    metrics.get('count', 0)
                )
            wandb.log({"test/classification_report": class_report_table})
            
            # Finish wandb run
            wandb.finish()
        
        return test_results


if __name__ == "__main__":
    print(f"[INFO] Initializing model with {NUM_CLASSES} classes")
    
    if DEBUG_DRY_RUN:
        print("[INFO] Using DummyClassifier for dry-run (no backbone loaded).")
        model = DummyClassifier(num_classes=NUM_CLASSES)
    else:
        # Initialize model with training strategy
        print(f"[INFO] Initializing OmniClassifier with training strategy: {FREEZE_BACKBONE}")
        model = OmniClassifier(
            num_classes=NUM_CLASSES, 
            freeze_backbone=FREEZE_BACKBONE,
            lora_config=LORA_CONFIG if FREEZE_BACKBONE == "lora" else None
        )
        # Print trainable parameters info
        model.get_trainable_parameters()

    trainer = OmniClassifierTrainer(
        data_files=DATA_FILES,
        val_data_files=VAL_DATA_FILES,
        test_data_files=TEST_DATA_FILES,
        tokenizer=tokenizer,
        processor=processor,
        config=config,
        batch_size=BATCH_SIZE,
        val_batch_size=VAL_BATCH_SIZE,
        lr=LR,
        epochs=EPOCHS,
        checkpoint_dir=CHECKPOINT_DIR,
        model=model
    )

    # 1) Dataloader probe (prints batch structure)
    # trainer.debug_batch_loader()

    # 2) Training with validation
    trainer.train(max_steps=10000)
    
    # 3) Testing
    test_results = trainer.test()