import os
import sys
import json
import torch
import time
import argparse
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
from datetime import datetime
from multi_task_evaluation import evaluate_predictions, compute_dataset_metrics
from wandb_utils import init_wandb, log_metrics, finish
from logger import log_training_metrics, log_validation_results, log_epoch_training_metrics

# Accelerate imports
from accelerate import Accelerator
from accelerate.utils import set_seed
from accelerate.logging import get_logger

logger = get_logger(__name__)

def parse_parameters():
    """
    Parse parameters from YAML config file with command-line argument overrides.
    
    Returns:
        dict: Dictionary containing all parsed parameters
    """
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Train OmniClassifier with Accelerate')
    
    # Data parameters
    parser.add_argument('--train_file', type=str, help='Training data file path')
    parser.add_argument('--val_file', type=str, help='Validation data file path')
    parser.add_argument('--test_file', type=str, help='Test data file path')
    parser.add_argument('--label_map_path', type=str, help='Path to label mapping JSON file')
    
    # Model parameters
    parser.add_argument('--tokenizer_name', type=str, help='Tokenizer model name')
    parser.add_argument('--processor_name', type=str, help='Processor model name')
    parser.add_argument('--training_strategy', type=str, 
                       choices=['head_only', 'lora', 'full'],
                       help='Training strategy: head_only, lora, or full')
    parser.add_argument('--device_map', type=str, help='Device mapping (auto, cpu, or specific devices)')
    parser.add_argument('--torch_dtype', type=str, 
                       choices=['float16', 'float32', 'bfloat16'],
                       help='PyTorch data type')
    
    # LoRA parameters
    parser.add_argument('--lora_r', type=int, help='LoRA rank')
    parser.add_argument('--lora_alpha', type=int, help='LoRA alpha')
    parser.add_argument('--lora_dropout', type=float, help='LoRA dropout')
    parser.add_argument('--lora_target_modules', type=str, nargs='+', 
                       help='LoRA target modules (space-separated list)')
    
    # Training parameters
    parser.add_argument('--train_batch_size', type=int, help='Training batch size')
    parser.add_argument('--val_batch_size', type=int, help='Validation batch size')
    parser.add_argument('--lr', type=float, help='Learning rate')
    parser.add_argument('--epochs', type=int, help='Number of training epochs')
    parser.add_argument('--save_checkpoint_dir', type=str, help='Directory to save checkpoints')
    parser.add_argument('--load_checkpoint_path', type=str, help='Path to load checkpoint from')
    parser.add_argument('--save_every_n_epochs', type=int, help='Save checkpoint every N epochs')
    parser.add_argument('--debug_dry_run', action='store_true', help='Enable debug dry run mode')
    parser.add_argument('--gradient_accumulation_steps', type=int, help='Gradient accumulation steps')
    parser.add_argument('--num_workers', type=int, help='Number of data loader workers')
    
    # Validation parameters
    parser.add_argument('--validate_every_n_epochs', type=str, 
                       help='Validate every N epochs (use "None" to disable)')
    parser.add_argument('--validate_every_n_steps', type=str, 
                       help='Validate every N steps (use "None" to disable)')
    parser.add_argument('--early_stopping_patience', type=int, help='Early stopping patience')
    
    # Dataset parameters
    parser.add_argument('--max_prompt_length', type=int, help='Maximum prompt length')
    parser.add_argument('--modalities', type=str, help='Comma-separated list of modalities')
    parser.add_argument('--prompt_key', type=str, help='Prompt key in dataset')
    parser.add_argument('--image_key', type=str, help='Image key in dataset')
    parser.add_argument('--video_key', type=str, help='Video key in dataset')
    parser.add_argument('--audio_key', type=str, help='Audio key in dataset')
    parser.add_argument('--label_key', type=str, help='Label key in dataset')
    parser.add_argument('--return_multi_modal_inputs', action='store_true', help='Return multi-modal inputs')
    parser.add_argument('--filter_overlong_prompts', action='store_true', help='Filter overlong prompts')
    parser.add_argument('--truncation', type=str, choices=['left', 'right'], help='Truncation direction')
    parser.add_argument('--format_prompt', type=str, help='Path to format prompt template')
    
    # Wandb parameters
    parser.add_argument('--use_wandb', action='store_true', help='Enable wandb logging')
    parser.add_argument('--project', type=str, help='Wandb project name')
    parser.add_argument('--entity', type=str, help='Wandb entity name')
    
    # System parameters
    parser.add_argument('--cuda_visible_devices', type=str, help='CUDA visible devices')
    parser.add_argument('--config_file', type=str, 
                       default='config_accelerate.yaml',
                       help='Path to config YAML file')
    
    args = parser.parse_args()
    
    # Load YAML config
    config_path = os.path.join(os.path.dirname(__file__), args.config_file)
    cfg = OmegaConf.load(config_path)
    
    # Override config with command line arguments
    
    # Data parameters
    if args.train_file is not None:
        cfg.data.train_file = args.train_file
    if args.val_file is not None:
        cfg.data.val_file = args.val_file
    if args.test_file is not None:
        cfg.data.test_file = args.test_file
    if args.label_map_path is not None:
        cfg.data.label_map_path = args.label_map_path
    
    # Model parameters
    if args.tokenizer_name is not None:
        cfg.model.tokenizer_name = args.tokenizer_name
    if args.processor_name is not None:
        cfg.model.processor_name = args.processor_name
    if args.training_strategy is not None:
        cfg.model.training_strategy = args.training_strategy
    if args.device_map is not None:
        cfg.model.device_map = args.device_map
    if args.torch_dtype is not None:
        cfg.model.torch_dtype = args.torch_dtype
    
    # LoRA parameters
    if args.lora_r is not None:
        cfg.model.lora_config.r = args.lora_r
    if args.lora_alpha is not None:
        cfg.model.lora_config.alpha = args.lora_alpha
    if args.lora_dropout is not None:
        cfg.model.lora_config.dropout = args.lora_dropout
    if args.lora_target_modules is not None:
        cfg.model.lora_config.target_modules = args.lora_target_modules
    
    # Training parameters
    if args.train_batch_size is not None:
        cfg.train.train_batch_size = args.train_batch_size
    if args.val_batch_size is not None:
        cfg.train.val_batch_size = args.val_batch_size
    if args.lr is not None:
        cfg.train.lr = args.lr
    if args.epochs is not None:
        cfg.train.epochs = args.epochs
    if args.save_checkpoint_dir is not None:
        cfg.train.save_checkpoint_dir = args.save_checkpoint_dir
    if args.load_checkpoint_path is not None:
        cfg.train.load_checkpoint_path = args.load_checkpoint_path
    if args.save_every_n_epochs is not None:
        cfg.train.save_every_n_epochs = args.save_every_n_epochs
    if args.debug_dry_run:
        cfg.train.debug_dry_run = True
    if args.gradient_accumulation_steps is not None:
        cfg.train.gradient_accumulation_steps = args.gradient_accumulation_steps
    if args.num_workers is not None:
        cfg.train.num_workers = args.num_workers
    
    # Validation parameters
    if args.validate_every_n_epochs is not None:
        if args.validate_every_n_epochs == "None":
            cfg.train.validate_every_n_epochs = None
        else:
            cfg.train.validate_every_n_epochs = int(args.validate_every_n_epochs)
    if args.validate_every_n_steps is not None:
        if args.validate_every_n_steps == "None":
            cfg.train.validate_every_n_steps = None
        else:
            cfg.train.validate_every_n_steps = int(args.validate_every_n_steps)
    if args.early_stopping_patience is not None:
        cfg.train.early_stopping_patience = args.early_stopping_patience
    
    # Dataset parameters
    if args.max_prompt_length is not None:
        cfg.dataset_config.max_prompt_length = args.max_prompt_length
    if args.modalities is not None:
        cfg.dataset_config.modalities = args.modalities
    if args.prompt_key is not None:
        cfg.dataset_config.prompt_key = args.prompt_key
    if args.image_key is not None:
        cfg.dataset_config.image_key = args.image_key
    if args.video_key is not None:
        cfg.dataset_config.video_key = args.video_key
    if args.audio_key is not None:
        cfg.dataset_config.audio_key = args.audio_key
    if args.label_key is not None:
        cfg.dataset_config.label_key = args.label_key
    if args.return_multi_modal_inputs:
        cfg.dataset_config.return_multi_modal_inputs = True
    if args.filter_overlong_prompts:
        cfg.dataset_config.filter_overlong_prompts = True
    if args.truncation is not None:
        cfg.dataset_config.truncation = args.truncation
    if args.format_prompt is not None:
        cfg.dataset_config.format_prompt = args.format_prompt
    
    # Wandb parameters
    if args.use_wandb:
        cfg.wandb.use = True
    if args.project is not None:
        cfg.wandb.project = args.project
    if args.entity is not None:
        cfg.wandb.entity = args.entity
    
    # System parameters
    if args.cuda_visible_devices is not None:
        if not hasattr(cfg, 'system'):
            cfg.system = OmegaConf.create({})
        cfg.system.cuda_visible_devices = args.cuda_visible_devices
    
    # Parse all parameters
    params = {}
    
    # Data parameters
    params['train_data_file'] = cfg.data.train_file
    params['val_data_file'] = cfg.data.val_file
    params['test_data_file'] = cfg.data.test_file
    params['label_map_path'] = cfg.data.label_map_path
    
    # Model parameters
    params['tokenizer_name'] = cfg.model.tokenizer_name
    params['processor_name'] = cfg.model.processor_name
    params['training_strategy'] = cfg.model.training_strategy
    params['device_map'] = cfg.model.device_map
    params['torch_dtype_str'] = cfg.model.torch_dtype
    
    # Convert torch_dtype string to actual torch dtype
    if params['torch_dtype_str'] == "float16":
        params['torch_dtype'] = torch.float16
    elif params['torch_dtype_str'] == "float32":
        params['torch_dtype'] = torch.float32
    elif params['torch_dtype_str'] == "bfloat16":
        params['torch_dtype'] = torch.bfloat16
    else:
        params['torch_dtype'] = torch.float16  # default
    
    # Training parameters
    params['train_batch_size'] = cfg.train.train_batch_size
    params['val_batch_size'] = cfg.train.val_batch_size
    params['lr'] = float(cfg.train.lr)
    params['epochs'] = int(cfg.train.epochs)
    params['save_checkpoint_dir'] = cfg.train.save_checkpoint_dir
    params['load_checkpoint_path'] = cfg.train.load_checkpoint_path
    params['save_every_n_epochs'] = int(cfg.train.save_every_n_epochs)
    params['debug_dry_run'] = bool(cfg.train.debug_dry_run)
    params['gradient_accumulation_steps'] = int(cfg.train.gradient_accumulation_steps)
    params['num_workers'] = int(cfg.train.num_workers)
    
    # Validation configuration
    params['validate_every_n_epochs'] = cfg.train.validate_every_n_epochs
    params['validate_every_n_steps'] = cfg.train.validate_every_n_steps
    if params['validate_every_n_steps'] is not None:
        params['validate_every_n_steps'] = int(params['validate_every_n_steps'])
    if params['validate_every_n_epochs'] is not None:
        params['validate_every_n_epochs'] = int(params['validate_every_n_epochs'])
    params['save_best_model'] = True
    params['early_stopping_patience'] = int(cfg.train.early_stopping_patience)
    
    # Wandb configuration
    params['use_wandb'] = bool(cfg.wandb.use)
    params['wandb_project'] = cfg.wandb.project
    params['wandb_entity'] = cfg.wandb.entity
    
    # Load label mapping from JSON file
    with open(params['label_map_path'], 'r') as f:
        label_config = json.load(f)
    
    params['label_map'] = label_config["label_mapping"]
    params['num_classes'] = label_config["num_classes"]
    
    # LoRA Configuration (only used when training_strategy = "lora")
    params['lora_config'] = {
        'r': int(cfg.model.lora_config.r),
        'alpha': int(cfg.model.lora_config.alpha),
        'dropout': float(cfg.model.lora_config.dropout),
        'target_modules': list(cfg.model.lora_config.target_modules),
    }
    
    # Dataset config
    params['dataset_config'] = OmegaConf.create(dict(cfg.dataset_config))
    
    # Print configuration summary
    print(f"[INFO] Training strategy: {params['training_strategy']}")
    print(f"[INFO] Loaded label mapping with {params['num_classes']} classes from {params['label_map_path']}")
    print(f"[INFO] Available datasets: {', '.join(label_config['datasets'])}")
    print(f"[INFO] Gradient accumulation: {params['gradient_accumulation_steps']} steps (effective batch size: {params['train_batch_size'] * params['gradient_accumulation_steps']})")
    print(f"[INFO] Data loading: {params['num_workers']} worker processes (0 = single-threaded, {params['num_workers']}+ = multi-threaded)")
    print(f"[INFO] Learning rate: {params['lr']}")
    print(f"[INFO] Epochs: {params['epochs']}")
    print(f"[INFO] Save checkpoint dir: {params['save_checkpoint_dir']}")
    if params['load_checkpoint_path']:
        print(f"[INFO] Load checkpoint path: {params['load_checkpoint_path']}")
    print(f"[INFO] Validate every N epochs: {params['validate_every_n_epochs']}")
    print(f"[INFO] Validate every N steps: {params['validate_every_n_steps']}")
    print(f"[INFO] Early stopping patience: {params['early_stopping_patience']}")
    print(f"[INFO] Wandb project: {params['wandb_project']}")
    
    return params, label_config

# Parse parameters
params, label_config = parse_parameters()

# Extract parameters to global variables for backward compatibility
TRAIN_DATA_FILE = params['train_data_file']
VAL_DATA_FILE = params['val_data_file']
TEST_DATA_FILE = params['test_data_file']
TOKENIZER_NAME = params['tokenizer_name']
PROCESSOR_NAME = params['processor_name']
TRAINING_STRATEGY = params['training_strategy']
DEVICE_MAP = params['device_map']
TORCH_DTYPE = params['torch_dtype']
TRAIN_BATCH_SIZE = params['train_batch_size']
VAL_BATCH_SIZE = params['val_batch_size']
LR = params['lr']
EPOCHS = params['epochs']
SAVE_CHECKPOINT_DIR = params['save_checkpoint_dir']
LOAD_CHECKPOINT_PATH = params['load_checkpoint_path']
SAVE_EVERY_N_EPOCHS = params['save_every_n_epochs']
DEBUG_DRY_RUN = params['debug_dry_run']
GRADIENT_ACCUMULATION_STEPS = params['gradient_accumulation_steps']
NUM_WORKERS = params['num_workers']
VALIDATE_EVERY_N_EPOCHS = params['validate_every_n_epochs']
VALIDATE_EVERY_N_STEPS = params['validate_every_n_steps']
SAVE_BEST_MODEL = params['save_best_model']
EARLY_STOPPING_PATIENCE = params['early_stopping_patience']
USE_WANDB = params['use_wandb']
WANDB_PROJECT = params['wandb_project']
WANDB_ENTITY = params['wandb_entity']
LABEL_MAP = params['label_map']
LABEL_MAP_PATH = params['label_map_path']
NUM_CLASSES = params['num_classes']
LORA_CONFIG = params['lora_config']
config = params['dataset_config']

# Load tokenizer and processor
tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_NAME)
processor = AutoProcessor.from_pretrained(PROCESSOR_NAME)

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
                # Set model to training mode (needed because validation sets it to eval mode)
                self.model.train()
                
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

                    # Log training metrics (inside torch.no_grad context)
                    log_training_metrics(
                        epoch=epoch,
                        batch_idx=batch_idx,
                        total_batches=len(train_dataloader),
                        loss=loss,
                        preds=preds,
                        labels=labels,
                        correct=correct,
                        total=total,
                        epoch_start_time=epoch_start_time,
                        start_time=self.start_time,
                        gradient_accumulation_steps=self.gradient_accumulation_steps,
                        batch_size=self.batch_size,
                        epochs=self.epochs,
                        accelerator=self.accelerator,
                        use_wandb=USE_WANDB
                    )

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
                            
                            # Add best_val_f1 and steps_without_improvement to val_results for logging
                            val_results['best_val_f1'] = self.best_val_acc
                            val_results['steps_without_improvement'] = self.steps_without_improvement
                            
                            # Log validation results
                            log_validation_results(
                                val_results=val_results,
                                current_step=current_step,
                                split_name="validation",
                                accelerator=self.accelerator,
                                use_wandb=USE_WANDB
                            )

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
                    
                    # Add best_val_f1 and epochs_without_improvement to val_results for logging
                    val_results['best_val_f1'] = self.best_val_acc
                    val_results['epochs_without_improvement'] = self.epochs_without_improvement
                    
                    # Log epoch training metrics
                    log_epoch_training_metrics(
                        epoch=epoch,
                        avg_train_loss=avg_train_loss,
                        train_acc=train_acc,
                        total_batches=len(train_dataloader),
                        accelerator=self.accelerator,
                        use_wandb=USE_WANDB
                    )
                    
                    # Log validation results
                    current_step = (epoch + 1) * len(train_dataloader)
                    log_validation_results(
                        val_results=val_results,
                        current_step=current_step,
                        split_name="validation",
                        accelerator=self.accelerator,
                        use_wandb=USE_WANDB
                    )
            else:
                # Log only training metrics
                log_epoch_training_metrics(
                    epoch=epoch,
                    avg_train_loss=avg_train_loss,
                    train_acc=train_acc,
                    total_batches=len(train_dataloader),
                    accelerator=self.accelerator,
                    use_wandb=USE_WANDB
                )

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

    # Training with validation
    trainer.train()
    
    # Testing
    test_results = trainer.test()
