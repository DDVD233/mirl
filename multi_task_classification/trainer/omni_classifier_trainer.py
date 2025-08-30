import os
import sys
import json
import torch
import time
from torch.utils.data import DataLoader
from torch.optim import Adam
from torch.nn import CrossEntropyLoss
from tqdm import tqdm
from datetime import datetime
from math import floor
from pathlib import Path
from transformers import get_scheduler
from math import ceil

# Add parent directory to path for imports
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

# Local imports
from mt_dataset.omni_classifier_dataset import OmniClassifierDataset
from verl.utils.dataset.rl_dataset import collate_fn
from utils.wandb_utils import init_wandb, log_metrics, finish
from utils.logger import log_batch_training_metrics, log_validation_results, log_epoch_training_metrics
from evaluate.multi_task_evaluation import evaluate_predictions

# Accelerate imports
from accelerate import Accelerator
from accelerate.utils import set_seed
from accelerate.logging import get_logger

logger = get_logger(__name__)

class OmniClassifierAccelerateTrainer:
    def __init__(self, data_files, val_data_files, test_data_files, tokenizer, processor, config, 
                 batch_size, val_batch_size, lr, epochs, save_checkpoint_dir, load_checkpoint_path, model, 
                 gradient_accumulation_steps, num_workers=0, use_lora=False, global_config=None):
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
        
        # Store global configuration for access to constants
        self.global_config = global_config or {}
        
        # Use the label map from global config
        self.label_map = self.global_config.get('LABEL_MAP', {})

        # Scheduler configuration
        self.use_scheduler = self.global_config.get("USE_SCHEDULER", True)
        self.scheduler_type = self.global_config.get("SCHEDULER_TYPE", "cosine")
        self.warmup_steps = self.global_config.get("WARMUP_STEPS", None)
        
        # Checkpoint IO setup
        self.checkpoint_dir = save_checkpoint_dir
        self.load_checkpoint_path = load_checkpoint_path

        # Training state
        self.best_val_acc = 0.0
        self.epochs_without_improvement = 0
        self.steps_without_improvement = 0  # For step-based early stopping

        # Initialize Accelerate
        use_wandb = self.global_config.get('USE_WANDB', False)
        self.accelerator = Accelerator(
            gradient_accumulation_steps=gradient_accumulation_steps,
            mixed_precision='fp16',  # Use fp16 for better memory efficiency
            log_with="wandb" if use_wandb else None,
            project_dir=save_checkpoint_dir if use_wandb else None,
        )
        
        # Set seed for reproducibility
        set_seed(42)
        
        # Initialize wandb
        if use_wandb and self.accelerator.is_main_process:
            self._init_wandb()
        
        # Initialize training start time
        self.start_time = time.time()
        
    def _init_wandb(self):
        """Initialize wandb logging via wandb_utils."""
        wandb_config = {
            "model_name": self.global_config.get('TOKENIZER_NAME', ''),
            "training_strategy": self.global_config.get('TRAINING_STRATEGY', ''),
            "batch_size": self.batch_size,
            "val_batch_size": self.val_batch_size,
            "gradient_accumulation_steps": self.gradient_accumulation_steps,
            "effective_batch_size": self.batch_size * self.gradient_accumulation_steps * self.accelerator.num_processes,
            "learning_rate": self.lr,
            "epochs": self.epochs,
            "num_classes": self.global_config.get('NUM_CLASSES', 0),
            "validate_every_n_epochs": self.global_config.get('VALIDATE_EVERY_N_EPOCHS', None),
            "validate_every_n_steps": self.global_config.get('VALIDATE_EVERY_N_STEPS', None),
            "early_stopping_patience": self.global_config.get('EARLY_STOPPING_PATIENCE', 0),
            "save_best_model": self.global_config.get('SAVE_BEST_MODEL', True),
            "num_workers": self.num_workers,
            "lora_config": self.global_config.get('LORA_CONFIG', None),
            "label_map_path": self.global_config.get('LABEL_MAP_PATH', ''),
            "datasets": self.global_config.get('label_config', {}).get('datasets', []),
            "accelerate": True,
            "mixed_precision": "fp16",
            "use_scheduler": self.use_scheduler,
            "scheduler_type": self.scheduler_type if self.use_scheduler else None,
            "warmup_steps": self.warmup_steps if self.use_scheduler else None
        }
        init_wandb(
            project=self.global_config.get('WANDB_PROJECT', ''),
            entity=self.global_config.get('WANDB_ENTITY', ''),
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


    def _latest_checkpoint_dir(self,base_dir: str):
        if not os.path.isdir(base_dir):
            return None
        # expect subdirs like step_00001234
        subs = [p for p in Path(base_dir).glob("step_*") if p.is_dir()]
        if not subs:
            return None
        subs.sort(key=lambda p: int(p.name.split("_")[-1]))
        return str(subs[-1])

    def save_checkpoint_unified(
        self,
        accelerator,
        model,
        epoch: int,
        batch_idx: int,
        len_train_dataloader: int,
        training_strategy: str,
        base_ckpt_dir: str,
    ):
        """
        Saves a resume-ready checkpoint+meta. Uses your definition:
        current_step = epoch * len_dl + (batch_idx + 1)
        """
        # accelerator.wait_for_everyone()
        
        global_step = epoch * len_train_dataloader + (batch_idx + 1)

        ckpt_dir = os.path.join(base_ckpt_dir, f"step_{global_step}")
        os.makedirs(ckpt_dir, exist_ok=True)

        # 1) Save Accelerate state: model, optimizer, scaler, RNG, registered objs
        accelerator.save_state(ckpt_dir)

        # 2) Minimal meta sidecar
        meta = {
            "epoch": int(epoch),
            "global_step": int(global_step),
            "len_train_dataloader": int(len_train_dataloader),
            "training_strategy": str(training_strategy),
            "saved_at_unix": time.time(),
        }
        with open(os.path.join(ckpt_dir, "meta.json"), "w") as f:
            json.dump(meta, f, indent=2)

        accelerator.print(f"[save] checkpoint @ step {global_step} → {ckpt_dir}")

        return ckpt_dir

    def load_checkpoint_unified(
        self,
        accelerator,
        model,                 # already built for the chosen strategy and wrapped by prepare()
        base_ckpt_dir: str,
        explicit_dir: str|None = None,
        expect_training_strategy: str|None = None,
    ):
        """
        Rebuild & accelerator.prepare() your model/optimizer/dataloaders first.
        Then call this loader to restore state and compute (start_epoch, start_batch_offset).
        Returns: (start_epoch, start_batch_offset, global_step, meta, ckpt_dir)
        """
        ckpt_dir = explicit_dir or self._latest_checkpoint_dir(base_ckpt_dir)
        if ckpt_dir is None:
            accelerator.print("[load] no checkpoint found; starting fresh.")
            return 0, 0, 0, None, None

        meta_path = os.path.join(ckpt_dir, "meta.json")
        if not os.path.isfile(meta_path):
            accelerator.print(f"[load] missing meta.json in {ckpt_dir}; starting fresh.")
            return 0, 0, 0, None, None

        with open(meta_path, "r") as f:
            meta = json.load(f)

        if expect_training_strategy and meta.get("training_strategy") != expect_training_strategy:
            accelerator.print(f"[warn] strategy mismatch: expected {expect_training_strategy}, got {meta.get('training_strategy')}")

        # 1) Restore everything the accelerator saved (model/opt/scaler/RNG/registered)
        accelerator.load_state(ckpt_dir)

        # 2) Compute resume positions using your step definition
        global_step = int(meta["global_step"])
        len_dl = int(meta["len_train_dataloader"])
        if len_dl <= 0:
            accelerator.print("[load] invalid len_train_dataloader; starting at epoch 0.")
            return 0, 0, 0, meta, ckpt_dir

        start_epoch = floor((global_step - 1) / len_dl)
        start_batch_offset = (global_step - 1) % len_dl

        accelerator.print(f"[load] resumed {ckpt_dir} → epoch={start_epoch}, step={global_step}, offset={start_batch_offset}")
    
        return start_epoch, start_batch_offset, global_step, meta, ckpt_dir


    def validate(self, val_dataloader, split_name="validation"):
        """Validate the model on the given dataloader."""
        self.model.eval()
        total_loss = 0.0
        all_predictions = []
        all_labels = []
        all_datasets = []
        criterion = CrossEntropyLoss()
        num_classes = self.global_config.get('NUM_CLASSES', 0)

        with torch.no_grad():
            for batch in tqdm(val_dataloader, desc="Validating", total=len(val_dataloader), disable=not self.accelerator.is_main_process):
                if 'input_ids' not in batch or 'labels' not in batch:
                    raise KeyError(f"Batch missing required keys. Got: {list(batch.keys())}")

                input_ids = batch['input_ids']
                labels = batch['labels']
                attention_mask = batch.get('attention_mask', None)

                # Handle labels shape
                if labels.dim() != 1:
                    if labels.dim() == 2 and labels.size(1) == num_classes:
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
                num_classes=num_classes,
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


    def train(self):
        train_dataloader = self.get_dataloader(self.data_files, self.batch_size, num_workers=self.num_workers, shuffle=True)
        val_dataloader = self.get_dataloader(self.val_data_files, self.val_batch_size, num_workers=self.num_workers, shuffle=False)
        
        optimizer = Adam(self.model.parameters(), lr=self.lr)
        criterion = CrossEntropyLoss()

        # ---- Compute update-steps-aware schedule sizes ----
        # updates_per_epoch = ceil(len(train_dataloader) / max(1, self.gradient_accumulation_steps))
        total_updates = self.epochs * len(train_dataloader)
        
        # Get the scheduler
        if self.use_scheduler:
            scheduler = get_scheduler(
                self.scheduler_type,
                optimizer,
                num_warmup_steps=self.warmup_steps,
                num_training_steps=total_updates
            )
            print(f"[INFO] Using {self.scheduler_type} scheduler with {self.warmup_steps} warmup steps")
        else:
            scheduler = None
            print("[INFO] Scheduler disabled - using constant learning rate")

        # Prepare everything with Accelerate
        if scheduler is not None:
            self.model, optimizer, train_dataloader, val_dataloader, scheduler = self.accelerator.prepare(
                self.model, optimizer, train_dataloader, val_dataloader, scheduler
            )
            # Register the scheduler for checkpointing
            self.accelerator.register_for_checkpointing(scheduler)
        else:
            self.model, optimizer, train_dataloader, val_dataloader = self.accelerator.prepare(
                self.model, optimizer, train_dataloader, val_dataloader
            )

        start_epoch, start_batch_offset, start_global_step, meta, ckpt_dir = self.load_checkpoint_unified(
            accelerator=self.accelerator,
            model=self.model,
            base_ckpt_dir=self.checkpoint_dir,
            explicit_dir=self.load_checkpoint_path or None,
            expect_training_strategy=self.global_config.get("TRAINING_STRATEGY"),
        )

        # Get configuration values
        validate_every_n_epochs = self.global_config.get('VALIDATE_EVERY_N_EPOCHS', None)
        validate_every_n_steps = self.global_config.get('VALIDATE_EVERY_N_STEPS', None)
        save_every_n_epochs = self.global_config.get('SAVE_EVERY_N_EPOCHS', None)
        early_stopping_patience = self.global_config.get('EARLY_STOPPING_PATIENCE', 0)
        use_wandb = self.global_config.get('USE_WANDB', False)
        num_classes = self.global_config.get('NUM_CLASSES', 0)

        for epoch in tqdm(range(start_epoch, self.epochs), desc="Epochs", position=0, disable=not self.accelerator.is_main_process):
            # Training phase
            self.model.train()

            # Handling the SAMPLER SHUFFLING
            if hasattr(train_dataloader, "sampler") and hasattr(train_dataloader.sampler, "set_epoch"):
                train_dataloader.sampler.set_epoch(epoch)  # required for proper per-epoch shuffling in DDP. :contentReference[oaicite:4]{index=4}

            total_loss = 0.0
            correct = 0
            total = 0
            epoch_start_time = time.time()

            # Variables for effective batch tracking (gradient updates)
            effective_batch_loss = 0.0
            effective_batch_correct = 0
            effective_batch_total = 0


            for batch_idx, batch in tqdm(enumerate(train_dataloader), desc="Training", total=len(train_dataloader), disable=not self.accelerator.is_main_process):
                # Set model to training mode (needed because validation sets it to eval mode)
                self.model.train()

                if epoch == start_epoch and batch_idx < start_batch_offset:
                    continue
                
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
                    if labels.dim() == 2 and labels.size(1) == num_classes:
                        labels = labels.argmax(dim=1)
                    else:
                        raise ValueError(f"Unexpected labels shape {labels.shape} (expected [B] or [B, C])")
                
                with self.accelerator.accumulate(self.model):
                    logits = self.model(input_ids, attention_mask=attention_mask)

                    if not torch.isfinite(logits).all():
                        raise FloatingPointError("Non-finite logits encountered")

                    loss = criterion(logits, labels)

                    if not torch.isfinite(loss):
                        raise FloatingPointError("Non-finite loss encountered")

                    # Accelerate handles gradient accumulation automatically
                    self.accelerator.backward(loss)

                    # NOTE: This if condition is not necessary as the accelerator 
                    # already syncs before stepping
                    # if self.accelerator.sync_gradients:
                    optimizer.step()

                    if scheduler is not None:
                        scheduler.step()

                    optimizer.zero_grad()
                    
                    # PURELY FOR LOGGING PURPOSES
                    if scheduler is not None:
                        did_update = False
                        if self.accelerator.sync_gradients:
                            did_update = True
                        if did_update:
                            current_lr = optimizer.param_groups[0]['lr'] 
                        else:
                            current_lr = None
                    else:
                        current_lr = self.lr
                        
                with torch.no_grad():
                    # Accumulate metrics for effective batch
                    effective_batch_loss += loss.item() * input_ids.size(0)
                    preds = logits.argmax(dim=1)
                    
                    # Gather accuracy metrics from all processes
                    gathered_preds = self.accelerator.gather_for_metrics(preds)
                    gathered_labels = self.accelerator.gather_for_metrics(labels)
                    
                    # Only compute global accuracy on main process
                    if self.accelerator.is_main_process:
                        effective_batch_correct += (gathered_preds == gathered_labels).sum().item()
                        effective_batch_total += gathered_labels.size(0)
                    # Also accumulate epoch-level metrics
                    total_loss += loss.item() * input_ids.size(0)
                    if self.accelerator.is_main_process:
                        correct += (gathered_preds == gathered_labels).sum().item()
                        total += gathered_labels.size(0)

                    # Check if this completes an effective batch (gradient update)
                    # An effective batch consists of gradient_accumulation_steps individual batches
       
                   
                    # Log training metrics for the effective batch
                    log_batch_training_metrics(
                        epoch=epoch,
                        batch_idx=batch_idx,
                        total_batches=len(train_dataloader),
                        loss=effective_batch_loss,  # Scalar value
                        correct=effective_batch_correct,
                        total=effective_batch_total,
                        epoch_start_time=epoch_start_time,
                        start_time=self.start_time,
                        gradient_accumulation_steps=self.gradient_accumulation_steps,
                        batch_size=self.batch_size,
                        epochs=self.epochs,
                        accelerator=self.accelerator,
                        use_wandb=use_wandb,
                        current_lr=current_lr
                    )

                    if (batch_idx + 1) % self.gradient_accumulation_steps == 0:  
                        # Reset the metrics at each gradient accumulation step
                        # After they are logged internally at every step
                        effective_batch_loss = 0.0
                        effective_batch_correct = 0
                        effective_batch_total = 0

                    # Step-based validation (if configured)
                    if validate_every_n_steps is not None and current_step % validate_every_n_steps == 0:
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
                                use_wandb=use_wandb
                            )

            # Calculate training metrics
            avg_train_loss = total_loss / max(1, total)
            train_acc = correct / max(1, total)
            
            if self.accelerator.is_main_process:
                print(f"Epoch {epoch+1}/{self.epochs} - Train Loss: {avg_train_loss:.4f} - Train Acc: {train_acc:.4f}")

            # Epoch-based validation phase (only if step-based validation is not configured)
            is_best = False
            if validate_every_n_epochs is not None and (epoch + 1) % validate_every_n_epochs == 0:
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
                        use_wandb=use_wandb
                    )

                    # Log validation results
                    current_step = (epoch + 1) * len(train_dataloader)
                    log_validation_results(
                        val_results=val_results,
                        current_step=current_step,
                        split_name="validation",
                        accelerator=self.accelerator,
                        use_wandb=use_wandb
                    )
            else:
                # Log only training metrics
                log_epoch_training_metrics(
                    epoch=epoch,
                    avg_train_loss=avg_train_loss,
                    train_acc=train_acc,
                    total_batches=len(train_dataloader),
                    accelerator=self.accelerator,
                    use_wandb=use_wandb
                )

            # Save checkpoint: every N epochs and also when best
            if save_every_n_epochs and ((epoch + 1) % save_every_n_epochs == 0):
                self.save_checkpoint_unified(
                accelerator=self.accelerator,
                model=self.model,
                epoch=epoch,
                batch_idx=batch_idx,
                len_train_dataloader=len(train_dataloader),
                training_strategy=self.global_config.get("TRAINING_STRATEGY"),
                base_ckpt_dir=self.checkpoint_dir,
            )
                
            elif is_best:
                # ensure best is saved even if not on save interval
                self.save_checkpoint_unified(
                accelerator=self.accelerator,
                model=self.model,
                epoch=epoch,
                batch_idx=batch_idx,
                len_train_dataloader=len(train_dataloader),
                training_strategy=self.global_config.get("TRAINING_STRATEGY"),
                base_ckpt_dir=self.checkpoint_dir,
            )
            
            # Early stopping
            if validate_every_n_steps is not None:
                # Step-based early stopping
                if self.steps_without_improvement >= early_stopping_patience:
                    if self.accelerator.is_main_process:
                        print(f"Early stopping triggered after {early_stopping_patience} steps without improvement")
                    break
            else:
                # Epoch-based early stopping
                if self.epochs_without_improvement >= early_stopping_patience:
                    if self.accelerator.is_main_process:
                        print(f"Early stopping triggered after {early_stopping_patience} epochs without improvement")
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
            
            # Load model state using helper method
            self._load_model_state_from_checkpoint(checkpoint, "best model")
            
            # Ensure the model is properly prepared with accelerate after loading
            # This is important because the model might have been unwrapped during loading
            
            self.model = self.accelerator.prepare(self.model)
   
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
            use_wandb = self.global_config.get('USE_WANDB', False)
            if use_wandb:
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
