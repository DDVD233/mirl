import os
import sys
import json
import torch
import time
import random
import numpy as np
from torch.utils.data import DataLoader
from torch.optim import Adam
from torch.nn import CrossEntropyLoss
from tqdm import tqdm
from datetime import datetime

# Add parent directory to path for imports
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

# Local imports
from mt_dataset.omni_classifier_dataset import OmniClassifierDataset
from verl.utils.dataset.rl_dataset import collate_fn
from utils.wandb_utils import init_wandb, log_metrics, finish
from utils.logger import log_training_metrics, log_validation_results, log_epoch_training_metrics
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
            "effective_batch_size": self.batch_size * self.gradient_accumulation_steps,
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
            "mixed_precision": "fp16"
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

    def _load_model_state_from_checkpoint(self, checkpoint, context="checkpoint"):
        """
        Load model weights according to the training strategy saved in `checkpoint`.
        - head_only:   loads only classifier/head module params
        - lora:        loads PEFT adapters (preferred) + classifier/head params
        - full:        loads full model state dict
        Fallbacks handle missing artifacts gracefully.
        """
        
        # ASSUME THAT THE MODEL IS ALREADY UNWRAPPED
        training_strategy = checkpoint.get("training_strategy", "head_only")
        # with open("/home/keaneong/human-behavior/verl/multi_task_classification/classifier_state_dict_KEY.txt", "a") as f:
        #     f.write(f"Classifier state dict {checkpoint['classifier_state_dict']}")
        # raise Exception("Stop here")
        
        def _report_load(res):
            if isinstance(res, tuple):
                missing, unexpected = res
            else:
                missing, unexpected = res.missing_keys, res.unexpected_keys
            if missing:
                print(f"[load:{context}] missing keys: {missing[:8]}{' ...' if len(missing) > 8 else ''}")
            if unexpected:
                print(f"[load:{context}] unexpected keys: {unexpected[:8]}{' ...' if len(unexpected) > 8 else ''}")

        if training_strategy == "head_only" and "classifier_state_dict" in checkpoint:
            print(f"Loading head-only {context} (classifier/head only)...")

            res = self.model.classifier.load_state_dict(checkpoint["classifier_state_dict"], strict=False)
            _report_load(res)

        elif training_strategy == "lora":
            print(f"Loading LoRA {context} (adapters + optional classifier/head)...")

            # 1) Load classifier/head if present
            cls_sd = checkpoint.get("classifier_state_dict")
            if cls_sd is not None:
                if hasattr(self.model, "classifier"):
                    _report_load(self.model.classifier.load_state_dict(cls_sd, strict=False))
                elif hasattr(self.model, "head"):
                    _report_load(self.model.head.load_state_dict(cls_sd, strict=False))
                else:
                    _report_load(self.model.load_state_dict(cls_sd, strict=False))

            # 2) Prefer loading adapters from a saved PEFT directory
            adapter_path = checkpoint.get("adapter_path")
            if not adapter_path:
                # Try to find one in the checkpoint dir: best first, then epoch dirs
                ckpt_dir = os.path.dirname(getattr(self, "load_checkpoint_path", "") or self.checkpoint_dir)
                best_dir = os.path.join(ckpt_dir, "best_lora_adapter")
                if os.path.isdir(best_dir):
                    adapter_path = best_dir
                else:
                    # find most recent 'lora_adapter_epoch_*'
                    candidates = [os.path.join(ckpt_dir, d)
                                for d in os.listdir(ckpt_dir) if "lora_adapter" in d]
                    if candidates:
                        adapter_path = sorted(candidates)[-1]

            loaded_adapter = False
            try:
                # If the model exposes PEFT's load_adapter, use it.
                if adapter_path and os.path.isdir(adapter_path) and hasattr(self.model, "load_adapter"):
                    print(f"Loading LoRA adapter directory: {adapter_path}")
                    self.model.load_adapter(adapter_path)
                    loaded_adapter = True
            except Exception as e:
                print(f"[WARN] PEFT adapter load failed from dir: {e}")

            # 3) Fallback: direct param load (when no adapter directory is available)
            if not loaded_adapter:
                lora_sd = checkpoint.get("lora_state_dict")
                if lora_sd:
                    print("No adapter directory found; loading LoRA parameters directly...")
                    _report_load(self.model.load_state_dict(lora_sd, strict=False))
                else:
                    print("[WARN] No LoRA adapter dir or lora_state_dict found—skipping adapter load.")

        elif training_strategy == "full" and "model_state_dict" in checkpoint:
            print(f"Loading full {context} (entire model state)...")
            _report_load(self.model.load_state_dict(checkpoint["model_state_dict"], strict=False))

        else:
            # Last-resort compatibility path
            print(f"Loading {context} as a raw/partial state dict (strict=False)...")
            _report_load(self.model.load_state_dict(checkpoint, strict=False))

    def load_checkpoint(self, optimizer=None, scheduler=None, scaler=None):
        """
        Load the latest (or best) checkpoint for *training resume*.
        Restores model weights, optimizer, scheduler, scaler, and RNG states.
        Returns: start_epoch (int)
        """
        # Decide which .pt to read
        # only if you specify .pt for the load_checkpoint_path
        if self.load_checkpoint_path and os.path.isfile(self.load_checkpoint_path):
            target_path = self.load_checkpoint_path
        else:
            # prefer best first if asked
            target_path = self._find_latest_checkpoint()

        if not target_path:
            print("[load] No checkpoint file found; starting fresh.")
            return 0

        print(f"[load] Reading checkpoint: {target_path}")
        try:
            checkpoint = torch.load(target_path, map_location="cpu", weights_only=False)

            # 1) Model weights - load before prepare() to avoid accelerate wrapper issues
            self._load_model_state_from_checkpoint(checkpoint, context=os.path.basename(target_path))

            # 2) Optimizer/Scheduler/Scaler (if provided)
            if optimizer is not None and "optimizer" in checkpoint:
                optimizer.load_state_dict(checkpoint["optimizer"])
            if scheduler is not None and checkpoint.get("scheduler") is not None:
                scheduler.load_state_dict(checkpoint["scheduler"])
            if scaler is not None and checkpoint.get("scaler") is not None:
                scaler.load_state_dict(checkpoint["scaler"])

            # 3) Restore trainer metadata
            self.best_val_acc = checkpoint.get("best_val_acc", getattr(self, "best_val_acc", None))
            self.epochs_without_improvement = checkpoint.get(
                "epochs_without_improvement",
                getattr(self, "epochs_without_improvement", None)
            )

            # 4) RNG states (optional but good for determinism)
            rng = checkpoint.get("rng_state", None)
            if rng:
                try:
                    random.setstate(rng["python"])
                    np.random.set_state(rng["numpy"])
                    torch.set_rng_state(rng["torch"])
                    if torch.cuda.is_available() and rng["cuda"] is not None:
                        torch.cuda.set_rng_state_all(rng["cuda"])
                except Exception as e:
                    print(f"[WARN] Could not fully restore RNG state: {e}")

            start_epoch = int(checkpoint.get("epoch", 0))
            print(f"[load] Resume from epoch={start_epoch}, best_val_acc={self.best_val_acc}")
            return start_epoch

        except Exception as e:
            print(f"[WARN] Failed to load checkpoint ({target_path}): {e}. Starting fresh.")
            return 0


    def _create_checkpoint_data(self, optimizer, epoch, scheduler=None, scaler=None):
        
        training_strategy = self.global_config.get('TRAINING_STRATEGY', 'head_only')
        
        unwrapped = self.accelerator.unwrap_model(self.model)

        classifier_sd = None
        lora_sd = None
        adapter_path = None
        full_model_sd = None

        if training_strategy == "head_only":
            if hasattr(unwrapped, "classifier"):
                classifier_sd = unwrapped.classifier.state_dict()
            elif hasattr(unwrapped, "head"):
                classifier_sd = unwrapped.head.state_dict()
            else:
                classifier_sd = {k: v for k, v in model_sd.items()
                                if k.startswith("classifier.") or k.startswith("head.")}
            # with open("/home/keaneong/human-behavior/verl/multi_task_classification/classifier_state_dict_KEY.txt", "a") as f:
            #     f.write(f"\nLatest Classifier state dict {classifier_sd}")
            # raise Exception("Stop here")

        elif training_strategy == "lora":
            if hasattr(unwrapped, "save_pretrained"):
                adapter_path = os.path.join(self.checkpoint_dir, f"lora_adapter_epoch_{epoch+1}")
                unwrapped.save_pretrained(adapter_path)

            try:
                from peft import get_peft_model_state_dict
                lora_sd = get_peft_model_state_dict(unwrapped)
            except Exception:
                lora_sd = {k: v for k, v in model_sd.items() if "lora" in k.lower()}

            if hasattr(unwrapped, "classifier"):
                classifier_sd = unwrapped.classifier.state_dict()
            elif hasattr(unwrapped, "head"):
                classifier_sd = unwrapped.head.state_dict()

        elif training_strategy == "full":
            model_sd = self.accelerator.get_state_dict(self.model)  # safe across wrappers
            full_model_sd = model_sd  # capture entire model

        state = {
            "epoch": epoch + 1,
            "best_val_acc": getattr(self, "best_val_acc", None),
            "epochs_without_improvement": getattr(self, "epochs_without_improvement", None),
            "training_strategy": training_strategy,
            "config": {
                "lr": self.lr,
                "batch_size": self.batch_size,
                "num_classes": self.global_config.get("NUM_CLASSES", 0),
                "freeze_backbone": training_strategy,
            },
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict() if scheduler else None,
            "scaler": scaler.state_dict() if scaler else None,
            "rng_state": {
                "python": random.getstate(),
                "numpy": np.random.get_state(),
                "torch": torch.get_rng_state().cpu(),
                "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
            },
        }

        if training_strategy == "head_only":
            state["classifier_state_dict"] = classifier_sd
        elif training_strategy == "lora":
            state["lora_state_dict"] = lora_sd
            state["classifier_state_dict"] = classifier_sd
            state["adapter_path"] = adapter_path
        elif training_strategy == "full":
            state["model_state_dict"] = full_model_sd

        return state

    def save_checkpoint(self, optimizer, epoch, is_best=False, scheduler=None, scaler=None):
        # make sure the accelerator waits first
        self.accelerator.wait_for_everyone()

        full_sd = self.accelerator.get_state_dict(self.model)   # wrapper/sharding-safe
        with open("/home/keaneong/human-behavior/verl/multi_task_classification/debug_save_2.txt", "a") as f:
            f.write(f"\nLatest Full model state dict {full_sd}")
        raise Exception("Stop here")
        
        # if not self.accelerator.is_main_process:
        #     return

        os.makedirs(self.checkpoint_dir, exist_ok=True)

        # self.accelerator.save_state(self.checkpoint_dir)


        ckpt = self._create_checkpoint_data(optimizer, epoch, scheduler, scaler)
        
        ckpt_path = os.path.join(self.checkpoint_dir, f"checkpoint_epoch_{epoch+1}.pt")
        self.accelerator.save(ckpt, ckpt_path)

        # with open("/home/keaneong/human-behavior/verl/multi_task_classification/debug_save.txt", "a") as f:
        #         f.write(f"\nSAVED THE CHECKPOINT")
        #         raise Exception("Stop here")

       
        # if is_best: # NOTE: Forget about this for now as it will be time consuming
        #     best_path = os.path.join(self.checkpoint_dir, "best_model.pt")
        #     self.accelerator.save(ckpt, best_path)

        #     strategy = self.global_config.get("TRAINING_STRATEGY")
        #     if strategy == "lora":
        #         try:
        #             unwrapped = self.accelerator.unwrap_model(self.model)
        #             best_adapter = os.path.join(self.checkpoint_dir, "best_lora_adapter")
        #             unwrapped.save_pretrained(best_adapter)
        #         except Exception as e:
        #             print(f"Warning: Could not save best LoRA adapter: {e}")

        #     elif strategy == "full":
        #         try:
        #             unwrapped = self.accelerator.unwrap_model(self.model)
        #             best_full = os.path.join(self.checkpoint_dir, "best_full_model")
        #             self.accelerator.save_model(unwrapped, best_full)
        #         except Exception as e:
        #             print(f"Warning: Could not save best full-model dir: {e}")

        print(f"Checkpoint saved: {ckpt_path}")

    def train(self):
        train_dataloader = self.get_dataloader(self.data_files, self.batch_size, num_workers=self.num_workers, shuffle=True)
        val_dataloader = self.get_dataloader(self.val_data_files, self.val_batch_size, num_workers=self.num_workers, shuffle=False)
        
        optimizer = Adam(self.model.parameters(), lr=self.lr)
        criterion = CrossEntropyLoss()

        # Load checkpoint if available (before prepare to avoid accelerate wrapper issues)
        start_epoch = self.load_checkpoint(optimizer)

        # Prepare everything with Accelerate
        self.model, optimizer, train_dataloader, val_dataloader = self.accelerator.prepare(
            self.model, optimizer, train_dataloader, val_dataloader
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
                    if labels.dim() == 2 and labels.size(1) == num_classes:
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
                        use_wandb=use_wandb
                    )

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
                self.save_checkpoint(optimizer, epoch, is_best)
            elif is_best:
                # ensure best is saved even if not on save interval
                self.save_checkpoint(optimizer, epoch, is_best)
            
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
