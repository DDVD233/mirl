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
import torch.nn.functional as F   # (already imported above)

# Add parent directory to path for imports
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

# Local imports
from mt_dataset.omni_classifier_dataset import OmniClassifierDataset
from verl.utils.dataset.rl_dataset import collate_fn
from utils.wandb_utils import init_wandb, log_metrics, finish
from utils.logger import log_batch_training_metrics, log_validation_results, log_epoch_training_metrics
# from evaluate.multi_task_evaluation import evaluate_predictions
from evaluate.detailed_multi_task_evaluation import evaluate_predictions

# Accelerate imports
from accelerate import Accelerator
from accelerate.utils import set_seed
from accelerate.logging import get_logger
from models.adapter_utils import maybe_build_adapters, apply_adapters, build_video_feats_batch, build_audio_feats_batch

logger = get_logger(__name__)

class RLAMultiHeadOmniClassifierAccelerateTrainer:
    def __init__(self, data_files, val_data_files, test_data_files, tokenizer, processor, config, 
                 batch_size, val_batch_size, test_batch_size, lr, epochs, save_checkpoint_dir, load_checkpoint_path, model, 
                 gradient_accumulation_steps, num_workers=0, use_lora=False, global_config=None):
        self.data_files = data_files
        self.val_data_files = val_data_files
        self.test_data_files = test_data_files
        self.tokenizer = tokenizer
        self.processor = processor
        self.config = config # basically the config for the dataset
        self.batch_size = batch_size
        self.val_batch_size = val_batch_size
        self.test_batch_size = test_batch_size
        self.gradient_accumulation_steps = gradient_accumulation_steps
        self.num_workers = num_workers
        self.lr = lr
        self.epochs = epochs
        self.model = model
        self.label_key = config.get("label_key", "answer")

        # Store global configuration for access to constants
        self.global_config = global_config or {}
        
        # Use the label map from global config
        self.full_label_scheme = self.global_config.get('FULL_LABEL_SCHEME', None)
        self.label_map = self.global_config.get('LABEL_MAP', {})
        self.label_map_path = self.global_config.get('LABEL_MAP_PATH', None)
        
        # Initialising of the RLA-related parameters
        # Whether to enable each modality adapter
        self.use_rla_video = self.global_config.get("USE_RLA_VIDEO", False)
        self.use_rla_audio = self.global_config.get("USE_RLA_AUDIO", False)


        # Training stage for adapters vs base:
        #   "base_only"      -> train base only (no adapters built/applied)
        #   "residual_only"  -> freeze base, train adapters only
        #   "joint"          -> train base and adapters together
        self.rla_stage     = self.global_config.get("RLA_STAGE", "base_only")
        # self.resume_diff_cfg = bool(self.global_config.get("RLA_RESUME_DIFFERENT_TRAINING_CONFIG", False))
        self.rla_resume_diff_training_stage = True

        # Adapter architecture / regularization
        self.rla_hidden    = self.global_config.get("RLA_HIDDEN", 128)
        self.rla_pv        = self.global_config.get("RLA_P_MODDROP_VIDEO", 0.30)
        self.rla_pa        = self.global_config.get("RLA_P_MODDROP_AUDIO", 0.30)

        # Feature dimensions (optional; inferred from dataset[0] if None)
        self.d_video_feat  = self.global_config.get("D_VIDEO_FEAT", None)
        self.d_audio_feat  = self.global_config.get("D_AUDIO_FEAT", None)

        # Adapter handles (populated later by maybe_build_adapters)
        self.video_adapter = None
        self.audio_adapter = None

        # after loading the label map, we need to build the domain routing tables
        self.build_domain_routing()

        # Scheduler configuration
        self.use_scheduler = self.global_config.get("USE_SCHEDULER", True)
        self.scheduler_type = self.global_config.get("SCHEDULER_TYPE", "cosine")
        self.warmup_steps = self.global_config.get("WARMUP_STEPS", None)
        
        # Checkpoint IO setup
        self.checkpoint_dir = save_checkpoint_dir
        self.load_checkpoint_path = load_checkpoint_path

        self.validation_result_dir = self.global_config.get('VALIDATION_RESULT_DIR', None)

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
            "test_batch_size": self.test_batch_size,
            "gradient_accumulation_steps": self.gradient_accumulation_steps,
            "effective_batch_size": self.batch_size * self.gradient_accumulation_steps * self.accelerator.num_processes,
            "learning_rate": self.lr,
            "epochs": self.epochs,
            "num_classes": self.global_config.get('NUM_CLASSES', 0),
            "validate_every_n_epochs": self.global_config.get('VALIDATE_EVERY_N_EPOCHS', None),
            "validate_every_n_steps": self.global_config.get('VALIDATE_EVERY_N_STEPS', None),
            "save_every_n_epochs": self.global_config.get('SAVE_EVERY_N_EPOCHS', None),
            "save_every_n_steps": self.global_config.get('SAVE_EVERY_N_STEPS', None),
            "validation_result_dir": self.validation_result_dir,
            "save_checkpoint_dir": self.checkpoint_dir,
            "load_checkpoint_path": self.load_checkpoint_path,
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

    def set_requires_grad(module, flag: bool):
        # Helper function to free/unfreeze the different parts of the model
        if module is None:
            return
        for p in module.parameters():
            p.requires_grad = flag

    def build_domain_routing(self):
        # === Build domain routing tables from label_map ===
        meta = self.full_label_scheme.get("meta", {})
        global_classes = meta.get("global_classes", {})
        domain_names = list(global_classes.keys())  # ['sentiment_intensity','emotion','mental_health']
        
        # obtain an id for the domain names e.g. {'sentiment_intensity': 0, 'emotion': 1, 'mental_health': 2}
        self.domain_name_to_id = {d:i for i,d in enumerate(domain_names)}
        
        # obtain the global indices for the different domains
        # e.g. [[0, 1, 2, 3, 4, 5, 6], [7, 8, 9, 10, 11, 12, 13, 14], [15, 16, 17, 18, 19, 20]]
        # where the first nested list is the global indices for the sentiment_intensity domain
        # the second nested list is the global indices for the emotion domain
        # the third nested list is the global indices for the mental_health domain
        self.domain_id_to_global_indices = [[x["index"] for x in global_classes[d]] for d in domain_names]

        # get the dataset that maps to the domain names;
        # e.g. {'mosei_senti': 'sentiment_intensity', 'mosei_emotion': 'emotion', 'ptsd_in_the_wild': 'mental_health'}
        dataset_to_domain = meta.get("dataset_domain", {})  # e.g. 'mosei_senti':'sentiment_intensity'

        # obtain the dataset to domain id mapping
        # e.g. {'mosei_senti': 0, 'mosei_emotion': 1, 'ptsd_in_the_wild': 2}
        self.dataset_to_domain_id = {ds: self.domain_name_to_id[dn] for ds, dn in dataset_to_domain.items()}

    def _datasets_to_domain_ids(self, dataset_names, device):
        # dataset_names is a list/sequence length B (strings)
        # Unknown datasets raise to fail-fast
        ids = []
        for ds in dataset_names:
            if isinstance(ds, bytes):  # sometimes collate/gather returns bytes
                ds = ds.decode("utf-8")
            if ds not in self.dataset_to_domain_id:
                raise KeyError(f"Dataset '{ds}' not in label_map.meta.dataset_domain")
            ids.append(self.dataset_to_domain_id[ds])

        # store the tensors for the domain ids
        return torch.tensor(ids, dtype=torch.long, device=device)

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

        accelerator.wait_for_everyone()

        # only save this in the main process
        if accelerator.is_main_process:
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
        rla_resume_diff_cfg=None  # whether to allow loading RLA checkpoints with different training config
    ):
        """
        Rebuild & accelerator.prepare() your model/optimizer/dataloaders first.
        Then call this loader to restore state and compute (start_epoch, start_batch_offset).
        Returns: (start_epoch, start_batch_offset, global_step, meta, ckpt_dir)
        """
        if explicit_dir:
            ckpt_dir = explicit_dir
        else:
            print(f"[load] finding latest checkpoint from {base_ckpt_dir}")
            ckpt_dir = self._latest_checkpoint_dir(base_ckpt_dir)
            print(f"[load] latest checkpoint found: {ckpt_dir}")
        
        print(f"[load] loading checkpoint from {ckpt_dir}")

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
        
        if rla_resume_diff_cfg:
            start_epoch = 0
            start_batch_offset = 0

        else:
            start_epoch = floor((global_step - 1) / len_dl)
            start_batch_offset = (global_step - 1) % len_dl

        accelerator.print(f"[load] resumed {ckpt_dir} → epoch={start_epoch}, step={global_step}, offset={start_batch_offset}")
    
        return start_epoch, start_batch_offset, global_step, meta, ckpt_dir
    
    def prepare_params_for_training(self, base_lr: float = None, rla_lr: float = None):
        """
        Freeze/unfreeze according to self.rla_stage ∈ {"base_only","residual_only","joint"}
        and build optimizer param groups. Simple, general, LoRA-compatible.
        """
        base_lr = self.lr if base_lr is None else base_lr
        rla_lr  = self.lr if rla_lr  is None else rla_lr

        def _set_requires_grad(module, flag: bool):
            if module is None:
                return
            for p in module.parameters():
                p.requires_grad = flag

        unfrozen_param_groups = []
        # NOTE: in the following, assume that the model is already has its required gradients
        # set by a preceding code
        if self.rla_stage == "base_only":
            print("Freezing adapters, training base model only")
            _set_requires_grad(self.video_adapter, False)
            _set_requires_grad(self.audio_adapter, False)
            base_params = self.model.parameters()
            if base_params:
                unfrozen_param_groups.append({"params": base_params, "lr": base_lr})

        elif self.rla_stage == "residual_only":
            print("Freezing base model, training adapters only")
            _set_requires_grad(self.model, False)
            # Train adapters only
            _set_requires_grad(self.video_adapter, True)
            _set_requires_grad(self.audio_adapter, True)
            rla_params = []
            if self.video_adapter is not None:
                rla_params += [p for p in self.video_adapter.parameters() if p.requires_grad]
            if self.audio_adapter is not None:
                rla_params += [p for p in self.audio_adapter.parameters() if p.requires_grad]
            if rla_params:
                unfrozen_param_groups.append({"params": rla_params, "lr": rla_lr})

        elif self.rla_stage == "joint":
            print("Training both base model and adapters")
            # Train base + adapters
            _set_requires_grad(self.video_adapter, True)
            _set_requires_grad(self.audio_adapter, True)

            base_params = self.model.parameters()
            if base_params:
                unfrozen_param_groups.append({"params": base_params, "lr": base_lr})

            rla_params = []
            if self.video_adapter is not None:
                rla_params += [p for p in self.video_adapter.parameters() if p.requires_grad]
            if self.audio_adapter is not None:
                rla_params += [p for p in self.audio_adapter.parameters() if p.requires_grad]
            if rla_params:
                unfrozen_param_groups.append({"params": rla_params, "lr": rla_lr})

        else:
            raise ValueError(f"Unknown RLA stage: {self.rla_stage}")

        return unfrozen_param_groups
    
    def _accelerate_prepare(
        self,
        train_dataloader,
        val_dataloader,
        optimizer=None,
        scheduler=None,
        prepare_adapters=False,
    ):
        """
        Prepare with Accelerate in two phases to avoid mixing modules with optimizer:
        (1) modules-only: model [+ adapters if requested] + dataloaders
        (2) opt-only: optimizer [+ scheduler]
        """
        # ----- Phase 1: modules-only -----
        modules = [self.model]

        if self.rla_stage in {"residual_only", "joint"} and prepare_adapters:
            if getattr(self, "video_adapter", None) is not None:
                modules.append(self.video_adapter)
            if getattr(self, "audio_adapter", None) is not None:
                modules.append(self.audio_adapter)

        modules += [train_dataloader, val_dataloader]

        prepared = self.accelerator.prepare(*modules)

        # Unpack Phase 1
        idx = 0
        self.model = prepared[idx]; idx += 1

        if self.rla_stage in {"residual_only", "joint"} and prepare_adapters:
            if getattr(self, "video_adapter", None) is not None:
                self.video_adapter = prepared[idx]; idx += 1
            if getattr(self, "audio_adapter", None) is not None:
                self.audio_adapter = prepared[idx]; idx += 1

        train_dataloader = prepared[idx]; idx += 1
        val_dataloader   = prepared[idx]; idx += 1

        # ----- Phase 2: optimizer/scheduler-only -----
        if optimizer is not None:
            if scheduler is not None:
                optimizer, scheduler = self.accelerator.prepare(optimizer, scheduler)
                self.accelerator.register_for_checkpointing(scheduler)
            else:
                (optimizer,) = self.accelerator.prepare(optimizer)

        return train_dataloader, val_dataloader, optimizer, scheduler

    def validate(self, val_dataloader, split_name="validation", current_step=None):
        """Validate the model on the given dataloader."""
        self.model.eval()

        if self.video_adapter is not None:
            self.video_adapter.eval()
        if self.audio_adapter is not None:
            self.audio_adapter.eval()
        
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


                # retrieve the batch and domain_ids for all the batch
                if 'dataset' not in batch:
                    raise KeyError("Batch missing 'dataset' needed for domain routing.")
                # batch['dataset'] is typically a list/tuple length B
                domain_ids = self._datasets_to_domain_ids(batch['dataset'], device=input_ids.device)


                # Each should be of shape (B, D_feat)
                if ("audio_feats" in batch) and (self.rla_stage in {"residual_only", "joint"}) and self.use_rla_audio:
                    audio_feats = batch["audio_feats"]

                    # folllowing the video_feats_batch, the pooled_audio_feats should be [B, X*K*C]
                    pooled_audio_feats = build_audio_feats_batch(
                        audio_feats,
                        device=input_ids.device,
                        temporal_mode=self.global_config.get("RLA_AUDIO_TEMPORAL", "none"),
                        norm=self.global_config.get("RLA_AUDIO_NORM", "l2"),
                        target_dim=self.d_audio_feat, 
                    )

                else:
                    pooled_audio_feats = None
                
                if ("video_feats" in batch) and (self.rla_stage in {"residual_only", "joint"}) and self.use_rla_video:
                    # assume a torch loaded batch of video features
                    video_feats = batch["video_feats"]
                    
                    # processing of the video feats to be of shape (B, D_feat)
                    # video_feats should be a list of dictionaries with 
                    # pose, face, left_hand, right_hand, audio, etc. as keys
                    # each value is a tensor of shape (num_frames, num_landmarks(i.e. feature values), (x,y,c)))
                    # and then we are pooling all of this into a single vector of shape (B, D_feat)
                    # list of dictionaries, meaning each element in the list is a dictionary corresponding to one sample
                    # within the batch ; batch encapsulates all the samples in the list
                    # i.e. all of the num frames; landmarks; and (x,y,c) values are pooled into a single vector

                    # the pooled_video_feats should look like [B, X*K*C], where K, C is basically the num_landmarks 
                    # and (x,y,c) values, but averaged across the temporal dimension (num_frames)
                    pooled_video_feats = build_video_feats_batch(
                        video_feats,   # list of dicts
                        device=input_ids.device,
                        temporal_mode=self.global_config.get("RLA_VIDEO_TEMPORAL", "meanstd"),
                        use_conf=self.global_config.get("RLA_VIDEO_USE_CONF", True),
                        target_dim=self.d_video_feat
                    )
            
                else:
                    pooled_video_feats = None

                # Handle labels shape
                if labels.dim() != 1:
                    if labels.dim() == 2 and labels.size(1) == num_classes:
                        labels = labels.argmax(dim=1)
                    else:
                        raise ValueError(f"Unexpected labels shape {labels.shape}")

                logits = self.model(input_ids, attention_mask=attention_mask, domain_ids=domain_ids)

                if (self.rla_stage in {"residual_only", "joint"}) and (self.use_rla_video or self.use_rla_audio):
                        # apply the adapters to the logits
                        # to get new logits
                        logits = apply_adapters(
                                logits,
                                domain_ids,
                                video_adapter=self.video_adapter,
                                audio_adapter=self.audio_adapter,
                                video_feats=pooled_video_feats,
                                audio_feats=pooled_audio_feats,
                                train_mode=False,
                        )

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
                split_name=split_name,
                save_path=self.validation_result_dir,
                global_steps=current_step,
                label_map_path=self.label_map_path
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

        # Building adapters if required;
        # NOTE WILL BE INITIALISED REGARDLESS OF WHETHER WE USE USE.RLA_VIDEO / USE.RLA_AUDIO
        self.video_adapter, self.audio_adapter = maybe_build_adapters(
            domain_id_to_global_indices=self.domain_id_to_global_indices,
            use_rla_video=self.use_rla_video,
            use_rla_audio=self.use_rla_audio,
            rla_hidden=self.rla_hidden,
            p_moddrop_video=self.rla_pv,
            p_moddrop_audio=self.rla_pa,
            d_video_feat=self.d_video_feat,
            d_audio_feat=self.d_audio_feat,
        )



        # optimizer, with an option to train the entire model or the adapters only etc.
        # Freeze/unfreeze + param groups
        # --- LR overrides (RLA > base) ---------------------------------------------
        base_lr = self.global_config.get("BASE_LR", self.lr * 0.25)
        rla_lr  = self.global_config.get("RLA_LR",  self.lr * 5.0)
        first_param_groups = self.prepare_params_for_training(base_lr=base_lr, rla_lr=rla_lr)
        # ---------------------------------------------------------------------------

        # first_param_groups = self.prepare_params_for_training()

        criterion = CrossEntropyLoss()

        # ---- Compute update-steps-aware schedule sizes ----
        # updates_per_epoch = ceil(len(train_dataloader) / max(1, self.gradient_accumulation_steps))
        total_updates = self.epochs * len(train_dataloader)
        
        if self.rla_resume_diff_training_stage:

            # ================= DIFFERENT TRAINING CONFIG RESUME =================
            # 1) Prepare ONLY model/adapters/dataloaders (optimizer=None toggles behavior)
            # 2) It will also not prepare the adapter if they self.rla_stage=="base_only"
            # 3) It will also not prepare the adapter if self.rla_resume_diff_cfg is True (which is in this case)
            # We do this because we do not want to load the state for the adapters via load_checkpoint_unified
            # since they are newly built and have no state to load.
            train_dataloader, val_dataloader, _, _ = self._accelerate_prepare(
                train_dataloader=train_dataloader,
                val_dataloader=val_dataloader,
                optimizer=None,
                scheduler=None,
                prepare_adapters=False
            )

            # 2) Load state now: restores model shards/RNG ONLY (no optimizer to rekey)
            # Since we didn't prepare the adapter, the adapter states will not be loaded.
            start_epoch, start_batch_offset, global_step, _, _ = self.load_checkpoint_unified(
                accelerator=self.accelerator,
                model=self.model,
                base_ckpt_dir=self.checkpoint_dir,
                explicit_dir=self.load_checkpoint_path or None,
                expect_training_strategy=self.global_config.get("TRAINING_STRATEGY"),
                rla_resume_diff_cfg=self.rla_resume_diff_training_stage,
            )

            # 3) Prepare adapters in a MODULES-ONLY call (so save_state will track them)
            # remember that they were previously unprepared
            if self.rla_stage in {"residual_only", "joint"}:
                adapters_to_prepare = []
                if self.video_adapter is not None:
                    adapters_to_prepare.append(self.video_adapter)
                if self.audio_adapter is not None:
                    adapters_to_prepare.append(self.audio_adapter)
                if len(adapters_to_prepare) > 0:
                    prepared = self.accelerator.prepare(*adapters_to_prepare)
                    i = 0
                    if self.video_adapter is not None:
                        self.video_adapter = prepared[i]; i += 1
                    if self.audio_adapter is not None:
                        self.audio_adapter = prepared[i] if i < len(prepared) else self.audio_adapter

            # prepare the new param groups after loading the wrapped models; particularly
            # for the model group.
            updated_param_groups = self.prepare_params_for_training(base_lr=base_lr, rla_lr=rla_lr)
            # the optimizer can just points to the updated param groups
            # because its a new optimizer instance that takes in the updated param groups
            # where the updated param groups now point to the wrapped model
            optimizer = Adam(updated_param_groups)
            
            if self.use_scheduler:
                scheduler = get_scheduler(
                    self.scheduler_type,
                    optimizer,
                    num_warmup_steps=self.warmup_steps,
                    num_training_steps=total_updates
                )
                # Prepare optimizer (and scheduler) seperately from the modules
                optimizer, scheduler = self.accelerator.prepare(optimizer, scheduler)
                self.accelerator.register_for_checkpointing(scheduler)
                print(f"[INFO] Using {self.scheduler_type} scheduler with {self.warmup_steps} warmup steps)")
            else:
                scheduler = None
                optimizer = self.accelerator.prepare(optimizer)
                print("[INFO] Scheduler disabled - using constant learning rate")

        else:
            optimizer = Adam(first_param_groups)
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
            # Note that under the hood, the models and adapters are also prepared here.
            train_dataloader, val_dataloader, optimizer, scheduler = self._accelerate_prepare(
                train_dataloader=train_dataloader,
                val_dataloader=val_dataloader,
                optimizer=optimizer,
                scheduler=scheduler
                )

            start_epoch, start_batch_offset, _, _, _ = self.load_checkpoint_unified(
                accelerator=self.accelerator,
                model=self.model,
                base_ckpt_dir=self.checkpoint_dir,
                explicit_dir=self.load_checkpoint_path or None,
                expect_training_strategy=self.global_config.get("TRAINING_STRATEGY"),
                rla_resume_diff_cfg=self.rla_resume_diff_training_stage
            )

        # Get configuration values
        validate_every_n_epochs = self.global_config.get('VALIDATE_EVERY_N_EPOCHS', None)
        validate_every_n_steps = self.global_config.get('VALIDATE_EVERY_N_STEPS', None)
        save_every_n_epochs = self.global_config.get('SAVE_EVERY_N_EPOCHS', None)
        save_every_n_steps = self.global_config.get('SAVE_EVERY_N_STEPS', None)
        early_stopping_patience = self.global_config.get('EARLY_STOPPING_PATIENCE', 0)
        use_wandb = self.global_config.get('USE_WANDB', False)
        num_classes = self.global_config.get('NUM_CLASSES', 0)

        for epoch in tqdm(range(start_epoch, self.epochs), desc="Epochs", position=0, disable=not self.accelerator.is_main_process):
            # Training phase
            self.model.train()
            if self.video_adapter is not None:
                self.video_adapter.train()
            if self.audio_adapter is not None:
                self.audio_adapter.train()

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

                # remember to set it back to train() mode after validation
                if self.video_adapter is not None:
                    self.video_adapter.train()
                if self.audio_adapter is not None:
                    self.audio_adapter.train()


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

                if 'dataset' not in batch:
                    raise KeyError("Batch missing 'dataset' needed for domain routing.")
                # batch['dataset'] is typically a list/tuple length B
                domain_ids = self._datasets_to_domain_ids(batch['dataset'], device=input_ids.device)

                # Each should be of shape (B, D_feat)
                if ("audio_feats" in batch) and (self.rla_stage in {"residual_only", "joint"}) and self.use_rla_audio:
                    audio_feats = batch["audio_feats"]

                    # folllowing the video_feats_batch, the pooled_audio_feats should be [B, X*K*C]
                    pooled_audio_feats = build_audio_feats_batch(
                        audio_feats,
                        device=input_ids.device,
                        temporal_mode=self.global_config.get("RLA_AUDIO_TEMPORAL", "none"),
                        norm=self.global_config.get("RLA_AUDIO_NORM", "l2"),
                        target_dim=self.d_audio_feat, 
                    )

                else:
                    pooled_audio_feats = None
                
                if ("video_feats" in batch) and (self.rla_stage in {"residual_only", "joint"}) and self.use_rla_video:
                    # assume a torch loaded batch of video features
                    video_feats = batch["video_feats"]
                    
                    # processing of the video feats to be of shape (B, D_feat)
                    # video_feats should be a list of dictionaries with 
                    # pose, face, left_hand, right_hand, audio, etc. as keys
                    # each value is a tensor of shape (num_frames, num_landmarks(i.e. feature values), (x,y,c)))
                    # and then we are pooling all of this into a single vector of shape (B, D_feat)
                    # list of dictionaries, meaning each element in the list is a dictionary corresponding to one sample
                    # within the batch ; batch encapsulates all the samples in the list
                    # i.e. all of the num frames; landmarks; and (x,y,c) values are pooled into a single vector

                    # the pooled_video_feats should look like [B, X*K*C], where K, C is basically the num_landmarks 
                    # and (x,y,c) values, but averaged across the temporal dimension (num_frames)
                    pooled_video_feats = build_video_feats_batch(
                        video_feats,   # list of dicts
                        device=input_ids.device,
                        temporal_mode=self.global_config.get("RLA_VIDEO_TEMPORAL", "meanstd"),
                        use_conf=self.global_config.get("RLA_VIDEO_USE_CONF", True),
                        target_dim=self.d_video_feat
                    )
            
                else:
                    pooled_video_feats = None

                # labels sanity
                if labels.dim() != 1:
                    # If your dataset sometimes emits multi-task/one-hot, squeeze or argmax here
                    if labels.dim() == 2 and labels.size(1) == num_classes:
                        labels = labels.argmax(dim=1)
                    else:
                        raise ValueError(f"Unexpected labels shape {labels.shape} (expected [B] or [B, C])")
                
                with self.accelerator.accumulate(self.model):
                    logits = self.model(input_ids, attention_mask=attention_mask, domain_ids=domain_ids)

                    if not torch.isfinite(logits).all():
                        raise FloatingPointError("Non-finite logits encountered")
                    
                    if (self.rla_stage in {"residual_only", "joint"}) and (self.use_rla_video or self.use_rla_audio):
                        # apply the adapters to the logits
                        logits = apply_adapters(
                                logits,
                                domain_ids,
                                video_adapter=self.video_adapter,
                                audio_adapter=self.audio_adapter,
                                video_feats=pooled_video_feats,
                                audio_feats=pooled_audio_feats,
                                train_mode=True,
                        )
                    
                    # NOTE: WEIGHING OF HARD EXAMPLES 
                    # TODO: ARE THE LOGITS/ CONFIDENCE IMPACTED BY OUR NEGATIVE INFINITY MASKING?
                    gamma = self.global_config.get("HARD_GAMMA", 0.0)  # set >0 to enable
                    if gamma > 0:
                        with torch.no_grad():
                            probs = torch.softmax(logits, dim=-1)
                            # p_true: prob assigned to ground-truth class per sample
                            p_true = probs.gather(1, labels.unsqueeze(1)).squeeze(1)  # [B]
                            # hardness in [0,1]; higher = harder
                            hardness = (1.0 - p_true).clamp_(1e-6, 1.0)
                            weights = hardness.pow(gamma)                              # [B]
                        ce_per_sample = F.cross_entropy(logits, labels, reduction='none')  # [B]
                        # Normalize by sum of weights to keep loss scale stable
                        loss = (weights * ce_per_sample).sum() / (weights.sum().clamp_min(1.0))
                    else:
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
                        val_results = self.validate(val_dataloader, "validation", current_step=current_step)
                        
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

                    # Step-based saving (if configured)
                    if save_every_n_steps is not None and current_step % save_every_n_steps == 0:
                        print(f"\n[STEP {current_step}] Saving checkpoint...")
                        self.save_checkpoint_unified(
                            accelerator=self.accelerator,
                            model=self.model,
                            epoch=epoch,
                            batch_idx=batch_idx,
                            len_train_dataloader=len(train_dataloader),
                            training_strategy=self.global_config.get("TRAINING_STRATEGY"),
                            base_ckpt_dir=self.checkpoint_dir,
                        )

            # Calculate training metrics
            avg_train_loss = total_loss / max(1, total)
            train_acc = correct / max(1, total)
            
            if self.accelerator.is_main_process:
                print(f"Epoch {epoch+1}/{self.epochs} - Train Loss: {avg_train_loss:.4f} - Train Acc: {train_acc:.4f}")

            # Epoch-based validation phase (only if step-based validation is not configured)
            if validate_every_n_epochs is not None and (epoch + 1) % validate_every_n_epochs == 0:
                val_results = self.validate(val_dataloader, "validation", current_step=current_step)
                
                if self.accelerator.is_main_process and val_results is not None:
                    
                    # Check if this is the best model (using micro F1 as primary metric)
                    val_f1 = val_results['f1']
                    if val_f1 > self.best_val_acc:
                        self.best_val_acc = val_f1
                        self.epochs_without_improvement = 0
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
                    
                    log_validation_results(
                        val_results=val_results,
                        current_step=current_step,
                        split_name="validation",
                        accelerator=self.accelerator,
                        use_wandb=use_wandb
                    )
            else:

                # Log only training metrics
                # NOTE: it should already have the main process check inside.
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
        print("\n" + "="*50)
        print("STARTING TESTING PHASE")
        print("="*50)

        # 1) Build loaders (train loader optional; used here only to mirror 'total_updates')
        train_dataloader = self.get_dataloader(self.data_files, self.batch_size, num_workers=self.num_workers, shuffle=True)
        test_dataloader  = self.get_dataloader(self.test_data_files, self.test_batch_size, num_workers=self.num_workers, shuffle=False)

        # 2) Build adapters exactly as in training
        self.video_adapter, self.audio_adapter = maybe_build_adapters(
            domain_id_to_global_indices=self.domain_id_to_global_indices,
            use_rla_video=self.use_rla_video,
            use_rla_audio=self.use_rla_audio,
            rla_hidden=self.rla_hidden,
            p_moddrop_video=self.rla_pv,
            p_moddrop_audio=self.rla_pa,
            d_video_feat=self.d_video_feat,
            d_audio_feat=self.d_audio_feat,
        )

        # 3) Mirror training’s (un)freeze and param-group wiring
        param_groups = self.prepare_params_for_training()

        # 4) Construct optimizer/scheduler (same classes as training)
        optimizer = Adam(param_groups, lr=self.lr)
        # We won’t step it during test; it just needs to exist for load_state().
        if self.use_scheduler:
            # The exact num_training_steps won’t matter: load_state will overwrite counters.
            total_updates = self.epochs * len(train_dataloader)
            scheduler = get_scheduler(
                self.scheduler_type,
                optimizer,
                num_warmup_steps=self.warmup_steps,
                num_training_steps=total_updates
            )
        else:
            scheduler = None

        # 5) Prepare model(+adapters)+dataloaders+optimizer(+scheduler) identically to training
        train_dataloader, test_dataloader, optimizer, scheduler = self._accelerate_prepare(
            train_dataloader=train_dataloader,
            val_dataloader=test_dataloader,
            optimizer=optimizer,
            scheduler=scheduler,
        )

        # 6) Load everything (model + optimizer + scheduler + RNG) from checkpoint
        #    same-config resume: allow optimizer state to be restored & rekeyed
        _ = self.load_checkpoint_unified(
            accelerator=self.accelerator,
            model=self.model,
            base_ckpt_dir=self.checkpoint_dir,
            explicit_dir=self.load_checkpoint_path or None,
            expect_training_strategy=self.global_config.get("TRAINING_STRATEGY"),
            rla_resume_diff_cfg=False,
        )

        test_results = self.validate(test_dataloader, "test", current_step=1)
        
        if self.accelerator.is_main_process and test_results is not None:
            print(f"\nOverall TEST RESULTS:")
            print(f"Test Loss: {test_results['loss']:.4f}")
            print(f"Test Micro Accuracy: {test_results['accuracy']:.4f}")
            print(f"Test Micro Precision: {test_results['precision']:.4f}")
            print(f"Test Micro Recall: {test_results['recall']:.4f}")
            print(f"Test Micro F1: {test_results['f1']:.4f}")
            
            # Print detailed aggregate metrics
            aggregate_metrics = test_results['aggregate_metrics']
            print(f"\nDetailed Test Metrics:")
            print(f"  Micro Accuracy: {aggregate_metrics.get('micro_accuracy', 0.0):.4f}")
            print(f"  Micro F1: {aggregate_metrics.get('micro_f1', 0.0):.4f}")
            print(f"  Macro F1: {aggregate_metrics.get('macro_f1', 0.0):.4f}")
            print(f"  Weighted F1: {aggregate_metrics.get('weighted_f1', 0.0):.4f}")
            
            # Log test results to wandb
            use_wandb = self.global_config.get('USE_WANDB', False)
    
            log_validation_results(
                    val_results=test_results,
                    current_step=1,
                    split_name="test",
                    accelerator=self.accelerator,
                    use_wandb=use_wandb
                )
        
            return test_results

        return None
