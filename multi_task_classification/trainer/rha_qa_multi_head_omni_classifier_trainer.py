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
from mt_dataset.addqa_omni_classifier_dataset import AddQAOmniClassifierDataset, log_failed_path
from verl.utils.dataset.rl_dataset import collate_fn
from utils.wandb_utils import init_wandb, log_metrics, finish
from utils.logger import log_batch_training_metrics, log_validation_results, log_epoch_training_metrics
# from evaluate.multi_task_evaluation import evaluate_predictions
from evaluate.detailed_multi_task_evaluation import evaluate_predictions

# Accelerate imports
from accelerate import Accelerator
from accelerate.utils import set_seed
from accelerate.logging import get_logger
from models.adapter_utils import build_video_feats_batch, build_audio_feats_batch
from models.rha_adapter_utils import (  
    maybe_build_hidden_adapters,
    apply_hidden_adapters,
)


logger = get_logger(__name__)

class QARHAMultiHeadOmniClassifierAccelerateTrainer:
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

        # QA DATASETS
        self.qa_datasets = set(self.global_config.get('QA_DATASETS', ['intentqa', 'mimeqa', 'siq2']))  # e.g. {"mmlu_qa","ptsd_qa","finance_qa"}
        self.qa_loss_weight = float(self.global_config.get('QA_LOSS_WEIGHT', 1.0))

        # Deterministic generation for evaluation
        self.max_val_qa_tokens = 30
        print(f"WARNING: Using max_val_qa_tokens={self.max_val_qa_tokens} for validation/test generation.")
        
        
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
        # inside __init__
        self.rla_stage  = self.global_config.get("RLA_STAGE", "base_only")
        self.rla_resume_diff_training_stage = bool(
            self.global_config.get("RLA_RESUME_DIFF_TRAINING_STAGE", False)  # <<< was hardcoded False
        )

        self.rla_hidden = self.global_config.get("RLA_HIDDEN", 128)
        self.rla_hidden_video = int(self.global_config.get("RLA_HIDDEN_VIDEO", self.rla_hidden))  # <<< NEW
        self.rla_hidden_audio = int(self.global_config.get("RLA_HIDDEN_AUDIO", self.rla_hidden))  # <<< NEW

        self.rla_pv = self.global_config.get("RLA_P_MODDROP_VIDEO", 0.30)
        self.rla_pa = self.global_config.get("RLA_P_MODDROP_AUDIO", 0.30)

        # Feature pipeline knobs
        self.video_temporal = self.global_config.get("RLA_VIDEO_TEMPORAL", "meanstd")
        self.video_norm     = self.global_config.get("RLA_VIDEO_NORM", None)          # <<< NEW

        self.audio_temporal = self.global_config.get("RLA_AUDIO_TEMPORAL", "none")
        self.audio_norm     = self.global_config.get("RLA_AUDIO_NORM", "l2")

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
            "train_batch_size": self.batch_size,
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
            "warmup_steps": self.warmup_steps if self.use_scheduler else None,

            # ==========================
            # RLA: high-level toggles
            # ==========================
            "rla_use_video": bool(self.global_config.get("USE_RLA_VIDEO", False)),
            "rla_use_audio": bool(self.global_config.get("USE_RLA_AUDIO", False)),
            "rla_stage": self.global_config.get("RLA_STAGE", "base_only"),
            "rla_resume_diff_training_stage": bool(self.global_config.get("RLA_RESUME_DIFF_TRAINING_STAGE", False)),

            # ==========================
            # RLA: feature dims / pooling / norms
            # ==========================
            "rla_d_video_feat": self.global_config.get("D_VIDEO_FEAT", None),
            "rla_d_audio_feat": self.global_config.get("D_AUDIO_FEAT", None),

            "rla_video_temporal": self.global_config.get("RLA_VIDEO_TEMPORAL", "meanstd"),
            "rla_video_norm": self.global_config.get("RLA_VIDEO_NORM", None),  # none|l2|zscore

            "rla_audio_temporal": self.global_config.get("RLA_AUDIO_TEMPORAL", "none"),
            "rla_audio_norm": self.global_config.get("RLA_AUDIO_NORM", "l2"),

            # ==========================
            # RLA: adapter architecture / regularization
            # ==========================
            "rla_hidden_global": self.global_config.get("RLA_HIDDEN", 128),
            "rla_hidden_video": int(self.global_config.get("RLA_HIDDEN_VIDEO", self.rla_hidden)),
            "rla_hidden_audio": int(self.global_config.get("RLA_HIDDEN_AUDIO", self.rla_hidden)),

            "rla_p_moddrop_video": self.global_config.get("RLA_P_MODDROP_VIDEO", 0.30),
            "rla_p_moddrop_audio": self.global_config.get("RLA_P_MODDROP_AUDIO", 0.30),

            "rla_video_use_ln": bool(self.global_config.get("RLA_VIDEO_USE_LN", False)),
            "rla_video_use_conf_gain": bool(self.global_config.get("RLA_VIDEO_USE_CONF_GAIN", False)),
            "rla_video_conf_init_gain": float(self.global_config.get("RLA_VIDEO_CONF_INIT_GAIN", 3.0)),
            "rla_video_alpha_init": float(self.global_config.get("RLA_VIDEO_ALPHA_INIT", 1.0)),

            "rla_audio_use_ln": bool(self.global_config.get("RLA_AUDIO_USE_LN", False)),
            "rla_audio_use_conf_gain": bool(self.global_config.get("RLA_AUDIO_USE_CONF_GAIN", False)),
            "rla_audio_conf_init_gain": float(self.global_config.get("RLA_AUDIO_CONF_INIT_GAIN", 3.0)),
            "rla_audio_alpha_init": float(self.global_config.get("RLA_AUDIO_ALPHA_INIT", 1.0)),

            # ==========================
            # RLA: optimization knobs
            # ==========================
            "base_lr": float(self.global_config.get("BASE_LR", self.lr * 0.25)),
            "rla_lr": float(self.global_config.get("RLA_LR", self.lr * 5.0)),
            "hard_gamma": float(self.global_config.get("HARD_GAMMA", 0.0)),
    
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
            if ds in self.qa_datasets:
                ids.append(-1)
            elif ds not in self.dataset_to_domain_id:
                raise KeyError(f"Dataset '{ds}' not in label_map.meta.dataset_domain")
            else:
                ids.append(self.dataset_to_domain_id[ds])
        # store the tensors for the domain ids
        return torch.tensor(ids, dtype=torch.long, device=device)
    

    def _build_tf_inputs_and_labels(self, batch, qa_rows, seq_len, device):
        """
        Returns:
        qa_input_ids: [Bq, T]  = prompt + answer(+EOS), padded/truncated to T
        qa_attn:      [Bq, T]  = 1 on real tokens, 0 on pads
        lm_labels_q:  [Bq, T]  = -100 on prompt/pads, answer(+EOS) tokens elsewhere
        """
        import numpy as np
        if qa_rows is None or qa_rows.numel() == 0:
            return None, None, None

        ids_all = batch["input_ids"]
        attn_all = batch.get("attention_mask", None)

        Bq = qa_rows.numel()
        T  = seq_len
        pad_id = self.tokenizer.pad_token_id
        eos_id = self.tokenizer.eos_token_id

        qa_input_ids = torch.full((Bq, T), pad_id, dtype=torch.long, device=device)
        qa_attn      = torch.zeros((Bq, T), dtype=torch.long, device=device)
        lm_labels_q  = torch.full((Bq, T), -100,  dtype=torch.long, device=device)

        for j, idx in enumerate(qa_rows.tolist()):
            # --- prompt slice ---
            prompt_ids = ids_all[idx]                          # [T]
            if attn_all is not None:
                prompt_len = int(attn_all[idx].sum().item())   # count non-pad
            else:
                # fallback: count until first pad
                prompt_len = (prompt_ids != pad_id).sum().item()

            prompt_len = min(prompt_len, T)
            # copy prompt first
            qa_input_ids[j, :prompt_len] = prompt_ids[:prompt_len]
            qa_attn[j, :prompt_len] = 1

            # --- tokenize answer (+EOS) ---
            ans = batch["lm_labels"][idx]
            if isinstance(ans, np.generic):
                ans = ans.item()
            if isinstance(ans, bytes):
                ans = ans.decode("utf-8", errors="ignore")
            ans = "" if ans is None else str(ans)

            ans_tok = self.tokenizer.encode(ans, add_special_tokens=False)
            if len(ans_tok) == 0 or ans_tok[-1] != eos_id:
                ans_tok = (ans_tok + [eos_id])

            # space left after prompt
            rem = T - prompt_len
            if rem > 0:
                ans_tok = ans_tok[:rem]
                qa_input_ids[j, prompt_len:prompt_len+len(ans_tok)] = torch.tensor(ans_tok, device=device)
                qa_attn[j,      prompt_len:prompt_len+len(ans_tok)] = 1

                # labels: -100 on prompt, copy answer(+EOS)
                lm_labels_q[j,  prompt_len:prompt_len+len(ans_tok)] = qa_input_ids[j, prompt_len:prompt_len+len(ans_tok)]

        # sanity: no loss on pads
        if attn_all is not None:
            assert not (((lm_labels_q != -100) & (qa_attn == 0)).any()), "Loss on pads detected."

        return qa_input_ids, qa_attn, lm_labels_q
    
    def _current_model_order(self):
        order = ["base"]
        if self.rla_stage in {"residual_only", "joint", "residual_and_decoder"} and getattr(self, "video_adapter", None) is not None:
            order.append("video")
        if self.rla_stage in {"residual_only", "joint", "residual_and_decoder"} and getattr(self, "audio_adapter", None) is not None:
            order.append("audio")
        return order

    def get_dataloader(self, data_files, batch_size, num_workers=0, shuffle=True):
        dataset = AddQAOmniClassifierDataset(
            data_files=data_files,
            tokenizer=self.tokenizer,
            config=self.config,
            processor=self.processor,
            label_key=self.label_key,
            label_map=self.label_map,
            qa_datasets=self.qa_datasets,
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
        global_step = epoch * len_train_dataloader + (batch_idx + 1)

        ckpt_dir = os.path.join(base_ckpt_dir, f"step_{global_step}")
        os.makedirs(ckpt_dir, exist_ok=True)

        accelerator.save_state(ckpt_dir)
        accelerator.wait_for_everyone()

        if accelerator.is_main_process:
            meta = {
                "epoch": int(epoch),
                "global_step": int(global_step),
                "len_train_dataloader": int(len_train_dataloader),
                "training_strategy": str(training_strategy),
                "model_order": self._current_model_order(),  # <— NEW
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
        and return per-module param bundles so we can build one optimizer per prepared model.
        This keeps adapters modular (separate nn.Modules, separate optimizers).
        """
        base_lr = self.lr if base_lr is None else base_lr
        rla_lr  = self.lr if rla_lr  is None else rla_lr

        def _set_requires_grad(module, flag: bool):
            if module is None:
                return
            for p in module.parameters():
                p.requires_grad = flag

        bundles = {"base": None, "video": None, "audio": None}

        if self.rla_stage == "base_only":
            # train base only
            print("Freezing adapters, training base model only")
            _set_requires_grad(self.video_adapter, False)
            _set_requires_grad(self.audio_adapter, False)
            base_params = [p for p in self.model.parameters() if p.requires_grad]
            if base_params:
                bundles["base"] = {"params": base_params, "lr": base_lr}

        elif self.rla_stage == "residual_only":
            # train adapters only
            print("Freezing base model, training adapters only")
            _set_requires_grad(self.model, False)
            _set_requires_grad(self.video_adapter, True)
            _set_requires_grad(self.audio_adapter, True)

            base_params = list(self.model.parameters())
            if base_params:
                bundles["base"] = {"params": base_params, "lr": 0.0, "weight_decay": 0.0}

            if self.video_adapter is not None:
                vid_params = [p for p in self.video_adapter.parameters() if p.requires_grad]
                if vid_params:
                    bundles["video"] = {"params": vid_params, "lr": rla_lr}

            if self.audio_adapter is not None:
                aud_params = [p for p in self.audio_adapter.parameters() if p.requires_grad]
                if aud_params:
                    bundles["audio"] = {"params": aud_params, "lr": rla_lr}

        elif self.rla_stage == "residual_and_decoder":
            # train adapters only
            print("Freezing base model, training adapters and classifier/ lm_heads only")
            # Freeze the model backbone
            _set_requires_grad(self.model, False)
            # but unfreeze the lm_head
            _set_requires_grad(self.model.backbone.lm_head, True)
            _set_requires_grad(self.video_adapter, True)
            _set_requires_grad(self.audio_adapter, True)

            head_params = [p for p in self.model.backbone.lm_head.parameters() if p.requires_grad]

            if head_params:
                bundles["base"] = {"params": head_params, "lr": base_lr}

            if self.video_adapter is not None:
                vid_params = [p for p in self.video_adapter.parameters() if p.requires_grad]
                if vid_params:
                    bundles["video"] = {"params": vid_params, "lr": rla_lr}

            if self.audio_adapter is not None:
                aud_params = [p for p in self.audio_adapter.parameters() if p.requires_grad]
                if aud_params:
                    bundles["audio"] = {"params": aud_params, "lr": rla_lr}
       
        elif self.rla_stage == "joint":
            # train base + adapters (still separate optimizers per module)
            print("Training both base model and adapters")
            _set_requires_grad(self.video_adapter, True)
            _set_requires_grad(self.audio_adapter, True)

            base_params = [p for p in self.model.parameters() if p.requires_grad]
            if base_params:
                bundles["base"] = {"params": base_params, "lr": base_lr}

            if self.video_adapter is not None:
                vid_params = [p for p in self.video_adapter.parameters() if p.requires_grad]
                if vid_params:
                    bundles["video"] = {"params": vid_params, "lr": rla_lr}

            if self.audio_adapter is not None:
                aud_params = [p for p in self.audio_adapter.parameters() if p.requires_grad]
                if aud_params:
                    bundles["audio"] = {"params": aud_params, "lr": rla_lr}
        else:
            raise ValueError(f"Unknown RLA stage: {self.rla_stage}")

        return bundles
    
    def _accelerate_prepare_modules(
        self, 
        train_dataloader, 
        val_dataloader, 
        *, 
        prepare_base_model: bool = True, 
        prepare_adapters: bool = False
    ):
        """
        Prepare ONLY the specified modules + dataloaders.
        - prepare_base_model: whether to include the main model.
        - prepare_adapters:   whether to include adapters (video/audio).
        Returns the prepared objects in the same order they were passed in.
        """
        modules = []

        if prepare_base_model:
            modules.append(self.model)

        if self.rla_stage in {"residual_only", "joint", "residual_and_decoder"} and prepare_adapters:
            if getattr(self, "video_adapter", None) is not None:
                modules.append(self.video_adapter)
            if getattr(self, "audio_adapter", None) is not None:
                modules.append(self.audio_adapter)

        modules += [train_dataloader, val_dataloader]

        prepared = self.accelerator.prepare(*modules)

        idx = 0
        if prepare_base_model:
            self.model = prepared[idx]; idx += 1

        if self.rla_stage in {"residual_only", "joint", "residual_and_decoder"} and prepare_adapters:
            if getattr(self, "video_adapter", None) is not None:
                self.video_adapter = prepared[idx]; idx += 1
            if getattr(self, "audio_adapter", None) is not None:
                self.audio_adapter = prepared[idx]; idx += 1

        train_dataloader = prepared[idx]; idx += 1
        val_dataloader   = prepared[idx]; idx += 1

        # Return exactly what was prepared in the same logical grouping
        return train_dataloader, val_dataloader, (
            self.model if prepare_base_model else None,
            self.video_adapter if prepare_adapters else None,
            self.audio_adapter if prepare_adapters else None
        )
    
    def _build_per_module_optimizers(self, bundles):
        """Return (opts_in_order, names_in_order) matching module prepare order: base, video?, audio?."""
        opt_base  = Adam(bundles["base"]["params"],  lr=bundles["base"]["lr"])   if bundles["base"]  else None
        opt_video = Adam(bundles["video"]["params"], lr=bundles["video"]["lr"])  if bundles["video"] else None
        opt_audio = Adam(bundles["audio"]["params"], lr=bundles["audio"]["lr"])  if bundles["audio"] else None

        # build all the optimizers regardless
        opts = [o for o in (opt_base, opt_video, opt_audio) if o is not None]
        names = [n for n, o in zip(["base","video","audio"], (opt_base, opt_video, opt_audio)) if o is not None]
        return opts, names

    def _build_schedulers(self, opts_in_order, total_updates):
        """Per-optimizer schedulers, same order as optimizers. May return []."""
        if not self.use_scheduler:
            return []
        from transformers import get_scheduler
        scheds = [
            get_scheduler(
                self.scheduler_type,
                o,
                num_warmup_steps=self.warmup_steps,
                num_training_steps=total_updates,
            ) for o in opts_in_order
        ]
        return scheds

    def _accelerate_prepare_opts_scheds(self, opts_in_order, scheds_in_order):
        """Prepare ONLY optimizers/schedulers and register schedulers for checkpointing."""
        if scheds_in_order:
            prepared = self.accelerator.prepare(*opts_in_order, *scheds_in_order)
            n_opt = len(opts_in_order)
            self.prepared_opts   = list(prepared[:n_opt])
            self.prepared_scheds = list(prepared[n_opt:])
            for s in self.prepared_scheds:
                self.accelerator.register_for_checkpointing(s)
        else:
            self.prepared_opts   = list(self.accelerator.prepare(*opts_in_order))
            self.prepared_scheds = []

    def _step_all_opts(self):
        for opt in self.prepared_opts:
            opt.step()

    def _zero_all_opts(self):
        for opt in self.prepared_opts:
            opt.zero_grad(set_to_none=True)

    def _step_all_scheds(self):
        for sch in self.prepared_scheds:
            sch.step()

    def _current_lr_for_logging(self):
        return (self.prepared_opts[0].param_groups[0]["lr"] if self.prepared_opts else self.lr)
    
    @torch.no_grad()
    def _greedy_decode_rha_no_generate(
        self,
        qa_input_ids: torch.Tensor,          # [Bq, T0] prompt
        qa_attn: torch.Tensor | None,        # [Bq, T0] or None
        *,
        domain_ids_q: torch.Tensor,          # [Bq] == -1 for QA
        pooled_video_feats: torch.Tensor | None,   # [Bq, Dv] or None (already prebuilt in the loop)
        pooled_audio_feats: torch.Tensor | None,   # [Bq, Da] or None
        max_new_tokens: int = 64,
    ):
        """
        Greedy decoding without .generate, but WITH adapter fusion every step:
        1) forward (labels=None) -> hidden_states[-2] -> pooled_base
        2) apply_hidden_adapters(..., train_mode=False) -> pooled_eff
        3) forward(..., video_pooled_rha/audio_pooled_rha=pooled_eff) -> logits
        4) take argmax at last position, append token, continue
        Returns:
            cont_ids: [Bq, L]
        """
        # Work on copies and keep everything on the same device/dtype
        input_ids = qa_input_ids.clone()
        attn      = qa_attn.clone() if qa_attn is not None else None
        device    = input_ids.device
        Bq        = input_ids.size(0)

        # EOS / PAD
        eos_id = self.tokenizer.eos_token_id
        pad_id = self.tokenizer.pad_token_id if self.tokenizer.pad_token_id is not None else eos_id

        finished  = torch.zeros(Bq, dtype=torch.bool, device=device)
        generated = []

        for _ in range(max_new_tokens):
            # --- pass 1: get pooled_base from penultimate hidden ---
            out0 = self.model(
                input_ids=input_ids,
                attention_mask=attn,
                domain_ids=domain_ids_q,
                lm_labels=None,               # no CE here
            )
            h_penult = out0["lm_output"].hidden_states[-2]  # [Bq, T, H]
            if attn is not None:
                pooled_base = (h_penult * attn.unsqueeze(-1)).sum(1) / (attn.sum(1, keepdim=True).clamp_min(1))
            else:
                pooled_base = h_penult.mean(dim=1)          # [Bq, H]

            # --- fuse adapters exactly like train (train_mode=False) ---
            pooled_eff = apply_hidden_adapters(
                h_base=pooled_base,
                domain_ids=domain_ids_q,                    # -1s are fine; adapters can ignore if desired
                prelim_global_logits=None,                  # not needed during QA
                video_hidden_adapter=self.video_adapter if self.use_rla_video else None,
                audio_hidden_adapter=self.audio_adapter if self.use_rla_audio else None,
                video_feats=pooled_video_feats,
                audio_feats=pooled_audio_feats,
                train_mode=False,
            )

            # We keep precedence consistent with forward(): audio > video > base
            video_pooled_rha = pooled_eff if (self.use_rla_video and pooled_video_feats is not None) else None
            audio_pooled_rha = pooled_eff if (self.use_rla_audio and pooled_audio_feats is not None) else None

            # --- pass 2: Δ injection + LM logits from modified states ---
            out = self.model(
                input_ids=input_ids,
                attention_mask=attn,
                domain_ids=domain_ids_q,
                lm_labels=None,                       # still no CE (free decoding)
                video_pooled_rha=video_pooled_rha,
                audio_pooled_rha=audio_pooled_rha,    # overrides video if both provided
            )

            next_logits = out["lm_output"].logits[:, -1, :]  # [Bq, V]
            next_tokens = next_logits.argmax(dim=-1)         # [Bq]
            next_tokens = torch.where(finished, torch.full_like(next_tokens, pad_id), next_tokens)
            generated.append(next_tokens.unsqueeze(1))

            # append token & update mask
            input_ids = torch.cat([input_ids, next_tokens.unsqueeze(1)], dim=1)
            if attn is not None:
                one = torch.ones((Bq, 1), dtype=attn.dtype, device=device)
                attn = torch.cat([attn, one], dim=1)

            # stop early if all are done
            finished = finished | (next_tokens == eos_id)

        return torch.cat(generated, dim=1) if generated else torch.empty((Bq, 0), dtype=input_ids.dtype, device=device)

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

        all_pred_texts = []
        all_gold_texts = []
        all_qa_datasets = []

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
                if ("audio_feats" in batch) and (self.rla_stage in {"residual_only", "joint", "residual_and_decoder"}) and self.use_rla_audio:
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
                
                if ("video_feats" in batch) and (self.rla_stage in {"residual_only", "joint", "residual_and_decoder"}) and self.use_rla_video:
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

                    # TODO: ideally we should log the failed paths, but nevermind for now
                else:
                    pooled_video_feats = None

                # Handle labels shape
                if labels.dim() != 1:
                    if labels.dim() == 2 and labels.size(1) == num_classes:
                        labels = labels.argmax(dim=1)
                    else:
                        raise ValueError(f"Unexpected labels shape {labels.shape}")
                    
                # ---- QA greedy decode (no .generate), WITH adapters ----
                qa_input_ids = input_ids
                qa_attn      = attention_mask

                # Domain sentinel = -1s for QA rows
                domain_ids_q = torch.full((qa_input_ids.size(0),), -1, dtype=torch.long, device=qa_input_ids.device)

                # Use the RHA-aware greedy decode
                cont_ids_local = self._greedy_decode_rha_no_generate(
                    qa_input_ids=qa_input_ids,
                    qa_attn=qa_attn,
                    domain_ids_q=domain_ids_q,
                    pooled_video_feats=pooled_video_feats if (self.use_rla_video and pooled_video_feats is not None) else None,
                    pooled_audio_feats=pooled_audio_feats if (self.use_rla_audio and pooled_audio_feats is not None) else None,
                    max_new_tokens=self.max_val_qa_tokens,
                )

                # Gather across processes
                g_cont_ids = self.accelerator.gather_for_metrics(cont_ids_local)
                g_prompts  = self.accelerator.gather_for_metrics(qa_input_ids)
                g_lm_labels = self.accelerator.gather_for_metrics(batch["lm_labels"])
                g_datasets  = self.accelerator.gather_for_metrics(batch["dataset"])

                # Decode on main proc
                if self.accelerator.is_main_process:
                    pred_texts = self.tokenizer.batch_decode(g_cont_ids, skip_special_tokens=True)
                    all_pred_texts.extend(pred_texts)
                    all_gold_texts.extend(g_lm_labels)   # raw gold strings from your dataset (unchanged)
                    all_qa_datasets.extend(g_datasets)
                    print(f"Pred: {pred_texts}")

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
        
        # Get the hidden size of the model
        H = getattr(self.model, "hidden_size", None)
        if H is None:
            raise RuntimeError("Model must expose .hidden_size for RHA out_dim")

        # Building adapters if required;
        # NOTE: INITALISED FOR HIDDEN ADAPETRS, but reusing the video_adapter/ audio_adapter functionality
        self.video_adapter, self.audio_adapter = maybe_build_hidden_adapters(
                domain_id_to_global_indices=self.domain_id_to_global_indices,
                use_rha_video=self.use_rla_video,
                use_rha_audio=self.use_rla_audio,
                rha_hidden_video=self.rla_hidden_video,
                rha_hidden_audio=self.rla_hidden_audio,
                p_moddrop_video=self.rla_pv,
                p_moddrop_audio=self.rla_pa,
                out_dim_hidden=H,
                d_video_feat=self.d_video_feat,
                d_audio_feat=self.d_audio_feat,
                # per-modality adapter knobs
                video_use_ln=bool(self.global_config.get("RLA_VIDEO_USE_LN", False)),
                video_use_conf_gain=bool(self.global_config.get("RLA_VIDEO_USE_CONF_GAIN", False)),
                video_conf_init_gain=float(self.global_config.get("RLA_VIDEO_CONF_INIT_GAIN", 3.0)),
                video_alpha_init=float(self.global_config.get("RLA_VIDEO_ALPHA_INIT", 1.0)),
                audio_use_ln=bool(self.global_config.get("RLA_AUDIO_USE_LN", False)),
                audio_use_conf_gain=bool(self.global_config.get("RLA_AUDIO_USE_CONF_GAIN", False)),
                audio_conf_init_gain=float(self.global_config.get("RLA_AUDIO_CONF_INIT_GAIN", 3.0)),
                audio_alpha_init=float(self.global_config.get("RLA_AUDIO_ALPHA_INIT", 1.0)),
            )
        criterion = CrossEntropyLoss()
        # ---- Compute update-steps-aware schedule sizes ----
        total_updates = self.epochs * len(train_dataloader)
        base_lr = self.global_config.get("BASE_LR", self.lr * 0.25)
        rla_lr  = self.global_config.get("RLA_LR",  self.lr * 5.0)
        # for hard examples learning (if used)
        gamma   = float(self.global_config.get("HARD_GAMMA", 0.0))

        # always init containers
        self.prepared_opts, self.prepared_scheds = [], []

        if self.rla_resume_diff_training_stage:
            # Phase 1: prepare modules WITHOUT adapters (we'll add fresh adapters next)
            # Because you'll likely be resuming from a base model only checkpoint
            train_dataloader, val_dataloader, (prepared_model, _, _) = self._accelerate_prepare_modules(
                train_dataloader, val_dataloader,
                prepare_base_model=True,
                prepare_adapters=False,
            )
            # Load only model/RNG
            start_epoch, start_batch_offset, _, _, _ = self.load_checkpoint_unified(
                accelerator=self.accelerator,
                model=self.model,
                base_ckpt_dir=self.checkpoint_dir,
                explicit_dir=self.load_checkpoint_path or None,
                expect_training_strategy=self.global_config.get("TRAINING_STRATEGY"),
                rla_resume_diff_cfg=True,
            )
            # Phase 1b: now prepare freshly built adapters as modules-only
            if self.rla_stage in {"residual_only", "joint", "residual_and_decoder"}:
                adapters = []
                if self.video_adapter is not None: adapters.append(self.video_adapter)
                if self.audio_adapter is not None: adapters.append(self.audio_adapter)
                if adapters:
                    prepared = self.accelerator.prepare(*adapters)
                    if not isinstance(prepared, (list, tuple)):
                        prepared = [prepared]
                    i = 0
                    if self.video_adapter is not None:
                        self.video_adapter = prepared[i]; i += 1
                    if self.audio_adapter is not None and i < len(prepared):
                        self.audio_adapter = prepared[i]

            # Phase 2: build per-module optimizers/schedulers for CURRENT prepared modules
            bundles = self.prepare_params_for_training(base_lr=base_lr, rla_lr=rla_lr)
            opts_in_order, self._opt_names_in_order = self._build_per_module_optimizers(bundles)
            sch_in_order = self._build_schedulers(opts_in_order, total_updates)
            self._accelerate_prepare_opts_scheds(opts_in_order, sch_in_order)

        else:
            # SAME-REGIME (or fresh)
            # Phase 1: prepare modules WITH adapters if active
            train_dataloader, val_dataloader, (prepared_model, prepared_video, prepared_audio) = self._accelerate_prepare_modules(
                                                                        train_dataloader, val_dataloader,
                                                                        prepare_base_model=True,  # don’t wrap the model again
                                                                        prepare_adapters=True,
                                                                    )
            
            # Load full state (model+opts+scheds+RNG) if compatible
            start_epoch, start_batch_offset, _, _, _ = self.load_checkpoint_unified(
                accelerator=self.accelerator,
                model=self.model,
                base_ckpt_dir=self.checkpoint_dir,
                explicit_dir=self.load_checkpoint_path or None,
                expect_training_strategy=self.global_config.get("TRAINING_STRATEGY"),
                rla_resume_diff_cfg=False,
            )
            # Phase 2: rebuild per-module optimizers/schedulers (so shapes/flags match current freeze plan)
            bundles = self.prepare_params_for_training(base_lr=base_lr, rla_lr=rla_lr)
            opts_in_order, self._opt_names_in_order = self._build_per_module_optimizers(bundles)
            sch_in_order = self._build_schedulers(opts_in_order, total_updates)
            self._accelerate_prepare_opts_scheds(opts_in_order, sch_in_order)
            
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

                # Calculate current step for validation checking
                current_step = (epoch * len(train_dataloader)) + batch_idx + 1
                
                # remember to set it back to train() mode after validation
                if self.video_adapter is not None:
                    self.video_adapter.train()
                if self.audio_adapter is not None:
                    self.audio_adapter.train()


                if epoch == start_epoch and batch_idx < start_batch_offset:
                    continue
                
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
                if ("audio_feats" in batch) and (batch["audio_feats"] is not None) \
                    and (self.rla_stage in {"residual_only", "joint", "residual_and_decoder"}) and self.use_rla_audio:
                    
                    audio_feats = batch["audio_feats"]

                    # folllowing the video_feats_batch, the pooled_audio_feats should be [B, X*K*C]
                    pooled_audio_feats = build_audio_feats_batch(
                        audio_feats,
                        device=input_ids.device,
                        temporal_mode=self.audio_temporal,
                        norm=self.audio_norm,
                        target_dim=self.d_audio_feat, 
                    )

                else:
                    pooled_audio_feats = None
                
                if ("video_feats" in batch) and (batch["video_feats"] is not None) \
                    and (self.rla_stage in {"residual_only", "joint", "residual_and_decoder"}) and self.use_rla_video:

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
                        norm=self.video_norm,                    # <<< NEW
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

                B, T = input_ids.size()
                device = input_ids.device

                # ---- start from the classification view of the batch ----
                input_ids_full = input_ids
                attn_full      = attention_mask

                # ---- build masked LM labels for QA rows ----

                lm_labels_full = None
                # has_qa = qa_rows is not None and qa_rows.numel() > 0 and ('lm_labels' in batch)
                # if has_qa:
                
                qa_rows = torch.arange(B, device=device)  # [0..B-1]
                
                # teacher-forced per-row tensors
                qa_input_ids, qa_attn, lm_labels_q = self._build_tf_inputs_and_labels(
                    batch=batch, qa_rows=qa_rows, seq_len=T, device=device
                )

                # (1) replace rows in input_ids / attention_mask for QA rows
                #     If you don't want to mutate originals, clone first:
                input_ids_full = input_ids.clone()
                attn_full      = attention_mask.clone() if attention_mask is not None else None
                input_ids_full.index_copy_(0, qa_rows, qa_input_ids)
                if attn_full is not None:
                    attn_full.index_copy_(0, qa_rows, qa_attn)

                # (2) build full labels with -100 everywhere except QA rows
                lm_labels_full = torch.full((B, T), -100, dtype=torch.long, device=device)
                lm_labels_full.index_copy_(0, qa_rows, lm_labels_q)

                with self.accelerator.accumulate(self.model):
                    # 1) first pass (no labels) to get pooled_base
                    out0 = self.model(
                        input_ids=input_ids_full,
                        attention_mask=attn_full,
                        domain_ids=domain_ids,
                        lm_labels=None,                   # we’ll compute CE after injection in pass 2
                    )
                    # compute pooled_base from out0.hidden_states[-2] exactly as in the forward above
                    h_penult = out0["lm_output"].hidden_states[-2]
                    if attn_full is not None:
                        pooled_base = (h_penult * attn_full.unsqueeze(-1)).sum(1) / attn_full.sum(1, keepdim=True)
                    else:
                        pooled_base = h_penult.mean(dim=1)

                    # 3) hidden fusion (RHA) if enabled
                    if (self.rla_stage in {"residual_only","joint", "residual_and_decoder"}) and (self.use_rla_video or self.use_rla_audio):
                        pooled_after_video = (
                            self.video_adapter(pooled_base, domain_ids.clamp(min=0), global_logits=None, feats=pooled_video_feats, train_mode=True)
                            if (self.video_adapter is not None and pooled_video_feats is not None) else pooled_base
                        )
                        pooled_after_audio = (
                            self.audio_adapter(pooled_after_video, domain_ids.clamp(min=0), global_logits=None, feats=pooled_audio_feats, train_mode=True)
                            if (self.audio_adapter is not None and pooled_audio_feats is not None) else pooled_after_video
                        )

                    # 3) second pass does Δ injection + LM CE + cls using the overrides
                    out = self.model(
                        input_ids=input_ids_full,
                        attention_mask=attn_full,
                        domain_ids=domain_ids,
                        lm_labels=lm_labels_full,               # teacher forcing
                        video_pooled_rha=pooled_after_video,    # optional
                        audio_pooled_rha=pooled_after_audio,    # precedence over video if both passed
                    )
                    qa_loss = out["lm_loss"]
                    lm_output = out["lm_output"]

                    # total_loss_this_step = cls_loss + self.qa_loss_weight * qa_loss
                    total_loss_this_step = self.qa_loss_weight * qa_loss

                    # Accelerate handles gradient accumulation automatically
                    self.accelerator.backward(total_loss_this_step)

                    # NOTE: This if condition is not necessary as the accelerator 
                    # already syncs before stepping
                    # if self.accelerator.sync_gradients:

                    self._step_all_opts()
                    self._step_all_scheds()
                    self._zero_all_opts()

                    current_lr = self._current_lr_for_logging()
                    
                    # PURELY FOR LOGGING PURPOSES
                    # if scheduler is not None:
                    #     did_update = False
                    #     if self.accelerator.sync_gradients:
                    #         did_update = True
                    #     if did_update:
                    #         if self.prepared_opts:
                    #             current_lr = self.prepared_opts[0].param_groups[0]['lr']
                    #         else:
                    #             current_lr = optimizer.param_groups[0]['lr'] 
                    #     else:
                    #         current_lr = None
                    # else:
                    #     current_lr = self.lr
                        
                with torch.no_grad():
                    # ---- metrics only on CLS rows ----
                    # if preds_cls is not None:
                    # gathered_preds = self.accelerator.gather_for_metrics(preds_cls)
                    # gathered_labels = self.accelerator.gather_for_metrics(labels.index_select(0, cls_rows))

                    if self.accelerator.is_main_process:
                        # effective_batch_correct += (gathered_preds == gathered_labels).sum().item()
                        # effective_batch_total += gathered_labels.size(0)
                        # correct += (gathered_preds == gathered_labels).sum().item()
                        # total_loss += gathered_labels.size(0)
                        effective_batch_correct += 0
                        effective_batch_total += 0
                        correct += 0
                        total_loss += 0

                    # if lm_output is not None:

                    # 1) Get token-level predictions (greedy) for the QA rows
                    pred_text_ids = lm_output.logits.argmax(dim=-1)  # [B,T]

                    # 2) Only evaluate/print tokens where labels are active (labels != -100)
                    active = (lm_labels_q != -100)
                    # For labels: make them decodable
                    text_labels_for_decode = lm_labels_q.masked_fill(lm_labels_q == -100,
                                                            self.tokenizer.pad_token_id)

                    # Optional: mask predictions to the same active positions
                    # (keeps inactive tokens as pad for clean decoding)
                    pred_text_ids_masked = torch.where(active, pred_text_ids, self.tokenizer.pad_token_id)

                    # 3) Gather across processes for consistent printing/metrics
                    gathered_text_pred = self.accelerator.gather_for_metrics(pred_text_ids_masked)
                    gathered_text_labels  = self.accelerator.gather_for_metrics(text_labels_for_decode)

                    # 4) Decode to strings
                    if self.accelerator.is_main_process:
                        pred_text = self.tokenizer.batch_decode(gathered_text_pred, skip_special_tokens=True)
                        gold_text = self.tokenizer.batch_decode(gathered_text_labels,  skip_special_tokens=True)

                    if self.accelerator.is_main_process and len(pred_text):
                        # show a couple of samples
                        for i in range(min(2, len(pred_text))):
                            print(f"[QA pred] {pred_text[i]}")
                            print(f"[QA gold] {gold_text[i]}")

                    # Accumulate epoch/effective losses using the combined loss
                    bs = input_ids.size(0)
                    total_loss += total_loss_this_step.item() * bs
                    effective_batch_loss += total_loss_this_step.item() * bs

                    # Check if this completes an effective batch (gradient update)
                    # An effective batch consists of gradient_accumulation_steps individual batches
       
                    # Log training metrics for the effective batch
                    # TODO: Make sure that you do this also for validation
                    # if effective_batch_total == 0:
                    effective_batch_total = 1  # to avoid div-by-zero in accuracy
       
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

        H = getattr(self.model, "hidden_size", None)
        if H is None:
            raise RuntimeError("Model must expose .hidden_size for RHA out_dim")

        # Building adapters if required;
        # NOTE: INITALISED FOR HIDDEN ADAPETRS, but reusing the video_adapter/ audio_adapter functionality
        
        self.video_adapter, self.audio_adapter = maybe_build_hidden_adapters(
                domain_id_to_global_indices=self.domain_id_to_global_indices,
                use_rha_video=self.use_rla_video,
                use_rha_audio=self.use_rla_audio,
                rha_hidden_video=self.rla_hidden_video,
                rha_hidden_audio=self.rla_hidden_audio,
                p_moddrop_video=self.rla_pv,
                p_moddrop_audio=self.rla_pa,
                out_dim_hidden=H,
                d_video_feat=self.d_video_feat,
                d_audio_feat=self.d_audio_feat,
                # per-modality adapter knobs
                video_use_ln=bool(self.global_config.get("RLA_VIDEO_USE_LN", False)),
                video_use_conf_gain=bool(self.global_config.get("RLA_VIDEO_USE_CONF_GAIN", False)),
                video_conf_init_gain=float(self.global_config.get("RLA_VIDEO_CONF_INIT_GAIN", 3.0)),
                video_alpha_init=float(self.global_config.get("RLA_VIDEO_ALPHA_INIT", 1.0)),
                audio_use_ln=bool(self.global_config.get("RLA_AUDIO_USE_LN", False)),
                audio_use_conf_gain=bool(self.global_config.get("RLA_AUDIO_USE_CONF_GAIN", False)),
                audio_conf_init_gain=float(self.global_config.get("RLA_AUDIO_CONF_INIT_GAIN", 3.0)),
                audio_alpha_init=float(self.global_config.get("RLA_AUDIO_ALPHA_INIT", 1.0)),
            )

        # --- Phase 1: modules-only prepare (model + adapters, plus loaders) ---
        train_dataloader, test_dataloader = self._accelerate_prepare_modules(
            train_dataloader=train_dataloader,
            val_dataloader=test_dataloader,
            prepare_base_model=True,
            prepare_adapters=(self.rla_stage in {"residual_only", "joint", "residual_and_decoder"}),
        )

        # --- Build per-module optimizers (no stepping during test; required so load_state can map) ---
        base_lr = self.global_config.get("BASE_LR", self.lr * 0.25)
        rla_lr  = self.global_config.get("RLA_LR",  self.lr * 5.0)
        bundles = self.prepare_params_for_training(base_lr=base_lr, rla_lr=rla_lr)

        opts_in_order, opt_names = self._build_per_module_optimizers(bundles)   # e.g., ["base","video","audio"]
        # Optional schedulers (harmless in test; needed if checkpoints contain them)
        sch_in_order = []
        if self.use_scheduler and len(opts_in_order) > 0:
            total_updates = max(1, self.epochs * len(train_dataloader))  # any positive int is fine for rekeying
            sch_in_order = [
                get_scheduler(
                    self.scheduler_type,
                    o,
                    num_warmup_steps=self.warmup_steps,
                    num_training_steps=total_updates
                ) for o in opts_in_order
            ]

        # --- Phase 2: prepare ONLY the optimizers/schedulers ---
        self._accelerate_prepare_opts_scheds(opts_in_order, sch_in_order)

        # --- Load (model + optimizer(s) + scheduler(s) + RNG) ---
        _ = self.load_checkpoint_unified(
            accelerator=self.accelerator,
            model=self.model,
            base_ckpt_dir=self.checkpoint_dir,
            explicit_dir=self.load_checkpoint_path or None,
            expect_training_strategy=self.global_config.get("TRAINING_STRATEGY"),
            rla_resume_diff_cfg=False,  # test assumes same regime graph
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
