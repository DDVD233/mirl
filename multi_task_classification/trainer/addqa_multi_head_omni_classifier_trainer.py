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
from mt_dataset.addqa_omni_classifier_dataset import AddQAOmniClassifierDataset, SkipBatchSampler
from verl.utils.dataset.rl_dataset import collate_fn
from utils.wandb_utils import init_wandb, log_metrics, finish
from utils.logger import log_batch_training_metrics, log_validation_results, log_epoch_training_metrics
# from evaluate.multi_task_evaluation import evaluate_predictions
from evaluate.detailed_multi_task_evaluation import evaluate_predictions

# Accelerate imports
from accelerate import Accelerator
from accelerate.utils import set_seed, DistributedDataParallelKwargs, InitProcessGroupKwargs
from accelerate.logging import get_logger

logger = get_logger(__name__)

class MultiHeadOmniClassifierAccelerateTrainer:
    def __init__(self, data_files, val_data_files, test_data_files, tokenizer, processor, config, 
                 batch_size, val_batch_size, test_batch_size, lr, epochs, save_checkpoint_dir, load_checkpoint_path, model, 
                 gradient_accumulation_steps, num_workers=0, use_lora=False, global_config=None):
        self.data_files = data_files
        self.val_data_files = val_data_files
        self.test_data_files = test_data_files
        self.tokenizer = tokenizer
        self.processor = processor
        # EOS / PAD sanity (needed for both training & generation)
        if self.tokenizer.eos_token_id is None:
            raise ValueError("Tokenizer needs an eos_token_id for QA SFT.")
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id  # common default
        self.config = config # basically the config for the dataset
        self.batch_size = batch_size
        self.val_batch_size = val_batch_size
        self.test_batch_size = test_batch_size
        self.gradient_accumulation_steps = gradient_accumulation_steps
        self.num_workers = num_workers
        self.lr = lr
        self.epochs = epochs
        self.model = model
        
        # (optional but nice) make generate() stop naturally
        self.model.backbone.generation_config.eos_token_id = self.tokenizer.eos_token_id
        self.model.backbone.generation_config.pad_token_id = self.tokenizer.pad_token_id

        self.label_key = config.get("label_key", "answer")
        # Store global configuration for access to constants
        self.global_config = global_config or {}
        self.qa_datasets = set(self.global_config.get('QA_DATASETS', ['intentqa', 'mimeqa', 'siq2']))  # e.g. {"mmlu_qa","ptsd_qa","finance_qa"}
        self.qa_loss_weight = float(self.global_config.get('QA_LOSS_WEIGHT', 1.0))

        # Deterministic generation for evaluation
        self.max_val_qa_tokens = 30
        print(f"WARNING: Using max_val_qa_tokens={self.max_val_qa_tokens} for validation/test generation.")
        
        # Use the label map from global config
        self.full_label_scheme = self.global_config.get('FULL_LABEL_SCHEME', None)
        self.label_map = self.global_config.get('LABEL_MAP', {})
        self.label_map_path = self.global_config.get('LABEL_MAP_PATH', None)

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
        ddp = DistributedDataParallelKwargs(find_unused_parameters=True)

        self.accelerator = Accelerator(
            gradient_accumulation_steps=gradient_accumulation_steps,
            mixed_precision='fp16',  # Use fp16 for better memory efficiency
            log_with="wandb" if use_wandb else None,
            project_dir=save_checkpoint_dir if use_wandb else None,
            kwargs_handlers=[ddp]
        )
        
        # Set seed for reproducibility
        # This makes sure that the dataloader shuffling is preserved, 
        # so that the random sampler uses this fixed generator
        # each epoch
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
            "warmup_steps": self.warmup_steps if self.use_scheduler else None,
            "qa_datasets": sorted(list(self.qa_datasets)),
            "qa_loss_weight": self.qa_loss_weight,

        }
        init_wandb(
            project=self.global_config.get('WANDB_PROJECT', ''),
            entity=self.global_config.get('WANDB_ENTITY', ''),
            config=wandb_config,
            run_name=f"omni_classifier_accelerate_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        )

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
    
    def _split_qa_cls_indices(self, dataset_names):
        qa_idx, cls_idx = [], []
        for i, ds in enumerate(dataset_names):
            if isinstance(ds, bytes):
                ds = ds.decode("utf-8", errors="ignore")
            ds = str(ds).lower()
            (qa_idx if ds in self.qa_datasets else cls_idx).append(i)
        qa = torch.tensor(qa_idx, dtype=torch.long) if qa_idx else None
        cl = torch.tensor(cls_idx, dtype=torch.long) if cls_idx else None
        return qa, cl
    
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

        start_epoch = floor((global_step - 1) / len_dl)
        start_batch_offset = (global_step - 1) % len_dl

        accelerator.print(f"[load] resumed {ckpt_dir} → epoch={start_epoch}, step={global_step}, offset={start_batch_offset}")
    
        return start_epoch, start_batch_offset, global_step, meta, ckpt_dir
    
    @torch.no_grad()
    def _greedy_decode_no_generate(self, qa_input_ids, qa_attn, max_new_tokens=64):
        """
        Manual greedy decoding using the same forward pass (no .generate, no teacher forcing).
        - qa_input_ids: [Bq, T] prompt tokens
        - qa_attn:      [Bq, T] or None
        Returns:
        cont_ids:  [Bq, L] newly generated token ids (L <= max_new_tokens)
        """
        # Work on copies to avoid mutating caller tensors
        input_ids = qa_input_ids.clone()
        attn      = qa_attn.clone() if qa_attn is not None else None

        device = input_ids.device
        Bq     = input_ids.size(0)

        # EOS / PAD
        eos_id = self.tokenizer.eos_token_id
        pad_id = self.tokenizer.pad_token_id if self.tokenizer.pad_token_id is not None else eos_id

        finished = torch.zeros(Bq, dtype=torch.bool, device=device)
        generated = []

        for _ in range(max_new_tokens):
            # Domain sentinel = -1 for QA (no head routing)
            # It should already be handled in the original domain_ids in the main loop, but just to be sure
            domain_ids_q = torch.full((Bq,), -1, dtype=torch.long, device=device)
            
            out = self.model(
                input_ids=input_ids,
                attention_mask=attn,
                domain_ids=domain_ids_q,
                lm_labels=None
            )
        
            # Same forward path; take LM logits from last position
            next_logits = out["lm_output"].logits[:, -1, :]        # [Bq, V]
            next_tokens = next_logits.argmax(dim=-1)               # [Bq]

            # If sequence already finished, keep emitting PAD to keep shape consistent
            next_tokens = torch.where(finished, torch.full_like(next_tokens, pad_id), next_tokens)

            generated.append(next_tokens.unsqueeze(1))             # accumulate [Bq, 1]

            # Append to input_ids (+ update attn if present)
            input_ids = torch.cat([input_ids, next_tokens.unsqueeze(1)], dim=1)
            if attn is not None:
                one = torch.ones((Bq, 1), dtype=attn.dtype, device=device)
                attn = torch.cat([attn, one], dim=1)

            # Early stop if all hit EOS
            finished = finished | (next_tokens == eos_id)
            if torch.all(finished):
                break

        # if not generated:
        #     # no tokens generated (edge-case max_new_tokens=0)
        #     return torch.empty((Bq, 0), dtype=input_ids.dtype, device=device)

        return torch.cat(generated, dim=1)  # [Bq, L]
    

    def validate(self, val_dataloader, split_name="validation", current_step=None):
        """Validate the model on the given dataloader."""
        self.model.eval()
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
                lm_labels = batch['lm_labels']
                datasets = batch["dataset"]

                domain_ids = self._datasets_to_domain_ids(batch['dataset'], device=input_ids.device)

                # retrieve the batch and domain_ids for all the batch
                if 'dataset' not in batch:
                    raise KeyError("Batch missing 'dataset' needed for domain routing.")
                # batch['dataset'] is typically a list/tuple length B
                # domain_ids = self._datasets_to_domain_ids(batch['dataset'], device=input_ids.device)

                # Handle labels shape
                if labels.dim() != 1:
                    if labels.dim() == 2 and labels.size(1) == num_classes:
                        labels = labels.argmax(dim=1)
                    else:
                        raise ValueError(f"Unexpected labels shape {labels.shape}")
                    
                #     # Split batch into QA vs CLS by dataset name
                # qa_rows, cls_rows = self._split_qa_cls_indices(batch['dataset'])
                # device = input_ids.device

                # if qa_rows is not None:  
                #     qa_rows = qa_rows.to(device)
                
                # if cls_rows is not None: 
                #     cls_rows = cls_rows.to(device)

                # # ---- Classification pass (unchanged) ----
                # if cls_rows is not None and cls_rows.numel() > 0:

                #     out = self.model(
                #         input_ids.index_select(0, cls_rows),
                #         attention_mask=attention_mask.index_select(0, cls_rows) if attention_mask is not None else None,
                #         domain_ids=domain_ids,
                #         lm_labels=None
                #     )

                #     cls_logits = out["cls_logits"]
                #     labels_cls = labels.index_select(0, cls_rows)

                #     loss = criterion(cls_logits, labels_cls)
                #     total_cls_loss += loss.item() * labels_cls.size(0)
                #     n_cls_samples += labels_cls.size(0)

                #     preds = cls_logits.argmax(dim=1)
                #     gathered_preds  = self.accelerator.gather_for_metrics(preds)
                #     gathered_labels = self.accelerator.gather_for_metrics(labels_cls)

                #     if self.accelerator.is_main_process:
                #         all_predictions.extend(gathered_preds.cpu().tolist())
                #         all_labels.extend(gathered_labels.cpu().tolist())
                #         all_datasets.extend(datasets)


                # # ---- QA: free generation, gather cont_ids then decode on main ----
                # has_qa = (qa_rows is not None and qa_rows.numel() > 0)
                # if has_qa:
                #     qa_input_ids = input_ids.index_select(0, qa_rows)
                #     qa_attn      = attention_mask.index_select(0, qa_rows) if attention_mask is not None else None

                # 1) Greedy decode continuation IDs ONLY (no .generate()), fixed L tokens
                qa_input_ids = input_ids
                qa_attn = attention_mask

                    # Your helper should return a [Bq, L] LongTensor, pad with tokenizer.pad_token_id if needed
                cont_ids_local = self._greedy_decode_no_generate(qa_input_ids, 
                                                                 qa_attn, 
                                                                 max_new_tokens=self.max_val_qa_tokens)  # [Bq, L]

                # 2) Gather IDs across processes (tensors only)
                g_cont_ids = self.accelerator.gather_for_metrics(cont_ids_local)        # [N_total, L]
                g_prompts  = self.accelerator.gather_for_metrics(qa_input_ids)          # [N_total, T]  (optional; only if you want full sequences)

                # collect them all
                gathered_lm_labels = self.accelerator.gather_for_metrics(lm_labels)  # [N_total]
                gathered_datasets  = self.accelerator.gather_for_metrics(datasets)   #
            
                # 4) Decode ONLY on main process (after gathering)
                if self.accelerator.is_main_process:
                    # Option A: decode only continuation
                    pred_texts = self.tokenizer.batch_decode(g_cont_ids, skip_special_tokens=True)

                    # Option B (optional): decode full sequences (prompt + continuation)
                    # full_ids = torch.cat([g_prompts, g_cont_ids], dim=1)
                    # pred_texts = self.tokenizer.batch_decode(full_ids, skip_special_tokens=True)

                    all_pred_texts.extend(pred_texts)
                    all_gold_texts.extend(gathered_lm_labels)
                    all_qa_datasets.extend(gathered_datasets)

        # Calculate average loss
        avg_loss = total_loss / max(1, len(all_labels)) if self.accelerator.is_main_process else 0.0

        if self.accelerator.is_main_process:
            out_dir = self.validation_result_dir or "."
            os.makedirs(out_dir, exist_ok=True)
            step_tag = str(current_step) if current_step is not None else "final"
            out_path = os.path.join(out_dir, f"{split_name}_qa_preds_step_{step_tag}.json")

            qa_records = [
                {"dataset": d, "pred": p, "gold": g}
                for d, p, g in zip(all_qa_datasets, all_pred_texts, all_gold_texts)
            ]
            with open(out_path, "w", encoding="utf-8") as f:
                json.dump(qa_records, f, ensure_ascii=False, indent=2)
            print(f"[QA] Saved {len(qa_records)} records to: {out_path}")
        
        # Use the new evaluation module (only on main process)
        if self.accelerator.is_main_process:
            if all_predictions and all_labels:
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

        start_epoch, start_batch_offset, _, _, _ = self.load_checkpoint_unified(
            accelerator=self.accelerator,
            model=self.model,
            base_ckpt_dir=self.checkpoint_dir,
            explicit_dir=self.load_checkpoint_path or None,
            expect_training_strategy=self.global_config.get("TRAINING_STRATEGY"),
        )

        # 3) OPTIONAL resumed loader (only used for the first resumed epoch)
        # Not required for now, as we always resume full epochs
        skipped_dataloader = None
        start_epoch = 0
        start_batch_offset = 0
        # if start_batch_offset > 0:
        #     skipped_dataloader = self.accelerator.skip_first_batches(
        #         train_dataloader, start_batch_offset
        #     )

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

            # Use the resume loader only for the first (partial) resumed epoch
            is_resumed_epoch = (epoch == start_epoch and skipped_dataloader is not None)
            cur_loader = skipped_dataloader if is_resumed_epoch else train_dataloader

            total_loss = 0.0
            correct = 0
            total_loss = 0
            epoch_start_time = time.time()

            # Variables for effective batch tracking (gradient updates)
            effective_batch_loss = 0.0
            effective_batch_correct = 0
            effective_batch_total = 0

            # Adjust step math so logs/checkpoints reflect true global position
            # So essentially if we are resuming mid checkpoint, then we need to use this base_offset
            base_offset = start_batch_offset if is_resumed_epoch else 0

            for batch_idx, batch in tqdm(enumerate(cur_loader), desc="Training", total=len(cur_loader), disable=not self.accelerator.is_main_process):
                # Set model to training mode (needed because validation sets it to eval mode)
                self.model.train()

                # current_step uses the *full* epoch length plus what we've already completed
                current_step = (epoch * len(train_dataloader)) + base_offset + batch_idx + 1
                
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

                # labels sanity
                if labels.dim() != 1:
                    # If your dataset sometimes emits multi-task/one-hot, squeeze or argmax here
                    if labels.dim() == 2 and labels.size(1) == num_classes:
                        labels = labels.argmax(dim=1)
                    else:
                        raise ValueError(f"Unexpected labels shape {labels.shape} (expected [B] or [B, C])")
              
                # --- split batch into QA vs CLS subsets based on dataset name ---
                # qa_rows, cls_rows = self._split_qa_cls_indices(batch['dataset'])
                                                                         
                # device = input_ids.device
                                                                         
                # # MOVE INDICES TO THE SAME DEVICE AS input_ids
                # if qa_rows is not None:
                #     qa_rows = qa_rows.to(device)
                # if cls_rows is not None:
                #     cls_rows = cls_rows.to(device)

                B, T = input_ids.size()
                device = input_ids.device

                # ---- domain ids: sentinel -1 for QA-only rows ----
                # domain_ids_full = torch.full((B,), -1, dtype=torch.long, device=device)
                
                # if cls_rows is not None and cls_rows.numel() > 0:
                #     # TODO: This may be causing your hang issues; the iterable list
                #     ds_cls = [batch['dataset'][i] for i in cls_rows.tolist()]
                #     domain_ids_cls = self._datasets_to_domain_ids(ds_cls, device=device)
                #     domain_ids_full.index_copy_(0, cls_rows, domain_ids_cls)

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
                    # ---- single forward pass ----
                    model_output = self.model(
                        input_ids=input_ids_full,
                        attention_mask=attn_full,
                        domain_ids=domain_ids,   # -1 means "no head" (QA-only row; and this will automatically be handled)
                        lm_labels=lm_labels_full      # None or masked by -100
                    )
                    # Classification loss only on cls_rows
                    # cls_loss = torch.zeros([], device=device)
                    # preds_cls = None
                    # if cls_rows is not None and cls_rows.numel() > 0:
                    #     # only do cls_loss over the cls_rows
                    #     labels_cls = labels.index_select(0, cls_rows)
                    #     cls_logits = model_output["cls_logits"].index_select(0, cls_rows)
                    #     cls_loss   = criterion(cls_logits, labels_cls)
                    #     preds_cls  = cls_logits.argmax(dim=1)

                    # LM loss is computed internally only for QA rows (thanks to -100 mask)
                    # qa_loss = model_output["lm_loss"] if has_qa else torch.zeros([], device=device)
                    qa_loss = model_output["lm_loss"]
                    # lm_output = None
                    # if has_qa:
                    lm_output = model_output["lm_output"]

                    # total_loss_this_step = cls_loss + self.qa_loss_weight * qa_loss
                    total_loss_this_step =self.qa_loss_weight * qa_loss

                    self.accelerator.backward(total_loss_this_step)

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
                        current_lr=current_lr,
                        current_step=current_step
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
                        # val_results = self.validate_off_accelerate_with_generate(val_dataloader_raw, "validation", current_step)

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
                            batch_idx=base_offset + batch_idx,
                            len_train_dataloader=len(train_dataloader),
                            training_strategy=self.global_config.get("TRAINING_STRATEGY"),
                            base_ckpt_dir=self.checkpoint_dir,
                        )
            # End of epoch
            skipped_dataloader = None  # Only use the resumed loader for one epoch

            # Calculate training metrics
            avg_train_loss = total_loss / max(1, total_loss)
            train_acc = correct / max(1, total_loss)
            
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
                        use_wandb=use_wandb,
                        current_step=current_step
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
                    use_wandb=use_wandb,
                    current_step=current_step
                )

            # Save checkpoint: every N epochs and also when best
            if save_every_n_epochs and ((epoch + 1) % save_every_n_epochs == 0):
                self.save_checkpoint_unified(
                accelerator=self.accelerator,
                model=self.model,
                epoch=epoch,
                batch_idx=base_offset + batch_idx, # save the offset within the epoch
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
        print("STARTING TESTING PHASE   ")
        print("="*50)
        train_dataloader = self.get_dataloader(self.data_files, self.batch_size, num_workers=self.num_workers, shuffle=True)
        test_dataloader = self.get_dataloader(self.test_data_files, self.test_batch_size, num_workers=self.num_workers, shuffle=False)

        optimizer = Adam(self.model.parameters(), lr=self.lr)
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


        # everything should be prepared again by accelerator, including the model
         # Prepare everything with Accelerate
        if scheduler is not None:
            self.model, optimizer, train_dataloader, test_dataloader, scheduler = self.accelerator.prepare(
                self.model, optimizer, train_dataloader, test_dataloader, scheduler
            )
            # Register the scheduler for checkpointing
            self.accelerator.register_for_checkpointing(scheduler)
        else:
            self.model, optimizer, train_dataloader, test_dataloader = self.accelerator.prepare(
                self.model, optimizer, train_dataloader, test_dataloader
            )

        # loading of the model etc. 

        _, _, _, _, _ = self.load_checkpoint_unified(
            accelerator=self.accelerator,
            model=self.model,
            base_ckpt_dir=self.checkpoint_dir, # essentially the save path; ignored if we specified load_checkpoint_path
            explicit_dir=self.load_checkpoint_path or None,
            expect_training_strategy=self.global_config.get("TRAINING_STRATEGY"),
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
