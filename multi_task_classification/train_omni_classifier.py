import os
import torch
from torch.utils.data import DataLoader
from torch.optim import Adam
from torch.nn import CrossEntropyLoss
import sys
from dummy_classifier import DummyClassifier
# from omni_classifier import OmniClassifier
from omni_classifier_dataset import OmniClassifierDataset
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from verl.utils.dataset.rl_dataset import collate_fn
from transformers import AutoTokenizer, AutoProcessor
from omegaconf import OmegaConf
from label_maps import *

# ---------------------------
# CONFIG
# ---------------------------
DATA_FILES = ["//Users/keane/Desktop/research/human-behavior/data/indiv/sample.jsonl"]
TOKENIZER_NAME = "Qwen/Qwen2.5-Omni-7B"
PROCESSOR_NAME = "Qwen/Qwen2.5-Omni-7B"
NUM_CLASSES = 5 #TODO: the num_classes should be dictated by the label map and the dataset/ task that we are doing
FREEZE_BACKBONE = True
BATCH_SIZE = 1
LR = 1e-3
EPOCHS = 3
CHECKPOINT_PATH = "/Users/keane/Desktop/research/human-behavior/verl/multi_task_heads/omni_classifier_ckpt.pt"
DEBUG_DRY_RUN = True  # <<< set True to avoid loading the real model

# TODO: this should wrap around the huggingface loading stuff
tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_NAME)
processor = AutoProcessor.from_pretrained(PROCESSOR_NAME)

config = OmegaConf.create({
    "max_prompt_length": 1024,
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
    "cache_dir": "/Users/keane/Desktop/research/human-behavior/data",
    "format_prompt": "/Users/keane/Desktop/research/human-behavior/verl/examples/format_prompt/default.jinja"
})


# ---------------------------
# TRAINER
# ---------------------------
class OmniClassifierTrainer:
    def __init__(self, data_files, tokenizer, processor, config, batch_size, lr, epochs, checkpoint_path, model):
        self.data_files = data_files
        self.tokenizer = tokenizer
        self.processor = processor
        self.config = config
        self.batch_size = batch_size
        self.lr = lr
        self.epochs = epochs
        self.checkpoint_path = checkpoint_path
        self.model = model
        self.label_key = config.get("label_key", "answer")
        # TODO code it such that we just inherit once from the dataset
        self.label_map = SENTIMENT_LABEL_MAP  # or EMOTION_LABEL_MAP based on your task
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    def get_dataloader(self):
        dataset = OmniClassifierDataset(
            data_files=self.data_files,
            tokenizer=self.tokenizer,
            config=self.config,
            processor=self.processor,
            label_key=self.label_key,
            label_map=self.label_map
        )
        # For debug, num_workers=0 avoids multiprocessing noise
        return DataLoader(dataset, batch_size=self.batch_size, shuffle=True, collate_fn=collate_fn,
                          num_workers=0, pin_memory=False, persistent_workers=False)

    def debug_batch_loader(self, n_batches: int = 1):
        print("[DEBUG] Running batch loader probe...")
        dl = self.get_dataloader()
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

    def train(self, max_steps: int | None = None):
        dataloader = self.get_dataloader()
        self.model.train().to(self.device)

        # If your real model exposes freeze_backbone, honor it; dummy doesn't.
        freeze_backbone = getattr(self.model, 'freeze_backbone', False)
        params = self.model.classifier.parameters() if freeze_backbone and hasattr(self.model, 'classifier') else self.model.parameters()
        optimizer = Adam(params, lr=self.lr)
        criterion = CrossEntropyLoss()

        # --- LOAD CHECKPOINT IF EXISTS (skip safely if state-dict keys don't match) ---
        start_epoch = 0
        # TODO: Please check whether the loading is the same as the original verl code.
        if os.path.exists(self.checkpoint_path):
            try:
                print(f"Loading checkpoint from {self.checkpoint_path}")
                checkpoint = torch.load(self.checkpoint_path, map_location=self.device)
                self.model.load_state_dict(checkpoint['model_state_dict'], strict=False)
                optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
                start_epoch = checkpoint.get('epoch', 0)
            except Exception as e:
                print(f"[WARN] Could not load checkpoint due to: {e}. Continuing fresh.")

        global_step = 0
        for epoch in range(start_epoch, self.epochs):
            total_loss = 0.0
            correct = 0
            total = 0

            for batch in dataloader:
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

            avg_loss = total_loss / max(1, total)
            acc = correct / max(1, total)
            print(f"Epoch {epoch+1}/{self.epochs} - Loss: {avg_loss:.4f} - Acc: {acc:.4f}")

            # --- SAVE CHECKPOINT ---
            try:
                torch.save({
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'epoch': epoch + 1,
                }, self.checkpoint_path)
                print(f"Checkpoint saved to {self.checkpoint_path}")
            except Exception as e:
                print(f"[WARN] Failed to save checkpoint: {e}")

            if max_steps is not None and global_step >= max_steps:
                break


if __name__ == "__main__":
    if DEBUG_DRY_RUN:
        print("[INFO] Using DummyClassifier for dry-run (no backbone loaded).")
        model = DummyClassifier(num_classes=NUM_CLASSES)
    else:
        model = OmniClassifier(num_classes=NUM_CLASSES, freeze_backbone=FREEZE_BACKBONE)

    trainer = OmniClassifierTrainer(
        data_files=DATA_FILES,
        tokenizer=tokenizer,
        processor=processor,
        config=config,
        batch_size=BATCH_SIZE,
        lr=LR,
        epochs=EPOCHS,
        checkpoint_path=CHECKPOINT_PATH,
        model=model,
    )

    # 1) Dataloader probe (prints batch structure)
    trainer.debug_batch_loader()

    # 2) Optional: run a VERY short training dry-run (e.g., 3 steps) to test loss/backward/optimizer/checkpoint
    trainer.train(max_steps=3)