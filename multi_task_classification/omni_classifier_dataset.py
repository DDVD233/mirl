import numpy as np
import os
import sys
import torch
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from verl.utils.dataset.rl_dataset import RLHFDataset

class OmniClassifierDataset(RLHFDataset):
    def __init__(self, *args, label_key='answer', label_map=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.label_key = label_key 
        
        self.label_map = label_map  # Optional: dict mapping raw label to class index

    def __getitem__(self, item):
        # connects to RLHFDataset __getitem__
        # gets the row dict which includes processed inputs 
        # from Qwen etc.
        # row = super().__getitem__(item)
        
        # -------------------------------------------------------------------
        # NOTE: ONLY FOR DEBUGGING, remove after done.
        row_dict: dict = self.dataframe[item]
        if "input_ids" not in row_dict:
            row_dict["input_ids"] = torch.randint(
                low=0, high=1000, size=(32,), dtype=torch.long)  # e.g., seq_len=32
        if "attention_mask" not in row_dict:
            row_dict["attention_mask"] = torch.ones_like(row_dict["input_ids"])

        row = row_dict
        # -------------------------------------------------------------------

        # --- your existing label extraction ---
        raw_label = row_dict.get(self.label_key)
        if self.label_map is not None:
            label = self.label_map.get(raw_label, 0)
        else:
            raise ValueError(f"label_map must be provided for mapping raw labels {raw_label} to class indices")

        # --- CRITICAL: ensure tensors so collate stacks them as tensors ---
        # labels must be Long for CrossEntropyLoss
        row_dict["labels"] = torch.tensor(label, dtype=torch.long)

        return row
